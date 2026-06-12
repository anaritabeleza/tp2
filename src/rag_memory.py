import os
import json
import re
import chromadb
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

class RagMemory:
    """Gestão da memória semântica baseada em RAG e ChromaDB."""
    
    def __init__(self, vectorstore_dir="vectorstore", api_key=None, strategy="hybrid", model_name="gemini-2.5-flash"):
        self.vstore_path = vectorstore_dir
        self.key = api_key or os.getenv("GEMINI_API_KEY")
        self.rag_strategy = strategy
        self.model_id = model_name
        self.embedding_model = None
        self.db_client = None
        self.collection = None
        
        os.makedirs(self.vstore_path, exist_ok=True)
        self._load_embeddings_engine()
        self._configure_persistent_db()

    def _load_embeddings_engine(self):
        """Carrega o modelo de embeddings multilingue local para vetores de proximidade."""
        print("[RAG_MEMORY] Inicializando modelo sentence-transformers...")
        try:
            self.embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            print("[RAG_MEMORY] SentenceTransformer configurado com sucesso.")
        except Exception as err:
            print(f"[RAG_ERROR] Erro ao instanciar sentence-transformers: {err}")
            raise err

    def _configure_persistent_db(self):
        """Inicializa a conexão persistente ao ChromaDB."""
        print(f"[RAG_MEMORY] Conectando ao ChromaDB em: {self.vstore_path}")
        try:
            self.db_client = chromadb.PersistentClient(path=self.vstore_path)
            
            c_name = f"inspections_{self.rag_strategy}"
            self.collection = self.db_client.get_or_create_collection(
                name=c_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"[RAG_MEMORY] Coleção ativa: {c_name}")
        except Exception as err:
            print(f"[RAG_ERROR] Falha na conexão com ChromaDB: {err}")
            raise err

    def call_gemini_with_backoff(self, prompt, max_retries=5, temperature=0.0, response_mime_type="text/plain"):
        """Executa a chamada da API Gemini aplicando backoff exponencial com jitter em caso de rate limit."""
        import time
        import random
        import google.generativeai as genai
        
        retry_delay = 2.0
        for attempt in range(max_retries):
            try:
                config = genai.GenerationConfig(
                    temperature=temperature,
                    response_mime_type=response_mime_type
                )
                
                gen_model = genai.GenerativeModel(self.model_id)
                response = gen_model.generate_content(prompt, generation_config=config)
                return response.text
            except Exception as ex:
                ex_str = str(ex)
                is_quota_error = any(term in ex_str for term in ["429", "ResourceExhausted", "Quota"])
                
                if is_quota_error and attempt < max_retries - 1:
                    wait_seconds = retry_delay * (2 ** attempt) + random.uniform(0.1, 0.9)
                    print(f"[RATE_LIMIT] Limite atingido na chamada do Gemini. Tentativa {attempt + 1}/{max_retries}. A aguardar {wait_seconds:.2f}s...")
                    time.sleep(wait_seconds)
                else:
                    print(f"[API_ERROR] Erro durante a chamada do Gemini: {ex}")
                    raise ex
        raise RuntimeError("Número máximo de tentativas de ligação à API foi excedido.")

    def generate_summary(self, inspection_data, trajectory_data=None):
        """Gera um resumo semântico textual rico a partir dos dados estruturados da inspeção."""
        if not self.key:
            raise ValueError("API Key do Gemini ausente.")
            
        genai.configure(api_key=self.key)
        
        prompt_file = "prompts/prompt_rag_summary.txt"
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Ficheiro de prompt não encontrado: {prompt_file}")
            
        with open(prompt_file, "r", encoding="utf-8") as stream:
            template = stream.read()
            
        json_inspection = json.dumps(inspection_data, indent=2, ensure_ascii=False)
        json_trajectory = json.dumps(trajectory_data, indent=2, ensure_ascii=False) if trajectory_data else "Sem histórico de trajetórias."
            
        prompt = template.replace("{INSPECTION_JSON}", json_inspection).replace("{TRAJECTORY_DATA}", json_trajectory)
        
        try:
            res_text = self.call_gemini_with_backoff(
                prompt=prompt,
                temperature=0.0
            )
            return res_text.strip()
        except Exception as err:
            print(f"[WARN] Falha ao gerar resumo com Gemini: {err}. Usando fallback offline.")
            issues_list = inspection_data.get("issues", [])
            issues_desc = ", ".join([f"{iss.get('type')}: {iss.get('description')}" for iss in issues_list])
            if not issues_desc:
                issues_desc = "Nenhuma anomalia detetada."
            f_rate = inspection_data.get('shelf_fill_rate', 1.0)
            # Garantir que não crasha se f_rate não for número
            try:
                f_rate_pct = f"{float(f_rate)*100:.1f}%" if f_rate is not None else "N/A"
            except Exception:
                f_rate_pct = str(f_rate)
            fallback_summary = (
                f"Resumo da Inspeção {inspection_data.get('inspection_id')} na Zona {inspection_data.get('zone_id')}. "
                f"Estado Geral: {inspection_data.get('overall_status')}. Taxa de Ocupação: {f_rate_pct}. "
                f"Problemas detetados: {issues_desc}"
            )
            return fallback_summary

    def add_inspection(self, inspection_data, trajectory_data=None):
        """Indexa os dados de uma inspeção na base de dados vetorial."""
        rich_summary = self.generate_summary(inspection_data, trajectory_data)
        inspection_data["summary"] = rich_summary
        
        # Guardar permanentemente em data/inspections
        dir_inspections = "data/inspections"
        os.makedirs(dir_inspections, exist_ok=True)
        file_path = os.path.join(dir_inspections, f"{inspection_data['inspection_id']}.json")
        with open(file_path, "w", encoding="utf-8") as stream:
            json.dump(inspection_data, stream, indent=2, ensure_ascii=False)
            
        iid = inspection_data["inspection_id"]
        zid = inspection_data.get("zone_id", "Z_UNKNOWN")
        ts = inspection_data.get("timestamp", "")
        f_rate = float(inspection_data.get("shelf_fill_rate", 1.0))
        status = inspection_data.get("overall_status", "ok")
        
        if self.rag_strategy == "hybrid":
            meta = {
                "inspection_id": iid,
                "zone_id": zid,
                "timestamp": ts,
                "fill_rate": f_rate,
                "overall_status": status,
                "chunk_type": "summary"
            }
            vector = self.embedding_model.encode(rich_summary).tolist()
            
            self.collection.add(
                ids=[iid],
                embeddings=[vector],
                documents=[rich_summary],
                metadatas=[meta]
            )
            print(f"[RAG_MEMORY] Inspeção {iid} indexada (Estratégia: Hybrid).")
            
        elif self.rag_strategy == "by_issue":
            issues = inspection_data.get("issues", [])
            
            if not issues:
                doc = f"Inspeção na zona {zid} normalizada. Prateleiras limpas e organizadas. Taxa de ocupação de {f_rate*100:.1f}%."
                doc_uid = f"{iid}_no_issue"
                meta = {
                    "inspection_id": iid,
                    "zone_id": zid,
                    "timestamp": ts,
                    "fill_rate": f_rate,
                    "overall_status": status,
                    "chunk_type": "no_issue"
                }
                vector = self.embedding_model.encode(doc).tolist()
                self.collection.add(
                    ids=[doc_uid],
                    embeddings=[vector],
                    documents=[doc],
                    metadatas=[meta]
                )
            else:
                for idx, issue in enumerate(issues):
                    issue_uid = issue.get("issue_id", f"ISS_{idx+1:03d}")
                    i_type = issue.get("type", "other")
                    sev = issue.get("severity", "low")
                    pos = issue.get("location", "any")
                    description = issue.get("description", "")
                    
                    doc = f"Inconformidade registada na zona {zid}: '{i_type}' na localização '{pos}' com nível '{sev}'. Nota: {description}"
                    doc_uid = f"{iid}_{issue_uid}"
                    
                    meta = {
                        "inspection_id": iid,
                        "zone_id": zid,
                        "timestamp": ts,
                        "issue_id": issue_uid,
                        "issue_type": i_type,
                        "severity": sev,
                        "chunk_type": "issue"
                    }
                    vector = self.embedding_model.encode(doc).tolist()
                    self.collection.add(
                        ids=[doc_uid],
                        embeddings=[vector],
                        documents=[doc],
                        metadatas=[meta]
                    )
            print(f"[RAG_MEMORY] Inspeção {iid} indexada (Estratégia: By Issue, {len(issues) or 1} chunks).")

    def search(self, query, k=3, zone_filter=None):
        """Pesquisa registos históricos por proximidade semântica."""
        q_vector = self.embedding_model.encode(query).tolist()
        
        filter_dict = {}
        if not zone_filter:
            match = re.search(r'\b(z_s[1-9])\b', query.lower())
            if match:
                zone_filter = match.group(1).upper()
                print(f"[RAG_MEMORY] Extrapolado filtro de zona dos metadados: '{zone_filter}'")
                
        if zone_filter:
            filter_dict["zone_id"] = zone_filter
            
        where_cond = filter_dict if filter_dict else None
        
        search_res = self.collection.query(
            query_embeddings=[q_vector],
            n_results=k,
            where=where_cond
        )
        
        matches = []
        if search_res and search_res.get("documents"):
            docs = search_res["documents"][0]
            metas = search_res["metadatas"][0]
            uids = search_res["ids"][0]
            dists = search_res["distances"][0] if "distances" in search_res else [0.0] * len(uids)
            
            for index in range(len(uids)):
                matches.append({
                    "id": uids[index],
                    "document": docs[index],
                    "metadata": metas[index],
                    "score": 1.0 - dists[index]
                })
        return matches

    def query_memory(self, query, k=3):
        """Efetua uma pesquisa na memória RAG e sintetiza a resposta usando o modelo Gemini."""
        retrieved_docs = self.search(query, k=k)
        
        if not retrieved_docs:
            return "Nenhum registo histórico condizente com a pesquisa foi detetado.", []

        formatted_context = []
        for idx, item in enumerate(retrieved_docs):
            meta = item["metadata"]
            formatted_context.append(
                f"[{idx+1}] Inspeção ID: {meta.get('inspection_id')} | Zona: {meta.get('zone_id')} | Data: {meta.get('timestamp')}\n"
                f"Descrição: {item['document']}\n"
                f"Grau Proximidade: {item['score']*100:.1f}%\n"
                f"----------------------------------------"
            )
        context_block = "\n".join(formatted_context)
        
        prompt_file = "prompts/prompt_rag_synthesis.txt"
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Template do prompt de síntese RAG em falta: {prompt_file}")
            
        with open(prompt_file, "r", encoding="utf-8") as stream:
            template = stream.read()
            
        prompt = template.replace("{QUERY}", query).replace("{CONTEXT}", context_block)
        
        if not self.key:
            offline_text = f"[OFFLINE] Encontrados {len(retrieved_docs)} registos históricos:\n\n"
            offline_text += "\n\n".join([item['document'] for item in retrieved_docs])
            return offline_text, retrieved_docs
            
        genai.configure(api_key=self.key)
        
        try:
            res_text = self.call_gemini_with_backoff(
                prompt=prompt,
                temperature=0.0
            )
            return res_text.strip(), retrieved_docs
        except Exception as err:
            print(f"[WARN] Falha na síntese RAG com Gemini: {err}. Usando fallback offline.")
            offline_text = f"[FALLBACK OFFLINE - QUOTA EXCEDIDA] Encontrados {len(retrieved_docs)} registos históricos:\n\n"
            offline_text += "\n\n".join([item['document'] for item in retrieved_docs])
            return offline_text, retrieved_docs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gestão de Memória Semântica RAG")
    parser.add_argument("--query", help="Pesquisa textual livre")
    parser.add_argument("--strategy", default="hybrid", choices=["hybrid", "by_issue"], help="Estratégia de Chunking")
    args = parser.parse_args()
    
    memory = RagMemory(strategy=args.strategy)
    if args.query:
        ans, docs = memory.query_memory(args.query)
        print("\n=== Resposta Sintetizada ===")
        print(ans)
        print("\n=== Documentos de Suporte ===")
        for d in docs:
            print(f"- {d['id']} (Score: {d['score']:.2f}) | {d['document'][:110]}...")
