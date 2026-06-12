import os
import json
import time
import hashlib
import random
import re
import argparse
import google.generativeai as genai
from datetime import datetime
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

def compute_file_md5(path):
    """Calcula a hash MD5 de um ficheiro de forma eficiente."""
    md5_obj = hashlib.md5()
    try:
        with open(path, "rb") as stream:
            while True:
                block = stream.read(8192)
                if not block:
                    break
                md5_obj.update(block)
        return md5_obj.hexdigest()
    except Exception as err:
        print(f"[MD5 ERROR] Não foi possível calcular hash para {path}: {err}")
        return None

def sanitize_json_str(raw_text):
    """Limpa e extrai uma string JSON válida de uma resposta do modelo."""
    raw_text = raw_text.strip()
    # Tenta encontrar formato markdown ```json ... ```
    json_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
    if json_block:
        cleaned = json_block.group(1)
    else:
        # Fallback: encontrar o primeiro '{' e o último '}'
        first_curly = raw_text.find('{')
        last_curly = raw_text.rfind('}')
        if first_curly != -1 and last_curly != -1:
            cleaned = raw_text[first_curly:last_curly+1]
        else:
            cleaned = raw_text
    # Remover caracteres não imprimíveis
    return re.sub(r'[\x00-\x1F\x7F]', '', cleaned)

class ShelfInspector:
    """Componente responsável pela análise visual das prateleiras usando a API do Gemini."""
    
    def __init__(self, api_key=None, cache_dir="cache", inspections_dir="data/inspections", model_name="gemini-2.5-flash"):
        self.key = api_key or os.getenv("GEMINI_API_KEY")
        self.cache_path = cache_dir
        self.inspections_path = inspections_dir
        self.model_id = model_name
        
        os.makedirs(self.cache_path, exist_ok=True)
        os.makedirs(self.inspections_path, exist_ok=True)
        
        self.sdk_client = None
        self._setup_sdk()
        
    def _setup_sdk(self):
        """Inicializa o cliente da API do Google Generative AI."""
        if not self.key:
            print("[WARN] A variável de ambiente GEMINI_API_KEY não foi detetada.")
            return
        try:
            genai.configure(api_key=self.key)
            self.sdk_client = genai
        except Exception as error:
            print(f"[ERROR] Erro ao configurar o SDK do Gemini: {error}")

    def call_gemini_with_backoff(self, model, prompt, image, max_retries=5):
        """Executa a chamada da API Gemini aplicando backoff exponencial com jitter em caso de rate limit."""
        if not self.sdk_client:
            raise ValueError("O cliente Gemini não foi configurado corretamente. Verifica a tua chave da API.")
        
        retry_delay = 2.0
        for attempt in range(max_retries):
            try:
                payload = [image, prompt] if image else [prompt]
                
                config = self.sdk_client.GenerationConfig(
                    temperature=0.0,
                    response_mime_type="application/json" if "json" in prompt.lower() else "text/plain"
                )
                
                gen_model = self.sdk_client.GenerativeModel(self.model_id)
                response = gen_model.generate_content(payload, generation_config=config)
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

    def inspect_image(self, image_path, zone_id, strategy="B", custom_prompt_path=None):
        """Inspeciona uma imagem de prateleira utilizando a estratégia de prompting definida."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagem ausente no caminho especificado: {image_path}")
        
        md5_sig = compute_file_md5(image_path)
        cache_filename = os.path.join(self.cache_path, f"{md5_sig}_{strategy}.json") if md5_sig else None
        
        # Tentativa de leitura do cache local
        if cache_filename and os.path.exists(cache_filename):
            print(f"[CACHE_HIT] A carregar inspeção existente de {image_path} (Estratégia {strategy})")
            try:
                with open(cache_filename, "r", encoding="utf-8") as stream:
                    return json.load(stream)
            except Exception as err:
                print(f"[CACHE_WARN] Erro ao ler ficheiro de cache: {err}. A reprocessar com a API...")

        # Fallback em modo offline se não houver chave ou falha no cliente
        if not self.key or not self.sdk_client:
            print("[OFFLINE_FALLBACK] Sem ligação à API. A procurar cache alternativo...")
            if md5_sig:
                for cached_item in os.listdir(self.cache_path):
                    if cached_item.startswith(md5_sig) and cached_item.endswith(".json"):
                        with open(os.path.join(self.cache_path, cached_item), "r", encoding="utf-8") as stream:
                            cached_data = json.load(stream)
                            cached_data["strategy_used"] = f"{strategy}_fallback_{cached_item.split('_')[-1].split('.')[0]}"
                            return cached_data
            raise ValueError("Incapaz de processar imagem: API indisponível e sem cache disponível.")

        # Determina o ficheiro do prompt
        prompt_file = custom_prompt_path or f"prompts/prompt_strat_{strategy.lower()}.txt"
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Ficheiro de prompt não encontrado: {prompt_file}")
            
        with open(prompt_file, "r", encoding="utf-8") as stream:
            prompt_text = stream.read()

        try:
            pil_image = Image.open(image_path)
        except Exception as err:
            raise ValueError(f"Não foi possível abrir a imagem {image_path}: {err}")

        print(f"[SHELF_INSPECTOR] A analisar prateleira {image_path} (Estratégia: {strategy})")
        try:
            api_response = self.call_gemini_with_backoff(self.model_id, prompt_text, pil_image)
        except Exception as err:
            # Fallback em caso de erro na chamada da API em runtime
            if md5_sig:
                for cached_item in os.listdir(self.cache_path):
                    if cached_item.startswith(md5_sig) and cached_item.endswith(".json"):
                        print(f"[FALLBACK] Erro na chamada da API. Usando cache alternativo: {cached_item}")
                        with open(os.path.join(self.cache_path, cached_item), "r", encoding="utf-8") as stream:
                            return json.load(stream)
            raise err

        # Limpeza e parsing da resposta
        cleaned_response = sanitize_json_str(api_response)
        try:
            parsed_data = json.loads(cleaned_response)
        except Exception as err:
            print(f"[PARSE_ERROR] Resposta do modelo não é um JSON válido: {err}")
            print(f"Resposta bruta:\n{api_response}")
            raise ValueError(f"Formato JSON inválido retornado pelo modelo: {err}")

        # Enriquecimento dos metadados da inspeção
        now_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        compact_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        uid = f"{random.randint(1, 999):03d}"
        
        if parsed_data.get("inspection_id") in [None, "", "INS_XXXXX"]:
            parsed_data["inspection_id"] = f"INS_{compact_ts}_{uid}"
            
        parsed_data["timestamp"] = now_utc
        parsed_data["image_path"] = image_path
        parsed_data["zone_id"] = zone_id
        parsed_data["strategy_used"] = strategy
        
        detected_issues = parsed_data.get("issues", [])
        shelf_fill = parsed_data.get("shelf_fill_rate", 1.0)
        
        # Determina o status geral se não vier na resposta
        if not parsed_data.get("overall_status"):
            if any(iss.get("severity") == "high" for iss in detected_issues) or shelf_fill < 0.5:
                parsed_data["overall_status"] = "critical"
            elif len(detected_issues) > 0 or shelf_fill < 0.8:
                parsed_data["overall_status"] = "warning"
            else:
                parsed_data["overall_status"] = "ok"

        # Garante identificadores para cada problema detetado
        for index, issue in enumerate(detected_issues):
            if issue.get("issue_id") in [None, "", "ISS_XXX"]:
                issue["issue_id"] = f"ISS_{index + 1:03d}"

        # Escrita no cache
        if cache_filename:
            with open(cache_filename, "w", encoding="utf-8") as stream:
                json.dump(parsed_data, stream, indent=2, ensure_ascii=False)
                
        # Escrita no registo permanente de inspeções
        archive_path = os.path.join(self.inspections_path, f"{parsed_data['inspection_id']}.json")
        with open(archive_path, "w", encoding="utf-8") as stream:
            json.dump(parsed_data, stream, indent=2, ensure_ascii=False)
            
        print(f"[SHELF_INSPECTOR] Inspeção guardada: {parsed_data['inspection_id']} (Estado: {parsed_data['overall_status']})")
        return parsed_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análise Visual de Prateleiras com Gemini")
    parser.add_argument("image", help="Caminho do ficheiro de imagem")
    parser.add_argument("--zone", default="Z_S3", help="ID da Zona da Loja")
    parser.add_argument("--strategy", default="B", choices=["A", "B", "C"], help="Estratégia de Prompting")
    args = parser.parse_args()
    
    try:
        inspector = ShelfInspector()
        inspection_res = inspector.inspect_image(args.image, args.zone, args.strategy)
        print(json.dumps(inspection_res, indent=2, ensure_ascii=False))
    except Exception as ex:
        print(f"[EXECUTION_ERROR] Ocorreu um erro: {ex}")
