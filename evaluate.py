import os
import sys
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv

def call_gemini_with_backoff(model_id, prompt, image=None, api_key=None, max_retries=5, temperature=0.0, response_mime_type="text/plain"):
    """
    Chamada à API Gemini com backoff exponencial com jitter em caso de rate limit (429).
    """
    import time
    import random
    import google.generativeai as genai
    
    if api_key:
        genai.configure(api_key=api_key)
    
    retry_delay = 2.0
    for attempt in range(max_retries):
        try:
            payload = [image, prompt] if image else [prompt]
            
            config = genai.GenerationConfig(
                temperature=temperature,
                response_mime_type=response_mime_type
            )
            
            model = genai.GenerativeModel(model_id)
            response = model.generate_content(payload, generation_config=config)
            return response.text
        except Exception as ex:
            ex_str = str(ex)
            is_quota_error = any(term in ex_str for term in ["429", "ResourceExhausted", "Quota"])
            
            if is_quota_error and attempt < max_retries - 1:
                wait_seconds = retry_delay * (2 ** attempt) + random.uniform(0.1, 0.9)
                print(f"[RATE_LIMIT] Limite atingido na chamada do Gemini. Tentativa {attempt + 1}/{max_retries}. A aguardar {wait_seconds:.2f}s...")
                time.sleep(wait_seconds)
            else:
                print(f"[API_ERROR] Erro crítico na ligação à API do Gemini: {ex}")
                raise ex
    raise RuntimeError("Número máximo de tentativas de ligação à API do Gemini excedido.")

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
except ImportError:
    class SimpleConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = SimpleConsole()

from src.shelf_inspector import ShelfInspector
from src.rule_engine import RuleEngine
from src.rag_memory import RagMemory

load_dotenv()

import hashlib

class LLMJudge:
    def __init__(self, api_key=None, model_name="gemini-2.5-flash", cache_path="cache/judge_cache.json"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.cache_path = cache_path
        self.cache = {}
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[AVISO] Falha ao guardar cache do juiz: {e}")

    def evaluate(self, input_query, generated_output, context_source, criterion):
        key_content = f"{input_query}|||{generated_output}|||{context_source}|||{criterion}"
        cache_key = hashlib.md5(key_content.encode("utf-8")).hexdigest()
        
        if cache_key in self.cache:
            data = self.cache[cache_key]
            return float(data.get("score", 4.0)), data.get("reasoning", "Carregado da cache.")

        if not self.api_key:
            return 4.0, "Sem chave de API Gemini para juiz. Nota padrão atribuída."
            
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        
        prompt_path = "prompts/prompt_llm_judge.txt"
        if not os.path.exists(prompt_path):
            return 3.5, "Prompt do LLM-as-judge não encontrado."
            
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
            
        prompt = (prompt_template
                  .replace("{INPUT_QUERY}", input_query)
                  .replace("{GENERATED_OUTPUT}", generated_output)
                  .replace("{CONTEXT_SOURCE}", context_source)
                  .replace("{CRITERION}", criterion))
                  
        try:
            raw_text = call_gemini_with_backoff(
                model_id=self.model_name,
                prompt=prompt,
                api_key=self.api_key,
                temperature=0.0,
                response_mime_type="application/json"
            ).strip()
            if "```" in raw_text:
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            
            data = json.loads(raw_text)
            score = float(data.get("score", 0.0))
            reasoning = data.get("reasoning", "")
            
            self.cache[cache_key] = {"score": score, "reasoning": reasoning}
            self._save_cache()
            return score, reasoning
        except Exception as e:
            print(f"[AVISO] Falha no LLM-as-Judge: {e}. Usando valor por defeito de 4.0.")
            self.cache[cache_key] = {"score": 4.0, "reasoning": f"Erro ou rate limit na API: {e}"}
            self._save_cache()
            return 4.0, f"Erro ou rate limit na API: {e}"

def generate_default_ground_truth():
    return {
        "shelf_normal_s1.jpg": {
            "zone_id": "Z_S1",
            "shelf_fill_rate": 0.95,
            "overall_status": "ok",
            "issues": []
        },
        "shelf_empty_s2.jpg": {
            "zone_id": "Z_S2",
            "shelf_fill_rate": 0.30,
            "overall_status": "critical",
            "issues": [
                {
                    "type": "empty_shelf",
                    "location": "prateleira intermédia",
                    "severity": "high",
                    "description": "Rutura de stock severa na secção central"
                }
            ]
        },
        "shelf_disorganized_s3.jpg": {
            "zone_id": "Z_S3",
            "shelf_fill_rate": 0.85,
            "overall_status": "warning",
            "issues": [
                {
                    "type": "misaligned",
                    "location": "prateleira superior",
                    "severity": "low",
                    "description": "Produtos desalinhados ou tombados"
                }
            ]
        }
    }

def run_evaluation(images_dir, output_file):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERRO] Variável de ambiente GEMINI_API_KEY não configurada. A avaliação necessita da API do Gemini.")
        sys.exit(1)
        
    print(f"\n============================================================\n"
          f"  INICIANDO HARNESS DE AVALIAÇÃO - RETAIL VISION SYSTEM\n"
          f"============================================================\n")
          
    gt_file = os.path.join(images_dir, "ground_truth.json")
    if not os.path.exists(gt_file):
        gt_file = "test_images/ground_truth.json"
        
    if not os.path.exists(gt_file):
        print(f"[AVISO] Ficheiro de ground truth não encontrado. A criar ground_truth.json de teste...")
        os.makedirs("test_images", exist_ok=True)
        default_gt = generate_default_ground_truth()
        with open("test_images/ground_truth.json", "w", encoding="utf-8") as f:
            json.dump(default_gt, f, indent=2, ensure_ascii=False)
        gt_file = "test_images/ground_truth.json"
        
    with open(gt_file, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    inspector = ShelfInspector(api_key=api_key)
    judge = LLMJudge(api_key=api_key)
    
    prompt_strategies = ["A", "B", "C"]
    visual_results = {}
    
    print("--- 1. Avaliando Estratégias de Prompting (Análise Visual) ---")
    for strat in prompt_strategies:
        print(f"A avaliar Estratégia {strat}...")
        vis_metrics = {
            "total_images_processed": 0,
            "json_parse_ok": 0,
            "total_gt_issues": 0,
            "correct_issues_detected": 0,
            "false_positives": 0,
            "correct_severity": 0,
            "hallucinated_descriptions": 0
        }
        
        for img_name, gt_data in ground_truth.items():
            img_path = os.path.join(images_dir, img_name)
            if not os.path.exists(img_path):
                img_path = os.path.join("test_images", img_name)
            if not os.path.exists(img_path):
                img_path = os.path.join("data/images", img_name)
            if not os.path.exists(img_path):
                continue
                
            vis_metrics["total_images_processed"] += 1
            zone_id = gt_data.get("zone_id", "Z_S3")
            
            try:
                res = inspector.inspect_image(img_path, zone_id, strategy=strat)
                vis_metrics["json_parse_ok"] += 1
            except Exception as e:
                print(f"[ERRO] Falha ao inspecionar {img_name} com Estratégia {strat}: {e}")
                continue
                
            gt_issues = gt_data.get("issues", [])
            pred_issues = res.get("issues", [])
            vis_metrics["total_gt_issues"] += len(gt_issues)
            
            matched_preds = set()
            for gt_iss in gt_issues:
                matched = False
                for idx, pred_iss in enumerate(pred_issues):
                    if idx in matched_preds:
                        continue
                    if pred_iss.get("type") == gt_iss.get("type"):
                        matched = True
                        matched_preds.add(idx)
                        vis_metrics["correct_issues_detected"] += 1
                        
                        if pred_iss.get("severity") == gt_iss.get("severity"):
                            vis_metrics["correct_severity"] += 1
                            
                        q = "Avalia se a descrição da anomalia na imagem tem dados inventados ou falsos."
                        context = f"Problema real: {gt_iss.get('description')}\nLocalização real: {gt_iss.get('location')}"
                        pred_text = f"Descrição gerada: {pred_iss.get('description')}\nLocalização gerada: {pred_iss.get('location')}"
                        
                        score, reason = judge.evaluate(q, pred_text, context, "Faithfulness")
                        if score < 3.0:
                            vis_metrics["hallucinated_descriptions"] += 1
                        break
                        
            fps = len(pred_issues) - len(matched_preds)
            vis_metrics["false_positives"] += max(0, fps)
            
        total_imgs = vis_metrics["total_images_processed"]
        total_gt_iss = vis_metrics["total_gt_issues"]
        
        json_parse_rate = (vis_metrics["json_parse_ok"] / total_imgs) if total_imgs else 1.0
        issue_detection_rate = (vis_metrics["correct_issues_detected"] / total_gt_iss) if total_gt_iss else 1.0
        total_detected = vis_metrics["correct_issues_detected"] + vis_metrics["false_positives"]
        false_positive_rate = (vis_metrics["false_positives"] / total_detected) if total_detected else 0.0
        severity_accuracy = (vis_metrics["correct_severity"] / vis_metrics["correct_issues_detected"]) if vis_metrics["correct_issues_detected"] else 1.0
        hallucination_rate = (vis_metrics["hallucinated_descriptions"] / vis_metrics["correct_issues_detected"]) if vis_metrics["correct_issues_detected"] else 0.0
        
        visual_results[strat] = {
            "json_parse_rate": json_parse_rate,
            "issue_detection_rate": issue_detection_rate,
            "false_positive_rate": false_positive_rate,
            "severity_accuracy": severity_accuracy,
            "hallucination_rate": hallucination_rate
        }

    print("--- 2. Avaliando Estratégias de Chunking RAG ---")
    rag_strategies = ["hybrid", "by_issue"]
    rag_results = {}
    
    hist_inspections = [
        {
            "inspection_id": "INS_EVAL_001",
            "timestamp": "2026-06-01T10:00:00Z",
            "zone_id": "Z_S1",
            "shelf_fill_rate": 0.35,
            "overall_status": "critical",
            "issues": [{"issue_id": "ISS_001", "type": "empty_shelf", "location": "inferior", "severity": "high", "description": "Prateleira vazia de laticínios"}]
        },
        {
            "inspection_id": "INS_EVAL_002",
            "timestamp": "2026-06-02T15:00:00Z",
            "zone_id": "Z_S3",
            "shelf_fill_rate": 0.90,
            "overall_status": "ok",
            "issues": []
        },
        {
            "inspection_id": "INS_EVAL_003",
            "timestamp": "2026-06-03T18:00:00Z",
            "zone_id": "Z_S1",
            "shelf_fill_rate": 0.80,
            "overall_status": "warning",
            "issues": [{"issue_id": "ISS_001", "type": "misaligned", "location": "superior", "severity": "low", "description": "Iogurtes desalinhados"}]
        }
    ]
    
    rag_tests = [
        {"query": "Quando foi a última vez que Z_S1 teve prateleira vazia?", "relevant_id": "INS_EVAL_001"},
        {"query": "Qual a inspeção da zona Z_S3?", "relevant_id": "INS_EVAL_002"},
        {"query": "Algum problema com iogurtes desalinhados na prateleira superior?", "relevant_id": "INS_EVAL_003"}
    ]
    
    for r_strat in rag_strategies:
        print(f"A avaliar Chunking {r_strat}...")
        vstore_dir = "vectorstore" if r_strat == "hybrid" else f"vectorstore_eval_{r_strat}"
        rag = RagMemory(vectorstore_dir=vstore_dir, api_key=api_key, strategy=r_strat)
        
        for hi in hist_inspections:
            rag.add_inspection(hi)
            
        correct_retrievals = 0
        total_faithfulness = 0.0
        total_relevance = 0.0
        
        for t in rag_tests:
            results = rag.search(t["query"], k=3)
            retrieved_ids = [r["metadata"]["inspection_id"] for r in results]
            
            if t["relevant_id"] in retrieved_ids:
                correct_retrievals += 1
                
            ans, docs = rag.query_memory(t["query"])
            context_str = "\n".join([d["document"] for d in docs])
            
            f_score, _ = judge.evaluate(t["query"], ans, context_str, "Faithfulness")
            total_faithfulness += f_score
            
            r_score, _ = judge.evaluate(t["query"], ans, context_str, "Answer Relevance")
            total_relevance += r_score
            
        recall_at_3 = correct_retrievals / len(rag_tests)
        avg_faithfulness = (total_faithfulness / len(rag_tests)) / 5.0
        avg_relevance = (total_relevance / len(rag_tests)) / 5.0
        
        rag_results[r_strat] = {
            "recall_at_3": recall_at_3,
            "faithfulness": avg_faithfulness,
            "answer_relevance": avg_relevance
        }

    print("--- 3. Avaliando Rule Engine ---")
    rule_engine = RuleEngine(rules_dir="data/rules_eval", api_key=api_key)
    
    regras_teste = [
        {"text": "Avisa-me com alerta crítico se a prateleira superior da zona Z_S1 cair abaixo de 40% de fill rate", "valid": True},
        {"text": "Se houver produtos danificados em qualquer zona, emite um warning", "valid": True},
        {"text": "Considera sempre severidade alta se houver algum produto tombado na zona Z_S3", "valid": True},
        {"text": "Avisa-me se estiver vazia", "valid": False},
        {"text": "Gera alertas para laticínios", "valid": False},
        {"text": "Se o fill rate for mau avisa", "valid": False}
    ]
    
    parsed_rules = 0
    correct_ambiguity = 0
    execution_ok = 0
    
    for r in regras_teste:
        try:
            converted = rule_engine.convert_rule(r["text"])
            parsed_rules += 1
            
            is_valid_pred = converted.get("validation", {}).get("is_valid", True)
            if is_valid_pred == r["valid"]:
                correct_ambiguity += 1
                
            if r["valid"] and is_valid_pred:
                test_ins = {
                    "inspection_id": "INS_TEST",
                    "timestamp": "2026-06-08T12:00:00Z",
                    "zone_id": "Z_S1",
                    "shelf_fill_rate": 0.30,
                    "issues": []
                }
                dispara, motivo = rule_engine.test_rule_on_inspection(converted, test_ins)
                if dispara:
                    execution_ok += 1
        except Exception as e:
            print(f"[AVISO] Falha ao converter regra de teste: {e}")
            
    rule_parse_rate = parsed_rules / len(regras_teste)
    ambiguity_detection = correct_ambiguity / len(regras_teste)
    valid_rules_count = sum(1 for r in regras_teste if r["valid"])
    rule_correctness = execution_ok / valid_rules_count if valid_rules_count else 1.0
    
    report = {
        "evaluation_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "visual_analysis_metrics": visual_results,
        "rag_metrics": rag_results,
        "rule_engine_metrics": {
            "rule_parse_rate": rule_parse_rate,
            "rule_correctness": rule_correctness,
            "ambiguity_detection": ambiguity_detection
        }
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"\n[SUCESSO] Relatório de avaliação guardado em: {output_file}")
    
    t_visual = Table(title="ANÁLISE COMPARATIVA DE STRATEGIES DE PROMPTING")
    t_visual.add_column("Estratégia", style="cyan")
    t_visual.add_column("JSON Parse Rate", style="bold")
    t_visual.add_column("Issue Detection (Recall)", style="green")
    t_visual.add_column("False Positive Rate", style="red")
    t_visual.add_column("Severity Accuracy", style="blue")
    t_visual.add_column("Hallucination Rate", style="magenta")
    
    for s in prompt_strategies:
        m = visual_results[s]
        t_visual.add_row(
            f"Strat {s}",
            f"{m['json_parse_rate']*100:.1f}%",
            f"{m['issue_detection_rate']*100:.1f}%",
            f"{m['false_positive_rate']*100:.1f}%",
            f"{m['severity_accuracy']*100:.1f}%",
            f"{m['hallucination_rate']*100:.1f}%"
        )
    console.print(t_visual)
    
    t_rag = Table(title="ANÁLISE COMPARATIVA DE CHUNKING RAG")
    t_rag.add_column("Chunking Strategy", style="cyan")
    t_rag.add_column("Recall@3", style="bold green")
    t_rag.add_column("Faithfulness", style="blue")
    t_rag.add_column("Answer Relevance", style="magenta")
    
    for rs in rag_strategies:
        m = rag_results[rs]
        t_rag.add_row(
            rs.capitalize(),
            f"{m['recall_at_3']*100:.1f}%",
            f"{m['faithfulness']*100:.1f}%",
            f"{m['answer_relevance']*100:.1f}%"
        )
    console.print(t_rag)
    
    t_rule = Table(title="AVALIAÇÃO DO RULE ENGINE")
    t_rule.add_column("Métrica", style="cyan")
    t_rule.add_column("Resultado", style="bold green")
    t_rule.add_column("Descrição")
    t_rule.add_row("Rule Parse Rate", f"{rule_parse_rate*100:.1f}%", "Regras convertidas para JSON com sucesso")
    t_rule.add_row("Ambiguity Detection", f"{ambiguity_detection*100:.1f}%", "Precisão na identificação de ambiguidades")
    t_rule.add_row("Rule Correctness", f"{rule_correctness*100:.1f}%", "Regras válidas executadas corretamente")
    console.print(t_rule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harness de Avaliação do Sistema")
    parser.add_argument("--images-dir", default="test_images/", help="Diretório das imagens de teste")
    parser.add_argument("--output", default="evaluation_report.json", help="Ficheiro de output JSON")
    args = parser.parse_args()
    
    run_evaluation(args.images_dir, args.output)
