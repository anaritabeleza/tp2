import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def get_severity_rank(sev_value):
    """Converte a representação de severidade textual para um nível numérico comparável."""
    if not sev_value:
        return 0
    normalized = str(sev_value).lower()
    if normalized in ("low", "baixa", "baixo"):
        return 1
    if normalized in ("medium", "media", "médio", "média", "normal"):
        return 2
    if normalized in ("high", "critical", "alta", "alto", "crítico", "critica", "critico"):
        return 3
    return 0

def is_matching_location(loc_filter, issue_loc):
    """Verifica se a localização descrita no problema corresponde ao filtro de localização da regra."""
    if not loc_filter or str(loc_filter).lower() == "any":
        return True
    if not issue_loc:
        return False
        
    normalized_filter = loc_filter.lower()
    normalized_issue_loc = issue_loc.lower()
    
    # Mapeamento semântico simplificado de posições
    if normalized_filter == "bottom":
        return any(x in normalized_issue_loc for x in ("inferior", "bottom", "baixo", "chão"))
    if normalized_filter == "top":
        return any(x in normalized_issue_loc for x in ("superior", "top", "cima"))
    if normalized_filter == "middle":
        return any(x in normalized_issue_loc for x in ("intermédia", "intermedia", "meio", "middle"))
        
    return normalized_filter in normalized_issue_loc

class RuleEngine:
    """Motor de execução e conversão de regras de negócio em linguagem natural."""
    
    def __init__(self, rules_dir="data/rules", api_key=None, model_name="gemini-2.5-flash"):
        self.rules_directory = rules_dir
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_id = model_name
        self.rules = {}
        
        os.makedirs(self.rules_directory, exist_ok=True)
        self.load_rules()
        
    def load_rules(self):
        """Carrega todas as regras persistidas em disco."""
        self.rules = {}
        if not os.path.exists(self.rules_directory):
            return
            
        for file_name in os.listdir(self.rules_directory):
            if file_name.endswith(".json") and file_name.startswith("RULE_"):
                path = os.path.join(self.rules_directory, file_name)
                try:
                    with open(path, "r", encoding="utf-8") as stream:
                        content = json.load(stream)
                        rid = content.get("rule_id")
                        if rid:
                            self.rules[rid] = content
                except Exception as err:
                    print(f"[RULE_LOAD_WARN] Falha ao ler ficheiro de regra {file_name}: {err}")
        print(f"[RULE_ENGINE] {len(self.rules)} regras de negócio carregadas com sucesso.")

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

    def convert_rule(self, natural_language_rule):
        """Converte uma regra descrita em português natural para estrutura JSON via API Gemini."""
        if not self.api_key:
            raise ValueError("Credencial da API Gemini ausente no ambiente.")
            
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        
        prompt_template_path = "prompts/prompt_rule_engine.txt"
        if not os.path.exists(prompt_template_path):
            raise FileNotFoundError(f"Template do prompt do Rule Engine não encontrado em: {prompt_template_path}")
            
        with open(prompt_template_path, "r", encoding="utf-8") as stream:
            template = stream.read()
            
        rendered_prompt = template.replace("{REGRA}", natural_language_rule)
        
        raw_text = self.call_gemini_with_backoff(
            prompt=rendered_prompt,
            temperature=0.0,
            response_mime_type="application/json"
        ).strip()
        
        # Extração de blocos JSON
        json_search = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
        if json_search:
            raw_text = json_search.group(1)
        else:
            first_curly = raw_text.find('{')
            last_curly = raw_text.rfind('}')
            if first_curly != -1 and last_curly != -1:
                raw_text = raw_text[first_curly:last_curly+1]
                
        try:
            return json.loads(raw_text)
        except Exception as err:
            print(f"[CONVERSION_ERROR] Falha ao decodificar JSON retornado: {err}")
            print(f"Texto bruto:\n{response.text}")
            raise ValueError(f"O modelo falhou em estruturar a regra no formato JSON exigido: {err}")

    def add_rule(self, natural_language_rule):
        """Adiciona e persiste uma nova regra no sistema."""
        parsed_rule = self.convert_rule(natural_language_rule)
        
        val_status = parsed_rule.get("validation", {})
        is_valid = val_status.get("is_valid", True)
        detected_ambiguities = val_status.get("ambiguities", [])
        
        if not is_valid or detected_ambiguities:
            parsed_rule["rule_id"] = "PENDING"
            return parsed_rule, detected_ambiguities
            
        nums = [int(rid.split("_")[1]) for rid in self.rules.keys() if rid.startswith("RULE_") and rid.split("_")[1].isdigit()]
        next_id = max(nums) + 1 if nums else 1
        rid = f"RULE_{next_id:03d}"
        
        parsed_rule["rule_id"] = rid
        parsed_rule["created_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        parsed_rule["natural_language"] = natural_language_rule
        
        file_path = os.path.join(self.rules_directory, f"{rid}.json")
        with open(file_path, "w", encoding="utf-8") as stream:
            json.dump(parsed_rule, stream, indent=2, ensure_ascii=False)
            
        self.rules[rid] = parsed_rule
        print(f"[RULE_ENGINE] Nova regra de negócio persistida: {rid}")
        return parsed_rule, None

    def delete_rule(self, rule_id):
        """Elimina uma regra persistida no disco e em memória."""
        if rule_id not in self.rules:
            return False
            
        file_path = os.path.join(self.rules_directory, f"{rule_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
            
        del self.rules[rule_id]
        print(f"[RULE_ENGINE] Regra de negócio eliminada: {rule_id}")
        return True

    def test_rule_on_inspection(self, rule, inspection_data):
        """Avalia se uma inspeção cumpre os critérios para disparar uma determinada regra."""
        conditions = rule.get("conditions", {})
        
        # 1. Filtro de Zonas
        zone_list = conditions.get("zone_filter", [])
        if zone_list:
            filtered_zones = [z.lower() for z in zone_list if z.lower() not in ("any", "all", "")]
            if filtered_zones:
                current_zone = str(inspection_data.get("zone_id", "")).lower()
                if current_zone not in filtered_zones:
                    return False, f"Zona atual '{current_zone}' não abrangida pelo filtro {zone_list}"

        # 2. Filtro Horário
        time_limits = conditions.get("time_filter")
        if time_limits and isinstance(time_limits, dict):
            h_start = time_limits.get("hours_start")
            h_end = time_limits.get("hours_end")
            if h_start is not None and h_end is not None:
                timestamp = inspection_data.get("timestamp", "")
                try:
                    dt_obj = datetime.strptime(timestamp.split(".")[0].replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
                    current_hour = dt_obj.hour
                except Exception:
                    try:
                        dt_obj = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        current_hour = dt_obj.hour
                    except Exception:
                        current_hour = datetime.utcnow().hour
                
                if h_start <= h_end:
                    if not (h_start <= current_hour <= h_end):
                        return False, f"Hora atual {current_hour}h fora do intervalo {h_start}h-{h_end}h"
                else:
                    if not (current_hour >= h_start or current_hour <= h_end):
                        return False, f"Hora atual {current_hour}h fora do intervalo {h_start}h-{h_end}h"

        # 3. Limiar de Fill Rate
        fill_limit = conditions.get("fill_rate_threshold")
        current_fill = inspection_data.get("shelf_fill_rate", 1.0)
        
        if fill_limit is not None:
            if current_fill >= fill_limit:
                return False, f"Rácio de fill rate de {current_fill*100:.1f}% dentro dos limites seguros (Limiar: {fill_limit*100:.1f}%)"

        # 4. Filtro por Tipos de Problemas e Detalhes
        target_issues = conditions.get("issue_types", [])
        min_severity = conditions.get("severity_threshold")
        loc_filter = conditions.get("location_filter", "any")
        
        all_issues = inspection_data.get("issues", [])
        requires_issue_check = target_issues or min_severity or (loc_filter and loc_filter.lower() != "any")
        
        if requires_issue_check:
            if not all_issues:
                return False, "Nenhum incidente detetado na inspeção para validar as condições da regra."
                
            valid_incidents = []
            for item in all_issues:
                if target_issues:
                    mapped_types = [t.lower() for t in target_issues if t.lower() not in ("any", "all", "")]
                    if mapped_types and item.get("type", "").lower() not in mapped_types:
                        continue
                        
                if min_severity:
                    target_rank = get_severity_rank(min_severity)
                    current_rank = get_severity_rank(item.get("severity", "low"))
                    if current_rank < target_rank:
                        continue
                        
                if loc_filter and loc_filter.lower() != "any":
                    if not is_matching_location(loc_filter, item.get("location", "")):
                        continue
                        
                valid_incidents.append(item)
                
            if not valid_incidents:
                return False, f"Nenhum incidente satisfaz os critérios da regra (Tipos: {target_issues}, Mín. Severidade: {min_severity}, Posição: {loc_filter})"
            
            trigger_item = valid_incidents[0]
            explanation = f"Ocorrência detetada: '{trigger_item.get('type')}' na localização '{trigger_item.get('location')}' com severidade '{trigger_item.get('severity')}'"
            return True, explanation

        # Dispara por fill rate baixo se nenhum filtro de problemas foi solicitado
        if fill_limit is not None and current_fill < fill_limit:
            return True, f"Rácio de preenchimento de {current_fill*100:.1f}% abaixo do limiar crítico de {fill_limit*100:.1f}%"
            
        return False, "Nenhum parâmetro de ativação de regras satisfeito."

    def evaluate_inspection(self, inspection_data):
        """Avalia todas as regras de negócio ativas contra os dados de uma inspeção."""
        alerts = []
        execution_logs = []
        now_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
        execution_logs.append(f"[{now_ts}] === Início do Motor de Regras ===")
        execution_logs.append(f"Inspeção Alvo: {inspection_data.get('inspection_id')}")
        execution_logs.append(f"Metadados: Zona={inspection_data.get('zone_id')} | Rácio Fill={inspection_data.get('shelf_fill_rate', 1.0)*100:.1f}% | Incidentes={len(inspection_data.get('issues', []))}")
        
        for rid, rule_content in self.rules.items():
            try:
                triggered, explanation = self.test_rule_on_inspection(rule_content, inspection_data)
                execution_logs.append(f"  - Avaliar regra {rid}: '{rule_content.get('description')}'")
                
                if triggered:
                    execution_logs.append(f"    -> [ATIVADA] Causa: {explanation}")
                    
                    rule_action = rule_content.get("action", {})
                    lvl = rule_action.get("alert_level", "info")
                    msg_template = rule_action.get("notification_message", "Alerta para {rule_id}")
                    
                    fill_pct = int(inspection_data.get("shelf_fill_rate", 1.0) * 100)
                    ts_info = inspection_data.get("timestamp", "")
                    
                    formatted_msg = msg_template
                    tokens = {
                        "{zone_id}": inspection_data.get("zone_id", ""),
                        "{fill_rate}": str(fill_pct),
                        "{time}": ts_info.split("T")[-1].replace("Z", "") if "T" in ts_info else ts_info,
                        "{rule_id}": rid,
                        "{motivo}": explanation
                    }
                    for token, repl in tokens.items():
                        formatted_msg = formatted_msg.replace(token, repl)
                        
                    alert_payload = {
                        "rule_id": rid,
                        "alert_level": lvl,
                        "message": formatted_msg,
                        "triggered_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "inspection_id": inspection_data.get("inspection_id"),
                        "reason": explanation
                    }
                    alerts.append(alert_payload)
                else:
                    execution_logs.append(f"    -> [NÃO ATIVADA] {explanation}")
            except Exception as error:
                execution_logs.append(f"    -> [ERRO] Insucesso no processamento de {rid}: {error}")
                
        execution_logs.append(f"=== Fim do Processamento. Notificações geradas: {len(alerts)} ===\n")
        
        full_log = "\n".join(execution_logs)
        return alerts, full_log

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Consola de Teste do Motor de Regras")
    parser.add_argument("--add", help="Adicionar nova diretriz em linguagem natural")
    parser.add_argument("--evaluate", help="Caminho do ficheiro JSON de inspeção")
    args = parser.parse_args()
    
    engine = RuleEngine()
    if args.add:
        regra_criada, problemas = engine.add_rule(args.add)
        if problemas:
            print("Regra apresenta problemas ou ambiguidades! Detalhes:")
            for p in problemas:
                print(f"- {p}")
        else:
            print("Regra guardada:")
            print(json.dumps(regra_criada, indent=2, ensure_ascii=False))
            
    elif args.evaluate:
        if not os.path.exists(args.evaluate):
            print(f"Ficheiro de inspeção não encontrado: {args.evaluate}")
        else:
            with open(args.evaluate, "r", encoding="utf-8") as stream:
                dados_inspeccao = json.load(stream)
            alertas, log_completo = engine.evaluate_inspection(dados_inspeccao)
            print(log_completo)
            print("Alertas Gerados:")
            print(json.dumps(alertas, indent=2, ensure_ascii=False))
