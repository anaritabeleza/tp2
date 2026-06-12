import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class ReportGenerator:
    """Gerador de relatórios executivos estruturados sobre as inspeções de prateleiras da loja."""
    
    def __init__(self, api_key=None, model_name="gemini-2.5-flash"):
        self.key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_id = model_name

    def generate_report(self, inspections, notifications, rag_memory, trajectory_data=None):
        """Produz um relatório completo em formato Markdown contendo análise visual, regras e RAG."""
        num_zones = len(inspections)
        critical_zones = sum(1 for item in inspections if item.get("overall_status") == "critical")
        warning_zones = sum(1 for item in inspections if item.get("overall_status") == "warning")
        
        incidents = []
        for item in inspections:
            for iss in item.get("issues", []):
                incidents.append((item.get("zone_id"), iss))
                
        executive_summary = self._build_summary_executive(num_zones, critical_zones, warning_zones, incidents)
        zone_problems_md = self._build_zone_problems(inspections, rag_memory)
        triggered_rules_md = self._build_rules_triggered(notifications)
        historical_context_md = self._build_history_context(inspections, rag_memory)
        recommendations_md = self._build_recommendations_list(incidents, trajectory_data)
        trajectory_md = self._build_trajectory_correlation(inspections, trajectory_data)
        
        full_markdown = f"""# Relatório de Inspeção de Prateleiras - Retail Vision Intelligence System
**Data do Relatório:** {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}
**Sessão:** {datetime.utcnow().strftime("%Y%m%d_%H%M")}

## 1. Sumário Executivo
{executive_summary}

## 2. Problemas por Zona
{zone_problems_md}

## 3. Regras Disparadas
{triggered_rules_md}

## 4. Contexto Histórico Relevante
{historical_context_md}

## 5. Recomendações
{recommendations_md}

"""
        if trajectory_md:
            full_markdown += f"""## 6. Integração com Trajetória (Afluência de Clientes)
{trajectory_md}
"""
            
        return full_markdown

    def _build_summary_executive(self, total_zones, critical_count, warning_count, all_issues):
        """Gera a secção de sumário executivo, recorrendo à API do Gemini se disponível."""
        if not self.key:
            return (
                f"Inspeção concluída com sucesso num total de {total_zones} zonas analisadas. "
                f"Identificaram-se {critical_count} zonas em estado crítico e {warning_count} zonas com avisos pendentes. "
                f"Registaram-se no total {len(all_issues)} inconformidades operacionais. "
                f"Recomenda-se reposição ou reordenação imediata nas prateleiras afetadas."
            )
            
        import google.generativeai as genai
        genai.configure(api_key=self.key)
        
        summary_issues = [f"Zona {z}: {i.get('type')} ({i.get('severity')})" for z, i in all_issues]
        prompt = (
            f"Por favor, elabora um sumário executivo corporativo sintetizando as seguintes métricas de inspeção:\n"
            f"- Zonas analisadas: {total_zones}\n"
            f"- Casos críticos: {critical_count}\n"
            f"- Avisos operacionais: {warning_count}\n"
            f"- Anomalias totais: {len(all_issues)}\n"
            f"- Descrição das anomalias: {summary_issues}\n\n"
            f"REQUISITOS:\n"
            f"1. Estilo executivo, formal e de fácil leitura.\n"
            f"2. Máximo estrito de 150 palavras.\n"
            f"3. Redigido obrigatoriamente em português de Portugal (europeu).\n"
            f"Gera apenas o parágrafo de texto, sem títulos."
        )
        
        try:
            model = genai.GenerativeModel(self.model_id)
            res = model.generate_content(prompt, generation_config=genai.GenerationConfig(temperature=0.3))
            return res.text.strip()
        except Exception as err:
            print(f"[REPORT_WARN] Insucesso ao criar sumário executivo via LLM: {err}")
            return f"Inspeção terminada. Zonas: {total_zones}, Críticos: {critical_count}, Avisos: {warning_count}."

    def _build_zone_problems(self, inspections, rag_memory):
        """Cria o detalhe de anomalias agrupado por zona da loja."""
        md_lines = []
        
        for item in inspections:
            zid = item.get("zone_id")
            status = item.get("overall_status")
            fill = item.get("shelf_fill_rate", 1.0)
            issues = item.get("issues", [])
            
            emoji = "🔴" if status == "critical" else "🟡" if status == "warning" else "🟢"
            md_lines.append(f"### Zona {zid} {emoji}")
            md_lines.append(f"- **Estado da Zona:** `{status.upper()}`")
            md_lines.append(f"- **Ocupação de Prateleira (Fill Rate):** {fill*100:.1f}%")
            
            if not issues:
                md_lines.append("- **Anomalias:** Prateleiras em conformidade nesta inspeção.")
            else:
                md_lines.append("- **Inconformidades Operacionais:**")
                for iss in issues:
                    md_lines.append(
                        f"  - **[{iss.get('issue_id')}] {str(iss.get('type')).replace('_', ' ').title()}** "
                        f"({iss.get('severity').upper()}) na zona *{iss.get('location')}*. "
                        f"Detalhe: {iss.get('description')} (Precisão: {iss.get('confidence', 0.0)*100:.1f}%)"
                    )
            
            # Contexto histórico do RAG para esta zona
            try:
                query = f"Histórico de fill rate e problemas registados na zona {zid}."
                docs = rag_memory.search(query, k=2, zone_filter=zid)
                
                if docs:
                    md_lines.append("- **Métricas Anteriores Recuperadas (RAG):**")
                    for doc in docs:
                        meta = doc["metadata"]
                        md_lines.append(
                            f"  - Na inspeção *{meta.get('inspection_id')}* ({meta.get('timestamp')}), "
                            f"a prateleira tinha fill rate de {meta.get('fill_rate', 0.0)*100:.1f}% com estado `{meta.get('overall_status', 'ok').upper()}`."
                        )
                else:
                    md_lines.append("- **Métricas Anteriores Recuperadas (RAG):** Sem registos históricos anteriores para esta zona.")
            except Exception as err:
                md_lines.append(f"- **Métricas Anteriores Recuperadas (RAG):** Não foi possível obter histórico ({err})")
                
            md_lines.append("")
            
        return "\n".join(md_lines)

    def _build_rules_triggered(self, notifications):
        """Formata a tabela de regras de negócio que dispararam alertas."""
        if not notifications:
            return "Nenhum alarme ou regra automática foi acionada nesta ronda."
            
        md_lines = []
        md_lines.append("| Regra | Nível | Alerta Gerado | Justificação |")
        md_lines.append("| --- | --- | --- | --- |")
        for alert in notifications:
            md_lines.append(
                f"| `{alert.get('rule_id')}` | **{alert.get('alert_level').upper()}** | "
                f"{alert.get('message')} | {alert.get('reason')} |"
            )
        return "\n".join(md_lines)

    def _build_history_context(self, inspections, rag_memory):
        """Procura na memória RAG por comportamentos recorrentes de stock."""
        query = "Identificar recorrências de ruturas, problemas de stock e desorganização crítica em auditorias anteriores."
        try:
            answer, _ = rag_memory.query_memory(query, k=3)
            return answer
        except Exception as err:
            return f"[ERROR] Incapacidade de consultar vector store: {err}"

    def _build_recommendations_list(self, all_issues, trajectory_data=None):
        """Gera recomendações específicas com base nos problemas encontrados."""
        if not all_issues:
            return "1. **Manter rotina padrão:** Nenhum problema registado. Prosseguir com a monitorização contínua por imagem."
            
        if not self.key:
            md_lines = []
            for idx, (zone, iss) in enumerate(all_issues[:5]):
                md_lines.append(
                    f"{idx+1}. **Ação Corretiva na Zona {zone}:** "
                    f"Corrigir inconformidade do tipo '{iss.get('type')}' na prateleira. "
                    f"Relevância classificada como nível {iss.get('severity').upper()}."
                )
            return "\n".join(md_lines)
            
        import google.generativeai as genai
        genai.configure(api_key=self.key)
        
        list_summary = [f"Zona {z}: {i.get('type')} ({i.get('severity')}) na posição {i.get('location')}" for z, i in all_issues]
        
        prompt = (
            f"Com base na seguinte listagem de inconformidades operacionais observadas:\n"
            f"{json.dumps(list_summary, indent=2, ensure_ascii=False)}\n\n"
            f"Desenvolve até 5 ações corretivas em formato Markdown numerado (1., 2., etc.).\n"
            f"REQUISITOS:\n"
            f"1. Ordena por importância de intervenção (críticos primeiro).\n"
            f"2. Devem ser recomendações acionáveis, precisas e indicando explicitamente a zona e prateleira afetadas.\n"
            f"3. Redige em português europeu formal.\n"
            f"Retorna apenas a lista final em Markdown."
        )
        
        try:
            model = genai.GenerativeModel(self.model_id)
            res = model.generate_content(prompt, generation_config=genai.GenerationConfig(temperature=0.3))
            return res.text.strip()
        except Exception as err:
            print(f"[REPORT_WARN] Erro ao criar recomendações via LLM: {err}")
            return "1. Repor prateleiras vazias.\n2. Reorganizar SKUs desalinhados."

    def _build_trajectory_correlation(self, inspections, trajectory_data):
        """Efetua a correlação entre afluência de clientes e ruturas de prateleiras."""
        if not trajectory_data:
            return None
            
        md_lines = []
        md_lines.append("Análise cruzada entre as anomalias visuais detetadas e a dinâmica de fluxo de clientes:")
        
        for item in inspections:
            zid = item.get("zone_id")
            issues = item.get("issues", [])
            fill = item.get("shelf_fill_rate", 1.0)
            
            if zid in trajectory_data:
                data = trajectory_data[zid]
                diff = data.get("flow_rate_diff_pct", 0.0)
                
                if diff > 15.0:
                    status = f"**{diff:+.1f}% acima da média padrão** (movimento elevado)"
                elif diff < -15.0:
                    status = f"**{diff:+.1f}% abaixo da média padrão** (movimento calmo)"
                else:
                    status = f"estável em relação ao fluxo médio ({diff:+.1f}%)"
                    
                md_lines.append(f"- **Zona {zid}:** O fluxo de clientes está {status}.")
                
                empty_items = [iss for iss in issues if iss.get("type") == "empty_shelf"]
                if empty_items and diff > 20.0:
                    md_lines.append(
                        f"  * ⚠️ *Fator Procura:* A rutura de stock identificada (fill rate de {fill*100:.1f}%) coincide com "
                        f"um aumento substancial de afluência ({diff:.1f}%). Recomenda-se reabastecimento urgente pelo volume de vendas rápido."
                    )
                elif empty_items and diff < -20.0:
                    md_lines.append(
                        f"  * ℹ️ *Fator Operacional:* Rutura detetada na zona apesar do fluxo estar anormalmente calmo. "
                        f"Isto indicia possível rutura logística ou atraso no armazém interno, não correspondendo a pico de procura recente."
                    )
                    
        if len(md_lines) <= 1:
            return "Nenhuma correlação relevante observada na sessão."
            
        return "\n".join(md_lines)

if __name__ == "__main__":
    generator = ReportGenerator()
    print("Módulo ReportGenerator pronto.")
