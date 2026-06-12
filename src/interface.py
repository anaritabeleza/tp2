import os
import sys
import json
import argparse
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich import print as rprint
    console = Console()
except ImportError:
    class SimpleConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = SimpleConsole()
    def rprint(*args, **kwargs):
        print(*args)

from src.shelf_inspector import ShelfInspector
from src.rule_engine import RuleEngine
from src.rag_memory import RagMemory
from src.report_generator import ReportGenerator

load_dotenv()

def print_welcome():
    """Imprime o cabeçalho inicial de boas-vindas do RVIS."""
    intro = """
    ============================================================
      RETAIL VISION INTELLIGENCE SYSTEM - INTERFACE DO GESTOR
    ============================================================
    Consola interativa para gestão de auditorias visuais e regras.
    
    Comandos CLI Disponíveis:
      * Análise Visual:
        - inspect <zona> --image <caminho_imagem>
        - inspect all --images-dir <diretorio_imagens>
      * Regras Operacionais:
        - add rule "<regras_em_linguagem_natural>"
        - list rules
        - delete rule <rule_id>
        - test rule <rule_id> --image <caminho_imagem>
      * Recuperação Semântica (RAG):
        - history "<pergunta_sobre_o_historico>"
        - compare <zona1> <zona2> --period "last 7 days"
      * Geração de Relatórios:
        - report --session today
        - report --zone <zona> --period "last 14 days"
        
    Executa sem subcomandos para iniciar o Modo Chat conversacional.
    ============================================================
    """
    if hasattr(console, "print") and "Panel" in globals():
        console.print(Panel(intro.strip(), title="SISTEMA INICIALIZADO", border_style="cyan"))
    else:
        print(intro)

class InteractiveInterface:
    """Interface de gestão da loja, lidando com CLI, Chat e integração de subsistemas."""
    
    def __init__(self, strategy="hybrid"):
        self.key_env = os.getenv("GEMINI_API_KEY")
        if not self.key_env:
            rprint("[yellow][ALERT][/yellow] GEMINI_API_KEY em falta. Operações offline ativadas.")
            
        self.inspector = ShelfInspector(api_key=self.key_env)
        self.rule_engine = RuleEngine(api_key=self.key_env)
        self.rag_memory = RagMemory(api_key=self.key_env, strategy=strategy)
        self.report_gen = ReportGenerator(api_key=self.key_env)

    def run_inspect(self, zone_id, image_path, strategy="B"):
        """Inspeciona uma prateleira, corre regras e atualiza a memória vetorial."""
        try:
            rprint(f"[cyan][STATUS] A analisar prateleira da zona {zone_id} ({image_path})...[/cyan]")
            res = self.inspector.inspect_image(image_path, zone_id, strategy=strategy)
            
            summary_info = (
                f"[bold]ID da Auditoria:[/bold] {res['inspection_id']}\n"
                f"[bold]Estado Geral:[/bold] {self._format_status_color(res['overall_status'])}\n"
                f"[bold]Taxa Fill Rate:[/bold] {res['shelf_fill_rate']*100:.1f}%\n"
                f"[bold]Produtos Identificados:[/bold] {', '.join(res['products_detected'])}\n"
                f"[bold]Raciocínio Operacional:[/bold]\n{res['model_reasoning']}"
            )
            rprint(Panel(summary_info, title="Resultados da Análise de Prateleira", border_style="green"))
            
            if res.get("issues"):
                rprint("[yellow]Inconformidades Operacionais Detetadas:[/yellow]")
                tbl = Table(show_header=True, header_style="bold magenta")
                tbl.add_column("Código", style="dim")
                tbl.add_column("Tipo de Falha")
                tbl.add_column("Localização")
                tbl.add_column("Gravidade")
                tbl.add_column("Observações")
                for item in res["issues"]:
                    tbl.add_row(item["issue_id"], item["type"], item["location"], item["severity"], item["description"])
                console.print(tbl)
            
            rprint("[cyan][STATUS] A processar regras de negócio...[/cyan]")
            notifications, logs = self.rule_engine.evaluate_inspection(res)
            rprint(logs)
            
            if notifications:
                rprint("[bold red]🚨 ALARMES OPERACIONAIS DISPARADOS:[/bold red]")
                for alert in notifications:
                    rprint(Panel(
                        f"[bold]Alarme:[/bold] {alert['rule_id']}\n"
                        f"[bold]Nível:[/bold] {alert['alert_level'].upper()}\n"
                        f"[bold]Mensagem:[/bold] {alert['message']}",
                        title="Disparo Crítico de Regra", border_style="red"
                    ))
            
            flow_data = self._get_mock_flow_data(zone_id)
            
            rprint("[cyan][STATUS] A indexar auditoria na Vector DB (RAG)...[/cyan]")
            self.rag_memory.add_inspection(res, trajectory_data=flow_data)
            rprint("[green][SUCCESS] Registo adicionado ao RAG com sucesso.[/green]")
            
            return res, notifications
        except Exception as error:
            rprint(f"[bold red][ERRO] Falha no fluxo de inspeção:[/bold red] {error}")
            return None, None

    def run_inspect_all(self, images_dir, strategy="B"):
        """Efetua a inspeção em lote de todas as imagens presentes num diretório."""
        if not os.path.exists(images_dir):
            rprint(f"[bold red][ERRO] Caminho não encontrado:[/bold red] {images_dir}")
            return
            
        valid_extensions = ('.png', '.jpg', '.jpeg')
        files = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_extensions)]
        if not files:
            rprint(f"[yellow][AVISO] Sem ficheiros de imagem válidos em {images_dir}[/yellow]")
            return
            
        rprint(f"[cyan]Ficheiros identificados para análise em lote: {len(files)}. A iniciar...[/cyan]")
        inspections = []
        all_notifs = []
        
        for idx, file in enumerate(files):
            rprint(f"\n[bold]Imagem {idx+1} de {len(files)}: {file}[/bold]")
            path = os.path.join(images_dir, file)
            match = re.search(r'\b(z_s[1-9])\b', file.lower())
            zid = match.group(1).upper() if match else "Z_S3"
            
            ins, notifs = self.run_inspect(zid, path, strategy=strategy)
            if ins:
                inspections.append(ins)
            if notifs:
                all_notifs.extend(notifs)
                
        rprint(f"\n[green][LOTE] Concluído. Processadas {len(inspections)} imagens com sucesso.[/green]")
        return inspections, all_notifs

    def run_add_rule(self, natural_language_rule):
        """Traduz, valida e adiciona uma nova regra operacional ao motor de regras."""
        try:
            rprint(f"[cyan][STATUS] A estruturar e validar a diretiva...[/cyan]")
            rule, ambiguities = self.rule_engine.add_rule(natural_language_rule)
            
            if ambiguities:
                rprint("[yellow][AMBIGUIDADE] Regra precisa de clarificação:[/yellow]")
                for idx, amb in enumerate(ambiguities):
                    rprint(f"  - {amb}")
                rprint("[yellow]Regra descartada. Escreve a regra de forma mais precisa e relança.[/yellow]")
                return False
            else:
                rule_summary = (
                    f"[bold]Regra ID:[/bold] {rule['rule_id']}\n"
                    f"[bold]Descrição Traduzida:[/bold] {rule['description']}\n"
                    f"[bold]Alerta:[/bold] {rule['action']['alert_level'].upper()}\n"
                    f"[bold]Condições de Validação:[/bold] {json.dumps(rule['conditions'], ensure_ascii=False)}"
                )
                rprint(Panel(rule_summary, title="Regra Registada no Sistema", border_style="green"))
                return True
        except Exception as error:
            rprint(f"[bold red][ERRO] Insucesso ao criar regra:[/bold red] {error}")
            return False

    def run_list_rules(self):
        """Mostra uma tabela com todas as regras operacionais guardadas no sistema."""
        rules = self.rule_engine.rules
        if not rules:
            rprint("[yellow]Sem regras configuradas no sistema.[/yellow]")
            return
            
        tbl = Table(title="Regras Ativas no RVIS")
        tbl.add_column("Código", style="bold cyan")
        tbl.add_column("Expressão Natural")
        tbl.add_column("Tradução Lógica")
        tbl.add_column("Alerta", style="magenta")
        tbl.add_column("Zonas Alvo", style="dim")
        
        for rid, item in rules.items():
            conds = item.get("conditions", {})
            zones = conds.get("zone_filter", [])
            zones_str = ", ".join(zones) if zones else "Todas"
            tbl.add_row(
                rid, 
                item.get("natural_language", "")[:40] + "...", 
                item.get("description", ""), 
                item.get("action", {}).get("alert_level", "info").upper(),
                zones_str
            )
        console.print(tbl)

    def run_delete_rule(self, rule_id):
        """Remove uma regra operacional do sistema."""
        if self.rule_engine.delete_rule(rule_id):
            rprint(f"[green][SUCCESS] Regra {rule_id} eliminada com sucesso.[/green]")
        else:
            rprint(f"[bold red][ERRO] Identificador de regra inválido: {rule_id}[/bold red]")

    def run_test_rule(self, rule_id, image_path):
        """Avalia se uma regra guardada seria despoletada por uma imagem específica."""
        if rule_id not in self.rule_engine.rules:
            rprint(f"[bold red][ERRO] Regra {rule_id} inexistente.[/bold red]")
            return
            
        try:
            rule = self.rule_engine.rules[rule_id]
            zid = "Z_S3"
            match = re.search(r'\b(z_s[1-9])\b', image_path.lower())
            if match:
                zid = match.group(1).upper()
                
            rprint(f"[cyan][TEST] A simular auditoria para teste...[/cyan]")
            ins_result = self.inspector.inspect_image(image_path, zid)
            
            rprint(f"[cyan][TEST] A avaliar comportamento de {rule_id}...[/cyan]")
            triggered, reason = self.rule_engine.test_rule_on_inspection(rule, ins_result)
            
            if triggered:
                rprint(f"[bold red]🚨 DISPARO CONFIRMADO![/bold red]")
                rprint(f"Razão: {reason}")
                action = rule.get("action", {})
                template = action.get("notification_message", "")
                formatted_message = template.replace('{zone_id}', zid).replace('{motivo}', reason)
                rprint(f"Texto formatado: [italic]{formatted_message}[/italic]")
            else:
                rprint(f"[green]Regra não dispararia nesta situação.[/green]")
                rprint(f"Razão: {reason}")
        except Exception as error:
            rprint(f"[bold red][ERRO] Falha no teste de regra:[/bold red] {error}")

    def run_history_query(self, query):
        """Efetua perguntas em linguagem natural e recolhe suporte da base de dados vetorial."""
        try:
            rprint(f"[cyan]A pesquisar memória semântica para: '{query}'...[/cyan]")
            answer, docs = self.rag_memory.query_memory(query)
            
            rprint(Panel(answer, title="Resultado RAG", border_style="blue"))
            
            if docs:
                rprint("[dim]Documentos históricos mais relevantes recuperados:[/dim]")
                for item in docs:
                    meta = item["metadata"]
                    rprint(f"  [dim]- [{meta.get('inspection_id')}] (Data: {meta.get('timestamp')}, Confiança: {item['score']*100:.1f}%): {item['document'][:120]}...[/dim]")
        except Exception as error:
            rprint(f"[bold red][ERRO] Erro na consulta de memória RAG:[/bold red] {error}")

    def run_compare(self, zone1, zone2, period):
        """Compara a afluência e conformidade histórica de duas zonas da loja."""
        try:
            q1 = f"histórico e problemas na zona {zone1}"
            q2 = f"histórico e problemas na zona {zone2}"
            
            docs1 = self.rag_memory.search(q1, k=5, zone_filter=zone1)
            docs2 = self.rag_memory.search(q2, k=5, zone_filter=zone2)
            
            if not docs1 and not docs2:
                rprint(f"[yellow]Histórico insuficiente para comparar {zone1} e {zone2}.[/yellow]")
                return
                
            status1 = docs1[0]['metadata'].get('overall_status','N/A').upper() if docs1 else 'N/A'
            status2 = docs2[0]['metadata'].get('overall_status','N/A').upper() if docs2 else 'N/A'
            fr1 = f"{docs1[0]['metadata'].get('fill_rate', 0.0)*100:.1f}%" if docs1 else 'N/A'
            fr2 = f"{docs2[0]['metadata'].get('fill_rate', 0.0)*100:.1f}%" if docs2 else 'N/A'
            
            rprint(Panel(
                f"[bold]Comparação entre a Zona {zone1} e Zona {zone2} ({period})[/bold]\n\n"
                f"[cyan]Zona {zone1}:[/cyan] Encontrados {len(docs1)} registos históricos.\n"
                f"  - Último estado: {status1}\n"
                f"  - Último fill rate: {fr1}\n\n"
                f"[cyan]Zona {zone2}:[/cyan] Encontrados {len(docs2)} registos históricos.\n"
                f"  - Último estado: {status2}\n"
                f"  - Último fill rate: {fr2}\n",
                title="Comparador de Zonas", border_style="cyan"
            ))
        except Exception as error:
            rprint(f"[bold red][ERRO] Erro na comparação de zonas:[/bold red] {error}")

    def run_report(self, session_arg=None, zone_id=None, period_arg=None):
        """Agrega os registos de inspeção e gera um relatório Markdown completo."""
        try:
            inspections_dir = "data/inspections"
            if not os.path.exists(inspections_dir) or not os.listdir(inspections_dir):
                rprint("[yellow]Sem dados de inspeções disponíveis para gerar relatórios.[/yellow]")
                return
                
            inspections = []
            for file in os.listdir(inspections_dir):
                if file.endswith(".json"):
                    with open(os.path.join(inspections_dir, file), "r", encoding="utf-8") as stream:
                        inspections.append(json.load(stream))
                        
            if not inspections:
                rprint("[yellow]Incapaz de carregar inspeções.[/yellow]")
                return
                
            if zone_id:
                inspections = [ins for ins in inspections if ins.get("zone_id") == zone_id]
                
            if not inspections:
                rprint(f"[yellow]Nenhum registo associado ao filtro selecionado (Zona: {zone_id}).[/yellow]")
                return
                
            all_notifications = []
            for ins in inspections:
                notifs, _ = self.rule_engine.evaluate_inspection(ins)
                all_notifications.extend(notifs)
                
            trajectory_data = {}
            for ins in inspections:
                z = ins.get("zone_id")
                trajectory_data[z] = self._get_mock_flow_data(z)
                
            rprint("[cyan]A estruturar o relatório final corporativo (RAG)...[/cyan]")
            report_content = self.report_gen.generate_report(inspections, all_notifications, self.rag_memory, trajectory_data)
            
            filename = f"inspection_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
            with open(filename, "w", encoding="utf-8") as stream:
                stream.write(report_content)
                
            rprint(f"[green][SUCCESS] Relatório guardado com sucesso em: {filename}[/green]")
            rprint(Markdown(report_content[:800] + "\n\n*(Visualização parcial. Relatório completo em ficheiro)*"))
        except Exception as error:
            rprint(f"[bold red][ERRO] Erro ao criar relatório Markdown:[/bold red] {error}")

    def run_chat_loop(self):
        """Executa a consola conversacional contínua para o gestor de loja."""
        print_welcome()
        rprint("[bold green]Consola Conversacional iniciada. Digita 'sair' ou 'exit' para fechar.[/bold green]")
        rprint("Questões válidas: 'Qual o estado de Z_S1?' ou 'Inspeciona Z_S3 com a imagem shelf.jpg'.\n")
        
        while True:
            try:
                user_input = console.input("[bold yellow]Gestor > [/bold yellow]").strip()
                if user_input.lower() in ["sair", "exit", "quit", "q"]:
                    rprint("[cyan]A encerrar a consola. Até breve![/cyan]")
                    break
                    
                if not user_input:
                    continue
                    
                inspect_match = re.search(r'(?:inspecciona|inspeciona|inspect|analisa|verificar)\s+(z_s[1-9])\s+(?:com a imagem|imagem|foto|file|img)?\s*([^\s]+)', user_input.lower())
                if inspect_match:
                    zone = inspect_match.group(1).upper()
                    img_path = inspect_match.group(2).strip("'\"")
                    if os.path.exists(img_path):
                        self.run_inspect(zone, img_path)
                    else:
                        rprint(f"[bold red]Imagem inexistente no caminho especificado:[/bold red] {img_path}")
                    continue
                    
                rule_match = re.search(r'(?:adiciona regra|cria regra|nova regra|criar regra|regra):?\s*(.+)', user_input, re.IGNORECASE)
                if rule_match:
                    rule_text = rule_match.group(1).strip()
                    self.run_add_rule(rule_text)
                    continue
                    
                if user_input.lower() in ["listar regras", "list rules", "regras", "rules"]:
                    self.run_list_rules()
                    continue
                    
                self.run_history_query(user_input)
                
            except KeyboardInterrupt:
                rprint("\n[cyan]Processo interrompido pelo utilizador. Adeus![/cyan]")
                break
            except Exception as error:
                rprint(f"[bold red][ERRO] Ocorreu uma exceção:[/bold red] {error}")

    def _format_status_color(self, status):
        """Aplica estilo Rich consoante a severidade do status da prateleira."""
        status = status.lower()
        if status == "ok":
            return "[green]OK[/green]"
        elif status == "warning":
            return "[yellow]WARNING[/yellow]"
        elif status == "critical":
            return "[red]CRITICAL[/red]"
        return f"[dim]{status}[/dim]"

    def _get_mock_flow_data(self, zone_id):
        """Simula a afluência de clientes nas zonas da loja para cruzamento de dados."""
        simulated = {
            "Z_S1": {"flow_rate_diff_pct": 38.5, "dwell_time_avg_seconds": 120},
            "Z_S2": {"flow_rate_diff_pct": -5.0, "dwell_time_avg_seconds": 45},
            "Z_S3": {"flow_rate_diff_pct": 42.1, "dwell_time_avg_seconds": 160},
            "Z_S4": {"flow_rate_diff_pct": -25.0, "dwell_time_avg_seconds": 30}
        }
        return simulated.get(zone_id, {"flow_rate_diff_pct": 5.0, "dwell_time_avg_seconds": 60})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interface Conversacional e CLI do Gestor de Loja")
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis")
    
    inspect_parser = subparsers.add_parser("inspect", help="Inspeciona uma prateleira ou diretório")
    inspect_parser.add_argument("zone", help="ID da zona (ex: Z_S3 ou 'all')")
    inspect_parser.add_argument("--image", help="Caminho para a imagem (obrigatório se zona for individual)")
    inspect_parser.add_argument("--images-dir", help="Diretório de imagens (obrigatório se zona for 'all')")
    inspect_parser.add_argument("--strategy", default="B", choices=["A", "B", "C"], help="Estratégia de prompting")
    
    add_parser = subparsers.add_parser("add", help="Adiciona uma nova regra")
    add_parser.add_argument("type", choices=["rule"], help="Tipo a adicionar")
    add_parser.add_argument("rule_text", help="Texto da regra em linguagem natural")
    
    subparsers.add_parser("list", help="Lista elementos do sistema").add_argument("type", choices=["rules"], help="O que listar")
    
    del_parser = subparsers.add_parser("delete", help="Apaga uma regra")
    del_parser.add_argument("type", choices=["rule"], help="Tipo a apagar")
    del_parser.add_argument("rule_id", help="ID da regra a apagar (ex: RULE_001)")
    
    test_parser = subparsers.add_parser("test", help="Testa uma regra numa imagem")
    test_parser.add_argument("type", choices=["rule"], help="Tipo a testar")
    test_parser.add_argument("rule_id", help="ID da regra")
    test_parser.add_argument("--image", required=True, help="Caminho para a imagem de teste")
    
    hist_parser = subparsers.add_parser("history", help="Consulta o histórico (RAG)")
    hist_parser.add_argument("query", help="Pergunta em linguagem natural à memória da loja")
    
    comp_parser = subparsers.add_parser("compare", help="Compara duas zonas")
    comp_parser.add_argument("zone1", help="Primeira zona")
    comp_parser.add_argument("zone2", help="Segunda zona")
    comp_parser.add_argument("--period", default="last 7 days", help="Período de tempo")
    
    rep_parser = subparsers.add_parser("report", help="Gera um relatório de inspeção")
    rep_parser.add_argument("--session", default="today", help="Sessão temporal")
    rep_parser.add_argument("--zone", help="Filtrar por zona específica")
    rep_parser.add_argument("--period", help="Período temporal")
    
    args = parser.parse_args()
    
    interface = InteractiveInterface()
    
    if not args.command:
        interface.run_chat_loop()
    else:
        if args.command == "inspect":
            if args.zone.lower() == "all":
                if not args.images_dir:
                    rprint("[bold red]Erro: --images-dir é obrigatório para inspect all[/bold red]")
                else:
                    interface.run_inspect_all(args.images_dir, strategy=args.strategy)
            else:
                if not args.image:
                    rprint("[bold red]Erro: --image é obrigatório para inspeção de zona individual[/bold red]")
                else:
                    interface.run_inspect(args.zone.upper(), args.image, strategy=args.strategy)
                    
        elif args.command == "add" and args.type == "rule":
            interface.run_add_rule(args.rule_text)
            
        elif args.command == "list" and args.type == "rules":
            interface.run_list_rules()
            
        elif args.command == "delete" and args.type == "rule":
            interface.run_delete_rule(args.rule_id.upper())
            
        elif args.command == "test" and args.type == "rule":
            interface.run_test_rule(args.rule_id.upper(), args.image)
            
        elif args.command == "history":
            interface.run_history_query(args.query)
            
        elif args.command == "compare":
            interface.run_compare(args.zone1.upper(), args.zone2.upper(), args.period)
            
        elif args.command == "report":
            interface.run_report(session_arg=args.session, zone_id=args.zone, period_arg=args.period)
