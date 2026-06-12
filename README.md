# Retail Vision Intelligence System (RVIS)
**Trabalho Prático #2 - LIACD (Interação com Modelos de Grande Escala)**
**Ano Letivo 2025/2026**

Este projeto implementa a camada de inteligência visual para uma loja de retalho. O sistema recebe imagens de prateleiras, analisa-as com modelos de linguagem multimodais (Google Gemini 1.5 Flash), deteta problemas operacionais (como rutura de stock, desalinhamento, planograma incorreto, etc.) e permite que o gestor de loja defina regras em linguagem natural que são automaticamente executadas. O histórico de inspeções é indexado numa base de dados vetorial (ChromaDB), permitindo a recuperação semântica RAG para contextualizar análises futuras e gerar relatórios executivos estruturados.

---

## 📂 Estrutura do Repositório

```
tp2/
├── README.md                 # Este ficheiro com documentação geral
├── requirements.txt          # Dependências do projeto Python
├── .env.example              # Template de variáveis de ambiente
├── evaluate.py               # Harness de avaliação das métricas do sistema
├── data/
│   ├── images/               # Dataset de imagens de trabalho da loja (500 imagens)
│   ├── inspections/          # Registos JSON históricos de inspeções arquivados
│   └── rules/                # Configurações JSON das regras de negócio ativas
├── src/
│   ├── shelf_inspector.py    # Componente 1: análise visual com Gemini (3 estratégias)
│   ├── rule_engine.py        # Componente 2: conversão e execução de regras
│   ├── rag_memory.py         # Componente 3: base de dados vetorial RAG (ChromaDB)
│   ├── report_generator.py   # Componente 4: fusão de dados e relatórios Markdown
│   └── interface.py          # Componente 5: interface de Chat e consola CLI
├── prompts/
│   ├── prompt_strat_a.txt    # Prompt Estratégia A (Zero-shot)
│   ├── prompt_strat_b.txt    # Prompt Estratégia B (Chain-of-Thought visual)
│   ├── prompt_strat_c.txt    # Prompt Estratégia C (Few-shot)
│   ├── prompt_rule_engine.txt# Prompt do motor de regras (tradução e ambiguidades)
│   ├── prompt_rag_summary.txt # Prompt para resumos semânticos RAG
│   ├── prompt_rag_synthesis.txt # Prompt de síntese de resposta RAG
│   └── prompt_llm_judge.txt  # Prompt do avaliador LLM-as-judge
├── vectorstore/              # ChromaDB persistente com os embeddings locais
└── cache/                    # Cache local de assinaturas MD5 para evitar chamadas à API
```

---

## 🛠️ Configuração e Instalação

### Pré-requisitos
Certifique-se de que tem o Python instalado (recomendado >= 3.9).

1. **Criar e Ativar Ambiente Virtual:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # No Linux/macOS
   # ou
   .venv\Scripts\activate     # No Windows
   ```

2. **Instalar Dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurar as Variáveis de Ambiente:**
   Crie um ficheiro `.env` no diretório raiz copiado do `.env.example`:
   ```bash
   cp .env.example .env
   ```
   Edite o `.env` e preencha a sua chave de API Gemini obtida no Google AI Studio:
   ```env
   GEMINI_API_KEY=AIzaSyYourGeminiApiKeyHere
   ```
---

## 🚀 Como Utilizar a Interface do Gestor

O script principal de interface do utilizador localiza-se em `src/interface.py`. Pode ser executado em modo de subcomandos CLI ou como um loop de Chat interativo e dinâmico.

### Modo Chat Conversacional Interativo
Para iniciar a consola interativa premium (onde pode conversar em português, inspecionar prateleiras e adicionar regras de forma conversacional), execute sem argumentos:
```bash
python src/interface.py
```

### Modo CLI (Subcomandos Individuais)

#### 1. Modo de Inspeção Visual
Inspeciona uma imagem de uma zona específica:
```bash
python src/interface.py inspect Z_S3 --image test_images/shelf_empty_1.jpg
```
Ou inspeciona todas as imagens num diretório em lote:
```bash
python src/interface.py inspect all --images-dir test_images/
```

#### 2. Modo de Definição de Regras
Adiciona uma regra em linguagem natural (converte para JSON e guarda):
```bash
python src/interface.py add rule "Avisa-me se o fill rate cair abaixo de 70% na prateleira superior de qualquer zona"
```
*Se a regra contiver ambiguidades (como "avisa-me se estiver vazia"), o sistema responderá indicando o que necessita de ser esclarecido e não guardará a regra até ser corrigida.*

Listar todas as regras ativas na loja:
```bash
python src/interface.py list rules
```

Apagar uma regra específica:
```bash
python src/interface.py delete rule RULE_001
```

Testar uma regra contra uma imagem de teste para verificar se dispara:
```bash
python src/interface.py test rule RULE_001 --image test_images/shelf_empty_1.jpg
```

#### 3. Modo de Consulta Histórica (RAG Memory)
Faz uma pergunta livre em linguagem natural baseada na memória de inspeções passadas da loja:
```bash
python src/interface.py history "Quais foram as anomalias detetadas na zona Z_S1?"
```

Comparar a situação de duas zonas no período de tempo:
```bash
python src/interface.py compare Z_S1 Z_S3 --period "last 7 days"
```

#### 4. Modo de Relatório
Gera e salva em disco um relatório de inspeção Markdown profissional (`inspection_report_*.md`) fundindo a análise da sessão, regras ativas e contexto histórico RAG:
```bash
python src/interface.py report --session today
```
Ou filtre o relatório por zona específica:
```bash
python src/interface.py report --zone Z_S3 --period "last 14 days"
```

---

## 📊 Harness de Avaliação (`evaluate.py`)

O script de avaliação automatizado executa de forma independente e calcula todas as métricas obrigatórias da Secção 9.
Para executar a avaliação com base na diretoria de imagens de teste e gerar o ficheiro JSON de relatório:
```bash
python evaluate.py --images-dir test_images/ --output evaluation_report.json
```

O harness calcula e apresenta:
- **Análise Visual:** JSON Parse Rate, Issue Detection Rate (Recall), False Positive Rate, Severity Accuracy, Hallucination Rate.
- **RAG Memory:** Recall@3 (capacidade de recuperar registos corretos), Faithfulness (fidelidade aos dados), Answer Relevance (relevância da resposta).
- **Rule Engine:** Rule Parse Rate, Rule Correctness (execução lógica), Ambiguity Detection.
- **LLM-as-Judge:** Integrado para avaliar qualitativamente a fidelidade do RAG e alucinações nas descrições de imagem.

---

## 💡 Arquitetura e Decisões de Implementação

### 1. Estratégias de Prompting (Shelf Inspector)
- **Estratégia A (Zero-shot):** Instrução de conversão direta da imagem para JSON. Rápida, mas com maior taxa de erro e descrições menos ricas.
- **Estratégia B (Chain-of-Thought visual):** Força o Gemini a realizar um raciocínio passo a passo no campo `model_reasoning` antes de gerar a classificação. Descreve o geral, analisa prateleira por prateleira e localizações específicas, reduzindo erros de severidade e anomalias falsas.
- **Estratégia C (Few-shot):** Fornece exemplos textuais ricos de imagens anteriores e respetivos JSONs de output no próprio prompt para instruir o modelo sem carregar imagens adicionais na API, protegendo a quota.

### 2. RAG e Estratégia de Chunking
O RVIS suporta duas estratégias no `src/rag_memory.py`:
- **Hybrid (Recomendado):** O resumo de texto livre estruturado gerado pelo Gemini sobre a inspeção é indexado, associado a metadados estruturados (`zone_id`, `timestamp`, `fill_rate`, `overall_status`). Isto permite uma filtragem prévia (ex: isolar a zona consultada nos metadados) antes do retrieval semântico por proximidade de cosseno.
- **By Issue:** Cada problema detetado é indexado como um chunk isolado de texto com metadados do pai correspondente. Permite maior granularidade, ideal para a detecção de problemas específicos repetitivos (ex: "embalagens amassadas").

### 3. Gestão de Limites e Fallbacks da API
Para evitar custos e lidar com o limite gratuito de 15 chamadas/minuto:
- **Cache Local:** Toda a imagem inspecionada é convertida em MD5 e o seu resultado JSON é guardado em disco. Se a mesma imagem for analisada de novo, o resultado é instantaneamente carregado de `cache/`.
- **Rate Limiting:** Implementado um wrapper com backoff exponencial e jitter (aleatoriedade) para erros 429.
- **Fallback Gracioso:** Em caso de perda de internet ou limite diário de API excedido, o sistema lê resultados compatíveis da cache e alerta de forma amigável o utilizador sem interromper o fluxo da aplicação.

---

## 📝 Relatório Académico e Autoria

### Relatório em LaTeX (Overleaf)
A documentação científica oficial do projeto encontra-se redigida no ficheiro [`relatorio_tp2.tex`](./relatorio_tp2.tex) (localizado na raiz do repositório). Este ficheiro foi desenhado para ser colado e compilado diretamente no **Overleaf** e cumpre os seguintes requisitos:
*   **Formato:** Artigo de 2 colunas estilo IEEE (`IEEEtran`).
*   **Estética Avançada:** Utilização de caixas de destaque coloridas com cantos arredondados (`tcolorbox`), tabelas científicas com linhas alternadas sombreadas (`colortbl` / `booktabs`) e micro-tipografia avançada (`microtype`).
*   **Parsing Resiliente:** Mapeamento integrado de caracteres acentuados portugueses em listings (`literate`), prevenindo erros de compilação de codificação UTF-8 no Overleaf.

### Autoria
*   **Autora:** Rita Alves
*   **Instituição:** Universidade da Beira Interior (UBI), Covilhã, Portugal
*   **Curso:** Mestrado em Inteligência Artificial e Ciência de Dados (LIACD)
*   **E-mail:** [Rita.Alves@ubi.pt](mailto:Rita.Alves@ubi.pt)

