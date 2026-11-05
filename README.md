# 🎯 SmartAlloc AI: Agente de Otimização de Recursos e Orçamento

Ferramenta de engenharia de decisão que combina **Pesquisa Operacional (Programação Linear Inteira Mista)** e **IA Generativa (Google Gemini)** para automatizar a alocação ótima de capital, orçamento corporativo (FP&A) e recursos operacionais.

## 💡 O Problema de Negócio

Em áreas de Finanças Corporativas (FP&A), Planejamento Estratégico e Operações, gestores precisam distribuir um orçamento limitado entre múltiplos projetos concorrentes (expansão logística, automação, infraestrutura, marketing) buscando maximizar o Retorno sobre o Investimento (ROI). 

Tomar essa decisão manualmente em planilhas costuma gerar alocações subótimas ou violar travas importantes de governança (como limites de concentração de risco, custos mínimos de viabilidade por projeto e diversificação obrigatória do portfólio). Por outro lado, motores matemáticos puros são complexos de operar e interpretar por líderes de negócio.

## 🚀 A Solução (Arquitetura AI Builder)

O **SmartAlloc AI** une o rigor determinístico da matemática aplicada com a capacidade analítica de LLMs em um pipeline integrado:

1. **Interface Interativa de Parâmetros (`Streamlit`):** Permite ao gestor editar tabelas de projetos (ROI esperado, piso e teto de investimento, score de risco) e configurar travas globais de governança em tempo real.
2. **Motor de Pesquisa Operacional (`PuLP` - Solver CBC):** Formula e resolve dinamicamente um modelo de Programação Linear Inteira Mista (MILP), utilizando variáveis de decisão contínuas (capital alocado $x_i$) e binárias (seleção de projetos $y_i \in \{0,1\}$) para encontrar a distribuição matematicamente ótima.
3. **Agente de Auditoria e Parecer Executivo (`google-genai`):** Recebe a matriz de decisão calculada pelo solver e gera automaticamente um relatório executivo detalhando a eficiência alocativa, os gargalos de restrição (por que determinados recursos ficaram em caixa ou projetos foram cortados) e um plano tático de execução.

## 🧮 Formulação Matemática do Modelo

O motor de otimização maximiza o retorno financeiro total esperado sujeito às restrições orçamentárias e operacionais:

$$\max \sum_{i=1}^{n} x_i \cdot \text{ROI}_i$$

**Sujeito a:**
* **Restrição Orçamentária Global:** $\sum_{i=1}^{n} x_i \leq B_{\text{total}}$
* **Piso e Teto de Viabilidade (Vínculo Binário):** $C_{\min, i} \cdot y_i \leq x_i \leq \min(C_{\max, i}, \alpha \cdot B_{\text{total}}) \cdot y_i$
* **Diversificação Mínima do Portfólio:** $\sum_{i=1}^{n} y_i \geq K_{\min}$

*(Onde $x_i \geq 0$ é o capital alocado no projeto $i$, $y_i \in \{0,1\}$ é a variável binária de ativação do projeto, $B_{\text{total}}$ é o orçamento disponível, $\alpha$ é o limite máximo de concentração por iniciativa e $K_{\min}$ é o número mínimo de projetos aprovados).*

## 🛠️ Tecnologias Utilizadas

* **Python 3.10+**
* **PuLP:** Modelagem algébrica e resolução de problemas de Programação Linear e Inteira Mista (CBC Solver).
* **Google GenAI SDK (`gemini-2.5-flash`):** Engenharia de prompt estruturada (*Data-to-Text*) para tradução de resultados matemáticos em recomendações estratégicas de negócios.
* **Streamlit:** Construção da interface web interativa e edição dinâmica de dados.
* **Plotly Express & Pandas:** Manipulação matricial e visualização gráfica da distribuição de capital.

## ⚙️ Como Executar Localmente

1. Clone este repositório:
```bash
git clone [https://github.com/Kelvinrgit/smartalloc-ai.git](https://github.com/Kelvinrgit/smartalloc-ai.git)
cd smartalloc-ai
```

2. Crie um ambiente virtual e instale as dependências:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3. Execute a aplicação:
```bash
streamlit run app.py
```

4. Insira sua chave da API do Google Gemini (obtida gratuitamente no [Google AI Studio](https://aistudio.google.com/)) na barra lateral e clique em **Rodar Solver de Otimização & IA**.

## 👤 Autor

**Kelvin Rammon**  
Graduando em Ciências Atuariais — Universidade Federal da Paraíba (UFPB)  
Foco em Finanças Quantitativas, Modelagem de Risco, Pesquisa Operacional e Soluções com IA.