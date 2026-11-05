import streamlit as st
import pandas as pd
import numpy as np
import pulp
import json
from google import genai
import plotly.express as px

# Configuração da página
st.set_page_config(page_title="SmartAlloc AI | Otimizador de Orçamento", layout="wide")

st.title("🎯 SmartAlloc AI: Agente de Otimização de Recursos e Orçamento")
st.markdown("Ferramenta de engenharia de decisão que combina Pesquisa Operacional (Programação Linear) e IA Generativa para alocação ótima de capital e recursos corporativos.")

with st.sidebar:
    st.header("⚙️ Configurações do Modelo")
    api_key = st.text_input("Chave da API (Google Gemini):", type="password", help="Obtenha no Google AI Studio.")
    
    st.subheader("Parâmetros Financeiros")
    orcamento_total = st.number_input("Orçamento Total Disponível (R$):", min_value=1000.0, value=500000.0, step=10000.0)
    
    st.subheader("Restrições de Alocação")
    min_projetos = st.slider("Mínimo de projetos a financiar:", min_value=1, max_value=6, value=3)
    max_concentracao = st.slider("Limite máximo por projeto (% do total):", min_value=10, max_value=100, value=40) / 100.0

st.divider()

# Dados simulados de projetos (ex: iniciativas de FP&A, marketing, expansão ou tecnologia)
st.subheader("📋 Projetos Disponíveis para Avaliação")
dados_iniciais = pd.DataFrame({
    'Projeto': [
        'Expansão Logística Nordeste', 
        'Automação de Atendimento (IA)', 
        'Campanha de Aquisição Q3', 
        'Modernização de Infraestrutura Cloud', 
        'Pesquisa de Novos Modelos de Crédito', 
        'Otimização de Frota Urbana'
    ],
    'Retorno_Estimado_ROI': [0.22, 0.35, 0.15, 0.18, 0.28, 0.30], # Retorno percentual esperado
    'Custo_Minimo': [50000, 30000, 20000, 40000, 25000, 35000],
    'Custo_Maximo': [150000, 120000, 80000, 100000, 90000, 110000],
    'Risco_Operacional': [0.3, 0.5, 0.2, 0.1, 0.4, 0.45] # Escala de 0 a 1
})

# Editor interativo para o usuário ajustar os dados se quiser
df_projetos = st.data_editor(dados_iniciais, num_rows="dynamic", use_container_width=True)

executar_otimizacao = st.button("🚀 Rodar Solver de Otimización & IA", type="primary", use_container_width=True)

if executar_otimizacao:
    if not api_key:
        st.warning("Insira sua chave de API do Gemini na barra lateral para gerar a análise interpretativa.")
    else:
        with st.spinner("Executando motor de Programação Linear (PuLP) e sintetizando via IA..."):
            try:
                # --- MODELAGEM MATEMÁTICA COM PULP ---
                n = len(df_projetos)
                prob = LpProb = pulp.LpProblem("Otimizacao_Orcamento_Corporativo", pulp.LpMaximize)

                # Variáveis de decisão: Quanto alocar em cada projeto
                x = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=orcamento_total) for i in range(n)]
                # Variáveis binárias: Se o projeto é escolhido ou não (1 ou 0)
                y = [pulp.LpVariable(f"y_{i}", cat='Binary') for i in range(n)]

                # Função Objetivo: Maximizar o Retorno Total ponderado pelo ROI
                prob += pulp.lpSum([x[i] * df_projetos.loc[i, 'Retorno_Estimado_ROI'] for i in range(n)])

                # Restrição 1: Orçamento total não pode ser ultrapassado
                prob += pulp.lpSum([x[i] for i in range(n)]) <= orcamento_total

                # Restrições de Limites Individuais e Vínculo com Variável Binária
                for i in range(n):
                    c_min = df_projetos.loc[i, 'Custo_Minimo']
                    c_max = df_projetos.loc[i, 'Custo_Maximo']
                    max_proj_val = orcamento_total * max_concentracao
                    
                    # Se o projeto for escolhido (y_i = 1), o gasto deve estar entre o mínimo e o teto máximo permitido
                    prob += x[i] >= c_min * y[i]
                    prob += x[i] <= min(c_max, max_proj_val) * y[i]

                # Restrição 2: Número mínimo de projetos ativos
                prob += pulp.lpSum([y[i] for i in range(n)]) >= min_projetos

                # Resolvendo o problema
                prob.solve(pulp.PULP_CBC_CMD(msg=False))

                # Coletando resultados
                resultados = []
                orcamento_usado = 0
                retorno_total_calculado = 0

                for i in range(n):
                    alocacao = x[i].varValue if x[i].varValue is not None else 0
                    ativo = y[i].varValue > 0.5 if y[i].varValue is not None else False
                    roi = df_projetos.loc[i, 'Retorno_Estimado_ROI']
                    retorno_proj = alocacao * roi
                    
                    orcamento_usado += alocacao
                    retorno_total_calculado += retorno_proj
                    
                    resultados.append({
                        'Projeto': df_projetos.loc[i, 'Projeto'],
                        'Selecionado': "Sim" if ativo else "Não",
                        'Alocação (R$)': round(alocacao, 2),
                        'Retorno Esperado (R$)': round(retorno_proj, 2),
                        'ROI (%)': roi * 100
                    })

                df_resultado = pd.DataFrame(resultados)
                
                # --- EXIBIÇÃO DOS RESULTADOS ---
                st.divider()
                st.subheader("📊 Plano de Alocação Ótima Calculado")
                
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Orçamento Alocado", f"R$ {orcamento_usado:,.2f}", f"{(orcamento_usado/orcamento_total)*100:.1f}% do total")
                col_m2.metric("Retorno Financeiro Esperado", f"R$ {retorno_total_calculado:,.2f}")
                col_m3.metric("Saldo em Caixa (Não Alocado)", f"R$ {orcamento_total - orcamento_usado:,.2f}")

                st.dataframe(df_resultado, use_container_width=True)

                # Gráfico de Alocação
                fig = px.bar(df_resultado[df_resultado['Alocação (R$)'] > 0], x='Projeto', y='Alocação (R$)', 
                             title="Distribuição de Capital por Iniciativa Aprovada", color='ROI (%)', text_auto='.2s')
                st.plotly_chart(fig, use_container_width=True)

                # --- SÍNTESE COM IA (GEMINI) ---
                st.subheader("🤖 Parecer Estratégico & Auditoria de Decisão (IA)")
                
                client = genai.Client(api_key=api_key)
                prompt_ia = f"""
                Você é um Diretor de Finanças e Planejamento Estratégico sênior.
                Analise o resultado da otimização matemática via Programação Linear (PuLP) para a alocação de um orçamento de R$ {orcamento_total:,.2f}:

                DADOS DA ALOCAÇÃO ÓTIMA:
                - Orçamento Total Utilizado: R$ {orcamento_usado:,.2f}
                - Retorno Financeiro Ponderado Esperado: R$ {retorno_total_calculado:,.2f}
                - Tabela de Decisão do Solver:
                {df_resultado.to_string()}

                Estruture um relatório executivo objetivo contendo:
                1. **Resumo da Eficiência Alocativa**: Explique por que o solver distribuiu o capital dessa forma com base no ROI e nas restrições de teto/mínimo.
                2. **Análise de Oportunidade e Gargalos**: O que o saldo remanescente em caixa (ou a ausência dele) diz sobre as restrições impostas?
                3. **Recomendação de Execução**: 3 orientações práticas para a liderança executar esse plano de investimento com segurança mitigando riscos.
                """

                with st.spinner("Gerando parecer analítico com Gemini..."):
                    resposta_ia = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt_ia
                    )
                    st.markdown(resposta_ia.text)

            except Exception as e:
                st.error(fot:=f"Erro ao executar a otimização ou chamada de IA: {e}")