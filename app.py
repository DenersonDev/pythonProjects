import streamlit as st
import yfinance as yf
from datetime import date
import pandas as pd
import numpy as np

np.float_ = np.float64
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go

DATA_INICIO = "2024-01-01"
DATA_FIM = date.today().strftime("%Y-%m-%d")

dict_valores = dict()

st.title("Analise de ações")

# criandoa sidebar
st.sidebar.header("Escolha a Ação")
acao = st.sidebar.text_input("Entre com o valor da ação", value="")
variacao_entrada = float(st.sidebar.text_input("Entre com a variação de entrada", value="0"))
variacao_saida = float(st.sidebar.text_input("Entre com a variação de saida", value="0"))

n_dias = st.slider("Quantidade de dias da previsão", 30, 365)

@st.cache_data
def get_data(acao):
    df = yf.download(acao, DATA_INICIO, DATA_FIM, multi_level_index=False)
    df.reset_index(inplace=True)
    return df


def trata_dados(df, variacao_ent=0, variacao_saida=0):
    df2 = df.copy()
    df2["variacao_DA"] = round((df2["Close"].shift(1) / df2["Low"]) - 1, 5)
    df2["queda_rs"] = round((df2["Close"].shift(1) - df2["Low"]), 2)
    mediavar = df2["variacao_DA"].drop(index=0).median() - variacao_ent
    mediavlr = df2["queda_rs"].drop(index=0).median()
    df2["preco_entrada"] = df2["Close"].shift(1) - (df2["Close"].shift(1) * mediavar)
    df2["entrou"] = df2["preco_entrada"] >= df2["Low"]
    df2["lucro"] = df2["High"] - df2["preco_entrada"]
    df2["variacao_lucro"] = round((df2["High"] / df2["preco_entrada"]) - 1, 5)
    medialucro = df2["variacao_lucro"].drop(index=0).median() - variacao_saida
    df2["preco_saida"] = df2["preco_entrada"] + (df2["preco_entrada"] * medialucro)
    df2["saida"] = df2["preco_saida"] <= df2["High"]
    df2[(df2["entrou"] == True) & (df2["saida"] == True)]
    total_dias = df2["Date"].count()
    trades = df2["entrou"][df2["entrou"] == True].count()
    trades_lucro = df2["Date"][(df2["entrou"] == True) & (df2["saida"] == True)].count()
    df2["lucro_certo"] = df2["preco_saida"] - df2["preco_entrada"]
    df2["prejuizo_certo"] = df2["preco_entrada"] - df2["Close"]
    lucroTotal = df2["lucro_certo"][
        (df2["entrou"] == True) & (df2["saida"] == True)
    ].sum()
    prejuizoTotal = df2["prejuizo_certo"][
        (df2["entrou"] == True) & (df2["saida"] == False)
    ].sum()
    preco_entrada = df2['Close'].tail(1) - (df2['Close'].tail(1)*mediavar)
    
    dict_valores['mediavar'] = mediavar
    dict_valores['mediavlr'] = mediavlr
    dict_valores['medialucro'] = medialucro
    dict_valores['total_dias'] = total_dias
    dict_valores['trades'] = trades
    dict_valores['trades_lucro'] = trades_lucro
    dict_valores['lucroTotal'] = lucroTotal
    dict_valores['prejuizototal'] = prejuizoTotal
    dict_valores['final'] = lucroTotal - prejuizoTotal
    dict_valores['entrada'] = round(preco_entrada,2)
    dict_valores['saida'] = round(preco_entrada + (preco_entrada*medialucro))
    print(dict_valores['saida'])

if acao == "":
    st.write("nenhum valor encontrado para esta ação")
else:
    df_valores = get_data(acao.upper())
    trata_dados(df_valores, variacao_ent=variacao_entrada, variacao_saida=variacao_saida)
    
    st.text(f'variação média é => {round(dict_valores['mediavar'],5)}')
    st.write(f'valor médio da queda é => {round(dict_valores['mediavlr'],2)}')
    st.write(f'lucro médio é => {round(dict_valores['medialucro'],4)}')
    st.write(f'qntd de dias é => {round(dict_valores['total_dias'],2)}')
    st.write(f'total de trades é => {round(dict_valores['trades'],2)}')
    st.write(f'trades com lucor é => {round(dict_valores['trades_lucro'],2)}')
    st.write(f'total de lucro é => {round(dict_valores['lucroTotal'],2)}')
    st.write(f'total de prejuizo é => {round(dict_valores['prejuizototal'],2)}')
    st.write(f'final é => {round(dict_valores['final'],2)}')
    st.write(f'vlr da próxima entrada é => {round(dict_valores['entrada'],2)}')
    st.write(f'vlr da próxima saida é => {round(dict_valores['saida'],4)}')
    
    
    st.subheader(f"Tabela de Valores - {acao.upper()}")
    st.write(df_valores.tail(10))

    # Criar Gráfico de Previsão
    st.subheader(f"Gráficos de preços - {acao.upper()}")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_valores["Date"],
            y=df_valores["Close"],
            name="Precos de Fechamento",
            line_color="yellow",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_valores["Date"],
            y=df_valores["Open"],
            name="Precos de Abertura",
            line_color="blue",
        )
    )
    st.plotly_chart(fig)

    # previsão

    df_treino = df_valores[["Date", "Close"]]
    df_treino = df_treino.rename(columns={"Date": "ds", "Close": "y"})

    print(df_treino)

    modelo = Prophet()
    modelo.fit(df_treino)
    future = modelo.make_future_dataframe(periods=n_dias, freq="B")
    previsao = modelo.predict(future)

    st.subheader(f"Previsão de {n_dias} dias - {acao.upper()}")
    st.write(previsao[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(n_dias))

    grafico1 = plot_plotly(modelo, previsao)
    st.plotly_chart(grafico1)

    grafico2 = plot_components_plotly(modelo, previsao)
    st.plotly_chart(grafico2)
