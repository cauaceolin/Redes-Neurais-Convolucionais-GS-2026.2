
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

st.set_page_config(page_title="CitizenTech", page_icon="📱", layout="centered")

from pathlib import Path

PASTA_PROJETO = Path(__file__).parent

@st.cache_resource
def carregar_artefatos():
    preprocessador = joblib.load(
        PASTA_PROJETO / "preprocessador.joblib"
    )

    modelo = load_model(
        PASTA_PROJETO / "modelo_final.keras",
        compile=False
    )

    return preprocessador, modelo

preprocessador, modelo = carregar_artefatos()

encoder = (preprocessador.named_transformers_["categoricas"].named_steps["codificador"])

colunas_categoricas = ["BAIRRO", "DESCR_TIPOLOCAL", "DESCR_SUBTIPOLOCAL"]

categorias = {}

for i, coluna in enumerate(colunas_categoricas):
    valores = [
        str(valor)
        for valor in encoder.categories_[i]
        if str(valor) != "NAO_INFORMADO"
    ]
    categorias[coluna] = ["Não informado"] + sorted(valores)

st.title("📱 CitizenTech")
st.subheader("Classificação de ocorrências envolvendo celulares")

st.write("Informe as características de uma ocorrência " "para estimar sua classificação como Furto ou Roubo.")

with st.form("formulario_ocorrencia"):

    st.subheader("Características do local")

    bairro = st.selectbox("Bairro",categorias["BAIRRO"])

    tipo_local = st.selectbox("Tipo de local", categorias["DESCR_TIPOLOCAL"])

    subtipo_local = st.selectbox("Subtipo do local", categorias["DESCR_SUBTIPOLOCAL"])

    st.subheader("Características temporais")

    dias_semana = {"Segunda-feira": 0, "Terça-feira": 1,"Quarta-feira": 2, "Quinta-feira": 3, "Sexta-feira": 4, "Sábado": 5, "Domingo": 6}

    dia = st.selectbox("Dia da semana",list(dias_semana.keys()))

    mes = st.selectbox("Mês", list(range(1, 13)))

    enviar = st.form_submit_button("Classificar ocorrência",use_container_width=True)

if enviar:

    entrada = pd.DataFrame([{
        "BAIRRO": np.nan if bairro == "Não informado" else bairro,
        "DESCR_TIPOLOCAL": np.nan if tipo_local == "Não informado" else tipo_local,
        "DESCR_SUBTIPOLOCAL": np.nan if subtipo_local == "Não informado" else subtipo_local,
        "DIA_SEMANA": dias_semana[dia],
        "MES": mes}])

    dados_processados = preprocessador.transform(entrada)

    if hasattr(dados_processados, "toarray"):
        dados_processados = dados_processados.toarray()

    dados_processados = np.asarray(
        dados_processados,
        dtype=np.float32
    )

    if dados_processados.shape[1] != modelo.input_shape[-1]:
        st.error("Quantidade de atributos incompatível com o modelo.")
        st.stop()

    probabilidade_roubo = float(
        modelo.predict(dados_processados, verbose=0)[0][0]
    )

    probabilidade_furto = 1 - probabilidade_roubo

    resultado = (
        "Roubo" if probabilidade_roubo >= 0.5 else "Furto"
    )

    st.divider()
    st.subheader("Resultado da classificação")

    st.success(f"Classe estimada: {resultado}")

    st.write(f"**Furto:** {probabilidade_furto * 100:.2f}%")
    st.progress(probabilidade_furto)

    st.write(f"**Roubo:** {probabilidade_roubo * 100:.2f}%")
    st.progress(probabilidade_roubo)

    st.caption(
        "Ferramenta experimental de classificação de registros. "
        "Os scores não representam a probabilidade de um crime futuro."
    )
