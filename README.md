
# CitizenTech

## 1. Sobre o projeto

O CitizenTech é um projeto acadêmico desenvolvido para a Global Solution da FIAP 2026.2.

O objetivo é desenvolver um modelo de Aprendizado de Máquina capaz de classificar registros de ocorrências envolvendo celulares subtraídos nas categorias Furto ou Roubo.

A solução utiliza uma rede neural Multilayer Perceptron (MLP) e uma aplicação web desenvolvida com Streamlit.

## 2. Fonte dos dados

Os dados utilizados foram obtidos no portal de estatísticas da Secretaria da Segurança Pública do Estado de São Paulo (SSP-SP).

Fonte: https://www.ssp.sp.gov.br/estatistica/consultas

A base utilizada contém registros de ocorrências envolvendo celulares subtraídos em 2026.

## 3. Preparação dos dados

Foram utilizadas as seguintes variáveis de entrada:

- BAIRRO
- DESCR_TIPOLOCAL
- DESCR_SUBTIPOLOCAL
- DIA_SEMANA
- MES

As variáveis categóricas foram codificadas com OneHotEncoder. As variáveis numéricas passaram por imputação de valores ausentes e padronização.

## 4. Modelo de classificação

Foram avaliadas três arquiteturas de redes neurais MLP:

- Compacta: 64 e 32 neurônios.
- Larga: 128 e 64 neurônios.
- Profunda: 64, 32 e 16 neurônios.

O modelo selecionado foi a arquitetura profunda.

## 5. Aplicação web

A aplicação foi desenvolvida com Streamlit e permite inserir características de uma ocorrência para obter uma classificação estimada entre Furto e Roubo.

Os resultados são experimentais e não representam previsão de crimes futuros.

## 6. Instalação e execução

Instale as dependências:

```bash
pip install -r requirements.txt
