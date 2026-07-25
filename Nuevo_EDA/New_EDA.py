import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from google import genai
from google.genai import types
from groq import Groq

# ---------------------------------------------------------
# Configuración de la Página
# ---------------------------------------------------------
st.set_page_config(page_title="Text to EDA Dashboard", page_icon="📊", layout="wide")

st.title("📊 Text-to-Data & EDA Dashboard")
st.caption(
    "Extrae tablas estructuradas de texto no estructurado usando IA y realiza un EDA automático."
)

# ---------------------------------------------------------
# Barra Lateral - Configuración de Proveedor y API Key
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")

    provider = st.radio("Selecciona el Proveedor de IA:", ["Groq", "Google Gemini"])

    if provider == "Groq":
        api_key = st.text_input(
            "Groq API Key", type="password", help="Ingresa tu API Key de Groq Cloud"
        )
        selected_model = st.selectbox(
            "Modelo Groq",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        )
    else:
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Ingresa tu API Key de Google AI Studio",
        )
        selected_model = st.selectbox(
            "Modelo Gemini", ["gemini-2.5-flash", "gemini-2.5-pro"]
        )

    st.markdown("---")
    st.markdown("**¿Cómo funciona?**")
    st.markdown(
        "1. Ingresa tu texto con datos.\n2. El LLM extrae un JSON estructurado.\n3. Convertimos los datos a Pandas.\n4. Generamos gráficos e insights."
    )

# Texto de ejemplo por defecto
DEFAULT_TEXT = """
En el primer trimestre de 2024, la división de América del Norte alcanzó ventas por $450,000 con un costo operativo de $280,000 y 120 empleados. 
La división de Europa registró ventas de $320,000, costos de $210,000 y cuenta con 85 empleados. 
Por su parte, Asia-Pacífico logró $580,000 en ventas, con $340,000 en costos y 150 empleados. 
Finalmente, América Latina obtuvo $190,000 en ventas, $130,000 en costos y opera con 45 empleados.
"""

# ---------------------------------------------------------
# Área de Entrada de Texto
# ---------------------------------------------------------
st.subheader("1. Texto de Entrada")
input_text = st.text_area(
    "Párrafo con cifras o métricas:", value=DEFAULT_TEXT, height=150
)


# ---------------------------------------------------------
# Funciones para Gemini
# ---------------------------------------------------------
def call_gemini(
    text: str, prompt_system: str, api_key: str, model_name: str, is_json: bool = False
) -> str:
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(temperature=0.1)
    if is_json:
        config.response_mime_type = "application/json"

    full_prompt = f"{prompt_system}\n\nTexto/Datos:\n{text}"
    response = client.models.generate_content(
        model=model_name, contents=full_prompt, config=config
    )
    return response.text


# ---------------------------------------------------------
# Funciones para Groq
# ---------------------------------------------------------
def call_groq(
    text: str, prompt_system: str, api_key: str, model_name: str, is_json: bool = False
) -> str:
    client = Groq(api_key=api_key)

    kwargs = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": text},
        ],
        "temperature": 0.1,
    }

    if is_json:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


# ---------------------------------------------------------
# Función Unificada de Extracción
# ---------------------------------------------------------
def extract_data_with_llm(
    text: str, provider: str, api_key: str, model_name: str
) -> pd.DataFrame:
    system_prompt = """
    Analiza el texto recibido y extrae TODAS las entidades, métricas y cifras numéricas contenidas en él.
    Organízalas como un objeto JSON donde la clave principal sea "datos" y contenga una lista de objetos.
    Cada objeto de la lista representa una fila de una tabla comparable.
    Asegúrate de convertir las cifras numéricas a números puros (float o int), sin símbolos de moneda ni comas.
    Ejemplo de respuesta esperada:
    {
      "datos": [
        {"Entidad": "A", "Ventas": 100, "Empleados": 10},
        {"Entidad": "B", "Ventas": 200, "Empleados": 20}
      ]
    }
    """

    if provider == "Google Gemini":
        raw_response = call_gemini(
            text, system_prompt, api_key, model_name, is_json=True
        )
    else:
        raw_response = call_groq(text, system_prompt, api_key, model_name, is_json=True)

    data = json.loads(raw_response)

    # Manejar variaciones en la estructura devuelta por el JSON
    if isinstance(data, dict):
        if "datos" in data:
            data = data["datos"]
        else:
            first_key = list(data.keys())[0]
            if isinstance(data[first_key], list):
                data = data[first_key]

    return pd.DataFrame(data)


# ---------------------------------------------------------
# Función Unificada para Insights
# ---------------------------------------------------------
def generate_eda_insights(
    df: pd.DataFrame, provider: str, api_key: str, model_name: str
) -> str:
    system_prompt = "Eres un analista de datos experto. Revisa la tabla entregada y proporciona de 3 a 5 hallazgos clave o insights en formato markdown (bullet points)."
    data_csv = df.to_csv(index=False)

    if provider == "Google Gemini":
        return call_gemini(data_csv, system_prompt, api_key, model_name, is_json=False)
    else:
        return call_groq(data_csv, system_prompt, api_key, model_name, is_json=False)


# ---------------------------------------------------------
# Ejecución Principal
# ---------------------------------------------------------
if st.button("🚀 Extraer Datos y Analizar", type="primary"):
    if not api_key:
        st.error(f"⚠️ Por favor ingresa tu API Key de {provider} en la barra lateral.")
    elif not input_text.strip():
        st.warning("⚠️ El texto de entrada está vacío.")
    else:
        try:
            with st.spinner(f"Procesando con {provider} ({selected_model})..."):
                df = extract_data_with_llm(
                    input_text, provider, api_key, selected_model
                )
                st.session_state["df"] = df
                st.session_state["insights"] = generate_eda_insights(
                    df, provider, api_key, selected_model
                )

        except Exception as e:
            st.error(f"Ocurrió un error al procesar el texto: {e}")

# ---------------------------------------------------------
# Dashboard & EDA
# ---------------------------------------------------------
if "df" in st.session_state:
    df = st.session_state["df"]

    st.markdown("---")
    st.subheader("2. Tabla Extraída")

    col_table, col_stats = st.columns([3, 2])

    with col_table:
        st.dataframe(df, use_container_width=True)

    with col_stats:
        st.markdown("**Resumen Estadístico**")
        st.dataframe(df.describe(), use_container_width=True)

    # Métricas clave arriba
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if num_cols:
        st.markdown("---")
        st.subheader("3. Indicadores Clave (KPIs)")
        kpi_cols = st.columns(min(len(num_cols), 4))
        for idx, col in enumerate(num_cols[:4]):
            with kpi_cols[idx]:
                st.metric(label=f"Total {col}", value=f"{df[col].sum():,.2f}")

    st.markdown("---")
    st.subheader("4. Visualización de Datos (EDA)")

    if num_cols and cat_cols:
        sns.set_theme(style="whitegrid")
        c1, c2 = st.columns(2)

        with c1:
            x_axis = st.selectbox("Eje X (Categoría):", cat_cols, index=0)
            y_axis = st.selectbox("Eje Y (Métrica Principal):", num_cols, index=0)

            fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
            sns.barplot(
                data=df,
                x=x_axis,
                y=y_axis,
                estimator="sum",
                errorbar=None,
                palette="viridis",
                ax=ax_bar,
            )
            ax_bar.set_title(f"{y_axis} por {x_axis}")
            ax_bar.set_xlabel(x_axis)
            ax_bar.set_ylabel(y_axis)
            ax_bar.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            st.pyplot(fig_bar, use_container_width=True)

        with c2:
            if len(num_cols) >= 2:
                y_axis2 = st.selectbox(
                    "Eje Y (Segunda Métrica):",
                    num_cols,
                    index=min(1, len(num_cols) - 1),
                )

                fig_scatter, ax_scatter = plt.subplots(figsize=(8, 4))
                sns.scatterplot(
                    data=df,
                    x=y_axis,
                    y=y_axis2,
                    hue=x_axis if x_axis else None,
                    palette="deep",
                    ax=ax_scatter,
                )
                ax_scatter.set_title(f"Relación: {y_axis} vs {y_axis2}")
                ax_scatter.set_xlabel(y_axis)
                ax_scatter.set_ylabel(y_axis2)
                plt.tight_layout()
                st.pyplot(fig_scatter, use_container_width=True)
            else:
                fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
                sns.barplot(
                    data=df,
                    x=x_axis,
                    y=y_axis,
                    estimator="sum",
                    errorbar=None,
                    palette="viridis",
                    ax=ax_dist,
                )
                ax_dist.set_title(f"Distribución de {y_axis}")
                ax_dist.set_xlabel(x_axis)
                ax_dist.set_ylabel(y_axis)
                ax_dist.tick_params(axis="x", rotation=30)
                plt.tight_layout()
                st.pyplot(fig_dist, use_container_width=True)
    else:
        st.info(
            "No se encontraron suficientes columnas numéricas y categóricas combinadas para generar gráficos dinámicos."
        )

    st.markdown("---")
    st.subheader("5. Insights Automáticos")
    if "insights" in st.session_state:
        st.markdown(st.session_state["insights"])
