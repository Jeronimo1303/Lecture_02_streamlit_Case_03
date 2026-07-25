import streamlit as st
import pandas as pd
import plotly.express as px
import json
from google import genai
from google.genai import types

# ---------------------------------------------------------
# Configuración de la Página
# ---------------------------------------------------------
st.set_page_config(page_title="Text to EDA Dashboard", page_icon="📊", layout="wide")

st.title("📊 Text-to-Data & EDA Dashboard")
st.caption(
    "Extrae tablas estructuradas de texto no estructurado usando IA y realiza un EDA automático."
)

# ---------------------------------------------------------
# Barra Lateral - Configuración de API Key
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    api_key = st.text_input(
        "Gemini API Key", type="password", help="Ingresa tu API Key de Google AI Studio"
    )
    selected_model = st.selectbox("Modelo", ["gemini-2.5-flash", "gemini-2.5-pro"])
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
    "Parrafo con cifras o métricas:", value=DEFAULT_TEXT, height=150
)


# ---------------------------------------------------------
# Función para Extracción de Datos
# ---------------------------------------------------------
def extract_data_with_llm(
    text: str, client: genai.Client, model_name: str
) -> pd.DataFrame:
    prompt = f"""
    Analiza el siguiente texto y extrae TODAS las entidades, métricas y cifras numéricas contenidas en él.
    Organízalas como una lista de objetos donde cada objeto representa una fila de una tabla comparable.
    Asegúrate de convertir las cifras numéricas a números puros (float o int), sin símbolos de moneda ni comas.

    Texto:
    {text}
    """

    # Forzamos respuesta en JSON estructurado
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json", temperature=0.1
        ),
    )

    data = json.loads(response.text)

    # Manejar posibles variaciones de estructura del JSON retornado
    if isinstance(data, dict):
        # Si el LLM envolvió la lista en una clave (ej: {"datos": [...]})
        first_key = list(data.keys())[0]
        data = data[first_key]

    return pd.DataFrame(data)


# ---------------------------------------------------------
# Función para Generar Insights
# ---------------------------------------------------------
def generate_eda_insights(
    df: pd.DataFrame, client: genai.Client, model_name: str
) -> str:
    prompt = f"""
    Eres un analista de datos experto. Revisa la siguiente tabla resumida y proporciona 3 a 5 hallazgos clave o insights relevantes en formato markdown (bullet points).
    
    Datos (CSV):
    {df.to_csv(index=False)}
    """

    response = client.models.generate_content(model=model_name, contents=prompt)
    return response.text


# ---------------------------------------------------------
# Ejecución Principal
# ---------------------------------------------------------
if st.button("🚀 Extraer Datos y Analizar", type="primary"):
    if not api_key:
        st.error("⚠️ Por favor ingresa tu API Key de Gemini en la barra lateral.")
    elif not input_text.strip():
        st.warning("⚠️ El texto de entrada está vacío.")
    else:
        try:
            client = genai.Client(api_key=api_key)

            with st.spinner("Procesando texto con el LLM y estructurando datos..."):
                df = extract_data_with_llm(input_text, client, selected_model)
                st.session_state["df"] = df
                st.session_state["insights"] = generate_eda_insights(
                    df, client, selected_model
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
        c1, c2 = st.columns(2)

        with c1:
            x_axis = st.selectbox("Eje X (Categoría):", cat_cols, index=0)
            y_axis = st.selectbox("Eje Y (Métrica Principal):", num_cols, index=0)
            fig_bar = px.bar(
                df,
                x=x_axis,
                y=y_axis,
                title=f"{y_axis} por {x_axis}",
                text_auto=True,
                template="plotly_white",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with c2:
            if len(num_cols) >= 2:
                y_axis2 = st.selectbox(
                    "Eje Y (Segunda Métrica):",
                    num_cols,
                    index=min(1, len(num_cols) - 1),
                )
                fig_scatter = px.scatter(
                    df,
                    x=y_axis,
                    y=y_axis2,
                    color=x_axis if x_axis else None,
                    title=f"Relación: {y_axis} vs {y_axis2}",
                    template="plotly_white",
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                fig_pie = px.pie(
                    df, names=x_axis, values=y_axis, title=f"Distribución de {y_axis}"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info(
            "No se encontraron suficientes columnas numéricas y categóricas combinadas para generar gráficos dinámicos."
        )

    st.markdown("---")
    st.subheader("5. Insights Automáticos")
    if "insights" in st.session_state:
        st.markdown(st.session_state["insights"])
