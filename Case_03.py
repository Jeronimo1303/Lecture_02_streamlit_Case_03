import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from groq import Groq

# ==========================================
# CONFIGURACIÓN DE PÁGINA Y TEMA DE COLOR
# ==========================================
st.set_page_config(
    page_title="EDA Agro Colombia",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos CSS personalizados para Paleta Clara y Legibilidad
st.markdown(
    """
    <style>
    /* Fondo general claro */
    .main {
        background-color: #F8FAF8;
        color: #1F2937;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Encabezados y títulos */
    h1, h2, h3 {
        color: #111827;
        font-weight: 700;
    }
    
    /* Tarjetas de Métricas */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #065F46;
        font-weight: bold;
    }
    
    /* Cajas de Insights / Storytelling */
    .insight-box {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 20px;
        color: #064E3B;
    }
    
    .alert-box {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 20px;
        color: #78350F;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Configuración de estilo global para Seaborn
sns.set_theme(style="whitegrid")
PALETA_SEABORN = ["#10B981", "#3B82F6", "#F59E0B", "#EF4444", "#8B5CF6"]


# ==========================================
# CARGA DE DATOS CON CACHÉ
# ==========================================
@st.cache_data
def cargar_datos():
    try:
        df = pd.read_csv("agro_colombia.csv")
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        np.random.seed(42)
        n = 500
        departamentos = [
            "Antioquia",
            "Cundinamarca",
            "Valle del Cauca",
            "Tolima",
            "Boyacá",
        ]
        cultivos = ["Café", "Aguacate", "Maíz", "Arroz", "Plátano"]
        riego = np.random.choice(["Con Riego", "Sin Riego"], size=n, p=[0.4, 0.6])

        base_prod = np.random.gamma(shape=3, scale=10, size=n)
        produccion = np.where(riego == "Con Riego", base_prod * 1.85, base_prod)
        area = np.random.uniform(5, 100, size=n)
        rendimiento = produccion / area

        df_dummy = pd.DataFrame(
            {
                "Departamento": np.random.choice(departamentos, size=n),
                "Cultivo": np.random.choice(cultivos, size=n),
                "Sistema_Riego": riego,
                "Area_Hectareas": np.round(area, 2),
                "Produccion_Toneladas": np.round(produccion, 2),
                "Rendimiento_Ton_Ha": np.round(rendimiento, 2),
                "Año": np.random.choice([2021, 2022, 2023], size=n),
            }
        )
        return df_dummy


df = cargar_datos()

col_riego = next((c for c in df.columns if "riego" in c.lower()), None)
col_prod = next((c for c in df.columns if "prod" in c.lower()), None)


def normalizar_riego(series: pd.Series) -> pd.Series:
    valores = series.astype(str).str.strip().str.lower()
    mapa = {
        "true": "Con Riego",
        "1": "Con Riego",
        "si": "Con Riego",
        "yes": "Con Riego",
        "t": "Con Riego",
        "con riego": "Con Riego",
        "conriego": "Con Riego",
        "false": "Sin Riego",
        "0": "Sin Riego",
        "no": "Sin Riego",
        "f": "Sin Riego",
        "sin riego": "Sin Riego",
        "sinriego": "Sin Riego",
    }
    resultado = valores.map(mapa)
    resultado = resultado.fillna(valores)
    resultado = resultado.replace({"con riego": "Con Riego", "sin riego": "Sin Riego"})
    resultado = resultado.replace({"true": "Con Riego", "false": "Sin Riego"})
    return resultado.where(resultado.isin(["Con Riego", "Sin Riego"]), "Sin Riego")


if col_riego is not None:
    df["Riego_Categorico"] = normalizar_riego(df[col_riego])
else:
    df["Riego_Categorico"] = pd.Series(["Sin Riego"] * len(df), index=df.index)

# ==========================================
# MENÚ NAVEGACIÓN (SIDEBAR)
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2908/2908122.png", width=70)
st.sidebar.title("📌 Menú Principal")
st.sidebar.markdown("---")

seccion = st.sidebar.radio(
    "Seleccione una sección:",
    [
        "🏠 Resumen Cualitativo",
        "📊 Análisis Cuantitativo",
        "⚖️ Riego vs. Producción (Boxplot)",
        "🎨 Galería de Gráficos (Plotly & Seaborn)",
        "🤖 Asistente de Interpretación de Datos",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔑 Configuración de API")
groq_api_key = st.sidebar.text_input(
    "Ingresa tu Groq API Key:",
    type="password",
    help="Consigue tu API key en https://console.groq.com/",
)

st.sidebar.markdown("---")
st.sidebar.caption("🌾 **EDA Agro Colombia** | Proyecto de Storytelling de Datos")

# ==========================================
# SECCIÓN 1: RESUMEN CUALITATIVO
# ==========================================
if seccion == "🏠 Resumen Cualitativo":
    st.title("🌾 Análisis Exploratorio del Sector Agrícola")
    st.markdown("### Visión General del Dataset")

    st.markdown(
        """
    <div class="insight-box">
    <b>💡 Mensaje Clave del Negocio:</b><br>
    Este panel evalúa la estructura de los datos recopilados en el sector agropecuario. 
    Permite identificar variables categóricas clave (regiones, tipos de cultivo, acceso a infraestructura) 
    y verificar la integridad cualitativa de la muestra recolectada.
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Registros", f"{len(df):,}")
    with col2:
        st.metric("Total Variables", f"{df.shape[1]}")
    with col3:
        cols_cat = df.select_dtypes(include=["object", "category"]).shape[1]
        st.metric("Variables Categóricas", f"{cols_cat}")
    with col4:
        cols_num = df.select_dtypes(include=["number"]).shape[1]
        st.metric("Variables Numéricas", f"{cols_num}")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📋 Muestra de Datos")
        st.dataframe(df.head(10), use_container_width=True)

    with col_right:
        st.subheader("🛠️ Tipos de Datos y Valores Faltantes")
        df_info = pd.DataFrame(
            {
                "Tipo de Dato": df.dtypes.astype(str),
                "Valores Nulos": df.isnull().sum(),
                "Completitud (%)": np.round((1 - df.isnull().sum() / len(df)) * 100, 2),
            }
        )
        st.dataframe(df_info, use_container_width=True)

    st.markdown("---")
    st.subheader("🏷️ Distribución Categórica Principal")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        col_sel = st.selectbox(
            "Seleccione variable categórica para analizar:", cat_cols
        )

        fig_cat = px.bar(
            df[col_sel].value_counts().reset_index(),
            x="count",
            y=col_sel,
            orientation="h",
            title=f"Distribución por {col_sel}",
            color_discrete_sequence=["#10B981"],
            text_auto=True,
        )
        fig_cat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Frecuencia",
            yaxis_title=col_sel,
            font=dict(color="#1F2937"),
        )
        st.plotly_chart(fig_cat, use_container_width=True)

# ==========================================
# SECCIÓN 2: ANÁLISIS CUANTITATIVO
# ==========================================
elif seccion == "📊 Análisis Cuantitativo":
    st.title("📊 Análisis Cuantitativo y Descriptivo")

    st.markdown(
        """
    <div class="insight-box">
    <b>📈 Hallazgo Estadístico:</b><br>
    Las variables de producción y rendimiento suelen presentar sesgos hacia la derecha debido a la concentración de fincas de gran escala. Revisa las desviaciones estándar e intercuartílicas para entender la dispersión real.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.subheader("📐 Estadísticas Descriptivas Globales")
    num_df = df.select_dtypes(include=["number"])
    st.dataframe(
        num_df.describe().T.style.background_gradient(cmap="Greens"),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("🔥 Matriz de Correlación Numérica")

    if len(num_df.columns) > 1:
        corr = num_df.corr()

        fig_corr, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            ax=ax,
            cbar=True,
            linewidths=0.5,
        )
        ax.set_title(
            "Matriz de Correlación de Pearson",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        st.pyplot(fig_corr)
    else:
        st.warning(
            "Se requieren al menos 2 variables numéricas para el análisis de"
            " correlación."
        )

# ==========================================
# SECCIÓN 3: BOXPLOT RIEGO VS PRODUCCIÓN
# ==========================================
elif seccion == "⚖️ Riego vs. Producción (Boxplot)":
    st.title("⚖️ Impacto de la Infraestructura: Riego vs. Producción")

    st.markdown(
        """
    <div class="insight-box">
    <b>🎯 Mensaje Central de Storytelling:</b><br>
    La implementación de <b>Sistemas de Riego</b> disminuye la variabilidad del rendimiento y eleva significativamente la mediana de producción en comparación con fincas sin infraestructura hídrica.
    </div>
    """,
        unsafe_allow_html=True,
    )

    if col_riego and col_prod:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### 🔵 Distribución Interactiva (Plotly)")
            fig_box_plotly = px.box(
                df,
                x="Riego_Categorico",
                y=col_prod,
                color="Riego_Categorico",
                color_discrete_map={"Con Riego": "#10B981", "Sin Riego": "#F59E0B"},
                points="outliers",
                title="Producción por Condición de Riego",
            )
            fig_box_plotly.update_layout(
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#F8FAF8",
                showlegend=False,
                xaxis_title="Sistema de Riego",
                yaxis_title="Producción",
                font=dict(color="#111827"),
            )
            st.plotly_chart(fig_box_plotly, use_container_width=True)
            st.caption(
                "🔍 **Interactividad:** Pasa el cursor para inspeccionar la mediana,"
                " Q1, Q3 y valores atípicos."
            )

        with col_b:
            st.markdown("### 🟢 Comparación Formal (Seaborn)")
            fig_sns, ax = plt.subplots(figsize=(6, 4.5))

            sns.boxplot(
                data=df,
                x="Riego_Categorico",
                y=col_prod,
                palette={"Con Riego": "#10B981", "Sin Riego": "#F59E0B"},
                ax=ax,
                width=0.4,
                boxprops=dict(alpha=0.85),
            )
            sns.stripplot(
                data=df,
                x="Riego_Categorico",
                y=col_prod,
                color="black",
                alpha=0.2,
                jitter=0.2,
                size=3,
                ax=ax,
            )

            ax.set_title(
                "Variabilidad de Producción según Acceso a Riego",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_xlabel("Presencia de Sistema de Riego", fontsize=10)
            ax.set_ylabel("Producción", fontsize=10)
            sns.despine(top=True, right=True)

            st.pyplot(fig_sns)
            st.caption(
                "📌 **Lectura:** Los puntos negros muestran la dispersión individual"
                " de las fincas observadas."
            )

        st.markdown("---")
        st.subheader("📊 Comparativo Cuantitativo de Rendimiento")

        stats_riego = (
            df.groupby("Riego_Categorico")[col_prod]
            .agg(
                Mediana="median",
                Promedio="mean",
                Desviacion_Estandar="std",
                Maximo="max",
            )
            .reset_index()
        )

        st.dataframe(
            stats_riego.style.highlight_max(axis=0, color="#D1FAE5"),
            use_container_width=True,
        )

    else:
        st.error(
            "No se detectaron automáticamente las columnas de Riego o Producción."
            f" Columnas disponibles: {list(df.columns)}"
        )

# ==========================================
# SECCIÓN 4: GALERÍA DE GRÁFICOS (STORYTELLING)
# ==========================================
elif seccion == "🎨 Galería de Gráficos (Plotly & Seaborn)":
    st.title("🎨 Visualización Avanzada con Storytelling")
    st.markdown(
        "Combinación de herramientas dinámicas (**Plotly**) e imprimibles"
        " (**Seaborn**) bajo una paleta clara y coherente."
    )

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    tab1, tab2 = st.tabs(
        [
            "📊 Visualizaciones Plotly (Dinámicas)",
            "🎨 Visualizaciones Seaborn (Estáticas)",
        ]
    )

    with tab1:
        st.markdown("### Visualización Interactiva")
        if len(num_cols) >= 2:
            var_x = st.selectbox("Eje X:", num_cols, index=0)
            var_y = st.selectbox("Eje Y:", num_cols, index=min(1, len(num_cols) - 1))
            col_color = st.selectbox("Agrupar por Color:", [None] + cat_cols)

            fig_scatter = px.scatter(
                df,
                x=var_x,
                y=var_y,
                color=col_color,
                color_discrete_sequence=[
                    "#10B981",
                    "#3B82F6",
                    "#F59E0B",
                    "#EF4444",
                ],
                title=f"Relación entre {var_x} y {var_y}",
            )
            fig_scatter.update_layout(
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#F8FAF8",
                font=dict(color="#111827"),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    with tab2:
        st.markdown("### Visualización Estática para Reportes")
        if num_cols:
            var_dist = st.selectbox(
                "Seleccione Variable Numérica para Histograma:", num_cols
            )

            fig_hist, ax = plt.subplots(figsize=(8, 3.5))
            sns.histplot(
                df[var_dist],
                kde=True,
                color="#0284C7",
                ax=ax,
                bins=20,
                line_kws={"linewidth": 2},
            )
            ax.set_title(f"Distribución de {var_dist}", fontsize=11, fontweight="bold")
            ax.set_xlabel(var_dist)
            ax.set_ylabel("Frecuencia")
            sns.despine()

            st.pyplot(fig_hist)

# ==========================================
# SECCIÓN 5: AGENTE DE IA ESPECIALIZADO EN DATOS
# ==========================================
elif seccion == "🤖 Asistente de Interpretación de Datos":
    st.title("🤖 Asistente Analista de Datos e Interpretación")

    st.markdown(
        """
    <div class="insight-box">
    <b>📊 Tu Interprete de Datos Personal:</b><br>
    Este agente tiene acceso completo a la estructura del dataset, métricas descriptivas, matrices de correlación y desglose por riego. 
    <b>Pregúntale cualquier duda sobre lo que ves en las tablas, gráficos, outliers o patrones estadísticos del Dashboard.</b>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if not groq_api_key:
        st.warning(
            "⚠️ **Requiere API Key de Groq para funcionar.**\n\nPor favor, ingresa"
            " tu API Key en la **barra lateral (Sidebar)** bajo la sección"
            " **'🔑 Configuración de API'**."
        )
    else:
        # 1. Extracción y preparación contextual detallada
        num_df = df.select_dtypes(include=["number"])
        corr_matrix = num_df.corr().to_string() if len(num_df.columns) > 1 else "N/A"

        stats_riego_str = "No disponible"
        if col_riego and col_prod:
            stats_riego_df = df.groupby("Riego_Categorico")[col_prod].agg(
                ["median", "mean", "std", "min", "max"]
            )
            stats_riego_str = stats_riego_df.to_string()

        frecuencias_cat = {}
        for col in df.select_dtypes(include=["object", "category"]).columns:
            frecuencias_cat[col] = df[col].value_counts().to_dict()

        # System prompt especializado en interpretación de datos
        SYSTEM_PROMPT = f"""
        Eres un Analista Senior de Datos e Inteligencia de Negocios especializado en el Sector Agrícola de Colombia.
        Tu misión principal es **INTERPRETAR LOS DATOS Y EXPLICAR EL DASHBOARD** al usuario. Ayúdalo a comprender las gráficas, métricas, correlaciones y hallazgos.

        --- INFORMACIÓN TÉCNICA DEL DASHBOARD ---
        1. ESTRUCTURA DEL DATASET:
           - Total de registros/filas: {len(df)}
           - Total de columnas: {df.shape[1]}
           - Nombres y tipos de columnas: {dict(df.dtypes.astype(str))}
           - Valores nulos por columna: {dict(df.isnull().sum())}

        2. ESTADÍSTICAS DESCRIPTIVAS NUMÉRICAS (Promedio, Mediana, Std, Min, Max, Quartiles):
           {num_df.describe().to_string()}

        3. ANÁLISIS DE IMPACTO DE RIEGO VS PRODUCCIÓN (Fila por condición):
           {stats_riego_str}

        4. MATRIZ DE CORRELACIÓN DE PEARSON:
           {corr_matrix}

        5. FRECUENCIAS DE VARIABLES CATEGÓRICAS:
           {frecuencias_cat}

        --- REGLAS DE RESPUESTA ---
        - Explica de forma clara e intuitiva qué significan los números (ej. explica la diferencia entre media y mediana si hay sesgo).
        - Si el usuario te pregunta sobre las gráficas de boxplot, explícale la interpretación del Riego (mediana, IQR, outliers).
        - Mantén un lenguaje accesible, profesional y enfocado en Storytelling de Datos.
        - Responde siempre en español.
        """

        # Inicializar chat
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [
                {
                    "role": "assistant",
                    "content": (
                        "¡Hola! 👋 Tengo acceso completo a todas las estadísticas, tablas"
                        " y gráficos de este Dashboard. \n\n¿Qué gráfico o dato"
                        " te gustaría que te interprete o te explique en detalle?"
                    ),
                }
            ]

        # Botones de sugerencias de preguntas
        st.markdown("##### 💡 Preguntas sugeridas para la IA:")
        col_s1, col_s2, col_s3 = st.columns(3)
        sugerencia = None
        if col_s1.button("📊 Explícame el Boxplot de Riego vs Producción"):
            sugerencia = (
                "¿Qué conclusiones principales puedo sacar de las gráficas de Boxplot"
                " de Riego vs Producción?"
            )
        if col_s2.button("📈 ¿Hay alguna correlación importante?"):
            sugerencia = (
                "¿Cuáles son las correlaciones más fuertes que se observan en la"
                " matriz de datos?"
            )
        if col_s3.button("🔍 Interpretación general del dataset"):
            sugerencia = (
                "Dame un resumen ejecutivo de la calidad y distribución general de"
                " los datos en pantalla."
            )

        # Botón Limpiar Chat
        col_title, col_clean = st.columns([4, 1])
        with col_clean:
            if st.button("🗑️ Limpiar Chat"):
                st.session_state.chat_messages = [
                    {
                        "role": "assistant",
                        "content": (
                            "Chat reiniciado. ¿En qué análisis de datos te puedo colaborar?"
                        ),
                    }
                ]
                st.rerun()

        # Mostrar mensajes
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Capturar la entrada del usuario o la sugerencia
        prompt_final = (
            sugerencia
            if sugerencia
            else st.chat_input("Pregúntale a la IA sobre cualquier dato o gráfica...")
        )

        if prompt_final:
            if not sugerencia:
                st.chat_message("user").markdown(prompt_final)
            st.session_state.chat_messages.append(
                {"role": "user", "content": prompt_final}
            )

            with st.chat_message("assistant"):
                with st.spinner("Analizando estadísticas con Llama 3.3..."):
                    try:
                        client = Groq(api_key=groq_api_key)

                        mensajes_api = [{"role": "system", "content": SYSTEM_PROMPT}]
                        for m in st.session_state.chat_messages:
                            mensajes_api.append(
                                {"role": m["role"], "content": str(m["content"])}
                            )

                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=mensajes_api,
                            temperature=0.4,
                            max_tokens=1024,
                        )

                        respuesta_texto = response.choices[0].message.content
                        st.markdown(respuesta_texto)

                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": respuesta_texto}
                        )

                    except Exception as e:
                        st.error(
                            f"Error al conectar con la API de Groq: {e}. Asegúrate de que"
                            " la API Key ingresada sea válida."
                        )
