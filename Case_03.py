import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

try:
    import plotly.express as px
except ImportError:  # pragma: no cover
    px = None

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 4)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    .stMetric {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 0.8rem 0.9rem;
    }
    [data-testid="stSidebar"] {
        background-color: #0f172a;
    }
    [data-testid="stSidebar"] * {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Exploratory Data Analysis")
st.caption(
    "Analiza datos de forma cuantitativa, cualitativa y visual para obtener una visión más completa del conjunto."
)


@st.cache_data
def load_default_data():
    return pd.read_csv("agro_colombia.csv")


# -------------------------------------------------------
# Data loading
# -------------------------------------------------------

with st.sidebar:
    st.header("🧭 Navegación")
    section = st.radio(
        "Ir a",
        ["Resumen", "Análisis cuantitativo", "Análisis cualitativo", "Visualización"],
        index=0,
    )

    st.divider()
    st.subheader("Cargar datos")
    uploaded_file = st.file_uploader("Subir un CSV", type=["csv"])
    st.caption("Si no subes un archivo, se usará el dataset por defecto.")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_default_data()


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

numeric_columns = df.select_dtypes(include="number").columns.tolist()
categorical_columns = df.select_dtypes(exclude="number").columns.tolist()


# -------------------------------------------------------
# Overview section
# -------------------------------------------------------

if section == "Resumen":
    st.header("Resumen general")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", f"{df.shape[0]:,}")
    c2.metric("Columnas", df.shape[1])
    c3.metric("Valores faltantes", int(df.isna().sum().sum()))
    c4.metric("Filas duplicadas", int(df.duplicated().sum()))

    st.subheader("📈 Boxplot comparativo: Producción con y sin riego")
    if (
        "Produccion_Anual_Ton" in df.columns
        and "Sistema_Riego_Tecnificado" in df.columns
    ):
        plot_df = df[["Produccion_Anual_Ton", "Sistema_Riego_Tecnificado"]].copy()
        plot_df["Riego"] = plot_df["Sistema_Riego_Tecnificado"].map(
            {True: "Con riego", False: "Sin riego"}
        )

        fig_seaborn, ax = plt.subplots(figsize=(9, 4))
        sns.boxplot(
            data=plot_df, x="Riego", y="Produccion_Anual_Ton", palette="Set2", ax=ax
        )
        ax.set_title("Producción anual por tipo de finca")
        ax.set_xlabel("Sistema de riego")
        ax.set_ylabel("Producción anual (ton)")
        st.pyplot(fig_seaborn)

        if px is not None:
            fig_plotly = px.box(
                plot_df,
                x="Riego",
                y="Produccion_Anual_Ton",
                color="Riego",
                title="Producción anual: con riego vs sin riego",
            )
            fig_plotly.update_layout(height=400)
            st.plotly_chart(fig_plotly, use_container_width=True)
        else:
            st.info(
                "Plotly no está disponible en este entorno, pero la visualización en Seaborn ya está lista."
            )
    else:
        st.info(
            "No se encontraron las columnas necesarias para construir este boxplot."
        )

    st.subheader("Vista previa del dataset")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Información de columnas")
    info = pd.DataFrame(
        {
            "Tipo": df.dtypes,
            "Faltantes": df.isnull().sum(),
            "Únicos": df.nunique(),
        }
    )
    st.dataframe(info, use_container_width=True)

# -------------------------------------------------------
# Quantitative analysis
# -------------------------------------------------------

elif section == "Análisis cuantitativo":
    st.header("Análisis cuantitativo")

    if numeric_columns:
        st.subheader("Estadísticas descriptivas")
        st.dataframe(df[numeric_columns].describe().T, use_container_width=True)

        st.subheader("Distribución de variables numéricas")
        col_a, col_b = st.columns(2)
        selected_num = col_a.selectbox(
            "Selecciona una variable numérica", numeric_columns
        )

        fig, ax = plt.subplots()
        sns.histplot(df[selected_num], kde=True, color="#4f46e5", ax=ax)
        ax.set_title(f"Distribución de {selected_num}")
        col_a.pyplot(fig)

        fig, ax = plt.subplots()
        sns.boxplot(x=df[selected_num], color="#818cf8", ax=ax)
        ax.set_title(f"Boxplot de {selected_num}")
        col_b.pyplot(fig)

        if len(numeric_columns) > 1:
            st.subheader("Matriz de correlación")
            corr = df[numeric_columns].corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Correlación entre variables numéricas")
            st.pyplot(fig)
    else:
        st.info("No hay variables numéricas disponibles para analizar.")

# -------------------------------------------------------
# Qualitative analysis
# -------------------------------------------------------

elif section == "Análisis cualitativo":
    st.header("Análisis cualitativo")

    if categorical_columns:
        selected_cat = st.selectbox(
            "Selecciona una variable categórica", categorical_columns
        )

        st.subheader(f"Frecuencia de {selected_cat}")
        counts = df[selected_cat].value_counts().head(10)
        st.dataframe(
            counts.rename_axis(selected_cat).reset_index(name="Frecuencia"),
            use_container_width=True,
        )

        fig, ax = plt.subplots()
        counts.plot(kind="bar", color="#10b981", ax=ax)
        ax.set_title(f"Top valores de {selected_cat}")
        ax.set_xlabel(selected_cat)
        ax.set_ylabel("Frecuencia")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    else:
        st.info("No hay variables categóricas disponibles para analizar.")

# -------------------------------------------------------
# Visual section
# -------------------------------------------------------

else:
    st.header("Visualización interactiva")

    if numeric_columns and len(numeric_columns) > 1:
        col_x, col_y = st.columns(2)
        x_var = col_x.selectbox("Eje X", numeric_columns, key="x_var")
        y_var = col_y.selectbox(
            "Eje Y",
            numeric_columns,
            index=1 if len(numeric_columns) > 1 else 0,
            key="y_var",
        )

        st.subheader("Relación entre variables")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_var, y=y_var, color="#2563eb", ax=ax)
        ax.set_title(f"{y_var} vs {x_var}")
        st.pyplot(fig)
    else:
        st.info(
            "Se necesitan al menos dos variables numéricas para mostrar gráficos de relación."
        )

    if numeric_columns:
        st.subheader("Resumen visual rápido")
        st.bar_chart(df[numeric_columns].isna().sum())
