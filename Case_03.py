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

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (8, 4)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        background: linear-gradient(135deg, #fcfefe 0%, #f7fbff 100%);
    }
    .stMetric {
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid #cbd5e1;
        border-radius: 0.85rem;
        padding: 0.8rem 0.9rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] * {
        color: white;
    }
    .stTextInput>div>div>input, .stSelectbox>div>div>div {
        border-radius: 0.6rem;
    }
    table, th, td {
        border: 1px solid #cbd5e1 !important;
    }
    th {
        background-color: #eef6ff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌿 Exploratory Data Analysis")
st.caption(
    "Una mirada clara y guiada a los patrones del dataset, diseñada para que cada gráfica cuente una historia útil y fácil de entender."
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

    st.markdown(
        "<div style='padding:0.8rem 1rem; border-left:4px solid #14b8a6; background:#f0fdf4; border-radius:0.5rem;'>"
        "<b>Historia del dataset:</b> esta vista inicial ayuda a identificar rápidamente el tamaño del problema, la calidad de los datos y los patrones que merecen atención."
        "</div>",
        unsafe_allow_html=True,
    )

    st.subheader("� Insight clave")
    st.info(
        "El resumen general ya no está vacío: aquí puedes ver de un vistazo la cantidad de datos, la calidad del dataset y los puntos que conviene explorar primero."
    )

    st.subheader("�📈 Boxplot comparativo: Producción con y sin riego")
    st.caption(
        "Esta comparación permite ver si la presencia de riego cambia de forma marcada la producción de las fincas."
    )
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
            data=plot_df,
            x="Riego",
            y="Produccion_Anual_Ton",
            palette=["#2563eb", "#14b8a6"],
            ax=ax,
        )
        ax.set_title("Producción anual: comparación entre fincas con y sin riego")
        ax.set_xlabel("Sistema de riego")
        ax.set_ylabel("Producción anual (ton)")
        ax.set_facecolor("#f8fbff")
        st.pyplot(fig_seaborn)

        if px is not None:
            fig_plotly = px.box(
                plot_df,
                x="Riego",
                y="Produccion_Anual_Ton",
                color="Riego",
                title="Producción anual: con riego vs sin riego",
                color_discrete_sequence=["#2563eb", "#14b8a6"],
            )
            fig_plotly.update_layout(
                height=400,
                template="plotly_white",
                paper_bgcolor="#f8fbff",
                plot_bgcolor="#f8fbff",
            )
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
        sns.histplot(
            df[selected_num],
            kde=True,
            bins=20,
            color="#2563eb",
            edgecolor="white",
            line_kws={"color": "#0f766e"},
            ax=ax,
        )
        ax.axvline(df[selected_num].mean(), color="#f59e0b", ls="--", lw=2)
        ax.set_title(f"Distribución de {selected_num}: patrones centrales y dispersión")
        ax.set_xlabel(selected_num)
        ax.set_ylabel("Frecuencia")
        col_a.pyplot(fig)

        fig, ax = plt.subplots()
        sns.boxplot(x=df[selected_num], color="#60a5fa", ax=ax)
        ax.set_title(f"Boxplot de {selected_num}: valores atípicos y rango")
        ax.set_xlabel(selected_num)
        col_b.pyplot(fig)

        if len(numeric_columns) > 1:
            st.subheader("Matriz de correlación")
            corr = df[numeric_columns].corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f", ax=ax)
            ax.set_title("Mapa de correlación: relaciones entre variables")
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

        st.caption(
            "Estas barras muestran qué categorías predominan y cómo se distribuyen los registros."
        )
        fig, ax = plt.subplots()
        counts.plot(kind="bar", color="#14b8a6", edgecolor="#0f766e", ax=ax)
        ax.set_title(f"Top valores de {selected_cat}: qué destaca más")
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
        st.caption(
            "Cada punto representa una observación; la línea de tendencia ayuda a interpretar si existe una relación clara entre ambas variables."
        )
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=df, x=x_var, y=y_var, color="#2563eb", alpha=0.7, s=90, ax=ax
        )
        sns.regplot(
            data=df,
            x=x_var,
            y=y_var,
            scatter=False,
            color="#f59e0b",
            line_kws={"lw": 2},
            ax=ax,
        )
        ax.set_title(f"{y_var} vs {x_var}: relación observada")
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        st.pyplot(fig)
    else:
        st.info(
            "Se necesitan al menos dos variables numéricas para mostrar gráficos de relación."
        )

    if numeric_columns:
        st.subheader("Resumen visual rápido")
        st.caption(
            "Este gráfico resume cuántos valores faltantes hay por variable, facilitando la priorización del limpieza de datos."
        )
        st.bar_chart(df[numeric_columns].isna().sum())
