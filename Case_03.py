import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

try:
    import plotly.express as px
except ImportError:  # pragma: no cover
    px = None

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
        background: linear-gradient(135deg, #fcfefe 0%, #f6fbff 100%);
    }
    .stApp {
        background: #f9fcff;
    }
    .stMetric {
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid #cbd5e1;
        border-radius: 0.85rem;
        padding: 0.8rem 0.9rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] * {
        color: white;
    }
    .hero-card {
        background: linear-gradient(135deg, #f8fcff 0%, #eefcf6 100%);
        border: 1px solid #dbeafe;
        border-radius: 1rem;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .section-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 0.9rem;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
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


@st.cache_data
def load_data():
    return pd.read_csv("agro_colombia.csv")


df = load_data()

numeric_columns = df.select_dtypes(include="number").columns.tolist()
categorical_columns = df.select_dtypes(exclude="number").columns.tolist()

with st.sidebar:
    st.header("🧭 Navegación")
    st.caption("Explora el dataset por secciones temáticas.")
    section = st.radio(
        "Ir a",
        ["Resumen", "Análisis cuantitativo", "Análisis cualitativo", "Visualización"],
        index=0,
    )

    st.divider()
    st.subheader("Cargar datos")
    uploaded_file = st.file_uploader("Subir un CSV", type=["csv"])
    st.caption("Si no cargas archivo, se usa el dataset por defecto.")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(exclude="number").columns.tolist()

st.title("🌿 Dashboard EDA para agro_colombia")
st.caption(
    "Una vista narrativa, clara y visual del dataset para entender patrones, comparaciones y tendencias con facilidad."
)

st.markdown(
    """
    <div class="hero-card">
        <h4 style="margin:0 0 0.35rem 0;">Historia del dataset</h4>
        <p style="margin:0; color:#334155;">Cada sección ayuda a interpretar el negocio agrícola desde una perspectiva cuantitativa, cualitativa y gráfica.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Filas", f"{df.shape[0]:,}")
c2.metric("Columnas", df.shape[1])
c3.metric("Valores faltantes", int(df.isna().sum().sum()))
c4.metric("Filas duplicadas", int(df.duplicated().sum()))

if section == "Resumen":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Resumen general")
    st.caption(
        "Esta vista inicial ofrece contexto rápido: tamaño del dataset, calidad de los datos y el punto de partida para explorar el problema."
    )

    if (
        "Produccion_Anual_Ton" in df.columns
        and "Sistema_Riego_Tecnificado" in df.columns
    ):
        plot_df = df[["Produccion_Anual_Ton", "Sistema_Riego_Tecnificado"]].copy()
        plot_df["Riego"] = plot_df["Sistema_Riego_Tecnificado"].map(
            {True: "Con riego", False: "Sin riego"}
        )

        st.subheader("📊 Boxplot comparativo: Producción con riego vs. sin riego")
        st.caption(
            "Este gráfico permite ver si la presencia de riego está asociada a diferencias claras en la producción anual de las fincas."
        )

        fig_seaborn, ax = plt.subplots(figsize=(9, 4))
        sns.boxplot(
            data=plot_df,
            x="Riego",
            y="Produccion_Anual_Ton",
            palette=["#2563eb", "#14b8a6"],
            ax=ax,
        )
        ax.set_title("Producción anual: fincas con riego vs. sin riego")
        ax.set_xlabel("Tipo de finca")
        ax.set_ylabel("Producción anual (ton)")
        ax.set_facecolor("#f8fbff")
        st.pyplot(fig_seaborn)

        if px is not None:
            fig_plotly = px.box(
                plot_df,
                x="Riego",
                y="Produccion_Anual_Ton",
                color="Riego",
                title="Producción anual: con riego vs. sin riego",
                color_discrete_sequence=["#2563eb", "#14b8a6"],
            )
            fig_plotly.update_layout(
                template="plotly_white",
                paper_bgcolor="#f8fbff",
                plot_bgcolor="#f8fbff",
                height=400,
                margin=dict(l=30, r=30, t=50, b=30),
            )
            st.plotly_chart(fig_plotly, use_container_width=True)

    st.subheader("Vista previa del dataset")
    st.caption(
        "Una muestra inicial del contenido del archivo para comprender la estructura de los datos."
    )
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
    st.markdown("</div>", unsafe_allow_html=True)

elif section == "Análisis cuantitativo":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Análisis cuantitativo")
    st.caption(
        "Aquí se observan los indicadores numéricos esenciales para entender la distribución y el comportamiento del dataset."
    )

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
            st.subheader("Mapa de correlación")
            corr = df[numeric_columns].corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f", ax=ax)
            ax.set_title("Relaciones entre variables numéricas")
            st.pyplot(fig)
    else:
        st.info("No hay variables numéricas disponibles para analizar.")
    st.markdown("</div>", unsafe_allow_html=True)

elif section == "Análisis cualitativo":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Análisis cualitativo")
    st.caption(
        "Estas visualizaciones ayudan a reconocer las categorías más frecuentes y cómo se distribuyen entre los registros."
    )

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
        counts.plot(kind="bar", color="#14b8a6", edgecolor="#0f766e", ax=ax)
        ax.set_title(f"Top valores de {selected_cat}: qué destaca más")
        ax.set_xlabel(selected_cat)
        ax.set_ylabel("Frecuencia")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

        if px is not None:
            fig_plotly = px.bar(
                counts.reset_index().rename(
                    columns={
                        counts.index.name or selected_cat: selected_cat,
                        "count": "Frecuencia",
                    }
                ),
                x=selected_cat,
                y="Frecuencia",
                color=selected_cat,
                title=f"Frecuencia de {selected_cat}",
                color_discrete_sequence=px.colors.sequential.Emerald,
            )
            fig_plotly.update_layout(
                template="plotly_white",
                height=400,
                paper_bgcolor="#f8fbff",
                plot_bgcolor="#f8fbff",
            )
            st.plotly_chart(fig_plotly, use_container_width=True)
    else:
        st.info("No hay variables categóricas disponibles para analizar.")
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Visualización interactiva")
    st.caption(
        "Aquí se unen las relaciones entre variables en un formato más interpretativo y visual."
    )

    if numeric_columns and len(numeric_columns) > 1:
        col_x, col_y = st.columns(2)
        x_var = col_x.selectbox("Eje X", numeric_columns, key="x_var")
        y_var = col_y.selectbox(
            "Eje Y",
            numeric_columns,
            index=1 if len(numeric_columns) > 1 else 0,
            key="y_var",
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

        if px is not None:
            fig_plotly = px.scatter(
                df,
                x=x_var,
                y=y_var,
                title=f"Relación entre {x_var} y {y_var}",
                color_discrete_sequence=["#2563eb"],
            )
            fig_plotly.update_layout(
                template="plotly_white",
                height=400,
                paper_bgcolor="#f8fbff",
                plot_bgcolor="#f8fbff",
            )
            st.plotly_chart(fig_plotly, use_container_width=True)
    else:
        st.info(
            "Se necesitan al menos dos variables numéricas para mostrar gráficos de relación."
        )

    if numeric_columns:
        st.subheader("Resumen visual rápido")
        st.caption(
            "Este gráfico permite localizar rápidamente los campos con más valores faltantes para priorizar limpieza de datos."
        )
        st.bar_chart(df[numeric_columns].isna().sum())
    st.markdown("</div>", unsafe_allow_html=True)
