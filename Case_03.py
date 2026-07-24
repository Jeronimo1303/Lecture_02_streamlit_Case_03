import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")

st.title("📊 Exploratory Data Analysis")

st.write("Upload a CSV file to automatically generate an exploratory data analysis.")

# -------------------------------------------------------
# File uploader
# -------------------------------------------------------


df = pd.read_csv("Lecture_02_streamlit_Case_03\\agro_colombia.csv")

# ===================================================
# Dataset Overview
# ===================================================

st.header("Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Missing Values", int(df.isna().sum().sum()))
c4.metric("Duplicated Rows", int(df.duplicated().sum()))

st.subheader("Preview")

st.dataframe(df.head())

# ===================================================
# Data Types
# ===================================================

st.header("Column Information")

info = pd.DataFrame(
    {"Type": df.dtypes, "Missing": df.isnull().sum(), "Unique": df.nunique()}
)

st.dataframe(info)

# ===================================================
# Summary Statistics
# ===================================================

st.header("Summary Statistics")

st.dataframe(df.describe(include="all").T)

# ===================================================
# Missing Values
# ===================================================

st.header("Missing Values")

missing = df.isnull().sum().sort_values(ascending=False)

st.bar_chart(missing)

# ===================================================
# Univariate Analysis
# ===================================================

st.header("Variable Distribution")

numeric_columns = df.select_dtypes(include="number").columns

if len(numeric_columns) > 0:

    column = st.selectbox("Select a numeric variable", numeric_columns)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[column], kde=True, ax=ax)

    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8, 2))
    sns.boxplot(x=df[column], ax=ax)

    st.pyplot(fig)

# ===================================================
# Correlation
# ===================================================

if len(numeric_columns) > 1:

    st.header("Correlation Matrix")

    corr = df[numeric_columns].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)

    st.pyplot(fig)

# ===================================================
# Scatter Plot
# ===================================================

if len(numeric_columns) > 1:

    st.header("Relationship Between Variables")

    c1, c2 = st.columns(2)

    x = c1.selectbox("X Axis", numeric_columns, key="x")

    y = c2.selectbox("Y Axis", numeric_columns, index=1, key="y")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(data=df, x=x, y=y, ax=ax)

    st.pyplot(fig)
