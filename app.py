import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/dataset.csv")

# Load models
lead_model = joblib.load("models/lead_model.pkl")
pip_model = joblib.load("models/pip_model.pkl")
salary_model = joblib.load("models/salary_model.pkl")

st.set_page_config(page_title="Euromax HR Analytics", layout="wide")

st.title("Euromax HR & Training Analytics Dashboard")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Lead Analytics", "Candidate Performance", "Employment Analysis", "Salary Analysis"]
)

# -------------------------
# Lead Analytics
# -------------------------
if page == "Lead Analytics":
    st.subheader("Lead Conversion Overview")

    fig, ax = plt.subplots()
    sns.countplot(x="Lead_Converted", data=df, ax=ax)
    st.pyplot(fig)

    st.dataframe(df[["Lead_Source", "Interest_Level", "Follow_Ups", "Lead_Converted"]].head(20))

# -------------------------
# Candidate Performance
# -------------------------
elif page == "Candidate Performance":
    st.subheader("Candidate Performance Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.histplot(df["Performance_Score"], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        pip_counts = df["PIP_Flag"].value_counts()
        st.bar_chart(pip_counts)

# -------------------------
# Employment Analysis
# -------------------------
elif page == "Employment Analysis":
    st.subheader("Employment Type Distribution")

    fig, ax = plt.subplots()
    sns.countplot(x="Employment_Type", data=df, ax=ax)
    st.pyplot(fig)

# -------------------------
# Salary Analysis
# -------------------------
elif page == "Salary Analysis":
    st.subheader("Salary (CTC) Analysis")

    fig, ax = plt.subplots()
    sns.boxplot(x="Employment_Type", y="CTC", data=df, ax=ax)
    st.pyplot(fig)

    st.write("Salary Summary")
    st.dataframe(df[["Basic_Salary", "HRA", "TA", "PF", "CTC"]].describe())
