
import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import os

# Page setup
st.set_page_config(page_title="FinSmart Analytics Web",
                   page_icon="📊",
                   layout="wide")

st.title("FinSmart Analytics Web")
st.caption("Web dashboard interaktif berbasis AI & Data Science")

# Load API
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

def generate_ai_commentary(prompt):
    try:
        chat = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":prompt}]
        )
        return chat.choices[0].message.content
    except Exception as e:
        return f"Error AI: {e}"

# Upload
uploaded = st.file_uploader("Upload CSV/Excel", type=["csv","xlsx"])

df = None
if uploaded:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        st.success("Data loaded!")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Gagal baca file: {e}")

# Dashboard
if df is not None and all(col in df.columns for col in ["date","amount","category"]):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    st.subheader("Grafik Time Series Amount")
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(df['date'], df['amount'])
    st.pyplot(fig)

    st.subheader("Pie Chart Category")
    fig2, ax2 = plt.subplots()
    df.groupby("category")["amount"].sum().plot(kind="pie", ax=ax2, autopct='%1.1f%%')
    ax2.set_ylabel("")
    st.pyplot(fig2)

    # Fraud Detection
    st.subheader("Fraud Detection (Isolation Forest)")
    model = IsolationForest(contamination=0.05)
    df['anomaly'] = model.fit_predict(df[['amount']])
    anomalies = df[df['anomaly']==-1]
    st.write(anomalies)

    # AI Insights
    st.subheader("AI Financial Insight")
    insight = generate_ai_commentary(f"Buat analisis singkat dari data ini:\n{df.head().to_string()}")
    st.write(insight)

# Chat Mode
st.subheader("AI Chat Mode")
if "chat" not in st.session_state:
    st.session_state.chat=[]

for role,msg in st.session_state.chat:
    st.chat_message(role).write(msg)

user_input = st.chat_input("Tanya AI tentang data Anda...")

if user_input:
    st.session_state.chat.append(("user", user_input))
    ai_reply = generate_ai_commentary(user_input)
    st.session_state.chat.append(("assistant", ai_reply))
    st.chat_message("assistant").write(ai_reply)
