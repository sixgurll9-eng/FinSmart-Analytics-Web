# app.py
"""
FinSmart Analytics Web - Single-file Streamlit app (ready to deploy)
Features:
 - Upload Excel/CSV (multi-sheet friendly)
 - Normalize columns: date, amount, category
 - KPIs, time-series, category breakdown, simple forecast
 - Anomaly detection (IsolationForest + z-score)
 - Rule-based commentary fallback
 - Groq LLM AI commentary & interactive chat (uses GROQ_API_KEY)
Notes:
 - Set GROQ_API_KEY in Streamlit Secrets (exact name) or environment variable.
 - Requirements (example): streamlit groq pandas numpy plotly scikit-learn openpyxl python-dotenv
"""

import os
import io
import traceback
from datetime import timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest

# Groq SDK
from groq import Groq

# ---------------------------------------
# Page setup
# ---------------------------------------
st.set_page_config(page_title="FinSmart Analytics Web",
                   page_icon="📊",
                   layout="wide")

st.title("FinSmart Analytics Web")
st.caption("Web dashboard interaktif berbasis AI & Data Science — upload Excel/CSV & dapatkan insight otomatis")

# ---------------------------------------
# Load Groq API safely
# ---------------------------------------
def get_groq_client():
    # Prefer Streamlit secrets
    key = None
    try:
        # Streamlit stores secrets keys in st.secrets as mapping
        if "GROQ_API_KEY" in st.secrets:
            key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    if not key:
        key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    try:
        return Groq(api_key=key)
    except Exception:
        return None

groq_client = get_groq_client()
AI_ENABLED = groq_client is not None

if not AI_ENABLED:
    st.info("AI disabled — GROQ_API_KEY not found. AI commentary will fall back to rule-based text. Set GROQ_API_KEY in Streamlit Secrets to enable AI.")

# ---------------------------------------
# Helpers: loading + normalizing data
# ---------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # strip whitespace from headers
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("date", "tanggal"):
            col_map[c] = "date"
        if lc in ("amount", "amt", "value", "total", "jumlah", "price", "amount_usd"):
            col_map[c] = "amount"
        if "cat" in lc or "kategori" in lc or "type" in lc or "category" in lc:
            col_map[c] = "category"
    if col_map:
        df = df.rename(columns=col_map)
    return df

def parse_date_column(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer column named 'date'. If absent, try heuristics.
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        for c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().sum() > 0:
                    df["date"] = parsed
                    break
            except Exception:
                continue
    return df

def load_file(file) -> pd.DataFrame:
    """Read CSV or Excel (multi-sheet fallback to first sheet) and normalize columns."""
    if file is None:
        return pd.DataFrame()
    name = getattr(file, "name", "")
    file.seek(0)
    try:
        if name.lower().endswith((".xls", ".xlsx")):
            # read all sheets; pick first non-empty
            xls = pd.read_excel(file, sheet_name=None, engine="openpyxl")
            # pick sheet with most rows
            best = None
            best_len = -1
            for sname, df in xls.items():
                if isinstance(df, pd.DataFrame) and len(df) > best_len:
                    best = df
                    best_len = len(df)
            df = best if best is not None else pd.DataFrame()
        else:
            # CSV
            df = pd.read_csv(file)
    except Exception:
        # fallback: try csv with latin1
        file.seek(0)
        df = pd.read_csv(file, encoding="latin1", low_memory=False)
    df = normalize_columns(df)
    df = parse_date_column(df)
    # try to compute amount from quantity & price if missing
    if "amount" not in df.columns:
        qty_cols = [c for c in df.columns if "qty" in c.lower() or "quantity" in c.lower()]
        price_cols = [c for c in df.columns if "price" in c.lower() or "unit price" in c.lower()]
        if qty_cols and price_cols:
            try:
                df["amount"] = pd.to_numeric(df[qty_cols[0]], errors="coerce") * pd.to_numeric(df[price_cols[0]], errors="coerce")
            except Exception:
                pass
    # convert amount numeric
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    # drop rows missing essential fields
    if "date" in df.columns and "amount" in df.columns:
        df = df.dropna(subset=["date", "amount"]).copy()
        df = df.sort_values("date")
    return df

# ---------------------------------------
# Anomaly detection helpers
# ---------------------------------------
def detect_anomalies_zscore(series: pd.Series, z_thresh: float = 3.0) -> pd.Series:
    if len(series) == 0:
        return pd.Series(dtype=bool)
    mean = series.mean()
    std = series.std(ddof=0) if series.std(ddof=0) != 0 else 1.0
    z = (series - mean) / std
    return z.abs() > z_thresh

def detect_anomalies_isolation(df: pd.DataFrame, column="amount", contamination=0.03):
    try:
        model = IsolationForest(contamination=contamination, random_state=42)
        vals = df[[column]].fillna(0).values
        preds = model.fit_predict(vals)
        df["_anomaly_iforest"] = preds  # -1 anomaly, 1 normal
        anomalies = df[df["_anomaly_iforest"] == -1]
        return anomalies
    except Exception:
        return pd.DataFrame()

# ---------------------------------------
# Rule-based summary (fallback)
# ---------------------------------------
def generate_rule_based_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "Tidak ada data untuk dianalisis."
    total_income = df.loc[df["amount"] > 0, "amount"].sum()
    total_expense = -df.loc[df["amount"] < 0, "amount"].sum() if (df["amount"] < 0).any() else 0
    net = total_income - total_expense
    avg_txn = df["amount"].mean()
    days = max((df["date"].max() - df["date"].min()).days, 1)
    daily = df.set_index("date").resample("D")["amount"].sum().fillna(0)
    anomalies = detect_anomalies_zscore(daily)
    anom_count = int(anomalies.sum()) if hasattr(anomalies, "sum") else 0
    text = (
        f"Ringkasan otomatis:\n"
        f"- Periode: {df['date'].min().date()} s/d {df['date'].max().date()} ({days} hari)\n"
        f"- Total pendapatan: Rp {int(total_income):,}\n"
        f"- Total pengeluaran: Rp {int(total_expense):,}\n"
        f"- Net: Rp {int(net):,}\n"
        f"- Rata-rata per transaksi: Rp {int(avg_txn):,}\n"
        f"- Detected anomaly days (z-score): {anom_count}\n"
    )
    return text

# ---------------------------------------
# AI commentary via Groq (safe)
# ---------------------------------------
def generate_ai_commentary(df: pd.DataFrame, chat_history: list = None, user_prompt: str = None) -> str:
    """
    Build a context prompt and call Groq. If Groq unavailable or error -> fallback to rule-based.
    """
    try:
        summary = generate_rule_based_summary(df)
    except Exception:
        summary = "Tidak dapat membuat ringkasan (error internal)."

    # Compose prompt
    prompt_items = []
    prompt_items.append("You are a concise financial analyst for small businesses. Provide short, actionable insights.")
    prompt_items.append("Data summary:\n" + summary)
    if user_prompt:
        prompt_items.append("User question: " + user_prompt)
    # include small snippet of latest transactions for context
    if not df.empty:
        try:
            snippet = df.tail(8)[["date", "amount", "category"]].to_string(index=False)
            prompt_items.append("Recent transactions (most recent first):\n" + snippet)
        except Exception:
            pass
    if chat_history:
        prompt_items.append("Conversation context:\n" + "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-6:]]))

    final_prompt = "\n\n".join(prompt_items)

    if not AI_ENABLED:
        return "AI disabled (no GROQ_API_KEY). Fallback:\n\n" + summary

    try:
        # Use a recommended Groq model that you have access to
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are Finance Twin AI — concise, helpful, and actionable."},
                {"role": "user", "content": final_prompt}
            ],
            max_tokens=600
        )
        # Newer Groq SDK: message is object -> use .content
        ai_text = response.choices[0].message.content
        return ai_text
    except Exception:
        traceback.print_exc()
        return "AI call failed. Fallback:\n\n" + summary

# ---------------------------------------
# Session state init
# ---------------------------------------
if "transactions_df" not in st.session_state:
    st.session_state["transactions_df"] = pd.DataFrame()
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # list of dicts {role, content}

# ---------------------------------------
# Sidebar: upload area
# ---------------------------------------
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload Excel (.xlsx/.xls) or CSV (min cols: date, amount, category)", type=["csv", "xls", "xlsx"])

contamination = st.sidebar.slider("Anomaly contamination (IsolationForest)", min_value=0.0, max_value=0.2, value=0.03, step=0.01)
forecast_days = st.sidebar.slider("Forecast horizon (days)", 7, 90, 30)

if uploaded_file is not None:
    with st.spinner("Memuat file dan menyiapkan data..."):
        df_loaded = load_file(uploaded_file)
        # quick validation
        missing = [c for c in ("date", "amount", "category") if c not in df_loaded.columns]
        if missing:
            st.sidebar.error(f"File tidak valid — kolom yang hilang: {missing}. Pastikan file minimal memiliki kolom: date, amount, category.")
        else:
            st.session_state["transactions_df"] = df_loaded.copy()
            st.sidebar.success(f"Data terload: {len(df_loaded)} baris. Periode: {df_loaded['date'].min().date()} - {df_loaded['date'].max().date()}")

# main df reference
df = st.session_state["transactions_df"]

# ---------------------------------------
# Dashboard: KPIs + charts + anomaly + forecast
# ---------------------------------------
st.markdown("---")
st.header("Dashboard Analytics")

# KPIs
k1, k2, k3, k4 = st.columns(4)
if not df.empty:
    total_income = int(df.loc[df["amount"] > 0, "amount"].sum())
    total_expense = int(-df.loc[df["amount"] < 0, "amount"].sum()) if (df["amount"] < 0).any() else 0
    net = total_income - total_expense
    days_covered = (df["date"].max() - df["date"].min()).days or 1
    k1.metric("Total Income", f"Rp {total_income:,}")
    k2.metric("Total Expense", f"Rp {total_expense:,}")
    k3.metric("Net", f"Rp {net:,}")
    k4.metric("Days covered", f"{days_covered} days")
else:
    k1.metric("Total Income", "Rp 0")
    k2.metric("Total Expense", "Rp 0")
    k3.metric("Net", "Rp 0")
    k4.metric("Days covered", "0 days")

if df.empty:
    st.info("Belum ada data. Upload file di sidebar untuk menampilkan dashboard.")
else:
    # Time series (daily)
    st.subheader("Cashflow: Time Series (daily)")
    daily = df.set_index("date").resample("D")["amount"].sum().fillna(0)
    fig_ts = px.line(daily.reset_index(), x="date", y="amount", title="Daily cashflow")
    fig_ts.update_layout(yaxis_title="Amount", template="plotly_dark")
    st.plotly_chart(fig_ts, use_container_width=True)

    # Category breakdown
    st.subheader("Breakdown by Category")
    if "category" in df.columns:
        cat_agg = df.groupby("category")["amount"].sum().reset_index().sort_values("amount", ascending=False)
        fig_cat = px.bar(cat_agg, x="category", y="amount", title="Amount by Category")
        fig_cat.update_layout(template="plotly_dark", xaxis_title="Category", yaxis_title="Amount")
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.info("Kolom 'category' tidak ditemukan.")

    # Anomaly detection: z-score on daily + IsolationForest on transactions
    st.subheader("Anomaly Detection")
    # z-score (daily)
    z_mask = detect_anomalies_zscore(daily, z_thresh=3.0)
    z_count = int(z_mask.sum())
    st.write(f"Z-score detected anomalous days: {z_count}")
    if z_count > 0:
        st.dataframe(daily[z_mask].reset_index().rename(columns={0: "amount"}).head(20))
    # IsolationForest (transaction-level)
    iso_anoms = detect_anomalies_isolation(df, column="amount", contamination=contamination)
    if not iso_anoms.empty:
        st.write(f"IsolationForest flagged {len(iso_anoms)} suspicious transactions (showing top 20):")
        st.dataframe(iso_anoms.head(20))
    else:
        st.write("No transaction-level anomalies detected by IsolationForest (with current contamination).")

    # Forecast (simple linear)
    st.subheader(f"Forecast (linear) — next {forecast_days} days")
    vals = daily.values
    if len(vals) >= 5:
        x = np.arange(len(vals))
        coeffs = np.polyfit(x, vals, 1)
        future_x = np.arange(len(vals), len(vals) + forecast_days)
        forecast = coeffs[0] * future_x + coeffs[1]
        future_index = pd.date_range(daily.index.max() + timedelta(days=1), periods=forecast_days)
        df_fore = pd.DataFrame({"date": list(daily.index) + list(future_index),
                                "amount": list(vals) + list(forecast),
                                "type": ["actual"] * len(vals) + ["forecast"] * len(forecast)})
        fig_fore = px.line(df_fore, x="date", y="amount", color="type", title="Actual vs Forecast (linear)")
        fig_fore.update_layout(template="plotly_dark")
        st.plotly_chart(fig_fore, use_container_width=True)
    else:
        st.info("Data tidak cukup untuk forecasting (butuh minimal ~5 hari data).")

    # Rule-based commentary (editable)
    st.subheader("Quick Rule-based Commentary")
    rb = generate_rule_based_summary(df)
    st.text_area("Rule-based commentary (editable)", value=rb, height=200)

# ---------------------------------------
# AI Commentary (single-shot) & Chat Mode
# ---------------------------------------
st.markdown("---")
st.header("AI Commentary & Chat (Finance Analyst)")

col_left, col_right = st.columns([3,1])
with col_left:
    if st.button("Generate AI Commentary (single-shot)"):
        with st.spinner("Menghubungi AI..."):
            ai_text = generate_ai_commentary(df, st.session_state.get("chat_history", []))
            st.success("AI commentary:")
            st.write(ai_text)

    st.markdown("#### Interactive Chat")
    # display chat history
    for msg in st.session_state["chat_history"]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            st.chat_message("user").write(content)
        else:
            st.chat_message("assistant").write(content)

    user_input = st.chat_input("Tanya AI tentang data / anomaly / forecast...")
    if user_input:
        # append user message
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.spinner("AI sedang menulis..."):
            ai_answer = generate_ai_commentary(df, st.session_state.get("chat_history", []), user_prompt=user_input)
            st.session_state["chat_history"].append({"role": "assistant", "content": ai_answer})
            st.chat_message("assistant").write(ai_answer)

with col_right:
    st.info("Tips:\n- Untuk AI: atur GROQ_API_KEY di Streamlit Secrets.\n- Untuk file besar: pre-sample data sebelum upload.\n- Hapus chat history via tombol di bawah.")
    if st.button("Clear Chat History"):
        st.session_state["chat_history"] = []
        st.experimental_rerun()

# ---------------------------------------
# Footer / Notes
# ---------------------------------------
st.markdown("---")
st.caption("FinSmart Analytics Web • Prototype • Developed for academic project. Untuk produksi: tambahkan auth, rate-limiting, dan validasi lebih ketat.")
