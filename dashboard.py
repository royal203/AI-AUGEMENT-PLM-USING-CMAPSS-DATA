import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Digital Twin – RUL & Degradation Rate Dashboard",
    layout="wide"
)

# ---------------------------------------------------------
# MODEL CLASS (same as training)
# ---------------------------------------------------------
class MultiTaskLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden_dim, 2)  # RUL + DR

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.out(last)


# ---------------------------------------------------------
# FEATURES + COLUMN NAMES
# ---------------------------------------------------------
FEATURES = ["setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
ALL_COLS = ["unit", "cycle"] + FEATURES   # 26 cols total
SEQ_LEN = 30


def ensure_cmapss_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure df has standard CMAPSS column names.
    Handles:
        - Raw txt from NASA (no header, numeric columns)
        - CSV with or without correct headers
    """
    # Case 1: already has all required columns
    if set(FEATURES).issubset(df.columns) and "unit" in df.columns and "cycle" in df.columns:
        return df

    # Case 2: raw CMAPSS layout (26 cols, unnamed or generic names)
    if df.shape[1] >= 26:
        df2 = df.iloc[:, :26].copy()
        df2.columns = ALL_COLS
        return df2

    # Otherwise: cannot map
    st.error(
        "Uploaded file does not match expected CMAPSS format.\n\n"
        f"Expected at least these columns: {ALL_COLS}"
    )
    st.stop()


def build_sequences(df: pd.DataFrame):
    X = []
    values = df[FEATURES].values

    if len(values) < SEQ_LEN:
        return None

    for i in range(len(values) - SEQ_LEN + 1):
        X.append(values[i:i + SEQ_LEN])

    return np.array(X, dtype=np.float32)


@st.cache_resource
def load_model(path="checkpoints/final_multitask_model.pth"):
    model = MultiTaskLSTM(len(FEATURES))
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# ---------------------------------------------------------
# DASHBOARD UI
# ---------------------------------------------------------
st.title("🔧 AI-Augmented Digital Twin for Predictive Maintenance")
st.write("""
This dashboard predicts **Remaining Useful Life (RUL)** and **Degradation Rate (DR)**  
using your **Multi-Task LSTM model** trained on NASA CMAPSS data.
""")

uploaded = st.file_uploader("📤 Upload engine sensor file (CMAPSS test_FD00X.txt or CSV)", type=["csv", "txt"])

if uploaded is not None:
    # -----------------------------------------------------
    # READ FILE: handle .txt (whitespace) and .csv
    # -----------------------------------------------------
    if uploaded.name.lower().endswith(".txt"):
        raw_df = pd.read_csv(uploaded, sep=r"\s+", header=None)
    else:
        raw_df = pd.read_csv(uploaded)

    df = ensure_cmapss_columns(raw_df)

    st.subheader("📄 Uploaded Data Preview (first 10 rows)")
    st.dataframe(df.head(10))

    # -----------------------------------------------------
    # SCALE FEATURES
    # -----------------------------------------------------
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[FEATURES] = scaler.fit_transform(df_scaled[FEATURES])

    # -----------------------------------------------------
    # BUILD SEQUENCES
    # -----------------------------------------------------
    X = build_sequences(df_scaled)
    if X is None:
        st.error(f"File must contain at least {SEQ_LEN} time steps.")
        st.stop()

    st.success(f"✔ Generated {X.shape[0]} sequences from uploaded data.")

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------
    model = load_model()
    with torch.no_grad():
        preds = model(torch.tensor(X))
        preds = preds.numpy()

    rul_pred = preds[:, 0]
    dr_pred = preds[:, 1]

    # -----------------------------------------------------
    # HEALTH SCORE
    # -----------------------------------------------------
    latest_rul = float(rul_pred[-1])
    health_score = max(0, min(100, latest_rul / 125 * 100))   # 125 = RUL clip in training

    st.subheader("🟢 Engine Health Indicator")
    st.metric("Predicted RUL (cycles)", f"{latest_rul:.2f}")
    st.progress(int(health_score))
    st.write(f"Estimated Engine Health: **{health_score:.1f}%**")

    # -----------------------------------------------------
    # PLOTS
    # -----------------------------------------------------
    st.subheader("📊 RUL and Degradation Rate Predictions")

    col1, col2 = st.columns(2)

    # RUL timeline
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rul_pred, label="Predicted RUL", color="tab:blue")
        ax.set_title("Predicted RUL Over Time")
        ax.set_xlabel("Sequence Index")
        ax.set_ylabel("RUL (cycles)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

    # DR timeline
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(dr_pred, label="Predicted DR", color="tab:red")
        ax.set_title("Predicted Degradation Rate Over Time")
        ax.set_xlabel("Sequence Index")
        ax.set_ylabel("DR (ΔRUL)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

    # Combined dual-axis
    st.subheader("📈 Combined RUL + DR (Dual Axis View)")
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(rul_pred, label="RUL", color="tab:blue")
    ax1.set_xlabel("Sequence Index")
    ax1.set_ylabel("RUL (cycles)", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(dr_pred, label="DR", color="tab:red", alpha=0.6)
    ax2.set_ylabel("DR (ΔRUL)", color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")

    ax1.grid(True)
    fig.suptitle("RUL and Degradation Rate Over Time")
    st.pyplot(fig)

    # Degradation curve only
    st.subheader("📉 Degradation Curve (Digital Twin Behavior)")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(rul_pred, color="purple")
    ax.set_title("Engine Degradation Curve (RUL vs Time)")
    ax.set_xlabel("Sequence Index")
    ax.set_ylabel("RUL (cycles)")
    ax.grid(True)
    st.pyplot(fig)

# end of file

