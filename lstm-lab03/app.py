import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_JIT"] = "0"

import torch
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime

from src.config import load_config
from src.vocabulary import Vocabulary
from src.model import SentimentModel
from src.gate_analysis import analyze_gates
from src.logger import setup_logging, get_logger

setup_logging("INFO", log_file="logs/app.log")
logger = get_logger("app")

st.set_page_config(page_title="LSTM Sentiment", page_icon="🎬", layout="wide")


@st.cache_resource
def load_model():
    cfg   = load_config("configs/lstm.yaml")
    vocab = Vocabulary.load(cfg["paths"]["artifacts"] + "/vocab.json")
    ckpt  = torch.load("artifacts/lstm_model.pt", weights_only=False, map_location="cpu")
    model = SentimentModel(**ckpt["model_cfg"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, vocab, cfg["data"]["max_len"]


def gate_heatmap(rows):
    """Vẽ heatmap forget gate và input gate theo từng token."""
    tokens   = [r["token"] for r in rows]
    f_values = [r["f_mean"] for r in rows]
    i_values = [r["i_mean"] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(tokens) * 0.55), 2.5))
    for ax, values, title, cmap in zip(
        axes,
        [f_values, i_values],
        ["Forget gate (f)", "Input gate (i)"],
        ["RdYlGn", "RdYlGn"],
    ):
        data = [values]
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
        ax.set_yticks([])
        ax.set_title(title, fontsize=10, pad=4)
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    fig.tight_layout()
    return fig


# --- UI ---
st.title("🎬 LSTM Sentiment Analysis")
st.caption("Custom LSTM implementation — Lab 3 Deep Learning")

model, vocab, max_len = load_model()

text = st.text_area(
    "Nhập review phim:",
    placeholder="This movie was absolutely wonderful...",
    height=120,
)

col1, col2 = st.columns([1, 3])
analyze = col1.button("Phân tích", type="primary", use_container_width=True)
show_gates = col2.checkbox("Hiển thị gate analysis", value=True)

if analyze and text.strip():
    with st.spinner("Đang phân tích..."):
        rows, label, prob = analyze_gates(model, text, vocab, max_len)
        logger.info("predict | label=%s | prob=%.4f | tokens=%d | text=%s",
                    label, prob, len(rows), text[:100].replace("\n", " "))

    # --- Kết quả chính ---
    st.divider()
    c1, c2, c3 = st.columns(3)
    emoji = "😊" if label == "positive" else "😞"
    c1.metric("Nhãn", f"{emoji} {label.upper()}")
    c2.metric("Confidence", f"{prob:.1%}")
    c3.metric("Số token", len(rows))

    # --- Gate analysis ---
    if show_gates and rows:
        st.subheader("Gate Analysis")
        st.caption("Forget gate cao → giữ thông tin cũ. Input gate cao → hấp thụ token mới mạnh.")

        fig = gate_heatmap(rows)
        st.pyplot(fig)
        plt.close(fig)

        # Bảng chi tiết
        with st.expander("Xem bảng chi tiết"):
            df = pd.DataFrame(rows)
            df["Δ‖c‖"] = df["delta_c"].apply(lambda x: f"{x:.3f}")
            df["f_mean"] = df["f_mean"].apply(lambda x: f"{x:.4f}")
            df["i_mean"] = df["i_mean"].apply(lambda x: f"{x:.4f}")
            st.dataframe(
                df[["token", "f_mean", "i_mean", "Δ‖c‖"]],
                use_container_width=True,
                hide_index=True,
            )

elif analyze and not text.strip():
    st.warning("Vui lòng nhập nội dung review.")

# --- Sidebar info ---
with st.sidebar:
    st.header("Thông tin model")
    st.markdown("""
    **Kiến trúc:** Custom LSTM (tự implement)

    **Dataset:** IMDb 25,000 reviews

    **Test accuracy:** 85.16%

    **Vocab size:** 20,000

    **Hidden size:** 128

    **Embed dim:** 64
    """)
    st.divider()
    st.caption("Lab 3 — MSc AI Engineering")
