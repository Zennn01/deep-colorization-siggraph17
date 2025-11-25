import streamlit as st
import numpy as np
import cv2
import torch
from PIL import Image

from colorizers.siggraph17 import siggraph17
from colorizers.util import load_img, preprocess_img, postprocess_tens
from enhance import auto_enhance


# ============================ MODEL =============================
model = siggraph17(pretrained=True).eval()


def run_colorizer(img_pil):
    """Convert PIL to deep-learning output"""
    img_path = "temp_input.jpg"
    img_pil.save(img_path)

    img = load_img(img_path)
    tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))

    with torch.no_grad():
        out_ab = model(tens_l_rs).cpu()

    out = postprocess_tens(tens_l_orig, out_ab)
    out_np = (out * 255).astype(np.uint8)

    return out_np


# ============================ STREAMLIT UI =============================
st.title("AI Image Colorizer - SIGGRAPH17 + Enhance")

uploaded = st.file_uploader("Upload Foto", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded:
    img_pil = Image.open(uploaded).convert("RGB")

    # Show original
    st.subheader("Original")
    st.image(img_pil, width=350)

    # PROSES BUTTON
    if st.button("Proses Warna (SIGGRAPH17)"):
        with st.spinner("Memproses..."):
            sig_np = run_colorizer(img_pil)

        st.session_state["sig"] = sig_np

    # Tampilkan hasil SIGGRAPH kalau sudah ada
    if "sig" in st.session_state:
        st.subheader("Hasil SIGGRAPH17")
        st.image(st.session_state["sig"], width=350)

        # Tombol enhance
        if st.button("Enhance"):
            with st.spinner("Enhance..."):
                sig_bgr = cv2.cvtColor(st.session_state["sig"], cv2.COLOR_RGB2BGR)
                enhanced = auto_enhance(sig_bgr)
                enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

            st.session_state["enhanced"] = enhanced_rgb

    # Tampilkan hasil enhance kalau ada
    if "enhanced" in st.session_state:
        st.subheader("Enhanced Result")
        st.image(st.session_state["enhanced"], width=350)


# ============================ 3 PANEL BERDAMPINGAN =============================
if uploaded and "sig" in st.session_state:
    st.subheader("Perbandingan 3 Model")

    col1, col2, col3 = st.columns(3)

    col1.image(img_pil, caption="Original", use_column_width=True)
    col2.image(st.session_state["sig"], caption="SIGGRAPH17", use_column_width=True)

    if "enhanced" in st.session_state:
        col3.image(st.session_state["enhanced"], caption="Enhanced", use_column_width=True)
    else:
        col3.write("Klik tombol *Enhance*")
