"""Streamlit demonstration app for PneumonAi.

This interface is for educational demonstration only and is not a medical
Diagnostic device.
"""

from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)
MODEL_PATH = Path("artifacts/pneumonai.keras")

st.set_page_config(page_title="PneumonAi", page_icon="🫁", layout="centered")

st.title("🫁 PneumonAi")
st.subheader("Chest X-Ray Pneumonia Classification")
st.warning(
    "Educational/research demonstration only. This tool is not a medical "
    "diagnostic device and must not be used for clinical decisions."
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

uploaded = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    if not MODEL_PATH.exists():
        st.error("Model not found. Train the model first with Code/train_modern.py.")
        st.stop()

    model = load_model()
    resized = image.resize(IMG_SIZE)
    array = np.asarray(resized, dtype=np.float32)
    probability = float(model.predict(array[None, ...], verbose=0)[0][0])

    if probability >= 0.5:
        st.error(f"Model prediction: PNEUMONIA ({probability:.1%})")
    else:
        st.success(f"Model prediction: NORMAL ({1 - probability:.1%})")

    st.caption(
        "The probability shown is the model's output, not a clinical probability "
        "or diagnosis."
    )
