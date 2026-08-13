"""
app.py -- Streamlit version
-----------------------------
A simple website: upload a chest X-ray, get a prediction + Grad-CAM heatmap.

Folder structure expected in the GitHub repo (all at ROOT level):
    app.py                  <- this file
    requirements.txt
    chest_xray_model.pth
    utils/
        model.py
        dataset.py
        gradcam.py
"""

import os
import streamlit as st
import torch
import numpy as np
import requests
from PIL import Image

from utils.model import load_model
from utils.dataset import eval_transforms
from utils.gradcam import GradCAM, overlay_heatmap

MODEL_PATH = "chest_xray_model.pth"
# Direct-download URL for the model file hosted on Hugging Face.
# If your file has a different name than chest_xray_model.pth, update this URL to match.
MODEL_URL = "https://huggingface.co/bach88/chest_xray/resolve/main/chest_xray_model.pth"
CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral_Pneumonia"]  # must match training order!

st.set_page_config(page_title="Chest X-ray Classifier", page_icon="🫁", layout="centered")


def download_model_if_needed():
    """
    On first run, the model file won't exist in the deployed app's storage
    (we're not uploading a 90MB+ file to GitHub). This downloads it once
    from Hugging Face and caches it on disk for subsequent runs.
    """
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (first run only, may take a minute)..."):
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)


@st.cache_resource
def get_model():
    download_model_if_needed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(MODEL_PATH, num_classes=len(CLASS_NAMES), device=device)
    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)
    return model, gradcam, device


st.title("🫁 Chest X-ray Abnormality Classifier")
st.caption(
    "Research/educational prototype. NOT a diagnostic device. "
    "Not validated for clinical use."
)

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    model, gradcam, device = get_model()

    image = Image.open(uploaded_file).convert("RGB")
    original_np = np.array(image.resize((224, 224)))

    input_tensor = eval_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    cam, predicted_idx = gradcam.generate(input_tensor)
    overlayed = overlay_heatmap(original_np, cam)

    predicted_class = CLASS_NAMES[predicted_idx]
    is_abnormal = predicted_class != "Normal"

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded X-ray", use_column_width=True)
    with col2:
        st.image(overlayed, caption="Grad-CAM: where the model looked", use_column_width=True)

    if is_abnormal:
        st.error(f"⚠️ Abnormality detected: **{predicted_class}**")
    else:
        st.success("✅ No abnormality detected (Normal)")

    st.subheader("Confidence scores")
    for i, cls in enumerate(CLASS_NAMES):
        st.write(f"{cls}: {probs[i]*100:.1f}%")
        st.progress(float(probs[i]))
else:
    st.info("Upload a chest X-ray image above to get a prediction.")
