import streamlit as st
import torch
from PIL import Image

from model import SwinClassifier
from utils.preprocessing import get_transform

st.set_page_config(page_title="DR Severity Grading", layout="centered")

CLASS_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
MODEL_PATH = "models/best_model_swin_gan_cpu.pt"

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = SwinClassifier().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, device


model, device = load_model()
transform = get_transform()

st.title("Diabetic Retinopathy Severity Grading")
st.write(
    "Upload a retinal fundus image to classify diabetic retinopathy severity "
    "(No DR → Proliferative DR)."
)

uploaded_file = st.file_uploader("Upload fundus image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    with st.spinner("Analyzing..."):
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_class = torch.argmax(probs).item()

    st.subheader(f"Prediction: {CLASS_NAMES[pred_class]}")
    st.write("Confidence per class:")
    for i, name in enumerate(CLASS_NAMES):
        st.progress(probs[i].item(), text=f"{name}: {probs[i].item():.2%}")

    st.caption(
        "This tool is for educational/portfolio purposes only and is not a "
        "diagnostic medical device."
    )
else:
    st.info("Upload an image to get a prediction.")
