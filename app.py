import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Smart Waste Classifier", page_icon="♻️")

st.title("♻️ Smart Waste Classifier")
st.write("Upload an image of waste and the model will classify it!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Send to FastAPI
    with st.spinner("Classifying..."):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://127.0.0.1:8000/predict", files={"file": uploaded_file.getvalue()})
        result = response.json()

    # Show results
    st.success(f"Predicted Class: **{result['predicted_class'].upper()}**")
    st.info(f"Confidence: **{result['confidence']}**")

    # Show all probabilities
    st.subheader("All Probabilities:")
    for class_name, prob in result['all_probabilities'].items():
        st.progress(float(prob.strip('%')) / 100, text=f"{class_name}: {prob}")