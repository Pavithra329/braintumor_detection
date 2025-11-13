import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
import os

# --- Configuration ---
IMAGE_SIZE = 128
MODEL_PATH = "my_brain_tumor_classifier.keras"
TRAIN_DIR = "dataset/Training/"  # used to get class names dynamically

# --- Load Model ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# --- Get Class Names ---
class_names = sorted(os.listdir(TRAIN_DIR))

# --- Image Preprocessing ---
def preprocess_image(image):
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = ImageEnhance.Brightness(image).enhance(1.1)
    image = ImageEnhance.Contrast(image).enhance(1.1)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# --- Streamlit UI ---
st.set_page_config(page_title="🧠 Brain Tumor Detection", layout="centered")

st.title("🧠 Brain Tumor Detection using VGG16")
st.markdown("Upload a **scanned brain MRI image**, and the model will predict the tumor type.")

uploaded_file = st.file_uploader("📤 Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    # Preprocess and predict
    with st.spinner("Analyzing image..."):
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = np.max(predictions) * 100

    # Display result
    st.subheader("🩻 Prediction Result:")
    st.write(f"**Predicted Tumor Type:** `{predicted_class}`")
    st.write(f"**Model Confidence:** `{confidence:.2f}%`")

    # Add an interpretation message
    if "no" in predicted_class.lower() or "normal" in predicted_class.lower():
        st.success("✅ No tumor detected. The scan appears normal.")
    else:
        st.error("⚠️ Tumor detected. Please consult a medical professional.")

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and TensorFlow.")
