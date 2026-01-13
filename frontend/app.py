import streamlit as st
import requests
from PIL import Image
import io

# --- FastAPI backend URL (LIVE) ---
API_URL = "https://leaf-disease-classifier.onrender.com/predict"

# st.title("Leaf Disease Classifier 🍀")
# st.write("Upload a potato leaf image to detect disease.")
st.title("Plant Leaf Disease Classifier 🌱")
st.write("Upload a leaf image (Potato / Tomato / Pepper) to detect disease.")


# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Classifying...")
    
    # Convert image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format=image.format)
    img_bytes = img_bytes.getvalue()
    
    # Send POST request to FastAPI
    files = {"file": (uploaded_file.name, img_bytes, uploaded_file.type)}
    try:
        response = requests.post(API_URL, files=files)
        response.raise_for_status()

        result = response.json()
        st.success(f"Prediction: {result['class']}")
        st.info(f"Confidence: {result['confidence']*100:.2f}%")
    except requests.exceptions.RequestException as e:
        st.error(f"Error in prediction: {e}")
