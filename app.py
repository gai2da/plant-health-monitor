import streamlit as st
import numpy as np
from PIL import Image
import os
from tensorflow.keras.models import load_model

st.title("Plant Disease Classifier")
st.write("Upload a plant leaf image and the model will predict the disease.")

# Path
MODEL_PATH = 'models/plant_disease_model'

# PlantVillage classes
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___healthy', 'Cherry___Powdery_mildew',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___healthy',
    'Corn___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy',
    'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

# Load the model 
@st.cache_resource
def load_saved_model(path):
    if os.path.exists(path):
        return load_model(path)
    else:
        st.error(f"Saved model not found at {path}")
        return None

model = load_saved_model(MODEL_PATH)

# Upload image and predict
uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg","png"])
if uploaded_file and model is not None:
    image = Image.open(uploaded_file).convert("RGB").resize((128,128))
    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    pred_index = np.argmax(preds)
    pred_class = class_names[pred_index]
    confidence = np.max(preds) * 100

    # healthy or diseased
    status = "Healthy" if "healthy" in pred_class.lower() else "Diseased"
    status_color = "green" if status == "Healthy" else "red"


    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="Uploaded Leaf", use_column_width=True)
    with col2:
        st.markdown(
        f"<span style='font-size:28px; font-weight:bold;'>Status: <span style='color:{status_color};'>{status}</span></span>",
        unsafe_allow_html=True
    )
        st.write(f"Class: {pred_class}")
        st.write(f"Confidence: {confidence:.2f}%")
        if confidence < 70:
            st.warning("Model is unsure — the leaf may not belong to any known class.")
