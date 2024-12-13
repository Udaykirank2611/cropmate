import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Load the pre-trained model
MODEL_PATH = 'cnn_model.h5'
model = load_model(MODEL_PATH)

# Define categories (same order as used during model training)
categories = ['Tomato verticulium wilt', 'Cassava green mite', 'Cassava mosaic', 'Cashew red rust', 'Cashew gumosis',
              'Tomato healthy', 'Cassava brown spot', 'Cassava bacterial blight', 'Maize leaf beetle', 'Cassava healthy',
              'Maize leaf spot', 'Maize healthy', 'Tomato leaf blight', 'Cashew healthy', 'Cashew leaf miner',
              'Maize streak virus', 'Tomato septoria leaf spot', 'Maize leaf blight',
              'Cashew anthracnose', 'Tomato leaf curl', 'Maize fall armyworm']

# Function to preprocess input image
def preprocess_image(image, image_size=(128, 128)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, image_size)  # Resize to match model input size
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=-1)  # Add channel dimension for grayscale
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app layout
st.title("Crop Pest and Disease Detection")
st.write("Upload an image of a crop to detect potential pests or diseases.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Run prediction
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class] * 100

    # Display the result
    st.subheader("Prediction")
    st.write(f"Predicted Class: {categories[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}%")
