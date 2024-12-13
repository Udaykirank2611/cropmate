import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load your model
@st.cache_resource
def load_trained_model():
    return load_model("cnn_model.h5")

model = load_trained_model()

# Define a function to preprocess the uploaded image
def preprocess_image(image, target_size):
    image = image.resize(target_size)  # Resize the image
    image = img_to_array(image)  # Convert to array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize the image
    return image

# Streamlit UI
st.title("Image Classification with ML Model")
st.write("Upload an image, and the ML model will classify it!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = load_img(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image, target_size=(224, 224))  # Change target size as per your model

    # Make predictions
    prediction = model.predict(processed_image)
    st.write("Prediction:", np.argmax(prediction, axis=1))  # Adjust for your model's output
