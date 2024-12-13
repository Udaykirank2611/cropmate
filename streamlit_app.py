import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import ImageOps

# Load your trained model
@st.cache_resource
def load_trained_model():
    return load_model("model.h5")

model = load_trained_model()

# Preprocess the uploaded image
def preprocess_image(image, target_size, grayscale=False):
    if grayscale:
        image = ImageOps.grayscale(image)  # Convert to grayscale if needed
        image = image.resize(target_size)  # Resize the image
        image = img_to_array(image)  # Convert to array
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
    else:
        image = image.resize(target_size)  # Resize the image
        image = img_to_array(image)  # Convert to array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize the image
    return image

# Streamlit UI
st.title("Image Classification with ML Model")
st.write("Upload an image, and the ML model will classify it!")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = load_img(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Get the model's expected input shape
    model_input_shape = model.input_shape[1:3]  # Extract height and width

    # Check if the model expects grayscale input
    is_grayscale = model.input_shape[-1] == 1

    # Preprocess the image
    processed_image = preprocess_image(image, target_size=model_input_shape, grayscale=is_grayscale)

    # Predict using the model
    try:
        prediction = model.predict(processed_image)
        st.write("Prediction (Raw Output):", prediction)
        st.write("Predicted Class:", np.argmax(prediction, axis=1)[0])
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
