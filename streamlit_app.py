import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import ImageOps
def explain_disease(category):
    explanations = {
        0: ("**Tomato Verticillium Wilt**: A fungal disease causing wilting, yellowing leaves, and stunted growth.\n\n"
            "**Preventive Measures**:\n"
            "- Use resistant tomato varieties.\n"
            "- Practice crop rotation.\n"
            "- Improve soil drainage.\n\n"
            "**Read More**:\n"
            "- [University of California Agriculture and Natural Resources](https://ipm.ucanr.edu/agriculture/tomato/verticillium-wilt/)\n"
            "- [Tomato Disease Guide by Cornell University](https://vegetablemdonline.ppath.cornell.edu/factsheets/Tomato_Verticillium.htm)"),
        1: ("**Cassava Green Mite**: A pest causing yellowing and curling of cassava leaves, reducing photosynthesis.\n\n"
            "**Preventive Measures**:\n"
            "- Introduce natural predators.\n"
            "- Use pest-resistant cassava varieties.\n"
            "- Avoid over-fertilization.\n\n"
            "**Read More**:\n"
            "- [International Institute of Tropical Agriculture (IITA)](https://www.iita.org/news-item/control-of-cassava-green-mites/)\n"
            "- [FAO Pest and Disease Guidelines](https://www.fao.org/agriculture/crops/thematic-sitemap/theme/pests/en/)"),
        2: ("**Cassava Mosaic**: A viral disease spread by whiteflies, causing leaf distortion and stunted plants.\n\n"
            "**Preventive Measures**:\n"
            "- Use disease-free planting materials.\n"
            "- Control whiteflies with biological or chemical methods.\n"
            "- Practice intercropping and crop rotation.\n\n"
            "**Read More**:\n"
            "- [IITA Cassava Mosaic Disease](https://www.iita.org/research/thematic-areas/crop-production-and-systems/cassava-mosaic-disease/)\n"
            "- [CGIAR Cassava Research Program](https://www.cgiar.org/impact/research/cassava-diseases/)"),
        3: ("**Cashew Red Rust**: A fungal disease causing reddish patches on leaves and twigs.\n\n"
            "**Preventive Measures**:\n"
            "- Apply fungicides like copper oxychloride.\n"
            "- Prune infected parts.\n"
            "- Ensure proper plant spacing for ventilation.\n\n"
            "**Read More**:\n"
            "- [Cashew Info](https://cashewinfo.com/)\n"
            "- [National Horticultural Board - Cashew Diseases](https://nhb.gov.in/)"),
        4: ("**Cashew Gumosis**: A fungal disease causing gum exudation from stems and branches.\n\n"
            "**Preventive Measures**:\n"
            "- Use disease-free seedlings.\n"
            "- Remove and burn infected parts.\n"
            "- Maintain proper soil health and drainage.\n\n"
            "**Read More**:\n"
            "- [Indian Council of Agricultural Research - Cashew Gumosis](https://icar.gov.in/)\n"
            "- [National Cashew Information Portal](https://cashew.gov.in/)"),
        5: ("**Tomato Healthy**: Your tomato plant shows no signs of disease or pests. Keep up the good work!\n\n"
            "**Tips**:\n"
            "- Monitor plants regularly.\n"
            "- Maintain soil fertility and organic matter.\n"
            "- Mulch to retain moisture.\n\n"
            "**Read More**:\n"
            "- [FAO Tomato Production Guide](https://www.fao.org/agriculture/tomatoes/production-guide/)"),
        # Add similar entries for categories 6 to 20
    }

    return explanations.get(category, "Unknown category. Please verify the category ID.")
# Load your trained model
@st.cache_resource
def load_trained_model():
    return load_model("cnn_model.h5")

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
        a = np.argmax(prediction, axis=1)[0]
        categories = {0:'Tomato verticulium wilt', 1:'Cassava green mite', 2:'Cassava mosaic',3:'Cashew red rust',4:'Cashew gumosis',
              5:'Tomato healthy',6:'Cassava brown spot',7:'Cassava bacterial blight',8:'Maize leaf beetle',9:'Cassava healthy',
              10:'Maize leaf spot',11:'Maize healthy',12:'Tomato leaf blight',13:'Cashew healthy',14:'Cashew leaf miner',
              15:'Maize streak virus',16:'Tomato septoria leaf spot',17:'Maize leaf blight',
              18:'Cashew anthracnose',19:'Tomato leaf curl',20:'Maize fall armyworm'}
        st.write("Predicted Class:", categories[a])
        st.write(explain_disease(a))
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
