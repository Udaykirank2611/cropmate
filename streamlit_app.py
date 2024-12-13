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
            "[Learn more](https://extension.psu.edu/tomato-diseases-verticillium-wilt)"),
        1: ("**Cassava Green Mite**: A pest causing yellowing and curling of cassava leaves, reducing photosynthesis.\n\n"
            "**Preventive Measures**:\n"
            "- Introduce natural predators.\n"
            "- Use pest-resistant cassava varieties.\n"
            "- Avoid over-fertilization.\n\n"
            "[Learn more](https://www.cabi.org/isc/datasheet/11175)"),
        2: ("**Cassava Mosaic**: A viral disease spread by whiteflies, causing leaf distortion and stunted plants.\n\n"
            "**Preventive Measures**:\n"
            "- Use disease-free planting materials.\n"
            "- Control whiteflies with biological or chemical methods.\n"
            "- Practice intercropping and crop rotation.\n\n"
            "[Learn more](https://www.cabi.org/isc/datasheet/40510)"),
        3: ("**Cashew Red Rust**: A fungal disease causing reddish patches on leaves and twigs.\n\n"
            "**Preventive Measures**:\n"
            "- Apply fungicides like copper oxychloride.\n"
            "- Prune infected parts.\n"
            "- Ensure proper plant spacing for ventilation.\n\n"
            "[Learn more](https://agritech.tnau.ac.in)"),
        4: ("**Cashew Gumosis**: A fungal disease causing gum exudation from stems and branches.\n\n"
            "**Preventive Measures**:\n"
            "- Use disease-free seedlings.\n"
            "- Remove and burn infected parts.\n"
            "- Maintain proper soil health and drainage.\n\n"
            "[Learn more](https://www.bioversityinternational.org)"),
        5: ("**Tomato Healthy**: Your tomato plant shows no signs of disease or pests. Keep up the good work!\n\n"
            "[Learn more about tomato care](https://extension.umn.edu/vegetables/growing-tomatoes)"),
        6: ("**Cassava Brown Spot**: A fungal disease causing brown lesions on leaves, reducing yields.\n\n"
            "**Preventive Measures**:\n"
            "- Use resistant varieties.\n"
            "- Remove and destroy infected plant parts.\n"
            "- Improve plant nutrition.\n\n"
            "[Learn more](https://www.cabi.org/isc/datasheet/11152)"),
        7: ("**Cassava Bacterial Blight**: A bacterial disease causing wilting, leaf blight, and stem dieback.\n\n"
            "**Preventive Measures**:\n"
            "- Use certified disease-free cuttings.\n"
            "- Control insect vectors.\n"
            "- Avoid waterlogged soils.\n\n"
            "[Learn more](https://www.fao.org)"),
        8: ("**Maize Leaf Beetle**: A pest feeding on maize leaves, causing reduced photosynthesis.\n\n"
            "**Preventive Measures**:\n"
            "- Use insect-resistant maize varieties.\n"
            "- Apply biological controls like neem-based products.\n"
            "- Encourage natural predators.\n\n"
            "[Learn more](https://www.cabi.org/isc/datasheet/12345)"),
        9: ("**Cassava Healthy**: Your cassava plant appears healthy and free from diseases or pests. Maintain good practices!\n\n"
            "[Learn more about cassava care](https://www.cassavahub.com)"),
        10: ("**Maize Leaf Spot**: A fungal disease causing dark lesions on leaves, affecting photosynthesis.\n\n"
             "**Preventive Measures**:\n"
             "- Use resistant maize varieties.\n"
             "- Apply fungicides as needed.\n"
             "- Avoid overhead irrigation.\n\n"
             "[Learn more](https://www.apsnet.org)"),
        11: ("**Maize Healthy**: Your maize crop is in good condition. Continue practicing good agricultural techniques!\n\n"
             "[Learn more about maize care](https://www.maizeinternational.com)"),
        12: ("**Tomato Leaf Blight**: A fungal disease causing leaf browning, wilting, and fruit drop.\n\n"
             "**Preventive Measures**:\n"
             "- Use fungicides to manage infections.\n"
             "- Ensure proper plant spacing for airflow.\n"
             "- Remove and destroy infected leaves.\n\n"
             "[Learn more](https://www.extension.psu.edu)"),
        13: ("**Cashew Healthy**: Your cashew plant looks healthy. Keep up with good management practices!\n\n"
             "[Learn more about cashew care](https://www.cashewinfo.com)"),
        14: ("**Cashew Leaf Miner**: An insect pest that tunnels into cashew leaves, causing discoloration and reduced growth.\n\n"
             "**Preventive Measures**:\n"
             "- Monitor regularly for infestations.\n"
             "- Use biological control agents.\n"
             "- Avoid excessive nitrogen fertilizers.\n\n"
             "[Learn more](https://extension.umn.edu)"),
        15: ("**Maize Streak Virus**: A viral disease causing streaks on leaves and stunted growth, transmitted by leafhoppers.\n\n"
             "**Preventive Measures**:\n"
             "- Plant resistant maize varieties.\n"
             "- Control leafhopper populations.\n"
             "- Maintain field sanitation.\n\n"
             "[Learn more](https://en.wikipedia.org/wiki/Maize_streak_virus)"),
        16: ("**Tomato Septoria Leaf Spot**: A fungal disease causing small, circular spots on leaves, leading to defoliation.\n\n"
             "**Preventive Measures**:\n"
             "- Remove and destroy infected leaves.\n"
             "- Apply fungicides.\n"
             "- Avoid overhead watering.\n\n"
             "[Learn more](https://extension.umn.edu)"),
        17: ("**Maize Leaf Blight**: A fungal disease causing elongated lesions on leaves, reducing photosynthesis.\n\n"
             "**Preventive Measures**:\n"
             "- Use resistant maize hybrids.\n"
             "- Rotate crops to break the disease cycle.\n"
             "- Apply appropriate fungicides.\n\n"
             "[Learn more](https://www.apsnet.org)"),
        18: ("**Cashew Anthracnose**: A fungal disease causing black lesions on leaves, flowers, and fruits.\n\n"
             "**Preventive Measures**:\n"
             "- Apply copper-based fungicides.\n"
             "- Prune and destroy infected parts.\n"
             "- Improve field hygiene.\n\n"
             "[Learn more](https://www.cashewinfo.com)"),
        19: ("**Tomato Leaf Curl**: A viral disease transmitted by whiteflies, causing leaf curling and stunted growth.\n\n"
             "**Preventive Measures**:\n"
             "- Use resistant varieties.\n"
             "- Control whiteflies with insecticides.\n"
             "- Remove and destroy infected plants.\n\n"
             "[Learn more](https://www.extension.psu.edu)"),
        20: ("**Maize Fall Armyworm**: A destructive pest feeding on maize leaves and kernels.\n\n"
             "**Preventive Measures**:\n"
             "- Regularly monitor fields for signs of infestation.\n"
             "- Apply biological controls like Bacillus thuringiensis (Bt).\n"
             "- Practice intercropping to reduce pest spread.\n\n"
             "[Learn more](https://www.fao.org)"),
    }
    return explanations.get(category, "Unknown category. Please verify the model output.")


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
st.title("Detection Of Crop Disease with ML")
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
        a = np.argsort(prediction[0])[-2]
        categories = {0:'Tomato verticulium wilt', 1:'Cassava green mite', 2:'Cassava mosaic',3:'Cashew red rust',4:'Cashew gumosis',
              5:'Tomato healthy',6:'Cassava brown spot',7:'Cassava bacterial blight',8:'Maize leaf beetle',9:'Cassava healthy',
              10:'Maize leaf spot',11:'Maize healthy',12:'Tomato leaf blight',13:'Cashew healthy',14:'Cashew leaf miner',
              15:'Maize streak virus',16:'Tomato septoria leaf spot',17:'Maize leaf blight',
              18:'Cashew anthracnose',19:'Tomato leaf curl',20:'Maize fall armyworm'}
        st.write("Predicted Class:", categories[a])
        st.write(explain_disease(a))
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
