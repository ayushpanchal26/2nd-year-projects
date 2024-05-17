import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from PIL import Image
import numpy as np
import joblib
import streamlit as st

# Load the model
model = tf.keras.models.load_model('modelcancerlung.h5')

# Define the predict function
def predict_image(img):
    # Resize the image to match the input shape of the model
    img = img.resize((224, 224))
    # Convert the image to RGB mode (if not already)
    img = img.convert('RGB')
    # Convert the image to a numpy array
    x = tf.keras.preprocessing.image.img_to_array(img)
    # Expand the dimensions to create a batch of size 1
    x = np.expand_dims(x, axis=0)
    # Normalize the image
    x = x / 255.0
    # Make prediction
    prediction = model.predict(x)
    return prediction

# Streamlit UI
st.title('Chest Cancer Type Detection')
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Make prediction
    prediction = predict_image(image)
    classes = ["AdenocarcinomaChest Lung Cancer", "Large cell carcinoma Lung Cancer", "NO Lung Cancer/ NORMAL", "Squamous cell carcinoma Lung Cancer"]
    predicted_class = classes[np.argmax(prediction)]
    
    st.write('Prediction:', predicted_class)




















# ayush panchal