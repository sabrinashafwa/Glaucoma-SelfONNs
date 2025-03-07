import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

model = load_model('C:/Users/LABKOM-12/Documents/saboy/SKRIPSI/self_onn_model.keras')

# Title of the app
st.title('glauGOma')

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    image_array = np.array(image)
    processed_image = preprocess_image(image_array)
    
    # Make prediction
    prediction = model.predict(processed_image)
    
    # Display the prediction
    if prediction[0][0] > 0.5:
        st.write('**Prediction:** Referable Glaucoma')
    else:
        st.write('**Prediction:** Non-Referable Glaucoma')
