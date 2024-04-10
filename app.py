import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
rad=st.sidebar.radio("Navigation",['Home','Potato Leaf Disease Detection','Apple Leaf Disease Detection','Bell Pepper Leaf Disease Detection','Corn Leaf Disease Detection'])
if rad=='Home':
    st.title('Plant Leaf Dection App')
    st.image("pldd.jpg")
    st.write("Welcome to the app!")
    st.write("Convolutional Neural Networks (CNNs) have revolutionized the field of image recognition and are now being applied to the critical task of plant leaf disease detection. By training on vast datasets of leaf images, CNNs learn to identify intricate patterns and anomalies that signify various diseases. This automated detection is not only remarkably accurate but also swift, enabling early intervention and treatment. The implications for agriculture are profound, as early disease detection is key to maintaining healthy crops, ensuring food security, and reducing economic losses for farmers. The use of CNNs in this domain exemplifies the transformative potential of artificial intelligence in addressing real-world challenge")
    st.balloons()
if rad=='Potato Leaf Disease Detection':

    st.image("pl.jpg")
    st.title("Potato Leaf Disease Detection")
    st.write("Disease that can be detected are:-['Early Blight','Late Blight']")
    image=st.file_uploader("Upload image")
    CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
    MODEL = tf.keras.models.load_model("potatoes_model.h5")
    if st.button("Submit"):
        size=(256,256)
        image = np.array((Image.open(image)).resize(size))
        img_batch = np.expand_dims(image, 0)
        
        predictions = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = (np.max(predictions[0])*100)
        st.write("Class:",predicted_class,)
        st.write("Confidence:",confidence)
if rad=='Corn Leaf Disease Detection':

    st.image("cl.jpg")
    st.title("Corn Leaf Disease Detection")
    st.write("Disease that can be detected are:-['Blight', 'Common_Rust', 'Gray_Leaf_Spot']")
    image=st.file_uploader("Upload image")
    CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
    MODEL = tf.keras.models.load_model("corn_model.h5")
    if st.button("Submit"):
        size=(256,256)
        image = np.array((Image.open(image)).resize(size))
        img_batch = np.expand_dims(image, 0)
        
        predictions = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]*100)
        st.write("Class:",predicted_class,)
        st.write("Confidence:",confidence)
if rad=='Apple Leaf Disease Detection':

    st.image("al.jpg")
    st.title("Apple Leaf Disease Detection")
    st.write("Disease that can be detected are:-['Apple___Apple_scab', 'Apple___Black_rot','Apple___Cedar_apple_rust']")
    image=st.file_uploader("Upload image")
    CLASS_NAMES = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy']
    MODEL = tf.keras.models.load_model("apple_model.h5")
    if st.button("Submit"):
        size=(256,256)
        image = np.array((Image.open(image)).resize(size))
        img_batch = np.expand_dims(image, 0)
        
        predictions = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence =(np.max(predictions[0])*100)
        st.write("Class:",predicted_class,)
        st.write("Confidence:",confidence)
if rad=='Bell Pepper Leaf Disease Detection':

    st.image("bpl.jpg")
    st.title("Bell Pepper Leaf Disease Detection")
    st.write("Disease that can be detected are:-['Pepper__bell___Bacterial_spot']")
    image=st.file_uploader("Upload image")
    CLASS_NAMES = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy']
    MODEL = tf.keras.models.load_model("bell.h5")
    if st.button("Submit"):
        size=(256,256)
        image = np.array((Image.open(image)).resize(size))
        img_batch = np.expand_dims(image, 0)
        
        predictions = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence =(np.max(predictions[0])*100)
        st.write("Class:",predicted_class,)
        st.write("Confidence:",confidence)
