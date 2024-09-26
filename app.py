import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
rad=st.sidebar.radio("Navigation",['Home','Potato Leaf Disease Detection','Apple Leaf Disease Detection','Corn Leaf Disease Detection','Tomato Disease Detection','Bell Pepper Leaf Disease Detection'])
if rad=='Home':
    st.title('Plant Leaf Dection App')
    st.image("pldd.jpg")
    st.write("Welcome to the app!")
    st.balloons()
if rad=='Potato Leaf Disease Detection':

    st.image("pl.jpg")
    st.title("Potato Leaf Disease Detection")
    st.write("Disease that can be detected are:-['Early Blight','Late Blight']")
    image=st.file_uploader("Upload image")
    CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
    MODEL = tf.keras.models.load_model(r"C:\Users\abhin\MACHINE LEARNING\ML_TENSORFLOW\Projects\Plant Leaf Detection\Disease app\1")
    if st.button("Submit"):
        size=(256,256)
        image = np.array((Image.open(image)).resize(size))
        img_batch = np.expand_dims(image, 0)
        
        predictions = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        st.write("Class:",predicted_class,)
        st.write("Confidence:",confidence)
        if predicted_class=="Early Blight":
            st.write("Pest: Alternaria Solani, a type of fungus ")
        elif predicted_class=="Late Blight":
            st.write("Pest: Phytophthora Infestans, a water mold")
        else:
            pass
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
        if predicted_class == "Blight":
            st.write("Pest: Stewart's Bacteria, Colletorichum Fungus, Bipolaris Maydis Fungus, Exserohilum turcicum Fungus")
        elif predicted_class == "Common_Rust":
            st.write("Pest: Puccinia Sorghi , a type of rust fungus")
        elif predicted_class == "Gray_Leaf_Spot":
            st.write("Pest: Cercospora zeae-maydis Fungus")
        else:
            pass

if rad=='Tomato Disease Detection':

    st.image("tl.jpg")
    st.title("Tomato Disease Detection")
    st.write("Disease that can be detected are:-['Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus']")
    image=st.file_uploader("Upload image")
    CLASS_NAMES = ['Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus']
    MODEL = tf.keras.models.load_model("tomato_model.h5")
    if st.button("Submit"):
        size=(128,128)
        image = np.array((Image.open(image)).resize(size))
        img_batch = np.expand_dims(image, 0)
        
        predictions = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]*100)
        st.write("Class:",predicted_class,)
        st.write("Confidence:",confidence)
        if predicted_class == "Tomato___Bacterial_spot":
            st.write("Pest: Xanthomonas campestris pv. Vesicatoria ")
        elif predicted_class == "Tomato___Early_blight":
            st.write("Pest: Alternaria tomatophila and Alternaria solani")
        elif predicted_class == "Tomato___Late_blight":
            st.write("Pest: Phytophthora infestanst")
        elif predicted_class == "Tomato___Leaf_Mold":
            st.write("Pest: Mycovellosiella fulva")
        elif predicted_class == "Tomato___Septoria_leaf_spot":
            st.write("Pest: Septoria lycopersici")
        elif predicted_class == "Tomato___Spider_mites Two-spotted_spider_mite":
            st.write("Pest: Two-Spotted Spider Mites")
        elif predicted_class == "Tomato___Target_Spot":
            st.write("Pest: Corynespora cassiicola")
        elif predicted_class == "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
            st.write("Pest: Silverleaf whitefly")
        elif predicted_class == "Tomato___Tomato_mosaic_virus":
            st.write("Pest: Human Activity, root,leaf and seed debris")
        else:
            pass

if rad=='Apple Leaf Disease Detection':

    st.image("al.jpg")
    st.title("Apple Leaf Disease Detection")
    st.write("Disease that can be detected are:-['Apple___Apple_scab', 'Apple___Black_rot','Apple___Cedar_apple_rust']")
    image=st.file_uploader("Upload image")
    CLASS_NAMES = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy']
    MODEL = tf.keras.models.load_model(r"C:\Users\abhin\MACHINE LEARNING\ML_TENSORFLOW\Projects\Plant Leaf Detection\Disease app\apple_model.h5")
    if st.button("Submit"):
        size=(256,256)
        image = np.array((Image.open(image)).resize(size))
        img_batch = np.expand_dims(image, 0)
        
        predictions = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = (np.max(predictions[0])*100)
        st.write("Class:",predicted_class,)
        st.write("Confidence:",confidence)
        if predicted_class == "Apple___Apple_scab":
            st.write("Pest: Venturia inaequalis")
        elif predicted_class == "Apple___Black_rot":
            st.write("Pest: Diplodia seriata")
        elif predicted_class == "Apple___Cedar_apple_rust":
            st.write("Pest: Gymnosporangium juniperi-virginianae , a fungal pathogen")
        else:
            pass

if rad=='Bell Pepper Leaf Disease Detection':

    st.image("bpl.jpg")
    st.title("Bell Pepper Leaf Disease Detection")
    st.write("Disease that can be detected are:-['Pepper__bell___Bacterial_spot']")
    image=st.file_uploader("Upload image")
    CLASS_NAMES = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy']
    MODEL = tf.keras.models.load_model(r"C:\Users\abhin\MACHINE LEARNING\ML_TENSORFLOW\Projects\Plant Leaf Detection\Disease app\3")
    if st.button("Submit"):
        size=(256,256)
        image = np.array((Image.open(image)).resize(size))
        img_batch = np.expand_dims(image, 0)
        
        predictions = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        st.write("Class:",predicted_class,)
        st.write("Confidence:",confidence)
        if predicted_class=="Pepper__bell___Bacterial_spot":
            st.write("Pest: Xanthomonas campestris pv. vesicatoria, a gram negative bacteria")
        else:
            pass
