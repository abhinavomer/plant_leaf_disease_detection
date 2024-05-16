# Plant Leaf Disease Detection using CNN

![banner](https://github.com/abhinavomer/plant_leaf_disease_detection/blob/main/pldd.jpg)
## Overview

- The Plant Leaf Detection App is an interactive web application that enables users to detect diseases in plant leaves using convolutional neural networks (CNNs). By leveraging deep learning models trained on vast datasets of leaf images, the app can accurately identify various diseases in plant leaves. This early detection capability is crucial for ensuring crop health, preventing the spread of diseases, and minimizing economic losses for farmers. The app covers a range of crops, including potatoes, tomatoes, corn, apples, and bell peppers, offering tailored disease detection for each plant type.
- Different Data Augmentation techniques are used for better accuracy.

## Features

- **Navigation Menu**: Users can navigate between different plant types (potatoes, tomatoes, corn, apples, bell peppers) to detect diseases specific to each plant.
- **Image Upload**: Users can upload images of plant leaves directly to the application for disease detection.
- **Disease Detection**: The app utilizes pre-trained deep learning models to predict the presence of diseases in the uploaded plant leaf images.
- **Prediction Display**: The predicted disease class along with the confidence score is displayed to the user.

## Supported Plant Types and Diseases

### Potato Leaf Disease Detection
- Early Blight
- Late Blight

### Tomato Leaf Disease Detection
- Tomato Bacterial Spot
- Tomato Early Blight
- Tomato Late Blight
- Tomato Leaf Mold
- Tomato Septoria Leaf Spot
- Tomato Spider Mites Two-Spotted Spider Mite
- Tomato Target Spot
- Tomato Tomato Yellow Leaf Curl Virus
- Tomato Tomato Mosaic Virus

### Corn Leaf Disease Detection
- Blight
- Common Rust
- Gray Leaf Spot

### Apple Leaf Disease Detection
- Apple Apple Scab
- Apple Black Rot
- Apple Cedar Apple Rust

### Bell Pepper Leaf Disease Detection
- Pepper Bell Bacterial Spot

## Technologies Used

- **Streamlit**: An open-source Python library used for building interactive web applications.
- **NumPy**: A fundamental package for scientific computing with Python used for numerical operations.
- **PIL (Python Imaging Library)**: A library for opening, manipulating, and saving many different image file formats.
- **TensorFlow**: An open-source machine learning framework for building and deploying machine learning models.

## Usage

1. **Navigation**: Select the desired plant type from the navigation menu.
2. **Upload Image**: Click on the "Upload image" button to select and upload an image of a plant leaf.
3. **Prediction**: After uploading the image, click the "Submit" button to detect diseases in the plant leaf.
4. **View Result**: The application will display the predicted disease class along with the confidence score.

## Features
- **Disease Identification**: The CNN model recognizes multiple diseases, providing a diagnosis based on leaf images.
- **User Interface**: A streamlit app deployed, link :- https://abhi-plantleafdiseasedetection.streamlit.app/.

## Installation
Provide step-by-step instructions on setting up the project locally. For example:
```bash
pip install -r requirements.txt
