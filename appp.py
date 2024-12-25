import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
import pandas as pd
from PIL import Image

# Custom CSS for decoration
st.markdown("""
    <style>
        .stApp {
            background-image: url('https://images.unsplash.com/photo-1478760329108-5c3ed9d495a0?fm=jpg&q=60&w=3000&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8ZGFyayUyMGJhY2tncm91bmR8ZW58MHx8MHx8fDA%3D');
            background-size: cover;
        }
        .stButton>button {
            background-color: white;
            color: red;
            font-size: 16px;
            font-weight: bold;
            border: none;
        }
        .stButton>button:hover {
            background-color: #d1cd50;
            color: white;  /* Ensures white text on hover */
        }
        .stButton>button:active {
            background-color: white; /* A darker shade on active */
            color: white;  /* Ensure the text stays white when button is pressed */
        }
        .stButton>button:focus {
            outline: none;  /* Remove any outline on focus */
            box-shadow: none;  /* Remove the box-shadow that may appear on focus */
        }
        .stHeader {
            color: black;
        }
        .stImage {
            border: 5px solid #ddd;
            border-radius: 10px;
            padding: 10px;
        }
        .stTitle {
            font-family: 'Arial', sans-serif;
            font-size: 40px;
            color: white;
        }
        .stSubheader {
            color: white;
        }
        .stMarkdown, .stText {
            color: white;
            font-size:150px;
        }
        .stWarning, .stError, .stSuccess {
            color: white;
        }
        
        /* Custom table styling */
        table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 20px;
            font-size:14px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border: 1px solid white;
        }
        th {
            background-color: black;
            color: #f2f2f2;
            font-size: 16px;
        }
        td {
            background-color: #f2f2f2;
            color: red;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
    </style>
""", unsafe_allow_html=True)

# Load the trained ResNet model (load it once to avoid repeated loading)
if 'model' not in st.session_state:
    model_path = r'C:/Users/svgau/Downloads/eye_disease_classifier_finetuned.h5'  # Update this path
    try:
        model = load_model(model_path)  # Load the pre-trained model
        st.session_state.model = model  # Save the model in the session state for reuse
        st.success("Model loaded successfully!")
    except Exception as e:
        st.session_state.model = None
        st.error(f"Error loading model: {e}")

# Set up the class labels (modify according to your dataset)
class_labels = ['Cataract', 'Diabetic retinopathy', 'Glaucoma', 'Normal']  # Replace with actual label

# Disease-related information
disease_info = {
    'Cataract': {
        'precautions': 'Wear sunglasses to protect eyes from UV rays, regular eye exams.',
        'causes': 'Aging, trauma to the eye, excessive UV light exposure.',
        'next_steps': 'Consult an ophthalmologist for surgery or other treatments.'
    },
    'Diabetic retinopathy': {
        'precautions': 'Control blood sugar levels, regular eye check-ups.',
        'causes': 'High blood sugar levels, high blood pressure.',
        'next_steps': 'Consult an ophthalmologist for laser treatment or surgery.'
    },
    'Glaucoma': {
        'precautions': 'Regular eye exams, protect eyes from injury, avoid smoking.',
        'causes': 'Increased pressure in the eye, age, family history.',
        'next_steps': 'Consult an ophthalmologist for pressure-lowering medications or surgery.'
    },
    'Normal': {
        'precautions': 'Maintain a healthy lifestyle, avoid smoking, protect eyes from UV light.',
        'causes': 'Healthy eye condition, no major issues.',
        'next_steps': 'No treatment needed, continue regular eye check-ups.'
    }
}

# Initialize session state for patient records
if 'patient_records' not in st.session_state:
    st.session_state.patient_records = pd.DataFrame(columns=[
        'Patient Name', 'Patient ID', 'Age', 'Gender', 
        'Left Eye Disease', 'Left Eye Confidence', 
        'Right Eye Disease', 'Right Eye Confidence'
    ])

# Streamlit App Setup
st.title('MEDSCAN - A Eye Disease Classifier')
st.header("Upload Eye Images to Predict Eye Diseases")

# Patient Information Input
st.sidebar.header("Patient Information")
patient_name = st.sidebar.text_input("Patient Name")
patient_id = st.sidebar.text_input("Patient ID")
age = st.sidebar.number_input("Age", min_value=0)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# Image Upload Options for Left and Right Eye
left_eye_image = st.file_uploader("Upload Left Eye Image", type=["jpg", "png", "jpeg"], key="left_eye")
right_eye_image = st.file_uploader("Upload Right Eye Image", type=["jpg", "png", "jpeg"], key="right_eye")

# Function to preprocess and predict the image
def predict_disease(image):
    try:
        # Preprocess the image
        img = Image.open(image)
        img = img.resize((224, 224))  # Resize image to fit ResNet input size
        img_array = np.array(img)  # Convert image to numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)  # Preprocess image
        
        # Extract features using ResNet50 model
        base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        base_model.trainable = False  # Freeze the ResNet50 base model
        features = base_model.predict(img_array)  # Extract features
        features = features.reshape(features.shape[0], -1)  # Flatten the 3D output to 1D
        
        # Predict the disease
        predictions = st.session_state.model.predict(features)
        predicted_class_idx = np.argmax(predictions)
        predicted_class = class_labels[predicted_class_idx]
        confidence = np.max(predictions) * 100
        
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        return None, None

# Add a Submit button
submit_button = st.button("Submit", key="submit")

# Predict diseases only when the Submit button is pressed and both images are uploaded
if submit_button:
    if left_eye_image is not None and right_eye_image is not None:
        # Predict diseases for the left and right eye images
        left_eye_disease, left_eye_confidence = predict_disease(left_eye_image)
        right_eye_disease, right_eye_confidence = predict_disease(right_eye_image)

        # Prepare patient details and prediction results
        patient_data = {
            'Patient Name': [patient_name],
            'Patient ID': [patient_id],
            'Age': [age],
            'Gender': [gender],
            'Left Eye Disease': [left_eye_disease],
            'Left Eye Confidence': [f"{left_eye_confidence:.2f}%"],
            'Right Eye Disease': [right_eye_disease],
            'Right Eye Confidence': [f"{right_eye_confidence:.2f}%"]
        }
        
        # Append new data to session state
        new_patient_df = pd.DataFrame(patient_data)
        st.session_state.patient_records = pd.concat([st.session_state.patient_records, new_patient_df], ignore_index=True)

        # Display patient details and predictions
        st.subheader("Patient Details and Predictions")
        st.write(st.session_state.patient_records)  # Display current records

        # Display disease-related information for each eye
        if left_eye_disease:
            disease = disease_info.get(left_eye_disease, {})
            if disease:
                st.subheader("Left Eye - Additional Information:")
                st.write(f"**Precautions**: {disease['precautions']}")
                st.write(f"**Causes**: {disease['causes']}")
                st.write(f"**Next Steps**: {disease['next_steps']}")
        
        if right_eye_disease:
            disease = disease_info.get(right_eye_disease, {})
            if disease:
                st.subheader("Right Eye - Additional Information:")
                st.write(f"**Precautions**: {disease['precautions']}")
                st.write(f"**Causes**: {disease['causes']}")
                st.write(f"**Next Steps**: {disease['next_steps']}")
        
    else:
        st.warning("Please upload images for both eyes before submitting.")
else:
    st.warning("Please upload images and click 'Submit' to get predictions.")

# Deleting patient record
if st.session_state.patient_records.shape[0] > 0:
    patient_to_delete = st.selectbox("Select Patient Record to Delete", st.session_state.patient_records['Patient Name'].tolist())
    if st.button("Delete Patient Record"):
        st.session_state.patient_records = st.session_state.patient_records[st.session_state.patient_records['Patient Name'] != patient_to_delete]
        st.success(f"Patient record for {patient_to_delete} has been deleted.")

# Always display patient records
if len(st.session_state.patient_records) > 0:
    st.subheader("Previous Patient Records")
    st.write(st.session_state.patient_records)  # Display patient records
