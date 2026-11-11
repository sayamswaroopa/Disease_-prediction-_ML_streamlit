import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

# Page Configuration
st.set_page_config(page_title="Disease Prediction from Medical Data", layout="wide")

st.title("Disease Prediction from Medical Data of Patient")
st.write("An AI/ML web application built with Python, Scikit-learn, and Streamlit to predict diseases based on patient symptoms.")

# 1. Upload Training Data (CSV)
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Training Data (CSV)", type=["csv"])

if uploaded_file is not None:
    # Read the dataset
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Show Training Data Sample")
    st.dataframe(df.head())
    
    # Assuming the last column is 'prognosis' (Target) and other columns are symptoms (Features)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_test_split=0.2, random_state=42)
    
    # Train Decision Tree Model
    model = DecisionTreeClassifier(criterion='gini', random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate Accuracy
    accuracy = model.score(X_test, y_test) * 100
    
    if st.sidebar.button("Test Model Accuracy"):
        st.sidebar.success(f"Model Accuracy: {accuracy:.2f}%")
        
    # Save the trained model using joblib
    joblib.dump(model, "disease_model.pkl")
    
    # 2. Select Symptoms for Prediction
    st.subheader("Select Symptoms")
    symptoms_list = list(X.columns)
    
    selected_symptoms = st.multiselect(
        "Choose the symptoms you are experiencing:",
        options=symptoms_list
    )
    
    # Predict Button
    if st.button("Predict Disease"):
        if len(selected_symptoms) == 0:
            st.warning("Please select at least one symptom for prediction.")
        else:
            # Create a dynamic input array filled with 0s
            input_data = np.zeros((1, len(symptoms_list)))
            
            # Set 1 for selected symptoms
            for symptom in selected_symptoms:
                symptom_index = symptoms_list.index(symptom)
                input_data[0, symptom_index] = 1
                
            # Make Prediction
            prediction = model.predict(input_data)
            
            # Display Result
            st.success(f"Predicted Disease: **{prediction[0]}**")
            
else:
    st.info("Please upload a training CSV file in the sidebar to start the application.")
    st.write("Your CSV should contain symptoms as columns (with 0 or 1 values) and the last column as the disease name.")
