import streamlit as stlt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cancer_prediction_model_with_random_forest import load_model as load_rf_model
from cancer_prediction_model_with_logistic import load_logistic_model
from selected_features import SELECTED_FEATURES, TARGET_COLUMN
from data_preprocessing import load_and_preprocess_data, encode_single_input
import os


#function to get input from user and parse it to the encoded model format
def get_user_input(data, encoding_rules):

    stlt.subheader("Enter Patient Information")
    
    user_input = {}
    
    # Create two columns for input fields to create horizontal grid sections
    col1, col2 = stlt.columns(2)
    
    # Get only the selected features, excluding the target to display in the above grids
    input_features = [col for col in SELECTED_FEATURES if col != TARGET_COLUMN]
    
    # Identify categorical to display them based on dropdown boxes
    categorical_features = list(encoding_rules.keys())
    
    # Create input fields for each feature
    for i, feature in enumerate(input_features):
        # Determine which column to place the field in
        current_col = col1 if i % 2 == 0 else col2
        
        if feature in categorical_features:
            # Create a drop down box for categorical variables
            unique_values = encoding_rules[feature]['categories']
            user_input[feature] = current_col.selectbox(
                f"{feature}",
                options=unique_values,
                key=f"input_{i}"
            )
        else:
            # For numerical variables, create range boxes
            if feature == 'Year of diagnosis':
                user_input[feature] = current_col.number_input(
                    f"{feature}",
                    min_value=2010,
                    max_value=2022,
                    value=2021,
                    step=1,
                    key=f"input_{i}"
                )
            elif "Regional nodes" in feature:
                user_input[feature] = current_col.number_input(
                    f"{feature}",
                    min_value=0,
                    max_value=50,
                    value=0,
                    step=1,
                    key=f"input_{i}"
                )
            elif "Histologic Type" in feature:
                user_input[feature] = current_col.number_input(
                    f"{feature}",
                    min_value=8000,
                    max_value=9000,
                    value=8500,
                    step=1,
                    key=f"input_{i}"
                )
            elif "Size" in feature:
                user_input[feature] = current_col.number_input(
                    f"{feature}",
                    min_value=0,
                    max_value=100,
                    value=10,
                    step=1,
                    key=f"input_{i}"
                )
            elif "Primary Site" in feature:
                user_input[feature] = current_col.number_input(
                    f"{feature}",
                    min_value=500,
                    max_value=509,
                    value=504,
                    step=1,
                    key=f"input_{i}"
                )

            elif "RX Summ" in feature:
                user_input[feature] = current_col.number_input(
                    f"{feature}",
                    min_value=0,
                    max_value=90,
                    value=22,
                    step=1,
                    key=f"input_{i}"
                )
            else:
                user_input[feature] = current_col.number_input(
                    f"{feature}",
                    min_value=0,
                    max_value=100,
                    value=10,
                    key=f"input_{i}"
                )
    
    return user_input

#make prediction based on user provided values in the application interface and user selected model
def make_prediction(model_info, user_input, encoding_rules):

    # Get the model
    model = model_info['model']
    
    # Encode the user input using the encoding rules established in the data preprocessing step
    encoded_input = encode_single_input(user_input, encoding_rules)
    
    # Create DataFrame with the encoded input
    input_df = pd.DataFrame([encoded_input])
    
    # If using logistic regression, apply scaling
    if 'scaler' in model_info:
        input_df = pd.DataFrame(model_info['scaler'].transform(input_df), columns=input_df.columns)
    
    # Ensure all required features are present
    model_features = model.feature_names_in_
    missing_features = set(model_features) - set(input_df.columns)
    if missing_features:
        stlt.error(f"Error: Missing required features: {missing_features}")
        return None, None
    
    # Reorder columns to match training data
    input_df = input_df[model_features]
    
    # Make prediction
    prediction = model.predict(input_df)
    probabilities = model.predict_proba(input_df)
    
    return prediction[0], probabilities[0]

def display_prediction_results(predicted_class, probabilities, unique_classes):
    #creates title for the prediction results section on the web app
    stlt.subheader("Prediction Results")
    
    # Create columns for displaying results
    col1, col2 = stlt.columns(2)
    
    with col1:
        stlt.write("**Predicted Outcome:**")
        stlt.write(predicted_class)
    
    with col2:
        stlt.write("**Prediction Probabilities:**")
        for class_label, prob in zip(unique_classes, probabilities):
            stlt.write(f"{class_label}: {prob:.2%}")

def main():
    # App title and description
    stlt.title("Cancer survivability rate prediction application")
    
    stlt.markdown("""
    This application predicts cancer survivability rate outcomes based on patient information using either a Random Forest
    or Logistic Regression model trained on the SEER (Surveillance, Epidemiology, and End Results) dataset.
    
    This version supports multiple models and uses preprocessed data with proper encoding of categorical variables.
    """)
    
    # Model selection in sidebar
    stlt.sidebar.title("Model Selection")
    model_type = stlt.sidebar.selectbox(
        "Select prediction model",
        ["Random Forest", "Logistic Regression"],
        help="Select which model to use for prediction"
    )
    
    # About section in sidebar
    stlt.sidebar.title("About")
    stlt.sidebar.info("""
    This application uses machine learning algorithms to predict cancer outcomes in terms of Patient Survivability rate.
    
    The classifier models are trained on the SEERstats breast cancer dataset from 2021 data. The dataset contains information about 
    cancer patients and their various demographics characteristics such as Race Ethnicily, Age, Income as well as tumor characteristics, treatments, and outcomes.
    
    **Target Variable**: SEER cause-specific death classification
    """)
    
    # Load the preprocessed data and model
    with stlt.spinner("Loading data and model....."):
        # Load preprocessed data
        data, encoding_rules = load_and_preprocess_data()
        
        # Load the selected model
        if model_type == "Random Forest":
            model_info = load_rf_model()
            stlt.info("Using Random Forest model")
        else:
            model_info = load_logistic_model()
            stlt.info("Using Logistic Regression model")
        
        if data is None or model_info is None:
            stlt.error("Failed to load data or model. Please ensure all necessary files are present.")
            return
    
    stlt.success("Data and model loaded successfully!")
    
    # Create tabs
    tab1, tab2, tab3 = stlt.tabs(["Make Prediction", "Model Information", "Feature Importance"])
    
    with tab1:
        # Get user input
        user_input = get_user_input(data, encoding_rules)
        
        # Make prediction button
        if stlt.button("Make Prediction"):
            with stlt.spinner("Making prediction....."):
                prediction_class, probabilities = make_prediction(model_info, user_input, encoding_rules)
                
                if prediction_class is not None:
                    # Get unique classes from the data
                    distinct_classes = sorted(data[TARGET_COLUMN].unique())
                    display_prediction_results(prediction_class, probabilities, distinct_classes)
                else:
                    stlt.error("Error: Could not go through with the prediction. Please check again the input values.")
    
    with tab2:
        stlt.subheader("Model Performance Metrics")
        metrics = model_info['metrics']
        
        # Create a metrics display
        col1, col2 = stlt.columns(2)
        
        with col1:
            stlt.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            stlt.metric("Precision", f"{metrics['precision']:.4f}")
        
        with col2:
            stlt.metric("Recall", f"{metrics['recall']:.4f}")
            stlt.metric("F1 Score", f"{metrics['f1']:.4f}")
        
        stlt.write("\n**Classification Report:**")
        if 'classification_report' in model_info:
            stlt.text(model_info['classification_report'])
        else:
            stlt.warning("Classification report not available for this model.")
    
    with tab3:
        stlt.subheader("Feature Importance")
        
        # Check if feature importance plot exists
        feature_importance_path = f"{'diagrams/Random_forest' if model_type == 'Random Forest' else 'diagrams/Logistic_regression'}_feature_importance_diagram.png"
        if os.path.exists(feature_importance_path):
            stlt.image(feature_importance_path)
        else:
            stlt.warning("Feature importance diagram not available for this model.")
        
        # Display confusion matrix
        stlt.subheader("Confusion Matrix") #Random_forest_confusion_matrix_diagram
        confusion_matrix_path = f"{'diagrams/Random_forest' if model_type == 'Random Forest' else 'diagrams/Logistic_regression'}_confusion_matrix_diagram.png"
        if os.path.exists(confusion_matrix_path):
            stlt.image(confusion_matrix_path)
        else:
            stlt.warning("Confusion matrix plot not available for this model.")

if __name__ == "__main__":
    main() 