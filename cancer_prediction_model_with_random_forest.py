import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,confusion_matrix
import pickle
import os
from selected_features import SELECTED_FEATURES, TARGET_COLUMN
from data_preprocessing import load_and_preprocess_data, encode_single_input
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score

#train random forest
def train_random_forest(X, y):
    # Split train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize random forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    #train model
    rf_model.fit(X_train, y_train)
    
    # make predictions on testing set
    y_pred = rf_model.predict(X_test)
    
    # performance metrics calculatiion
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    clf_report = classification_report(y_test, y_pred)

        # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    
    # Storing model iinformation using dictionary
    model_info = {
        'model': rf_model,
        'feature_names': X.columns.tolist(),
        'target_name': TARGET_COLUMN,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        },
        'class_mapping': {i: label for i, label in enumerate(np.unique(y))},
        'classification_report': clf_report,
        'confusion_matrix': conf_matrix
    }
    
    return model_info

#Save model from the previous function to a pickle file
def save_model(model_info, filepath='random_forest_model.pkl'):
    with open(filepath, 'wb') as f:
        pickle.dump(model_info, f)

#load model using streamlit cache to improve performance. It will get executed from the streamlit app
@st.cache_resource
def load_model(filepath='random_forest_model.pkl'):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

def feature_importance_diagram(feature_importance_df, save_path='diagrams/Random_forest_feature_importance_diagram.png'):
  
    # Sort and get top N features
    top_features = feature_importance_df.sort_values('importance', ascending=False).head(10)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title('Random Forest : Feature Importance Diagram )'.format(10))
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()

def precision_recall_diagram(model, X, y, save_path='diagrams/Random_forest_precision_recall_diagram.png'):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Get unique classes
    classes = np.unique(y)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # For each class, create a precision-recall curve
    for i, class_label in enumerate(classes):
        # Create binary labels for this class
        y_test_binary = (y_test == class_label).astype(int)
        
        # Get predicted probabilities for this class
        y_scores = model.predict_proba(X_test)[:, i]
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test_binary, y_scores)
        average_precision = average_precision_score(y_test_binary, y_scores)
        
        # Plot the curve
        plt.plot(recall, precision, 
                label=f'Class {class_label} (AP = {average_precision:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Random Forest : Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()

def confusion_matrix_diagram(model_info, save_path='diagrams/Random_forest_confusion_matrix_diagram.png'):
    # Get confusion matrix from model info
    conf_matrix = model_info['confusion_matrix']
    
    # Get class labels
    class_labels = list(model_info['class_mapping'].values())
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title('Random Forest : Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()

# main execution point of script to run it in a standalone mode
if __name__ == "__main__":
    # Load and preprocess the data. If ran in the data_preprocessing.py file, it will not be necessary   
    data, encoding_rules = load_and_preprocess_data()
    
    if data is not None:
        #Seperate features and targent into seperate variables
        X = data.drop(columns=[TARGET_COLUMN])
        y = data[TARGET_COLUMN]
        
        #model training
        model_info = train_random_forest(X, y)
        
        # Adding encoding rules to parse the user input data from the application
        model_info['encoding_rules'] = encoding_rules
        
        # Save the model
        save_model(model_info)
        
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': model_info['feature_names'],
            'importance': model_info['model'].feature_importances_
        })

        # Create and save feature importance diagram
        feature_importance_diagram(feature_importance)
        
        # Create and save precision-recall curve
        precision_recall_diagram(model_info['model'], X, y)
        
        # Create and save confusion matrix diagram
        confusion_matrix_diagram(model_info)
    else:
        print("Error: Failed to load and preprocess data.") 