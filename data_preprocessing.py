import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from selected_features import SELECTED_FEATURES, TARGET_COLUMN
import pickle
import os
import re
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings
import textwrap
warnings.filterwarnings('ignore')

# Process raw data and establish encoding rules for the categorical features
@st.cache_data #to cache data to prevent the app from loading the data again and again
def load_and_preprocess_data(file_path='seer_data.xlsx', null_threshold=0.5):
    print("**************************************************")
    print("Loading raw data, preprocessing it and encoding it using One-hot encoding")
    print("**************************************************")
    #load the base excel file
    data = pd.read_excel(file_path)
    
    # convert all ('Blank(s)') values to NULL values
    data = data.replace('Blank(s)', np.nan)
    data = data.drop_duplicates()
    

    a= data.isnull().sum()/len(data)
    cols = a[a > null_threshold].index.tolist()
    data = data.drop(columns = cols)    

    # For numerical columns, fill with median
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        if data[col].isnull().any():
            data[col] = data[col].fillna(data[col].median())
            print(f"Filled null values in {col} using median")
    
    # For categorical columns, fill with mode
    cat_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if data[col].isnull().any():
            data[col] = data[col].fillna(data[col].mode()[0])
    
    #convert all object types to category if not done in previous step
    for col in cat_cols:
        data[col] = data[col].astype('category')
    
    # Select only the required features
    required_features = SELECTED_FEATURES
    
    # Include target columns 
    if TARGET_COLUMN not in required_features:
        required_features.append(TARGET_COLUMN)

    existing_features = list(set(required_features) & set(data.columns))
    data = data[existing_features]
    
    #one-hot encoding to convert categorical variables to columns of 1 and 0 (TRUE OR FALSE))
    final_cat_cols = [col for col in data.select_dtypes(include=['category']).columns 
                            if col != TARGET_COLUMN]
    
    # Apply one-hot encoding to final categorical variables
    data, encoding_rules = apply_one_hot_encoding(data, final_cat_cols)
    
    return data, encoding_rules

def save_to_file(data, output_path='preprocessed_seer_data.xlsx'):
    print("\n**************************************************")
    print("\n Loading pre processed data to excel file")
    print("**************************************************")
    data.to_excel(output_path, index=False)



def create_one_hot_encoding_rules(data, categorical_cols):
    print("**************************************************")
    print("Executing encoding rules function and store rules in a pickle file")
    print("**************************************************")
    encoding_rules = {}
    
    for col in categorical_cols:
        # Get unique values and convert them to strings for consistent sorting
        unique_values = [str(val) for val in data[col].unique().tolist()]
        unique_values = sorted(unique_values)
        
        # Create encoding rules
        encoding_rules[col] = {
            'categories': unique_values,
            'mapping': {val: idx for idx, val in enumerate(unique_values)}
        }
    
    # Save encoding rules to a file
    with open('encoding_rules.pkl', 'wb') as f:
        pickle.dump(encoding_rules, f)
    
    return encoding_rules

def apply_one_hot_encoding(data, categorical_cols):
    print("**************************************************")
    print("Applying one-hot encoding")
    print("**************************************************")
    # create encoding rules
    encoding_rules = create_one_hot_encoding_rules(data, categorical_cols)
    
    # Create a copy of the data to avoid modifying the original
    encoded_data = data.copy()
    
    # Apply encoding to each categorical column
    for col in categorical_cols:
        if col in encoding_rules:
            # Create dummy variables
            dummies = pd.get_dummies(encoded_data[col], prefix=col)
            
            # Drop the original column
            encoded_data = encoded_data.drop(columns=[col])
            
            # Add the dummy variables
            encoded_data = pd.concat([encoded_data, dummies], axis=1)
    
    return encoded_data, encoding_rules

def load_encoding_rules():
    print("**************************************************")
    print("Loading encoding rules from pickel file")
    print("**************************************************")
    if os.path.exists('encoding_rules.pkl'):
        with open('encoding_rules.pkl', 'rb') as f:
            return pickle.load(f)
    return None

def encode_single_input(input, encoding_rules):
    print("**************************************************")
    print("Encoding user input using the encoding_rules.pkl file")
    print("**************************************************")
 
    encoded_input = {}
    
    # First, copy all numerical values
    for key, value in input.items():
        if key not in encoding_rules:
            encoded_input[key] = value
    
    # Then, encode categorical variables
    for col, rules in encoding_rules.items():
        if col in input:
            value = input[col]
            # Create dummy variables for this categorical value
            for category in rules['categories']:
                encoded_input[f"{col}_{category}"] = 1 if value == category else 0
    
    return encoded_input

def visualize_data_attributes(data, selected_features):
    print("**************************************************")
    print("Visualizing data attributes in the data preprocessing step")
    print("**************************************************")
    # Create directory if it doesn't exist
    os.makedirs('diagrams', exist_ok=True)
    
    # Separate categorical and numerical features
    categorical_features = []
    numerical_features = []
    
    for feature in selected_features:
        if feature in data.columns:
            if data[feature].dtype == 'object' or data[feature].nunique() < 10:
                categorical_features.append(feature)
            else:
                numerical_features.append(feature)
    
    # Visualize categorical features
    for feature in categorical_features:
        plt.figure(figsize=(10, 6))
        value_counts = data[feature].value_counts()
        sns.barplot(x=value_counts.index, y=value_counts.values)
        plt.title(f'Distribution of {feature}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'diagrams/{feature}_distribution.png')
        plt.close()
        
        # Print value counts
        print(f"\n{feature} value counts:")
        print(value_counts)
    
    # Visualize numerical features
    if numerical_features:
        # Create a figure with subplots for numerical features
        n_cols = 2
        n_rows = (len(numerical_features) + 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()
        
        for i, feature in enumerate(numerical_features):
            sns.histplot(data[feature], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {feature}')
        
        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(f'diagrams/numerical_features_distribution.png')
        plt.close()
        
    
    # Create correlation matrix for numerical features
    if len(numerical_features) > 1:
        plt.figure(figsize=(12, 8))
        corr_matrix = data[numerical_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.savefig(f'diagrams/correlation_matrix.png')
        plt.close()
        
    


def analyze_selected_features(file_path='seer_data.xlsx'):
   print("**************************************************")
   print("Analyzing selected features in the data preprocessing step")
   print("**************************************************")
   # Load the data
   data = pd.read_excel(file_path)
  
   # Initialize list to store feature information
   feature_information = []
  
   # Analyze each selected feature
   for feature in SELECTED_FEATURES:
       if feature in data.columns:
           # Determine variable type
           if data[feature].dtype == 'object' or data[feature].nunique() < 10:
               type = 'Categorical'
           else:
               type = 'Numeric'
 

           # Get distinct values
           if type == 'Categorical':
               values = data[feature].dropna().unique().tolist()
               # Convert to string representation
               values = [str(val) for val in values]
           else:
               values = ['Numeric Range: {:.2f} to {:.2f}'.format(
                   data[feature].min(), data[feature].max())]
          
           # Add to feature info
           feature_information.append({
               'Feature_Name': feature,
               'Type_Of_Variable': type,
               'Distinct_Values': values
           })
  
   # Create DataFrame
   df_feature = pd.DataFrame(feature_information)
  
   # Save to Excel
   df_feature.to_excel('feature_information_analysis.xlsx', index=False)
  
   return df_feature

def visualize_target_distribution(file_path='seer_data.xlsx'):
    """
    Creates a bar plot showing the distribution of target variable values.
    
    Args:
        file_path (str): Path to the SEER data Excel file.
    """
    print("**************************************************")
    print("Visualizing target variable distribution")
    print("**************************************************")
    
    # Load the data
    data = pd.read_excel(file_path)
    
    # Ensure target column exists
    if TARGET_COLUMN not in data.columns:
        print(f"Target column '{TARGET_COLUMN}' not found in the dataset.")
        return
    
    # Count records for each target value
    target_counts = data[TARGET_COLUMN].value_counts().sort_index()
    
    # Create directory if it doesn't exist
    os.makedirs('diagrams', exist_ok=True)
    
    # Create the bar plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=target_counts.index, y=target_counts.values, color="#154734")
    
    # Add value labels on top of each bar with default size
    for i, count in enumerate(target_counts.values):
        ax.text(i, count + (count * 0.01), f'{count:,}', ha='center', fontsize=10, weight='bold')
    
    # Set title and labels with bold font for axis labels - increased by 2 points
    plt.title(f'Distribution of {TARGET_COLUMN}', fontsize=14)
    plt.xlabel('Target Values', fontsize=14, weight='bold')
    plt.ylabel('Count', fontsize=14, weight='bold')
    
    # Horizontal x-axis labels with text wrapping
    # Create wrapped labels
    labels = ['\n'.join(textwrap.wrap(str(label), width=20)) for label in target_counts.index]
    ax.set_xticklabels(labels, rotation=0, ha='center')
    
    # Make tick labels bold and increase their size by 2 points
    plt.setp(ax.get_xticklabels(), weight='bold', fontsize=12)
    plt.setp(ax.get_yticklabels(), weight='bold', fontsize=12)
    
    # Remove grid lines
    plt.grid(False)
    
    # Add more bottom padding to accommodate wrapped text
    plt.subplots_adjust(bottom=0.2)
    
    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f'diagrams/target_distribution.png', dpi=300)
    plt.close()
    
    # Print counts information
    print(f"\nDistribution of {TARGET_COLUMN}:")
    for value, count in target_counts.items():
        print(f"{value}: {count:,} records ({count/len(data)*100:.2f}%)")
    
    return target_counts

#main execution point of the program if run as a standalone code
if __name__ == "__main__":
    analyze_selected_features()
    preprocessed_data, encoding_rules = load_and_preprocess_data()
    save_to_file(preprocessed_data)
    visualize_data_attributes(preprocessed_data, SELECTED_FEATURES)
    visualize_target_distribution()

