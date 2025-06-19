# Cancer Survivability Rate Prediction Application

A comprehensive machine learning application that predicts cancer survivability rates using the SEER (Surveillance, Epidemiology, and End Results) dataset. This application provides an interactive web interface for healthcare professionals and researchers to input patient information and receive predictions using either Random Forest or Logistic Regression models.

##  Project Overview

This application leverages machine learning algorithms to predict cancer outcomes based on patient demographics, tumor characteristics, treatment information, and socioeconomic factors. The system is trained on the SEER breast cancer dataset from 2021 and provides real-time predictions through an intuitive Streamlit web interface.

### Key Features

- **Dual Model Support**: Choose between Random Forest and Logistic Regression models
- **Interactive Web Interface**: User-friendly Streamlit application with real-time predictions
- **Comprehensive Data Preprocessing**: Automated handling of missing values, categorical encoding, and feature selection
- **Model Performance Visualization**: Confusion matrices, feature importance plots, and precision-recall curves
- **Real-time Predictions**: Instant cancer survivability predictions based on patient input

## Dataset Information

The application uses the **SEER (Surveillance, Epidemiology, and End Results)** dataset, which is a comprehensive cancer surveillance database maintained by the National Cancer Institute. The dataset includes:

- **Patient Demographics**: Age, race, sex, marital status
- **Tumor Characteristics**: Primary site, histologic type, grade, stage, size
- **Treatment Information**: Surgery, radiation, chemotherapy
- **Socioeconomic Factors**: Income levels, rural-urban classification
- **Outcome Data**: SEER cause-specific death classification

### Selected Features

The model uses 17 carefully selected features:

1. Age recode with <1 year olds
2. Race recode (White, Black, Other)
3. Sex
4. Year of diagnosis
5. Primary Site
6. Histologic Type ICD-O-3
7. Grade Recode (thru 2017)
8. Combined Summary Stage (2004+)
9. Regional nodes examined (1988+)
10. Regional nodes positive (1988+)
11. Tumor Size Summary (2016+)
12. Radiation recode
13. Chemotherapy recode (yes, no/unk)
14. RX Summ--Surg Prim Site (1998+)
15. Marital status at diagnosis
16. Median household income inflation adj to 2022
17. Rural-Urban Continuum Code

**Target Variable**: SEER cause-specific death classification

## Project Architecture

```
Final_Cancer/
├── cancer_prediction_app.py          # Main Streamlit application
├── data_preprocessing.py             # Data preprocessing and encoding
├── cancer_prediction_model_with_random_forest.py    # Random Forest model
├── cancer_prediction_model_with_logistic.py         # Logistic Regression model
├── selected_features.py              # Feature selection configuration
├── requirements.txt                  # Python dependencies
├── seer_data.xlsx                   # Raw SEER dataset
├── preprocessed_seer_data.xlsx      # Preprocessed dataset
├── random_forest_model.pkl          # Trained Random Forest model
├── logistic_model.pkl               # Trained Logistic Regression model
├── encoding_rules.pkl               # Categorical encoding rules
├── diagrams/                        # Model performance visualizations
│   ├── Random_forest_*.png
│   ├── Logistic_regression_*.png
│   └── data_analysis_*.png
└── README.md                        # This file
```

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation Steps

1. **Clone or download the project**
   ```bash
   # If using git
   git clone <repository-url>
   cd Final_Cancer
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import streamlit, pandas, numpy, matplotlib, seaborn, sklearn; print('All dependencies installed successfully!')"
   ```

## Dependencies

The application requires the following Python packages:

```
streamlit==1.28.0          # Web application framework
pandas==2.0.3              # Data manipulation and analysis
numpy==1.24.3              # Numerical computing
matplotlib==3.7.2          # Plotting and visualization
seaborn==0.12.2            # Statistical data visualization
scikit-learn==1.3.0        # Machine learning algorithms
xgboost>=1.7.0             # Gradient boosting (optional)
openpyxl>=3.0.0            # Excel file handling
```

## Usage

### Running the Application

1. **Start the Streamlit application**
   ```bash
   streamlit run cancer_prediction_app.py
   ```

2. **Open your web browser**
   - The application will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, navigate to the URL manually

### Using the Application

1. **Model Selection**
   - Use the sidebar to choose between "Random Forest" or "Logistic Regression"
   - Each model has different characteristics and performance metrics

2. **Input Patient Information**
   - Navigate to the "Make Prediction" tab
   - Fill in the patient information fields:
     - **Demographics**: Age, race, sex, marital status
     - **Diagnosis**: Year of diagnosis, primary site, histologic type
     - **Tumor Details**: Grade, stage, size, regional nodes
     - **Treatment**: Surgery, radiation, chemotherapy
     - **Socioeconomic**: Income, rural-urban classification

3. **Get Predictions**
   - Click the "Make Prediction" button
   - View the predicted outcome and probability scores
   - Results show both the predicted class and confidence levels

4. **Explore Model Information**
   - **Model Information Tab**: View accuracy, precision, recall, and F1 scores
   - **Feature Importance Tab**: See which features most influence predictions

## Model Training

### Training Individual Models

You can train the models separately by running the individual model scripts:

```bash
# Train Random Forest model
python cancer_prediction_model_with_random_forest.py

# Train Logistic Regression model
python cancer_prediction_model_with_logistic.py
```

### Data Preprocessing

The preprocessing pipeline includes:

1. **Data Cleaning**
   - Handling missing values (threshold-based removal)
   - Duplicate removal
   - Data type conversion

2. **Feature Engineering**
   - One-hot encoding for categorical variables
   - Feature selection based on domain expertise
   - Encoding rules preservation for consistent predictions

3. **Model Preparation**
   - Train-test split (80-20)
   - Stratified sampling for balanced classes
   - Feature scaling for Logistic Regression

## Model Performance

### Random Forest Model
- **Algorithm**: Random Forest Classifier
- **Parameters**: 100 estimators, random_state=42
- **Strengths**: Handles non-linear relationships, robust to outliers
- **Use Case**: When you need high accuracy and interpretable feature importance

### Logistic Regression Model
- **Algorithm**: Logistic Regression with balanced class weights
- **Parameters**: max_iter=1000, class_weight='balanced'
- **Strengths**: Interpretable coefficients, fast predictions
- **Use Case**: When you need interpretable results and linear relationships

### Performance Metrics
Both models are evaluated using:
- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1 Score**: Harmonic mean of precision and recall

## Visualizations

The application generates comprehensive visualizations:

### Model Performance
- **Confusion Matrices**: Show prediction accuracy for each class
- **Precision-Recall Curves**: Display model performance across different thresholds
- **Feature Importance Plots**: Identify most influential features

### Data Analysis
- **Target Distribution**: Class balance visualization
- **Feature Distributions**: Histograms and bar plots for all features
- **Correlation Matrix**: Feature relationships heatmap

## Technical Details

### Data Preprocessing Pipeline

```python
def load_and_preprocess_data(file_path='seer_data.xlsx', null_threshold=0.5):
    # 1. Load raw data
    # 2. Handle missing values
    # 3. Remove high-null columns
    # 4. Apply one-hot encoding
    # 5. Return processed data and encoding rules
```

### Model Training Process

```python
def train_model(X, y):
    # 1. Split data (80-20)
    # 2. Initialize model
    # 3. Train on training set
    # 4. Evaluate on test set
    # 5. Calculate metrics
    # 6. Generate visualizations
```

### Prediction Pipeline

```python
def make_prediction(model_info, user_input, encoding_rules):
    # 1. Encode user input
    # 2. Apply scaling (if needed)
    # 3. Ensure feature compatibility
    # 4. Make prediction
    # 5. Return results
```

## Customization

### Adding New Features

1. Update `selected_features.py` with new feature names
2. Ensure the feature exists in the raw dataset
3. Retrain models to include the new feature

### Modifying Model Parameters

Edit the model training functions in:
- `cancer_prediction_model_with_random_forest.py`
- `cancer_prediction_model_with_logistic.py`

### Changing Preprocessing

Modify `data_preprocessing.py` to:
- Adjust null threshold values
- Change encoding strategies
- Add new preprocessing steps

## Important Notes

### Data Privacy
- The application uses publicly available SEER data
- No patient identifiers are included in the dataset
- All predictions are for research/educational purposes

### Model Limitations
- Models are trained on 2021 SEER breast cancer data
- Predictions may not generalize to other cancer types or time periods
- Clinical decisions should not be based solely on these predictions

### Performance Considerations
- Large dataset processing may take time on first run
- Models are cached using Streamlit for faster subsequent runs
- Consider using GPU acceleration for large-scale deployments

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request
