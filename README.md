# Amazon Product Price Prediction

A machine learning project for predicting product prices based on catalog content using feature engineering and XGBoost.

## Libraries Used

### Core Machine Learning Libraries
- **numpy** - Numerical computations and array operations
- **pandas** - Data manipulation and analysis
- **scikit-learn** - Feature extraction (TF-IDF, LabelEncoder) and evaluation metrics
- **xgboost** - Gradient boosting model for price prediction
- **sentence-transformers** - Text embeddings using pre-trained models (all-MiniLM-L6-v2)

### Standard Python Libraries
- **re** - Regular expressions for text parsing
- **math** - Mathematical functions
- **pathlib** - File path operations
- **warnings** - Warning control
- **typing** - Type hints

## Installation

Install all required libraries:

```bash
pip install -r requirements.txt
```

## Project Files

- `main.py` - Main pipeline execution
- `feature_engg.py` - Feature engineering with TF-IDF and embeddings
- `model_train.py` - XGBoost model training with cross-validation
- `predict.py` - Generate predictions on test data
- `quick_test.py` - Quick testing script
- `uitil.py` - Utility functions for parsing and metrics
- `dataset/` - Training and test data
- `requirements.txt` - Python dependencies

## Usage

Run the complete pipeline:

```bash
python main.py
```

## Features

- Text feature extraction from catalog content
- TF-IDF vectorization
- Sentence embeddings
- Numeric and categorical feature engineering
- XGBoost regression with K-Fold cross-validation
- Log transformation for target variable
- SMAPE, MAE, RMSE, and R² evaluation metrics
