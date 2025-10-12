# ML Challenge 2025 - Smart Product Pricing Solution

## Overview

This solution uses **XGBoost** with advanced text-based feature engineering to predict product prices from catalog content. The approach is optimized for the SMAPE evaluation metric and handles both textual and numerical features extracted from product descriptions.

## Solution Architecture

### 1. Feature Engineering Pipeline (`feature_engineering.py`)
- **Text Parsing**: Extracts item name, bullet points, descriptions, IPQ values, and units
- **TF-IDF Features**: Top 100 TF-IDF features from product text (1-2 grams)
- **Numeric Features**: Extracts numbers, calculates statistics (min, max, avg, sum)
- **Category Features**: Binary flags for organic, vegan, kosher, gluten-free, GMO-free, etc.
- **Length Features**: Text length, word count, description length
- **Unit Encoding**: Standardized and encoded product units

### 2. XGBoost Model (`model_training.py`)
- **Algorithm**: XGBoost Regressor with optimized hyperparameters
- **Cross-Validation**: 5-fold CV for robust model training
- **Ensemble**: Averages predictions from all folds
- **Optimization**: Tuned for SMAPE metric with regularization

### 3. Prediction Pipeline (`predict.py`)
- Generates predictions for test data
- Validates output format
- Ensures all prices are positive

## File Structure

```
project/
├── main.py                    # Main execution script
├── feature_engineering.py     # Feature extraction pipeline
├── model_training.py          # XGBoost model training
├── model.py                   # Model wrapper
├── predict.py                 # Prediction generation
├── utils.py                   # Utility functions
├── quick_test.py             # Quick test with sample data
├── requirements.txt          # Python dependencies
├── dataset/
│   ├── train.csv            # Training data (75k samples)
│   ├── test.csv             # Test data (75k samples)
│   ├── sample_test.csv      # Sample test data
│   └── sample_test_out.csv  # Sample output format
└── test_out.csv             # Generated predictions (output)
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0

## Usage

### Quick Test (Recommended First)

Test the pipeline with sample data:

```bash
python quick_test.py
```

This creates synthetic training data and runs a quick 2-fold CV test.

### Full Training and Prediction

```bash
python main.py
```

This will:
1. Load `dataset/train.csv` and `dataset/test.csv`
2. Engineer features from catalog content
3. Train XGBoost model with 5-fold cross-validation
4. Generate predictions
5. Save results to `test_out.csv`

## Key Features

### Text Feature Engineering
- **TF-IDF Vectorization**: Captures important words and phrases
- **N-gram Analysis**: Uses both unigrams and bigrams
- **Stop Word Removal**: Filters common English words

### Numeric Feature Extraction
- Extracts all numbers from text
- Computes statistics: min, max, mean, sum, count
- Handles IPQ (Item Pack Quantity) values

### Category Detection
- Identifies product categories (food, beverage, snack)
- Detects certifications (organic, kosher, vegan, gluten-free)
- Pack detection for bulk items

### Model Optimization
- Learning rate: 0.05 (conservative for better generalization)
- Max depth: 7 (prevents overfitting)
- L1 & L2 regularization (alpha=0.1, lambda=1.0)
- Early stopping (50 rounds)
- Histogram-based tree method (faster training)

## Output Format

The solution generates `test_out.csv` with exactly 2 columns:

```csv
sample_id,price
217392,45.67
209156,12.34
...
```

- All sample IDs from test set are included
- Prices are positive float values (minimum 0.01)
- Predictions are rounded to 2 decimal places

## Performance Metrics

The model is evaluated using **SMAPE** (Symmetric Mean Absolute Percentage Error):

```
SMAPE = (1/n) * Σ |predicted - actual| / ((|actual| + |predicted|)/2)
```

- Lower values indicate better performance
- Range: 0% to 200%
- Cross-validation provides out-of-fold SMAPE scores

## Optimization Strategy

### 1. Feature Selection
- TF-IDF captures product-specific keywords
- Numeric features capture quantity/size information
- Category features capture product types

### 2. Model Tuning
- Conservative learning rate for stability
- Regularization prevents overfitting
- Cross-validation ensures generalization

### 3. Ensemble Approach
- Averages predictions from 5 models
- Reduces variance and improves robustness

## Customization

### Adjust Hyperparameters

Edit `main.py` to modify XGBoost parameters:

```python
xgb_params = {
    'learning_rate': 0.05,      # Lower = slower but more accurate
    'max_depth': 7,              # Higher = more complex trees
    'n_estimators': 1000,        # More trees (with early stopping)
    'subsample': 0.8,            # Row sampling
    'colsample_bytree': 0.8,     # Column sampling
    # ... other parameters
}
```

### Change Feature Count

Modify TF-IDF feature count in `feature_engineering.py`:

```python
feature_engineer = FeatureEngineer(max_tfidf_features=100)  # Change to 50, 200, etc.
```

### Adjust Cross-Validation Folds

Change the number of folds in `main.py`:

```python
model = train_xgboost_model(X_train, y_train, params=xgb_params, n_folds=5)  # Change to 3, 7, 10, etc.
```

## Troubleshooting

### Memory Issues
- Reduce `max_tfidf_features` to 50
- Reduce `n_folds` to 3
- Process data in batches

### Slow Training
- Increase `learning_rate` to 0.1
- Reduce `n_estimators` to 500
- Use fewer CV folds

### Poor Performance
- Increase `max_tfidf_features` to 200
- Tune regularization parameters
- Add more feature engineering

## Academic Integrity

This solution adheres to competition rules:
- ✓ Uses only provided training data
- ✓ No external price lookups
- ✓ No web scraping
- ✓ Text-based feature engineering only
- ✓ Open-source model (MIT License)

## Model License

- XGBoost: Apache 2.0 License
- Scikit-learn: BSD License
- All code is original and competition-compliant

## Next Steps

1. Run `quick_test.py` to verify setup
2. Run `main.py` with full dataset
3. Review `test_out.csv` output
4. Submit to competition portal
5. Prepare methodology documentation

## Contact & Support

For questions about this solution, please refer to the competition guidelines and documentation.

---

**Good luck with the ML Challenge 2025!** 🚀
