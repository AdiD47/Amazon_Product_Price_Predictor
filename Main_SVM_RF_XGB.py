#!/usr/bin/env python3
"""
ML Challenge 2025 - Smart Product Pricing
SVM + Random Forest + XGBoost-based Price Prediction Solution
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')


def load_data(train_path: str, test_path: str):
    """
    Load training and test datasets.
    """
    print("Loading datasets...")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Training data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")

    if 'price' in train_df.columns:
        print(f"\nTraining Price Statistics:")
        print(f"  Mean: ${train_df['price'].mean():.2f}")
        print(f"  Median: ${train_df['price'].median():.2f}")
        print(f"  Min: ${train_df['price'].min():.2f}")
        print(f"  Max: ${train_df['price'].max():.2f}")
        print(f"  Std Dev: ${train_df['price'].std():.2f}")

    return train_df, test_df


# def train_svm_model(X_train, y_train):
#     """
#     Train Support Vector Machine model.
#     """
#     svm_model = SVR(kernel='rbf', C=100, epsilon=0.1)
#     svm_model.fit(X_train, y_train)
#     return svm_model
from sklearn.svm import LinearSVR

def train_svm_model(X_train, y_train):
    print("Training LinearSVR...")
    svm_model = LinearSVR(epsilon=0.1, C=1.0, max_iter=10000, random_state=42)
    svm_model.fit(X_train, y_train)
    print("LinearSVR training done...")
    return svm_model



def train_rf_model(X_train, y_train):
    """
    Train Random Forest model.
    """
    print("Training RF...")
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    print("RF training done...")
    return rf_model


def train_xgboost_model(X_train, y_train, params=None, n_folds=5):
    """
    Train XGBoost model using ensemble learning from SVM and RF predictions.
    """
    print("Training XGB...")
    model = XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    model.fit(X_train, y_train)
    print("XGB training done...")
    return model


def engineer_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    # Your original feature engineering code here
    # For example:
    feature_engineer = FeatureEngineer(max_tfidf_features=100)
    X_train = feature_engineer.fit_transform(train_df)
    X_test = feature_engineer.transform(test_df)
    y_train = train_df['price'].values if 'price' in train_df.columns else None
    
    return X_train, X_test, y_train, feature_engineer


def generate_predictions(model, X_test, sample_ids, output_path):
    """
    Generate predictions and save them to a CSV file.
    """
    predictions = model.predict(X_test)
    output_df = pd.DataFrame({'sample_id': sample_ids, 'price': predictions})
    output_df.to_csv(output_path, index=False)
    return output_df


def validate_output_format(output_path, sample_ids):
    """
    Validate output format by checking the generated file.
    """
    output_df = pd.read_csv(output_path)
    if set(output_df.columns) == {'sample_id', 'price'} and len(output_df) == len(sample_ids):
        return True
    return False


def main():
    """
    Main execution pipeline.
    """
    print("="*70)
    print(" ML Challenge 2025 - Smart Product Pricing ")
    print(" SVM + Random Forest + XGBoost Price Prediction Pipeline ")
    print("="*70)

    TRAIN_PATH = '/content/train.csv'
    TEST_PATH = '/content/test.csv'
    OUTPUT_PATH = 'test_out.csv'

    if not Path(TRAIN_PATH).exists():
        print(f"\nERROR: Training file not found at {TRAIN_PATH}")
        return

    if not Path(TEST_PATH).exists():
        print(f"\nERROR: Test file not found at {TEST_PATH}")
        return

    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

    print("\n" + "="*70)
    print(" Feature Engineering ")
    print("="*70)

    X_train, X_test, y_train, feature_engineer = engineer_features(train_df, test_df)

    print(f"\nEngineered Features:")
    print(f"  Training shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}")
    print(f"  Number of features: {X_train.shape[1]}")

    print("\n" + "="*70)
    print(" Model Training ")
    print("="*70)

    # Train SVM and RF models
    svm_model = train_svm_model(X_train, y_train)
    rf_model = train_rf_model(X_train, y_train)

    # Get predictions from SVM and RF to use as new features for XGBoost
    svm_preds_train = svm_model.predict(X_train)
    rf_preds_train = rf_model.predict(X_train)

    # Create a new feature matrix with the original features + SVM + RF predictions
    X_train_ensemble = np.column_stack((X_train, svm_preds_train, rf_preds_train))

    svm_preds_test = svm_model.predict(X_test)
    rf_preds_test = rf_model.predict(X_test)

    X_test_ensemble = np.column_stack((X_test, svm_preds_test, rf_preds_test))

    # Train XGBoost model using ensemble predictions as new features
    xgb_model = train_xgboost_model(X_train_ensemble, y_train)

    print("\n" + "="*70)
    print(" Generating Predictions ")
    print("="*70)

    # Generate predictions on the test set using the XGBoost model
    output_df = generate_predictions(
        xgb_model,
        X_test_ensemble,
        test_df['sample_id'],
        OUTPUT_PATH
    )

    print("\n" + "="*70)
    print(" Validating Output ")
    print("="*70)

    is_valid = validate_output_format(OUTPUT_PATH, test_df['sample_id'])

    if is_valid:
        print("\n" + "="*70)
        print(" SUCCESS! ")
        print("="*70)
        print(f"\nPredictions ready for submission: {OUTPUT_PATH}")
        print("\nNext steps:")
        print("  1. Review the output file format")
        print("  2. Submit test_out.csv to the competition portal")
        print("  3. Prepare 1-page methodology documentation")
    else:
        print("\n" + "="*70)
        print(" WARNING: Output validation failed ")
        print("="*70)
        print("Please review the errors above and fix the issues.")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
