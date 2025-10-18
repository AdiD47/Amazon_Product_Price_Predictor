#!/usr/bin/env python3
"""
ML Challenge 2025 - Smart Product Pricing
XGBoost-based Price Prediction Solution

This solution uses text-based feature engineering + embeddings and XGBoost
to predict product prices based on catalog content.
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

from feature_engg import engineer_features
from model_train import train_xgboost_model
from predict import generate_predictions, validate_output_format

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


def main():
    """
    Main execution pipeline.
    """
    print("="*70)
    print(" ML Challenge 2025 - Smart Product Pricing ")
    print(" XGBoost Price Prediction Pipeline with Embeddings ")
    print("="*70)

    TRAIN_PATH = 'dataset/train.csv'
    TEST_PATH = 'dataset/test.csv'
    OUTPUT_PATH = 'test_out.csv'

    if not Path(TRAIN_PATH).exists():
        print(f"\nERROR: Training file not found at {TRAIN_PATH}")
        print("Please ensure dataset/train.csv exists in the project directory.")
        return

    if not Path(TEST_PATH).exists():
        print(f"\nERROR: Test file not found at {TEST_PATH}")
        print("Please ensure dataset/test.csv exists in the project directory.")
        return

    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

    print("\n" + "="*70)
    print(" Feature Engineering ")
    print("="*70)

    # ✅ Enable embeddings here
    X_train, X_test, y_train, feature_engineer = engineer_features(
        train_df, test_df, use_embeddings=True
    )

    print(f"\nEngineered Features:")
    print(f"  Training shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}")
    print(f"  Number of features: {X_train.shape[1]}")

    print("\n" + "="*70)
    print(" Model Training ")
    print("="*70)

    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.05,
        'max_depth': 7,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_estimators': 1000,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    }

    model = train_xgboost_model(X_train, y_train, params=xgb_params, n_folds=5)

    print("\n" + "="*70)
    print(" Generating Predictions ")
    print("="*70)

    output_df = generate_predictions(
        model,
        X_test,
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
    