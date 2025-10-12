#!/usr/bin/env python3
"""
Quick test script using sample_test.csv to verify the pipeline works.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from feature_engg import engineer_features
from model_train import train_xgboost_model
from predict import generate_predictions, validate_output_format


def create_synthetic_train_data(sample_test_df: pd.DataFrame, n_samples: int = 1000) -> pd.DataFrame:
    """
    Create synthetic training data based on sample test structure.
    """
    print("Creating synthetic training data for testing...")

    train_rows = []

    for _ in range(n_samples):
        idx = np.random.randint(0, len(sample_test_df))
        row = sample_test_df.iloc[idx].copy()

        base_price = np.random.uniform(5, 100)

        catalog_lower = str(row['catalog_content']).lower()
        if 'pack' in catalog_lower:
            base_price *= 1.5
        if 'organic' in catalog_lower:
            base_price *= 1.3
        if 'oz' in catalog_lower or 'ounce' in catalog_lower:
            try:
                import re
                numbers = re.findall(r'\d+\.?\d*', str(row['catalog_content']))
                if numbers:
                    base_price *= (1 + float(numbers[0]) / 100)
            except:
                pass

        row['price'] = base_price + np.random.normal(0, base_price * 0.1)
        row['price'] = max(0.01, row['price'])

        train_rows.append(row)

    train_df = pd.DataFrame(train_rows)
    train_df = train_df.reset_index(drop=True)

    return train_df


def quick_test():
    """
    Run a quick test with sample data.
    """
    print("="*70)
    print(" Quick Test with Sample Data ")
    print("="*70)

    SAMPLE_TEST_PATH = 'data/sample_test.csv'

    if not Path(SAMPLE_TEST_PATH).exists():
        print(f"\nERROR: Sample test file not found at {SAMPLE_TEST_PATH}")
        return

    sample_test_df = pd.read_csv(SAMPLE_TEST_PATH)
    print(f"\nLoaded sample test data: {sample_test_df.shape}")

    train_df = create_synthetic_train_data(sample_test_df, n_samples=500)
    print(f"Created synthetic training data: {train_df.shape}")

    print("\n" + "="*70)
    print(" Feature Engineering ")
    print("="*70)

    X_train, X_test, y_train, feature_engineer = engineer_features(train_df, sample_test_df)

    print(f"\nFeatures created:")
    print(f"  Training: {X_train.shape}")
    print(f"  Test: {X_test.shape}")

    print("\n" + "="*70)
    print(" Model Training (Quick Test - 2 Folds) ")
    print("="*70)

    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.1,
        'max_depth': 5,
        'n_estimators': 100,
        'random_state': 42,
        'n_jobs': -1
    }

    model = train_xgboost_model(X_train, y_train, params=xgb_params, n_folds=2)

    print("\n" + "="*70)
    print(" Generating Predictions ")
    print("="*70)

    output_df = generate_predictions(
        model,
        X_test,
        sample_test_df['sample_id'],
        'sample_test_predictions.csv'
    )

    print("\n" + "="*70)
    print(" Validation ")
    print("="*70)

    is_valid = validate_output_format('sample_test_predictions.csv', sample_test_df['sample_id'])

    if is_valid:
        print("\n✓ Quick test passed successfully!")
        print("\nYou can now run main.py with the full dataset.")
    else:
        print("\n✗ Quick test failed. Please check the errors above.")


if __name__ == "__main__":
    quick_test()
