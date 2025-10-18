import pandas as pd
import numpy as np
from pathlib import Path


def generate_predictions(
    model,
    X_test: pd.DataFrame,
    test_sample_ids: pd.Series,
    output_path: str = 'test_out.csv'
) -> pd.DataFrame:
    """
    Generate predictions and save to CSV file.
    """
    print("\nGenerating predictions...")

    predictions = model.predict(X_test)

    output_df = pd.DataFrame({
        'sample_id': test_sample_ids,
        'price': predictions
    })

    output_df['price'] = output_df['price'].round(2)

    output_df.to_csv(output_path, index=False)

    print(f"\nPredictions saved to: {output_path}")
    print(f"Total predictions: {len(output_df)}")
    print(f"\nPrediction Statistics:")
    print(f"  Mean Price: ${output_df['price'].mean():.2f}")
    print(f"  Median Price: ${output_df['price'].median():.2f}")
    print(f"  Min Price: ${output_df['price'].min():.2f}")
    print(f"  Max Price: ${output_df['price'].max():.2f}")
    print(f"  Std Dev: ${output_df['price'].std():.2f}")

    return output_df


def validate_output_format(output_path: str, expected_sample_ids: pd.Series = None) -> bool:
    """
    Validate that output file matches required format.
    """
    try:
        df = pd.read_csv(output_path)

        if list(df.columns) != ['sample_id', 'price']:
            print(f"ERROR: Columns should be ['sample_id', 'price'], got {list(df.columns)}")
            return False

        if df['sample_id'].isna().any():
            print("ERROR: Found missing sample_ids")
            return False

        if df['price'].isna().any():
            print("ERROR: Found missing prices")
            return False

        if (df['price'] <= 0).any():
            print("WARNING: Found non-positive prices")
            return False

        if expected_sample_ids is not None:
            if len(df) != len(expected_sample_ids):
                print(f"ERROR: Expected {len(expected_sample_ids)} predictions, got {len(df)}")
                return False

            if not df['sample_id'].isin(expected_sample_ids).all():
                print("ERROR: Sample IDs don't match test set")
                return False

        print("\n✓ Output format validation passed!")
        return True

    except Exception as e:
        print(f"ERROR validating output: {e}")
        return False


def evaluate_on_sample(
    model,
    sample_test_path: str = 'data/sample_test.csv',
    sample_out_path: str = 'data/sample_test_out.csv'
):
    """
    Evaluate model on sample test data if available.
    """
    from feature_engg import FeatureEngineer
    from uitil import calculate_smape

    try:
        sample_test_df = pd.read_csv(sample_test_path)
        sample_out_df = pd.read_csv(sample_out_path)

        print("\nEvaluating on sample test data...")

        feature_engineer = FeatureEngineer()
        feature_engineer.fit(sample_test_df)
        X_sample = feature_engineer.transform(sample_test_df)

        predictions = model.predict(X_sample)

        merged = sample_out_df.merge(
            pd.DataFrame({'sample_id': sample_test_df['sample_id'], 'predicted': predictions}),
            on='sample_id'
        )

        smape = calculate_smape(merged['price'].values, merged['predicted'].values)
        print(f"Sample Test SMAPE: {smape:.6f}")

    except FileNotFoundError:
        print("\nSample test files not found. Skipping evaluation.")
    except Exception as e:
        print(f"\nError during sample evaluation: {e}")


    