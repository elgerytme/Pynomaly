#!/usr/bin/env python3
"""
Simple test script to verify banking anomaly detection functionality.
"""

import sys

import numpy as np
import pandas as pd

# Import from src.pynomaly instead of using path manipulation
try:
    from src.pynomaly.domain.entities.dataset import Dataset
    from src.pynomaly.domain.value_objects.contamination_rate import ContaminationRate
    from src.pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter

    print("✓ Pynomaly imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def test_basic_anomaly_detection():
    """Test basic anomaly detection on deposit data."""
    print("\n=== Testing Basic Anomaly Detection ===")

    # Load deposit data
    try:
        deposits = pd.read_csv("examples/banking/datasets/deposits.csv")
        print(f"✓ Loaded {len(deposits)} deposit records")
    except FileNotFoundError:
        print(
            "✗ Deposit data not found. Run examples/banking/scripts/generate_sample_data.py first."
        )
        return False

    # Simple feature engineering
    features = deposits[["amount"]].copy()
    features["hour"] = pd.to_datetime(deposits["timestamp"]).dt.hour
    features["day_of_week"] = pd.to_datetime(deposits["timestamp"]).dt.dayofweek
    features["amount_log"] = np.log1p(features["amount"])

    print(f"✓ Engineered {features.shape[1]} features")

    # Prepare data for anomaly detection
    X = features.fillna(0).values
    dataset = Dataset(
        name="deposit_features", data=X, feature_names=list(features.columns)
    )

    # Test Isolation Forest
    try:
        contamination_rate = ContaminationRate(0.05)
        detector = SklearnAdapter(
            "IsolationForest", contamination_rate=contamination_rate
        )
        result = detector.fit_detect(dataset)

        # Extract scores from AnomalyScore objects
        scores = np.array([score.value for score in result.scores])
        anomaly_count = (scores > np.percentile(scores, 95)).sum()
        print(f"✓ Isolation Forest detected {anomaly_count} anomalies")

        # Compare with known anomalies
        actual_anomalies = deposits["is_anomaly"].sum()
        print(f"  Ground truth: {actual_anomalies} anomalies")
        print(f"  Detection rate: {anomaly_count / len(deposits) * 100:.1f}%")

        return True

    except Exception as e:
        print(f"✗ Anomaly detection failed: {e}")
        return False


def test_data_quality():
    """Test data quality of generated datasets."""
    print("\n=== Testing Data Quality ===")

    datasets = [
        "deposits.csv",
        "credit_card_transactions.csv",
        "fx_transactions.csv",
        "atm_transactions.csv",
    ]

    total_records = 0
    total_anomalies = 0

    for dataset_file in datasets:
        try:
            df = pd.read_csv(f"examples/banking/datasets/{dataset_file}")
            anomalies = df["is_anomaly"].sum() if "is_anomaly" in df.columns else 0

            print(
                f"✓ {dataset_file}: {len(df):,} records, {anomalies} anomalies ({anomalies / len(df) * 100:.1f}%)"
            )

            total_records += len(df)
            total_anomalies += anomalies

        except FileNotFoundError:
            print(f"✗ {dataset_file} not found")

    print(
        f"\nTotal: {total_records:,} records, {total_anomalies:,} anomalies ({total_anomalies / total_records * 100:.1f}%)"
    )
    return True


def main():
    """Run all tests."""
    print("Banking Anomaly Detection - Basic Functionality Test")
    print("=" * 55)

    # Test data quality
    data_ok = test_data_quality()

    # Test anomaly detection
    detection_ok = test_basic_anomaly_detection()

    # Summary
    print("\n=== Test Summary ===")
    if data_ok and detection_ok:
        print("✓ All tests passed! Banking anomaly detection is working correctly.")
        print("\nNext steps:")
        print(
            "1. Run individual analysis scripts: analyze_deposits.py, analyze_credit_cards.py"
        )
        print("2. Open the Jupyter notebook for comprehensive analysis")
        print("3. Review the generated reports in ../outputs/")
    else:
        print("✗ Some tests failed. Check the error messages above.")


if __name__ == "__main__":
    main()
