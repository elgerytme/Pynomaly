#!/usr/bin/env python3
"""
Quick test example using generated datasets.
This demonstrates how to use the example datasets with the anomaly detection package.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Mock anomaly detection classes for testing
class MockDetectionService:
    def detect(self, data, algorithm='isolation_forest', contamination=0.1, **kwargs):
        """Mock detection service for testing datasets."""
        n_samples = len(data)
        n_anomalies = int(contamination * n_samples)
        
        # Create mock results
        predictions = np.ones(n_samples)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        predictions[anomaly_indices] = -1
        
        scores = np.random.random(n_samples)
        scores[anomaly_indices] *= 2  # Higher scores for anomalies
        
        return MockResult(predictions, scores, n_samples, processing_time=0.1)

class MockResult:
    def __init__(self, predictions, scores, total_samples, processing_time):
        self.predictions = predictions
        self.scores = scores
        self.total_samples = total_samples
        self.processing_time = processing_time
        self.anomaly_count = np.sum(predictions == -1)


def test_credit_card_dataset():
    """Test the credit card transactions dataset."""
    print("🏧 Testing Credit Card Transactions Dataset")
    print("-" * 50)
    
    # Load dataset
    df = pd.read_csv('datasets/credit_card_transactions.csv')
    print(f"📊 Loaded {len(df)} transactions")
    print(f"🚨 Contains {df['is_fraud'].sum()} actual fraud cases ({df['is_fraud'].sum()/len(df)*100:.1f}%)")
    
    # Prepare features for detection
    feature_columns = ['amount', 'hour', 'merchant_category', 'days_since_last', 'location_risk']
    features = df[feature_columns].values
    
    print(f"🔧 Using {len(feature_columns)} features: {feature_columns}")
    
    # Mock detection
    service = MockDetectionService()
    result = service.detect(
        data=features,
        algorithm='isolation_forest',
        contamination=0.02  # Expect 2% fraud
    )
    
    print(f"✅ Detection complete in {result.processing_time:.2f}s")
    print(f"🎯 Detected {result.anomaly_count} potential fraud cases")
    print(f"📈 Detection rate: {result.anomaly_count/len(df)*100:.1f}%")
    
    return True


def test_sensor_dataset():
    """Test the IoT sensor readings dataset."""
    print("\n🌡️ Testing IoT Sensor Readings Dataset")
    print("-" * 50)
    
    # Load dataset
    df = pd.read_csv('datasets/sensor_readings.csv')
    print(f"📊 Loaded {len(df)} sensor readings")
    print(f"🚨 Contains {df['is_malfunction'].sum()} actual malfunctions ({df['is_malfunction'].sum()/len(df)*100:.1f}%)")
    
    # Check timestamp parsing
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    time_range = df['timestamp'].max() - df['timestamp'].min()
    print(f"⏱️ Time range: {time_range}")
    
    # Prepare features
    feature_columns = ['temperature', 'humidity', 'pressure', 'vibration', 'power_consumption']
    features = df[feature_columns].values
    
    print(f"🔧 Using {len(feature_columns)} features: {feature_columns}")
    
    # Show some basic statistics
    print("📈 Feature statistics:")
    for col in feature_columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        print(f"   {col}: mean={mean_val:.2f}, std={std_val:.2f}")
    
    # Mock detection
    service = MockDetectionService()
    result = service.detect(
        data=features,
        algorithm='isolation_forest',
        contamination=0.03  # Expect 3% malfunctions
    )
    
    print(f"✅ Detection complete in {result.processing_time:.2f}s")
    print(f"🎯 Detected {result.anomaly_count} potential malfunctions")
    
    return True


def test_mixed_features_dataset():
    """Test the mixed features dataset."""
    print("\n🔀 Testing Mixed Features Dataset")
    print("-" * 50)
    
    # Load dataset
    df = pd.read_csv('datasets/mixed_features.csv')
    print(f"📊 Loaded {len(df)} samples")
    print(f"🚨 Contains {df['is_anomaly'].sum()} actual anomalies ({df['is_anomaly'].sum()/len(df)*100:.1f}%)")
    
    # Show feature types
    print("📋 Feature information:")
    for col in df.columns:
        if col != 'is_anomaly':
            dtype = df[col].dtype
            unique_vals = df[col].nunique()
            print(f"   {col}: {dtype}, {unique_vals} unique values")
    
    # Prepare features (exclude ground truth)
    feature_columns = [col for col in df.columns if col != 'is_anomaly']
    features = df[feature_columns].values
    
    # Mock detection
    service = MockDetectionService()
    result = service.detect(
        data=features,
        algorithm='isolation_forest',
        contamination=0.15  # Expect 15% anomalies
    )
    
    print(f"✅ Detection complete in {result.processing_time:.2f}s")
    print(f"🎯 Detected {result.anomaly_count} potential anomalies")
    
    return True


def main():
    """Run all dataset tests."""
    print("🧪 Testing Generated Example Datasets")
    print("=" * 60)
    
    # Check if datasets directory exists
    datasets_dir = Path('datasets')
    if not datasets_dir.exists():
        print("❌ Datasets directory not found. Please run generate_example_data.py first.")
        return False
    
    # List available datasets
    csv_files = list(datasets_dir.glob('*.csv'))
    print(f"📁 Found {len(csv_files)} datasets:")
    for csv_file in csv_files:
        size_mb = csv_file.stat().st_size / (1024 * 1024)
        print(f"   • {csv_file.name} ({size_mb:.1f} MB)")
    
    # Run specific tests
    tests = [
        test_credit_card_dataset,
        test_sensor_dataset,
        test_mixed_features_dataset
    ]
    
    passed = 0
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All datasets are working correctly!")
        print("\n💡 Next steps:")
        print("   1. Use these datasets with the quickstart templates")
        print("   2. Try different algorithms and parameters")
        print("   3. Compare results with ground truth labels")
        print("   4. Experiment with feature engineering")
        return True
    else:
        print("⚠️ Some tests failed. Check the datasets and try again.")
        return False


if __name__ == "__main__":
    main()