#!/usr/bin/env python3
"""
Working example of Pynomaly Detection package.

This example demonstrates the basic usage of the anomaly detection package
with a minimal, working implementation.
"""

import numpy as np
import sys
from pathlib import Path

# Add the src directory to the path for development
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from pynomaly_detection import AnomalyDetector, get_default_detector


def generate_sample_data():
    """Generate sample data with known anomalies."""
    np.random.seed(42)
    
    # Create normal data (90 samples)
    normal_data = np.random.randn(90, 5)
    
    # Create anomalous data (10 samples with clear outliers)
    anomalous_data = np.random.randn(10, 5) + 4
    
    # Combine and shuffle
    X = np.vstack([normal_data, anomalous_data])
    indices = np.random.permutation(len(X))
    X = X[indices]
    
    return X


def basic_usage_example():
    """Basic usage example."""
    print("🚀 Basic Usage Example")
    print("=" * 30)
    
    # Generate sample data
    X = generate_sample_data()
    print(f"Generated {len(X)} samples with {X.shape[1]} features")
    
    # Create detector
    detector = AnomalyDetector()
    
    # Fit the model
    detector.fit(X)
    print("✓ Model fitted successfully")
    
    # Make predictions
    predictions = detector.predict(X)
    anomaly_count = np.sum(predictions) if hasattr(predictions, 'sum') else sum(predictions)
    
    print(f"✓ Detected {anomaly_count} anomalies out of {len(X)} samples")
    print(f"✓ Anomaly rate: {anomaly_count/len(X)*100:.1f}%")
    
    return detector, predictions


def advanced_usage_example():
    """Advanced usage with configuration."""
    print("\n🔧 Advanced Usage Example")
    print("=" * 30)
    
    # Generate sample data
    X = generate_sample_data()
    
    # Create detector with configuration
    detector = AnomalyDetector()
    
    # Fit with specific contamination rate
    detector.fit(X, contamination=0.1, random_state=42)  # Expect 10% anomalies
    
    # Make predictions
    predictions = detector.predict(X)
    anomaly_count = np.sum(predictions) if hasattr(predictions, 'sum') else sum(predictions)
    
    print(f"✓ With contamination=0.1: {anomaly_count} anomalies detected")
    print(f"✓ Anomaly rate: {anomaly_count/len(X)*100:.1f}%")
    
    return detector, predictions


def default_detector_example():
    """Example using the default detector."""
    print("\n⚡ Default Detector Example")
    print("=" * 30)
    
    # Generate sample data
    X = generate_sample_data()
    
    # Get default detector
    detector = get_default_detector()
    
    # Fit and predict
    detector.fit(X)
    predictions = detector.predict(X)
    anomaly_count = np.sum(predictions) if hasattr(predictions, 'sum') else sum(predictions)
    
    print(f"✓ Default detector: {anomaly_count} anomalies detected")
    print(f"✓ Anomaly rate: {anomaly_count/len(X)*100:.1f}%")
    
    return detector, predictions


def main():
    """Run all examples."""
    print("Pynomaly Detection - Working Examples")
    print("=" * 50)
    
    try:
        # Basic usage
        basic_detector, basic_predictions = basic_usage_example()
        
        # Advanced usage
        advanced_detector, advanced_predictions = advanced_usage_example()
        
        # Default detector
        default_detector, default_predictions = default_detector_example()
        
        print("\n🎉 All examples completed successfully!")
        print("\nNext steps:")
        print("- Try with your own data")
        print("- Explore different contamination rates")
        print("- Check the full API documentation")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your installation and try again.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)