#!/usr/bin/env python3
"""
Quick Start Example - Get Started in 5 Minutes

This is the fastest way to get started with anomaly detection.
Simple, focused example that gets you up and running quickly.
"""

import numpy as np
from sklearn.datasets import make_blobs
from pynomaly_detection import AnomalyDetector


def main():
    """
    5-minute quick start example.
    """
    print("🚀 Quick Start - Anomaly Detection in 5 Minutes")
    print("=" * 50)
    
    # Generate some sample data
    print("📊 Generating sample data...")
    X, _ = make_blobs(n_samples=300, centers=1, n_features=2, random_state=42)
    
    # Add a few obvious outliers
    outliers = np.array([[10, 10], [-10, -10], [10, -10]])
    X = np.vstack([X, outliers])
    
    print(f"   ✓ Created {len(X)} samples (including 3 outliers)")
    
    # Create detector
    print("🔧 Setting up detector...")
    detector = AnomalyDetector()
    print(f"   ✓ Using IsolationForest algorithm")
    
    # Detect anomalies
    print("🔍 Detecting anomalies...")
    predictions = detector.detect(X, contamination=0.1)
    
    anomalies_found = sum(predictions)
    print(f"   ✓ Found {anomalies_found} anomalies")
    
    # Show results
    print("\n📈 Results:")
    anomaly_indices = np.where(predictions == 1)[0]
    for i, idx in enumerate(anomaly_indices):
        print(f"   🚨 Sample {idx}: ANOMALY")
    
    print("\n🎉 Success! You've completed your first anomaly detection.")
    print("\n📚 What's next?")
    print("   - Try basic_usage.py for a more detailed example")
    print("   - Explore ensemble_detection.py for advanced techniques")
    print("   - Check out the docs/ folder for comprehensive guides")
    
    return True


if __name__ == "__main__":
    main()