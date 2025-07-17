#!/usr/bin/env python3
"""
Quick Start Example - Get Started in 5 Minutes

This is the fastest way to get started with anomaly detection.
Simple, focused example that gets you up and running quickly.
"""

import numpy as np
from sklearn.datasets import make_blobs
from pynomaly_detection.application.use_cases.detect_anomalies import DetectAnomaliesUseCase
from pynomaly_detection.domain.entities.detector import Detector
from pynomaly_detection.domain.value_objects.contamination_rate import ContaminationRate


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
    detector = Detector(
        name="quick_detector",
        algorithm="isolation_forest",
        contamination=ContaminationRate(0.1)
    )
    print(f"   ✓ Using {detector.algorithm} algorithm")
    
    # Detect anomalies
    print("🔍 Detecting anomalies...")
    use_case = DetectAnomaliesUseCase()
    
    # Convert to DataFrame format
    import pandas as pd
    df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
    
    result = use_case.execute(detector, df)
    
    anomalies_found = sum(result.predictions)
    print(f"   ✓ Found {anomalies_found} anomalies")
    
    # Show results
    print("\n📈 Results:")
    for i, (pred, score) in enumerate(zip(result.predictions, result.scores)):
        if pred == 1:  # Anomaly
            print(f"   🚨 Sample {i}: ANOMALY (score: {score:.3f})")
    
    print("\n🎉 Success! You've completed your first anomaly detection.")
    print("\n📚 What's next?")
    print("   - Try basic_usage.py for a more detailed example")
    print("   - Explore ensemble_detection.py for advanced techniques")
    print("   - Check out the docs/ folder for comprehensive guides")
    
    return True


if __name__ == "__main__":
    main()