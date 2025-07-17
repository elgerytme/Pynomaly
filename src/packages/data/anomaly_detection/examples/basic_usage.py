#!/usr/bin/env python3
"""
Basic Anomaly Detection Example

This example demonstrates the fundamental usage of the anomaly detection package.
It shows how to:
1. Load and prepare data
2. Initialize a detector
3. Fit the model
4. Detect anomalies
5. Analyze results
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from pynomaly_detection.application.use_cases.detect_anomalies import DetectAnomaliesUseCase
from pynomaly_detection.domain.entities.detector import Detector
from pynomaly_detection.domain.value_objects.contamination_rate import ContaminationRate


def generate_sample_data(n_samples=1000, contamination=0.1):
    """
    Generate sample data with anomalies.
    
    Args:
        n_samples: Number of samples to generate
        contamination: Fraction of outliers
    
    Returns:
        tuple: (X, y) where X is features and y is true labels
    """
    # Generate normal data
    X, _ = make_classification(
        n_samples=int(n_samples * (1 - contamination)),
        n_features=20,
        n_informative=10,
        n_redundant=10,
        random_state=42
    )
    
    # Add some anomalies
    n_anomalies = int(n_samples * contamination)
    anomalies = np.random.uniform(
        low=X.min() - 2, 
        high=X.max() + 2, 
        size=(n_anomalies, X.shape[1])
    )
    
    # Combine normal and anomalous data
    X_combined = np.vstack([X, anomalies])
    y_true = np.hstack([np.zeros(len(X)), np.ones(n_anomalies)])
    
    return X_combined, y_true


def main():
    """
    Main function demonstrating basic anomaly detection workflow.
    """
    print("Basic Anomaly Detection Example")
    print("=" * 40)
    
    # Step 1: Generate sample data
    print("\n1. Generating sample data...")
    X, y_true = generate_sample_data(n_samples=1000, contamination=0.1)
    print(f"   Generated {len(X)} samples with {sum(y_true)} anomalies")
    
    # Step 2: Initialize detector
    print("\n2. Initializing detector...")
    detector = Detector(
        name="basic_detector",
        algorithm="isolation_forest",
        contamination=ContaminationRate(0.1)
    )
    print(f"   Detector: {detector.name} using {detector.algorithm}")
    
    # Step 3: Create use case and detect anomalies
    print("\n3. Running anomaly detection...")
    use_case = DetectAnomaliesUseCase()
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    
    try:
        result = use_case.execute(detector, df)
        print(f"   Detection completed successfully")
        print(f"   Found {sum(result.predictions)} anomalies")
        
        # Step 4: Analyze results
        print("\n4. Analysis Results:")
        print(f"   True anomalies: {sum(y_true)}")
        print(f"   Detected anomalies: {sum(result.predictions)}")
        
        # Calculate basic metrics
        if len(result.predictions) == len(y_true):
            true_positives = sum((y_true == 1) & (result.predictions == 1))
            false_positives = sum((y_true == 0) & (result.predictions == 1))
            false_negatives = sum((y_true == 1) & (result.predictions == 0))
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            print(f"   Precision: {precision:.3f}")
            print(f"   Recall: {recall:.3f}")
        
        # Show score distribution
        scores = result.scores
        print(f"\n5. Score Statistics:")
        print(f"   Min score: {min(scores):.3f}")
        print(f"   Max score: {max(scores):.3f}")
        print(f"   Mean score: {np.mean(scores):.3f}")
        print(f"   Std score: {np.std(scores):.3f}")
        
        # Show top anomalies
        print(f"\n6. Top 5 Anomalies (by score):")
        sorted_indices = np.argsort(scores)[::-1]
        for i, idx in enumerate(sorted_indices[:5]):
            print(f"   {i+1}. Sample {idx}: score = {scores[idx]:.3f}")
        
    except Exception as e:
        print(f"   Error during detection: {str(e)}")
        return False
    
    print("\n" + "=" * 40)
    print("Basic example completed successfully!")
    print("\nNext steps:")
    print("- Try different algorithms (isolation_forest, local_outlier_factor, etc.)")
    print("- Experiment with different contamination rates")
    print("- Check out ensemble_detection.py for advanced techniques")
    print("- Explore automl_optimization.py for automated tuning")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)