#!/usr/bin/env python3
"""
Basic anomaly_detection Usage Example

This example demonstrates the fundamental features of anomaly_detection:
- Loading and preparing data
- Initializing anomaly detectors
- Fitting models and detecting anomalies
- Visualizing results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from typing import Tuple, Optional

# Note: These imports would be actual anomaly_detection imports in a real implementation
# from anomaly_detection import AnomalyDetector
# from anomaly_detection.visualization import plot_anomalies


class AnomalyDetector:
    """Demo anomaly detector for example purposes."""
    
    def __init__(self, algorithm: str = 'isolation_forest', contamination: float = 0.1):
        self.algorithm = algorithm
        self.contamination = contamination
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame) -> 'AnomalyDetector':
        """Fit the anomaly detector."""
        print(f"Fitting {self.algorithm} on {len(X)} samples...")
        self.is_fitted = True
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict anomalies (1 for normal, -1 for anomaly)."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
            
        # Demo prediction - randomly mark some points as anomalies
        np.random.seed(42)
        n_anomalies = int(len(X) * self.contamination)
        predictions = np.ones(len(X))
        anomaly_indices = np.random.choice(len(X), n_anomalies, replace=False)
        predictions[anomaly_indices] = -1
        
        return predictions
        
    def decision_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Get anomaly scores for each sample."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before scoring")
            
        # Demo scores - random scores with anomalies having higher scores
        np.random.seed(42)
        scores = np.random.normal(0, 1, len(X))
        predictions = self.predict(X)
        scores[predictions == -1] += 2  # Anomalies get higher scores
        
        return scores


def create_sample_data(n_samples: int = 1000, n_anomalies: int = 50) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Create sample data with known anomalies for demonstration.
    
    Args:
        n_samples: Total number of samples
        n_anomalies: Number of anomalies to inject
        
    Returns:
        DataFrame with features and ground truth labels
    """
    # Create normal data
    X_normal, _ = make_blobs(
        n_samples=n_samples - n_anomalies,
        centers=2,
        cluster_std=1.0,
        center_box=(-5.0, 5.0),
        random_state=42
    )
    
    # Create anomalous data
    np.random.seed(42)
    X_anomalies = np.random.uniform(low=-10, high=10, size=(n_anomalies, 2))
    
    # Combine data
    X = np.vstack([X_normal, X_anomalies])
    y_true = np.hstack([
        np.ones(n_samples - n_anomalies),  # Normal points
        -np.ones(n_anomalies)              # Anomalous points
    ])
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
    
    return df, y_true


def visualize_results(
    data: pd.DataFrame,
    predictions: np.ndarray,
    scores: Optional[np.ndarray] = None,
    y_true: Optional[np.ndarray] = None
) -> None:
    """
    Visualize detection results.
    
    Args:
        data: Input features
        predictions: Predicted labels (-1 for anomaly, 1 for normal)
        scores: Anomaly scores (optional)
        y_true: Ground truth labels (optional)
    """
    fig, axes = plt.subplots(1, 3 if scores is not None else 2, figsize=(15, 5))
    
    # Plot 1: Predictions
    ax1 = axes[0]
    normal_mask = predictions == 1
    anomaly_mask = predictions == -1
    
    ax1.scatter(
        data.loc[normal_mask, 'feature_1'],
        data.loc[normal_mask, 'feature_2'],
        c='blue', alpha=0.6, label='Normal', s=50
    )
    ax1.scatter(
        data.loc[anomaly_mask, 'feature_1'],
        data.loc[anomaly_mask, 'feature_2'],
        c='red', alpha=0.8, label='Anomaly', s=80, marker='x'
    )
    ax1.set_title('Anomaly Detection Results')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Ground Truth (if available)
    if y_true is not None:
        ax2 = axes[1]
        true_normal_mask = y_true == 1
        true_anomaly_mask = y_true == -1
        
        ax2.scatter(
            data.loc[true_normal_mask, 'feature_1'],
            data.loc[true_normal_mask, 'feature_2'],
            c='lightblue', alpha=0.6, label='True Normal', s=50
        )
        ax2.scatter(
            data.loc[true_anomaly_mask, 'feature_1'],
            data.loc[true_anomaly_mask, 'feature_2'],
            c='darkred', alpha=0.8, label='True Anomaly', s=80, marker='s'
        )
        ax2.set_title('Ground Truth')
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Anomaly Scores (if available)
    if scores is not None:
        ax3 = axes[2 if y_true is not None else 1]
        scatter = ax3.scatter(
            data['feature_1'],
            data['feature_2'],
            c=scores,
            cmap='RdYlBu_r',
            alpha=0.7,
            s=60
        )
        ax3.set_title('Anomaly Scores')
        ax3.set_xlabel('Feature 1')
        ax3.set_ylabel('Feature 2')
        plt.colorbar(scatter, ax=ax3, label='Anomaly Score')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate performance metrics for detection.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with performance metrics
    """
    # Convert to binary format (0 for normal, 1 for anomaly)
    y_true_binary = (y_true == -1).astype(int)
    y_pred_binary = (y_pred == -1).astype(int)
    
    # Calculate metrics
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))  # True Positives
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))  # True Negatives
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))  # False Positives
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))  # False Negatives
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }


def main():
    """Main example execution."""
    print("🔍 anomaly_detection Basic Usage Example")
    print("=" * 50)
    
    # 1. Create sample data
    print("\n1. Creating sample data...")
    data, y_true = create_sample_data(n_samples=1000, n_anomalies=50)
    print(f"   Created dataset with {len(data)} samples")
    print(f"   Features: {list(data.columns)}")
    print(f"   True anomalies: {np.sum(y_true == -1)}")
    
    # 2. Initialize detector
    print("\n2. Initializing anomaly detector...")
    detector = AnomalyDetector(
        algorithm='isolation_forest',
        contamination=0.05  # Expect 5% anomalies
    )
    print(f"   Algorithm: {detector.algorithm}")
    print(f"   Expected contamination: {detector.contamination}")
    
    # 3. Fit the detector
    print("\n3. Fitting the detector...")
    detector.fit(data)
    print("   ✓ Detector fitted successfully")
    
    # 4. Make predictions
    print("\n4. Making predictions...")
    predictions = detector.predict(data)
    scores = detector.decision_scores(data)
    
    n_predicted_anomalies = np.sum(predictions == -1)
    print(f"   Predicted anomalies: {n_predicted_anomalies}")
    print(f"   Anomaly rate: {n_predicted_anomalies / len(data):.2%}")
    
    # 5. Calculate metrics
    print("\n5. Performance Metrics:")
    metrics = calculate_metrics(y_true, predictions)
    
    print(f"   Accuracy:  {metrics['accuracy']:.3f}")
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall:    {metrics['recall']:.3f}")
    print(f"   F1-Score:  {metrics['f1_score']:.3f}")
    
    # 6. Visualize results
    print("\n6. Visualizing results...")
    try:
        visualize_results(data, predictions, scores, y_true)
        print("   ✓ Visualization completed")
    except Exception as e:
        print(f"   ⚠ Visualization failed: {e}")
        print("   (This is normal if running in a headless environment)")
    
    # 7. Advanced usage examples
    print("\n7. Advanced Usage Examples:")
    
    # Multiple algorithms comparison
    algorithms = ['isolation_forest', 'lof', 'ocsvm']
    results = {}
    
    for algo in algorithms:
        detector_algo = AnomalyDetector(algorithm=algo, contamination=0.05)
        detector_algo.fit(data)
        pred_algo = detector_algo.predict(data)
        metrics_algo = calculate_metrics(y_true, pred_algo)
        results[algo] = metrics_algo
        
        print(f"   {algo}: F1={metrics_algo['f1_score']:.3f}, "
              f"Precision={metrics_algo['precision']:.3f}, "
              f"Recall={metrics_algo['recall']:.3f}")
    
    # Find best algorithm
    best_algo = max(results.keys(), key=lambda x: results[x]['f1_score'])
    print(f"\n   🏆 Best algorithm: {best_algo} (F1={results[best_algo]['f1_score']:.3f})")
    
    print("\n" + "=" * 50)
    print("✅ Example completed successfully!")
    print("\nNext steps:")
    print("- Try different algorithms and parameters")
    print("- Experiment with your own data")
    print("- Explore advanced features in other examples")


if __name__ == "__main__":
    main()