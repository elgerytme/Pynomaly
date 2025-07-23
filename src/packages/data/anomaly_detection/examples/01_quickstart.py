#!/usr/bin/env python3
"""
Quickstart Example for Anomaly Detection Package

This example demonstrates basic usage of the anomaly detection package
with various algorithms and simple visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import anomaly detection components
try:
    from anomaly_detection import DetectionService, EnsembleService
    from anomaly_detection.domain.entities.detection_result import DetectionResult
except ImportError:
    print("Please install the anomaly_detection package first:")
    print("pip install -e .")
    exit(1)


def generate_sample_data(n_samples: int = 1000, contamination: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data with known anomalies for demonstration.
    
    Args:
        n_samples: Total number of samples
        contamination: Fraction of anomalies
        
    Returns:
        X: Feature matrix
        y_true: True labels (1 for normal, -1 for anomaly)
    """
    np.random.seed(42)
    
    n_anomalies = int(n_samples * contamination)
    n_normal = n_samples - n_anomalies
    
    # Generate normal data (centered around origin)
    normal_data = np.random.randn(n_normal, 2)
    normal_data = normal_data @ np.array([[1.5, 0.5], [0.5, 1.0]])  # Add correlation
    
    # Generate anomalies (far from normal data)
    anomaly_angles = np.random.uniform(0, 2 * np.pi, n_anomalies)
    anomaly_radius = np.random.uniform(4, 6, n_anomalies)
    anomaly_data = np.column_stack([
        anomaly_radius * np.cos(anomaly_angles),
        anomaly_radius * np.sin(anomaly_angles)
    ])
    
    # Combine data
    X = np.vstack([normal_data, anomaly_data])
    y_true = np.hstack([np.ones(n_normal), -np.ones(n_anomalies)])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y_true[indices]


def visualize_results(X: np.ndarray, result: Dict[str, Any], title: str = "Anomaly Detection Results"):
    """
    Visualize anomaly detection results in 2D.
    
    Args:
        X: Feature matrix
        result: Detection results
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    
    # Get predictions
    predictions = result.get('predictions', np.ones(len(X)))
    anomaly_mask = predictions == -1
    normal_mask = ~anomaly_mask
    
    # Plot normal points
    plt.scatter(X[normal_mask, 0], X[normal_mask, 1], 
                c='blue', alpha=0.6, label='Normal', s=50)
    
    # Plot anomalies
    plt.scatter(X[anomaly_mask, 0], X[anomaly_mask, 1], 
                c='red', alpha=0.8, label='Anomaly', 
                marker='x', s=100, linewidths=2)
    
    # Add contour plot if scores available
    if 'anomaly_scores' in result and len(result['anomaly_scores']) == len(X):
        scores = result['anomaly_scores']
        
        # Create grid for contour plot
        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 50),
            np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 50)
        )
        
        # Interpolate scores on grid (simplified visualization)
        from scipy.interpolate import griddata
        zz = griddata((X[:, 0], X[:, 1]), scores, (xx, yy), method='linear')
        
        # Plot contours
        contour = plt.contour(xx, yy, zz, levels=10, colors='black', alpha=0.4, linewidths=0.5)
        plt.clabel(contour, inline=True, fontsize=8)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_results(result: Dict[str, Any], y_true: np.ndarray = None):
    """
    Print detection results and metrics.
    
    Args:
        result: Detection results
        y_true: True labels (optional, for evaluation)
    """
    print(f"\nAlgorithm: {result.get('algorithm', 'Unknown')}")
    print(f"Total samples: {result.get('total_samples', 0)}")
    print(f"Anomalies detected: {result.get('anomaly_count', 0)}")
    print(f"Anomaly rate: {result.get('anomaly_rate', 0):.2%}")
    
    if y_true is not None and 'predictions' in result:
        predictions = result['predictions']
        
        # Calculate metrics
        true_positives = np.sum((y_true == -1) & (predictions == -1))
        false_positives = np.sum((y_true == 1) & (predictions == -1))
        true_negatives = np.sum((y_true == 1) & (predictions == 1))
        false_negatives = np.sum((y_true == -1) & (predictions == 1))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nEvaluation Metrics:")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1_score:.3f}")
    
    print("-" * 50)


def example_1_basic_detection():
    """Example 1: Basic anomaly detection with Isolation Forest."""
    print("\n" + "="*60)
    print("Example 1: Basic Anomaly Detection with Isolation Forest")
    print("="*60)
    
    # Generate data
    X, y_true = generate_sample_data(n_samples=500, contamination=0.1)
    print(f"Generated {len(X)} samples with {np.sum(y_true == -1)} true anomalies")
    
    # Initialize detection service
    service = DetectionService()
    
    # Detect anomalies
    result = service.detect_anomalies(
        data=X,
        algorithm='iforest',
        contamination=0.1,
        n_estimators=100,
        random_state=42
    )
    
    # Convert to dict for compatibility
    result_dict = {
        'algorithm': 'iforest',
        'predictions': result.predictions,
        'anomaly_scores': result.anomaly_scores,
        'total_samples': len(X),
        'anomaly_count': result.anomaly_count,
        'anomaly_rate': result.anomaly_rate
    }
    
    print_results(result_dict, y_true)
    visualize_results(X, result_dict, "Isolation Forest Results")


def example_2_multiple_algorithms():
    """Example 2: Compare multiple algorithms on the same data."""
    print("\n" + "="*60)
    print("Example 2: Comparing Multiple Algorithms")
    print("="*60)
    
    # Generate data
    X, y_true = generate_sample_data(n_samples=500, contamination=0.1)
    
    # Initialize service
    service = DetectionService()
    
    # Algorithms to test
    algorithms = ['iforest', 'lof', 'ocsvm']
    
    # Store results
    results = {}
    
    for algorithm in algorithms:
        try:
            result = service.detect_anomalies(
                data=X,
                algorithm=algorithm,
                contamination=0.1,
                random_state=42
            )
            
            results[algorithm] = {
                'algorithm': algorithm,
                'predictions': result.predictions,
                'anomaly_scores': result.anomaly_scores,
                'total_samples': len(X),
                'anomaly_count': result.anomaly_count,
                'anomaly_rate': result.anomaly_rate
            }
            
            print(f"\n{algorithm.upper()}:")
            print_results(results[algorithm], y_true)
            
        except Exception as e:
            print(f"Error with {algorithm}: {e}")
    
    # Visualize all results
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
    for idx, (algo, result) in enumerate(results.items()):
        ax = axes[idx]
        
        predictions = result['predictions']
        anomaly_mask = predictions == -1
        normal_mask = ~anomaly_mask
        
        ax.scatter(X[normal_mask, 0], X[normal_mask, 1], 
                   c='blue', alpha=0.6, s=30)
        ax.scatter(X[anomaly_mask, 0], X[anomaly_mask, 1], 
                   c='red', alpha=0.8, marker='x', s=60)
        ax.set_title(algo.upper())
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def example_3_ensemble_detection():
    """Example 3: Ensemble anomaly detection."""
    print("\n" + "="*60)
    print("Example 3: Ensemble Anomaly Detection")
    print("="*60)
    
    # Generate data
    X, y_true = generate_sample_data(n_samples=500, contamination=0.1)
    
    # Initialize ensemble service
    ensemble = EnsembleService()
    
    # Detect using ensemble
    result = ensemble.detect_with_ensemble(
        data=X,
        algorithms=['iforest', 'lof'],
        method='majority',
        contamination=0.1
    )
    
    result_dict = {
        'algorithm': 'ensemble_majority',
        'predictions': result.predictions,
        'anomaly_scores': result.anomaly_scores if hasattr(result, 'anomaly_scores') else None,
        'total_samples': len(X),
        'anomaly_count': result.anomaly_count,
        'anomaly_rate': result.anomaly_rate
    }
    
    print_results(result_dict, y_true)
    visualize_results(X, result_dict, "Ensemble Detection Results (Majority Voting)")


def example_4_parameter_sensitivity():
    """Example 4: Parameter sensitivity analysis."""
    print("\n" + "="*60)
    print("Example 4: Parameter Sensitivity Analysis")
    print("="*60)
    
    # Generate data
    X, y_true = generate_sample_data(n_samples=500, contamination=0.1)
    
    # Initialize service
    service = DetectionService()
    
    # Test different contamination rates
    contamination_rates = [0.05, 0.1, 0.15, 0.2]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, contamination in enumerate(contamination_rates):
        result = service.detect_anomalies(
            data=X,
            algorithm='iforest',
            contamination=contamination,
            random_state=42
        )
        
        ax = axes[idx]
        predictions = result.predictions
        anomaly_mask = predictions == -1
        normal_mask = ~anomaly_mask
        
        ax.scatter(X[normal_mask, 0], X[normal_mask, 1], 
                   c='blue', alpha=0.6, s=30)
        ax.scatter(X[anomaly_mask, 0], X[anomaly_mask, 1], 
                   c='red', alpha=0.8, marker='x', s=60)
        ax.set_title(f'Contamination = {contamination}')
        ax.grid(True, alpha=0.3)
        
        # Calculate metrics
        if y_true is not None:
            tp = np.sum((y_true == -1) & (predictions == -1))
            fp = np.sum((y_true == 1) & (predictions == -1))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / np.sum(y_true == -1) if np.sum(y_true == -1) > 0 else 0
            ax.text(0.02, 0.98, f'Prec: {precision:.2f}\nRec: {recall:.2f}', 
                    transform=ax.transAxes, va='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Effect of Contamination Parameter on Detection Results', fontsize=14)
    plt.tight_layout()
    plt.show()


def example_5_realtime_detection():
    """Example 5: Simulated real-time anomaly detection."""
    print("\n" + "="*60)
    print("Example 5: Real-time Anomaly Detection Simulation")
    print("="*60)
    
    # Initialize service
    service = DetectionService()
    
    # Train initial model on normal data
    print("Training initial model on normal data...")
    normal_data = np.random.randn(200, 2)
    service.fit(normal_data, algorithm='iforest')
    
    # Simulate streaming data
    print("\nSimulating real-time data stream...")
    print("(Press Ctrl+C to stop)")
    
    plt.ion()  # Interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Initialize plot data
    all_data = []
    anomaly_counts = []
    
    try:
        for i in range(100):
            # Generate new data point
            if np.random.random() < 0.9:  # 90% normal
                new_point = np.random.randn(1, 2)
            else:  # 10% anomaly
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(4, 6)
                new_point = np.array([[radius * np.cos(angle), radius * np.sin(angle)]])
            
            # Detect anomaly
            result = service.predict(new_point, algorithm='iforest')
            is_anomaly = result.predictions[0] == -1
            
            # Store data
            all_data.append((new_point[0], is_anomaly))
            anomaly_counts.append(len([d for d in all_data[-50:] if d[1]]))
            
            # Update plots
            ax1.clear()
            ax2.clear()
            
            # Plot 1: Scatter plot
            if len(all_data) > 0:
                normal_points = np.array([d[0] for d in all_data if not d[1]])
                anomaly_points = np.array([d[0] for d in all_data if d[1]])
                
                if len(normal_points) > 0:
                    ax1.scatter(normal_points[:, 0], normal_points[:, 1], 
                               c='blue', alpha=0.6, s=30, label='Normal')
                if len(anomaly_points) > 0:
                    ax1.scatter(anomaly_points[:, 0], anomaly_points[:, 1], 
                               c='red', alpha=0.8, marker='x', s=60, label='Anomaly')
                
                # Highlight current point
                color = 'red' if is_anomaly else 'green'
                ax1.scatter(new_point[0, 0], new_point[0, 1], 
                           c=color, s=200, marker='o', edgecolors='black', linewidths=2)
            
            ax1.set_xlim(-8, 8)
            ax1.set_ylim(-8, 8)
            ax1.set_title('Real-time Detection')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Anomaly count over time
            ax2.plot(anomaly_counts, 'b-', linewidth=2)
            ax2.fill_between(range(len(anomaly_counts)), 0, anomaly_counts, alpha=0.3)
            ax2.set_xlabel('Time Window')
            ax2.set_ylabel('Anomaly Count (last 50 points)')
            ax2.set_title('Anomaly Trend')
            ax2.grid(True, alpha=0.3)
            
            plt.pause(0.1)  # Small pause for animation
            
            if i % 10 == 0:
                print(f"Processed {i+1} points, {sum(d[1] for d in all_data)} anomalies detected")
    
    except KeyboardInterrupt:
        print("\nStopping real-time detection...")
    
    plt.ioff()
    plt.show()
    
    print(f"\nTotal points processed: {len(all_data)}")
    print(f"Total anomalies detected: {sum(d[1] for d in all_data)}")
    print(f"Detection rate: {sum(d[1] for d in all_data) / len(all_data) * 100:.1f}%")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("ANOMALY DETECTION PACKAGE - QUICKSTART EXAMPLES")
    print("="*60)
    
    examples = [
        ("Basic Detection", example_1_basic_detection),
        ("Multiple Algorithms", example_2_multiple_algorithms),
        ("Ensemble Detection", example_3_ensemble_detection),
        ("Parameter Sensitivity", example_4_parameter_sensitivity),
        ("Real-time Detection", example_5_realtime_detection)
    ]
    
    while True:
        print("\nAvailable Examples:")
        for i, (name, _) in enumerate(examples, 1):
            print(f"{i}. {name}")
        print("0. Exit")
        
        try:
            choice = int(input("\nSelect an example (0-5): "))
            if choice == 0:
                print("Exiting...")
                break
            elif 1 <= choice <= len(examples):
                examples[choice-1][1]()
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error running example: {e}")


if __name__ == "__main__":
    main()