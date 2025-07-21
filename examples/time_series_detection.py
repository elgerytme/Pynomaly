#!/usr/bin/env python3
"""
Time Series Anomaly Detection Example

This example demonstrates time series anomaly detection using anomaly_detection:
- Creating synthetic time series data with seasonal patterns
- Detecting point anomalies, collective anomalies, and trend changes
- Visualizing results with interactive plots
- Handling different types of temporal anomalies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Note: These would be actual anomaly_detection imports in real implementation
# from anomaly_detection import TimeSeriesDetector
# from anomaly_detection.preprocessing import TimeSeriesPreprocessor


class TimeSeriesDetector:
    """Demo time series anomaly detector."""
    
    def __init__(
        self,
        algorithm: str = 'lstm_autoencoder',
        window_size: int = 50,
        contamination: float = 0.05
    ):
        self.algorithm = algorithm
        self.window_size = window_size
        self.contamination = contamination
        self.is_fitted = False
        
    def fit(self, data: pd.DataFrame) -> 'TimeSeriesDetector':
        """Fit the time series detector."""
        print(f"Fitting {self.algorithm} with window_size={self.window_size}")
        print(f"Training on {len(data)} time points...")
        self.is_fitted = True
        return self
        
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict anomalies in time series."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
            
        # Demo: detect anomalies based on z-score threshold
        values = data.iloc[:, -1].values  # Last column is the target
        z_scores = np.abs((values - np.mean(values)) / np.std(values))
        threshold = np.percentile(z_scores, (1 - self.contamination) * 100)
        
        return (z_scores > threshold).astype(int)
        
    def anomaly_scores(self, data: pd.DataFrame) -> np.ndarray:
        """Get anomaly scores."""
        values = data.iloc[:, -1].values
        z_scores = np.abs((values - np.mean(values)) / np.std(values))
        return z_scores
        
    def plot_results(self, data: pd.DataFrame, predictions: np.ndarray) -> None:
        """Plot time series with anomalies highlighted."""
        plt.figure(figsize=(15, 8))
        
        # Plot time series
        plt.subplot(2, 1, 1)
        plt.plot(data.index, data.iloc[:, -1], 'b-', alpha=0.7, label='Time Series')
        
        # Highlight anomalies
        anomaly_mask = predictions == 1
        if np.any(anomaly_mask):
            plt.scatter(
                data.index[anomaly_mask],
                data.iloc[anomaly_mask, -1],
                color='red', s=100, alpha=0.8, label='Anomalies', marker='x'
            )
        
        plt.title(f'Time Series Anomaly Detection ({self.algorithm})')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot anomaly scores
        plt.subplot(2, 1, 2)
        scores = self.anomaly_scores(data)
        plt.plot(data.index, scores, 'g-', alpha=0.7, label='Anomaly Scores')
        
        # Add threshold line
        threshold = np.percentile(scores, (1 - self.contamination) * 100)
        plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label='Threshold')
        
        plt.title('Anomaly Scores')
        plt.xlabel('Time')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def create_synthetic_time_series(
    start_date: str = "2023-01-01",
    end_date: str = "2024-01-01",
    freq: str = "H",
    base_value: float = 100.0,
    trend_slope: float = 0.001,
    seasonal_amplitude: float = 20.0,
    noise_level: float = 5.0
) -> pd.DataFrame:
    """
    Create synthetic time series with seasonal patterns and trend.
    
    Args:
        start_date: Start date for time series
        end_date: End date for time series  
        freq: Frequency of observations
        base_value: Base value around which series oscillates
        trend_slope: Linear trend slope
        seasonal_amplitude: Amplitude of seasonal component
        noise_level: Standard deviation of random noise
        
    Returns:
        DataFrame with timestamp and value columns
    """
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Create base components
    n_points = len(dates)
    time_numeric = np.arange(n_points)
    
    # Linear trend
    trend = trend_slope * time_numeric
    
    # Seasonal components
    # Daily seasonality (24 hour cycle)
    daily_seasonal = seasonal_amplitude * np.sin(2 * np.pi * time_numeric / 24)
    
    # Weekly seasonality (7 day cycle)
    weekly_seasonal = (seasonal_amplitude / 2) * np.sin(2 * np.pi * time_numeric / (24 * 7))
    
    # Random noise
    np.random.seed(42)
    noise = np.random.normal(0, noise_level, n_points)
    
    # Combine all components
    values = base_value + trend + daily_seasonal + weekly_seasonal + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'value': values
    })
    df.set_index('timestamp', inplace=True)
    
    return df


def inject_anomalies(
    data: pd.DataFrame,
    point_anomalies: int = 10,
    collective_anomalies: int = 2,
    trend_changes: int = 1
) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Inject different types of anomalies into time series.
    
    Args:
        data: Original time series data
        point_anomalies: Number of point anomalies to inject
        collective_anomalies: Number of collective anomalies to inject
        trend_changes: Number of trend changes to inject
        
    Returns:
        Modified DataFrame and list of injected anomalies
    """
    data_copy = data.copy()
    anomalies_injected = []
    
    np.random.seed(42)
    
    # 1. Point anomalies (sudden spikes or drops)
    if point_anomalies > 0:
        point_indices = np.random.choice(
            len(data_copy), size=point_anomalies, replace=False
        )
        
        for idx in point_indices:
            original_value = data_copy.iloc[idx, 0]
            # Random spike (positive or negative)
            multiplier = np.random.choice([-3, -2, 3, 4, 5])
            data_copy.iloc[idx, 0] = original_value + (multiplier * data_copy.iloc[:, 0].std())
            
            anomalies_injected.append({
                'type': 'point',
                'index': idx,
                'timestamp': data_copy.index[idx],
                'original_value': original_value,
                'anomaly_value': data_copy.iloc[idx, 0]
            })
    
    # 2. Collective anomalies (unusual subsequences)
    if collective_anomalies > 0:
        for _ in range(collective_anomalies):
            # Random start position and length
            start_idx = np.random.randint(50, len(data_copy) - 100)
            length = np.random.randint(20, 50)
            end_idx = start_idx + length
            
            # Create anomalous pattern
            original_values = data_copy.iloc[start_idx:end_idx, 0].copy()
            
            # Add systematic shift or oscillation
            anomaly_type = np.random.choice(['shift', 'oscillation'])
            if anomaly_type == 'shift':
                # Sudden level shift
                shift_amount = np.random.choice([-2, -1.5, 1.5, 2]) * data_copy.iloc[:, 0].std()
                data_copy.iloc[start_idx:end_idx, 0] += shift_amount
            else:
                # High-frequency oscillation
                t = np.arange(length)
                oscillation = 2 * data_copy.iloc[:, 0].std() * np.sin(2 * np.pi * t / 5)
                data_copy.iloc[start_idx:end_idx, 0] += oscillation
            
            anomalies_injected.append({
                'type': 'collective',
                'start_index': start_idx,
                'end_index': end_idx,
                'start_timestamp': data_copy.index[start_idx],
                'end_timestamp': data_copy.index[end_idx],
                'pattern': anomaly_type,
                'length': length
            })
    
    # 3. Trend changes (change points)
    if trend_changes > 0:
        for _ in range(trend_changes):
            # Random change point
            change_idx = np.random.randint(len(data_copy) // 3, 2 * len(data_copy) // 3)
            
            # Apply trend change from this point onwards
            original_values = data_copy.iloc[change_idx:, 0].copy()
            trend_change = np.random.choice([-0.01, -0.005, 0.005, 0.01])  # per time step
            
            for i in range(change_idx, len(data_copy)):
                time_since_change = i - change_idx
                data_copy.iloc[i, 0] += trend_change * time_since_change
            
            anomalies_injected.append({
                'type': 'trend_change',
                'change_point': change_idx,
                'timestamp': data_copy.index[change_idx],
                'trend_change': trend_change
            })
    
    return data_copy, anomalies_injected


def visualize_time_series_with_anomalies(
    data: pd.DataFrame,
    predictions: np.ndarray,
    injected_anomalies: List[dict],
    scores: Optional[np.ndarray] = None
) -> None:
    """Visualize time series with detected and true anomalies."""
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Original time series with injected anomalies
    ax1 = axes[0]
    ax1.plot(data.index, data.iloc[:, 0], 'b-', alpha=0.7, label='Time Series')
    
    # Mark injected anomalies
    for anomaly in injected_anomalies:
        if anomaly['type'] == 'point':
            ax1.scatter(
                anomaly['timestamp'],
                anomaly['anomaly_value'],
                color='orange', s=150, alpha=0.8, marker='o',
                label='Injected Point Anomaly' if 'Injected Point Anomaly' not in [t.get_text() for t in ax1.get_legend_handles_labels()[1]] else ""
            )
        elif anomaly['type'] == 'collective':
            ax1.axvspan(
                anomaly['start_timestamp'],
                anomaly['end_timestamp'],
                alpha=0.3, color='yellow',
                label='Injected Collective Anomaly' if 'Injected Collective Anomaly' not in [t.get_text() for t in ax1.get_legend_handles_labels()[1]] else ""
            )
        elif anomaly['type'] == 'trend_change':
            ax1.axvline(
                anomaly['timestamp'],
                color='purple', linestyle='--', alpha=0.7,
                label='Injected Trend Change' if 'Injected Trend Change' not in [t.get_text() for t in ax1.get_legend_handles_labels()[1]] else ""
            )
    
    ax1.set_title('Time Series with Injected Anomalies')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Detected anomalies
    ax2 = axes[1]
    ax2.plot(data.index, data.iloc[:, 0], 'b-', alpha=0.7, label='Time Series')
    
    # Mark detected anomalies
    detected_mask = predictions == 1
    if np.any(detected_mask):
        ax2.scatter(
            data.index[detected_mask],
            data.iloc[detected_mask, 0],
            color='red', s=100, alpha=0.8, marker='x',
            label='Detected Anomalies'
        )
    
    ax2.set_title('Detected Anomalies')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Anomaly scores
    if scores is not None:
        ax3 = axes[2]
        ax3.plot(data.index, scores, 'g-', alpha=0.7, label='Anomaly Scores')
        
        # Add threshold
        threshold = np.percentile(scores, 95)  # Top 5% as anomalies
        ax3.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label='Threshold')
        
        ax3.set_title('Anomaly Scores Over Time')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def evaluate_time_series_detection(
    predictions: np.ndarray,
    injected_anomalies: List[dict],
    data: pd.DataFrame,
    tolerance_window: int = 5
) -> dict:
    """
    Evaluate time series anomaly detection performance.
    
    Args:
        predictions: Binary predictions (1 for anomaly, 0 for normal)
        injected_anomalies: List of injected anomalies
        data: Time series data
        tolerance_window: Window for considering detection as correct
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Create ground truth array
    ground_truth = np.zeros(len(data))
    
    # Mark injected anomalies
    for anomaly in injected_anomalies:
        if anomaly['type'] == 'point':
            ground_truth[anomaly['index']] = 1
        elif anomaly['type'] == 'collective':
            ground_truth[anomaly['start_index']:anomaly['end_index']] = 1
        elif anomaly['type'] == 'trend_change':
            # Mark some points around trend change
            start_idx = max(0, anomaly['change_point'] - tolerance_window)
            end_idx = min(len(data), anomaly['change_point'] + tolerance_window)
            ground_truth[start_idx:end_idx] = 1
    
    # Calculate metrics with tolerance
    detected_indices = np.where(predictions == 1)[0]
    true_indices = np.where(ground_truth == 1)[0]
    
    # Count true positives with tolerance
    tp = 0
    for true_idx in true_indices:
        # Check if any detection is within tolerance window
        distances = np.abs(detected_indices - true_idx)
        if len(distances) > 0 and np.min(distances) <= tolerance_window:
            tp += 1
    
    fp = len(detected_indices) - tp
    fn = len(true_indices) - tp
    tn = len(data) - len(detected_indices) - len(true_indices) + tp
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn,
        'total_injected_anomalies': len(true_indices),
        'total_detected_anomalies': len(detected_indices)
    }


def main():
    """Main example execution."""
    print("⏰ anomaly_detection Time Series Anomaly Detection Example")
    print("=" * 60)
    
    # 1. Create synthetic time series
    print("\n1. Creating synthetic time series...")
    ts_data = create_synthetic_time_series(
        start_date="2023-01-01",
        end_date="2023-02-01",  # One month of hourly data
        freq="H",
        base_value=100.0,
        trend_slope=0.002,
        seasonal_amplitude=15.0,
        noise_level=3.0
    )
    
    print(f"   Created time series with {len(ts_data)} data points")
    print(f"   Date range: {ts_data.index.min()} to {ts_data.index.max()}")
    print(f"   Value range: {ts_data.iloc[:, 0].min():.2f} to {ts_data.iloc[:, 0].max():.2f}")
    
    # 2. Inject anomalies
    print("\n2. Injecting anomalies...")
    anomalous_data, injected_anomalies = inject_anomalies(
        ts_data,
        point_anomalies=8,
        collective_anomalies=3,
        trend_changes=2
    )
    
    print(f"   Injected {len(injected_anomalies)} anomalies:")
    for i, anomaly in enumerate(injected_anomalies):
        if anomaly['type'] == 'point':
            print(f"     {i+1}. Point anomaly at {anomaly['timestamp']}")
        elif anomaly['type'] == 'collective':
            print(f"     {i+1}. Collective anomaly ({anomaly['pattern']}) from {anomaly['start_timestamp']} to {anomaly['end_timestamp']}")
        elif anomaly['type'] == 'trend_change':
            print(f"     {i+1}. Trend change at {anomaly['timestamp']}")
    
    # 3. Initialize time series detector
    print("\n3. Initializing time series detector...")
    detector = TimeSeriesDetector(
        algorithm='lstm_autoencoder',
        window_size=24,  # 24-hour window
        contamination=0.08  # Expect 8% anomalies
    )
    
    # 4. Fit and detect
    print("\n4. Fitting detector and detecting anomalies...")
    detector.fit(anomalous_data)
    predictions = detector.predict(anomalous_data)
    scores = detector.anomaly_scores(anomalous_data)
    
    n_detected = np.sum(predictions)
    print(f"   Detected {n_detected} anomalies ({n_detected/len(anomalous_data):.1%} of data)")
    
    # 5. Evaluate performance
    print("\n5. Evaluating detection performance...")
    metrics = evaluate_time_series_detection(
        predictions, injected_anomalies, anomalous_data, tolerance_window=3
    )
    
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall:    {metrics['recall']:.3f}")
    print(f"   F1-Score:  {metrics['f1_score']:.3f}")
    print(f"   True Positives:  {metrics['true_positives']}")
    print(f"   False Positives: {metrics['false_positives']}")
    print(f"   False Negatives: {metrics['false_negatives']}")
    
    # 6. Visualize results
    print("\n6. Visualizing results...")
    try:
        visualize_time_series_with_anomalies(
            anomalous_data, predictions, injected_anomalies, scores
        )
        print("   ✓ Visualization completed")
    except Exception as e:
        print(f"   ⚠ Visualization failed: {e}")
    
    # 7. Advanced analysis
    print("\n7. Advanced Time Series Analysis:")
    
    # Seasonal decomposition simulation
    print("\n   Seasonal Decomposition Analysis:")
    daily_pattern = anomalous_data.groupby(anomalous_data.index.hour).mean()
    print(f"     Peak hours: {daily_pattern.idxmax().values[0]:02d}:00")
    print(f"     Lowest hours: {daily_pattern.idxmin().values[0]:02d}:00")
    print(f"     Daily variation: ±{(daily_pattern.max() - daily_pattern.min()).values[0]:.2f}")
    
    # Anomaly distribution analysis
    if n_detected > 0:
        detected_hours = anomalous_data.index[predictions == 1].hour
        most_common_hour = pd.Series(detected_hours).mode().iloc[0]
        print(f"\n   Anomaly Distribution:")
        print(f"     Most anomalous hour: {most_common_hour:02d}:00")
        print(f"     Hourly spread: {len(np.unique(detected_hours))} different hours")
    
    # Compare algorithms
    print("\n8. Algorithm Comparison:")
    algorithms = ['lstm_autoencoder', 'isolation_forest', 'statistical_threshold']
    
    for algo in algorithms:
        detector_algo = TimeSeriesDetector(
            algorithm=algo,
            window_size=24,
            contamination=0.08
        )
        detector_algo.fit(anomalous_data)
        pred_algo = detector_algo.predict(anomalous_data)
        metrics_algo = evaluate_time_series_detection(
            pred_algo, injected_anomalies, anomalous_data
        )
        
        print(f"   {algo:20} F1={metrics_algo['f1_score']:.3f} "
              f"P={metrics_algo['precision']:.3f} "
              f"R={metrics_algo['recall']:.3f}")
    
    print("\n" + "=" * 60)
    print("✅ Time Series Example completed!")
    print("\nKey Insights:")
    print("- Point anomalies are easiest to detect")
    print("- Collective anomalies require context awareness")
    print("- Trend changes need change-point detection")
    print("- Seasonal patterns help distinguish normal vs. anomalous behavior")
    
    print("\nNext Steps:")
    print("- Try different window sizes and contamination rates")
    print("- Experiment with real-world time series data")
    print("- Combine multiple detection methods for better performance")


if __name__ == "__main__":
    main()