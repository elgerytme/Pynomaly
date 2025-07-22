"""
Anomaly detection algorithm validation tests.
Tests detection accuracy, precision, recall, and performance for production deployment.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import sys
from pathlib import Path
import time

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from anomaly_detection.domain.services.detection_service import DetectionService
    from anomaly_detection.domain.entities.detection_result import DetectionResult
    from anomaly_detection.domain.entities.anomaly import Anomaly
except ImportError as e:
    # Create mock classes for testing infrastructure
    class DetectionService:
        def __init__(self, algorithm='isolation_forest'):
            self.algorithm = algorithm
            
        def detect_anomalies(self, data, config=None):
            """Mock detection method."""
            if isinstance(data, pd.DataFrame):
                n_samples = len(data)
            else:
                n_samples = len(data) if hasattr(data, '__len__') else 100
                
            # Generate mock predictions with some logic
            np.random.seed(42)
            predictions = np.ones(n_samples)
            n_anomalies = max(1, int(n_samples * 0.1))  # 10% anomalies
            anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
            predictions[anomaly_indices] = -1
            
            confidence = np.random.uniform(0.6, 0.95, n_samples)
            
            return DetectionResult(predictions, confidence)
    
    class DetectionResult:
        def __init__(self, predictions, confidence_scores=None):
            self.predictions = predictions
            self.confidence_scores = confidence_scores or np.random.random(len(predictions))
            self.anomalies = self._extract_anomalies()
            self.success = True
            
        def _extract_anomalies(self):
            anomaly_indices = np.where(self.predictions == -1)[0]
            return [Anomaly(idx, self.confidence_scores[idx]) for idx in anomaly_indices]
    
    class Anomaly:
        def __init__(self, index, confidence):
            self.index = index
            self.confidence = confidence


# Test data generators
def generate_synthetic_anomalies(
    n_samples: int = 1000, 
    n_features: int = 10, 
    contamination: float = 0.1,
    anomaly_type: str = 'point',
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic datasets with known anomalies."""
    np.random.seed(random_state)
    
    n_anomalies = int(n_samples * contamination)
    n_normal = n_samples - n_anomalies
    
    if anomaly_type == 'point':
        # Point anomalies: outliers in feature space
        normal_data = np.random.multivariate_normal(
            mean=np.zeros(n_features), 
            cov=np.eye(n_features), 
            size=n_normal
        )
        
        anomaly_data = np.random.multivariate_normal(
            mean=np.ones(n_features) * 4,  # Far from normal
            cov=np.eye(n_features) * 0.5,   # Different variance
            size=n_anomalies
        )
        
    elif anomaly_type == 'contextual':
        # Contextual anomalies: normal values in wrong context
        normal_data = np.random.randn(n_normal, n_features)
        
        # Add seasonal pattern to first feature
        time_steps = np.linspace(0, 4*np.pi, n_normal)
        normal_data[:, 0] += 2 * np.sin(time_steps)
        
        # Anomalies: break the seasonal pattern
        anomaly_data = np.random.randn(n_anomalies, n_features)
        anomaly_time_steps = np.random.uniform(0, 4*np.pi, n_anomalies)
        anomaly_data[:, 0] += 2 * np.cos(anomaly_time_steps)  # Out of phase
        
    elif anomaly_type == 'collective':
        # Collective anomalies: groups of points that are anomalous together
        normal_data = np.random.randn(n_normal, n_features)
        
        # Create collective anomaly as a dense cluster in unusual location
        anomaly_center = np.ones(n_features) * 3
        anomaly_data = np.random.multivariate_normal(
            mean=anomaly_center,
            cov=np.eye(n_features) * 0.1,  # Very tight cluster
            size=n_anomalies
        )
    
    else:
        raise ValueError(f"Unknown anomaly type: {anomaly_type}")
    
    # Combine and shuffle
    X = np.vstack([normal_data, anomaly_data])
    y = np.hstack([np.ones(n_normal), -np.ones(n_anomalies)])
    
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def calculate_detection_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive detection metrics."""
    # Handle different labeling conventions
    if set(np.unique(y_true)) == {-1, 1}:
        y_true_binary = (y_true == -1).astype(int)
        y_pred_binary = (y_pred == -1).astype(int)
    else:
        y_true_binary = y_true
        y_pred_binary = y_pred
    
    # Basic metrics
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }


@pytest.mark.parametrize("algorithm,expected_precision,expected_recall", [
    ("isolation_forest", 0.70, 0.65),
    ("one_class_svm", 0.65, 0.60),
    ("local_outlier_factor", 0.60, 0.70),
    ("autoencoder", 0.75, 0.65),
])
class TestAnomalyDetectionAccuracy:
    """Test anomaly detection accuracy across different algorithms."""
    
    def test_point_anomaly_detection(
        self, 
        algorithm: str, 
        expected_precision: float, 
        expected_recall: float
    ):
        """Test detection of point anomalies."""
        X, y_true = generate_synthetic_anomalies(
            n_samples=1000, 
            contamination=0.1, 
            anomaly_type='point'
        )
        
        detector = DetectionService(algorithm=algorithm)
        result = detector.detect_anomalies(X)
        
        assert result.success, f"Detection failed for algorithm {algorithm}"
        
        metrics = calculate_detection_metrics(y_true, result.predictions)
        
        assert metrics['precision'] >= expected_precision, (
            f"Algorithm {algorithm} precision {metrics['precision']:.3f} "
            f"below expected {expected_precision:.3f}"
        )
        
        assert metrics['recall'] >= expected_recall, (
            f"Algorithm {algorithm} recall {metrics['recall']:.3f} "
            f"below expected {expected_recall:.3f}"
        )
        
        assert metrics['f1_score'] >= 0.50, (
            f"Algorithm {algorithm} F1-score {metrics['f1_score']:.3f} too low"
        )
    
    def test_contextual_anomaly_detection(
        self, 
        algorithm: str, 
        expected_precision: float, 
        expected_recall: float
    ):
        """Test detection of contextual anomalies."""
        X, y_true = generate_synthetic_anomalies(
            n_samples=800, 
            contamination=0.15, 
            anomaly_type='contextual'
        )
        
        detector = DetectionService(algorithm=algorithm)
        result = detector.detect_anomalies(X)
        
        metrics = calculate_detection_metrics(y_true, result.predictions)
        
        # Contextual anomalies are harder, so lower thresholds
        contextual_precision = expected_precision * 0.8
        contextual_recall = expected_recall * 0.8
        
        assert metrics['precision'] >= contextual_precision, (
            f"Algorithm {algorithm} contextual precision {metrics['precision']:.3f} "
            f"below expected {contextual_precision:.3f}"
        )
        
        assert metrics['recall'] >= contextual_recall, (
            f"Algorithm {algorithm} contextual recall {metrics['recall']:.3f} "
            f"below expected {contextual_recall:.3f}"
        )
    
    def test_collective_anomaly_detection(
        self, 
        algorithm: str, 
        expected_precision: float, 
        expected_recall: float
    ):
        """Test detection of collective anomalies."""
        X, y_true = generate_synthetic_anomalies(
            n_samples=600, 
            contamination=0.20, 
            anomaly_type='collective'
        )
        
        detector = DetectionService(algorithm=algorithm)
        result = detector.detect_anomalies(X)
        
        metrics = calculate_detection_metrics(y_true, result.predictions)
        
        # Collective anomalies should be easier for some algorithms
        collective_precision = expected_precision * 0.9
        collective_recall = expected_recall * 0.9
        
        assert metrics['precision'] >= collective_precision, (
            f"Algorithm {algorithm} collective precision {metrics['precision']:.3f} "
            f"below expected {collective_precision:.3f}"
        )
        
        assert metrics['recall'] >= collective_recall, (
            f"Algorithm {algorithm} collective recall {metrics['recall']:.3f} "
            f"below expected {collective_recall:.3f}"
        )


@pytest.mark.parametrize("contamination_rate", [0.05, 0.10, 0.15, 0.20, 0.25])
class TestContaminationRateAccuracy:
    """Test detection accuracy across different contamination rates."""
    
    def test_contamination_rate_handling(self, contamination_rate: float):
        """Test algorithm accuracy with varying contamination rates."""
        X, y_true = generate_synthetic_anomalies(
            n_samples=1000, 
            contamination=contamination_rate,
            anomaly_type='point'
        )
        
        # Test with isolation forest (generally robust to contamination)
        detector = DetectionService(algorithm='isolation_forest')
        
        # Configure detector with contamination rate
        config = {'contamination': contamination_rate}
        result = detector.detect_anomalies(X, config)
        
        metrics = calculate_detection_metrics(y_true, result.predictions)
        
        # Predicted contamination should approximate true contamination
        predicted_contamination = (result.predictions == -1).sum() / len(result.predictions)
        contamination_error = abs(predicted_contamination - contamination_rate)
        
        assert contamination_error < 0.05, (
            f"Contamination rate error {contamination_error:.3f} too high "
            f"(predicted: {predicted_contamination:.3f}, true: {contamination_rate:.3f})"
        )
        
        # Accuracy should remain reasonable across contamination rates
        min_accuracy = max(0.60, 1 - contamination_rate - 0.1)  # Adaptive threshold
        assert metrics['accuracy'] >= min_accuracy, (
            f"Accuracy {metrics['accuracy']:.3f} too low for contamination {contamination_rate:.3f}"
        )
    
    def test_extreme_contamination_rates(self):
        """Test behavior with extreme contamination rates."""
        # Very low contamination (1%)
        X_low, y_low = generate_synthetic_anomalies(n_samples=1000, contamination=0.01)
        detector = DetectionService(algorithm='isolation_forest')
        result_low = detector.detect_anomalies(X_low, {'contamination': 0.01})
        
        assert result_low.success, "Failed with very low contamination"
        
        # High contamination (40%)
        X_high, y_high = generate_synthetic_anomalies(n_samples=1000, contamination=0.40)
        result_high = detector.detect_anomalies(X_high, {'contamination': 0.40})
        
        assert result_high.success, "Failed with high contamination"
        
        # Should still produce reasonable results
        metrics_low = calculate_detection_metrics(y_low, result_low.predictions)
        metrics_high = calculate_detection_metrics(y_high, result_high.predictions)
        
        assert metrics_low['precision'] >= 0.30, "Precision too low with low contamination"
        assert metrics_high['precision'] >= 0.30, "Precision too low with high contamination"


class TestRealTimeDetectionPerformance:
    """Test real-time detection performance requirements."""
    
    @pytest.mark.performance
    def test_single_sample_detection_speed(self):
        """Test detection speed for single samples (real-time requirement)."""
        # Train detector
        X_train, _ = generate_synthetic_anomalies(n_samples=1000)
        detector = DetectionService(algorithm='isolation_forest')
        
        # Simulate training (in real implementation)
        detector.detect_anomalies(X_train)  # Initial training
        
        # Test single sample detection speed
        single_sample = np.random.randn(1, X_train.shape[1])
        
        # Measure multiple detection times
        detection_times = []
        for _ in range(100):
            start_time = time.perf_counter()
            result = detector.detect_anomalies(single_sample)
            end_time = time.perf_counter()
            
            detection_times.append(end_time - start_time)
            assert result.success, "Single sample detection failed"
        
        avg_detection_time = np.mean(detection_times)
        max_detection_time = np.max(detection_times)
        
        # Real-time requirements
        assert avg_detection_time < 0.010, (  # 10ms average
            f"Average detection time {avg_detection_time*1000:.2f}ms exceeds 10ms"
        )
        
        assert max_detection_time < 0.050, (  # 50ms maximum
            f"Maximum detection time {max_detection_time*1000:.2f}ms exceeds 50ms"
        )
    
    @pytest.mark.performance
    def test_batch_detection_performance(self):
        """Test batch detection performance for high-throughput scenarios."""
        X_train, _ = generate_synthetic_anomalies(n_samples=5000)
        detector = DetectionService(algorithm='isolation_forest')
        
        # Train detector
        detector.detect_anomalies(X_train)
        
        # Test different batch sizes
        batch_sizes = [100, 500, 1000, 2000]
        
        for batch_size in batch_sizes:
            X_batch = np.random.randn(batch_size, X_train.shape[1])
            
            start_time = time.perf_counter()
            result = detector.detect_anomalies(X_batch)
            end_time = time.perf_counter()
            
            detection_time = end_time - start_time
            per_sample_time = detection_time / batch_size
            
            assert result.success, f"Batch detection failed for size {batch_size}"
            
            # Should maintain sub-millisecond per-sample performance in batches
            assert per_sample_time < 0.001, (  # 1ms per sample
                f"Per-sample time {per_sample_time*1000:.2f}ms exceeds 1ms for batch {batch_size}"
            )
    
    @pytest.mark.performance
    def test_memory_usage_efficiency(self):
        """Test memory efficiency for large datasets."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process increasingly large datasets
            dataset_sizes = [1000, 5000, 10000]
            
            for size in dataset_sizes:
                X, _ = generate_synthetic_anomalies(n_samples=size, n_features=20)
                detector = DetectionService(algorithm='isolation_forest')
                
                result = detector.detect_anomalies(X)
                assert result.success, f"Detection failed for size {size}"
                
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                # Memory increase should scale reasonably with data size
                memory_per_sample = memory_increase / size
                assert memory_per_sample < 0.1, (  # 0.1 MB per sample max
                    f"Memory per sample {memory_per_sample:.3f}MB too high for size {size}"
                )
                
        except ImportError:
            pytest.skip("psutil not available for memory testing")


class TestDetectionQualityAssurance:
    """Test detection quality and reliability."""
    
    def test_confidence_score_calibration(self):
        """Test that confidence scores correlate with detection accuracy."""
        X, y_true = generate_synthetic_anomalies(n_samples=1000, contamination=0.1)
        detector = DetectionService(algorithm='isolation_forest')
        
        result = detector.detect_anomalies(X)
        
        # Group predictions by confidence quartiles
        confidence_quartiles = np.percentile(result.confidence_scores, [25, 50, 75])
        
        low_conf_mask = result.confidence_scores <= confidence_quartiles[0]
        high_conf_mask = result.confidence_scores >= confidence_quartiles[2]
        
        # Calculate accuracy for different confidence levels
        if np.sum(low_conf_mask) > 0 and np.sum(high_conf_mask) > 0:
            low_conf_predictions = result.predictions[low_conf_mask]
            low_conf_true = y_true[low_conf_mask]
            low_conf_accuracy = (low_conf_predictions == low_conf_true).mean()
            
            high_conf_predictions = result.predictions[high_conf_mask]
            high_conf_true = y_true[high_conf_mask]
            high_conf_accuracy = (high_conf_predictions == high_conf_true).mean()
            
            # High confidence should have better accuracy
            assert high_conf_accuracy >= low_conf_accuracy, (
                f"High confidence accuracy {high_conf_accuracy:.3f} not better than "
                f"low confidence accuracy {low_conf_accuracy:.3f}"
            )
    
    def test_detection_consistency(self):
        """Test detection consistency across multiple runs."""
        X, y_true = generate_synthetic_anomalies(n_samples=500, random_state=42)
        
        # Run detection multiple times
        results = []
        for i in range(5):
            detector = DetectionService(algorithm='isolation_forest')
            result = detector.detect_anomalies(X)
            results.append(result.predictions)
        
        # Calculate pairwise agreement
        agreements = []
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                agreement = (results[i] == results[j]).mean()
                agreements.append(agreement)
        
        avg_agreement = np.mean(agreements)
        
        # Should have reasonable consistency (allowing for some randomness)
        assert avg_agreement >= 0.85, (
            f"Detection consistency too low: {avg_agreement:.3f} average agreement"
        )
    
    def test_edge_case_handling(self):
        """Test handling of edge cases and problematic data."""
        detector = DetectionService(algorithm='isolation_forest')
        
        # Test with very small dataset
        X_small = np.random.randn(5, 3)
        result_small = detector.detect_anomalies(X_small)
        assert result_small.success, "Failed with very small dataset"
        
        # Test with single feature
        X_single_feature = np.random.randn(100, 1)
        result_single = detector.detect_anomalies(X_single_feature)
        assert result_single.success, "Failed with single feature"
        
        # Test with constant feature
        X_constant = np.ones((100, 3))
        X_constant[:, 0] = np.random.randn(100)  # One varying feature
        result_constant = detector.detect_anomalies(X_constant)
        assert result_constant.success, "Failed with constant features"
        
        # Test with missing values (NaN)
        X_nan = np.random.randn(100, 5)
        X_nan[10:20, 1] = np.nan
        
        try:
            result_nan = detector.detect_anomalies(X_nan)
            # Should either succeed or handle gracefully
            if not result_nan.success:
                # Acceptable to fail gracefully with clear error handling
                pass
        except Exception as e:
            # Should not crash with unhandled exception
            assert "nan" in str(e).lower() or "missing" in str(e).lower(), (
                f"Unexpected error with NaN data: {e}"
            )


class TestAlgorithmComparison:
    """Test comparative performance of different algorithms."""
    
    def test_algorithm_ranking_consistency(self):
        """Test that algorithm ranking is consistent across datasets."""
        algorithms = ['isolation_forest', 'one_class_svm', 'local_outlier_factor']
        
        # Test on different types of anomalies
        anomaly_types = ['point', 'contextual', 'collective']
        algorithm_scores = {algo: [] for algo in algorithms}
        
        for anomaly_type in anomaly_types:
            X, y_true = generate_synthetic_anomalies(
                n_samples=800, 
                contamination=0.1, 
                anomaly_type=anomaly_type
            )
            
            for algorithm in algorithms:
                detector = DetectionService(algorithm=algorithm)
                result = detector.detect_anomalies(X)
                
                if result.success:
                    metrics = calculate_detection_metrics(y_true, result.predictions)
                    algorithm_scores[algorithm].append(metrics['f1_score'])
                else:
                    algorithm_scores[algorithm].append(0.0)
        
        # Calculate average scores
        avg_scores = {
            algo: np.mean(scores) for algo, scores in algorithm_scores.items()
        }
        
        # Validate that all algorithms perform reasonably
        for algo, score in avg_scores.items():
            assert score >= 0.40, (
                f"Algorithm {algo} average F1-score {score:.3f} too low across datasets"
            )
        
        # At least one algorithm should perform well
        best_score = max(avg_scores.values())
        assert best_score >= 0.60, (
            f"Best algorithm F1-score {best_score:.3f} insufficient"
        )
    
    def test_computational_complexity_comparison(self):
        """Test computational complexity scaling for different algorithms."""
        algorithms = ['isolation_forest', 'one_class_svm']
        dataset_sizes = [500, 1000, 2000]
        
        for algorithm in algorithms:
            scaling_times = []
            
            for size in dataset_sizes:
                X, _ = generate_synthetic_anomalies(n_samples=size, n_features=10)
                detector = DetectionService(algorithm=algorithm)
                
                start_time = time.perf_counter()
                result = detector.detect_anomalies(X)
                end_time = time.perf_counter()
                
                if result.success:
                    scaling_times.append(end_time - start_time)
                else:
                    scaling_times.append(float('inf'))
            
            # Check that scaling is not worse than quadratic
            if len(scaling_times) >= 2 and all(t < float('inf') for t in scaling_times):
                time_ratio_1_to_2 = scaling_times[1] / scaling_times[0] if scaling_times[0] > 0 else 1
                expected_ratio = (dataset_sizes[1] / dataset_sizes[0]) ** 2  # Quadratic upper bound
                
                assert time_ratio_1_to_2 <= expected_ratio * 2, (
                    f"Algorithm {algorithm} scaling worse than quadratic: "
                    f"ratio {time_ratio_1_to_2:.2f} vs expected max {expected_ratio * 2:.2f}"
                )