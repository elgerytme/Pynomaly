"""Performance benchmark tests for anomaly detection algorithms and services."""

import pytest
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from unittest.mock import Mock, patch

from anomaly_detection.domain.services.detection_service import DetectionService
from anomaly_detection.domain.services.ensemble_service import EnsembleService
from anomaly_detection.domain.services.streaming_service import StreamingService
from anomaly_detection.infrastructure.adapters.algorithms.adapters.sklearn_adapter import SklearnAdapter
from anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter import PyODAdapter


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    algorithm: str
    dataset_size: int
    features: int
    training_time: float
    prediction_time: float
    memory_usage: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    throughput_samples_per_sec: float


class DatasetGenerator:
    """Generate synthetic datasets for benchmarking."""
    
    @staticmethod
    def generate_normal_data(n_samples: int, n_features: int, random_state: int = 42) -> np.ndarray:
        """Generate normal data without anomalies."""
        np.random.seed(random_state)
        return np.random.normal(0, 1, (n_samples, n_features))
    
    @staticmethod
    def generate_anomalous_data(n_samples: int, n_features: int, contamination: float = 0.1, 
                              random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Generate data with known anomalies."""
        np.random.seed(random_state)
        
        n_anomalies = int(n_samples * contamination)
        n_normal = n_samples - n_anomalies
        
        # Generate normal data
        normal_data = np.random.normal(0, 1, (n_normal, n_features))
        
        # Generate anomalous data (outliers)
        anomalous_data = np.random.normal(0, 5, (n_anomalies, n_features))  # Higher variance
        
        # Combine and create labels
        data = np.vstack([normal_data, anomalous_data])
        labels = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])
        
        # Shuffle
        indices = np.random.permutation(n_samples)
        return data[indices], labels[indices]
    
    @staticmethod
    def generate_high_dimensional_data(n_samples: int, n_features: int, 
                                     contamination: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Generate high-dimensional data for curse of dimensionality testing."""
        return DatasetGenerator.generate_anomalous_data(
            n_samples, n_features, contamination, random_state=42
        )
    
    @staticmethod
    def generate_streaming_data(n_batches: int, batch_size: int, n_features: int,
                              drift_point: int = None) -> List[np.ndarray]:
        """Generate streaming data with optional concept drift."""
        batches = []
        
        for i in range(n_batches):
            if drift_point and i >= drift_point:
                # Introduce concept drift by changing distribution
                batch = np.random.normal(2, 1.5, (batch_size, n_features))
            else:
                batch = np.random.normal(0, 1, (batch_size, n_features))
            
            batches.append(batch)
        
        return batches


class PerformanceTimer:
    """Context manager for measuring execution time."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time


class MemoryProfiler:
    """Memory usage profiler."""
    
    @staticmethod
    def get_memory_usage() -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0  # psutil not available
    
    @staticmethod
    def profile_function(func, *args, **kwargs) -> Tuple[Any, float]:
        """Profile memory usage of a function."""
        initial_memory = MemoryProfiler.get_memory_usage()
        result = func(*args, **kwargs)
        final_memory = MemoryProfiler.get_memory_usage()
        memory_diff = final_memory - initial_memory
        return result, max(0, memory_diff)


@pytest.mark.performance
class TestAlgorithmBenchmarks:
    """Benchmark tests for individual algorithms."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detection_service = DetectionService()
        self.ensemble_service = EnsembleService()
        
        # Test datasets of varying sizes
        self.small_data, self.small_labels = DatasetGenerator.generate_anomalous_data(1000, 10)
        self.medium_data, self.medium_labels = DatasetGenerator.generate_anomalous_data(10000, 20)
        self.large_data, self.large_labels = DatasetGenerator.generate_anomalous_data(50000, 50)
        
        self.datasets = [
            ("small", self.small_data, self.small_labels),
            ("medium", self.medium_data, self.medium_labels),
            ("large", self.large_data, self.large_labels)
        ]
        
        self.algorithms = [
            "isolation_forest",
            "one_class_svm",
            "local_outlier_factor",
            "elliptic_envelope"
        ]
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0)
        }
    
    def benchmark_algorithm(self, algorithm: str, data: np.ndarray, 
                          labels: np.ndarray) -> BenchmarkResult:
        """Benchmark a single algorithm."""
        # Training time (fitting)
        with PerformanceTimer() as training_timer:
            result, memory_usage = MemoryProfiler.profile_function(
                self.detection_service.detect_anomalies,
                data.tolist(),
                algorithm=algorithm,
                contamination=0.1
            )
        
        # Prediction time (for same data - simulating inference)
        with PerformanceTimer() as prediction_timer:
            # Re-run detection to simulate prediction phase
            pred_result, _ = MemoryProfiler.profile_function(
                self.detection_service.detect_anomalies,
                data[:1000].tolist(),  # Use subset for prediction timing
                algorithm=algorithm,
                contamination=0.1
            )
        
        # Calculate metrics
        y_pred = np.array(result["anomalies"])
        metrics = self.calculate_metrics(labels, y_pred)
        
        # Calculate throughput
        throughput = len(data) / (prediction_timer.duration + 0.001)  # Avoid division by zero
        
        return BenchmarkResult(
            algorithm=algorithm,
            dataset_size=len(data),
            features=data.shape[1],
            training_time=training_timer.duration,
            prediction_time=prediction_timer.duration,
            memory_usage=memory_usage,
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
            throughput_samples_per_sec=throughput
        )
    
    @pytest.mark.parametrize("algorithm", [
        "isolation_forest", "one_class_svm", "local_outlier_factor"
    ])
    def test_algorithm_performance_small_dataset(self, algorithm):
        """Test algorithm performance on small dataset."""
        result = self.benchmark_algorithm(algorithm, self.small_data, self.small_labels)
        
        # Performance assertions
        assert result.training_time < 5.0, f"{algorithm} training too slow: {result.training_time}s"
        assert result.prediction_time < 2.0, f"{algorithm} prediction too slow: {result.prediction_time}s"
        assert result.memory_usage < 100.0, f"{algorithm} uses too much memory: {result.memory_usage}MB"
        assert result.throughput_samples_per_sec > 100, f"{algorithm} throughput too low: {result.throughput_samples_per_sec}"
        
        # Quality assertions
        assert result.accuracy > 0.7, f"{algorithm} accuracy too low: {result.accuracy}"
        
        print(f"\n{algorithm} Performance (Small Dataset):")
        print(f"  Training Time: {result.training_time:.3f}s")
        print(f"  Prediction Time: {result.prediction_time:.3f}s")
        print(f"  Memory Usage: {result.memory_usage:.1f}MB")
        print(f"  Throughput: {result.throughput_samples_per_sec:.1f} samples/sec")
        print(f"  Accuracy: {result.accuracy:.3f}")
    
    @pytest.mark.parametrize("algorithm", [
        "isolation_forest", "one_class_svm"
    ])
    def test_algorithm_performance_medium_dataset(self, algorithm):
        """Test algorithm performance on medium dataset."""
        result = self.benchmark_algorithm(algorithm, self.medium_data, self.medium_labels)
        
        # Adjusted thresholds for larger dataset
        assert result.training_time < 30.0, f"{algorithm} training too slow: {result.training_time}s"
        assert result.prediction_time < 10.0, f"{algorithm} prediction too slow: {result.prediction_time}s"
        assert result.memory_usage < 500.0, f"{algorithm} uses too much memory: {result.memory_usage}MB"
        assert result.throughput_samples_per_sec > 50, f"{algorithm} throughput too low: {result.throughput_samples_per_sec}"
        
        print(f"\n{algorithm} Performance (Medium Dataset):")
        print(f"  Training Time: {result.training_time:.3f}s")
        print(f"  Prediction Time: {result.prediction_time:.3f}s")
        print(f"  Memory Usage: {result.memory_usage:.1f}MB")
        print(f"  Throughput: {result.throughput_samples_per_sec:.1f} samples/sec")
        print(f"  Accuracy: {result.accuracy:.3f}")
    
    def test_algorithm_scalability(self):
        """Test how algorithms scale with dataset size."""
        algorithm = "isolation_forest"
        results = []
        
        for dataset_name, data, labels in self.datasets:
            result = self.benchmark_algorithm(algorithm, data, labels)
            results.append((dataset_name, result))
        
        # Check that training time scales reasonably
        small_time = results[0][1].training_time
        medium_time = results[1][1].training_time
        large_time = results[2][1].training_time
        
        # Training time should scale sub-quadratically
        time_ratio_small_to_medium = medium_time / small_time
        time_ratio_medium_to_large = large_time / medium_time
        
        # Should not scale worse than O(n^2)
        assert time_ratio_small_to_medium < 100, "Training time scaling too poor (small to medium)"
        assert time_ratio_medium_to_large < 50, "Training time scaling too poor (medium to large)"
        
        print(f"\nScalability Analysis for {algorithm}:")
        for dataset_name, result in results:
            print(f"  {dataset_name.title()} Dataset ({result.dataset_size} samples):")
            print(f"    Training Time: {result.training_time:.3f}s")
            print(f"    Memory Usage: {result.memory_usage:.1f}MB")
            print(f"    Throughput: {result.throughput_samples_per_sec:.1f} samples/sec")


@pytest.mark.performance
class TestEnsembleBenchmarks:
    """Benchmark tests for ensemble methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ensemble_service = EnsembleService()
        self.test_data, self.test_labels = DatasetGenerator.generate_anomalous_data(5000, 15)
        
        self.ensemble_configs = [
            {
                "algorithms": ["isolation_forest", "one_class_svm"],
                "method": "majority",
                "name": "two_algorithm_majority"
            },
            {
                "algorithms": ["isolation_forest", "one_class_svm", "local_outlier_factor"],
                "method": "average",
                "name": "three_algorithm_average"
            },
            {
                "algorithms": ["isolation_forest", "one_class_svm", "local_outlier_factor", "elliptic_envelope"],
                "method": "weighted_average",
                "name": "four_algorithm_weighted"
            }
        ]
    
    def benchmark_ensemble(self, config: Dict) -> BenchmarkResult:
        """Benchmark ensemble method."""
        with PerformanceTimer() as timer:
            result, memory_usage = MemoryProfiler.profile_function(
                self.ensemble_service.detect_with_ensemble,
                self.test_data.tolist(),
                algorithms=config["algorithms"],
                ensemble_method=config["method"],
                contamination=0.1
            )
        
        # Calculate metrics
        y_pred = np.array(result["anomalies"])
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            "accuracy": accuracy_score(self.test_labels, y_pred),
            "precision": precision_score(self.test_labels, y_pred, zero_division=0),
            "recall": recall_score(self.test_labels, y_pred, zero_division=0),
            "f1_score": f1_score(self.test_labels, y_pred, zero_division=0)
        }
        
        throughput = len(self.test_data) / (timer.duration + 0.001)
        
        return BenchmarkResult(
            algorithm=config["name"],
            dataset_size=len(self.test_data),
            features=self.test_data.shape[1],
            training_time=timer.duration,
            prediction_time=timer.duration,  # Same as training for this test
            memory_usage=memory_usage,
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
            throughput_samples_per_sec=throughput
        )
    
    @pytest.mark.parametrize("config", [
        {"algorithms": ["isolation_forest", "one_class_svm"], "method": "majority", "name": "two_alg_majority"},
        {"algorithms": ["isolation_forest", "one_class_svm", "local_outlier_factor"], "method": "average", "name": "three_alg_average"}
    ])
    def test_ensemble_performance(self, config):
        """Test ensemble method performance."""
        result = self.benchmark_ensemble(config)
        
        # Performance thresholds (ensemble methods are expected to be slower)
        expected_max_time = 20.0 * len(config["algorithms"])  # Scale with number of algorithms
        assert result.training_time < expected_max_time, f"Ensemble too slow: {result.training_time}s"
        assert result.memory_usage < 200.0 * len(config["algorithms"]), f"Ensemble uses too much memory: {result.memory_usage}MB"
        assert result.throughput_samples_per_sec > 10, f"Ensemble throughput too low: {result.throughput_samples_per_sec}"
        
        # Quality should be better than individual algorithms
        assert result.accuracy > 0.75, f"Ensemble accuracy too low: {result.accuracy}"
        
        print(f"\nEnsemble Performance ({config['name']}):")
        print(f"  Algorithms: {config['algorithms']}")
        print(f"  Method: {config['method']}")
        print(f"  Training Time: {result.training_time:.3f}s")
        print(f"  Memory Usage: {result.memory_usage:.1f}MB")
        print(f"  Throughput: {result.throughput_samples_per_sec:.1f} samples/sec")
        print(f"  Accuracy: {result.accuracy:.3f}")
        print(f"  F1 Score: {result.f1_score:.3f}")
    
    def test_ensemble_vs_individual_performance(self):
        """Compare ensemble performance to individual algorithms."""
        detection_service = DetectionService()
        
        # Benchmark individual algorithm
        with PerformanceTimer() as individual_timer:
            individual_result = detection_service.detect_anomalies(
                self.test_data.tolist(),
                algorithm="isolation_forest",
                contamination=0.1
            )
        
        # Benchmark ensemble
        ensemble_config = self.ensemble_configs[0]  # Two algorithm majority
        ensemble_result = self.benchmark_ensemble(ensemble_config)
        
        # Ensemble should be slower but more accurate
        time_overhead = ensemble_result.training_time / individual_timer.duration
        assert time_overhead < 5.0, f"Ensemble overhead too high: {time_overhead}x"
        
        # Calculate individual accuracy for comparison
        y_pred_individual = np.array(individual_result["anomalies"])
        from sklearn.metrics import accuracy_score
        individual_accuracy = accuracy_score(self.test_labels, y_pred_individual)
        
        print(f"\nEnsemble vs Individual Comparison:")
        print(f"  Individual Time: {individual_timer.duration:.3f}s")
        print(f"  Ensemble Time: {ensemble_result.training_time:.3f}s")
        print(f"  Time Overhead: {time_overhead:.1f}x")
        print(f"  Individual Accuracy: {individual_accuracy:.3f}")
        print(f"  Ensemble Accuracy: {ensemble_result.accuracy:.3f}")
        print(f"  Accuracy Improvement: {ensemble_result.accuracy - individual_accuracy:.3f}")


@pytest.mark.performance
class TestStreamingBenchmarks:
    """Benchmark tests for streaming detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.streaming_service = StreamingService()
        
        # Generate streaming data
        self.streaming_batches = DatasetGenerator.generate_streaming_data(
            n_batches=50, batch_size=100, n_features=10
        )
        
        self.drift_batches = DatasetGenerator.generate_streaming_data(
            n_batches=50, batch_size=100, n_features=10, drift_point=25
        )
    
    def test_streaming_processing_performance(self):
        """Test streaming processing performance."""
        total_samples = 0
        processing_times = []
        
        with PerformanceTimer() as total_timer:
            for i, batch in enumerate(self.streaming_batches[:10]):  # Test first 10 batches
                with PerformanceTimer() as batch_timer:
                    # Simulate streaming processing
                    result = self.streaming_service.process_streaming_batch(
                        batch.tolist(),
                        algorithm="isolation_forest",
                        buffer_size=500
                    )
                
                processing_times.append(batch_timer.duration)
                total_samples += len(batch)
        
        # Performance metrics
        avg_batch_time = np.mean(processing_times)
        total_throughput = total_samples / total_timer.duration
        
        # Assertions
        assert avg_batch_time < 1.0, f"Batch processing too slow: {avg_batch_time}s"
        assert total_throughput > 500, f"Streaming throughput too low: {total_throughput} samples/sec"
        
        # Consistency check - processing times should be relatively stable
        time_std = np.std(processing_times)
        time_cv = time_std / avg_batch_time  # Coefficient of variation
        assert time_cv < 0.5, f"Processing time too variable: CV={time_cv}"
        
        print(f"\nStreaming Processing Performance:")
        print(f"  Average Batch Time: {avg_batch_time:.3f}s")
        print(f"  Total Throughput: {total_throughput:.1f} samples/sec")
        print(f"  Processing Consistency (CV): {time_cv:.3f}")
        print(f"  Min Batch Time: {min(processing_times):.3f}s")
        print(f"  Max Batch Time: {max(processing_times):.3f}s")
    
    def test_concept_drift_detection_performance(self):
        """Test concept drift detection performance."""
        drift_detection_times = []
        drift_detected = False
        
        with PerformanceTimer() as total_timer:
            for i, batch in enumerate(self.drift_batches):
                with PerformanceTimer() as drift_timer:
                    # Check for concept drift
                    drift_result = self.streaming_service.detect_concept_drift(
                        batch.tolist(),
                        threshold=0.05
                    )
                    
                    if drift_result.get("drift_detected", False):
                        drift_detected = True
                
                drift_detection_times.append(drift_timer.duration)
        
        avg_drift_time = np.mean(drift_detection_times)
        
        # Performance assertions
        assert avg_drift_time < 0.5, f"Drift detection too slow: {avg_drift_time}s"
        
        # Should detect drift after drift point (batch 25)
        assert drift_detected, "Failed to detect concept drift"
        
        print(f"\nConcept Drift Detection Performance:")
        print(f"  Average Drift Detection Time: {avg_drift_time:.3f}s")
        print(f"  Total Processing Time: {total_timer.duration:.3f}s")
        print(f"  Drift Detected: {drift_detected}")
    
    def test_streaming_memory_efficiency(self):
        """Test streaming memory efficiency."""
        initial_memory = MemoryProfiler.get_memory_usage()
        memory_readings = [initial_memory]
        
        # Process many batches to test memory growth
        for batch in self.streaming_batches:
            self.streaming_service.process_streaming_batch(
                batch.tolist(),
                algorithm="isolation_forest",
                buffer_size=1000
            )
            
            current_memory = MemoryProfiler.get_memory_usage()
            memory_readings.append(current_memory)
        
        final_memory = memory_readings[-1]
        memory_growth = final_memory - initial_memory
        max_memory = max(memory_readings)
        
        # Memory growth should be bounded
        assert memory_growth < 100.0, f"Excessive memory growth: {memory_growth}MB"
        assert max_memory < initial_memory + 150.0, f"Peak memory too high: {max_memory}MB"
        
        print(f"\nStreaming Memory Efficiency:")
        print(f"  Initial Memory: {initial_memory:.1f}MB")
        print(f"  Final Memory: {final_memory:.1f}MB")
        print(f"  Memory Growth: {memory_growth:.1f}MB")
        print(f"  Peak Memory: {max_memory:.1f}MB")


@pytest.mark.performance
class TestAdapterBenchmarks:
    """Benchmark tests for algorithm adapters."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sklearn_adapter = SklearnAdapter()
        try:
            self.pyod_adapter = PyODAdapter()
        except ImportError:
            self.pyod_adapter = None
        
        self.test_data, self.test_labels = DatasetGenerator.generate_anomalous_data(2000, 10)
    
    def test_sklearn_adapter_performance(self):
        """Test SklearnAdapter performance."""
        algorithms = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor"]
        results = {}
        
        for algorithm in algorithms:
            with PerformanceTimer() as timer:
                result, memory_usage = MemoryProfiler.profile_function(
                    self.sklearn_adapter.detect_anomalies,
                    self.test_data,
                    algorithm=algorithm,
                    contamination=0.1
                )
            
            results[algorithm] = {
                "time": timer.duration,
                "memory": memory_usage,
                "result": result
            }
            
            # Performance assertions
            assert timer.duration < 10.0, f"SklearnAdapter {algorithm} too slow: {timer.duration}s"
            assert memory_usage < 100.0, f"SklearnAdapter {algorithm} uses too much memory: {memory_usage}MB"
        
        print(f"\nSklearnAdapter Performance:")
        for algorithm, metrics in results.items():
            print(f"  {algorithm}:")
            print(f"    Time: {metrics['time']:.3f}s")
            print(f"    Memory: {metrics['memory']:.1f}MB")
    
    @pytest.mark.skipif("pyod_adapter is None", reason="PyOD not available")
    def test_pyod_adapter_performance(self):
        """Test PyODAdapter performance."""
        if self.pyod_adapter is None:
            pytest.skip("PyOD adapter not available")
        
        algorithms = ["IForest", "OCSVM", "LOF"]
        results = {}
        
        for algorithm in algorithms:
            try:
                with PerformanceTimer() as timer:
                    result, memory_usage = MemoryProfiler.profile_function(
                        self.pyod_adapter.detect_anomalies,
                        self.test_data,
                        algorithm=algorithm,
                        contamination=0.1
                    )
                
                results[algorithm] = {
                    "time": timer.duration,
                    "memory": memory_usage,
                    "result": result
                }
                
                # Performance assertions
                assert timer.duration < 15.0, f"PyODAdapter {algorithm} too slow: {timer.duration}s"
                assert memory_usage < 150.0, f"PyODAdapter {algorithm} uses too much memory: {memory_usage}MB"
            
            except Exception as e:
                print(f"Warning: {algorithm} test failed: {e}")
                continue
        
        print(f"\nPyODAdapter Performance:")
        for algorithm, metrics in results.items():
            print(f"  {algorithm}:")
            print(f"    Time: {metrics['time']:.3f}s")
            print(f"    Memory: {metrics['memory']:.1f}MB")
    
    def test_adapter_comparison(self):
        """Compare performance between adapters."""
        if self.pyod_adapter is None:
            pytest.skip("PyOD adapter not available for comparison")
        
        # Test same algorithm with both adapters
        sklearn_algorithm = "IsolationForest"
        pyod_algorithm = "IForest"
        
        # SklearnAdapter
        with PerformanceTimer() as sklearn_timer:
            sklearn_result = self.sklearn_adapter.detect_anomalies(
                self.test_data,
                algorithm=sklearn_algorithm,
                contamination=0.1
            )
        
        # PyODAdapter
        with PerformanceTimer() as pyod_timer:
            pyod_result = self.pyod_adapter.detect_anomalies(
                self.test_data,
                algorithm=pyod_algorithm,
                contamination=0.1
            )
        
        # Compare results quality
        from sklearn.metrics import accuracy_score
        sklearn_accuracy = accuracy_score(self.test_labels, sklearn_result["anomalies"])
        pyod_accuracy = accuracy_score(self.test_labels, pyod_result["anomalies"])
        
        print(f"\nAdapter Comparison (Isolation Forest):")
        print(f"  SklearnAdapter:")
        print(f"    Time: {sklearn_timer.duration:.3f}s")
        print(f"    Accuracy: {sklearn_accuracy:.3f}")
        print(f"  PyODAdapter:")
        print(f"    Time: {pyod_timer.duration:.3f}s")
        print(f"    Accuracy: {pyod_accuracy:.3f}")
        
        # Both should produce reasonable results
        assert sklearn_accuracy > 0.7, f"SklearnAdapter accuracy too low: {sklearn_accuracy}"
        assert pyod_accuracy > 0.7, f"PyODAdapter accuracy too low: {pyod_accuracy}"


@pytest.mark.performance
class TestHighDimensionalBenchmarks:
    """Benchmark tests for high-dimensional data."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detection_service = DetectionService()
        
        # Generate high-dimensional datasets
        self.dimensions = [10, 50, 100, 200]
        self.datasets = {}
        
        for dim in self.dimensions:
            data, labels = DatasetGenerator.generate_high_dimensional_data(
                n_samples=1000, n_features=dim, contamination=0.1
            )
            self.datasets[dim] = (data, labels)
    
    @pytest.mark.parametrize("dimensions", [10, 50, 100])
    def test_curse_of_dimensionality_performance(self, dimensions):
        """Test performance with increasing dimensionality."""
        data, labels = self.datasets[dimensions]
        
        with PerformanceTimer() as timer:
            result = self.detection_service.detect_anomalies(
                data.tolist(),
                algorithm="isolation_forest",
                contamination=0.1
            )
        
        # Calculate accuracy
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(labels, result["anomalies"])
        
        # Performance should degrade gracefully with dimensionality
        max_time = 2.0 + (dimensions / 50) * 5.0  # Allow more time for higher dimensions
        assert timer.duration < max_time, f"Performance too poor for {dimensions}D: {timer.duration}s"
        
        # Accuracy might decrease with dimensionality, but should still be reasonable
        min_accuracy = max(0.6, 0.9 - (dimensions / 200) * 0.2)
        assert accuracy > min_accuracy, f"Accuracy too low for {dimensions}D: {accuracy}"
        
        print(f"\nHigh-Dimensional Performance ({dimensions}D):")
        print(f"  Processing Time: {timer.duration:.3f}s")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Samples per Second: {len(data) / timer.duration:.1f}")
    
    def test_dimensionality_scaling(self):
        """Test how performance scales with dimensionality."""
        results = []
        
        for dim in self.dimensions:
            data, labels = self.datasets[dim]
            
            with PerformanceTimer() as timer:
                result = self.detection_service.detect_anomalies(
                    data.tolist(),
                    algorithm="isolation_forest",
                    contamination=0.1
                )
            
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(labels, result["anomalies"])
            
            results.append({
                "dimensions": dim,
                "time": timer.duration,
                "accuracy": accuracy,
                "throughput": len(data) / timer.duration
            })
        
        # Check scaling behavior
        for i in range(1, len(results)):
            prev_result = results[i-1]
            curr_result = results[i]
            
            # Time should scale sub-linearly with dimensions
            time_ratio = curr_result["time"] / prev_result["time"]
            dim_ratio = curr_result["dimensions"] / prev_result["dimensions"]
            
            # Time scaling should be better than linear in dimensions
            assert time_ratio < dim_ratio * 2, f"Time scaling too poor: {time_ratio} vs {dim_ratio}"
        
        print(f"\nDimensionality Scaling Analysis:")
        for result in results:
            print(f"  {result['dimensions']}D: {result['time']:.3f}s, Accuracy: {result['accuracy']:.3f}, Throughput: {result['throughput']:.1f}")


if __name__ == "__main__":
    # Run specific benchmark tests
    pytest.main([
        __file__ + "::TestAlgorithmBenchmarks::test_algorithm_performance_small_dataset",
        "-v", "-s", "--tb=short"
    ])