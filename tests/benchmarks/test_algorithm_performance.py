"""Performance benchmarks for anomaly detection algorithms."""

import pytest
import time
import psutil
import numpy as np
from typing import List, Dict, Any
import pandas as pd

from pynomaly.domain.entities import Dataset
from pynomaly.domain.value_objects import ContaminationRate
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


class PerformanceMetrics:
    """Container for performance metrics."""
    
    def __init__(self):
        self.fit_time: float = 0.0
        self.predict_time: float = 0.0
        self.memory_usage_mb: float = 0.0
        self.precision: float = 0.0
        self.recall: float = 0.0
        self.f1_score: float = 0.0
        self.roc_auc: float = 0.0
        self.n_anomalies_detected: int = 0
        self.anomaly_rate: float = 0.0


class TestAlgorithmPerformance:
    """Performance benchmarks for anomaly detection algorithms."""
    
    def measure_memory_usage(self, func, *args, **kwargs):
        """Measure memory usage of a function."""
        process = psutil.Process()
        
        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Get peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return result, peak_memory - baseline_memory
    
    def calculate_detection_metrics(self, true_labels: np.ndarray, 
                                  predicted_labels: np.ndarray,
                                  anomaly_scores: np.ndarray) -> Dict[str, float]:
        """Calculate detection performance metrics."""
        metrics = {}
        
        # Basic classification metrics
        if len(np.unique(predicted_labels)) > 1:
            metrics['precision'] = precision_score(true_labels, predicted_labels)
            metrics['recall'] = recall_score(true_labels, predicted_labels)
            metrics['f1_score'] = f1_score(true_labels, predicted_labels)
        else:
            # Handle case where only one class is predicted
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
            metrics['f1_score'] = 0.0
        
        # ROC AUC (requires probability scores)
        if len(np.unique(true_labels)) > 1:
            try:
                metrics['roc_auc'] = roc_auc_score(true_labels, anomaly_scores)
            except ValueError:
                metrics['roc_auc'] = 0.5  # Random performance
        else:
            metrics['roc_auc'] = 0.5
        
        return metrics
    
    @pytest.mark.benchmark
    def test_small_dataset_performance(self, small_dataset: Dataset, 
                                     benchmark_algorithms: List[dict],
                                     performance_thresholds: dict):
        """Test algorithm performance on small dataset."""
        thresholds = performance_thresholds["small_dataset"]
        results = []
        
        for algo_config in benchmark_algorithms:
            metrics = self._benchmark_algorithm(
                algo_config, small_dataset, thresholds
            )
            results.append({
                "algorithm": algo_config["name"],
                "dataset": "small",
                **metrics.__dict__
            })
        
        # Create performance report
        self._create_performance_report(results, "small_dataset_performance.json")
        
        # Verify all algorithms meet thresholds
        for result in results:
            assert result["fit_time"] <= thresholds["max_fit_time"], \
                f"{result['algorithm']} fit time exceeded threshold"
            assert result["predict_time"] <= thresholds["max_predict_time"], \
                f"{result['algorithm']} predict time exceeded threshold"
    
    @pytest.mark.benchmark
    def test_medium_dataset_performance(self, medium_dataset: Dataset,
                                      benchmark_algorithms: List[dict],
                                      performance_thresholds: dict):
        """Test algorithm performance on medium dataset."""
        thresholds = performance_thresholds["medium_dataset"]
        results = []
        
        for algo_config in benchmark_algorithms:
            metrics = self._benchmark_algorithm(
                algo_config, medium_dataset, thresholds
            )
            results.append({
                "algorithm": algo_config["name"],
                "dataset": "medium",
                **metrics.__dict__
            })
        
        # Create performance report
        self._create_performance_report(results, "medium_dataset_performance.json")
        
        # Verify performance doesn't degrade significantly
        for result in results:
            assert result["fit_time"] <= thresholds["max_fit_time"], \
                f"{result['algorithm']} fit time exceeded threshold"
    
    @pytest.mark.benchmark 
    @pytest.mark.slow
    def test_large_dataset_performance(self, large_dataset: Dataset,
                                     benchmark_algorithms: List[dict],
                                     performance_thresholds: dict):
        """Test algorithm performance on large dataset."""
        thresholds = performance_thresholds["large_dataset"]
        results = []
        
        # Only test fast algorithms on large dataset
        fast_algorithms = [
            algo for algo in benchmark_algorithms 
            if "LOF" not in algo["name"]  # LOF is O(n^2), skip for large data
        ]
        
        for algo_config in fast_algorithms:
            metrics = self._benchmark_algorithm(
                algo_config, large_dataset, thresholds
            )
            results.append({
                "algorithm": algo_config["name"],
                "dataset": "large",
                **metrics.__dict__
            })
        
        # Create performance report
        self._create_performance_report(results, "large_dataset_performance.json")
        
        # Verify scalability
        for result in results:
            assert result["fit_time"] <= thresholds["max_fit_time"], \
                f"{result['algorithm']} doesn't scale to large data"
    
    def _benchmark_algorithm(self, algo_config: dict, dataset: Dataset,
                           thresholds: dict) -> PerformanceMetrics:
        """Benchmark a single algorithm."""
        metrics = PerformanceMetrics()
        
        # Create detector
        adapter_class = algo_config["adapter_class"]
        detector = adapter_class(
            algorithm_name=algo_config["algorithm_name"],
            contamination_rate=ContaminationRate(0.1),
            **algo_config["params"]
        )
        
        # Measure fit time and memory
        start_time = time.time()
        _, fit_memory = self.measure_memory_usage(detector.fit, dataset)
        metrics.fit_time = time.time() - start_time
        
        # Measure predict time and memory
        start_time = time.time()
        result, predict_memory = self.measure_memory_usage(detector.predict, dataset)
        metrics.predict_time = time.time() - start_time
        
        # Total memory usage
        metrics.memory_usage_mb = max(fit_memory, predict_memory)
        
        # Calculate detection metrics
        true_labels = dataset.data[dataset.target_column].values
        predicted_labels = np.zeros(len(true_labels))
        anomaly_scores = np.zeros(len(true_labels))
        
        # Extract predictions and scores from result
        for anomaly in result.anomalies:
            predicted_labels[anomaly.index] = 1
            anomaly_scores[anomaly.index] = anomaly.score.value
        
        detection_metrics = self.calculate_detection_metrics(
            true_labels, predicted_labels, anomaly_scores
        )
        
        metrics.precision = detection_metrics['precision']
        metrics.recall = detection_metrics['recall']
        metrics.f1_score = detection_metrics['f1_score']
        metrics.roc_auc = detection_metrics['roc_auc']
        metrics.n_anomalies_detected = len(result.anomalies)
        metrics.anomaly_rate = result.anomaly_rate
        
        return metrics
    
    def _create_performance_report(self, results: List[dict], filename: str):
        """Create a performance report."""
        report_df = pd.DataFrame(results)
        
        # Save to file (would normally save to tests/benchmarks/reports/)
        print(f"\n--- Performance Report: {filename} ---")
        print(report_df.to_string(index=False))
        print(f"{'='*60}")
    
    @pytest.mark.benchmark
    def test_algorithm_scaling_characteristics(self, benchmark_algorithms: List[dict]):
        """Test how algorithms scale with data size."""
        dataset_sizes = [1000, 5000, 10000]
        scaling_results = []
        
        for size in dataset_sizes:
            # Generate dataset of specific size
            np.random.seed(42)
            n_anomalies = int(size * 0.1)
            
            normal_data = np.random.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], size - n_anomalies)
            anomaly_data = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], n_anomalies)
            
            data = np.vstack([normal_data, anomaly_data])
            labels = np.hstack([np.zeros(size - n_anomalies), np.ones(n_anomalies)])
            
            df = pd.DataFrame(data, columns=['feature_1', 'feature_2'])
            df['label'] = labels
            
            dataset = Dataset(
                name=f"scaling_test_{size}",
                data=df,
                target_column="label"
            )
            
            # Test IsolationForest scaling (representative algorithm)
            iso_config = next(algo for algo in benchmark_algorithms 
                            if algo["name"] == "IsolationForest_PyOD")
            
            detector = iso_config["adapter_class"](
                algorithm_name=iso_config["algorithm_name"],
                contamination_rate=ContaminationRate(0.1),
                **iso_config["params"]
            )
            
            # Measure fit time
            start_time = time.time()
            detector.fit(dataset)
            fit_time = time.time() - start_time
            
            # Measure predict time
            start_time = time.time()
            result = detector.predict(dataset)
            predict_time = time.time() - start_time
            
            scaling_results.append({
                "dataset_size": size,
                "fit_time": fit_time,
                "predict_time": predict_time,
                "total_time": fit_time + predict_time
            })
        
        # Analyze scaling characteristics
        scaling_df = pd.DataFrame(scaling_results)
        print(f"\n--- Scaling Analysis ---")
        print(scaling_df.to_string(index=False))
        
        # Verify that scaling is reasonable (not exponential)
        # For IsolationForest, should be roughly O(n log n)
        size_ratios = []
        time_ratios = []
        
        for i in range(1, len(scaling_results)):
            size_ratio = scaling_results[i]["dataset_size"] / scaling_results[i-1]["dataset_size"]
            time_ratio = scaling_results[i]["total_time"] / scaling_results[i-1]["total_time"]
            
            size_ratios.append(size_ratio)
            time_ratios.append(time_ratio)
        
        # Time ratio should not exceed size ratio significantly
        for size_r, time_r in zip(size_ratios, time_ratios):
            assert time_r <= size_r * 2, f"Poor scaling: time ratio {time_r} >> size ratio {size_r}"
    
    @pytest.mark.benchmark
    def test_memory_efficiency(self, medium_dataset: Dataset, benchmark_algorithms: List[dict]):
        """Test memory efficiency of algorithms."""
        memory_results = []
        
        for algo_config in benchmark_algorithms:
            detector = algo_config["adapter_class"](
                algorithm_name=algo_config["algorithm_name"],
                contamination_rate=ContaminationRate(0.1),
                **algo_config["params"]
            )
            
            # Measure memory during fit
            _, fit_memory = self.measure_memory_usage(detector.fit, medium_dataset)
            
            # Measure memory during predict
            _, predict_memory = self.measure_memory_usage(detector.predict, medium_dataset)
            
            memory_results.append({
                "algorithm": algo_config["name"],
                "fit_memory_mb": fit_memory,
                "predict_memory_mb": predict_memory,
                "total_memory_mb": fit_memory + predict_memory
            })
        
        # Create memory efficiency report
        memory_df = pd.DataFrame(memory_results)
        print(f"\n--- Memory Efficiency Report ---")
        print(memory_df.to_string(index=False))
        
        # Verify memory usage is reasonable
        for result in memory_results:
            assert result["total_memory_mb"] < 1000, \
                f"{result['algorithm']} uses excessive memory: {result['total_memory_mb']:.2f}MB"
    
    @pytest.mark.benchmark
    def test_accuracy_vs_speed_tradeoff(self, medium_dataset: Dataset, 
                                       benchmark_algorithms: List[dict]):
        """Analyze accuracy vs speed tradeoffs."""
        tradeoff_results = []
        
        for algo_config in benchmark_algorithms:
            detector = algo_config["adapter_class"](
                algorithm_name=algo_config["algorithm_name"],
                contamination_rate=ContaminationRate(0.1),
                **algo_config["params"]
            )
            
            # Measure total time
            start_time = time.time()
            detector.fit(medium_dataset)
            result = detector.predict(medium_dataset)
            total_time = time.time() - start_time
            
            # Calculate accuracy metrics
            true_labels = medium_dataset.data[medium_dataset.target_column].values
            predicted_labels = np.zeros(len(true_labels))
            anomaly_scores = np.zeros(len(true_labels))
            
            for anomaly in result.anomalies:
                predicted_labels[anomaly.index] = 1
                anomaly_scores[anomaly.index] = anomaly.score.value
            
            detection_metrics = self.calculate_detection_metrics(
                true_labels, predicted_labels, anomaly_scores
            )
            
            tradeoff_results.append({
                "algorithm": algo_config["name"],
                "total_time": total_time,
                "f1_score": detection_metrics['f1_score'],
                "roc_auc": detection_metrics['roc_auc'],
                "efficiency_score": detection_metrics['f1_score'] / total_time  # F1 per second
            })
        
        # Create tradeoff analysis
        tradeoff_df = pd.DataFrame(tradeoff_results)
        print(f"\n--- Accuracy vs Speed Tradeoff ---")
        print(tradeoff_df.to_string(index=False))
        
        # Verify that we have algorithms with different tradeoff profiles
        efficiency_scores = [r["efficiency_score"] for r in tradeoff_results]
        assert max(efficiency_scores) / min(efficiency_scores) > 2, \
            "Algorithms should show different efficiency profiles"