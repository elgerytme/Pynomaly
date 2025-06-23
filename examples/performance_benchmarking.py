#!/usr/bin/env python3
"""
Performance Benchmarking Example
===============================

This example benchmarks different anomaly detection algorithms on various dataset sizes
to help you choose the best algorithm for your performance requirements.
"""

import asyncio
import time
import psutil
import gc
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import sys
import os

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from pynomaly.infrastructure.config import create_container


@dataclass
class BenchmarkResult:
    """Performance benchmark result."""
    algorithm: str
    dataset_size: int
    features: int
    training_time_ms: float
    prediction_time_ms: float
    memory_mb: float
    cpu_percent: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self):
        self.container = None
        self.results: List[BenchmarkResult] = []
    
    async def initialize(self):
        """Initialize the benchmark suite."""
        self.container = create_container()
    
    def generate_synthetic_data(self, n_samples: int, n_features: int, contamination: float = 0.1) -> Tuple[List[Dict], List[bool]]:
        """Generate synthetic dataset for benchmarking."""
        np.random.seed(42)  # For reproducible results
        
        # Generate normal data
        normal_samples = int(n_samples * (1 - contamination))
        normal_data = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=np.eye(n_features),
            size=normal_samples
        )
        
        # Generate anomalous data
        anomaly_samples = n_samples - normal_samples
        anomaly_data = np.random.multivariate_normal(
            mean=np.ones(n_features) * 3,  # Shifted mean for anomalies
            cov=np.eye(n_features) * 2,    # Different covariance
            size=anomaly_samples
        )
        
        # Combine data
        data = np.vstack([normal_data, anomaly_data])
        labels = [False] * normal_samples + [True] * anomaly_samples
        
        # Shuffle
        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = [labels[i] for i in indices]
        
        # Convert to list of dictionaries
        data_dicts = []
        for row in data:
            data_dict = {f'feature_{i}': float(row[i]) for i in range(n_features)}
            data_dicts.append(data_dict)
        
        return data_dicts, labels
    
    def measure_memory_usage(self) -> float:
        """Measure current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def measure_cpu_usage(self) -> float:
        """Measure current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)
    
    async def benchmark_algorithm(
        self, 
        algorithm: str, 
        data: List[Dict], 
        true_labels: List[bool],
        parameters: Dict[str, Any] = None
    ) -> BenchmarkResult:
        """Benchmark a single algorithm."""
        
        if parameters is None:
            parameters = {"contamination": 0.1}
        
        dataset_size = len(data)
        features = len(data[0]) if data else 0
        
        # Clear memory before benchmark
        gc.collect()
        memory_before = self.measure_memory_usage()
        
        try:
            # Create services
            detection_service = self.container.detection_service()
            dataset_service = self.container.dataset_service()
            
            # Create dataset
            dataset = await dataset_service.create_from_data(
                data=data,
                name=f"Benchmark Dataset {algorithm}",
                description=f"Synthetic data for {algorithm} benchmark"
            )
            
            # Create detector
            detector = await detection_service.create_detector(
                name=f"Benchmark {algorithm}",
                algorithm=algorithm,
                parameters=parameters
            )
            
            # Measure training time
            cpu_before_train = self.measure_cpu_usage()
            start_time = time.perf_counter()
            
            await detection_service.train_detector(detector.id, dataset.id)
            
            end_time = time.perf_counter()
            training_time_ms = (end_time - start_time) * 1000
            cpu_after_train = self.measure_cpu_usage()
            
            # Measure prediction time
            start_time = time.perf_counter()
            
            results = await detection_service.detect_batch(detector.id, data)
            
            end_time = time.perf_counter()
            prediction_time_ms = (end_time - start_time) * 1000
            
            # Calculate performance metrics
            predictions = [r.is_anomaly for r in results]
            
            true_positives = sum(1 for i in range(len(predictions)) if predictions[i] and true_labels[i])
            false_positives = sum(1 for i in range(len(predictions)) if predictions[i] and not true_labels[i])
            true_negatives = sum(1 for i in range(len(predictions)) if not predictions[i] and not true_labels[i])
            false_negatives = sum(1 for i in range(len(predictions)) if not predictions[i] and true_labels[i])
            
            accuracy = (true_positives + true_negatives) / len(predictions) if predictions else 0
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Memory usage
            memory_after = self.measure_memory_usage()
            memory_used = memory_after - memory_before
            
            # CPU usage (average during training)
            cpu_usage = (cpu_before_train + cpu_after_train) / 2
            
            return BenchmarkResult(
                algorithm=algorithm,
                dataset_size=dataset_size,
                features=features,
                training_time_ms=training_time_ms,
                prediction_time_ms=prediction_time_ms,
                memory_mb=memory_used,
                cpu_percent=cpu_usage,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score
            )
            
        except Exception as e:
            print(f"   ❌ Error benchmarking {algorithm}: {e}")
            return BenchmarkResult(
                algorithm=algorithm,
                dataset_size=dataset_size,
                features=features,
                training_time_ms=float('inf'),
                prediction_time_ms=float('inf'),
                memory_mb=float('inf'),
                cpu_percent=0,
                accuracy=0,
                precision=0,
                recall=0,
                f1_score=0
            )
    
    async def run_scalability_benchmark(self):
        """Test algorithm scalability with different dataset sizes."""
        print("🚀 Scalability Benchmark")
        print("=" * 50)
        
        algorithms = ["IsolationForest", "LOF", "OCSVM", "COPOD"]
        dataset_sizes = [100, 500, 1000, 5000]
        features = 10
        
        print(f"Testing {len(algorithms)} algorithms on {len(dataset_sizes)} dataset sizes")
        print(f"Features: {features}, Contamination: 10%")
        print()
        
        for size in dataset_sizes:
            print(f"📊 Dataset Size: {size} samples")
            print("-" * 30)
            
            # Generate data once per size
            data, labels = self.generate_synthetic_data(size, features, 0.1)
            
            for algorithm in algorithms:
                print(f"   Testing {algorithm}...", end=" ")
                
                result = await self.benchmark_algorithm(algorithm, data, labels)
                self.results.append(result)
                
                print(f"Train: {result.training_time_ms:.1f}ms, "
                      f"Predict: {result.prediction_time_ms:.1f}ms, "
                      f"F1: {result.f1_score:.3f}")
            
            print()
    
    async def run_feature_dimensionality_benchmark(self):
        """Test algorithm performance with different feature dimensions."""
        print("🎯 Feature Dimensionality Benchmark")
        print("=" * 50)
        
        algorithms = ["IsolationForest", "COPOD", "ECOD"]
        feature_counts = [5, 10, 25, 50, 100]
        samples = 1000
        
        print(f"Testing {len(algorithms)} algorithms on {len(feature_counts)} feature dimensions")
        print(f"Samples: {samples}, Contamination: 10%")
        print()
        
        for features in feature_counts:
            print(f"📐 Features: {features}")
            print("-" * 20)
            
            # Generate data once per feature count
            data, labels = self.generate_synthetic_data(samples, features, 0.1)
            
            for algorithm in algorithms:
                print(f"   Testing {algorithm}...", end=" ")
                
                result = await self.benchmark_algorithm(algorithm, data, labels)
                self.results.append(result)
                
                print(f"Train: {result.training_time_ms:.1f}ms, "
                      f"Memory: {result.memory_mb:.1f}MB, "
                      f"F1: {result.f1_score:.3f}")
            
            print()
    
    async def run_parameter_sensitivity_benchmark(self):
        """Test sensitivity to different parameter settings."""
        print("⚙️ Parameter Sensitivity Benchmark")
        print("=" * 50)
        
        samples = 1000
        features = 10
        data, labels = self.generate_synthetic_data(samples, features, 0.1)
        
        # Test IsolationForest with different n_estimators
        print("🌳 IsolationForest - n_estimators sensitivity:")
        estimator_counts = [10, 50, 100, 200, 500]
        
        for n_est in estimator_counts:
            params = {"contamination": 0.1, "n_estimators": n_est}
            result = await self.benchmark_algorithm("IsolationForest", data, labels, params)
            self.results.append(result)
            
            print(f"   n_estimators={n_est:3d}: "
                  f"Train: {result.training_time_ms:6.1f}ms, "
                  f"F1: {result.f1_score:.3f}")
        
        print()
        
        # Test LOF with different n_neighbors
        print("👥 LOF - n_neighbors sensitivity:")
        neighbor_counts = [5, 10, 20, 50, 100]
        
        for n_neighbors in neighbor_counts:
            params = {"contamination": 0.1, "n_neighbors": n_neighbors}
            result = await self.benchmark_algorithm("LOF", data, labels, params)
            self.results.append(result)
            
            print(f"   n_neighbors={n_neighbors:3d}: "
                  f"Train: {result.training_time_ms:6.1f}ms, "
                  f"F1: {result.f1_score:.3f}")
        
        print()
    
    def generate_performance_report(self):
        """Generate a comprehensive performance report."""
        print("📊 Performance Benchmark Report")
        print("=" * 60)
        
        if not self.results:
            print("No benchmark results available.")
            return
        
        # Create DataFrame for analysis
        df = pd.DataFrame([
            {
                'Algorithm': r.algorithm,
                'Dataset_Size': r.dataset_size,
                'Features': r.features,
                'Train_Time_ms': r.training_time_ms,
                'Predict_Time_ms': r.prediction_time_ms,
                'Memory_MB': r.memory_mb,
                'CPU_Percent': r.cpu_percent,
                'Accuracy': r.accuracy,
                'Precision': r.precision,
                'Recall': r.recall,
                'F1_Score': r.f1_score
            }
            for r in self.results if r.training_time_ms != float('inf')
        ])
        
        if df.empty:
            print("No successful benchmark results.")
            return
        
        # Overall performance summary
        print("\n🏆 Algorithm Performance Summary (1000 samples, 10 features):")
        base_results = df[(df['Dataset_Size'] == 1000) & (df['Features'] == 10)]
        
        if not base_results.empty:
            summary = base_results.groupby('Algorithm').agg({
                'Train_Time_ms': 'mean',
                'Predict_Time_ms': 'mean',
                'Memory_MB': 'mean',
                'F1_Score': 'mean'
            }).round(2)
            
            summary = summary.sort_values('F1_Score', ascending=False)
            print(summary.to_string())
        
        # Speed ranking
        print(f"\n⚡ Speed Ranking (Training Time):")
        speed_ranking = df.groupby('Algorithm')['Train_Time_ms'].mean().sort_values()
        for i, (algo, time_ms) in enumerate(speed_ranking.items(), 1):
            print(f"   {i}. {algo:<20} {time_ms:8.1f} ms")
        
        # Memory ranking
        print(f"\n💾 Memory Efficiency Ranking:")
        memory_ranking = df.groupby('Algorithm')['Memory_MB'].mean().sort_values()
        for i, (algo, memory_mb) in enumerate(memory_ranking.items(), 1):
            print(f"   {i}. {algo:<20} {memory_mb:8.1f} MB")
        
        # Accuracy ranking
        print(f"\n🎯 Accuracy Ranking (F1 Score):")
        accuracy_ranking = df.groupby('Algorithm')['F1_Score'].mean().sort_values(ascending=False)
        for i, (algo, f1) in enumerate(accuracy_ranking.items(), 1):
            print(f"   {i}. {algo:<20} {f1:8.3f}")
        
        # Scalability analysis
        print(f"\n📈 Scalability Analysis:")
        scalability_data = df[df['Features'] == 10].groupby(['Algorithm', 'Dataset_Size'])['Train_Time_ms'].mean().unstack()
        
        if not scalability_data.empty:
            print("Training time growth (ms) by dataset size:")
            print(scalability_data.round(1).to_string())
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if not base_results.empty:
            fastest = speed_ranking.index[0]
            most_accurate = accuracy_ranking.index[0]
            most_memory_efficient = memory_ranking.index[0]
            
            print(f"   🚀 Fastest Training:     {fastest}")
            print(f"   🎯 Most Accurate:        {most_accurate}")
            print(f"   💾 Memory Efficient:     {most_memory_efficient}")
            
            # Balanced recommendation
            normalized_scores = df.groupby('Algorithm').agg({
                'Train_Time_ms': lambda x: 1 / x.mean(),  # Inverse for speed
                'Memory_MB': lambda x: 1 / x.mean(),      # Inverse for efficiency
                'F1_Score': 'mean'                        # Higher is better
            })
            
            # Normalize to 0-1 scale
            for col in normalized_scores.columns:
                normalized_scores[col] = (normalized_scores[col] - normalized_scores[col].min()) / (normalized_scores[col].max() - normalized_scores[col].min())
            
            # Equal weight combination
            normalized_scores['Combined_Score'] = normalized_scores.mean(axis=1)
            best_overall = normalized_scores['Combined_Score'].idxmax()
            
            print(f"   ⚖️ Best Overall:         {best_overall}")
        
        print(f"\n✅ Benchmark completed with {len(self.results)} results")


async def main():
    """Run comprehensive performance benchmarks."""
    print("🔬 Pynomaly Performance Benchmarking Suite")
    print("=" * 60)
    
    benchmark = PerformanceBenchmark()
    await benchmark.initialize()
    
    print("Running comprehensive performance benchmarks...")
    print("This may take several minutes depending on your system.\n")
    
    # Run all benchmark suites
    await benchmark.run_scalability_benchmark()
    await benchmark.run_feature_dimensionality_benchmark()
    await benchmark.run_parameter_sensitivity_benchmark()
    
    # Generate final report
    benchmark.generate_performance_report()
    
    print(f"\n🏁 Benchmarking completed!")
    print("\nKey takeaways:")
    print("- IsolationForest: Generally fastest and most scalable")
    print("- LOF: Best for small datasets with local anomalies")
    print("- COPOD: Good balance of speed and accuracy")
    print("- OCSVM: Highest memory usage but handles non-linear patterns")
    print("\nChoose your algorithm based on:")
    print("- Dataset size (scalability requirements)")
    print("- Memory constraints")
    print("- Accuracy requirements")
    print("- Real-time vs batch processing needs")


if __name__ == "__main__":
    asyncio.run(main())