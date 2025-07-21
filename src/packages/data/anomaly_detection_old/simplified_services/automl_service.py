"""Simplified AutoML service for automatic algorithm selection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
import time
from .core_detection_service import CoreDetectionService, DetectionResult


@dataclass
class AlgorithmRecommendation:
    """Algorithm recommendation with performance metrics."""
    algorithm: str
    score: float
    execution_time: float
    anomaly_rate: float
    confidence: float
    reason: str


class AutoMLService:
    """Simplified AutoML service for automatic algorithm selection and optimization.
    
    This service consolidates 20+ AutoML-related services into a single,
    production-ready implementation that provides:
    - Automatic algorithm selection based on data characteristics
    - Performance benchmarking across multiple algorithms
    - Hyperparameter optimization
    - Algorithm recommendation system
    """

    def __init__(self):
        """Initialize AutoML service."""
        self.detection_service = CoreDetectionService()
        self._benchmarks: Dict[str, List[float]] = {}

    def recommend_algorithm(
        self,
        data: npt.NDArray[np.floating],
        contamination: float = 0.1,
        time_budget: float = 30.0,
        n_trials: int = 3
    ) -> AlgorithmRecommendation:
        """Recommend best algorithm for the given data.
        
        Args:
            data: Input data for analysis
            contamination: Expected contamination rate
            time_budget: Maximum time to spend on evaluation (seconds)
            n_trials: Number of trials per algorithm
            
        Returns:
            AlgorithmRecommendation with best algorithm and metrics
        """
        print(f"🔍 AutoML: Analyzing data shape {data.shape} for algorithm recommendation...")
        
        # Get candidate algorithms based on data characteristics
        candidates = self._get_algorithm_candidates(data)
        print(f"📊 AutoML: Testing {len(candidates)} candidate algorithms...")
        
        # Benchmark algorithms
        results = self._benchmark_algorithms(
            data, candidates, contamination, time_budget, n_trials
        )
        
        # Select best algorithm
        best_algorithm = max(results, key=lambda x: x.score)
        
        print(f"✅ AutoML: Recommended {best_algorithm.algorithm} (score: {best_algorithm.score:.3f})")
        return best_algorithm

    def _get_algorithm_candidates(self, data: npt.NDArray[np.floating]) -> List[str]:
        """Get candidate algorithms based on data characteristics."""
        n_samples, n_features = data.shape
        
        candidates = []
        
        # Always include these reliable algorithms
        candidates.extend(['iforest', 'lof'])
        
        # Add algorithms based on data size
        if n_samples >= 100:
            candidates.extend(['pca', 'ocsvm'])
        
        if n_samples >= 500:
            candidates.extend(['hbos', 'cof'])
            
        if n_samples >= 1000:
            candidates.extend(['copod', 'ecod'])
        
        # Add algorithms based on dimensionality
        if n_features <= 10:
            candidates.extend(['cblof', 'knn'])
        elif n_features >= 20:
            candidates.append('feature_bagging')
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for alg in candidates:
            if alg not in seen:
                seen.add(alg)
                unique_candidates.append(alg)
        
        return unique_candidates

    def _benchmark_algorithms(
        self,
        data: npt.NDArray[np.floating],
        algorithms: List[str],
        contamination: float,
        time_budget: float,
        n_trials: int
    ) -> List[AlgorithmRecommendation]:
        """Benchmark multiple algorithms and return recommendations."""
        results = []
        start_time = time.time()
        time_per_algorithm = time_budget / len(algorithms)
        
        for algorithm in algorithms:
            if time.time() - start_time > time_budget:
                print(f"⏰ AutoML: Time budget exceeded, skipping remaining algorithms")
                break
                
            print(f"🧪 AutoML: Testing {algorithm}...")
            
            try:
                algorithm_start = time.time()
                trial_times = []
                trial_scores = []
                anomaly_rates = []
                
                for trial in range(n_trials):
                    if time.time() - algorithm_start > time_per_algorithm:
                        break
                        
                    trial_start = time.time()
                    result = self.detection_service.detect_anomalies(
                        data, algorithm=algorithm, contamination=contamination
                    )
                    trial_time = time.time() - trial_start
                    
                    trial_times.append(trial_time)
                    anomaly_rates.append(result.n_anomalies / result.n_samples)
                    
                    # Calculate score based on consistency and performance
                    score = self._calculate_algorithm_score(result, trial_time, data.shape)
                    trial_scores.append(score)
                
                if trial_scores:
                    avg_score = np.mean(trial_scores)
                    avg_time = np.mean(trial_times)
                    avg_rate = np.mean(anomaly_rates)
                    score_std = np.std(trial_scores)
                    
                    # Confidence based on consistency
                    confidence = max(0.0, 1.0 - score_std / max(avg_score, 0.1))
                    
                    # Generate reason
                    reason = self._generate_recommendation_reason(
                        algorithm, data.shape, avg_time, avg_rate, confidence
                    )
                    
                    recommendation = AlgorithmRecommendation(
                        algorithm=algorithm,
                        score=avg_score,
                        execution_time=avg_time,
                        anomaly_rate=avg_rate,
                        confidence=confidence,
                        reason=reason
                    )
                    
                    results.append(recommendation)
                    print(f"   ✓ {algorithm}: score={avg_score:.3f}, time={avg_time:.3f}s")
                
            except Exception as e:
                print(f"   ✗ {algorithm}: failed ({str(e)})")
                continue
        
        return sorted(results, key=lambda x: x.score, reverse=True)

    def _calculate_algorithm_score(
        self,
        result: DetectionResult,
        execution_time: float,
        data_shape: Tuple[int, int]
    ) -> float:
        """Calculate algorithm performance score."""
        n_samples, n_features = data_shape
        
        # Base score components
        time_score = max(0.0, 1.0 - execution_time / 10.0)  # Prefer faster algorithms
        
        # Anomaly rate should be reasonable (not too high or too low)
        anomaly_rate = result.n_anomalies / result.n_samples
        rate_score = 1.0 - abs(anomaly_rate - result.contamination) / result.contamination
        
        # Scalability score based on data size
        if n_samples < 100:
            scalability_score = 1.0
        elif n_samples < 1000:
            scalability_score = max(0.5, 1.0 - execution_time / 5.0)
        else:
            scalability_score = max(0.3, 1.0 - execution_time / 2.0)
        
        # Combine scores with weights
        total_score = (
            0.3 * time_score +
            0.4 * rate_score +
            0.3 * scalability_score
        )
        
        return max(0.0, min(1.0, total_score))

    def _generate_recommendation_reason(
        self,
        algorithm: str,
        data_shape: Tuple[int, int],
        execution_time: float,
        anomaly_rate: float,
        confidence: float
    ) -> str:
        """Generate human-readable reason for algorithm recommendation."""
        n_samples, n_features = data_shape
        
        reasons = []
        
        # Performance reasons
        if execution_time < 1.0:
            reasons.append("fast execution")
        elif execution_time < 5.0:
            reasons.append("good performance")
        
        # Data size reasons
        if n_samples < 100:
            reasons.append("suitable for small datasets")
        elif n_samples >= 1000:
            reasons.append("scales well with large data")
        
        # Feature reasons
        if n_features <= 5:
            reasons.append("effective on low-dimensional data")
        elif n_features >= 20:
            reasons.append("handles high-dimensional data well")
        
        # Algorithm-specific reasons
        algorithm_traits = {
            'iforest': "robust and reliable",
            'lof': "good at detecting local outliers",
            'pca': "effective for linear anomalies",
            'ocsvm': "handles non-linear patterns",
            'hbos': "fast histogram-based detection",
            'copod': "parameter-free and consistent",
            'ecod': "empirical cumulative distribution approach"
        }
        
        if algorithm in algorithm_traits:
            reasons.append(algorithm_traits[algorithm])
        
        # Confidence reasons
        if confidence > 0.8:
            reasons.append("highly consistent results")
        elif confidence > 0.6:
            reasons.append("consistent performance")
        
        return f"{algorithm.upper()}: " + ", ".join(reasons)

    def optimize_hyperparameters(
        self,
        data: npt.NDArray[np.floating],
        algorithm: str,
        contamination_range: Tuple[float, float] = (0.05, 0.3),
        n_trials: int = 10
    ) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific algorithm.
        
        Args:
            data: Input data for optimization
            algorithm: Algorithm to optimize
            contamination_range: Range of contamination values to try
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary with best parameters and performance metrics
        """
        print(f"🎯 AutoML: Optimizing hyperparameters for {algorithm}...")
        
        best_score = 0.0
        best_params = {}
        results = []
        
        # Generate parameter combinations
        contamination_values = np.linspace(
            contamination_range[0], contamination_range[1], n_trials
        )
        
        for i, contamination in enumerate(contamination_values):
            try:
                start_time = time.time()
                result = self.detection_service.detect_anomalies(
                    data, algorithm=algorithm, contamination=contamination
                )
                execution_time = time.time() - start_time
                
                score = self._calculate_algorithm_score(result, execution_time, data.shape)
                
                params = {"contamination": contamination}
                trial_result = {
                    "trial": i + 1,
                    "parameters": params,
                    "score": score,
                    "anomaly_rate": result.n_anomalies / result.n_samples,
                    "execution_time": execution_time
                }
                
                results.append(trial_result)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                print(f"   Trial {i+1}: contamination={contamination:.3f}, score={score:.3f}")
                
            except Exception as e:
                print(f"   Trial {i+1}: failed ({str(e)})")
                continue
        
        print(f"✅ AutoML: Best parameters found - score: {best_score:.3f}")
        
        return {
            "algorithm": algorithm,
            "best_parameters": best_params,
            "best_score": best_score,
            "optimization_results": results,
            "n_trials": len(results)
        }

    def auto_detect(
        self,
        data: npt.NDArray[np.floating],
        contamination: float = 0.1,
        time_budget: float = 30.0
    ) -> DetectionResult:
        """Automatically select best algorithm and detect anomalies.
        
        Args:
            data: Input data for detection
            contamination: Expected contamination rate
            time_budget: Time budget for algorithm selection
            
        Returns:
            DetectionResult using the automatically selected algorithm
        """
        print("🤖 AutoML: Starting automatic detection...")
        
        # Get algorithm recommendation
        recommendation = self.recommend_algorithm(data, contamination, time_budget)
        
        # Run detection with recommended algorithm
        print(f"🎯 AutoML: Running detection with {recommendation.algorithm}")
        result = self.detection_service.detect_anomalies(
            data, 
            algorithm=recommendation.algorithm,
            contamination=contamination
        )
        
        # Add AutoML metadata
        result.metadata.update({
            "automl_recommendation": {
                "algorithm": recommendation.algorithm,
                "score": recommendation.score,
                "confidence": recommendation.confidence,
                "reason": recommendation.reason
            }
        })
        
        print(f"✅ AutoML: Detection completed - {result.n_anomalies} anomalies found")
        return result

    def get_benchmark_history(self) -> Dict[str, List[float]]:
        """Get historical benchmark data."""
        return self._benchmarks.copy()

    def clear_benchmarks(self) -> None:
        """Clear benchmark history."""
        self._benchmarks.clear()