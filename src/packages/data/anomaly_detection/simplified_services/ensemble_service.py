"""Simplified ensemble service for combining multiple anomaly detection algorithms."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .core_detection_service import CoreDetectionService, DetectionResult


@dataclass
class EnsembleConfig:
    """Configuration for ensemble detection."""
    algorithms: List[str]
    weights: Optional[List[float]] = None
    voting_method: str = "majority"  # "majority", "weighted", "unanimous"
    contamination: float = 0.1
    parallel: bool = True
    max_workers: int = 4


class EnsembleService:
    """Simplified ensemble service that combines multiple algorithms.
    
    This service consolidates 10+ ensemble-related services into a single,
    production-ready implementation that provides:
    - Multiple algorithm ensemble detection
    - Various voting strategies (majority, weighted, unanimous)
    - Parallel algorithm execution
    - Confidence scoring based on agreement
    - Performance optimization
    """

    def __init__(self):
        """Initialize ensemble service."""
        self.detection_service = CoreDetectionService()
        self._ensemble_history: List[Dict[str, Any]] = []

    def detect_ensemble(
        self,
        data: npt.NDArray[np.floating],
        config: EnsembleConfig
    ) -> DetectionResult:
        """Detect anomalies using ensemble of algorithms.
        
        Args:
            data: Input data for detection
            config: Ensemble configuration
            
        Returns:
            DetectionResult with ensemble predictions and metadata
        """
        print(f"🎭 Ensemble: Running {len(config.algorithms)} algorithms...")
        start_time = time.time()
        
        # Get individual algorithm results
        if config.parallel and len(config.algorithms) > 1:
            individual_results = self._run_parallel_detection(data, config)
        else:
            individual_results = self._run_sequential_detection(data, config)
        
        # Combine results using voting strategy
        ensemble_predictions, confidence_scores = self._combine_predictions(
            individual_results, config
        )
        
        execution_time = time.time() - start_time
        
        # Create ensemble result
        ensemble_result = DetectionResult(
            predictions=ensemble_predictions,
            scores=confidence_scores,
            algorithm=f"ensemble({','.join(config.algorithms)})",
            contamination=config.contamination,
            execution_time=execution_time,
            metadata={
                "ensemble_config": {
                    "algorithms": config.algorithms,
                    "voting_method": config.voting_method,
                    "weights": config.weights,
                    "parallel": config.parallel
                },
                "individual_results": [
                    {
                        "algorithm": result.algorithm,
                        "n_anomalies": result.n_anomalies,
                        "execution_time": result.execution_time
                    }
                    for result in individual_results
                ],
                "agreement_metrics": self._calculate_agreement_metrics(individual_results)
            }
        )
        
        # Update history
        self._ensemble_history.append({
            "timestamp": ensemble_result.timestamp,
            "algorithms": config.algorithms,
            "n_samples": len(data),
            "n_anomalies": ensemble_result.n_anomalies,
            "execution_time": execution_time,
            "agreement_score": ensemble_result.metadata["agreement_metrics"]["overall_agreement"]
        })
        
        print(f"✅ Ensemble: {ensemble_result.n_anomalies} anomalies detected (agreement: {ensemble_result.metadata['agreement_metrics']['overall_agreement']:.2f})")
        return ensemble_result

    def _run_parallel_detection(
        self,
        data: npt.NDArray[np.floating],
        config: EnsembleConfig
    ) -> List[DetectionResult]:
        """Run detection algorithms in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            # Submit all algorithms
            future_to_algorithm = {
                executor.submit(
                    self.detection_service.detect_anomalies,
                    data,
                    algorithm,
                    config.contamination
                ): algorithm
                for algorithm in config.algorithms
            }
            
            # Collect results
            for future in as_completed(future_to_algorithm):
                algorithm = future_to_algorithm[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"   ✓ {algorithm}: {result.n_anomalies} anomalies")
                except Exception as e:
                    print(f"   ✗ {algorithm}: failed ({str(e)})")
                    continue
        
        return results

    def _run_sequential_detection(
        self,
        data: npt.NDArray[np.floating],
        config: EnsembleConfig
    ) -> List[DetectionResult]:
        """Run detection algorithms sequentially."""
        results = []
        
        for algorithm in config.algorithms:
            try:
                result = self.detection_service.detect_anomalies(
                    data, algorithm, config.contamination
                )
                results.append(result)
                print(f"   ✓ {algorithm}: {result.n_anomalies} anomalies")
            except Exception as e:
                print(f"   ✗ {algorithm}: failed ({str(e)})")
                continue
        
        return results

    def _combine_predictions(
        self,
        results: List[DetectionResult],
        config: EnsembleConfig
    ) -> Tuple[npt.NDArray[np.integer], npt.NDArray[np.floating]]:
        """Combine individual predictions using voting strategy."""
        if not results:
            raise ValueError("No successful algorithm results to combine")
        
        n_samples = len(results[0].predictions)
        n_algorithms = len(results)
        
        # Stack all predictions
        all_predictions = np.stack([result.predictions for result in results])
        
        if config.voting_method == "majority":
            # Majority voting
            vote_counts = np.sum(all_predictions, axis=0)
            ensemble_predictions = (vote_counts > n_algorithms / 2).astype(int)
            confidence_scores = vote_counts / n_algorithms
            
        elif config.voting_method == "weighted":
            # Weighted voting
            if config.weights is None:
                weights = np.ones(n_algorithms) / n_algorithms
            else:
                weights = np.array(config.weights)
                weights = weights / np.sum(weights)  # Normalize
            
            weighted_votes = np.sum(all_predictions * weights[:, np.newaxis], axis=0)
            ensemble_predictions = (weighted_votes > 0.5).astype(int)
            confidence_scores = weighted_votes
            
        elif config.voting_method == "unanimous":
            # Unanimous voting (all algorithms must agree)
            ensemble_predictions = np.all(all_predictions, axis=0).astype(int)
            confidence_scores = np.mean(all_predictions, axis=0)
            
        else:
            raise ValueError(f"Unknown voting method: {config.voting_method}")
        
        return ensemble_predictions, confidence_scores

    def _calculate_agreement_metrics(self, results: List[DetectionResult]) -> Dict[str, float]:
        """Calculate agreement metrics between algorithms."""
        if len(results) < 2:
            return {"overall_agreement": 1.0, "pairwise_agreements": []}
        
        predictions = [result.predictions for result in results]
        n_algorithms = len(predictions)
        
        # Calculate pairwise agreements
        pairwise_agreements = []
        for i in range(n_algorithms):
            for j in range(i + 1, n_algorithms):
                agreement = np.mean(predictions[i] == predictions[j])
                pairwise_agreements.append(agreement)
        
        overall_agreement = np.mean(pairwise_agreements) if pairwise_agreements else 1.0
        
        return {
            "overall_agreement": float(overall_agreement),
            "pairwise_agreements": pairwise_agreements,
            "algorithm_count": n_algorithms
        }

    def create_smart_ensemble(
        self,
        data: npt.NDArray[np.floating],
        contamination: float = 0.1,
        max_algorithms: int = 5,
        time_budget: float = 60.0
    ) -> DetectionResult:
        """Create an intelligent ensemble based on data characteristics.
        
        Args:
            data: Input data for detection
            contamination: Expected contamination rate
            max_algorithms: Maximum number of algorithms to include
            time_budget: Time budget for ensemble creation
            
        Returns:
            DetectionResult from optimized ensemble
        """
        print("🧠 Ensemble: Creating smart ensemble...")
        
        # Get candidate algorithms based on data characteristics
        candidates = self._get_smart_candidates(data, max_algorithms)
        
        # Create ensemble configuration
        config = EnsembleConfig(
            algorithms=candidates,
            voting_method="weighted",
            contamination=contamination,
            parallel=True,
            max_workers=min(len(candidates), 4)
        )
        
        # Calculate weights based on algorithm suitability
        config.weights = self._calculate_smart_weights(data, candidates)
        
        print(f"🎯 Ensemble: Selected algorithms: {candidates}")
        print(f"⚖️  Ensemble: Weights: {[f'{w:.2f}' for w in config.weights]}")
        
        return self.detect_ensemble(data, config)

    def _get_smart_candidates(
        self,
        data: npt.NDArray[np.floating],
        max_algorithms: int
    ) -> List[str]:
        """Select smart algorithm candidates based on data characteristics."""
        n_samples, n_features = data.shape
        
        # Base algorithms (always include)
        candidates = ["iforest", "lof"]
        
        # Add algorithms based on data size and dimensionality
        if n_samples >= 200:
            candidates.append("pca")
        
        if n_samples >= 500 and n_features <= 20:
            candidates.append("ocsvm")
        
        if n_samples >= 300:
            candidates.append("hbos")
        
        if n_features >= 5:
            candidates.append("copod")
        
        if n_samples >= 1000:
            candidates.append("ecod")
        
        # Limit to max_algorithms
        return candidates[:max_algorithms]

    def _calculate_smart_weights(
        self,
        data: npt.NDArray[np.floating],
        algorithms: List[str]
    ) -> List[float]:
        """Calculate smart weights for algorithms based on data characteristics."""
        n_samples, n_features = data.shape
        weights = []
        
        for algorithm in algorithms:
            if algorithm == "iforest":
                # IForest is generally reliable
                weight = 0.8 if n_samples >= 100 else 0.6
            elif algorithm == "lof":
                # LOF is good for local outliers
                weight = 0.7 if n_samples <= 1000 else 0.5
            elif algorithm == "pca":
                # PCA is good for linear anomalies
                weight = 0.6 if n_features >= 5 else 0.4
            elif algorithm == "ocsvm":
                # OCSVM handles non-linear patterns
                weight = 0.7 if n_features <= 20 else 0.4
            elif algorithm == "hbos":
                # HBOS is fast and effective
                weight = 0.6
            elif algorithm == "copod":
                # COPOD is parameter-free
                weight = 0.7
            elif algorithm == "ecod":
                # ECOD is good for large datasets
                weight = 0.8 if n_samples >= 1000 else 0.5
            else:
                weight = 0.5  # Default weight
            
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        return [w / total_weight for w in weights]

    def benchmark_ensemble_strategies(
        self,
        data: npt.NDArray[np.floating],
        algorithms: List[str],
        contamination: float = 0.1
    ) -> Dict[str, Any]:
        """Benchmark different ensemble strategies.
        
        Args:
            data: Input data for benchmarking
            algorithms: List of algorithms to use
            contamination: Expected contamination rate
            
        Returns:
            Dictionary with benchmark results for each strategy
        """
        print("📊 Ensemble: Benchmarking ensemble strategies...")
        
        strategies = ["majority", "weighted", "unanimous"]
        results = {}
        
        for strategy in strategies:
            print(f"   Testing {strategy} voting...")
            
            config = EnsembleConfig(
                algorithms=algorithms,
                voting_method=strategy,
                contamination=contamination,
                parallel=True
            )
            
            try:
                start_time = time.time()
                result = self.detect_ensemble(data, config)
                execution_time = time.time() - start_time
                
                results[strategy] = {
                    "n_anomalies": result.n_anomalies,
                    "anomaly_rate": result.n_anomalies / len(data),
                    "execution_time": execution_time,
                    "agreement_score": result.metadata["agreement_metrics"]["overall_agreement"],
                    "success": True
                }
                
            except Exception as e:
                results[strategy] = {
                    "error": str(e),
                    "success": False
                }
        
        return results

    def get_ensemble_history(self) -> List[Dict[str, Any]]:
        """Get ensemble execution history."""
        return self._ensemble_history.copy()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get ensemble performance statistics."""
        if not self._ensemble_history:
            return {"total_ensembles": 0}
        
        execution_times = [h["execution_time"] for h in self._ensemble_history]
        agreement_scores = [h["agreement_score"] for h in self._ensemble_history]
        algorithm_counts = [len(h["algorithms"]) for h in self._ensemble_history]
        
        return {
            "total_ensembles": len(self._ensemble_history),
            "average_execution_time": np.mean(execution_times),
            "average_agreement_score": np.mean(agreement_scores),
            "average_algorithm_count": np.mean(algorithm_counts),
            "best_agreement_score": np.max(agreement_scores),
            "most_used_algorithms": self._get_most_used_algorithms()
        }

    def _get_most_used_algorithms(self) -> Dict[str, int]:
        """Get count of most used algorithms in ensembles."""
        algorithm_counts = {}
        
        for history in self._ensemble_history:
            for algorithm in history["algorithms"]:
                algorithm_counts[algorithm] = algorithm_counts.get(algorithm, 0) + 1
        
        return dict(sorted(algorithm_counts.items(), key=lambda x: x[1], reverse=True))