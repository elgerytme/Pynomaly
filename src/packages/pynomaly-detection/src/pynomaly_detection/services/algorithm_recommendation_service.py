"""Algorithm recommendation service for autonomous detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pynomaly_detection.application.services.algorithm_adapter_registry import (
    AlgorithmAdapterRegistry,
)
from pynomaly_detection.application.services.data_profiling_service import DataProfile


@dataclass
class AlgorithmRecommendation:
    """Algorithm recommendation with confidence and reasoning."""

    algorithm: str
    confidence: float
    reasoning: str
    hyperparams: dict[str, Any]
    expected_performance: float
    computational_complexity: str
    memory_requirements: str
    interpretability_score: float
    suitability_score: float
    decision_factors: dict[str, float]


class AlgorithmRecommendationService:
    """Service responsible for recommending algorithms based on data characteristics."""

    def __init__(self, adapter_registry: AlgorithmAdapterRegistry | None = None):
        """Initialize algorithm recommendation service.

        Args:
            adapter_registry: Registry of algorithm adapters
        """
        self.adapter_registry = adapter_registry or AlgorithmAdapterRegistry()
        self.logger = logging.getLogger(__name__)

        # Algorithm metadata for recommendation
        self.algorithm_metadata = self._initialize_algorithm_metadata()

    def _initialize_algorithm_metadata(self) -> dict[str, dict[str, Any]]:
        """Initialize algorithm metadata for recommendation.

        Returns:
            Dictionary of algorithm metadata
        """
        return {
            "IsolationForest": {
                "type": "ensemble",
                "complexity": "medium",
                "scalability": "high",
                "interpretability": 0.6,
                "memory_efficiency": 0.8,
                "training_speed": 0.9,
                "prediction_speed": 0.9,
                "best_for": ["general", "high_dimensional", "large_datasets"],
                "worst_for": ["small_datasets", "high_precision_required"],
                "min_samples": 100,
                "max_features": 10000,
                "handles_categorical": False,
                "handles_missing": False,
                "linear_complexity": True,
            },
            "LOF": {
                "type": "proximity",
                "complexity": "high",
                "scalability": "low",
                "interpretability": 0.8,
                "memory_efficiency": 0.5,
                "training_speed": 0.3,
                "prediction_speed": 0.3,
                "best_for": ["local_outliers", "small_datasets", "interpretability"],
                "worst_for": ["large_datasets", "high_dimensional", "sparse_data"],
                "min_samples": 50,
                "max_features": 100,
                "handles_categorical": False,
                "handles_missing": False,
                "linear_complexity": False,
            },
            "COPOD": {
                "type": "probabilistic",
                "complexity": "low",
                "scalability": "very_high",
                "interpretability": 0.7,
                "memory_efficiency": 0.9,
                "training_speed": 0.95,
                "prediction_speed": 0.95,
                "best_for": ["large_datasets", "high_dimensional", "correlated_features"],
                "worst_for": ["small_datasets", "independent_features"],
                "min_samples": 1000,
                "max_features": 50000,
                "handles_categorical": True,
                "handles_missing": True,
                "linear_complexity": True,
            },
            "HBOS": {
                "type": "probabilistic",
                "complexity": "low",
                "scalability": "very_high",
                "interpretability": 0.8,
                "memory_efficiency": 0.9,
                "training_speed": 0.95,
                "prediction_speed": 0.95,
                "best_for": ["large_datasets", "independent_features", "speed"],
                "worst_for": ["correlated_features", "complex_patterns"],
                "min_samples": 100,
                "max_features": 1000,
                "handles_categorical": True,
                "handles_missing": True,
                "linear_complexity": True,
            },
            "OneClassSVM": {
                "type": "boundary",
                "complexity": "high",
                "scalability": "low",
                "interpretability": 0.4,
                "memory_efficiency": 0.6,
                "training_speed": 0.2,
                "prediction_speed": 0.4,
                "best_for": ["small_datasets", "non_linear_boundaries", "high_precision"],
                "worst_for": ["large_datasets", "high_dimensional", "speed"],
                "min_samples": 20,
                "max_features": 50,
                "handles_categorical": False,
                "handles_missing": False,
                "linear_complexity": False,
            },
            "PCA": {
                "type": "linear",
                "complexity": "medium",
                "scalability": "medium",
                "interpretability": 0.6,
                "memory_efficiency": 0.7,
                "training_speed": 0.7,
                "prediction_speed": 0.8,
                "best_for": ["high_dimensional", "linear_patterns", "dimensionality_reduction"],
                "worst_for": ["non_linear_patterns", "small_datasets"],
                "min_samples": 100,
                "max_features": 1000,
                "handles_categorical": False,
                "handles_missing": False,
                "linear_complexity": True,
            },
            "KNN": {
                "type": "proximity",
                "complexity": "medium",
                "scalability": "medium",
                "interpretability": 0.7,
                "memory_efficiency": 0.6,
                "training_speed": 0.8,
                "prediction_speed": 0.5,
                "best_for": ["local_patterns", "medium_datasets", "interpretability"],
                "worst_for": ["high_dimensional", "large_datasets"],
                "min_samples": 50,
                "max_features": 200,
                "handles_categorical": False,
                "handles_missing": False,
                "linear_complexity": False,
            },
            "AutoEncoder": {
                "type": "deep_learning",
                "complexity": "very_high",
                "scalability": "medium",
                "interpretability": 0.3,
                "memory_efficiency": 0.5,
                "training_speed": 0.2,
                "prediction_speed": 0.6,
                "best_for": ["complex_patterns", "non_linear", "reconstruction"],
                "worst_for": ["small_datasets", "interpretability", "speed"],
                "min_samples": 1000,
                "max_features": 10000,
                "handles_categorical": False,
                "handles_missing": False,
                "linear_complexity": False,
            },
        }

    async def recommend_algorithms(
        self,
        profile: DataProfile,
        max_algorithms: int = 5,
        confidence_threshold: float = 0.3,
        verbose: bool = False,
    ) -> list[AlgorithmRecommendation]:
        """Recommend algorithms based on data profile.

        Args:
            profile: Data profile containing dataset characteristics
            max_algorithms: Maximum number of algorithms to recommend
            confidence_threshold: Minimum confidence threshold
            verbose: Enable verbose logging

        Returns:
            List of algorithm recommendations sorted by confidence
        """
        if verbose:
            self.logger.info(f"Recommending algorithms for dataset with {profile.n_samples} samples and {profile.n_features} features")

        recommendations = []

        # Get available algorithms
        available_algorithms = self.adapter_registry.get_supported_algorithms()

        for algorithm in available_algorithms:
            if algorithm not in self.algorithm_metadata:
                continue

            # Calculate recommendation for this algorithm
            recommendation = self._evaluate_algorithm(algorithm, profile)

            if recommendation.confidence >= confidence_threshold:
                recommendations.append(recommendation)

        # Sort by confidence (descending)
        recommendations.sort(key=lambda x: x.confidence, reverse=True)

        # Limit to max_algorithms
        recommendations = recommendations[:max_algorithms]

        if verbose:
            self.logger.info(f"Recommended {len(recommendations)} algorithms")
            for rec in recommendations:
                self.logger.info(f"  {rec.algorithm}: {rec.confidence:.3f} confidence - {rec.reasoning}")

        return recommendations

    def _evaluate_algorithm(self, algorithm: str, profile: DataProfile) -> AlgorithmRecommendation:
        """Evaluate a single algorithm for the given data profile.

        Args:
            algorithm: Algorithm name
            profile: Data profile

        Returns:
            Algorithm recommendation
        """
        metadata = self.algorithm_metadata[algorithm]

        # Calculate decision factors
        decision_factors = self._calculate_decision_factors(algorithm, profile, metadata)

        # Calculate overall suitability score
        suitability_score = self._calculate_suitability_score(decision_factors)

        # Calculate confidence based on suitability and constraints
        confidence = self._calculate_confidence(algorithm, profile, metadata, suitability_score)

        # Generate reasoning
        reasoning = self._generate_reasoning(algorithm, profile, metadata, decision_factors)

        # Recommend hyperparameters
        hyperparams = self._recommend_hyperparameters(algorithm, profile, metadata)

        # Estimate performance
        expected_performance = self._estimate_performance(algorithm, profile, metadata, suitability_score)

        return AlgorithmRecommendation(
            algorithm=algorithm,
            confidence=confidence,
            reasoning=reasoning,
            hyperparams=hyperparams,
            expected_performance=expected_performance,
            computational_complexity=metadata["complexity"],
            memory_requirements=self._estimate_memory_requirements(algorithm, profile, metadata),
            interpretability_score=metadata["interpretability"],
            suitability_score=suitability_score,
            decision_factors=decision_factors,
        )

    def _calculate_decision_factors(
        self, algorithm: str, profile: DataProfile, metadata: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate decision factors for algorithm evaluation.

        Args:
            algorithm: Algorithm name
            profile: Data profile
            metadata: Algorithm metadata

        Returns:
            Dictionary of decision factors
        """
        factors = {}

        # Sample size factor
        if profile.n_samples < metadata["min_samples"]:
            factors["sample_size"] = 0.1
        elif profile.n_samples > metadata.get("optimal_samples", 10000):
            factors["sample_size"] = 1.0
        else:
            factors["sample_size"] = profile.n_samples / metadata.get("optimal_samples", 10000)

        # Feature count factor
        if profile.n_features > metadata["max_features"]:
            factors["feature_count"] = 0.1
        else:
            factors["feature_count"] = 1.0 - (profile.n_features / metadata["max_features"])

        # Missing values factor
        if metadata["handles_missing"]:
            factors["missing_values"] = 1.0
        else:
            factors["missing_values"] = 1.0 - profile.missing_values_ratio

        # Categorical data factor
        if metadata["handles_categorical"]:
            factors["categorical_data"] = 1.0
        else:
            if profile.categorical_features > 0:
                factors["categorical_data"] = 0.2
            else:
                factors["categorical_data"] = 1.0

        # Correlation factor
        if "correlated_features" in metadata["best_for"]:
            factors["correlation"] = profile.correlation_score
        elif "independent_features" in metadata["best_for"]:
            factors["correlation"] = 1.0 - profile.correlation_score
        else:
            factors["correlation"] = 0.8

        # Complexity factor
        if profile.complexity_score > 0.7 and metadata["complexity"] in ["high", "very_high"]:
            factors["complexity_match"] = 1.0
        elif profile.complexity_score < 0.3 and metadata["complexity"] in ["low", "medium"]:
            factors["complexity_match"] = 1.0
        else:
            factors["complexity_match"] = 0.6

        # Scalability factor
        if profile.n_samples > 100000:
            scalability_scores = {
                "very_high": 1.0,
                "high": 0.8,
                "medium": 0.4,
                "low": 0.1,
            }
            factors["scalability"] = scalability_scores.get(metadata["scalability"], 0.5)
        else:
            factors["scalability"] = 1.0

        # Sparsity factor
        if profile.sparsity_ratio > 0.5:
            if "sparse_data" in metadata["worst_for"]:
                factors["sparsity"] = 0.3
            else:
                factors["sparsity"] = 0.7
        else:
            factors["sparsity"] = 1.0

        # Outlier ratio factor
        if profile.outlier_ratio_estimate > 0.2:
            if algorithm in ["IsolationForest", "LOF", "COPOD"]:
                factors["outlier_ratio"] = 1.0
            else:
                factors["outlier_ratio"] = 0.6
        else:
            factors["outlier_ratio"] = 1.0

        return factors

    def _calculate_suitability_score(self, decision_factors: dict[str, float]) -> float:
        """Calculate overall suitability score.

        Args:
            decision_factors: Dictionary of decision factors

        Returns:
            Suitability score between 0 and 1
        """
        # Weighted average of decision factors
        weights = {
            "sample_size": 0.2,
            "feature_count": 0.15,
            "missing_values": 0.1,
            "categorical_data": 0.1,
            "correlation": 0.1,
            "complexity_match": 0.15,
            "scalability": 0.1,
            "sparsity": 0.05,
            "outlier_ratio": 0.05,
        }

        total_score = 0.0
        total_weight = 0.0

        for factor, score in decision_factors.items():
            weight = weights.get(factor, 0.1)
            total_score += score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.5

    def _calculate_confidence(
        self, algorithm: str, profile: DataProfile, metadata: dict[str, Any], suitability_score: float
    ) -> float:
        """Calculate confidence score for algorithm recommendation.

        Args:
            algorithm: Algorithm name
            profile: Data profile
            metadata: Algorithm metadata
            suitability_score: Calculated suitability score

        Returns:
            Confidence score between 0 and 1
        """
        base_confidence = suitability_score

        # Adjust confidence based on hard constraints
        if profile.n_samples < metadata["min_samples"]:
            base_confidence *= 0.3

        if profile.n_features > metadata["max_features"]:
            base_confidence *= 0.2

        if not metadata["handles_missing"] and profile.missing_values_ratio > 0.1:
            base_confidence *= 0.7

        if not metadata["handles_categorical"] and profile.categorical_features > 0:
            base_confidence *= 0.5

        # Boost confidence for algorithms that match data characteristics
        if profile.complexity_score > 0.7 and metadata["complexity"] in ["high", "very_high"]:
            base_confidence *= 1.2

        if profile.n_samples > 50000 and metadata["scalability"] == "very_high":
            base_confidence *= 1.3

        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, base_confidence))

    def _generate_reasoning(
        self, algorithm: str, profile: DataProfile, metadata: dict[str, Any], decision_factors: dict[str, float]
    ) -> str:
        """Generate human-readable reasoning for algorithm recommendation.

        Args:
            algorithm: Algorithm name
            profile: Data profile
            metadata: Algorithm metadata
            decision_factors: Decision factors

        Returns:
            Human-readable reasoning string
        """
        reasons = []

        # Positive factors
        if decision_factors.get("sample_size", 0) > 0.8:
            reasons.append(f"Good fit for dataset size ({profile.n_samples:,} samples)")

        if decision_factors.get("scalability", 0) > 0.8:
            reasons.append("High scalability matches dataset size")

        if decision_factors.get("complexity_match", 0) > 0.8:
            reasons.append("Algorithm complexity matches data complexity")

        if metadata["handles_missing"] and profile.missing_values_ratio > 0.1:
            reasons.append(f"Handles missing values ({profile.missing_values_ratio:.1%})")

        if metadata["handles_categorical"] and profile.categorical_features > 0:
            reasons.append(f"Handles categorical features ({profile.categorical_features})")

        # Negative factors
        if decision_factors.get("sample_size", 0) < 0.3:
            reasons.append("Small dataset may limit performance")

        if decision_factors.get("feature_count", 0) < 0.3:
            reasons.append("High dimensionality may be challenging")

        if not metadata["handles_missing"] and profile.missing_values_ratio > 0.1:
            reasons.append("Requires preprocessing for missing values")

        # Algorithm-specific reasoning
        if algorithm == "IsolationForest":
            reasons.append("Excellent general-purpose anomaly detector")
        elif algorithm == "LOF":
            reasons.append("Effective for local outlier detection")
        elif algorithm == "COPOD":
            reasons.append("Fast and scalable for large datasets")
        elif algorithm == "HBOS":
            reasons.append("Very fast histogram-based detection")
        elif algorithm == "OneClassSVM":
            reasons.append("Effective for non-linear boundaries")

        if not reasons:
            reasons.append("Standard algorithm suitable for this dataset")

        return "; ".join(reasons)

    def _recommend_hyperparameters(
        self, algorithm: str, profile: DataProfile, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Recommend hyperparameters for the algorithm.

        Args:
            algorithm: Algorithm name
            profile: Data profile
            metadata: Algorithm metadata

        Returns:
            Dictionary of recommended hyperparameters
        """
        hyperparams = {}

        # Common contamination parameter
        hyperparams["contamination"] = profile.recommended_contamination

        # Algorithm-specific hyperparameters
        if algorithm == "IsolationForest":
            hyperparams.update({
                "n_estimators": min(200, max(100, profile.n_samples // 1000)),
                "max_samples": min(1.0, max(0.1, 10000 / profile.n_samples)),
                "max_features": min(1.0, max(0.1, 100 / profile.n_features)) if profile.n_features > 100 else 1.0,
                "random_state": 42,
            })

        elif algorithm == "LOF":
            hyperparams.update({
                "n_neighbors": min(50, max(5, profile.n_samples // 100)),
                "algorithm": "auto",
                "leaf_size": 30,
                "metric": "minkowski",
                "p": 2,
            })

        elif algorithm == "COPOD":
            hyperparams.update({
                "n_jobs": -1,
            })

        elif algorithm == "HBOS":
            hyperparams.update({
                "n_bins": min(50, max(10, profile.n_samples // 100)),
                "alpha": 0.1,
                "tol": 0.5,
            })

        elif algorithm == "OneClassSVM":
            hyperparams.update({
                "kernel": "rbf",
                "gamma": "scale",
                "nu": profile.recommended_contamination,
                "degree": 3,
                "coef0": 0.0,
            })

        elif algorithm == "PCA":
            hyperparams.update({
                "n_components": min(profile.n_features - 1, max(2, profile.n_features // 2)),
                "contamination": profile.recommended_contamination,
                "weighted": True,
                "standardization": True,
            })

        elif algorithm == "KNN":
            hyperparams.update({
                "n_neighbors": min(50, max(5, profile.n_samples // 100)),
                "method": "largest",
                "radius": 1.0,
                "algorithm": "auto",
                "leaf_size": 30,
                "metric": "minkowski",
                "p": 2,
            })

        elif algorithm == "AutoEncoder":
            hyperparams.update({
                "hidden_neurons": [
                    min(128, max(32, profile.n_features)),
                    min(64, max(16, profile.n_features // 2)),
                    min(32, max(8, profile.n_features // 4)),
                ],
                "epochs": min(200, max(50, profile.n_samples // 1000)),
                "batch_size": min(256, max(32, profile.n_samples // 100)),
                "learning_rate": 0.001,
                "preprocessing": True,
            })

        return hyperparams

    def _estimate_performance(
        self, algorithm: str, profile: DataProfile, metadata: dict[str, Any], suitability_score: float
    ) -> float:
        """Estimate expected performance for the algorithm.

        Args:
            algorithm: Algorithm name
            profile: Data profile
            metadata: Algorithm metadata
            suitability_score: Calculated suitability score

        Returns:
            Expected performance score between 0 and 1
        """
        # Base performance estimate from suitability score
        base_performance = suitability_score

        # Adjust based on algorithm characteristics
        if algorithm == "IsolationForest":
            base_performance *= 0.9  # Generally good performance
        elif algorithm == "LOF":
            base_performance *= 0.85  # Good for local outliers
        elif algorithm == "COPOD":
            base_performance *= 0.8  # Good for large datasets
        elif algorithm == "HBOS":
            base_performance *= 0.75  # Fast but may miss complex patterns
        elif algorithm == "OneClassSVM":
            base_performance *= 0.8  # Good for non-linear boundaries
        elif algorithm == "PCA":
            base_performance *= 0.7  # Good for linear patterns
        elif algorithm == "AutoEncoder":
            base_performance *= 0.85  # Good for complex patterns

        # Adjust based on dataset characteristics
        if profile.complexity_score > 0.7:
            if algorithm in ["AutoEncoder", "OneClassSVM"]:
                base_performance *= 1.1
            elif algorithm in ["HBOS", "PCA"]:
                base_performance *= 0.9

        if profile.n_samples > 100000:
            if algorithm in ["COPOD", "HBOS", "IsolationForest"]:
                base_performance *= 1.1
            elif algorithm in ["LOF", "OneClassSVM"]:
                base_performance *= 0.8

        # Ensure performance is between 0 and 1
        return max(0.0, min(1.0, base_performance))

    def _estimate_memory_requirements(
        self, algorithm: str, profile: DataProfile, metadata: dict[str, Any]
    ) -> str:
        """Estimate memory requirements for the algorithm.

        Args:
            algorithm: Algorithm name
            profile: Data profile
            metadata: Algorithm metadata

        Returns:
            Memory requirements description
        """
        # Base memory factor
        memory_factor = profile.n_samples * profile.n_features * 8  # bytes

        # Algorithm-specific multipliers
        multipliers = {
            "IsolationForest": 2.0,
            "LOF": 3.0,
            "COPOD": 1.5,
            "HBOS": 1.2,
            "OneClassSVM": 4.0,
            "PCA": 2.0,
            "KNN": 2.5,
            "AutoEncoder": 5.0,
        }

        estimated_memory = memory_factor * multipliers.get(algorithm, 2.0)

        # Convert to human-readable format
        if estimated_memory < 1e6:
            return f"~{estimated_memory/1e3:.0f} KB"
        elif estimated_memory < 1e9:
            return f"~{estimated_memory/1e6:.0f} MB"
        else:
            return f"~{estimated_memory/1e9:.1f} GB"

    def get_algorithm_alternatives(self, algorithm: str) -> list[str]:
        """Get alternative algorithms for the given algorithm.

        Args:
            algorithm: Algorithm name

        Returns:
            List of alternative algorithm names
        """
        alternatives_map = {
            "IsolationForest": ["COPOD", "HBOS", "LOF"],
            "LOF": ["KNN", "OneClassSVM", "IsolationForest"],
            "COPOD": ["HBOS", "IsolationForest", "PCA"],
            "HBOS": ["COPOD", "IsolationForest", "PCA"],
            "OneClassSVM": ["LOF", "KNN", "AutoEncoder"],
            "PCA": ["COPOD", "HBOS", "IsolationForest"],
            "KNN": ["LOF", "OneClassSVM", "IsolationForest"],
            "AutoEncoder": ["OneClassSVM", "IsolationForest", "PCA"],
        }

        return alternatives_map.get(algorithm, [])

    def explain_recommendation(self, recommendation: AlgorithmRecommendation) -> str:
        """Generate detailed explanation for a recommendation.

        Args:
            recommendation: Algorithm recommendation

        Returns:
            Detailed explanation string
        """
        explanation_parts = [
            f"Algorithm: {recommendation.algorithm}",
            f"Confidence: {recommendation.confidence:.3f}",
            f"Expected Performance: {recommendation.expected_performance:.3f}",
            f"Interpretability: {recommendation.interpretability_score:.3f}",
            f"Computational Complexity: {recommendation.computational_complexity}",
            f"Memory Requirements: {recommendation.memory_requirements}",
            f"Reasoning: {recommendation.reasoning}",
        ]

        if recommendation.decision_factors:
            explanation_parts.append("Decision Factors:")
            for factor, score in recommendation.decision_factors.items():
                explanation_parts.append(f"  {factor}: {score:.3f}")

        if recommendation.hyperparams:
            explanation_parts.append("Recommended Hyperparameters:")
            for param, value in recommendation.hyperparams.items():
                explanation_parts.append(f"  {param}: {value}")

        return "\n".join(explanation_parts)
