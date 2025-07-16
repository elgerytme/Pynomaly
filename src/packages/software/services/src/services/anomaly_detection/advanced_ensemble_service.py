"""Advanced ensemble methods and meta-learning service.

This service provides sophisticated ensemble capabilities including:
- Dynamic ensemble selection with intelligent algorithm combination
- Meta-learning framework for cross-dataset knowledge transfer
- Automated ensemble optimization with diversity analysis
- Advanced voting strategies with learned weights
- Cross-domain knowledge transfer and adaptation
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

# Application layer imports
# Core domain imports
from monorepo.domain.entities import Dataset, Detector

# Infrastructure imports - handle optional dependencies
try:
    from sklearn.metrics import accuracy_score, roc_auc_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Suppress warnings during optimization
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class EnsembleStrategy(BaseModel):
    """Ensemble combination strategy configuration."""

    name: str = Field(description="Strategy name")
    description: str = Field(description="Strategy description")
    requires_training: bool = Field(
        default=False, description="Whether strategy requires training"
    )
    supports_weights: bool = Field(
        default=True, description="Whether strategy supports weights"
    )
    complexity: str = Field(default="medium", description="Computational complexity")
    interpretability: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Interpretability score"
    )


class DiversityMetrics(BaseModel):
    """Diversity analysis metrics for ensemble validation."""

    disagreement_measure: float = Field(
        description="Average disagreement between classifiers"
    )
    double_fault_measure: float = Field(description="Double fault diversity measure")
    q_statistic: float = Field(description="Q-statistic correlation measure")
    correlation_coefficient: float = Field(description="Correlation coefficient")
    kappa_statistic: float = Field(description="Kappa inter-rater agreement")
    entropy_measure: float = Field(description="Entropy-based diversity measure")
    overall_diversity: float = Field(description="Overall diversity score")


class MetaLearningKnowledge(BaseModel):
    """Meta-learning knowledge representation."""

    dataset_characteristics: dict[str, Any] = Field(
        description="Dataset characteristics"
    )
    algorithm_performance: dict[str, float] = Field(
        description="Algorithm performance mapping"
    )
    ensemble_composition: list[str] = Field(description="Optimal ensemble composition")
    optimal_weights: dict[str, float] = Field(description="Optimal ensemble weights")
    diversity_requirements: dict[str, float] = Field(
        description="Diversity requirements"
    )
    performance_metrics: dict[str, float] = Field(description="Performance metrics")
    confidence_score: float = Field(description="Confidence in recommendation")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Knowledge creation time"
    )


class EnsembleConfiguration(BaseModel):
    """Configuration for ensemble creation and optimization."""

    base_algorithms: list[str] = Field(description="Base algorithms for ensemble")
    ensemble_strategy: str = Field(
        default="voting", description="Ensemble combination strategy"
    )
    max_ensemble_size: int = Field(
        default=5, ge=2, le=10, description="Maximum ensemble size"
    )
    min_diversity_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Minimum diversity threshold"
    )
    weight_optimization: bool = Field(
        default=True, description="Enable weight optimization"
    )
    diversity_weighting: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Diversity vs performance trade-off"
    )
    cross_validation_folds: int = Field(
        default=3, ge=2, le=10, description="CV folds for validation"
    )
    meta_learning_enabled: bool = Field(
        default=True, description="Enable meta-learning"
    )


class AdvancedEnsembleService:
    """Advanced ensemble methods and meta-learning service."""

    def __init__(
        self,
        meta_knowledge_path: Path = Path("./meta_knowledge"),
        enable_meta_learning: bool = True,
        diversity_threshold: float = 0.3,
    ):
        """Initialize advanced ensemble service.

        Args:
            meta_knowledge_path: Path for storing meta-learning knowledge
            enable_meta_learning: Enable meta-learning capabilities
            diversity_threshold: Minimum diversity threshold for ensembles
        """
        self.meta_knowledge_path = meta_knowledge_path
        self.meta_knowledge_path.mkdir(parents=True, exist_ok=True)

        self.enable_meta_learning = enable_meta_learning
        self.diversity_threshold = diversity_threshold

        # Meta-learning knowledge base
        self.meta_knowledge: list[MetaLearningKnowledge] = []
        self.load_meta_knowledge()

        # Ensemble strategies
        self.ensemble_strategies = self._initialize_ensemble_strategies()

        # Algorithm compatibility matrix
        self.algorithm_compatibility = self._initialize_algorithm_compatibility()

    def _initialize_ensemble_strategies(self) -> dict[str, EnsembleStrategy]:
        """Initialize available ensemble strategies."""
        return {
            "voting": EnsembleStrategy(
                name="voting",
                description="Simple majority/average voting",
                requires_training=False,
                supports_weights=True,
                complexity="low",
                interpretability=0.9,
            ),
            "weighted_voting": EnsembleStrategy(
                name="weighted_voting",
                description="Performance-weighted voting",
                requires_training=True,
                supports_weights=True,
                complexity="medium",
                interpretability=0.8,
            ),
            "stacking": EnsembleStrategy(
                name="stacking",
                description="Meta-learner stacking",
                requires_training=True,
                supports_weights=False,
                complexity="high",
                interpretability=0.4,
            ),
            "dynamic_selection": EnsembleStrategy(
                name="dynamic_selection",
                description="Dynamic algorithm selection per sample",
                requires_training=True,
                supports_weights=True,
                complexity="high",
                interpretability=0.6,
            ),
            "adaptive_boosting": EnsembleStrategy(
                name="adaptive_boosting",
                description="Adaptive boosting with reweighting",
                requires_training=True,
                supports_weights=True,
                complexity="high",
                interpretability=0.5,
            ),
            "diversity_optimized": EnsembleStrategy(
                name="diversity_optimized",
                description="Diversity-optimized combination",
                requires_training=True,
                supports_weights=True,
                complexity="medium",
                interpretability=0.7,
            ),
        }

    def _initialize_algorithm_compatibility(self) -> dict[str, dict[str, float]]:
        """Initialize algorithm compatibility matrix."""
        return {
            "IsolationForest": {
                "LocalOutlierFactor": 0.8,
                "OneClassSVM": 0.7,
                "EllipticEnvelope": 0.6,
                "ABOD": 0.9,
                "KNN": 0.8,
                "COPOD": 0.7,
            },
            "LocalOutlierFactor": {
                "IsolationForest": 0.8,
                "OneClassSVM": 0.6,
                "EllipticEnvelope": 0.5,
                "ABOD": 0.7,
                "KNN": 0.9,
                "COPOD": 0.6,
            },
            "OneClassSVM": {
                "IsolationForest": 0.7,
                "LocalOutlierFactor": 0.6,
                "EllipticEnvelope": 0.8,
                "ABOD": 0.5,
                "KNN": 0.6,
                "COPOD": 0.5,
            },
        }

    async def create_intelligent_ensemble(
        self,
        dataset: Dataset,
        algorithms: list[str] | None = None,
        config: EnsembleConfiguration | None = None,
    ) -> tuple[list[Detector], dict[str, Any]]:
        """Create an intelligent ensemble using meta-learning and optimization.

        Args:
            dataset: Dataset for ensemble creation
            algorithms: List of algorithms to consider (None for auto-selection)
            config: Ensemble configuration

        Returns:
            Tuple of (ensemble_detectors, ensemble_report)
        """
        config = config or EnsembleConfiguration()

        logger.info("Creating intelligent ensemble with meta-learning")

        # Step 1: Analyze dataset characteristics
        dataset_chars = self._analyze_dataset_for_ensemble(dataset)

        # Step 2: Apply meta-learning for algorithm selection
        if self.enable_meta_learning and algorithms is None:
            algorithms = await self._meta_learning_algorithm_selection(
                dataset_chars, config
            )
        else:
            algorithms = algorithms or [
                "IsolationForest",
                "LocalOutlierFactor",
                "OneClassSVM",
            ]

        # Step 3: Create and train base detectors
        base_detectors = await self._create_base_detectors(dataset, algorithms, config)

        # Step 4: Evaluate individual detector performance
        individual_performance = await self._evaluate_individual_detectors(
            base_detectors, dataset, config
        )

        # Step 5: Analyze ensemble diversity
        diversity_metrics = await self._analyze_ensemble_diversity(
            base_detectors, dataset
        )

        # Step 6: Optimize ensemble composition
        optimized_ensemble = await self._optimize_ensemble_composition(
            base_detectors, individual_performance, diversity_metrics, config
        )

        # Step 7: Learn optimal ensemble weights
        ensemble_weights = await self._learn_ensemble_weights(
            optimized_ensemble, dataset, config
        )

        # Step 8: Store meta-learning knowledge
        if self.enable_meta_learning:
            await self._store_meta_knowledge(
                dataset_chars,
                optimized_ensemble,
                ensemble_weights,
                individual_performance,
                diversity_metrics,
            )

        # Step 9: Generate ensemble report
        ensemble_report = self._generate_ensemble_report(
            optimized_ensemble,
            individual_performance,
            diversity_metrics,
            ensemble_weights,
            dataset_chars,
            config,
        )

        logger.info(f"Created ensemble with {len(optimized_ensemble)} detectors")

        return optimized_ensemble, ensemble_report

    def _analyze_dataset_for_ensemble(self, dataset: Dataset) -> dict[str, Any]:
        """Analyze dataset characteristics for ensemble optimization."""
        data = dataset.data
        n_samples, n_features = data.shape

        characteristics = {
            "n_samples": n_samples,
            "n_features": n_features,
            "size_ratio": n_samples / n_features if n_features > 0 else 0,
            "feature_density": self._calculate_feature_density(data),
            "data_complexity": self._estimate_data_complexity(data),
            "noise_level": self._estimate_noise_level(data),
            "cluster_structure": self._analyze_cluster_structure(data),
            "anomaly_likelihood": self._estimate_anomaly_likelihood(data),
            "dimensionality_category": self._categorize_dimensionality(n_features),
            "sample_efficiency": self._calculate_sample_efficiency(
                n_samples, n_features
            ),
        }

        return characteristics

    def _calculate_feature_density(self, data: np.ndarray) -> float:
        """Calculate feature density (information content per feature)."""
        try:
            # Calculate variance per feature
            feature_variances = np.var(data, axis=0)
            # Normalize by mean variance
            mean_variance = np.mean(feature_variances)
            if mean_variance > 0:
                return np.mean(feature_variances / mean_variance)
            return 1.0
        except Exception:
            return 1.0

    def _estimate_data_complexity(self, data: np.ndarray) -> float:
        """Estimate intrinsic data complexity."""
        try:
            # Use PCA to estimate intrinsic dimensionality
            from sklearn.decomposition import PCA

            # Limit samples for efficiency
            sample_data = data[: min(1000, len(data))]

            pca = PCA()
            pca.fit(sample_data)

            # Calculate explained variance ratio
            explained_variance = pca.explained_variance_ratio_

            # Find number of components explaining 90% variance
            cumsum = np.cumsum(explained_variance)
            intrinsic_dim = np.argmax(cumsum >= 0.9) + 1

            # Complexity as ratio of intrinsic to actual dimensions
            return intrinsic_dim / data.shape[1] if data.shape[1] > 0 else 1.0

        except Exception:
            # Fallback to simple complexity estimate
            return min(1.0, data.shape[1] / 10.0)

    def _estimate_noise_level(self, data: np.ndarray) -> float:
        """Estimate noise level in the data."""
        try:
            # Use nearest neighbor distance distribution
            from sklearn.neighbors import NearestNeighbors

            # Sample for efficiency
            sample_data = data[: min(500, len(data))]

            # Find k-nearest neighbors
            k = min(5, len(sample_data) - 1)
            nn = NearestNeighbors(n_neighbors=k + 1)
            nn.fit(sample_data)

            distances, _ = nn.kneighbors(sample_data)
            # Use distances to nearest neighbor (excluding self)
            nearest_distances = distances[:, 1]

            # Noise level as coefficient of variation of distances
            mean_dist = np.mean(nearest_distances)
            std_dist = np.std(nearest_distances)

            if mean_dist > 0:
                noise_level = std_dist / mean_dist
            else:
                noise_level = 0.0

            # Normalize to [0, 1]
            return min(1.0, noise_level)

        except Exception:
            return 0.5  # Default moderate noise level

    def _analyze_cluster_structure(self, data: np.ndarray) -> dict[str, float]:
        """Analyze cluster structure in the data."""
        try:
            # Use silhouette analysis for cluster structure
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score

            # Sample for efficiency
            sample_data = data[: min(500, len(data))]

            if len(sample_data) < 10:
                return {"cluster_tendency": 0.5, "separability": 0.5}

            # Try different numbers of clusters
            max_clusters = min(10, len(sample_data) // 2)
            silhouette_scores = []

            for k in range(2, max_clusters + 1):
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(sample_data)
                    score = silhouette_score(sample_data, cluster_labels)
                    silhouette_scores.append(score)
                except Exception:
                    continue

            if silhouette_scores:
                cluster_tendency = max(silhouette_scores)
                separability = np.mean(silhouette_scores)
            else:
                cluster_tendency = 0.5
                separability = 0.5

            return {
                "cluster_tendency": max(0.0, cluster_tendency),
                "separability": max(0.0, separability),
            }

        except Exception:
            return {"cluster_tendency": 0.5, "separability": 0.5}

    def _estimate_anomaly_likelihood(self, data: np.ndarray) -> float:
        """Estimate likelihood of anomalies in the dataset."""
        try:
            # Use statistical measures to estimate anomaly likelihood

            # Calculate z-scores for all points
            mean_vals = np.mean(data, axis=0)
            std_vals = np.std(data, axis=0)

            # Avoid division by zero
            std_vals = np.where(std_vals == 0, 1, std_vals)

            z_scores = np.abs((data - mean_vals) / std_vals)
            max_z_scores = np.max(z_scores, axis=1)

            # Count points with high z-scores (potential anomalies)
            outlier_threshold = 3.0  # 3-sigma rule
            potential_outliers = np.sum(max_z_scores > outlier_threshold)

            # Estimate anomaly likelihood
            anomaly_likelihood = potential_outliers / len(data)

            # Cap at reasonable maximum
            return min(0.5, anomaly_likelihood)

        except Exception:
            return 0.1  # Default low anomaly likelihood

    def _categorize_dimensionality(self, n_features: int) -> str:
        """Categorize dataset dimensionality."""
        if n_features <= 5:
            return "low"
        elif n_features <= 20:
            return "medium"
        elif n_features <= 100:
            return "high"
        else:
            return "very_high"

    def _calculate_sample_efficiency(self, n_samples: int, n_features: int) -> float:
        """Calculate sample efficiency ratio."""
        if n_features == 0:
            return 1.0

        ratio = n_samples / n_features

        # Ideal ratio is around 10:1 or higher
        if ratio >= 10:
            return 1.0
        elif ratio >= 5:
            return 0.8
        elif ratio >= 2:
            return 0.6
        else:
            return 0.3

    async def _meta_learning_algorithm_selection(
        self, dataset_chars: dict[str, Any], config: EnsembleConfiguration
    ) -> list[str]:
        """Select algorithms using meta-learning knowledge."""
        if not self.meta_knowledge:
            # Default algorithms if no meta-knowledge
            return ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]

        # Find similar datasets in meta-knowledge
        similar_cases = []

        for knowledge in self.meta_knowledge:
            similarity = self._calculate_dataset_similarity(
                dataset_chars, knowledge.dataset_characteristics
            )

            if similarity > 0.6:  # Similarity threshold
                similar_cases.append((knowledge, similarity))

        if not similar_cases:
            # No similar cases found, use default
            return ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]

        # Weight algorithms by similarity and performance
        algorithm_scores = {}

        for knowledge, similarity in similar_cases:
            confidence_weight = knowledge.confidence_score * similarity

            for algorithm, performance in knowledge.algorithm_performance.items():
                if algorithm not in algorithm_scores:
                    algorithm_scores[algorithm] = 0

                algorithm_scores[algorithm] += performance * confidence_weight

        # Select top algorithms
        sorted_algorithms = sorted(
            algorithm_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Return top algorithms up to max ensemble size
        selected = [alg for alg, _ in sorted_algorithms[: config.max_ensemble_size]]

        # Ensure minimum diversity
        selected = self._ensure_algorithm_diversity(selected, config)

        logger.info(f"Meta-learning selected algorithms: {selected}")

        return selected

    def _calculate_dataset_similarity(
        self, chars1: dict[str, Any], chars2: dict[str, Any]
    ) -> float:
        """Calculate similarity between dataset characteristics."""
        # Key characteristics for similarity
        key_features = [
            "n_samples",
            "n_features",
            "size_ratio",
            "feature_density",
            "data_complexity",
            "noise_level",
        ]

        similarities = []

        for feature in key_features:
            val1 = chars1.get(feature, 0)
            val2 = chars2.get(feature, 0)

            if isinstance(val1, int | float) and isinstance(val2, int | float):
                # Normalize similarity for numeric values
                max_val = max(abs(val1), abs(val2), 1e-8)
                similarity = 1.0 - abs(val1 - val2) / max_val
                similarities.append(similarity)

        # Handle nested characteristics
        if "cluster_structure" in chars1 and "cluster_structure" in chars2:
            cluster1 = chars1["cluster_structure"]
            cluster2 = chars2["cluster_structure"]

            if isinstance(cluster1, dict) and isinstance(cluster2, dict):
                cluster_sim = 1.0 - abs(
                    cluster1.get("cluster_tendency", 0)
                    - cluster2.get("cluster_tendency", 0)
                )
                similarities.append(cluster_sim)

        return np.mean(similarities) if similarities else 0.0

    def _ensure_algorithm_diversity(
        self, algorithms: list[str], config: EnsembleConfiguration
    ) -> list[str]:
        """Ensure selected algorithms have sufficient diversity."""
        if len(algorithms) <= 1:
            return algorithms

        # Check pairwise compatibility
        diverse_algorithms = [algorithms[0]]  # Start with first algorithm

        for candidate in algorithms[1:]:
            is_diverse = True

            for selected in diverse_algorithms:
                compatibility = self.algorithm_compatibility.get(selected, {}).get(
                    candidate, 0.5
                )

                # High compatibility means low diversity
                if compatibility > 0.8:
                    is_diverse = False
                    break

            if is_diverse and len(diverse_algorithms) < config.max_ensemble_size:
                diverse_algorithms.append(candidate)

        # Ensure minimum size
        if len(diverse_algorithms) < 2:
            # Add default diverse algorithms
            defaults = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]
            for default in defaults:
                if default not in diverse_algorithms:
                    diverse_algorithms.append(default)
                    if len(diverse_algorithms) >= 2:
                        break

        return diverse_algorithms[: config.max_ensemble_size]

    async def _create_base_detectors(
        self, dataset: Dataset, algorithms: list[str], config: EnsembleConfiguration
    ) -> list[Detector]:
        """Create and train base detectors."""
        detectors = []

        for algorithm in algorithms:
            try:
                # Create detector using existing adapter system
                from monorepo.infrastructure.adapters import SklearnAdapter

                detector = SklearnAdapter(algorithm)
                detector.fit(dataset)
                detectors.append(detector)

                logger.debug(f"Created detector: {algorithm}")

            except Exception as e:
                logger.warning(f"Failed to create detector {algorithm}: {e}")
                continue

        if not detectors:
            # Fallback to default detector
            from monorepo.infrastructure.adapters import SklearnAdapter

            default_detector = SklearnAdapter("IsolationForest")
            default_detector.fit(dataset)
            detectors = [default_detector]

        return detectors

    async def _evaluate_individual_detectors(
        self, detectors: list[Detector], dataset: Dataset, config: EnsembleConfiguration
    ) -> dict[str, dict[str, float]]:
        """Evaluate individual detector performance."""
        performance = {}

        for i, detector in enumerate(detectors):
            detector_name = f"detector_{i}"

            try:
                # Generate predictions
                scores = detector.predict(dataset)

                # Create synthetic evaluation data
                eval_data = self._create_evaluation_dataset(dataset)
                eval_scores = detector.predict(eval_data)

                # Calculate performance metrics
                if "labels" in eval_data.metadata:
                    labels = eval_data.metadata["labels"]

                    # Calculate various metrics
                    metrics = self._calculate_performance_metrics(eval_scores, labels)
                else:
                    # Fallback metrics without ground truth
                    metrics = {
                        "anomaly_ratio": np.mean(scores > np.percentile(scores, 90)),
                        "score_variance": np.var(scores),
                        "score_range": np.max(scores) - np.min(scores),
                        "estimated_accuracy": 0.7,  # Default estimate
                    }

                performance[detector_name] = metrics

            except Exception as e:
                logger.warning(f"Failed to evaluate detector {detector_name}: {e}")
                performance[detector_name] = {"estimated_accuracy": 0.5}

        return performance

    def _create_evaluation_dataset(self, dataset: Dataset) -> Dataset:
        """Create evaluation dataset with synthetic anomalies."""
        # Use portion of original data as normal
        normal_data = dataset.data[: int(len(dataset.data) * 0.8)]

        # Generate synthetic anomalies
        n_anomalies = max(1, int(len(normal_data) * 0.1))
        anomalies = self._generate_synthetic_anomalies(normal_data, n_anomalies)

        # Combine data
        eval_data = np.vstack([normal_data, anomalies])
        eval_labels = np.hstack([np.zeros(len(normal_data)), np.ones(len(anomalies))])

        return Dataset(
            name=f"{dataset.name}_eval",
            data=eval_data,
            features=dataset.features,
            metadata={"labels": eval_labels},
        )

    def _generate_synthetic_anomalies(
        self, normal_data: np.ndarray, n_anomalies: int
    ) -> np.ndarray:
        """Generate synthetic anomalies for evaluation."""
        anomalies = []

        for _ in range(n_anomalies):
            # Method: Add significant noise to random normal sample
            base_sample = normal_data[np.random.randint(len(normal_data))]
            noise_scale = np.std(normal_data, axis=0) * 3  # 3-sigma noise
            noise = np.random.normal(0, noise_scale, base_sample.shape)
            anomaly = base_sample + noise
            anomalies.append(anomaly)

        return np.array(anomalies)

    def _calculate_performance_metrics(
        self, scores: np.ndarray, labels: np.ndarray
    ) -> dict[str, float]:
        """Calculate performance metrics for detector evaluation."""
        try:
            metrics = {}

            if SKLEARN_AVAILABLE and len(np.unique(labels)) > 1:
                # ROC AUC
                metrics["roc_auc"] = roc_auc_score(labels, scores)

                # Accuracy with threshold
                threshold = np.percentile(scores, 90)
                predictions = (scores > threshold).astype(int)
                metrics["accuracy"] = accuracy_score(labels, predictions)

                # Precision and recall
                true_positives = np.sum((predictions == 1) & (labels == 1))
                false_positives = np.sum((predictions == 1) & (labels == 0))
                false_negatives = np.sum((predictions == 0) & (labels == 1))

                precision = (
                    true_positives / (true_positives + false_positives)
                    if (true_positives + false_positives) > 0
                    else 0
                )
                recall = (
                    true_positives / (true_positives + false_negatives)
                    if (true_positives + false_negatives) > 0
                    else 0
                )

                metrics["precision"] = precision
                metrics["recall"] = recall
                metrics["f1_score"] = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )

            else:
                # Fallback metrics
                threshold = np.percentile(scores, 90)
                predictions = (scores > threshold).astype(int)
                accuracy = np.mean(predictions == labels) if len(labels) > 0 else 0.5
                metrics["estimated_accuracy"] = accuracy

            # Additional metrics
            metrics["anomaly_ratio"] = np.mean(scores > np.percentile(scores, 90))
            metrics["score_variance"] = np.var(scores)

            return metrics

        except Exception as e:
            logger.warning(f"Failed to calculate metrics: {e}")
            return {"estimated_accuracy": 0.5, "anomaly_ratio": 0.1}

    async def _analyze_ensemble_diversity(
        self, detectors: list[Detector], dataset: Dataset
    ) -> DiversityMetrics:
        """Analyze diversity among ensemble detectors."""
        if len(detectors) < 2:
            # No diversity with single detector
            return DiversityMetrics(
                disagreement_measure=0.0,
                double_fault_measure=0.0,
                q_statistic=0.0,
                correlation_coefficient=0.0,
                kappa_statistic=0.0,
                entropy_measure=0.0,
                overall_diversity=0.0,
            )

        # Get predictions from all detectors
        predictions_matrix = []

        for detector in detectors:
            scores = detector.predict(dataset)
            # Convert to binary predictions
            threshold = np.percentile(scores, 90)
            binary_predictions = (scores > threshold).astype(int)
            predictions_matrix.append(binary_predictions)

        predictions_matrix = np.array(predictions_matrix)

        # Calculate diversity metrics
        diversity_metrics = self._calculate_diversity_metrics(predictions_matrix)

        return diversity_metrics

    def _calculate_diversity_metrics(
        self, predictions_matrix: np.ndarray
    ) -> DiversityMetrics:
        """Calculate comprehensive diversity metrics."""
        n_classifiers, n_samples = predictions_matrix.shape

        # Disagreement measure
        disagreement = 0.0
        correlation_sum = 0.0
        q_statistics = []
        double_faults = []

        for i in range(n_classifiers):
            for j in range(i + 1, n_classifiers):
                pred_i = predictions_matrix[i]
                pred_j = predictions_matrix[j]

                # Disagreement
                disagreement += np.mean(pred_i != pred_j)

                # Correlation
                if SCIPY_AVAILABLE:
                    corr, _ = stats.pearsonr(pred_i, pred_j)
                    correlation_sum += abs(corr) if not np.isnan(corr) else 0

                # Q-statistic
                n11 = np.sum((pred_i == 1) & (pred_j == 1))
                n10 = np.sum((pred_i == 1) & (pred_j == 0))
                n01 = np.sum((pred_i == 0) & (pred_j == 1))
                n00 = np.sum((pred_i == 0) & (pred_j == 0))

                if (n11 * n00 + n01 * n10) != 0:
                    q_stat = (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10)
                    q_statistics.append(abs(q_stat))

                # Double fault
                double_fault = np.mean((pred_i == 0) & (pred_j == 0))
                double_faults.append(double_fault)

        n_pairs = n_classifiers * (n_classifiers - 1) / 2

        # Average metrics
        avg_disagreement = disagreement / n_pairs if n_pairs > 0 else 0
        avg_correlation = correlation_sum / n_pairs if n_pairs > 0 else 0
        avg_q_statistic = np.mean(q_statistics) if q_statistics else 0
        avg_double_fault = np.mean(double_faults) if double_faults else 0

        # Kappa statistic (inter-rater agreement)
        kappa = self._calculate_fleiss_kappa(predictions_matrix)

        # Entropy measure
        entropy = self._calculate_entropy_diversity(predictions_matrix)

        # Overall diversity score
        overall_diversity = (
            0.3 * avg_disagreement
            + 0.2 * (1 - avg_correlation)
            + 0.2 * (1 - avg_q_statistic)
            + 0.1 * (1 - avg_double_fault)
            + 0.1 * (1 - kappa)
            + 0.1 * entropy
        )

        return DiversityMetrics(
            disagreement_measure=avg_disagreement,
            double_fault_measure=avg_double_fault,
            q_statistic=avg_q_statistic,
            correlation_coefficient=avg_correlation,
            kappa_statistic=kappa,
            entropy_measure=entropy,
            overall_diversity=overall_diversity,
        )

    def _calculate_fleiss_kappa(self, predictions_matrix: np.ndarray) -> float:
        """Calculate Fleiss' kappa for inter-rater agreement."""
        try:
            n_classifiers, n_samples = predictions_matrix.shape

            # Count agreements for each sample
            agreements = []
            for i in range(n_samples):
                sample_predictions = predictions_matrix[:, i]
                # Count how many agree on positive class
                positive_votes = np.sum(sample_predictions == 1)
                negative_votes = np.sum(sample_predictions == 0)

                # Agreement score for this sample
                max_agreement = max(positive_votes, negative_votes)
                agreement = (
                    (max_agreement - 1) / (n_classifiers - 1)
                    if n_classifiers > 1
                    else 1
                )
                agreements.append(agreement)

            return np.mean(agreements)

        except Exception:
            return 0.5  # Default moderate agreement

    def _calculate_entropy_diversity(self, predictions_matrix: np.ndarray) -> float:
        """Calculate entropy-based diversity measure."""
        try:
            n_classifiers, n_samples = predictions_matrix.shape

            entropies = []
            for i in range(n_samples):
                sample_predictions = predictions_matrix[:, i]

                # Count votes
                positive_votes = np.sum(sample_predictions == 1)
                negative_votes = np.sum(sample_predictions == 0)

                # Calculate entropy
                if positive_votes > 0 and negative_votes > 0:
                    p_pos = positive_votes / n_classifiers
                    p_neg = negative_votes / n_classifiers
                    entropy = -p_pos * np.log2(p_pos) - p_neg * np.log2(p_neg)
                else:
                    entropy = 0  # No diversity

                entropies.append(entropy)

            # Normalize by maximum possible entropy
            max_entropy = 1.0  # log2(2) for binary classification
            return np.mean(entropies) / max_entropy if max_entropy > 0 else 0

        except Exception:
            return 0.5  # Default moderate entropy

    async def _optimize_ensemble_composition(
        self,
        base_detectors: list[Detector],
        individual_performance: dict[str, dict[str, float]],
        diversity_metrics: DiversityMetrics,
        config: EnsembleConfiguration,
    ) -> list[Detector]:
        """Optimize ensemble composition using performance and diversity."""
        if len(base_detectors) <= config.max_ensemble_size:
            return base_detectors

        # Calculate combined scores for each detector
        detector_scores = []

        for i, detector in enumerate(base_detectors):
            detector_name = f"detector_{i}"
            performance = individual_performance.get(detector_name, {})

            # Performance score
            perf_score = performance.get(
                "roc_auc",
                performance.get("accuracy", performance.get("estimated_accuracy", 0.5)),
            )

            # Diversity contribution (simplified)
            diversity_score = diversity_metrics.overall_diversity

            # Combined score
            combined_score = (
                1 - config.diversity_weighting
            ) * perf_score + config.diversity_weighting * diversity_score

            detector_scores.append((detector, combined_score))

        # Sort by combined score
        detector_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top detectors
        optimized_ensemble = [
            detector for detector, _ in detector_scores[: config.max_ensemble_size]
        ]

        return optimized_ensemble

    async def _learn_ensemble_weights(
        self,
        ensemble_detectors: list[Detector],
        dataset: Dataset,
        config: EnsembleConfiguration,
    ) -> dict[str, float]:
        """Learn optimal ensemble weights."""
        if not config.weight_optimization or len(ensemble_detectors) <= 1:
            # Equal weights
            weight = 1.0 / len(ensemble_detectors)
            return {f"detector_{i}": weight for i in range(len(ensemble_detectors))}

        # Create evaluation dataset
        eval_dataset = self._create_evaluation_dataset(dataset)

        # Get predictions from all detectors
        predictions = []
        for detector in ensemble_detectors:
            scores = detector.predict(eval_dataset)
            predictions.append(scores)

        predictions = np.array(predictions)

        # Learn weights using cross-validation
        weights = await self._optimize_weights_cv(predictions, eval_dataset, config)

        return {f"detector_{i}": weight for i, weight in enumerate(weights)}

    async def _optimize_weights_cv(
        self,
        predictions: np.ndarray,
        eval_dataset: Dataset,
        config: EnsembleConfiguration,
    ) -> np.ndarray:
        """Optimize ensemble weights using cross-validation."""
        try:
            from scipy.optimize import minimize

            n_detectors = predictions.shape[0]
            labels = eval_dataset.metadata.get("labels", np.zeros(predictions.shape[1]))

            def objective_function(weights):
                # Ensure weights sum to 1
                weights = weights / np.sum(weights)

                # Weighted ensemble prediction
                ensemble_pred = np.dot(weights, predictions)

                # Calculate loss (negative AUC)
                try:
                    if SKLEARN_AVAILABLE and len(np.unique(labels)) > 1:
                        auc = roc_auc_score(labels, ensemble_pred)
                        return -auc
                    else:
                        # Fallback: minimize variance while maximizing separation
                        threshold = np.percentile(ensemble_pred, 90)
                        binary_pred = (ensemble_pred > threshold).astype(int)
                        accuracy = (
                            np.mean(binary_pred == labels) if len(labels) > 0 else 0.5
                        )
                        return -accuracy
                except Exception:
                    return 1.0  # High loss for invalid predictions

            # Initialize with equal weights
            initial_weights = np.ones(n_detectors) / n_detectors

            # Constraints: weights sum to 1 and are non-negative
            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
            bounds = [(0, 1) for _ in range(n_detectors)]

            # Optimize
            result = minimize(
                objective_function,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            if result.success:
                weights = result.x
                # Ensure weights sum to 1
                weights = weights / np.sum(weights)
                return weights
            else:
                # Fallback to equal weights
                return initial_weights

        except Exception as e:
            logger.warning(f"Weight optimization failed: {e}")
            # Return equal weights
            n_detectors = predictions.shape[0]
            return np.ones(n_detectors) / n_detectors

    async def _store_meta_knowledge(
        self,
        dataset_chars: dict[str, Any],
        ensemble_detectors: list[Detector],
        ensemble_weights: dict[str, float],
        individual_performance: dict[str, dict[str, float]],
        diversity_metrics: DiversityMetrics,
    ):
        """Store meta-learning knowledge for future use."""
        try:
            # Extract algorithm names (simplified)
            algorithm_names = [f"algorithm_{i}" for i in range(len(ensemble_detectors))]

            # Calculate overall ensemble performance
            overall_performance = np.mean(
                [
                    perf.get("roc_auc", perf.get("estimated_accuracy", 0.5))
                    for perf in individual_performance.values()
                ]
            )

            # Create meta-learning knowledge
            knowledge = MetaLearningKnowledge(
                dataset_characteristics=dataset_chars,
                algorithm_performance={
                    name: individual_performance.get(f"detector_{i}", {}).get(
                        "roc_auc", 0.5
                    )
                    for i, name in enumerate(algorithm_names)
                },
                ensemble_composition=algorithm_names,
                optimal_weights=ensemble_weights,
                diversity_requirements={
                    "min_diversity": diversity_metrics.overall_diversity,
                    "target_disagreement": diversity_metrics.disagreement_measure,
                },
                performance_metrics={
                    "ensemble_performance": overall_performance,
                    "diversity_score": diversity_metrics.overall_diversity,
                },
                confidence_score=min(
                    1.0, overall_performance + diversity_metrics.overall_diversity * 0.3
                ),
            )

            # Add to knowledge base
            self.meta_knowledge.append(knowledge)

            # Limit knowledge base size
            if len(self.meta_knowledge) > 100:
                # Keep most recent and high-confidence knowledge
                self.meta_knowledge.sort(
                    key=lambda k: (k.confidence_score, k.timestamp), reverse=True
                )
                self.meta_knowledge = self.meta_knowledge[:100]

            # Save to disk
            self.save_meta_knowledge()

            logger.debug("Stored meta-learning knowledge")

        except Exception as e:
            logger.warning(f"Failed to store meta-knowledge: {e}")

    def save_meta_knowledge(self):
        """Save meta-knowledge to disk."""
        try:
            knowledge_file = self.meta_knowledge_path / "meta_knowledge.json"

            knowledge_data = [
                {**knowledge.dict(), "timestamp": knowledge.timestamp.isoformat()}
                for knowledge in self.meta_knowledge
            ]

            with open(knowledge_file, "w") as f:
                json.dump(knowledge_data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save meta-knowledge: {e}")

    def load_meta_knowledge(self):
        """Load meta-knowledge from disk."""
        try:
            knowledge_file = self.meta_knowledge_path / "meta_knowledge.json"

            if not knowledge_file.exists():
                return

            with open(knowledge_file) as f:
                knowledge_data = json.load(f)

            self.meta_knowledge = []
            for item in knowledge_data:
                # Convert timestamp back to datetime
                if "timestamp" in item:
                    item["timestamp"] = datetime.fromisoformat(item["timestamp"])

                knowledge = MetaLearningKnowledge(**item)
                self.meta_knowledge.append(knowledge)

        except Exception as e:
            logger.error(f"Failed to load meta-knowledge: {e}")
            self.meta_knowledge = []

    def _generate_ensemble_report(
        self,
        ensemble_detectors: list[Detector],
        individual_performance: dict[str, dict[str, float]],
        diversity_metrics: DiversityMetrics,
        ensemble_weights: dict[str, float],
        dataset_chars: dict[str, Any],
        config: EnsembleConfiguration,
    ) -> dict[str, Any]:
        """Generate comprehensive ensemble report."""
        report = {
            "ensemble_summary": {
                "n_detectors": len(ensemble_detectors),
                "strategy": config.ensemble_strategy,
                "creation_timestamp": datetime.now().isoformat(),
                "meta_learning_enabled": self.enable_meta_learning,
            },
            "dataset_characteristics": dataset_chars,
            "individual_performance": individual_performance,
            "diversity_analysis": diversity_metrics.dict(),
            "ensemble_weights": ensemble_weights,
            "configuration": config.dict(),
            "performance_summary": self._calculate_ensemble_performance_summary(
                individual_performance, diversity_metrics, ensemble_weights
            ),
            "recommendations": self._generate_ensemble_recommendations(
                ensemble_detectors, diversity_metrics, config
            ),
            "meta_learning_insights": self._generate_meta_learning_insights(),
        }

        return report

    def _calculate_ensemble_performance_summary(
        self,
        individual_performance: dict[str, dict[str, float]],
        diversity_metrics: DiversityMetrics,
        ensemble_weights: dict[str, float],
    ) -> dict[str, float]:
        """Calculate ensemble performance summary."""
        # Weighted average performance
        weighted_performance = 0.0
        total_weight = 0.0

        for detector_name, weight in ensemble_weights.items():
            perf = individual_performance.get(detector_name, {})
            detector_perf = perf.get("roc_auc", perf.get("estimated_accuracy", 0.5))
            weighted_performance += weight * detector_perf
            total_weight += weight

        if total_weight > 0:
            weighted_performance /= total_weight

        # Diversity bonus
        diversity_bonus = diversity_metrics.overall_diversity * 0.1

        # Estimated ensemble performance
        estimated_ensemble_performance = min(
            1.0, weighted_performance + diversity_bonus
        )

        return {
            "weighted_individual_performance": weighted_performance,
            "diversity_score": diversity_metrics.overall_diversity,
            "estimated_ensemble_performance": estimated_ensemble_performance,
            "performance_improvement": max(
                0, estimated_ensemble_performance - weighted_performance
            ),
            "confidence_score": min(
                1.0, diversity_metrics.overall_diversity + weighted_performance * 0.5
            ),
        }

    def _generate_ensemble_recommendations(
        self,
        ensemble_detectors: list[Detector],
        diversity_metrics: DiversityMetrics,
        config: EnsembleConfiguration,
    ) -> list[str]:
        """Generate recommendations for ensemble improvement."""
        recommendations = []

        # Diversity recommendations
        if diversity_metrics.overall_diversity < 0.3:
            recommendations.append(
                "Consider adding more diverse algorithms to improve ensemble diversity"
            )

        if diversity_metrics.correlation_coefficient > 0.8:
            recommendations.append(
                "High correlation detected between detectors - consider removing similar algorithms"
            )

        # Size recommendations
        if len(ensemble_detectors) < 3:
            recommendations.append(
                "Consider adding more detectors to the ensemble for better robustness"
            )
        elif len(ensemble_detectors) > 7:
            recommendations.append(
                "Large ensemble detected - consider pruning to reduce computational cost"
            )

        # Strategy recommendations
        if (
            config.ensemble_strategy == "voting"
            and diversity_metrics.overall_diversity > 0.7
        ):
            recommendations.append(
                "High diversity detected - consider weighted voting or stacking for better performance"
            )

        # Meta-learning recommendations
        if self.enable_meta_learning and len(self.meta_knowledge) < 10:
            recommendations.append(
                "Build more meta-learning knowledge by creating ensembles on diverse datasets"
            )

        if not recommendations:
            recommendations.append("Ensemble configuration appears well-optimized")

        return recommendations

    def _generate_meta_learning_insights(self) -> dict[str, Any]:
        """Generate insights from meta-learning knowledge."""
        if not self.meta_knowledge:
            return {"message": "No meta-learning knowledge available"}

        # Analyze knowledge base
        algorithm_popularity = {}
        performance_trends = {}

        for knowledge in self.meta_knowledge:
            # Track algorithm usage
            for algorithm in knowledge.ensemble_composition:
                algorithm_popularity[algorithm] = (
                    algorithm_popularity.get(algorithm, 0) + 1
                )

            # Track performance trends
            knowledge.performance_metrics.get("ensemble_performance", 0.5)
            for algorithm, perf in knowledge.algorithm_performance.items():
                if algorithm not in performance_trends:
                    performance_trends[algorithm] = []
                performance_trends[algorithm].append(perf)

        # Generate insights
        insights = {
            "knowledge_base_size": len(self.meta_knowledge),
            "most_popular_algorithms": sorted(
                algorithm_popularity.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "average_performance_by_algorithm": {
                alg: np.mean(perfs) for alg, perfs in performance_trends.items()
            },
            "learning_confidence": np.mean(
                [k.confidence_score for k in self.meta_knowledge]
            ),
            "diversity_patterns": {
                "avg_diversity_requirement": np.mean(
                    [
                        k.diversity_requirements.get("min_diversity", 0.3)
                        for k in self.meta_knowledge
                    ]
                )
            },
        }

        return insights

    async def predict_ensemble_performance(
        self, dataset_chars: dict[str, Any], proposed_algorithms: list[str]
    ) -> dict[str, Any]:
        """Predict ensemble performance using meta-learning."""
        if not self.meta_knowledge:
            return {
                "predicted_performance": 0.7,
                "confidence": 0.3,
                "message": "No meta-learning knowledge available for prediction",
            }

        # Find similar cases
        similar_cases = []
        for knowledge in self.meta_knowledge:
            similarity = self._calculate_dataset_similarity(
                dataset_chars, knowledge.dataset_characteristics
            )
            if similarity > 0.5:
                similar_cases.append((knowledge, similarity))

        if not similar_cases:
            return {
                "predicted_performance": 0.7,
                "confidence": 0.2,
                "message": "No similar cases found in meta-knowledge",
            }

        # Weight predictions by similarity
        predicted_performance = 0.0
        total_weight = 0.0

        for knowledge, similarity in similar_cases:
            weight = similarity * knowledge.confidence_score
            performance = knowledge.performance_metrics.get("ensemble_performance", 0.5)

            predicted_performance += weight * performance
            total_weight += weight

        if total_weight > 0:
            predicted_performance /= total_weight

        # Calculate confidence
        confidence = (
            min(1.0, total_weight / len(similar_cases)) if similar_cases else 0.0
        )

        return {
            "predicted_performance": predicted_performance,
            "confidence": confidence,
            "similar_cases_found": len(similar_cases),
            "recommendation_strength": (
                "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"
            ),
        }
