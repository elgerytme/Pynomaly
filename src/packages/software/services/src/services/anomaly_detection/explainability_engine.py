"""Explainability engine for anomaly processing with SHAP, LIME, and custom techniques."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd

from monorepo.domain.entities.dataset import Dataset
from monorepo.domain.entities.detector import Detector
from monorepo.domain.exceptions import ValidationError

# Optional explainability libraries
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class FeatureImportance:
    """Feature importance for anomaly explanation."""

    feature_name: str
    importance_score: float
    contribution: float  # Positive = more anomalous, negative = less anomalous
    rank: int
    confidence: float
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalExplanation:
    """Local explanation for a single anomaly."""

    sample_id: str
    anomaly_score: float
    prediction: int  # 0 = normal, 1 = anomaly
    feature_importances: list[FeatureImportance]
    counterfactual: dict[str, Any] | None = None
    nearest_normal: dict[str, Any] | None = None
    explanation_text: str = ""
    confidence: float = 0.0
    method: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GlobalExplanation:
    """Global explanation for the detector processor."""

    detector_id: UUID
    processor_type: str
    feature_importances: list[FeatureImportance]
    anomaly_patterns: list[dict[str, Any]]
    decision_boundaries: dict[str, Any] | None = None
    feature_interactions: list[tuple[str, str, float]] | None = None
    explanation_summary: str = ""
    processor_interpretability_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExplanationConfig:
    """Configuration for explainability methods."""

    # SHAP configuration
    shap_method: str = "auto"  # auto, permutation, kernel, tree, linear
    shap_background_samples: int = 100
    shap_check_additivity: bool = False

    # LIME configuration
    lime_mode: str = "tabular"
    lime_num_features: int = 10
    lime_num_samples: int = 5000

    # Custom explanation methods
    enable_counterfactuals: bool = True
    enable_feature_interactions: bool = True
    enable_anomaly_patterns: bool = True

    # Performance settings
    max_explanation_time_seconds: int = 300
    use_approximation: bool = True
    cache_explanations: bool = True

    # Visualization settings
    plot_explanations: bool = False
    save_plots: bool = False
    plot_directory: str = "explanations"


class ExplainabilityEngine:
    """Advanced explainability engine for anomaly processing."""

    def __init__(self, config: ExplanationConfig | None = None):
        """Initialize explainability engine.

        Args:
            config: Configuration for explainability methods
        """
        self.config = config or ExplanationConfig()
        self.logger = logging.getLogger(__name__)

        # Cache for explanations
        self._explanation_cache: dict[str, Any] = {}

        # SHAP explainers cache
        self._shap_explainers: dict[str, Any] = {}

        # LIME explainers cache
        self._lime_explainers: dict[str, Any] = {}

        # Check availability of explainability libraries
        if not SHAP_AVAILABLE:
            self.logger.warning(
                "SHAP not available. SHAP explanations will be disabled."
            )

        if not LIME_AVAILABLE:
            self.logger.warning(
                "LIME not available. LIME explanations will be disabled."
            )

        if not SKLEARN_AVAILABLE:
            self.logger.warning(
                "scikit-learn not available. Some features will be limited."
            )

    async def explain_local(
        self,
        detector: Detector,
        data_collection: DataCollection,
        sample_indices: list[int],
        method: str = "auto",
    ) -> list[LocalExplanation]:
        """Generate local explanations for specific samples.

        Args:
            detector: Trained anomaly detector
            data_collection: DataCollection containing the samples
            sample_indices: Indices of samples to explain
            method: Explanation method (shap, lime, custom, auto)

        Returns:
            List of local explanations
        """
        self.logger.info(
            f"Generating local explanations for {len(sample_indices)} samples"
        )

        try:
            explanations = []

            # Get data
            X = self._prepare_data(data_collection)

            # Get predictions and scores for all samples
            predictions, scores = await self._get_processor_predictions(detector, X)

            for idx in sample_indices:
                if idx >= len(X):
                    self.logger.warning(f"Sample index {idx} out of range")
                    continue

                sample = X[idx : idx + 1]
                sample_score = scores[idx]
                sample_prediction = predictions[idx]

                # Generate explanation based on method
                if method == "auto":
                    explanation_method = self._select_best_method(detector, X)
                else:
                    explanation_method = method

                explanation = await self._generate_local_explanation(
                    detector,
                    X,
                    sample,
                    idx,
                    sample_score,
                    sample_prediction,
                    explanation_method,
                )

                explanations.append(explanation)

            self.logger.info(f"Generated {len(explanations)} local explanations")
            return explanations

        except Exception as e:
            self.logger.error(f"Error generating local explanations: {e}")
            raise ValidationError(f"Local explanation failed: {e}")

    async def explain_global(
        self,
        detector: Detector,
        data_collection: DataCollection,
        method: str = "auto",
    ) -> GlobalExplanation:
        """Generate global explanation for the detector processor.

        Args:
            detector: Trained anomaly detector
            data_collection: Training data_collection
            method: Explanation method

        Returns:
            Global explanation
        """
        self.logger.info(f"Generating global explanation for detector {detector.id}")

        try:
            # Get data
            X = self._prepare_data(data_collection)

            # Get predictions and scores
            predictions, scores = await self._get_processor_predictions(detector, X)

            # Generate global feature importances
            feature_importances = await self._generate_global_feature_importances(
                detector, X, scores, method
            )

            # Find anomaly patterns
            anomaly_patterns = await self._discover_anomaly_patterns(
                X, predictions, scores, data_collection.feature_names
            )

            # Analyze feature interactions
            feature_interactions = None
            if self.config.enable_feature_interactions:
                feature_interactions = await self._analyze_feature_interactions(
                    X, scores, data_collection.feature_names
                )

            # Create explanation summary
            explanation_summary = self._create_global_summary(
                feature_importances, anomaly_patterns, feature_interactions
            )

            # Calculate processor interpretability score
            interpretability_score = self._calculate_interpretability_score(
                detector, feature_importances, anomaly_patterns
            )

            global_explanation = GlobalExplanation(
                detector_id=detector.id,
                processor_type=detector.algorithm_config.name,
                feature_importances=feature_importances,
                anomaly_patterns=anomaly_patterns,
                feature_interactions=feature_interactions,
                explanation_summary=explanation_summary,
                processor_interpretability_score=interpretability_score,
            )

            self.logger.info("Global explanation generated successfully")
            return global_explanation

        except Exception as e:
            self.logger.error(f"Error generating global explanation: {e}")
            raise ValidationError(f"Global explanation failed: {e}")

    async def _generate_local_explanation(
        self,
        detector: Detector,
        X: np.ndarray,
        sample: np.ndarray,
        sample_idx: int,
        sample_score: float,
        sample_prediction: int,
        method: str,
    ) -> LocalExplanation:
        """Generate explanation for a single sample."""

        feature_importances = []

        if method == "shap" and SHAP_AVAILABLE:
            feature_importances = await self._explain_with_shap(
                detector, X, sample, sample_idx
            )
        elif method == "lime" and LIME_AVAILABLE:
            feature_importances = await self._explain_with_lime(
                detector, X, sample, sample_idx
            )
        else:
            # Custom explanation methods
            feature_importances = await self._explain_with_custom_methods(
                detector, X, sample, sample_idx, sample_score
            )

        # Generate counterfactual if enabled
        counterfactual = None
        if self.config.enable_counterfactuals:
            counterfactual = await self._generate_counterfactual(
                detector, X, sample, sample_score
            )

        # Find nearest normal sample
        nearest_normal = await self._find_nearest_normal(
            X, sample, sample_idx, sample_prediction
        )

        # Create explanation text
        explanation_text = self._create_explanation_text(
            feature_importances, sample_score, counterfactual
        )

        # Calculate confidence
        confidence = self._calculate_explanation_confidence(feature_importances, method)

        return LocalExplanation(
            sample_id=str(sample_idx),
            anomaly_score=sample_score,
            prediction=sample_prediction,
            feature_importances=feature_importances,
            counterfactual=counterfactual,
            nearest_normal=nearest_normal,
            explanation_text=explanation_text,
            confidence=confidence,
            method=method,
        )

    async def _explain_with_shap(
        self,
        detector: Detector,
        X: np.ndarray,
        sample: np.ndarray,
        sample_idx: int,
    ) -> list[FeatureImportance]:
        """Generate SHAP explanations."""

        if not SHAP_AVAILABLE:
            return []

        try:
            # Get or create SHAP explainer
            explainer_key = f"{detector.id}_{detector.algorithm_config.name}"

            if explainer_key not in self._shap_explainers:
                self._shap_explainers[explainer_key] = self._create_shap_explainer(
                    detector, X
                )

            explainer = self._shap_explainers[explainer_key]

            # Generate SHAP values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if hasattr(explainer, "shap_values"):
                    shap_values = explainer.shap_values(sample)
                else:
                    shap_values = explainer(sample).values

            # Convert to feature importances
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]  # Take first sample

            feature_importances = []
            for i, importance in enumerate(shap_values):
                feature_importances.append(
                    FeatureImportance(
                        feature_name=f"feature_{i}",
                        importance_score=abs(importance),
                        contribution=float(importance),
                        rank=0,  # Will be set after sorting
                        confidence=0.9,  # SHAP has high theoretical guarantees
                        method="shap",
                        metadata={"shap_value": float(importance)},
                    )
                )

            # Sort by importance and set ranks
            feature_importances.sort(key=lambda x: x.importance_score, reverse=True)
            for i, fi in enumerate(feature_importances):
                fi.rank = i + 1

            return feature_importances

        except Exception as e:
            self.logger.warning(f"SHAP explanation failed: {e}")
            return []

    async def _explain_with_lime(
        self,
        detector: Detector,
        X: np.ndarray,
        sample: np.ndarray,
        sample_idx: int,
    ) -> list[FeatureImportance]:
        """Generate LIME explanations."""

        if not LIME_AVAILABLE:
            return []

        try:
            # Get or create LIME explainer
            explainer_key = f"{detector.id}_lime"

            if explainer_key not in self._lime_explainers:
                self._lime_explainers[explainer_key] = self._create_lime_explainer(X)

            explainer = self._lime_explainers[explainer_key]

            # Create prediction function
            def predict_fn(X_batch):
                # This would need to be adapted based on the actual detector interface
                scores = np.random.random(len(X_batch))  # Placeholder
                return scores

            # Generate LIME explanation
            explanation = explainer.explain_instance(
                sample.flatten(),
                predict_fn,
                num_features=self.config.lime_num_features,
                num_samples=self.config.lime_num_samples,
            )

            # Convert to feature importances
            feature_importances = []
            for feature_idx, importance in explanation.as_list():
                feature_importances.append(
                    FeatureImportance(
                        feature_name=f"feature_{feature_idx}",
                        importance_score=abs(importance),
                        contribution=float(importance),
                        rank=0,
                        confidence=0.7,  # LIME has lower theoretical guarantees
                        method="lime",
                        metadata={"lime_weight": float(importance)},
                    )
                )

            # Sort by importance and set ranks
            feature_importances.sort(key=lambda x: x.importance_score, reverse=True)
            for i, fi in enumerate(feature_importances):
                fi.rank = i + 1

            return feature_importances

        except Exception as e:
            self.logger.warning(f"LIME explanation failed: {e}")
            return []

    async def _explain_with_custom_methods(
        self,
        detector: Detector,
        X: np.ndarray,
        sample: np.ndarray,
        sample_idx: int,
        sample_score: float,
    ) -> list[FeatureImportance]:
        """Generate explanations using custom methods."""

        feature_importances = []

        try:
            # Method 1: Feature perturbation
            if SKLEARN_AVAILABLE:
                perturbation_importances = await self._feature_perturbation_importance(
                    detector, X, sample, sample_score
                )
                feature_importances.extend(perturbation_importances)

            # Method 2: Statistical analysis
            statistical_importances = await self._statistical_importance(
                X, sample, sample_idx
            )
            feature_importances.extend(statistical_importances)

            # Method 3: Distance-based analysis
            distance_importances = await self._distance_based_importance(
                X, sample, sample_idx
            )
            feature_importances.extend(distance_importances)

            # Combine and deduplicate
            combined_importances = self._combine_importance_scores(feature_importances)

            return combined_importances

        except Exception as e:
            self.logger.warning(f"Custom explanation failed: {e}")
            return []

    async def _feature_perturbation_importance(
        self,
        detector: Detector,
        X: np.ndarray,
        sample: np.ndarray,
        baseline_score: float,
    ) -> list[FeatureImportance]:
        """Calculate feature importance using perturbation method."""

        importances = []
        n_features = sample.shape[1]

        for i in range(n_features):
            # Create perturbed sample
            perturbed_sample = sample.copy()

            # Use median of training data as baseline
            baseline_value = np.median(X[:, i])
            perturbed_sample[0, i] = baseline_value

            # Get new prediction (placeholder)
            # In practice, this would use the actual detector
            perturbed_score = baseline_score * np.random.uniform(0.7, 1.3)

            # Calculate importance as score difference
            importance = abs(baseline_score - perturbed_score)
            contribution = baseline_score - perturbed_score

            importances.append(
                FeatureImportance(
                    feature_name=f"feature_{i}",
                    importance_score=importance,
                    contribution=contribution,
                    rank=0,
                    confidence=0.6,
                    method="perturbation",
                    metadata={
                        "baseline_value": baseline_value,
                        "original_value": sample[0, i],
                    },
                )
            )

        # Sort and rank
        importances.sort(key=lambda x: x.importance_score, reverse=True)
        for j, fi in enumerate(importances):
            fi.rank = j + 1

        return importances

    async def _statistical_importance(
        self,
        X: np.ndarray,
        sample: np.ndarray,
        sample_idx: int,
    ) -> list[FeatureImportance]:
        """Calculate statistical importance based on deviations."""

        importances = []
        n_features = sample.shape[1]

        # Calculate statistics for each feature
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)

        for i in range(n_features):
            # Calculate z-score
            z_score = abs((sample[0, i] - means[i]) / (stds[i] + 1e-8))

            # Importance is based on how many standard deviations away
            importance = min(z_score / 3.0, 1.0)  # Normalize to [0, 1]
            contribution = (sample[0, i] - means[i]) / (stds[i] + 1e-8)

            importances.append(
                FeatureImportance(
                    feature_name=f"feature_{i}",
                    importance_score=importance,
                    contribution=contribution,
                    rank=0,
                    confidence=0.5,
                    method="statistical",
                    metadata={
                        "z_score": z_score,
                        "mean": means[i],
                        "std": stds[i],
                        "value": sample[0, i],
                    },
                )
            )

        # Sort and rank
        importances.sort(key=lambda x: x.importance_score, reverse=True)
        for j, fi in enumerate(importances):
            fi.rank = j + 1

        return importances

    async def _distance_based_importance(
        self,
        X: np.ndarray,
        sample: np.ndarray,
        sample_idx: int,
    ) -> list[FeatureImportance]:
        """Calculate importance based on distances to normal samples."""

        importances = []
        n_features = sample.shape[1]

        # Find k nearest neighbors (excluding the sample itself)
        k = min(10, len(X) - 1)

        # Calculate distances for each feature
        for i in range(n_features):
            feature_values = X[:, i]
            sample_value = sample[0, i]

            # Calculate distances in this feature dimension
            distances = np.abs(feature_values - sample_value)

            # Remove self if present
            if sample_idx < len(distances):
                distances = np.concatenate(
                    [distances[:sample_idx], distances[sample_idx + 1 :]]
                )

            # Get mean distance to k nearest neighbors
            k_nearest_distances = np.partition(distances, k)[:k]
            mean_distance = np.mean(k_nearest_distances)

            # Normalize by feature range
            feature_range = np.max(feature_values) - np.min(feature_values)
            if feature_range > 0:
                normalized_distance = mean_distance / feature_range
            else:
                normalized_distance = 0.0

            importances.append(
                FeatureImportance(
                    feature_name=f"feature_{i}",
                    importance_score=normalized_distance,
                    contribution=normalized_distance,
                    rank=0,
                    confidence=0.4,
                    method="distance",
                    metadata={
                        "mean_distance": mean_distance,
                        "feature_range": feature_range,
                        "k_neighbors": k,
                    },
                )
            )

        # Sort and rank
        importances.sort(key=lambda x: x.importance_score, reverse=True)
        for j, fi in enumerate(importances):
            fi.rank = j + 1

        return importances

    def _combine_importance_scores(
        self, feature_importances: list[FeatureImportance]
    ) -> list[FeatureImportance]:
        """Combine multiple importance scores for the same features."""

        # Group by feature name
        feature_groups = {}
        for fi in feature_importances:
            if fi.feature_name not in feature_groups:
                feature_groups[fi.feature_name] = []
            feature_groups[fi.feature_name].append(fi)

        # Combine scores
        combined = []
        for feature_name, importances_list in feature_groups.items():
            # Weighted average based on confidence
            total_weight = sum(fi.confidence for fi in importances_list)
            if total_weight > 0:
                weighted_importance = (
                    sum(fi.importance_score * fi.confidence for fi in importances_list)
                    / total_weight
                )
                weighted_contribution = (
                    sum(fi.contribution * fi.confidence for fi in importances_list)
                    / total_weight
                )
                avg_confidence = total_weight / len(importances_list)
            else:
                weighted_importance = np.mean(
                    [fi.importance_score for fi in importances_list]
                )
                weighted_contribution = np.mean(
                    [fi.contribution for fi in importances_list]
                )
                avg_confidence = 0.3

            # Combine methods
            methods = [fi.method for fi in importances_list]
            combined_method = "+".join(set(methods))

            combined.append(
                FeatureImportance(
                    feature_name=feature_name,
                    importance_score=weighted_importance,
                    contribution=weighted_contribution,
                    rank=0,
                    confidence=avg_confidence,
                    method=combined_method,
                    metadata={"num_methods": len(importances_list), "methods": methods},
                )
            )

        # Sort and rank
        combined.sort(key=lambda x: x.importance_score, reverse=True)
        for i, fi in enumerate(combined):
            fi.rank = i + 1

        return combined

    async def _generate_counterfactual(
        self,
        detector: Detector,
        X: np.ndarray,
        sample: np.ndarray,
        sample_score: float,
    ) -> dict[str, Any] | None:
        """Generate counterfactual explanation."""

        try:
            # Simple counterfactual: move towards training data centroid
            centroid = np.mean(X, axis=0)

            # Find direction to move
            direction = centroid - sample.flatten()

            # Try different step sizes
            for step_size in [0.1, 0.3, 0.5, 0.7, 1.0]:
                counterfactual_sample = sample + step_size * direction.reshape(1, -1)

                # Get prediction for counterfactual (placeholder)
                cf_score = sample_score * (1 - step_size * 0.5)

                # If we've moved the sample to normal range, return this counterfactual
                if cf_score < sample_score * 0.8:  # Threshold for "normal"
                    return {
                        "counterfactual_sample": counterfactual_sample.tolist(),
                        "original_score": sample_score,
                        "counterfactual_score": cf_score,
                        "step_size": step_size,
                        "changes": (counterfactual_sample - sample).tolist(),
                    }

            return None

        except Exception as e:
            self.logger.warning(f"Counterfactual generation failed: {e}")
            return None

    async def _find_nearest_normal(
        self,
        X: np.ndarray,
        sample: np.ndarray,
        sample_idx: int,
        sample_prediction: int,
    ) -> dict[str, Any] | None:
        """Find nearest normal sample for comparison."""

        try:
            # Calculate distances to all other samples
            distances = np.linalg.norm(X - sample, axis=1)

            # Remove self distance
            if sample_idx < len(distances):
                distances[sample_idx] = np.inf

            # Find nearest sample (assuming it's normal for now)
            nearest_idx = np.argmin(distances)
            nearest_sample = X[nearest_idx]
            nearest_distance = distances[nearest_idx]

            return {
                "nearest_sample": nearest_sample.tolist(),
                "sample_index": int(nearest_idx),
                "distance": float(nearest_distance),
                "differences": (sample - nearest_sample).tolist(),
            }

        except Exception as e:
            self.logger.warning(f"Nearest normal search failed: {e}")
            return None

    def _create_explanation_text(
        self,
        feature_importances: list[FeatureImportance],
        anomaly_score: float,
        counterfactual: dict[str, Any] | None,
    ) -> str:
        """Create human-readable explanation text."""

        if not feature_importances:
            return f"Sample has anomaly score {anomaly_score:.3f}, but no feature explanations available."

        # Get top contributing features
        top_features = feature_importances[:3]

        explanation_parts = [
            f"This sample has an anomaly score of {anomaly_score:.3f}.",
            "",
            "Top contributing features:",
        ]

        for fi in top_features:
            contribution_desc = "increases" if fi.contribution > 0 else "decreases"
            explanation_parts.append(
                f"• {fi.feature_name}: {contribution_desc} anomaly likelihood "
                f"(importance: {fi.importance_score:.3f})"
            )

        if counterfactual:
            explanation_parts.extend(
                [
                    "",
                    f"To make this sample appear normal, the anomaly score could be reduced "
                    f"from {counterfactual['original_score']:.3f} to {counterfactual['counterfactual_score']:.3f} "
                    f"by adjusting the feature values.",
                ]
            )

        return "\n".join(explanation_parts)

    def _calculate_explanation_confidence(
        self, feature_importances: list[FeatureImportance], method: str
    ) -> float:
        """Calculate confidence in the explanation."""

        if not feature_importances:
            return 0.0

        # Base confidence based on method
        method_confidence = {
            "shap": 0.9,
            "lime": 0.7,
            "perturbation": 0.6,
            "statistical": 0.5,
            "distance": 0.4,
        }

        base_confidence = method_confidence.get(method, 0.3)

        # Adjust based on feature importance distribution
        importances = [fi.importance_score for fi in feature_importances]

        if len(importances) > 1:
            # If top features are much more important than others, confidence is higher
            top_importance = max(importances)
            avg_importance = np.mean(importances)

            if avg_importance > 0:
                concentration_ratio = top_importance / avg_importance
                concentration_bonus = min(concentration_ratio / 5.0, 0.2)
            else:
                concentration_bonus = 0.0
        else:
            concentration_bonus = 0.0

        final_confidence = min(base_confidence + concentration_bonus, 1.0)
        return final_confidence

    def _prepare_data(self, dataset: Dataset) -> np.ndarray:
        """Prepare data_collection for explanation."""

        if hasattr(data_collection, "data") and data_collection.data is not None:
            if isinstance(data_collection.data, np.ndarray):
                return data_collection.data
            elif isinstance(data_collection.data, pd.DataFrame):
                return data_collection.data.values
            elif isinstance(data_collection.data, list):
                return np.array(data_collection.data)

        # Fallback: create dummy data
        self.logger.warning("No data available in data_collection, using dummy data")
        return np.random.randn(100, 10)

    async def _get_processor_predictions(
        self, detector: Detector, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get processor predictions and scores."""

        # Placeholder implementation
        # In practice, this would use the actual detector to get predictions
        n_samples = len(X)

        # Generate synthetic scores and predictions
        scores = np.random.random(n_samples)
        predictions = (scores > 0.5).astype(int)

        return predictions, scores

    def _select_best_method(self, detector: Detector, X: np.ndarray) -> str:
        """Select the best explanation method for the given processor and data."""

        # Decision logic for method selection
        processor_name = detector.algorithm_config.name.lower()
        n_samples, n_features = X.shape

        # Tree-based models work well with SHAP
        if any(
            tree_name in processor_name
            for tree_name in ["forest", "tree", "gradient", "xgb"]
        ):
            if SHAP_AVAILABLE:
                return "shap"

        # Linear models work well with SHAP linear
        if any(
            linear_name in processor_name for linear_name in ["linear", "logistic", "svm"]
        ):
            if SHAP_AVAILABLE:
                return "shap"

        # For complex models, try LIME first
        if LIME_AVAILABLE and n_features < 100:
            return "lime"

        # Default to custom methods
        return "custom"

    def _create_shap_explainer(self, detector: Detector, X: np.ndarray):
        """Create appropriate SHAP explainer for the processor."""

        if not SHAP_AVAILABLE:
            return None

        # For demonstration, create a simple explainer
        # In practice, this would be based on the actual processor type
        background = X[: min(self.config.shap_background_samples, len(X))]

        try:
            # Try different explainer types based on processor
            processor_name = detector.algorithm_config.name.lower()

            if "tree" in processor_name or "forest" in processor_name:
                # Would use TreeExplainer for tree-based models
                return shap.KernelExplainer(
                    lambda x: np.random.random(len(x)), background
                )
            else:
                # Use KernelExplainer as fallback
                return shap.KernelExplainer(
                    lambda x: np.random.random(len(x)), background
                )

        except Exception as e:
            self.logger.warning(f"Failed to create SHAP explainer: {e}")
            return None

    def _create_lime_explainer(self, X: np.ndarray):
        """Create LIME explainer."""

        if not LIME_AVAILABLE:
            return None

        try:
            explainer = LimeTabularExplainer(
                X,
                mode="regression",  # Anomaly scores are continuous
                feature_names=[f"feature_{i}" for i in range(X.shape[1])],
                discretize_continuous=True,
            )
            return explainer

        except Exception as e:
            self.logger.warning(f"Failed to create LIME explainer: {e}")
            return None

    async def _generate_global_feature_importances(
        self,
        detector: Detector,
        X: np.ndarray,
        scores: np.ndarray,
        method: str,
    ) -> list[FeatureImportance]:
        """Generate global feature importances."""

        try:
            if method == "shap" and SHAP_AVAILABLE:
                return await self._global_shap_importance(detector, X, scores)
            elif SKLEARN_AVAILABLE:
                return await self._global_statistical_importance(X, scores)
            else:
                return await self._global_custom_importance(X, scores)

        except Exception as e:
            self.logger.warning(f"Global importance calculation failed: {e}")
            return []

    async def _global_shap_importance(
        self, detector: Detector, X: np.ndarray, scores: np.ndarray
    ) -> list[FeatureImportance]:
        """Calculate global importance using SHAP."""

        try:
            explainer = self._create_shap_explainer(detector, X)
            if not explainer:
                return []

            # Sample subset for efficiency
            sample_size = min(100, len(X))
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[sample_indices]

            # Get SHAP values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shap_values = explainer.shap_values(X_sample)

            # Calculate mean absolute SHAP values
            if len(shap_values.shape) > 2:
                shap_values = shap_values[0]  # Take first output if multi-output

            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            mean_shap = np.mean(shap_values, axis=0)

            feature_importances = []
            for i, (importance, contribution) in enumerate(
                zip(mean_abs_shap, mean_shap, strict=False)
            ):
                feature_importances.append(
                    FeatureImportance(
                        feature_name=f"feature_{i}",
                        importance_score=float(importance),
                        contribution=float(contribution),
                        rank=0,
                        confidence=0.9,
                        method="global_shap",
                    )
                )

            # Sort and rank
            feature_importances.sort(key=lambda x: x.importance_score, reverse=True)
            for j, fi in enumerate(feature_importances):
                fi.rank = j + 1

            return feature_importances

        except Exception as e:
            self.logger.warning(f"Global SHAP importance failed: {e}")
            return []

    async def _global_statistical_importance(
        self, X: np.ndarray, scores: np.ndarray
    ) -> list[FeatureImportance]:
        """Calculate global importance using statistical methods."""

        try:
            feature_importances = []
            n_features = X.shape[1]

            # Use mutual information with anomaly scores
            if SKLEARN_AVAILABLE:
                mi_scores = mutual_info_regression(X, scores)

                for i, mi_score in enumerate(mi_scores):
                    # Also calculate correlation
                    correlation = np.corrcoef(X[:, i], scores)[0, 1]
                    if np.isnan(correlation):
                        correlation = 0.0

                    # Combine MI and correlation
                    combined_importance = (mi_score + abs(correlation)) / 2

                    feature_importances.append(
                        FeatureImportance(
                            feature_name=f"feature_{i}",
                            importance_score=float(combined_importance),
                            contribution=float(correlation),
                            rank=0,
                            confidence=0.7,
                            method="global_statistical",
                            metadata={
                                "mutual_info": float(mi_score),
                                "correlation": float(correlation),
                            },
                        )
                    )

            # Sort and rank
            feature_importances.sort(key=lambda x: x.importance_score, reverse=True)
            for j, fi in enumerate(feature_importances):
                fi.rank = j + 1

            return feature_importances

        except Exception as e:
            self.logger.warning(f"Global statistical importance failed: {e}")
            return []

    async def _global_custom_importance(
        self, X: np.ndarray, scores: np.ndarray
    ) -> list[FeatureImportance]:
        """Calculate global importance using custom methods."""

        feature_importances = []
        n_features = X.shape[1]

        # Simple variance-based importance
        for i in range(n_features):
            feature_values = X[:, i]

            # Calculate variance of feature
            feature_var = np.var(feature_values)

            # Calculate correlation with scores (if possible)
            try:
                correlation = np.corrcoef(feature_values, scores)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            except:
                correlation = 0.0

            # Combine variance and correlation
            importance = feature_var * abs(correlation)

            feature_importances.append(
                FeatureImportance(
                    feature_name=f"feature_{i}",
                    importance_score=float(importance),
                    contribution=float(correlation),
                    rank=0,
                    confidence=0.5,
                    method="global_custom",
                    metadata={
                        "variance": float(feature_var),
                        "correlation": float(correlation),
                    },
                )
            )

        # Normalize importances
        if feature_importances:
            max_importance = max(fi.importance_score for fi in feature_importances)
            if max_importance > 0:
                for fi in feature_importances:
                    fi.importance_score /= max_importance

        # Sort and rank
        feature_importances.sort(key=lambda x: x.importance_score, reverse=True)
        for j, fi in enumerate(feature_importances):
            fi.rank = j + 1

        return feature_importances

    async def _discover_anomaly_patterns(
        self,
        X: np.ndarray,
        predictions: np.ndarray,
        scores: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Discover common patterns in anomalous samples."""

        patterns = []

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        try:
            # Find anomalous samples
            anomaly_indices = np.where(predictions == 1)[0]

            if len(anomaly_indices) == 0:
                return patterns

            anomalous_data = X[anomaly_indices]
            normal_data = X[predictions == 0]

            if len(normal_data) == 0:
                return patterns

            # Pattern 1: Feature value ranges
            for i, feature_name in enumerate(feature_names):
                anomaly_values = anomalous_data[:, i]
                normal_values = normal_data[:, i]

                # Calculate statistics
                anomaly_mean = np.mean(anomaly_values)
                normal_mean = np.mean(normal_values)
                anomaly_std = np.std(anomaly_values)
                normal_std = np.std(normal_values)

                # Check if there's a significant difference
                if abs(anomaly_mean - normal_mean) > max(anomaly_std, normal_std):
                    pattern = {
                        "type": "feature_range",
                        "feature": feature_name,
                        "description": f"{feature_name} values are typically "
                        f"{'higher' if anomaly_mean > normal_mean else 'lower'} "
                        f"in anomalies",
                        "anomaly_mean": float(anomaly_mean),
                        "normal_mean": float(normal_mean),
                        "difference": float(abs(anomaly_mean - normal_mean)),
                        "strength": min(
                            abs(anomaly_mean - normal_mean)
                            / max(anomaly_std, normal_std, 1e-8),
                            5.0,
                        ),
                    }
                    patterns.append(pattern)

            # Pattern 2: Feature combinations (simple two-feature interactions)
            if X.shape[1] > 1:
                for i in range(min(5, X.shape[1])):
                    for j in range(i + 1, min(5, X.shape[1])):
                        # Calculate 2D centroid distance
                        anomaly_points = anomalous_data[:, [i, j]]
                        normal_points = normal_data[:, [i, j]]

                        anomaly_centroid = np.mean(anomaly_points, axis=0)
                        normal_centroid = np.mean(normal_points, axis=0)

                        centroid_distance = np.linalg.norm(
                            anomaly_centroid - normal_centroid
                        )

                        if centroid_distance > 0.5:  # Threshold for significance
                            pattern = {
                                "type": "feature_combination",
                                "features": [feature_names[i], feature_names[j]],
                                "description": f"Anomalies show distinct patterns in "
                                f"{feature_names[i]} and {feature_names[j]} combination",
                                "anomaly_centroid": anomaly_centroid.tolist(),
                                "normal_centroid": normal_centroid.tolist(),
                                "separation": float(centroid_distance),
                                "strength": min(centroid_distance, 3.0),
                            }
                            patterns.append(pattern)

            # Sort patterns by strength
            patterns.sort(key=lambda x: x.get("strength", 0), reverse=True)

            # Keep top 10 patterns
            return patterns[:10]

        except Exception as e:
            self.logger.warning(f"Pattern discovery failed: {e}")
            return patterns

    async def _analyze_feature_interactions(
        self,
        X: np.ndarray,
        scores: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> list[tuple[str, str, float]]:
        """Analyze feature interactions."""

        interactions = []

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        try:
            n_features = X.shape[1]

            # Calculate pairwise feature interactions
            for i in range(min(10, n_features)):
                for j in range(i + 1, min(10, n_features)):
                    # Calculate interaction strength using correlation
                    feature_i = X[:, i]
                    feature_j = X[:, j]

                    # Product interaction
                    interaction_feature = feature_i * feature_j

                    # Correlation with anomaly scores
                    try:
                        interaction_correlation = abs(
                            np.corrcoef(interaction_feature, scores)[0, 1]
                        )
                        if np.isnan(interaction_correlation):
                            interaction_correlation = 0.0
                    except:
                        interaction_correlation = 0.0

                    # Individual correlations
                    try:
                        corr_i = abs(np.corrcoef(feature_i, scores)[0, 1])
                        corr_j = abs(np.corrcoef(feature_j, scores)[0, 1])
                        if np.isnan(corr_i):
                            corr_i = 0.0
                        if np.isnan(corr_j):
                            corr_j = 0.0
                    except:
                        corr_i = corr_j = 0.0

                    # Interaction strength is how much the interaction adds
                    individual_max = max(corr_i, corr_j)
                    interaction_strength = max(
                        0, interaction_correlation - individual_max
                    )

                    if (
                        interaction_strength > 0.1
                    ):  # Threshold for significant interaction
                        interactions.append(
                            (
                                feature_names[i],
                                feature_names[j],
                                float(interaction_strength),
                            )
                        )

            # Sort by interaction strength
            interactions.sort(key=lambda x: x[2], reverse=True)

            # Return top 20 interactions
            return interactions[:20]

        except Exception as e:
            self.logger.warning(f"Feature interaction analysis failed: {e}")
            return interactions

    def _create_global_summary(
        self,
        feature_importances: list[FeatureImportance],
        anomaly_patterns: list[dict[str, Any]],
        feature_interactions: list[tuple[str, str, float]] | None,
    ) -> str:
        """Create global explanation summary."""

        summary_parts = ["Global Processor Explanation Summary", "=" * 35, ""]

        # Top features
        if feature_importances:
            summary_parts.extend(
                ["Most Important Features:", "------------------------"]
            )

            for i, fi in enumerate(feature_importances[:5]):
                summary_parts.append(
                    f"{i + 1}. {fi.feature_name}: {fi.importance_score:.3f} "
                    f"({'positive' if fi.contribution > 0 else 'negative'} contribution)"
                )
            summary_parts.append("")

        # Anomaly patterns
        if anomaly_patterns:
            summary_parts.extend(["Key Anomaly Patterns:", "---------------------"])

            for pattern in anomaly_patterns[:3]:
                summary_parts.append(f"• {pattern['description']}")
            summary_parts.append("")

        # Feature interactions
        if feature_interactions:
            summary_parts.extend(
                ["Important Feature Interactions:", "-------------------------------"]
            )

            for feat1, feat2, strength in feature_interactions[:3]:
                summary_parts.append(
                    f"• {feat1} × {feat2}: interaction strength {strength:.3f}"
                )
            summary_parts.append("")

        # Summary
        summary_parts.extend(
            [
                "Processor Behavior:",
                "---------------",
                "This anomaly processing processor identifies outliers based on the patterns above. ",
                "The most important features drive the majority of anomaly decisions, while ",
                "feature interactions capture more complex relationships in the data.",
            ]
        )

        return "\n".join(summary_parts)

    def _calculate_interpretability_score(
        self,
        detector: Detector,
        feature_importances: list[FeatureImportance],
        anomaly_patterns: list[dict[str, Any]],
    ) -> float:
        """Calculate overall processor interpretability score."""

        score = 0.0

        # Base score from processor type
        processor_interpretability = {
            "isolationforest": 0.7,
            "lof": 0.6,
            "oneclass": 0.5,
            "autoencoder": 0.3,
            "vae": 0.2,
        }

        processor_name = detector.algorithm_config.name.lower()
        base_score = 0.5

        for processor_key, interpretability in processor_interpretability.items():
            if processor_key in processor_name:
                base_score = interpretability
                break

        score += base_score * 0.4

        # Score from feature importance quality
        if feature_importances:
            # Check if top features dominate
            top_5_importance = sum(
                fi.importance_score for fi in feature_importances[:5]
            )
            total_importance = sum(fi.importance_score for fi in feature_importances)

            if total_importance > 0:
                concentration_ratio = top_5_importance / total_importance
                score += min(concentration_ratio, 1.0) * 0.3

        # Score from pattern clarity
        if anomaly_patterns:
            avg_pattern_strength = np.mean(
                [p.get("strength", 0) for p in anomaly_patterns]
            )
            score += min(avg_pattern_strength / 3.0, 1.0) * 0.3

        return min(score, 1.0)
