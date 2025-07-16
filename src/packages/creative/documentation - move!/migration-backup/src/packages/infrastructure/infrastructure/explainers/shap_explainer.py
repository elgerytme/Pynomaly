"""SHAP-based explainer implementation with comprehensive features.

This module provides advanced SHAP integration including:
- Multiple explainer types (Tree, Kernel, Linear, Permutation, Additive)
- Interaction effects analysis
- Partial dependence plots
- Waterfall plots
- Force plots
- Clustering explanations
- Batch processing
- Caching for performance
"""

from __future__ import annotations

import logging
import uuid
import warnings
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np

from monorepo.domain.services.explainability_service import (
    CohortExplanation,
    ExplainerProtocol,
    ExplanationMethod,
    FeatureContribution,
    GlobalExplanation,
    LocalExplanation,
)

logger = logging.getLogger(__name__)

# Optional SHAP import
try:
    import shap
    from shap.explainers import (
        Additive,
        Exact,
        Kernel,
        KernelExplainer,
        Linear,
        LinearExplainer,
        Permutation,
        PermutationExplainer,
        Sampling,
        Tree,
        TreeExplainer,
    )
    from shap.plots import beeswarm, force, partial_dependence, waterfall

    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")

# Optional plotting dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class SHAPExplainer(ExplainerProtocol):
    """Advanced SHAP-based explainer for anomaly detection models."""

    def __init__(
        self,
        explainer_type: str = "auto",
        background_data: np.ndarray | None = None,
        n_background_samples: int = 100,
        enable_interactions: bool = True,
        enable_clustering: bool = True,
        cache_explainers: bool = True,
        max_evals: int = 2000,
        batch_size: int = 100,
        **kwargs,
    ):
        """Initialize comprehensive SHAP explainer.

        Args:
            explainer_type: Type of SHAP explainer ('auto', 'tree', 'linear', 'kernel', 'permutation', 'additive')
            background_data: Background data for explainer initialization
            n_background_samples: Number of background samples to use
            enable_interactions: Enable interaction effects analysis
            enable_clustering: Enable clustering-based explanations
            cache_explainers: Cache explainers for performance
            max_evals: Maximum evaluations for kernel explainer
            batch_size: Batch size for processing large datasets
            **kwargs: Additional arguments for SHAP explainer
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP is required for SHAPExplainer. Install with: pip install shap"
            )

        self.explainer_type = explainer_type
        self.background_data = background_data
        self.n_background_samples = n_background_samples
        self.enable_interactions = enable_interactions
        self.enable_clustering = enable_clustering
        self.cache_explainers = cache_explainers
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.explainer_kwargs = kwargs

        # Internal state
        self._explainer_cache = {} if cache_explainers else None
        self._interaction_cache = {}
        self._clustering_cache = {}

        # Suppress SHAP warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="shap")

        logger.info(f"Advanced SHAP explainer initialized with type: {explainer_type}")

    def _get_explainer(
        self, model: Any, data: np.ndarray | None = None
    ) -> shap.Explainer:
        """Get or create SHAP explainer for the model."""
        if self._explainer is None or self._model != model:
            self._model = model

            # Determine background data
            background = self._get_background_data(data)

            # Create explainer based on type
            if self.explainer_type == "auto":
                try:
                    # Try to auto-detect explainer type
                    self._explainer = shap.Explainer(
                        model, background, **self.explainer_kwargs
                    )
                except Exception as e:
                    logger.warning(
                        f"Auto explainer failed: {e}. Falling back to Permutation explainer."
                    )
                    self._explainer = shap.explainers.Permutation(
                        model.predict if hasattr(model, "predict") else model,
                        background,
                        **self.explainer_kwargs,
                    )
            elif self.explainer_type == "tree":
                self._explainer = shap.TreeExplainer(
                    model, background, **self.explainer_kwargs
                )
            elif self.explainer_type == "linear":
                self._explainer = shap.LinearExplainer(
                    model, background, **self.explainer_kwargs
                )
            elif self.explainer_type == "kernel":
                self._explainer = shap.KernelExplainer(
                    model.predict if hasattr(model, "predict") else model,
                    background,
                    **self.explainer_kwargs,
                )
            elif self.explainer_type == "permutation":
                self._explainer = shap.explainers.Permutation(
                    model.predict if hasattr(model, "predict") else model,
                    background,
                    **self.explainer_kwargs,
                )
            else:
                raise ValueError(f"Unknown explainer type: {self.explainer_type}")

        return self._explainer

    def _get_background_data(self, data: np.ndarray | None = None) -> np.ndarray:
        """Get background data for explainer."""
        if self.background_data is not None:
            return self.background_data
        elif data is not None:
            # Sample background data from provided data
            n_samples = min(self.n_background_samples, len(data))
            indices = np.random.choice(len(data), n_samples, replace=False)
            return data[indices]
        else:
            raise ValueError("Background data is required for SHAP explainer")

    def explain_local(
        self, instance: np.ndarray, model: Any, feature_names: list[str], **kwargs
    ) -> LocalExplanation:
        """Generate local explanation using SHAP."""
        try:
            # Get explainer
            explainer = self._get_explainer(model, None)

            # Ensure instance is 2D
            if instance.ndim == 1:
                instance = instance.reshape(1, -1)

            # Calculate SHAP values
            shap_values = explainer(instance)

            # Handle different SHAP value formats
            if hasattr(shap_values, "values"):
                values = shap_values.values[0]  # Get first instance
                (
                    shap_values.base_values[0]
                    if hasattr(shap_values, "base_values")
                    else 0
                )
            else:
                values = shap_values[0]  # Get first instance

            # Get model prediction
            if hasattr(model, "decision_function"):
                anomaly_score = model.decision_function(instance)[0]
            elif hasattr(model, "predict_proba"):
                anomaly_score = model.predict_proba(instance)[
                    0, 1
                ]  # Anomaly probability
            else:
                anomaly_score = model.predict(instance)[0]

            # Create feature contributions
            feature_contributions = []
            for i, (feature_name, contribution) in enumerate(
                zip(feature_names, values, strict=False)
            ):
                feature_contributions.append(
                    FeatureContribution(
                        feature_name=feature_name,
                        value=float(instance[0, i]),
                        contribution=float(contribution),
                        importance=abs(float(contribution)),
                        rank=i + 1,
                        description=f"SHAP contribution for {feature_name}",
                    )
                )

            # Sort by importance
            feature_contributions.sort(key=lambda x: x.importance, reverse=True)
            for i, contrib in enumerate(feature_contributions):
                contrib.rank = i + 1

            # Determine prediction
            threshold = kwargs.get("threshold", 0.5)
            prediction = "anomaly" if anomaly_score > threshold else "normal"
            confidence = float(abs(anomaly_score - threshold))

            return LocalExplanation(
                instance_id=str(uuid.uuid4()),
                anomaly_score=float(anomaly_score),
                prediction=prediction,
                confidence=confidence,
                feature_contributions=feature_contributions,
                explanation_method=ExplanationMethod.SHAP,
                model_name=model.__class__.__name__,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"SHAP local explanation failed: {e}")
            raise

    def explain_global(
        self, data: np.ndarray, model: Any, feature_names: list[str], **kwargs
    ) -> GlobalExplanation:
        """Generate global explanation using SHAP."""
        try:
            # Get explainer
            explainer = self._get_explainer(model, data)

            # Calculate SHAP values for all data
            max_samples = kwargs.get("max_samples", 1000)
            if len(data) > max_samples:
                indices = np.random.choice(len(data), max_samples, replace=False)
                sample_data = data[indices]
            else:
                sample_data = data

            shap_values = explainer(sample_data)

            # Handle different SHAP value formats
            if hasattr(shap_values, "values"):
                values = shap_values.values
            else:
                values = shap_values

            # Calculate feature importances (mean absolute SHAP values)
            feature_importances = {}
            mean_abs_shap = np.mean(np.abs(values), axis=0)

            for i, feature_name in enumerate(feature_names):
                feature_importances[feature_name] = float(mean_abs_shap[i])

            # Get top features
            sorted_features = sorted(
                feature_importances.items(), key=lambda x: x[1], reverse=True
            )
            top_features = [f[0] for f in sorted_features[:10]]

            # Calculate model performance if possible
            model_performance = {}
            try:
                if hasattr(model, "score"):
                    model_performance["score"] = float(model.score(data))
            except Exception:
                pass

            # Create summary
            top_3_features = [f[0] for f in sorted_features[:3]]
            summary = f"Model is most sensitive to: {', '.join(top_3_features)}"

            return GlobalExplanation(
                model_name=model.__class__.__name__,
                feature_importances=feature_importances,
                top_features=top_features,
                explanation_method=ExplanationMethod.SHAP,
                model_performance=model_performance,
                timestamp=datetime.now().isoformat(),
                summary=summary,
            )

        except Exception as e:
            logger.error(f"SHAP global explanation failed: {e}")
            raise

    def explain_cohort(
        self,
        instances: np.ndarray,
        model: Any,
        feature_names: list[str],
        cohort_id: str,
        **kwargs,
    ) -> CohortExplanation:
        """Generate cohort explanation using SHAP."""
        try:
            # Get explainer
            explainer = self._get_explainer(model, None)

            # Calculate SHAP values for cohort
            shap_values = explainer(instances)

            # Handle different SHAP value formats
            if hasattr(shap_values, "values"):
                values = shap_values.values
            else:
                values = shap_values

            # Calculate average contributions for cohort
            mean_contributions = np.mean(values, axis=0)
            mean_abs_contributions = np.mean(np.abs(values), axis=0)

            # Create common feature contributions
            common_features = []
            for i, feature_name in enumerate(feature_names):
                common_features.append(
                    FeatureContribution(
                        feature_name=feature_name,
                        value=float(np.mean(instances[:, i])),
                        contribution=float(mean_contributions[i]),
                        importance=float(mean_abs_contributions[i]),
                        rank=i + 1,
                        description=f"Average SHAP contribution for {feature_name} in cohort",
                    )
                )

            # Sort by importance
            common_features.sort(key=lambda x: x.importance, reverse=True)
            for i, contrib in enumerate(common_features):
                contrib.rank = i + 1

            # Create cohort description
            top_features = [f.feature_name for f in common_features[:3]]
            cohort_description = (
                f"Cohort characterized by high importance of: {', '.join(top_features)}"
            )

            return CohortExplanation(
                cohort_id=cohort_id,
                cohort_description=cohort_description,
                instance_count=len(instances),
                common_features=common_features,
                explanation_method=ExplanationMethod.SHAP,
                model_name=model.__class__.__name__,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"SHAP cohort explanation failed: {e}")
            raise

    def get_interaction_values(
        self,
        instances: np.ndarray,
        model: Any,
        feature_names: list[str],
        max_interactions: int = 20,
    ) -> dict[str, float]:
        """Get SHAP interaction values for feature pairs.

        Args:
            instances: Instances to analyze
            model: Trained model
            feature_names: List of feature names
            max_interactions: Maximum number of interactions to return

        Returns:
            Dictionary mapping feature pairs to interaction values
        """
        if not self.enable_interactions:
            logger.warning("Interaction analysis is disabled")
            return {}

        try:
            # Check cache
            cache_key = f"{id(model)}_{instances.shape}"
            if cache_key in self._interaction_cache:
                return self._interaction_cache[cache_key]

            # Get explainer
            explainer = self._get_explainer(model, instances)

            # Only tree explainer supports interaction values
            if hasattr(explainer, "shap_interaction_values"):
                # Calculate interaction values
                interaction_values = explainer.shap_interaction_values(instances)

                # Calculate mean absolute interaction values
                mean_interactions = np.mean(np.abs(interaction_values), axis=0)

                # Extract pairwise interactions
                interactions = {}
                for i in range(len(feature_names)):
                    for j in range(i + 1, len(feature_names)):
                        interaction_name = f"{feature_names[i]} × {feature_names[j]}"
                        interaction_value = mean_interactions[i, j]
                        interactions[interaction_name] = float(interaction_value)

                # Sort and return top interactions
                sorted_interactions = sorted(
                    interactions.items(), key=lambda x: x[1], reverse=True
                )

                result = dict(sorted_interactions[:max_interactions])

                # Cache result
                self._interaction_cache[cache_key] = result
                return result

            else:
                logger.warning(
                    "Interaction values not available for this explainer type"
                )
                return {}

        except Exception as e:
            logger.error(f"Failed to get interaction values: {e}")
            return {}

    def get_partial_dependence(
        self,
        feature_name: str,
        model: Any,
        background_data: np.ndarray,
        feature_names: list[str],
        num_points: int = 50,
        percentile_range: tuple[float, float] = (0.05, 0.95),
    ) -> dict[str, list[float]]:
        """Calculate partial dependence for a feature.

        Args:
            feature_name: Name of the feature
            model: Trained model
            background_data: Background data
            feature_names: List of feature names
            num_points: Number of points to evaluate
            percentile_range: Percentile range for feature values

        Returns:
            Dictionary with feature values and partial dependence values
        """
        try:
            if feature_name not in feature_names:
                raise ValueError(f"Feature {feature_name} not found in feature names")

            feature_idx = feature_names.index(feature_name)

            # Get feature value range using percentiles
            feature_values = background_data[:, feature_idx]
            min_val = np.percentile(feature_values, percentile_range[0] * 100)
            max_val = np.percentile(feature_values, percentile_range[1] * 100)

            # Create evaluation points
            eval_points = np.linspace(min_val, max_val, num_points)

            # Calculate partial dependence
            partial_deps = []
            for point in eval_points:
                # Create modified data
                modified_data = background_data.copy()
                modified_data[:, feature_idx] = point

                # Get model predictions
                predictions = self._get_model_predictions(model, modified_data)

                # Calculate mean prediction
                mean_pred = np.mean(predictions)
                partial_deps.append(float(mean_pred))

            return {
                "feature_values": eval_points.tolist(),
                "partial_dependence": partial_deps,
                "feature_name": feature_name,
                "num_points": num_points,
                "percentile_range": percentile_range,
            }

        except Exception as e:
            logger.error(f"Failed to calculate partial dependence: {e}")
            return {"error": str(e)}

    def get_clustering_explanation(
        self,
        instances: np.ndarray,
        model: Any,
        feature_names: list[str],
        n_clusters: int = 5,
        clustering_method: str = "kmeans",
    ) -> dict[str, Any]:
        """Get clustering-based explanations for instances.

        Args:
            instances: Instances to cluster and explain
            model: Trained model
            feature_names: List of feature names
            n_clusters: Number of clusters
            clustering_method: Clustering method ('kmeans', 'hierarchical')

        Returns:
            Dictionary with clustering results and explanations
        """
        if not self.enable_clustering:
            logger.warning("Clustering analysis is disabled")
            return {}

        try:
            # Check cache
            cache_key = (
                f"{id(model)}_{instances.shape}_{n_clusters}_{clustering_method}"
            )
            if cache_key in self._clustering_cache:
                return self._clustering_cache[cache_key]

            # Get explainer
            explainer = self._get_explainer(model, instances)

            # Calculate SHAP values for clustering
            shap_values = explainer(instances)

            # Handle different SHAP value formats
            if hasattr(shap_values, "values"):
                values = shap_values.values
            else:
                values = shap_values

            # Perform clustering on SHAP values
            if clustering_method == "kmeans":
                from sklearn.cluster import KMeans

                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            elif clustering_method == "hierarchical":
                from sklearn.cluster import AgglomerativeClustering

                clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            else:
                raise ValueError(f"Unknown clustering method: {clustering_method}")

            # Cluster based on SHAP values
            cluster_labels = clusterer.fit_predict(values)

            # Analyze clusters
            cluster_explanations = {}
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_instances = instances[cluster_mask]
                cluster_shap = values[cluster_mask]

                if len(cluster_instances) == 0:
                    continue

                # Calculate cluster characteristics
                mean_shap = np.mean(cluster_shap, axis=0)
                std_shap = np.std(cluster_shap, axis=0)

                # Find most important features for this cluster
                feature_importance = np.abs(mean_shap)
                top_features = np.argsort(feature_importance)[-5:][::-1]

                cluster_explanations[f"cluster_{cluster_id}"] = {
                    "size": int(np.sum(cluster_mask)),
                    "mean_shap": mean_shap.tolist(),
                    "std_shap": std_shap.tolist(),
                    "top_features": [feature_names[i] for i in top_features],
                    "top_feature_values": [
                        float(feature_importance[i]) for i in top_features
                    ],
                    "instances": cluster_instances.tolist(),
                }

            result = {
                "cluster_labels": cluster_labels.tolist(),
                "n_clusters": n_clusters,
                "clustering_method": clustering_method,
                "cluster_explanations": cluster_explanations,
            }

            # Cache result
            self._clustering_cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"Failed to get clustering explanation: {e}")
            return {"error": str(e)}

    def generate_waterfall_data(
        self,
        instance: np.ndarray,
        model: Any,
        feature_names: list[str],
        max_features: int = 10,
    ) -> dict[str, Any]:
        """Generate data for waterfall plot visualization.

        Args:
            instance: Single instance to explain
            model: Trained model
            feature_names: List of feature names
            max_features: Maximum number of features to include

        Returns:
            Dictionary with waterfall plot data
        """
        try:
            # Get local explanation
            local_explanation = self.explain_local(instance, model, feature_names)

            # Get top features by importance
            top_features = local_explanation.feature_contributions[:max_features]

            # Prepare waterfall data
            feature_names_list = [f.feature_name for f in top_features]
            feature_values = [f.value for f in top_features]
            contributions = [f.contribution for f in top_features]

            # Calculate cumulative contributions
            cumulative = [0]
            for contrib in contributions:
                cumulative.append(cumulative[-1] + contrib)

            return {
                "feature_names": feature_names_list,
                "feature_values": feature_values,
                "contributions": contributions,
                "cumulative": cumulative,
                "base_value": 0.0,  # Baseline
                "final_value": local_explanation.anomaly_score,
                "instance_id": local_explanation.instance_id,
            }

        except Exception as e:
            logger.error(f"Failed to generate waterfall data: {e}")
            return {"error": str(e)}

    def batch_explain_local(
        self,
        instances: np.ndarray,
        model: Any,
        feature_names: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[LocalExplanation]:
        """Batch processing for local explanations.

        Args:
            instances: Batch of instances to explain
            model: Trained model
            feature_names: List of feature names
            progress_callback: Optional callback for progress updates

        Returns:
            List of local explanations
        """
        try:
            explanations = []
            total_instances = len(instances)

            # Process in batches
            for i in range(0, total_instances, self.batch_size):
                batch_end = min(i + self.batch_size, total_instances)
                batch_instances = instances[i:batch_end]

                # Explain batch
                for j, instance in enumerate(batch_instances):
                    explanation = self.explain_local(instance, model, feature_names)
                    explanations.append(explanation)

                    # Update progress
                    if progress_callback:
                        progress_callback(i + j + 1, total_instances)

            return explanations

        except Exception as e:
            logger.error(f"Failed to batch explain local: {e}")
            raise

    def compute_feature_statistics(
        self,
        explanations: list[LocalExplanation],
        confidence_level: float = 0.95,
    ) -> dict[str, dict[str, float]]:
        """Compute comprehensive feature statistics across explanations.

        Args:
            explanations: List of local explanations
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary with feature statistics
        """
        try:
            feature_stats = {}

            # Collect feature data
            for explanation in explanations:
                for contrib in explanation.feature_contributions:
                    feature_name = contrib.feature_name

                    if feature_name not in feature_stats:
                        feature_stats[feature_name] = {
                            "values": [],
                            "contributions": [],
                            "importances": [],
                        }

                    feature_stats[feature_name]["values"].append(contrib.value)
                    feature_stats[feature_name]["contributions"].append(
                        contrib.contribution
                    )
                    feature_stats[feature_name]["importances"].append(
                        contrib.importance
                    )

            # Calculate statistics
            stats_result = {}
            alpha = 1 - confidence_level

            for feature_name, data in feature_stats.items():
                values = np.array(data["values"])
                contributions = np.array(data["contributions"])
                importances = np.array(data["importances"])

                stats_result[feature_name] = {
                    # Basic statistics
                    "count": len(values),
                    "mean_value": float(np.mean(values)),
                    "std_value": float(np.std(values)),
                    "min_value": float(np.min(values)),
                    "max_value": float(np.max(values)),
                    # Contribution statistics
                    "mean_contribution": float(np.mean(contributions)),
                    "std_contribution": float(np.std(contributions)),
                    "min_contribution": float(np.min(contributions)),
                    "max_contribution": float(np.max(contributions)),
                    # Importance statistics
                    "mean_importance": float(np.mean(importances)),
                    "std_importance": float(np.std(importances)),
                    "total_importance": float(np.sum(importances)),
                    # Confidence intervals
                    "value_ci_lower": float(np.percentile(values, alpha / 2 * 100)),
                    "value_ci_upper": float(
                        np.percentile(values, (1 - alpha / 2) * 100)
                    ),
                    "contribution_ci_lower": float(
                        np.percentile(contributions, alpha / 2 * 100)
                    ),
                    "contribution_ci_upper": float(
                        np.percentile(contributions, (1 - alpha / 2) * 100)
                    ),
                    # Distribution metrics
                    "value_skewness": float(self._calculate_skewness(values)),
                    "value_kurtosis": float(self._calculate_kurtosis(values)),
                    "contribution_skewness": float(
                        self._calculate_skewness(contributions)
                    ),
                    "contribution_kurtosis": float(
                        self._calculate_kurtosis(contributions)
                    ),
                }

            return stats_result

        except Exception as e:
            logger.error(f"Failed to compute feature statistics: {e}")
            return {}

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return 0.0

        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        if len(data) < 4:
            return 0.0

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return 0.0

        return np.mean(((data - mean) / std) ** 4) - 3.0

    def _get_model_predictions(self, model: Any, data: np.ndarray) -> np.ndarray:
        """Get predictions from model in a consistent format."""
        if hasattr(model, "decision_function"):
            return model.decision_function(data)
        elif hasattr(model, "predict_proba"):
            return model.predict_proba(data)[:, 1]  # Anomaly probability
        else:
            return model.predict(data)

    def clear_cache(self):
        """Clear all caches."""
        if self._explainer_cache:
            self._explainer_cache.clear()
        self._interaction_cache.clear()
        self._clustering_cache.clear()
        logger.info("SHAP explainer caches cleared")

    def get_explainer_info(self) -> dict[str, Any]:
        """Get information about the explainer state."""
        return {
            "explainer_type": self.explainer_type,
            "n_background_samples": self.n_background_samples,
            "enable_interactions": self.enable_interactions,
            "enable_clustering": self.enable_clustering,
            "cache_explainers": self.cache_explainers,
            "max_evals": self.max_evals,
            "batch_size": self.batch_size,
            "cached_explainers": len(self._explainer_cache)
            if self._explainer_cache
            else 0,
            "cached_interactions": len(self._interaction_cache),
            "cached_clusterings": len(self._clustering_cache),
        }
