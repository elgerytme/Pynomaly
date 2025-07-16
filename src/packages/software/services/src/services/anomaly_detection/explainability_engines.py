"""Explainability engines for different explanation methods."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from monorepo.application.services.explainability_core import (
    ExplanationConfig,
    GlobalExplanation,
    LocalExplanation,
)
from monorepo.shared.protocols import DetectorProtocol

# Optional explainability imports with fallbacks
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False

try:
    import lime

    LIME_AVAILABLE = True
except ImportError:
    lime = None
    LIME_AVAILABLE = False

try:
    from sklearn.inspection import permutation_importance

    SKLEARN_AVAILABLE = True
except ImportError:
    permutation_importance = None
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExplanationEngine:
    """Base class for explanation engines."""

    def __init__(self, cache_explanations: bool = True):
        self.cache_explanations = cache_explanations
        self.explanation_cache: dict[str, Any] = {}
        self.explainer_cache: dict[str, Any] = {}

    def clear_caches(self) -> None:
        """Clear all caches."""
        self.explanation_cache.clear()
        self.explainer_cache.clear()


class LocalExplanationEngine(ExplanationEngine):
    """Engine for generating local explanations."""

    def __init__(self, enable_shap: bool = True, enable_lime: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.enable_shap = enable_shap and SHAP_AVAILABLE
        self.enable_lime = enable_lime and LIME_AVAILABLE

    async def generate_explanations(
        self,
        detector: DetectorProtocol,
        X: np.ndarray,
        feature_names: list[str],
        config: ExplanationConfig,
    ) -> list[LocalExplanation]:
        """Generate local explanations for individual samples."""
        try:
            explanations = []
            n_samples = min(config.n_samples, len(X))

            # Select representative samples
            indices = self._select_representative_samples(X, n_samples)

            for idx in indices:
                sample = X[idx : idx + 1]
                prediction = detector.decision_function(sample)[0]

                # Generate explanation using available methods
                feature_contributions = {}

                if self.enable_shap:
                    try:
                        shap_values = await self._compute_shap_values(
                            detector, sample, X, feature_names
                        )
                        if shap_values is not None:
                            feature_contributions.update(shap_values)
                    except Exception as e:
                        logger.warning(f"SHAP explanation failed for sample {idx}: {e}")

                if self.enable_lime and not feature_contributions:
                    try:
                        lime_values = await self._compute_lime_values(
                            detector, sample, X, feature_names
                        )
                        if lime_values is not None:
                            feature_contributions.update(lime_values)
                    except Exception as e:
                        logger.warning(f"LIME explanation failed for sample {idx}: {e}")

                # Fallback to gradient-based explanation
                if not feature_contributions:
                    feature_contributions = self._compute_gradient_explanation(
                        detector, sample, feature_names
                    )

                # Calculate confidence
                confidence = self._calculate_prediction_confidence(detector, sample, X)

                explanation = LocalExplanation(
                    sample_id=f"sample_{idx}",
                    prediction=float(prediction),
                    confidence=confidence,
                    feature_contributions=feature_contributions,
                    explanation_method=(
                        "shap"
                        if self.enable_shap
                        else "lime"
                        if self.enable_lime
                        else "gradient"
                    ),
                    metadata={"sample_index": int(idx)},
                )

                explanations.append(explanation)

            return explanations

        except Exception as e:
            logger.error(f"Local explanation generation failed: {e}")
            return []

    async def _compute_shap_values(
        self,
        detector: DetectorProtocol,
        sample: np.ndarray,
        background: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, float] | None:
        """Compute SHAP values for explanation."""
        if not SHAP_AVAILABLE:
            return None

        try:
            cache_key = f"shap_{id(detector)}"

            if cache_key not in self.explainer_cache:

                def predict_fn(x) -> Any:
                    return detector.decision_function(x)

                # Sample background data for efficiency
                background_sample = background[
                    np.random.choice(
                        len(background), min(100, len(background)), replace=False
                    )
                ]
                explainer = shap.KernelExplainer(predict_fn, background_sample)
                self.explainer_cache[cache_key] = explainer
            else:
                explainer = self.explainer_cache[cache_key]

            # Compute SHAP values
            shap_values = explainer.shap_values(sample, nsamples=100)

            # Convert to dictionary
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]

            return {
                feature_names[i]: float(shap_values[i])
                for i in range(min(len(feature_names), len(shap_values)))
            }

        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return None

    async def _compute_lime_values(
        self,
        detector: DetectorProtocol,
        sample: np.ndarray,
        background: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, float] | None:
        """Compute LIME values for explanation."""
        if not LIME_AVAILABLE:
            return None

        try:
            from lime.lime_tabular import LimeTabularExplainer

            cache_key = f"lime_{id(detector)}"

            if cache_key not in self.explainer_cache:
                explainer = LimeTabularExplainer(
                    background,
                    feature_names=feature_names,
                    mode="regression",
                    discretize_continuous=True,
                )
                self.explainer_cache[cache_key] = explainer
            else:
                explainer = self.explainer_cache[cache_key]

            # Generate explanation
            explanation = explainer.explain_instance(
                sample[0], detector.decision_function, num_features=len(feature_names)
            )

            # Convert to dictionary
            lime_values = {}
            for feature_idx, importance in explanation.as_list():
                if isinstance(feature_idx, int) and feature_idx < len(feature_names):
                    lime_values[feature_names[feature_idx]] = importance
                elif isinstance(feature_idx, str):
                    lime_values[feature_idx] = importance

            return lime_values

        except Exception as e:
            logger.warning(f"LIME computation failed: {e}")
            return None

    def _compute_gradient_explanation(
        self, detector: DetectorProtocol, sample: np.ndarray, feature_names: list[str]
    ) -> dict[str, float]:
        """Compute gradient-based explanation (fallback method)."""
        try:
            # Simple numerical gradient approximation
            epsilon = 1e-5
            base_prediction = detector.decision_function(sample)[0]
            gradients = {}

            for i, feature_name in enumerate(feature_names):
                # Perturb feature
                perturbed_sample = sample.copy()
                perturbed_sample[0, i] += epsilon

                perturbed_prediction = detector.decision_function(perturbed_sample)[0]
                gradient = (perturbed_prediction - base_prediction) / epsilon

                gradients[feature_name] = float(gradient * sample[0, i])

            return gradients

        except Exception as e:
            logger.warning(f"Gradient explanation failed: {e}")
            return dict.fromkeys(feature_names, 0.0)

    def _select_representative_samples(
        self, X: np.ndarray, n_samples: int
    ) -> list[int]:
        """Select representative samples for explanation."""
        if len(X) <= n_samples:
            return list(range(len(X)))

        try:
            # Use k-means clustering to select diverse samples
            from sklearn.cluster import KMeans

            n_clusters = min(n_samples, len(X))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)

            # Select one sample from each cluster
            selected_indices = []
            for cluster_id in range(n_clusters):
                cluster_mask = clusters == cluster_id
                cluster_indices = np.where(cluster_mask)[0]

                if len(cluster_indices) > 0:
                    # Select sample closest to centroid
                    centroid = kmeans.cluster_centers_[cluster_id]
                    distances = np.linalg.norm(X[cluster_indices] - centroid, axis=1)
                    closest_idx = cluster_indices[np.argmin(distances)]
                    selected_indices.append(closest_idx)

            return selected_indices

        except Exception as e:
            logger.warning(f"Representative sampling failed: {e}")
            return np.random.choice(len(X), n_samples, replace=False).tolist()

    def _calculate_prediction_confidence(
        self, detector: DetectorProtocol, sample: np.ndarray, X: np.ndarray
    ) -> float:
        """Calculate prediction confidence."""
        try:
            # Get prediction
            prediction = detector.decision_function(sample)[0]

            # Get all predictions for context
            all_predictions = detector.decision_function(X)

            # Calculate percentile rank as confidence measure
            percentile = np.percentile(all_predictions, 50)  # Median

            # Higher deviation from median -> higher confidence
            deviation = abs(prediction - percentile)
            max_deviation = max(
                abs(np.max(all_predictions) - percentile),
                abs(np.min(all_predictions) - percentile),
            )

            if max_deviation > 0:
                confidence = deviation / max_deviation
            else:
                confidence = 0.5

            return float(np.clip(confidence, 0.0, 1.0))

        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5


class GlobalExplanationEngine(ExplanationEngine):
    """Engine for generating global explanations."""

    def __init__(self, enable_permutation: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.enable_permutation = enable_permutation and SKLEARN_AVAILABLE

    async def generate_explanation(
        self,
        detector: DetectorProtocol,
        X: np.ndarray,
        feature_names: list[str],
        config: ExplanationConfig,
    ) -> GlobalExplanation:
        """Generate global model explanation."""
        try:
            feature_importance = {}
            feature_interactions = {}
            method_used = "unknown"

            # Try permutation importance first
            if self.enable_permutation:
                try:
                    perm_importance = await self._compute_permutation_importance(
                        detector, X, feature_names
                    )
                    if perm_importance:
                        feature_importance.update(perm_importance)
                        method_used = "permutation"
                except Exception as e:
                    logger.warning(f"Permutation importance failed: {e}")

            # Try SHAP global explanation
            if SHAP_AVAILABLE and not feature_importance:
                try:
                    shap_importance = await self._compute_shap_global_importance(
                        detector, X, feature_names
                    )
                    if shap_importance:
                        feature_importance.update(shap_importance)
                        method_used = "shap"
                except Exception as e:
                    logger.warning(f"SHAP global explanation failed: {e}")

            # Fallback to variance-based importance
            if not feature_importance:
                feature_importance = self._compute_variance_importance(X, feature_names)
                method_used = "variance"

            # Compute feature interactions
            feature_interactions = self._compute_feature_interactions(X, feature_names)

            # Model summary
            model_summary = {
                "n_features": len(feature_names),
                "n_samples": len(X),
                "prediction_range": {
                    "min": float(np.min(detector.decision_function(X))),
                    "max": float(np.max(detector.decision_function(X))),
                    "mean": float(np.mean(detector.decision_function(X))),
                },
            }

            return GlobalExplanation(
                feature_importance=feature_importance,
                feature_interactions=feature_interactions,
                model_summary=model_summary,
                explanation_method=method_used,
                coverage=0.95,
                reliability=0.85,
            )

        except Exception as e:
            logger.error(f"Global explanation generation failed: {e}")
            return self._create_fallback_global_explanation(feature_names)

    async def _compute_permutation_importance(
        self, detector: DetectorProtocol, X: np.ndarray, feature_names: list[str]
    ) -> dict[str, float] | None:
        """Compute permutation importance."""
        if not SKLEARN_AVAILABLE:
            return None

        try:
            # Create a scorer function
            def scorer(estimator, X, y) -> float:
                predictions = estimator.decision_function(X)
                return -np.std(predictions)

            # Dummy y values
            y = np.zeros(len(X))

            # Compute permutation importance
            perm_importance = permutation_importance(
                detector, X, y, scoring=scorer, n_repeats=10, random_state=42
            )

            return {
                feature_names[i]: float(perm_importance.importances_mean[i])
                for i in range(
                    min(len(feature_names), len(perm_importance.importances_mean))
                )
            }

        except Exception as e:
            logger.warning(f"Permutation importance failed: {e}")
            return None

    async def _compute_shap_global_importance(
        self, detector: DetectorProtocol, X: np.ndarray, feature_names: list[str]
    ) -> dict[str, float] | None:
        """Compute global SHAP importance."""
        if not SHAP_AVAILABLE:
            return None

        try:
            cache_key = f"shap_global_{id(detector)}"

            if cache_key not in self.explainer_cache:

                def predict_fn(x) -> Any:
                    return detector.decision_function(x)

                # Sample for efficiency
                sample_indices = np.random.choice(
                    len(X), min(200, len(X)), replace=False
                )
                X_sample = X[sample_indices]

                explainer = shap.KernelExplainer(predict_fn, X_sample[:50])
                self.explainer_cache[cache_key] = (explainer, X_sample)
            else:
                explainer, X_sample = self.explainer_cache[cache_key]

            # Compute SHAP values for sample
            shap_values = explainer.shap_values(X_sample, nsamples=50)

            # Calculate mean absolute SHAP values
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

            return {
                feature_names[i]: float(mean_abs_shap[i])
                for i in range(min(len(feature_names), len(mean_abs_shap)))
            }

        except Exception as e:
            logger.warning(f"SHAP global importance failed: {e}")
            return None

    def _compute_variance_importance(
        self, X: np.ndarray, feature_names: list[str]
    ) -> dict[str, float]:
        """Compute variance-based feature importance (fallback)."""
        variances = np.var(X, axis=0)
        normalized_variances = (
            variances / np.sum(variances) if np.sum(variances) > 0 else variances
        )

        return {
            feature_names[i]: float(normalized_variances[i])
            for i in range(min(len(feature_names), len(normalized_variances)))
        }

    def _compute_feature_interactions(
        self, X: np.ndarray, feature_names: list[str]
    ) -> dict[str, float]:
        """Compute feature interactions (simplified)."""
        if len(feature_names) < 2:
            return {}

        try:
            # Compute correlation matrix
            corr_matrix = np.corrcoef(X.T)
            interactions = {}

            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    if i < corr_matrix.shape[0] and j < corr_matrix.shape[1]:
                        interaction_key = f"{feature_names[i]}_x_{feature_names[j]}"
                        interactions[interaction_key] = float(abs(corr_matrix[i, j]))

            return interactions

        except Exception as e:
            logger.warning(f"Feature interaction computation failed: {e}")
            return {}

    def _create_fallback_global_explanation(
        self, feature_names: list[str]
    ) -> GlobalExplanation:
        """Create fallback global explanation."""
        equal_importance = 1.0 / len(feature_names) if feature_names else 0.0

        return GlobalExplanation(
            feature_importance=dict.fromkeys(feature_names, equal_importance),
            feature_interactions={},
            model_summary={"note": "Fallback explanation due to computation failures"},
            explanation_method="fallback",
            coverage=0.5,
            reliability=0.3,
        )
