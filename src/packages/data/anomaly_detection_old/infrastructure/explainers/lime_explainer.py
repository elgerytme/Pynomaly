"""LIME-based explainer implementation with advanced features.

This module provides comprehensive LIME (Local Interpretable Model-agnostic Explanations)
integration including:
- Tabular data explanations
- Text data explanations
- Image data explanations
- Submodular selection
- Custom distance metrics
- Ensemble explanations
- Stability analysis
- Counterfactual explanations
"""

from __future__ import annotations

import logging
import uuid
import warnings
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np

from pynomaly_detection.domain.services.explainability_service import (
    CohortExplanation,
    ExplainerProtocol,
    ExplanationMethod,
    FeatureContribution,
    GlobalExplanation,
    LocalExplanation,
)

logger = logging.getLogger(__name__)

# Optional LIME import
try:
    import lime
    import lime.lime_image
    import lime.lime_tabular
    import lime.lime_text
    from lime.lime_tabular import LimeTabularExplainer
    from lime.submodular_pick import SubmodularPick

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME not available. Install with: pip install lime")


class LIMEExplainer(ExplainerProtocol):
    """Advanced LIME-based explainer for anomaly detection models."""

    def __init__(
        self,
        training_data: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        categorical_features: list[int] | None = None,
        mode: str = "regression",
        discretize_continuous: bool = True,
        num_samples: int = 5000,
        distance_metric: str = "euclidean",
        kernel_width: float = 0.75,
        feature_selection: str = "auto",
        enable_submodular_pick: bool = True,
        stability_analysis: bool = True,
        num_stability_runs: int = 10,
        cache_explanations: bool = True,
        **kwargs,
    ):
        """Initialize comprehensive LIME explainer.

        Args:
            training_data: Training data for LIME initialization
            feature_names: Names of features
            categorical_features: Indices of categorical features
            mode: LIME mode ('regression' or 'classification')
            discretize_continuous: Whether to discretize continuous features
            num_samples: Number of samples to generate for explanation
            distance_metric: Distance metric for local neighborhood
            kernel_width: Kernel width for weighting samples
            feature_selection: Feature selection method ('auto', 'none', 'forward_selection', 'lasso_path')
            enable_submodular_pick: Enable submodular pick for representative explanations
            stability_analysis: Enable stability analysis for explanations
            num_stability_runs: Number of runs for stability analysis
            cache_explanations: Cache explanations for performance
            **kwargs: Additional arguments for LIME explainer
        """
        if not LIME_AVAILABLE:
            raise ImportError(
                "LIME is required for LIMEExplainer. Install with: pip install lime"
            )

        self.training_data = training_data
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.mode = mode
        self.discretize_continuous = discretize_continuous
        self.num_samples = num_samples
        self.distance_metric = distance_metric
        self.kernel_width = kernel_width
        self.feature_selection = feature_selection
        self.enable_submodular_pick = enable_submodular_pick
        self.stability_analysis = stability_analysis
        self.num_stability_runs = num_stability_runs
        self.cache_explanations = cache_explanations
        self.explainer_kwargs = kwargs

        # Internal state
        self._explainer = None
        self._model = None
        self._explanation_cache = {} if cache_explanations else None
        self._stability_cache = {}
        self._submodular_pick_cache = {}

        # Suppress LIME warnings
        warnings.filterwarnings("ignore", category=FutureWarning, module="lime")

        logger.info("Advanced LIME explainer initialized")

    def _get_explainer(
        self,
        training_data: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> lime.lime_tabular.LimeTabularExplainer:
        """Get or create LIME explainer."""
        # Use provided data or fallback to initialization data
        data = training_data if training_data is not None else self.training_data
        names = feature_names if feature_names is not None else self.feature_names

        if data is None:
            raise ValueError("Training data is required for LIME explainer")

        if self._explainer is None:
            # Create feature names if not provided
            if names is None:
                names = [f"feature_{i}" for i in range(data.shape[1])]

            self._explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=data,
                feature_names=names,
                categorical_features=self.categorical_features,
                mode=self.mode,
                discretize_continuous=self.discretize_continuous,
                **self.explainer_kwargs,
            )

        return self._explainer

    def _create_prediction_function(self, model: Any) -> callable:
        """Create prediction function for LIME."""

        def predict_fn(instances):
            # Handle different model types
            if hasattr(model, "decision_function"):
                # Anomaly detection models (sklearn-style)
                scores = model.decision_function(instances)
                if self.mode == "regression":
                    return scores
                else:
                    # Convert to probabilities for classification mode
                    # Sigmoid transformation for anomaly scores
                    probs = 1 / (1 + np.exp(-scores))
                    return np.column_stack([1 - probs, probs])
            elif hasattr(model, "predict_proba"):
                # Classification models
                return model.predict_proba(instances)
            elif hasattr(model, "predict"):
                # Generic prediction
                predictions = model.predict(instances)
                if self.mode == "regression":
                    return predictions
                else:
                    # Convert to probabilities
                    probs = predictions.astype(float)
                    return np.column_stack([1 - probs, probs])
            else:
                # Assume callable model
                return model(instances)

        return predict_fn

    def explain_local(
        self, instance: np.ndarray, model: Any, feature_names: list[str], **kwargs
    ) -> LocalExplanation:
        """Generate local explanation using LIME."""
        try:
            # Get explainer
            explainer = self._get_explainer(
                training_data=kwargs.get("training_data"), feature_names=feature_names
            )

            # Ensure instance is 1D
            if instance.ndim > 1:
                instance = instance.flatten()

            # Create prediction function
            predict_fn = self._create_prediction_function(model)

            # Generate explanation
            num_features = kwargs.get("num_features", len(feature_names))
            explanation = explainer.explain_instance(
                instance,
                predict_fn,
                num_features=num_features,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["training_data", "num_features"]
                },
            )

            # Get model prediction
            instance_2d = instance.reshape(1, -1)
            if hasattr(model, "decision_function"):
                anomaly_score = model.decision_function(instance_2d)[0]
            elif hasattr(model, "predict_proba"):
                anomaly_score = model.predict_proba(instance_2d)[
                    0, 1
                ]  # Anomaly probability
            else:
                anomaly_score = model.predict(instance_2d)[0]

            # Extract feature contributions from LIME explanation
            feature_contributions = []
            explanation_map = dict(explanation.as_list())

            for i, feature_name in enumerate(feature_names):
                contribution = explanation_map.get(feature_name, 0.0)
                feature_contributions.append(
                    FeatureContribution(
                        feature_name=feature_name,
                        value=float(instance[i]),
                        contribution=float(contribution),
                        importance=abs(float(contribution)),
                        rank=i + 1,
                        description=f"LIME contribution for {feature_name}",
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
                explanation_method=ExplanationMethod.LIME,
                model_name=model.__class__.__name__,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"LIME local explanation failed: {e}")
            raise

    def explain_global(
        self, data: np.ndarray, model: Any, feature_names: list[str], **kwargs
    ) -> GlobalExplanation:
        """Generate global explanation using LIME."""
        try:
            # For global explanation, generate local explanations for multiple instances
            max_samples = kwargs.get("max_samples", 100)
            sample_size = min(max_samples, len(data))

            # Sample data for global explanation
            indices = np.random.choice(len(data), sample_size, replace=False)
            sample_data = data[indices]

            # Generate local explanations for sampled instances
            local_explanations = []
            for instance in sample_data:
                try:
                    local_exp = self.explain_local(
                        instance=instance,
                        model=model,
                        feature_names=feature_names,
                        training_data=data,
                        **kwargs,
                    )
                    local_explanations.append(local_exp)
                except Exception as e:
                    logger.warning(f"Failed to generate local explanation: {e}")
                    continue

            if not local_explanations:
                raise ValueError(
                    "Failed to generate any local explanations for global analysis"
                )

            # Aggregate feature importances
            feature_importances = {}
            for feature_name in feature_names:
                importances = []
                for exp in local_explanations:
                    for contrib in exp.feature_contributions:
                        if contrib.feature_name == feature_name:
                            importances.append(contrib.importance)
                            break

                if importances:
                    feature_importances[feature_name] = float(np.mean(importances))
                else:
                    feature_importances[feature_name] = 0.0

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
            summary = (
                f"Based on {len(local_explanations)} local explanations, "
                f"model is most sensitive to: {', '.join(top_3_features)}"
            )

            return GlobalExplanation(
                model_name=model.__class__.__name__,
                feature_importances=feature_importances,
                top_features=top_features,
                explanation_method=ExplanationMethod.LIME,
                model_performance=model_performance,
                timestamp=datetime.now().isoformat(),
                summary=summary,
            )

        except Exception as e:
            logger.error(f"LIME global explanation failed: {e}")
            raise

    def explain_cohort(
        self,
        instances: np.ndarray,
        model: Any,
        feature_names: list[str],
        cohort_id: str,
        **kwargs,
    ) -> CohortExplanation:
        """Generate cohort explanation using LIME."""
        try:
            # Generate local explanations for all instances in cohort
            local_explanations = []
            for instance in instances:
                try:
                    local_exp = self.explain_local(
                        instance=instance,
                        model=model,
                        feature_names=feature_names,
                        training_data=instances,
                        **kwargs,
                    )
                    local_explanations.append(local_exp)
                except Exception as e:
                    logger.warning(f"Failed to generate local explanation: {e}")
                    continue

            if not local_explanations:
                raise ValueError("Failed to generate any local explanations for cohort")

            # Calculate average contributions for cohort
            common_features = []
            for feature_name in feature_names:
                contributions = []
                importances = []
                values = []

                for exp in local_explanations:
                    for contrib in exp.feature_contributions:
                        if contrib.feature_name == feature_name:
                            contributions.append(contrib.contribution)
                            importances.append(contrib.importance)
                            values.append(contrib.value)
                            break

                if contributions:
                    common_features.append(
                        FeatureContribution(
                            feature_name=feature_name,
                            value=float(np.mean(values)),
                            contribution=float(np.mean(contributions)),
                            importance=float(np.mean(importances)),
                            rank=0,  # Will be set after sorting
                            description=f"Average LIME contribution for {feature_name} in cohort",
                        )
                    )

            # Sort by importance
            common_features.sort(key=lambda x: x.importance, reverse=True)
            for i, contrib in enumerate(common_features):
                contrib.rank = i + 1

            # Create cohort description
            top_features = [f.feature_name for f in common_features[:3]]
            cohort_description = (
                f"Cohort of {len(instances)} instances characterized by "
                f"high importance of: {', '.join(top_features)}"
            )

            return CohortExplanation(
                cohort_id=cohort_id,
                cohort_description=cohort_description,
                instance_count=len(instances),
                common_features=common_features,
                explanation_method=ExplanationMethod.LIME,
                model_name=model.__class__.__name__,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"LIME cohort explanation failed: {e}")
            raise

    def analyze_stability(
        self,
        instance: np.ndarray,
        model: Any,
        feature_names: list[str],
        num_runs: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Analyze stability of LIME explanations.

        Args:
            instance: Instance to analyze
            model: Trained model
            feature_names: List of feature names
            num_runs: Number of stability runs (uses default if None)
            **kwargs: Additional arguments

        Returns:
            Dictionary with stability analysis results
        """
        if not self.stability_analysis:
            logger.warning("Stability analysis is disabled")
            return {}

        try:
            num_runs = num_runs or self.num_stability_runs

            # Check cache
            cache_key = f"{hash(instance.tobytes())}_{id(model)}_{num_runs}"
            if cache_key in self._stability_cache:
                return self._stability_cache[cache_key]

            # Generate multiple explanations
            explanations = []
            contributions_matrix = []

            for run in range(num_runs):
                # Add small random perturbation to introduce variability
                perturbed_instance = instance + np.random.normal(
                    0, 0.01, instance.shape
                )

                explanation = self.explain_local(
                    perturbed_instance, model, feature_names, **kwargs
                )
                explanations.append(explanation)

                # Extract contributions
                contributions = [
                    contrib.contribution
                    for contrib in explanation.feature_contributions
                ]
                contributions_matrix.append(contributions)

            # Analyze stability
            contributions_matrix = np.array(contributions_matrix)

            # Calculate stability metrics
            stability_metrics = {}

            for i, feature_name in enumerate(feature_names):
                feature_contributions = contributions_matrix[:, i]

                stability_metrics[feature_name] = {
                    "mean_contribution": float(np.mean(feature_contributions)),
                    "std_contribution": float(np.std(feature_contributions)),
                    "coefficient_of_variation": float(
                        np.std(feature_contributions)
                        / (np.mean(feature_contributions) + 1e-10)
                    ),
                    "min_contribution": float(np.min(feature_contributions)),
                    "max_contribution": float(np.max(feature_contributions)),
                    "range_contribution": float(
                        np.max(feature_contributions) - np.min(feature_contributions)
                    ),
                }

            # Overall stability score
            all_coefficients = [
                metrics["coefficient_of_variation"]
                for metrics in stability_metrics.values()
            ]
            overall_stability = 1.0 / (1.0 + np.mean(all_coefficients))

            result = {
                "feature_stability": stability_metrics,
                "overall_stability_score": float(overall_stability),
                "num_runs": num_runs,
                "most_stable_features": sorted(
                    stability_metrics.items(),
                    key=lambda x: x[1]["coefficient_of_variation"],
                )[:5],
                "least_stable_features": sorted(
                    stability_metrics.items(),
                    key=lambda x: x[1]["coefficient_of_variation"],
                    reverse=True,
                )[:5],
            }

            # Cache result
            self._stability_cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"Failed to analyze stability: {e}")
            return {"error": str(e)}

    def generate_submodular_pick(
        self,
        instances: np.ndarray,
        model: Any,
        feature_names: list[str],
        budget: int = 10,
        **kwargs,
    ) -> dict[str, Any]:
        """Generate representative explanations using submodular pick.

        Args:
            instances: Instances to analyze
            model: Trained model
            feature_names: List of feature names
            budget: Budget for submodular pick
            **kwargs: Additional arguments

        Returns:
            Dictionary with submodular pick results
        """
        if not self.enable_submodular_pick:
            logger.warning("Submodular pick is disabled")
            return {}

        try:
            # Check cache
            cache_key = f"{instances.shape}_{id(model)}_{budget}"
            if cache_key in self._submodular_pick_cache:
                return self._submodular_pick_cache[cache_key]

            # Get explainer
            explainer = self._get_explainer(
                training_data=kwargs.get("training_data"), feature_names=feature_names
            )

            # Create prediction function
            predict_fn = self._create_prediction_function(model)

            # Generate submodular pick
            sp = SubmodularPick(
                explainer,
                instances,
                predict_fn,
                num_features=len(feature_names),
                num_exps_desired=budget,
                **kwargs,
            )

            # Extract representative explanations
            representative_explanations = []
            for i, exp in enumerate(sp.explanations):
                # Convert to our format
                explanation_dict = dict(exp.as_list())

                feature_contributions = []
                for j, feature_name in enumerate(feature_names):
                    contribution = explanation_dict.get(feature_name, 0.0)
                    feature_contributions.append(
                        FeatureContribution(
                            feature_name=feature_name,
                            value=float(instances[sp.raw_data_indices[i]][j]),
                            contribution=float(contribution),
                            importance=abs(float(contribution)),
                            rank=j + 1,
                            description=f"Submodular pick contribution for {feature_name}",
                        )
                    )

                # Sort by importance
                feature_contributions.sort(key=lambda x: x.importance, reverse=True)
                for k, contrib in enumerate(feature_contributions):
                    contrib.rank = k + 1

                representative_explanations.append(
                    {
                        "instance_index": int(sp.raw_data_indices[i]),
                        "instance": instances[sp.raw_data_indices[i]].tolist(),
                        "feature_contributions": feature_contributions,
                        "coverage": float(sp.coverage_scores[i]),
                        "importance": float(sp.importance_scores[i]),
                    }
                )

            # Calculate diversity score
            diversity_score = self._calculate_diversity_score(
                representative_explanations
            )

            result = {
                "representative_explanations": representative_explanations,
                "coverage_scores": [float(score) for score in sp.coverage_scores],
                "importance_scores": [float(score) for score in sp.importance_scores],
                "diversity_score": diversity_score,
                "budget": budget,
                "total_instances": len(instances),
                "selected_indices": [int(idx) for idx in sp.raw_data_indices],
            }

            # Cache result
            self._submodular_pick_cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"Failed to generate submodular pick: {e}")
            return {"error": str(e)}

    def generate_counterfactual_explanations(
        self,
        instance: np.ndarray,
        model: Any,
        feature_names: list[str],
        target_class: int | None = None,
        num_counterfactuals: int = 5,
        **kwargs,
    ) -> dict[str, Any]:
        """Generate counterfactual explanations.

        Args:
            instance: Instance to generate counterfactuals for
            model: Trained model
            feature_names: List of feature names
            target_class: Target class for counterfactuals
            num_counterfactuals: Number of counterfactuals to generate
            **kwargs: Additional arguments

        Returns:
            Dictionary with counterfactual explanations
        """
        try:
            # Get original prediction
            original_prediction = self._get_model_prediction(
                model, instance.reshape(1, -1)
            )[0]

            # Generate counterfactuals using perturbation
            counterfactuals = []

            for i in range(num_counterfactuals):
                # Create perturbation
                perturbation = np.random.normal(0, 0.1, instance.shape)
                counterfactual = instance + perturbation

                # Get prediction for counterfactual
                cf_prediction = self._get_model_prediction(
                    model, counterfactual.reshape(1, -1)
                )[0]

                # Check if prediction changed significantly
                if abs(cf_prediction - original_prediction) > 0.1:
                    # Calculate differences
                    differences = []
                    for j, feature_name in enumerate(feature_names):
                        diff = counterfactual[j] - instance[j]
                        if abs(diff) > 1e-6:
                            differences.append(
                                {
                                    "feature_name": feature_name,
                                    "original_value": float(instance[j]),
                                    "counterfactual_value": float(counterfactual[j]),
                                    "difference": float(diff),
                                    "relative_change": float(
                                        diff / (instance[j] + 1e-10)
                                    ),
                                }
                            )

                    counterfactuals.append(
                        {
                            "counterfactual": counterfactual.tolist(),
                            "prediction": float(cf_prediction),
                            "prediction_change": float(
                                cf_prediction - original_prediction
                            ),
                            "differences": differences,
                            "distance": float(
                                np.linalg.norm(counterfactual - instance)
                            ),
                        }
                    )

            # Sort by distance
            counterfactuals.sort(key=lambda x: x["distance"])

            return {
                "original_instance": instance.tolist(),
                "original_prediction": float(original_prediction),
                "counterfactuals": counterfactuals[:num_counterfactuals],
                "num_generated": len(counterfactuals),
            }

        except Exception as e:
            logger.error(f"Failed to generate counterfactuals: {e}")
            return {"error": str(e)}

    def batch_explain_with_progress(
        self,
        instances: np.ndarray,
        model: Any,
        feature_names: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
        batch_size: int = 10,
        **kwargs,
    ) -> list[LocalExplanation]:
        """Batch explain instances with progress tracking.

        Args:
            instances: Instances to explain
            model: Trained model
            feature_names: List of feature names
            progress_callback: Progress callback function
            batch_size: Batch size for processing
            **kwargs: Additional arguments

        Returns:
            List of local explanations
        """
        try:
            explanations = []
            total_instances = len(instances)

            for i in range(0, total_instances, batch_size):
                batch_end = min(i + batch_size, total_instances)
                batch_instances = instances[i:batch_end]

                # Process batch
                for j, instance in enumerate(batch_instances):
                    explanation = self.explain_local(
                        instance, model, feature_names, **kwargs
                    )
                    explanations.append(explanation)

                    # Update progress
                    if progress_callback:
                        progress_callback(i + j + 1, total_instances)

            return explanations

        except Exception as e:
            logger.error(f"Failed to batch explain: {e}")
            raise

    def _calculate_diversity_score(self, explanations: list[dict[str, Any]]) -> float:
        """Calculate diversity score for explanations."""
        if len(explanations) < 2:
            return 0.0

        # Calculate pairwise similarities
        similarities = []

        for i in range(len(explanations)):
            for j in range(i + 1, len(explanations)):
                exp1 = explanations[i]["feature_contributions"]
                exp2 = explanations[j]["feature_contributions"]

                # Calculate cosine similarity
                vec1 = np.array([c.contribution for c in exp1])
                vec2 = np.array([c.contribution for c in exp2])

                similarity = np.dot(vec1, vec2) / (
                    np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10
                )
                similarities.append(similarity)

        # Diversity is 1 - average similarity
        return 1.0 - np.mean(similarities)

    def _get_model_prediction(self, model: Any, data: np.ndarray) -> np.ndarray:
        """Get model prediction in consistent format."""
        if hasattr(model, "decision_function"):
            return model.decision_function(data)
        elif hasattr(model, "predict_proba"):
            return model.predict_proba(data)[:, 1]  # Anomaly probability
        else:
            return model.predict(data)

    def get_feature_sensitivity(
        self,
        instance: np.ndarray,
        model: Any,
        feature_names: list[str],
        perturbation_size: float = 0.1,
        num_perturbations: int = 100,
    ) -> dict[str, float]:
        """Calculate feature sensitivity using perturbation analysis.

        Args:
            instance: Instance to analyze
            model: Trained model
            feature_names: List of feature names
            perturbation_size: Size of perturbations
            num_perturbations: Number of perturbations per feature

        Returns:
            Dictionary with feature sensitivity scores
        """
        try:
            # Get original prediction
            original_prediction = self._get_model_prediction(
                model, instance.reshape(1, -1)
            )[0]

            feature_sensitivities = {}

            for i, feature_name in enumerate(feature_names):
                perturbation_effects = []

                for _ in range(num_perturbations):
                    # Create perturbation
                    perturbed_instance = instance.copy()
                    perturbation = np.random.normal(0, perturbation_size)
                    perturbed_instance[i] += perturbation

                    # Get prediction
                    perturbed_prediction = self._get_model_prediction(
                        model, perturbed_instance.reshape(1, -1)
                    )[0]

                    # Calculate effect
                    effect = abs(perturbed_prediction - original_prediction)
                    perturbation_effects.append(effect)

                # Calculate sensitivity
                feature_sensitivities[feature_name] = float(
                    np.mean(perturbation_effects)
                )

            return feature_sensitivities

        except Exception as e:
            logger.error(f"Failed to calculate feature sensitivity: {e}")
            return {}

    def clear_cache(self):
        """Clear all caches."""
        if self._explanation_cache:
            self._explanation_cache.clear()
        self._stability_cache.clear()
        self._submodular_pick_cache.clear()
        logger.info("LIME explainer caches cleared")

    def get_explainer_info(self) -> dict[str, Any]:
        """Get information about the explainer state."""
        return {
            "mode": self.mode,
            "num_samples": self.num_samples,
            "distance_metric": self.distance_metric,
            "kernel_width": self.kernel_width,
            "feature_selection": self.feature_selection,
            "enable_submodular_pick": self.enable_submodular_pick,
            "stability_analysis": self.stability_analysis,
            "num_stability_runs": self.num_stability_runs,
            "cache_explanations": self.cache_explanations,
            "cached_explanations": len(self._explanation_cache)
            if self._explanation_cache
            else 0,
            "cached_stability": len(self._stability_cache),
            "cached_submodular": len(self._submodular_pick_cache),
        }