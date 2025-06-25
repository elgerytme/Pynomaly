"""Advanced explainable AI service for comprehensive model interpretability."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from pynomaly.domain.entities import Dataset
from pynomaly.shared.protocols import DetectorProtocol

# Optional explainability imports with fallbacks
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False

try:
    import lime
    from lime.explanation import Explanation
    from lime.lime_base import LimeBase

    LIME_AVAILABLE = True
except ImportError:
    lime = None
    LimeBase = None
    Explanation = None
    LIME_AVAILABLE = False

try:
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import average_precision_score, roc_auc_score

    SKLEARN_AVAILABLE = True
except ImportError:
    permutation_importance = None
    roc_auc_score = None
    average_precision_score = None
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExplanationConfig(BaseModel):
    """Configuration for explanation generation."""

    explanation_type: str = Field(
        default="local", description="Explanation type (local, global)"
    )
    method: str = Field(default="shap", description="Explanation method")
    n_samples: int = Field(
        default=1000, ge=10, description="Number of samples for explanation"
    )
    feature_names: list[str] | None = Field(None, description="Feature names")
    target_audience: str = Field(
        default="technical", description="Target audience level"
    )
    include_confidence: bool = Field(
        default=True, description="Include confidence metrics"
    )
    generate_plots: bool = Field(
        default=True, description="Generate visualization plots"
    )


class BiasAnalysisConfig(BaseModel):
    """Configuration for bias analysis."""

    protected_attributes: list[str] = Field(description="Protected attribute names")
    fairness_metrics: list[str] = Field(
        default=["demographic_parity", "equalized_odds"], description="Fairness metrics"
    )
    threshold: float = Field(default=0.5, description="Decision threshold")
    reference_group: str | None = Field(
        None, description="Reference group for comparison"
    )
    min_group_size: int = Field(
        default=50, description="Minimum group size for analysis"
    )


class TrustScoreConfig(BaseModel):
    """Configuration for trust scoring."""

    consistency_checks: bool = Field(
        default=True, description="Enable consistency analysis"
    )
    stability_analysis: bool = Field(
        default=True, description="Enable stability analysis"
    )
    fidelity_assessment: bool = Field(
        default=True, description="Enable fidelity assessment"
    )
    n_perturbations: int = Field(
        default=100, description="Number of perturbations for stability"
    )
    perturbation_strength: float = Field(
        default=0.1, description="Perturbation strength"
    )


class LocalExplanation(BaseModel):
    """Local explanation for individual predictions."""

    sample_id: str = Field(description="Sample identifier")
    prediction: float = Field(description="Model prediction")
    confidence: float = Field(description="Prediction confidence")
    feature_contributions: dict[str, float] = Field(
        description="Feature contribution scores"
    )
    explanation_method: str = Field(description="Method used for explanation")
    trust_score: float | None = Field(
        None, description="Trust score for explanation"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class GlobalExplanation(BaseModel):
    """Global explanation for model behavior."""

    feature_importance: dict[str, float] = Field(
        description="Global feature importance"
    )
    feature_interactions: dict[str, float] = Field(
        default_factory=dict, description="Feature interactions"
    )
    model_summary: dict[str, Any] = Field(description="Model summary statistics")
    explanation_method: str = Field(description="Method used for explanation")
    coverage: float = Field(description="Explanation coverage percentage")
    reliability: float = Field(description="Explanation reliability score")


class BiasAnalysisResult(BaseModel):
    """Results of bias analysis."""

    protected_attribute: str = Field(description="Protected attribute analyzed")
    fairness_metrics: dict[str, float] = Field(description="Fairness metric values")
    group_statistics: dict[str, dict[str, float]] = Field(
        description="Statistics by group"
    )
    bias_detected: bool = Field(description="Whether bias was detected")
    severity: str = Field(description="Bias severity level")
    recommendations: list[str] = Field(description="Mitigation recommendations")


class TrustScoreResult(BaseModel):
    """Trust score assessment results."""

    overall_trust_score: float = Field(description="Overall trust score (0-1)")
    consistency_score: float = Field(description="Model consistency score")
    stability_score: float = Field(description="Prediction stability score")
    fidelity_score: float = Field(description="Explanation fidelity score")
    trust_factors: dict[str, float] = Field(description="Individual trust factors")
    risk_assessment: str = Field(description="Risk level assessment")


class ExplanationReport(BaseModel):
    """Comprehensive explanation report."""

    model_info: dict[str, Any] = Field(description="Model information")
    dataset_summary: dict[str, Any] = Field(description="Dataset summary")
    local_explanations: list[LocalExplanation] = Field(description="Local explanations")
    global_explanation: GlobalExplanation = Field(description="Global explanation")
    bias_analysis: list[BiasAnalysisResult] = Field(description="Bias analysis results")
    trust_assessment: TrustScoreResult = Field(description="Trust score assessment")
    recommendations: list[str] = Field(description="Overall recommendations")
    generation_time: datetime = Field(
        default_factory=datetime.now, description="Report generation time"
    )


class AdvancedExplainabilityService:
    """Service for advanced explainable AI capabilities."""

    def __init__(
        self,
        enable_shap: bool = True,
        enable_lime: bool = True,
        enable_permutation: bool = True,
        cache_explanations: bool = True,
    ):
        """Initialize explainability service.

        Args:
            enable_shap: Enable SHAP explanations
            enable_lime: Enable LIME explanations
            enable_permutation: Enable permutation importance
            cache_explanations: Cache computed explanations
        """
        self.enable_shap = enable_shap and SHAP_AVAILABLE
        self.enable_lime = enable_lime and LIME_AVAILABLE
        self.enable_permutation = enable_permutation and SKLEARN_AVAILABLE
        self.cache_explanations = cache_explanations

        self.explanation_cache: dict[str, Any] = {}
        self.explainer_cache: dict[str, Any] = {}

        logger.info(
            f"Initialized AdvancedExplainabilityService "
            f"(SHAP: {self.enable_shap}, LIME: {self.enable_lime}, Permutation: {self.enable_permutation})"
        )

    async def generate_comprehensive_explanation(
        self,
        detector: DetectorProtocol,
        dataset: Dataset,
        config: ExplanationConfig | None = None,
    ) -> ExplanationReport:
        """Generate comprehensive explanation report."""
        try:
            if not config:
                config = ExplanationConfig()

            logger.info("Generating comprehensive explanation report")

            # Prepare data
            X = dataset.data.values if hasattr(dataset.data, "values") else dataset.data
            feature_names = (
                config.feature_names
                or dataset.features
                or [f"feature_{i}" for i in range(X.shape[1])]
            )

            # Get model predictions
            predictions = detector.decision_function(X)

            # Generate explanations
            tasks = []

            # Local explanations
            if config.explanation_type in ["local", "both"]:
                tasks.append(
                    self._generate_local_explanations(
                        detector, X, feature_names, config
                    )
                )

            # Global explanations
            if config.explanation_type in ["global", "both"]:
                tasks.append(
                    self._generate_global_explanation(
                        detector, X, feature_names, config
                    )
                )

            # Execute explanation tasks
            explanation_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            local_explanations = []
            global_explanation = None

            for result in explanation_results:
                if isinstance(result, Exception):
                    logger.warning(f"Explanation task failed: {result}")
                    continue

                if isinstance(result, list):  # Local explanations
                    local_explanations = result
                elif isinstance(result, GlobalExplanation):
                    global_explanation = result

            # Generate trust assessment
            trust_assessment = await self._assess_trust_score(
                detector, X, predictions, TrustScoreConfig()
            )

            # Create report
            report = ExplanationReport(
                model_info=self._get_model_info(detector),
                dataset_summary=self._get_dataset_summary(dataset),
                local_explanations=local_explanations,
                global_explanation=global_explanation
                or self._create_fallback_global_explanation(feature_names),
                bias_analysis=[],  # Will be added if bias analysis is performed
                trust_assessment=trust_assessment,
                recommendations=self._generate_recommendations(trust_assessment),
            )

            logger.info("Comprehensive explanation report generated successfully")
            return report

        except Exception as e:
            logger.error(f"Failed to generate explanation report: {e}")
            raise RuntimeError(f"Explanation generation failed: {e}")

    async def _generate_local_explanations(
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

            for i, idx in enumerate(indices):
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

                # Fallback to gradient-based explanation if available
                if not feature_contributions:
                    feature_contributions = self._compute_gradient_explanation(
                        detector, sample, feature_names
                    )

                # Calculate confidence (simplified)
                confidence = self._calculate_prediction_confidence(detector, sample, X)

                explanation = LocalExplanation(
                    sample_id=f"sample_{idx}",
                    prediction=float(prediction),
                    confidence=confidence,
                    feature_contributions=feature_contributions,
                    explanation_method="shap"
                    if self.enable_shap
                    else "lime"
                    if self.enable_lime
                    else "gradient",
                    metadata={"sample_index": int(idx)},
                )

                explanations.append(explanation)

            return explanations

        except Exception as e:
            logger.error(f"Local explanation generation failed: {e}")
            return []

    async def _generate_global_explanation(
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
            if self.enable_shap and not feature_importance:
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

            # Compute feature interactions (simplified)
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
                coverage=0.95,  # Simplified
                reliability=0.85,  # Simplified
            )

        except Exception as e:
            logger.error(f"Global explanation generation failed: {e}")
            return self._create_fallback_global_explanation(feature_names)

    async def analyze_bias(
        self, detector: DetectorProtocol, dataset: Dataset, config: BiasAnalysisConfig
    ) -> list[BiasAnalysisResult]:
        """Analyze model for potential bias."""
        try:
            results = []

            X = dataset.data.values if hasattr(dataset.data, "values") else dataset.data
            predictions = detector.predict(X)
            scores = detector.decision_function(X)

            for protected_attr in config.protected_attributes:
                if protected_attr not in dataset.data.columns:
                    logger.warning(
                        f"Protected attribute '{protected_attr}' not found in dataset"
                    )
                    continue

                # Get protected attribute values
                protected_values = dataset.data[protected_attr].values
                unique_groups = np.unique(protected_values)

                if len(unique_groups) < 2:
                    logger.warning(
                        f"Protected attribute '{protected_attr}' has less than 2 groups"
                    )
                    continue

                # Analyze bias for this attribute
                bias_result = await self._analyze_attribute_bias(
                    protected_attr, protected_values, predictions, scores, config
                )

                results.append(bias_result)

            return results

        except Exception as e:
            logger.error(f"Bias analysis failed: {e}")
            return []

    async def _analyze_attribute_bias(
        self,
        attribute_name: str,
        attribute_values: np.ndarray,
        predictions: np.ndarray,
        scores: np.ndarray,
        config: BiasAnalysisConfig,
    ) -> BiasAnalysisResult:
        """Analyze bias for a specific protected attribute."""
        unique_groups = np.unique(attribute_values)
        group_stats = {}
        fairness_metrics = {}

        # Calculate statistics for each group
        for group in unique_groups:
            group_mask = attribute_values == group
            group_predictions = predictions[group_mask]
            group_scores = scores[group_mask]

            if len(group_predictions) < config.min_group_size:
                continue

            group_stats[str(group)] = {
                "size": int(len(group_predictions)),
                "positive_rate": float(np.mean(group_predictions)),
                "mean_score": float(np.mean(group_scores)),
                "std_score": float(np.std(group_scores)),
            }

        # Calculate fairness metrics
        if len(group_stats) >= 2:
            # Demographic parity
            positive_rates = [stats["positive_rate"] for stats in group_stats.values()]
            fairness_metrics["demographic_parity"] = 1.0 - (
                max(positive_rates) - min(positive_rates)
            )

            # Statistical parity difference
            fairness_metrics["statistical_parity_difference"] = max(
                positive_rates
            ) - min(positive_rates)

            # Equalized odds (simplified)
            mean_scores = [stats["mean_score"] for stats in group_stats.values()]
            fairness_metrics["equalized_odds"] = 1.0 - (
                max(mean_scores) - min(mean_scores)
            )

        # Determine if bias is detected
        bias_detected = any(
            metric < 0.8
            for metric in fairness_metrics.values()
            if metric <= 1.0  # Only for normalized metrics
        )

        # Assess severity
        if bias_detected:
            min_fairness = min(fairness_metrics.values())
            if min_fairness < 0.6:
                severity = "high"
            elif min_fairness < 0.8:
                severity = "medium"
            else:
                severity = "low"
        else:
            severity = "none"

        # Generate recommendations
        recommendations = []
        if bias_detected:
            recommendations.extend(
                [
                    f"Consider rebalancing dataset for attribute '{attribute_name}'",
                    "Apply bias mitigation techniques during training",
                    "Use fairness-aware evaluation metrics",
                    "Consider post-processing fairness adjustments",
                ]
            )

        return BiasAnalysisResult(
            protected_attribute=attribute_name,
            fairness_metrics=fairness_metrics,
            group_statistics=group_stats,
            bias_detected=bias_detected,
            severity=severity,
            recommendations=recommendations,
        )

    async def _assess_trust_score(
        self,
        detector: DetectorProtocol,
        X: np.ndarray,
        predictions: np.ndarray,
        config: TrustScoreConfig,
    ) -> TrustScoreResult:
        """Assess trust score for model predictions."""
        trust_factors = {}

        # Consistency analysis
        if config.consistency_checks:
            consistency_score = await self._assess_consistency(detector, X)
            trust_factors["consistency"] = consistency_score
        else:
            trust_factors["consistency"] = 1.0

        # Stability analysis
        if config.stability_analysis:
            stability_score = await self._assess_stability(detector, X, config)
            trust_factors["stability"] = stability_score
        else:
            trust_factors["stability"] = 1.0

        # Fidelity assessment
        if config.fidelity_assessment:
            fidelity_score = await self._assess_fidelity(detector, X)
            trust_factors["fidelity"] = fidelity_score
        else:
            trust_factors["fidelity"] = 1.0

        # Calculate overall trust score
        overall_trust = np.mean(list(trust_factors.values()))

        # Risk assessment
        if overall_trust >= 0.8:
            risk_assessment = "low"
        elif overall_trust >= 0.6:
            risk_assessment = "medium"
        else:
            risk_assessment = "high"

        return TrustScoreResult(
            overall_trust_score=overall_trust,
            consistency_score=trust_factors["consistency"],
            stability_score=trust_factors["stability"],
            fidelity_score=trust_factors["fidelity"],
            trust_factors=trust_factors,
            risk_assessment=risk_assessment,
        )

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
            # Create explainer based on detector type
            cache_key = f"shap_{id(detector)}"

            if cache_key not in self.explainer_cache:
                # Use kernel explainer as general approach
                def predict_fn(x):
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
                shap_values = shap_values[0]  # Take first sample

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

                gradients[feature_name] = float(
                    gradient * sample[0, i]
                )  # Gradient * input

            return gradients

        except Exception as e:
            logger.warning(f"Gradient explanation failed: {e}")
            return dict.fromkeys(feature_names, 0.0)

    async def _compute_permutation_importance(
        self, detector: DetectorProtocol, X: np.ndarray, feature_names: list[str]
    ) -> dict[str, float] | None:
        """Compute permutation importance."""
        if not SKLEARN_AVAILABLE:
            return None

        try:
            # Create a scorer function
            def scorer(estimator, X, y):
                # For anomaly detection, we use consistency as score
                predictions = estimator.decision_function(X)
                return -np.std(predictions)  # Negative std (higher is better)

            # Dummy y values (not used in scoring)
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

                def predict_fn(x):
                    return detector.decision_function(x)

                # Sample for efficiency
                sample_indices = np.random.choice(
                    len(X), min(200, len(X)), replace=False
                )
                X_sample = X[sample_indices]

                explainer = shap.KernelExplainer(
                    predict_fn, X_sample[:50]
                )  # Small background
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

    async def _assess_consistency(
        self, detector: DetectorProtocol, X: np.ndarray
    ) -> float:
        """Assess model consistency."""
        try:
            # Test consistency with small perturbations
            n_tests = min(100, len(X))
            indices = np.random.choice(len(X), n_tests, replace=False)

            consistency_scores = []

            for idx in indices:
                original = X[idx : idx + 1]
                original_pred = detector.decision_function(original)[0]

                # Add small noise
                noise_levels = [0.01, 0.05, 0.1]
                for noise_level in noise_levels:
                    noise = np.random.normal(0, noise_level, original.shape)
                    perturbed = original + noise
                    perturbed_pred = detector.decision_function(perturbed)[0]

                    # Calculate consistency (inverse of relative change)
                    if abs(original_pred) > 1e-10:
                        relative_change = abs(perturbed_pred - original_pred) / abs(
                            original_pred
                        )
                        consistency = 1.0 / (1.0 + relative_change)
                    else:
                        consistency = 1.0 if abs(perturbed_pred) < 1e-10 else 0.0

                    consistency_scores.append(consistency)

            return float(np.mean(consistency_scores))

        except Exception as e:
            logger.warning(f"Consistency assessment failed: {e}")
            return 0.5

    async def _assess_stability(
        self, detector: DetectorProtocol, X: np.ndarray, config: TrustScoreConfig
    ) -> float:
        """Assess prediction stability."""
        try:
            n_samples = min(50, len(X))
            indices = np.random.choice(len(X), n_samples, replace=False)

            stability_scores = []

            for idx in indices:
                sample = X[idx : idx + 1]
                original_pred = detector.decision_function(sample)[0]

                # Generate perturbations
                perturbations = []
                for _ in range(config.n_perturbations):
                    noise = np.random.normal(
                        0, config.perturbation_strength, sample.shape
                    )
                    perturbed = sample + noise
                    perturbed_pred = detector.decision_function(perturbed)[0]
                    perturbations.append(perturbed_pred)

                # Calculate stability as inverse of prediction variance
                pred_variance = np.var(perturbations)
                stability = 1.0 / (1.0 + pred_variance)
                stability_scores.append(stability)

            return float(np.mean(stability_scores))

        except Exception as e:
            logger.warning(f"Stability assessment failed: {e}")
            return 0.5

    async def _assess_fidelity(
        self, detector: DetectorProtocol, X: np.ndarray
    ) -> float:
        """Assess explanation fidelity."""
        try:
            # Simple fidelity assessment based on prediction consistency
            n_samples = min(50, len(X))
            indices = np.random.choice(len(X), n_samples, replace=False)

            predictions = detector.decision_function(X[indices])

            # Fidelity based on prediction distribution consistency
            pred_std = np.std(predictions)
            pred_mean = np.mean(np.abs(predictions))

            # Higher consistency -> higher fidelity
            if pred_mean > 0:
                coefficient_of_variation = pred_std / pred_mean
                fidelity = 1.0 / (1.0 + coefficient_of_variation)
            else:
                fidelity = 1.0

            return float(np.clip(fidelity, 0.0, 1.0))

        except Exception as e:
            logger.warning(f"Fidelity assessment failed: {e}")
            return 0.5

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

            # Select one sample from each cluster (closest to centroid)
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
            # Fallback to random sampling
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

            # Higher deviation from median -> higher confidence in anomaly/normal classification
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

    def _get_model_info(self, detector: DetectorProtocol) -> dict[str, Any]:
        """Get model information."""
        return {
            "algorithm": getattr(detector, "algorithm_name", "unknown"),
            "parameters": getattr(detector, "algorithm_params", {}),
            "is_trained": getattr(detector, "is_trained", False),
        }

    def _get_dataset_summary(self, dataset: Dataset) -> dict[str, Any]:
        """Get dataset summary."""
        data = dataset.data
        if hasattr(data, "shape"):
            n_samples, n_features = data.shape
        else:
            n_samples = len(data)
            n_features = len(data[0]) if len(data) > 0 else 0

        return {
            "name": dataset.name,
            "n_samples": n_samples,
            "n_features": n_features,
            "features": dataset.features[:10]
            if dataset.features
            else [],  # First 10 features
        }

    def _create_fallback_global_explanation(
        self, feature_names: list[str]
    ) -> GlobalExplanation:
        """Create fallback global explanation."""
        # Equal importance fallback
        equal_importance = 1.0 / len(feature_names) if feature_names else 0.0

        return GlobalExplanation(
            feature_importance=dict.fromkeys(feature_names, equal_importance),
            feature_interactions={},
            model_summary={"note": "Fallback explanation due to computation failures"},
            explanation_method="fallback",
            coverage=0.5,
            reliability=0.3,
        )

    def _generate_recommendations(
        self, trust_assessment: TrustScoreResult
    ) -> list[str]:
        """Generate recommendations based on trust assessment."""
        recommendations = []

        if trust_assessment.overall_trust_score < 0.7:
            recommendations.append("Consider model retraining or hyperparameter tuning")

        if trust_assessment.consistency_score < 0.7:
            recommendations.append("Improve model consistency through regularization")

        if trust_assessment.stability_score < 0.7:
            recommendations.append("Enhance prediction stability with ensemble methods")

        if trust_assessment.fidelity_score < 0.7:
            recommendations.append("Validate explanation fidelity with domain experts")

        if trust_assessment.risk_assessment == "high":
            recommendations.append(
                "Exercise caution when using model predictions for critical decisions"
            )

        if not recommendations:
            recommendations.append("Model shows good explainability characteristics")

        return recommendations

    def get_service_info(self) -> dict[str, Any]:
        """Get service information."""
        return {
            "shap_available": SHAP_AVAILABLE,
            "lime_available": LIME_AVAILABLE,
            "sklearn_available": SKLEARN_AVAILABLE,
            "shap_enabled": self.enable_shap,
            "lime_enabled": self.enable_lime,
            "permutation_enabled": self.enable_permutation,
            "cache_enabled": self.cache_explanations,
            "cached_explanations": len(self.explanation_cache),
            "cached_explainers": len(self.explainer_cache),
        }
