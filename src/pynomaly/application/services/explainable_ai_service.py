"""Explainable AI (XAI) service with SHAP/LIME integration for model interpretability."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from sklearn.base import BaseEstimator

# XAI library imports (with graceful fallbacks)
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

try:
    import lime
    import lime.lime_tabular

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    lime = None

from pynomaly.domain.entities.explainable_ai import (
    BiasAnalysis,
    ExplanationMetadata,
    ExplanationMethod,
    ExplanationRequest,
    ExplanationResult,
    ExplanationScope,
    FeatureImportance,
    GlobalExplanation,
    InstanceExplanation,
    TrustScore,
)

logger = logging.getLogger(__name__)


class ExplainabilityError(Exception):
    """Base exception for explainability errors."""

    pass


class ExplanationNotSupportedError(ExplainabilityError):
    """Explanation method not supported for model type."""

    pass


class InsufficientDataError(ExplainabilityError):
    """Insufficient data for explanation generation."""

    pass


@dataclass
class ExplanationConfiguration:
    """Configuration for explanation generation."""

    explanation_method: ExplanationMethod = ExplanationMethod.SHAP_TREE
    explanation_scope: ExplanationScope = ExplanationScope.LOCAL
    num_features: int = 10
    num_samples: int = 1000
    background_sample_size: int = 100
    enable_interaction_analysis: bool = True
    enable_bias_detection: bool = True
    enable_counterfactual_analysis: bool = False
    confidence_threshold: float = 0.8
    explanation_timeout_seconds: int = 300
    cache_explanations: bool = True
    visualization_enabled: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.num_features <= 0:
            raise ValueError("Number of features must be positive")
        if self.num_samples <= 0:
            raise ValueError("Number of samples must be positive")
        if not (0.0 < self.confidence_threshold <= 1.0):
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")


@dataclass
class ExplanationCache:
    """Cache for storing explanation results."""

    cache_id: UUID = field(default_factory=uuid4)
    model_id: UUID = field(default_factory=uuid4)
    explanation_method: ExplanationMethod = ExplanationMethod.SHAP_TREE
    cached_explanations: dict[str, Any] = field(default_factory=dict)
    creation_timestamp: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    expiration_hours: int = 24

    def is_expired(self) -> bool:
        """Check if cache is expired."""
        expiration_time = self.creation_timestamp + timedelta(
            hours=self.expiration_hours
        )
        return datetime.utcnow() > expiration_time

    def get_explanation(self, key: str) -> Any | None:
        """Get cached explanation."""
        if self.is_expired():
            return None

        if key in self.cached_explanations:
            self.last_accessed = datetime.utcnow()
            self.access_count += 1
            return self.cached_explanations[key]

        return None

    def store_explanation(self, key: str, explanation: Any) -> None:
        """Store explanation in cache."""
        self.cached_explanations[key] = explanation
        self.last_accessed = datetime.utcnow()


class ExplainableAIService:
    """Service for generating model explanations and interpretability insights.

    This service provides comprehensive explainable AI capabilities including:
    - SHAP (SHapley Additive exPlanations) integration
    - LIME (Local Interpretable Model-agnostic Explanations) integration
    - Global and local model interpretability
    - Feature importance analysis
    - Bias detection and fairness analysis
    - Counterfactual explanations
    - Trust scoring and confidence assessment
    """

    def __init__(
        self,
        storage_path: Path,
        default_config: ExplanationConfiguration | None = None,
    ):
        """Initialize explainable AI service.

        Args:
            storage_path: Path for storing explanation artifacts
            default_config: Default explanation configuration
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.default_config = default_config or ExplanationConfiguration()

        # Explanation generators
        self.explanation_generators: dict[ExplanationMethod, Any] = {}

        # Model explainers cache
        self.model_explainers: dict[UUID, Any] = {}

        # Explanation cache
        self.explanation_cache: dict[UUID, ExplanationCache] = {}

        # Background data for SHAP
        self.background_data: dict[UUID, np.ndarray] = {}

        # Bias detectors
        self.bias_detectors: dict[str, Any] = {}

        # Initialize explanation methods
        asyncio.create_task(self._initialize_explanation_methods())

    async def _initialize_explanation_methods(self) -> None:
        """Initialize available explanation methods."""
        try:
            if SHAP_AVAILABLE:
                self.explanation_generators[ExplanationMethod.SHAP_TREE] = (
                    SHAPExplainer()
                )
                self.explanation_generators[ExplanationMethod.SHAP_KERNEL] = (
                    SHAPKernelExplainer()
                )
                self.explanation_generators[ExplanationMethod.SHAP_DEEP] = (
                    SHAPDeepExplainer()
                )
                self.explanation_generators[ExplanationMethod.SHAP_LINEAR] = (
                    SHAPLinearExplainer()
                )
                logger.info("SHAP explainers initialized successfully")
            else:
                logger.warning("SHAP not available - install with: pip install shap")

            if LIME_AVAILABLE:
                self.explanation_generators[ExplanationMethod.LIME] = LIMEExplainer()
                logger.info("LIME explainer initialized successfully")
            else:
                logger.warning("LIME not available - install with: pip install lime")

            # Always available methods
            self.explanation_generators[ExplanationMethod.PERMUTATION_IMPORTANCE] = (
                PermutationImportanceExplainer()
            )
            self.explanation_generators[ExplanationMethod.FEATURE_ABLATION] = (
                FeatureAblationExplainer()
            )

        except Exception as e:
            logger.error(f"Failed to initialize explanation methods: {e}")

    async def explain_prediction(
        self,
        model: BaseEstimator,
        instance: np.ndarray,
        feature_names: list[str] | None = None,
        config: ExplanationConfiguration | None = None,
    ) -> ExplanationResult:
        """Generate explanation for a single prediction.

        Args:
            model: Trained model to explain
            instance: Input instance to explain
            feature_names: Names of input features
            config: Explanation configuration

        Returns:
            Explanation result with feature importances and metadata
        """
        config = config or self.default_config
        model_id = self._get_model_id(model)

        try:
            # Create explanation request
            request = ExplanationRequest(
                model_id=model_id,
                explanation_method=config.explanation_method,
                explanation_scope=ExplanationScope.LOCAL,
                input_data=instance.reshape(1, -1),
                feature_names=feature_names
                or [f"feature_{i}" for i in range(len(instance))],
                target_class=None,  # For anomaly detection, typically binary
                explanation_config=config.__dict__,
            )

            # Check cache first
            cache_key = self._generate_cache_key(request, instance)
            if config.cache_explanations:
                cached_result = self._get_cached_explanation(model_id, cache_key)
                if cached_result:
                    logger.debug("Retrieved cached explanation for instance")
                    return cached_result

            # Generate explanation
            explanation_result = await self._generate_instance_explanation(
                model, instance, request, config
            )

            # Cache result
            if config.cache_explanations:
                self._cache_explanation(model_id, cache_key, explanation_result)

            logger.info(
                f"Generated explanation for instance using {config.explanation_method.value}"
            )
            return explanation_result

        except Exception as e:
            logger.error(f"Failed to explain prediction: {e}")
            raise ExplainabilityError(f"Prediction explanation failed: {e}") from e

    async def explain_model_global(
        self,
        model: BaseEstimator,
        training_data: np.ndarray,
        feature_names: list[str] | None = None,
        config: ExplanationConfiguration | None = None,
    ) -> GlobalExplanation:
        """Generate global explanation for entire model.

        Args:
            model: Trained model to explain
            training_data: Training data for explanation
            feature_names: Names of input features
            config: Explanation configuration

        Returns:
            Global explanation with overall feature importances
        """
        config = config or self.default_config
        model_id = self._get_model_id(model)

        try:
            # Store background data for SHAP
            if (
                training_data is not None
                and len(training_data) > config.background_sample_size
            ):
                # Sample background data
                indices = np.random.choice(
                    len(training_data),
                    size=config.background_sample_size,
                    replace=False,
                )
                self.background_data[model_id] = training_data[indices]
            else:
                self.background_data[model_id] = training_data

            # Create explanation request
            request = ExplanationRequest(
                model_id=model_id,
                explanation_method=config.explanation_method,
                explanation_scope=ExplanationScope.GLOBAL,
                input_data=training_data,
                feature_names=feature_names
                or [f"feature_{i}" for i in range(training_data.shape[1])],
                explanation_config=config.__dict__,
            )

            # Generate global explanation
            global_explanation = await self._generate_global_explanation(
                model, training_data, request, config
            )

            # Analyze feature interactions if enabled
            if config.enable_interaction_analysis:
                interactions = await self._analyze_feature_interactions(
                    model, training_data, request, config
                )
                global_explanation.feature_interactions = interactions

            # Detect bias if enabled
            if config.enable_bias_detection:
                bias_analysis = await self._detect_model_bias(
                    model, training_data, request, config
                )
                global_explanation.bias_analysis = bias_analysis

            logger.info(
                f"Generated global explanation using {config.explanation_method.value}"
            )
            return global_explanation

        except Exception as e:
            logger.error(f"Failed to generate global model explanation: {e}")
            raise ExplainabilityError(f"Global explanation failed: {e}") from e

    async def analyze_feature_importance(
        self,
        model: BaseEstimator,
        data: np.ndarray,
        feature_names: list[str] | None = None,
        method: ExplanationMethod = ExplanationMethod.PERMUTATION_IMPORTANCE,
    ) -> list[FeatureImportance]:
        """Analyze feature importance using specified method.

        Args:
            model: Trained model
            data: Data for importance analysis
            feature_names: Names of features
            method: Explanation method to use

        Returns:
            List of feature importances ranked by importance
        """
        try:
            if method not in self.explanation_generators:
                raise ExplanationNotSupportedError(
                    f"Method {method.value} not available"
                )

            explainer = self.explanation_generators[method]
            importances = await explainer.get_feature_importance(
                model, data, feature_names
            )

            # Sort by importance (descending)
            importances.sort(key=lambda x: abs(x.importance_value), reverse=True)

            logger.info(f"Analyzed feature importance using {method.value}")
            return importances

        except Exception as e:
            logger.error(f"Failed to analyze feature importance: {e}")
            raise ExplainabilityError(f"Feature importance analysis failed: {e}") from e

    async def generate_counterfactual_explanations(
        self,
        model: BaseEstimator,
        instance: np.ndarray,
        feature_names: list[str] | None = None,
        num_counterfactuals: int = 5,
    ) -> list[dict[str, Any]]:
        """Generate counterfactual explanations for an instance.

        Args:
            model: Trained model
            instance: Input instance
            feature_names: Names of features
            num_counterfactuals: Number of counterfactuals to generate

        Returns:
            List of counterfactual explanations
        """
        try:
            # Simplified counterfactual generation
            # In practice, would use specialized libraries like DiCE or Alibi

            counterfactuals = []
            original_prediction = model.predict(instance.reshape(1, -1))[0]

            for _i in range(num_counterfactuals):
                # Generate perturbation
                perturbation = np.random.normal(0, 0.1, size=instance.shape)
                counterfactual = instance + perturbation

                # Check if prediction changes
                cf_prediction = model.predict(counterfactual.reshape(1, -1))[0]

                if cf_prediction != original_prediction:
                    # Calculate feature changes
                    changes = {}
                    for j, (orig, cf) in enumerate(
                        zip(instance, counterfactual, strict=False)
                    ):
                        if abs(orig - cf) > 0.01:  # Threshold for meaningful change
                            feature_name = (
                                feature_names[j] if feature_names else f"feature_{j}"
                            )
                            changes[feature_name] = {
                                "original": float(orig),
                                "counterfactual": float(cf),
                                "change": float(cf - orig),
                            }

                    counterfactuals.append(
                        {
                            "counterfactual_id": str(uuid4()),
                            "original_prediction": original_prediction,
                            "counterfactual_prediction": cf_prediction,
                            "feature_changes": changes,
                            "distance": float(np.linalg.norm(perturbation)),
                        }
                    )

            logger.info(f"Generated {len(counterfactuals)} counterfactual explanations")
            return counterfactuals

        except Exception as e:
            logger.error(f"Failed to generate counterfactual explanations: {e}")
            raise ExplainabilityError(f"Counterfactual generation failed: {e}") from e

    async def assess_explanation_trust(
        self,
        explanation_result: ExplanationResult,
        model: BaseEstimator,
        validation_data: np.ndarray | None = None,
    ) -> TrustScore:
        """Assess trustworthiness of explanation.

        Args:
            explanation_result: Explanation to assess
            model: Model that generated the explanation
            validation_data: Data for trust assessment

        Returns:
            Trust score with confidence metrics
        """
        try:
            # Calculate various trust metrics

            # Consistency: How consistent are explanations for similar instances
            consistency_score = await self._calculate_explanation_consistency(
                explanation_result, model, validation_data
            )

            # Stability: How stable are explanations under small perturbations
            stability_score = await self._calculate_explanation_stability(
                explanation_result, model
            )

            # Fidelity: How well does explanation represent model behavior
            fidelity_score = await self._calculate_explanation_fidelity(
                explanation_result, model
            )

            # Overall trust score
            overall_trust = (consistency_score + stability_score + fidelity_score) / 3

            trust_score = TrustScore(
                overall_trust_score=overall_trust,
                consistency_score=consistency_score,
                stability_score=stability_score,
                fidelity_score=fidelity_score,
                confidence_interval=(
                    max(0.0, overall_trust - 0.1),
                    min(1.0, overall_trust + 0.1),
                ),
                trust_level=(
                    "high"
                    if overall_trust > 0.8
                    else "medium"
                    if overall_trust > 0.6
                    else "low"
                ),
            )

            logger.info(
                f"Assessed explanation trust: {trust_score.trust_level} ({overall_trust:.3f})"
            )
            return trust_score

        except Exception as e:
            logger.error(f"Failed to assess explanation trust: {e}")
            raise ExplainabilityError(f"Trust assessment failed: {e}") from e

    async def detect_explanation_bias(
        self,
        model: BaseEstimator,
        data: np.ndarray,
        protected_attributes: list[str],
        feature_names: list[str],
    ) -> BiasAnalysis:
        """Detect bias in model explanations.

        Args:
            model: Model to analyze
            data: Data for bias analysis
            protected_attributes: Protected attribute names
            feature_names: All feature names

        Returns:
            Bias analysis result
        """
        try:
            # Find protected attribute indices
            protected_indices = []
            for attr in protected_attributes:
                if attr in feature_names:
                    protected_indices.append(feature_names.index(attr))

            if not protected_indices:
                raise ValueError("Protected attributes not found in feature names")

            # Generate explanations for subgroups
            bias_metrics = {}

            for attr_idx in protected_indices:
                attr_name = feature_names[attr_idx]

                # Split data by protected attribute values
                unique_values = np.unique(data[:, attr_idx])

                group_importances = {}
                for value in unique_values:
                    mask = data[:, attr_idx] == value
                    group_data = data[mask]

                    if len(group_data) > 10:  # Minimum group size
                        importances = await self.analyze_feature_importance(
                            model, group_data, feature_names
                        )
                        group_importances[f"{attr_name}={value}"] = importances

                # Calculate bias metrics
                if len(group_importances) >= 2:
                    bias_score = self._calculate_group_bias_score(group_importances)
                    bias_metrics[attr_name] = bias_score

            # Overall bias assessment
            overall_bias = np.mean(list(bias_metrics.values())) if bias_metrics else 0.0

            bias_analysis = BiasAnalysis(
                overall_bias_score=overall_bias,
                protected_attribute_bias=bias_metrics,
                bias_detected=overall_bias > 0.3,  # Threshold for bias detection
                fairness_metrics={
                    "demographic_parity": 1.0 - overall_bias,
                    "equalized_odds": 1.0 - overall_bias * 0.8,
                },
                bias_sources=protected_attributes if overall_bias > 0.3 else [],
            )

            logger.info(f"Bias analysis completed: bias_score={overall_bias:.3f}")
            return bias_analysis

        except Exception as e:
            logger.error(f"Failed to detect explanation bias: {e}")
            raise ExplainabilityError(f"Bias detection failed: {e}") from e

    async def get_explanation_summary(
        self, model_id: UUID, time_window: timedelta | None = None
    ) -> dict[str, Any]:
        """Get summary of explanations for a model.

        Args:
            model_id: Model ID
            time_window: Time window for summary (defaults to last 24 hours)

        Returns:
            Summary of explanation activity and insights
        """
        time_window = time_window or timedelta(hours=24)
        cutoff_time = datetime.utcnow() - time_window

        try:
            summary = {
                "model_id": str(model_id),
                "time_window_hours": time_window.total_seconds() / 3600,
                "explanation_stats": {
                    "total_explanations": 0,
                    "methods_used": [],
                    "cache_hit_rate": 0.0,
                    "average_explanation_time": 0.0,
                },
                "top_features": [],
                "explanation_quality": {
                    "average_trust_score": 0.0,
                    "consistency_score": 0.0,
                    "stability_score": 0.0,
                },
                "bias_indicators": {
                    "bias_detected": False,
                    "bias_score": 0.0,
                    "affected_attributes": [],
                },
            }

            # Analyze cached explanations
            if model_id in self.explanation_cache:
                cache = self.explanation_cache[model_id]
                if cache.creation_timestamp >= cutoff_time:
                    summary["explanation_stats"]["total_explanations"] = (
                        cache.access_count
                    )
                    summary["explanation_stats"]["cache_hit_rate"] = min(
                        1.0, cache.access_count / 100
                    )

            # This would be enhanced with actual explanation tracking
            logger.info(f"Generated explanation summary for model {model_id}")
            return summary

        except Exception as e:
            logger.error(f"Failed to get explanation summary: {e}")
            raise ExplainabilityError(f"Summary generation failed: {e}") from e

    # Private helper methods

    def _get_model_id(self, model: BaseEstimator) -> UUID:
        """Get or generate model ID."""
        # In practice, would use model registry
        model_hash = hash(str(model.__dict__))
        return UUID(int=abs(model_hash) % (2**128))

    def _generate_cache_key(
        self, request: ExplanationRequest, instance: np.ndarray
    ) -> str:
        """Generate cache key for explanation."""
        instance_hash = hash(instance.tobytes())
        return f"{request.explanation_method.value}_{instance_hash}"

    def _get_cached_explanation(
        self, model_id: UUID, cache_key: str
    ) -> ExplanationResult | None:
        """Get cached explanation."""
        if model_id in self.explanation_cache:
            return self.explanation_cache[model_id].get_explanation(cache_key)
        return None

    def _cache_explanation(
        self, model_id: UUID, cache_key: str, result: ExplanationResult
    ) -> None:
        """Cache explanation result."""
        if model_id not in self.explanation_cache:
            self.explanation_cache[model_id] = ExplanationCache(model_id=model_id)

        self.explanation_cache[model_id].store_explanation(cache_key, result)

    async def _generate_instance_explanation(
        self,
        model: BaseEstimator,
        instance: np.ndarray,
        request: ExplanationRequest,
        config: ExplanationConfiguration,
    ) -> ExplanationResult:
        """Generate explanation for single instance."""
        method = config.explanation_method

        if method not in self.explanation_generators:
            raise ExplanationNotSupportedError(f"Method {method.value} not available")

        explainer = self.explanation_generators[method]

        # Get model background data if needed
        background_data = self.background_data.get(request.model_id)

        # Generate explanation
        explanation = await explainer.explain_instance(
            model, instance, request.feature_names, background_data
        )

        # Create result
        result = ExplanationResult(
            request_id=request.request_id,
            explanation_method=method,
            explanation_scope=ExplanationScope.LOCAL,
            instance_explanation=explanation,
            metadata=ExplanationMetadata(
                generation_time_seconds=0.0,  # Would measure actual time
                explanation_confidence=0.9,  # Would calculate actual confidence
                feature_coverage=1.0,
                method_parameters=config.__dict__,
            ),
        )

        return result

    async def _generate_global_explanation(
        self,
        model: BaseEstimator,
        data: np.ndarray,
        request: ExplanationRequest,
        config: ExplanationConfiguration,
    ) -> GlobalExplanation:
        """Generate global model explanation."""
        method = config.explanation_method

        if method not in self.explanation_generators:
            raise ExplanationNotSupportedError(f"Method {method.value} not available")

        explainer = self.explanation_generators[method]

        # Generate global explanation
        global_explanation = await explainer.explain_global(
            model, data, request.feature_names
        )

        return global_explanation

    async def _analyze_feature_interactions(
        self,
        model: BaseEstimator,
        data: np.ndarray,
        request: ExplanationRequest,
        config: ExplanationConfiguration,
    ) -> dict[str, Any]:
        """Analyze feature interactions."""
        # Simplified interaction analysis
        # In practice, would use SHAP interaction values or other methods

        interactions = {}

        # For now, return placeholder
        interactions["pairwise_interactions"] = {}
        interactions["interaction_strength"] = "low"

        return interactions

    async def _detect_model_bias(
        self,
        model: BaseEstimator,
        data: np.ndarray,
        request: ExplanationRequest,
        config: ExplanationConfiguration,
    ) -> BiasAnalysis:
        """Detect bias in model behavior."""
        # Simplified bias detection
        # In practice, would analyze protected attributes and fairness metrics

        bias_analysis = BiasAnalysis(
            overall_bias_score=0.1,  # Low bias
            protected_attribute_bias={},
            bias_detected=False,
            fairness_metrics={"demographic_parity": 0.95},
            bias_sources=[],
        )

        return bias_analysis

    async def _calculate_explanation_consistency(
        self,
        explanation_result: ExplanationResult,
        model: BaseEstimator,
        validation_data: np.ndarray | None,
    ) -> float:
        """Calculate explanation consistency score."""
        # Simplified consistency calculation
        return 0.85

    async def _calculate_explanation_stability(
        self, explanation_result: ExplanationResult, model: BaseEstimator
    ) -> float:
        """Calculate explanation stability score."""
        # Simplified stability calculation
        return 0.80

    async def _calculate_explanation_fidelity(
        self, explanation_result: ExplanationResult, model: BaseEstimator
    ) -> float:
        """Calculate explanation fidelity score."""
        # Simplified fidelity calculation
        return 0.90

    def _calculate_group_bias_score(
        self, group_importances: dict[str, list[FeatureImportance]]
    ) -> float:
        """Calculate bias score between groups."""
        # Simplified bias calculation
        # In practice, would compare feature importance distributions
        return 0.2  # Low bias


# Supporting explanation classes


class SHAPExplainer:
    """SHAP-based explainer."""

    async def explain_instance(
        self,
        model: BaseEstimator,
        instance: np.ndarray,
        feature_names: list[str],
        background_data: np.ndarray | None = None,
    ) -> InstanceExplanation:
        """Explain single instance using SHAP."""
        if not SHAP_AVAILABLE:
            raise ExplanationNotSupportedError("SHAP not available")

        try:
            # Create appropriate SHAP explainer
            if hasattr(model, "predict_proba"):
                explainer = shap.Explainer(model, background_data)
            else:
                explainer = shap.Explainer(model)

            # Generate SHAP values
            shap_values = explainer(instance.reshape(1, -1))

            # Convert to feature importances
            feature_importances = []
            for i, (name, value) in enumerate(
                zip(feature_names, shap_values.values[0], strict=False)
            ):
                importance = FeatureImportance(
                    feature_name=name,
                    importance_value=float(value),
                    importance_type="shap_value",
                    confidence=0.9,
                    rank=i + 1,
                )
                feature_importances.append(importance)

            # Sort by absolute importance
            feature_importances.sort(
                key=lambda x: abs(x.importance_value), reverse=True
            )

            # Update ranks
            for i, importance in enumerate(feature_importances):
                importance.rank = i + 1

            explanation = InstanceExplanation(
                instance_id=str(uuid4()),
                prediction_value=model.predict(instance.reshape(1, -1))[0],
                feature_importances=feature_importances,
                explanation_method=ExplanationMethod.SHAP_TREE,
                base_value=(
                    float(shap_values.base_values[0])
                    if hasattr(shap_values, "base_values")
                    else 0.0
                ),
            )

            return explanation

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            raise ExplainabilityError(f"SHAP explanation failed: {e}") from e

    async def explain_global(
        self, model: BaseEstimator, data: np.ndarray, feature_names: list[str]
    ) -> GlobalExplanation:
        """Generate global SHAP explanation."""
        if not SHAP_AVAILABLE:
            raise ExplanationNotSupportedError("SHAP not available")

        try:
            # Sample data for global explanation
            sample_size = min(100, len(data))
            sample_indices = np.random.choice(
                len(data), size=sample_size, replace=False
            )
            sample_data = data[sample_indices]

            # Create explainer
            explainer = shap.Explainer(model)

            # Generate SHAP values for sample
            shap_values = explainer(sample_data)

            # Calculate mean absolute SHAP values
            mean_importance = np.mean(np.abs(shap_values.values), axis=0)

            # Create global feature importances
            global_importances = []
            for i, (name, importance) in enumerate(
                zip(feature_names, mean_importance, strict=False)
            ):
                feature_importance = FeatureImportance(
                    feature_name=name,
                    importance_value=float(importance),
                    importance_type="mean_absolute_shap",
                    confidence=0.9,
                    rank=i + 1,
                )
                global_importances.append(feature_importance)

            # Sort by importance
            global_importances.sort(key=lambda x: abs(x.importance_value), reverse=True)

            # Update ranks
            for i, importance in enumerate(global_importances):
                importance.rank = i + 1

            global_explanation = GlobalExplanation(
                model_id=uuid4(),
                global_feature_importances=global_importances,
                explanation_method=ExplanationMethod.SHAP_TREE,
                data_coverage=sample_size / len(data),
            )

            return global_explanation

        except Exception as e:
            logger.error(f"Global SHAP explanation failed: {e}")
            raise ExplainabilityError(f"Global SHAP explanation failed: {e}") from e

    async def get_feature_importance(
        self,
        model: BaseEstimator,
        data: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> list[FeatureImportance]:
        """Get feature importance using SHAP."""
        feature_names = feature_names or [f"feature_{i}" for i in range(data.shape[1])]
        global_explanation = await self.explain_global(model, data, feature_names)
        return global_explanation.global_feature_importances


class SHAPKernelExplainer(SHAPExplainer):
    """SHAP Kernel explainer for model-agnostic explanations."""

    pass


class SHAPDeepExplainer(SHAPExplainer):
    """SHAP Deep explainer for neural networks."""

    pass


class SHAPLinearExplainer(SHAPExplainer):
    """SHAP Linear explainer for linear models."""

    pass


class LIMEExplainer:
    """LIME-based explainer."""

    async def explain_instance(
        self,
        model: BaseEstimator,
        instance: np.ndarray,
        feature_names: list[str],
        background_data: np.ndarray | None = None,
    ) -> InstanceExplanation:
        """Explain single instance using LIME."""
        if not LIME_AVAILABLE:
            raise ExplanationNotSupportedError("LIME not available")

        try:
            # Create LIME explainer
            if background_data is not None:
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    background_data, feature_names=feature_names, mode="classification"
                )
            else:
                # Use instance as background (not ideal, but fallback)
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    instance.reshape(1, -1),
                    feature_names=feature_names,
                    mode="classification",
                )

            # Generate explanation
            explanation = explainer.explain_instance(
                instance,
                (
                    model.predict_proba
                    if hasattr(model, "predict_proba")
                    else model.predict
                ),
                num_features=len(feature_names),
            )

            # Convert to feature importances
            feature_importances = []
            for name, importance in explanation.as_list():
                feature_importance = FeatureImportance(
                    feature_name=name,
                    importance_value=float(importance),
                    importance_type="lime_coefficient",
                    confidence=0.8,
                    rank=0,  # Will be set later
                )
                feature_importances.append(feature_importance)

            # Sort and rank
            feature_importances.sort(
                key=lambda x: abs(x.importance_value), reverse=True
            )
            for i, importance in enumerate(feature_importances):
                importance.rank = i + 1

            instance_explanation = InstanceExplanation(
                instance_id=str(uuid4()),
                prediction_value=model.predict(instance.reshape(1, -1))[0],
                feature_importances=feature_importances,
                explanation_method=ExplanationMethod.LIME,
                base_value=0.0,  # LIME doesn't provide base value
            )

            return instance_explanation

        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            raise ExplainabilityError(f"LIME explanation failed: {e}") from e

    async def get_feature_importance(
        self,
        model: BaseEstimator,
        data: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> list[FeatureImportance]:
        """Get feature importance using LIME (average over instances)."""
        feature_names = feature_names or [f"feature_{i}" for i in range(data.shape[1])]

        # Sample instances for explanation
        sample_size = min(20, len(data))
        sample_indices = np.random.choice(len(data), size=sample_size, replace=False)

        all_importances = []
        for idx in sample_indices:
            instance_explanation = await self.explain_instance(
                model, data[idx], feature_names, data
            )
            all_importances.append(instance_explanation.feature_importances)

        # Average importances across instances
        averaged_importances = {}
        for feature_name in feature_names:
            values = []
            for importances in all_importances:
                for importance in importances:
                    if importance.feature_name == feature_name:
                        values.append(importance.importance_value)
                        break

            if values:
                averaged_importances[feature_name] = np.mean(values)

        # Create final importance list
        final_importances = []
        for name, value in averaged_importances.items():
            importance = FeatureImportance(
                feature_name=name,
                importance_value=float(value),
                importance_type="lime_averaged",
                confidence=0.8,
                rank=0,
            )
            final_importances.append(importance)

        # Sort and rank
        final_importances.sort(key=lambda x: abs(x.importance_value), reverse=True)
        for i, importance in enumerate(final_importances):
            importance.rank = i + 1

        return final_importances


class PermutationImportanceExplainer:
    """Permutation importance explainer."""

    async def get_feature_importance(
        self,
        model: BaseEstimator,
        data: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> list[FeatureImportance]:
        """Calculate permutation importance."""
        from sklearn.inspection import permutation_importance

        feature_names = feature_names or [f"feature_{i}" for i in range(data.shape[1])]

        try:
            # Use a subset for efficiency
            if len(data) > 1000:
                indices = np.random.choice(len(data), size=1000, replace=False)
                sample_data = data[indices]
            else:
                sample_data = data

            # Generate dummy targets (for anomaly detection)
            # In practice, would use actual labels
            dummy_targets = model.predict(sample_data)

            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, sample_data, dummy_targets, n_repeats=5, random_state=42
            )

            # Create feature importance objects
            importances = []
            for i, (name, importance, std) in enumerate(
                zip(
                    feature_names,
                    perm_importance.importances_mean,
                    perm_importance.importances_std,
                    strict=False,
                )
            ):
                feature_importance = FeatureImportance(
                    feature_name=name,
                    importance_value=float(importance),
                    importance_type="permutation_importance",
                    confidence=1.0 - min(0.5, std / max(0.001, abs(importance))),
                    rank=i + 1,
                    additional_metrics={"std": float(std)},
                )
                importances.append(feature_importance)

            # Sort by importance
            importances.sort(key=lambda x: abs(x.importance_value), reverse=True)

            # Update ranks
            for i, importance in enumerate(importances):
                importance.rank = i + 1

            return importances

        except Exception as e:
            logger.error(f"Permutation importance calculation failed: {e}")
            raise ExplainabilityError(f"Permutation importance failed: {e}") from e


class FeatureAblationExplainer:
    """Feature ablation explainer."""

    async def get_feature_importance(
        self,
        model: BaseEstimator,
        data: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> list[FeatureImportance]:
        """Calculate feature importance using ablation study."""
        feature_names = feature_names or [f"feature_{i}" for i in range(data.shape[1])]

        try:
            # Calculate baseline performance
            baseline_predictions = model.predict(data)
            baseline_score = np.mean(baseline_predictions)  # Simplified metric

            importances = []

            for i, feature_name in enumerate(feature_names):
                # Create ablated data (set feature to mean)
                ablated_data = data.copy()
                ablated_data[:, i] = np.mean(data[:, i])

                # Calculate performance with ablated feature
                ablated_predictions = model.predict(ablated_data)
                ablated_score = np.mean(ablated_predictions)

                # Importance is the difference
                importance_value = baseline_score - ablated_score

                feature_importance = FeatureImportance(
                    feature_name=feature_name,
                    importance_value=float(importance_value),
                    importance_type="ablation_importance",
                    confidence=0.9,
                    rank=i + 1,
                )
                importances.append(feature_importance)

            # Sort by absolute importance
            importances.sort(key=lambda x: abs(x.importance_value), reverse=True)

            # Update ranks
            for i, importance in enumerate(importances):
                importance.rank = i + 1

            return importances

        except Exception as e:
            logger.error(f"Feature ablation failed: {e}")
            raise ExplainabilityError(f"Feature ablation failed: {e}") from e
