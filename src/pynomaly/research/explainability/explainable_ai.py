"""Advanced explainable AI with counterfactual explanations, concept vectors, and model-agnostic methods."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ExplanationMethod(str, Enum):
    """Types of explanation methods."""

    LIME = "lime"  # Local Interpretable Model-agnostic Explanations
    SHAP = "shap"  # SHapley Additive exPlanations
    COUNTERFACTUAL = "counterfactual"  # Counterfactual explanations
    CONCEPT_ACTIVATION = "concept_activation"  # Concept Activation Vectors
    GRADIENT_BASED = "gradient_based"  # Gradient-based explanations
    ATTENTION = "attention"  # Attention-based explanations
    ANCHOR = "anchor"  # Anchor explanations
    PROTOTYPES = "prototypes"  # Prototype-based explanations


class ExplanationScope(str, Enum):
    """Scope of explanation."""

    LOCAL = "local"  # Individual prediction
    GLOBAL = "global"  # Model behavior
    COHORT = "cohort"  # Group of similar instances
    FEATURE = "feature"  # Feature importance
    CONCEPT = "concept"  # Concept-level


class CounterfactualType(str, Enum):
    """Types of counterfactual explanations."""

    NEAREST = "nearest"  # Nearest counterfactual
    DIVERSE = "diverse"  # Diverse counterfactuals
    FEASIBLE = "feasible"  # Feasible counterfactuals
    ACTIONABLE = "actionable"  # Actionable counterfactuals
    MINIMAL = "minimal"  # Minimal change counterfactuals


@dataclass
class ExplanationResult:
    """Result from explanation method."""

    explanation_id: str
    method: ExplanationMethod
    scope: ExplanationScope
    sample_id: str
    feature_importance: dict[str, float]
    explanation_text: str
    confidence: float
    processing_time: float
    metadata: dict[str, Any] = field(default_factory=dict)
    visual_data: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CounterfactualExplanation:
    """Counterfactual explanation result."""

    original_sample: np.ndarray
    counterfactual_samples: list[np.ndarray]
    feature_changes: list[dict[str, Any]]
    validity_scores: list[float]
    plausibility_scores: list[float]
    diversity_score: float
    explanation_text: str
    generation_method: CounterfactualType
    constraints_satisfied: bool = True


@dataclass
class ConceptActivation:
    """Concept Activation Vector result."""

    concept_name: str
    activation_vector: np.ndarray
    directional_derivative: float
    concept_importance: float
    layer_name: str
    sensitivity_score: float
    statistical_significance: float
    examples: list[np.ndarray] = field(default_factory=list)


@dataclass
class PrototypeExplanation:
    """Prototype-based explanation."""

    prototypes: list[np.ndarray]
    prototype_labels: list[str]
    similarity_scores: list[float]
    influence_scores: list[float]
    explanation_text: str
    prototype_features: list[dict[str, float]]


class BaseExplainer(ABC):
    """Base class for explanation methods."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model = None
        self.feature_names: list[str] | None = None
        self.is_fitted = False

    @abstractmethod
    async def explain(
        self,
        sample: np.ndarray,
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        **kwargs,
    ) -> ExplanationResult:
        """Generate explanation for a sample."""
        pass

    async def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        feature_names: list[str] | None = None,
    ) -> None:
        """Fit explainer on training data."""
        self.feature_names = feature_names or [
            f"feature_{i}" for i in range(X.shape[1])
        ]
        self.is_fitted = True


class LIMEExplainer(BaseExplainer):
    """LIME (Local Interpretable Model-agnostic Explanations) implementation."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.num_features = config.get("num_features", 10)
        self.num_samples = config.get("num_samples", 1000)
        self.kernel_width = config.get("kernel_width", 0.75)

    async def explain(
        self,
        sample: np.ndarray,
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        **kwargs,
    ) -> ExplanationResult:
        """Generate LIME explanation."""
        try:
            start_time = datetime.now()

            # Generate perturbed samples around the instance
            perturbed_samples = await self._generate_perturbed_samples(sample)

            # Get model predictions for perturbed samples
            predictions = model_predict_fn(perturbed_samples)

            # Calculate distances and weights
            distances = np.array(
                [np.linalg.norm(sample - perturbed) for perturbed in perturbed_samples]
            )
            weights = np.exp(-(distances**2) / (self.kernel_width**2))

            # Fit interpretable model (linear regression)
            feature_importance = await self._fit_interpretable_model(
                perturbed_samples, predictions, weights
            )

            # Generate explanation text
            explanation_text = await self._generate_lime_explanation(feature_importance)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ExplanationResult(
                explanation_id=f"lime_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                method=ExplanationMethod.LIME,
                scope=ExplanationScope.LOCAL,
                sample_id=kwargs.get("sample_id", "unknown"),
                feature_importance=feature_importance,
                explanation_text=explanation_text,
                confidence=0.8,  # Simplified confidence
                processing_time=processing_time,
                metadata={
                    "num_samples": len(perturbed_samples),
                    "kernel_width": self.kernel_width,
                    "weights_range": [float(np.min(weights)), float(np.max(weights))],
                },
            )

        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return self._create_fallback_explanation("lime", str(e))

    async def _generate_perturbed_samples(self, sample: np.ndarray) -> np.ndarray:
        """Generate perturbed samples around the instance."""
        perturbed_samples = []

        for _ in range(self.num_samples):
            # Create perturbation
            perturbation = np.random.normal(0, 0.1, size=sample.shape)
            perturbed_sample = sample + perturbation
            perturbed_samples.append(perturbed_sample)

        return np.array(perturbed_samples)

    async def _fit_interpretable_model(
        self, X: np.ndarray, y: np.ndarray, weights: np.ndarray
    ) -> dict[str, float]:
        """Fit weighted linear regression as interpretable model."""
        try:
            from sklearn.linear_model import LinearRegression

            # Weighted linear regression
            model = LinearRegression()

            # Apply weights by duplicating samples
            weighted_X = []
            weighted_y = []

            for i, weight in enumerate(weights):
                if weight > 0.01:  # Skip very low weight samples
                    num_copies = max(1, int(weight * 10))
                    for _ in range(num_copies):
                        weighted_X.append(X[i])
                        weighted_y.append(y[i])

            if weighted_X:
                model.fit(weighted_X, weighted_y)
                coefficients = model.coef_

                # Create feature importance dictionary
                feature_importance = {}
                for i, coef in enumerate(coefficients):
                    feature_name = (
                        self.feature_names[i]
                        if self.feature_names and i < len(self.feature_names)
                        else f"feature_{i}"
                    )
                    feature_importance[feature_name] = float(abs(coef))

                return feature_importance
            else:
                return {}

        except Exception as e:
            logger.error(f"Interpretable model fitting failed: {e}")
            return {}

    async def _generate_lime_explanation(
        self, feature_importance: dict[str, float]
    ) -> str:
        """Generate human-readable LIME explanation."""
        if not feature_importance:
            return "Unable to generate explanation due to insufficient data."

        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        top_features = sorted_features[: self.num_features]

        explanation_parts = [
            "LIME Explanation:",
            "Most important features for this prediction:",
        ]

        for i, (feature, importance) in enumerate(top_features):
            explanation_parts.append(f"{i+1}. {feature}: {importance:.4f}")

        return " | ".join(explanation_parts)


class SHAPExplainer(BaseExplainer):
    """SHAP (SHapley Additive exPlanations) implementation."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.background_samples = None
        self.num_background = config.get("num_background", 100)
        self.max_evals = config.get("max_evals", 500)

    async def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        feature_names: list[str] | None = None,
    ) -> None:
        """Fit SHAP explainer."""
        await super().fit(X, y, model_predict_fn, feature_names)

        # Sample background data
        if len(X) > self.num_background:
            indices = np.random.choice(len(X), self.num_background, replace=False)
            self.background_samples = X[indices]
        else:
            self.background_samples = X

    async def explain(
        self,
        sample: np.ndarray,
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        **kwargs,
    ) -> ExplanationResult:
        """Generate SHAP explanation."""
        try:
            start_time = datetime.now()

            # Calculate SHAP values using sampling approximation
            shap_values = await self._calculate_shap_values(sample, model_predict_fn)

            # Create feature importance dictionary
            feature_importance = {}
            for i, shap_val in enumerate(shap_values):
                feature_name = (
                    self.feature_names[i]
                    if self.feature_names and i < len(self.feature_names)
                    else f"feature_{i}"
                )
                feature_importance[feature_name] = float(abs(shap_val))

            # Generate explanation text
            explanation_text = await self._generate_shap_explanation(
                shap_values, sample
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            return ExplanationResult(
                explanation_id=f"shap_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                method=ExplanationMethod.SHAP,
                scope=ExplanationScope.LOCAL,
                sample_id=kwargs.get("sample_id", "unknown"),
                feature_importance=feature_importance,
                explanation_text=explanation_text,
                confidence=0.85,
                processing_time=processing_time,
                metadata={
                    "shap_values": shap_values.tolist(),
                    "baseline_prediction": float(
                        np.mean(model_predict_fn(self.background_samples))
                    ),
                    "current_prediction": float(
                        model_predict_fn(sample.reshape(1, -1))[0]
                    ),
                },
            )

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return self._create_fallback_explanation("shap", str(e))

    async def _calculate_shap_values(
        self, sample: np.ndarray, model_predict_fn: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """Calculate SHAP values using sampling approximation."""
        try:
            n_features = len(sample)
            shap_values = np.zeros(n_features)

            # Calculate baseline (expected value)
            baseline = np.mean(model_predict_fn(self.background_samples))

            # Calculate marginal contributions
            for i in range(n_features):
                marginal_contribution = 0.0
                num_evaluations = min(self.max_evals // n_features, 50)

                for _ in range(num_evaluations):
                    # Random coalition
                    coalition = np.random.choice([True, False], size=n_features)
                    coalition[i] = False  # Remove feature i

                    # Create sample with coalition
                    coalition_sample = sample.copy()
                    background_sample = self.background_samples[
                        np.random.randint(len(self.background_samples))
                    ]

                    for j in range(n_features):
                        if not coalition[j]:
                            coalition_sample[j] = background_sample[j]

                    # Prediction without feature i
                    pred_without = model_predict_fn(coalition_sample.reshape(1, -1))[0]

                    # Prediction with feature i
                    coalition[i] = True
                    coalition_sample[i] = sample[i]
                    pred_with = model_predict_fn(coalition_sample.reshape(1, -1))[0]

                    # Marginal contribution
                    marginal_contribution += pred_with - pred_without

                shap_values[i] = marginal_contribution / num_evaluations

            return shap_values

        except Exception as e:
            logger.error(f"SHAP value calculation failed: {e}")
            return np.zeros(len(sample))

    async def _generate_shap_explanation(
        self, shap_values: np.ndarray, sample: np.ndarray
    ) -> str:
        """Generate SHAP explanation text."""
        # Sort by absolute SHAP value
        sorted_indices = np.argsort(np.abs(shap_values))[::-1]

        explanation_parts = [
            "SHAP Explanation:",
            "Feature contributions to prediction:",
        ]

        for i, idx in enumerate(sorted_indices[:5]):  # Top 5 features
            feature_name = (
                self.feature_names[idx] if self.feature_names else f"feature_{idx}"
            )
            shap_val = shap_values[idx]
            feature_val = sample[idx]

            direction = "increases" if shap_val > 0 else "decreases"
            explanation_parts.append(
                f"{i+1}. {feature_name} ({feature_val:.3f}) {direction} prediction by {abs(shap_val):.4f}"
            )

        return " | ".join(explanation_parts)


class CounterfactualExplainer(BaseExplainer):
    """Counterfactual explanation generator."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.counterfactual_type = CounterfactualType(config.get("type", "nearest"))
        self.num_counterfactuals = config.get("num_counterfactuals", 5)
        self.max_iterations = config.get("max_iterations", 100)
        self.feature_constraints = config.get("feature_constraints", {})
        self.distance_metric = config.get("distance_metric", "euclidean")

    async def generate_counterfactuals(
        self,
        sample: np.ndarray,
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        target_class: int | None = None,
        **kwargs,
    ) -> CounterfactualExplanation:
        """Generate counterfactual explanations."""
        try:
            logger.info(f"Generating {self.counterfactual_type} counterfactuals")

            # Get original prediction
            original_pred = model_predict_fn(sample.reshape(1, -1))[0]

            if target_class is None:
                # Flip the prediction
                target_class = (
                    1 - round(original_pred)
                    if original_pred < 0.5 or original_pred > 0.5
                    else 0.5
                )

            # Generate counterfactuals based on type
            if self.counterfactual_type == CounterfactualType.NEAREST:
                counterfactuals = await self._generate_nearest_counterfactuals(
                    sample, model_predict_fn, target_class
                )
            elif self.counterfactual_type == CounterfactualType.DIVERSE:
                counterfactuals = await self._generate_diverse_counterfactuals(
                    sample, model_predict_fn, target_class
                )
            elif self.counterfactual_type == CounterfactualType.MINIMAL:
                counterfactuals = await self._generate_minimal_counterfactuals(
                    sample, model_predict_fn, target_class
                )
            else:
                counterfactuals = await self._generate_nearest_counterfactuals(
                    sample, model_predict_fn, target_class
                )

            # Calculate metrics
            feature_changes = await self._calculate_feature_changes(
                sample, counterfactuals
            )
            validity_scores = await self._calculate_validity_scores(
                counterfactuals, model_predict_fn, target_class
            )
            plausibility_scores = await self._calculate_plausibility_scores(
                sample, counterfactuals
            )
            diversity_score = await self._calculate_diversity_score(counterfactuals)

            # Generate explanation text
            explanation_text = await self._generate_counterfactual_explanation(
                sample, counterfactuals, feature_changes
            )

            return CounterfactualExplanation(
                original_sample=sample,
                counterfactual_samples=counterfactuals,
                feature_changes=feature_changes,
                validity_scores=validity_scores,
                plausibility_scores=plausibility_scores,
                diversity_score=diversity_score,
                explanation_text=explanation_text,
                generation_method=self.counterfactual_type,
            )

        except Exception as e:
            logger.error(f"Counterfactual generation failed: {e}")
            return CounterfactualExplanation(
                original_sample=sample,
                counterfactual_samples=[],
                feature_changes=[],
                validity_scores=[],
                plausibility_scores=[],
                diversity_score=0.0,
                explanation_text=f"Counterfactual generation failed: {e}",
                generation_method=self.counterfactual_type,
            )

    async def _generate_nearest_counterfactuals(
        self,
        sample: np.ndarray,
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        target_class: float,
    ) -> list[np.ndarray]:
        """Generate nearest counterfactuals using gradient-free optimization."""
        counterfactuals = []

        for _ in range(self.num_counterfactuals):
            current_sample = sample.copy()

            for iteration in range(self.max_iterations):
                # Random perturbation
                perturbation = np.random.normal(0, 0.1, size=sample.shape)
                candidate = current_sample + perturbation

                # Apply feature constraints
                candidate = await self._apply_constraints(candidate)

                # Check if prediction changed
                pred = model_predict_fn(candidate.reshape(1, -1))[0]

                if abs(pred - target_class) < abs(
                    model_predict_fn(current_sample.reshape(1, -1))[0] - target_class
                ):
                    current_sample = candidate

                # Early stopping if target reached
                if (target_class > 0.5 and pred > 0.5) or (
                    target_class <= 0.5 and pred <= 0.5
                ):
                    break

            counterfactuals.append(current_sample)

        return counterfactuals

    async def _generate_diverse_counterfactuals(
        self,
        sample: np.ndarray,
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        target_class: float,
    ) -> list[np.ndarray]:
        """Generate diverse counterfactuals."""
        counterfactuals = []

        # Generate candidates with different perturbation strategies
        strategies = [
            lambda x: x + np.random.normal(0, 0.05, x.shape),  # Small perturbations
            lambda x: x + np.random.normal(0, 0.2, x.shape),  # Large perturbations
            lambda x: x * (1 + np.random.normal(0, 0.1, x.shape)),  # Multiplicative
            lambda x: np.where(
                np.random.random(x.shape) < 0.3, np.random.normal(0, 0.5, x.shape), x
            ),  # Sparse
        ]

        for strategy in strategies:
            current_sample = sample.copy()

            for iteration in range(self.max_iterations // len(strategies)):
                candidate = strategy(current_sample)
                candidate = await self._apply_constraints(candidate)

                pred = model_predict_fn(candidate.reshape(1, -1))[0]

                if abs(pred - target_class) < abs(
                    model_predict_fn(current_sample.reshape(1, -1))[0] - target_class
                ):
                    current_sample = candidate

                if (target_class > 0.5 and pred > 0.5) or (
                    target_class <= 0.5 and pred <= 0.5
                ):
                    break

            counterfactuals.append(current_sample)

        return counterfactuals[: self.num_counterfactuals]

    async def _generate_minimal_counterfactuals(
        self,
        sample: np.ndarray,
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        target_class: float,
    ) -> list[np.ndarray]:
        """Generate minimal change counterfactuals."""
        counterfactuals = []

        for _ in range(self.num_counterfactuals):
            best_candidate = sample.copy()
            best_distance = float("inf")

            # Try changing each feature individually first
            for feature_idx in range(len(sample)):
                for change_factor in [0.8, 0.9, 1.1, 1.2, 1.5, 2.0]:
                    candidate = sample.copy()
                    candidate[feature_idx] *= change_factor

                    candidate = await self._apply_constraints(candidate)
                    pred = model_predict_fn(candidate.reshape(1, -1))[0]

                    if (target_class > 0.5 and pred > 0.5) or (
                        target_class <= 0.5 and pred <= 0.5
                    ):
                        distance = np.linalg.norm(candidate - sample)
                        if distance < best_distance:
                            best_candidate = candidate
                            best_distance = distance

            # If single feature change doesn't work, try combinations
            if best_distance == float("inf"):
                for iteration in range(50):
                    candidate = sample.copy()

                    # Change random subset of features
                    feature_indices = np.random.choice(
                        len(sample),
                        size=np.random.randint(1, min(3, len(sample))),
                        replace=False,
                    )

                    for idx in feature_indices:
                        candidate[idx] += np.random.normal(0, 0.1)

                    candidate = await self._apply_constraints(candidate)
                    pred = model_predict_fn(candidate.reshape(1, -1))[0]

                    if (target_class > 0.5 and pred > 0.5) or (
                        target_class <= 0.5 and pred <= 0.5
                    ):
                        distance = np.linalg.norm(candidate - sample)
                        if distance < best_distance:
                            best_candidate = candidate
                            best_distance = distance

            counterfactuals.append(best_candidate)

        return counterfactuals

    async def _apply_constraints(self, sample: np.ndarray) -> np.ndarray:
        """Apply feature constraints to sample."""
        constrained_sample = sample.copy()

        for feature_idx, constraints in self.feature_constraints.items():
            if isinstance(feature_idx, int) and feature_idx < len(constrained_sample):
                if "min" in constraints:
                    constrained_sample[feature_idx] = max(
                        constrained_sample[feature_idx], constraints["min"]
                    )
                if "max" in constraints:
                    constrained_sample[feature_idx] = min(
                        constrained_sample[feature_idx], constraints["max"]
                    )
                if "categorical" in constraints and constraints["categorical"]:
                    # Round to nearest integer for categorical features
                    constrained_sample[feature_idx] = round(
                        constrained_sample[feature_idx]
                    )

        return constrained_sample

    async def _calculate_feature_changes(
        self, original: np.ndarray, counterfactuals: list[np.ndarray]
    ) -> list[dict[str, Any]]:
        """Calculate feature changes for each counterfactual."""
        changes = []

        for cf in counterfactuals:
            change_dict = {}
            for i in range(len(original)):
                feature_name = (
                    self.feature_names[i] if self.feature_names else f"feature_{i}"
                )
                original_val = original[i]
                cf_val = cf[i]

                if abs(original_val - cf_val) > 1e-6:
                    change_dict[feature_name] = {
                        "original": float(original_val),
                        "counterfactual": float(cf_val),
                        "change": float(cf_val - original_val),
                        "relative_change": float(
                            (cf_val - original_val) / (original_val + 1e-8)
                        ),
                    }

            changes.append(change_dict)

        return changes

    async def _calculate_validity_scores(
        self,
        counterfactuals: list[np.ndarray],
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        target_class: float,
    ) -> list[float]:
        """Calculate validity scores (how well counterfactuals achieve target)."""
        scores = []

        for cf in counterfactuals:
            pred = model_predict_fn(cf.reshape(1, -1))[0]

            if target_class > 0.5:
                # Target is positive class
                score = pred
            else:
                # Target is negative class
                score = 1.0 - pred

            scores.append(float(score))

        return scores

    async def _calculate_plausibility_scores(
        self, original: np.ndarray, counterfactuals: list[np.ndarray]
    ) -> list[float]:
        """Calculate plausibility scores (how realistic counterfactuals are)."""
        scores = []

        for cf in counterfactuals:
            # Distance-based plausibility
            distance = np.linalg.norm(cf - original)

            # Convert to plausibility score (closer = more plausible)
            max_distance = np.linalg.norm(original) * 2  # Heuristic max distance
            plausibility = max(0.0, 1.0 - distance / max_distance)

            scores.append(float(plausibility))

        return scores

    async def _calculate_diversity_score(
        self, counterfactuals: list[np.ndarray]
    ) -> float:
        """Calculate diversity score among counterfactuals."""
        if len(counterfactuals) < 2:
            return 0.0

        total_distance = 0.0
        count = 0

        for i in range(len(counterfactuals)):
            for j in range(i + 1, len(counterfactuals)):
                distance = np.linalg.norm(counterfactuals[i] - counterfactuals[j])
                total_distance += distance
                count += 1

        return float(total_distance / count) if count > 0 else 0.0

    async def _generate_counterfactual_explanation(
        self,
        original: np.ndarray,
        counterfactuals: list[np.ndarray],
        feature_changes: list[dict[str, Any]],
    ) -> str:
        """Generate human-readable counterfactual explanation."""
        if not counterfactuals:
            return "No valid counterfactuals found."

        explanation_parts = [
            f"Generated {len(counterfactuals)} counterfactual explanations:"
        ]

        for i, changes in enumerate(feature_changes[:3]):  # Show top 3
            if changes:
                change_descriptions = []
                for feature, change_info in list(changes.items())[
                    :3
                ]:  # Top 3 changed features
                    direction = "increase" if change_info["change"] > 0 else "decrease"
                    change_descriptions.append(
                        f"{direction} {feature} to {change_info['counterfactual']:.3f}"
                    )

                explanation_parts.append(
                    f"Counterfactual {i+1}: {', '.join(change_descriptions)}"
                )

        return " | ".join(explanation_parts)


class ConceptActivationVectorExplainer(BaseExplainer):
    """Concept Activation Vector (CAV) explainer."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.concepts = config.get("concepts", {})
        self.layer_name = config.get("layer_name", "hidden")
        self.num_random_concepts = config.get("num_random_concepts", 10)

    async def fit_concept(
        self,
        concept_name: str,
        positive_examples: list[np.ndarray],
        negative_examples: list[np.ndarray],
        model_activation_fn: Callable[[np.ndarray, str], np.ndarray],
    ) -> ConceptActivation:
        """Fit concept activation vector."""
        try:
            logger.info(f"Fitting concept: {concept_name}")

            # Get activations for positive and negative examples
            positive_activations = []
            for example in positive_examples:
                activation = model_activation_fn(
                    example.reshape(1, -1), self.layer_name
                )
                positive_activations.append(activation.flatten())

            negative_activations = []
            for example in negative_examples:
                activation = model_activation_fn(
                    example.reshape(1, -1), self.layer_name
                )
                negative_activations.append(activation.flatten())

            if not positive_activations or not negative_activations:
                raise ValueError("Need both positive and negative examples")

            # Combine activations and labels
            X = np.vstack([positive_activations, negative_activations])
            y = np.array(
                [1] * len(positive_activations) + [0] * len(negative_activations)
            )

            # Train linear classifier to separate concept
            concept_vector = await self._train_concept_classifier(X, y)

            # Calculate statistical significance
            significance = await self._calculate_statistical_significance(
                positive_activations, negative_activations, concept_vector
            )

            return ConceptActivation(
                concept_name=concept_name,
                activation_vector=concept_vector,
                directional_derivative=0.0,  # Will be calculated per sample
                concept_importance=significance,
                layer_name=self.layer_name,
                sensitivity_score=0.0,  # Will be calculated per sample
                statistical_significance=significance,
                examples=positive_examples[:5],  # Store some examples
            )

        except Exception as e:
            logger.error(f"Concept fitting failed: {e}")
            raise

    async def explain_with_concepts(
        self,
        sample: np.ndarray,
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        model_activation_fn: Callable[[np.ndarray, str], np.ndarray],
        concepts: list[ConceptActivation],
        **kwargs,
    ) -> ExplanationResult:
        """Explain prediction using concept activation vectors."""
        try:
            start_time = datetime.now()

            # Get sample activation
            sample_activation = model_activation_fn(
                sample.reshape(1, -1), self.layer_name
            ).flatten()

            # Calculate concept importance for this sample
            concept_scores = {}

            for concept in concepts:
                # Directional derivative (simplified)
                directional_deriv = await self._calculate_directional_derivative(
                    sample,
                    concept.activation_vector,
                    model_predict_fn,
                    model_activation_fn,
                )

                # Concept activation level
                activation_level = np.dot(sample_activation, concept.activation_vector)

                # Combined concept score
                concept_score = abs(directional_deriv * activation_level)
                concept_scores[concept.concept_name] = float(concept_score)

            # Generate explanation
            explanation_text = await self._generate_concept_explanation(concept_scores)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ExplanationResult(
                explanation_id=f"cav_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                method=ExplanationMethod.CONCEPT_ACTIVATION,
                scope=ExplanationScope.CONCEPT,
                sample_id=kwargs.get("sample_id", "unknown"),
                feature_importance=concept_scores,
                explanation_text=explanation_text,
                confidence=0.75,
                processing_time=processing_time,
                metadata={
                    "layer_name": self.layer_name,
                    "num_concepts": len(concepts),
                    "activation_vector_size": len(sample_activation),
                },
            )

        except Exception as e:
            logger.error(f"Concept explanation failed: {e}")
            return self._create_fallback_explanation("concept_activation", str(e))

    async def _train_concept_classifier(
        self, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Train linear classifier to learn concept direction."""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train classifier
            classifier = LogisticRegression(random_state=42)
            classifier.fit(X_scaled, y)

            # Get concept vector (normal to decision boundary)
            concept_vector = classifier.coef_[0]

            # Normalize
            concept_vector = concept_vector / (np.linalg.norm(concept_vector) + 1e-8)

            return concept_vector

        except Exception as e:
            logger.error(f"Concept classifier training failed: {e}")
            return np.random.random(X.shape[1])

    async def _calculate_directional_derivative(
        self,
        sample: np.ndarray,
        concept_vector: np.ndarray,
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        model_activation_fn: Callable[[np.ndarray, str], np.ndarray],
    ) -> float:
        """Calculate directional derivative of model output w.r.t. concept."""
        try:
            epsilon = 1e-4

            # Get current activation and prediction
            current_activation = model_activation_fn(
                sample.reshape(1, -1), self.layer_name
            ).flatten()
            current_pred = model_predict_fn(sample.reshape(1, -1))[0]

            # Approximate activation change (simplified)
            # In practice, this would require gradient computation through the model
            perturbed_activation = current_activation + epsilon * concept_vector

            # For simplification, assume linear relationship for derivative approximation
            # This would be more sophisticated in a real implementation
            activation_change = np.dot(concept_vector, concept_vector) * epsilon

            # Approximate prediction change
            pred_change = activation_change * 0.1  # Simplified

            # Directional derivative
            directional_deriv = pred_change / (epsilon + 1e-8)

            return float(directional_deriv)

        except Exception as e:
            logger.warning(f"Directional derivative calculation failed: {e}")
            return 0.0

    async def _calculate_statistical_significance(
        self,
        positive_activations: list[np.ndarray],
        negative_activations: list[np.ndarray],
        concept_vector: np.ndarray,
    ) -> float:
        """Calculate statistical significance of concept."""
        try:
            # Project activations onto concept vector
            pos_projections = [
                np.dot(act, concept_vector) for act in positive_activations
            ]
            neg_projections = [
                np.dot(act, concept_vector) for act in negative_activations
            ]

            # Simple t-test like measure
            pos_mean = np.mean(pos_projections)
            neg_mean = np.mean(neg_projections)
            pos_std = np.std(pos_projections)
            neg_std = np.std(neg_projections)

            pooled_std = np.sqrt((pos_std**2 + neg_std**2) / 2)
            t_statistic = abs(pos_mean - neg_mean) / (pooled_std + 1e-8)

            # Convert to significance score [0, 1]
            significance = min(1.0, t_statistic / 5.0)

            return float(significance)

        except Exception as e:
            logger.warning(f"Statistical significance calculation failed: {e}")
            return 0.5

    async def _generate_concept_explanation(
        self, concept_scores: dict[str, float]
    ) -> str:
        """Generate concept-based explanation."""
        if not concept_scores:
            return "No concept activations found."

        sorted_concepts = sorted(
            concept_scores.items(), key=lambda x: x[1], reverse=True
        )

        explanation_parts = [
            "Concept Activation Explanation:",
            "Most activated concepts:",
        ]

        for i, (concept, score) in enumerate(sorted_concepts[:5]):
            explanation_parts.append(f"{i+1}. {concept}: {score:.4f}")

        return " | ".join(explanation_parts)


class ExplainableAIOrchestrator:
    """Main orchestrator for explainable AI methods."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.explainers: dict[ExplanationMethod, BaseExplainer] = {}
        self.explanation_history: list[ExplanationResult] = []

        # Initialize explainers
        self._initialize_explainers()

    def _initialize_explainers(self) -> None:
        """Initialize explanation methods."""
        try:
            if ExplanationMethod.LIME in self.config.get("methods", []):
                self.explainers[ExplanationMethod.LIME] = LIMEExplainer(
                    self.config.get("lime_config", {})
                )

            if ExplanationMethod.SHAP in self.config.get("methods", []):
                self.explainers[ExplanationMethod.SHAP] = SHAPExplainer(
                    self.config.get("shap_config", {})
                )

            if ExplanationMethod.COUNTERFACTUAL in self.config.get("methods", []):
                self.explainers[ExplanationMethod.COUNTERFACTUAL] = (
                    CounterfactualExplainer(
                        self.config.get("counterfactual_config", {})
                    )
                )

            if ExplanationMethod.CONCEPT_ACTIVATION in self.config.get("methods", []):
                self.explainers[ExplanationMethod.CONCEPT_ACTIVATION] = (
                    ConceptActivationVectorExplainer(
                        self.config.get("concept_config", {})
                    )
                )

            logger.info(f"Initialized {len(self.explainers)} explanation methods")

        except Exception as e:
            logger.error(f"Explainer initialization failed: {e}")

    async def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        feature_names: list[str] | None = None,
    ) -> None:
        """Fit all explainers."""
        try:
            logger.info("Fitting explainable AI methods")

            for method, explainer in self.explainers.items():
                await explainer.fit(X, y, model_predict_fn, feature_names)
                logger.info(f"Fitted {method} explainer")

        except Exception as e:
            logger.error(f"Explainer fitting failed: {e}")
            raise

    async def explain_prediction(
        self,
        sample: np.ndarray,
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        methods: list[ExplanationMethod] | None = None,
        **kwargs,
    ) -> dict[ExplanationMethod, ExplanationResult]:
        """Generate explanations using specified methods."""
        try:
            if methods is None:
                methods = list(self.explainers.keys())

            explanations = {}

            for method in methods:
                if method in self.explainers:
                    try:
                        explanation = await self.explainers[method].explain(
                            sample, model_predict_fn, **kwargs
                        )
                        explanations[method] = explanation
                        self.explanation_history.append(explanation)

                    except Exception as e:
                        logger.error(f"Explanation with {method} failed: {e}")
                        explanations[method] = self._create_error_explanation(
                            method, str(e)
                        )

            return explanations

        except Exception as e:
            logger.error(f"Prediction explanation failed: {e}")
            return {}

    async def generate_counterfactuals(
        self,
        sample: np.ndarray,
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        **kwargs,
    ) -> CounterfactualExplanation | None:
        """Generate counterfactual explanations."""
        try:
            if ExplanationMethod.COUNTERFACTUAL in self.explainers:
                cf_explainer = self.explainers[ExplanationMethod.COUNTERFACTUAL]
                return await cf_explainer.generate_counterfactuals(
                    sample, model_predict_fn, **kwargs
                )
            else:
                logger.warning("Counterfactual explainer not available")
                return None

        except Exception as e:
            logger.error(f"Counterfactual generation failed: {e}")
            return None

    async def explain_with_concepts(
        self,
        sample: np.ndarray,
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        model_activation_fn: Callable[[np.ndarray, str], np.ndarray],
        concepts: list[ConceptActivation],
        **kwargs,
    ) -> ExplanationResult | None:
        """Generate concept-based explanations."""
        try:
            if ExplanationMethod.CONCEPT_ACTIVATION in self.explainers:
                cav_explainer = self.explainers[ExplanationMethod.CONCEPT_ACTIVATION]
                return await cav_explainer.explain_with_concepts(
                    sample, model_predict_fn, model_activation_fn, concepts, **kwargs
                )
            else:
                logger.warning("Concept activation explainer not available")
                return None

        except Exception as e:
            logger.error(f"Concept explanation failed: {e}")
            return None

    async def generate_global_explanation(
        self,
        X: np.ndarray,
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        sample_size: int = 100,
    ) -> dict[str, Any]:
        """Generate global model explanation."""
        try:
            logger.info("Generating global explanation")

            # Sample data for global explanation
            if len(X) > sample_size:
                indices = np.random.choice(len(X), sample_size, replace=False)
                sample_data = X[indices]
            else:
                sample_data = X

            # Collect feature importance across samples
            global_importance = {}
            explanation_count = 0

            for sample in sample_data:
                explanations = await self.explain_prediction(sample, model_predict_fn)

                for method, explanation in explanations.items():
                    if explanation.feature_importance:
                        explanation_count += 1
                        for (
                            feature,
                            importance,
                        ) in explanation.feature_importance.items():
                            if feature not in global_importance:
                                global_importance[feature] = []
                            global_importance[feature].append(importance)

            # Aggregate feature importance
            aggregated_importance = {}
            for feature, importance_list in global_importance.items():
                aggregated_importance[feature] = {
                    "mean": float(np.mean(importance_list)),
                    "std": float(np.std(importance_list)),
                    "count": len(importance_list),
                }

            return {
                "global_feature_importance": aggregated_importance,
                "num_samples_analyzed": len(sample_data),
                "num_explanations_generated": explanation_count,
                "most_important_features": sorted(
                    aggregated_importance.items(),
                    key=lambda x: x[1]["mean"],
                    reverse=True,
                )[:10],
            }

        except Exception as e:
            logger.error(f"Global explanation generation failed: {e}")
            return {}

    def _create_error_explanation(
        self, method: ExplanationMethod, error_msg: str
    ) -> ExplanationResult:
        """Create error explanation result."""
        return ExplanationResult(
            explanation_id=f"error_{method}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            method=method,
            scope=ExplanationScope.LOCAL,
            sample_id="unknown",
            feature_importance={},
            explanation_text=f"Explanation failed: {error_msg}",
            confidence=0.0,
            processing_time=0.0,
        )

    def _create_fallback_explanation(
        self, method_name: str, error_msg: str
    ) -> ExplanationResult:
        """Create fallback explanation."""
        return ExplanationResult(
            explanation_id=f"fallback_{method_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            method=ExplanationMethod(method_name),
            scope=ExplanationScope.LOCAL,
            sample_id="unknown",
            feature_importance={},
            explanation_text=f"Fallback explanation due to error: {error_msg}",
            confidence=0.1,
            processing_time=0.0,
        )

    async def get_explanation_summary(self) -> dict[str, Any]:
        """Get summary of explanation history."""
        try:
            if not self.explanation_history:
                return {"message": "No explanations generated yet"}

            # Group by method
            method_counts = {}
            method_avg_confidence = {}
            method_avg_time = {}

            for explanation in self.explanation_history:
                method = explanation.method.value

                if method not in method_counts:
                    method_counts[method] = 0
                    method_avg_confidence[method] = []
                    method_avg_time[method] = []

                method_counts[method] += 1
                method_avg_confidence[method].append(explanation.confidence)
                method_avg_time[method].append(explanation.processing_time)

            # Calculate averages
            summary = {
                "total_explanations": len(self.explanation_history),
                "methods_used": list(method_counts.keys()),
                "method_statistics": {},
            }

            for method in method_counts:
                summary["method_statistics"][method] = {
                    "count": method_counts[method],
                    "avg_confidence": float(np.mean(method_avg_confidence[method])),
                    "avg_processing_time": float(np.mean(method_avg_time[method])),
                }

            return summary

        except Exception as e:
            logger.error(f"Explanation summary generation failed: {e}")
            return {"error": str(e)}
