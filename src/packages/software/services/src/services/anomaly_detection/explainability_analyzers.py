"""Analyzers for bias analysis and trust scoring."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from monorepo.application.services.explainability_core import (
    BiasAnalysisConfig,
    BiasAnalysisResult,
    TrustScoreConfig,
    TrustScoreResult,
)
from monorepo.domain.entities import Dataset
from monorepo.shared.protocols import DetectorProtocol

logger = logging.getLogger(__name__)


class BiasAnalyzer:
    """Analyzer for detecting and measuring model bias."""

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
            metric < 0.8 for metric in fairness_metrics.values() if metric <= 1.0
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


class TrustScoreAnalyzer:
    """Analyzer for computing trust scores."""

    async def assess_trust_score(
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


class CounterfactualAnalyzer:
    """Analyzer for generating counterfactual explanations."""

    async def generate_counterfactual_explanations(
        self,
        detector: DetectorProtocol,
        dataset: Dataset,
        target_instances: list[int],
        n_counterfactuals: int = 5,
        optimization_method: str = "random",
    ) -> dict[str, Any]:
        """Generate counterfactual explanations for given instances."""
        try:
            logger.info(
                f"Generating counterfactual explanations for {len(target_instances)} instances"
            )

            X = dataset.data.values if hasattr(dataset.data, "values") else dataset.data
            counterfactuals = {}

            for instance_idx in target_instances:
                if instance_idx >= len(X):
                    continue

                original_instance = X[instance_idx]
                original_prediction = detector.decision_function(
                    original_instance.reshape(1, -1)
                )[0]

                # Generate counterfactuals
                cf_instances = []

                for _ in range(n_counterfactuals * 10):
                    if optimization_method == "random":
                        # Random perturbation
                        noise = np.random.normal(0, 0.1, original_instance.shape)
                        candidate = original_instance + noise
                    elif optimization_method == "genetic":
                        # Simple genetic algorithm approach
                        candidate = self._genetic_counterfactual(
                            original_instance, detector, X
                        )
                    else:
                        # Gradient-based (simplified)
                        candidate = self._gradient_counterfactual(
                            original_instance, detector
                        )

                    # Check if prediction changed significantly
                    candidate_prediction = detector.decision_function(
                        candidate.reshape(1, -1)
                    )[0]

                    if abs(candidate_prediction - original_prediction) > 0.1:
                        distance = np.linalg.norm(candidate - original_instance)
                        cf_instances.append(
                            {
                                "instance": candidate.tolist(),
                                "prediction": float(candidate_prediction),
                                "distance": float(distance),
                                "changes": self._calculate_feature_changes(
                                    original_instance, candidate, dataset.features or []
                                ),
                            }
                        )

                # Select best counterfactuals
                cf_instances.sort(key=lambda x: x["distance"])

                counterfactuals[f"instance_{instance_idx}"] = {
                    "original": {
                        "instance": original_instance.tolist(),
                        "prediction": float(original_prediction),
                    },
                    "counterfactuals": cf_instances[:n_counterfactuals],
                    "summary": {
                        "generated": len(cf_instances),
                        "selected": min(n_counterfactuals, len(cf_instances)),
                        "avg_distance": float(
                            np.mean(
                                [
                                    cf["distance"]
                                    for cf in cf_instances[:n_counterfactuals]
                                ]
                            )
                        )
                        if cf_instances
                        else 0.0,
                    },
                }

            return counterfactuals

        except Exception as e:
            logger.error(f"Counterfactual generation failed: {e}")
            return {"error": str(e)}

    def _genetic_counterfactual(
        self, original: np.ndarray, detector: DetectorProtocol, X: np.ndarray
    ) -> np.ndarray:
        """Generate counterfactual using genetic algorithm approach."""
        population_size = 50
        generations = 10

        # Initialize population
        population = []
        for _ in range(population_size):
            noise = np.random.normal(0, 0.1, original.shape)
            candidate = original + noise
            population.append(candidate)

        original_pred = detector.decision_function(original.reshape(1, -1))[0]

        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for candidate in population:
                candidate_pred = detector.decision_function(candidate.reshape(1, -1))[0]

                # Fitness: prediction change + distance penalty
                pred_change = abs(candidate_pred - original_pred)
                distance_penalty = np.linalg.norm(candidate - original) * 0.1
                fitness = pred_change - distance_penalty

                fitness_scores.append(fitness)

            # Selection (tournament selection)
            new_population = []
            for _ in range(population_size):
                tournament_size = 5
                tournament_indices = np.random.choice(
                    len(population), tournament_size, replace=False
                )
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                new_population.append(population[winner_idx].copy())

            # Mutation
            for i in range(len(new_population)):
                if np.random.random() < 0.1:
                    mutation = np.random.normal(0, 0.05, new_population[i].shape)
                    new_population[i] += mutation

            population = new_population

        # Return best individual
        best_idx = np.argmax(fitness_scores)
        return population[best_idx]

    def _gradient_counterfactual(
        self, original: np.ndarray, detector: DetectorProtocol
    ) -> np.ndarray:
        """Generate counterfactual using gradient-based approach."""
        current = original.copy()
        learning_rate = 0.01
        steps = 100

        original_pred = detector.decision_function(original.reshape(1, -1))[0]

        for _ in range(steps):
            # Compute gradient
            epsilon = 1e-5
            gradients = np.zeros_like(current)

            for i in range(len(current)):
                perturbed = current.copy()
                perturbed[i] += epsilon

                pred_plus = detector.decision_function(perturbed.reshape(1, -1))[0]
                pred_minus = detector.decision_function(current.reshape(1, -1))[0]

                gradient = (pred_plus - pred_minus) / epsilon
                gradients[i] = gradient

            # Update in direction that changes prediction most
            current += learning_rate * gradients

            # Check if prediction changed significantly
            current_pred = detector.decision_function(current.reshape(1, -1))[0]
            if abs(current_pred - original_pred) > 0.1:
                break

        return current

    def _calculate_feature_changes(
        self, original: np.ndarray, modified: np.ndarray, feature_names: list[str]
    ) -> dict[str, dict[str, float]]:
        """Calculate changes between original and modified instances."""
        changes = {}

        for i, feature_name in enumerate(feature_names):
            if i < len(original) and i < len(modified):
                original_val = original[i]
                modified_val = modified[i]

                if abs(modified_val - original_val) > 1e-6:
                    changes[feature_name] = {
                        "original": float(original_val),
                        "modified": float(modified_val),
                        "absolute_change": float(modified_val - original_val),
                        "relative_change": float(
                            (modified_val - original_val) / (original_val + 1e-10)
                        ),
                    }

        return changes
