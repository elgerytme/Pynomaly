"""
Quantify uncertainty use case implementation.

This module implements the business logic for quantifying uncertainty in
anomaly detection predictions through various statistical methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monorepo.application.dto.uncertainty_dto import (
    EnsembleUncertaintyRequest,
    EnsembleUncertaintyResponse,
    UncertaintyRequest,
    UncertaintyResponse,
)
from monorepo.domain.entities.detection_result import DetectionResult
from monorepo.domain.services.uncertainty_service import (
    UncertaintyQuantificationService,
)
from monorepo.domain.value_objects.confidence_interval import ConfidenceInterval


@dataclass
class QuantifyUncertaintyUseCase:
    """
    Use case for quantifying uncertainty in anomaly detection predictions.

    Provides various methods for calculating confidence intervals and
    uncertainty measures to help users understand the reliability of
    anomaly detection results.
    """

    uncertainty_service: UncertaintyQuantificationService

    def execute(self, request: UncertaintyRequest) -> UncertaintyResponse:
        """
        Execute uncertainty quantification for detection results.

        Args:
            request: Uncertainty quantification request

        Returns:
            UncertaintyResponse with confidence intervals and uncertainty metrics
        """
        if not request.detection_results:
            raise ValueError("Cannot quantify uncertainty without detection results")

        # Calculate uncertainty metrics
        uncertainty_metrics = self.uncertainty_service.calculate_prediction_uncertainty(
            detection_results=request.detection_results, method=request.method
        )

        # Calculate specific confidence intervals based on method
        confidence_intervals = self._calculate_confidence_intervals(
            request.detection_results, request.confidence_level, request.method
        )

        # Calculate additional uncertainty measures
        additional_metrics = self._calculate_additional_metrics(
            request.detection_results,
            request.include_prediction_intervals,
            request.include_entropy,
        )

        return UncertaintyResponse(
            confidence_intervals=confidence_intervals,
            uncertainty_metrics=uncertainty_metrics,
            additional_metrics=additional_metrics,
            method=request.method,
            confidence_level=request.confidence_level,
            n_samples=len(request.detection_results),
        )

    def execute_ensemble_uncertainty(
        self, request: EnsembleUncertaintyRequest
    ) -> EnsembleUncertaintyResponse:
        """
        Execute uncertainty quantification for ensemble predictions.

        Args:
            request: Ensemble uncertainty quantification request

        Returns:
            EnsembleUncertaintyResponse with ensemble-specific uncertainty metrics
        """
        if not request.ensemble_results:
            raise ValueError("Cannot quantify ensemble uncertainty without results")

        # Extract scores from ensemble results
        ensemble_scores = []
        for model_results in request.ensemble_results:
            model_scores = [result.score.value for result in model_results]
            ensemble_scores.append(model_scores)

        # Calculate ensemble uncertainty
        ensemble_metrics = self.uncertainty_service.calculate_ensemble_uncertainty(
            ensemble_scores=ensemble_scores, confidence_level=request.confidence_level
        )

        # Calculate per-model uncertainties
        model_uncertainties = []
        for i, model_results in enumerate(request.ensemble_results):
            model_uncertainty = (
                self.uncertainty_service.calculate_prediction_uncertainty(
                    detection_results=model_results, method=request.method
                )
            )
            model_uncertainties.append(
                {"model_index": i, "uncertainty_metrics": model_uncertainty}
            )

        # Calculate disagreement metrics
        disagreement_metrics = self._calculate_ensemble_disagreement(
            request.ensemble_results
        )

        return EnsembleUncertaintyResponse(
            ensemble_metrics=ensemble_metrics,
            model_uncertainties=model_uncertainties,
            disagreement_metrics=disagreement_metrics,
            method=request.method,
            confidence_level=request.confidence_level,
            n_models=len(request.ensemble_results),
        )

    def calculate_bootstrap_interval(
        self,
        scores: list[float],
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        statistic: str = "mean",
    ) -> ConfidenceInterval:
        """
        Calculate bootstrap confidence interval for a specific statistic.

        Args:
            scores: List of anomaly scores
            confidence_level: Desired confidence level
            n_bootstrap: Number of bootstrap samples
            statistic: Statistic to calculate ("mean", "median", "std")

        Returns:
            ConfidenceInterval for the specified statistic
        """
        return self.uncertainty_service.calculate_bootstrap_confidence_interval(
            scores=scores,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            statistic_function=statistic,
        )

    def calculate_bayesian_interval(
        self,
        binary_scores: list[int],
        confidence_level: float = 0.95,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> ConfidenceInterval:
        """
        Calculate Bayesian confidence interval for anomaly rate.

        Args:
            binary_scores: List of binary anomaly indicators (0 or 1)
            confidence_level: Desired confidence level
            prior_alpha: Alpha parameter of Beta prior
            prior_beta: Beta parameter of Beta prior

        Returns:
            ConfidenceInterval for anomaly rate
        """
        return self.uncertainty_service.calculate_bayesian_confidence_interval(
            scores=binary_scores,
            confidence_level=confidence_level,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
        )

    def calculate_prediction_interval(
        self, training_scores: list[float], confidence_level: float = 0.95
    ) -> ConfidenceInterval:
        """
        Calculate prediction interval for individual predictions.

        Args:
            training_scores: Historical anomaly scores for calibration
            confidence_level: Desired confidence level

        Returns:
            ConfidenceInterval for individual predictions
        """
        return self.uncertainty_service._calculate_prediction_interval(
            scores=np.array(training_scores), confidence_level=confidence_level
        )

    def _calculate_confidence_intervals(
        self,
        detection_results: list[DetectionResult],
        confidence_level: float,
        method: str,
    ) -> dict[str, ConfidenceInterval]:
        """Calculate various confidence intervals based on method."""
        scores = [result.score.value for result in detection_results]

        intervals = {}

        if method == "bootstrap":
            intervals["mean"] = (
                self.uncertainty_service.calculate_bootstrap_confidence_interval(
                    scores, confidence_level, statistic_function="mean"
                )
            )
            intervals["std"] = (
                self.uncertainty_service.calculate_bootstrap_confidence_interval(
                    scores, confidence_level, statistic_function="std"
                )
            )
        elif method == "normal":
            intervals["mean"] = (
                self.uncertainty_service.calculate_normal_confidence_interval(
                    scores, confidence_level
                )
            )
        elif method == "bayesian":
            # Convert scores to binary for Bayesian analysis
            binary_scores = [(score > 0.5) for score in scores]
            intervals["anomaly_rate"] = (
                self.uncertainty_service.calculate_bayesian_confidence_interval(
                    binary_scores, confidence_level
                )
            )

        return intervals

    def _calculate_additional_metrics(
        self,
        detection_results: list[DetectionResult],
        include_prediction_intervals: bool,
        include_entropy: bool,
    ) -> dict[str, float | ConfidenceInterval]:
        """Calculate additional uncertainty metrics."""
        additional_metrics = {}

        scores = np.array([result.score.value for result in detection_results])

        if include_prediction_intervals:
            prediction_interval = (
                self.uncertainty_service._calculate_prediction_interval(scores)
            )
            additional_metrics["prediction_interval"] = prediction_interval

        if include_entropy:
            entropy = self.uncertainty_service._calculate_entropy(scores)
            additional_metrics["entropy"] = entropy

        # Calculate other useful metrics
        additional_metrics["range"] = float(np.max(scores) - np.min(scores))
        additional_metrics["iqr"] = float(
            np.percentile(scores, 75) - np.percentile(scores, 25)
        )
        additional_metrics["median_absolute_deviation"] = float(
            np.median(np.abs(scores - np.median(scores)))
        )

        return additional_metrics

    def _calculate_ensemble_disagreement(
        self, ensemble_results: list[list[DetectionResult]]
    ) -> dict[str, float]:
        """Calculate disagreement metrics between ensemble models."""
        # Extract predictions (binary) from each model
        model_predictions = []
        for model_results in ensemble_results:
            predictions = [result.is_anomaly for result in model_results]
            model_predictions.append(predictions)

        model_predictions = np.array(model_predictions)

        # Calculate disagreement metrics
        disagreement_metrics = {}

        # Fraction of cases where models disagree
        n_samples = model_predictions.shape[1]
        disagreements = []

        for i in range(n_samples):
            sample_predictions = model_predictions[:, i]
            # Disagreement if not all models agree
            disagreement = len(set(sample_predictions)) > 1
            disagreements.append(disagreement)

        disagreement_metrics["disagreement_rate"] = float(np.mean(disagreements))

        # Average pairwise disagreement
        pairwise_disagreements = []
        n_models = model_predictions.shape[0]

        for i in range(n_models):
            for j in range(i + 1, n_models):
                pairwise_disagreement = np.mean(
                    model_predictions[i] != model_predictions[j]
                )
                pairwise_disagreements.append(pairwise_disagreement)

        disagreement_metrics["avg_pairwise_disagreement"] = float(
            np.mean(pairwise_disagreements)
        )

        # Entropy of ensemble predictions
        ensemble_probs = np.mean(model_predictions.astype(float), axis=0)
        epsilon = 1e-15
        ensemble_probs = np.clip(ensemble_probs, epsilon, 1 - epsilon)

        entropy = -np.mean(
            ensemble_probs * np.log2(ensemble_probs)
            + (1 - ensemble_probs) * np.log2(1 - ensemble_probs)
        )
        disagreement_metrics["ensemble_entropy"] = float(entropy)

        return disagreement_metrics
