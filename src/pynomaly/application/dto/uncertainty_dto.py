"""
Data Transfer Objects for uncertainty quantification operations.

This module defines the request and response DTOs for uncertainty
quantification use cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.value_objects.confidence_interval import ConfidenceInterval


@dataclass
class UncertaintyRequest:
    """
    Request for uncertainty quantification.

    Attributes:
        detection_results: List of detection results to analyze
        method: Method for uncertainty calculation ("bootstrap", "normal", "bayesian")
        confidence_level: Desired confidence level (0.0 to 1.0)
        include_prediction_intervals: Whether to include prediction intervals
        include_entropy: Whether to include entropy-based uncertainty
        n_bootstrap: Number of bootstrap samples (for bootstrap method)
    """

    detection_results: List[DetectionResult]
    method: str = "bootstrap"
    confidence_level: float = 0.95
    include_prediction_intervals: bool = True
    include_entropy: bool = True
    n_bootstrap: int = 1000

    def __post_init__(self) -> None:
        """Validate request parameters."""
        if not self.detection_results:
            raise ValueError("Detection results cannot be empty")

        if self.method not in ["bootstrap", "normal", "bayesian"]:
            raise ValueError(f"Unknown method: {self.method}")

        if not 0.0 <= self.confidence_level <= 1.0:
            raise ValueError(
                f"Confidence level must be between 0 and 1, got {self.confidence_level}"
            )

        if self.n_bootstrap < 100:
            raise ValueError(
                f"n_bootstrap must be at least 100, got {self.n_bootstrap}"
            )


@dataclass
class UncertaintyResponse:
    """
    Response from uncertainty quantification.

    Attributes:
        confidence_intervals: Dictionary of confidence intervals by type
        uncertainty_metrics: General uncertainty metrics
        additional_metrics: Additional uncertainty measures
        method: Method used for calculation
        confidence_level: Confidence level used
        n_samples: Number of samples analyzed
    """

    confidence_intervals: Dict[str, ConfidenceInterval]
    uncertainty_metrics: Dict[str, Union[float, ConfidenceInterval]]
    additional_metrics: Dict[str, Union[float, ConfidenceInterval]]
    method: str
    confidence_level: float
    n_samples: int

    def to_dict(self) -> dict:
        """Convert response to dictionary representation."""
        return {
            "confidence_intervals": {
                key: interval.to_dict()
                for key, interval in self.confidence_intervals.items()
            },
            "uncertainty_metrics": {
                key: (
                    value.to_dict() if isinstance(value, ConfidenceInterval) else value
                )
                for key, value in self.uncertainty_metrics.items()
            },
            "additional_metrics": {
                key: (
                    value.to_dict() if isinstance(value, ConfidenceInterval) else value
                )
                for key, value in self.additional_metrics.items()
            },
            "method": self.method,
            "confidence_level": self.confidence_level,
            "n_samples": self.n_samples,
        }

    def get_summary(self) -> Dict[str, Union[str, float]]:
        """Get a summary of key uncertainty metrics."""
        summary = {
            "method": self.method,
            "confidence_level": self.confidence_level,
            "n_samples": self.n_samples,
        }

        # Add key metrics
        if "mean_score" in self.uncertainty_metrics:
            summary["mean_score"] = self.uncertainty_metrics["mean_score"]

        if "std_score" in self.uncertainty_metrics:
            summary["std_score"] = self.uncertainty_metrics["std_score"]

        if "coefficient_of_variation" in self.uncertainty_metrics:
            summary["coefficient_of_variation"] = self.uncertainty_metrics[
                "coefficient_of_variation"
            ]

        if "entropy" in self.additional_metrics:
            summary["entropy"] = self.additional_metrics["entropy"]

        # Add confidence interval widths
        for key, interval in self.confidence_intervals.items():
            summary[f"{key}_ci_width"] = interval.width()

        return summary


@dataclass
class EnsembleUncertaintyRequest:
    """
    Request for ensemble uncertainty quantification.

    Attributes:
        ensemble_results: List of detection results from each model in ensemble
        method: Method for uncertainty calculation
        confidence_level: Desired confidence level
        include_disagreement: Whether to include disagreement metrics
    """

    ensemble_results: List[List[DetectionResult]]
    method: str = "bootstrap"
    confidence_level: float = 0.95
    include_disagreement: bool = True

    def __post_init__(self) -> None:
        """Validate ensemble request parameters."""
        if not self.ensemble_results:
            raise ValueError("Ensemble results cannot be empty")

        if len(self.ensemble_results) < 2:
            raise ValueError("Ensemble must contain at least 2 models")

        # Check that all models have the same number of results
        result_counts = [len(model_results) for model_results in self.ensemble_results]
        if len(set(result_counts)) > 1:
            raise ValueError(
                "All models must have the same number of detection results"
            )

        if self.method not in ["bootstrap", "normal", "bayesian"]:
            raise ValueError(f"Unknown method: {self.method}")

        if not 0.0 <= self.confidence_level <= 1.0:
            raise ValueError(
                f"Confidence level must be between 0 and 1, got {self.confidence_level}"
            )


@dataclass
class EnsembleUncertaintyResponse:
    """
    Response from ensemble uncertainty quantification.

    Attributes:
        ensemble_metrics: Ensemble-level uncertainty metrics
        model_uncertainties: Per-model uncertainty metrics
        disagreement_metrics: Model disagreement metrics
        method: Method used for calculation
        confidence_level: Confidence level used
        n_models: Number of models in ensemble
    """

    ensemble_metrics: Dict[str, Union[float, ConfidenceInterval]]
    model_uncertainties: List[Dict[str, Union[int, Dict]]]
    disagreement_metrics: Dict[str, float]
    method: str
    confidence_level: float
    n_models: int

    def to_dict(self) -> dict:
        """Convert response to dictionary representation."""
        return {
            "ensemble_metrics": {
                key: (
                    value.to_dict() if isinstance(value, ConfidenceInterval) else value
                )
                for key, value in self.ensemble_metrics.items()
            },
            "model_uncertainties": self.model_uncertainties,
            "disagreement_metrics": self.disagreement_metrics,
            "method": self.method,
            "confidence_level": self.confidence_level,
            "n_models": self.n_models,
        }

    def get_summary(self) -> Dict[str, Union[str, float]]:
        """Get a summary of key ensemble uncertainty metrics."""
        summary = {
            "method": self.method,
            "confidence_level": self.confidence_level,
            "n_models": self.n_models,
        }

        # Add key ensemble metrics
        if "ensemble_mean" in self.ensemble_metrics:
            summary["ensemble_mean"] = self.ensemble_metrics["ensemble_mean"]

        if "ensemble_std" in self.ensemble_metrics:
            summary["ensemble_std"] = self.ensemble_metrics["ensemble_std"]

        if "aleatoric_uncertainty" in self.ensemble_metrics:
            summary["aleatoric_uncertainty"] = self.ensemble_metrics[
                "aleatoric_uncertainty"
            ]

        if "epistemic_uncertainty" in self.ensemble_metrics:
            summary["epistemic_uncertainty"] = self.ensemble_metrics[
                "epistemic_uncertainty"
            ]

        # Add disagreement metrics
        summary.update(self.disagreement_metrics)

        return summary


@dataclass
class BootstrapRequest:
    """
    Request for bootstrap confidence interval calculation.

    Attributes:
        scores: List of anomaly scores
        confidence_level: Desired confidence level
        n_bootstrap: Number of bootstrap samples
        statistic: Statistic to calculate ("mean", "median", "std")
    """

    scores: List[float]
    confidence_level: float = 0.95
    n_bootstrap: int = 1000
    statistic: str = "mean"

    def __post_init__(self) -> None:
        """Validate bootstrap request parameters."""
        if not self.scores:
            raise ValueError("Scores cannot be empty")

        if not 0.0 <= self.confidence_level <= 1.0:
            raise ValueError(
                f"Confidence level must be between 0 and 1, got {self.confidence_level}"
            )

        if self.n_bootstrap < 100:
            raise ValueError(
                f"n_bootstrap must be at least 100, got {self.n_bootstrap}"
            )

        if self.statistic not in ["mean", "median", "std", "var"]:
            raise ValueError(f"Unknown statistic: {self.statistic}")


@dataclass
class BayesianRequest:
    """
    Request for Bayesian confidence interval calculation.

    Attributes:
        binary_scores: List of binary anomaly indicators (0 or 1)
        confidence_level: Desired confidence level
        prior_alpha: Alpha parameter of Beta prior
        prior_beta: Beta parameter of Beta prior
    """

    binary_scores: List[int]
    confidence_level: float = 0.95
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    def __post_init__(self) -> None:
        """Validate Bayesian request parameters."""
        if not self.binary_scores:
            raise ValueError("Binary scores cannot be empty")

        # Check that scores are binary
        unique_scores = set(self.binary_scores)
        if not unique_scores.issubset({0, 1}):
            raise ValueError("Binary scores must contain only 0 and 1")

        if not 0.0 <= self.confidence_level <= 1.0:
            raise ValueError(
                f"Confidence level must be between 0 and 1, got {self.confidence_level}"
            )

        if self.prior_alpha <= 0 or self.prior_beta <= 0:
            raise ValueError("Prior parameters must be positive")


@dataclass
class PredictionIntervalRequest:
    """
    Request for prediction interval calculation.

    Attributes:
        training_scores: Historical anomaly scores for calibration
        confidence_level: Desired confidence level
    """

    training_scores: List[float]
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        """Validate prediction interval request parameters."""
        if not self.training_scores:
            raise ValueError("Training scores cannot be empty")

        if len(self.training_scores) < 10:
            raise ValueError(
                "Need at least 10 training scores for reliable prediction intervals"
            )

        if not 0.0 <= self.confidence_level <= 1.0:
            raise ValueError(
                f"Confidence level must be between 0 and 1, got {self.confidence_level}"
            )
