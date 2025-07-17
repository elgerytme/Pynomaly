"""
FastAPI endpoints for uncertainty quantification functionality.

This module provides REST API endpoints for calculating confidence intervals
and uncertainty measures for anomaly processing predictions.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from interfaces.application.dto.uncertainty_dto import (
    EnsembleUncertaintyRequest,
    UncertaintyRequest,
    UncertaintyResponse,
)
from interfaces.application.use_cases.quantify_uncertainty import (
    QuantifyUncertaintyUseCase,
)
from interfaces.domain.entities.detection_result import DetectionResult
from interfaces.domain.services.uncertainty_service import (
    UncertaintyQuantificationService,
)
from interfaces.domain.value_objects.anomaly_score import AnomalyScore
from interfaces.domain.value_objects.confidence_interval import ConfidenceInterval

router = APIRouter(prefix="/uncertainty", tags=["uncertainty"])


class DetectionResultModel(BaseModel):
    """Pydantic processor for processing result API representation."""

    sample_id: str = Field(..., description="Unique identifier for the sample")
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Anomaly score between 0 and 1"
    )
    is_anomaly: bool = Field(
        ..., description="Whether the sample is classified as anomaly"
    )
    timestamp: str | None = Field(None, description="Timestamp of processing")
    processor_version: str = Field(..., description="Version of the processor used")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class UncertaintyRequestModel(BaseModel):
    """Pydantic processor for uncertainty quantification request."""

    processing_results: list[DetectionResultModel] = Field(
        ..., min_items=1, description="List of processing results to analyze"
    )
    method: str = Field(
        "bootstrap",
        description="Uncertainty calculation method",
        regex="^(bootstrap|normal|bayesian)$",
    )
    confidence_level: float = Field(
        0.95, ge=0.0, le=1.0, description="Desired confidence level"
    )
    include_prediction_intervals: bool = Field(
        True, description="Whether to include prediction intervals"
    )
    include_entropy: bool = Field(
        True, description="Whether to include entropy-based uncertainty"
    )
    n_bootstrap: int = Field(
        1000, ge=100, le=10000, description="Number of bootstrap samples"
    )


class EnsembleUncertaintyRequestModel(BaseModel):
    """Pydantic processor for ensemble uncertainty quantification request."""

    ensemble_results: list[list[DetectionResultModel]] = Field(
        ..., min_items=2, description="List of processing results from each processor"
    )
    method: str = Field(
        "bootstrap",
        description="Uncertainty calculation method",
        regex="^(bootstrap|normal|bayesian)$",
    )
    confidence_level: float = Field(
        0.95, ge=0.0, le=1.0, description="Desired confidence level"
    )
    include_disagreement: bool = Field(
        True, description="Whether to include disagreement measurements"
    )


class BootstrapRequestModel(BaseModel):
    """Pydantic processor for bootstrap confidence interval request."""

    scores: list[float] = Field(..., min_items=1, description="List of anomaly scores")
    confidence_level: float = Field(
        0.95, ge=0.0, le=1.0, description="Confidence level"
    )
    n_bootstrap: int = Field(
        1000, ge=100, le=10000, description="Number of bootstrap samples"
    )
    statistic: str = Field(
        "mean", description="Statistic to calculate", regex="^(mean|median|std|var)$"
    )


class BayesianRequestModel(BaseModel):
    """Pydantic processor for Bayesian confidence interval request."""

    binary_scores: list[int] = Field(
        ..., min_items=1, description="Binary anomaly indicators"
    )
    confidence_level: float = Field(
        0.95, ge=0.0, le=1.0, description="Confidence level"
    )
    prior_alpha: float = Field(1.0, gt=0.0, description="Alpha parameter of Beta prior")
    prior_beta: float = Field(1.0, gt=0.0, description="Beta parameter of Beta prior")


class PredictionIntervalRequestModel(BaseModel):
    """Pydantic processor for prediction interval request."""

    training_scores: list[float] = Field(
        ..., min_items=10, description="Historical anomaly scores"
    )
    confidence_level: float = Field(
        0.95, ge=0.0, le=1.0, description="Confidence level"
    )


class ConfidenceIntervalModel(BaseModel):
    """Pydantic processor for confidence interval response."""

    lower: float = Field(..., description="Lower bound of interval")
    upper: float = Field(..., description="Upper bound of interval")
    confidence_level: float = Field(..., description="Confidence level used")
    method: str = Field(..., description="Method used for calculation")
    width: float = Field(..., description="Width of interval")
    midpoint: float = Field(..., description="Midpoint of interval")
    margin_of_error: float = Field(..., description="Margin of error")


class UncertaintyResponseModel(BaseModel):
    """Pydantic processor for uncertainty quantification response."""

    confidence_intervals: dict[str, ConfidenceIntervalModel] = Field(
        ..., description="Confidence intervals by type"
    )
    uncertainty_measurements: dict[str, float | ConfidenceIntervalModel] = Field(
        ..., description="General uncertainty measurements"
    )
    additional_measurements: dict[str, float | ConfidenceIntervalModel] = Field(
        ..., description="Additional uncertainty measures"
    )
    method: str = Field(..., description="Method used for calculation")
    confidence_level: float = Field(..., description="Confidence level used")
    n_samples: int = Field(..., description="Number of samples analyzed")
    summary: dict[str, str | float] = Field(..., description="Summary of key measurements")


def _convert_to_detection_result(model: DetectionResultModel) -> DetectionResult:
    """Convert Pydantic processor to domain entity."""
    score = AnomalyScore(value=processor.score)
    return DetectionResult(
        sample_id=processor.sample_id,
        score=score,
        is_anomaly=processor.is_anomaly,
        timestamp=processor.timestamp,
        processor_version=processor.processor_version,
        metadata=processor.metadata,
    )


def _convert_confidence_interval_to_model(
    ci: ConfidenceInterval,
) -> ConfidenceIntervalModel:
    """Convert domain confidence interval to Pydantic processor."""
    return ConfidenceIntervalModel(
        lower=ci.lower,
        upper=ci.upper,
        confidence_level=ci.confidence_level,
        method=ci.method,
        width=ci.width(),
        midpoint=ci.midpoint(),
        margin_of_error=ci.margin_of_error,
    )


def get_uncertainty_use_case() -> QuantifyUncertaintyUseCase:
    """Dependency injection for uncertainty use case."""
    uncertainty_service = UncertaintyQuantificationService(random_seed=42)
    return QuantifyUncertaintyUseCase(uncertainty_service=uncertainty_service)


def _convert_uncertainty_response_to_model(
    response: UncertaintyResponse,
) -> UncertaintyResponseModel:
    """Convert application response to Pydantic processor."""
    # Convert confidence intervals
    confidence_intervals = {
        key: _convert_confidence_interval_to_processor(ci)
        for key, ci in response.confidence_intervals.items()
    }

    # Convert uncertainty measurements (handling mixed types)
    uncertainty_measurements = {}
    for key, value in response.uncertainty_measurements.items():
        if isinstance(value, ConfidenceInterval):
            uncertainty_measurements[key] = _convert_confidence_interval_to_processor(value)
        else:
            uncertainty_measurements[key] = value

    # Convert additional measurements
    additional_measurements = {}
    for key, value in response.additional_measurements.items():
        if isinstance(value, ConfidenceInterval):
            additional_measurements[key] = _convert_confidence_interval_to_processor(value)
        else:
            additional_measurements[key] = value

    return UncertaintyResponseModel(
        confidence_intervals=confidence_intervals,
        uncertainty_measurements=uncertainty_measurements,
        additional_measurements=additional_measurements,
        method=response.method,
        confidence_level=response.confidence_level,
        n_samples=response.n_samples,
        summary=response.get_summary(),
    )


@router.post(
    "/quantify",
    response_processor=UncertaintyResponseModel,
    status_code=status.HTTP_200_OK,
    summary="Quantify uncertainty in anomaly processing results",
    description="""
    Calculate uncertainty measurements and confidence intervals for anomaly processing predictions.

    Supports multiple statistical methods:
    - **Bootstrap**: Resampling-based confidence intervals
    - **Normal**: Assumes normal distribution of scores
    - **Bayesian**: Beta-binomial processor for binary outcomes

    Returns comprehensive uncertainty analysis including confidence intervals,
    prediction intervals, and entropy-based uncertainty measures.
    """,
)
async def quantify_uncertainty(
    request: UncertaintyRequestModel,
    use_case: QuantifyUncertaintyUseCase = Depends(get_uncertainty_use_case),
) -> UncertaintyResponseModel:
    """Quantify uncertainty in anomaly processing results."""
    try:
        # Convert API models to domain entities
        processing_results = [
            _convert_to_processing_result(result) for result in request.processing_results
        ]

        # Create application request
        app_request = UncertaintyRequest(
            processing_results=processing_results,
            method=request.method,
            confidence_level=request.confidence_level,
            include_prediction_intervals=request.include_prediction_intervals,
            include_entropy=request.include_entropy,
            n_bootstrap=request.n_bootstrap,
        )

        # Execute use case
        response = use_case.execute(app_request)

        # Convert to API response processor
        return _convert_uncertainty_response_to_processor(response)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.post(
    "/ensemble",
    response_processor=dict,
    status_code=status.HTTP_200_OK,
    summary="Quantify uncertainty in ensemble predictions",
    description="""
    Calculate uncertainty measurements for ensemble anomaly processing models.

    Analyzes uncertainty across multiple models including:
    - **Ensemble measurements**: Overall ensemble uncertainty
    - **Processor disagreement**: How much models disagree
    - **Aleatoric vs Epistemic**: Data vs processor uncertainty

    Useful for understanding processor reliability and confidence in ensemble predictions.
    """,
)
async def quantify_ensemble_uncertainty(
    request: EnsembleUncertaintyRequestModel,
    use_case: QuantifyUncertaintyUseCase = Depends(get_uncertainty_use_case),
) -> dict:
    """Quantify uncertainty in ensemble predictions."""
    try:
        # Convert API models to domain entities
        ensemble_results = []
        for processor_results in request.ensemble_results:
            processor_processing_results = [
                _convert_to_processing_result(result) for result in processor_results
            ]
            ensemble_results.append(processor_processing_results)

        # Create application request
        app_request = EnsembleUncertaintyRequest(
            ensemble_results=ensemble_results,
            method=request.method,
            confidence_level=request.confidence_level,
            include_disagreement=request.include_disagreement,
        )

        # Execute use case
        response = use_case.execute_ensemble_uncertainty(app_request)

        # Convert to dictionary for response
        return response.to_dict()

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.post(
    "/bootstrap",
    response_processor=ConfidenceIntervalModel,
    status_code=status.HTTP_200_OK,
    summary="Calculate bootstrap confidence interval",
    description="""
    Calculate confidence interval using bootstrap resampling method.

    Bootstrap is a non-parametric method that doesn't assume any specific
    distribution. It's robust and works well for various statistics.

    Supported statistics:
    - **mean**: Average of scores
    - **median**: Middle value of scores
    - **std**: Standard deviation of scores
    - **var**: Variance of scores
    """,
)
async def calculate_bootstrap_interval(
    request: BootstrapRequestModel,
    use_case: QuantifyUncertaintyUseCase = Depends(get_uncertainty_use_case),
) -> ConfidenceIntervalModel:
    """Calculate bootstrap confidence interval for a specific statistic."""
    try:
        # Validate binary scores for Bayesian
        if any(score < 0 or score > 1 for score in request.scores):
            raise ValueError("All scores must be between 0 and 1")

        # Execute bootstrap calculation
        ci = use_case.calculate_bootstrap_interval(
            scores=request.scores,
            confidence_level=request.confidence_level,
            n_bootstrap=request.n_bootstrap,
            statistic=request.statistic,
        )

        return _convert_confidence_interval_to_processor(ci)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.post(
    "/bayesian",
    response_processor=ConfidenceIntervalModel,
    status_code=status.HTTP_200_OK,
    summary="Calculate Bayesian confidence interval",
    description="""
    Calculate confidence interval using Bayesian inference with Beta prior.

    This method is particularly useful for binary classification problems
    where you want to estimate the anomaly rate with prior knowledge.

    The Beta prior allows you to incorporate domain knowledge:
    - **prior_alpha**: Strength of belief in positive outcomes
    - **prior_beta**: Strength of belief in negative outcomes
    - Equal values (e.g., 1,1) represent uniform prior (no preference)
    """,
)
async def calculate_bayesian_interval(
    request: BayesianRequestModel,
    use_case: QuantifyUncertaintyUseCase = Depends(get_uncertainty_use_case),
) -> ConfidenceIntervalModel:
    """Calculate Bayesian confidence interval for anomaly rate."""
    try:
        # Validate binary scores
        if not all(score in [0, 1] for score in request.binary_scores):
            raise ValueError("Binary scores must be 0 or 1")

        # Execute Bayesian calculation
        ci = use_case.calculate_bayesian_interval(
            binary_scores=request.binary_scores,
            confidence_level=request.confidence_level,
            prior_alpha=request.prior_alpha,
            prior_beta=request.prior_beta,
        )

        return _convert_confidence_interval_to_processor(ci)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.post(
    "/prediction-interval",
    response_processor=ConfidenceIntervalModel,
    status_code=status.HTTP_200_OK,
    summary="Calculate prediction interval",
    description="""
    Calculate prediction interval for individual future predictions.

    Prediction intervals are wider than confidence intervals because they
    account for both:
    1. **Sampling uncertainty**: Uncertainty in estimating population parameters
    2. **Individual variation**: Natural variation in individual observations

    Use this when you want to predict where a single new observation will fall,
    rather than estimating a population parameter.
    """,
)
async def calculate_prediction_interval(
    request: PredictionIntervalRequestModel,
    use_case: QuantifyUncertaintyUseCase = Depends(get_uncertainty_use_case),
) -> ConfidenceIntervalModel:
    """Calculate prediction interval for individual predictions."""
    try:
        # Validate training scores
        if any(score < 0 or score > 1 for score in request.training_scores):
            raise ValueError("All training scores must be between 0 and 1")

        # Execute prediction interval calculation
        ci = use_case.calculate_prediction_interval(
            training_scores=request.training_scores,
            confidence_level=request.confidence_level,
        )

        return _convert_confidence_interval_to_processor(ci)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}",
        )


@router.get(
    "/methods",
    response_processor=dict[str, dict[str, str]],
    status_code=status.HTTP_200_OK,
    summary="Get available uncertainty quantification methods",
    description="""
    Get information about available uncertainty quantification methods
    and their characteristics.
    """,
)
async def get_uncertainty_methods() -> dict[str, dict[str, str]]:
    """Get available uncertainty quantification methods."""
    return {
        "bootstrap": {
            "description": "Non-parametric resampling method",
            "assumptions": "None (distribution-free)",
            "use_case": "General purpose, robust for any distribution",
            "computational_cost": "Medium to High",
            "sample_size": "Works with small samples",
        },
        "normal": {
            "description": "Assumes normal distribution of data",
            "assumptions": "Normality of underlying distribution",
            "use_case": "When data follows normal distribution",
            "computational_cost": "Low",
            "sample_size": "Works better with larger samples (n>30)",
        },
        "bayesian": {
            "description": "Incorporates prior knowledge using Beta-binomial processor",
            "assumptions": "Beta prior distribution for parameters",
            "use_case": "Binary outcomes with prior knowledge",
            "computational_cost": "Low",
            "sample_size": "Works with any sample size",
        },
    }
