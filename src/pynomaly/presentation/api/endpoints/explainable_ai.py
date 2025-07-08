"""REST API endpoints for explainable AI (XAI) service."""

from datetime import timedelta
from typing import Any, Optional, Union
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from pynomaly.application.services.explainable_ai_service import (
    ExplainableAIService,
    ExplanationConfiguration,
    ExplanationNotSupportedError,
    InsufficientDataError,
)
from pynomaly.domain.entities.explainable_ai import (
    BiasType,
    ExplanationAudience,
    ExplanationMethod,
    TrustLevel,
)
from pynomaly.presentation.api.deps import (
    get_current_user,
    get_explainable_ai_service,
    require_read,
)

router = APIRouter(prefix="/explainable-ai", tags=["Explainable AI"])


# ==================== Request/Response Models ====================


class ExplainPredictionRequest(BaseModel):
    """Request model for single prediction explanation."""

    model_id: str = Field(..., description="Model identifier")
    instance_data: list[float] = Field(..., description="Input instance data")
    feature_names: Optional[list[str]] = Field(None, description="Feature names")
    explanation_method: Optional[str] = Field(
        None, description="Explanation method to use"
    )
    num_features: int = Field(10, description="Number of top features to explain")
    target_audience: Optional[str] = Field(
        None, description="Target audience for explanation"
    )
    enable_counterfactuals: bool = Field(
        False, description="Generate counterfactual explanations"
    )

    @field_validator("instance_data")
    @classmethod
    def validate_instance_data(cls, v):
        if not v:
            raise ValueError("Instance data cannot be empty")
        return v

    @field_validator("num_features")
    @classmethod
    def validate_num_features(cls, v):
        if v <= 0:
            raise ValueError("Number of features must be positive")
        return v


class ExplainModelGlobalRequest(BaseModel):
    """Request model for global model explanation."""

    model_id: str = Field(..., description="Model identifier")
    feature_names: Optional[list[str]] = Field(None, description="Feature names")
    explanation_method: Optional[str] = Field(
        None, description="Explanation method to use"
    )
    num_features: int = Field(10, description="Number of top features to explain")
    sample_size: int = Field(1000, description="Sample size for global explanation")
    enable_interaction_analysis: bool = Field(
        True, description="Analyze feature interactions"
    )
    enable_bias_detection: bool = Field(True, description="Detect model bias")

    @field_validator("sample_size")
    @classmethod
    def validate_sample_size(cls, v):
        if v <= 0:
            raise ValueError("Sample size must be positive")
        return v


class FeatureImportanceRequest(BaseModel):
    """Request model for feature importance analysis."""

    model_id: str = Field(..., description="Model identifier")
    feature_names: Optional[list[str]] = Field(None, description="Feature names")
    explanation_method: str = Field(
        "permutation_importance", description="Method for importance calculation"
    )
    sample_size: int = Field(1000, description="Sample size for analysis")

    @field_validator("explanation_method")
    @classmethod
    def validate_method(cls, v):
        valid_methods = [
            "permutation_importance",
            "feature_ablation",
            "shap_tree",
            "shap_kernel",
            "lime",
        ]
        if v not in valid_methods:
            raise ValueError(f"Method must be one of: {valid_methods}")
        return v


class BiasDetectionRequest(BaseModel):
    """Request model for bias detection."""

    model_id: str = Field(..., description="Model identifier")
    protected_attributes: list[str] = Field(
        ..., description="Protected attribute names"
    )
    feature_names: list[str] = Field(..., description="All feature names")
    sample_size: int = Field(1000, description="Sample size for bias analysis")
    bias_threshold: float = Field(0.3, description="Bias detection threshold")

    @field_validator("protected_attributes")
    @classmethod
    def validate_protected_attributes(cls, v):
        if not v:
            raise ValueError("Protected attributes cannot be empty")
        return v

    @field_validator("bias_threshold")
    @classmethod
    def validate_bias_threshold(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("Bias threshold must be between 0.0 and 1.0")
        return v


class CounterfactualRequest(BaseModel):
    """Request model for counterfactual explanations."""

    model_id: str = Field(..., description="Model identifier")
    instance_data: list[float] = Field(..., description="Input instance data")
    feature_names: Optional[list[str]] = Field(None, description="Feature names")
    num_counterfactuals: int = Field(5, description="Number of counterfactuals")
    max_distance: Optional[float] = Field(
        None, description="Maximum distance for counterfactuals"
    )

    @field_validator("num_counterfactuals")
    @classmethod
    def validate_num_counterfactuals(cls, v):
        if not (1 <= v <= 20):
            raise ValueError("Number of counterfactuals must be between 1 and 20")
        return v


class TrustAssessmentRequest(BaseModel):
    """Request model for trust assessment."""

    explanation_id: str = Field(..., description="Explanation ID to assess")
    model_id: str = Field(..., description="Model identifier")
    include_validation: bool = Field(True, description="Include validation tests")
    validation_sample_size: int = Field(100, description="Sample size for validation")


# Response Models


class FeatureImportanceResponse(BaseModel):
    """Response model for feature importance."""

    feature_name: str = Field(..., description="Feature name")
    importance_value: float = Field(..., description="Importance value")
    importance_type: str = Field(..., description="Type of importance measure")
    confidence: float = Field(..., description="Confidence in importance")
    rank: int = Field(..., description="Feature rank")
    contribution_direction: str = Field(..., description="Positive/negative/neutral")
    additional_metrics: dict[str, float] = Field(
        default_factory=dict, description="Additional metrics"
    )


class InstanceExplanationResponse(BaseModel):
    """Response model for instance explanation."""

    instance_id: str = Field(..., description="Instance identifier")
    prediction_value: Union[float, int, str] = Field(
        ..., description="Model prediction"
    )
    prediction_confidence: float = Field(..., description="Prediction confidence")
    base_value: float = Field(..., description="Base value for additive explanations")
    feature_importances: list[FeatureImportanceResponse] = Field(
        ..., description="Feature importance scores"
    )
    local_fidelity_score: float = Field(..., description="Local explanation fidelity")
    explanation_completeness: float = Field(
        ..., description="Explanation completeness score"
    )


class GlobalExplanationResponse(BaseModel):
    """Response model for global explanation."""

    model_id: str = Field(..., description="Model identifier")
    explanation_method: str = Field(..., description="Explanation method used")
    global_feature_importances: list[FeatureImportanceResponse] = Field(
        ..., description="Global feature importance scores"
    )
    data_coverage: float = Field(..., description="Fraction of data used")
    feature_stability_score: float = Field(..., description="Feature stability score")
    has_bias_issues: bool = Field(..., description="Whether bias was detected")
    fairness_metrics: dict[str, float] = Field(
        default_factory=dict, description="Fairness metrics"
    )


class BiasAnalysisResponse(BaseModel):
    """Response model for bias analysis."""

    analysis_id: str = Field(..., description="Analysis identifier")
    overall_bias_score: float = Field(..., description="Overall bias score")
    bias_severity: str = Field(..., description="Bias severity level")
    bias_detected: bool = Field(..., description="Whether bias was detected")
    protected_attribute_bias: dict[str, float] = Field(
        ..., description="Bias scores for protected attributes"
    )
    fairness_metrics: dict[str, float] = Field(..., description="Fairness metrics")
    requires_attention: bool = Field(
        ..., description="Whether bias requires immediate attention"
    )
    bias_sources: list[str] = Field(..., description="Sources of bias")
    mitigation_recommendations: list[str] = Field(
        default_factory=list, description="Bias mitigation recommendations"
    )


class TrustScoreResponse(BaseModel):
    """Response model for trust score."""

    overall_trust_score: float = Field(..., description="Overall trust score")
    trust_level: str = Field(..., description="Trust level")
    consistency_score: float = Field(..., description="Consistency score")
    stability_score: float = Field(..., description="Stability score")
    fidelity_score: float = Field(..., description="Fidelity score")
    completeness_score: float = Field(..., description="Completeness score")
    confidence_interval: list[float] = Field(
        ..., description="Trust score confidence interval"
    )
    is_trustworthy: bool = Field(..., description="Whether explanation is trustworthy")


class CounterfactualResponse(BaseModel):
    """Response model for counterfactual explanations."""

    counterfactual_id: str = Field(..., description="Counterfactual identifier")
    original_prediction: Union[float, int, str] = Field(
        ..., description="Original prediction"
    )
    counterfactual_prediction: Union[float, int, str] = Field(
        ..., description="Counterfactual prediction"
    )
    feature_changes: dict[str, dict[str, float]] = Field(
        ..., description="Changes made to features"
    )
    distance: float = Field(..., description="Distance from original instance")
    feasibility_score: float = Field(
        default=1.0, description="Feasibility of counterfactual"
    )


class ExplanationMetadataResponse(BaseModel):
    """Response model for explanation metadata."""

    generation_time_seconds: float = Field(..., description="Time to generate")
    explanation_confidence: float = Field(..., description="Explanation confidence")
    model_version: str = Field(..., description="Model version")
    explanation_version: str = Field(..., description="Explanation framework version")
    feature_coverage: float = Field(..., description="Feature coverage")
    method_parameters: dict[str, Any] = Field(..., description="Method parameters used")


class ExplainPredictionResponse(BaseModel):
    """Response model for prediction explanation."""

    request_id: str = Field(..., description="Request identifier")
    explanation_id: str = Field(..., description="Explanation identifier")
    success: bool = Field(..., description="Whether explanation succeeded")
    explanation_method: str = Field(..., description="Explanation method used")
    instance_explanation: InstanceExplanationResponse = Field(
        ..., description="Instance explanation details"
    )
    trust_score: Optional[TrustScoreResponse] = Field(
        None, description="Trust assessment"
    )
    counterfactuals: list[CounterfactualResponse] = Field(
        default_factory=list, description="Counterfactual explanations"
    )
    metadata: ExplanationMetadataResponse = Field(
        ..., description="Explanation metadata"
    )
    warnings: list[str] = Field(default_factory=list, description="Warning messages")


class ExplanationSummaryResponse(BaseModel):
    """Response model for explanation summary."""

    model_id: str = Field(..., description="Model identifier")
    time_window_hours: float = Field(..., description="Time window for summary")
    total_explanations: int = Field(..., description="Total explanations generated")
    methods_used: list[str] = Field(..., description="Explanation methods used")
    cache_hit_rate: float = Field(..., description="Cache hit rate")
    average_explanation_time: float = Field(
        ..., description="Average explanation generation time"
    )
    top_features: list[str] = Field(..., description="Most important features")
    average_trust_score: float = Field(..., description="Average trust score")
    bias_detected: bool = Field(..., description="Whether bias was detected")
    quality_metrics: dict[str, float] = Field(
        ..., description="Explanation quality metrics"
    )


# ==================== Explanation Generation Endpoints ====================


@router.post("/predictions/explain", response_model=ExplainPredictionResponse)
async def explain_prediction(
    request: ExplainPredictionRequest,
    current_user: dict = Depends(get_current_user),
    xai_service: ExplainableAIService = Depends(get_explainable_ai_service),
    _: None = Depends(require_read),
) -> ExplainPredictionResponse:
    """Generate explanation for a single prediction."""
    try:
        # Load model (mock implementation)
        model = await _load_model(request.model_id)

        # Convert input data
        instance = np.array(request.instance_data)

        # Set up configuration
        config = ExplanationConfiguration(
            explanation_method=(
                ExplanationMethod(request.explanation_method)
                if request.explanation_method
                else ExplanationMethod.PERMUTATION_IMPORTANCE
            ),
            num_features=request.num_features,
            enable_counterfactual_analysis=request.enable_counterfactuals,
        )

        # Generate explanation
        result = await xai_service.explain_prediction(
            model=model,
            instance=instance,
            feature_names=request.feature_names,
            config=config,
        )

        # Convert to response format
        instance_explanation = InstanceExplanationResponse(
            instance_id=result.instance_explanation.instance_id,
            prediction_value=result.instance_explanation.prediction_value,
            prediction_confidence=result.instance_explanation.prediction_confidence,
            base_value=result.instance_explanation.base_value,
            feature_importances=[
                FeatureImportanceResponse(
                    feature_name=fi.feature_name,
                    importance_value=fi.importance_value,
                    importance_type=fi.importance_type,
                    confidence=fi.confidence,
                    rank=fi.rank,
                    contribution_direction=fi.contribution_direction,
                    additional_metrics=fi.additional_metrics,
                )
                for fi in result.instance_explanation.feature_importances
            ],
            local_fidelity_score=result.instance_explanation.local_fidelity_score,
            explanation_completeness=result.instance_explanation.calculate_explanation_completeness(),
        )

        # Generate counterfactuals if requested
        counterfactuals = []
        if request.enable_counterfactuals:
            cf_results = await xai_service.generate_counterfactual_explanations(
                model=model,
                instance=instance,
                feature_names=request.feature_names,
                num_counterfactuals=5,
            )

            counterfactuals = [
                CounterfactualResponse(
                    counterfactual_id=cf["counterfactual_id"],
                    original_prediction=cf["original_prediction"],
                    counterfactual_prediction=cf["counterfactual_prediction"],
                    feature_changes=cf["feature_changes"],
                    distance=cf["distance"],
                )
                for cf in cf_results
            ]

        # Generate trust score
        trust_score = None
        if result.trust_score:
            trust_score = TrustScoreResponse(
                overall_trust_score=result.trust_score.overall_trust_score,
                trust_level=result.trust_score.trust_level.value,
                consistency_score=result.trust_score.consistency_score,
                stability_score=result.trust_score.stability_score,
                fidelity_score=result.trust_score.fidelity_score,
                completeness_score=result.trust_score.completeness_score,
                confidence_interval=list(result.trust_score.confidence_interval),
                is_trustworthy=result.trust_score.is_trustworthy(),
            )

        return ExplainPredictionResponse(
            request_id=str(uuid4()),
            explanation_id=str(result.result_id),
            success=result.success,
            explanation_method=result.explanation_method.value,
            instance_explanation=instance_explanation,
            trust_score=trust_score,
            counterfactuals=counterfactuals,
            metadata=ExplanationMetadataResponse(
                generation_time_seconds=result.metadata.generation_time_seconds,
                explanation_confidence=result.metadata.explanation_confidence,
                model_version=result.metadata.model_version,
                explanation_version=result.metadata.explanation_version,
                feature_coverage=result.metadata.feature_coverage,
                method_parameters=result.metadata.method_parameters,
            ),
            warnings=result.warnings,
        )

    except ExplanationNotSupportedError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Explanation method not supported: {str(e)}",
        )
    except InsufficientDataError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Insufficient data for explanation: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate explanation: {str(e)}",
        )


@router.post("/models/explain-global", response_model=GlobalExplanationResponse)
async def explain_model_global(
    request: ExplainModelGlobalRequest,
    current_user: dict = Depends(get_current_user),
    xai_service: ExplainableAIService = Depends(get_explainable_ai_service),
    _: None = Depends(require_read),
) -> GlobalExplanationResponse:
    """Generate global explanation for entire model."""
    try:
        # Load model and training data
        model = await _load_model(request.model_id)
        training_data = await _load_training_data(request.model_id, request.sample_size)

        # Set up configuration
        config = ExplanationConfiguration(
            explanation_method=(
                ExplanationMethod(request.explanation_method)
                if request.explanation_method
                else ExplanationMethod.PERMUTATION_IMPORTANCE
            ),
            num_features=request.num_features,
            enable_interaction_analysis=request.enable_interaction_analysis,
            enable_bias_detection=request.enable_bias_detection,
        )

        # Generate global explanation
        global_explanation = await xai_service.explain_model_global(
            model=model,
            training_data=training_data,
            feature_names=request.feature_names,
            config=config,
        )

        return GlobalExplanationResponse(
            model_id=str(global_explanation.model_id),
            explanation_method=global_explanation.explanation_method.value,
            global_feature_importances=[
                FeatureImportanceResponse(
                    feature_name=fi.feature_name,
                    importance_value=fi.importance_value,
                    importance_type=fi.importance_type,
                    confidence=fi.confidence,
                    rank=fi.rank,
                    contribution_direction=fi.contribution_direction,
                    additional_metrics=fi.additional_metrics,
                )
                for fi in global_explanation.global_feature_importances
            ],
            data_coverage=global_explanation.data_coverage,
            feature_stability_score=global_explanation.get_feature_stability_score(),
            has_bias_issues=global_explanation.has_bias_issues(),
            fairness_metrics=global_explanation.fairness_metrics,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate global explanation: {str(e)}",
        )


# ==================== Feature Analysis Endpoints ====================


@router.post("/features/importance", response_model=list[FeatureImportanceResponse])
async def analyze_feature_importance(
    request: FeatureImportanceRequest,
    current_user: dict = Depends(get_current_user),
    xai_service: ExplainableAIService = Depends(get_explainable_ai_service),
    _: None = Depends(require_read),
) -> list[FeatureImportanceResponse]:
    """Analyze feature importance using specified method."""
    try:
        # Load model and data
        model = await _load_model(request.model_id)
        data = await _load_training_data(request.model_id, request.sample_size)

        # Analyze feature importance
        importances = await xai_service.analyze_feature_importance(
            model=model,
            data=data,
            feature_names=request.feature_names,
            method=ExplanationMethod(request.explanation_method),
        )

        return [
            FeatureImportanceResponse(
                feature_name=fi.feature_name,
                importance_value=fi.importance_value,
                importance_type=fi.importance_type,
                confidence=fi.confidence,
                rank=fi.rank,
                contribution_direction=fi.contribution_direction,
                additional_metrics=fi.additional_metrics,
            )
            for fi in importances
        ]

    except ExplanationNotSupportedError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Analysis method not supported: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze feature importance: {str(e)}",
        )


@router.post("/bias/detect", response_model=BiasAnalysisResponse)
async def detect_bias(
    request: BiasDetectionRequest,
    current_user: dict = Depends(get_current_user),
    xai_service: ExplainableAIService = Depends(get_explainable_ai_service),
    _: None = Depends(require_read),
) -> BiasAnalysisResponse:
    """Detect bias in model explanations."""
    try:
        # Load model and data
        model = await _load_model(request.model_id)
        data = await _load_training_data(request.model_id, request.sample_size)

        # Detect bias
        bias_analysis = await xai_service.detect_explanation_bias(
            model=model,
            data=data,
            protected_attributes=request.protected_attributes,
            feature_names=request.feature_names,
        )

        return BiasAnalysisResponse(
            analysis_id=str(bias_analysis.analysis_id),
            overall_bias_score=bias_analysis.overall_bias_score,
            bias_severity=bias_analysis.get_bias_severity(),
            bias_detected=bias_analysis.bias_detected,
            protected_attribute_bias=bias_analysis.protected_attribute_bias,
            fairness_metrics=bias_analysis.fairness_metrics,
            requires_attention=bias_analysis.requires_immediate_attention(),
            bias_sources=bias_analysis.bias_sources,
            mitigation_recommendations=bias_analysis.mitigation_recommendations,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect bias: {str(e)}",
        )


# ==================== Counterfactual and Trust Endpoints ====================


@router.post("/counterfactuals/generate", response_model=list[CounterfactualResponse])
async def generate_counterfactuals(
    request: CounterfactualRequest,
    current_user: dict = Depends(get_current_user),
    xai_service: ExplainableAIService = Depends(get_explainable_ai_service),
    _: None = Depends(require_read),
) -> list[CounterfactualResponse]:
    """Generate counterfactual explanations."""
    try:
        # Load model
        model = await _load_model(request.model_id)

        # Convert input data
        instance = np.array(request.instance_data)

        # Generate counterfactuals
        counterfactuals = await xai_service.generate_counterfactual_explanations(
            model=model,
            instance=instance,
            feature_names=request.feature_names,
            num_counterfactuals=request.num_counterfactuals,
        )

        return [
            CounterfactualResponse(
                counterfactual_id=cf["counterfactual_id"],
                original_prediction=cf["original_prediction"],
                counterfactual_prediction=cf["counterfactual_prediction"],
                feature_changes=cf["feature_changes"],
                distance=cf["distance"],
            )
            for cf in counterfactuals
        ]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate counterfactuals: {str(e)}",
        )


@router.post("/trust/assess", response_model=TrustScoreResponse)
async def assess_trust(
    request: TrustAssessmentRequest,
    current_user: dict = Depends(get_current_user),
    xai_service: ExplainableAIService = Depends(get_explainable_ai_service),
    _: None = Depends(require_read),
) -> TrustScoreResponse:
    """Assess trust in explanation."""
    try:
        # This is a simplified implementation
        # In practice, you'd retrieve the explanation result by ID

        # Load model and validation data
        model = await _load_model(request.model_id)
        validation_data = await _load_training_data(
            request.model_id, request.validation_sample_size
        )

        # Mock explanation result for trust assessment
        # In practice, retrieve actual explanation result
        from pynomaly.domain.entities.explainable_ai import ExplanationResult

        mock_result = ExplanationResult(
            explanation_method=ExplanationMethod.PERMUTATION_IMPORTANCE,
            success=True,
        )

        # Assess trust
        trust_score = await xai_service.assess_explanation_trust(
            explanation_result=mock_result,
            model=model,
            validation_data=validation_data,
        )

        return TrustScoreResponse(
            overall_trust_score=trust_score.overall_trust_score,
            trust_level=trust_score.trust_level.value,
            consistency_score=trust_score.consistency_score,
            stability_score=trust_score.stability_score,
            fidelity_score=trust_score.fidelity_score,
            completeness_score=trust_score.completeness_score,
            confidence_interval=list(trust_score.confidence_interval),
            is_trustworthy=trust_score.is_trustworthy(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to assess trust: {str(e)}",
        )


# ==================== Management and Information Endpoints ====================


@router.get("/models/{model_id}/summary", response_model=ExplanationSummaryResponse)
async def get_explanation_summary(
    model_id: str,
    time_window_hours: float = 24.0,
    xai_service: ExplainableAIService = Depends(get_explainable_ai_service),
) -> ExplanationSummaryResponse:
    """Get explanation summary for a model."""
    try:
        from uuid import UUID

        # Convert string to UUID
        model_uuid = UUID(model_id)
        time_window = timedelta(hours=time_window_hours)

        # Get summary
        summary = await xai_service.get_explanation_summary(
            model_id=model_uuid,
            time_window=time_window,
        )

        return ExplanationSummaryResponse(
            model_id=model_id,
            time_window_hours=time_window_hours,
            total_explanations=summary["explanation_stats"]["total_explanations"],
            methods_used=summary["explanation_stats"]["methods_used"],
            cache_hit_rate=summary["explanation_stats"]["cache_hit_rate"],
            average_explanation_time=summary["explanation_stats"][
                "average_explanation_time"
            ],
            top_features=summary["top_features"],
            average_trust_score=summary["explanation_quality"]["average_trust_score"],
            bias_detected=summary["bias_indicators"]["bias_detected"],
            quality_metrics=summary["explanation_quality"],
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model ID format: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get explanation summary: {str(e)}",
        )


@router.get("/methods", response_model=dict[str, Any])
async def get_available_methods() -> dict[str, Any]:
    """Get information about available explanation methods."""
    return {
        "explanation_methods": [
            {
                "name": method.value,
                "description": _get_method_description(method),
                "scope": _get_method_scope(method),
                "requirements": _get_method_requirements(method),
            }
            for method in ExplanationMethod
        ],
        "bias_types": [
            {
                "name": bias_type.value,
                "description": _get_bias_type_description(bias_type),
            }
            for bias_type in BiasType
        ],
        "audiences": [
            {
                "name": audience.value,
                "description": _get_audience_description(audience),
            }
            for audience in ExplanationAudience
        ],
        "trust_levels": [level.value for level in TrustLevel],
    }


@router.get("/config/defaults", response_model=dict[str, Any])
async def get_default_configuration() -> dict[str, Any]:
    """Get default explanation configuration."""
    config = ExplanationConfiguration()
    return {
        "explanation_method": config.explanation_method.value,
        "explanation_scope": config.explanation_scope.value,
        "num_features": config.num_features,
        "num_samples": config.num_samples,
        "background_sample_size": config.background_sample_size,
        "enable_interaction_analysis": config.enable_interaction_analysis,
        "enable_bias_detection": config.enable_bias_detection,
        "confidence_threshold": config.confidence_threshold,
        "explanation_timeout_seconds": config.explanation_timeout_seconds,
        "cache_explanations": config.cache_explanations,
    }


# ==================== Helper Functions ====================


async def _load_model(model_id: str):
    """Load model by ID (mock implementation)."""
    # In practice, this would load from model registry
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(contamination=0.1, random_state=42)

    # Mock training data for model fitting
    from sklearn.datasets import make_classification

    X, _ = make_classification(n_samples=1000, n_features=10, random_state=42)
    model.fit(X)

    return model


async def _load_training_data(model_id: str, sample_size: int) -> np.ndarray:
    """Load training data for model (mock implementation)."""
    # In practice, this would load from data storage
    from sklearn.datasets import make_classification

    X, _ = make_classification(n_samples=sample_size, n_features=10, random_state=42)
    return X


def _get_method_description(method: ExplanationMethod) -> str:
    """Get description for explanation method."""
    descriptions = {
        ExplanationMethod.SHAP_TREE: "SHAP values for tree-based models",
        ExplanationMethod.SHAP_KERNEL: "Model-agnostic SHAP explanations",
        ExplanationMethod.LIME: "Local interpretable model-agnostic explanations",
        ExplanationMethod.PERMUTATION_IMPORTANCE: "Feature importance via permutation",
        ExplanationMethod.FEATURE_ABLATION: "Importance via feature removal",
    }
    return descriptions.get(method, "Advanced explanation method")


def _get_method_scope(method: ExplanationMethod) -> list[str]:
    """Get supported scopes for explanation method."""
    local_methods = [
        ExplanationMethod.LIME,
        ExplanationMethod.SHAP_KERNEL,
        ExplanationMethod.COUNTERFACTUAL,
    ]
    global_methods = [
        ExplanationMethod.PERMUTATION_IMPORTANCE,
        ExplanationMethod.FEATURE_ABLATION,
        ExplanationMethod.SHAP_TREE,
    ]

    scopes = []
    if method in local_methods:
        scopes.append("local")
    if method in global_methods:
        scopes.append("global")
    if not scopes:
        scopes = ["local", "global"]

    return scopes


def _get_method_requirements(method: ExplanationMethod) -> list[str]:
    """Get requirements for explanation method."""
    requirements = {
        ExplanationMethod.SHAP_TREE: ["tree-based model", "shap library"],
        ExplanationMethod.LIME: ["lime library", "background data"],
        ExplanationMethod.PERMUTATION_IMPORTANCE: ["scikit-learn"],
        ExplanationMethod.FEATURE_ABLATION: ["model prediction interface"],
    }
    return requirements.get(method, ["model prediction interface"])


def _get_bias_type_description(bias_type: BiasType) -> str:
    """Get description for bias type."""
    descriptions = {
        BiasType.DEMOGRAPHIC_PARITY: "Equal positive prediction rates across groups",
        BiasType.EQUALIZED_ODDS: "Equal true positive and false positive rates",
        BiasType.EQUALITY_OF_OPPORTUNITY: "Equal true positive rates across groups",
        BiasType.CALIBRATION: "Equal prediction calibration across groups",
    }
    return descriptions.get(bias_type, "Fairness metric for bias detection")


def _get_audience_description(audience: ExplanationAudience) -> str:
    """Get description for target audience."""
    descriptions = {
        ExplanationAudience.TECHNICAL: "Data scientists and ML engineers",
        ExplanationAudience.BUSINESS: "Business stakeholders and managers",
        ExplanationAudience.REGULATORY: "Compliance officers and auditors",
        ExplanationAudience.END_USER: "End users of the application",
        ExplanationAudience.DOMAIN_EXPERT: "Subject matter experts",
    }
    return descriptions.get(audience, "Explanation audience")
