"""Explainability endpoints."""

import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ...domain.services.explainability_service import ExplainabilityService, ExplainerType
from ...domain.services.detection_service import DetectionService
from ...infrastructure.repositories.model_repository import ModelRepository
from ...infrastructure.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


class ExplanationRequest(BaseModel):
    """Request model for single sample explanation."""
    data: List[float] = Field(..., description="Feature vector to explain")
    algorithm: str = Field("isolation_forest", description="Detection algorithm")
    explainer_type: str = Field("feature_importance", description="Type of explainer to use")
    feature_names: Optional[List[str]] = Field(None, description="Names of features")
    training_data: Optional[List[List[float]]] = Field(None, description="Training data (required for LIME)")
    model_id: Optional[str] = Field(None, description="Saved model ID to use")
    contamination: float = Field(0.1, description="Contamination rate if fitting new model")


class BatchExplanationRequest(BaseModel):
    """Request model for batch explanations."""
    data: List[List[float]] = Field(..., description="List of feature vectors to explain")
    algorithm: str = Field("isolation_forest", description="Detection algorithm")
    explainer_type: str = Field("feature_importance", description="Type of explainer to use")
    feature_names: Optional[List[str]] = Field(None, description="Names of features")
    max_samples: int = Field(10, description="Maximum samples to explain")
    anomalies_only: bool = Field(False, description="Only explain detected anomalies")
    contamination: float = Field(0.1, description="Contamination rate")


class TopFeature(BaseModel):
    """Top contributing feature."""
    feature_name: str = Field(..., description="Name of the feature")
    importance: float = Field(..., description="Importance score")
    value: float = Field(..., description="Feature value")
    rank: int = Field(..., description="Ranking by importance")


class ExplanationResponse(BaseModel):
    """Response model for explanations."""
    success: bool = Field(..., description="Whether explanation was successful")
    explainer_type: str = Field(..., description="Type of explainer used")
    feature_names: List[str] = Field(..., description="Names of features")
    feature_importance: Dict[str, float] = Field(..., description="Feature importance scores")
    is_anomaly: bool = Field(..., description="Whether sample is anomalous")
    prediction_confidence: Optional[float] = Field(None, description="Prediction confidence")
    top_features: List[TopFeature] = Field(..., description="Top contributing features")
    sample_data: List[float] = Field(..., description="Original sample data")
    base_value: Optional[float] = Field(None, description="Base prediction value")
    timestamp: str = Field(..., description="Explanation timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class BatchExplanationResponse(BaseModel):
    """Response model for batch explanations."""
    success: bool = Field(..., description="Whether explanations were successful")
    total_explained: int = Field(..., description="Number of samples explained")
    explainer_type: str = Field(..., description="Type of explainer used")
    explanations: List[ExplanationResponse] = Field(..., description="Individual explanations")
    global_feature_importance: Dict[str, float] = Field(..., description="Average feature importance")
    timestamp: str = Field(..., description="Explanation timestamp")


class GlobalImportanceRequest(BaseModel):
    """Request for global feature importance."""
    training_data: List[List[float]] = Field(..., description="Training dataset")
    algorithm: str = Field("isolation_forest", description="Detection algorithm")
    feature_names: Optional[List[str]] = Field(None, description="Names of features")
    n_samples: int = Field(100, description="Number of samples to analyze")
    contamination: float = Field(0.1, description="Contamination rate")


class GlobalImportanceResponse(BaseModel):
    """Response for global feature importance."""
    success: bool = Field(..., description="Whether analysis was successful")
    algorithm: str = Field(..., description="Algorithm analyzed")
    samples_analyzed: int = Field(..., description="Number of samples analyzed")
    total_features: int = Field(..., description="Total number of features")
    feature_importance: Dict[str, float] = Field(..., description="Global feature importance")
    importance_ranking: List[Dict[str, Any]] = Field(..., description="Features ranked by importance")
    statistics: Dict[str, float] = Field(..., description="Importance statistics")
    timestamp: str = Field(..., description="Analysis timestamp")


# Dependency injection
_explainability_service: Optional[ExplainabilityService] = None
_detection_service: Optional[DetectionService] = None
_model_repository: Optional[ModelRepository] = None


def get_explainability_service() -> ExplainabilityService:
    """Get explainability service instance."""
    global _explainability_service, _detection_service
    if _explainability_service is None:
        if _detection_service is None:
            _detection_service = DetectionService()
        _explainability_service = ExplainabilityService(_detection_service)
    return _explainability_service


def get_detection_service() -> DetectionService:
    """Get detection service instance."""
    global _detection_service
    if _detection_service is None:
        _detection_service = DetectionService()
    return _detection_service


def get_model_repository() -> ModelRepository:
    """Get model repository instance."""
    global _model_repository
    if _model_repository is None:
        _model_repository = ModelRepository()
    return _model_repository


@router.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(
    request: ExplanationRequest,
    explainability_service: ExplainabilityService = Depends(get_explainability_service),
    detection_service: DetectionService = Depends(get_detection_service),
    model_repository: ModelRepository = Depends(get_model_repository)
) -> ExplanationResponse:
    """Explain a single prediction."""
    try:
        logger.info("Processing explanation request", 
                   algorithm=request.algorithm,
                   explainer_type=request.explainer_type)
        
        # Convert data to numpy array
        sample_array = np.array(request.data, dtype=np.float64)
        
        # Algorithm mapping
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'lof': 'lof'
        }
        mapped_algorithm = algorithm_map.get(request.algorithm, request.algorithm)
        
        # Load model if model_id provided
        if request.model_id:
            try:
                model = model_repository.load(request.model_id)
                detection_service._fitted_models[mapped_algorithm] = model.model_object
                logger.info("Loaded saved model", model_id=request.model_id)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{request.model_id}' not found: {str(e)}"
                )
        else:
            # Fit model if not already fitted
            if mapped_algorithm not in detection_service._fitted_models:
                if request.training_data:
                    training_array = np.array(request.training_data, dtype=np.float64)
                    detection_service.fit(training_array, mapped_algorithm, request.contamination)
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Either model_id or training_data must be provided"
                    )
        
        # Map explainer type
        explainer_type_map = {
            'shap': ExplainerType.SHAP,
            'lime': ExplainerType.LIME,
            'permutation': ExplainerType.PERMUTATION,
            'feature_importance': ExplainerType.FEATURE_IMPORTANCE
        }
        explainer_enum = explainer_type_map.get(request.explainer_type, ExplainerType.FEATURE_IMPORTANCE)
        
        # Prepare training data for LIME if needed
        training_data = None
        if request.explainer_type == 'lime' and request.training_data:
            training_data = np.array(request.training_data, dtype=np.float64)
        
        # Generate explanation
        explanation = explainability_service.explain_prediction(
            sample=sample_array,
            algorithm=mapped_algorithm,
            explainer_type=explainer_enum,
            training_data=training_data,
            feature_names=request.feature_names
        )
        
        # Convert top features to response model
        top_features = []
        if explanation.top_features:
            for feature in explanation.top_features:
                top_features.append(TopFeature(
                    feature_name=feature["feature_name"],
                    importance=feature["importance"],
                    value=feature["value"],
                    rank=feature["rank"]
                ))
        
        return ExplanationResponse(
            success=True,
            explainer_type=explanation.explainer_type,
            feature_names=explanation.feature_names,
            feature_importance=explanation.feature_importance,
            is_anomaly=explanation.is_anomaly,
            prediction_confidence=explanation.prediction_confidence,
            top_features=top_features,
            sample_data=explanation.data_sample,
            base_value=explanation.base_value,
            timestamp=datetime.utcnow().isoformat(),
            metadata=explanation.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Explanation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {str(e)}"
        )


@router.post("/explain/batch", response_model=BatchExplanationResponse)
async def explain_batch(
    request: BatchExplanationRequest,
    explainability_service: ExplainabilityService = Depends(get_explainability_service),
    detection_service: DetectionService = Depends(get_detection_service)
) -> BatchExplanationResponse:
    """Explain multiple predictions."""
    try:
        logger.info("Processing batch explanation request", 
                   samples=len(request.data),
                   explainer_type=request.explainer_type)
        
        # Convert data to numpy array
        data_array = np.array(request.data, dtype=np.float64)
        
        # Algorithm mapping
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'lof': 'lof'
        }
        mapped_algorithm = algorithm_map.get(request.algorithm, request.algorithm)
        
        # Fit model
        detection_service.fit(data_array, mapped_algorithm, request.contamination)
        
        # Filter samples if anomalies_only is True
        samples_to_explain = data_array[:request.max_samples]
        
        if request.anomalies_only:
            # Get predictions to filter anomalies
            result = detection_service.detect_anomalies(data_array, mapped_algorithm, request.contamination)
            anomaly_indices = np.where(result.predictions == -1)[0]
            samples_to_explain = data_array[anomaly_indices[:request.max_samples]]
        
        # Map explainer type
        explainer_type_map = {
            'shap': ExplainerType.SHAP,
            'lime': ExplainerType.LIME,
            'permutation': ExplainerType.PERMUTATION,
            'feature_importance': ExplainerType.FEATURE_IMPORTANCE
        }
        explainer_enum = explainer_type_map.get(request.explainer_type, ExplainerType.FEATURE_IMPORTANCE)
        
        # Generate explanations
        explanations = explainability_service.explain_batch(
            samples=samples_to_explain,
            algorithm=mapped_algorithm,
            explainer_type=explainer_enum,
            feature_names=request.feature_names
        )
        
        # Convert to response format
        response_explanations = []
        global_importance = {}
        
        for explanation in explanations:
            # Convert top features
            top_features = []
            if explanation.top_features:
                for feature in explanation.top_features:
                    top_features.append(TopFeature(
                        feature_name=feature["feature_name"],
                        importance=feature["importance"],
                        value=feature["value"],
                        rank=feature["rank"]
                    ))
            
            response_explanations.append(ExplanationResponse(
                success=True,
                explainer_type=explanation.explainer_type,
                feature_names=explanation.feature_names,
                feature_importance=explanation.feature_importance,
                is_anomaly=explanation.is_anomaly,
                prediction_confidence=explanation.prediction_confidence,
                top_features=top_features,
                sample_data=explanation.data_sample,
                base_value=explanation.base_value,
                timestamp=datetime.utcnow().isoformat(),
                metadata=explanation.metadata
            ))
            
            # Accumulate global importance
            for feature_name, importance in explanation.feature_importance.items():
                if feature_name not in global_importance:
                    global_importance[feature_name] = []
                global_importance[feature_name].append(importance)
        
        # Calculate average global importance
        avg_global_importance = {}
        for feature_name, importance_list in global_importance.items():
            avg_global_importance[feature_name] = float(np.mean(importance_list))
        
        return BatchExplanationResponse(
            success=True,
            total_explained=len(response_explanations),
            explainer_type=request.explainer_type,
            explanations=response_explanations,
            global_feature_importance=avg_global_importance,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error("Batch explanation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch explanation failed: {str(e)}"
        )


@router.post("/global-importance", response_model=GlobalImportanceResponse)
async def analyze_global_importance(
    request: GlobalImportanceRequest,
    explainability_service: ExplainabilityService = Depends(get_explainability_service),
    detection_service: DetectionService = Depends(get_detection_service)
) -> GlobalImportanceResponse:
    """Analyze global feature importance."""
    try:
        logger.info("Processing global importance analysis", 
                   samples=len(request.training_data),
                   algorithm=request.algorithm)
        
        # Convert data to numpy array
        training_array = np.array(request.training_data, dtype=np.float64)
        
        # Algorithm mapping
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'lof': 'lof'
        }
        mapped_algorithm = algorithm_map.get(request.algorithm, request.algorithm)
        
        # Fit model
        detection_service.fit(training_array, mapped_algorithm, request.contamination)
        
        # Get global feature importance
        global_importance = explainability_service.get_global_feature_importance(
            algorithm=mapped_algorithm,
            training_data=training_array,
            feature_names=request.feature_names,
            n_samples=request.n_samples
        )
        
        # Create importance ranking
        sorted_features = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)
        importance_ranking = [
            {"rank": i+1, "feature": name, "importance": importance}
            for i, (name, importance) in enumerate(sorted_features)
        ]
        
        # Calculate statistics
        importance_values = list(global_importance.values())
        statistics = {}
        if importance_values:
            statistics = {
                "max_importance": float(max(importance_values)),
                "min_importance": float(min(importance_values)),
                "mean_importance": float(np.mean(importance_values)),
                "std_importance": float(np.std(importance_values))
            }
        
        return GlobalImportanceResponse(
            success=True,
            algorithm=request.algorithm,
            samples_analyzed=min(request.n_samples, len(training_array)),
            total_features=len(global_importance),
            feature_importance=global_importance,
            importance_ranking=importance_ranking,
            statistics=statistics,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error("Global importance analysis failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Global importance analysis failed: {str(e)}"
        )


@router.get("/explainers")
async def get_available_explainers(
    explainability_service: ExplainabilityService = Depends(get_explainability_service)
) -> Dict[str, List[str]]:
    """Get available explainer types."""
    try:
        available_explainers = explainability_service.get_available_explainers()
        
        return {
            "available_explainers": available_explainers,
            "descriptions": {
                "shap": "SHAP (SHapley Additive exPlanations) - Advanced model-agnostic explanations",
                "lime": "LIME (Local Interpretable Model-agnostic Explanations) - Local linear approximations",
                "permutation": "Permutation Importance - Feature importance via permutation testing",
                "feature_importance": "Simple Feature Importance - Based on feature magnitude"
            }
        }
        
    except Exception as e:
        logger.error("Error getting available explainers", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get available explainers: {str(e)}"
        )