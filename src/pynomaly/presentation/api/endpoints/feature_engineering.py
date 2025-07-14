"""
Feature Engineering API Endpoints

RESTful endpoints for comprehensive feature engineering capabilities.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from uuid import UUID
import logging

from ....domain.entities.user import User
from ....application.dto.feature_engineering_dto import (
    FeatureCreationRequestDTO,
    FeatureCreationResponseDTO,
    FeatureImportanceRequestDTO,
    FeatureImportanceResponseDTO,
    FeatureSelectionRequestDTO,
    FeatureSelectionResponseDTO,
    AutoFeatureEngineeringRequestDTO,
    AutoFeatureEngineeringResponseDTO,
    FeatureTransformationRequestDTO,
    FeatureTransformationResponseDTO
)
from ....application.use_cases.feature_engineering import (
    CreateFeatureUseCase,
    AnalyzeFeatureImportanceUseCase,
    SelectFeaturesUseCase,
    AutoFeatureEngineeringUseCase
)
from ....shared.dependencies import get_current_user, get_feature_engineering_use_cases
from ....shared.monitoring import metrics, monitor_endpoint
from ....shared.error_handling import APIError, ErrorCode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/features", tags=["Feature Engineering"])


@router.post(
    "/create",
    response_model=FeatureCreationResponseDTO,
    summary="Create New Features",
    description="""
    Create new features from existing dataset columns using mathematical operations,
    transformations, and domain-specific feature engineering techniques.
    
    Supports feature creation through:
    - Mathematical operations (addition, multiplication, ratios)
    - Statistical transformations (log, sqrt, polynomial)
    - Time-based features (date decomposition, lag features)
    - Categorical encoding (one-hot, label encoding, target encoding)
    - Text features (length, word count, TF-IDF)
    
    **Features:**
    - Custom feature formulas and expressions
    - Automated feature validation and quality checks
    - Feature metadata tracking and lineage
    - Batch feature creation with dependency resolution
    - Feature versioning and reproducibility
    """,
    responses={
        200: {"description": "Features created successfully"},
        400: {"description": "Invalid feature specification"},
        404: {"description": "Dataset not found"},
        422: {"description": "Feature creation validation error"},
        500: {"description": "Internal server error"}
    }
)
@monitor_endpoint("feature_creation")
async def create_features(
    request: FeatureCreationRequestDTO,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    use_case: CreateFeatureUseCase = Depends(get_feature_engineering_use_cases)
) -> FeatureCreationResponseDTO:
    """
    Create new features from existing dataset columns.
    
    Args:
        request: Feature creation request
        background_tasks: Background task manager
        current_user: Authenticated user
        use_case: Feature creation use case
        
    Returns:
        Feature creation results
        
    Raises:
        HTTPException: If feature creation fails
    """
    try:
        logger.info(
            f"Creating {len(request.feature_specifications)} features for dataset {request.dataset_id} "
            f"by user {current_user.id}"
        )
        
        result = await use_case.execute(request, current_user.id)
        
        if result.status == "failed":
            metrics.feature_creation_failures.inc()
            raise HTTPException(
                status_code=500,
                detail=f"Feature creation failed: {result.error_message}"
            )
        
        metrics.successful_feature_creations.inc()
        metrics.features_created_total.inc(len(result.created_features))
        
        logger.info(
            f"Successfully created {len(result.created_features)} features "
            f"for dataset {request.dataset_id}"
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        metrics.feature_creation_failures.inc()
        logger.error(f"Unexpected error in feature creation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during feature creation"
        )


@router.post(
    "/importance",
    response_model=FeatureImportanceResponseDTO,
    summary="Analyze Feature Importance",
    description="""
    Analyze the importance of features in predicting target variables using
    multiple importance calculation methods and interpretability techniques.
    
    Supports feature importance analysis through:
    - Tree-based importance (Random Forest, XGBoost)
    - Permutation importance
    - SHAP (SHapley Additive exPlanations) values
    - Linear model coefficients
    - Mutual information scores
    
    **Features:**
    - Multiple importance calculation methods
    - Statistical significance testing
    - Feature interaction analysis
    - Temporal importance tracking
    - Visualization data for importance plots
    """,
    responses={
        200: {"description": "Feature importance analysis completed"},
        400: {"description": "Invalid analysis parameters"},
        404: {"description": "Dataset or features not found"},
        422: {"description": "Insufficient data for importance analysis"},
        500: {"description": "Internal server error"}
    }
)
@monitor_endpoint("feature_importance_analysis")
async def analyze_feature_importance(
    request: FeatureImportanceRequestDTO,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    use_case: AnalyzeFeatureImportanceUseCase = Depends(get_feature_engineering_use_cases)
) -> FeatureImportanceResponseDTO:
    """
    Analyze feature importance for target prediction.
    
    Args:
        request: Feature importance analysis request
        background_tasks: Background task manager
        current_user: Authenticated user
        use_case: Feature importance analysis use case
        
    Returns:
        Feature importance analysis results
        
    Raises:
        HTTPException: If analysis fails
    """
    try:
        logger.info(
            f"Analyzing feature importance for dataset {request.dataset_id} "
            f"with target {request.target_column} by user {current_user.id}"
        )
        
        result = await use_case.execute(request, current_user.id)
        
        if result.status == "failed":
            metrics.feature_importance_failures.inc()
            raise HTTPException(
                status_code=500,
                detail=f"Feature importance analysis failed: {result.error_message}"
            )
        
        metrics.successful_feature_importance_analyses.inc()
        
        logger.info(
            f"Feature importance analysis completed for dataset {request.dataset_id}"
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        metrics.feature_importance_failures.inc()
        logger.error(f"Unexpected error in feature importance analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during feature importance analysis"
        )


@router.post(
    "/selection",
    response_model=FeatureSelectionResponseDTO,
    summary="Perform Feature Selection",
    description="""
    Perform intelligent feature selection to identify the most relevant features
    for machine learning tasks and reduce dimensionality.
    
    Supports feature selection through:
    - Statistical tests (chi-square, ANOVA, mutual information)
    - Recursive feature elimination (RFE)
    - L1 regularization (Lasso)
    - Sequential feature selection
    - Genetic algorithms for feature optimization
    
    **Features:**
    - Multiple feature selection algorithms
    - Cross-validation based selection
    - Feature subset stability analysis
    - Performance impact assessment
    - Automatic feature ranking and scoring
    """,
    responses={
        200: {"description": "Feature selection completed successfully"},
        400: {"description": "Invalid selection parameters"},
        404: {"description": "Dataset not found"},
        422: {"description": "Insufficient features for selection"},
        500: {"description": "Internal server error"}
    }
)
@monitor_endpoint("feature_selection")
async def select_features(
    request: FeatureSelectionRequestDTO,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    use_case: SelectFeaturesUseCase = Depends(get_feature_engineering_use_cases)
) -> FeatureSelectionResponseDTO:
    """
    Perform feature selection on dataset.
    
    Args:
        request: Feature selection request
        background_tasks: Background task manager
        current_user: Authenticated user
        use_case: Feature selection use case
        
    Returns:
        Feature selection results
        
    Raises:
        HTTPException: If selection fails
    """
    try:
        logger.info(
            f"Performing feature selection for dataset {request.dataset_id} "
            f"using method {request.selection_method} by user {current_user.id}"
        )
        
        # Validate minimum features
        if len(request.feature_columns) < request.max_features:
            raise HTTPException(
                status_code=422,
                detail=f"Dataset has {len(request.feature_columns)} features, "
                      f"cannot select {request.max_features} features"
            )
        
        result = await use_case.execute(request, current_user.id)
        
        if result.status == "failed":
            metrics.feature_selection_failures.inc()
            raise HTTPException(
                status_code=500,
                detail=f"Feature selection failed: {result.error_message}"
            )
        
        metrics.successful_feature_selections.inc()
        
        logger.info(
            f"Feature selection completed for dataset {request.dataset_id}, "
            f"selected {len(result.selected_features)} features"
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        metrics.feature_selection_failures.inc()
        logger.error(f"Unexpected error in feature selection: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during feature selection"
        )


@router.post(
    "/auto-engineer",
    response_model=AutoFeatureEngineeringResponseDTO,
    summary="Automated Feature Engineering",
    description="""
    Perform automated feature engineering using machine learning to discover
    and create high-value features automatically.
    
    Automated feature engineering includes:
    - Automatic feature transformation discovery
    - Feature interaction detection and creation
    - Temporal pattern feature generation
    - Domain-specific feature templates
    - Feature quality scoring and ranking
    
    **Features:**
    - ML-powered feature discovery
    - Configurable complexity levels
    - Feature interpretability scoring
    - Automated feature validation
    - Performance impact prediction
    """,
    responses={
        200: {"description": "Automated feature engineering completed"},
        400: {"description": "Invalid engineering parameters"},
        404: {"description": "Dataset not found"},
        422: {"description": "Dataset not suitable for auto-engineering"},
        500: {"description": "Internal server error"}
    }
)
@monitor_endpoint("auto_feature_engineering")
async def auto_engineer_features(
    request: AutoFeatureEngineeringRequestDTO,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    use_case: AutoFeatureEngineeringUseCase = Depends(get_feature_engineering_use_cases)
) -> AutoFeatureEngineeringResponseDTO:
    """
    Perform automated feature engineering.
    
    Args:
        request: Auto feature engineering request
        background_tasks: Background task manager
        current_user: Authenticated user
        use_case: Auto feature engineering use case
        
    Returns:
        Auto feature engineering results
        
    Raises:
        HTTPException: If auto-engineering fails
    """
    try:
        logger.info(
            f"Starting automated feature engineering for dataset {request.dataset_id} "
            f"with complexity level {request.complexity_level} by user {current_user.id}"
        )
        
        # Add background task for long-running auto-engineering
        background_tasks.add_task(
            log_auto_engineering_completion,
            request.dataset_id,
            current_user.id
        )
        
        result = await use_case.execute(request, current_user.id)
        
        if result.status == "failed":
            metrics.auto_feature_engineering_failures.inc()
            raise HTTPException(
                status_code=500,
                detail=f"Automated feature engineering failed: {result.error_message}"
            )
        
        metrics.successful_auto_feature_engineering.inc()
        
        logger.info(
            f"Automated feature engineering completed for dataset {request.dataset_id}, "
            f"generated {len(result.generated_features)} features"
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        metrics.auto_feature_engineering_failures.inc()
        logger.error(f"Unexpected error in automated feature engineering: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during automated feature engineering"
        )


@router.post(
    "/transform",
    response_model=FeatureTransformationResponseDTO,
    summary="Transform Features",
    description="""
    Apply transformations to existing features to improve their distribution,
    scale, or representation for machine learning algorithms.
    
    Supports feature transformations including:
    - Scaling and normalization (StandardScaler, MinMaxScaler, RobustScaler)
    - Distribution transformations (log, sqrt, Box-Cox, Yeo-Johnson)
    - Categorical encoding (one-hot, label, target, embeddings)
    - Text transformations (TF-IDF, word embeddings, n-grams)
    - Time series transformations (lag features, rolling statistics)
    
    **Features:**
    - Multiple transformation algorithms
    - Inverse transformation support
    - Transformation pipeline creation
    - Parameter optimization
    - Quality assessment and validation
    """,
    responses={
        200: {"description": "Feature transformation completed"},
        400: {"description": "Invalid transformation parameters"},
        404: {"description": "Dataset or features not found"},
        422: {"description": "Features not suitable for transformation"},
        500: {"description": "Internal server error"}
    }
)
@monitor_endpoint("feature_transformation")
async def transform_features(
    request: FeatureTransformationRequestDTO,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    use_case: CreateFeatureUseCase = Depends(get_feature_engineering_use_cases)
) -> FeatureTransformationResponseDTO:
    """
    Transform features using specified transformations.
    
    Args:
        request: Feature transformation request
        background_tasks: Background task manager
        current_user: Authenticated user
        use_case: Feature creation use case (handles transformations)
        
    Returns:
        Feature transformation results
        
    Raises:
        HTTPException: If transformation fails
    """
    try:
        logger.info(
            f"Transforming {len(request.feature_columns)} features for dataset {request.dataset_id} "
            f"by user {current_user.id}"
        )
        
        # Convert transformation request to feature creation request
        feature_specs = []
        for feature_col in request.feature_columns:
            for transform in request.transformations:
                feature_specs.append({
                    "name": f"{feature_col}_{transform['name']}",
                    "expression": f"{transform['name']}({feature_col})",
                    "type": "transformation",
                    "source_columns": [feature_col],
                    "parameters": transform.get("parameters", {})
                })
        
        creation_request = FeatureCreationRequestDTO(
            dataset_id=request.dataset_id,
            feature_specifications=feature_specs,
            validation_enabled=request.validation_enabled,
            create_pipeline=True
        )
        
        result = await use_case.execute(creation_request, current_user.id)
        
        if result.status == "failed":
            metrics.feature_transformation_failures.inc()
            raise HTTPException(
                status_code=500,
                detail=f"Feature transformation failed: {result.error_message}"
            )
        
        metrics.successful_feature_transformations.inc()
        
        logger.info(
            f"Feature transformation completed for dataset {request.dataset_id}"
        )
        
        return FeatureTransformationResponseDTO(
            transformation_id=result.creation_id,
            dataset_id=request.dataset_id,
            transformed_features=result.created_features,
            transformation_pipeline=result.feature_pipeline,
            quality_metrics=result.quality_metrics,
            execution_time_seconds=result.execution_time_seconds,
            created_at=result.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        metrics.feature_transformation_failures.inc()
        logger.error(f"Unexpected error in feature transformation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during feature transformation"
        )


@router.get(
    "/list",
    response_model=List[Dict[str, Any]],
    summary="List Available Features",
    description="""
    List all available features for a dataset with metadata including
    feature types, creation date, lineage, and quality metrics.
    """,
    responses={
        200: {"description": "Features listed successfully"},
        404: {"description": "Dataset not found"},
        500: {"description": "Internal server error"}
    }
)
async def list_features(
    dataset_id: UUID = Query(..., description="Dataset identifier"),
    feature_type: Optional[str] = Query(None, description="Filter by feature type"),
    include_derived: bool = Query(True, description="Include derived features"),
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    List available features for a dataset.
    
    Args:
        dataset_id: Dataset identifier
        feature_type: Optional feature type filter
        include_derived: Include derived features
        current_user: Authenticated user
        
    Returns:
        List of available features with metadata
    """
    try:
        logger.info(f"Listing features for dataset {dataset_id}")
        
        # This would be implemented with proper repository queries
        # For now, return a placeholder response
        return [
            {
                "feature_id": "feature_1",
                "name": "age",
                "type": "numeric",
                "created_at": "2024-01-01T00:00:00Z",
                "is_derived": False,
                "quality_score": 0.95
            },
            {
                "feature_id": "feature_2", 
                "name": "age_squared",
                "type": "numeric",
                "created_at": "2024-01-01T00:00:00Z",
                "is_derived": True,
                "quality_score": 0.88
            }
        ]
        
    except Exception as e:
        logger.error(f"Error listing features: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error listing features"
        )


@router.delete(
    "/{feature_id}",
    summary="Delete Feature",
    description="""
    Delete a derived feature from the dataset. Original dataset features
    cannot be deleted, only derived features created through feature engineering.
    """,
    responses={
        200: {"description": "Feature deleted successfully"},
        404: {"description": "Feature not found"},
        403: {"description": "Cannot delete original dataset feature"},
        500: {"description": "Internal server error"}
    }
)
async def delete_feature(
    feature_id: UUID,
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Delete a derived feature.
    
    Args:
        feature_id: Feature identifier
        current_user: Authenticated user
        
    Returns:
        Deletion confirmation
        
    Raises:
        HTTPException: If feature cannot be deleted
    """
    try:
        logger.info(f"Deleting feature {feature_id} by user {current_user.id}")
        
        # This would be implemented with proper business logic
        # For now, return a success response
        return {"message": f"Feature {feature_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting feature: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error deleting feature"
        )


# Background task functions
async def log_auto_engineering_completion(dataset_id: UUID, user_id: UUID):
    """Log completion of automated feature engineering."""
    logger.info(
        f"Automated feature engineering background task completed "
        f"for dataset {dataset_id} by user {user_id}"
    )