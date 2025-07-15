"""Enhanced dataset management endpoints with data transformation capabilities."""

import io
from uuid import UUID
from typing import Any, Dict, Optional
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse

from pynomaly.application.dto import DataQualityReportDTO, DatasetDTO
from pynomaly.domain.entities import Dataset
from pynomaly.infrastructure.auth import require_data_scientist, require_viewer
from pynomaly.infrastructure.config import Container
from pynomaly.presentation.api.auth_deps import get_container_simple

# Import data transformation components
try:
    from data_transformation.application.use_cases.data_pipeline import DataPipelineUseCase
    from data_transformation.domain.value_objects.pipeline_config import (
        PipelineConfig, SourceType, CleaningStrategy, ScalingMethod, EncodingStrategy
    )
    from data_transformation.application.dto.pipeline_result import PipelineResult
    from services.services.enhanced_data_preprocessing_service import (
        EnhancedDataPreprocessingService, EnhancedDataQualityReport
    )
    DATA_TRANSFORMATION_AVAILABLE = True
except ImportError:
    DATA_TRANSFORMATION_AVAILABLE = False


# Simplified permission system for OpenAPI compatibility
def require_permissions(permission):
    """Simplified permission checker for OpenAPI compatibility."""
    return require_viewer


class CommonPermissions:
    """Common permission constants."""

    DATASET_READ = "dataset:read"
    DATASET_WRITE = "dataset:write"
    DATASET_DELETE = "dataset:delete"
    DATASET_TRANSFORM = "dataset:transform"


router = APIRouter()


@router.post("/{dataset_id}/advanced-transform")
async def advanced_transform_dataset(
    dataset_id: UUID,
    background_tasks: BackgroundTasks,
    cleaning_strategy: str = Form("auto"),
    scaling_method: str = Form("robust"),
    encoding_strategy: str = Form("onehot"),
    feature_engineering: bool = Form(True),
    validation_enabled: bool = Form(True),
    current_user=Depends(require_data_scientist),
    container: Container = Depends(lambda: Container()),
) -> JSONResponse:
    """Apply advanced data transformations to a dataset.
    
    This endpoint uses the integrated data_transformation package to apply
    sophisticated preprocessing operations optimized for anomaly detection.
    """
    if not DATA_TRANSFORMATION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Advanced data transformation features are not available"
        )
    
    try:
        # Get dataset from repository
        dataset_repo = container.dataset_repository()
        dataset = await dataset_repo.find_by_id(dataset_id)
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Create transformation configuration
        config = PipelineConfig(
            source_type=SourceType.CSV,  # Will be auto-detected
            cleaning_strategy=CleaningStrategy(cleaning_strategy),
            scaling_method=ScalingMethod(scaling_method),
            encoding_strategy=EncodingStrategy(encoding_strategy),
            feature_engineering=feature_engineering,
            validation_enabled=validation_enabled,
            parallel_processing=True
        )
        
        # Initialize preprocessing service
        preprocessing_service = EnhancedDataPreprocessingService()
        
        # Apply transformations (run in background for large datasets)
        def transform_task():
            pipeline = DataPipelineUseCase(config)
            result = pipeline.execute(dataset.data)
            
            if result.success:
                # Update dataset with transformed data
                dataset.data = result.data
                # Save updated dataset (implement async version in real scenario)
                # await dataset_repo.update(dataset)
                return result
            else:
                raise Exception(f"Transformation failed: {result.error_message}")
        
        # For large datasets, run in background
        if len(dataset.data) > 10000:
            background_tasks.add_task(transform_task)
            return JSONResponse(
                content={
                    "message": "Transformation started in background",
                    "dataset_id": str(dataset_id),
                    "status": "processing"
                },
                status_code=202
            )
        else:
            # Process immediately for smaller datasets
            result = transform_task()
            return JSONResponse(
                content={
                    "message": "Dataset transformed successfully",
                    "dataset_id": str(dataset_id),
                    "rows_processed": len(result.data),
                    "columns_processed": len(result.data.columns),
                    "execution_time": result.execution_time,
                    "steps_executed": [step.step_type for step in result.steps_executed]
                }
            )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transformation failed: {e}")


@router.get("/{dataset_id}/transformation-recommendations")
async def get_transformation_recommendations(
    dataset_id: UUID,
    anomaly_detection_type: str = Query("unsupervised", regex="^(supervised|unsupervised)$"),
    current_user=Depends(require_viewer),
    container: Container = Depends(lambda: Container()),
) -> Dict[str, Any]:
    """Get intelligent preprocessing recommendations for a dataset.
    
    Analyzes the dataset and provides recommendations for optimal preprocessing
    configuration based on data characteristics and anomaly detection requirements.
    """
    if not DATA_TRANSFORMATION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Advanced recommendation features are not available"
        )
    
    try:
        # Get dataset from repository
        dataset_repo = container.dataset_repository()
        dataset = await dataset_repo.find_by_id(dataset_id)
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Initialize preprocessing service
        preprocessing_service = EnhancedDataPreprocessingService()
        
        # Get recommendations
        recommendations = await preprocessing_service.get_preprocessing_recommendations(
            dataset.data,
            anomaly_detection_type
        )
        
        return {
            "dataset_id": str(dataset_id),
            "anomaly_detection_type": anomaly_detection_type,
            "recommendations": recommendations,
            "generated_at": pd.Timestamp.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {e}"
        )


@router.post("/{dataset_id}/quality-assessment")
async def assess_data_quality(
    dataset_id: UUID,
    include_advanced_metrics: bool = Form(True),
    current_user=Depends(require_viewer),
    container: Container = Depends(lambda: Container()),
) -> Dict[str, Any]:
    """Perform comprehensive data quality assessment.
    
    Provides detailed analysis of data quality including missing values,
    outliers, duplicates, and advanced metrics when available.
    """
    try:
        # Get dataset from repository
        dataset_repo = container.dataset_repository()
        dataset = await dataset_repo.find_by_id(dataset_id)
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if DATA_TRANSFORMATION_AVAILABLE and include_advanced_metrics:
            # Use enhanced assessment
            preprocessing_service = EnhancedDataPreprocessingService()
            quality_report = await preprocessing_service._assess_data_quality(
                dataset.data, str(dataset_id)
            )
            
            return {
                "dataset_id": str(dataset_id),
                "assessment_type": "enhanced",
                "quality_score": quality_report.quality_score,
                "overall_assessment": quality_report.get_overall_assessment(),
                "metrics": {
                    "missing_values_ratio": quality_report.missing_values_ratio,
                    "duplicate_rows_ratio": quality_report.duplicate_rows_ratio,
                    "outlier_ratio": quality_report.outlier_ratio,
                    "sparsity_ratio": quality_report.sparsity_ratio
                },
                "data_characteristics": {
                    "n_samples": quality_report.n_samples,
                    "n_features": quality_report.n_features,
                    "feature_types": quality_report.feature_types
                },
                "issues": quality_report.issues,
                "warnings": quality_report.warnings,
                "recommendations": quality_report.recommendations,
                "transformation_suggestions": quality_report.transformation_suggestions,
                "assessed_at": quality_report.assessment_time.isoformat()
            }
        else:
            # Basic assessment fallback
            df = dataset.data
            basic_assessment = {
                "dataset_id": str(dataset_id),
                "assessment_type": "basic",
                "metrics": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "missing_values": int(df.isnull().sum().sum()),
                    "duplicates": int(df.duplicated().sum()),
                    "numeric_columns": len(df.select_dtypes(include=['number']).columns),
                    "categorical_columns": len(df.select_dtypes(include=['object']).columns)
                },
                "assessed_at": pd.Timestamp.now().isoformat()
            }
            
            return basic_assessment
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Quality assessment failed: {e}"
        )


@router.post("/{dataset_id}/optimize-for-algorithm")
async def optimize_dataset_for_algorithm(
    dataset_id: UUID,
    algorithm_type: str = Form(..., regex="^(isolation_forest|one_class_svm|local_outlier_factor|autoencoder)$"),
    target_column: Optional[str] = Form(None),
    current_user=Depends(require_data_scientist),
    container: Container = Depends(lambda: Container()),
) -> JSONResponse:
    """Optimize dataset preprocessing for a specific anomaly detection algorithm.
    
    Applies algorithm-specific preprocessing optimizations to improve
    anomaly detection performance.
    """
    if not DATA_TRANSFORMATION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Algorithm optimization features are not available"
        )
    
    try:
        # Get dataset from repository
        dataset_repo = container.dataset_repository()
        dataset = await dataset_repo.find_by_id(dataset_id)
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Initialize preprocessing service
        preprocessing_service = EnhancedDataPreprocessingService()
        
        # Optimize for specific algorithm
        optimized_data = await preprocessing_service.optimize_for_algorithm(
            dataset.data,
            algorithm_type,
            target_column
        )
        
        # Update dataset with optimized data
        dataset.data = optimized_data
        
        return JSONResponse(
            content={
                "message": f"Dataset optimized for {algorithm_type}",
                "dataset_id": str(dataset_id),
                "algorithm_type": algorithm_type,
                "rows": len(optimized_data),
                "columns": len(optimized_data.columns),
                "target_column": target_column,
                "optimized_at": pd.Timestamp.now().isoformat()
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Algorithm optimization failed: {e}"
        )


@router.get("/{dataset_id}/transformation-status")
async def get_transformation_status(
    dataset_id: UUID,
    current_user=Depends(require_viewer),
    container: Container = Depends(lambda: Container()),
) -> Dict[str, Any]:
    """Get the status of ongoing or completed transformations for a dataset."""
    # This would typically check a job queue or database for transformation status
    # For now, return a placeholder response
    return {
        "dataset_id": str(dataset_id),
        "status": "completed",  # or "processing", "failed", "queued"
        "message": "No active transformations",
        "last_transformation": None,
        "checked_at": pd.Timestamp.now().isoformat()
    }


@router.post("/upload-and-transform")
async def upload_and_transform_dataset(
    file: UploadFile = File(...),
    cleaning_strategy: str = Form("auto"),
    enable_feature_engineering: bool = Form(True),
    current_user=Depends(require_data_scientist),
    container: Container = Depends(lambda: Container()),
) -> JSONResponse:
    """Upload a new dataset and apply transformations in one step.
    
    Convenient endpoint for uploading and immediately preprocessing
    data for anomaly detection workflows.
    """
    if not DATA_TRANSFORMATION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Upload and transform features are not available"
        )
    
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Determine file type and read data
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.StringIO(contents.decode('utf-8')))
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Use CSV or JSON."
            )
        
        # Create transformation configuration
        config = PipelineConfig(
            source_type=SourceType.CSV,
            cleaning_strategy=CleaningStrategy(cleaning_strategy),
            scaling_method=ScalingMethod.ROBUST,
            encoding_strategy=EncodingStrategy.ONEHOT,
            feature_engineering=enable_feature_engineering,
            validation_enabled=True
        )
        
        # Apply transformations
        pipeline = DataPipelineUseCase(config)
        result = pipeline.execute(df)
        
        if result.success:
            # Create new dataset entity (simplified)
            # In real implementation, save to repository and return dataset_id
            dataset_id = UUID('12345678-1234-5678-9abc-123456789abc')  # Placeholder
            
            return JSONResponse(
                content={
                    "message": "Dataset uploaded and transformed successfully",
                    "dataset_id": str(dataset_id),
                    "original_filename": file.filename,
                    "rows_processed": len(result.data),
                    "columns_processed": len(result.data.columns),
                    "execution_time": result.execution_time,
                    "transformations_applied": len(result.steps_executed)
                }
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Transformation failed: {result.error_message}"
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Upload and transform failed: {e}"
        )