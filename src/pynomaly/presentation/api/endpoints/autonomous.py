"""Autonomous detection API endpoints."""

from __future__ import annotations

from typing import Dict, List, Optional, Union
from uuid import UUID
import asyncio

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from pynomaly.application.services.autonomous_service import (
    AutonomousDetectionService,
    AutonomousConfig
)
from pynomaly.application.services.automl_service import (
    AutoMLService,
    OptimizationObjective,
    AlgorithmFamily
)
from pynomaly.application.services.ensemble_service import EnsembleService
from pynomaly.infrastructure.config import Container
from pynomaly.infrastructure.data_loaders.csv_loader import CSVLoader
from pynomaly.infrastructure.data_loaders.parquet_loader import ParquetLoader
from pynomaly.infrastructure.data_loaders.json_loader import JSONLoader
from pynomaly.infrastructure.data_loaders.excel_loader import ExcelLoader
from pynomaly.presentation.api.deps import get_container, get_current_user


router = APIRouter()


class AutonomousDetectionRequest(BaseModel):
    """Request for autonomous detection."""
    data_source: Optional[str] = None  # File path or URL
    max_algorithms: int = Field(5, ge=1, le=15)
    confidence_threshold: float = Field(0.8, ge=0.1, le=1.0)
    auto_tune: bool = True
    save_results: bool = True
    export_format: str = "json"
    enable_preprocessing: bool = True
    quality_threshold: float = Field(0.8, ge=0.1, le=1.0)
    max_preprocessing_time: float = Field(300.0, gt=0.0)
    preprocessing_strategy: str = "auto"


class AutoMLRequest(BaseModel):
    """Request for AutoML optimization."""
    dataset_id: UUID
    objective: OptimizationObjective = OptimizationObjective.AUC
    max_algorithms: int = Field(5, ge=1, le=10)
    optimization_time: int = Field(3600, ge=60, le=7200)  # seconds
    n_trials: int = Field(100, ge=10, le=1000)
    enable_ensemble: bool = True


class EnsembleRequest(BaseModel):
    """Request for ensemble creation."""
    name: str
    detector_ids: List[UUID] = Field(..., min_items=2)
    weights: Optional[Dict[str, float]] = None
    aggregation_method: str = "weighted_voting"


class FamilyEnsembleRequest(BaseModel):
    """Request for family-based ensemble."""
    dataset_id: UUID
    families: List[str] = Field(..., min_items=1)
    enable_family_ensembles: bool = True
    enable_meta_ensemble: bool = True
    optimization_time: int = Field(1800, ge=300, le=3600)


class ExplainChoicesRequest(BaseModel):
    """Request for algorithm choice explanation."""
    dataset_id: Optional[UUID] = None
    max_algorithms: int = Field(5, ge=1, le=15)
    include_alternatives: bool = True
    include_data_analysis: bool = True


@router.post("/detect")
async def autonomous_detect(
    request: AutonomousDetectionRequest,
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    container: Container = Depends(get_container),
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict:
    """Run autonomous anomaly detection.
    
    Supports both file upload and data source path/URL.
    Automatically detects format, profiles data, selects algorithms,
    and runs optimized detection.
    """
    
    # Determine data source
    data_source = None
    if file:
        # Handle file upload
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp:
            content = await file.read()
            tmp.write(content)
            data_source = tmp.name
    elif request.data_source:
        data_source = request.data_source
    else:
        raise HTTPException(
            status_code=400,
            detail="Either file upload or data_source path must be provided"
        )
    
    # Create configuration
    config = AutonomousConfig(
        max_algorithms=request.max_algorithms,
        confidence_threshold=request.confidence_threshold,
        auto_tune_hyperparams=request.auto_tune,
        save_results=request.save_results,
        export_results=False,  # API doesn't export to files
        export_format=request.export_format,
        verbose=False,  # API doesn't need verbose console output
        enable_preprocessing=request.enable_preprocessing,
        quality_threshold=request.quality_threshold,
        max_preprocessing_time=request.max_preprocessing_time,
        preprocessing_strategy=request.preprocessing_strategy
    )
    
    # Setup data loaders
    data_loaders = {
        "csv": CSVLoader(),
        "parquet": ParquetLoader(),
        "json": JSONLoader(),
        "excel": ExcelLoader()
    }
    
    # Create autonomous service
    autonomous_service = AutonomousDetectionService(
        detector_repository=container.detector_repository(),
        result_repository=container.result_repository(),
        data_loaders=data_loaders
    )
    
    try:
        # Run autonomous detection
        results = await autonomous_service.detect_autonomous(data_source, config)
        
        # Clean up temporary file if created
        if file:
            import os
            os.unlink(data_source)
        
        return {
            "success": True,
            "results": results,
            "metadata": {
                "algorithms_tested": len(results.get("autonomous_detection_results", {}).get("detection_results", {})),
                "data_source_type": "upload" if file else "path",
                "preprocessing_applied": results.get("autonomous_detection_results", {}).get("data_profile", {}).get("preprocessing_applied", False)
            }
        }
        
    except Exception as e:
        # Clean up temporary file if created
        if file:
            import os
            os.unlink(data_source)
        
        raise HTTPException(
            status_code=400,
            detail=f"Autonomous detection failed: {str(e)}"
        )


@router.post("/automl/optimize")
async def automl_optimize(
    request: AutoMLRequest,
    background_tasks: BackgroundTasks,
    container: Container = Depends(get_container),
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict:
    """Run AutoML optimization for dataset.
    
    Automatically profiles dataset, recommends algorithms,
    optimizes hyperparameters, and optionally creates ensembles.
    """
    
    # Validate dataset exists
    dataset_repo = container.dataset_repository()
    dataset = await dataset_repo.get(str(request.dataset_id))
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Create AutoML service
    automl_service = AutoMLService(
        detector_repository=container.detector_repository(),
        dataset_repository=dataset_repo,
        adapter_registry=container.adapter_registry(),
        max_optimization_time=request.optimization_time,
        n_trials=request.n_trials
    )
    
    try:
        # Run AutoML optimization
        automl_result = await automl_service.auto_select_and_optimize(
            dataset_id=str(request.dataset_id),
            objective=request.objective,
            max_algorithms=request.max_algorithms,
            enable_ensemble=request.enable_ensemble
        )
        
        # Create optimized detector
        detector_id = await automl_service.create_optimized_detector(
            automl_result,
            detector_name=f"AutoML_{automl_result.best_algorithm}"
        )
        
        return {
            "success": True,
            "automl_result": {
                "best_algorithm": automl_result.best_algorithm,
                "best_params": automl_result.best_params,
                "best_score": automl_result.best_score,
                "optimization_time": automl_result.optimization_time,
                "trials_completed": automl_result.trials_completed,
                "algorithm_rankings": automl_result.algorithm_rankings,
                "has_ensemble": automl_result.ensemble_config is not None
            },
            "detector_id": detector_id,
            "summary": automl_service.get_optimization_summary(automl_result)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"AutoML optimization failed: {str(e)}"
        )


@router.post("/ensemble/create")
async def create_ensemble(
    request: EnsembleRequest,
    container: Container = Depends(get_container),
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict:
    """Create ensemble from multiple detectors."""
    
    # Create ensemble service
    ensemble_service = EnsembleService(
        detector_repository=container.detector_repository(),
        ensemble_aggregator=container.ensemble_aggregator(),
        anomaly_scorer=container.anomaly_scorer()
    )
    
    try:
        # Create ensemble
        ensemble = await ensemble_service.create_ensemble(
            name=request.name,
            detector_ids=request.detector_ids,
            weights=request.weights,
            aggregation_method=request.aggregation_method
        )
        
        return {
            "success": True,
            "ensemble_id": str(ensemble.id),
            "ensemble_name": ensemble.name,
            "base_detectors": [str(d.id) for d in ensemble.base_detectors],
            "aggregation_method": ensemble.aggregation_method
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Ensemble creation failed: {str(e)}"
        )


@router.post("/ensemble/create-by-family")
async def create_family_ensemble(
    request: FamilyEnsembleRequest,
    background_tasks: BackgroundTasks,
    container: Container = Depends(get_container),
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict:
    """Create hierarchical ensemble organized by algorithm families."""
    
    # Validate dataset
    dataset_repo = container.dataset_repository()
    dataset = await dataset_repo.get(str(request.dataset_id))
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Algorithm family mapping
    family_algorithms = {
        "statistical": ["ECOD", "COPOD"],
        "distance_based": ["KNN", "LOF", "OneClassSVM"],
        "isolation_based": ["IsolationForest"],
        "density_based": ["LOF"],
        "neural_networks": ["AutoEncoder", "VAE"]
    }
    
    # Validate families
    invalid_families = [f for f in request.families if f not in family_algorithms]
    if invalid_families:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid algorithm families: {invalid_families}"
        )
    
    try:
        # Create AutoML service for each family
        automl_service = AutoMLService(
            detector_repository=container.detector_repository(),
            dataset_repository=dataset_repo,
            adapter_registry=container.adapter_registry(),
            max_optimization_time=request.optimization_time // len(request.families)
        )
        
        family_results = {}
        family_ensembles = {}
        
        # Process each family
        for family in request.families:
            algorithms = family_algorithms[family]
            
            # Run optimization for family algorithms
            # This is a simplified implementation - would need algorithm filtering
            family_result = await automl_service.auto_select_and_optimize(
                dataset_id=str(request.dataset_id),
                max_algorithms=len(algorithms),
                enable_ensemble=request.enable_family_ensembles
            )
            
            family_results[family] = family_result
            
            if request.enable_family_ensembles and family_result.ensemble_config:
                # Create family ensemble detector
                detector_id = await automl_service.create_optimized_detector(
                    family_result,
                    detector_name=f"Ensemble_{family}"
                )
                family_ensembles[family] = detector_id
        
        # Create meta-ensemble if requested
        meta_ensemble_id = None
        if request.enable_meta_ensemble and len(family_ensembles) > 1:
            ensemble_service = EnsembleService(
                detector_repository=container.detector_repository(),
                ensemble_aggregator=container.ensemble_aggregator(),
                anomaly_scorer=container.anomaly_scorer()
            )
            
            meta_ensemble = await ensemble_service.create_ensemble(
                name="Meta_Ensemble",
                detector_ids=list(family_ensembles.values()),
                aggregation_method="weighted_voting"
            )
            meta_ensemble_id = str(meta_ensemble.id)
        
        return {
            "success": True,
            "family_results": {
                family: {
                    "best_algorithm": result.best_algorithm,
                    "best_score": result.best_score,
                    "ensemble_id": family_ensembles.get(family)
                }
                for family, result in family_results.items()
            },
            "meta_ensemble_id": meta_ensemble_id,
            "total_algorithms_tested": sum(len(family_algorithms[f]) for f in request.families)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Family ensemble creation failed: {str(e)}"
        )


@router.post("/explain/choices")
async def explain_algorithm_choices(
    request: ExplainChoicesRequest,
    data_file: Optional[UploadFile] = File(None),
    container: Container = Depends(get_container),
    current_user: Optional[str] = Depends(get_current_user)
) -> Dict:
    """Explain algorithm selection choices for a dataset."""
    
    dataset = None
    temp_file = None
    
    try:
        # Get dataset
        if request.dataset_id:
            dataset_repo = container.dataset_repository()
            dataset = await dataset_repo.get(str(request.dataset_id))
            if not dataset:
                raise HTTPException(status_code=404, detail="Dataset not found")
        
        elif data_file:
            # Handle file upload for analysis
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{data_file.filename.split('.')[-1]}") as tmp:
                content = await data_file.read()
                tmp.write(content)
                temp_file = tmp.name
            
            # Load dataset for analysis
            data_loaders = {
                "csv": CSVLoader(),
                "parquet": ParquetLoader(),
                "json": JSONLoader(),
                "excel": ExcelLoader()
            }
            
            autonomous_service = AutonomousDetectionService(
                detector_repository=container.detector_repository(),
                result_repository=container.result_repository(),
                data_loaders=data_loaders
            )
            
            config = AutonomousConfig(verbose=False)
            dataset = await autonomous_service._auto_load_data(temp_file, config)
        
        else:
            raise HTTPException(
                status_code=400,
                detail="Either dataset_id or data file must be provided"
            )
        
        # Create services for analysis
        data_loaders = {
            "csv": CSVLoader(),
            "parquet": ParquetLoader(),
            "json": JSONLoader(),
            "excel": ExcelLoader()
        }
        
        autonomous_service = AutonomousDetectionService(
            detector_repository=container.detector_repository(),
            result_repository=container.result_repository(),
            data_loaders=data_loaders
        )
        
        automl_service = AutoMLService(
            detector_repository=container.detector_repository(),
            dataset_repository=container.dataset_repository(),
            adapter_registry=container.adapter_registry()
        )
        
        # Profile dataset
        config = AutonomousConfig(max_algorithms=request.max_algorithms)
        profile = await autonomous_service._profile_data(dataset, config)
        
        # Get algorithm recommendations
        recommendations = await autonomous_service._recommend_algorithms(profile, config)
        
        # Get AutoML dataset profile for additional insights
        automl_profile = await automl_service.profile_dataset(str(dataset.id) if hasattr(dataset, 'id') else "temp")
        
        # Build explanation response
        explanations = {
            "dataset_analysis": {
                "basic_stats": {
                    "samples": profile.n_samples,
                    "features": profile.n_features,
                    "numeric_features": profile.numeric_features,
                    "categorical_features": profile.categorical_features,
                    "missing_ratio": profile.missing_values_ratio,
                    "complexity_score": profile.complexity_score
                },
                "data_characteristics": {
                    "sparsity_ratio": profile.sparsity_ratio,
                    "correlation_score": profile.correlation_score,
                    "outlier_estimate": profile.outlier_ratio_estimate,
                    "recommended_contamination": profile.recommended_contamination
                }
            } if request.include_data_analysis else None,
            
            "algorithm_recommendations": [
                {
                    "rank": i + 1,
                    "algorithm": rec.algorithm,
                    "confidence": rec.confidence,
                    "reasoning": rec.reasoning,
                    "expected_performance": rec.expected_performance,
                    "hyperparameters": rec.hyperparams,
                    "suitability_factors": {
                        "data_size_match": "good" if profile.n_samples >= 100 else "poor",
                        "complexity_match": "good" if abs(profile.complexity_score - 0.5) < 0.3 else "moderate",
                        "feature_support": "good" if profile.numeric_features > 0 else "limited"
                    }
                }
                for i, rec in enumerate(recommendations[:request.max_algorithms])
            ],
            
            "alternatives_considered": [
                {
                    "algorithm": rec.algorithm,
                    "confidence": rec.confidence,
                    "why_not_chosen": f"Lower confidence ({rec.confidence:.1%}) than top recommendations"
                }
                for rec in recommendations[request.max_algorithms:request.max_algorithms+3]
            ] if request.include_alternatives else [],
            
            "recommendation_summary": {
                "top_choice": recommendations[0].algorithm if recommendations else None,
                "confidence_level": "high" if recommendations and recommendations[0].confidence > 0.8 else "moderate",
                "key_factors": [
                    f"Dataset size: {profile.n_samples:,} samples",
                    f"Complexity: {profile.complexity_score:.2f}",
                    f"Feature types: {profile.numeric_features} numeric, {profile.categorical_features} categorical"
                ]
            }
        }
        
        return {
            "success": True,
            "explanations": explanations
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Algorithm choice explanation failed: {str(e)}"
        )
    
    finally:
        # Clean up temporary file if created
        if temp_file:
            import os
            os.unlink(temp_file)


@router.get("/algorithms/families")
async def list_algorithm_families() -> Dict:
    """List available algorithm families and their algorithms."""
    
    families = {
        "statistical": {
            "algorithms": ["ECOD", "COPOD"],
            "description": "Statistical methods for outlier detection",
            "strengths": ["Fast computation", "Good for Gaussian data"],
            "limitations": ["Assumes data distribution"]
        },
        "distance_based": {
            "algorithms": ["KNN", "LOF", "OneClassSVM"],
            "description": "Distance and density-based methods",
            "strengths": ["No distribution assumptions", "Local anomaly detection"],
            "limitations": ["Sensitive to dimensionality", "Computation intensive"]
        },
        "isolation_based": {
            "algorithms": ["IsolationForest"],
            "description": "Isolation-based anomaly detection",
            "strengths": ["Efficient for large datasets", "Good for high dimensions"],
            "limitations": ["May miss clustered anomalies"]
        },
        "neural_networks": {
            "algorithms": ["AutoEncoder", "VAE"],
            "description": "Deep learning approaches",
            "strengths": ["Captures complex patterns", "Non-linear relationships"],
            "limitations": ["Requires large datasets", "Longer training time"]
        }
    }
    
    return {
        "success": True,
        "families": families,
        "total_algorithms": sum(len(f["algorithms"]) for f in families.values())
    }


@router.get("/status")
async def get_autonomous_status(
    container: Container = Depends(get_container)
) -> Dict:
    """Get status of autonomous detection capabilities."""
    
    # Check available adapters
    adapter_registry = container.adapter_registry()
    available_adapters = []
    
    try:
        pyod_adapter = container.pyod_adapter()
        available_adapters.append("pyod")
    except:
        pass
    
    try:
        sklearn_adapter = container.sklearn_adapter()
        available_adapters.append("sklearn")
    except:
        pass
    
    try:
        pytorch_adapter = container.pytorch_adapter()
        available_adapters.append("pytorch")
    except:
        pass
    
    return {
        "autonomous_available": True,
        "automl_available": True,
        "ensemble_available": True,
        "available_adapters": available_adapters,
        "supported_formats": ["csv", "parquet", "json", "excel"],
        "capabilities": {
            "data_profiling": True,
            "algorithm_selection": True,
            "hyperparameter_optimization": True,
            "ensemble_creation": True,
            "preprocessing": True,
            "explanation": True
        }
    }