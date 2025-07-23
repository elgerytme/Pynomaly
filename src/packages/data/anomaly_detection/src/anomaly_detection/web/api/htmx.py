"""HTMX endpoints for dynamic content."""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ...domain.services.detection_service import DetectionService
from ...domain.services.ensemble_service import EnsembleService
from ...infrastructure.repositories.model_repository import ModelRepository
from ...infrastructure.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# Dependency injection
_detection_service: DetectionService = None
_ensemble_service: EnsembleService = None
_model_repository: ModelRepository = None


def get_detection_service() -> DetectionService:
    """Get detection service instance."""
    global _detection_service
    if _detection_service is None:
        _detection_service = DetectionService()
    return _detection_service


def get_ensemble_service() -> EnsembleService:
    """Get ensemble service instance."""
    global _ensemble_service
    if _ensemble_service is None:
        _ensemble_service = EnsembleService()
    return _ensemble_service


def get_model_repository() -> ModelRepository:
    """Get model repository instance."""
    global _model_repository
    if _model_repository is None:
        _model_repository = ModelRepository()
    return _model_repository


@router.post("/detect", response_class=HTMLResponse)
async def run_detection(
    request: Request,
    algorithm: str = Form(...),
    contamination: float = Form(0.1),
    sample_data: str = Form(""),
    detection_service: DetectionService = Depends(get_detection_service)
):
    """Run anomaly detection and return results."""
    try:
        # Parse sample data (expecting JSON array)
        if not sample_data.strip():
            # Generate sample data if none provided
            np.random.seed(42)
            normal_data = np.random.normal(0, 1, (100, 5))
            anomaly_data = np.random.normal(3, 1, (10, 5))
            data_array = np.vstack([normal_data, anomaly_data])
        else:
            try:
                data_list = json.loads(sample_data)
                data_array = np.array(data_list, dtype=np.float64)
            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(status_code=400, detail=f"Invalid data format: {str(e)}")
        
        # Map algorithm names
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'lof': 'lof'
        }
        
        algorithm_name = algorithm_map.get(algorithm, algorithm)
        
        # Run detection
        start_time = datetime.utcnow()
        result = detection_service.detect_anomalies(
            data=data_array,
            algorithm=algorithm_name,
            contamination=contamination
        )
        end_time = datetime.utcnow()
        
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        # Prepare results for template
        results = {
            "success": result.success,
            "algorithm": algorithm,
            "total_samples": result.total_samples,
            "anomalies_detected": result.anomaly_count,
            "anomaly_rate": f"{result.anomaly_rate:.1%}",
            "processing_time_ms": f"{processing_time:.2f}",
            "timestamp": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "anomaly_indices": result.anomalies[:10],  # Show first 10
            "contamination": contamination
        }
        
        return templates.TemplateResponse(
            "components/detection_results.html",
            {"request": request, "results": results}
        )
        
    except Exception as e:
        logger.error("Detection failed", error=str(e))
        error_context = {
            "request": request,
            "error": str(e),
            "algorithm": algorithm
        }
        return templates.TemplateResponse(
            "components/error_message.html",
            error_context,
            status_code=500
        )


@router.post("/ensemble", response_class=HTMLResponse)
async def run_ensemble_detection(
    request: Request,
    algorithms: List[str] = Form(...),
    method: str = Form("majority"),
    contamination: float = Form(0.1),
    sample_data: str = Form(""),
    ensemble_service: EnsembleService = Depends(get_ensemble_service)
):
    """Run ensemble anomaly detection."""
    try:
        # Parse sample data
        if not sample_data.strip():
            # Generate sample data if none provided
            np.random.seed(42)
            normal_data = np.random.normal(0, 1, (100, 5))
            anomaly_data = np.random.normal(3, 1, (10, 5))
            data_array = np.vstack([normal_data, anomaly_data])
        else:
            try:
                data_list = json.loads(sample_data)
                data_array = np.array(data_list, dtype=np.float64)
            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(status_code=400, detail=f"Invalid data format: {str(e)}")
        
        # Map algorithm names
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'lof': 'lof'
        }
        
        mapped_algorithms = [algorithm_map.get(alg, alg) for alg in algorithms]
        
        # Get individual results
        start_time = datetime.utcnow()
        individual_results = []
        for algorithm in mapped_algorithms:
            detection_service = DetectionService()
            result = detection_service.detect_anomalies(
                data=data_array,
                algorithm=algorithm,
                contamination=contamination
            )
            individual_results.append(result)
        
        # Combine using ensemble method
        predictions_array = np.array([result.predictions for result in individual_results])
        scores_array = np.array([result.confidence_scores for result in individual_results if result.confidence_scores is not None])
        
        if method == 'majority':
            ensemble_predictions = ensemble_service.majority_vote(predictions_array)
        elif method in ['average', 'weighted_average', 'max'] and len(scores_array) > 0:
            if method == 'average':
                ensemble_predictions, _ = ensemble_service.average_combination(predictions_array, scores_array)
            elif method == 'max':
                ensemble_predictions, _ = ensemble_service.max_combination(predictions_array, scores_array)
            else:  # weighted_average
                weights = np.ones(len(algorithms)) / len(algorithms)
                ensemble_predictions, _ = ensemble_service.weighted_combination(
                    predictions_array, scores_array, weights
                )
        else:
            ensemble_predictions = ensemble_service.majority_vote(predictions_array)
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        # Calculate statistics
        anomaly_count = int(np.sum(ensemble_predictions == -1))
        total_samples = len(ensemble_predictions)
        anomaly_rate = anomaly_count / total_samples if total_samples > 0 else 0.0
        anomaly_indices = np.where(ensemble_predictions == -1)[0].tolist()
        
        # Prepare results
        results = {
            "success": True,
            "method": method,
            "algorithms": algorithms,
            "total_samples": total_samples,
            "anomalies_detected": anomaly_count,
            "anomaly_rate": f"{anomaly_rate:.1%}",
            "processing_time_ms": f"{processing_time:.2f}",
            "timestamp": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "anomaly_indices": anomaly_indices[:10],  # Show first 10
            "individual_results": [
                {
                    "algorithm": alg,
                    "anomalies": result.anomaly_count,
                    "rate": f"{result.anomaly_rate:.1%}"
                }
                for alg, result in zip(algorithms, individual_results)
            ]
        }
        
        return templates.TemplateResponse(
            "components/ensemble_results.html",
            {"request": request, "results": results}
        )
        
    except Exception as e:
        logger.error("Ensemble detection failed", error=str(e))
        error_context = {
            "request": request,
            "error": str(e),
            "method": method
        }
        return templates.TemplateResponse(
            "components/error_message.html",
            error_context,
            status_code=500
        )


@router.get("/models/list", response_class=HTMLResponse)
async def list_models_htmx(
    request: Request,
    model_repository: ModelRepository = Depends(get_model_repository)
):
    """Get updated model list."""
    try:
        models = model_repository.list_models()
        return templates.TemplateResponse(
            "components/model_list.html",
            {"request": request, "models": models}
        )
    except Exception as e:
        logger.error("Error loading models", error=str(e))
        return templates.TemplateResponse(
            "components/error_message.html",
            {"request": request, "error": str(e)},
            status_code=500
        )


@router.get("/models/{model_id}/info", response_class=HTMLResponse)
async def get_model_info_htmx(
    request: Request,
    model_id: str,
    model_repository: ModelRepository = Depends(get_model_repository)
):
    """Get model information."""
    try:
        metadata = model_repository.get_model_metadata(model_id)
        return templates.TemplateResponse(
            "components/model_info.html",
            {"request": request, "model": metadata}
        )
    except FileNotFoundError:
        return templates.TemplateResponse(
            "components/error_message.html",
            {"request": request, "error": f"Model {model_id} not found"},
            status_code=404
        )
    except Exception as e:
        logger.error("Error getting model info", model_id=model_id, error=str(e))
        return templates.TemplateResponse(
            "components/error_message.html",
            {"request": request, "error": str(e)},
            status_code=500
        )


@router.get("/dashboard/stats", response_class=HTMLResponse)
async def get_dashboard_stats(
    request: Request,
    model_repository: ModelRepository = Depends(get_model_repository)
):
    """Get updated dashboard statistics."""
    try:
        models = model_repository.list_models()
        
        stats = {
            "total_models": len(models),
            "active_models": len([m for m in models if m.get('status') == 'trained']),
            "recent_detections": 156,  # Mock data
            "anomalies_found": 23,     # Mock data
            "last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return templates.TemplateResponse(
            "components/dashboard_stats.html",
            {"request": request, "stats": stats}
        )
        
    except Exception as e:
        logger.error("Error getting dashboard stats", error=str(e))
        return templates.TemplateResponse(
            "components/error_message.html",
            {"request": request, "error": str(e)},
            status_code=500
        )