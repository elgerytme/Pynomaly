"""HTMX endpoints for dynamic content."""

import json
import numpy as np
import pandas as pd
import uuid
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ...domain.services.detection_service import DetectionService
from ...domain.services.ensemble_service import EnsembleService
from ...domain.services.streaming_service import StreamingService
from ...domain.services.explainability_service import ExplainabilityService, ExplainerType
from ...infrastructure.repositories.model_repository import ModelRepository
from ...domain.entities.model import Model, ModelMetadata, ModelStatus, SerializationFormat
from ...domain.entities.dataset import Dataset, DatasetType, DatasetMetadata
from ...infrastructure.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# Dependency injection
_detection_service: DetectionService = None
_ensemble_service: EnsembleService = None
_streaming_service: StreamingService = None
_explainability_service: ExplainabilityService = None
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


def get_streaming_service() -> StreamingService:
    """Get streaming service instance."""
    global _streaming_service
    if _streaming_service is None:
        _streaming_service = StreamingService()
    return _streaming_service


def get_explainability_service() -> ExplainabilityService:
    """Get explainability service instance."""
    global _explainability_service, _detection_service
    if _explainability_service is None:
        if _detection_service is None:
            _detection_service = DetectionService()
        _explainability_service = ExplainabilityService(_detection_service)
    return _explainability_service


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


@router.post("/train", response_class=HTMLResponse)
async def train_model_htmx(
    request: Request,
    model_name: str = Form(...),
    algorithm: str = Form("isolation_forest"),
    contamination: float = Form(0.1),
    training_data: str = Form(""),
    has_labels: bool = Form(False),
    detection_service: DetectionService = Depends(get_detection_service),
    model_repository: ModelRepository = Depends(get_model_repository)
):
    """Train a new model via web interface."""
    try:
        start_time = datetime.utcnow()
        
        # Parse training data
        if not training_data.strip():
            # Generate sample training data
            np.random.seed(42)
            normal_data = np.random.normal(0, 1, (200, 5))
            anomaly_data = np.random.normal(3, 1, (20, 5))
            data_array = np.vstack([normal_data, anomaly_data])
            labels_array = np.array([1] * 200 + [-1] * 20) if has_labels else None
        else:
            try:
                data_parsed = json.loads(training_data)
                if isinstance(data_parsed, dict) and 'data' in data_parsed:
                    data_array = np.array(data_parsed['data'], dtype=np.float64)
                    labels_array = np.array(data_parsed.get('labels')) if has_labels and 'labels' in data_parsed else None
                else:
                    data_array = np.array(data_parsed, dtype=np.float64)
                    labels_array = None
            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(status_code=400, detail=f"Invalid training data format: {str(e)}")
        
        # Validate data
        if len(data_array.shape) != 2:
            raise HTTPException(status_code=400, detail="Training data must be a 2D array")
        
        # Create dataset entity
        df = pd.DataFrame(data_array, columns=[f"feature_{i}" for i in range(data_array.shape[1])])
        dataset = Dataset(
            data=df,
            dataset_type=DatasetType.TRAINING,
            labels=labels_array,
            metadata=DatasetMetadata(
                name=f"{model_name}_training_data",
                source="web_interface",
                description=f"Training dataset for {model_name} via web interface"
            )
        )
        
        # Algorithm mapping
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'lof': 'lof'
        }
        
        mapped_algorithm = algorithm_map.get(algorithm, algorithm)
        
        # Fit the model
        detection_service.fit(data_array, mapped_algorithm, contamination=contamination)
        
        # Get predictions for evaluation
        detection_result = detection_service.detect_anomalies(
            data=data_array,
            algorithm=mapped_algorithm,
            contamination=contamination
        )
        
        end_time = datetime.utcnow()
        training_duration = (end_time - start_time).total_seconds()
        
        # Calculate metrics if labels available
        accuracy, precision, recall, f1_score_val = None, None, None, None
        if has_labels and labels_array is not None:
            pred_labels = detection_result.predictions
            accuracy = float(accuracy_score(labels_array, pred_labels))
            precision = float(precision_score(labels_array, pred_labels, pos_label=-1, zero_division=0, average='binary'))
            recall = float(recall_score(labels_array, pred_labels, pos_label=-1, zero_division=0, average='binary'))
            f1_score_val = float(f1_score(labels_array, pred_labels, pos_label=-1, zero_division=0, average='binary'))
        
        # Create and save model
        model_id = str(uuid.uuid4())
        metadata = ModelMetadata(
            model_id=model_id,
            name=model_name,
            algorithm=algorithm,
            status=ModelStatus.TRAINED,
            training_samples=dataset.n_samples,
            training_features=dataset.n_features,
            contamination_rate=contamination,
            training_duration_seconds=training_duration,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score_val,
            feature_names=dataset.feature_names,
            hyperparameters={'contamination': contamination},
            description=f"Trained {algorithm} model via web interface",
        )
        
        # Get the trained model object from the service
        trained_model_obj = detection_service._fitted_models.get(mapped_algorithm)
        
        model = Model(
            metadata=metadata,
            model_object=trained_model_obj
        )
        
        # Save model
        saved_model_id = model_repository.save(model, SerializationFormat.PICKLE)
        
        # Prepare results
        results = {
            "success": True,
            "model_id": saved_model_id,
            "model_name": model_name,
            "algorithm": algorithm,
            "training_samples": dataset.n_samples,
            "training_features": dataset.n_features,
            "contamination": contamination,
            "training_duration": f"{training_duration:.2f}",
            "accuracy": f"{accuracy:.3f}" if accuracy is not None else "N/A",
            "precision": f"{precision:.3f}" if precision is not None else "N/A",
            "recall": f"{recall:.3f}" if recall is not None else "N/A",
            "f1_score": f"{f1_score_val:.3f}" if f1_score_val is not None else "N/A",
            "timestamp": end_time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return templates.TemplateResponse(
            "components/training_results.html",
            {"request": request, "results": results}
        )
        
    except Exception as e:
        logger.error("Model training failed", error=str(e))
        error_context = {
            "request": request,
            "error": str(e),
            "model_name": model_name,
            "algorithm": algorithm
        }
        return templates.TemplateResponse(
            "components/error_message.html",
            error_context,
            status_code=500
        )


@router.post("/streaming/start", response_class=HTMLResponse)
async def start_streaming_monitor(
    request: Request,
    algorithm: str = Form("isolation_forest"),
    window_size: int = Form(1000),
    update_frequency: int = Form(100),
    streaming_service: StreamingService = Depends(get_streaming_service)
):
    """Start streaming detection monitoring."""
    try:
        # Reset the streaming service
        streaming_service.reset_stream()
        streaming_service.set_window_size(window_size)
        streaming_service.set_update_frequency(update_frequency)
        
        # Initialize with sample data
        np.random.seed(42)
        initial_data = np.random.normal(0, 1, (50, 5))
        
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'lof': 'lof'
        }
        mapped_algorithm = algorithm_map.get(algorithm, algorithm)
        
        # Process initial batch to fit model
        streaming_service.process_batch(initial_data, mapped_algorithm)
        
        # Get initial stats
        stats = streaming_service.get_streaming_stats()
        
        context = {
            "request": request,
            "status": "started",
            "algorithm": algorithm,
            "window_size": window_size,
            "update_frequency": update_frequency,
            "stats": stats
        }
        
        return templates.TemplateResponse(
            "components/streaming_status.html",
            context
        )
        
    except Exception as e:
        logger.error("Streaming start error", error=str(e))
        return templates.TemplateResponse(
            "components/error_message.html",
            {"request": request, "error": str(e)},
            status_code=500
        )


@router.post("/streaming/sample", response_class=HTMLResponse)
async def process_streaming_sample(
    request: Request,
    sample_data: str = Form(""),
    algorithm: str = Form("isolation_forest"),
    streaming_service: StreamingService = Depends(get_streaming_service)
):
    """Process a single streaming sample."""
    try:
        # Parse or generate sample data
        if not sample_data.strip():
            # Generate random sample (with 20% chance of anomaly)
            if np.random.random() < 0.2:
                sample = np.random.normal(3, 1, 5)  # Anomalous sample
            else:
                sample = np.random.normal(0, 1, 5)  # Normal sample
        else:
            try:
                sample = np.array(json.loads(sample_data), dtype=np.float64)
            except (json.JSONDecodeError, ValueError):
                raise HTTPException(status_code=400, detail="Invalid sample data format")
        
        # Algorithm mapping
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'lof': 'lof'
        }
        mapped_algorithm = algorithm_map.get(algorithm, algorithm)
        
        # Process sample
        result = streaming_service.process_sample(sample, mapped_algorithm)
        stats = streaming_service.get_streaming_stats()
        
        # Prepare response
        context = {
            "request": request,
            "success": result.success,
            "is_anomaly": bool(result.predictions[0] == -1),
            "confidence_score": float(result.confidence_scores[0]) if result.confidence_scores is not None else None,
            "algorithm": algorithm,
            "sample_data": sample.tolist(),
            "timestamp": datetime.utcnow().strftime("%H:%M:%S"),
            "buffer_size": stats['buffer_size'],
            "model_fitted": stats['model_fitted'],
            "samples_processed": stats['total_samples']
        }
        
        return templates.TemplateResponse(
            "components/streaming_result.html",
            context
        )
        
    except Exception as e:
        logger.error("Streaming sample processing error", error=str(e))
        return templates.TemplateResponse(
            "components/error_message.html",
            {"request": request, "error": str(e)},
            status_code=500
        )


@router.get("/streaming/stats", response_class=HTMLResponse)
async def get_streaming_stats(
    request: Request,
    streaming_service: StreamingService = Depends(get_streaming_service)
):
    """Get current streaming statistics."""
    try:
        stats = streaming_service.get_streaming_stats()
        
        # Check for concept drift
        drift_result = streaming_service.detect_concept_drift()
        
        context = {
            "request": request,
            "stats": stats,
            "drift": drift_result,
            "timestamp": datetime.utcnow().strftime("%H:%M:%S")
        }
        
        return templates.TemplateResponse(
            "components/streaming_stats.html",
            context
        )
        
    except Exception as e:
        logger.error("Error getting streaming stats", error=str(e))
        return templates.TemplateResponse(
            "components/error_message.html",
            {"request": request, "error": str(e)},
            status_code=500
        )


@router.post("/streaming/reset", response_class=HTMLResponse)
async def reset_streaming(
    request: Request,
    streaming_service: StreamingService = Depends(get_streaming_service)
):
    """Reset streaming service state."""
    try:
        streaming_service.reset_stream()
        
        context = {
            "request": request,
            "message": "Streaming service reset successfully",
            "timestamp": datetime.utcnow().strftime("%H:%M:%S")
        }
        
        return templates.TemplateResponse(
            "components/streaming_reset.html",
            context
        )
        
    except Exception as e:
        logger.error("Streaming reset error", error=str(e))
        return templates.TemplateResponse(
            "components/error_message.html",
            {"request": request, "error": str(e)},
            status_code=500
        )


@router.post("/explain", response_class=HTMLResponse)
async def explain_prediction_htmx(
    request: Request,
    sample_data: str = Form(...),
    algorithm: str = Form("isolation_forest"),
    explainer_type: str = Form("feature_importance"),
    feature_names: str = Form(""),
    detection_service: DetectionService = Depends(get_detection_service),
    explainability_service: ExplainabilityService = Depends(get_explainability_service)
):
    """Explain a prediction via web interface."""
    try:
        # Parse sample data
        try:
            sample_array = np.array(json.loads(sample_data), dtype=np.float64)
        except (json.JSONDecodeError, ValueError):
            raise HTTPException(status_code=400, detail="Invalid sample data format")
        
        # Parse feature names
        if feature_names.strip():
            try:
                feature_names_list = json.loads(feature_names)
            except (json.JSONDecodeError, ValueError):
                feature_names_list = feature_names.split(',')
        else:
            feature_names_list = [f"feature_{i}" for i in range(len(sample_array))]
        
        # Algorithm mapping
        algorithm_map = {
            'isolation_forest': 'iforest',
            'one_class_svm': 'ocsvm',
            'lof': 'lof'
        }
        mapped_algorithm = algorithm_map.get(algorithm, algorithm)
        
        # Fit model if not already fitted
        if mapped_algorithm not in detection_service._fitted_models:
            # Generate training data for demo
            np.random.seed(42)
            training_data = np.random.normal(0, 1, (100, len(sample_array)))
            detection_service.fit(training_data, mapped_algorithm)
        
        # Map explainer type
        explainer_type_map = {
            'shap': ExplainerType.SHAP,
            'lime': ExplainerType.LIME,
            'permutation': ExplainerType.PERMUTATION,
            'feature_importance': ExplainerType.FEATURE_IMPORTANCE
        }
        explainer_enum = explainer_type_map.get(explainer_type, ExplainerType.FEATURE_IMPORTANCE)
        
        # Generate explanation
        explanation = explainability_service.explain_prediction(
            sample=sample_array,
            algorithm=mapped_algorithm,
            explainer_type=explainer_enum,
            feature_names=feature_names_list
        )
        
        # Prepare context
        context = {
            "request": request,
            "success": True,
            "algorithm": algorithm,
            "explainer_type": explainer_type,
            "is_anomaly": explanation.is_anomaly,
            "confidence": explanation.prediction_confidence,
            "base_value": explanation.base_value,
            "top_features": explanation.top_features,
            "feature_importance": explanation.feature_importance,
            "sample_data": dict(zip(feature_names_list, explanation.data_sample)),
            "timestamp": datetime.utcnow().strftime("%H:%M:%S")
        }
        
        return templates.TemplateResponse(
            "components/explanation_result.html",
            context
        )
        
    except Exception as e:
        logger.error("Explanation failed", error=str(e))
        return templates.TemplateResponse(
            "components/error_message.html",
            {"request": request, "error": str(e)},
            status_code=500
        )


@router.get("/explain/available", response_class=HTMLResponse)
async def get_available_explainers_htmx(
    request: Request,
    explainability_service: ExplainabilityService = Depends(get_explainability_service)
):
    """Get available explainer types for web interface."""
    try:
        available_explainers = explainability_service.get_available_explainers()
        
        descriptions = {
            'shap': 'SHAP - Advanced model explanations',
            'lime': 'LIME - Local linear approximations', 
            'permutation': 'Permutation - Feature importance via testing',
            'feature_importance': 'Simple - Based on feature magnitude'
        }
        
        explainers_info = []
        for explainer in available_explainers:
            explainers_info.append({
                'name': explainer,
                'description': descriptions.get(explainer, 'Feature importance method'),
                'available': True
            })
        
        context = {
            "request": request,
            "explainers": explainers_info,
            "timestamp": datetime.utcnow().strftime("%H:%M:%S")
        }
        
        return templates.TemplateResponse(
            "components/explainers_list.html",
            context
        )
        
    except Exception as e:
        logger.error("Error getting available explainers", error=str(e))
        return templates.TemplateResponse(
            "components/error_message.html",
            {"request": request, "error": str(e)},
            status_code=500
        )


# Worker Management HTMX Endpoints

# Global worker instance (in production, use proper dependency injection)
_worker_instance = None

def get_worker_instance():
    """Get or create worker instance."""
    global _worker_instance
    if _worker_instance is None:
        from ...worker import AnomalyDetectionWorker
        _worker_instance = AnomalyDetectionWorker(
            models_dir="./models",
            max_concurrent_jobs=5,
            enable_monitoring=True
        )
        logger.info("Created new worker instance for web interface")
    return _worker_instance


@router.post("/worker/submit", response_class=HTMLResponse)
async def submit_worker_job_htmx(
    request: Request,
    job_type: str = Form(...),
    data_source: str = Form(""),
    algorithm: str = Form("isolation_forest"),
    contamination: float = Form(0.1),
    priority: str = Form("normal"),
    model_name: str = Form(""),
    output_path: str = Form("")
):
    """Submit a job to the background worker via web interface."""
    try:
        from ...worker import JobType, JobPriority
        
        # Map string values to enums
        job_type_map = {
            'detection': JobType.DETECTION,
            'ensemble': JobType.ENSEMBLE,
            'batch_training': JobType.BATCH_TRAINING,
            'stream_monitoring': JobType.STREAM_MONITORING,
            'model_validation': JobType.MODEL_VALIDATION,
            'data_preprocessing': JobType.DATA_PREPROCESSING,
            'explanation_generation': JobType.EXPLANATION_GENERATION,
            'scheduled_analysis': JobType.SCHEDULED_ANALYSIS
        }
        
        priority_map = {
            'low': JobPriority.LOW,
            'normal': JobPriority.NORMAL,
            'high': JobPriority.HIGH,
            'critical': JobPriority.CRITICAL
        }
        
        if job_type not in job_type_map:
            raise HTTPException(status_code=400, detail=f"Invalid job type: {job_type}")
        
        if priority not in priority_map:
            raise HTTPException(status_code=400, detail=f"Invalid priority: {priority}")
        
        # Build job payload
        payload = {
            'algorithm': algorithm,
            'contamination': contamination
        }
        
        if data_source.strip():
            # Try to parse as JSON data first, then treat as file path
            try:
                data_list = json.loads(data_source)
                payload['data_source'] = data_list
            except json.JSONDecodeError:
                payload['data_source'] = data_source  # Treat as file path
        else:
            # Generate sample data for demo
            np.random.seed(42)
            sample_data = np.random.randn(100, 5).tolist()
            payload['data_source'] = sample_data
        
        if model_name.strip():
            payload['model_name'] = model_name
        
        if output_path.strip():
            payload['output_path'] = output_path
        
        # Get worker instance and submit job
        worker = get_worker_instance()
        job_id = await worker.submit_job(
            job_type_map[job_type],
            payload,
            priority=priority_map[priority]
        )
        
        # Start worker if not running
        if not worker.is_running:
            import asyncio
            asyncio.create_task(worker.start())
        
        # Get queue status for position estimation
        queue_status = await worker.job_queue.get_queue_status()
        queue_position = queue_status.get('pending_jobs', 0)
        
        context = {
            "request": request,
            "success": True,
            "job_id": job_id,
            "job_type": job_type,
            "priority": priority,
            "algorithm": algorithm,
            "queue_position": queue_position,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return templates.TemplateResponse(
            "components/worker_job_submitted.html",
            context
        )
        
    except Exception as e:
        logger.error("Worker job submission failed", error=str(e))
        return templates.TemplateResponse(
            "components/error_message.html",
            {"request": request, "error": str(e)},
            status_code=500
        )


@router.get("/worker/job/{job_id}/status", response_class=HTMLResponse)
async def get_worker_job_status_htmx(
    request: Request,
    job_id: str
):
    """Get status of a specific worker job."""
    try:
        worker = get_worker_instance()
        job_status = await worker.get_job_status(job_id)
        
        if not job_status:
            return templates.TemplateResponse(
                "components/error_message.html",
                {"request": request, "error": f"Job {job_id} not found"},
                status_code=404
            )
        
        context = {
            "request": request,
            "job": job_status,
            "timestamp": datetime.utcnow().strftime("%H:%M:%S")
        }
        
        return templates.TemplateResponse(
            "components/worker_job_status.html",
            context
        )
        
    except Exception as e:
        logger.error("Failed to get worker job status", job_id=job_id, error=str(e))
        return templates.TemplateResponse(
            "components/error_message.html",
            {"request": request, "error": str(e)},
            status_code=500
        )


@router.get("/worker/dashboard", response_class=HTMLResponse)
async def get_worker_dashboard_htmx(
    request: Request
):
    """Get worker dashboard information."""
    try:
        worker = get_worker_instance()
        worker_status = await worker.get_worker_status()
        queue_status = worker_status['queue_status']
        
        # Calculate health metrics
        current_jobs = worker_status['currently_running_jobs']
        max_jobs = worker_status['max_concurrent_jobs']
        utilization = (current_jobs / max_jobs) * 100 if max_jobs > 0 else 0
        
        pending_jobs = queue_status.get('pending_jobs', 0)
        total_jobs = queue_status.get('total_jobs', 0)
        
        # Determine health status
        health_status = "healthy"
        if utilization > 90 or pending_jobs > 50:
            health_status = "critical"
        elif utilization > 70 or pending_jobs > 20:
            health_status = "warning"
        
        context = {
            "request": request,
            "worker_status": worker_status,
            "utilization": utilization,
            "health_status": health_status,
            "pending_jobs": pending_jobs,
            "total_jobs": total_jobs,
            "timestamp": datetime.utcnow().strftime("%H:%M:%S")
        }
        
        return templates.TemplateResponse(
            "components/worker_dashboard.html",
            context
        )
        
    except Exception as e:
        logger.error("Failed to get worker dashboard", error=str(e))
        return templates.TemplateResponse(
            "components/error_message.html",
            {"request": request, "error": str(e)},
            status_code=500
        )


@router.post("/worker/job/{job_id}/cancel", response_class=HTMLResponse)
async def cancel_worker_job_htmx(
    request: Request,
    job_id: str
):
    """Cancel a pending or running worker job."""
    try:
        worker = get_worker_instance()
        success = await worker.cancel_job(job_id)
        
        if success:
            context = {
                "request": request,
                "success": True,
                "job_id": job_id,
                "message": f"Job {job_id} cancelled successfully",
                "timestamp": datetime.utcnow().strftime("%H:%M:%S")
            }
        else:
            context = {
                "request": request,
                "success": False,
                "job_id": job_id,
                "message": f"Job {job_id} not found or cannot be cancelled",
                "timestamp": datetime.utcnow().strftime("%H:%M:%S")
            }
        
        return templates.TemplateResponse(
            "components/worker_job_cancelled.html",
            context
        )
        
    except Exception as e:
        logger.error("Failed to cancel worker job", job_id=job_id, error=str(e))
        return templates.TemplateResponse(
            "components/error_message.html",
            {"request": request, "error": str(e)},
            status_code=500
        )


@router.get("/worker/jobs", response_class=HTMLResponse)
async def list_worker_jobs_htmx(
    request: Request,
    status_filter: str = "",
    job_type_filter: str = "",
    limit: int = 20
):
    """List worker jobs with optional filtering."""
    try:
        worker = get_worker_instance()
        worker_status = await worker.get_worker_status()
        queue_status = worker_status['queue_status']
        
        # Get currently running jobs
        jobs_list = []
        for job_id in worker_status['running_job_ids']:
            job_status = await worker.get_job_status(job_id)
            if job_status:
                # Apply filters
                if status_filter and job_status['status'] != status_filter:
                    continue
                if job_type_filter and job_status['job_type'] != job_type_filter:
                    continue
                
                jobs_list.append(job_status)
        
        # Limit results
        jobs_list = jobs_list[:limit]
        
        context = {
            "request": request,
            "jobs": jobs_list,
            "total_jobs": queue_status.get('total_jobs', 0),
            "pending_jobs": queue_status.get('pending_jobs', 0),
            "filtered_count": len(jobs_list),
            "status_filter": status_filter,
            "job_type_filter": job_type_filter,
            "timestamp": datetime.utcnow().strftime("%H:%M:%S")
        }
        
        return templates.TemplateResponse(
            "components/worker_jobs_list.html",
            context
        )
        
    except Exception as e:
        logger.error("Failed to list worker jobs", error=str(e))
        return templates.TemplateResponse(
            "components/error_message.html",
            {"request": request, "error": str(e)},
            status_code=500
        )


@router.get("/worker/health", response_class=HTMLResponse)
async def get_worker_health_htmx(
    request: Request
):
    """Get worker health metrics for web interface."""
    try:
        worker = get_worker_instance()
        worker_status = await worker.get_worker_status()
        
        # Calculate health score
        health_factors = []
        
        # Worker running status
        health_factors.append(100 if worker_status['is_running'] else 0)
        
        # Resource utilization
        current_jobs = worker_status['currently_running_jobs']
        max_jobs = worker_status['max_concurrent_jobs']
        utilization = (current_jobs / max_jobs) * 100 if max_jobs > 0 else 0
        
        if utilization < 50:
            health_factors.append(100)
        elif utilization < 80:
            health_factors.append(80)
        else:
            health_factors.append(60)
        
        # Queue health
        queue_status = worker_status['queue_status']
        pending_jobs = queue_status.get('pending_jobs', 0)
        
        if pending_jobs < 10:
            health_factors.append(100)
        elif pending_jobs < 50:
            health_factors.append(80)
        else:
            health_factors.append(60)
        
        # Calculate overall health score
        health_score = sum(health_factors) / len(health_factors)
        is_healthy = health_score >= 70
        
        # Mock resource utilization and performance metrics
        resource_utilization = {
            "cpu_usage_percent": 25.3,
            "memory_usage_mb": 1024,
            "memory_usage_percent": 15.2,
            "disk_usage_percent": 45.8,
            "worker_utilization_percent": utilization
        }
        
        performance_metrics = {
            "jobs_completed_24h": 156,
            "jobs_failed_24h": 8,
            "success_rate_percent": 95.1,
            "avg_processing_time_seconds": 42.3,
            "throughput_jobs_per_hour": 12.8
        }
        
        # Recent issues
        recent_issues = []
        if utilization > 80:
            recent_issues.append({
                "severity": "warning",
                "message": "High worker utilization detected",
                "timestamp": datetime.utcnow().strftime("%H:%M:%S")
            })
        
        if pending_jobs > 20:
            recent_issues.append({
                "severity": "info",
                "message": "High queue length detected",
                "timestamp": datetime.utcnow().strftime("%H:%M:%S")
            })
        
        context = {
            "request": request,
            "is_healthy": is_healthy,
            "health_score": health_score,
            "resource_utilization": resource_utilization,
            "performance_metrics": performance_metrics,
            "recent_issues": recent_issues,
            "worker_status": worker_status,
            "timestamp": datetime.utcnow().strftime("%H:%M:%S")
        }
        
        return templates.TemplateResponse(
            "components/worker_health.html",
            context
        )
        
    except Exception as e:
        logger.error("Failed to get worker health", error=str(e))
        return templates.TemplateResponse(
            "components/error_message.html",
            {"request": request, "error": str(e)},
            status_code=500
        )


@router.post("/worker/start", response_class=HTMLResponse)
async def start_worker_htmx(
    request: Request
):
    """Start the worker service via web interface."""
    try:
        worker = get_worker_instance()
        
        if worker.is_running:
            context = {
                "request": request,
                "success": True,
                "message": "Worker is already running",
                "timestamp": datetime.utcnow().strftime("%H:%M:%S")
            }
        else:
            # Start worker in background
            import asyncio
            asyncio.create_task(worker.start())
            
            context = {
                "request": request,
                "success": True,
                "message": "Worker start initiated",
                "timestamp": datetime.utcnow().strftime("%H:%M:%S")
            }
        
        return templates.TemplateResponse(
            "components/worker_control_result.html",
            context
        )
        
    except Exception as e:
        logger.error("Failed to start worker", error=str(e))
        return templates.TemplateResponse(
            "components/error_message.html",
            {"request": request, "error": str(e)},
            status_code=500
        )


@router.post("/worker/stop", response_class=HTMLResponse)
async def stop_worker_htmx(
    request: Request
):
    """Stop the worker service via web interface."""
    try:
        worker = get_worker_instance()
        
        if not worker.is_running:
            context = {
                "request": request,
                "success": True,
                "message": "Worker is not running",
                "timestamp": datetime.utcnow().strftime("%H:%M:%S")
            }
        else:
            await worker.stop()
            
            context = {
                "request": request,
                "success": True,
                "message": "Worker stopped successfully",
                "timestamp": datetime.utcnow().strftime("%H:%M:%S")
            }
        
        return templates.TemplateResponse(
            "components/worker_control_result.html",
            context
        )
        
    except Exception as e:
        logger.error("Failed to stop worker", error=str(e))
        return templates.TemplateResponse(
            "components/error_message.html",
            {"request": request, "error": str(e)},
            status_code=500
        )