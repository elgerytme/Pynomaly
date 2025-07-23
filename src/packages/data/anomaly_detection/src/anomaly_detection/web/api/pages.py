"""Page routes for the web interface."""

from pathlib import Path
from typing import List, Dict, Any

from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ...domain.services.detection_service import DetectionService
from ...infrastructure.repositories.model_repository import ModelRepository
from ...infrastructure.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# Dependency injection
_detection_service: DetectionService = None
_model_repository: ModelRepository = None


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


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page."""
    return templates.TemplateResponse(
        "pages/home.html",
        {"request": request, "title": "Anomaly Detection Dashboard"}
    )


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    model_repository: ModelRepository = Depends(get_model_repository)
):
    """Main dashboard page."""
    try:
        # Get summary statistics
        models = model_repository.list_models()
        total_models = len(models)
        
        # Get recent activity (mock data for now)
        recent_detections = [
            {"id": "det_001", "algorithm": "isolation_forest", "anomalies": 15, "timestamp": "2024-01-20 10:30:00"},
            {"id": "det_002", "algorithm": "ensemble_majority", "anomalies": 23, "timestamp": "2024-01-20 09:15:00"},
            {"id": "det_003", "algorithm": "one_class_svm", "anomalies": 8, "timestamp": "2024-01-20 08:45:00"},
        ]
        
        context = {
            "request": request,
            "title": "Dashboard",
            "total_models": total_models,
            "recent_detections": recent_detections,
            "active_algorithms": ["isolation_forest", "one_class_svm", "lof", "ensemble"]
        }
        
        return templates.TemplateResponse("pages/dashboard.html", context)
        
    except Exception as e:
        logger.error("Error loading dashboard", error=str(e))
        return templates.TemplateResponse(
            "pages/500.html",
            {"request": request, "error": str(e)},
            status_code=500
        )


@router.get("/detection", response_class=HTMLResponse)
async def detection_page(request: Request):
    """Anomaly detection page."""
    algorithms = [
        {"value": "isolation_forest", "label": "Isolation Forest"},
        {"value": "one_class_svm", "label": "One-Class SVM"},
        {"value": "lof", "label": "Local Outlier Factor"},
    ]
    
    ensemble_methods = [
        {"value": "majority", "label": "Majority Vote"},
        {"value": "average", "label": "Average"},
        {"value": "weighted_average", "label": "Weighted Average"},
        {"value": "max", "label": "Maximum"},
    ]
    
    return templates.TemplateResponse(
        "pages/detection.html",
        {
            "request": request,
            "title": "Run Detection",
            "algorithms": algorithms,
            "ensemble_methods": ensemble_methods
        }
    )


@router.get("/models", response_class=HTMLResponse)
async def models_page(
    request: Request,
    model_repository: ModelRepository = Depends(get_model_repository)
):
    """Model management page."""
    try:
        models = model_repository.list_models()
        
        return templates.TemplateResponse(
            "pages/models.html",
            {
                "request": request,
                "title": "Model Management",
                "models": models
            }
        )
        
    except Exception as e:
        logger.error("Error loading models page", error=str(e))
        return templates.TemplateResponse(
            "pages/500.html",
            {"request": request, "error": str(e)},
            status_code=500
        )


@router.get("/monitoring", response_class=HTMLResponse)
async def monitoring_page(request: Request):
    """Monitoring and metrics page."""
    return templates.TemplateResponse(
        "pages/monitoring.html",
        {
            "request": request,
            "title": "System Monitoring"
        }
    )


@router.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    """About page."""
    return templates.TemplateResponse(
        "pages/about.html",
        {
            "request": request,
            "title": "About Anomaly Detection"
        }
    )