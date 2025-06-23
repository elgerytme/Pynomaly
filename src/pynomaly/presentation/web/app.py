"""Web UI application with HTMX."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pynomaly.infrastructure.config import Container
from pynomaly.presentation.api.deps import get_container, get_current_user


# Get template and static directories
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Create directories if they don't exist
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
(STATIC_DIR / "css").mkdir(exist_ok=True)
(STATIC_DIR / "js").mkdir(exist_ok=True)
(STATIC_DIR / "img").mkdir(exist_ok=True)

# Initialize templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Create router
router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    container: Container = Depends(get_container),
    current_user: Optional[str] = Depends(get_current_user)
):
    """Main dashboard page."""
    settings = container.config()
    
    # Check if auth is enabled and user is not authenticated
    if settings.auth_enabled and not current_user:
        return templates.TemplateResponse(
            "login.html",
            {"request": request}
        )
    
    # Get counts for dashboard
    detector_count = container.detector_repository().count()
    dataset_count = container.dataset_repository().count()
    result_count = container.result_repository().count()
    
    # Get recent results
    recent_results = container.result_repository().find_recent(5)
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "current_user": current_user,
            "auth_enabled": settings.auth_enabled,
            "detector_count": detector_count,
            "dataset_count": dataset_count,
            "result_count": result_count,
            "recent_results": recent_results
        }
    )


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page."""
    return templates.TemplateResponse(
        "login.html",
        {"request": request}
    )


@router.post("/login", response_class=HTMLResponse)
async def login_post(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    container: Container = Depends(get_container)
):
    """Handle login form submission."""
    settings = container.config()
    
    if not settings.auth_enabled:
        return RedirectResponse(url="/web/", status_code=302)
    
    try:
        from pynomaly.infrastructure.auth import get_auth
        from pynomaly.domain.exceptions import AuthenticationError
        
        auth_service = get_auth()
        if not auth_service:
            raise HTTPException(status_code=503, detail="Authentication service not available")
        
        # Authenticate user
        user = auth_service.authenticate_user(username, password)
        token_response = auth_service.create_access_token(user)
        
        # Create redirect response with token as cookie
        response = RedirectResponse(url="/web/", status_code=302)
        response.set_cookie(
            key="access_token",
            value=f"Bearer {token_response.access_token}",
            max_age=token_response.expires_in,
            httponly=True,
            secure=settings.is_production,
            samesite="lax"
        )
        
        return response
        
    except AuthenticationError as e:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": str(e)
            }
        )


@router.post("/logout")
async def logout():
    """Handle logout."""
    response = RedirectResponse(url="/web/login", status_code=302)
    response.delete_cookie("access_token")
    return response


@router.get("/detectors", response_class=HTMLResponse)
async def detectors_page(
    request: Request,
    container: Container = Depends(get_container)
):
    """Detectors management page."""
    detectors = container.detector_repository().find_all()
    pyod_adapter = container.pyod_adapter()
    algorithms = pyod_adapter.list_algorithms()
    
    return templates.TemplateResponse(
        "detectors.html",
        {
            "request": request,
            "detectors": detectors,
            "algorithms": algorithms
        }
    )


@router.get("/detectors/{detector_id}", response_class=HTMLResponse)
async def detector_detail(
    request: Request,
    detector_id: str,
    container: Container = Depends(get_container)
):
    """Detector detail page."""
    from uuid import UUID
    
    detector = container.detector_repository().find_by_id(UUID(detector_id))
    if not detector:
        return templates.TemplateResponse(
            "404.html",
            {"request": request, "message": "Detector not found"},
            status_code=404
        )
    
    # Get detection results for this detector
    results = container.result_repository().find_by_detector(detector.id)
    
    return templates.TemplateResponse(
        "detector_detail.html",
        {
            "request": request,
            "detector": detector,
            "results": results[:10]  # Last 10 results
        }
    )


@router.get("/datasets", response_class=HTMLResponse)
async def datasets_page(
    request: Request,
    container: Container = Depends(get_container)
):
    """Datasets management page."""
    datasets = container.dataset_repository().find_all()
    
    return templates.TemplateResponse(
        "datasets.html",
        {
            "request": request,
            "datasets": datasets
        }
    )


@router.get("/datasets/{dataset_id}", response_class=HTMLResponse)
async def dataset_detail(
    request: Request,
    dataset_id: str,
    container: Container = Depends(get_container)
):
    """Dataset detail page."""
    from uuid import UUID
    
    dataset = container.dataset_repository().find_by_id(UUID(dataset_id))
    if not dataset:
        return templates.TemplateResponse(
            "404.html",
            {"request": request, "message": "Dataset not found"},
            status_code=404
        )
    
    # Get data quality report
    feature_validator = container.feature_validator()
    quality_report = feature_validator.check_data_quality(dataset)
    
    # Get sample data
    sample_data = dataset.data.head(10).to_dict(orient="records")
    
    return templates.TemplateResponse(
        "dataset_detail.html",
        {
            "request": request,
            "dataset": dataset,
            "quality_report": quality_report,
            "sample_data": sample_data,
            "columns": list(dataset.data.columns)
        }
    )


@router.get("/detection", response_class=HTMLResponse)
async def detection_page(
    request: Request,
    container: Container = Depends(get_container)
):
    """Detection page."""
    detectors = container.detector_repository().find_all()
    datasets = container.dataset_repository().find_all()
    
    # Filter trained detectors
    trained_detectors = [d for d in detectors if d.is_fitted]
    
    return templates.TemplateResponse(
        "detection.html",
        {
            "request": request,
            "detectors": detectors,
            "trained_detectors": trained_detectors,
            "datasets": datasets
        }
    )


@router.get("/experiments", response_class=HTMLResponse)
async def experiments_page(
    request: Request,
    container: Container = Depends(get_container)
):
    """Experiments tracking page."""
    experiment_service = container.experiment_tracking_service()
    
    # Load experiments
    experiment_service._load_experiments()
    experiments = []
    
    for exp_id, exp_data in experiment_service.experiments.items():
        experiments.append({
            "id": exp_id,
            "name": exp_data["name"],
            "description": exp_data.get("description", ""),
            "created_at": exp_data["created_at"],
            "run_count": len(exp_data.get("runs", []))
        })
    
    # Sort by creation date
    experiments.sort(key=lambda e: e["created_at"], reverse=True)
    
    return templates.TemplateResponse(
        "experiments.html",
        {
            "request": request,
            "experiments": experiments
        }
    )


@router.get("/visualizations", response_class=HTMLResponse)
async def visualizations_page(
    request: Request,
    container: Container = Depends(get_container)
):
    """Visualizations page with D3.js and ECharts."""
    # Get recent results for visualization
    results = container.result_repository().find_recent(20)
    
    # Prepare data for charts
    detection_timeline = []
    anomaly_rates = []
    
    for result in results:
        detection_timeline.append({
            "timestamp": result.timestamp.isoformat(),
            "anomalies": result.n_anomalies,
            "samples": result.n_samples
        })
        anomaly_rates.append({
            "detector_id": str(result.detector_id),
            "rate": result.anomaly_rate
        })
    
    return templates.TemplateResponse(
        "visualizations.html",
        {
            "request": request,
            "detection_timeline": detection_timeline,
            "anomaly_rates": anomaly_rates
        }
    )


# HTMX endpoints for partial updates
@router.get("/htmx/detector-list", response_class=HTMLResponse)
async def htmx_detector_list(
    request: Request,
    container: Container = Depends(get_container)
):
    """HTMX endpoint for detector list."""
    detectors = container.detector_repository().find_all()
    
    return templates.TemplateResponse(
        "partials/detector_list.html",
        {
            "request": request,
            "detectors": detectors
        }
    )


@router.post("/htmx/detector-create", response_class=HTMLResponse)
async def htmx_detector_create(
    request: Request,
    container: Container = Depends(get_container)
):
    """HTMX endpoint for creating detector."""
    from pynomaly.domain.entities import Detector
    
    # Get form data
    form_data = await request.form()
    
    # Create detector
    detector = Detector(
        name=form_data["name"],
        algorithm=form_data["algorithm"],
        description=form_data.get("description", ""),
        parameters={"contamination": float(form_data.get("contamination", 0.1))}
    )
    
    # Save detector
    detector_repo = container.detector_repository()
    detector_repo.save(detector)
    
    # Return updated list
    detectors = detector_repo.find_all()
    
    return templates.TemplateResponse(
        "partials/detector_list.html",
        {
            "request": request,
            "detectors": detectors,
            "message": f"Detector '{detector.name}' created successfully"
        }
    )


@router.get("/htmx/dataset-list", response_class=HTMLResponse)
async def htmx_dataset_list(
    request: Request,
    container: Container = Depends(get_container)
):
    """HTMX endpoint for dataset list."""
    datasets = container.dataset_repository().find_all()
    
    return templates.TemplateResponse(
        "partials/dataset_list.html",
        {
            "request": request,
            "datasets": datasets
        }
    )


@router.get("/htmx/results-table", response_class=HTMLResponse)
async def htmx_results_table(
    request: Request,
    limit: int = 10,
    container: Container = Depends(get_container)
):
    """HTMX endpoint for results table."""
    results = container.result_repository().find_recent(limit)
    detector_repo = container.detector_repository()
    dataset_repo = container.dataset_repository()
    
    # Enrich results with names
    enriched_results = []
    for result in results:
        detector = detector_repo.find_by_id(result.detector_id)
        dataset = dataset_repo.find_by_id(result.dataset_id)
        
        enriched_results.append({
            "result": result,
            "detector_name": detector.name if detector else "Unknown",
            "dataset_name": dataset.name if dataset else "Unknown"
        })
    
    return templates.TemplateResponse(
        "partials/results_table.html",
        {
            "request": request,
            "results": enriched_results
        }
    )


@router.post("/htmx/train-detector", response_class=HTMLResponse)
async def htmx_train_detector(
    request: Request,
    container: Container = Depends(get_container)
):
    """HTMX endpoint for training detector."""
    from uuid import UUID
    from pynomaly.application.use_cases import TrainDetectorRequest
    
    # Get form data
    form_data = await request.form()
    detector_id = UUID(form_data["detector_id"])
    dataset_id = UUID(form_data["dataset_id"])
    
    # Get entities
    detector_repo = container.detector_repository()
    dataset_repo = container.dataset_repository()
    
    detector = detector_repo.find_by_id(detector_id)
    dataset = dataset_repo.find_by_id(dataset_id)
    
    if not detector or not dataset:
        return HTMLResponse(
            '<div class="alert alert-error">Invalid detector or dataset</div>'
        )
    
    # Train detector
    train_use_case = container.train_detector_use_case()
    
    try:
        request_obj = TrainDetectorRequest(
            detector_id=detector_id,
            dataset=dataset,
            validate_data=True,
            save_model=True
        )
        
        import asyncio
        response = await train_use_case.execute(request_obj)
        
        return HTMLResponse(
            f'<div class="alert alert-success">Training completed in {response.training_time_ms}ms</div>'
        )
        
    except Exception as e:
        return HTMLResponse(
            f'<div class="alert alert-error">Training failed: {str(e)}</div>'
        )


@router.post("/htmx/detect-anomalies", response_class=HTMLResponse)
async def htmx_detect_anomalies(
    request: Request,
    container: Container = Depends(get_container)
):
    """HTMX endpoint for anomaly detection."""
    from uuid import UUID
    from pynomaly.application.use_cases import DetectAnomaliesRequest
    
    # Get form data
    form_data = await request.form()
    detector_id = UUID(form_data["detector_id"])
    dataset_id = UUID(form_data["dataset_id"])
    
    # Get entities
    detector_repo = container.detector_repository()
    dataset_repo = container.dataset_repository()
    
    detector = detector_repo.find_by_id(detector_id)
    dataset = dataset_repo.find_by_id(dataset_id)
    
    if not detector or not dataset:
        return HTMLResponse(
            '<div class="alert alert-error">Invalid detector or dataset</div>'
        )
    
    if not detector.is_fitted:
        return HTMLResponse(
            '<div class="alert alert-error">Detector must be trained first</div>'
        )
    
    # Run detection
    detect_use_case = container.detect_anomalies_use_case()
    
    try:
        request_obj = DetectAnomaliesRequest(
            detector_id=detector_id,
            dataset=dataset,
            validate_features=True,
            save_results=True
        )
        
        response = await detect_use_case.execute(request_obj)
        result = response.result
        
        return templates.TemplateResponse(
            "partials/detection_result.html",
            {
                "request": request,
                "result": result,
                "detector_name": detector.name,
                "dataset_name": dataset.name
            }
        )
        
    except Exception as e:
        return HTMLResponse(
            f'<div class="alert alert-error">Detection failed: {str(e)}</div>'
        )


def mount_web_ui(app):
    """Mount web UI to FastAPI app."""
    # Mount static files
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    
    # Include web routes
    app.include_router(router, prefix="/web", tags=["Web UI"])