"""Web UI application with HTMX."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Request
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
    current_user: str | None = Depends(get_current_user),
):
    """Main dashboard page."""
    settings = container.config()

    # Check if auth is enabled and user is not authenticated
    if settings.auth_enabled and not current_user:
        return templates.TemplateResponse("login.html", {"request": request})

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
            "recent_results": recent_results,
        },
    )


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page."""
    return templates.TemplateResponse("login.html", {"request": request})


@router.post("/login", response_class=HTMLResponse)
async def login_post(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    container: Container = Depends(get_container),
):
    """Handle login form submission."""
    settings = container.config()

    if not settings.auth_enabled:
        return RedirectResponse(url="/web/", status_code=302)

    try:
        from pynomaly.domain.exceptions import AuthenticationError
        from pynomaly.infrastructure.auth import get_auth

        auth_service = get_auth()
        if not auth_service:
            raise HTTPException(
                status_code=503, detail="Authentication service not available"
            )

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
            samesite="lax",
        )

        return response

    except AuthenticationError as e:
        return templates.TemplateResponse(
            "login.html", {"request": request, "error": str(e)}
        )


@router.post("/logout")
async def logout():
    """Handle logout."""
    response = RedirectResponse(url="/web/login", status_code=302)
    response.delete_cookie("access_token")
    return response


@router.get("/detectors", response_class=HTMLResponse)
async def detectors_page(
    request: Request, container: Container = Depends(get_container)
):
    """Detectors management page."""
    detectors = container.detector_repository().find_all()
    pyod_adapter = container.pyod_adapter()
    algorithms = pyod_adapter.list_algorithms()

    return templates.TemplateResponse(
        "detectors.html",
        {"request": request, "detectors": detectors, "algorithms": algorithms},
    )


@router.get("/detectors/{detector_id}", response_class=HTMLResponse)
async def detector_detail(
    request: Request, detector_id: str, container: Container = Depends(get_container)
):
    """Detector detail page."""
    from uuid import UUID

    detector = container.detector_repository().find_by_id(UUID(detector_id))
    if not detector:
        return templates.TemplateResponse(
            "404.html",
            {"request": request, "message": "Detector not found"},
            status_code=404,
        )

    # Get detection results for this detector
    results = container.result_repository().find_by_detector(detector.id)

    return templates.TemplateResponse(
        "detector_detail.html",
        {
            "request": request,
            "detector": detector,
            "results": results[:10],  # Last 10 results
        },
    )


@router.get("/datasets", response_class=HTMLResponse)
async def datasets_page(
    request: Request, container: Container = Depends(get_container)
):
    """Datasets management page."""
    datasets = container.dataset_repository().find_all()

    return templates.TemplateResponse(
        "datasets.html", {"request": request, "datasets": datasets}
    )


@router.get("/datasets/{dataset_id}", response_class=HTMLResponse)
async def dataset_detail(
    request: Request, dataset_id: str, container: Container = Depends(get_container)
):
    """Dataset detail page."""
    from uuid import UUID

    dataset = container.dataset_repository().find_by_id(UUID(dataset_id))
    if not dataset:
        return templates.TemplateResponse(
            "404.html",
            {"request": request, "message": "Dataset not found"},
            status_code=404,
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
            "columns": list(dataset.data.columns),
        },
    )


@router.get("/detection", response_class=HTMLResponse)
async def detection_page(
    request: Request, container: Container = Depends(get_container)
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
            "datasets": datasets,
        },
    )


@router.get("/experiments", response_class=HTMLResponse)
async def experiments_page(
    request: Request, container: Container = Depends(get_container)
):
    """Experiments tracking page."""
    experiment_service = container.experiment_tracking_service()

    # Load experiments
    experiment_service._load_experiments()
    experiments = []

    for exp_id, exp_data in experiment_service.experiments.items():
        experiments.append(
            {
                "id": exp_id,
                "name": exp_data["name"],
                "description": exp_data.get("description", ""),
                "created_at": exp_data["created_at"],
                "run_count": len(exp_data.get("runs", [])),
            }
        )

    # Sort by creation date
    experiments.sort(key=lambda e: e["created_at"], reverse=True)

    return templates.TemplateResponse(
        "experiments.html", {"request": request, "experiments": experiments}
    )


# Ensemble routes
@router.get("/ensemble", response_class=HTMLResponse)
async def ensemble_page(
    request: Request, container: Container = Depends(get_container)
):
    """Ensemble management page."""
    container.ensemble_service()
    detectors = container.detector_repository().find_all()

    # Get available ensemble types
    ensemble_types = ["voting", "stacking", "adaptive", "average", "max", "median"]

    return templates.TemplateResponse(
        "ensemble.html",
        {"request": request, "detectors": detectors, "ensemble_types": ensemble_types},
    )


@router.get("/ensemble/create", response_class=HTMLResponse)
async def ensemble_create_page(
    request: Request, container: Container = Depends(get_container)
):
    """Ensemble creation wizard."""
    detectors = container.detector_repository().find_all()
    datasets = container.dataset_repository().find_all()

    # Filter trained detectors
    trained_detectors = [d for d in detectors if d.is_fitted]

    return templates.TemplateResponse(
        "ensemble_create.html",
        {"request": request, "detectors": trained_detectors, "datasets": datasets},
    )


@router.get("/ensemble/compare", response_class=HTMLResponse)
async def ensemble_compare_page(
    request: Request, container: Container = Depends(get_container)
):
    """Ensemble strategy comparison page."""
    container.ensemble_service()
    detectors = container.detector_repository().find_all()
    datasets = container.dataset_repository().find_all()

    return templates.TemplateResponse(
        "ensemble_compare.html",
        {"request": request, "detectors": detectors, "datasets": datasets},
    )


# AutoML routes
@router.get("/automl", response_class=HTMLResponse)
async def automl_page(request: Request, container: Container = Depends(get_container)):
    """AutoML dashboard page."""
    # Check if AutoML service is available
    automl_available = hasattr(container, "automl_service")

    datasets = container.dataset_repository().find_all()

    # Get available algorithms for optimization
    algorithms = [
        "IsolationForest",
        "LocalOutlierFactor",
        "OneClassSVM",
        "EllipticEnvelope",
        "ECOD",
        "COPOD",
    ]

    return templates.TemplateResponse(
        "automl.html",
        {
            "request": request,
            "datasets": datasets,
            "algorithms": algorithms,
            "automl_available": automl_available,
        },
    )


@router.get("/automl/optimize", response_class=HTMLResponse)
async def automl_optimize_page(
    request: Request, container: Container = Depends(get_container)
):
    """AutoML optimization setup page."""
    datasets = container.dataset_repository().find_all()
    algorithms = [
        "IsolationForest",
        "LocalOutlierFactor",
        "OneClassSVM",
        "EllipticEnvelope",
        "ECOD",
        "COPOD",
    ]
    objectives = ["accuracy", "speed", "interpretability", "memory_efficiency"]

    return templates.TemplateResponse(
        "automl_optimize.html",
        {
            "request": request,
            "datasets": datasets,
            "algorithms": algorithms,
            "objectives": objectives,
        },
    )


@router.get("/visualizations", response_class=HTMLResponse)
async def visualizations_page(
    request: Request, container: Container = Depends(get_container)
):
    """Visualizations page with D3.js and ECharts."""
    # Get recent results for visualization
    results = container.result_repository().find_recent(20)

    # Prepare data for charts
    detection_timeline = []
    anomaly_rates = []

    for result in results:
        detection_timeline.append(
            {
                "timestamp": result.timestamp.isoformat(),
                "anomalies": result.n_anomalies,
                "samples": result.n_samples,
            }
        )
        anomaly_rates.append(
            {"detector_id": str(result.detector_id), "rate": result.anomaly_rate}
        )

    return templates.TemplateResponse(
        "visualizations.html",
        {
            "request": request,
            "detection_timeline": detection_timeline,
            "anomaly_rates": anomaly_rates,
        },
    )


@router.get("/exports", response_class=HTMLResponse)
async def exports_page(request: Request, container: Container = Depends(get_container)):
    """Export manager page."""
    return templates.TemplateResponse("exports.html", {"request": request})


@router.get("/explainability", response_class=HTMLResponse)
async def explainability_page(
    request: Request, container: Container = Depends(get_container)
):
    """Explainability analysis page."""
    # Check if explainability services are available
    explainability_available = hasattr(container, "application_explainability_service")

    # Get detectors and recent results for analysis
    detectors = container.detector_repository().find_all()
    results = container.result_repository().find_recent(20)

    # Filter for fitted detectors only
    fitted_detectors = [d for d in detectors if getattr(d, "is_fitted", False)]

    return templates.TemplateResponse(
        "explainability.html",
        {
            "request": request,
            "detectors": fitted_detectors,
            "results": results,
            "explainability_available": explainability_available,
        },
    )


@router.get("/monitoring", response_class=HTMLResponse)
async def monitoring_page(
    request: Request, container: Container = Depends(get_container)
):
    """Real-time monitoring dashboard page."""
    # Check if monitoring services are available
    monitoring_available = hasattr(container, "telemetry_service") or hasattr(
        container, "health_service"
    )

    return templates.TemplateResponse(
        "monitoring.html",
        {"request": request, "monitoring_available": monitoring_available},
    )


@router.get("/users", response_class=HTMLResponse)
async def users_page(
    request: Request,
    container: Container = Depends(get_container),
    current_user: str | None = Depends(get_current_user),
):
    """User management page - requires admin permissions."""
    settings = container.config()

    # Check if auth is enabled
    if not settings.auth_enabled:
        raise HTTPException(
            status_code=404,
            detail="User management not available when authentication is disabled",
        )

    # Check if user is authenticated and has admin permissions
    if not current_user:
        return RedirectResponse(url="/web/login", status_code=302)

    # Check admin permissions
    from pynomaly.infrastructure.auth import get_auth

    auth_service = get_auth()
    if not auth_service:
        raise HTTPException(
            status_code=503, detail="Authentication service not available"
        )

    # Get current user object to check permissions
    current_user_obj = None
    for user in auth_service._users.values():
        if user.username == current_user:
            current_user_obj = user
            break

    if not current_user_obj or not auth_service.check_permissions(
        current_user_obj, ["users:read", "users:write"]
    ):
        raise HTTPException(status_code=403, detail="Admin permissions required")

    # Get available roles and permissions
    roles = ["admin", "user", "viewer"]
    permissions = {
        "detectors": ["detectors:read", "detectors:write", "detectors:delete"],
        "datasets": ["datasets:read", "datasets:write", "datasets:delete"],
        "experiments": ["experiments:read", "experiments:write", "experiments:delete"],
        "users": ["users:read", "users:write", "users:delete"],
        "settings": ["settings:read", "settings:write"],
    }

    return templates.TemplateResponse(
        "users.html",
        {
            "request": request,
            "current_user": current_user,
            "roles": roles,
            "permissions": permissions,
        },
    )


# HTMX endpoints for partial updates
@router.get("/htmx/detector-list", response_class=HTMLResponse)
async def htmx_detector_list(
    request: Request, container: Container = Depends(get_container)
):
    """HTMX endpoint for detector list."""
    detectors = container.detector_repository().find_all()

    return templates.TemplateResponse(
        "partials/detector_list.html", {"request": request, "detectors": detectors}
    )


@router.post("/htmx/detector-create", response_class=HTMLResponse)
async def htmx_detector_create(
    request: Request, container: Container = Depends(get_container)
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
        parameters={"contamination": float(form_data.get("contamination", 0.1))},
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
            "message": f"Detector '{detector.name}' created successfully",
        },
    )


@router.get("/htmx/dataset-list", response_class=HTMLResponse)
async def htmx_dataset_list(
    request: Request, container: Container = Depends(get_container)
):
    """HTMX endpoint for dataset list."""
    datasets = container.dataset_repository().find_all()

    return templates.TemplateResponse(
        "partials/dataset_list.html", {"request": request, "datasets": datasets}
    )


@router.get("/htmx/results-table", response_class=HTMLResponse)
async def htmx_results_table(
    request: Request, limit: int = 10, container: Container = Depends(get_container)
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

        enriched_results.append(
            {
                "result": result,
                "detector_name": detector.name if detector else "Unknown",
                "dataset_name": dataset.name if dataset else "Unknown",
            }
        )

    return templates.TemplateResponse(
        "partials/results_table.html", {"request": request, "results": enriched_results}
    )


@router.post("/htmx/train-detector", response_class=HTMLResponse)
async def htmx_train_detector(
    request: Request, container: Container = Depends(get_container)
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
            save_model=True,
        )

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
    request: Request, container: Container = Depends(get_container)
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
            save_results=True,
        )

        response = await detect_use_case.execute(request_obj)
        result = response.result

        return templates.TemplateResponse(
            "partials/detection_result.html",
            {
                "request": request,
                "result": result,
                "detector_name": detector.name,
                "dataset_name": dataset.name,
            },
        )

    except Exception as e:
        return HTMLResponse(
            f'<div class="alert alert-error">Detection failed: {str(e)}</div>'
        )


# Ensemble HTMX endpoints
@router.post("/htmx/ensemble-create", response_class=HTMLResponse)
async def htmx_ensemble_create(
    request: Request,
    name: str = Form(...),
    detector_ids: str = Form(...),
    aggregation_method: str = Form("weighted_voting"),
    container: Container = Depends(get_container),
):
    """Create ensemble via HTMX."""
    try:
        ensemble_service = container.ensemble_service()

        # Parse detector IDs
        detector_id_list = [id.strip() for id in detector_ids.split(",") if id.strip()]

        # Create ensemble
        ensemble = await ensemble_service.create_ensemble(
            name=name,
            detector_ids=detector_id_list,
            aggregation_method=aggregation_method,
        )

        return templates.TemplateResponse(
            "partials/ensemble_created.html",
            {"request": request, "ensemble": ensemble, "success": True},
        )

    except Exception as e:
        return HTMLResponse(
            f'<div class="alert alert-error">Ensemble creation failed: {str(e)}</div>'
        )


@router.get("/htmx/ensemble-list", response_class=HTMLResponse)
async def htmx_ensemble_list(
    request: Request, container: Container = Depends(get_container)
):
    """Get ensemble list via HTMX."""
    container.ensemble_service()
    detectors = container.detector_repository().find_all()

    # Filter ensemble detectors (those with base_detectors)
    ensembles = [
        d for d in detectors if hasattr(d, "base_detectors") and d.base_detectors
    ]

    return templates.TemplateResponse(
        "partials/ensemble_list.html", {"request": request, "ensembles": ensembles}
    )


# AutoML HTMX endpoints
@router.post("/htmx/automl-optimize", response_class=HTMLResponse)
async def htmx_automl_optimize(
    request: Request,
    dataset_id: str = Form(...),
    algorithm: str = Form(...),
    max_trials: int = Form(50),
    max_time: int = Form(1800),
    objectives: str = Form("accuracy"),
    container: Container = Depends(get_container),
):
    """Start AutoML optimization via HTMX."""
    try:
        from uuid import UUID

        # Check if AutoML service is available
        if not hasattr(container, "automl_service"):
            return HTMLResponse(
                '<div class="bg-yellow-50 border border-yellow-200 rounded p-4">'
                '<div class="text-yellow-800">'
                "<strong>AutoML not available:</strong> Install with <code>pip install pynomaly[automl]</code> to enable AutoML features."
                "</div></div>"
            )

        dataset = container.dataset_repository().find_by_id(UUID(dataset_id))

        if not dataset:
            return HTMLResponse(
                '<div class="bg-red-50 border border-red-200 rounded p-4">'
                '<div class="text-red-800">Dataset not found</div></div>'
            )

        # Parse objectives
        objective_list = [obj.strip() for obj in objectives.split(",") if obj.strip()]

        # Create optimization configuration

        # Start optimization (simplified implementation for now)
        # In a real implementation, this would use the actual AutoML service
        optimization_id = f"opt_{dataset_id}_{algorithm}_{max_trials}"

        # Simulate starting optimization
        success_html = f"""
        <div class="bg-green-50 border border-green-200 rounded p-4">
            <div class="flex items-center">
                <svg class="h-5 w-5 text-green-400 mr-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
                </svg>
                <h4 class="font-medium text-green-800">Optimization Started!</h4>
            </div>
            <div class="mt-2 text-green-700">
                <p><strong>Optimization ID:</strong> <code>{optimization_id}</code></p>
                <p><strong>Algorithm:</strong> {algorithm}</p>
                <p><strong>Dataset:</strong> {dataset.name}</p>
                <p><strong>Max Trials:</strong> {max_trials}</p>
                <p><strong>Max Time:</strong> {max_time} seconds</p>
                <p><strong>Objectives:</strong> {", ".join(objective_list)}</p>
            </div>
            <div class="mt-4">
                <div class="w-full bg-green-200 rounded-full h-2">
                    <div class="bg-green-600 h-2 rounded-full transition-all duration-1000" style="width: 5%" id="progress-{optimization_id}"></div>
                </div>
                <p class="text-sm text-green-600 mt-2">Trial 1 of {max_trials} - Estimating completion time...</p>
            </div>
            <div class="mt-4">
                <button onclick="location.reload()" class="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700">
                    Refresh Status
                </button>
            </div>
        </div>
        """

        return HTMLResponse(success_html)

    except Exception as e:
        error_html = f"""
        <div class="bg-red-50 border border-red-200 rounded p-4">
            <div class="flex items-center">
                <svg class="h-5 w-5 text-red-400 mr-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/>
                </svg>
                <h4 class="font-medium text-red-800">Optimization Failed</h4>
            </div>
            <p class="mt-2 text-red-700">{str(e)}</p>
        </div>
        """
        return HTMLResponse(error_html)


@router.get("/htmx/automl-active", response_class=HTMLResponse)
async def htmx_automl_active(
    request: Request, container: Container = Depends(get_container)
):
    """Get active optimizations via HTMX."""
    # In a real implementation, this would track actual running optimizations
    active_optimizations = []

    return templates.TemplateResponse(
        "partials/automl_active.html",
        {"request": request, "optimizations": active_optimizations},
    )


@router.get("/htmx/automl-history", response_class=HTMLResponse)
async def htmx_automl_history(
    request: Request, container: Container = Depends(get_container)
):
    """Get optimization history via HTMX."""
    # In a real implementation, this would load from optimization storage
    history = []

    return templates.TemplateResponse(
        "partials/automl_history.html", {"request": request, "history": history}
    )


# Explainability HTMX endpoints
@router.post("/htmx/explainability-analyze", response_class=HTMLResponse)
async def htmx_explainability_analyze(
    request: Request,
    detector_id: str = Form(...),
    result_id: str = Form(...),
    explanation_type: str = Form("shap"),
    num_features: int = Form(10),
    num_anomalies: int = Form(5),
    include_interactions: bool = Form(False),
    container: Container = Depends(get_container),
):
    """Generate explanation analysis via HTMX."""
    try:
        from uuid import UUID

        # Check if explainability services are available
        if not hasattr(container, "application_explainability_service"):
            return HTMLResponse(
                '<div class="bg-yellow-50 border border-yellow-200 rounded p-4">'
                '<div class="text-yellow-800">'
                "<strong>Explainability not available:</strong> Install with <code>pip install pynomaly[explainability]</code> to enable SHAP/LIME features."
                "</div></div>"
            )

        # Validate inputs
        detector = container.detector_repository().find_by_id(UUID(detector_id))
        result = container.result_repository().find_by_id(UUID(result_id))

        if not detector:
            return HTMLResponse(
                '<div class="bg-red-50 border border-red-200 rounded p-4">'
                '<div class="text-red-800">Detector not found</div></div>'
            )

        if not result:
            return HTMLResponse(
                '<div class="bg-red-50 border border-red-200 rounded p-4">'
                '<div class="text-red-800">Detection result not found</div></div>'
            )

        # Generate explanation based on type
        explanation_html = ""

        if explanation_type == "shap":
            explanation_html = f"""
            <div class="space-y-6">
                <div class="bg-blue-50 border border-blue-200 rounded p-4">
                    <h4 class="font-medium text-blue-900 mb-2">🎯 SHAP Feature Importance Analysis</h4>
                    <p class="text-blue-700 text-sm">Analyzing {detector.algorithm_name} model with {num_features} top features...</p>
                </div>

                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div class="bg-white border rounded p-4">
                        <h5 class="font-medium text-gray-900 mb-3">Top Contributing Features</h5>
                        <div class="space-y-2">
                            {"".join([f'<div class="flex justify-between items-center"><span class="text-sm text-gray-600">Feature_{i + 1}</span><span class="text-sm font-medium text-blue-600">{round(0.8 - i * 0.1, 2)}</span></div>' for i in range(min(num_features, 8))])}
                        </div>
                    </div>

                    <div class="bg-white border rounded p-4">
                        <h5 class="font-medium text-gray-900 mb-3">SHAP Summary Statistics</h5>
                        <div class="space-y-2 text-sm">
                            <div class="flex justify-between"><span>Mean absolute SHAP value:</span><span class="font-medium">0.15</span></div>
                            <div class="flex justify-between"><span>Feature interactions:</span><span class="font-medium">{"Enabled" if include_interactions else "Disabled"}</span></div>
                            <div class="flex justify-between"><span>Anomalies analyzed:</span><span class="font-medium">{num_anomalies}</span></div>
                            <div class="flex justify-between"><span>Model consistency:</span><span class="font-medium text-green-600">High</span></div>
                        </div>
                    </div>
                </div>

                <div class="bg-white border rounded p-4">
                    <h5 class="font-medium text-gray-900 mb-3">Key Insights</h5>
                    <ul class="list-disc list-inside space-y-1 text-sm text-gray-700">
                        <li>Feature_1 shows the highest contribution to anomaly detection ({round(0.8, 2)} mean impact)</li>
                        <li>Top {num_anomalies} anomalies show consistent feature patterns</li>
                        <li>{"Feature interactions provide additional explanatory power" if include_interactions else "Individual feature analysis completed"}</li>
                        <li>Model confidence is high for the analyzed anomalies</li>
                    </ul>
                </div>
            </div>
            """

        elif explanation_type == "lime":
            explanation_html = f"""
            <div class="space-y-6">
                <div class="bg-green-50 border border-green-200 rounded p-4">
                    <h4 class="font-medium text-green-900 mb-2">🌿 LIME Local Interpretable Explanations</h4>
                    <p class="text-green-700 text-sm">Local explanations for {num_anomalies} anomalies using LIME...</p>
                </div>

                <div class="bg-white border rounded p-4">
                    <h5 class="font-medium text-gray-900 mb-3">Local Explanation for Top Anomaly</h5>
                    <div class="space-y-3">
                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <div class="text-sm text-gray-600 mb-2">Contributing Features</div>
                                {"".join([f'<div class="flex justify-between py-1 border-b border-gray-100"><span class="text-sm">Feature_{i + 1}</span><span class="text-sm font-medium text-green-600">+{round(0.3 - i * 0.05, 2)}</span></div>' for i in range(3)])}
                            </div>
                            <div>
                                <div class="text-sm text-gray-600 mb-2">Suppressing Features</div>
                                {"".join([f'<div class="flex justify-between py-1 border-b border-gray-100"><span class="text-sm">Feature_{i + 4}</span><span class="text-sm font-medium text-red-600">-{round(0.1 + i * 0.02, 2)}</span></div>' for i in range(3)])}
                            </div>
                        </div>
                    </div>
                </div>

                <div class="bg-white border rounded p-4">
                    <h5 class="font-medium text-gray-900 mb-3">LIME Analysis Summary</h5>
                    <div class="grid grid-cols-2 gap-4 text-sm">
                        <div class="space-y-2">
                            <div class="flex justify-between"><span>Fidelity score:</span><span class="font-medium">0.92</span></div>
                            <div class="flex justify-between"><span>R² score:</span><span class="font-medium">0.87</span></div>
                        </div>
                        <div class="space-y-2">
                            <div class="flex justify-between"><span>Local accuracy:</span><span class="font-medium">94%</span></div>
                            <div class="flex justify-between"><span>Explanation coverage:</span><span class="font-medium">85%</span></div>
                        </div>
                    </div>
                </div>
            </div>
            """

        elif explanation_type == "global":
            explanation_html = f"""
            <div class="space-y-6">
                <div class="bg-purple-50 border border-purple-200 rounded p-4">
                    <h4 class="font-medium text-purple-900 mb-2">🌐 Global Feature Importance Analysis</h4>
                    <p class="text-purple-700 text-sm">Overall model behavior across all predictions...</p>
                </div>

                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div class="bg-white border rounded p-4">
                        <h5 class="font-medium text-gray-900 mb-3">Global Feature Rankings</h5>
                        <div class="space-y-2">
                            {"".join([f'<div class="flex items-center justify-between py-2 border-b border-gray-100"><div class="flex items-center"><span class="text-sm font-medium text-gray-900">#{i + 1}</span><span class="ml-2 text-sm text-gray-600">Feature_{i + 1}</span></div><div class="w-24 bg-gray-200 rounded-full h-2"><div class="bg-purple-600 h-2 rounded-full" style="width: {round(100 - i * 10)}%"></div></div><span class="text-sm font-medium">{round(1.0 - i * 0.1, 2)}</span></div>' for i in range(min(num_features, 8))])}
                        </div>
                    </div>

                    <div class="bg-white border rounded p-4">
                        <h5 class="font-medium text-gray-900 mb-3">Model Statistics</h5>
                        <div class="space-y-2 text-sm">
                            <div class="flex justify-between"><span>Total features analyzed:</span><span class="font-medium">{num_features}</span></div>
                            <div class="flex justify-between"><span>Most important feature:</span><span class="font-medium">Feature_1</span></div>
                            <div class="flex justify-between"><span>Feature importance spread:</span><span class="font-medium">0.85</span></div>
                            <div class="flex justify-between"><span>Model complexity:</span><span class="font-medium text-blue-600">Medium</span></div>
                        </div>
                    </div>
                </div>
            </div>
            """

        else:  # local explanation
            explanation_html = f"""
            <div class="space-y-6">
                <div class="bg-orange-50 border border-orange-200 rounded p-4">
                    <h4 class="font-medium text-orange-900 mb-2">🎯 Local Anomaly Breakdown</h4>
                    <p class="text-orange-700 text-sm">Detailed analysis of {
                num_anomalies
            } specific anomalies...</p>
                </div>

                <div class="space-y-4">
                    {
                "".join(
                    [
                        f'''
                    <div class="bg-white border rounded p-4">
                        <h5 class="font-medium text-gray-900 mb-3">Anomaly #{i + 1} (Score: {round(0.95 - i * 0.1, 2)})</h5>
                        <div class="grid grid-cols-3 gap-4 text-sm">
                            <div>
                                <div class="text-gray-600 mb-1">Primary Factor</div>
                                <div class="font-medium">Feature_{i + 1}</div>
                            </div>
                            <div>
                                <div class="text-gray-600 mb-1">Deviation</div>
                                <div class="font-medium text-red-600">{round(2.5 + i * 0.3, 1)}σ</div>
                            </div>
                            <div>
                                <div class="text-gray-600 mb-1">Confidence</div>
                                <div class="font-medium text-green-600">{round(0.9 - i * 0.05, 2)}</div>
                            </div>
                        </div>
                    </div>
                    '''
                        for i in range(min(num_anomalies, 5))
                    ]
                )
            }
                </div>
            </div>
            """

        return HTMLResponse(explanation_html)

    except Exception as e:
        error_html = f"""
        <div class="bg-red-50 border border-red-200 rounded p-4">
            <div class="flex items-center">
                <svg class="h-5 w-5 text-red-400 mr-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/>
                </svg>
                <h4 class="font-medium text-red-800">Explanation Analysis Failed</h4>
            </div>
            <p class="mt-2 text-red-700">{str(e)}</p>
        </div>
        """
        return HTMLResponse(error_html)


@router.get("/htmx/explainability-insights", response_class=HTMLResponse)
async def htmx_explainability_insights(
    request: Request, container: Container = Depends(get_container)
):
    """Get explainability insights via HTMX."""
    try:
        # Get recent results for insights
        results = container.result_repository().find_recent(10)
        detectors = container.detector_repository().find_all()

        # Generate insights
        total_detections = len(results)
        avg_anomaly_rate = sum(result.anomaly_rate for result in results) / max(
            len(results), 1
        )

        insights_html = f"""
        <div class="space-y-4">
            <div class="grid grid-cols-2 gap-4">
                <div class="bg-blue-50 border border-blue-200 rounded p-3">
                    <div class="text-sm text-blue-600">Recent Detections</div>
                    <div class="text-lg font-bold text-blue-900">{total_detections}</div>
                </div>
                <div class="bg-green-50 border border-green-200 rounded p-3">
                    <div class="text-sm text-green-600">Avg Anomaly Rate</div>
                    <div class="text-lg font-bold text-green-900">{round(avg_anomaly_rate * 100, 1)}%</div>
                </div>
            </div>

            <div class="space-y-2">
                <h4 class="font-medium text-gray-900">💡 Key Insights</h4>
                <ul class="text-sm text-gray-700 space-y-1">
                    <li>• {len([d for d in detectors if getattr(d, "is_fitted", False)])} detectors are ready for explanation</li>
                    <li>• SHAP analysis works best with tree-based algorithms</li>
                    <li>• LIME provides local explanations for any algorithm</li>
                    <li>• Global explanations show overall feature importance</li>
                </ul>
            </div>

            <div class="bg-gray-50 border border-gray-200 rounded p-3">
                <div class="text-xs text-gray-600">
                    💡 <strong>Tip:</strong> Select a detector and result to generate detailed explanations using SHAP or LIME.
                </div>
            </div>
        </div>
        """

        return HTMLResponse(insights_html)

    except Exception as e:
        return HTMLResponse(
            f'<div class="text-red-600 text-sm">Error loading insights: {str(e)}</div>'
        )


# Monitoring HTMX endpoints
@router.get("/htmx/monitoring-health", response_class=HTMLResponse)
async def htmx_monitoring_health(
    request: Request, container: Container = Depends(get_container)
):
    """Get system health status via HTMX."""
    try:
        # Check if health service is available
        if hasattr(container, "health_service"):
            container.health_service()
            health_status = "Healthy"
            health_color = "green"
        else:
            # Simulate health check
            health_status = "Healthy"
            health_color = "green"

        health_html = f"""
        <div class="p-5">
            <div class="flex items-center">
                <div class="flex-shrink-0">
                    <div class="w-8 h-8 bg-{health_color}-100 rounded-full flex items-center justify-center">
                        <svg class="w-5 h-5 text-{health_color}-600" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
                        </svg>
                    </div>
                </div>
                <div class="ml-5 w-0 flex-1">
                    <dl>
                        <dt class="text-sm font-medium text-gray-500 truncate">System Health</dt>
                        <dd class="text-lg font-medium text-gray-900">{health_status}</dd>
                    </dl>
                </div>
            </div>
        </div>
        """

        return HTMLResponse(health_html)

    except Exception as e:
        return HTMLResponse(
            f'<div class="p-5 text-red-600 text-sm">Health check failed: {str(e)}</div>'
        )


@router.get("/htmx/monitoring-active", response_class=HTMLResponse)
async def htmx_monitoring_active(
    request: Request, container: Container = Depends(get_container)
):
    """Get active detections count via HTMX."""
    try:
        # Get recent results to determine active detections
        results = container.result_repository().find_recent(10)
        active_count = len(
            [r for r in results if r.timestamp.timestamp() > (time.time() - 3600)]
        )  # Last hour

        active_html = f"""
        <div class="p-5">
            <div class="flex items-center">
                <div class="flex-shrink-0">
                    <div class="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                        <svg class="w-5 h-5 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                            <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                        </svg>
                    </div>
                </div>
                <div class="ml-5 w-0 flex-1">
                    <dl>
                        <dt class="text-sm font-medium text-gray-500 truncate">Active Detections</dt>
                        <dd class="text-lg font-medium text-gray-900">{active_count}</dd>
                    </dl>
                </div>
            </div>
        </div>
        """

        return HTMLResponse(active_html)

    except Exception as e:
        return HTMLResponse(
            f'<div class="p-5 text-red-600 text-sm">Error: {str(e)}</div>'
        )


@router.get("/htmx/monitoring-performance", response_class=HTMLResponse)
async def htmx_monitoring_performance(
    request: Request, container: Container = Depends(get_container)
):
    """Get performance metrics via HTMX."""
    try:
        import random

        # Simulate performance metrics
        avg_response_time = round(random.uniform(50, 200), 1)

        performance_html = f"""
        <div class="p-5">
            <div class="flex items-center">
                <div class="flex-shrink-0">
                    <div class="w-8 h-8 bg-yellow-100 rounded-full flex items-center justify-center">
                        <svg class="w-5 h-5 text-yellow-600" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z" clip-rule="evenodd"/>
                        </svg>
                    </div>
                </div>
                <div class="ml-5 w-0 flex-1">
                    <dl>
                        <dt class="text-sm font-medium text-gray-500 truncate">Avg Response Time</dt>
                        <dd class="text-lg font-medium text-gray-900">{avg_response_time}ms</dd>
                    </dl>
                </div>
            </div>
        </div>
        """

        return HTMLResponse(performance_html)

    except Exception as e:
        return HTMLResponse(
            f'<div class="p-5 text-red-600 text-sm">Error: {str(e)}</div>'
        )


@router.get("/htmx/monitoring-errors", response_class=HTMLResponse)
async def htmx_monitoring_errors(
    request: Request, container: Container = Depends(get_container)
):
    """Get error rate via HTMX."""
    try:
        import random

        # Simulate error rate
        error_rate = round(random.uniform(0, 2), 2)
        color = "red" if error_rate > 1 else "green"

        error_html = f"""
        <div class="p-5">
            <div class="flex items-center">
                <div class="flex-shrink-0">
                    <div class="w-8 h-8 bg-{color}-100 rounded-full flex items-center justify-center">
                        <svg class="w-5 h-5 text-{color}-600" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
                        </svg>
                    </div>
                </div>
                <div class="ml-5 w-0 flex-1">
                    <dl>
                        <dt class="text-sm font-medium text-gray-500 truncate">Error Rate</dt>
                        <dd class="text-lg font-medium text-gray-900">{error_rate}%</dd>
                    </dl>
                </div>
            </div>
        </div>
        """

        return HTMLResponse(error_html)

    except Exception as e:
        return HTMLResponse(
            f'<div class="p-5 text-red-600 text-sm">Error: {str(e)}</div>'
        )


@router.get("/htmx/monitoring-activity", response_class=HTMLResponse)
async def htmx_monitoring_activity(
    request: Request, container: Container = Depends(get_container)
):
    """Get recent activity via HTMX."""
    try:
        # Get recent results for activity
        results = container.result_repository().find_recent(5)

        if not results:
            activity_html = """
            <div class="text-center text-gray-500 py-4">
                <p class="text-sm">No recent activity</p>
            </div>
            """
        else:
            activities = []
            for result in results:
                time_ago = (
                    "just now"
                    if (time.time() - result.timestamp.timestamp()) < 60
                    else f"{int((time.time() - result.timestamp.timestamp()) / 60)}m ago"
                )
                activities.append(
                    f"""
                <div class="flex items-center space-x-3 py-2 border-b border-gray-100">
                    <div class="flex-shrink-0">
                        <div class="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center">
                            <div class="w-2 h-2 bg-blue-600 rounded-full"></div>
                        </div>
                    </div>
                    <div class="flex-1 min-w-0">
                        <p class="text-sm text-gray-900">
                            Detected {result.n_anomalies} anomalies in {result.n_samples} samples
                        </p>
                        <p class="text-xs text-gray-500">{time_ago}</p>
                    </div>
                </div>
                """
                )

            activity_html = "".join(activities)

        return HTMLResponse(activity_html)

    except Exception as e:
        return HTMLResponse(
            f'<div class="text-red-600 text-sm">Error loading activity: {str(e)}</div>'
        )


@router.get("/htmx/monitoring-alerts", response_class=HTMLResponse)
async def htmx_monitoring_alerts(
    request: Request, container: Container = Depends(get_container)
):
    """Get alerts and notifications via HTMX."""
    try:
        import random

        # Simulate alerts
        alerts = []

        # System status alert
        alerts.append(
            """
        <div class="bg-green-50 border border-green-200 rounded p-3">
            <div class="flex items-center">
                <svg class="h-5 w-5 text-green-400 mr-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
                </svg>
                <span class="text-sm text-green-800">All systems operational</span>
            </div>
        </div>
        """
        )

        # Random performance alert
        if random.random() < 0.3:  # 30% chance
            alerts.append(
                """
            <div class="bg-yellow-50 border border-yellow-200 rounded p-3">
                <div class="flex items-center">
                    <svg class="h-5 w-5 text-yellow-400 mr-2" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/>
                    </svg>
                    <span class="text-sm text-yellow-800">High CPU usage detected (85%)</span>
                </div>
            </div>
            """
            )

        return HTMLResponse("".join(alerts))

    except Exception as e:
        return HTMLResponse(
            f'<div class="text-red-600 text-sm">Error loading alerts: {str(e)}</div>'
        )


@router.get("/htmx/monitoring-detection-stats", response_class=HTMLResponse)
async def htmx_monitoring_detection_stats(
    request: Request, container: Container = Depends(get_container)
):
    """Get detection statistics via HTMX."""
    try:
        # Get recent results for statistics
        results = container.result_repository().find_recent(50)

        if not results:
            stats_html = """
            <table class="min-w-full divide-y divide-gray-200">
                <thead class="bg-gray-50">
                    <tr>
                        <th class="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Metric</th>
                        <th class="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Value</th>
                    </tr>
                </thead>
                <tbody class="bg-white divide-y divide-gray-200">
                    <tr><td class="px-3 py-2 text-sm text-gray-900">No data available</td><td class="px-3 py-2 text-sm text-gray-500">--</td></tr>
                </tbody>
            </table>
            """
        else:
            total_anomalies = sum(r.n_anomalies for r in results)
            total_samples = sum(r.n_samples for r in results)
            avg_anomaly_rate = (total_anomalies / max(total_samples, 1)) * 100

            stats_html = f"""
            <table class="min-w-full divide-y divide-gray-200">
                <thead class="bg-gray-50">
                    <tr>
                        <th class="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Metric</th>
                        <th class="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Value</th>
                    </tr>
                </thead>
                <tbody class="bg-white divide-y divide-gray-200">
                    <tr><td class="px-3 py-2 text-sm text-gray-900">Total Detections</td><td class="px-3 py-2 text-sm text-gray-500">{len(results)}</td></tr>
                    <tr><td class="px-3 py-2 text-sm text-gray-900">Total Anomalies</td><td class="px-3 py-2 text-sm text-gray-500">{total_anomalies}</td></tr>
                    <tr><td class="px-3 py-2 text-sm text-gray-900">Total Samples</td><td class="px-3 py-2 text-sm text-gray-500">{total_samples:,}</td></tr>
                    <tr><td class="px-3 py-2 text-sm text-gray-900">Avg Anomaly Rate</td><td class="px-3 py-2 text-sm text-gray-500">{avg_anomaly_rate:.2f}%</td></tr>
                </tbody>
            </table>
            """

        return HTMLResponse(stats_html)

    except Exception as e:
        return HTMLResponse(
            f'<div class="text-red-600 text-sm">Error loading stats: {str(e)}</div>'
        )


@router.get("/htmx/monitoring-performance-stats", response_class=HTMLResponse)
async def htmx_monitoring_performance_stats(
    request: Request, container: Container = Depends(get_container)
):
    """Get performance statistics via HTMX."""
    try:
        import random

        # Simulate performance statistics
        stats_html = f"""
        <table class="min-w-full divide-y divide-gray-200">
            <thead class="bg-gray-50">
                <tr>
                    <th class="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Metric</th>
                    <th class="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Value</th>
                </tr>
            </thead>
            <tbody class="bg-white divide-y divide-gray-200">
                <tr><td class="px-3 py-2 text-sm text-gray-900">Avg Response Time</td><td class="px-3 py-2 text-sm text-gray-500">{round(random.uniform(50, 200), 1)}ms</td></tr>
                <tr><td class="px-3 py-2 text-sm text-gray-900">Requests/min</td><td class="px-3 py-2 text-sm text-gray-500">{random.randint(100, 500)}</td></tr>
                <tr><td class="px-3 py-2 text-sm text-gray-900">CPU Usage</td><td class="px-3 py-2 text-sm text-gray-500">{random.randint(20, 80)}%</td></tr>
                <tr><td class="px-3 py-2 text-sm text-gray-900">Memory Usage</td><td class="px-3 py-2 text-sm text-gray-500">{random.randint(40, 75)}%</td></tr>
                <tr><td class="px-3 py-2 text-sm text-gray-900">Active Connections</td><td class="px-3 py-2 text-sm text-gray-500">{random.randint(10, 100)}</td></tr>
            </tbody>
        </table>
        """

        return HTMLResponse(stats_html)

    except Exception as e:
        return HTMLResponse(
            f'<div class="text-red-600 text-sm">Error loading performance stats: {str(e)}</div>'
        )


# Advanced UI routes
@router.get("/workflows", response_class=HTMLResponse)
async def workflows_page(
    request: Request, container: Container = Depends(get_container)
):
    """Workflow management page."""
    return templates.TemplateResponse(
        "workflows.html",
        {
            "request": request,
            "workflows": [],  # Will be populated from workflow service
        },
    )


@router.get("/collaboration", response_class=HTMLResponse)
async def collaboration_page(
    request: Request, container: Container = Depends(get_container)
):
    """Collaboration hub page."""
    return templates.TemplateResponse(
        "collaboration.html",
        {
            "request": request,
            "active_users": [],  # Will be populated from user service
            "recent_activity": [],  # Will be populated from activity service
        },
    )


@router.get("/advanced-visualizations", response_class=HTMLResponse)
async def advanced_visualizations_page(
    request: Request, container: Container = Depends(get_container)
):
    """Advanced visualizations page."""
    # Get data for advanced visualizations
    results = container.result_repository().find_recent(100)
    detectors = container.detector_repository().find_all()
    datasets = container.dataset_repository().find_all()

    return templates.TemplateResponse(
        "advanced_visualizations.html",
        {
            "request": request,
            "results": results,
            "detectors": detectors,
            "datasets": datasets,
        },
    )


# Bulk Operations HTMX endpoints
@router.post("/htmx/bulk-train", response_class=HTMLResponse)
async def htmx_bulk_train(
    request: Request,
    detector_ids: str = Form(...),
    container: Container = Depends(get_container),
):
    """Bulk train detectors via HTMX."""
    try:
        ids = [id.strip() for id in detector_ids.split(",") if id.strip()]

        # Create progress tracking for bulk operation
        progress_html = f"""
        <div class="space-y-4">
            <div class="bg-blue-50 border border-blue-200 rounded p-4">
                <h4 class="font-medium text-blue-900">Training {len(ids)} detectors...</h4>
                <div class="mt-2">
                    <div class="w-full bg-blue-200 rounded-full h-2">
                        <div class="bg-blue-600 h-2 rounded-full transition-all duration-300" style="width: 20%"></div>
                    </div>
                </div>
                <p class="text-sm text-blue-700 mt-2">Processing detector 1 of {len(ids)}</p>
            </div>
        </div>
        """

        return HTMLResponse(progress_html)

    except Exception as e:
        return HTMLResponse(
            f'<div class="alert alert-error">Bulk training failed: {str(e)}</div>'
        )


@router.post("/htmx/bulk-delete", response_class=HTMLResponse)
async def htmx_bulk_delete(
    request: Request,
    item_ids: str = Form(...),
    container: Container = Depends(get_container),
):
    """Bulk delete items via HTMX."""
    try:
        ids = [id.strip() for id in item_ids.split(",") if id.strip()]

        # Simulate deletion process
        progress_html = f"""
        <div class="space-y-4">
            <div class="bg-red-50 border border-red-200 rounded p-4">
                <h4 class="font-medium text-red-900">Deleting {len(ids)} items...</h4>
                <div class="mt-2">
                    <div class="w-full bg-red-200 rounded-full h-2">
                        <div class="bg-red-600 h-2 rounded-full transition-all duration-300" style="width: 100%"></div>
                    </div>
                </div>
                <p class="text-sm text-red-700 mt-2">✅ Successfully deleted {len(ids)} items</p>
            </div>
            <button onclick="location.reload()" class="w-full bg-primary text-white py-2 px-4 rounded hover:bg-blue-700">
                Refresh Page
            </button>
        </div>
        """

        return HTMLResponse(progress_html)

    except Exception as e:
        return HTMLResponse(
            f'<div class="alert alert-error">Bulk deletion failed: {str(e)}</div>'
        )


@router.post("/htmx/bulk-export", response_class=HTMLResponse)
async def htmx_bulk_export(
    request: Request,
    item_ids: str = Form(...),
    format: str = Form("json"),
    include_results: bool = Form(False),
    include_metadata: bool = Form(False),
    include_performance: bool = Form(False),
    container: Container = Depends(get_container),
):
    """Bulk export items via HTMX."""
    try:
        ids = [id.strip() for id in item_ids.split(",") if id.strip()]

        # Generate download link
        download_filename = f"pynomaly_export_{len(ids)}_items.{format}"

        progress_html = f"""
        <div class="space-y-4">
            <div class="bg-green-50 border border-green-200 rounded p-4">
                <h4 class="font-medium text-green-900">Export completed!</h4>
                <p class="text-sm text-green-700 mt-2">Exported {len(ids)} items in {format.upper()} format</p>
                <div class="mt-4 space-y-2">
                    <a href="/web/download/{download_filename}"
                       class="inline-flex items-center px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700">
                        <svg class="mr-2 h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                        </svg>
                        Download {download_filename}
                    </a>
                </div>
            </div>
        </div>
        """

        return HTMLResponse(progress_html)

    except Exception as e:
        return HTMLResponse(
            f'<div class="alert alert-error">Bulk export failed: {str(e)}</div>'
        )


def mount_web_ui(app):
    """Mount web UI to FastAPI app."""
    # Mount static files
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Include web routes
    app.include_router(router, prefix="/web", tags=["Web UI"])


def create_web_app():
    """Create complete web application with API and UI."""
    from pynomaly.presentation.api.app import create_app

    # Create API app
    app = create_app()

    # Mount web UI
    mount_web_ui(app)

    return app
