# Contributing to Interfaces Package

Thank you for your interest in contributing to the Interfaces package! This package provides all user-facing interfaces including REST APIs, CLI tools, web applications, and client SDKs for the Pynomaly platform.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Interface Development Guidelines](#interface-development-guidelines)
- [Testing Requirements](#testing-requirements)
- [Documentation Standards](#documentation-standards)
- [Pull Request Process](#pull-request-process)
- [User Experience Standards](#user-experience-standards)
- [Community](#community)

## Code of Conduct

This project adheres to our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+ (for JavaScript SDK and web development)
- Modern web browser (Chrome 90+, Firefox 88+, Safari 14+)
- Docker (for containerized testing)
- Understanding of web development and API design principles

### Repository Setup

```bash
# Clone the repository
git clone https://github.com/your-org/monorepo.git
cd monorepo/src/packages/software/interfaces

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,test,api,cli,web,sdk]"

# Install Node.js dependencies (for web and JS SDK)
cd interfaces/web/static && npm install
cd ../../../interfaces/sdk/javascript && npm install

# Install pre-commit hooks
pre-commit install
```

### Environment Setup

```bash
# Start development services
docker-compose -f docker/dev-services.yml up -d

# Start API server in development mode
uvicorn interfaces.api.main:app --reload --host 0.0.0.0 --port 8000

# Start web development server
cd interfaces/web && python -m http.server 3000

# Run CLI in development mode
python -m interfaces.cli --help
```

## Development Environment

### IDE Configuration

Recommended VS Code extensions:
- Python
- REST Client
- Live Server
- Tailwind CSS IntelliSense
- GraphQL
- JavaScript/TypeScript

### Development Services

```yaml
# docker/dev-services.yml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: pynomaly_dev
      POSTGRES_USER: dev
      POSTGRES_PASSWORD: dev123
    ports:
      - "5432:5432"
  
  mailhog:
    image: mailhog/mailhog
    ports:
      - "1025:1025"
      - "8025:8025"
```

### Environment Variables

Create a `.env` file for local development:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=true
API_RELOAD=true

# Database
DATABASE_URL=postgresql://dev:dev123@localhost:5432/pynomaly_dev
REDIS_URL=redis://localhost:6379

# Authentication
JWT_SECRET_KEY=dev-secret-key-change-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=30

# CORS Configuration
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]
CORS_ALLOW_CREDENTIALS=true

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# CLI Configuration
CLI_THEME=dark
CLI_OUTPUT_FORMAT=table
CLI_PROGRESS_BARS=true

# Web Configuration
WEB_DEBUG=true
WEB_THEME=auto
WEB_ENABLE_PWA=true

# WebSocket Configuration
WS_MAX_CONNECTIONS=1000
WS_HEARTBEAT_INTERVAL=30

# Logging
LOG_LEVEL=DEBUG
LOG_FORMAT=json
```

## Interface Development Guidelines

### Architecture Principles

This package follows presentation layer patterns:

1. **Separation of Concerns**: Clear separation between presentation logic and business logic
2. **Dependency Inversion**: Interfaces depend on abstractions, not implementations
3. **Single Responsibility**: Each interface has a focused purpose
4. **User-Centric Design**: Prioritize user experience and accessibility
5. **API-First**: Design APIs before implementing user interfaces

### Package Structure

```
interfaces/
├── api/                    # REST API implementation
│   ├── routers/           # FastAPI route handlers
│   │   ├── detection.py   # Anomaly detection endpoints
│   │   ├── datasets.py    # Dataset management endpoints
│   │   └── models.py      # Model management endpoints
│   ├── middleware/        # HTTP middleware
│   │   ├── auth.py        # Authentication middleware
│   │   ├── cors.py        # CORS configuration
│   │   └── rate_limit.py  # Rate limiting
│   ├── schemas/           # Pydantic request/response models
│   └── dependencies/      # Dependency injection
├── cli/                   # Command-line interface
│   ├── commands/          # CLI command implementations
│   ├── interactive/       # Interactive prompts and wizards
│   └── utils/            # CLI utilities
├── web/                   # Web application
│   ├── static/           # Static assets
│   ├── templates/        # HTML templates
│   └── components/       # Reusable components
├── sdk/                   # Client SDKs
│   ├── python/           # Python client library
│   └── javascript/       # JavaScript/TypeScript client
└── graphql/              # GraphQL API (optional)
```

### API Development Patterns

**FastAPI Router Pattern:**
```python
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional

from interfaces.api.schemas import DetectionRequest, DetectionResponse
from interfaces.api.dependencies import get_current_user, get_detection_service
from core.application.use_cases import DetectAnomaliesUseCase

router = APIRouter(prefix="/detection", tags=["detection"])

@router.post(
    "/detect",
    response_model=DetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect anomalies in dataset",
    description="Run anomaly detection on a dataset using specified algorithm",
    responses={
        200: {"description": "Detection completed successfully"},
        400: {"description": "Invalid request data"},
        401: {"description": "Authentication required"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    }
)
async def detect_anomalies(
    request: DetectionRequest,
    current_user = Depends(get_current_user),
    detection_service: DetectAnomaliesUseCase = Depends(get_detection_service)
) -> DetectionResponse:
    """
    Detect anomalies in a dataset.
    
    This endpoint accepts a dataset and detection configuration,
    runs the specified anomaly detection algorithm, and returns
    the results including identified anomalies and their scores.
    
    - **dataset_id**: ID of the dataset to analyze
    - **algorithm**: Detection algorithm to use
    - **parameters**: Algorithm-specific parameters
    """
    try:
        # Validate request
        if not request.dataset_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dataset ID is required"
            )
        
        # Execute use case
        result = await detection_service.execute(
            dataset_id=request.dataset_id,
            algorithm=request.algorithm,
            parameters=request.parameters,
            user_context=current_user
        )
        
        # Format response
        return DetectionResponse(
            job_id=result.job_id,
            status="completed",
            anomalies=result.anomalies,
            statistics=result.statistics,
            execution_time=result.execution_time
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}"
        )
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    except Exception as e:
        # Log error for debugging
        logger.error(f"Detection error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
```

**Request/Response Schemas:**
```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Any
from datetime import datetime

class DetectionRequest(BaseModel):
    """Request model for anomaly detection."""
    
    dataset_id: str = Field(
        ...,
        description="ID of the dataset to analyze",
        example="dataset_123"
    )
    algorithm: str = Field(
        ...,
        description="Detection algorithm to use",
        example="isolation_forest"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default={},
        description="Algorithm-specific parameters",
        example={"contamination": 0.1, "n_estimators": 100}
    )
    
    @validator('algorithm')
    def validate_algorithm(cls, v):
        """Validate algorithm name."""
        allowed_algorithms = [
            'isolation_forest', 'lof', 'one_class_svm', 'autoencoder'
        ]
        if v not in allowed_algorithms:
            raise ValueError(f"Algorithm must be one of: {allowed_algorithms}")
        return v
    
    @validator('parameters')
    def validate_parameters(cls, v, values):
        """Validate algorithm parameters."""
        algorithm = values.get('algorithm')
        if algorithm == 'isolation_forest':
            if 'contamination' in v:
                contamination = v['contamination']
                if not 0.0 <= contamination <= 1.0:
                    raise ValueError("Contamination must be between 0.0 and 1.0")
        return v

class AnomalyItem(BaseModel):
    """Individual anomaly item."""
    
    index: int = Field(..., description="Index in original dataset")
    score: float = Field(..., description="Anomaly score (0.0 to 1.0)")
    features: Dict[str, Any] = Field(..., description="Feature values")
    timestamp: Optional[datetime] = Field(None, description="Timestamp if available")

class DetectionResponse(BaseModel):
    """Response model for anomaly detection."""
    
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    anomalies: List[AnomalyItem] = Field(..., description="Detected anomalies")
    statistics: Dict[str, Any] = Field(..., description="Detection statistics")
    execution_time: float = Field(..., description="Execution time in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "job_456",
                "status": "completed",
                "anomalies": [
                    {
                        "index": 42,
                        "score": 0.95,
                        "features": {"temperature": 120.5, "pressure": 2.1},
                        "timestamp": "2024-12-16T10:30:00Z"
                    }
                ],
                "statistics": {
                    "total_samples": 1000,
                    "anomalies_detected": 1,
                    "contamination_rate": 0.001
                },
                "execution_time": 2.5
            }
        }
```

### CLI Development Patterns

**Command Structure:**
```python
import typer
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from interfaces.cli.utils import (
    create_progress_bar, handle_error, format_output
)

app = typer.Typer(help="Pynomaly CLI - Advanced anomaly detection")
console = Console()

@app.command()
def detect(
    dataset: Path = typer.Argument(
        ...,
        help="Path to dataset file (CSV, JSON, or Parquet)",
        exists=True,
        file_okay=True,
        dir_okay=False
    ),
    algorithm: str = typer.Option(
        "isolation_forest",
        "--algorithm", "-a",
        help="Detection algorithm to use"
    ),
    contamination: float = typer.Option(
        0.1,
        "--contamination", "-c",
        min=0.0,
        max=1.0,
        help="Expected contamination rate"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file for results"
    ),
    format: str = typer.Option(
        "table",
        "--format", "-f",
        help="Output format (table, json, csv)"
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive", "-i",
        help="Run in interactive mode"
    )
):
    """
    Detect anomalies in a dataset.
    
    This command loads a dataset from the specified file and runs
    anomaly detection using the chosen algorithm. Results can be
    displayed in various formats or saved to a file.
    
    Examples:
        pynomaly detect data.csv
        pynomaly detect data.csv --algorithm lof --contamination 0.05
        pynomaly detect data.csv --output results.json --format json
    """
    try:
        if interactive:
            return run_interactive_detection()
        
        # Validate inputs
        if not dataset.exists():
            raise typer.BadParameter(f"Dataset file not found: {dataset}")
        
        # Load dataset with progress
        with create_progress_bar() as progress:
            task = progress.add_task("Loading dataset...", total=100)
            
            # Load data
            data = load_dataset(dataset)
            progress.update(task, advance=30)
            
            # Run detection
            progress.update(task, description="Running detection...")
            result = run_detection(data, algorithm, contamination)
            progress.update(task, advance=60)
            
            # Format results
            progress.update(task, description="Formatting results...")
            formatted_result = format_output(result, format)
            progress.update(task, advance=10)
        
        # Display or save results
        if output:
            save_results(formatted_result, output)
            console.print(f"✅ Results saved to {output}")
        else:
            console.print(formatted_result)
            
    except Exception as e:
        handle_error(e, console)
        raise typer.Exit(1)

def run_interactive_detection():
    """Run interactive detection wizard."""
    from interfaces.cli.interactive import DetectionWizard
    
    wizard = DetectionWizard()
    return wizard.run()
```

**Interactive Components:**
```python
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
from typing import Dict, Any

class DetectionWizard:
    """Interactive wizard for anomaly detection."""
    
    def __init__(self):
        self.console = Console()
        self.config = {}
    
    def run(self) -> Dict[str, Any]:
        """Run the interactive detection wizard."""
        self.console.print(Panel(
            "[bold blue]Pynomaly Interactive Detection Wizard[/bold blue]\n"
            "This wizard will guide you through setting up anomaly detection.",
            title="Welcome"
        ))
        
        # Dataset selection
        self.select_dataset()
        
        # Algorithm selection
        self.select_algorithm()
        
        # Parameter configuration
        self.configure_parameters()
        
        # Execution confirmation
        if self.confirm_execution():
            return self.execute_detection()
        
        return {"cancelled": True}
    
    def select_dataset(self):
        """Interactive dataset selection."""
        dataset_path = Prompt.ask(
            "📁 Enter the path to your dataset file",
            default="data.csv"
        )
        
        # Validate file exists
        from pathlib import Path
        if not Path(dataset_path).exists():
            self.console.print("❌ File not found. Please try again.")
            return self.select_dataset()
        
        self.config['dataset'] = dataset_path
        self.console.print(f"✅ Dataset selected: {dataset_path}")
    
    def select_algorithm(self):
        """Interactive algorithm selection."""
        algorithms = [
            "isolation_forest", "lof", "one_class_svm", "autoencoder"
        ]
        
        # Display algorithm table
        table = Table(title="Available Algorithms")
        table.add_column("Number", style="cyan")
        table.add_column("Algorithm", style="green")
        table.add_column("Description", style="white")
        
        for i, algo in enumerate(algorithms, 1):
            descriptions = {
                "isolation_forest": "Fast, good for high-dimensional data",
                "lof": "Local outlier detection, good for clusters",
                "one_class_svm": "Support vector machines, robust",
                "autoencoder": "Neural network, learns complex patterns"
            }
            table.add_row(str(i), algo, descriptions[algo])
        
        self.console.print(table)
        
        choice = Prompt.ask(
            "🔍 Select an algorithm",
            choices=[str(i) for i in range(1, len(algorithms) + 1)],
            default="1"
        )
        
        selected_algorithm = algorithms[int(choice) - 1]
        self.config['algorithm'] = selected_algorithm
        self.console.print(f"✅ Algorithm selected: {selected_algorithm}")
```

### Web Development Patterns

**HTMX + Tailwind CSS Components:**
```html
<!-- templates/dashboard.html -->
<!DOCTYPE html>
<html lang="en" class="h-full">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pynomaly Dashboard</title>
    <script src="https://unpkg.com/htmx.org@1.9.9"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        'primary': '#3b82f6',
                        'secondary': '#64748b'
                    }
                }
            }
        }
    </script>
</head>
<body class="h-full bg-gray-50">
    <div class="min-h-full">
        <!-- Navigation -->
        <nav class="bg-white shadow">
            <div class="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
                <div class="flex h-16 justify-between">
                    <div class="flex items-center">
                        <h1 class="text-xl font-semibold text-gray-900">
                            Pynomaly Dashboard
                        </h1>
                    </div>
                    <div class="flex items-center space-x-4">
                        <button 
                            class="btn-primary"
                            hx-get="/components/upload-modal"
                            hx-target="#modal-container"
                            hx-trigger="click"
                        >
                            Upload Dataset
                        </button>
                    </div>
                </div>
            </div>
        </nav>

        <!-- Main content -->
        <main class="mx-auto max-w-7xl py-6 sm:px-6 lg:px-8">
            <!-- Stats grid -->
            <div 
                class="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4"
                hx-get="/components/stats"
                hx-trigger="load, every 30s"
            >
                <!-- Stats will be loaded here -->
            </div>

            <!-- Real-time detection results -->
            <div class="mt-8">
                <div class="bg-white shadow rounded-lg">
                    <div class="px-4 py-5 sm:p-6">
                        <h3 class="text-lg font-medium text-gray-900 mb-4">
                            Live Anomaly Detection
                        </h3>
                        <div 
                            id="live-results"
                            class="space-y-4"
                            hx-ext="ws"
                            ws-connect="/ws/live-detection"
                        >
                            <!-- Live results will appear here -->
                        </div>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <!-- Modal container -->
    <div id="modal-container"></div>

    <style>
        .btn-primary {
            @apply bg-primary text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2;
        }
    </style>
</body>
</html>
```

**WebSocket Integration:**
```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict, Any
import json
import asyncio

class WebSocketManager:
    """Manage WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str = None):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_metadata[websocket] = {
            "user_id": user_id,
            "connected_at": datetime.utcnow(),
            "subscriptions": set()
        }
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            del self.connection_metadata[websocket]
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send message to specific connection."""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception:
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any], subscription: str = None):
        """Broadcast message to all or filtered connections."""
        if not self.active_connections:
            return
        
        # Filter connections by subscription if specified
        target_connections = self.active_connections
        if subscription:
            target_connections = [
                ws for ws in self.active_connections
                if subscription in self.connection_metadata[ws]["subscriptions"]
            ]
        
        # Send to all target connections
        disconnected = []
        for connection in target_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

# WebSocket endpoint
websocket_manager = WebSocketManager()

@app.websocket("/ws/live-detection")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    
    try:
        while True:
            # Receive subscription preferences
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "subscribe":
                subscriptions = message.get("subscriptions", [])
                websocket_manager.connection_metadata[websocket]["subscriptions"].update(subscriptions)
                
                await websocket_manager.send_personal_message({
                    "type": "subscription_confirmed",
                    "subscriptions": list(subscriptions)
                }, websocket)
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
```

## Testing Requirements

### Test Categories

1. **Unit Tests**: Test individual components (API endpoints, CLI commands, UI components)
2. **Integration Tests**: Test interface interactions with services
3. **End-to-End Tests**: Test complete user workflows
4. **Load Tests**: Test performance under load
5. **Accessibility Tests**: Test web interface accessibility
6. **Browser Tests**: Cross-browser compatibility testing

### Test Structure

```bash
tests/
├── unit/                    # Unit tests
│   ├── api/                # API endpoint tests
│   ├── cli/                # CLI command tests
│   ├── web/                # Web component tests
│   └── sdk/                # SDK functionality tests
├── integration/            # Integration tests
│   ├── api_integration/    # API with services integration
│   ├── cli_integration/    # CLI with backend integration
│   └── websocket/         # WebSocket integration tests
├── e2e/                   # End-to-end tests
│   ├── api_workflows/     # Complete API workflows
│   ├── web_workflows/     # Complete web user journeys
│   └── cli_workflows/     # Complete CLI workflows
├── load/                  # Load and performance tests
├── accessibility/         # Accessibility compliance tests
└── fixtures/              # Test data and fixtures
```

### Testing Requirements

- **Coverage**: Minimum 90% code coverage for interfaces
- **Response Time**: API endpoints under 200ms for simple operations
- **Accessibility**: WCAG 2.1 AA compliance for web interfaces
- **Cross-Browser**: Support for latest 2 versions of major browsers
- **Mobile**: Responsive design testing for mobile devices

### API Testing

```python
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from interfaces.api.main import app

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)

@pytest.fixture
async def async_client():
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

class TestDetectionAPI:
    """Test detection API endpoints."""
    
    def test_detect_anomalies_success(self, client, mock_detection_service):
        """Test successful anomaly detection."""
        request_data = {
            "dataset_id": "test_dataset",
            "algorithm": "isolation_forest",
            "parameters": {"contamination": 0.1}
        }
        
        response = client.post("/api/v1/detection/detect", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert "anomalies" in data
        assert data["status"] == "completed"
    
    def test_detect_anomalies_validation_error(self, client):
        """Test validation error handling."""
        request_data = {
            "dataset_id": "",  # Invalid empty dataset_id
            "algorithm": "invalid_algorithm"
        }
        
        response = client.post("/api/v1/detection/detect", json=request_data)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, async_client):
        """Test WebSocket connection."""
        async with async_client.websocket_connect("/ws/live-detection") as websocket:
            # Send subscription message
            await websocket.send_json({
                "type": "subscribe",
                "subscriptions": ["anomaly_detection"]
            })
            
            # Receive confirmation
            data = await websocket.receive_json()
            assert data["type"] == "subscription_confirmed"
            assert "anomaly_detection" in data["subscriptions"]
```

### CLI Testing

```python
import pytest
from typer.testing import CliRunner
from pathlib import Path

from interfaces.cli.main import app

@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()

@pytest.fixture
def sample_dataset(tmp_path):
    """Create sample dataset file."""
    dataset_file = tmp_path / "test_data.csv"
    dataset_file.write_text("feature1,feature2\n1,2\n3,4\n100,200\n")
    return dataset_file

class TestCLI:
    """Test CLI commands."""
    
    def test_detect_command_success(self, runner, sample_dataset):
        """Test successful detection command."""
        result = runner.invoke(app, [
            "detect",
            str(sample_dataset),
            "--algorithm", "isolation_forest",
            "--contamination", "0.1"
        ])
        
        assert result.exit_code == 0
        assert "✅" in result.stdout  # Success indicator
        assert "anomalies" in result.stdout.lower()
    
    def test_detect_command_file_not_found(self, runner):
        """Test detection with non-existent file."""
        result = runner.invoke(app, [
            "detect",
            "nonexistent.csv"
        ])
        
        assert result.exit_code != 0
        assert "not found" in result.stdout.lower()
    
    def test_detect_interactive_mode(self, runner, sample_dataset, monkeypatch):
        """Test interactive detection mode."""
        # Mock user inputs
        inputs = iter([
            str(sample_dataset),  # Dataset path
            "1",                  # Algorithm choice (isolation_forest)
            "0.1",               # Contamination rate
            "y"                  # Confirm execution
        ])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        
        result = runner.invoke(app, ["detect", "--interactive"])
        
        assert result.exit_code == 0
        assert "Welcome" in result.stdout
```

### Web Testing

```python
import pytest
from playwright.async_api import async_playwright
import asyncio

@pytest.fixture
async def browser():
    """Create browser for testing."""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        yield browser
        await browser.close()

@pytest.fixture
async def page(browser):
    """Create page for testing."""
    page = await browser.new_page()
    yield page
    await page.close()

class TestWebInterface:
    """Test web interface functionality."""
    
    @pytest.mark.asyncio
    async def test_dashboard_loads(self, page):
        """Test dashboard page loads correctly."""
        await page.goto("http://localhost:3000/dashboard")
        
        # Check page title
        title = await page.title()
        assert "Pynomaly Dashboard" in title
        
        # Check main heading
        heading = await page.text_content("h1")
        assert "Pynomaly Dashboard" in heading
        
        # Check upload button exists
        upload_button = await page.query_selector("button:has-text('Upload Dataset')")
        assert upload_button is not None
    
    @pytest.mark.asyncio
    async def test_file_upload_workflow(self, page, sample_dataset):
        """Test complete file upload workflow."""
        await page.goto("http://localhost:3000/dashboard")
        
        # Click upload button
        await page.click("button:has-text('Upload Dataset')")
        
        # Wait for modal to appear
        await page.wait_for_selector("[data-testid='upload-modal']")
        
        # Upload file
        await page.set_input_files("input[type='file']", str(sample_dataset))
        
        # Submit upload
        await page.click("button:has-text('Upload')")
        
        # Wait for success message
        await page.wait_for_selector(".success-message")
        success_text = await page.text_content(".success-message")
        assert "successfully uploaded" in success_text.lower()
    
    @pytest.mark.asyncio
    async def test_accessibility_compliance(self, page):
        """Test accessibility compliance."""
        await page.goto("http://localhost:3000/dashboard")
        
        # Check for alt text on images
        images = await page.query_selector_all("img")
        for img in images:
            alt_text = await img.get_attribute("alt")
            assert alt_text is not None and alt_text.strip() != ""
        
        # Check for proper heading hierarchy
        headings = await page.query_selector_all("h1, h2, h3, h4, h5, h6")
        assert len(headings) > 0  # Should have at least one heading
        
        # Check for form labels
        inputs = await page.query_selector_all("input[type='text'], input[type='email']")
        for input_elem in inputs:
            # Check for associated label or aria-label
            label = await page.query_selector(f"label[for='{await input_elem.get_attribute('id')}']")
            aria_label = await input_elem.get_attribute("aria-label")
            assert label is not None or aria_label is not None
```

## Documentation Standards

### API Documentation

- **OpenAPI Specifications**: Comprehensive OpenAPI 3.0 documentation
- **Interactive Docs**: FastAPI automatic interactive documentation
- **Code Examples**: Examples in multiple programming languages
- **Error Responses**: Document all possible error responses
- **Authentication**: Clear authentication requirements and examples

### User Documentation

- **Getting Started Guide**: Step-by-step setup and first usage
- **API Reference**: Complete endpoint documentation with examples
- **CLI Reference**: All commands with usage examples and options
- **Web User Guide**: Screenshots and workflows for web interface
- **SDK Documentation**: Client library documentation and examples

### Interface Documentation

```python
from fastapi import APIRouter
from typing import List

router = APIRouter()

@router.post(
    "/detect",
    response_model=DetectionResponse,
    summary="Detect anomalies in dataset",
    description="""
    Run anomaly detection on a dataset using the specified algorithm.
    
    This endpoint processes the provided dataset and returns identified
    anomalies along with their confidence scores and relevant statistics.
    
    **Algorithm Options:**
    - `isolation_forest`: Fast algorithm suitable for high-dimensional data
    - `lof`: Local Outlier Factor, good for clustered data
    - `one_class_svm`: Support Vector Machine approach, robust to outliers
    - `autoencoder`: Neural network approach for complex patterns
    
    **Rate Limiting:**
    - 100 requests per minute per user
    - Large datasets may have longer processing times
    
    **Examples:**
    
    Basic detection:
    ```python
    import requests
    
    response = requests.post("/api/v1/detection/detect", json={
        "dataset_id": "my_dataset",
        "algorithm": "isolation_forest"
    })
    ```
    
    Advanced configuration:
    ```python
    response = requests.post("/api/v1/detection/detect", json={
        "dataset_id": "my_dataset",
        "algorithm": "isolation_forest",
        "parameters": {
            "contamination": 0.05,
            "n_estimators": 200,
            "random_state": 42
        }
    })
    ```
    """,
    responses={
        200: {
            "description": "Detection completed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "job_id": "job_123",
                        "status": "completed",
                        "anomalies": [
                            {
                                "index": 42,
                                "score": 0.95,
                                "features": {"temp": 120.5, "pressure": 2.1}
                            }
                        ],
                        "statistics": {
                            "total_samples": 1000,
                            "anomalies_detected": 1,
                            "contamination_rate": 0.001
                        },
                        "execution_time": 2.5
                    }
                }
            }
        },
        400: {"description": "Invalid request data"},
        401: {"description": "Authentication required"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"}
    },
    tags=["Detection"]
)
async def detect_anomalies(request: DetectionRequest):
    """Implementation here"""
    pass
```

## Pull Request Process

### Before Submitting

1. **Test Coverage**: Ensure comprehensive test coverage
2. **API Documentation**: Update OpenAPI specifications
3. **User Documentation**: Update user guides and examples
4. **Accessibility**: Verify web interface accessibility
5. **Performance**: Run load tests for API changes
6. **Cross-Browser**: Test web changes across browsers

### Pull Request Template

```markdown
## Description
Brief description of interface changes and user impact.

## Type of Change
- [ ] New API endpoint
- [ ] CLI command enhancement
- [ ] Web interface improvement
- [ ] SDK functionality addition
- [ ] Bug fix
- [ ] Performance improvement
- [ ] Documentation update

## Interface Components Affected
- [ ] REST API
- [ ] Command Line Interface
- [ ] Web Application
- [ ] Python SDK
- [ ] JavaScript SDK
- [ ] WebSocket API
- [ ] GraphQL API

## User Experience Impact
- [ ] No user-facing changes
- [ ] Improved usability
- [ ] New functionality
- [ ] Breaking change (requires migration guide)

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests run
- [ ] End-to-end tests pass
- [ ] Load tests pass (if applicable)
- [ ] Accessibility tests pass
- [ ] Cross-browser tests pass

## Documentation
- [ ] API documentation updated
- [ ] CLI help text updated
- [ ] User guide updated
- [ ] SDK documentation updated
- [ ] Examples provided
- [ ] Migration guide (if breaking change)

## Security Considerations
- [ ] Input validation implemented
- [ ] Authentication/authorization verified
- [ ] CORS configuration reviewed
- [ ] Rate limiting appropriate
- [ ] No sensitive data exposed

## Performance Impact
- [ ] Response times measured
- [ ] Memory usage assessed
- [ ] Concurrent user capacity tested
- [ ] Database query optimization verified
```

## User Experience Standards

### Design Principles

- **Accessibility First**: WCAG 2.1 AA compliance
- **Mobile Responsive**: Works on all device sizes
- **Performance**: Fast loading and responsive interactions
- **Consistency**: Consistent design patterns across interfaces
- **Error Handling**: Clear, helpful error messages

### Interface Guidelines

**API Design:**
- RESTful principles
- Consistent naming conventions
- Proper HTTP status codes
- Comprehensive error responses
- Pagination for large datasets

**CLI Design:**
- Intuitive command structure
- Rich help documentation
- Progress indicators for long operations
- Colored output for better readability
- Interactive modes for complex operations

**Web Design:**
- Clean, modern interface
- Responsive design
- Real-time updates
- Keyboard navigation support
- Screen reader compatibility

## Community

### Communication Channels

- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for interface design questions
- **Slack**: #interfaces-dev channel for real-time discussion
- **Email**: interfaces-team@yourorg.com for design consultations

### Interface Expertise Areas

- **API Design**: REST/GraphQL API architecture and best practices
- **CLI Development**: Command-line interface design and usability
- **Web Development**: Frontend development and user experience
- **SDK Development**: Client library design and documentation
- **Accessibility**: Web accessibility and inclusive design

### Getting Help

1. **Design Questions**: Post in GitHub Discussions with "design" label
2. **Technical Issues**: Create GitHub Issues with relevant labels
3. **UX Feedback**: Use user testing channels and feedback forms
4. **Performance Issues**: Include benchmarks and profiling data

Thank you for contributing to the Interfaces package! Your contributions help create better user experiences for the entire Pynomaly community.