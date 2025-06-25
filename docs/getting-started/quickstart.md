# Quick Start Guide

Get started with anomaly detection in minutes using Pynomaly's modern development stack.

## Prerequisites

- **Python 3.11+** (3.12 recommended)
- **Hatch** for environment management (recommended)
- **Git** for cloning the repository

## Installation Methods

### Method 1: Hatch (Recommended)

```bash
# Install Hatch (one-time setup)
pip install hatch

# Clone and setup
git clone https://github.com/yourusername/pynomaly.git
cd pynomaly

# Automated setup
make setup              # Install Hatch and create environments
make dev-install        # Install in development mode
make test               # Verify installation
```

### Method 2: Traditional pip

```bash
# Clone and setup
git clone https://github.com/yourusername/pynomaly.git
cd pynomaly

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install with features
pip install -e ".[server]"  # API + CLI + basic ML

# Test installation
pynomaly --help
```

### Method 3: Simple pip Setup

```bash
# Automated setup script (handles environment issues)
python scripts/setup_simple.py
```

## Basic Workflow

The typical anomaly detection workflow consists of:

1. **Load Data** - Import your dataset
2. **Create Detector** - Choose an algorithm
3. **Train** - Fit the detector to your data
4. **Detect** - Find anomalies
5. **Analyze** - Review results

## Using the CLI

### Step 1: Load Your Data

```bash
# Primary method (recommended)
pynomaly dataset load data.csv --name "Sales Data"

# Load with specific options
pynomaly dataset load transactions.csv \
  --name "Transactions" \
  --target fraud_label \
  --sample 10000

# Alternative methods
python scripts/pynomaly_cli.py dataset load data.csv --name "Sales Data"
hatch env run cli:run dataset load data.csv --name "Sales Data"
```

### Step 2: Create a Detector

```bash
# List available algorithms
pynomaly detector algorithms

# Create an Isolation Forest detector
pynomaly detector create \
  --name "Fraud Detector" \
  --algorithm IsolationForest \
  --contamination 0.05

# Create with custom parameters
pynomaly detector create \
  --name "Custom LOF" \
  --algorithm LOF \
  --contamination 0.1 \
  --description "Local Outlier Factor for transaction anomalies"
```

### Step 3: Train the Detector

```bash
# List your detectors and datasets
pynomaly detector list
pynomaly dataset list

# Train using IDs (first 8 characters are enough)
pynomaly detect train <detector_id> <dataset_id>

# Example with partial IDs
pynomaly detect train a1b2c3d4 e5f6g7h8
```

### Step 4: Run Detection

```bash
# Detect anomalies in the same dataset
pynomaly detect run <detector_id> <dataset_id>

# Detect on new data
pynomaly dataset load new_data.csv --name "New Data"
pynomaly detect run <detector_id> <new_dataset_id>

# Export results
pynomaly detect run <detector_id> <dataset_id> \
  --output results.csv
```

### Step 5: View Results

```bash
# Show recent results
pynomaly detect results --latest

# Show results for specific detector
pynomaly detect results --detector <detector_id>

# Start web UI to visualize
# Method 1: Using Hatch
make prod-api-dev

# Method 2: Traditional CLI
pynomaly server start

# Method 3: Direct Hatch command
hatch env run prod:serve-api

# Access at: http://localhost:8000
# API docs: http://localhost:8000/docs
```

## Using the Python API

### Basic Example

```python
import pandas as pd
from pynomaly.infrastructure.config import create_container
from pynomaly.domain.entities import Detector, Dataset
from pynomaly.application.use_cases import (
    TrainDetectorRequest,
    DetectAnomaliesRequest
)

# Initialize container
container = create_container()

# 1. Create detector
detector = Detector(
    name="Python API Detector",
    algorithm="IsolationForest",
    parameters={"contamination": 0.1, "random_state": 42}
)
detector_repo = container.detector_repository()
detector_repo.save(detector)

# 2. Load data
data = pd.read_csv("your_data.csv")
dataset = Dataset(name="My Data", data=data)
dataset_repo = container.dataset_repository()
dataset_repo.save(dataset)

# 3. Train detector
train_use_case = container.train_detector_use_case()
train_request = TrainDetectorRequest(
    detector_id=detector.id,
    dataset=dataset,
    validate_data=True,
    save_model=True
)

import asyncio
train_response = asyncio.run(train_use_case.execute(train_request))
print(f"Training completed in {train_response.training_time_ms}ms")

# 4. Detect anomalies
detect_use_case = container.detect_anomalies_use_case()
detect_request = DetectAnomaliesRequest(
    detector_id=detector.id,
    dataset=dataset,
    validate_features=True,
    save_results=True
)

detect_response = asyncio.run(detect_use_case.execute(detect_request))
result = detect_response.result

print(f"Found {result.n_anomalies} anomalies ({result.anomaly_rate:.1%})")
print(f"Anomaly indices: {result.anomaly_indices[:10]}...")  # First 10
```

### Using Different Algorithms

```python
# Statistical methods
detectors = [
    Detector(name="IF", algorithm="IsolationForest", 
             parameters={"contamination": 0.1}),
    Detector(name="LOF", algorithm="LOF", 
             parameters={"n_neighbors": 20, "contamination": 0.1}),
    Detector(name="OCSVM", algorithm="OCSVM", 
             parameters={"nu": 0.1, "kernel": "rbf"}),
]

# Train and compare
for detector in detectors:
    detector_repo.save(detector)
    
    # Train
    train_request = TrainDetectorRequest(
        detector_id=detector.id,
        dataset=dataset,
        validate_data=True,
        save_model=True
    )
    await train_use_case.execute(train_request)
    
    # Detect
    detect_request = DetectAnomaliesRequest(
        detector_id=detector.id,
        dataset=dataset,
        validate_features=True,
        save_results=True
    )
    response = await detect_use_case.execute(detect_request)
    
    print(f"{detector.name}: {response.result.n_anomalies} anomalies")
```

### Ensemble Detection

```python
# Use multiple detectors together
ensemble_service = container.ensemble_service()

# Create ensemble result
ensemble_result = await ensemble_service.detect_with_ensemble(
    detector_ids=[d.id for d in detectors],
    dataset=dataset,
    aggregation_method="average"  # or "voting", "maximum"
)

print(f"Ensemble found {ensemble_result.n_anomalies} anomalies")
```

## Using the Web Interface

### Start the Server

```bash
# Hatch methods (recommended)
make prod-api           # Production server
make prod-api-dev       # Development server with reload

# Direct Hatch commands
hatch env run prod:serve-api-prod    # Production with workers
hatch env run prod:serve-api         # Development mode

# Traditional methods
pynomaly server start               # If installed with pip
python scripts/pynomaly_cli.py server start  # Alternative CLI

# Direct uvicorn
uvicorn pynomaly.presentation.api.app:app --reload
```

### Access the UI

1. **Main Application**: http://localhost:8000
2. **API Documentation**: http://localhost:8000/docs  
3. **Progressive Web App**: http://localhost:8000/app
4. **Health Check**: http://localhost:8000/api/health

### Web Interface Features

- **📊 Real-time Dashboard** - Live anomaly detection with WebSocket updates
- **🎯 Interactive Visualizations** - D3.js custom charts and Apache ECharts
- **📱 Progressive Web App** - Install on desktop and mobile like a native app
- **⚡ HTMX Simplicity** - Server-side rendering with minimal JavaScript
- **🎨 Modern UI** - Tailwind CSS for responsive, accessible design
- **🔄 Offline Capability** - Service worker enables offline operation
- **📈 Experiment Tracking** - Compare models, track performance metrics
- **🔍 Dataset Analysis** - Data quality reports, drift detection

### Navigation

- **Dashboard** - Overview and recent results
- **Detectors** - Manage algorithms and parameters
- **Datasets** - Upload, explore, and validate data
- **Detection** - Run anomaly detection workflows
- **Experiments** - Track and compare model performance
- **Visualizations** - Interactive charts and analysis
- **Settings** - Configuration and preferences

## Example: Credit Card Fraud Detection

```python
# Complete example for fraud detection
import pandas as pd
from pynomaly.infrastructure.config import create_container
from pynomaly.domain.entities import Detector, Dataset

# Setup
container = create_container()

# Load credit card transactions
transactions = pd.read_csv("creditcard.csv")
dataset = Dataset(
    name="Credit Card Transactions",
    data=transactions.drop(columns=["Class"]),  # Remove labels for unsupervised
    target_column="Class",  # Keep reference for evaluation
    metadata={"source": "kaggle", "type": "fraud_detection"}
)
container.dataset_repository().save(dataset)

# Create specialized detector
fraud_detector = Detector(
    name="Fraud Detector",
    algorithm="IsolationForest",
    parameters={
        "contamination": 0.001,  # Expect 0.1% fraud
        "n_estimators": 200,     # More trees for stability
        "max_samples": "auto",
        "random_state": 42
    }
)
container.detector_repository().save(fraud_detector)

# Train and detect
# ... (training code as above)

# Evaluate if labels available
if dataset.has_target:
    evaluate_use_case = container.evaluate_model_use_case()
    from pynomaly.application.use_cases import EvaluateModelRequest
    
    eval_request = EvaluateModelRequest(
        detector_id=fraud_detector.id,
        test_dataset=dataset,
        metrics=["precision", "recall", "f1", "roc_auc"]
    )
    
    eval_response = await evaluate_use_case.execute(eval_request)
    print(f"Performance metrics: {eval_response.metrics}")
```

## Development Workflow

### Daily Development Commands

```bash
# Code quality and testing
make format             # Auto-format code with Ruff
make test               # Run core tests
make lint               # Check code quality
make ci                 # Full CI pipeline locally

# Environment management
make status             # Show project status
make env-show           # List environments
make clean              # Clean build artifacts
```

### Building and Deployment

```bash
# Build package
make build              # Build wheel and source distribution
make version            # Show current version

# Docker deployment
make docker             # Build Docker image

# Production deployment
hatch env run prod:serve-api-prod  # Production server with workers
```

## Next Steps

### Learn More
- **[Development Guide](../development/README.md)** - Modern development workflow
- **[Hatch Guide](../development/HATCH_GUIDE.md)** - Detailed Hatch usage
- **[Available Algorithms](../guides/algorithms.md)** - Algorithm selection guide
- **[Data Processing](../guides/datasets.md)** - Data preparation and validation

### Advanced Usage
- **[Production Deployment](../deployment/PRODUCTION_DEPLOYMENT_GUIDE.md)** - Deploy to production
- **[Docker Guide](../deployment/DOCKER_DEPLOYMENT_GUIDE.md)** - Containerized deployment
- **[API Reference](../api/rest-api.md)** - Complete API documentation
- **[Troubleshooting](../guides/troubleshooting.md)** - Common issues and solutions

### Integrations
- **[Business Intelligence](../../README.md#business-intelligence-integrations)** - Export to Excel, Power BI, Google Sheets
- **[Monitoring](../guides/monitoring.md)** - Observability and metrics
- **[Security](../deployment/SECURITY.md)** - Security best practices