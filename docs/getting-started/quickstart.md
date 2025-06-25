# Quick Start Guide

Get started with anomaly detection in minutes using Pynomaly.

## Prerequisites

Make sure you have Python 3.11+ installed. No need for Poetry, Docker, or Make!

### Simple Setup (Python + pip only)

```bash
# Clone and setup
git clone https://github.com/pynomaly/pynomaly.git
cd pynomaly
python scripts/setup_simple.py

# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Test installation
pynomaly --help
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

# Alternative method
python scripts/cli.py dataset load data.csv --name "Sales Data"
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
pynomaly server start
# Open http://localhost:8000
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
# Using simple Python CLI
python cli.py server start

# Or if package is installed
pynomaly server start

# Start with custom host/port
python cli.py server start --host 0.0.0.0 --port 8080

# Start with auto-reload for development
python cli.py server start --reload

# Or run directly with uvicorn
python -m uvicorn pynomaly.presentation.api.app:app --reload
```

### Access the UI

1. Open http://localhost:8000 in your browser
2. Navigate through:
   - **Dashboard** - Overview and recent results
   - **Detectors** - Manage algorithms
   - **Datasets** - Upload and explore data
   - **Detection** - Run anomaly detection
   - **Visualizations** - Interactive charts

### Key Features

- **Real-time Updates** - HTMX-powered dynamic content
- **Interactive Charts** - D3.js and ECharts visualizations
- **Experiment Tracking** - Compare model performance
- **Progressive Web App** - Install as desktop app

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

## Next Steps

- Explore [available algorithms](../guide/algorithms.md)
- Learn about [data preprocessing](../guide/datasets.md)
- Set up [experiment tracking](../guide/experiments.md)
- Deploy to [production](../advanced/deployment.md)