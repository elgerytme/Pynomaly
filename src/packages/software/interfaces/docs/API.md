# Interfaces API Reference

## Overview

The Interfaces package provides all user-facing APIs and interfaces for the Pynomaly platform. This includes REST APIs, CLI commands, web interfaces, and client SDKs.

## REST API

### Base URL

```
http://localhost:8000/api/v1
```

### Authentication

All API endpoints require authentication via JWT tokens:

```http
Authorization: Bearer <jwt_token>
```

### Common Response Format

```json
{
  "success": true,
  "data": {},
  "message": "Operation completed successfully",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Error Response Format

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {}
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Detection API

### POST /api/v1/detection/detect

Run anomaly detection on a dataset.

**Request:**
```json
{
  "dataset_id": "string",
  "algorithm": "isolation_forest",
  "parameters": {
    "contamination": 0.1,
    "n_estimators": 100
  },
  "preprocessing": {
    "normalize": true,
    "handle_missing": "mean"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "detection_id": "det_123",
    "anomaly_count": 45,
    "anomaly_rate": 0.045,
    "processing_time": 2.34,
    "anomalies": [
      {
        "id": "anom_001",
        "score": 0.95,
        "index": 123,
        "features": {
          "temperature": 85.2,
          "pressure": 1.2
        }
      }
    ]
  }
}
```

### POST /api/v1/detection/batch

Run batch anomaly detection on multiple datasets.

**Request:**
```json
{
  "datasets": [
    {
      "dataset_id": "dataset_001",
      "algorithm": "isolation_forest"
    },
    {
      "dataset_id": "dataset_002", 
      "algorithm": "lof"
    }
  ],
  "parallel": true
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "batch_id": "batch_456",
    "results": [
      {
        "dataset_id": "dataset_001",
        "detection_id": "det_124",
        "status": "completed",
        "anomaly_count": 23
      }
    ]
  }
}
```

### GET /api/v1/detection/results/{detection_id}

Get detection results by ID.

**Response:**
```json
{
  "success": true,
  "data": {
    "detection_id": "det_123",
    "dataset_id": "dataset_001",
    "algorithm": "isolation_forest",
    "status": "completed",
    "created_at": "2024-01-01T00:00:00Z",
    "completed_at": "2024-01-01T00:00:02Z",
    "anomaly_count": 45,
    "anomaly_rate": 0.045,
    "processing_time": 2.34,
    "anomalies": [...]
  }
}
```

### WebSocket /ws/detection/live

Real-time anomaly detection streaming.

**Connect:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/detection/live');
```

**Send Data:**
```json
{
  "type": "detect",
  "data": {
    "algorithm": "isolation_forest",
    "sample": [1.2, 3.4, 5.6, 7.8]
  }
}
```

**Receive Result:**
```json
{
  "type": "anomaly_result",
  "data": {
    "is_anomaly": true,
    "score": 0.92,
    "timestamp": "2024-01-01T00:00:00Z"
  }
}
```

## Dataset API

### GET /api/v1/datasets

List all datasets.

**Query Parameters:**
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 20)
- `search`: Search query
- `sort`: Sort field (name, created_at, size)
- `order`: Sort order (asc, desc)

**Response:**
```json
{
  "success": true,
  "data": {
    "datasets": [
      {
        "id": "dataset_001",
        "name": "sensor_data",
        "size": 10000,
        "features": 5,
        "created_at": "2024-01-01T00:00:00Z",
        "metadata": {
          "source": "production"
        }
      }
    ],
    "pagination": {
      "page": 1,
      "limit": 20,
      "total": 100,
      "pages": 5
    }
  }
}
```

### POST /api/v1/datasets

Create a new dataset.

**Request:**
```json
{
  "name": "sensor_data",
  "description": "Production sensor readings",
  "data": [
    [1.2, 3.4, 5.6, 7.8, 9.0],
    [2.1, 4.3, 6.5, 8.7, 0.9]
  ],
  "features": ["temp", "pressure", "humidity", "vibration", "voltage"],
  "metadata": {
    "source": "production",
    "collection_date": "2024-01-01"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "dataset_id": "dataset_002",
    "name": "sensor_data",
    "size": 2,
    "features": 5,
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

### GET /api/v1/datasets/{dataset_id}

Get dataset by ID.

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "dataset_001",
    "name": "sensor_data",
    "description": "Production sensor readings",
    "size": 10000,
    "features": 5,
    "feature_names": ["temp", "pressure", "humidity", "vibration", "voltage"],
    "created_at": "2024-01-01T00:00:00Z",
    "metadata": {
      "source": "production"
    },
    "statistics": {
      "mean": [25.5, 1.2, 45.6, 0.02, 12.5],
      "std": [5.2, 0.1, 8.9, 0.005, 2.1]
    }
  }
}
```

### PUT /api/v1/datasets/{dataset_id}

Update dataset.

**Request:**
```json
{
  "name": "updated_sensor_data",
  "description": "Updated description",
  "metadata": {
    "source": "production",
    "updated": true
  }
}
```

### DELETE /api/v1/datasets/{dataset_id}

Delete dataset.

**Response:**
```json
{
  "success": true,
  "message": "Dataset deleted successfully"
}
```

### POST /api/v1/datasets/{dataset_id}/upload

Upload dataset from file.

**Request:**
```multipart/form-data
file: <csv_file>
separator: ","
has_header: true
```

**Response:**
```json
{
  "success": true,
  "data": {
    "rows_processed": 10000,
    "features_detected": 5,
    "preview": [
      [1.2, 3.4, 5.6, 7.8, 9.0]
    ]
  }
}
```

## Model API

### GET /api/v1/models

List all trained models.

**Response:**
```json
{
  "success": true,
  "data": {
    "models": [
      {
        "id": "model_001",
        "name": "isolation_forest_v1",
        "algorithm": "isolation_forest",
        "status": "trained",
        "dataset_id": "dataset_001",
        "created_at": "2024-01-01T00:00:00Z",
        "metrics": {
          "accuracy": 0.95,
          "precision": 0.92,
          "recall": 0.88
        }
      }
    ]
  }
}
```

### POST /api/v1/models/train

Train a new model.

**Request:**
```json
{
  "name": "isolation_forest_v2",
  "algorithm": "isolation_forest",
  "dataset_id": "dataset_001",
  "parameters": {
    "contamination": 0.1,
    "n_estimators": 100,
    "max_samples": "auto"
  },
  "validation": {
    "split": 0.2,
    "cross_validation": 5
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "model_id": "model_002",
    "name": "isolation_forest_v2",
    "status": "training",
    "estimated_time": 120
  }
}
```

### GET /api/v1/models/{model_id}

Get model details.

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "model_001",
    "name": "isolation_forest_v1",
    "algorithm": "isolation_forest",
    "status": "trained",
    "dataset_id": "dataset_001",
    "created_at": "2024-01-01T00:00:00Z",
    "training_time": 45.6,
    "parameters": {
      "contamination": 0.1,
      "n_estimators": 100
    },
    "metrics": {
      "accuracy": 0.95,
      "precision": 0.92,
      "recall": 0.88,
      "f1_score": 0.90
    }
  }
}
```

### POST /api/v1/models/{model_id}/predict

Make predictions using a trained model.

**Request:**
```json
{
  "data": [
    [1.2, 3.4, 5.6, 7.8, 9.0],
    [2.1, 4.3, 6.5, 8.7, 0.9]
  ]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "predictions": [1, 0],
    "scores": [0.95, 0.23],
    "processing_time": 0.05
  }
}
```

## CLI API

### PynomaCLI Class

Main CLI application class.

```python
from pynomaly.interfaces.cli import PynomaCLI

cli = PynomaCLI()
cli.run(["detect", "--dataset", "data.csv"])
```

#### Methods

```python
def run(self, args: List[str]) -> int:
    """Run CLI command with arguments."""
    
def interactive(self) -> None:
    """Start interactive mode."""
    
def version(self) -> str:
    """Get version information."""
```

### CLI Commands

#### detect

Run anomaly detection on a dataset.

```bash
pynomaly detect --dataset data.csv --algorithm isolation_forest --output results.json
```

**Options:**
- `--dataset, -d`: Input dataset file
- `--algorithm, -a`: Detection algorithm (default: isolation_forest)
- `--contamination, -c`: Contamination rate (default: 0.1)
- `--output, -o`: Output file (default: stdout)
- `--format, -f`: Output format (json, csv, table)

#### train

Train a new model.

```bash
pynomaly train --dataset data.csv --algorithm isolation_forest --name my_model
```

**Options:**
- `--dataset, -d`: Training dataset file
- `--algorithm, -a`: Algorithm to train
- `--name, -n`: Model name
- `--validate`: Enable validation
- `--save-model`: Save trained model

#### predict

Make predictions using a trained model.

```bash
pynomaly predict --model my_model --data new_data.csv
```

**Options:**
- `--model, -m`: Model name or ID
- `--data, -d`: Input data file
- `--output, -o`: Output file
- `--batch-size`: Batch size for processing

#### dataset

Manage datasets.

```bash
# List datasets
pynomaly dataset list

# Create dataset
pynomaly dataset create --name sensor_data --file data.csv

# Show dataset info
pynomaly dataset info --id dataset_001

# Delete dataset
pynomaly dataset delete --id dataset_001
```

#### model

Manage models.

```bash
# List models
pynomaly model list

# Show model info
pynomaly model info --id model_001

# Delete model
pynomaly model delete --id model_001
```

#### config

Manage configuration.

```bash
# Show current config
pynomaly config show

# Set configuration value
pynomaly config set api.host localhost

# Reset configuration
pynomaly config reset
```

#### Interactive Mode

```bash
pynomaly --interactive
```

Interactive mode provides:
- Guided setup wizard
- Parameter selection help
- Progress indicators
- Error handling assistance

## Web Interface API

### WebApplication Class

Main web application class.

```python
from pynomaly.interfaces.web import WebApplication

app = WebApplication()
app.run(host="0.0.0.0", port=8080)
```

#### Methods

```python
def run(self, host: str = "localhost", port: int = 8080) -> None:
    """Start web server."""
    
def add_custom_route(self, path: str, handler: Callable) -> None:
    """Add custom route handler."""
    
def enable_websockets(self) -> None:
    """Enable WebSocket support."""
```

### Web Routes

#### Dashboard

- `GET /` - Main dashboard
- `GET /dashboard` - Detailed dashboard
- `GET /datasets` - Dataset management
- `GET /models` - Model management
- `GET /detection` - Detection interface

#### API Integration

- `GET /api/docs` - API documentation
- `GET /api/health` - Health check
- `POST /api/upload` - File upload interface

### WebSocket Manager

```python
from pynomaly.interfaces.web.websockets import WebSocketManager

manager = WebSocketManager()
```

#### Methods

```python
async def connect(self, websocket: WebSocket) -> None:
    """Connect new WebSocket client."""
    
async def disconnect(self, websocket: WebSocket) -> None:
    """Disconnect WebSocket client."""
    
async def broadcast(self, message: dict) -> None:
    """Broadcast message to all clients."""
    
async def send_personal(self, websocket: WebSocket, message: dict) -> None:
    """Send message to specific client."""
```

## Python SDK

### PynomalyClient

Main Python SDK client.

```python
from pynomaly.interfaces.sdk.python import PynomalyClient

client = PynomalyClient(base_url="http://localhost:8000")
```

#### Authentication

```python
# Login with credentials
await client.login(username="admin", password="password")

# Use API key
client = PynomalyClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)
```

#### Methods

```python
async def detect_anomalies(
    self,
    dataset_id: str,
    algorithm: str = "isolation_forest",
    parameters: dict = None
) -> DetectionResult:
    """Run anomaly detection."""
    
async def create_dataset(
    self,
    name: str,
    data: List[List[float]],
    features: List[str] = None
) -> str:
    """Create new dataset."""
    
async def train_model(
    self,
    name: str,
    algorithm: str,
    dataset_id: str,
    parameters: dict = None
) -> str:
    """Train new model."""
    
async def get_model(self, model_id: str) -> Model:
    """Get model by ID."""
    
async def predict(
    self,
    model_id: str,
    data: List[List[float]]
) -> List[float]:
    """Make predictions."""
```

### AsyncClient

Async version of the Python client.

```python
from pynomaly.interfaces.sdk.python import AsyncClient

async with AsyncClient(base_url="http://localhost:8000") as client:
    result = await client.detect_anomalies("dataset_001")
```

### Batch Operations

```python
# Batch detection
results = await client.batch_detect([
    {"dataset_id": "dataset_001", "algorithm": "isolation_forest"},
    {"dataset_id": "dataset_002", "algorithm": "lof"}
])

# Batch predictions
predictions = await client.batch_predict(
    model_id="model_001",
    data_batches=[batch1, batch2, batch3]
)
```

## JavaScript SDK

### Installation

```bash
npm install pynomaly-js
```

### Usage

```javascript
import { PynomalyClient } from 'pynomaly-js';

const client = new PynomalyClient({
  baseUrl: 'http://localhost:8000',
  apiKey: 'your-api-key'
});
```

#### Methods

```javascript
// Detect anomalies
const result = await client.detectAnomalies({
  datasetId: 'dataset_001',
  algorithm: 'isolation_forest'
});

// Create dataset
const datasetId = await client.createDataset({
  name: 'sensor_data',
  data: [[1, 2, 3], [4, 5, 6]],
  features: ['temp', 'pressure', 'humidity']
});

// Train model
const modelId = await client.trainModel({
  name: 'my_model',
  algorithm: 'isolation_forest',
  datasetId: 'dataset_001'
});

// Make predictions
const predictions = await client.predict({
  modelId: 'model_001',
  data: [[1, 2, 3], [4, 5, 6]]
});
```

#### Real-time Detection

```javascript
// WebSocket connection
const ws = client.createWebSocket('/ws/detection/live');

ws.onMessage((message) => {
  console.log('Anomaly detected:', message);
});

// Send data for real-time detection
ws.send({
  type: 'detect',
  data: {
    algorithm: 'isolation_forest',
    sample: [1.2, 3.4, 5.6]
  }
});
```

## GraphQL API

### Schema

```graphql
type Dataset {
  id: ID!
  name: String!
  size: Int!
  features: Int!
  createdAt: DateTime!
  metadata: JSON
}

type Model {
  id: ID!
  name: String!
  algorithm: String!
  status: String!
  metrics: ModelMetrics
}

type DetectionResult {
  id: ID!
  datasetId: ID!
  anomalyCount: Int!
  anomalyRate: Float!
  anomalies: [Anomaly!]!
}

type Query {
  datasets(limit: Int, offset: Int): [Dataset!]!
  dataset(id: ID!): Dataset
  models(limit: Int, offset: Int): [Model!]!
  model(id: ID!): Model
  detectionResult(id: ID!): DetectionResult
}

type Mutation {
  createDataset(input: CreateDatasetInput!): Dataset!
  trainModel(input: TrainModelInput!): Model!
  detectAnomalies(input: DetectAnomaliesInput!): DetectionResult!
}

type Subscription {
  anomalyDetected(datasetId: ID!): Anomaly!
  modelTrainingProgress(modelId: ID!): TrainingProgress!
}
```

### Queries

```graphql
# Get datasets
query GetDatasets {
  datasets(limit: 10) {
    id
    name
    size
    features
  }
}

# Get detection results
query GetDetectionResult($id: ID!) {
  detectionResult(id: $id) {
    id
    anomalyCount
    anomalyRate
    anomalies {
      id
      score
      features
    }
  }
}
```

### Mutations

```graphql
# Create dataset
mutation CreateDataset($input: CreateDatasetInput!) {
  createDataset(input: $input) {
    id
    name
    size
  }
}

# Train model
mutation TrainModel($input: TrainModelInput!) {
  trainModel(input: $input) {
    id
    name
    status
  }
}
```

### Subscriptions

```graphql
# Real-time anomaly detection
subscription AnomalyDetected($datasetId: ID!) {
  anomalyDetected(datasetId: $datasetId) {
    id
    score
    timestamp
  }
}
```

## Configuration

### Interface Settings

```python
from pynomaly.interfaces.config import InterfaceSettings

settings = InterfaceSettings(
    # API settings
    api_host="0.0.0.0",
    api_port=8000,
    api_workers=4,
    
    # Security settings
    enable_cors=True,
    cors_origins=["http://localhost:3000"],
    rate_limit="100/minute",
    
    # WebSocket settings
    websocket_enabled=True,
    websocket_max_connections=1000,
    
    # CLI settings
    cli_theme="dark",
    cli_verbose=False,
    
    # Web settings
    web_host="0.0.0.0",
    web_port=8080,
    web_theme="auto"
)
```

### Authentication Configuration

```python
from pynomaly.interfaces.auth import AuthSettings

auth_settings = AuthSettings(
    jwt_secret="your-secret-key",
    jwt_algorithm="HS256",
    jwt_expiration=3600,
    
    # OAuth settings
    oauth_enabled=True,
    oauth_providers=["google", "github"],
    
    # API key settings
    api_key_enabled=True,
    api_key_header="X-API-Key"
)
```

## Error Handling

### HTTP Status Codes

- `200 OK` - Success
- `201 Created` - Resource created
- `400 Bad Request` - Invalid request
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Access denied
- `404 Not Found` - Resource not found
- `422 Unprocessable Entity` - Validation error
- `500 Internal Server Error` - Server error

### Error Response Format

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "dataset_id",
      "issue": "Dataset not found"
    }
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Exception Classes

```python
from pynomaly.interfaces.exceptions import (
    InterfaceError,
    AuthenticationError,
    ValidationError,
    NotFoundError,
    RateLimitError
)

try:
    result = await client.detect_anomalies("invalid_dataset")
except NotFoundError as e:
    print(f"Dataset not found: {e}")
except ValidationError as e:
    print(f"Validation error: {e}")
```

## Testing

### API Testing

```python
import pytest
from fastapi.testclient import TestClient
from pynomaly.interfaces.api import create_app

@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)

def test_detect_anomalies(client):
    response = client.post("/api/v1/detection/detect", json={
        "dataset_id": "test_dataset",
        "algorithm": "isolation_forest"
    })
    assert response.status_code == 200
    assert response.json()["success"] is True
```

### CLI Testing

```python
from pynomaly.interfaces.cli import PynomaCLI

def test_cli_detect():
    cli = PynomaCLI()
    result = cli.run(["detect", "--dataset", "test_data.csv"])
    assert result == 0  # Success exit code
```

### WebSocket Testing

```python
import pytest
from fastapi.testclient import TestClient

def test_websocket_connection():
    with client.websocket_connect("/ws/detection/live") as websocket:
        websocket.send_json({"type": "detect", "data": {"sample": [1, 2, 3]}})
        data = websocket.receive_json()
        assert data["type"] == "anomaly_result"
```

This comprehensive API reference covers all the interfaces provided by the package. For additional examples and integration patterns, refer to the [examples](../examples/) directory.