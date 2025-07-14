# Pynomaly API Documentation

## Overview

The Pynomaly API provides comprehensive endpoints for anomaly detection, model management, and system monitoring. This RESTful API is built with FastAPI and includes authentication, rate limiting, and comprehensive error handling.

## Base URL

- **Production**: `https://api.pynomaly.com`
- **Development**: `http://localhost:8000`

## Authentication

### API Key Authentication

```bash
# Include API key in headers
curl -H "X-API-Key: your-api-key" https://api.pynomaly.com/api/v1/detectors
```

### JWT Authentication

```bash
# Login to get JWT token
curl -X POST \
  https://api.pynomaly.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "password"}'

# Use JWT token in subsequent requests
curl -H "Authorization: Bearer your-jwt-token" https://api.pynomaly.com/api/v1/detectors
```

## Rate Limiting

- **General API**: 100 requests per minute
- **Detection**: 10 requests per minute
- **Training**: 5 requests per minute

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "contamination_rate",
      "error": "Must be between 0 and 1"
    },
    "timestamp": "2023-01-01T12:00:00Z",
    "trace_id": "abc123"
  }
}
```

### HTTP Status Codes

- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `422` - Validation Error
- `429` - Rate Limited
- `500` - Internal Server Error

## Endpoints

### Health Check

#### GET /health

Check system health and status.

```bash
curl https://api.pynomaly.com/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2023-01-01T12:00:00Z",
  "version": "1.0.0",
  "components": {
    "database": "healthy",
    "redis": "healthy",
    "ml_models": "healthy"
  }
}
```

### Detector Management

#### GET /api/v1/detectors

List all available detectors.

```bash
curl -H "Authorization: Bearer your-jwt-token" \
  https://api.pynomaly.com/api/v1/detectors
```

**Response:**
```json
{
  "detectors": [
    {
      "id": "uuid",
      "name": "isolation_forest_detector",
      "algorithm": "IsolationForest",
      "status": "active",
      "created_at": "2023-01-01T12:00:00Z",
      "updated_at": "2023-01-01T12:00:00Z",
      "parameters": {
        "contamination": 0.1,
        "n_estimators": 100
      },
      "performance": {
        "accuracy": 0.95,
        "precision": 0.92,
        "recall": 0.88
      }
    }
  ],
  "total": 1,
  "page": 1,
  "size": 10
}
```

#### POST /api/v1/detectors

Create a new detector.

```bash
curl -X POST \
  -H "Authorization: Bearer your-jwt-token" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_detector",
    "algorithm": "IsolationForest",
    "parameters": {
      "contamination": 0.1,
      "n_estimators": 100
    }
  }' \
  https://api.pynomaly.com/api/v1/detectors
```

**Request Body:**
```json
{
  "name": "string",
  "algorithm": "IsolationForest|LOF|OneClassSVM",
  "parameters": {
    "contamination": 0.1,
    "n_estimators": 100
  },
  "description": "string"
}
```

**Response:**
```json
{
  "id": "uuid",
  "name": "my_detector",
  "algorithm": "IsolationForest",
  "status": "created",
  "created_at": "2023-01-01T12:00:00Z",
  "parameters": {
    "contamination": 0.1,
    "n_estimators": 100
  }
}
```

#### GET /api/v1/detectors/{detector_id}

Get detector details.

```bash
curl -H "Authorization: Bearer your-jwt-token" \
  https://api.pynomaly.com/api/v1/detectors/uuid
```

#### PUT /api/v1/detectors/{detector_id}

Update detector configuration.

```bash
curl -X PUT \
  -H "Authorization: Bearer your-jwt-token" \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {
      "contamination": 0.15,
      "n_estimators": 200
    }
  }' \
  https://api.pynomaly.com/api/v1/detectors/uuid
```

#### DELETE /api/v1/detectors/{detector_id}

Delete a detector.

```bash
curl -X DELETE \
  -H "Authorization: Bearer your-jwt-token" \
  https://api.pynomaly.com/api/v1/detectors/uuid
```

### Dataset Management

#### POST /api/v1/datasets

Upload a dataset.

```bash
curl -X POST \
  -H "Authorization: Bearer your-jwt-token" \
  -F "file=@data.csv" \
  -F "name=my_dataset" \
  -F "description=Training data" \
  https://api.pynomaly.com/api/v1/datasets
```

**Response:**
```json
{
  "id": "uuid",
  "name": "my_dataset",
  "filename": "data.csv",
  "size": 1048576,
  "rows": 1000,
  "columns": 10,
  "created_at": "2023-01-01T12:00:00Z",
  "status": "uploaded"
}
```

#### GET /api/v1/datasets

List datasets.

```bash
curl -H "Authorization: Bearer your-jwt-token" \
  https://api.pynomaly.com/api/v1/datasets
```

#### GET /api/v1/datasets/{dataset_id}

Get dataset details.

```bash
curl -H "Authorization: Bearer your-jwt-token" \
  https://api.pynomaly.com/api/v1/datasets/uuid
```

### Model Training

#### POST /api/v1/train

Train a detector on a dataset.

```bash
curl -X POST \
  -H "Authorization: Bearer your-jwt-token" \
  -H "Content-Type: application/json" \
  -d '{
    "detector_id": "detector-uuid",
    "dataset_id": "dataset-uuid",
    "validation_split": 0.2,
    "cross_validation": true,
    "hyperparameter_tuning": true
  }' \
  https://api.pynomaly.com/api/v1/train
```

**Request Body:**
```json
{
  "detector_id": "string",
  "dataset_id": "string",
  "validation_split": 0.2,
  "cross_validation": true,
  "hyperparameter_tuning": false,
  "parameters": {
    "n_folds": 5,
    "scoring": "roc_auc"
  }
}
```

**Response:**
```json
{
  "training_id": "uuid",
  "status": "started",
  "detector_id": "detector-uuid",
  "dataset_id": "dataset-uuid",
  "started_at": "2023-01-01T12:00:00Z",
  "estimated_duration": 300
}
```

#### GET /api/v1/train/{training_id}

Get training status.

```bash
curl -H "Authorization: Bearer your-jwt-token" \
  https://api.pynomaly.com/api/v1/train/uuid
```

**Response:**
```json
{
  "training_id": "uuid",
  "status": "completed",
  "progress": 100,
  "started_at": "2023-01-01T12:00:00Z",
  "completed_at": "2023-01-01T12:05:00Z",
  "duration": 300,
  "results": {
    "accuracy": 0.95,
    "precision": 0.92,
    "recall": 0.88,
    "f1_score": 0.90,
    "roc_auc": 0.94
  }
}
```

### Anomaly Detection

#### POST /api/v1/detect

Detect anomalies in data.

```bash
curl -X POST \
  -H "Authorization: Bearer your-jwt-token" \
  -H "Content-Type: application/json" \
  -d '{
    "detector_id": "detector-uuid",
    "data": [
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
      [7.0, 8.0, 9.0]
    ],
    "return_probabilities": true,
    "return_explanations": true
  }' \
  https://api.pynomaly.com/api/v1/detect
```

**Request Body:**
```json
{
  "detector_id": "string",
  "data": [[1.0, 2.0], [3.0, 4.0]],
  "return_probabilities": true,
  "return_explanations": false,
  "threshold": 0.5
}
```

**Response:**
```json
{
  "detection_id": "uuid",
  "detector_id": "detector-uuid",
  "timestamp": "2023-01-01T12:00:00Z",
  "results": [
    {
      "index": 0,
      "is_anomaly": false,
      "score": 0.3,
      "probability": 0.15,
      "explanation": {
        "top_features": [
          {"feature": "feature_1", "contribution": 0.8},
          {"feature": "feature_2", "contribution": 0.2}
        ]
      }
    }
  ],
  "summary": {
    "total_samples": 3,
    "anomalies_detected": 1,
    "anomaly_rate": 0.33
  }
}
```

#### POST /api/v1/detect/batch

Batch anomaly detection.

```bash
curl -X POST \
  -H "Authorization: Bearer your-jwt-token" \
  -F "file=@test_data.csv" \
  -F "detector_id=detector-uuid" \
  https://api.pynomaly.com/api/v1/detect/batch
```

**Response:**
```json
{
  "batch_id": "uuid",
  "detector_id": "detector-uuid",
  "status": "processing",
  "submitted_at": "2023-01-01T12:00:00Z",
  "estimated_completion": "2023-01-01T12:10:00Z"
}
```

### Model Management

#### GET /api/v1/models

List trained models.

```bash
curl -H "Authorization: Bearer your-jwt-token" \
  https://api.pynomaly.com/api/v1/models
```

#### GET /api/v1/models/{model_id}

Get model details.

```bash
curl -H "Authorization: Bearer your-jwt-token" \
  https://api.pynomaly.com/api/v1/models/uuid
```

#### POST /api/v1/models/{model_id}/deploy

Deploy a model to production.

```bash
curl -X POST \
  -H "Authorization: Bearer your-jwt-token" \
  -H "Content-Type: application/json" \
  -d '{
    "environment": "production",
    "auto_scaling": true,
    "min_instances": 2,
    "max_instances": 10
  }' \
  https://api.pynomaly.com/api/v1/models/uuid/deploy
```

### Metrics and Monitoring

#### GET /api/v1/metrics

Get system metrics.

```bash
curl -H "Authorization: Bearer your-jwt-token" \
  https://api.pynomaly.com/api/v1/metrics
```

**Response:**
```json
{
  "timestamp": "2023-01-01T12:00:00Z",
  "system": {
    "cpu_usage": 45.2,
    "memory_usage": 67.8,
    "disk_usage": 23.4
  },
  "application": {
    "active_detectors": 5,
    "detections_per_hour": 1500,
    "average_response_time": 250
  },
  "models": {
    "total_models": 10,
    "active_models": 8,
    "training_jobs": 2
  }
}
```

#### GET /api/v1/metrics/detectors/{detector_id}

Get detector-specific metrics.

```bash
curl -H "Authorization: Bearer your-jwt-token" \
  https://api.pynomaly.com/api/v1/metrics/detectors/uuid
```

### Explainable AI

#### POST /api/v1/explain

Get explanations for predictions.

```bash
curl -X POST \
  -H "Authorization: Bearer your-jwt-token" \
  -H "Content-Type: application/json" \
  -d '{
    "detector_id": "detector-uuid",
    "data": [[1.0, 2.0, 3.0]],
    "method": "shap",
    "top_features": 5
  }' \
  https://api.pynomaly.com/api/v1/explain
```

**Response:**
```json
{
  "explanation_id": "uuid",
  "detector_id": "detector-uuid",
  "method": "shap",
  "explanations": [
    {
      "sample_index": 0,
      "prediction": 0.8,
      "feature_contributions": [
        {"feature": "feature_1", "contribution": 0.4, "value": 1.0},
        {"feature": "feature_2", "contribution": 0.3, "value": 2.0},
        {"feature": "feature_3", "contribution": 0.1, "value": 3.0}
      ]
    }
  ]
}
```

## WebSocket API

### Real-time Detection

Connect to WebSocket for real-time anomaly detection:

```javascript
const ws = new WebSocket('wss://api.pynomaly.com/ws/detect');

ws.onopen = function() {
    // Send configuration
    ws.send(JSON.stringify({
        'detector_id': 'detector-uuid',
        'threshold': 0.5
    }));
};

ws.onmessage = function(event) {
    const result = JSON.parse(event.data);
    console.log('Anomaly detected:', result);
};

// Send data for detection
ws.send(JSON.stringify({
    'data': [1.0, 2.0, 3.0],
    'timestamp': new Date().toISOString()
}));
```

## SDKs and Libraries

### Python SDK

```python
from pynomaly_client import PynamolyClient

client = PynamolyClient(
    base_url="https://api.pynomaly.com",
    api_key="your-api-key"
)

# Create detector
detector = client.detectors.create(
    name="my_detector",
    algorithm="IsolationForest",
    parameters={"contamination": 0.1}
)

# Train detector
training = client.training.start(
    detector_id=detector.id,
    dataset_id="dataset-uuid"
)

# Detect anomalies
results = client.detection.detect(
    detector_id=detector.id,
    data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
)
```

### JavaScript SDK

```javascript
import { PynamolyClient } from 'pynomaly-js';

const client = new PynamolyClient({
    baseUrl: 'https://api.pynomaly.com',
    apiKey: 'your-api-key'
});

// Create detector
const detector = await client.detectors.create({
    name: 'my_detector',
    algorithm: 'IsolationForest',
    parameters: { contamination: 0.1 }
});

// Detect anomalies
const results = await client.detection.detect({
    detectorId: detector.id,
    data: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
});
```

## Examples

### Complete Workflow

```python
import requests
import json

# Configuration
base_url = "https://api.pynomaly.com"
headers = {"Authorization": "Bearer your-jwt-token"}

# 1. Create detector
detector_data = {
    "name": "fraud_detector",
    "algorithm": "IsolationForest",
    "parameters": {"contamination": 0.05}
}
response = requests.post(f"{base_url}/api/v1/detectors", 
                        headers=headers, json=detector_data)
detector = response.json()

# 2. Upload dataset
with open("fraud_data.csv", "rb") as f:
    files = {"file": f}
    data = {"name": "fraud_dataset"}
    response = requests.post(f"{base_url}/api/v1/datasets", 
                            headers=headers, files=files, data=data)
dataset = response.json()

# 3. Train detector
training_data = {
    "detector_id": detector["id"],
    "dataset_id": dataset["id"],
    "validation_split": 0.2,
    "cross_validation": True
}
response = requests.post(f"{base_url}/api/v1/train", 
                        headers=headers, json=training_data)
training = response.json()

# 4. Check training status
while True:
    response = requests.get(f"{base_url}/api/v1/train/{training['training_id']}", 
                           headers=headers)
    status = response.json()
    if status["status"] == "completed":
        break
    time.sleep(10)

# 5. Detect anomalies
detection_data = {
    "detector_id": detector["id"],
    "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    "return_probabilities": True
}
response = requests.post(f"{base_url}/api/v1/detect", 
                        headers=headers, json=detection_data)
results = response.json()

print(f"Detected {results['summary']['anomalies_detected']} anomalies")
```

## Best Practices

### Performance Optimization

1. **Batch Processing**: Use batch detection for large datasets
2. **Caching**: Cache detector models for faster inference
3. **Async Operations**: Use async endpoints for long-running tasks
4. **Data Streaming**: Use WebSocket for real-time processing

### Error Handling

```python
try:
    response = client.detection.detect(data=data)
except PynamolyAPIError as e:
    if e.status_code == 429:
        # Rate limited - implement backoff
        time.sleep(60)
        retry_request()
    elif e.status_code == 422:
        # Validation error - fix data
        print(f"Validation error: {e.details}")
    else:
        # Other errors
        print(f"API error: {e.message}")
```

### Security

1. **API Keys**: Store API keys securely
2. **Rate Limiting**: Implement client-side rate limiting
3. **Input Validation**: Validate data before sending
4. **HTTPS**: Always use HTTPS in production

## Changelog

### v1.0.0
- Initial API release
- Basic detector management
- Anomaly detection endpoints
- Authentication system

### v1.1.0
- Added batch detection
- WebSocket support
- Model deployment
- Explainable AI features

### v1.2.0
- Enhanced monitoring
- Performance improvements
- Additional algorithms
- SDK libraries

## Support

For API support and questions:

- **Documentation**: https://docs.pynomaly.com/api
- **Status Page**: https://status.pynomaly.com
- **Support**: support@pynomaly.com
- **Community**: https://github.com/elgerytme/Pynomaly/discussions