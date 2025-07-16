# 🔌 Pynomaly API Documentation

## Overview

The Pynomaly API provides comprehensive REST endpoints for anomaly detection, model management, and system monitoring. This documentation covers all 65+ available endpoints with detailed examples and usage instructions.

## Base URL

```
Production: https://api.pynomaly.com
Development: http://localhost:8000
```

## Authentication

All API endpoints require authentication via API key:

```bash
# Header-based authentication
curl -H "X-API-Key: your-api-key" https://api.pynomaly.com/api/v1/detectors

# Query parameter authentication
curl "https://api.pynomaly.com/api/v1/detectors?api_key=your-api-key"
```

## Response Format

All responses follow a consistent JSON format:

```json
{
  "success": true,
  "data": {...},
  "message": "Operation completed successfully",
  "timestamp": "2023-12-01T10:00:00Z",
  "request_id": "req_abc123"
}
```

Error responses:

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {...}
  },
  "timestamp": "2023-12-01T10:00:00Z",
  "request_id": "req_abc123"
}
```

## Core Endpoints

### 1. Health & Status

#### GET /health
Basic health check endpoint.

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "timestamp": "2023-12-01T10:00:00Z"
  }
}
```

#### GET /health/detailed
Detailed system health information.

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "components": {
      "database": "healthy",
      "redis": "healthy",
      "workers": "healthy"
    },
    "metrics": {
      "cpu_usage": 45.2,
      "memory_usage": 1024,
      "disk_usage": 15.5
    },
    "uptime": 86400
  }
}
```

#### GET /health/ready
Kubernetes readiness probe endpoint.

#### GET /health/live
Kubernetes liveness probe endpoint.

### 2. Detector Management

#### GET /api/v1/detectors
List all available detectors.

**Query Parameters:**
- `limit` (integer): Number of results per page (default: 20)
- `offset` (integer): Results offset (default: 0)
- `algorithm` (string): Filter by algorithm type
- `status` (string): Filter by status (active, training, etc.)

**Response:**
```json
{
  "success": true,
  "data": {
    "detectors": [
      {
        "id": "detector_abc123",
        "name": "Production Detector",
        "algorithm": "IsolationForest",
        "status": "active",
        "created_at": "2023-12-01T10:00:00Z",
        "updated_at": "2023-12-01T10:00:00Z",
        "performance": {
          "accuracy": 0.95,
          "precision": 0.92,
          "recall": 0.89,
          "f1_score": 0.90
        }
      }
    ],
    "pagination": {
      "total": 100,
      "limit": 20,
      "offset": 0,
      "has_next": true,
      "has_prev": false
    }
  }
}
```

#### POST /api/v1/detectors
Create a new detector.

**Request Body:**
```json
{
  "name": "My Detector",
  "algorithm": "IsolationForest",
  "parameters": {
    "contamination": 0.1,
    "n_estimators": 100,
    "random_state": 42
  },
  "description": "Production anomaly detector"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "detector_id": "detector_abc123",
    "name": "My Detector",
    "algorithm": "IsolationForest",
    "status": "created",
    "created_at": "2023-12-01T10:00:00Z"
  }
}
```

#### GET /api/v1/detectors/{detector_id}
Get detector details.

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "detector_abc123",
    "name": "My Detector",
    "algorithm": "IsolationForest",
    "status": "active",
    "parameters": {
      "contamination": 0.1,
      "n_estimators": 100,
      "random_state": 42
    },
    "performance": {
      "accuracy": 0.95,
      "precision": 0.92,
      "recall": 0.89,
      "f1_score": 0.90
    },
    "training_history": [
      {
        "timestamp": "2023-12-01T10:00:00Z",
        "dataset_size": 10000,
        "training_time": 45.2,
        "accuracy": 0.95
      }
    ],
    "created_at": "2023-12-01T10:00:00Z",
    "updated_at": "2023-12-01T10:00:00Z"
  }
}
```

#### PUT /api/v1/detectors/{detector_id}
Update detector configuration.

**Request Body:**
```json
{
  "name": "Updated Detector",
  "parameters": {
    "contamination": 0.05,
    "n_estimators": 200
  }
}
```

#### DELETE /api/v1/detectors/{detector_id}
Delete a detector.

**Response:**
```json
{
  "success": true,
  "data": {
    "message": "Detector deleted successfully"
  }
}
```

### 3. Training

#### POST /api/v1/detectors/{detector_id}/train
Train a detector with provided data.

**Request Body:**
```json
{
  "dataset_id": "dataset_abc123",
  "validation_split": 0.2,
  "cross_validation": true,
  "cv_folds": 5
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "training_id": "training_abc123",
    "status": "started",
    "estimated_time": 300
  }
}
```

#### GET /api/v1/training/{training_id}/status
Get training status.

**Response:**
```json
{
  "success": true,
  "data": {
    "training_id": "training_abc123",
    "status": "training",
    "progress": 0.65,
    "estimated_remaining": 120,
    "current_metrics": {
      "accuracy": 0.92,
      "loss": 0.08
    }
  }
}
```

### 4. Detection

#### POST /api/v1/detectors/{detector_id}/detect
Perform anomaly detection on new data.

**Request Body:**
```json
{
  "data": [
    [1.2, 3.4, 5.6],
    [2.3, 4.5, 6.7],
    [3.4, 5.6, 7.8]
  ],
  "feature_names": ["feature1", "feature2", "feature3"],
  "return_explanations": true
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "predictions": [
      {
        "index": 0,
        "is_anomaly": false,
        "anomaly_score": 0.2,
        "confidence": 0.8,
        "explanation": {
          "top_features": ["feature1", "feature2"],
          "feature_contributions": {
            "feature1": 0.3,
            "feature2": -0.1,
            "feature3": 0.05
          }
        }
      }
    ],
    "summary": {
      "total_samples": 3,
      "anomalies_detected": 1,
      "anomaly_rate": 0.33,
      "average_confidence": 0.75
    }
  }
}
```

#### POST /api/v1/detectors/{detector_id}/batch-detect
Batch detection for large datasets.

**Request Body:**
```json
{
  "dataset_id": "dataset_abc123",
  "batch_size": 1000,
  "async": true
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "batch_id": "batch_abc123",
    "status": "queued",
    "estimated_time": 600
  }
}
```

### 5. Dataset Management

#### GET /api/v1/datasets
List all datasets.

**Response:**
```json
{
  "success": true,
  "data": {
    "datasets": [
      {
        "id": "dataset_abc123",
        "name": "Production Data",
        "size": 100000,
        "features": 25,
        "created_at": "2023-12-01T10:00:00Z",
        "status": "ready"
      }
    ]
  }
}
```

#### POST /api/v1/datasets
Upload a new dataset.

**Request Body (multipart/form-data):**
```
file: [CSV/Parquet file]
name: "Dataset Name"
description: "Dataset description"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "dataset_id": "dataset_abc123",
    "name": "Dataset Name",
    "status": "processing",
    "upload_id": "upload_abc123"
  }
}
```

#### GET /api/v1/datasets/{dataset_id}
Get dataset details.

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "dataset_abc123",
    "name": "Production Data",
    "size": 100000,
    "features": 25,
    "feature_names": ["feature1", "feature2", "..."],
    "statistics": {
      "mean": [1.2, 3.4, 5.6],
      "std": [0.5, 1.2, 2.1],
      "min": [0.1, 0.5, 1.0],
      "max": [5.0, 10.0, 15.0]
    },
    "quality_metrics": {
      "completeness": 0.95,
      "consistency": 0.92,
      "validity": 0.98
    },
    "created_at": "2023-12-01T10:00:00Z",
    "status": "ready"
  }
}
```

### 6. AutoML

#### POST /api/v1/automl/optimize
Start AutoML optimization.

**Request Body:**
```json
{
  "dataset_id": "dataset_abc123",
  "algorithms": ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"],
  "objectives": ["accuracy", "speed", "interpretability"],
  "constraints": {
    "max_time": 3600,
    "max_trials": 100,
    "max_memory": 4096
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "optimization_id": "opt_abc123",
    "status": "started",
    "estimated_time": 3600
  }
}
```

#### GET /api/v1/automl/optimization/{optimization_id}
Get optimization status.

**Response:**
```json
{
  "success": true,
  "data": {
    "optimization_id": "opt_abc123",
    "status": "running",
    "progress": 0.45,
    "current_best": {
      "algorithm": "IsolationForest",
      "score": 0.92,
      "parameters": {
        "contamination": 0.1,
        "n_estimators": 150
      }
    },
    "trials_completed": 45,
    "trials_total": 100,
    "estimated_remaining": 1800
  }
}
```

### 7. Explainability

#### POST /api/v1/explainability/explain
Generate explanations for model predictions.

**Request Body:**
```json
{
  "detector_id": "detector_abc123",
  "data": [[1.2, 3.4, 5.6]],
  "explanation_type": "both",
  "methods": ["shap", "lime"],
  "top_features": 5
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "explanations": [
      {
        "instance_id": "inst_abc123",
        "prediction": "anomaly",
        "confidence": 0.85,
        "methods": {
          "shap": {
            "feature_importance": {
              "feature1": 0.45,
              "feature2": -0.23,
              "feature3": 0.12
            },
            "base_value": 0.1,
            "plot_url": "/plots/shap_abc123.png"
          },
          "lime": {
            "feature_importance": {
              "feature1": 0.42,
              "feature2": -0.25,
              "feature3": 0.15
            },
            "local_fidelity": 0.92
          }
        }
      }
    ]
  }
}
```

### 8. Monitoring & Metrics

#### GET /api/v1/metrics
Get system metrics.

**Response:**
```json
{
  "success": true,
  "data": {
    "system": {
      "cpu_usage": 45.2,
      "memory_usage": 1024,
      "disk_usage": 15.5,
      "uptime": 86400
    },
    "detection": {
      "total_detections": 1000000,
      "anomalies_detected": 50000,
      "average_accuracy": 0.95,
      "average_response_time": 0.15
    },
    "models": {
      "total_models": 25,
      "active_models": 20,
      "training_models": 2
    }
  }
}
```

#### GET /api/v1/metrics/performance
Get performance metrics.

**Query Parameters:**
- `start_time` (ISO datetime): Start time for metrics
- `end_time` (ISO datetime): End time for metrics
- `granularity` (string): Metrics granularity (hour, day, week)

**Response:**
```json
{
  "success": true,
  "data": {
    "time_range": {
      "start": "2023-12-01T00:00:00Z",
      "end": "2023-12-01T23:59:59Z"
    },
    "metrics": [
      {
        "timestamp": "2023-12-01T10:00:00Z",
        "accuracy": 0.95,
        "precision": 0.92,
        "recall": 0.89,
        "f1_score": 0.90,
        "response_time": 0.15,
        "throughput": 100
      }
    ]
  }
}
```

### 9. Drift Detection

#### POST /api/v1/drift/detect
Detect data drift.

**Request Body:**
```json
{
  "reference_dataset_id": "dataset_abc123",
  "current_dataset_id": "dataset_def456",
  "methods": ["ks_test", "psi", "wasserstein"],
  "threshold": 0.05
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "drift_detected": true,
    "overall_drift_score": 0.75,
    "feature_drift": {
      "feature1": {
        "drift_score": 0.45,
        "p_value": 0.02,
        "method": "ks_test",
        "drift_detected": true
      }
    },
    "recommendations": [
      "Retrain model with recent data",
      "Investigate feature1 distribution change"
    ]
  }
}
```

### 10. Results & Reports

#### GET /api/v1/results
Get detection results.

**Query Parameters:**
- `detector_id` (string): Filter by detector
- `start_time` (ISO datetime): Start time
- `end_time` (ISO datetime): End time
- `limit` (integer): Results limit

**Response:**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "id": "result_abc123",
        "detector_id": "detector_abc123",
        "dataset_id": "dataset_abc123",
        "timestamp": "2023-12-01T10:00:00Z",
        "summary": {
          "total_samples": 1000,
          "anomalies_detected": 50,
          "anomaly_rate": 0.05,
          "accuracy": 0.95
        }
      }
    ]
  }
}
```

#### GET /api/v1/results/{result_id}
Get detailed result.

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "result_abc123",
    "detector_id": "detector_abc123",
    "dataset_id": "dataset_abc123",
    "timestamp": "2023-12-01T10:00:00Z",
    "predictions": [
      {
        "index": 0,
        "is_anomaly": false,
        "anomaly_score": 0.2,
        "confidence": 0.8
      }
    ],
    "summary": {
      "total_samples": 1000,
      "anomalies_detected": 50,
      "anomaly_rate": 0.05,
      "accuracy": 0.95,
      "precision": 0.92,
      "recall": 0.89,
      "f1_score": 0.90
    },
    "execution_time": 15.2,
    "resource_usage": {
      "cpu_time": 12.5,
      "memory_peak": 512,
      "disk_io": 10.2
    }
  }
}
```

## Advanced Features

### 11. Real-time Detection

#### WebSocket: /ws/detection/{detector_id}
Real-time anomaly detection stream.

**Connection Example:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/detection/detector_abc123');

ws.onmessage = function(event) {
  const result = JSON.parse(event.data);
  console.log('Anomaly detected:', result);
};

// Send data for detection
ws.send(JSON.stringify({
  data: [1.2, 3.4, 5.6],
  timestamp: new Date().toISOString()
}));
```

### 12. Batch Processing

#### POST /api/v1/batch/submit
Submit batch processing job.

**Request Body:**
```json
{
  "job_type": "detection",
  "detector_id": "detector_abc123",
  "dataset_id": "dataset_abc123",
  "parameters": {
    "batch_size": 1000,
    "parallel_jobs": 4
  }
}
```

#### GET /api/v1/batch/jobs
List batch jobs.

#### GET /api/v1/batch/jobs/{job_id}
Get batch job status.

### 13. Model Management

#### POST /api/v1/models/export
Export trained model.

**Request Body:**
```json
{
  "detector_id": "detector_abc123",
  "format": "onnx",
  "include_metadata": true
}
```

#### POST /api/v1/models/import
Import external model.

**Request Body (multipart/form-data):**
```
file: [Model file]
format: "onnx"
name: "Imported Model"
```

### 14. A/B Testing

#### POST /api/v1/experiments/create
Create A/B test experiment.

**Request Body:**
```json
{
  "name": "Model Comparison",
  "detector_a": "detector_abc123",
  "detector_b": "detector_def456",
  "traffic_split": 0.5,
  "duration": 86400
}
```

#### GET /api/v1/experiments/{experiment_id}/results
Get A/B test results.

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid input parameters |
| 401 | Unauthorized - Invalid API key |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource not found |
| 409 | Conflict - Resource already exists |
| 422 | Unprocessable Entity - Validation error |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable - System overloaded |

## Rate Limits

- **Default**: 1000 requests per minute
- **Burst**: 100 requests per second
- **Training**: 10 concurrent jobs per user
- **Detection**: 10,000 requests per minute

## SDKs & Libraries

### Python SDK
```python
from pynomaly_client import PynomalyClient

client = PynomalyClient(api_key="your-api-key")
detector = client.detectors.create(
    name="My Detector",
    algorithm="IsolationForest"
)
```

### JavaScript SDK
```javascript
import { PynomalyClient } from 'pynomaly-js';

const client = new PynomalyClient('your-api-key');
const detector = await client.detectors.create({
  name: 'My Detector',
  algorithm: 'IsolationForest'
});
```

### cURL Examples
```bash
# Create detector
curl -X POST "https://api.pynomaly.com/api/v1/detectors" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "My Detector", "algorithm": "IsolationForest"}'

# Run detection
curl -X POST "https://api.pynomaly.com/api/v1/detectors/detector_abc123/detect" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"data": [[1.2, 3.4, 5.6]]}'
```

## Webhooks

Register webhooks to receive real-time notifications:

### POST /api/v1/webhooks
Create webhook endpoint.

**Request Body:**
```json
{
  "url": "https://your-app.com/webhook",
  "events": ["detection.completed", "training.finished", "drift.detected"],
  "secret": "webhook-secret"
}
```

### Webhook Events

- `detection.completed`: Anomaly detection finished
- `training.started`: Model training started
- `training.finished`: Model training completed
- `drift.detected`: Data drift detected
- `alert.triggered`: System alert triggered

## OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:
- **JSON**: `/api/v1/openapi.json`
- **YAML**: `/api/v1/openapi.yaml`
- **Interactive Docs**: `/docs`
- **ReDoc**: `/redoc`

## Support

- **Documentation**: https://docs.pynomaly.com
- **API Status**: https://status.pynomaly.com
- **Support**: support@pynomaly.com
- **Community**: https://github.com/pynomaly/pynomaly