# REST API Reference

This document provides comprehensive documentation for the Pynomaly REST API endpoints.

## Overview

The Pynomaly REST API is built with FastAPI and provides programmatic access to all anomaly detection functionality. The API follows RESTful principles and returns JSON responses.

**Base URL**: `http://localhost:8000/api/v1`  
**OpenAPI Documentation**: `http://localhost:8000/docs`  
**ReDoc Documentation**: `http://localhost:8000/redoc`

## Authentication

The API supports multiple authentication methods:

### JWT Authentication
```bash
# Login to get access token
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "password"}'

# Use token in subsequent requests
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  "http://localhost:8000/api/v1/detectors"
```

### API Key Authentication
```bash
# Create API key
curl -X POST "http://localhost:8000/api/v1/auth/api-keys" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{"name": "My API Key"}'

# Use API key
curl -H "X-API-Key: YOUR_API_KEY" \
  "http://localhost:8000/api/v1/detectors"
```

## Health and Status

### GET /health
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0",
  "components": {
    "database": "healthy",
    "cache": "healthy",
    "algorithms": "healthy"
  }
}
```

### GET /status
Get detailed system status.

**Response:**
```json
{
  "status": "operational",
  "uptime": 3600,
  "active_detectors": 5,
  "total_detections": 1000,
  "memory_usage_mb": 512.5,
  "cpu_usage_percent": 25.3
}
```

## Authentication Endpoints

### POST /auth/login
Authenticate user and get access token.

**Request:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### POST /auth/refresh
Refresh access token.

**Request:**
```json
{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

### POST /auth/api-keys
Create a new API key.

**Request:**
```json
{
  "name": "My Application Key",
  "description": "API key for my application"
}
```

**Response:**
```json
{
  "id": "api_key_123",
  "name": "My Application Key",
  "key": "pk_live_abc123...",
  "created_at": "2024-01-01T12:00:00Z"
}
```

### DELETE /auth/api-keys/{key_id}
Revoke an API key.

## Detector Management

### GET /detectors
List all detectors.

**Query Parameters:**
- `limit` (int, default: 100): Maximum number of results
- `offset` (int, default: 0): Number of results to skip
- `algorithm` (string): Filter by algorithm type
- `status` (string): Filter by training status

**Response:**
```json
{
  "detectors": [
    {
      "id": "detector_123",
      "name": "Fraud Detector",
      "algorithm": "IsolationForest",
      "parameters": {
        "contamination": 0.1,
        "n_estimators": 100
      },
      "is_trained": true,
      "created_at": "2024-01-01T12:00:00Z",
      "updated_at": "2024-01-01T12:30:00Z"
    }
  ],
  "total": 1,
  "limit": 100,
  "offset": 0
}
```

### POST /detectors
Create a new detector.

**Request:**
```json
{
  "name": "Credit Card Fraud Detector",
  "algorithm": "IsolationForest",
  "parameters": {
    "contamination": 0.1,
    "n_estimators": 100,
    "random_state": 42
  },
  "description": "Detects fraudulent credit card transactions"
}
```

**Response:**
```json
{
  "id": "detector_456",
  "name": "Credit Card Fraud Detector",
  "algorithm": "IsolationForest",
  "parameters": {
    "contamination": 0.1,
    "n_estimators": 100,
    "random_state": 42
  },
  "is_trained": false,
  "created_at": "2024-01-01T12:00:00Z"
}
```

### GET /detectors/{detector_id}
Get detector details.

**Response:**
```json
{
  "id": "detector_123",
  "name": "Fraud Detector",
  "algorithm": "IsolationForest",
  "parameters": {
    "contamination": 0.1,
    "n_estimators": 100
  },
  "is_trained": true,
  "training_info": {
    "trained_at": "2024-01-01T12:15:00Z",
    "training_samples": 1000,
    "training_duration_ms": 5000
  },
  "performance_metrics": {
    "last_evaluation": {
      "precision": 0.85,
      "recall": 0.78,
      "f1_score": 0.81,
      "evaluated_at": "2024-01-01T12:20:00Z"
    }
  }
}
```

### PUT /detectors/{detector_id}
Update detector configuration.

**Request:**
```json
{
  "name": "Updated Fraud Detector",
  "parameters": {
    "contamination": 0.15,
    "n_estimators": 200
  }
}
```

### DELETE /detectors/{detector_id}
Delete a detector.

**Response:**
```json
{
  "message": "Detector deleted successfully"
}
```

### GET /detectors/algorithms
Get available algorithms and their parameters.

**Response:**
```json
{
  "algorithms": [
    {
      "name": "IsolationForest",
      "category": "tree_based",
      "description": "Isolation Forest for anomaly detection",
      "parameters": {
        "contamination": {
          "type": "float",
          "default": 0.1,
          "min": 0.0,
          "max": 0.5,
          "description": "Expected proportion of outliers"
        },
        "n_estimators": {
          "type": "integer",
          "default": 100,
          "min": 1,
          "max": 1000,
          "description": "Number of base estimators"
        }
      },
      "complexity": "O(n log n)",
      "scalability": "excellent"
    }
  ]
}
```

## Dataset Management

### GET /datasets
List all datasets.

**Query Parameters:**
- `limit` (int): Maximum number of results
- `offset` (int): Number of results to skip
- `format` (string): Filter by data format

**Response:**
```json
{
  "datasets": [
    {
      "id": "dataset_123",
      "name": "Credit Transactions",
      "description": "Historical credit card transactions",
      "sample_count": 10000,
      "feature_count": 15,
      "created_at": "2024-01-01T10:00:00Z",
      "data_source": "uploaded_file"
    }
  ],
  "total": 1
}
```

### POST /datasets
Create a new dataset.

**Request (JSON data):**
```json
{
  "name": "Transaction Data",
  "description": "Credit card transaction dataset",
  "data": [
    {"amount": 100.0, "merchant": "grocery", "location": "NY"},
    {"amount": 5000.0, "merchant": "jewelry", "location": "FL"}
  ]
}
```

**Request (File upload):**
```bash
curl -X POST "http://localhost:8000/api/v1/datasets/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@transactions.csv" \
  -F "name=Transaction Data" \
  -F "description=Credit card transactions"
```

### GET /datasets/{dataset_id}
Get dataset details.

**Response:**
```json
{
  "id": "dataset_123",
  "name": "Credit Transactions",
  "description": "Historical credit card transactions",
  "sample_count": 10000,
  "feature_count": 15,
  "features": [
    {"name": "amount", "type": "float", "min": 0.1, "max": 10000.0},
    {"name": "merchant_category", "type": "string", "unique_values": 50}
  ],
  "statistics": {
    "numerical_features": 8,
    "categorical_features": 7,
    "missing_values": 0
  }
}
```

### GET /datasets/{dataset_id}/sample
Get sample data from dataset.

**Query Parameters:**
- `limit` (int, default: 10): Number of samples to return

**Response:**
```json
{
  "samples": [
    {"amount": 100.0, "merchant": "grocery", "location": "NY"},
    {"amount": 250.0, "merchant": "restaurant", "location": "CA"}
  ],
  "total_samples": 10000
}
```

### DELETE /datasets/{dataset_id}
Delete a dataset.

## Detection Operations

### POST /detectors/{detector_id}/train
Train a detector with a dataset.

**Request:**
```json
{
  "dataset_id": "dataset_123",
  "validation_split": 0.2
}
```

**Response:**
```json
{
  "training_id": "training_789",
  "status": "completed",
  "training_samples": 8000,
  "validation_samples": 2000,
  "training_duration_ms": 5000,
  "metrics": {
    "training_loss": 0.15,
    "validation_score": 0.82
  }
}
```

### POST /detectors/{detector_id}/detect
Run anomaly detection on data.

**Request:**
```json
{
  "data": [
    {"amount": 100.0, "merchant": "grocery", "location": "NY"},
    {"amount": 10000.0, "merchant": "jewelry", "location": "FL"}
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "data_point": {"amount": 100.0, "merchant": "grocery", "location": "NY"},
      "is_anomaly": false,
      "anomaly_score": 0.3,
      "confidence": 0.85,
      "explanation": "Normal transaction pattern"
    },
    {
      "data_point": {"amount": 10000.0, "merchant": "jewelry", "location": "FL"},
      "is_anomaly": true,
      "anomaly_score": 0.95,
      "confidence": 0.92,
      "explanation": "Unusually high amount for this merchant category"
    }
  ],
  "anomalies_detected": 1,
  "total_samples": 2
}
```

### POST /detectors/{detector_id}/detect/batch
Run batch detection on dataset.

**Request:**
```json
{
  "dataset_id": "dataset_123",
  "output_format": "json"
}
```

**Response:**
```json
{
  "batch_id": "batch_456",
  "status": "completed",
  "processed_samples": 10000,
  "anomalies_detected": 150,
  "anomaly_rate": 0.015,
  "processing_duration_ms": 15000,
  "results_url": "/api/v1/results/batch_456"
}
```

### GET /detectors/{detector_id}/results
Get detection results.

**Query Parameters:**
- `limit` (int): Maximum results to return
- `start_date` (datetime): Filter results after this date
- `end_date` (datetime): Filter results before this date
- `anomalies_only` (bool): Return only anomalous results

**Response:**
```json
{
  "results": [
    {
      "id": "result_123",
      "detector_id": "detector_456",
      "is_anomaly": true,
      "anomaly_score": 0.95,
      "detected_at": "2024-01-01T12:00:00Z",
      "data_point": {"amount": 10000.0}
    }
  ],
  "total": 150,
  "anomaly_count": 150
}
```

## Experiment Management

### GET /experiments
List experiments.

**Response:**
```json
{
  "experiments": [
    {
      "id": "exp_123",
      "name": "Algorithm Comparison",
      "description": "Compare multiple algorithms on fraud data",
      "status": "completed",
      "created_at": "2024-01-01T10:00:00Z",
      "results_summary": {
        "best_algorithm": "IsolationForest",
        "best_f1_score": 0.85
      }
    }
  ]
}
```

### POST /experiments
Create a new experiment.

**Request:**
```json
{
  "name": "Fraud Detection Comparison",
  "description": "Compare algorithms for fraud detection",
  "dataset_id": "dataset_123",
  "algorithms": [
    {
      "name": "IsolationForest",
      "parameters": {"contamination": 0.1}
    },
    {
      "name": "LOF",
      "parameters": {"contamination": 0.1, "n_neighbors": 20}
    }
  ],
  "evaluation_metrics": ["precision", "recall", "f1_score"]
}
```

### GET /experiments/{experiment_id}
Get experiment details and results.

**Response:**
```json
{
  "id": "exp_123",
  "name": "Algorithm Comparison",
  "status": "completed",
  "results": [
    {
      "algorithm": "IsolationForest",
      "metrics": {
        "precision": 0.85,
        "recall": 0.78,
        "f1_score": 0.81
      },
      "training_time_ms": 5000
    },
    {
      "algorithm": "LOF",
      "metrics": {
        "precision": 0.82,
        "recall": 0.75,
        "f1_score": 0.78
      },
      "training_time_ms": 15000
    }
  ],
  "winner": {
    "algorithm": "IsolationForest",
    "reason": "Highest F1 score and fastest training"
  }
}
```

## Ensemble Methods

### POST /ensembles
Create an ensemble detector.

**Request:**
```json
{
  "name": "Fraud Detection Ensemble",
  "detector_ids": ["detector_1", "detector_2", "detector_3"],
  "voting_strategy": "majority",
  "weights": [0.4, 0.3, 0.3]
}
```

### POST /ensembles/{ensemble_id}/detect
Run ensemble detection.

**Request:**
```json
{
  "data": [
    {"amount": 100.0, "merchant": "grocery"}
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "data_point": {"amount": 100.0, "merchant": "grocery"},
      "is_anomaly": false,
      "ensemble_score": 0.3,
      "individual_scores": [0.2, 0.4, 0.3],
      "votes": {"anomaly": 0, "normal": 3},
      "confidence": 0.9
    }
  ]
}
```

## Monitoring and Analytics

### GET /metrics
Get system performance metrics.

**Response:**
```json
{
  "total_detectors": 25,
  "active_detectors": 18,
  "total_detections_today": 5000,
  "anomalies_detected_today": 150,
  "avg_detection_time_ms": 45.2,
  "system_health": "healthy"
}
```

### GET /analytics/performance
Get performance analytics.

**Query Parameters:**
- `detector_id` (string): Filter by detector
- `period` (string): Time period (hour, day, week, month)

**Response:**
```json
{
  "period": "day",
  "detector_performance": [
    {
      "detector_id": "detector_123",
      "detections_count": 1000,
      "anomalies_count": 50,
      "avg_score": 0.3,
      "avg_processing_time_ms": 42.5
    }
  ],
  "trends": {
    "detection_volume": [850, 920, 1000, 1100],
    "anomaly_rate": [0.048, 0.052, 0.050, 0.045]
  }
}
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error information.

### Error Response Format
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid detector parameters",
    "details": {
      "field": "contamination",
      "issue": "Value must be between 0 and 0.5"
    },
    "request_id": "req_123456"
  }
}
```

### Common HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request data |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 422 | Unprocessable Entity | Validation error |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Standard users**: 1000 requests per hour
- **Premium users**: 10000 requests per hour
- **Enterprise**: Custom limits

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1577836800
```

## Pagination

List endpoints support pagination:

**Request:**
```
GET /api/v1/detectors?limit=50&offset=100
```

**Response includes pagination metadata:**
```json
{
  "data": [...],
  "pagination": {
    "limit": 50,
    "offset": 100,
    "total": 500,
    "has_more": true
  }
}
```

## WebSocket Support

Real-time updates are available via WebSocket connections:

### Connect to WebSocket
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/detections');

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Real-time detection:', data);
};
```

### WebSocket Message Types
- `detection_result` - New anomaly detection result
- `detector_trained` - Detector training completed
- `system_alert` - System health alerts

## SDK Examples

### Python SDK
```python
from pynomaly import PynomalyClient

client = PynomalyClient(
    api_key="your_api_key",
    base_url="http://localhost:8000"
)

# Create detector
detector = client.detectors.create(
    name="My Detector",
    algorithm="IsolationForest",
    parameters={"contamination": 0.1}
)

# Run detection
results = client.detectors.detect(
    detector.id,
    data=[{"feature1": 1.0, "feature2": 2.0}]
)
```

### cURL Examples
```bash
# Create detector
curl -X POST "http://localhost:8000/api/v1/detectors" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Fraud Detector",
    "algorithm": "IsolationForest",
    "parameters": {"contamination": 0.1}
  }'

# Run detection
curl -X POST "http://localhost:8000/api/v1/detectors/123/detect" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{"amount": 100.0, "merchant": "grocery"}]
  }'
```

This REST API provides comprehensive access to all Pynomaly functionality with production-ready features including authentication, rate limiting, error handling, and real-time capabilities.