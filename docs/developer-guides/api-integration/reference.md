# API Reference Documentation

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 👨‍💻 [Developer Guides](../README.md) > 🔌 [API Integration](README.md) > 📄 Reference

---


This comprehensive API reference provides complete documentation for all Pynomaly REST API endpoints, including request/response schemas, authentication, error handling, and code examples.

## Table of Contents

1. [API Overview](#api-overview)
2. [Authentication](#authentication)
3. [Error Handling](#error-handling)
4. [Rate Limiting](#rate-limiting)
5. [Core Endpoints](#core-endpoints)
6. [Detector Management](#detector-management)
7. [Dataset Management](#dataset-management)
8. [Detection Operations](#detection-operations)
9. [Experiment Management](#experiment-management)
10. [Model Management](#model-management)
11. [WebSocket API](#websocket-api)
12. [SDK Examples](#sdk-examples)

## API Overview

### Base URL

```
Production: https://api.pynomaly.com/v1
Development: http://localhost:8000/v1
```

### Content Types

- **Request**: `application/json`
- **Response**: `application/json`
- **File Upload**: `multipart/form-data`

### API Versioning

The API uses URL versioning with the version number in the path. Current version: `v1`

### OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:
- **Interactive Documentation**: `/docs` (Swagger UI)
- **OpenAPI JSON**: `/openapi.json`
- **ReDoc**: `/redoc`

## Authentication

### API Key Authentication

Include your API key in the `Authorization` header:

```http
Authorization: Bearer your-api-key-here
```

### JWT Token Authentication

For user sessions, use JWT tokens:

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Authentication Endpoints

#### POST /auth/login
Authenticate with username/password and receive JWT tokens.

**Request:**
```json
{
  "username": "string",
  "password": "string",
  "remember_me": false
}
```

**Response:**
```json
{
  "access_token": "string",
  "refresh_token": "string",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "id": "uuid",
    "username": "string",
    "email": "string",
    "roles": ["string"]
  }
}
```

#### POST /auth/refresh
Refresh access token using refresh token.

**Request:**
```json
{
  "refresh_token": "string"
}
```

**Response:**
```json
{
  "access_token": "string",
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### POST /auth/logout
Invalidate current tokens.

**Request:**
```json
{
  "refresh_token": "string"
}
```

**Response:**
```json
{
  "message": "Successfully logged out"
}
```

### API Key Management

#### GET /auth/api-keys
List user's API keys.

**Response:**
```json
{
  "api_keys": [
    {
      "id": "uuid",
      "name": "string",
      "prefix": "pyn_****",
      "created_at": "2024-01-01T00:00:00Z",
      "last_used": "2024-01-01T00:00:00Z",
      "expires_at": "2024-12-31T23:59:59Z",
      "permissions": ["read", "write"]
    }
  ]
}
```

#### POST /auth/api-keys
Create new API key.

**Request:**
```json
{
  "name": "string",
  "permissions": ["read", "write"],
  "expires_at": "2024-12-31T23:59:59Z"
}
```

**Response:**
```json
{
  "id": "uuid",
  "name": "string",
  "api_key": "pyn_1234567890abcdef",
  "permissions": ["read", "write"],
  "expires_at": "2024-12-31T23:59:59Z"
}
```

#### DELETE /auth/api-keys/{key_id}
Revoke API key.

**Response:**
```json
{
  "message": "API key revoked successfully"
}
```

## Error Handling

### Standard Error Response

All error responses follow this format:

```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": "string",
    "timestamp": "2024-01-01T00:00:00Z",
    "request_id": "uuid"
  }
}
```

### HTTP Status Codes

- **200 OK**: Request successful
- **201 Created**: Resource created successfully
- **204 No Content**: Request successful, no content to return
- **400 Bad Request**: Invalid request parameters
- **401 Unauthorized**: Authentication required or invalid
- **403 Forbidden**: Insufficient permissions
- **404 Not Found**: Resource not found
- **409 Conflict**: Resource conflict (e.g., duplicate name)
- **422 Unprocessable Entity**: Validation errors
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server error
- **503 Service Unavailable**: Service temporarily unavailable

### Common Error Codes

```json
{
  "VALIDATION_ERROR": "Request validation failed",
  "AUTHENTICATION_FAILED": "Authentication credentials invalid",
  "AUTHORIZATION_FAILED": "Insufficient permissions",
  "RESOURCE_NOT_FOUND": "Requested resource not found",
  "RESOURCE_CONFLICT": "Resource already exists",
  "RATE_LIMIT_EXCEEDED": "API rate limit exceeded",
  "DETECTOR_TRAINING_FAILED": "Detector training failed",
  "DATASET_INVALID": "Dataset validation failed",
  "INTERNAL_ERROR": "Internal server error occurred"
}
```

## Rate Limiting

### Rate Limits

- **Authenticated users**: 1000 requests per hour
- **Unauthenticated users**: 100 requests per hour
- **Premium users**: 10000 requests per hour

### Rate Limit Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

## Core Endpoints

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "storage": "healthy"
  }
}
```

### GET /status
Detailed system status.

**Response:**
```json
{
  "system": {
    "uptime": 3600,
    "memory_usage": 0.65,
    "cpu_usage": 0.25,
    "disk_usage": 0.45
  },
  "metrics": {
    "active_detectors": 15,
    "total_datasets": 42,
    "detections_today": 1234
  }
}
```

### GET /version
API version information.

**Response:**
```json
{
  "api_version": "1.0.0",
  "pynomaly_version": "1.0.0",
  "supported_algorithms": ["IsolationForest", "LOF", "OCSVM"],
  "features": ["ensemble", "streaming", "explainability"]
}
```

## Detector Management

### GET /detectors
List all detectors with filtering and pagination.

**Query Parameters:**
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 20, max: 100)
- `algorithm`: Filter by algorithm name
- `status`: Filter by status (trained, untrained, training, failed)
- `search`: Search in detector names and descriptions

**Response:**
```json
{
  "detectors": [
    {
      "id": "uuid",
      "name": "string",
      "description": "string",
      "algorithm": "IsolationForest",
      "status": "trained",
      "parameters": {
        "contamination": 0.1,
        "n_estimators": 100
      },
      "performance": {
        "training_time": 12.5,
        "memory_usage": 256,
        "accuracy": 0.95
      },
      "created_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-01-01T00:00:00Z",
      "trained_at": "2024-01-01T00:00:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 100,
    "pages": 5
  }
}
```

### POST /detectors
Create a new detector.

**Request:**
```json
{
  "name": "string",
  "description": "string",
  "algorithm": "IsolationForest",
  "parameters": {
    "contamination": 0.1,
    "n_estimators": 100,
    "max_samples": "auto"
  },
  "tags": ["production", "fraud-detection"]
}
```

**Response:**
```json
{
  "id": "uuid",
  "name": "string",
  "description": "string",
  "algorithm": "IsolationForest",
  "status": "untrained",
  "parameters": {
    "contamination": 0.1,
    "n_estimators": 100,
    "max_samples": "auto"
  },
  "tags": ["production", "fraud-detection"],
  "created_at": "2024-01-01T00:00:00Z"
}
```

### GET /detectors/{detector_id}
Get detector details.

**Response:**
```json
{
  "id": "uuid",
  "name": "string",
  "description": "string",
  "algorithm": "IsolationForest",
  "status": "trained",
  "parameters": {
    "contamination": 0.1,
    "n_estimators": 100
  },
  "performance": {
    "training_time": 12.5,
    "memory_usage": 256,
    "accuracy": 0.95,
    "f1_score": 0.92,
    "precision": 0.94,
    "recall": 0.90
  },
  "training_history": [
    {
      "dataset_id": "uuid",
      "started_at": "2024-01-01T00:00:00Z",
      "completed_at": "2024-01-01T00:00:10Z",
      "metrics": {
        "accuracy": 0.95,
        "f1_score": 0.92
      }
    }
  ],
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

### PUT /detectors/{detector_id}
Update detector configuration.

**Request:**
```json
{
  "name": "string",
  "description": "string",
  "parameters": {
    "contamination": 0.15,
    "n_estimators": 200
  },
  "tags": ["production", "updated"]
}
```

### DELETE /detectors/{detector_id}
Delete detector.

**Response:**
```json
{
  "message": "Detector deleted successfully"
}
```

### POST /detectors/{detector_id}/train
Train detector with dataset.

**Request:**
```json
{
  "dataset_id": "uuid",
  "validation_split": 0.2,
  "cross_validation": {
    "enabled": true,
    "folds": 5
  },
  "early_stopping": {
    "enabled": true,
    "patience": 10,
    "metric": "f1_score"
  }
}
```

**Response:**
```json
{
  "training_job_id": "uuid",
  "status": "started",
  "estimated_duration": 300,
  "started_at": "2024-01-01T00:00:00Z"
}
```

### GET /detectors/{detector_id}/training-status
Get training status.

**Response:**
```json
{
  "training_job_id": "uuid",
  "status": "training",
  "progress": 0.65,
  "current_step": "validation",
  "estimated_remaining": 120,
  "metrics": {
    "current_accuracy": 0.87,
    "best_accuracy": 0.89,
    "training_loss": 0.23
  },
  "started_at": "2024-01-01T00:00:00Z"
}
```

### POST /detectors/{detector_id}/evaluate
Evaluate detector performance.

**Request:**
```json
{
  "dataset_id": "uuid",
  "metrics": ["accuracy", "precision", "recall", "f1_score", "auc"],
  "cross_validation": {
    "enabled": true,
    "folds": 5
  }
}
```

**Response:**
```json
{
  "evaluation_id": "uuid",
  "metrics": {
    "accuracy": 0.95,
    "precision": 0.94,
    "recall": 0.90,
    "f1_score": 0.92,
    "auc": 0.96
  },
  "cross_validation": {
    "mean_accuracy": 0.93,
    "std_accuracy": 0.02,
    "fold_scores": [0.95, 0.92, 0.94, 0.91, 0.93]
  },
  "confusion_matrix": [[850, 50], [30, 70]],
  "classification_report": {
    "normal": {"precision": 0.96, "recall": 0.94, "f1-score": 0.95},
    "anomaly": {"precision": 0.58, "recall": 0.70, "f1-score": 0.64}
  }
}
```

## Dataset Management

### GET /datasets
List all datasets.

**Query Parameters:**
- `page`: Page number
- `limit`: Items per page
- `format`: Filter by format (csv, parquet, json)
- `status`: Filter by status (valid, invalid, processing)
- `search`: Search in dataset names

**Response:**
```json
{
  "datasets": [
    {
      "id": "uuid",
      "name": "string",
      "description": "string",
      "format": "csv",
      "size_bytes": 1048576,
      "row_count": 10000,
      "column_count": 20,
      "status": "valid",
      "statistics": {
        "numerical_columns": 15,
        "categorical_columns": 5,
        "missing_values": 0.02,
        "duplicate_rows": 0
      },
      "created_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-01-01T00:00:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 50,
    "pages": 3
  }
}
```

### POST /datasets
Create dataset by uploading file.

**Request (multipart/form-data):**
```
file: [binary data]
name: "My Dataset"
description: "Sample dataset for testing"
format: "csv"
has_header: true
delimiter: ","
encoding: "utf-8"
```

**Response:**
```json
{
  "id": "uuid",
  "name": "My Dataset",
  "description": "Sample dataset for testing",
  "format": "csv",
  "size_bytes": 1048576,
  "status": "processing",
  "upload_id": "uuid",
  "created_at": "2024-01-01T00:00:00Z"
}
```

### GET /datasets/{dataset_id}
Get dataset details.

**Response:**
```json
{
  "id": "uuid",
  "name": "string",
  "description": "string",
  "format": "csv",
  "size_bytes": 1048576,
  "row_count": 10000,
  "column_count": 20,
  "status": "valid",
  "columns": [
    {
      "name": "feature_1",
      "type": "float64",
      "null_count": 0,
      "unique_count": 8743,
      "statistics": {
        "mean": 45.2,
        "std": 12.8,
        "min": 0.1,
        "max": 99.9,
        "quartiles": [25.1, 45.2, 65.3]
      }
    }
  ],
  "quality_report": {
    "overall_score": 0.92,
    "issues": [
      {
        "type": "missing_values",
        "severity": "low",
        "count": 42,
        "description": "Missing values in optional columns"
      }
    ]
  },
  "created_at": "2024-01-01T00:00:00Z"
}
```

### GET /datasets/{dataset_id}/sample
Get sample data from dataset.

**Query Parameters:**
- `rows`: Number of rows to return (default: 100, max: 1000)
- `random`: Whether to return random sample (default: false)

**Response:**
```json
{
  "columns": ["feature_1", "feature_2", "label"],
  "data": [
    [1.5, 2.3, 0],
    [2.1, 3.7, 1],
    [0.8, 1.2, 0]
  ],
  "row_count": 100,
  "total_rows": 10000
}
```

### POST /datasets/{dataset_id}/validate
Validate dataset for anomaly detection.

**Request:**
```json
{
  "checks": ["missing_values", "data_types", "outliers", "duplicates"],
  "strict_mode": false,
  "quality_threshold": 0.8
}
```

**Response:**
```json
{
  "validation_id": "uuid",
  "status": "completed",
  "overall_score": 0.92,
  "checks": {
    "missing_values": {
      "passed": true,
      "score": 0.98,
      "details": "2% missing values, within acceptable range"
    },
    "data_types": {
      "passed": true,
      "score": 1.0,
      "details": "All columns have consistent data types"
    },
    "outliers": {
      "passed": true,
      "score": 0.85,
      "details": "15% outliers detected, may indicate anomalies"
    },
    "duplicates": {
      "passed": true,
      "score": 1.0,
      "details": "No duplicate rows found"
    }
  },
  "recommendations": [
    "Consider feature scaling for better performance",
    "Remove or impute missing values in critical columns"
  ]
}
```

### DELETE /datasets/{dataset_id}
Delete dataset.

**Response:**
```json
{
  "message": "Dataset deleted successfully"
}
```

## Detection Operations

### POST /detect
Perform anomaly detection on new data.

**Request:**
```json
{
  "detector_id": "uuid",
  "data": [
    [1.5, 2.3, 4.1],
    [2.1, 3.7, 5.2],
    [0.8, 1.2, 2.9]
  ],
  "options": {
    "return_scores": true,
    "return_explanations": true,
    "threshold": 0.5
  }
}
```

**Response:**
```json
{
  "detection_id": "uuid",
  "predictions": [0, 1, 0],
  "scores": [0.25, 0.85, 0.15],
  "explanations": [
    {
      "sample_index": 1,
      "prediction": 1,
      "score": 0.85,
      "feature_importance": {
        "feature_0": 0.4,
        "feature_1": 0.35,
        "feature_2": 0.25
      },
      "local_explanation": "High values in feature_0 and feature_1 indicate anomaly"
    }
  ],
  "processed_at": "2024-01-01T00:00:00Z"
}
```

### POST /detect/batch
Batch anomaly detection on dataset.

**Request:**
```json
{
  "detector_id": "uuid",
  "dataset_id": "uuid",
  "options": {
    "return_scores": true,
    "return_explanations": false,
    "chunk_size": 1000,
    "output_format": "csv"
  }
}
```

**Response:**
```json
{
  "batch_job_id": "uuid",
  "status": "started",
  "estimated_duration": 180,
  "output_location": "s3://bucket/results/batch_job_uuid.csv",
  "started_at": "2024-01-01T00:00:00Z"
}
```

### GET /detect/batch/{job_id}
Get batch detection status.

**Response:**
```json
{
  "batch_job_id": "uuid",
  "status": "completed",
  "progress": 1.0,
  "processed_samples": 10000,
  "anomalies_found": 1250,
  "anomaly_rate": 0.125,
  "output_location": "s3://bucket/results/batch_job_uuid.csv",
  "statistics": {
    "processing_time": 165,
    "samples_per_second": 60.6,
    "memory_peak": 512
  },
  "started_at": "2024-01-01T00:00:00Z",
  "completed_at": "2024-01-01T00:02:45Z"
}
```

### POST /detect/stream
Start streaming detection.

**Request:**
```json
{
  "detector_id": "uuid",
  "stream_config": {
    "input_source": "kafka",
    "topic": "sensor_data",
    "batch_size": 100,
    "window_size": 3600
  },
  "output_config": {
    "destination": "webhook",
    "url": "https://api.example.com/anomalies",
    "threshold": 0.7
  }
}
```

**Response:**
```json
{
  "stream_id": "uuid",
  "status": "active",
  "input_source": "kafka",
  "started_at": "2024-01-01T00:00:00Z"
}
```

### GET /detect/stream/{stream_id}
Get streaming detection status.

**Response:**
```json
{
  "stream_id": "uuid",
  "status": "active",
  "statistics": {
    "samples_processed": 45230,
    "anomalies_detected": 127,
    "current_rate": 15.2,
    "uptime": 3600
  },
  "last_activity": "2024-01-01T01:00:00Z"
}
```

### DELETE /detect/stream/{stream_id}
Stop streaming detection.

**Response:**
```json
{
  "message": "Stream stopped successfully",
  "final_statistics": {
    "total_samples": 45230,
    "total_anomalies": 127,
    "runtime": 3600
  }
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
      "id": "uuid",
      "name": "string",
      "description": "string",
      "status": "completed",
      "detector_configs": [
        {
          "algorithm": "IsolationForest",
          "parameters": {"contamination": 0.1}
        }
      ],
      "dataset_id": "uuid",
      "results": {
        "best_algorithm": "IsolationForest",
        "best_score": 0.95,
        "completion_time": 300
      },
      "created_at": "2024-01-01T00:00:00Z"
    }
  ]
}
```

### POST /experiments
Create new experiment.

**Request:**
```json
{
  "name": "Algorithm Comparison",
  "description": "Compare different algorithms on fraud dataset",
  "dataset_id": "uuid",
  "detector_configs": [
    {
      "algorithm": "IsolationForest",
      "parameters": {"contamination": 0.1, "n_estimators": 100}
    },
    {
      "algorithm": "LOF",
      "parameters": {"contamination": 0.1, "n_neighbors": 20}
    }
  ],
  "evaluation_metrics": ["accuracy", "precision", "recall", "f1_score"],
  "cross_validation": {
    "enabled": true,
    "folds": 5
  }
}
```

**Response:**
```json
{
  "experiment_id": "uuid",
  "name": "Algorithm Comparison",
  "status": "started",
  "estimated_duration": 600,
  "started_at": "2024-01-01T00:00:00Z"
}
```

### GET /experiments/{experiment_id}
Get experiment results.

**Response:**
```json
{
  "id": "uuid",
  "name": "Algorithm Comparison",
  "status": "completed",
  "results": {
    "algorithms": [
      {
        "algorithm": "IsolationForest",
        "metrics": {
          "accuracy": 0.95,
          "precision": 0.92,
          "recall": 0.88,
          "f1_score": 0.90
        },
        "training_time": 12.5,
        "prediction_time": 0.003
      },
      {
        "algorithm": "LOF",
        "metrics": {
          "accuracy": 0.91,
          "precision": 0.89,
          "recall": 0.85,
          "f1_score": 0.87
        },
        "training_time": 25.3,
        "prediction_time": 0.015
      }
    ],
    "best_algorithm": "IsolationForest",
    "best_metric": "f1_score",
    "best_score": 0.90
  },
  "completed_at": "2024-01-01T00:10:00Z"
}
```

## Model Management

### GET /models
List trained models.

**Response:**
```json
{
  "models": [
    {
      "id": "uuid",
      "detector_id": "uuid",
      "version": "1.0.0",
      "algorithm": "IsolationForest",
      "size_bytes": 1048576,
      "performance": {
        "accuracy": 0.95,
        "training_time": 12.5
      },
      "created_at": "2024-01-01T00:00:00Z"
    }
  ]
}
```

### POST /models/{model_id}/export
Export trained model.

**Request:**
```json
{
  "format": "pickle",
  "include_metadata": true
}
```

**Response:**
```json
{
  "export_id": "uuid",
  "download_url": "https://api.pynomaly.com/downloads/model_uuid.pkl",
  "expires_at": "2024-01-01T01:00:00Z",
  "size_bytes": 1048576
}
```

### POST /models/import
Import pre-trained model.

**Request (multipart/form-data):**
```
file: [binary model file]
metadata: {
  "algorithm": "IsolationForest",
  "version": "1.0.0",
  "parameters": {"contamination": 0.1}
}
```

**Response:**
```json
{
  "model_id": "uuid",
  "detector_id": "uuid",
  "status": "imported",
  "validation_results": {
    "format_valid": true,
    "parameters_valid": true,
    "compatibility_score": 1.0
  }
}
```

## WebSocket API

### Connection

Connect to WebSocket for real-time updates:

```javascript
const ws = new WebSocket('wss://api.pynomaly.com/ws');
```

### Authentication

Send authentication message after connection:

```json
{
  "type": "auth",
  "token": "your-jwt-token"
}
```

### Subscriptions

Subscribe to specific events:

```json
{
  "type": "subscribe",
  "channels": ["detector.training", "detection.anomaly", "system.alerts"]
}
```

### Event Types

#### Training Updates
```json
{
  "type": "detector.training",
  "detector_id": "uuid",
  "status": "training",
  "progress": 0.65,
  "metrics": {
    "current_accuracy": 0.87
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

#### Anomaly Alerts
```json
{
  "type": "detection.anomaly",
  "detector_id": "uuid",
  "anomaly": {
    "score": 0.95,
    "sample": [1.5, 2.3, 4.1],
    "explanation": "High values in features 0 and 1"
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

#### System Alerts
```json
{
  "type": "system.alert",
  "level": "warning",
  "message": "High memory usage detected",
  "details": {
    "memory_usage": 0.85,
    "threshold": 0.80
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## SDK Examples

### Python SDK

```python
from pynomaly import PynomalyClient

# Initialize client
client = PynomalyClient(
    api_key="your-api-key",
    base_url="https://api.pynomaly.com/v1"
)

# Create detector
detector = client.detectors.create(
    name="Fraud Detector",
    algorithm="IsolationForest",
    parameters={"contamination": 0.1, "n_estimators": 100}
)

# Upload dataset
dataset = client.datasets.upload(
    file_path="data.csv",
    name="Training Data",
    has_header=True
)

# Train detector
training_job = client.detectors.train(
    detector_id=detector.id,
    dataset_id=dataset.id
)

# Wait for training completion
training_job.wait_for_completion()

# Perform detection
results = client.detect(
    detector_id=detector.id,
    data=[[1.5, 2.3, 4.1], [2.1, 3.7, 5.2]],
    return_scores=True
)

print(f"Predictions: {results.predictions}")
print(f"Scores: {results.scores}")
```

### JavaScript SDK

```javascript
import { PynomalyClient } from '@pynomaly/client';

// Initialize client
const client = new PynomalyClient({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.pynomaly.com/v1'
});

// Create detector
const detector = await client.detectors.create({
  name: 'Fraud Detector',
  algorithm: 'IsolationForest',
  parameters: { contamination: 0.1, n_estimators: 100 }
});

// Upload dataset
const dataset = await client.datasets.upload({
  file: fileObject,
  name: 'Training Data',
  hasHeader: true
});

// Train detector
const trainingJob = await client.detectors.train({
  detectorId: detector.id,
  datasetId: dataset.id
});

// Monitor training progress
trainingJob.onProgress((progress) => {
  console.log(`Training progress: ${progress.percentage}%`);
});

await trainingJob.waitForCompletion();

// Perform detection
const results = await client.detect({
  detectorId: detector.id,
  data: [[1.5, 2.3, 4.1], [2.1, 3.7, 5.2]],
  returnScores: true
});

console.log('Predictions:', results.predictions);
console.log('Scores:', results.scores);
```

### cURL Examples

#### Create Detector
```bash
curl -X POST "https://api.pynomaly.com/v1/detectors" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Fraud Detector",
    "algorithm": "IsolationForest",
    "parameters": {
      "contamination": 0.1,
      "n_estimators": 100
    }
  }'
```

#### Upload Dataset
```bash
curl -X POST "https://api.pynomaly.com/v1/datasets" \
  -H "Authorization: Bearer your-api-key" \
  -F "file=@data.csv" \
  -F "name=Training Data" \
  -F "has_header=true"
```

#### Perform Detection
```bash
curl -X POST "https://api.pynomaly.com/v1/detect" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "detector_id": "detector-uuid",
    "data": [[1.5, 2.3, 4.1], [2.1, 3.7, 5.2]],
    "options": {
      "return_scores": true,
      "return_explanations": true
    }
  }'
```

### Error Handling Examples

#### Python
```python
from pynomaly.exceptions import PynomalyAPIError, ValidationError

try:
    detector = client.detectors.create(
        name="Test Detector",
        algorithm="InvalidAlgorithm"
    )
except ValidationError as e:
    print(f"Validation error: {e.message}")
    print(f"Details: {e.details}")
except PynomalyAPIError as e:
    print(f"API error: {e.message}")
    print(f"Status code: {e.status_code}")
```

#### JavaScript
```javascript
try {
  const detector = await client.detectors.create({
    name: 'Test Detector',
    algorithm: 'InvalidAlgorithm'
  });
} catch (error) {
  if (error instanceof ValidationError) {
    console.error('Validation error:', error.message);
    console.error('Details:', error.details);
  } else if (error instanceof PynomalyAPIError) {
    console.error('API error:', error.message);
    console.error('Status code:', error.statusCode);
  }
}
```

This comprehensive API reference provides complete documentation for all Pynomaly endpoints, including authentication, error handling, and practical examples for common use cases. The API is designed to be RESTful, well-documented, and easy to integrate with various programming languages and frameworks.

---

## 🔗 **Related Documentation**

### **Development**
- **[Contributing Guidelines](../contributing/CONTRIBUTING.md)** - How to contribute
- **[Development Setup](../contributing/README.md)** - Local development environment
- **[Architecture Overview](../architecture/overview.md)** - System design
- **[Implementation Guide](../contributing/IMPLEMENTATION_GUIDE.md)** - Coding standards

### **API Integration**
- **[REST API](../api-integration/rest-api.md)** - HTTP API reference
- **[Python SDK](../api-integration/python-sdk.md)** - Python client library
- **[CLI Reference](../api-integration/cli.md)** - Command-line interface
- **[Authentication](../api-integration/authentication.md)** - Security and auth

### **User Documentation**
- **[User Guides](../../user-guides/README.md)** - Feature usage guides
- **[Getting Started](../../getting-started/README.md)** - Installation and setup
- **[Examples](../../examples/README.md)** - Real-world use cases

### **Deployment**
- **[Production Deployment](../../deployment/README.md)** - Production deployment
- **[Security Setup](../../deployment/SECURITY.md)** - Security configuration
- **[Monitoring](../../user-guides/basic-usage/monitoring.md)** - System observability

---

## 🆘 **Getting Help**

- **[Development Troubleshooting](../contributing/troubleshooting/)** - Development issues
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs
- **[Contributing Guidelines](../contributing/CONTRIBUTING.md)** - Contribution process
