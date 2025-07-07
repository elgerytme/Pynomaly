# Pynomaly API Documentation

This document provides comprehensive documentation for the Pynomaly REST API, which offers 65+ endpoints for anomaly detection, model management, and system administration.

## Quick Start

### Authentication
Most API endpoints require authentication. Obtain a JWT token using the login endpoint:

```bash
curl -X POST "http://localhost:8000/api/auth/login" \
     -H "Content-Type: application/json" \
     -d '{"username": "admin", "password": "admin"}'
```

Use the token in subsequent requests:
```bash
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     "http://localhost:8000/api/health"
```

### Base URL
- **Development**: `http://localhost:8000`
- **Production**: Configure according to your deployment

## API Endpoints Overview

### 🩺 Health & Status
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Basic health check |
| `/api/health/detailed` | GET | Detailed system health |
| `/api/health/dependencies` | GET | Check external dependencies |

### 🔐 Authentication & Authorization
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/login` | POST | User login |
| `/api/auth/logout` | POST | User logout |
| `/api/auth/refresh` | POST | Refresh JWT token |
| `/api/auth/register` | POST | User registration |

### 👤 Administration
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/admin/users` | GET | List users |
| `/api/admin/users` | POST | Create user |
| `/api/admin/users/{user_id}` | PUT | Update user |
| `/api/admin/users/{user_id}` | DELETE | Delete user |
| `/api/admin/system/status` | GET | System status |

### 🤖 Autonomous Operations
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/autonomous/enable` | POST | Enable autonomous mode |
| `/api/autonomous/disable` | POST | Disable autonomous mode |
| `/api/autonomous/status` | GET | Get autonomous status |

### 🔍 Detector Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/detectors` | GET | List detectors |
| `/api/detectors` | POST | Create detector |
| `/api/detectors/{detector_id}` | GET | Get detector details |
| `/api/detectors/{detector_id}` | PUT | Update detector |
| `/api/detectors/{detector_id}` | DELETE | Delete detector |
| `/api/detectors/{detector_id}/train` | POST | Train detector |
| `/api/detectors/algorithms` | GET | List available algorithms |

### 📊 Dataset Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/datasets` | GET | List datasets |
| `/api/datasets` | POST | Create dataset |
| `/api/datasets/{dataset_id}` | GET | Get dataset details |
| `/api/datasets/{dataset_id}` | PUT | Update dataset |
| `/api/datasets/{dataset_id}` | DELETE | Delete dataset |
| `/api/datasets/upload` | POST | Upload dataset file |
| `/api/datasets/{dataset_id}/preview` | GET | Preview dataset |

### 🎯 Anomaly Detection
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/detection/predict` | POST | Run anomaly detection |
| `/api/detection/batch` | POST | Batch anomaly detection |
| `/api/detection/stream` | POST | Stream anomaly detection |
| `/api/detection/results` | GET | List detection results |
| `/api/detection/results/{result_id}` | GET | Get detection result |

### 🚀 AutoML
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/automl/create` | POST | Create AutoML pipeline |
| `/api/automl/pipelines` | GET | List pipelines |
| `/api/automl/pipelines/{pipeline_id}` | GET | Get pipeline details |
| `/api/automl/pipelines/{pipeline_id}/start` | POST | Start pipeline |
| `/api/automl/pipelines/{pipeline_id}/stop` | POST | Stop pipeline |

### 🎭 Ensemble Methods
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ensemble/create` | POST | Create ensemble |
| `/api/ensemble/list` | GET | List ensembles |
| `/api/ensemble/{ensemble_id}` | GET | Get ensemble details |
| `/api/ensemble/{ensemble_id}/predict` | POST | Ensemble prediction |

### 🔍 Explainability
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/explainability/shap` | POST | Generate SHAP explanations |
| `/api/explainability/lime` | POST | Generate LIME explanations |
| `/api/explainability/feature-importance` | POST | Feature importance analysis |

### 🧪 Experiments
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/experiments` | GET | List experiments |
| `/api/experiments` | POST | Create experiment |
| `/api/experiments/{experiment_id}` | GET | Get experiment details |
| `/api/experiments/{experiment_id}/run` | POST | Run experiment |

### 📈 Performance
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/performance/metrics` | GET | Get performance metrics |
| `/api/performance/benchmark` | POST | Run performance benchmark |
| `/api/performance/monitor` | GET | Real-time monitoring |

### 📤 Export
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/export/formats` | GET | List export formats |
| `/api/export/csv` | POST | Export to CSV |
| `/api/export/excel` | POST | Export to Excel |
| `/api/export/json` | POST | Export to JSON |

### 🌊 Streaming
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/streaming/start` | POST | Start streaming detection |
| `/api/streaming/stop` | POST | Stop streaming detection |
| `/api/streaming/status` | GET | Get streaming status |

## Detailed API Reference

### Authentication

#### Login
```http
POST /api/auth/login
Content-Type: application/json

{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Detector Management

#### Create Detector
```http
POST /api/detectors
Authorization: Bearer {token}
Content-Type: application/json

{
  "name": "isolation_forest_detector",
  "algorithm": "IsolationForest",
  "parameters": {
    "contamination": 0.1,
    "n_estimators": 100,
    "random_state": 42
  },
  "description": "Isolation Forest detector for outlier detection"
}
```

**Response:**
```json
{
  "id": "uuid-string",
  "name": "isolation_forest_detector",
  "algorithm": "IsolationForest",
  "parameters": {
    "contamination": 0.1,
    "n_estimators": 100,
    "random_state": 42
  },
  "status": "created",
  "created_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T12:00:00Z"
}
```

#### Train Detector
```http
POST /api/detectors/{detector_id}/train
Authorization: Bearer {token}
Content-Type: application/json

{
  "dataset_id": "uuid-string",
  "validation_split": 0.2,
  "training_options": {
    "enable_validation": true,
    "early_stopping": false
  }
}
```

### Dataset Management

#### Upload Dataset
```http
POST /api/datasets/upload
Authorization: Bearer {token}
Content-Type: multipart/form-data

file: [CSV/Parquet file]
name: "my_dataset"
description: "Sample dataset for anomaly detection"
```

**Response:**
```json
{
  "id": "uuid-string",
  "name": "my_dataset",
  "description": "Sample dataset for anomaly detection",
  "file_path": "/path/to/dataset.csv",
  "rows": 10000,
  "columns": 15,
  "numeric_columns": 12,
  "categorical_columns": 3,
  "created_at": "2024-01-01T12:00:00Z"
}
```

### Anomaly Detection

#### Run Detection
```http
POST /api/detection/predict
Authorization: Bearer {token}
Content-Type: application/json

{
  "detector_id": "uuid-string",
  "dataset_id": "uuid-string",
  "options": {
    "return_confidence": true,
    "return_feature_importance": false,
    "batch_size": 1000
  }
}
```

**Response:**
```json
{
  "result_id": "uuid-string",
  "detector_id": "uuid-string",
  "dataset_id": "uuid-string",
  "anomalies_count": 156,
  "anomalies": [
    {
      "index": 42,
      "score": 0.8532,
      "confidence": 0.9234,
      "features": {...}
    }
  ],
  "execution_time_seconds": 2.45,
  "created_at": "2024-01-01T12:00:00Z"
}
```

### Performance Monitoring

#### Get Performance Metrics
```http
GET /api/performance/metrics
Authorization: Bearer {token}
```

**Response:**
```json
{
  "system": {
    "cpu_usage_percent": 45.2,
    "memory_usage_percent": 68.5,
    "disk_usage_percent": 32.1
  },
  "application": {
    "active_detectors": 5,
    "cached_models": 12,
    "detection_requests_per_minute": 23.4
  },
  "performance": {
    "average_detection_time_ms": 125.6,
    "cache_hit_rate_percent": 87.3,
    "error_rate_percent": 0.1
  }
}
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error information:

### Error Response Format
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "The provided data is invalid",
    "details": {
      "field": "contamination",
      "reason": "Value must be between 0.0 and 1.0"
    },
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "uuid-string"
  }
}
```

### Common HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Access denied |
| 404 | Not Found | Resource not found |
| 422 | Unprocessable Entity | Validation error |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |

### Error Codes

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Input validation failed |
| `AUTHENTICATION_ERROR` | Authentication failed |
| `AUTHORIZATION_ERROR` | Insufficient permissions |
| `RESOURCE_NOT_FOUND` | Requested resource not found |
| `TRAINING_ERROR` | Model training failed |
| `DETECTION_ERROR` | Anomaly detection failed |
| `SYSTEM_ERROR` | Internal system error |

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Default**: 100 requests per minute per user
- **Burst**: Up to 20 requests in 10 seconds
- **Headers**: Rate limit information in response headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 85
X-RateLimit-Reset: 1640995200
```

## Webhooks (Beta)

Configure webhooks to receive notifications for important events:

### Events
- `detection.completed` - Anomaly detection completed
- `training.completed` - Model training completed
- `alert.triggered` - System alert triggered

### Webhook Configuration
```http
POST /api/webhooks
Authorization: Bearer {token}
Content-Type: application/json

{
  "url": "https://your-app.com/webhooks/pynomaly",
  "events": ["detection.completed", "training.completed"],
  "secret": "your-webhook-secret"
}
```

## SDK Integration

### Python SDK
```python
from pynomaly_client import PynomalyClient

client = PynomalyClient(
    base_url="http://localhost:8000",
    token="your-jwt-token"
)

# Create detector
detector = client.detectors.create(
    name="my_detector",
    algorithm="IsolationForest",
    parameters={"contamination": 0.1}
)

# Upload dataset
dataset = client.datasets.upload(
    file_path="data.csv",
    name="my_dataset"
)

# Train detector
training_result = client.detectors.train(
    detector_id=detector.id,
    dataset_id=dataset.id
)

# Run detection
results = client.detection.predict(
    detector_id=detector.id,
    dataset_id=dataset.id
)
```

### JavaScript SDK
```javascript
import { PynomalyClient } from 'pynomaly-js';

const client = new PynomalyClient({
  baseUrl: 'http://localhost:8000',
  token: 'your-jwt-token'
});

// Create detector
const detector = await client.detectors.create({
  name: 'my_detector',
  algorithm: 'IsolationForest',
  parameters: { contamination: 0.1 }
});

// Run detection
const results = await client.detection.predict({
  detectorId: detector.id,
  datasetId: dataset.id
});
```

## Interactive API Explorer

Visit the interactive API documentation at:
- **Swagger UI**: `http://localhost:8000/api/docs`
- **ReDoc**: `http://localhost:8000/api/redoc`
- **OpenAPI Spec**: `http://localhost:8000/api/openapi.json`

## Support

- **Documentation**: [https://pynomaly.readthedocs.io](https://pynomaly.readthedocs.io)
- **GitHub Issues**: [https://github.com/pynomaly/pynomaly/issues](https://github.com/pynomaly/pynomaly/issues)
- **API Support**: team@pynomaly.io

## Changelog

### v1.3.0 (Latest)
- ✅ Added performance monitoring endpoints
- ✅ Enhanced batch operations for cache
- ✅ Improved memory management
- ✅ Added comprehensive testing infrastructure

### v1.2.0
- ✅ Added streaming detection support
- ✅ Enhanced export functionality
- ✅ Improved authentication system

### v1.1.0
- ✅ Added AutoML endpoints
- ✅ Enhanced explainability features
- ✅ Improved error handling

### v1.0.0
- ✅ Initial stable API release
- ✅ Core detection and management endpoints
- ✅ JWT authentication
- ✅ Basic monitoring
