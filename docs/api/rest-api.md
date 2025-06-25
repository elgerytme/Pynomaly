# Pynomaly REST API Documentation

## Overview

The Pynomaly REST API provides a comprehensive interface for anomaly detection operations, including detector management, dataset handling, model training, and real-time prediction. Built with FastAPI, the API follows RESTful principles and returns JSON responses.

**🏗️ Built with Modern Stack:**
- **FastAPI** - High-performance async web framework
- **Pydantic** - Data validation and serialization  
- **Hatch** - Modern build system and environment management
- **OpenTelemetry** - Observability and monitoring
- **Prometheus** - Metrics collection

## Base URLs

- **Development**: `http://localhost:8000`
- **API Endpoints**: `http://localhost:8000/api`
- **Interactive Documentation**: `http://localhost:8000/docs`
- **Alternative Documentation**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

## Starting the API Server

Before using the API, start the server:

### Using Hatch (Recommended)

```bash
# Development server with auto-reload
hatch env run prod:serve-api

# Production server with workers
hatch env run prod:serve-api-prod

# Using Makefile shortcuts
make prod-api-dev       # Development mode
make prod-api           # Production mode
```

### Traditional Methods

```bash
# Direct uvicorn
uvicorn pynomaly.presentation.api.app:app --reload

# Using CLI (if installed)
pynomaly server start

# Alternative CLI
python scripts/pynomaly_cli.py server start
```

### Docker Deployment

```bash
# Build and run with Docker
make docker
docker run -p 8000:8000 pynomaly:latest

# Or use docker-compose
docker-compose up -d
```

## API Endpoint Groups

The API is organized into the following endpoint groups:

- **`/api/health`** - Health checks and system status
- **`/api/auth`** - Authentication and authorization
- **`/api/detectors`** - Anomaly detector management
- **`/api/datasets`** - Dataset upload and management
- **`/api/detection`** - Training and anomaly detection
- **`/api/experiments`** - Experiment tracking and comparison
- **`/api/export`** - Data export and business intelligence
- **`/api/performance`** - Performance monitoring and optimization
- **`/api/admin`** - Administrative functions
- **`/api/autonomous`** - Autonomous mode operations

## Authentication

All API endpoints (except health checks) require authentication using JWT tokens.

### Getting an Access Token

```bash
curl -X POST "http://localhost:8000/api/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=your_password"
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Using the Token

Include the token in the Authorization header for all subsequent requests:

```bash
curl -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

## Rate Limiting

API requests are rate-limited to prevent abuse:
- **Authenticated users**: 100 requests per minute
- **Unauthenticated users**: 10 requests per minute

When rate limits are exceeded, the API returns a `429 Too Many Requests` status code.

## Error Handling

The API uses standard HTTP status codes and returns detailed error information:

### Success Codes
- `200 OK` - Request successful
- `201 Created` - Resource created successfully
- `204 No Content` - Resource deleted successfully

### Error Codes
- `400 Bad Request` - Invalid request parameters
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource doesn't exist
- `422 Unprocessable Entity` - Validation error
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

### Error Response Format

```json
{
  "detail": "Human-readable error description",
  "error_code": "MACHINE_READABLE_CODE",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Validation Errors

For validation errors (422), the response includes detailed field-level errors:

```json
{
  "detail": "Validation error",
  "errors": [
    {
      "field": "contamination_rate",
      "message": "Value must be between 0.0 and 0.5",
      "value": "0.8"
    }
  ]
}
```

## API Endpoints

### Health and Monitoring

#### GET /health
Check the basic health status of the API.

**No authentication required**

```bash
curl "http://localhost:8000/api/health"
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0",
  "uptime_seconds": 3600
}
```

#### GET /health/detailed
Get detailed health information about all system components.

```bash
curl "http://localhost:8000/api/health/detailed"
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "components": {
    "database": {
      "status": "healthy",
      "details": {
        "connection_pool": "5/10 active",
        "response_time_ms": 2.5
      }
    },
    "cache": {
      "status": "healthy",
      "details": {
        "memory_usage": "45MB",
        "hit_rate": "85%"
      }
    }
  }
}
```

### Detector Management

#### GET /detectors
List all anomaly detectors with optional filtering.

**Parameters:**
- `algorithm` (string, optional) - Filter by algorithm name
- `is_fitted` (boolean, optional) - Filter by fitted status
- `limit` (integer, optional) - Maximum results (1-1000, default: 100)

```bash
curl -H "Authorization: Bearer TOKEN" \
  "http://localhost:8000/api/detectors?algorithm=IsolationForest&limit=10"
```

**Response:**
```json
[
  {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "name": "Fraud Detection Model",
    "algorithm_name": "IsolationForest",
    "contamination_rate": 0.1,
    "is_fitted": true,
    "hyperparameters": {
      "n_estimators": 100,
      "max_samples": "auto"
    },
    "description": "Model for detecting fraudulent transactions",
    "created_at": "2024-01-01T12:00:00Z",
    "updated_at": "2024-01-01T12:30:00Z"
  }
]
```

#### POST /detectors
Create a new anomaly detector.

**Request Body:**
```json
{
  "name": "Fraud Detection Model",
  "algorithm_name": "IsolationForest",
  "contamination_rate": 0.1,
  "hyperparameters": {
    "n_estimators": 100,
    "max_samples": "auto",
    "contamination": 0.1
  },
  "description": "Model for detecting fraudulent transactions"
}
```

**Response (201):**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "name": "Fraud Detection Model",
  "algorithm_name": "IsolationForest",
  "contamination_rate": 0.1,
  "is_fitted": false,
  "hyperparameters": {
    "n_estimators": 100,
    "max_samples": "auto"
  },
  "created_at": "2024-01-01T12:00:00Z"
}
```

#### GET /detectors/{detector_id}
Get details of a specific detector.

```bash
curl -H "Authorization: Bearer TOKEN" \
  "http://localhost:8000/api/detectors/123e4567-e89b-12d3-a456-426614174000"
```

#### PUT /detectors/{detector_id}
Update detector configuration.

**Request Body:**
```json
{
  "name": "Updated Fraud Detection Model",
  "description": "Enhanced model with better parameters",
  "hyperparameters": {
    "n_estimators": 200,
    "max_samples": 0.8
  }
}
```

#### DELETE /detectors/{detector_id}
Delete a detector and all associated data.

```bash
curl -X DELETE -H "Authorization: Bearer TOKEN" \
  "http://localhost:8000/api/detectors/123e4567-e89b-12d3-a456-426614174000"
```

### Dataset Management

#### GET /datasets
List all datasets with optional filtering.

**Parameters:**
- `format` (string, optional) - Filter by data format (csv, parquet, json)
- `has_target` (boolean, optional) - Filter by presence of target column
- `limit` (integer, optional) - Maximum results (1-1000, default: 100)

```bash
curl -H "Authorization: Bearer TOKEN" \
  "http://localhost:8000/api/datasets?format=csv&has_target=false"
```

**Response:**
```json
[
  {
    "id": "456e7890-e89b-12d3-a456-426614174000",
    "name": "Transaction Data",
    "description": "Credit card transactions for fraud detection",
    "file_format": "csv",
    "n_samples": 10000,
    "n_features": 15,
    "has_target": false,
    "feature_names": ["amount", "merchant_category", "hour_of_day"],
    "file_size_mb": 2.5,
    "created_at": "2024-01-01T11:00:00Z"
  }
]
```

#### POST /datasets
Upload a new dataset.

**Content-Type: multipart/form-data**

```bash
curl -X POST -H "Authorization: Bearer TOKEN" \
  -F "file=@transaction_data.csv" \
  -F "name=Transaction Data" \
  -F "description=Credit card transactions for fraud detection" \
  -F "target_column=is_fraud" \
  "http://localhost:8000/api/datasets"
```

**Response (201):**
```json
{
  "id": "456e7890-e89b-12d3-a456-426614174000",
  "name": "Transaction Data",
  "description": "Credit card transactions for fraud detection",
  "file_format": "csv",
  "n_samples": 10000,
  "n_features": 15,
  "has_target": true,
  "target_column": "is_fraud",
  "file_size_mb": 2.5,
  "created_at": "2024-01-01T11:00:00Z"
}
```

#### GET /datasets/{dataset_id}
Get details of a specific dataset.

#### GET /datasets/{dataset_id}/sample
Get a sample of data from the dataset.

**Parameters:**
- `size` (integer, optional) - Number of samples (1-1000, default: 10)

```bash
curl -H "Authorization: Bearer TOKEN" \
  "http://localhost:8000/api/datasets/456e7890-e89b-12d3-a456-426614174000/sample?size=5"
```

**Response:**
```json
{
  "data": [
    {
      "amount": 150.25,
      "merchant_category": "grocery",
      "hour_of_day": 14
    },
    {
      "amount": 89.99,
      "merchant_category": "gas_station",
      "hour_of_day": 8
    }
  ],
  "total_samples": 10000,
  "sample_size": 5,
  "feature_names": ["amount", "merchant_category", "hour_of_day"]
}
```

#### DELETE /datasets/{dataset_id}
Delete a dataset and all associated data.

### Anomaly Detection

#### POST /detection/train
Train a detector on a specific dataset.

**Request Body:**
```json
{
  "detector_id": "123e4567-e89b-12d3-a456-426614174000",
  "dataset_id": "456e7890-e89b-12d3-a456-426614174000",
  "validation_split": 0.2,
  "cross_validation": false,
  "save_model": true
}
```

**Response:**
```json
{
  "success": true,
  "training_time_ms": 5420,
  "validation_metrics": {
    "precision": 0.92,
    "recall": 0.88,
    "f1_score": 0.90,
    "auc_score": 0.94
  },
  "model_info": {
    "n_samples_trained": 8000,
    "n_features": 15,
    "algorithm": "IsolationForest"
  }
}
```

#### POST /detection/predict
Run anomaly detection on data.

**Request Body (using dataset):**
```json
{
  "detector_id": "123e4567-e89b-12d3-a456-426614174000",
  "dataset_id": "456e7890-e89b-12d3-a456-426614174000",
  "threshold": 0.5,
  "return_scores": true
}
```

**Request Body (using inline data):**
```json
{
  "detector_id": "123e4567-e89b-12d3-a456-426614174000",
  "data": [
    {
      "amount": 1500.00,
      "merchant_category": "online",
      "hour_of_day": 3
    },
    {
      "amount": 45.99,
      "merchant_category": "grocery",
      "hour_of_day": 12
    }
  ],
  "return_scores": true
}
```

**Response:**
```json
{
  "predictions": [1, 0],
  "anomaly_scores": [0.85, 0.15],
  "anomaly_count": 1,
  "anomaly_rate": 0.5,
  "processing_time_ms": 124,
  "threshold_used": 0.5,
  "summary": {
    "total_samples": 2,
    "max_score": 0.85,
    "min_score": 0.15,
    "avg_score": 0.50
  }
}
```

#### POST /detection/explain
Get explanations for anomaly predictions using SHAP or LIME.

**Request Body:**
```json
{
  "detector_id": "123e4567-e89b-12d3-a456-426614174000",
  "instance": {
    "amount": 1500.00,
    "merchant_category": "online",
    "hour_of_day": 3
  },
  "method": "shap",
  "feature_names": ["amount", "merchant_category", "hour_of_day"]
}
```

**Response:**
```json
{
  "explanation": {
    "feature_importance": {
      "amount": 0.75,
      "hour_of_day": 0.20,
      "merchant_category": 0.05
    },
    "values": [0.75, 0.05, 0.20]
  },
  "method_used": "shap",
  "prediction": 0.85,
  "confidence": 0.92
}
```

### Experiment Tracking

#### GET /experiments
List all experiments.

```bash
curl -H "Authorization: Bearer TOKEN" \
  "http://localhost:8000/api/experiments"
```

**Response:**
```json
[
  {
    "id": "789e1234-e89b-12d3-a456-426614174000",
    "name": "Fraud Detection Experiment 1",
    "description": "Testing different algorithms for fraud detection",
    "tags": ["fraud", "comparison"],
    "created_at": "2024-01-01T10:00:00Z",
    "metrics": {
      "best_algorithm": "IsolationForest",
      "best_f1_score": 0.90
    }
  }
]
```

#### POST /experiments
Create a new experiment.

**Request Body:**
```json
{
  "name": "Fraud Detection Experiment 1",
  "description": "Testing different algorithms for fraud detection",
  "tags": ["fraud", "comparison"]
}
```

### Performance Monitoring

#### GET /performance/metrics
Get current performance metrics.

```bash
curl -H "Authorization: Bearer TOKEN" \
  "http://localhost:8000/api/performance/metrics"
```

**Response:**
```json
{
  "cpu_usage_percent": 45.2,
  "memory_usage_mb": 512.8,
  "active_requests": 3,
  "total_requests": 1542,
  "average_response_time_ms": 125.4,
  "error_rate_percent": 0.2
}
```

## SDKs and Integration

### Python SDK

```python
from pynomaly_client import PynomalyClient

# Initialize client
client = PynomalyClient(
    base_url="http://localhost:8000/api",
    username="admin",
    password="password123"
)

# Create detector
detector = client.detectors.create(
    name="Fraud Detection",
    algorithm_name="IsolationForest",
    contamination_rate=0.1
)

# Upload dataset
dataset = client.datasets.upload(
    file_path="data.csv",
    name="Transaction Data"
)

# Train detector
training_result = client.detection.train(
    detector_id=detector.id,
    dataset_id=dataset.id
)

# Make predictions
predictions = client.detection.predict(
    detector_id=detector.id,
    data=[{"amount": 1500.00, "hour": 3}]
)
```

### JavaScript/Node.js SDK

```javascript
const PynomalyClient = require('pynomaly-client');

const client = new PynomalyClient({
  baseUrl: 'http://localhost:8000/api',
  username: 'admin',
  password: 'password123'
});

// Create detector
const detector = await client.detectors.create({
  name: 'Fraud Detection',
  algorithm_name: 'IsolationForest',
  contamination_rate: 0.1
});

// Make predictions
const predictions = await client.detection.predict({
  detector_id: detector.id,
  data: [{ amount: 1500.00, hour: 3 }]
});
```

### cURL Examples

See the individual endpoint documentation above for cURL examples.

## Best Practices

### 1. Authentication
- Store JWT tokens securely
- Refresh tokens before expiration
- Use HTTPS in production

### 2. Error Handling
- Always check response status codes
- Parse error messages for debugging
- Implement retry logic for transient errors

### 3. Performance
- Use appropriate page sizes for list endpoints
- Cache static data (algorithm lists, etc.)
- Implement client-side rate limiting

### 4. Data Management
- Validate data before uploading
- Use appropriate data formats (Parquet for large datasets)
- Clean up unused datasets and detectors

### 5. Security
- Never log or expose JWT tokens
- Validate all input data
- Use API keys for service-to-service communication

## Troubleshooting

### Common Issues

#### 401 Unauthorized
- Check if token is valid and not expired
- Ensure token is included in Authorization header
- Verify username/password for token generation

#### 422 Validation Error
- Check request body format and required fields
- Verify data types match API specification
- Ensure contamination_rate is between 0.0 and 0.5

#### 429 Rate Limited
- Implement exponential backoff in client
- Check current rate limits
- Consider upgrading API plan for higher limits

#### 500 Internal Server Error
- Check server logs for detailed error information
- Verify all required services are running
- Contact support if issue persists

### Getting Help

- **Documentation**: Check the OpenAPI specification at `/api/docs`
- **Support**: Contact support@pynomaly.io
- **Issues**: Report bugs at https://github.com/your-org/pynomaly/issues
- **Community**: Join our Discord server for community support