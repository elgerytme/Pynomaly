# Pynomaly API Documentation


# Pynomaly - Enterprise Anomaly Detection Platform

A comprehensive, production-ready anomaly detection system with enterprise-grade features.

## Features

### Core Detection
- **Anomaly Detection**: Detect anomalies in time series, tabular, and streaming data
- **Multiple Algorithms**: Support for Isolation Forest, One-Class SVM, LSTM Autoencoders, and custom models
- **Ensemble Methods**: Combine multiple detection algorithms for improved accuracy
- **Real-time Processing**: Stream processing for continuous anomaly detection

### MLOps Platform
- **Model Registry**: Centralized model management with versioning and metadata
- **Experiment Tracking**: Track experiments, parameters, and metrics
- **Model Deployment**: Deploy models to development, staging, and production environments
- **Automated Retraining**: Automatic model retraining based on data drift and performance degradation

### Enterprise Features
- **Multi-tenancy**: Complete tenant isolation with role-based access control
- **Audit Logging**: Comprehensive audit trails with compliance support (GDPR, HIPAA, SOX)
- **Security**: JWT authentication, data encryption, and tamper detection
- **Analytics Dashboard**: Real-time insights and business metrics

### Monitoring & Observability
- **Health Monitoring**: System health checks and performance metrics
- **Alerting**: Real-time alerts for anomalies and system issues
- **Compliance Reporting**: Generate compliance reports for regulatory requirements
- **Performance Tracking**: Track model performance and system metrics

## Quick Start

### 1. Authentication
```bash
# Get JWT token
curl -X POST "https://api.pynomaly.com/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'
```

### 2. Basic Anomaly Detection
```bash
# Detect anomalies in data
curl -X POST "https://api.pynomaly.com/api/v1/detection/detect" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
    "algorithm": "isolation_forest",
    "parameters": {"contamination": 0.1}
  }'
```

### 3. Train Custom Model
```bash
# Train a new model
curl -X POST "https://api.pynomaly.com/api/v1/detection/train" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "training_data": "path/to/training/data.csv",
    "algorithm": "lstm_autoencoder",
    "parameters": {"epochs": 100, "batch_size": 32}
  }'
```

## Authentication

Most endpoints require authentication using JWT tokens. Include the token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

### API Key Authentication
For service-to-service authentication, use API keys:

```
X-API-Key: <your-api-key>
```

## Rate Limiting

API calls are rate-limited to ensure fair usage and system stability:
- **Standard users**: 1000 requests per hour
- **Enterprise users**: 10000 requests per hour
- **Internal services**: Unlimited

## Error Handling

The API uses standard HTTP status codes and returns detailed error messages following RFC 7807:

```json
{
  "type": "https://pynomaly.com/errors/validation-error",
  "title": "Validation Error",
  "status": 422,
  "detail": "The request data is invalid",
  "instance": "/api/v1/detection/detect",
  "errors": [
    {
      "field": "data",
      "message": "Field required"
    }
  ]
}
```

## Support

- **Documentation**: https://docs.pynomaly.com
- **Support**: support@pynomaly.com
- **Community**: https://community.pynomaly.com


## Generated Documentation

This directory contains comprehensive API documentation for the Pynomaly platform:

- **`openapi.json`** - OpenAPI 3.0 specification in JSON format
- **`openapi.yaml`** - OpenAPI 3.0 specification in YAML format
- **`examples/`** - Client code examples in various languages

## API Endpoints

### Authentication
- `POST /api/v1/auth/login` - User authentication
- `POST /api/v1/auth/refresh` - Token refresh
- `GET /api/v1/auth/me` - Get current user profile

### Anomaly Detection
- `POST /api/v1/detection/detect` - Detect anomalies in data
- `POST /api/v1/detection/train` - Train anomaly detection model
- `POST /api/v1/detection/batch` - Batch anomaly detection

### Model Management
- `GET /api/v1/models` - List available models
- `GET /api/v1/models/{model_id}` - Get model details

### Health & Monitoring
- `GET /api/v1/health` - System health check
- `GET /api/v1/health/metrics` - System metrics

## Usage Examples

### Authentication
```bash
curl -X POST "https://api.pynomaly.com/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'
```

### Anomaly Detection
```bash
curl -X POST "https://api.pynomaly.com/api/v1/detection/detect" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
    "algorithm": "isolation_forest",
    "parameters": {"contamination": 0.1}
  }'
```

## Interactive Documentation

- **Swagger UI**: Available at `/api/v1/docs`
- **ReDoc**: Available at `/api/v1/redoc`

## Client Libraries

See the `examples/` directory for client code examples in:
- Python
- JavaScript/TypeScript
- cURL
- Java
- Go

## Support

- **Documentation**: https://docs.pynomaly.com
- **Support**: support@pynomaly.com
- **Community**: https://community.pynomaly.com

---

Generated on: 2025-07-09 16:46:53
