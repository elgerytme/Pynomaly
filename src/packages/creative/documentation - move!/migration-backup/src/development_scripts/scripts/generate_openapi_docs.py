#!/usr/bin/env python3
"""
Enhanced OpenAPI Documentation Generator for Pynomaly API

This script generates comprehensive OpenAPI documentation with:
- Complete endpoint documentation
- Request/response examples
- Schema definitions
- Authentication details
- Error handling documentation
- Client code examples
"""

import json

# Add src to path for imports
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class OpenAPIDocumentationGenerator:
    """Generate enhanced OpenAPI documentation for Pynomaly API."""

    def __init__(self):
        """Initialize the documentation generator."""
        self.app_info = {
            "title": "Pynomaly API",
            "version": "1.0.0",
            "description": self._get_api_description(),
            "contact": {
                "name": "Pynomaly Support",
                "url": "https://pynomaly.com/support",
                "email": "support@pynomaly.com",
            },
            "license": {
                "name": "MIT License",
                "url": "https://opensource.org/licenses/MIT",
            },
        }

        self.servers = [
            {"url": "https://api.pynomaly.com", "description": "Production server"},
            {
                "url": "https://staging-api.pynomaly.com",
                "description": "Staging server",
            },
            {"url": "http://localhost:8000", "description": "Development server"},
        ]

    def _get_api_description(self) -> str:
        """Get comprehensive API description."""
        return """
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
curl -X POST "https://api.pynomaly.com/api/v1/auth/login" \\
  -H "Content-Type: application/json" \\
  -d '{"username": "your_username", "password": "your_password"}'
```

### 2. Basic Anomaly Detection
```bash
# Detect anomalies in data
curl -X POST "https://api.pynomaly.com/api/v1/detection/detect" \\
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{
    "data": [1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
    "algorithm": "isolation_forest",
    "parameters": {"contamination": 0.1}
  }'
```

### 3. Train Custom Model
```bash
# Train a new model
curl -X POST "https://api.pynomaly.com/api/v1/detection/train" \\
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \\
  -H "Content-Type: application/json" \\
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
"""

    def generate_comprehensive_schema(self) -> dict[str, Any]:
        """Generate comprehensive OpenAPI schema."""
        schema = {
            "openapi": "3.0.3",
            "info": self.app_info,
            "servers": self.servers,
            "tags": self._get_tags(),
            "paths": self._get_paths(),
            "components": {
                "schemas": self._get_schemas(),
                "responses": self._get_responses(),
                "parameters": self._get_parameters(),
                "examples": self._get_examples(),
                "requestBodies": self._get_request_bodies(),
                "headers": self._get_headers(),
                "securitySchemes": self._get_security_schemes(),
            },
            "security": [{"BearerAuth": []}, {"ApiKeyAuth": []}],
            "externalDocs": {
                "description": "Pynomaly Documentation",
                "url": "https://docs.pynomaly.com",
            },
        }

        return schema

    def _get_tags(self) -> list[dict[str, Any]]:
        """Get API tags for organization."""
        return [
            {
                "name": "Authentication",
                "description": "User authentication and authorization",
                "externalDocs": {
                    "description": "Authentication Guide",
                    "url": "https://docs.pynomaly.com/authentication",
                },
            },
            {
                "name": "Anomaly Detection",
                "description": "Core anomaly detection capabilities",
                "externalDocs": {
                    "description": "Detection Guide",
                    "url": "https://docs.pynomaly.com/detection",
                },
            },
            {
                "name": "Model Management",
                "description": "Model lifecycle management and deployment",
                "externalDocs": {
                    "description": "Model Management Guide",
                    "url": "https://docs.pynomaly.com/models",
                },
            },
            {
                "name": "MLOps",
                "description": "MLOps platform features",
                "externalDocs": {
                    "description": "MLOps Guide",
                    "url": "https://docs.pynomaly.com/mlops",
                },
            },
            {
                "name": "Enterprise",
                "description": "Enterprise features and multi-tenancy",
                "externalDocs": {
                    "description": "Enterprise Guide",
                    "url": "https://docs.pynomaly.com/enterprise",
                },
            },
            {
                "name": "Monitoring",
                "description": "System monitoring and health checks",
                "externalDocs": {
                    "description": "Monitoring Guide",
                    "url": "https://docs.pynomaly.com/monitoring",
                },
            },
            {
                "name": "Analytics",
                "description": "Analytics dashboard and insights",
                "externalDocs": {
                    "description": "Analytics Guide",
                    "url": "https://docs.pynomaly.com/analytics",
                },
            },
            {
                "name": "Compliance",
                "description": "Audit logging and compliance reporting",
                "externalDocs": {
                    "description": "Compliance Guide",
                    "url": "https://docs.pynomaly.com/compliance",
                },
            },
        ]

    def _get_paths(self) -> dict[str, Any]:
        """Get comprehensive API paths."""
        return {
            # Authentication endpoints
            "/api/v1/auth/login": {
                "post": {
                    "tags": ["Authentication"],
                    "summary": "User login",
                    "description": "Authenticate user and return JWT token",
                    "operationId": "login",
                    "requestBody": {"$ref": "#/components/requestBodies/LoginRequest"},
                    "responses": {
                        "200": {"$ref": "#/components/responses/LoginResponse"},
                        "401": {"$ref": "#/components/responses/UnauthorizedError"},
                        "422": {"$ref": "#/components/responses/ValidationError"},
                    },
                }
            },
            "/api/v1/auth/refresh": {
                "post": {
                    "tags": ["Authentication"],
                    "summary": "Refresh JWT token",
                    "description": "Refresh an existing JWT token",
                    "operationId": "refresh_token",
                    "security": [{"BearerAuth": []}],
                    "responses": {
                        "200": {"$ref": "#/components/responses/TokenResponse"},
                        "401": {"$ref": "#/components/responses/UnauthorizedError"},
                    },
                }
            },
            "/api/v1/auth/me": {
                "get": {
                    "tags": ["Authentication"],
                    "summary": "Get current user profile",
                    "description": "Get the current authenticated user's profile",
                    "operationId": "get_current_user",
                    "security": [{"BearerAuth": []}],
                    "responses": {
                        "200": {"$ref": "#/components/responses/UserProfile"},
                        "401": {"$ref": "#/components/responses/UnauthorizedError"},
                    },
                }
            },
            # Anomaly Detection endpoints
            "/api/v1/detection/detect": {
                "post": {
                    "tags": ["Anomaly Detection"],
                    "summary": "Detect anomalies in data",
                    "description": "Detect anomalies in the provided dataset using the specified algorithm",
                    "operationId": "detect_anomalies",
                    "security": [{"BearerAuth": []}],
                    "requestBody": {
                        "$ref": "#/components/requestBodies/DetectionRequest"
                    },
                    "responses": {
                        "200": {"$ref": "#/components/responses/DetectionResponse"},
                        "400": {"$ref": "#/components/responses/BadRequestError"},
                        "401": {"$ref": "#/components/responses/UnauthorizedError"},
                        "422": {"$ref": "#/components/responses/ValidationError"},
                    },
                }
            },
            "/api/v1/detection/train": {
                "post": {
                    "tags": ["Anomaly Detection"],
                    "summary": "Train anomaly detection model",
                    "description": "Train a new anomaly detection model with the provided data",
                    "operationId": "train_model",
                    "security": [{"BearerAuth": []}],
                    "requestBody": {
                        "$ref": "#/components/requestBodies/TrainingRequest"
                    },
                    "responses": {
                        "201": {"$ref": "#/components/responses/TrainingResponse"},
                        "400": {"$ref": "#/components/responses/BadRequestError"},
                        "401": {"$ref": "#/components/responses/UnauthorizedError"},
                        "422": {"$ref": "#/components/responses/ValidationError"},
                    },
                }
            },
            "/api/v1/detection/batch": {
                "post": {
                    "tags": ["Anomaly Detection"],
                    "summary": "Batch anomaly detection",
                    "description": "Perform batch anomaly detection with multiple detectors",
                    "operationId": "batch_detect",
                    "security": [{"BearerAuth": []}],
                    "requestBody": {
                        "$ref": "#/components/requestBodies/BatchDetectionRequest"
                    },
                    "responses": {
                        "200": {
                            "$ref": "#/components/responses/BatchDetectionResponse"
                        },
                        "400": {"$ref": "#/components/responses/BadRequestError"},
                        "401": {"$ref": "#/components/responses/UnauthorizedError"},
                    },
                }
            },
            # Model Management endpoints
            "/api/v1/models": {
                "get": {
                    "tags": ["Model Management"],
                    "summary": "List models",
                    "description": "Get a list of all available models",
                    "operationId": "list_models",
                    "security": [{"BearerAuth": []}],
                    "parameters": [
                        {"$ref": "#/components/parameters/PageParameter"},
                        {"$ref": "#/components/parameters/SizeParameter"},
                    ],
                    "responses": {
                        "200": {"$ref": "#/components/responses/ModelListResponse"},
                        "401": {"$ref": "#/components/responses/UnauthorizedError"},
                    },
                }
            },
            "/api/v1/models/{model_id}": {
                "get": {
                    "tags": ["Model Management"],
                    "summary": "Get model details",
                    "description": "Get detailed information about a specific model",
                    "operationId": "get_model",
                    "security": [{"BearerAuth": []}],
                    "parameters": [
                        {"$ref": "#/components/parameters/ModelIdParameter"}
                    ],
                    "responses": {
                        "200": {"$ref": "#/components/responses/ModelDetailsResponse"},
                        "404": {"$ref": "#/components/responses/NotFoundError"},
                        "401": {"$ref": "#/components/responses/UnauthorizedError"},
                    },
                }
            },
            # Health and Monitoring endpoints
            "/api/v1/health": {
                "get": {
                    "tags": ["Monitoring"],
                    "summary": "Health check",
                    "description": "Get system health status",
                    "operationId": "health_check",
                    "responses": {
                        "200": {"$ref": "#/components/responses/HealthResponse"},
                        "503": {
                            "$ref": "#/components/responses/ServiceUnavailableError"
                        },
                    },
                }
            },
            "/api/v1/health/metrics": {
                "get": {
                    "tags": ["Monitoring"],
                    "summary": "System metrics",
                    "description": "Get detailed system metrics",
                    "operationId": "get_metrics",
                    "security": [{"BearerAuth": []}],
                    "responses": {
                        "200": {"$ref": "#/components/responses/MetricsResponse"},
                        "401": {"$ref": "#/components/responses/UnauthorizedError"},
                    },
                }
            },
        }

    def _get_schemas(self) -> dict[str, Any]:
        """Get comprehensive schema definitions."""
        return {
            "LoginRequest": {
                "type": "object",
                "required": ["username", "password"],
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "Username or email address",
                        "example": "user@example.com",
                    },
                    "password": {
                        "type": "string",
                        "format": "password",
                        "description": "User password",
                        "example": "secretpassword123",
                    },
                },
            },
            "TokenResponse": {
                "type": "object",
                "properties": {
                    "access_token": {
                        "type": "string",
                        "description": "JWT access token",
                    },
                    "refresh_token": {
                        "type": "string",
                        "description": "JWT refresh token",
                    },
                    "token_type": {
                        "type": "string",
                        "description": "Token type",
                        "example": "bearer",
                    },
                    "expires_in": {
                        "type": "integer",
                        "description": "Token expiration time in seconds",
                        "example": 3600,
                    },
                },
            },
            "DetectionRequest": {
                "type": "object",
                "required": ["data", "algorithm"],
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Input data for anomaly detection",
                        "example": [1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
                    },
                    "algorithm": {
                        "type": "string",
                        "enum": [
                            "isolation_forest",
                            "one_class_svm",
                            "lstm_autoencoder",
                            "ensemble",
                        ],
                        "description": "Anomaly detection algorithm",
                        "example": "isolation_forest",
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Algorithm-specific parameters",
                        "example": {"contamination": 0.1, "n_estimators": 100},
                    },
                },
            },
            "DetectionResponse": {
                "type": "object",
                "properties": {
                    "anomalies": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Indices of detected anomalies",
                        "example": [3],
                    },
                    "scores": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Anomaly scores for each data point",
                        "example": [0.1, 0.2, 0.15, 0.95, 0.18, 0.12],
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Anomaly detection threshold",
                        "example": 0.5,
                    },
                    "model_id": {
                        "type": "string",
                        "description": "ID of the model used for detection",
                        "example": "isolation_forest_20240101_001",
                    },
                    "processing_time_ms": {
                        "type": "number",
                        "description": "Processing time in milliseconds",
                        "example": 45.6,
                    },
                },
            },
            "TrainingRequest": {
                "type": "object",
                "required": ["training_data", "algorithm"],
                "properties": {
                    "training_data": {
                        "type": "string",
                        "description": "Path to training data file or inline data",
                        "example": "path/to/training/data.csv",
                    },
                    "algorithm": {
                        "type": "string",
                        "enum": [
                            "isolation_forest",
                            "one_class_svm",
                            "lstm_autoencoder",
                            "ensemble",
                        ],
                        "description": "Algorithm to train",
                        "example": "lstm_autoencoder",
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Training parameters",
                        "example": {
                            "epochs": 100,
                            "batch_size": 32,
                            "learning_rate": 0.001,
                        },
                    },
                    "model_name": {
                        "type": "string",
                        "description": "Name for the trained model",
                        "example": "production_model_v1",
                    },
                },
            },
            "TrainingResponse": {
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "ID of the trained model",
                        "example": "lstm_autoencoder_20240101_001",
                    },
                    "model_name": {
                        "type": "string",
                        "description": "Name of the trained model",
                        "example": "production_model_v1",
                    },
                    "training_metrics": {
                        "type": "object",
                        "description": "Training performance metrics",
                        "example": {
                            "loss": 0.0234,
                            "accuracy": 0.967,
                            "precision": 0.923,
                            "recall": 0.889,
                        },
                    },
                    "training_time_ms": {
                        "type": "number",
                        "description": "Training time in milliseconds",
                        "example": 45600.0,
                    },
                },
            },
            "HealthResponse": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["healthy", "unhealthy", "degraded"],
                        "description": "Overall system health status",
                        "example": "healthy",
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Health check timestamp",
                        "example": "2024-01-01T12:00:00Z",
                    },
                    "version": {
                        "type": "string",
                        "description": "Application version",
                        "example": "1.0.0",
                    },
                    "services": {
                        "type": "object",
                        "description": "Individual service health statuses",
                        "example": {
                            "database": "healthy",
                            "cache": "healthy",
                            "model_registry": "healthy",
                        },
                    },
                },
            },
            "ErrorResponse": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Error type URI",
                        "example": "https://pynomaly.com/errors/validation-error",
                    },
                    "title": {
                        "type": "string",
                        "description": "Error title",
                        "example": "Validation Error",
                    },
                    "status": {
                        "type": "integer",
                        "description": "HTTP status code",
                        "example": 422,
                    },
                    "detail": {
                        "type": "string",
                        "description": "Error details",
                        "example": "The request data is invalid",
                    },
                    "instance": {
                        "type": "string",
                        "description": "Request instance that caused the error",
                        "example": "/api/v1/detection/detect",
                    },
                },
            },
        }

    def _get_responses(self) -> dict[str, Any]:
        """Get standard response definitions."""
        return {
            "LoginResponse": {
                "description": "Successful login response",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/TokenResponse"}
                    }
                },
            },
            "TokenResponse": {
                "description": "JWT token response",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/TokenResponse"}
                    }
                },
            },
            "DetectionResponse": {
                "description": "Anomaly detection response",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/DetectionResponse"}
                    }
                },
            },
            "TrainingResponse": {
                "description": "Model training response",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/TrainingResponse"}
                    }
                },
            },
            "HealthResponse": {
                "description": "System health response",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/HealthResponse"}
                    }
                },
            },
            "BadRequestError": {
                "description": "Bad request error",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "example": {
                            "type": "https://pynomaly.com/errors/bad-request",
                            "title": "Bad Request",
                            "status": 400,
                            "detail": "Invalid request parameters",
                        },
                    }
                },
            },
            "UnauthorizedError": {
                "description": "Unauthorized error",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "example": {
                            "type": "https://pynomaly.com/errors/unauthorized",
                            "title": "Unauthorized",
                            "status": 401,
                            "detail": "Authentication required",
                        },
                    }
                },
            },
            "NotFoundError": {
                "description": "Resource not found error",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "example": {
                            "type": "https://pynomaly.com/errors/not-found",
                            "title": "Not Found",
                            "status": 404,
                            "detail": "Resource not found",
                        },
                    }
                },
            },
            "ValidationError": {
                "description": "Validation error",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "example": {
                            "type": "https://pynomaly.com/errors/validation-error",
                            "title": "Validation Error",
                            "status": 422,
                            "detail": "The request data is invalid",
                        },
                    }
                },
            },
            "ServiceUnavailableError": {
                "description": "Service unavailable error",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "example": {
                            "type": "https://pynomaly.com/errors/service-unavailable",
                            "title": "Service Unavailable",
                            "status": 503,
                            "detail": "Service temporarily unavailable",
                        },
                    }
                },
            },
        }

    def _get_parameters(self) -> dict[str, Any]:
        """Get reusable parameter definitions."""
        return {
            "PageParameter": {
                "name": "page",
                "in": "query",
                "description": "Page number for pagination",
                "required": False,
                "schema": {"type": "integer", "minimum": 1, "default": 1},
            },
            "SizeParameter": {
                "name": "size",
                "in": "query",
                "description": "Number of items per page",
                "required": False,
                "schema": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 20,
                },
            },
            "ModelIdParameter": {
                "name": "model_id",
                "in": "path",
                "description": "Unique model identifier",
                "required": True,
                "schema": {"type": "string", "format": "uuid"},
            },
        }

    def _get_examples(self) -> dict[str, Any]:
        """Get comprehensive examples."""
        return {
            "SimpleDetectionRequest": {
                "summary": "Simple anomaly detection",
                "description": "Basic anomaly detection with Isolation Forest",
                "value": {
                    "data": [1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
                    "algorithm": "isolation_forest",
                    "parameters": {"contamination": 0.1, "n_estimators": 100},
                },
            },
            "TimeSeriesDetectionRequest": {
                "summary": "Time series anomaly detection",
                "description": "Anomaly detection for time series data",
                "value": {
                    "data": [1.0, 1.1, 1.2, 1.1, 1.0, 5.0, 1.1, 1.2],
                    "algorithm": "lstm_autoencoder",
                    "parameters": {"sequence_length": 10, "epochs": 50},
                },
            },
            "ModelTrainingRequest": {
                "summary": "Model training example",
                "description": "Training a custom LSTM autoencoder model",
                "value": {
                    "training_data": "s3://pynomaly-data/training/timeseries.csv",
                    "algorithm": "lstm_autoencoder",
                    "parameters": {
                        "epochs": 100,
                        "batch_size": 32,
                        "learning_rate": 0.001,
                        "hidden_units": 64,
                    },
                    "model_name": "production_lstm_v1",
                },
            },
        }

    def _get_request_bodies(self) -> dict[str, Any]:
        """Get request body definitions."""
        return {
            "LoginRequest": {
                "description": "User login credentials",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/LoginRequest"}
                    }
                },
                "required": True,
            },
            "DetectionRequest": {
                "description": "Anomaly detection request",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/DetectionRequest"},
                        "examples": {
                            "simple": {
                                "$ref": "#/components/examples/SimpleDetectionRequest"
                            },
                            "timeseries": {
                                "$ref": "#/components/examples/TimeSeriesDetectionRequest"
                            },
                        },
                    }
                },
                "required": True,
            },
            "TrainingRequest": {
                "description": "Model training request",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/TrainingRequest"},
                        "examples": {
                            "lstm_training": {
                                "$ref": "#/components/examples/ModelTrainingRequest"
                            }
                        },
                    }
                },
                "required": True,
            },
        }

    def _get_headers(self) -> dict[str, Any]:
        """Get header definitions."""
        return {
            "X-Request-ID": {
                "description": "Unique request identifier",
                "schema": {"type": "string", "format": "uuid"},
            },
            "X-Rate-Limit-Remaining": {
                "description": "Number of requests remaining in current window",
                "schema": {"type": "integer"},
            },
            "X-Rate-Limit-Reset": {
                "description": "Time when rate limit resets",
                "schema": {"type": "integer"},
            },
        }

    def _get_security_schemes(self) -> dict[str, Any]:
        """Get security scheme definitions."""
        return {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT token obtained from the authentication endpoint",
            },
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key for service-to-service authentication",
            },
        }

    def generate_documentation(self, output_dir: Path) -> None:
        """Generate comprehensive API documentation."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate OpenAPI schema
        schema = self.generate_comprehensive_schema()

        # Save as JSON
        json_path = output_dir / "openapi.json"
        with open(json_path, "w") as f:
            json.dump(schema, f, indent=2)

        # Save as YAML
        yaml_path = output_dir / "openapi.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(schema, f, default_flow_style=False, sort_keys=False)

        # Generate README
        readme_path = output_dir / "README.md"
        self._generate_readme(readme_path, schema)

        # Generate client examples
        examples_dir = output_dir / "examples"
        self._generate_client_examples(examples_dir)

        print(f"✅ OpenAPI documentation generated in: {output_dir}")
        print(f"📄 OpenAPI JSON: {json_path}")
        print(f"📄 OpenAPI YAML: {yaml_path}")
        print(f"📖 README: {readme_path}")
        print(f"💡 Examples: {examples_dir}")

    def _generate_readme(self, readme_path: Path, schema: dict[str, Any]) -> None:
        """Generate README.md for the API documentation."""
        readme_content = f"""# Pynomaly API Documentation

{schema['info']['description']}

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
- `GET /api/v1/models/{{model_id}}` - Get model details

### Health & Monitoring
- `GET /api/v1/health` - System health check
- `GET /api/v1/health/metrics` - System metrics

## Usage Examples

### Authentication
```bash
curl -X POST "https://api.pynomaly.com/api/v1/auth/login" \\
  -H "Content-Type: application/json" \\
  -d '{{"username": "your_username", "password": "your_password"}}'
```

### Anomaly Detection
```bash
curl -X POST "https://api.pynomaly.com/api/v1/detection/detect" \\
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "data": [1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
    "algorithm": "isolation_forest",
    "parameters": {{"contamination": 0.1}}
  }}'
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

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        with open(readme_path, "w") as f:
            f.write(readme_content)

    def _generate_client_examples(self, examples_dir: Path) -> None:
        """Generate client code examples."""
        examples_dir.mkdir(parents=True, exist_ok=True)

        # Python example
        python_example = '''"""
Pynomaly API Client Example - Python
"""

import requests
import json
from typing import Dict, Any, List

class PynomaliClient:
    """Python client for Pynomaly API."""

    def __init__(self, base_url: str = "https://api.pynomaly.com"):
        self.base_url = base_url
        self.session = requests.Session()
        self.access_token = None

    def login(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate and get JWT token."""
        response = self.session.post(
            f"{self.base_url}/api/v1/auth/login",
            json={"username": username, "password": password}
        )
        response.raise_for_status()

        token_data = response.json()
        self.access_token = token_data["access_token"]
        self.session.headers.update({
            "Authorization": f"Bearer {self.access_token}"
        })

        return token_data

    def detect_anomalies(self, data: List[float], algorithm: str = "isolation_forest",
                        parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Detect anomalies in data."""
        if parameters is None:
            parameters = {"contamination": 0.1}

        response = self.session.post(
            f"{self.base_url}/api/v1/detection/detect",
            json={
                "data": data,
                "algorithm": algorithm,
                "parameters": parameters
            }
        )
        response.raise_for_status()
        return response.json()

    def train_model(self, training_data: str, algorithm: str,
                   parameters: Dict[str, Any] = None, model_name: str = None) -> Dict[str, Any]:
        """Train a new anomaly detection model."""
        payload = {
            "training_data": training_data,
            "algorithm": algorithm
        }

        if parameters:
            payload["parameters"] = parameters
        if model_name:
            payload["model_name"] = model_name

        response = self.session.post(
            f"{self.base_url}/api/v1/detection/train",
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def get_health(self) -> Dict[str, Any]:
        """Get system health status."""
        response = self.session.get(f"{self.base_url}/api/v1/health")
        response.raise_for_status()
        return response.json()


# Example usage
if __name__ == "__main__":
    client = PynomaliClient()

    # Login
    client.login("your_username", "your_password")

    # Detect anomalies
    result = client.detect_anomalies([1.0, 2.0, 3.0, 100.0, 4.0, 5.0])
    print(f"Detected anomalies: {result['anomalies']}")

    # Check health
    health = client.get_health()
    print(f"System status: {health['status']}")
'''

        with open(examples_dir / "python_client.py", "w") as f:
            f.write(python_example)

        # JavaScript example
        js_example = """/**
 * Pynomaly API Client Example - JavaScript
 */

class PynomaliClient {
    constructor(baseUrl = 'https://api.pynomaly.com') {
        this.baseUrl = baseUrl;
        this.accessToken = null;
    }

    async login(username, password) {
        const response = await fetch(`${this.baseUrl}/api/v1/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password })
        });

        if (!response.ok) {
            throw new Error(`Login failed: ${response.statusText}`);
        }

        const tokenData = await response.json();
        this.accessToken = tokenData.access_token;

        return tokenData;
    }

    async detectAnomalies(data, algorithm = 'isolation_forest', parameters = { contamination: 0.1 }) {
        const response = await fetch(`${this.baseUrl}/api/v1/detection/detect`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.accessToken}`
            },
            body: JSON.stringify({
                data,
                algorithm,
                parameters
            })
        });

        if (!response.ok) {
            throw new Error(`Detection failed: ${response.statusText}`);
        }

        return await response.json();
    }

    async trainModel(trainingData, algorithm, parameters = null, modelName = null) {
        const payload = {
            training_data: trainingData,
            algorithm
        };

        if (parameters) payload.parameters = parameters;
        if (modelName) payload.model_name = modelName;

        const response = await fetch(`${this.baseUrl}/api/v1/detection/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.accessToken}`
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`Training failed: ${response.statusText}`);
        }

        return await response.json();
    }

    async getHealth() {
        const response = await fetch(`${this.baseUrl}/api/v1/health`);

        if (!response.ok) {
            throw new Error(`Health check failed: ${response.statusText}`);
        }

        return await response.json();
    }
}

// Example usage
(async () => {
    const client = new PynomaliClient();

    try {
        // Login
        await client.login('your_username', 'your_password');

        // Detect anomalies
        const result = await client.detectAnomalies([1.0, 2.0, 3.0, 100.0, 4.0, 5.0]);
        console.log('Detected anomalies:', result.anomalies);

        // Check health
        const health = await client.getHealth();
        console.log('System status:', health.status);

    } catch (error) {
        console.error('Error:', error.message);
    }
})();
"""

        with open(examples_dir / "javascript_client.js", "w") as f:
            f.write(js_example)

        # cURL examples
        curl_example = """#!/bin/bash
# Pynomaly API Client Example - cURL

# Base URL
BASE_URL="https://api.pynomaly.com"

# Login and get JWT token
echo "Logging in..."
LOGIN_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/auth/login" \\
  -H "Content-Type: application/json" \\
  -d '{"username": "your_username", "password": "your_password"}')

# Extract access token
ACCESS_TOKEN=$(echo $LOGIN_RESPONSE | jq -r '.access_token')

if [ "$ACCESS_TOKEN" = "null" ]; then
    echo "Login failed"
    exit 1
fi

echo "Login successful"

# Detect anomalies
echo "Detecting anomalies..."
DETECTION_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/detection/detect" \\
  -H "Authorization: Bearer $ACCESS_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{
    "data": [1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
    "algorithm": "isolation_forest",
    "parameters": {"contamination": 0.1}
  }')

echo "Detection result:"
echo $DETECTION_RESPONSE | jq '.'

# Train model
echo "Training model..."
TRAINING_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/detection/train" \\
  -H "Authorization: Bearer $ACCESS_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{
    "training_data": "s3://pynomaly-data/training/sample.csv",
    "algorithm": "lstm_autoencoder",
    "parameters": {"epochs": 50, "batch_size": 32},
    "model_name": "test_model"
  }')

echo "Training result:"
echo $TRAINING_RESPONSE | jq '.'

# Health check
echo "Checking health..."
HEALTH_RESPONSE=$(curl -s -X GET "$BASE_URL/api/v1/health")

echo "Health status:"
echo $HEALTH_RESPONSE | jq '.'
"""

        with open(examples_dir / "curl_examples.sh", "w") as f:
            f.write(curl_example)

        # Make shell script executable
        (examples_dir / "curl_examples.sh").chmod(0o755)


def main():
    """Main function to generate OpenAPI documentation."""
    generator = OpenAPIDocumentationGenerator()
    output_dir = Path("docs/api")

    print("🚀 Generating comprehensive OpenAPI documentation...")
    generator.generate_documentation(output_dir)
    print("✅ Documentation generation complete!")


if __name__ == "__main__":
    main()
