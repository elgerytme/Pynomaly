#!/usr/bin/env python3
"""
Comprehensive API Documentation Configuration for Software.
This module configures OpenAPI/Swagger documentation for all API endpoints.
"""

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def get_custom_openapi_schema(app: FastAPI) -> dict[str, Any]:
    """Generate custom OpenAPI schema for Software APIs."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Software API",
        version="1.0.0",
        description="""
        # Software - Enterprise Anomaly Processing Platform

        A comprehensive, production-ready anomaly processing system with enterprise-grade features.

        ## Features

        ### Core Processing
        - **Anomaly Processing**: Detect anomalies in time series, tabular, and streaming data
        - **Multiple Algorithms**: Support for Isolation Forest, One-Class SVM, LSTM Autoencoders, and custom models
        - **Ensemble Methods**: Combine multiple processing algorithms for improved accuracy
        - **Real-time Processing**: Stream processing for continuous anomaly processing

        ### MLOps Platform
        - **Processor Registry**: Centralized processor management with versioning and metadata
        - **Experiment Tracking**: Track experiments, parameters, and measurements
        - **Processor Deployment**: Deploy models to development, staging, and production environments
        - **Automated Retraining**: Automatic processor retraining based on data drift and performance degradation

        ### Enterprise Features
        - **Multi-tenancy**: Complete tenant isolation with role-based access control
        - **Audit Logging**: Comprehensive audit trails with compliance support (GDPR, HIPAA, SOX)
        - **Security**: JWT authentication, data encryption, and tamper processing
        - **Analytics Dashboard**: Real-time insights and business measurements

        ### Monitoring & Observability
        - **Health Monitoring**: System health checks and performance measurements
        - **Alerting**: Real-time alerts for anomalies and system issues
        - **Compliance Reporting**: Generate compliance reports for regulatory requirements
        - **Performance Tracking**: Track processor performance and system measurements

        ## Authentication

        Most endpoints require authentication using JWT tokens. Include the token in the Authorization header:

        ```
        Authorization: Bearer <your-jwt-token>
        ```

        ## Rate Limiting

        API calls are rate-limited to ensure fair usage and system stability:
        - **Standard users**: 1000 requests per hour
        - **Enterprise users**: 10000 requests per hour
        - **Internal services**: Unlimited

        ## Error Handling

        The API uses standard HTTP status codes and returns detailed error messages:
        - **200**: Success
        - **201**: Created
        - **400**: Bad Request
        - **401**: Unauthorized
        - **403**: Forbidden
        - **404**: Not Found
        - **422**: Validation Error
        - **500**: Internal Server Error
        """,
        routes=app.routes,
    )

    # Add custom metadata
    openapi_schema["info"]["contact"] = {
        "name": "Software Support",
        "url": "https://example.com/support",
        "email": "support@example.com"
    }

    openapi_schema["info"]["license"] = {
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }

    # Add servers
    openapi_schema["servers"] = [
        {
            "url": "https://api.example.com",
            "description": "Production server"
        },
        {
            "url": "https://staging-api.example.com",
            "description": "Staging server"
        },
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        }
    ]

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token obtained from the authentication endpoint"
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for service-to-service authentication"
        }
    }

    # Add tags for better organization
    openapi_schema["tags"] = [
        {
            "name": "Authentication",
            "description": "User authentication and authorization"
        },
        {
            "name": "Anomaly Processing",
            "description": "Core anomaly processing capabilities"
        },
        {
            "name": "Processor Management",
            "description": "Processor lifecycle management and deployment"
        },
        {
            "name": "MLOps",
            "description": "MLOps platform features"
        },
        {
            "name": "Enterprise",
            "description": "Enterprise features and multi-tenancy"
        },
        {
            "name": "Monitoring",
            "description": "System monitoring and health checks"
        },
        {
            "name": "Analytics",
            "description": "Analytics dashboard and insights"
        },
        {
            "name": "Compliance",
            "description": "Audit logging and compliance reporting"
        }
    ]

    # Add example responses
    openapi_schema["components"]["examples"] = {
        "AnomalyDetectionRequest": {
            "summary": "Simple anomaly processing request",
            "value": {
                "data": [1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
                "algorithm": "isolation_forest",
                "parameters": {
                    "contamination": 0.1,
                    "n_estimators": 100
                }
            }
        },
        "AnomalyDetectionResponse": {
            "summary": "Anomaly processing response",
            "value": {
                "anomalies": [3],
                "scores": [0.1, 0.2, 0.15, 0.95, 0.18, 0.12],
                "threshold": 0.5,
                "processor_id": "isolation_forest_20240101_001",
                "processing_time_ms": 45.6
            }
        },
        "ErrorResponse": {
            "summary": "Error response",
            "value": {
                "error": "ValidationError",
                "message": "Invalid input data format",
                "details": {
                    "field": "data",
                    "issue": "Expected array of numbers"
                }
            }
        }
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


def configure_api_docs(app: FastAPI) -> None:
    """Configure API documentation for the FastAPI application."""

    # Set custom OpenAPI schema
    app.openapi = lambda: get_custom_openapi_schema(app)

    # Configure Swagger UI
    app.swagger_ui_parameters = {
        "defaultModelsExpandDepth": -1,
        "docExpansion": "none",
        "filter": True,
        "showExtensions": True,
        "showCommonExtensions": True,
        "tryItOutEnabled": True,
        "displayRequestDuration": True,
        "persistAuthorization": True,
        "layout": "StandaloneLayout"
    }


# Common response models for documentation
COMMON_RESPONSES = {
    400: {
        "description": "Bad Request",
        "content": {
            "application/json": {
                "example": {
                    "error": "BadRequest",
                    "message": "Invalid request parameters"
                }
            }
        }
    },
    401: {
        "description": "Unauthorized",
        "content": {
            "application/json": {
                "example": {
                    "error": "Unauthorized",
                    "message": "Authentication required"
                }
            }
        }
    },
    403: {
        "description": "Forbidden",
        "content": {
            "application/json": {
                "example": {
                    "error": "Forbidden",
                    "message": "Insufficient permissions"
                }
            }
        }
    },
    404: {
        "description": "Not Found",
        "content": {
            "application/json": {
                "example": {
                    "error": "NotFound",
                    "message": "Resource not found"
                }
            }
        }
    },
    422: {
        "description": "Validation Error",
        "content": {
            "application/json": {
                "example": {
                    "error": "ValidationError",
                    "message": "Request validation failed",
                    "details": [
                        {
                            "field": "data",
                            "message": "Field required"
                        }
                    ]
                }
            }
        }
    },
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {
                    "error": "InternalServerError",
                    "message": "An unexpected error occurred"
                }
            }
        }
    }
}


# API endpoint documentation metadata
ENDPOINT_METADATA = {
    "detect_anomalies": {
        "summary": "Detect anomalies in data",
        "description": """
        Detect anomalies in the provided data_collection using the specified algorithm.

        **Supported Algorithms:**
        - `isolation_forest`: Isolation Forest algorithm
        - `one_class_svm`: One-Class Support Vector Machine
        - `lstm_autoencoder`: LSTM Autoencoder for time series
        - `ensemble`: Ensemble of multiple algorithms

        **Parameters:**
        - Algorithm-specific parameters can be provided in the `parameters` field
        - Common parameters include `contamination`, `n_estimators`, etc.

        **Response:**
        - Returns anomaly scores, detected anomalies, and processor metadata
        """,
        "tags": ["Anomaly Processing"]
    },
    "train_processor": {
        "summary": "Train a new anomaly processing processor",
        "description": """
        Train a new anomaly processing processor with the provided training data.

        **Training Process:**
        1. Data validation and preprocessing
        2. Processor training with specified parameters
        3. Processor evaluation and measurements calculation
        4. Processor registration in the processor registry

        **Response:**
        - Returns trained processor ID, performance measurements, and training metadata
        """,
        "tags": ["Processor Management"]
    },
    "get_processor_info": {
        "summary": "Get processor information",
        "description": """
        Retrieve detailed information about a specific processor.

        **Information Included:**
        - Processor metadata (name, version, type, author)
        - Training parameters and configuration
        - Performance measurements and evaluation results
        - Deployment history and current status
        """,
        "tags": ["Processor Management"]
    },
    "deploy_processor": {
        "summary": "Deploy processor to environment",
        "description": """
        Deploy a trained processor to a specific environment.

        **Deployment Environments:**
        - `development`: For testing and development
        - `staging`: For pre-production testing
        - `production`: For live production workloads

        **Deployment Process:**
        1. Processor validation and compatibility checks
        2. Environment preparation and resource allocation
        3. Processor deployment and health checks
        4. Monitoring and alerting setup
        """,
        "tags": ["MLOps"]
    },
    "health_check": {
        "summary": "System health check",
        "description": """
        Get the current health status of the Software system.

        **Health Checks:**
        - Database connectivity
        - Processor registry status
        - Cache system status
        - External service dependencies
        - Resource utilization measurements
        """,
        "tags": ["Monitoring"]
    },
    "enterprise_dashboard": {
        "summary": "Enterprise dashboard data",
        "description": """
        Get comprehensive dashboard data for enterprise users.

        **Dashboard Measurements:**
        - Tenant resource usage and limits
        - Recent audit events and security alerts
        - Processor performance and deployment status
        - User activity and system measurements
        """,
        "tags": ["Enterprise", "Analytics"]
    },
    "compliance_report": {
        "summary": "Generate compliance report",
        "description": """
        Generate a comprehensive compliance report for regulatory requirements.

        **Supported Compliance Levels:**
        - GDPR: General Data Protection Regulation
        - HIPAA: Health Insurance Portability and Accountability Act
        - SOX: Sarbanes-Oxley Act
        - PCI DSS: Payment Card Industry Data Security Standard
        - ISO 27001: Information Security Management

        **Report Contents:**
        - Audit trail summary and detailed events
        - User activity and access patterns
        - Resource usage and data handling
        - Security incidents and violations
        """,
        "tags": ["Compliance", "Enterprise"]
    }
}


# Schema examples for different data types
SCHEMA_EXAMPLES = {
    "TimeSeriesData": {
        "description": "Time series data format",
        "example": {
            "timestamps": ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"],
            "values": [1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
            "features": {
                "sensor_id": "sensor_001",
                "location": "facility_A"
            }
        }
    },
    "TabularData": {
        "description": "Tabular data format",
        "example": {
            "columns": ["feature1", "feature2", "feature3"],
            "data": [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0]
            ]
        }
    },
    "StreamingData": {
        "description": "Streaming data format",
        "example": {
            "stream_id": "stream_001",
            "batch_size": 100,
            "data_points": [
                {"timestamp": "2024-01-01T00:00:00Z", "value": 1.0},
                {"timestamp": "2024-01-01T00:01:00Z", "value": 2.0}
            ]
        }
    }
}
