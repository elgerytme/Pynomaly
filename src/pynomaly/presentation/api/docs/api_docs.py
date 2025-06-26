"""API documentation routes and utilities."""

from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from pynomaly.presentation.api.docs.openapi_config import (
    get_custom_redoc_html,
    get_custom_swagger_ui_html,
)

router = APIRouter(prefix="/docs", include_in_schema=False)


@router.get("/swagger", response_class=HTMLResponse)
async def custom_swagger_ui_html(request: Request) -> HTMLResponse:
    """Custom Swagger UI with Pynomaly branding."""
    openapi_url = str(request.url_for("openapi"))

    return HTMLResponse(
        get_custom_swagger_ui_html(
            openapi_url=openapi_url,
            title="Pynomaly API Documentation",
        )
    )


@router.get("/redoc", response_class=HTMLResponse)
async def custom_redoc_html(request: Request) -> HTMLResponse:
    """Custom ReDoc documentation with Pynomaly branding."""
    openapi_url = str(request.url_for("openapi"))

    return HTMLResponse(
        get_custom_redoc_html(
            openapi_url=openapi_url,
            title="Pynomaly API Reference",
        )
    )


@router.get("/postman", response_class=JSONResponse)
async def generate_postman_collection(request: Request) -> JSONResponse:
    """Generate Postman collection from OpenAPI spec."""
    # Get the OpenAPI schema
    openapi_url = str(request.url_for("openapi"))
    base_url = str(request.base_url).rstrip("/")

    # Generate basic Postman collection structure
    collection = {
        "info": {
            "name": "Pynomaly API",
            "description": "Comprehensive anomaly detection API collection",
            "version": "1.0.0",
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
        },
        "auth": {
            "type": "bearer",
            "bearer": [
                {
                    "key": "token",
                    "value": "{{jwt_token}}",
                    "type": "string"
                }
            ]
        },
        "variable": [
            {
                "key": "base_url",
                "value": base_url,
                "type": "string"
            },
            {
                "key": "jwt_token",
                "value": "",
                "type": "string"
            }
        ],
        "item": _generate_postman_items()
    }

    return JSONResponse(
        content=collection,
        headers={
            "Content-Disposition": "attachment; filename=pynomaly-api.postman_collection.json"
        }
    )


@router.get("/openapi-summary")
async def openapi_summary(request: Request) -> dict:
    """Get a summary of the OpenAPI specification."""
    # This would analyze the OpenAPI spec and provide a summary
    # For now, return a basic summary
    return {
        "api_name": "Pynomaly API",
        "version": "1.0.0",
        "total_endpoints": 65,  # Updated count with new endpoints
        "endpoint_categories": [
            "Authentication",
            "Health",
            "Datasets",
            "Detectors", 
            "Detection",
            "AutoML",
            "Enhanced AutoML",
            "Ensemble",
            "Explainability",
            "Experiments",
            "Export",
            "Performance",
            "Model Lineage",
            "Streaming",
            "Events",
            "Administration",
            "Autonomous"
        ],
        "new_features": [
            "Advanced AutoML with meta-learning",
            "Multi-objective optimization",
            "Ensemble learning capabilities", 
            "Model explainability and interpretation",
            "Real-time streaming detection",
            "Comprehensive model lineage tracking"
        ],
        "authentication_methods": ["JWT Bearer Token", "API Key", "OAuth2"],
        "response_formats": ["JSON", "CSV", "Excel", "Parquet"],
        "rate_limits": {
            "default": "100 requests/minute",
            "training": "10 requests/minute",
            "automl": "5 requests/minute", 
            "export": "20 requests/minute",
            "streaming": "1000 requests/minute"
        },
        "advanced_capabilities": {
            "automl": {
                "algorithms_supported": 8,
                "optimization_strategies": ["bayesian", "hyperband", "bohb", "multi_objective"],
                "meta_learning": True,
                "ensemble_methods": ["voting", "stacking", "blending"]
            },
            "explainability": {
                "methods": ["shap", "lime", "feature_importance", "partial_dependence"],
                "global_explanations": True,
                "local_explanations": True
            },
            "monitoring": {
                "health_checks": True,
                "performance_metrics": True,
                "drift_detection": True,
                "alerts": True
            }
        }
    }


@router.get("/sdk-info")
async def sdk_information() -> dict:
    """Get information about available SDKs and code generation."""
    return {
        "available_sdks": [
            {
                "language": "Python",
                "status": "official",
                "installation": "pip install pynomaly-client",
                "documentation": "https://pynomaly.readthedocs.io/python-sdk",
                "repository": "https://github.com/pynomaly/pynomaly-python-client"
            },
            {
                "language": "JavaScript/TypeScript",
                "status": "community",
                "installation": "npm install pynomaly-js",
                "documentation": "https://pynomaly.readthedocs.io/js-sdk",
                "repository": "https://github.com/pynomaly/pynomaly-js-client"
            },
            {
                "language": "R",
                "status": "planned",
                "installation": "install.packages('pynomaly')",
                "documentation": "https://pynomaly.readthedocs.io/r-sdk",
                "repository": "https://github.com/pynomaly/pynomaly-r-client"
            }
        ],
        "code_generation": {
            "openapi_generator": {
                "supported_languages": [
                    "python", "javascript", "typescript", "java", "csharp",
                    "php", "ruby", "go", "rust", "kotlin", "swift"
                ],
                "command": "openapi-generator-cli generate -i {openapi_url} -g {language} -o ./generated-client",
                "documentation": "https://openapi-generator.tech/"
            },
            "swagger_codegen": {
                "supported_languages": [
                    "python", "javascript", "java", "csharp", "php", "ruby"
                ],
                "command": "swagger-codegen generate -i {openapi_url} -l {language} -o ./generated-client",
                "documentation": "https://swagger.io/tools/swagger-codegen/"
            }
        }
    }


def _generate_postman_items() -> list[dict]:
    """Generate Postman collection items for major API endpoints."""
    return [
        {
            "name": "Authentication",
            "item": [
                {
                    "name": "Login",
                    "request": {
                        "method": "POST",
                        "header": [
                            {
                                "key": "Content-Type",
                                "value": "application/json"
                            }
                        ],
                        "body": {
                            "mode": "raw",
                            "raw": json.dumps({
                                "username": "admin",
                                "password": "your_password"
                            })
                        },
                        "url": {
                            "raw": "{{base_url}}/auth/login",
                            "host": ["{{base_url}}"],
                            "path": ["auth", "login"]
                        }
                    }
                }
            ]
        },
        {
            "name": "Health Checks",
            "item": [
                {
                    "name": "Comprehensive Health Check",
                    "request": {
                        "method": "GET",
                        "header": [],
                        "url": {
                            "raw": "{{base_url}}/health/",
                            "host": ["{{base_url}}"],
                            "path": ["health", ""]
                        }
                    }
                },
                {
                    "name": "System Metrics",
                    "request": {
                        "method": "GET",
                        "header": [],
                        "url": {
                            "raw": "{{base_url}}/health/metrics",
                            "host": ["{{base_url}}"],
                            "path": ["health", "metrics"]
                        }
                    }
                }
            ]
        },
        {
            "name": "Datasets",
            "item": [
                {
                    "name": "List Datasets",
                    "request": {
                        "method": "GET",
                        "header": [
                            {
                                "key": "Authorization",
                                "value": "Bearer {{jwt_token}}"
                            }
                        ],
                        "url": {
                            "raw": "{{base_url}}/datasets/",
                            "host": ["{{base_url}}"],
                            "path": ["datasets", ""]
                        }
                    }
                },
                {
                    "name": "Upload Dataset",
                    "request": {
                        "method": "POST",
                        "header": [
                            {
                                "key": "Authorization",
                                "value": "Bearer {{jwt_token}}"
                            }
                        ],
                        "body": {
                            "mode": "formdata",
                            "formdata": [
                                {
                                    "key": "file",
                                    "type": "file",
                                    "src": []
                                },
                                {
                                    "key": "name",
                                    "value": "Sample Dataset",
                                    "type": "text"
                                }
                            ]
                        },
                        "url": {
                            "raw": "{{base_url}}/datasets/upload",
                            "host": ["{{base_url}}"],
                            "path": ["datasets", "upload"]
                        }
                    }
                }
            ]
        },
        {
            "name": "Detectors",
            "item": [
                {
                    "name": "Create Detector",
                    "request": {
                        "method": "POST",
                        "header": [
                            {
                                "key": "Authorization",
                                "value": "Bearer {{jwt_token}}"
                            },
                            {
                                "key": "Content-Type",
                                "value": "application/json"
                            }
                        ],
                        "body": {
                            "mode": "raw",
                            "raw": json.dumps({
                                "name": "Isolation Forest Detector",
                                "algorithm": "IsolationForest",
                                "parameters": {
                                    "n_estimators": 100,
                                    "contamination": 0.1
                                }
                            })
                        },
                        "url": {
                            "raw": "{{base_url}}/detectors/",
                            "host": ["{{base_url}}"],
                            "path": ["detectors", ""]
                        }
                    }
                }
            ]
        },
        {
            "name": "Detection",
            "item": [
                {
                    "name": "Train Detector",
                    "request": {
                        "method": "POST",
                        "header": [
                            {
                                "key": "Authorization",
                                "value": "Bearer {{jwt_token}}"
                            },
                            {
                                "key": "Content-Type",
                                "value": "application/json"
                            }
                        ],
                        "body": {
                            "mode": "raw",
                            "raw": json.dumps({
                                "detector_id": "detector_123",
                                "dataset_id": "dataset_456"
                            })
                        },
                        "url": {
                            "raw": "{{base_url}}/detection/train",
                            "host": ["{{base_url}}"],
                            "path": ["detection", "train"]
                        }
                    }
                },
                {
                    "name": "Detect Anomalies",
                    "request": {
                        "method": "POST",
                        "header": [
                            {
                                "key": "Authorization",
                                "value": "Bearer {{jwt_token}}"
                            },
                            {
                                "key": "Content-Type",
                                "value": "application/json"
                            }
                        ],
                        "body": {
                            "mode": "raw",
                            "raw": json.dumps({
                                "detector_id": "detector_123",
                                "data": [
                                    {"feature1": 1.5, "feature2": 2.3},
                                    {"feature1": 5.7, "feature2": 1.2}
                                ]
                            })
                        },
                        "url": {
                            "raw": "{{base_url}}/detection/predict",
                            "host": ["{{base_url}}"],
                            "path": ["detection", "predict"]
                        }
                    }
                }
            ]
        }
    ]
