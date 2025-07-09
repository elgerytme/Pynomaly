#!/usr/bin/env python3
"""
Pynomaly API Documentation Generator

This script generates comprehensive API documentation including:
- OpenAPI 3.0 specification (JSON and YAML formats)
- Interactive Swagger UI documentation
- Code examples for Python, JavaScript, and cURL
- Postman collection for API testing
- Comprehensive endpoint documentation
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DOCS_DIR = PROJECT_DIR / "docs" / "api"
OUTPUT_DIR = DOCS_DIR / "generated"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Colors for output
class Colors:
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[0;31m"
    BLUE = "\033[0;34m"
    NC = "\033[0m"  # No Color


def log(message: str, color: str = Colors.GREEN):
    """Log a message with color."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{color}[{timestamp}] {message}{Colors.NC}")


def error(message: str):
    """Log an error message."""
    log(f"ERROR: {message}", Colors.RED)


def warning(message: str):
    """Log a warning message."""
    log(f"WARNING: {message}", Colors.YELLOW)


def info(message: str):
    """Log an info message."""
    log(f"INFO: {message}", Colors.BLUE)


def generate_openapi_spec() -> dict[str, Any]:
    """Generate OpenAPI 3.0 specification."""
    log("Generating OpenAPI 3.0 specification...")

    spec = {
        "openapi": "3.0.3",
        "info": {
            "title": "Pynomaly API",
            "description": (
                "State-of-the-art anomaly detection API with comprehensive "
                "machine learning capabilities"
            ),
            "version": "1.0.0",
            "contact": {
                "name": "Pynomaly Support",
                "email": "support@pynomaly.com",
                "url": "https://docs.pynomaly.com",
            },
            "license": {"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
        },
        "servers": [
            {"url": "https://api.pynomaly.com", "description": "Production server"},
            {
                "url": "https://staging-api.pynomaly.com",
                "description": "Staging server",
            },
            {"url": "http://localhost:8000", "description": "Development server"},
        ],
        "security": [{"ApiKeyAuth": []}],
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health check endpoint",
                    "description": "Basic health check to verify API availability",
                    "operationId": "healthCheck",
                    "tags": ["Health"],
                    "responses": {
                        "200": {
                            "description": "API is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/HealthResponse"
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/health/detailed": {
                "get": {
                    "summary": "Detailed health check",
                    "description": "Comprehensive health check with system metrics",
                    "operationId": "detailedHealthCheck",
                    "tags": ["Health"],
                    "responses": {
                        "200": {
                            "description": "Detailed health information",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": (
                                            "#/components/schemas/"
                                            "DetailedHealthResponse"
                                        )
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/api/v1/detectors": {
                "get": {
                    "summary": "List detectors",
                    "description": "Retrieve a list of all available anomaly detectors",
                    "operationId": "listDetectors",
                    "tags": ["Detectors"],
                    "parameters": [
                        {
                            "name": "limit",
                            "in": "query",
                            "description": "Number of results per page",
                            "schema": {
                                "type": "integer",
                                "default": 20,
                                "minimum": 1,
                                "maximum": 100,
                            },
                        },
                        {
                            "name": "offset",
                            "in": "query",
                            "description": "Results offset",
                            "schema": {"type": "integer", "default": 0, "minimum": 0},
                        },
                        {
                            "name": "algorithm",
                            "in": "query",
                            "description": "Filter by algorithm type",
                            "schema": {
                                "type": "string",
                                "enum": [
                                    "IsolationForest",
                                    "LocalOutlierFactor",
                                    "OneClassSVM",
                                    "EllipticEnvelope",
                                    "DBSCAN",
                                ],
                            },
                        },
                    ],
                    "responses": {
                        "200": {
                            "description": "List of detectors",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": (
                                            "#/components/schemas/DetectorListResponse"
                                        )
                                    }
                                }
                            },
                        }
                    },
                },
                "post": {
                    "summary": "Create detector",
                    "description": "Create a new anomaly detector",
                    "operationId": "createDetector",
                    "tags": ["Detectors"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/CreateDetectorRequest"
                                }
                            }
                        },
                    },
                    "responses": {
                        "201": {
                            "description": "Detector created successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/DetectorResponse"
                                    }
                                }
                            },
                        },
                        "400": {
                            "description": "Invalid request",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    }
                                }
                            },
                        },
                    },
                },
            },
            "/api/v1/detectors/{detector_id}": {
                "get": {
                    "summary": "Get detector details",
                    "description": (
                        "Retrieve detailed information about a specific detector"
                    ),
                    "operationId": "getDetector",
                    "tags": ["Detectors"],
                    "parameters": [
                        {
                            "name": "detector_id",
                            "in": "path",
                            "required": True,
                            "description": "Detector ID",
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Detector details",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/DetectorResponse"
                                    }
                                }
                            },
                        },
                        "404": {
                            "description": "Detector not found",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    }
                                }
                            },
                        },
                    },
                },
                "put": {
                    "summary": "Update detector",
                    "description": "Update detector configuration",
                    "operationId": "updateDetector",
                    "tags": ["Detectors"],
                    "parameters": [
                        {
                            "name": "detector_id",
                            "in": "path",
                            "required": True,
                            "description": "Detector ID",
                            "schema": {"type": "string"},
                        }
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/UpdateDetectorRequest"
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Detector updated successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/DetectorResponse"
                                    }
                                }
                            },
                        }
                    },
                },
                "delete": {
                    "summary": "Delete detector",
                    "description": "Delete a detector",
                    "operationId": "deleteDetector",
                    "tags": ["Detectors"],
                    "parameters": [
                        {
                            "name": "detector_id",
                            "in": "path",
                            "required": True,
                            "description": "Detector ID",
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Detector deleted successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/SuccessResponse"
                                    }
                                }
                            },
                        }
                    },
                },
            },
            "/api/v1/detectors/{detector_id}/detect": {
                "post": {
                    "summary": "Run anomaly detection",
                    "description": "Perform anomaly detection on new data",
                    "operationId": "detectAnomalies",
                    "tags": ["Detection"],
                    "parameters": [
                        {
                            "name": "detector_id",
                            "in": "path",
                            "required": True,
                            "description": "Detector ID",
                            "schema": {"type": "string"},
                        }
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/DetectionRequest"
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Detection results",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/DetectionResponse"
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/api/v1/automl/optimize": {
                "post": {
                    "summary": "Start AutoML optimization",
                    "description": (
                        "Begin automated machine learning optimization process"
                    ),
                    "operationId": "startAutoMLOptimization",
                    "tags": ["AutoML"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/AutoMLRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "AutoML optimization started",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/AutoMLResponse"
                                    }
                                }
                            },
                        }
                    },
                }
            },
        },
        "components": {
            "securitySchemes": {
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                    "description": "API key for authentication",
                }
            },
            "schemas": {
                "HealthResponse": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean", "example": True},
                        "data": {
                            "type": "object",
                            "properties": {
                                "status": {"type": "string", "example": "healthy"},
                                "timestamp": {"type": "string", "format": "date-time"},
                            },
                        },
                    },
                },
                "DetailedHealthResponse": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "data": {
                            "type": "object",
                            "properties": {
                                "status": {"type": "string"},
                                "components": {
                                    "type": "object",
                                    "properties": {
                                        "database": {"type": "string"},
                                        "redis": {"type": "string"},
                                        "workers": {"type": "string"},
                                    },
                                },
                                "metrics": {
                                    "type": "object",
                                    "properties": {
                                        "cpu_usage": {"type": "number"},
                                        "memory_usage": {"type": "number"},
                                        "disk_usage": {"type": "number"},
                                    },
                                },
                            },
                        },
                    },
                },
                "DetectorListResponse": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "data": {
                            "type": "object",
                            "properties": {
                                "detectors": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/Detector"},
                                },
                                "pagination": {
                                    "$ref": "#/components/schemas/Pagination"
                                },
                            },
                        },
                    },
                },
                "Detector": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "example": "detector_abc123"},
                        "name": {"type": "string", "example": "Production Detector"},
                        "algorithm": {
                            "type": "string",
                            "enum": [
                                "IsolationForest",
                                "LocalOutlierFactor",
                                "OneClassSVM",
                                "EllipticEnvelope",
                                "DBSCAN",
                            ],
                        },
                        "status": {
                            "type": "string",
                            "enum": ["active", "training", "inactive", "error"],
                        },
                        "created_at": {"type": "string", "format": "date-time"},
                        "updated_at": {"type": "string", "format": "date-time"},
                        "performance": {
                            "type": "object",
                            "properties": {
                                "accuracy": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "precision": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "recall": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "f1_score": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                            },
                        },
                    },
                },
                "CreateDetectorRequest": {
                    "type": "object",
                    "required": ["name", "algorithm"],
                    "properties": {
                        "name": {"type": "string", "example": "My Detector"},
                        "algorithm": {
                            "type": "string",
                            "enum": [
                                "IsolationForest",
                                "LocalOutlierFactor",
                                "OneClassSVM",
                                "EllipticEnvelope",
                                "DBSCAN",
                            ],
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Algorithm-specific parameters",
                        },
                        "description": {
                            "type": "string",
                            "example": "Production anomaly detector",
                        },
                    },
                },
                "UpdateDetectorRequest": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "parameters": {"type": "object"},
                        "description": {"type": "string"},
                    },
                },
                "DetectorResponse": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "data": {"$ref": "#/components/schemas/Detector"},
                    },
                },
                "DetectionRequest": {
                    "type": "object",
                    "required": ["data"],
                    "properties": {
                        "data": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                            "example": [[1.2, 3.4, 5.6], [2.3, 4.5, 6.7]],
                        },
                        "feature_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "example": ["feature1", "feature2", "feature3"],
                        },
                        "return_explanations": {"type": "boolean", "default": False},
                    },
                },
                "DetectionResponse": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "data": {
                            "type": "object",
                            "properties": {
                                "predictions": {
                                    "type": "array",
                                    "items": {
                                        "$ref": "#/components/schemas/Prediction"
                                    },
                                },
                                "summary": {
                                    "$ref": "#/components/schemas/DetectionSummary"
                                },
                            },
                        },
                    },
                },
                "Prediction": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "is_anomaly": {"type": "boolean"},
                        "anomaly_score": {"type": "number"},
                        "confidence": {"type": "number"},
                    },
                },
                "DetectionSummary": {
                    "type": "object",
                    "properties": {
                        "total_samples": {"type": "integer"},
                        "anomalies_detected": {"type": "integer"},
                        "anomaly_rate": {"type": "number"},
                        "average_confidence": {"type": "number"},
                    },
                },
                "AutoMLRequest": {
                    "type": "object",
                    "required": ["dataset_id"],
                    "properties": {
                        "dataset_id": {"type": "string"},
                        "algorithms": {"type": "array", "items": {"type": "string"}},
                        "objectives": {"type": "array", "items": {"type": "string"}},
                        "constraints": {
                            "type": "object",
                            "properties": {
                                "max_time": {"type": "integer"},
                                "max_trials": {"type": "integer"},
                                "max_memory": {"type": "integer"},
                            },
                        },
                    },
                },
                "AutoMLResponse": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "data": {
                            "type": "object",
                            "properties": {
                                "optimization_id": {"type": "string"},
                                "status": {"type": "string"},
                                "estimated_time": {"type": "integer"},
                            },
                        },
                    },
                },
                "Pagination": {
                    "type": "object",
                    "properties": {
                        "total": {"type": "integer"},
                        "limit": {"type": "integer"},
                        "offset": {"type": "integer"},
                        "has_next": {"type": "boolean"},
                        "has_prev": {"type": "boolean"},
                    },
                },
                "SuccessResponse": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "data": {
                            "type": "object",
                            "properties": {"message": {"type": "string"}},
                        },
                    },
                },
                "ErrorResponse": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean", "example": False},
                        "error": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string"},
                                "message": {"type": "string"},
                                "details": {"type": "object"},
                            },
                        },
                    },
                },
            },
        },
        "tags": [
            {
                "name": "Health",
                "description": "Health check and system status endpoints",
            },
            {"name": "Detectors", "description": "Anomaly detector management"},
            {"name": "Detection", "description": "Anomaly detection operations"},
            {
                "name": "AutoML",
                "description": "Automated machine learning optimization",
            },
        ],
    }

    return spec


def save_openapi_spec(spec: dict[str, Any]):
    """Save OpenAPI specification in JSON and YAML formats."""
    log("Saving OpenAPI specification...")

    # Save JSON format
    json_path = OUTPUT_DIR / "openapi.json"
    with open(json_path, "w") as f:
        json.dump(spec, f, indent=2)

    # Save YAML format
    yaml_path = OUTPUT_DIR / "openapi.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(spec, f, default_flow_style=False, sort_keys=False)

    log(f"OpenAPI specification saved to {json_path} and {yaml_path}")


def generate_swagger_ui():
    """Generate Swagger UI for interactive documentation."""
    log("Generating Swagger UI...")

    swagger_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Pynomaly API Documentation</title>
    <link rel="stylesheet" type="text/css" 
          href="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui.css" />
    <style>
        html {
            box-sizing: border-box;
            overflow: -moz-scrollbars-vertical;
            overflow-y: scroll;
        }
        *, *:before, *:after {
            box-sizing: inherit;
        }
        body {
            margin:0;
            background: #fafafa;
        }
        .swagger-ui .topbar {
            background-color: #2E3440;
        }
        .swagger-ui .topbar .link {
            color: #ECEFF4;
        }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-bundle.js">
    </script>
    <script 
        src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-standalone-preset.js">
    </script>
    <script>
        window.onload = function() {
            const ui = SwaggerUIBundle({
                url: './openapi.json',
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                docExpansion: "none",
                defaultModelExpandDepth: 1,
                defaultModelsExpandDepth: 1,
                showExtensions: true,
                showCommonExtensions: true,
                tryItOutEnabled: true,
                requestInterceptor: function(request) {
                    // Add API key if available
                    const apiKey = localStorage.getItem('pynomaly_api_key');
                    if (apiKey) {
                        request.headers['X-API-Key'] = apiKey;
                    }
                    return request;
                }
            });

            // Add API key input
            setTimeout(() => {
                const topbar = document.querySelector('.topbar');
                if (topbar) {
                    const apiKeyInput = document.createElement('input');
                    apiKeyInput.type = 'text';
                    apiKeyInput.placeholder = 'Enter API Key';
                    apiKeyInput.style.marginLeft = '20px';
                    apiKeyInput.style.padding = '5px';
                    apiKeyInput.style.borderRadius = '3px';
                    apiKeyInput.style.border = '1px solid #ccc';

                    const savedKey = localStorage.getItem('pynomaly_api_key');
                    if (savedKey) {
                        apiKeyInput.value = savedKey;
                    }

                    apiKeyInput.addEventListener('change', function() {
                        localStorage.setItem('pynomaly_api_key', this.value);
                    });

                    topbar.appendChild(apiKeyInput);
                }
            }, 1000);
        };
    </script>
</body>
</html>
"""

    swagger_path = OUTPUT_DIR / "index.html"
    with open(swagger_path, "w") as f:
        f.write(swagger_template)

    log(f"Swagger UI generated at {swagger_path}")


def generate_code_examples():
    """Generate code examples for different programming languages."""
    log("Generating code examples...")

    examples = {
        "python": {
            "install": "pip install requests",
            "basic_usage": """
import requests

# Configuration
API_KEY = "your-api-key-here"
BASE_URL = "https://api.pynomaly.com"

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Health check
response = requests.get(f"{BASE_URL}/health", headers=headers)
print(f"Health status: {response.json()}")

# List detectors
response = requests.get(f"{BASE_URL}/api/v1/detectors", headers=headers)
detectors = response.json()
print(f"Available detectors: {len(detectors['data']['detectors'])}")

# Create a detector
detector_data = {
    "name": "My Detector",
    "algorithm": "IsolationForest",
    "parameters": {
        "contamination": 0.1,
        "n_estimators": 100,
        "random_state": 42
    },
    "description": "Production anomaly detector"
}

response = requests.post(
    f"{BASE_URL}/api/v1/detectors",
    headers=headers,
    json=detector_data
)
detector = response.json()
detector_id = detector['data']['id']
print(f"Created detector: {detector_id}")

# Run detection
detection_data = {
    "data": [
        [1.2, 3.4, 5.6],
        [2.3, 4.5, 6.7],
        [3.4, 5.6, 7.8]
    ],
    "feature_names": ["feature1", "feature2", "feature3"],
    "return_explanations": True
}

response = requests.post(
    f"{BASE_URL}/api/v1/detectors/{detector_id}/detect",
    headers=headers,
    json=detection_data
)
results = response.json()
print(f"Detection results: {results['data']['summary']}")
""",
            "sdk_usage": """
# Using the Pynomaly Python SDK
from pynomaly_client import PynomalyClient

# Initialize client
client = PynomalyClient(api_key="your-api-key-here")

# Create detector
detector = client.detectors.create(
    name="My Detector",
    algorithm="IsolationForest",
    parameters={
        "contamination": 0.1,
        "n_estimators": 100
    }
)

# Run detection
results = client.detectors.detect(
    detector_id=detector.id,
    data=[[1.2, 3.4, 5.6], [2.3, 4.5, 6.7]],
    feature_names=["feature1", "feature2", "feature3"]
)

print(f"Anomalies detected: {results.summary.anomalies_detected}")
""",
        },
        "javascript": {
            "install": "npm install axios",
            "basic_usage": """
const axios = require('axios');

// Configuration
const API_KEY = 'your-api-key-here';
const BASE_URL = 'https://api.pynomaly.com';

const headers = {
    'X-API-Key': API_KEY,
    'Content-Type': 'application/json'
};

// Health check
async function checkHealth() {
    try {
        const response = await axios.get(`${BASE_URL}/health`, { headers });
        console.log('Health status:', response.data);
    } catch (error) {
        console.error('Health check failed:', error.response?.data || error.message);
    }
}

// List detectors
async function listDetectors() {
    try {
        const response = await axios.get(`${BASE_URL}/api/v1/detectors`, { headers });
        const detectors = response.data;
        console.log(`Available detectors: ${detectors.data.detectors.length}`);
        return detectors;
    } catch (error) {
        console.error(
            'Failed to list detectors:', 
            error.response?.data || error.message
        );
    }
}

// Create detector
async function createDetector() {
    const detectorData = {
        name: 'My Detector',
        algorithm: 'IsolationForest',
        parameters: {
            contamination: 0.1,
            n_estimators: 100,
            random_state: 42
        },
        description: 'Production anomaly detector'
    };

    try {
        const response = await axios.post(
            `${BASE_URL}/api/v1/detectors`,
            detectorData,
            { headers }
        );
        const detector = response.data;
        console.log(`Created detector: ${detector.data.id}`);
        return detector.data.id;
    } catch (error) {
        console.error(
            'Failed to create detector:', 
            error.response?.data || error.message
        );
    }
}

// Run detection
async function runDetection(detectorId) {
    const detectionData = {
        data: [
            [1.2, 3.4, 5.6],
            [2.3, 4.5, 6.7],
            [3.4, 5.6, 7.8]
        ],
        feature_names: ['feature1', 'feature2', 'feature3'],
        return_explanations: true
    };

    try {
        const response = await axios.post(
            `${BASE_URL}/api/v1/detectors/${detectorId}/detect`,
            detectionData,
            { headers }
        );
        const results = response.data;
        console.log('Detection results:', results.data.summary);
        return results;
    } catch (error) {
        console.error('Detection failed:', error.response?.data || error.message);
    }
}

// Example usage
(async () => {
    await checkHealth();
    await listDetectors();
    const detectorId = await createDetector();
    if (detectorId) {
        await runDetection(detectorId);
    }
})();
""",
            "sdk_usage": """
// Using the Pynomaly JavaScript SDK
import { PynomalyClient } from 'pynomaly-js';

// Initialize client
const client = new PynomalyClient('your-api-key-here');

// Create detector
const detector = await client.detectors.create({
    name: 'My Detector',
    algorithm: 'IsolationForest',
    parameters: {
        contamination: 0.1,
        n_estimators: 100
    }
});

// Run detection
const results = await client.detectors.detect({
    detectorId: detector.id,
    data: [[1.2, 3.4, 5.6], [2.3, 4.5, 6.7]],
    featureNames: ['feature1', 'feature2', 'feature3']
});

console.log(`Anomalies detected: ${results.summary.anomaliesDetected}`);
""",
        },
        "curl": {
            "health_check": """
# Health check
curl -X GET "https://api.pynomaly.com/health" \\
  -H "X-API-Key: your-api-key-here"
""",
            "list_detectors": """
# List detectors
curl -X GET "https://api.pynomaly.com/api/v1/detectors?limit=10" \\
  -H "X-API-Key: your-api-key-here" \\
  -H "Content-Type: application/json"
""",
            "create_detector": """
# Create detector
curl -X POST "https://api.pynomaly.com/api/v1/detectors" \\
  -H "X-API-Key: your-api-key-here" \\
  -H "Content-Type: application/json" \\
  -d '{
    "name": "My Detector",
    "algorithm": "IsolationForest",
    "parameters": {
      "contamination": 0.1,
      "n_estimators": 100,
      "random_state": 42
    },
    "description": "Production anomaly detector"
  }'
""",
            "run_detection": """
# Run detection
curl -X POST "https://api.pynomaly.com/api/v1/detectors/detector_abc123/detect" \\
  -H "X-API-Key: your-api-key-here" \\
  -H "Content-Type: application/json" \\
  -d '{
    "data": [
      [1.2, 3.4, 5.6],
      [2.3, 4.5, 6.7],
      [3.4, 5.6, 7.8]
    ],
    "feature_names": ["feature1", "feature2", "feature3"],
    "return_explanations": true
  }'
""",
            "automl_optimization": """
# Start AutoML optimization
curl -X POST "https://api.pynomaly.com/api/v1/automl/optimize" \\
  -H "X-API-Key: your-api-key-here" \\
  -H "Content-Type: application/json" \\
  -d '{
    "dataset_id": "dataset_abc123",
    "algorithms": ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"],
    "objectives": ["accuracy", "speed", "interpretability"],
    "constraints": {
      "max_time": 3600,
      "max_trials": 100,
      "max_memory": 4096
    }
  }'
""",
        },
    }

    # Save examples
    examples_dir = OUTPUT_DIR / "examples"
    examples_dir.mkdir(exist_ok=True)

    for language, code_examples in examples.items():
        lang_dir = examples_dir / language
        lang_dir.mkdir(exist_ok=True)

        for example_name, code in code_examples.items():
            example_path = lang_dir / f"{example_name}.txt"
            with open(example_path, "w") as f:
                f.write(code.strip())

    log(f"Code examples generated in {examples_dir}")


def generate_postman_collection():
    """Generate Postman collection for API testing."""
    log("Generating Postman collection...")

    collection = {
        "info": {
            "name": "Pynomaly API",
            "description": (
                "Comprehensive API collection for Pynomaly anomaly detection platform"
            ),
            "version": "1.0.0",
            "schema": (
                "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
            ),
        },
        "auth": {
            "type": "apikey",
            "apikey": [
                {"key": "key", "value": "X-API-Key", "type": "string"},
                {"key": "value", "value": "{{api_key}}", "type": "string"},
            ],
        },
        "variable": [
            {"key": "base_url", "value": "https://api.pynomaly.com", "type": "string"},
            {"key": "api_key", "value": "your-api-key-here", "type": "string"},
        ],
        "item": [
            {
                "name": "Health",
                "item": [
                    {
                        "name": "Health Check",
                        "request": {
                            "method": "GET",
                            "header": [],
                            "url": {
                                "raw": "{{base_url}}/health",
                                "host": ["{{base_url}}"],
                                "path": ["health"],
                            },
                        },
                    },
                    {
                        "name": "Detailed Health Check",
                        "request": {
                            "method": "GET",
                            "header": [],
                            "url": {
                                "raw": "{{base_url}}/health/detailed",
                                "host": ["{{base_url}}"],
                                "path": ["health", "detailed"],
                            },
                        },
                    },
                ],
            },
            {
                "name": "Detectors",
                "item": [
                    {
                        "name": "List Detectors",
                        "request": {
                            "method": "GET",
                            "header": [],
                            "url": {
                                "raw": (
                                    "{{base_url}}/api/v1/detectors?limit=20&offset=0"
                                ),
                                "host": ["{{base_url}}"],
                                "path": ["api", "v1", "detectors"],
                                "query": [
                                    {"key": "limit", "value": "20"},
                                    {"key": "offset", "value": "0"},
                                ],
                            },
                        },
                    },
                    {
                        "name": "Create Detector",
                        "request": {
                            "method": "POST",
                            "header": [
                                {"key": "Content-Type", "value": "application/json"}
                            ],
                            "body": {
                                "mode": "raw",
                                "raw": json.dumps(
                                    {
                                        "name": "My Detector",
                                        "algorithm": "IsolationForest",
                                        "parameters": {
                                            "contamination": 0.1,
                                            "n_estimators": 100,
                                            "random_state": 42,
                                        },
                                        "description": "Production anomaly detector",
                                    },
                                    indent=2,
                                ),
                            },
                            "url": {
                                "raw": "{{base_url}}/api/v1/detectors",
                                "host": ["{{base_url}}"],
                                "path": ["api", "v1", "detectors"],
                            },
                        },
                    },
                    {
                        "name": "Get Detector",
                        "request": {
                            "method": "GET",
                            "header": [],
                            "url": {
                                "raw": "{{base_url}}/api/v1/detectors/{{detector_id}}",
                                "host": ["{{base_url}}"],
                                "path": ["api", "v1", "detectors", "{{detector_id}}"],
                            },
                        },
                    },
                    {
                        "name": "Update Detector",
                        "request": {
                            "method": "PUT",
                            "header": [
                                {"key": "Content-Type", "value": "application/json"}
                            ],
                            "body": {
                                "mode": "raw",
                                "raw": json.dumps(
                                    {
                                        "name": "Updated Detector",
                                        "parameters": {
                                            "contamination": 0.05,
                                            "n_estimators": 200,
                                        },
                                    },
                                    indent=2,
                                ),
                            },
                            "url": {
                                "raw": "{{base_url}}/api/v1/detectors/{{detector_id}}",
                                "host": ["{{base_url}}"],
                                "path": ["api", "v1", "detectors", "{{detector_id}}"],
                            },
                        },
                    },
                    {
                        "name": "Delete Detector",
                        "request": {
                            "method": "DELETE",
                            "header": [],
                            "url": {
                                "raw": "{{base_url}}/api/v1/detectors/{{detector_id}}",
                                "host": ["{{base_url}}"],
                                "path": ["api", "v1", "detectors", "{{detector_id}}"],
                            },
                        },
                    },
                ],
            },
            {
                "name": "Detection",
                "item": [
                    {
                        "name": "Run Detection",
                        "request": {
                            "method": "POST",
                            "header": [
                                {"key": "Content-Type", "value": "application/json"}
                            ],
                            "body": {
                                "mode": "raw",
                                "raw": json.dumps(
                                    {
                                        "data": [
                                            [1.2, 3.4, 5.6],
                                            [2.3, 4.5, 6.7],
                                            [3.4, 5.6, 7.8],
                                        ],
                                        "feature_names": [
                                            "feature1",
                                            "feature2",
                                            "feature3",
                                        ],
                                        "return_explanations": True,
                                    },
                                    indent=2,
                                ),
                            },
                            "url": {
                                "raw": (
                                    "{{base_url}}/api/v1/detectors/"
                                    "{{detector_id}}/detect"
                                ),
                                "host": ["{{base_url}}"],
                                "path": [
                                    "api",
                                    "v1",
                                    "detectors",
                                    "{{detector_id}}",
                                    "detect",
                                ],
                            },
                        },
                    }
                ],
            },
            {
                "name": "AutoML",
                "item": [
                    {
                        "name": "Start AutoML Optimization",
                        "request": {
                            "method": "POST",
                            "header": [
                                {"key": "Content-Type", "value": "application/json"}
                            ],
                            "body": {
                                "mode": "raw",
                                "raw": json.dumps(
                                    {
                                        "dataset_id": "dataset_abc123",
                                        "algorithms": [
                                            "IsolationForest",
                                            "LocalOutlierFactor",
                                            "OneClassSVM",
                                        ],
                                        "objectives": [
                                            "accuracy",
                                            "speed",
                                            "interpretability",
                                        ],
                                        "constraints": {
                                            "max_time": 3600,
                                            "max_trials": 100,
                                            "max_memory": 4096,
                                        },
                                    },
                                    indent=2,
                                ),
                            },
                            "url": {
                                "raw": "{{base_url}}/api/v1/automl/optimize",
                                "host": ["{{base_url}}"],
                                "path": ["api", "v1", "automl", "optimize"],
                            },
                        },
                    }
                ],
            },
        ],
    }

    postman_path = OUTPUT_DIR / "pynomaly_api.postman_collection.json"
    with open(postman_path, "w") as f:
        json.dump(collection, f, indent=2)

    log(f"Postman collection generated at {postman_path}")


def generate_readme():
    """Generate README for the API documentation."""
    log("Generating README...")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    readme_content = f"""# Pynomaly API Documentation

Welcome to the Pynomaly API documentation! This directory contains comprehensive 
documentation for the Pynomaly anomaly detection platform.

## 📁 Documentation Structure

```
docs/api/
├── API_DOCUMENTATION.md         # Main API documentation
├── generated/                   # Generated documentation files
│   ├── openapi.json            # OpenAPI 3.0 specification (JSON)
│   ├── openapi.yaml            # OpenAPI 3.0 specification (YAML)
│   ├── index.html              # Interactive Swagger UI
│   ├── examples/               # Code examples
│   │   ├── python/            # Python examples
│   │   ├── javascript/        # JavaScript examples
│   │   └── curl/              # cURL examples
│   └── pynomaly_api.postman_collection.json  # Postman collection
└── README.md                   # This file
```

## 🚀 Getting Started

### 1. Interactive Documentation

Open `generated/index.html` in your browser to explore the API interactively 
with Swagger UI.

### 2. API Specification

- **OpenAPI JSON**: `generated/openapi.json`
- **OpenAPI YAML**: `generated/openapi.yaml`

### 3. Code Examples

Choose your preferred language:

- **Python**: `generated/examples/python/`
- **JavaScript**: `generated/examples/javascript/`
- **cURL**: `generated/examples/curl/`

### 4. Postman Collection

Import `generated/pynomaly_api.postman_collection.json` into Postman for easy 
API testing.

## 📖 API Overview

The Pynomaly API provides comprehensive REST endpoints for:

- **Health Monitoring**: System health and status checks
- **Detector Management**: Create, update, and manage anomaly detectors
- **Detection Operations**: Run anomaly detection on data
- **AutoML**: Automated machine learning optimization
- **Dataset Management**: Handle training and testing datasets
- **Metrics & Monitoring**: Performance and system metrics
- **Explainability**: Model interpretability and explanations

## 🔐 Authentication

All API endpoints require authentication via API key:

```bash
# Header authentication (recommended)
curl -H "X-API-Key: your-api-key" https://api.pynomaly.com/api/v1/detectors

# Query parameter authentication
curl "https://api.pynomaly.com/api/v1/detectors?api_key=your-api-key"
```

## 📚 Quick Examples

### Python

```python
import requests

headers = {{"X-API-Key": "your-api-key"}}
response = requests.get("https://api.pynomaly.com/health", headers=headers)
print(response.json())
```

### JavaScript

```javascript
const response = await fetch('https://api.pynomaly.com/health', {{
    headers: {{ 'X-API-Key': 'your-api-key' }}
}});
const data = await response.json();
console.log(data);
```

### cURL

```bash
curl -H "X-API-Key: your-api-key" https://api.pynomaly.com/health
```

## 🌐 Base URLs

- **Production**: `https://api.pynomaly.com`
- **Staging**: `https://staging-api.pynomaly.com`
- **Development**: `http://localhost:8000`

## 📊 Rate Limits

- **Default**: 1000 requests per minute
- **Burst**: 100 requests per second
- **Training**: 10 concurrent jobs per user
- **Detection**: 10,000 requests per minute

## 📧 Support

- **Documentation**: https://docs.pynomaly.com
- **API Status**: https://status.pynomaly.com
- **Support**: support@pynomaly.com
- **Community**: https://github.com/pynomaly/pynomaly

## 🔄 Regenerating Documentation

To regenerate this documentation:

```bash
cd /path/to/pynomaly
python scripts/generate_api_docs.py
```

---

*Last updated: {timestamp}*
"""

    readme_path = DOCS_DIR / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)

    log(f"README generated at {readme_path}")


def main():
    """Main function to generate all API documentation."""
    log("Starting API documentation generation...")

    try:
        # Generate OpenAPI specification
        spec = generate_openapi_spec()
        save_openapi_spec(spec)

        # Generate interactive documentation
        generate_swagger_ui()

        # Generate code examples
        generate_code_examples()

        # Generate Postman collection
        generate_postman_collection()

        # Generate README
        generate_readme()

        log("API documentation generation completed successfully!")

        # Display summary
        print("\n" + "=" * 60)
        print("📚 API Documentation Generated!")
        print("=" * 60)
        print(f"📁 Output Directory: {OUTPUT_DIR}")
        print(f"🌐 Interactive Docs: {OUTPUT_DIR / 'index.html'}")
        print(f"📄 OpenAPI Spec: {OUTPUT_DIR / 'openapi.json'}")
        print(
            f"🔧 Postman Collection: "
            f"{OUTPUT_DIR / 'pynomaly_api.postman_collection.json'}"
        )
        print(f"💻 Code Examples: {OUTPUT_DIR / 'examples'}")
        print("=" * 60)

        # Instructions
        print("\n🚀 Next Steps:")
        print("1. Open the interactive documentation:")
        print(f"   file://{OUTPUT_DIR / 'index.html'}")
        print("2. Import the Postman collection for API testing")
        print("3. Use the code examples to integrate with your applications")
        print("4. Review the comprehensive API documentation")
        print("\n✅ All documentation files are ready for use!")

    except Exception as e:
        error(f"Documentation generation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
