#!/usr/bin/env python3
"""
Generate comprehensive API documentation from Pynomaly API Gateway endpoints.
"""

import json
import yaml
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Import Pynomaly components
from src.pynomaly.features.api_gateway import (
    APIGateway, 
    get_api_gateway,
    HTTPMethod,
    ResponseFormat,
    EndpointStatus
)

class APIDocumentationGenerator:
    """Generate comprehensive API documentation."""
    
    def __init__(self, output_dir: str = "docs/api"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gateway = get_api_gateway()
        
    async def generate_all_documentation(self) -> Dict[str, Any]:
        """Generate all API documentation formats."""
        
        # Register standard endpoints
        await self.gateway.register_anomaly_detection_endpoints()
        
        # Generate different documentation formats
        docs = {
            'openapi': await self.generate_openapi_spec(),
            'postman': await self.generate_postman_collection(),
            'markdown': await self.generate_markdown_docs(),
            'interactive': await self.generate_interactive_docs(),
            'curl_examples': await self.generate_curl_examples(),
            'sdk_examples': await self.generate_sdk_examples()
        }
        
        # Save all documentation
        await self.save_documentation(docs)
        
        return docs
    
    async def generate_openapi_spec(self) -> Dict[str, Any]:
        """Generate OpenAPI 3.0 specification."""
        
        # Get all registered endpoints
        endpoints = self.gateway.endpoint_manager.list_endpoints()
        
        openapi_spec = {
            "openapi": "3.0.3",
            "info": {
                "title": "Pynomaly API",
                "description": "Production-ready Anomaly Detection API",
                "version": "1.0.0",
                "contact": {
                    "name": "Pynomaly Team",
                    "url": "https://github.com/your-org/pynomaly",
                    "email": "support@pynomaly.com"
                },
                "license": {
                    "name": "MIT",
                    "url": "https://opensource.org/licenses/MIT"
                }
            },
            "servers": [
                {
                    "url": "https://api.pynomaly.com/v1",
                    "description": "Production server"
                },
                {
                    "url": "https://staging-api.pynomaly.com/v1",
                    "description": "Staging server"
                },
                {
                    "url": "http://localhost:8000/v1",
                    "description": "Development server"
                }
            ],
            "paths": {},
            "components": {
                "schemas": self._generate_schemas(),
                "securitySchemes": {
                    "bearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT"
                    },
                    "apiKey": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key"
                    }
                }
            },
            "security": [
                {"bearerAuth": []},
                {"apiKey": []}
            ]
        }
        
        # Add endpoint paths
        for endpoint in endpoints:
            path = endpoint.path
            method = endpoint.method.value.lower()
            
            if path not in openapi_spec["paths"]:
                openapi_spec["paths"][path] = {}
                
            openapi_spec["paths"][path][method] = {
                "summary": endpoint.name or f"{method.upper()} {path}",
                "description": endpoint.description or f"Endpoint: {endpoint.name}",
                "operationId": f"{method}_{path.replace('/', '_').replace('-', '_')}",
                "tags": endpoint.tags or ["general"],
                "parameters": self._generate_parameters(endpoint),
                "responses": self._generate_responses(endpoint),
                "security": [{"bearerAuth": []}] if endpoint.auth_required else []
            }
            
            # Add request body for POST/PUT methods
            if method in ['post', 'put', 'patch']:
                openapi_spec["paths"][path][method]["requestBody"] = self._generate_request_body(endpoint)
        
        return openapi_spec
    
    async def generate_postman_collection(self) -> Dict[str, Any]:
        """Generate Postman collection."""
        
        endpoints = self.gateway.endpoint_manager.list_endpoints()
        
        collection = {
            "info": {
                "name": "Pynomaly API",
                "description": "Complete Pynomaly API collection for testing",
                "version": "1.0.0",
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
            },
            "auth": {
                "type": "bearer",
                "bearer": [
                    {
                        "key": "token",
                        "value": "{{access_token}}",
                        "type": "string"
                    }
                ]
            },
            "variable": [
                {
                    "key": "base_url",
                    "value": "https://api.pynomaly.com",
                    "type": "string"
                },
                {
                    "key": "access_token",
                    "value": "your_jwt_token_here",
                    "type": "string"
                }
            ],
            "item": []
        }
        
        # Group endpoints by tags
        endpoint_groups = {}
        for endpoint in endpoints:
            tags = endpoint.tags or ["General"]
            for tag in tags:
                if tag not in endpoint_groups:
                    endpoint_groups[tag] = []
                endpoint_groups[tag].append(endpoint)
        
        # Create Postman folders and requests
        for group_name, group_endpoints in endpoint_groups.items():
            folder = {
                "name": group_name,
                "item": []
            }
            
            for endpoint in group_endpoints:
                request = {
                    "name": endpoint.name or f"{endpoint.method.value} {endpoint.path}",
                    "request": {
                        "method": endpoint.method.value,
                        "header": [
                            {
                                "key": "Content-Type",
                                "value": "application/json"
                            }
                        ],
                        "url": {
                            "raw": "{{base_url}}" + endpoint.path,
                            "host": ["{{base_url}}"],
                            "path": endpoint.path.strip("/").split("/")
                        }
                    },
                    "response": []
                }
                
                # Add request body for POST/PUT methods
                if endpoint.method in [HTTPMethod.POST, HTTPMethod.PUT, HTTPMethod.PATCH]:
                    request["request"]["body"] = {
                        "mode": "raw",
                        "raw": json.dumps(self._generate_example_request_body(endpoint), indent=2)
                    }
                
                folder["item"].append(request)
            
            collection["item"].append(folder)
        
        return collection
    
    async def generate_markdown_docs(self) -> str:
        """Generate Markdown documentation."""
        
        endpoints = self.gateway.endpoint_manager.list_endpoints()
        
        markdown = f"""# Pynomaly API Documentation

Generated on: {datetime.now().isoformat()}

## Overview

Pynomaly is a production-ready anomaly detection API that provides:

- **Real-time anomaly detection** with streaming capabilities
- **Advanced analytics** with explainable AI
- **Model management** with MLOps lifecycle
- **Feature engineering** with preprocessing pipelines
- **Comprehensive monitoring** and observability

## Base URL

```
Production: https://api.pynomaly.com
Staging: https://staging-api.pynomaly.com
Development: http://localhost:8000
```

## Authentication

All API endpoints require authentication. Pynomaly supports:

1. **Bearer Token (JWT)**
   ```
   Authorization: Bearer <your_jwt_token>
   ```

2. **API Key**
   ```
   X-API-Key: <your_api_key>
   ```

## Rate Limiting

- **Global limit**: 10,000 requests per hour
- **Per user limit**: 1,000 requests per hour
- **Per IP limit**: 500 requests per hour

Rate limit headers are included in all responses:
- `X-RateLimit-Limit`: Request limit per hour
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Reset time (Unix timestamp)

## Error Handling

All errors follow the standard format:

```json
{{
  "error": "error_type",
  "message": "Human readable error message",
  "details": {{
    "field": "Additional error details"
  }},
  "timestamp": "2024-01-01T12:00:00Z",
  "request_id": "req_123456789"
}}
```

## API Endpoints

"""
        
        # Group endpoints by tags
        endpoint_groups = {}
        for endpoint in endpoints:
            tags = endpoint.tags or ["General"]
            for tag in tags:
                if tag not in endpoint_groups:
                    endpoint_groups[tag] = []
                endpoint_groups[tag].append(endpoint)
        
        # Generate documentation for each group
        for group_name, group_endpoints in endpoint_groups.items():
            markdown += f"\n### {group_name}\n\n"
            
            for endpoint in group_endpoints:
                markdown += f"#### {endpoint.method.value} {endpoint.path}\n\n"
                
                if endpoint.description:
                    markdown += f"{endpoint.description}\n\n"
                
                # Status badge
                status_color = {
                    EndpointStatus.ACTIVE: "green",
                    EndpointStatus.DEPRECATED: "orange",
                    EndpointStatus.INACTIVE: "red",
                    EndpointStatus.MAINTENANCE: "yellow"
                }
                
                markdown += f"![Status](https://img.shields.io/badge/status-{endpoint.status.value}-{status_color.get(endpoint.status, 'gray')})\n\n"
                
                # Request example
                markdown += "**Request:**\n\n"
                markdown += f"```http\n{endpoint.method.value} {endpoint.path} HTTP/1.1\n"
                markdown += "Host: api.pynomaly.com\n"
                markdown += "Content-Type: application/json\n"
                
                if endpoint.auth_required:
                    markdown += "Authorization: Bearer <your_jwt_token>\n"
                
                if endpoint.method in [HTTPMethod.POST, HTTPMethod.PUT, HTTPMethod.PATCH]:
                    markdown += "\n"
                    markdown += json.dumps(self._generate_example_request_body(endpoint), indent=2)
                
                markdown += "\n```\n\n"
                
                # Response example
                markdown += "**Response:**\n\n"
                markdown += "```json\n"
                markdown += json.dumps(self._generate_example_response(endpoint), indent=2)
                markdown += "\n```\n\n"
                
                # Rate limiting info
                if endpoint.rate_limit:
                    markdown += f"**Rate Limit:** {endpoint.rate_limit} requests per minute\n\n"
                
                markdown += "---\n\n"
        
        return markdown
    
    async def generate_interactive_docs(self) -> Dict[str, Any]:
        """Generate interactive documentation configuration."""
        
        return {
            "swagger_ui": {
                "url": "/docs",
                "title": "Pynomaly API - Interactive Documentation",
                "description": "Try out the API endpoints directly from your browser",
                "version": "1.0.0"
            },
            "redoc": {
                "url": "/redoc",
                "title": "Pynomaly API - Documentation",
                "description": "Comprehensive API documentation with examples",
                "version": "1.0.0"
            },
            "rapidoc": {
                "url": "/rapidoc",
                "title": "Pynomaly API - RapiDoc",
                "description": "Modern API documentation with try-it-out features",
                "version": "1.0.0"
            }
        }
    
    async def generate_curl_examples(self) -> Dict[str, List[str]]:
        """Generate cURL examples for all endpoints."""
        
        endpoints = self.gateway.endpoint_manager.list_endpoints()
        curl_examples = {}
        
        for endpoint in endpoints:
            group = endpoint.tags[0] if endpoint.tags else "general"
            
            if group not in curl_examples:
                curl_examples[group] = []
            
            # Base curl command
            curl_cmd = f"curl -X {endpoint.method.value} \\\n"
            curl_cmd += f"  'https://api.pynomaly.com{endpoint.path}' \\\n"
            curl_cmd += f"  -H 'Content-Type: application/json' \\\n"
            
            if endpoint.auth_required:
                curl_cmd += f"  -H 'Authorization: Bearer $ACCESS_TOKEN' \\\n"
            
            # Add request body for POST/PUT methods
            if endpoint.method in [HTTPMethod.POST, HTTPMethod.PUT, HTTPMethod.PATCH]:
                example_body = self._generate_example_request_body(endpoint)
                curl_cmd += f"  -d '{json.dumps(example_body)}'"
            else:
                curl_cmd = curl_cmd.rstrip(" \\\n")
            
            curl_examples[group].append({
                "name": endpoint.name or f"{endpoint.method.value} {endpoint.path}",
                "description": endpoint.description or "",
                "command": curl_cmd
            })
        
        return curl_examples
    
    async def generate_sdk_examples(self) -> Dict[str, Dict[str, str]]:
        """Generate SDK examples in multiple languages."""
        
        endpoints = self.gateway.endpoint_manager.list_endpoints()
        sdk_examples = {
            "python": {},
            "javascript": {},
            "java": {},
            "go": {}
        }
        
        for endpoint in endpoints:
            endpoint_name = endpoint.name or f"{endpoint.method.value}_{endpoint.path.replace('/', '_')}"
            
            # Python example
            sdk_examples["python"][endpoint_name] = self._generate_python_example(endpoint)
            
            # JavaScript example
            sdk_examples["javascript"][endpoint_name] = self._generate_javascript_example(endpoint)
            
            # Java example
            sdk_examples["java"][endpoint_name] = self._generate_java_example(endpoint)
            
            # Go example
            sdk_examples["go"][endpoint_name] = self._generate_go_example(endpoint)
        
        return sdk_examples
    
    def _generate_schemas(self) -> Dict[str, Any]:
        """Generate OpenAPI schemas for common data structures."""
        
        return {
            "AnomalyDetectionRequest": {
                "type": "object",
                "required": ["data"],
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "description": "Data point for anomaly detection"
                        },
                        "description": "Array of data points to analyze"
                    },
                    "algorithm": {
                        "type": "string",
                        "enum": ["isolation_forest", "one_class_svm", "local_outlier_factor"],
                        "description": "Algorithm to use for detection"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Algorithm-specific parameters"
                    }
                }
            },
            "AnomalyDetectionResponse": {
                "type": "object",
                "properties": {
                    "detection_id": {
                        "type": "string",
                        "description": "Unique identifier for this detection"
                    },
                    "anomalies_detected": {
                        "type": "integer",
                        "description": "Number of anomalies detected"
                    },
                    "anomalies": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/Anomaly"}
                    },
                    "processing_time_ms": {
                        "type": "number",
                        "description": "Processing time in milliseconds"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Detection timestamp"
                    }
                }
            },
            "Anomaly": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Unique anomaly identifier"
                    },
                    "score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Anomaly score (0-1)"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["point", "contextual", "collective"],
                        "description": "Type of anomaly"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "When the anomaly was detected"
                    },
                    "data_point": {
                        "type": "object",
                        "description": "The original data point"
                    },
                    "explanation": {
                        "type": "object",
                        "description": "Anomaly explanation details"
                    }
                }
            },
            "HealthResponse": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["healthy", "unhealthy", "degraded"],
                        "description": "Overall system health status"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Health check timestamp"
                    },
                    "version": {
                        "type": "string",
                        "description": "API version"
                    },
                    "uptime": {
                        "type": "number",
                        "description": "System uptime in seconds"
                    },
                    "components": {
                        "type": "object",
                        "description": "Component health status"
                    }
                }
            },
            "Error": {
                "type": "object",
                "properties": {
                    "error": {
                        "type": "string",
                        "description": "Error type"
                    },
                    "message": {
                        "type": "string",
                        "description": "Human-readable error message"
                    },
                    "details": {
                        "type": "object",
                        "description": "Additional error details"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Error timestamp"
                    },
                    "request_id": {
                        "type": "string",
                        "description": "Request identifier for debugging"
                    }
                }
            }
        }
    
    def _generate_parameters(self, endpoint) -> List[Dict[str, Any]]:
        """Generate OpenAPI parameters for an endpoint."""
        
        parameters = []
        
        # Add common parameters based on endpoint
        if endpoint.path == "/v1/detect":
            parameters.extend([
                {
                    "name": "async",
                    "in": "query",
                    "description": "Process request asynchronously",
                    "required": False,
                    "schema": {"type": "boolean", "default": False}
                },
                {
                    "name": "explain",
                    "in": "query",
                    "description": "Include anomaly explanations",
                    "required": False,
                    "schema": {"type": "boolean", "default": False}
                }
            ])
        
        return parameters
    
    def _generate_responses(self, endpoint) -> Dict[str, Any]:
        """Generate OpenAPI responses for an endpoint."""
        
        responses = {
            "200": {
                "description": "Success",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/AnomalyDetectionResponse"}
                    }
                }
            },
            "400": {
                "description": "Bad Request",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "401": {
                "description": "Unauthorized",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "429": {
                "description": "Rate Limit Exceeded",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "500": {
                "description": "Internal Server Error",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            }
        }
        
        # Customize responses based on endpoint
        if endpoint.path == "/health":
            responses["200"]["content"]["application/json"]["schema"] = {
                "$ref": "#/components/schemas/HealthResponse"
            }
        
        return responses
    
    def _generate_request_body(self, endpoint) -> Dict[str, Any]:
        """Generate OpenAPI request body for an endpoint."""
        
        if endpoint.path == "/v1/detect":
            return {
                "description": "Anomaly detection request",
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/AnomalyDetectionRequest"},
                        "example": self._generate_example_request_body(endpoint)
                    }
                }
            }
        
        return {
            "description": "Request body",
            "required": True,
            "content": {
                "application/json": {
                    "schema": {"type": "object"}
                }
            }
        }
    
    def _generate_example_request_body(self, endpoint) -> Dict[str, Any]:
        """Generate example request body for an endpoint."""
        
        if endpoint.path == "/v1/detect":
            return {
                "data": [
                    {"feature_1": 1.5, "feature_2": -0.8, "feature_3": 2.1},
                    {"feature_1": 0.3, "feature_2": 1.2, "feature_3": -0.5},
                    {"feature_1": 3.7, "feature_2": -2.1, "feature_3": 0.9}
                ],
                "algorithm": "isolation_forest",
                "parameters": {
                    "contamination": 0.1,
                    "n_estimators": 100
                }
            }
        
        return {}
    
    def _generate_example_response(self, endpoint) -> Dict[str, Any]:
        """Generate example response for an endpoint."""
        
        if endpoint.path == "/health":
            return {
                "status": "healthy",
                "timestamp": "2024-01-01T12:00:00Z",
                "version": "1.0.0",
                "uptime": 3600,
                "components": {
                    "database": "healthy",
                    "redis": "healthy",
                    "streaming": "healthy"
                }
            }
        elif endpoint.path == "/v1/detect":
            return {
                "detection_id": "det_123456789",
                "anomalies_detected": 1,
                "anomalies": [
                    {
                        "id": "anomaly_001",
                        "score": 0.87,
                        "type": "point",
                        "timestamp": "2024-01-01T12:00:00Z",
                        "data_point": {"feature_1": 3.7, "feature_2": -2.1, "feature_3": 0.9},
                        "explanation": {
                            "top_features": ["feature_1", "feature_2"],
                            "feature_contributions": {"feature_1": 0.6, "feature_2": 0.4}
                        }
                    }
                ],
                "processing_time_ms": 45.2,
                "timestamp": "2024-01-01T12:00:00Z"
            }
        
        return {"message": "Success"}
    
    def _generate_python_example(self, endpoint) -> str:
        """Generate Python SDK example."""
        
        if endpoint.path == "/v1/detect":
            return """
import requests
import json

# Configure API client
base_url = "https://api.pynomaly.com"
headers = {
    "Authorization": "Bearer YOUR_ACCESS_TOKEN",
    "Content-Type": "application/json"
}

# Prepare detection request
data = {
    "data": [
        {"feature_1": 1.5, "feature_2": -0.8, "feature_3": 2.1},
        {"feature_1": 0.3, "feature_2": 1.2, "feature_3": -0.5},
        {"feature_1": 3.7, "feature_2": -2.1, "feature_3": 0.9}
    ],
    "algorithm": "isolation_forest",
    "parameters": {
        "contamination": 0.1,
        "n_estimators": 100
    }
}

# Make request
response = requests.post(
    f"{base_url}/v1/detect",
    headers=headers,
    json=data
)

# Process response
if response.status_code == 200:
    result = response.json()
    print(f"Detected {result['anomalies_detected']} anomalies")
    for anomaly in result['anomalies']:
        print(f"Anomaly {anomaly['id']}: score={anomaly['score']}")
else:
    print(f"Error: {response.status_code} - {response.text}")
"""
        
        return f"""
import requests

response = requests.{endpoint.method.value.lower()}(
    "https://api.pynomaly.com{endpoint.path}",
    headers={{"Authorization": "Bearer YOUR_ACCESS_TOKEN"}}
)

print(response.json())
"""
    
    def _generate_javascript_example(self, endpoint) -> str:
        """Generate JavaScript SDK example."""
        
        if endpoint.path == "/v1/detect":
            return """
const axios = require('axios');

// Configure API client
const baseURL = 'https://api.pynomaly.com';
const headers = {
    'Authorization': 'Bearer YOUR_ACCESS_TOKEN',
    'Content-Type': 'application/json'
};

// Prepare detection request
const data = {
    data: [
        { feature_1: 1.5, feature_2: -0.8, feature_3: 2.1 },
        { feature_1: 0.3, feature_2: 1.2, feature_3: -0.5 },
        { feature_1: 3.7, feature_2: -2.1, feature_3: 0.9 }
    ],
    algorithm: 'isolation_forest',
    parameters: {
        contamination: 0.1,
        n_estimators: 100
    }
};

// Make request
axios.post(`${baseURL}/v1/detect`, data, { headers })
    .then(response => {
        const result = response.data;
        console.log(`Detected ${result.anomalies_detected} anomalies`);
        result.anomalies.forEach(anomaly => {
            console.log(`Anomaly ${anomaly.id}: score=${anomaly.score}`);
        });
    })
    .catch(error => {
        console.error('Error:', error.response?.data || error.message);
    });
"""
        
        return f"""
const axios = require('axios');

axios.{endpoint.method.value.lower()}('https://api.pynomaly.com{endpoint.path}', {{
    headers: {{
        'Authorization': 'Bearer YOUR_ACCESS_TOKEN'
    }}
}})
.then(response => console.log(response.data))
.catch(error => console.error(error));
"""
    
    def _generate_java_example(self, endpoint) -> str:
        """Generate Java SDK example."""
        
        return f"""
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.URI;

HttpClient client = HttpClient.newHttpClient();
HttpRequest request = HttpRequest.newBuilder()
    .uri(URI.create("https://api.pynomaly.com{endpoint.path}"))
    .header("Authorization", "Bearer YOUR_ACCESS_TOKEN")
    .header("Content-Type", "application/json")
    .{endpoint.method.value.upper()}()
    .build();

HttpResponse<String> response = client.send(request, 
    HttpResponse.BodyHandlers.ofString());

System.out.println(response.body());
"""
    
    def _generate_go_example(self, endpoint) -> str:
        """Generate Go SDK example."""
        
        return f"""
package main

import (
    "fmt"
    "net/http"
    "io/ioutil"
)

func main() {{
    client := &http.Client{{}}
    
    req, err := http.NewRequest("{endpoint.method.value}", 
        "https://api.pynomaly.com{endpoint.path}", nil)
    if err != nil {{
        panic(err)
    }}
    
    req.Header.Add("Authorization", "Bearer YOUR_ACCESS_TOKEN")
    req.Header.Add("Content-Type", "application/json")
    
    resp, err := client.Do(req)
    if err != nil {{
        panic(err)
    }}
    defer resp.Body.Close()
    
    body, err := ioutil.ReadAll(resp.Body)
    if err != nil {{
        panic(err)
    }}
    
    fmt.Println(string(body))
}}
"""
    
    async def save_documentation(self, docs: Dict[str, Any]) -> None:
        """Save all documentation to files."""
        
        # Save OpenAPI spec
        with open(self.output_dir / "openapi.json", "w") as f:
            json.dump(docs["openapi"], f, indent=2)
        
        with open(self.output_dir / "openapi.yaml", "w") as f:
            yaml.dump(docs["openapi"], f, default_flow_style=False)
        
        # Save Postman collection
        with open(self.output_dir / "postman_collection.json", "w") as f:
            json.dump(docs["postman"], f, indent=2)
        
        # Save Markdown documentation
        with open(self.output_dir / "README.md", "w") as f:
            f.write(docs["markdown"])
        
        # Save interactive docs config
        with open(self.output_dir / "interactive_docs.json", "w") as f:
            json.dump(docs["interactive"], f, indent=2)
        
        # Save cURL examples
        with open(self.output_dir / "curl_examples.json", "w") as f:
            json.dump(docs["curl_examples"], f, indent=2)
        
        # Save SDK examples
        for language, examples in docs["sdk_examples"].items():
            lang_dir = self.output_dir / "sdk_examples" / language
            lang_dir.mkdir(parents=True, exist_ok=True)
            
            for endpoint_name, example_code in examples.items():
                with open(lang_dir / f"{endpoint_name}.{self._get_file_extension(language)}", "w") as f:
                    f.write(example_code)
    
    def _get_file_extension(self, language: str) -> str:
        """Get file extension for programming language."""
        extensions = {
            "python": "py",
            "javascript": "js",
            "java": "java",
            "go": "go"
        }
        return extensions.get(language, "txt")

async def main():
    """Generate all API documentation."""
    
    print("🚀 Generating comprehensive API documentation...")
    
    generator = APIDocumentationGenerator()
    docs = await generator.generate_all_documentation()
    
    print("✅ Documentation generated successfully!")
    print(f"📁 Output directory: {generator.output_dir}")
    print(f"📄 Generated files:")
    print(f"  - OpenAPI spec: openapi.json, openapi.yaml")
    print(f"  - Postman collection: postman_collection.json")
    print(f"  - Markdown docs: README.md")
    print(f"  - Interactive docs: interactive_docs.json")
    print(f"  - cURL examples: curl_examples.json")
    print(f"  - SDK examples: sdk_examples/")
    
    # Print summary
    openapi_endpoints = len(docs["openapi"]["paths"])
    print(f"\n📊 Summary:")
    print(f"  - Total endpoints documented: {openapi_endpoints}")
    print(f"  - Languages supported: Python, JavaScript, Java, Go")
    print(f"  - Documentation formats: OpenAPI, Postman, Markdown, Interactive")

if __name__ == "__main__":
    asyncio.run(main())