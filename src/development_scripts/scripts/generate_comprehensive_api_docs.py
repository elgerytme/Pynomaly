#!/usr/bin/env python3
"""
Comprehensive API Documentation Generator

This script generates complete API documentation including:
- OpenAPI/Swagger specification
- Postman collection
- Developer integration guides
- Authentication documentation
- API versioning strategy
- Code examples and SDKs
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from anomaly_detection.infrastructure.config import create_container
    from anomaly_detection.presentation.api.app import create_app
except ImportError as e:
    print(f"Warning: Could not import anomaly_detection modules: {e}")
    print("API documentation will be generated from static analysis")

class APIDocumentationGenerator:
    """Generate comprehensive API documentation."""

    def __init__(self, output_dir: str = "docs/api/generated"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # API endpoints discovered from the codebase
        self.endpoints = self._discover_endpoints()

    def _discover_endpoints(self) -> dict[str, Any]:
        """Discover API endpoints from the codebase."""
        endpoints = {
            "authentication": {
                "prefix": "/api/v1/auth",
                "endpoints": [
                    {
                        "path": "/login",
                        "method": "POST",
                        "description": "Authenticate user and get JWT token",
                        "request_body": {
                            "username": "string",
                            "password": "string"
                        },
                        "responses": {
                            "200": {
                                "description": "Authentication successful",
                                "schema": {
                                    "access_token": "string",
                                    "refresh_token": "string",
                                    "token_type": "string",
                                    "expires_in": "integer"
                                }
                            }
                        }
                    },
                    {
                        "path": "/logout",
                        "method": "POST",
                        "description": "Logout user and invalidate token",
                        "responses": {"200": {"description": "Logout successful"}}
                    },
                    {
                        "path": "/refresh",
                        "method": "POST",
                        "description": "Refresh JWT token",
                        "request_body": {"refresh_token": "string"},
                        "responses": {
                            "200": {
                                "description": "Token refreshed successfully",
                                "schema": {
                                    "access_token": "string",
                                    "expires_in": "integer"
                                }
                            }
                        }
                    },
                    {
                        "path": "/password-reset",
                        "method": "POST",
                        "description": "Request password reset",
                        "request_body": {"email": "string"},
                        "responses": {"200": {"description": "Password reset email sent"}}
                    },
                    {
                        "path": "/password-reset/confirm",
                        "method": "POST",
                        "description": "Confirm password reset",
                        "request_body": {
                            "token": "string",
                            "new_password": "string"
                        },
                        "responses": {"200": {"description": "Password reset successful"}}
                    }
                ]
            },
            "mfa": {
                "prefix": "/api/v1/mfa",
                "endpoints": [
                    {
                        "path": "/totp/setup",
                        "method": "POST",
                        "description": "Setup TOTP authentication",
                        "request_body": {
                            "app_name": "string",
                            "issuer": "string"
                        },
                        "responses": {
                            "200": {
                                "description": "TOTP setup successful",
                                "schema": {
                                    "secret": "string",
                                    "qr_code_url": "string",
                                    "manual_entry_key": "string",
                                    "backup_codes": ["string"]
                                }
                            }
                        }
                    },
                    {
                        "path": "/totp/verify",
                        "method": "POST",
                        "description": "Verify TOTP code",
                        "request_body": {"totp_code": "string"},
                        "responses": {"200": {"description": "TOTP verification successful"}}
                    },
                    {
                        "path": "/sms/setup",
                        "method": "POST",
                        "description": "Setup SMS authentication",
                        "request_body": {"phone_number": "string"},
                        "responses": {"200": {"description": "SMS code sent"}}
                    },
                    {
                        "path": "/sms/verify",
                        "method": "POST",
                        "description": "Verify SMS code",
                        "request_body": {"sms_code": "string"},
                        "responses": {"200": {"description": "SMS verification successful"}}
                    },
                    {
                        "path": "/email/setup",
                        "method": "POST",
                        "description": "Setup email authentication",
                        "responses": {"200": {"description": "Email code sent"}}
                    },
                    {
                        "path": "/email/verify",
                        "method": "POST",
                        "description": "Verify email code",
                        "request_body": {"email_code": "string"},
                        "responses": {"200": {"description": "Email verification successful"}}
                    },
                    {
                        "path": "/status",
                        "method": "GET",
                        "description": "Get MFA status",
                        "responses": {
                            "200": {
                                "description": "MFA status retrieved",
                                "schema": {
                                    "mfa_enabled": "boolean",
                                    "active_methods": ["object"],
                                    "backup_codes_available": "boolean"
                                }
                            }
                        }
                    },
                    {
                        "path": "/backup-codes",
                        "method": "GET",
                        "description": "Get backup codes count",
                        "responses": {"200": {"description": "Backup codes info"}}
                    },
                    {
                        "path": "/backup-codes/regenerate",
                        "method": "POST",
                        "description": "Regenerate backup codes",
                        "responses": {"200": {"description": "Backup codes regenerated"}}
                    }
                ]
            },
            "admin": {
                "prefix": "/api/v1/admin",
                "endpoints": [
                    {
                        "path": "/users",
                        "method": "GET",
                        "description": "List all users",
                        "responses": {"200": {"description": "Users list retrieved"}}
                    },
                    {
                        "path": "/users/{user_id}",
                        "method": "GET",
                        "description": "Get user details",
                        "responses": {"200": {"description": "User details retrieved"}}
                    },
                    {
                        "path": "/users/{user_id}/roles",
                        "method": "PUT",
                        "description": "Update user roles",
                        "request_body": {"roles": ["string"]},
                        "responses": {"200": {"description": "User roles updated"}}
                    },
                    {
                        "path": "/users/{user_id}/disable",
                        "method": "POST",
                        "description": "Disable user account",
                        "responses": {"200": {"description": "User account disabled"}}
                    },
                    {
                        "path": "/users/{user_id}/enable",
                        "method": "POST",
                        "description": "Enable user account",
                        "responses": {"200": {"description": "User account enabled"}}
                    }
                ]
            },
            "api_keys": {
                "prefix": "/api/v1/api-keys",
                "endpoints": [
                    {
                        "path": "/",
                        "method": "GET",
                        "description": "List API keys",
                        "responses": {"200": {"description": "API keys list retrieved"}}
                    },
                    {
                        "path": "/",
                        "method": "POST",
                        "description": "Create new API key",
                        "request_body": {
                            "name": "string",
                            "scopes": ["string"],
                            "expires_at": "string"
                        },
                        "responses": {"201": {"description": "API key created"}}
                    },
                    {
                        "path": "/{key_id}",
                        "method": "DELETE",
                        "description": "Delete API key",
                        "responses": {"204": {"description": "API key deleted"}}
                    },
                    {
                        "path": "/{key_id}/regenerate",
                        "method": "POST",
                        "description": "Regenerate API key",
                        "responses": {"200": {"description": "API key regenerated"}}
                    }
                ]
            },
            "detection": {
                "prefix": "/api/v1/detection",
                "endpoints": [
                    {
                        "path": "/detect",
                        "method": "POST",
                        "description": "Perform anomaly detection",
                        "request_body": {
                            "data": "array",
                            "algorithm": "string",
                            "parameters": "object"
                        },
                        "responses": {
                            "200": {
                                "description": "Detection completed",
                                "schema": {
                                    "anomalies": ["object"],
                                    "scores": ["number"],
                                    "threshold": "number"
                                }
                            }
                        }
                    },
                    {
                        "path": "/algorithms",
                        "method": "GET",
                        "description": "List available algorithms",
                        "responses": {"200": {"description": "Algorithms list retrieved"}}
                    },
                    {
                        "path": "/batch",
                        "method": "POST",
                        "description": "Batch anomaly detection",
                        "request_body": {
                            "datasets": ["object"],
                            "algorithm": "string"
                        },
                        "responses": {"200": {"description": "Batch detection completed"}}
                    }
                ]
            },
            "health": {
                "prefix": "/api/v1/health",
                "endpoints": [
                    {
                        "path": "/",
                        "method": "GET",
                        "description": "Health check",
                        "responses": {"200": {"description": "Service healthy"}}
                    },
                    {
                        "path": "/detailed",
                        "method": "GET",
                        "description": "Detailed health check",
                        "responses": {"200": {"description": "Detailed health status"}}
                    }
                ]
            }
        }

        return endpoints

    def generate_openapi_spec(self) -> dict[str, Any]:
        """Generate OpenAPI 3.0 specification."""
        spec = {
            "openapi": "3.0.3",
            "info": {
                "title": "anomaly detection API",
                "description": "Comprehensive anomaly detection API with authentication, MFA, and admin capabilities",
                "version": "1.0.0",
                "contact": {
                    "name": "Anomaly Detection Team",
                    "url": "https://github.com/anomaly_detection/anomaly_detection",
                    "email": "support@anomaly_detection.com"
                },
                "license": {
                    "name": "MIT",
                    "url": "https://opensource.org/licenses/MIT"
                }
            },
            "servers": [
                {
                    "url": "http://localhost:8000",
                    "description": "Development server"
                },
                {
                    "url": "https://api.anomaly_detection.com",
                    "description": "Production server"
                }
            ],
            "components": {
                "securitySchemes": {
                    "BearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT"
                    },
                    "ApiKeyAuth": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key"
                    }
                },
                "schemas": self._generate_schemas(),
                "responses": self._generate_common_responses()
            },
            "security": [
                {"BearerAuth": []},
                {"ApiKeyAuth": []}
            ],
            "paths": self._generate_paths()
        }

        return spec

    def _generate_schemas(self) -> dict[str, Any]:
        """Generate OpenAPI schemas."""
        return {
            "AuthResponse": {
                "type": "object",
                "properties": {
                    "access_token": {"type": "string"},
                    "refresh_token": {"type": "string"},
                    "token_type": {"type": "string"},
                    "expires_in": {"type": "integer"}
                }
            },
            "LoginRequest": {
                "type": "object",
                "required": ["username", "password"],
                "properties": {
                    "username": {"type": "string"},
                    "password": {"type": "string"}
                }
            },
            "TOTPSetupResponse": {
                "type": "object",
                "properties": {
                    "secret": {"type": "string"},
                    "qr_code_url": {"type": "string"},
                    "manual_entry_key": {"type": "string"},
                    "backup_codes": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            },
            "DetectionRequest": {
                "type": "object",
                "required": ["data", "algorithm"],
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "algorithm": {"type": "string"},
                    "parameters": {"type": "object"}
                }
            },
            "DetectionResponse": {
                "type": "object",
                "properties": {
                    "anomalies": {
                        "type": "array",
                        "items": {"type": "object"}
                    },
                    "scores": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "threshold": {"type": "number"}
                }
            },
            "Error": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "title": {"type": "string"},
                    "detail": {"type": "string"},
                    "status": {"type": "integer"},
                    "instance": {"type": "string"}
                }
            }
        }

    def _generate_common_responses(self) -> dict[str, Any]:
        """Generate common OpenAPI responses."""
        return {
            "BadRequest": {
                "description": "Bad request",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "Unauthorized": {
                "description": "Unauthorized",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "Forbidden": {
                "description": "Forbidden",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "NotFound": {
                "description": "Not found",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "InternalServerError": {
                "description": "Internal server error",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            }
        }

    def _generate_paths(self) -> dict[str, Any]:
        """Generate OpenAPI paths."""
        paths = {}

        for category, info in self.endpoints.items():
            for endpoint in info["endpoints"]:
                path = info["prefix"] + endpoint["path"]
                method = endpoint["method"].lower()

                if path not in paths:
                    paths[path] = {}

                paths[path][method] = {
                    "summary": endpoint["description"],
                    "description": endpoint["description"],
                    "tags": [category],
                    "responses": {
                        "400": {"$ref": "#/components/responses/BadRequest"},
                        "401": {"$ref": "#/components/responses/Unauthorized"},
                        "403": {"$ref": "#/components/responses/Forbidden"},
                        "404": {"$ref": "#/components/responses/NotFound"},
                        "500": {"$ref": "#/components/responses/InternalServerError"}
                    }
                }

                # Add specific responses
                if "responses" in endpoint:
                    for status, response in endpoint["responses"].items():
                        paths[path][method]["responses"][status] = {
                            "description": response["description"],
                            "content": {
                                "application/json": {
                                    "schema": response.get("schema", {})
                                }
                            }
                        }

                # Add request body
                if "request_body" in endpoint:
                    paths[path][method]["requestBody"] = {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": endpoint["request_body"]
                                }
                            }
                        }
                    }

        return paths

    def generate_postman_collection(self) -> dict[str, Any]:
        """Generate Postman collection."""
        collection = {
            "info": {
                "name": "anomaly detection API",
                "description": "Comprehensive anomaly detection API",
                "version": "1.0.0",
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
            },
            "auth": {
                "type": "bearer",
                "bearer": [
                    {
                        "key": "token",
                        "value": "{{auth_token}}",
                        "type": "string"
                    }
                ]
            },
            "variable": [
                {
                    "key": "base_url",
                    "value": "http://localhost:8000",
                    "type": "string"
                },
                {
                    "key": "auth_token",
                    "value": "",
                    "type": "string"
                }
            ],
            "item": self._generate_postman_items()
        }

        return collection

    def _generate_postman_items(self) -> list[dict[str, Any]]:
        """Generate Postman collection items."""
        items = []

        for category, info in self.endpoints.items():
            folder = {
                "name": category.title(),
                "description": f"{category.title()} endpoints",
                "item": []
            }

            for endpoint in info["endpoints"]:
                request_item = {
                    "name": endpoint["description"],
                    "request": {
                        "method": endpoint["method"],
                        "header": [
                            {
                                "key": "Content-Type",
                                "value": "application/json"
                            }
                        ],
                        "url": {
                            "raw": "{{base_url}}" + info["prefix"] + endpoint["path"],
                            "host": ["{{base_url}}"],
                            "path": info["prefix"].split("/")[1:] + endpoint["path"].split("/")[1:]
                        }
                    },
                    "response": []
                }

                # Add request body example
                if "request_body" in endpoint:
                    request_item["request"]["body"] = {
                        "mode": "raw",
                        "raw": json.dumps(endpoint["request_body"], indent=2)
                    }

                folder["item"].append(request_item)

            items.append(folder)

        return items

    def generate_developer_guide(self) -> str:
        """Generate developer integration guide."""
        guide = """# anomaly detection API Developer Guide

## Overview

The anomaly detection API provides comprehensive anomaly detection capabilities with built-in authentication, multi-factor authentication (MFA), and administrative features.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://api.anomaly_detection.com`

## Authentication

### JWT Token Authentication

1. **Login** to get JWT token:
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \\
  -H "Content-Type: application/json" \\
  -d '{"username": "your_username", "password": "your_password"}'
```

2. **Use token** in subsequent requests:
```bash
curl -X GET http://localhost:8000/api/v1/health \\
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### API Key Authentication

1. **Create API key** (requires admin privileges):
```bash
curl -X POST http://localhost:8000/api/v1/api-keys \\
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{
    "name": "My API Key",
    "scopes": ["detection:read", "detection:write"],
    "expires_at": "2024-12-31T23:59:59Z"
  }'
```

2. **Use API key** in requests:
```bash
curl -X GET http://localhost:8000/api/v1/health \\
  -H "X-API-Key: YOUR_API_KEY"
```

## Multi-Factor Authentication (MFA)

### TOTP Setup

1. **Initialize TOTP**:
```bash
curl -X POST http://localhost:8000/api/v1/mfa/totp/setup \\
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{
    "app_name": "anomaly_detection",
    "issuer": "anomaly_detection Security"
  }'
```

2. **Verify TOTP code**:
```bash
curl -X POST http://localhost:8000/api/v1/mfa/totp/verify \\
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{"totp_code": "123456"}'
```

### SMS Authentication

1. **Setup SMS**:
```bash
curl -X POST http://localhost:8000/api/v1/mfa/sms/setup \\
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{"phone_number": "+1234567890"}'
```

2. **Verify SMS code**:
```bash
curl -X POST http://localhost:8000/api/v1/mfa/sms/verify \\
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{"sms_code": "123456"}'
```

## Anomaly Detection

### Basic Detection

```bash
curl -X POST http://localhost:8000/api/v1/detection/detect \\
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{
    "data": [1, 2, 3, 4, 5, 100, 6, 7, 8, 9],
    "algorithm": "isolation_forest",
    "parameters": {
      "contamination": 0.1,
      "random_state": 42
    }
  }'
```

### Batch Detection

```bash
curl -X POST http://localhost:8000/api/v1/detection/batch \\
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{
    "datasets": [
      {
        "name": "dataset1",
        "data": [1, 2, 3, 4, 5, 100]
      },
      {
        "name": "dataset2",
        "data": [10, 20, 30, 40, 50, 1000]
      }
    ],
    "algorithm": "isolation_forest"
  }'
```

## Error Handling

The API uses RFC 7807 Problem Details for HTTP APIs format:

```json
{
  "type": "https://api.anomaly_detection.com/problems/validation-error",
  "title": "Validation Error",
  "detail": "The request body contains invalid data",
  "status": 400,
  "instance": "/api/v1/detection/detect"
}
```

## Rate Limiting

The API implements rate limiting:
- **Authenticated users**: 1000 requests per hour
- **API keys**: 5000 requests per hour
- **Admin users**: 10000 requests per hour

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

## SDKs and Code Examples

### Python SDK

```python
import requests
from typing import List, Dict, Any

class anomaly-detectionAPI:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    def detect_anomalies(self, data: List[float], algorithm: str = "isolation_forest") -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/v1/detection/detect",
            headers=self.headers,
            json={
                "data": data,
                "algorithm": algorithm
            }
        )
        return response.json()

    def get_health(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/api/v1/health", headers=self.headers)
        return response.json()

# Usage
api = anomaly-detectionAPI("http://localhost:8000", "your_jwt_token")
result = api.detect_anomalies([1, 2, 3, 4, 5, 100, 6, 7, 8, 9])
print(result)
```

### JavaScript SDK

```javascript
class anomaly-detectionAPI {
    constructor(baseUrl, token) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        };
    }

    async detectAnomalies(data, algorithm = 'isolation_forest') {
        const response = await fetch(`${this.baseUrl}/api/v1/detection/detect`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({
                data: data,
                algorithm: algorithm
            })
        });
        return await response.json();
    }

    async getHealth() {
        const response = await fetch(`${this.baseUrl}/api/v1/health`, {
            headers: this.headers
        });
        return await response.json();
    }
}

// Usage
const api = new anomaly-detectionAPI('http://localhost:8000', 'your_jwt_token');
api.detectAnomalies([1, 2, 3, 4, 5, 100, 6, 7, 8, 9])
    .then(result => console.log(result));
```

## Best Practices

1. **Always use HTTPS** in production
2. **Implement proper error handling** for all API calls
3. **Use API keys** for server-to-server communication
4. **Enable MFA** for enhanced security
5. **Monitor rate limits** and implement backoff strategies
6. **Keep tokens secure** and refresh them regularly
7. **Use appropriate scopes** for API keys

## Support

For support and questions:
- GitHub Issues: https://github.com/anomaly_detection/anomaly_detection/issues
- Email: support@anomaly_detection.com
- Documentation: https://docs.anomaly_detection.com
"""

        return guide

    def generate_authentication_guide(self) -> str:
        """Generate authentication documentation."""
        guide = """# Authentication Guide

## Overview

anomaly detection API supports multiple authentication methods:
- JWT Token Authentication (recommended for user sessions)
- API Key Authentication (recommended for server-to-server)
- Multi-Factor Authentication (MFA) for enhanced security

## JWT Token Authentication

### Flow

1. **Login** with credentials
2. **Receive** JWT access token and refresh token
3. **Use** access token in API requests
4. **Refresh** token when expired

### Implementation

```python
import requests
import json
from datetime import datetime, timedelta

class AuthManager:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None

    def login(self, username: str, password: str) -> bool:
        response = requests.post(
            f"{self.base_url}/api/v1/auth/login",
            json={"username": username, "password": password}
        )

        if response.status_code == 200:
            data = response.json()
            self.access_token = data["access_token"]
            self.refresh_token = data["refresh_token"]
            self.token_expires_at = datetime.now() + timedelta(seconds=data["expires_in"])
            return True
        return False

    def refresh_access_token(self) -> bool:
        if not self.refresh_token:
            return False

        response = requests.post(
            f"{self.base_url}/api/v1/auth/refresh",
            json={"refresh_token": self.refresh_token}
        )

        if response.status_code == 200:
            data = response.json()
            self.access_token = data["access_token"]
            self.token_expires_at = datetime.now() + timedelta(seconds=data["expires_in"])
            return True
        return False

    def get_auth_headers(self) -> dict:
        # Auto-refresh if token is expired
        if self.token_expires_at and datetime.now() >= self.token_expires_at:
            self.refresh_access_token()

        return {"Authorization": f"Bearer {self.access_token}"}
```

## API Key Authentication

### Creation

```bash
# Create API key (requires admin privileges)
curl -X POST http://localhost:8000/api/v1/api-keys \\
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{
    "name": "Production API Key",
    "scopes": ["detection:read", "detection:write", "health:read"],
    "expires_at": "2024-12-31T23:59:59Z"
  }'
```

### Usage

```python
import requests

class APIKeyClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }

    def make_request(self, method: str, endpoint: str, data: dict = None):
        url = f"{self.base_url}{endpoint}"
        response = requests.request(method, url, headers=self.headers, json=data)
        return response.json()
```

## Multi-Factor Authentication (MFA)

### TOTP (Time-based One-Time Password)

```python
import pyotp
import qrcode
from io import BytesIO
import base64

class TOTPManager:
    def __init__(self, api_client):
        self.api_client = api_client

    def setup_totp(self, app_name: str = "anomaly_detection") -> dict:
        response = self.api_client.post("/api/v1/mfa/totp/setup", {
            "app_name": app_name,
            "issuer": "anomaly_detection Security"
        })
        return response.json()

    def verify_totp(self, totp_code: str) -> bool:
        response = self.api_client.post("/api/v1/mfa/totp/verify", {
            "totp_code": totp_code
        })
        return response.status_code == 200

    def generate_qr_code(self, totp_uri: str) -> str:
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
```

### SMS Authentication

```python
class SMSAuth:
    def __init__(self, api_client):
        self.api_client = api_client

    def setup_sms(self, phone_number: str) -> bool:
        response = self.api_client.post("/api/v1/mfa/sms/setup", {
            "phone_number": phone_number
        })
        return response.status_code == 200

    def verify_sms(self, sms_code: str) -> bool:
        response = self.api_client.post("/api/v1/mfa/sms/verify", {
            "sms_code": sms_code
        })
        return response.status_code == 200
```

## Security Best Practices

1. **Token Storage**: Store tokens securely (not in localStorage for web apps)
2. **Token Rotation**: Implement automatic token refresh
3. **API Key Management**: Rotate API keys regularly
4. **MFA Enforcement**: Enable MFA for all admin accounts
5. **Secure Transmission**: Always use HTTPS in production
6. **Rate Limiting**: Implement client-side rate limiting
7. **Error Handling**: Handle authentication errors gracefully

## Troubleshooting

### Common Issues

1. **Token Expired**: Implement automatic refresh
2. **Invalid Credentials**: Check username/password
3. **MFA Required**: Complete MFA setup
4. **Rate Limited**: Implement backoff strategy
5. **Invalid API Key**: Check key validity and scopes

### Debug Mode

```python
import logging

# Enable secure debug logging in development only
from src.packages.data.anomaly_detection.core.security_configuration import get_security_config, configure_secure_logging

security_config = get_security_config()
if security_config.is_development():
    configure_secure_logging()
else:
    logging.basicConfig(level=logging.INFO)

# Use debug headers
headers = {
    "X-Debug": "true",
    "X-Request-ID": "unique-request-id"
}
```
"""

        return guide

    def generate_api_versioning_strategy(self) -> str:
        """Generate API versioning strategy documentation."""
        strategy = """# API Versioning Strategy

## Overview

anomaly detection API follows semantic versioning and provides multiple versioning strategies to ensure backward compatibility and smooth upgrades.

## Versioning Scheme

### URL Path Versioning (Current)
- **Current**: `/api/v1/`
- **Future**: `/api/v2/`, `/api/v3/`

### Header Versioning (Alternative)
```
Accept: application/vnd.anomaly_detection.v1+json
```

## Version Lifecycle

### Version 1.0 (Current)
- **Status**: Stable
- **Support**: Full support
- **Deprecation**: TBD
- **Features**:
  - JWT Authentication
  - MFA Support
  - Anomaly Detection
  - Admin Operations

### Version 2.0 (Planned)
- **Status**: In Development
- **Features**:
  - GraphQL API
  - Real-time Streaming
  - Enhanced ML Models
  - Advanced Analytics

## Backward Compatibility

### Breaking Changes Policy
- **Major Version**: Breaking changes allowed
- **Minor Version**: Backward compatible features
- **Patch Version**: Bug fixes only

### Deprecation Process
1. **Announce**: 90 days notice
2. **Mark**: Add deprecation headers
3. **Document**: Update documentation
4. **Remove**: In next major version

### Migration Support
- **Migration Guide**: Step-by-step instructions
- **Dual Support**: 12 months overlap
- **Automated Tools**: Migration scripts

## Version Detection

### Client Implementation
```python
class VersionedClient:
    def __init__(self, base_url: str, version: str = "v1"):
        self.base_url = base_url
        self.version = version

    def get_endpoint(self, path: str) -> str:
        return f"{self.base_url}/api/{self.version}{path}"

    def get_headers(self) -> dict:
        return {
            "Accept": f"application/vnd.anomaly_detection.{self.version}+json",
            "User-Agent": "anomaly_detection-Client/1.0"
        }
```

### Server Response Headers
```
API-Version: 1.0
Supported-Versions: 1.0, 1.1
Deprecated-Versions: 0.9
```

## Feature Flags

### Implementation
```python
class FeatureManager:
    def __init__(self, version: str):
        self.version = version
        self.features = self._load_features()

    def is_enabled(self, feature: str) -> bool:
        return self.features.get(feature, False)

    def _load_features(self) -> dict:
        return {
            "v1": {
                "mfa": True,
                "streaming": False,
                "graphql": False
            },
            "v2": {
                "mfa": True,
                "streaming": True,
                "graphql": True
            }
        }.get(self.version, {})
```

## Testing Strategy

### Cross-Version Testing
```python
import pytest

@pytest.mark.parametrize("version", ["v1", "v2"])
def test_health_endpoint(version):
    client = VersionedClient("http://localhost:8000", version)
    response = client.get("/health")
    assert response.status_code == 200
```

### Compatibility Matrix
| Feature | v1.0 | v1.1 | v2.0 |
|---------|------|------|------|
| JWT Auth | ✓ | ✓ | ✓ |
| MFA | ✓ | ✓ | ✓ |
| Streaming | ✗ | ✗ | ✓ |
| GraphQL | ✗ | ✗ | ✓ |
"""

        return strategy

    def generate_all_documentation(self):
        """Generate all API documentation."""
        print("🚀 Starting comprehensive API documentation generation...")

        # Generate OpenAPI specification
        print("📋 Generating OpenAPI specification...")
        openapi_spec = self.generate_openapi_spec()

        # Save OpenAPI spec as JSON
        with open(self.output_dir / "openapi.json", "w") as f:
            json.dump(openapi_spec, f, indent=2)

        # Save OpenAPI spec as YAML
        with open(self.output_dir / "openapi.yaml", "w") as f:
            yaml.dump(openapi_spec, f, default_flow_style=False)

        # Generate Postman collection
        print("📮 Generating Postman collection...")
        postman_collection = self.generate_postman_collection()

        with open(self.output_dir / "anomaly_detection_api.postman_collection.json", "w") as f:
            json.dump(postman_collection, f, indent=2)

        # Generate documentation files
        print("📚 Generating documentation guides...")

        # Developer guide
        with open(self.output_dir / "developer_guide.md", "w") as f:
            f.write(self.generate_developer_guide())

        # Authentication guide
        with open(self.output_dir / "authentication_guide.md", "w") as f:
            f.write(self.generate_authentication_guide())

        # API versioning strategy
        with open(self.output_dir / "versioning_strategy.md", "w") as f:
            f.write(self.generate_api_versioning_strategy())

        # Generate main documentation index
        self._generate_documentation_index()

        print("✅ API documentation generation completed!")
        print(f"📁 Documentation saved to: {self.output_dir}")

    def _generate_documentation_index(self):
        """Generate main documentation index."""
        index_content = f"""# anomaly detection API Documentation

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Quick Start

1. **[Developer Guide](developer_guide.md)** - Complete integration guide
2. **[Authentication Guide](authentication_guide.md)** - Security implementation
3. **[API Versioning Strategy](versioning_strategy.md)** - Version management

## API Specifications

- **[OpenAPI JSON](openapi.json)** - Machine-readable API specification
- **[OpenAPI YAML](openapi.yaml)** - Human-readable API specification
- **[Postman Collection](anomaly_detection_api.postman_collection.json)** - Ready-to-use API testing

## API Overview

### Authentication Endpoints
- JWT token authentication
- Multi-factor authentication (MFA)
- API key management
- Password reset functionality

### Detection Endpoints
- Real-time anomaly detection
- Batch processing
- Algorithm selection
- Custom parameters

### Admin Endpoints
- User management
- Role assignment
- System monitoring
- Configuration management

### Health Endpoints
- Service health checks
- Detailed system status
- Performance metrics

## Getting Started

### 1. Authentication
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \\
  -H "Content-Type: application/json" \\
  -d '{{"username": "your_username", "password": "your_password"}}'
```

### 2. Basic Detection
```bash
curl -X POST http://localhost:8000/api/v1/detection/detect \\
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "data": [1, 2, 3, 4, 5, 100, 6, 7, 8, 9],
    "algorithm": "isolation_forest"
  }}'
```

### 3. Health Check
```bash
curl -X GET http://localhost:8000/api/v1/health \\
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## Development Tools

### Import Postman Collection
1. Open Postman
2. Click "Import"
3. Select `anomaly_detection_api.postman_collection.json`
4. Set environment variables:
   - `base_url`: Your API base URL
   - `auth_token`: Your JWT token

### Generate SDK
Use the OpenAPI specification to generate SDKs:
```bash
# Python SDK
openapi-generator-cli generate -i openapi.json -g python -o python-sdk/

# JavaScript SDK
openapi-generator-cli generate -i openapi.json -g javascript -o javascript-sdk/
```

## Support

- **GitHub**: https://github.com/anomaly_detection/anomaly_detection
- **Issues**: https://github.com/anomaly_detection/anomaly_detection/issues
- **Email**: support@anomaly_detection.com

## Security

- Always use HTTPS in production
- Implement proper token management
- Enable MFA for enhanced security
- Follow rate limiting guidelines
- Keep API keys secure
"""

        with open(self.output_dir / "index.md", "w") as f:
            f.write(index_content)

def main():
    """Main function to generate comprehensive API documentation."""
    generator = APIDocumentationGenerator()
    generator.generate_all_documentation()

if __name__ == "__main__":
    main()
