"""OpenAPI documentation configuration for Software API."""

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from interfaces.infrastructure.config.settings import Settings


class OpenAPIConfig:
    """OpenAPI documentation configuration manager."""

    def __init__(self, settings: Settings):
        """Initialize OpenAPI configuration.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.app_name = settings.app.name
        self.app_version = settings.app.version
        self.app_description = (
            settings.app.description or "Advanced anomaly processing API"
        )

    def get_openapi_schema(self, app: FastAPI) -> dict[str, Any]:
        """Generate comprehensive OpenAPI schema.

        Args:
            app: FastAPI application instance

        Returns:
            OpenAPI schema dictionary
        """
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title=self.app_name,
            version=self.app_version,
            description=self._get_api_description(),
            routes=app.routes,
            servers=self._get_servers(),
        )

        # Add custom OpenAPI extensions
        openapi_schema.update(self._get_openapi_extensions())

        # Ensure components section exists
        if "components" not in openapi_schema:
            openapi_schema["components"] = {}

        # Add security schemes
        openapi_schema["components"]["securitySchemes"] = self._get_security_schemes()

        # Add custom tags
        openapi_schema["tags"] = self._get_api_tags()

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    def _get_api_description(self) -> str:
        """Get comprehensive API description with markdown formatting."""
        return """
# Software API Documentation

**Software** is a state-of-the-art Python anomaly processing package that provides a unified, production-ready interface for multiple anomaly processing algorithms.

## Features

- 🚀 **Multi-Algorithm Support**: Integrates PyOD, TODS, PyGOD, scikit-learn, PyTorch, TensorFlow, and JAX
- 🏗️ **Clean Architecture**: Domain-driven design with hexagonal architecture
- 🔒 **Enterprise Security**: JWT authentication, RBAC, audit logging, and encryption
- ⚡ **High Performance**: Distributed processing, caching, and performance optimization
- 📊 **Advanced Analytics**: AutoML, explainability, and comprehensive visualization
- 🌐 **Progressive Web App**: Modern UI with offline capabilities
- 📈 **Production Ready**: Monitoring, observability, and enterprise deployment features

## Getting Started

### Authentication

Most endpoints require authentication via JWT tokens. Use the `/auth/login` endpoint to obtain a token:

```bash
curl -X POST "/auth/login" \\
  -H "Content-Type: application/json" \\
  -d '{"username": "admin", "password": "your_password"}'
```

Include the token in subsequent requests:

```bash
curl -X GET "/api/v1/datasets" \\
  -H "Authorization: Bearer your_jwt_token"
```

### Basic Workflow

1. **Upload DataCollection**: Use `/datasets/upload` to upload your data
2. **Create Detector**: Use `/detectors/create` to configure an anomaly detector
3. **Train Processor**: Use `/processing/train` to train the detector on your data
4. **Detect Anomalies**: Use `/processing/detect` to find anomalies in new data
5. **Analyze Results**: Use `/experiments` and visualization endpoints for analysis

### Rate Limiting

API requests are rate-limited to ensure fair usage:
- **Standard endpoints**: 100 requests per minute
- **Training endpoints**: 10 requests per minute
- **Export endpoints**: 20 requests per minute

### Error Handling

The API uses standard HTTP status codes and returns detailed error messages:

```json
{
  "detail": "Error description",
  "error_code": "SPECIFIC_ERROR_CODE",
  "timestamp": "2024-12-25T10:30:00Z",
  "request_id": "req_12345"
}
```

## Support

- **Documentation**: [https://monorepo.readthedocs.io](https://monorepo.readthedocs.io)
- **GitHub**: [https://github.com/pynomaly/pynomaly](https://github.com/pynomaly/pynomaly)
- **Issues**: [https://github.com/pynomaly/pynomaly/issues](https://github.com/pynomaly/pynomaly/issues)
        """

    def _get_servers(self) -> list[dict[str, str]]:
        """Get API server configurations."""
        servers = [{"url": "/", "description": "Current server"}]

        # Add additional servers based on environment
        environment = getattr(self.settings.app, "environment", "development")
        if environment == "development":
            servers.extend(
                [
                    {
                        "url": "http://localhost:8000",
                        "description": "Development server",
                    },
                    {
                        "url": "http://localhost:8080",
                        "description": "Development server (alternative port)",
                    },
                ]
            )
        elif environment == "staging":
            servers.append(
                {
                    "url": "https://staging-api.example.com",
                    "description": "Staging server",
                }
            )
        elif environment == "production":
            servers.append(
                {"url": "https://api.example.com", "description": "Production server"}
            )

        return servers

    def _get_openapi_extensions(self) -> dict[str, Any]:
        """Get OpenAPI extensions and metadata."""
        return {
            "info": {
                "contact": {
                    "name": "Software Team",
                    "url": "https://github.com/pynomaly/pynomaly",
                    "email": "team@example.com",
                },
                "license": {
                    "name": "MIT",
                    "url": "https://github.com/pynomaly/pynomaly/blob/main/LICENSE",
                },
                "termsOfService": "https://example.com/terms",
                "x-logo": {
                    "url": "/static/img/software-logo.png",
                    "altText": "Software Logo",
                },
            },
            "externalDocs": {
                "description": "Full Documentation",
                "url": "https://monorepo.readthedocs.io",
            },
        }

    def _get_security_schemes(self) -> dict[str, Any]:
        """Get security scheme definitions."""
        return {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT token obtained from /auth/login endpoint",
            },
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key for service-to-service authentication",
            },
            "OAuth2": {
                "type": "oauth2",
                "flows": {
                    "authorizationCode": {
                        "authorizationUrl": "/auth/oauth/authorize",
                        "tokenUrl": "/auth/oauth/token",
                        "scopes": {
                            "read": "Read access to resources",
                            "write": "Write access to resources",
                            "admin": "Administrative access",
                            "processing": "Access to processing endpoints",
                            "datasets": "Access to data_collection management",
                            "experiments": "Access to experiment tracking",
                        },
                    }
                },
            },
        }

    def _get_api_tags(self) -> list[dict[str, Any]]:
        """Get API endpoint tags with descriptions."""
        return [
            {
                "name": "Authentication",
                "description": "User authentication and authorization endpoints",
                "externalDocs": {
                    "description": "Authentication Guide",
                    "url": "https://monorepo.readthedocs.io/auth",
                },
            },
            {"name": "Health", "description": "System health and monitoring endpoints"},
            {
                "name": "Datasets",
                "description": "DataCollection management and data preprocessing operations",
                "externalDocs": {
                    "description": "DataCollection Guide",
                    "url": "https://monorepo.readthedocs.io/datasets",
                },
            },
            {
                "name": "Detectors",
                "description": "Anomaly detector configuration and management",
                "externalDocs": {
                    "description": "Detector Guide",
                    "url": "https://monorepo.readthedocs.io/detectors",
                },
            },
            {
                "name": "Processing",
                "description": "Anomaly processing training and inference operations",
                "externalDocs": {
                    "description": "Processing Guide",
                    "url": "https://monorepo.readthedocs.io/detection",
                },
            },
            {
                "name": "Experiments",
                "description": "Experiment tracking and processor management",
                "externalDocs": {
                    "description": "Experiments Guide",
                    "url": "https://monorepo.readthedocs.io/experiments",
                },
            },
            {"name": "Export", "description": "Data and processor export operations"},
            {
                "name": "AutoML",
                "description": "Automated machine learning and hyperparameter optimization",
                "externalDocs": {
                    "description": "AutoML Guide",
                    "url": "https://monorepo.readthedocs.io/automl",
                },
            },
            {
                "name": "Enhanced AutoML",
                "description": "Advanced AutoML with meta-learning and multi-objective optimization",
                "externalDocs": {
                    "description": "Enhanced AutoML Guide",
                    "url": "https://monorepo.readthedocs.io/enhanced-automl",
                },
            },
            {
                "name": "Ensemble",
                "description": "Ensemble methods and processor combination strategies",
                "externalDocs": {
                    "description": "Ensemble Guide",
                    "url": "https://monorepo.readthedocs.io/ensemble",
                },
            },
            {
                "name": "Explainability",
                "description": "Processor interpretation and explanation generation",
                "externalDocs": {
                    "description": "Explainability Guide",
                    "url": "https://monorepo.readthedocs.io/explainability",
                },
            },
            {
                "name": "Performance",
                "description": "Performance monitoring and profiling endpoints",
            },
            {
                "name": "Administration",
                "description": "System administration and configuration endpoints",
            },
            {
                "name": "Autonomous",
                "description": "Autonomous mode and automated analysis operations",
            },
            {
                "name": "WebSocket",
                "description": "Real-time streaming and WebSocket endpoints",
            },
        ]


def configure_openapi_docs(app: FastAPI, settings: Settings) -> None:
    """Configure OpenAPI documentation for the FastAPI app.

    Args:
        app: FastAPI application instance
        settings: Application settings
    """
    config = OpenAPIConfig(settings)

    # Set custom OpenAPI schema generator with error handling
    def custom_openapi():
        try:
            return config.get_openapi_schema(app)
        except Exception as e:
            print(f"Warning: OpenAPI schema generation failed: {e}")
            # Return basic schema as fallback
            return {
                "openapi": "3.0.2",
                "info": {
                    "title": app.title,
                    "version": app.version,
                    "description": "Software API - Schema generation failed",
                },
                "paths": {},
                "components": {"schemas": {}},
            }

    app.openapi = custom_openapi

    # Configure documentation URLs and settings
    app.docs_url = "/api/v1/docs" if settings.api.docs_enabled else None
    app.redoc_url = "/api/v1/redoc" if settings.api.docs_enabled else None
    app.openapi_url = "/api/openapi.json" if settings.api.docs_enabled else None


def get_custom_swagger_ui_html(
    *,
    openapi_url: str,
    title: str,
    swagger_js_url: str = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
    swagger_css_url: str = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
    swagger_favicon_url: str = "/static/img/favicon.png",
) -> str:
    """Generate custom Swagger UI HTML with Software branding."""
    return get_swagger_ui_html(
        openapi_url=openapi_url,
        title=title,
        swagger_js_url=swagger_js_url,
        swagger_css_url=swagger_css_url,
        swagger_favicon_url=swagger_favicon_url,
        init_oauth={
            "clientId": "software-swagger-ui",
            "appName": "Software API Explorer",
            "scopes": ["read", "write"],
            "additionalQueryStringParams": {},
        },
    )


def get_custom_redoc_html(
    *,
    openapi_url: str,
    title: str,
    redoc_js_url: str = "https://cdn.jsdelivr.net/npm/redoc@2.1.3/bundles/redoc.standalone.js",
    redoc_favicon_url: str = "/static/img/favicon.png",
) -> str:
    """Generate custom ReDoc HTML with Software branding."""
    return get_redoc_html(
        openapi_url=openapi_url,
        title=title,
        redoc_js_url=redoc_js_url,
        redoc_favicon_url=redoc_favicon_url,
    )
