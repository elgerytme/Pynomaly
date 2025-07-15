"""FastAPI middleware integration for configuration capture.

This module provides utilities for integrating configuration capture middleware
with FastAPI applications.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI

from pynomaly.application.services.configuration_capture_service import (
    ConfigurationCaptureService,
)
from pynomaly.infrastructure.config.feature_flags import feature_flags
from pynomaly.infrastructure.middleware.configuration_middleware import (
    ConfigurationAPIMiddleware,
    ConfigurationCaptureMiddleware,
)

logger = logging.getLogger(__name__)


def setup_configuration_middleware(
    app: FastAPI,
    configuration_service: ConfigurationCaptureService,
    environment: str = "development",
) -> None:
    """Setup configuration capture middleware for FastAPI app.

    Args:
        app: FastAPI application instance
        configuration_service: Configuration capture service
        environment: Application environment (development/production)
    """
    if not feature_flags.is_enabled("advanced_automl"):
        logger.info(
            "Configuration middleware disabled - advanced_automl feature not enabled"
        )
        return

    try:
        if environment == "production":
            # Production middleware with security considerations
            middleware_factory = (
                ConfigurationAPIMiddleware.create_production_middleware(
                    configuration_service
                )
            )
        else:
            # Development middleware with verbose capture
            middleware_factory = (
                ConfigurationAPIMiddleware.create_development_middleware(
                    configuration_service
                )
            )

        # Add middleware to app
        app.add_middleware(
            ConfigurationCaptureMiddleware,
            configuration_service=configuration_service,
            **(
                middleware_factory.__dict__
                if hasattr(middleware_factory, "__dict__")
                else {}
            ),
        )

        logger.info(
            f"Configuration capture middleware enabled for {environment} environment"
        )

    except Exception as e:
        logger.error(f"Failed to setup configuration middleware: {e}")
        # Don't fail app startup due to middleware issues
        pass


def add_configuration_endpoints(
    app: FastAPI,
    configuration_service: ConfigurationCaptureService,
    integration_service: object | None = None,
) -> None:
    """Add configuration management endpoints to FastAPI app.

    Args:
        app: FastAPI application instance
        configuration_service: Configuration capture service
        integration_service: Web API configuration integration service
    """

    from fastapi import HTTPException

    @app.get("/api/v1/configurations/middleware/stats")
    async def get_middleware_stats() -> dict[str, Any]:
        """Get middleware capture statistics."""
        try:
            # This would need to be accessed from the middleware instance
            # For now, return placeholder
            return {
                "message": "Middleware statistics endpoint",
                "note": "Statistics would be retrieved from middleware instance",
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/configurations/api-usage")
    async def get_api_usage_analysis(days: int = 7) -> dict[str, Any]:
        """Get API usage analysis."""
        if not integration_service:
            raise HTTPException(
                status_code=404, detail="Integration service not available"
            )

        try:
            analysis = await integration_service.analyze_api_usage_patterns(days)
            return analysis
        except Exception as e:
            logger.error(f"API usage analysis failed: {e}")
            raise HTTPException(status_code=500, detail="Analysis failed")

    @app.get("/api/v1/configurations/endpoints/{endpoint:path}/performance")
    async def get_endpoint_performance(endpoint: str, days: int = 7) -> dict[str, Any]:
        """Get performance metrics for specific endpoint."""
        if not integration_service:
            raise HTTPException(
                status_code=404, detail="Integration service not available"
            )

        try:
            # URL decode the endpoint
            import urllib.parse

            decoded_endpoint = urllib.parse.unquote(endpoint)

            metrics = await integration_service.get_endpoint_performance_metrics(
                decoded_endpoint, days
            )
            return metrics
        except Exception as e:
            logger.error(f"Endpoint performance analysis failed: {e}")
            raise HTTPException(status_code=500, detail="Analysis failed")

    @app.get("/api/v1/configurations/api-report")
    async def get_api_configuration_report(days: int = 30) -> dict[str, Any]:
        """Get comprehensive API configuration report."""
        if not integration_service:
            raise HTTPException(
                status_code=404, detail="Integration service not available"
            )

        try:
            report = await integration_service.generate_api_configuration_report(days)
            return report
        except Exception as e:
            logger.error(f"API report generation failed: {e}")
            raise HTTPException(status_code=500, detail="Report generation failed")

    logger.info("Configuration API endpoints added")


def setup_configuration_logging(app: FastAPI) -> None:
    """Setup configuration-specific logging for the API.

    Args:
        app: FastAPI application instance
    """

    @app.middleware("http")
    async def log_configuration_requests(request, call_next):
        """Log configuration-related requests."""
        response = await call_next(request)

        # Log configuration API calls
        if "/configurations" in str(request.url.path):
            logger.info(
                f"Configuration API: {request.method} {request.url.path} "
                f"-> {response.status_code}"
            )

        return response


# Example usage in main API application
def create_configured_app(
    configuration_service: ConfigurationCaptureService, environment: str = "development"
) -> FastAPI:
    """Create FastAPI app with configuration middleware enabled.

    Args:
        configuration_service: Configuration capture service
        environment: Application environment

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Pynomaly API",
        description="State-of-the-art anomaly detection API with configuration capture",
        version="1.0.0",
    )

    # Setup configuration middleware
    setup_configuration_middleware(app, configuration_service, environment)

    # Setup configuration logging
    setup_configuration_logging(app)

    # Add configuration endpoints
    try:
        from pynomaly.application.services.web_api_configuration_integration import (
            WebAPIConfigurationIntegration,
        )

        integration_service = WebAPIConfigurationIntegration(configuration_service)
        add_configuration_endpoints(app, configuration_service, integration_service)
    except ImportError:
        logger.warning("Web API configuration integration not available")
        add_configuration_endpoints(app, configuration_service, None)

    return app


# Health check with configuration status
def add_configuration_health_check(app: FastAPI) -> None:
    """Add health check endpoint that includes configuration status.

    Args:
        app: FastAPI application instance
    """

    @app.get("/health/configuration")
    async def configuration_health_check() -> dict[str, Any]:
        """Health check for configuration system."""
        try:
            # Check if configuration features are enabled
            config_enabled = feature_flags.is_enabled("advanced_automl")

            health_status = {
                "status": "healthy" if config_enabled else "disabled",
                "configuration_capture": config_enabled,
                "timestamp": "datetime.now().isoformat()",
                "features": {
                    "middleware": config_enabled,
                    "api_endpoints": config_enabled,
                    "pattern_analysis": config_enabled,
                },
            }

            return health_status

        except Exception as e:
            logger.error(f"Configuration health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": "datetime.now().isoformat()",
            }


def setup_middleware_stack(app: FastAPI) -> None:
    """Setup complete middleware stack for the FastAPI application.

    Args:
        app: FastAPI application instance
    """
    from pynomaly.presentation.api.middleware.security_headers import (
        SecurityHeadersMiddleware,
    )

    # Add security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)


def configure_cors(app: FastAPI, allow_origins: list = None) -> None:
    """Configure CORS middleware for the FastAPI application.

    Args:
        app: FastAPI application instance
        allow_origins: List of allowed origins, defaults to ["*"]
    """
    from fastapi.middleware.cors import CORSMiddleware

    if allow_origins is None:
        allow_origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def configure_rate_limiting(app: FastAPI, requests_per_minute: int = 60) -> None:
    """Configure rate limiting middleware for the FastAPI application.

    Args:
        app: FastAPI application instance
        requests_per_minute: Maximum requests per minute per client
    """
    # Rate limiting middleware would be implemented here
    # For now, this is a placeholder
    pass
