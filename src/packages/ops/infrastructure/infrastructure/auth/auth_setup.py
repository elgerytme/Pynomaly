"""Authentication and security setup utilities."""

import logging
from typing import Any

from pynomaly.infrastructure.config.settings import Settings
from pynomaly.infrastructure.services.email_service import init_email_service

logger = logging.getLogger(__name__)


def setup_authentication_system(settings: Settings) -> dict[str, Any]:
    """Set up the complete authentication system.

    Args:
        settings: Application settings

    Returns:
        Dictionary with setup status and initialized services
    """
    setup_status = {
        "email_service": False,
        "database_repositories": False,
        "mfa_service": False,
        "security_headers": False,
        "audit_logging": False,
        "errors": [],
    }

    try:
        # Initialize email service
        email_service = init_email_service(settings)
        setup_status["email_service"] = email_service is not None
        if email_service:
            logger.info("Email service initialized successfully")
        else:
            logger.warning("Email service not configured - email features disabled")

        # Set up database repositories
        try:
            from pynomaly.infrastructure.persistence.repository_factory import (
                get_repository_factory,
            )

            # Initialize repository factory (this creates tables if needed)
            factory = get_repository_factory()

            # Test repository connections
            user_repo = factory.get_user_repository()
            tenant_repo = factory.get_tenant_repository()
            session_repo = factory.get_session_repository()

            setup_status["database_repositories"] = True
            logger.info("Database repositories initialized successfully")

        except Exception as e:
            setup_status["errors"].append(f"Database setup failed: {e}")
            logger.error(f"Failed to set up database repositories: {e}")

        # Set up MFA service
        try:
            from pynomaly.domain.services.mfa_service import MFAService
            from pynomaly.infrastructure.cache import get_cache

            cache = get_cache()
            mfa_service = MFAService(redis_client=cache)
            setup_status["mfa_service"] = True
            logger.info("MFA service initialized successfully")

        except Exception as e:
            setup_status["errors"].append(f"MFA setup failed: {e}")
            logger.error(f"Failed to set up MFA service: {e}")

        # Configure security headers
        if settings.security.security_headers_enabled:
            setup_status["security_headers"] = True
            logger.info("Security headers enabled")

        # Configure audit logging
        if settings.security.enable_audit_logging:
            setup_status["audit_logging"] = True
            logger.info("Audit logging enabled")

        # Log overall setup status
        successful_components = sum(
            1 for v in setup_status.values() if isinstance(v, bool) and v
        )
        total_components = len([k for k in setup_status.keys() if k != "errors"])

        logger.info(
            f"Authentication system setup completed: {successful_components}/{total_components} components initialized"
        )

        if setup_status["errors"]:
            logger.warning(
                f"Setup completed with {len(setup_status['errors'])} errors: {setup_status['errors']}"
            )

    except Exception as e:
        setup_status["errors"].append(f"Critical setup failure: {e}")
        logger.error(f"Critical failure during authentication system setup: {e}")

    return setup_status


def validate_security_configuration(settings: Settings) -> dict[str, Any]:
    """Validate security configuration and provide recommendations.

    Args:
        settings: Application settings

    Returns:
        Dictionary with validation results and recommendations
    """
    validation_results = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "recommendations": [],
    }

    try:
        # Check secret key security
        if (
            settings.security.secret_key
            == "change-me-in-production-this-is-32-chars-long-default-key"
        ):
            if settings.is_production:
                validation_results["errors"].append(
                    "CRITICAL: Default secret key detected in production environment"
                )
                validation_results["valid"] = False
            else:
                validation_results["warnings"].append(
                    "Default secret key in use - change for production"
                )

        # Check JWT configuration
        if settings.security.jwt_expiration > 3600:
            validation_results["warnings"].append(
                f"JWT expiration time is high ({settings.security.jwt_expiration}s) - consider shorter duration for security"
            )

        # Check email configuration
        email_configured = all(
            [
                settings.security.smtp_server,
                settings.security.smtp_username,
                settings.security.smtp_password,
                settings.security.sender_email,
            ]
        )

        if not email_configured:
            validation_results["warnings"].append(
                "Email service not configured - password reset and user invitations will not work"
            )
            validation_results["recommendations"].append(
                "Configure SMTP settings in environment variables for full authentication features"
            )

        # Check database configuration
        if not settings.database.database_url:
            validation_results["warnings"].append(
                "Database URL not configured - using in-memory storage"
            )
            validation_results["recommendations"].append(
                "Configure DATABASE_URL for persistent data storage"
            )

        # Check production security settings
        if settings.is_production:
            if not settings.security.security_headers_enabled:
                validation_results["errors"].append(
                    "Security headers should be enabled in production"
                )
                validation_results["valid"] = False

            if not settings.security.enable_audit_logging:
                validation_results["warnings"].append(
                    "Audit logging should be enabled in production for compliance"
                )

            if not settings.security.hsts_enabled:
                validation_results["warnings"].append(
                    "HSTS should be enabled in production for HTTPS enforcement"
                )

        # Security recommendations
        if not settings.security.threat_detection_enabled:
            validation_results["recommendations"].append(
                "Enable threat detection for enhanced security monitoring"
            )

        if settings.security.brute_force_max_attempts > 10:
            validation_results["recommendations"].append(
                "Consider lowering brute force max attempts for better security"
            )

        logger.info(
            f"Security configuration validation completed with {len(validation_results['errors'])} errors and {len(validation_results['warnings'])} warnings"
        )

    except Exception as e:
        validation_results["errors"].append(f"Validation failed: {e}")
        validation_results["valid"] = False
        logger.error(f"Security configuration validation failed: {e}")

    return validation_results


def get_security_status() -> dict[str, Any]:
    """Get current security system status.

    Returns:
        Dictionary with current status of security components
    """
    status = {"timestamp": None, "components": {}, "overall_health": "unknown"}

    try:
        from datetime import datetime

        status["timestamp"] = datetime.utcnow().isoformat()

        # Check email service
        from pynomaly.infrastructure.services.email_service import get_email_service

        email_service = get_email_service()
        status["components"]["email_service"] = {
            "status": "configured" if email_service else "not_configured",
            "available": email_service is not None,
        }

        # Check database repositories
        try:
            from pynomaly.infrastructure.persistence.repository_factory import (
                get_repository_factory,
            )

            factory = get_repository_factory()
            status["components"]["database_repositories"] = {
                "status": "available",
                "available": True,
            }
        except Exception:
            status["components"]["database_repositories"] = {
                "status": "error",
                "available": False,
            }

        # Check MFA service
        try:
            from pynomaly.infrastructure.cache import get_cache

            cache = get_cache()
            status["components"]["mfa_service"] = {
                "status": "available" if cache else "limited",
                "available": True,
                "cache_available": cache is not None,
            }
        except Exception:
            status["components"]["mfa_service"] = {
                "status": "error",
                "available": False,
            }

        # Check cache service
        try:
            from pynomaly.infrastructure.cache import get_cache

            cache = get_cache()
            status["components"]["cache_service"] = {
                "status": "available" if cache and cache.enabled else "not_configured",
                "available": cache is not None and cache.enabled if cache else False,
            }
        except Exception:
            status["components"]["cache_service"] = {
                "status": "error",
                "available": False,
            }

        # Determine overall health
        available_components = sum(
            1 for comp in status["components"].values() if comp["available"]
        )
        total_components = len(status["components"])

        if available_components == total_components:
            status["overall_health"] = "healthy"
        elif available_components >= total_components * 0.75:
            status["overall_health"] = "degraded"
        else:
            status["overall_health"] = "unhealthy"

        logger.debug(
            f"Security system status: {status['overall_health']} ({available_components}/{total_components} components available)"
        )

    except Exception as e:
        status["overall_health"] = "error"
        status["error"] = str(e)
        logger.error(f"Failed to get security system status: {e}")

    return status
