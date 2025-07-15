"""
Security Integration Module

This module integrates all security hardening measures into the application.
It provides a centralized way to apply security configurations and policies.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from pynomaly.infrastructure.config.settings import Settings
from pynomaly.infrastructure.security.secure_database import get_query_auditor
from pynomaly.infrastructure.security.security_hardening import (
    SecureConfigurationManager,
    get_config_manager,
    get_input_validator,
    get_secure_serializer,
    initialize_security_hardening,
    set_config_manager,
)

logger = logging.getLogger(__name__)


class SecurityIntegrationManager:
    """
    Central manager for security integration and hardening.

    This class orchestrates all security measures and provides a unified
    interface for applying security configurations.
    """

    def __init__(self, settings: Settings):
        """Initialize security integration manager."""
        self.settings = settings
        self.config_manager: SecureConfigurationManager | None = None
        self.security_warnings: list[str] = []
        self.security_status: dict[str, bool] = {}

    def initialize_security(self) -> bool:
        """
        Initialize all security hardening measures.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing security hardening...")

            # Initialize secure configuration manager
            self.config_manager = initialize_security_hardening(self.settings)
            set_config_manager(self.config_manager)

            # Validate security configuration
            self.security_warnings = (
                self.config_manager.validate_security_configuration()
            )

            # Apply security configurations
            self._apply_security_headers()
            self._configure_input_validation()
            self._setup_secure_serialization()
            self._validate_environment_security()

            # Initialize database security
            self._initialize_database_security()

            # Generate security report
            self._generate_security_report()

            logger.info("Security hardening initialization completed")
            return True

        except Exception as e:
            logger.error(f"Security initialization failed: {str(e)}")
            return False

    def _apply_security_headers(self) -> None:
        """Apply security headers configuration."""
        if not self.config_manager:
            return

        try:
            headers = self.config_manager.get_security_headers()

            # Store headers for use by middleware
            self.settings.security_headers = headers
            self.security_status["security_headers"] = True

            logger.info("Security headers configured successfully")

        except Exception as e:
            logger.error(f"Failed to configure security headers: {str(e)}")
            self.security_status["security_headers"] = False

    def _configure_input_validation(self) -> None:
        """Configure input validation."""
        try:
            validator = get_input_validator()

            # Test validator
            test_input = "test_input"
            validator.validate_and_sanitize(test_input)

            self.security_status["input_validation"] = True
            logger.info("Input validation configured successfully")

        except Exception as e:
            logger.error(f"Failed to configure input validation: {str(e)}")
            self.security_status["input_validation"] = False

    def _setup_secure_serialization(self) -> None:
        """Setup secure serialization."""
        try:
            serializer = get_secure_serializer()

            # Test serialization
            test_data = {"test": "data"}
            from pathlib import Path
            from tempfile import NamedTemporaryFile

            with NamedTemporaryFile(delete=False) as tmp:
                tmp_path = Path(tmp.name)
                serializer.serialize_model(test_data, tmp_path)
                deserialized = serializer.deserialize_model(tmp_path)
                tmp_path.unlink()

            self.security_status["secure_serialization"] = True
            logger.info("Secure serialization configured successfully")

        except Exception as e:
            logger.error(f"Failed to setup secure serialization: {str(e)}")
            self.security_status["secure_serialization"] = False

    def _validate_environment_security(self) -> None:
        """Validate environment security configuration."""
        try:
            # Check for secure environment variables
            required_env_vars = [
                "PYNOMALY_SECRET_KEY",
                "PYNOMALY_MASTER_KEY",
            ]

            missing_vars = []
            for var in required_env_vars:
                if not os.getenv(var):
                    missing_vars.append(var)

            if missing_vars:
                self.security_warnings.append(
                    f"Missing required environment variables: {', '.join(missing_vars)}"
                )

            # Check for development/debug settings in production
            if not self.settings.app.debug:
                if (
                    self.settings.security.secret_key
                    == "change-me-in-production-this-is-32-chars-long-default-key"
                ):
                    self.security_warnings.append(
                        "Using default secret key in production"
                    )

            self.security_status["environment_security"] = len(missing_vars) == 0
            logger.info("Environment security validation completed")

        except Exception as e:
            logger.error(f"Environment security validation failed: {str(e)}")
            self.security_status["environment_security"] = False

    def _initialize_database_security(self) -> None:
        """Initialize database security measures."""
        try:
            # Initialize query auditing
            auditor = get_query_auditor()

            # Test audit functionality
            auditor.audit_query("SELECT 1", {}, "system")

            self.security_status["database_security"] = True
            logger.info("Database security initialized successfully")

        except Exception as e:
            logger.error(f"Database security initialization failed: {str(e)}")
            self.security_status["database_security"] = False

    def _generate_security_report(self) -> None:
        """Generate security status report."""
        logger.info("Security Status Report:")
        logger.info("=" * 50)

        for component, status in self.security_status.items():
            status_str = "✓ ENABLED" if status else "✗ FAILED"
            logger.info(f"  {component}: {status_str}")

        if self.security_warnings:
            logger.warning("Security Warnings:")
            for warning in self.security_warnings:
                logger.warning(f"  - {warning}")
        else:
            logger.info("No security warnings")

    def get_security_status(self) -> dict[str, Any]:
        """Get current security status."""
        return {
            "status": self.security_status,
            "warnings": self.security_warnings,
            "overall_status": all(self.security_status.values())
            and len(self.security_warnings) == 0,
        }

    def validate_request_security(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """
        Validate request security.

        Args:
            request_data: Request data to validate

        Returns:
            Validated and sanitized request data
        """
        try:
            validator = get_input_validator()

            # Validate and sanitize all request data
            sanitized_data = {}
            for key, value in request_data.items():
                sanitized_data[key] = validator.validate_and_sanitize(value, key)

            return sanitized_data

        except Exception as e:
            logger.error(f"Request security validation failed: {str(e)}")
            raise ValueError(f"Request validation failed: {str(e)}")

    def audit_database_operation(
        self, query: str, parameters: dict[str, Any], user_id: str
    ) -> bool:
        """
        Audit database operation.

        Args:
            query: SQL query
            parameters: Query parameters
            user_id: User executing the query

        Returns:
            True if query is safe, False if suspicious
        """
        try:
            auditor = get_query_auditor()
            return auditor.audit_query(query, parameters, user_id)

        except Exception as e:
            logger.error(f"Database audit failed: {str(e)}")
            return False


def initialize_application_security(settings: Settings) -> SecurityIntegrationManager:
    """
    Initialize application security.

    Args:
        settings: Application settings

    Returns:
        Configured security integration manager
    """
    security_manager = SecurityIntegrationManager(settings)

    if not security_manager.initialize_security():
        logger.error("Security initialization failed - application may be vulnerable")
        # In production, you might want to exit here
        if not settings.app.debug:
            sys.exit(1)

    return security_manager


def get_security_middleware_config() -> dict[str, Any]:
    """
    Get security middleware configuration.

    Returns:
        Security middleware configuration
    """
    config_manager = get_config_manager()
    if not config_manager:
        return {}

    return {
        "headers": config_manager.get_security_headers(),
        "csp_policy": config_manager.harden_csp_policy(),
        "force_https": True,
        "hsts_max_age": 31536000,
        "csrf_protection": True,
        "xss_protection": True,
    }


def create_security_startup_check() -> bool:
    """
    Perform security startup check.

    Returns:
        True if security check passes, False otherwise
    """
    try:
        # Check for critical security environment variables
        critical_vars = ["PYNOMALY_SECRET_KEY"]

        for var in critical_vars:
            if not os.getenv(var):
                logger.error(f"Critical security variable missing: {var}")
                return False

        # Check for insecure defaults
        settings = Settings()
        if (
            settings.security.secret_key
            == "change-me-in-production-this-is-32-chars-long-default-key"
        ):
            logger.error("Using default secret key - security risk")
            return False

        logger.info("Security startup check passed")
        return True

    except Exception as e:
        logger.error(f"Security startup check failed: {str(e)}")
        return False


# Global security manager instance
_security_manager: SecurityIntegrationManager | None = None


def get_security_manager() -> SecurityIntegrationManager | None:
    """Get global security manager instance."""
    return _security_manager


def set_security_manager(manager: SecurityIntegrationManager) -> None:
    """Set global security manager instance."""
    global _security_manager
    _security_manager = manager
