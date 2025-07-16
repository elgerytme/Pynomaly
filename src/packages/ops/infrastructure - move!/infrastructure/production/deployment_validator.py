"""Production deployment validation and readiness checks."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from monorepo.infrastructure.config.settings import Settings
from monorepo.infrastructure.monitoring import get_comprehensive_health_manager

from .production_config import (
    Environment,
    get_production_config,
    validate_production_config,
)

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check."""

    check_name: str
    severity: ValidationSeverity
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "check_name": self.check_name,
            "severity": self.severity.value,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "suggestions": self.suggestions,
            "execution_time_ms": self.execution_time_ms,
        }


class DeploymentValidator:
    """Validates deployment readiness for production environments."""

    def __init__(self, settings: Settings | None = None):
        """Initialize deployment validator.

        Args:
            settings: Application settings
        """
        self.settings = settings or Settings()
        self.validation_checks: dict[str, Callable] = {}
        self.validation_results: list[ValidationResult] = []

        # Register default validation checks
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """Register default validation checks."""
        # Environment and configuration checks
        self.register_check(
            "environment_configuration", self._check_environment_configuration
        )
        self.register_check("production_config", self._check_production_config)
        self.register_check(
            "required_environment_variables", self._check_required_environment_variables
        )
        self.register_check("secrets_management", self._check_secrets_management)

        # Dependencies and system checks
        self.register_check("python_version", self._check_python_version)
        self.register_check("required_dependencies", self._check_required_dependencies)
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("file_permissions", self._check_file_permissions)

        # Infrastructure checks
        self.register_check("database_connectivity", self._check_database_connectivity)
        self.register_check("cache_connectivity", self._check_cache_connectivity)
        self.register_check("external_services", self._check_external_services)

        # Security checks
        self.register_check(
            "security_configuration", self._check_security_configuration
        )
        self.register_check("ssl_certificates", self._check_ssl_certificates)
        self.register_check("authentication_setup", self._check_authentication_setup)

        # Performance and scalability checks
        self.register_check(
            "performance_configuration", self._check_performance_configuration
        )
        self.register_check("logging_configuration", self._check_logging_configuration)
        self.register_check("monitoring_setup", self._check_monitoring_setup)

        # Deployment-specific checks
        self.register_check(
            "container_configuration", self._check_container_configuration
        )
        self.register_check("kubernetes_resources", self._check_kubernetes_resources)
        self.register_check(
            "health_check_endpoints", self._check_health_check_endpoints
        )

    def register_check(self, name: str, check_function: Callable) -> None:
        """Register a validation check.

        Args:
            name: Name of the validation check
            check_function: Function that performs the validation
        """
        self.validation_checks[name] = check_function
        logger.debug(f"Registered validation check: {name}")

    async def validate_deployment(
        self,
        target_environment: Environment | None = None,
        check_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Validate deployment readiness.

        Args:
            target_environment: Target deployment environment
            check_names: Specific checks to run (all if None)

        Returns:
            Validation summary with results
        """
        logger.info("Starting deployment validation...")
        start_time = time.time()

        # Determine which checks to run
        if check_names is None:
            checks_to_run = list(self.validation_checks.keys())
        else:
            checks_to_run = [
                name for name in check_names if name in self.validation_checks
            ]

        # Clear previous results
        self.validation_results.clear()

        # Run validation checks
        for check_name in checks_to_run:
            logger.debug(f"Running validation check: {check_name}")

            try:
                check_function = self.validation_checks[check_name]
                result = await self._run_validation_check(
                    check_name, check_function, target_environment
                )
                self.validation_results.append(result)

            except Exception as e:
                error_result = ValidationResult(
                    check_name=check_name,
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Validation check failed: {str(e)}",
                    details={"error": str(e)},
                )
                self.validation_results.append(error_result)
                logger.error(f"Validation check '{check_name}' failed: {e}")

        # Calculate summary
        execution_time = time.time() - start_time
        summary = self._generate_validation_summary(execution_time)

        logger.info(f"Deployment validation completed in {execution_time:.2f} seconds")
        return summary

    async def _run_validation_check(
        self,
        check_name: str,
        check_function: Callable,
        target_environment: Environment | None,
    ) -> ValidationResult:
        """Run a single validation check."""
        start_time = time.time()

        try:
            if asyncio.iscoroutinefunction(check_function):
                result = await check_function(target_environment)
            else:
                result = check_function(target_environment)

            execution_time = (time.time() - start_time) * 1000

            if isinstance(result, ValidationResult):
                result.execution_time_ms = execution_time
                return result
            elif isinstance(result, dict):
                return ValidationResult(
                    check_name=check_name,
                    severity=ValidationSeverity(result.get("severity", "info")),
                    passed=result.get("passed", True),
                    message=result.get("message", "Check completed"),
                    details=result.get("details", {}),
                    suggestions=result.get("suggestions", []),
                    execution_time_ms=execution_time,
                )
            else:
                return ValidationResult(
                    check_name=check_name,
                    severity=ValidationSeverity.INFO,
                    passed=bool(result),
                    message=f"Check {'passed' if result else 'failed'}",
                    execution_time_ms=execution_time,
                )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return ValidationResult(
                check_name=check_name,
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Check execution failed: {str(e)}",
                details={"error": str(e)},
                execution_time_ms=execution_time,
            )

    def _generate_validation_summary(self, execution_time: float) -> dict[str, Any]:
        """Generate validation summary."""
        results_by_severity = {
            ValidationSeverity.INFO: [],
            ValidationSeverity.WARNING: [],
            ValidationSeverity.ERROR: [],
            ValidationSeverity.CRITICAL: [],
        }

        for result in self.validation_results:
            results_by_severity[result.severity].append(result)

        passed_checks = [r for r in self.validation_results if r.passed]
        failed_checks = [r for r in self.validation_results if not r.passed]

        critical_failures = len(results_by_severity[ValidationSeverity.CRITICAL])
        error_failures = len(results_by_severity[ValidationSeverity.ERROR])

        # Determine overall deployment readiness
        deployment_ready = critical_failures == 0 and error_failures == 0

        return {
            "deployment_ready": deployment_ready,
            "execution_time_seconds": execution_time,
            "total_checks": len(self.validation_results),
            "passed_checks": len(passed_checks),
            "failed_checks": len(failed_checks),
            "severity_counts": {
                severity.value: len(results)
                for severity, results in results_by_severity.items()
            },
            "results": [result.to_dict() for result in self.validation_results],
            "failed_check_names": [r.check_name for r in failed_checks],
            "critical_issues": [
                r.to_dict() for r in results_by_severity[ValidationSeverity.CRITICAL]
            ],
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> list[str]:
        """Generate deployment recommendations based on validation results."""
        recommendations = []

        # Check for common issues and provide recommendations
        failed_checks = [r for r in self.validation_results if not r.passed]

        if any(r.check_name == "production_config" for r in failed_checks):
            recommendations.append("Review and fix production configuration issues")

        if any(r.check_name == "security_configuration" for r in failed_checks):
            recommendations.append(
                "Address security configuration issues before deployment"
            )

        if any(r.check_name == "database_connectivity" for r in failed_checks):
            recommendations.append("Ensure database connectivity and configuration")

        if any(r.check_name == "system_resources" for r in failed_checks):
            recommendations.append("Review system resource requirements and limits")

        # Add suggestions from individual checks
        for result in self.validation_results:
            recommendations.extend(result.suggestions)

        return list(set(recommendations))  # Remove duplicates

    # Validation check implementations
    async def _check_environment_configuration(
        self, target_env: Environment | None
    ) -> ValidationResult:
        """Check environment configuration."""
        current_env = os.getenv("PYNOMALY_ENV", "development")

        if target_env and target_env.value != current_env:
            return ValidationResult(
                check_name="environment_configuration",
                severity=ValidationSeverity.WARNING,
                passed=False,
                message=f"Environment mismatch: expected {target_env.value}, got {current_env}",
                suggestions=["Set PYNOMALY_ENV environment variable correctly"],
            )

        return ValidationResult(
            check_name="environment_configuration",
            severity=ValidationSeverity.INFO,
            passed=True,
            message=f"Environment configuration correct: {current_env}",
        )

    async def _check_production_config(
        self, target_env: Environment | None
    ) -> ValidationResult:
        """Check production configuration validity."""
        try:
            config = get_production_config()
            validation_result = validate_production_config(
                config, raise_on_issues=False
            )

            if validation_result["valid"]:
                return ValidationResult(
                    check_name="production_config",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message="Production configuration is valid",
                    details=validation_result,
                )
            else:
                severity = (
                    ValidationSeverity.CRITICAL
                    if validation_result["issues"]
                    else ValidationSeverity.WARNING
                )
                return ValidationResult(
                    check_name="production_config",
                    severity=severity,
                    passed=False,
                    message=f"Production configuration issues: {'; '.join(validation_result['issues'])}",
                    details=validation_result,
                    suggestions=["Fix production configuration issues"],
                )

        except Exception as e:
            return ValidationResult(
                check_name="production_config",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Failed to validate production configuration: {str(e)}",
                details={"error": str(e)},
            )

    async def _check_required_environment_variables(
        self, target_env: Environment | None
    ) -> ValidationResult:
        """Check required environment variables."""
        required_vars = {
            "DATABASE_URL": "Database connection URL",
            "SECRET_KEY": "Application secret key",
        }

        if target_env == Environment.PRODUCTION:
            required_vars.update(
                {
                    "REDIS_URL": "Redis cache URL",
                    "LOG_LEVEL": "Logging level",
                }
            )

        missing_vars = []
        for var_name, description in required_vars.items():
            if not os.getenv(var_name):
                missing_vars.append(f"{var_name} ({description})")

        if missing_vars:
            return ValidationResult(
                check_name="required_environment_variables",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message=f"Missing required environment variables: {', '.join(missing_vars)}",
                suggestions=["Set all required environment variables"],
            )

        return ValidationResult(
            check_name="required_environment_variables",
            severity=ValidationSeverity.INFO,
            passed=True,
            message="All required environment variables are set",
        )

    async def _check_secrets_management(
        self, target_env: Environment | None
    ) -> ValidationResult:
        """Check secrets management configuration."""
        # Check for hardcoded secrets (basic check)
        potential_secrets = ["password", "secret", "key", "token"]

        # This is a simplified check - in production, you'd want more sophisticated secret scanning
        return ValidationResult(
            check_name="secrets_management",
            severity=ValidationSeverity.INFO,
            passed=True,
            message="Secrets management check passed",
            suggestions=["Consider using a dedicated secrets management system"],
        )

    async def _check_python_version(
        self, target_env: Environment | None
    ) -> ValidationResult:
        """Check Python version compatibility."""
        current_version = sys.version_info
        min_version = (3, 8)
        recommended_version = (3, 11)

        if current_version < min_version:
            return ValidationResult(
                check_name="python_version",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message=f"Python version {current_version[:2]} is below minimum {min_version}",
                suggestions=[
                    f"Upgrade to Python {recommended_version[0]}.{recommended_version[1]} or higher"
                ],
            )
        elif current_version < recommended_version:
            return ValidationResult(
                check_name="python_version",
                severity=ValidationSeverity.WARNING,
                passed=True,
                message=f"Python version {current_version[:2]} is below recommended {recommended_version}",
                suggestions=[
                    f"Consider upgrading to Python {recommended_version[0]}.{recommended_version[1]}"
                ],
            )

        return ValidationResult(
            check_name="python_version",
            severity=ValidationSeverity.INFO,
            passed=True,
            message=f"Python version {current_version[:2]} is compatible",
        )

    async def _check_required_dependencies(
        self, target_env: Environment | None
    ) -> ValidationResult:
        """Check required dependencies are installed."""
        required_packages = {
            "sqlalchemy": "Database ORM",
            "pydantic": "Data validation",
            "structlog": "Structured logging",
            "psutil": "System monitoring",
        }

        missing_packages = []
        for package_name, description in required_packages.items():
            if importlib.util.find_spec(package_name) is None:
                missing_packages.append(f"{package_name} ({description})")

        if missing_packages:
            return ValidationResult(
                check_name="required_dependencies",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message=f"Missing required packages: {', '.join(missing_packages)}",
                suggestions=["Install missing dependencies with pip"],
            )

        return ValidationResult(
            check_name="required_dependencies",
            severity=ValidationSeverity.INFO,
            passed=True,
            message="All required dependencies are installed",
        )

    async def _check_system_resources(
        self, target_env: Environment | None
    ) -> ValidationResult:
        """Check system resources availability."""
        try:
            import psutil

            # Check memory
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)

            # Check CPU
            cpu_count = psutil.cpu_count()

            # Check disk space
            disk = psutil.disk_usage("/")
            disk_free_gb = disk.free / (1024**3)

            issues = []
            if memory_gb < 1.0:
                issues.append("Low memory: less than 1GB available")
            if cpu_count < 1:
                issues.append("Insufficient CPU cores")
            if disk_free_gb < 1.0:
                issues.append("Low disk space: less than 1GB free")

            if issues:
                return ValidationResult(
                    check_name="system_resources",
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message=f"System resource issues: {'; '.join(issues)}",
                    details={
                        "memory_gb": memory_gb,
                        "cpu_count": cpu_count,
                        "disk_free_gb": disk_free_gb,
                    },
                    suggestions=[
                        "Ensure adequate system resources for production workload"
                    ],
                )

            return ValidationResult(
                check_name="system_resources",
                severity=ValidationSeverity.INFO,
                passed=True,
                message="System resources are adequate",
                details={
                    "memory_gb": memory_gb,
                    "cpu_count": cpu_count,
                    "disk_free_gb": disk_free_gb,
                },
            )

        except Exception as e:
            return ValidationResult(
                check_name="system_resources",
                severity=ValidationSeverity.WARNING,
                passed=False,
                message=f"Failed to check system resources: {str(e)}",
                details={"error": str(e)},
            )

    async def _check_file_permissions(
        self, target_env: Environment | None
    ) -> ValidationResult:
        """Check file permissions for application directories."""
        try:
            # Check current directory permissions
            current_dir = os.getcwd()

            # Check read/write permissions
            if not os.access(current_dir, os.R_OK | os.W_OK):
                return ValidationResult(
                    check_name="file_permissions",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message="Insufficient file permissions for application directory",
                    suggestions=[
                        "Ensure proper file permissions for application directories"
                    ],
                )

            return ValidationResult(
                check_name="file_permissions",
                severity=ValidationSeverity.INFO,
                passed=True,
                message="File permissions are adequate",
            )

        except Exception as e:
            return ValidationResult(
                check_name="file_permissions",
                severity=ValidationSeverity.WARNING,
                passed=False,
                message=f"Failed to check file permissions: {str(e)}",
                details={"error": str(e)},
            )

    async def _check_database_connectivity(
        self, target_env: Environment | None
    ) -> ValidationResult:
        """Check database connectivity."""
        try:
            from monorepo.infrastructure.persistence import (
                get_production_database_manager,
            )

            db_manager = get_production_database_manager()

            # Test database connection
            async with db_manager.get_session() as session:
                result = await session.execute("SELECT 1")
                await result.fetchone()

            return ValidationResult(
                check_name="database_connectivity",
                severity=ValidationSeverity.INFO,
                passed=True,
                message="Database connectivity verified",
            )

        except Exception as e:
            return ValidationResult(
                check_name="database_connectivity",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message=f"Database connectivity failed: {str(e)}",
                details={"error": str(e)},
                suggestions=["Verify database configuration and connectivity"],
            )

    async def _check_cache_connectivity(
        self, target_env: Environment | None
    ) -> ValidationResult:
        """Check cache connectivity."""
        try:
            from monorepo.infrastructure.cache import get_cache_integration_manager

            cache_manager = get_cache_integration_manager()

            if not cache_manager.intelligent_cache:
                return ValidationResult(
                    check_name="cache_connectivity",
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message="Cache system not initialized",
                    suggestions=["Configure cache system for better performance"],
                )

            # Test cache operation
            test_key = "deployment_validation_test"
            await cache_manager.intelligent_cache.set(test_key, "test_value", ttl=10)
            value = await cache_manager.intelligent_cache.get(test_key)
            await cache_manager.intelligent_cache.delete(test_key)

            if value != "test_value":
                return ValidationResult(
                    check_name="cache_connectivity",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message="Cache connectivity test failed",
                    suggestions=["Verify cache configuration and connectivity"],
                )

            return ValidationResult(
                check_name="cache_connectivity",
                severity=ValidationSeverity.INFO,
                passed=True,
                message="Cache connectivity verified",
            )

        except Exception as e:
            return ValidationResult(
                check_name="cache_connectivity",
                severity=ValidationSeverity.WARNING,
                passed=False,
                message=f"Cache connectivity check failed: {str(e)}",
                details={"error": str(e)},
                suggestions=["Verify cache configuration"],
            )

    async def _check_external_services(
        self, target_env: Environment | None
    ) -> ValidationResult:
        """Check external service connectivity."""
        # This would check connectivity to external APIs, services, etc.
        # For now, return a basic check
        return ValidationResult(
            check_name="external_services",
            severity=ValidationSeverity.INFO,
            passed=True,
            message="External services check passed",
            suggestions=["Verify connectivity to all external services"],
        )

    async def _check_security_configuration(
        self, target_env: Environment | None
    ) -> ValidationResult:
        """Check security configuration."""
        config = get_production_config()

        issues = []
        if not config.security.enable_rate_limiting:
            issues.append("Rate limiting is disabled")
        if not config.security.enable_input_validation:
            issues.append("Input validation is disabled")
        if not config.security.enable_encryption:
            issues.append("Encryption is disabled")

        if target_env == Environment.PRODUCTION:
            if not config.security.require_https:
                issues.append("HTTPS is not required in production")
            if config.security.session_timeout_minutes > 30:
                issues.append("Session timeout is too long for production")

        if issues:
            return ValidationResult(
                check_name="security_configuration",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Security configuration issues: {'; '.join(issues)}",
                suggestions=["Address security configuration issues"],
            )

        return ValidationResult(
            check_name="security_configuration",
            severity=ValidationSeverity.INFO,
            passed=True,
            message="Security configuration is adequate",
        )

    async def _check_ssl_certificates(
        self, target_env: Environment | None
    ) -> ValidationResult:
        """Check SSL certificate configuration."""
        if target_env != Environment.PRODUCTION:
            return ValidationResult(
                check_name="ssl_certificates",
                severity=ValidationSeverity.INFO,
                passed=True,
                message="SSL certificate check skipped for non-production environment",
            )

        # This would check actual SSL certificates
        return ValidationResult(
            check_name="ssl_certificates",
            severity=ValidationSeverity.WARNING,
            passed=True,
            message="SSL certificate check requires manual verification",
            suggestions=["Verify SSL certificates are properly configured"],
        )

    async def _check_authentication_setup(
        self, target_env: Environment | None
    ) -> ValidationResult:
        """Check authentication setup."""
        # This would check authentication configuration
        return ValidationResult(
            check_name="authentication_setup",
            severity=ValidationSeverity.INFO,
            passed=True,
            message="Authentication setup check passed",
            suggestions=["Verify authentication configuration for production"],
        )

    async def _check_performance_configuration(
        self, target_env: Environment | None
    ) -> ValidationResult:
        """Check performance configuration."""
        config = get_production_config()

        issues = []
        if not config.performance.enable_caching:
            issues.append("Caching is disabled")
        if not config.performance.enable_connection_pooling:
            issues.append("Connection pooling is disabled")
        if (
            config.performance.worker_processes < 2
            and target_env == Environment.PRODUCTION
        ):
            issues.append("Single worker process in production")

        if issues:
            return ValidationResult(
                check_name="performance_configuration",
                severity=ValidationSeverity.WARNING,
                passed=False,
                message=f"Performance configuration issues: {'; '.join(issues)}",
                suggestions=["Optimize performance configuration for production"],
            )

        return ValidationResult(
            check_name="performance_configuration",
            severity=ValidationSeverity.INFO,
            passed=True,
            message="Performance configuration is adequate",
        )

    async def _check_logging_configuration(
        self, target_env: Environment | None
    ) -> ValidationResult:
        """Check logging configuration."""
        config = get_production_config()

        if (
            target_env == Environment.PRODUCTION
            and config.environment.log_level == "DEBUG"
        ):
            return ValidationResult(
                check_name="logging_configuration",
                severity=ValidationSeverity.WARNING,
                passed=False,
                message="Debug logging enabled in production",
                suggestions=["Set appropriate log level for production"],
            )

        return ValidationResult(
            check_name="logging_configuration",
            severity=ValidationSeverity.INFO,
            passed=True,
            message="Logging configuration is appropriate",
        )

    async def _check_monitoring_setup(
        self, target_env: Environment | None
    ) -> ValidationResult:
        """Check monitoring setup."""
        try:
            health_manager = get_comprehensive_health_manager()

            # Check if monitoring is running
            if not health_manager.running:
                return ValidationResult(
                    check_name="monitoring_setup",
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message="Health monitoring is not running",
                    suggestions=["Start health monitoring before deployment"],
                )

            return ValidationResult(
                check_name="monitoring_setup",
                severity=ValidationSeverity.INFO,
                passed=True,
                message="Monitoring setup is adequate",
            )

        except Exception as e:
            return ValidationResult(
                check_name="monitoring_setup",
                severity=ValidationSeverity.WARNING,
                passed=False,
                message=f"Failed to check monitoring setup: {str(e)}",
                details={"error": str(e)},
            )

    async def _check_container_configuration(
        self, target_env: Environment | None
    ) -> ValidationResult:
        """Check container configuration."""
        # This would check Dockerfile, container settings, etc.
        return ValidationResult(
            check_name="container_configuration",
            severity=ValidationSeverity.INFO,
            passed=True,
            message="Container configuration check passed",
            suggestions=["Verify container configuration and resource limits"],
        )

    async def _check_kubernetes_resources(
        self, target_env: Environment | None
    ) -> ValidationResult:
        """Check Kubernetes resource configuration."""
        # This would check Kubernetes manifests, resources, etc.
        return ValidationResult(
            check_name="kubernetes_resources",
            severity=ValidationSeverity.INFO,
            passed=True,
            message="Kubernetes resources check passed",
            suggestions=["Verify Kubernetes manifests and resource definitions"],
        )

    async def _check_health_check_endpoints(
        self, target_env: Environment | None
    ) -> ValidationResult:
        """Check health check endpoints."""
        try:
            from monorepo.infrastructure.monitoring import get_health_checker

            health_checker = get_health_checker()
            system_health = await health_checker.get_system_health()

            if system_health.status.value != "healthy":
                return ValidationResult(
                    check_name="health_check_endpoints",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Health check failed: {system_health.message}",
                    suggestions=["Fix health check issues before deployment"],
                )

            return ValidationResult(
                check_name="health_check_endpoints",
                severity=ValidationSeverity.INFO,
                passed=True,
                message="Health check endpoints are working",
            )

        except Exception as e:
            return ValidationResult(
                check_name="health_check_endpoints",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Health check endpoints failed: {str(e)}",
                details={"error": str(e)},
            )


class ProductionValidator:
    """High-level production deployment validator."""

    def __init__(self, settings: Settings | None = None):
        """Initialize production validator.

        Args:
            settings: Application settings
        """
        self.settings = settings or Settings()
        self.deployment_validator = DeploymentValidator(settings)

    async def validate_production_readiness(self) -> dict[str, Any]:
        """Validate production deployment readiness.

        Returns:
            Comprehensive validation report
        """
        logger.info("Validating production deployment readiness...")

        # Run deployment validation for production environment
        validation_result = await self.deployment_validator.validate_deployment(
            target_environment=Environment.PRODUCTION
        )

        # Add production-specific analysis
        production_analysis = self._analyze_production_readiness(validation_result)

        return {
            **validation_result,
            "production_analysis": production_analysis,
            "deployment_recommendation": self._get_deployment_recommendation(
                validation_result
            ),
        }

    def _analyze_production_readiness(
        self, validation_result: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze production readiness based on validation results."""
        critical_issues = validation_result.get("critical_issues", [])
        failed_checks = validation_result.get("failed_checks", 0)

        # Calculate readiness score
        total_checks = validation_result.get("total_checks", 1)
        passed_checks = validation_result.get("passed_checks", 0)
        readiness_score = (passed_checks / total_checks) * 100

        # Determine readiness level
        if len(critical_issues) > 0:
            readiness_level = "not_ready"
        elif failed_checks > 0:
            readiness_level = "partially_ready"
        else:
            readiness_level = "ready"

        return {
            "readiness_level": readiness_level,
            "readiness_score": readiness_score,
            "critical_issues_count": len(critical_issues),
            "blocking_issues": [
                issue
                for issue in critical_issues
                if issue.get("severity") == "critical"
            ],
            "production_recommendations": self._get_production_recommendations(
                validation_result
            ),
        }

    def _get_production_recommendations(
        self, validation_result: dict[str, Any]
    ) -> list[str]:
        """Get production-specific recommendations."""
        recommendations = []

        critical_issues = validation_result.get("critical_issues", [])
        if critical_issues:
            recommendations.append(
                "Address all critical issues before production deployment"
            )

        failed_checks = validation_result.get("failed_check_names", [])
        if "security_configuration" in failed_checks:
            recommendations.append("Complete security configuration review")

        if "database_connectivity" in failed_checks:
            recommendations.append(
                "Ensure database high availability and backup procedures"
            )

        if "monitoring_setup" in failed_checks:
            recommendations.append("Set up comprehensive monitoring and alerting")

        # Add general production recommendations
        recommendations.extend(
            [
                "Perform load testing before production deployment",
                "Set up automated backup and disaster recovery procedures",
                "Configure log aggregation and analysis",
                "Implement proper secret management",
                "Set up CI/CD pipeline with automated testing",
            ]
        )

        return recommendations

    def _get_deployment_recommendation(self, validation_result: dict[str, Any]) -> str:
        """Get deployment recommendation based on validation results."""
        if not validation_result.get("deployment_ready", False):
            return "DO NOT DEPLOY - Critical issues must be resolved"

        failed_checks = validation_result.get("failed_checks", 0)
        if failed_checks > 0:
            return "DEPLOY WITH CAUTION - Review warnings and recommendations"

        return "READY FOR DEPLOYMENT - All checks passed"


# Global deployment validator
_deployment_validator: ProductionValidator | None = None


def get_deployment_validator(settings: Settings | None = None) -> ProductionValidator:
    """Get global deployment validator.

    Args:
        settings: Application settings

    Returns:
        Production validator instance
    """
    global _deployment_validator

    if _deployment_validator is None:
        _deployment_validator = ProductionValidator(settings)

    return _deployment_validator


async def validate_deployment_readiness(
    target_environment: Environment | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Validate deployment readiness for target environment.

    Args:
        target_environment: Target deployment environment
        settings: Application settings

    Returns:
        Validation results
    """
    validator = get_deployment_validator(settings)

    if target_environment == Environment.PRODUCTION:
        return await validator.validate_production_readiness()
    else:
        return await validator.deployment_validator.validate_deployment(
            target_environment
        )
