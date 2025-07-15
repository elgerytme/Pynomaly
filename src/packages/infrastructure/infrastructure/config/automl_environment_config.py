"""Environment-based AutoML feature configuration and validation.

This module provides environment-specific AutoML feature control with production
safety, development flexibility, and comprehensive validation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from .automl_feature_manager import AutoMLEnvironment, automl_manager
from .feature_flags import get_automl_configuration, validate_automl_environment

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration for AutoML features."""

    name: str
    allowed_features: set[str]
    restricted_features: set[str] = field(default_factory=set)
    required_validations: list[str] = field(default_factory=list)
    resource_limits: dict[str, Any] = field(default_factory=dict)
    security_requirements: dict[str, bool] = field(default_factory=dict)
    fallback_strategies: dict[str, str] = field(default_factory=dict)


class AutoMLEnvironmentConfigurator:
    """Manages environment-specific AutoML feature configurations."""

    def __init__(self):
        """Initialize the environment configurator."""
        self._environment_configs = self._initialize_environment_configs()
        self._current_environment = automl_manager.get_environment()

    def _initialize_environment_configs(
        self,
    ) -> dict[AutoMLEnvironment, EnvironmentConfig]:
        """Initialize predefined environment configurations."""
        configs = {}

        # Development Environment - All features allowed
        configs[AutoMLEnvironment.DEVELOPMENT] = EnvironmentConfig(
            name="development",
            allowed_features={
                "automl_hyperparameter_optimization",
                "automl_feature_engineering",
                "automl_model_selection",
                "automl_ensemble_creation",
                "automl_pipeline_optimization",
                "automl_distributed_search",
                "automl_neural_architecture_search",
                "automl_time_series_features",
                "automl_transfer_learning",
                "automl_early_stopping",
                "automl_warm_start",
                "automl_cross_validation",
                "automl_validation_strategies",
                "automl_resource_management",
                "automl_experiment_tracking",
            },
            required_validations=["package_availability"],
            resource_limits={
                "max_optimization_time_minutes": 60,
                "max_trials": 1000,
                "max_concurrent_jobs": 4,
                "max_memory_gb": 16,
            },
            security_requirements={
                "require_authentication": False,
                "audit_logging": False,
                "data_encryption": False,
            },
        )

        # Testing Environment - All features for testing
        configs[AutoMLEnvironment.TESTING] = EnvironmentConfig(
            name="testing",
            allowed_features={
                "automl_hyperparameter_optimization",
                "automl_feature_engineering",
                "automl_model_selection",
                "automl_ensemble_creation",
                "automl_pipeline_optimization",
                "automl_distributed_search",
                "automl_neural_architecture_search",
                "automl_time_series_features",
                "automl_transfer_learning",
                "automl_early_stopping",
                "automl_warm_start",
                "automl_cross_validation",
                "automl_validation_strategies",
                "automl_resource_management",
                "automl_experiment_tracking",
            },
            required_validations=["package_availability", "feature_compatibility"],
            resource_limits={
                "max_optimization_time_minutes": 30,
                "max_trials": 100,
                "max_concurrent_jobs": 2,
                "max_memory_gb": 8,
            },
            security_requirements={
                "require_authentication": False,
                "audit_logging": True,
                "data_encryption": False,
            },
        )

        # Staging Environment - Limited experimental features
        configs[AutoMLEnvironment.STAGING] = EnvironmentConfig(
            name="staging",
            allowed_features={
                "automl_hyperparameter_optimization",
                "automl_feature_engineering",
                "automl_model_selection",
                "automl_ensemble_creation",
                "automl_pipeline_optimization",
                "automl_early_stopping",
                "automl_warm_start",
                "automl_cross_validation",
                "automl_validation_strategies",
                "automl_resource_management",
                "automl_experiment_tracking",
            },
            restricted_features={
                "automl_distributed_search",
                "automl_neural_architecture_search",
                "automl_transfer_learning",
            },
            required_validations=[
                "package_availability",
                "feature_compatibility",
                "resource_availability",
                "security_compliance",
            ],
            resource_limits={
                "max_optimization_time_minutes": 45,
                "max_trials": 200,
                "max_concurrent_jobs": 3,
                "max_memory_gb": 12,
            },
            security_requirements={
                "require_authentication": True,
                "audit_logging": True,
                "data_encryption": True,
            },
            fallback_strategies={
                "automl_distributed_search": "automl_hyperparameter_optimization",
                "automl_neural_architecture_search": "automl_model_selection",
            },
        )

        # Production Environment - Only stable features
        configs[AutoMLEnvironment.PRODUCTION] = EnvironmentConfig(
            name="production",
            allowed_features={
                "automl_hyperparameter_optimization",
                "automl_feature_engineering",
                "automl_model_selection",
                "automl_ensemble_creation",
                "automl_early_stopping",
                "automl_cross_validation",
                "automl_resource_management",
            },
            restricted_features={
                "automl_distributed_search",
                "automl_neural_architecture_search",
                "automl_transfer_learning",
                "automl_warm_start",  # May cause instability
                "automl_validation_strategies",  # Experimental
                "automl_experiment_tracking",  # May impact performance
            },
            required_validations=[
                "package_availability",
                "feature_compatibility",
                "resource_availability",
                "security_compliance",
                "performance_benchmarks",
                "data_quality_checks",
            ],
            resource_limits={
                "max_optimization_time_minutes": 30,
                "max_trials": 50,
                "max_concurrent_jobs": 2,
                "max_memory_gb": 8,
            },
            security_requirements={
                "require_authentication": True,
                "audit_logging": True,
                "data_encryption": True,
            },
            fallback_strategies={
                "automl_distributed_search": "automl_hyperparameter_optimization",
                "automl_neural_architecture_search": "automl_model_selection",
                "automl_transfer_learning": "automl_model_selection",
                "automl_warm_start": "automl_hyperparameter_optimization",
                "automl_validation_strategies": "automl_cross_validation",
                "automl_experiment_tracking": None,  # No fallback, disable completely
            },
        )

        return configs

    def get_environment_config(
        self, environment: AutoMLEnvironment | None = None
    ) -> EnvironmentConfig:
        """Get configuration for the specified or current environment."""
        env = environment or self._current_environment
        return self._environment_configs.get(
            env, self._environment_configs[AutoMLEnvironment.DEVELOPMENT]
        )

    def is_feature_allowed_in_environment(
        self, feature_name: str, environment: AutoMLEnvironment | None = None
    ) -> bool:
        """Check if a feature is allowed in the specified environment."""
        config = self.get_environment_config(environment)

        # Check if explicitly restricted
        if feature_name in config.restricted_features:
            return False

        # Check if in allowed features
        return feature_name in config.allowed_features

    def get_feature_fallback(
        self, feature_name: str, environment: AutoMLEnvironment | None = None
    ) -> str | None:
        """Get fallback feature for a restricted feature."""
        config = self.get_environment_config(environment)
        return config.fallback_strategies.get(feature_name)

    def validate_environment_compliance(
        self, environment: AutoMLEnvironment | None = None
    ) -> dict[str, Any]:
        """Validate that current AutoML configuration complies with environment requirements."""
        config = self.get_environment_config(environment)
        env = environment or self._current_environment

        validation_result = {
            "environment": env.value,
            "compliant": True,
            "violations": [],
            "warnings": [],
            "recommendations": [],
            "required_actions": [],
        }

        # Check enabled features against environment restrictions
        enabled_features = automl_manager.get_feature_status()["enabled_features"]

        for feature_name, is_enabled in enabled_features.items():
            if is_enabled and not self.is_feature_allowed_in_environment(
                feature_name, env
            ):
                validation_result["violations"].append(
                    f"Feature '{feature_name}' is enabled but not allowed in {env.value} environment"
                )
                validation_result["compliant"] = False

                # Suggest fallback if available
                fallback = self.get_feature_fallback(feature_name, env)
                if fallback:
                    validation_result["recommendations"].append(
                        f"Use fallback feature '{fallback}' instead of '{feature_name}'"
                    )
                else:
                    validation_result["required_actions"].append(
                        f"Disable feature '{feature_name}' by setting PYNOMALY_{feature_name.upper()}=false"
                    )

        # Check resource limits
        resource_violations = self._validate_resource_limits(config)
        validation_result["violations"].extend(resource_violations)
        if resource_violations:
            validation_result["compliant"] = False

        # Check security requirements
        security_violations = self._validate_security_requirements(config)
        validation_result["violations"].extend(security_violations)
        if security_violations:
            validation_result["compliant"] = False

        # Run required validations
        for validation_type in config.required_validations:
            validation_result_specific = self._run_specific_validation(validation_type)
            if not validation_result_specific["valid"]:
                validation_result["warnings"].extend(
                    validation_result_specific.get("warnings", [])
                )
                validation_result["violations"].extend(
                    validation_result_specific.get("errors", [])
                )

        return validation_result

    def _validate_resource_limits(self, config: EnvironmentConfig) -> list[str]:
        """Validate resource limit compliance."""
        violations = []

        # Check if current AutoML configuration exceeds limits
        current_config = get_automl_configuration()

        # This is a simplified check - in practice, you'd check actual runtime values
        for limit_name, limit_value in config.resource_limits.items():
            if limit_name == "max_memory_gb":
                # Check if memory-intensive features are enabled beyond limits
                memory_intensive_features = [
                    "automl_distributed_search",
                    "automl_neural_architecture_search",
                ]
                enabled_intensive = [
                    f
                    for f in memory_intensive_features
                    if current_config["enabled_features"].get(f, False)
                ]
                if enabled_intensive and limit_value < 12:
                    violations.append(
                        f"Memory-intensive features {enabled_intensive} enabled "
                        f"but memory limit is {limit_value}GB (recommended: 12GB+)"
                    )

        return violations

    def _validate_security_requirements(self, config: EnvironmentConfig) -> list[str]:
        """Validate security requirement compliance."""
        violations = []

        # Check environment variables for security settings
        for requirement, required_value in config.security_requirements.items():
            if requirement == "require_authentication":
                auth_enabled = (
                    os.getenv("PYNOMALY_AUTH_ENABLED", "false").lower() == "true"
                )
                if required_value and not auth_enabled:
                    violations.append(
                        "Authentication is required in this environment but not enabled. "
                        "Set PYNOMALY_AUTH_ENABLED=true"
                    )
            elif requirement == "audit_logging":
                audit_enabled = (
                    os.getenv("PYNOMALY_AUDIT_LOGGING", "false").lower() == "true"
                )
                if required_value and not audit_enabled:
                    violations.append(
                        "Audit logging is required in this environment but not enabled. "
                        "Set PYNOMALY_AUDIT_LOGGING=true"
                    )
            elif requirement == "data_encryption":
                encryption_enabled = (
                    os.getenv("PYNOMALY_DATA_ENCRYPTION", "false").lower() == "true"
                )
                if required_value and not encryption_enabled:
                    violations.append(
                        "Data encryption is required in this environment but not enabled. "
                        "Set PYNOMALY_DATA_ENCRYPTION=true"
                    )

        return violations

    def _run_specific_validation(self, validation_type: str) -> dict[str, Any]:
        """Run a specific type of validation."""
        if validation_type == "package_availability":
            return validate_automl_environment()
        elif validation_type == "feature_compatibility":
            return {"valid": True, "warnings": [], "errors": []}  # Placeholder
        elif validation_type == "resource_availability":
            return self._validate_system_resources()
        elif validation_type == "security_compliance":
            return {"valid": True, "warnings": [], "errors": []}  # Placeholder
        elif validation_type == "performance_benchmarks":
            return self._validate_performance_requirements()
        elif validation_type == "data_quality_checks":
            return {"valid": True, "warnings": [], "errors": []}  # Placeholder
        else:
            return {
                "valid": True,
                "warnings": [f"Unknown validation type: {validation_type}"],
            }

    def _validate_system_resources(self) -> dict[str, Any]:
        """Validate system resource availability."""
        import psutil

        validation_result = {"valid": True, "warnings": [], "errors": []}

        try:
            # Check memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)

            if available_gb < 4:
                validation_result["errors"].append(
                    f"Insufficient memory: {available_gb:.1f}GB available (minimum: 4GB)"
                )
                validation_result["valid"] = False
            elif available_gb < 8:
                validation_result["warnings"].append(
                    f"Low memory: {available_gb:.1f}GB available (recommended: 8GB+)"
                )

            # Check CPU
            cpu_count = psutil.cpu_count()
            if cpu_count < 2:
                validation_result["warnings"].append(
                    f"Low CPU count: {cpu_count} cores (recommended: 4+ cores)"
                )

        except Exception as e:
            validation_result["warnings"].append(
                f"Could not validate system resources: {e}"
            )

        return validation_result

    def _validate_performance_requirements(self) -> dict[str, Any]:
        """Validate performance requirements for production."""
        validation_result = {"valid": True, "warnings": [], "errors": []}

        # This would include actual performance benchmarks in a real implementation
        validation_result["warnings"].append(
            "Performance validation not yet implemented"
        )

        return validation_result

    def apply_environment_defaults(
        self, environment: AutoMLEnvironment | None = None
    ) -> dict[str, Any]:
        """Apply environment-specific default configurations."""
        config = self.get_environment_config(environment)
        env = environment or self._current_environment

        applied_changes = {
            "environment": env.value,
            "changes_applied": [],
            "features_enabled": [],
            "features_disabled": [],
            "fallbacks_applied": [],
        }

        logger.info(f"Applying environment defaults for {env.value}")

        # Enable allowed features that aren't explicitly configured
        for feature in config.allowed_features:
            if automl_manager.is_feature_enabled(feature):
                applied_changes["features_enabled"].append(feature)

        # Disable restricted features and apply fallbacks
        current_status = automl_manager.get_feature_status()
        for feature_name, is_enabled in current_status["enabled_features"].items():
            if is_enabled and feature_name in config.restricted_features:
                applied_changes["features_disabled"].append(feature_name)

                # Apply fallback if available
                fallback = config.fallback_strategies.get(feature_name)
                if fallback:
                    applied_changes["fallbacks_applied"].append(
                        {"from": feature_name, "to": fallback}
                    )
                    automl_manager.disable_feature(feature_name, temporary=True)
                    if fallback and not automl_manager.is_feature_enabled(fallback):
                        automl_manager.enable_feature(fallback, temporary=True)

        return applied_changes

    def get_environment_summary(self) -> dict[str, Any]:
        """Get a comprehensive summary of the current environment configuration."""
        current_env = self._current_environment
        config = self.get_environment_config(current_env)
        compliance = self.validate_environment_compliance(current_env)

        return {
            "current_environment": current_env.value,
            "environment_config": {
                "allowed_features": list(config.allowed_features),
                "restricted_features": list(config.restricted_features),
                "resource_limits": config.resource_limits,
                "security_requirements": config.security_requirements,
                "required_validations": config.required_validations,
            },
            "compliance_status": compliance,
            "automl_manager_status": automl_manager.get_feature_status(),
            "recommendations": self._generate_environment_recommendations(
                config, compliance
            ),
        }

    def _generate_environment_recommendations(
        self, config: EnvironmentConfig, compliance: dict[str, Any]
    ) -> list[str]:
        """Generate environment-specific recommendations."""
        recommendations = []

        if not compliance["compliant"]:
            recommendations.append(
                "Address compliance violations before proceeding with AutoML operations"
            )

        if compliance["warnings"]:
            recommendations.append(
                "Review and address warnings for optimal AutoML performance"
            )

        # Environment-specific recommendations
        if self._current_environment == AutoMLEnvironment.PRODUCTION:
            recommendations.extend(
                [
                    "Ensure comprehensive monitoring and alerting for AutoML operations",
                    "Implement proper backup and recovery procedures for AutoML artifacts",
                    "Validate performance benchmarks before deploying AutoML models",
                ]
            )
        elif self._current_environment == AutoMLEnvironment.DEVELOPMENT:
            recommendations.extend(
                [
                    "Experiment with different AutoML features to find optimal configurations",
                    "Document successful configurations for staging/production deployment",
                ]
            )

        return recommendations


# Global environment configurator instance
environment_configurator = AutoMLEnvironmentConfigurator()


# Convenience functions
def get_environment_configurator() -> AutoMLEnvironmentConfigurator:
    """Get the global environment configurator instance."""
    return environment_configurator


def validate_current_environment() -> dict[str, Any]:
    """Validate the current environment configuration."""
    return environment_configurator.validate_environment_compliance()


def apply_environment_defaults() -> dict[str, Any]:
    """Apply environment-specific defaults to AutoML configuration."""
    return environment_configurator.apply_environment_defaults()


def get_current_environment_summary() -> dict[str, Any]:
    """Get a summary of the current environment configuration."""
    return environment_configurator.get_environment_summary()


def is_feature_production_ready(feature_name: str) -> bool:
    """Check if a feature is production-ready."""
    return environment_configurator.is_feature_allowed_in_environment(
        feature_name, AutoMLEnvironment.PRODUCTION
    )
