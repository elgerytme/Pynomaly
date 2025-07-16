"""Comprehensive AutoML feature flag validation system.

This module provides extensive validation capabilities for AutoML features,
including dependency checking, performance impact analysis, security validation,
and production readiness assessment.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .automl_environment_config import environment_configurator
from .automl_feature_manager import automl_manager

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Categories of validation checks."""

    DEPENDENCY = "dependency"
    COMPATIBILITY = "compatibility"
    PERFORMANCE = "performance"
    SECURITY = "security"
    RESOURCE = "resource"
    ENVIRONMENT = "environment"
    CONFIGURATION = "configuration"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""

    category: ValidationCategory
    severity: ValidationSeverity
    feature_name: str
    message: str
    details: str
    recommendation: str
    fix_command: str | None = None
    documentation_link: str | None = None


@dataclass
class ValidationReport:
    """Comprehensive validation report."""

    timestamp: str
    environment: str
    total_features_checked: int
    issues: list[ValidationIssue] = field(default_factory=list)
    passed_checks: list[str] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)
    overall_status: str = "unknown"
    recommendations: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate summary statistics."""
        self.summary = {
            "total_issues": len(self.issues),
            "critical": len(
                [i for i in self.issues if i.severity == ValidationSeverity.CRITICAL]
            ),
            "errors": len(
                [i for i in self.issues if i.severity == ValidationSeverity.ERROR]
            ),
            "warnings": len(
                [i for i in self.issues if i.severity == ValidationSeverity.WARNING]
            ),
            "info": len(
                [i for i in self.issues if i.severity == ValidationSeverity.INFO]
            ),
        }

        # Determine overall status
        if self.summary["critical"] > 0:
            self.overall_status = "critical"
        elif self.summary["errors"] > 0:
            self.overall_status = "error"
        elif self.summary["warnings"] > 0:
            self.overall_status = "warning"
        else:
            self.overall_status = "passed"


class AutoMLFeatureValidator:
    """Comprehensive validator for AutoML features."""

    def __init__(self):
        """Initialize the feature validator."""
        self.dependency_graph = self._build_dependency_graph()
        self.performance_profiles = self._load_performance_profiles()
        self.security_requirements = self._load_security_requirements()

    def _build_dependency_graph(self) -> dict[str, set[str]]:
        """Build feature dependency graph."""
        return {
            "automl_hyperparameter_optimization": {"advanced_automl"},
            "automl_feature_engineering": {"advanced_automl"},
            "automl_model_selection": {"advanced_automl"},
            "automl_ensemble_creation": {
                "automl_model_selection",
                "ensemble_optimization",
            },
            "automl_pipeline_optimization": {
                "automl_feature_engineering",
                "automl_hyperparameter_optimization",
            },
            "automl_distributed_search": {"automl_hyperparameter_optimization"},
            "automl_neural_architecture_search": {
                "deep_learning",
                "automl_hyperparameter_optimization",
            },
            "automl_time_series_features": {"automl_feature_engineering"},
            "automl_transfer_learning": {"deep_learning", "meta_learning"},
            "automl_warm_start": {"meta_learning", "automl_experiment_tracking"},
            "automl_validation_strategies": {"automl_cross_validation"},
            "automl_resource_management": {"performance_monitoring"},
            "automl_experiment_tracking": {"advanced_automl"},
        }

    def _load_performance_profiles(self) -> dict[str, dict[str, Any]]:
        """Load performance profiles for features."""
        return {
            "automl_hyperparameter_optimization": {
                "cpu_intensity": "medium",
                "memory_usage": "medium",
                "io_intensity": "low",
                "estimated_overhead": 0.2,
                "scaling_factor": 1.5,
            },
            "automl_feature_engineering": {
                "cpu_intensity": "high",
                "memory_usage": "high",
                "io_intensity": "medium",
                "estimated_overhead": 0.3,
                "scaling_factor": 2.0,
            },
            "automl_model_selection": {
                "cpu_intensity": "medium",
                "memory_usage": "medium",
                "io_intensity": "low",
                "estimated_overhead": 0.15,
                "scaling_factor": 1.3,
            },
            "automl_ensemble_creation": {
                "cpu_intensity": "high",
                "memory_usage": "high",
                "io_intensity": "medium",
                "estimated_overhead": 0.4,
                "scaling_factor": 2.5,
            },
            "automl_distributed_search": {
                "cpu_intensity": "very_high",
                "memory_usage": "very_high",
                "io_intensity": "high",
                "estimated_overhead": 0.6,
                "scaling_factor": 4.0,
            },
            "automl_neural_architecture_search": {
                "cpu_intensity": "very_high",
                "memory_usage": "very_high",
                "io_intensity": "high",
                "estimated_overhead": 0.8,
                "scaling_factor": 5.0,
            },
        }

    def _load_security_requirements(self) -> dict[str, dict[str, Any]]:
        """Load security requirements for features."""
        return {
            "automl_experiment_tracking": {
                "data_access_level": "high",
                "network_requirements": ["outbound_http", "outbound_https"],
                "sensitive_data": True,
                "audit_required": True,
            },
            "automl_distributed_search": {
                "data_access_level": "high",
                "network_requirements": ["outbound_tcp", "inbound_tcp"],
                "sensitive_data": True,
                "audit_required": True,
                "resource_sharing": True,
            },
            "automl_transfer_learning": {
                "data_access_level": "high",
                "network_requirements": ["outbound_https"],
                "sensitive_data": True,
                "audit_required": True,
                "external_dependencies": True,
            },
        }

    def validate_all_features(self) -> ValidationReport:
        """Run comprehensive validation on all AutoML features."""
        from datetime import datetime

        report = ValidationReport(
            timestamp=datetime.now().isoformat(),
            environment=automl_manager.get_environment().value,
            total_features_checked=0,
        )

        enabled_features = automl_manager.get_feature_status()["enabled_features"]
        report.total_features_checked = len(enabled_features)

        logger.info(
            f"Starting comprehensive validation of {report.total_features_checked} AutoML features"
        )

        for feature_name, is_enabled in enabled_features.items():
            if is_enabled:
                issues = self._validate_single_feature(feature_name)
                report.issues.extend(issues)

                if not issues:
                    report.passed_checks.append(feature_name)

        # Run cross-feature validations
        cross_issues = self._validate_feature_interactions(enabled_features)
        report.issues.extend(cross_issues)

        # Run environment-specific validations
        env_issues = self._validate_environment_compliance()
        report.issues.extend(env_issues)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report.issues)

        logger.info(
            f"Validation completed: {report.overall_status} status with {len(report.issues)} issues"
        )

        return report

    def _validate_single_feature(self, feature_name: str) -> list[ValidationIssue]:
        """Validate a single feature comprehensively."""
        issues = []

        # Dependency validation
        issues.extend(self._validate_dependencies(feature_name))

        # Package validation
        issues.extend(self._validate_packages(feature_name))

        # Performance validation
        issues.extend(self._validate_performance_impact(feature_name))

        # Security validation
        issues.extend(self._validate_security_requirements(feature_name))

        # Environment validation
        issues.extend(self._validate_environment_compatibility(feature_name))

        # Configuration validation
        issues.extend(self._validate_configuration(feature_name))

        return issues

    def _validate_dependencies(self, feature_name: str) -> list[ValidationIssue]:
        """Validate feature dependencies."""
        issues = []
        dependencies = self.dependency_graph.get(feature_name, set())
        enabled_features = automl_manager.get_feature_status()["enabled_features"]

        for dependency in dependencies:
            if not enabled_features.get(dependency, False):
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.DEPENDENCY,
                        severity=ValidationSeverity.ERROR,
                        feature_name=feature_name,
                        message=f"Missing required dependency: {dependency}",
                        details=f"Feature '{feature_name}' requires '{dependency}' to be enabled",
                        recommendation=f"Enable dependency '{dependency}' before using '{feature_name}'",
                        fix_command=f"export PYNOMALY_{dependency.upper()}=true",
                    )
                )

        return issues

    def _validate_packages(self, feature_name: str) -> list[ValidationIssue]:
        """Validate package availability for features."""
        issues = []

        package_requirements = {
            "automl_hyperparameter_optimization": ["optuna", "scikit-optimize"],
            "automl_feature_engineering": ["scikit-learn", "pandas", "numpy"],
            "automl_distributed_search": ["dask", "ray"],
            "automl_neural_architecture_search": ["torch", "tensorflow"],
            "automl_transfer_learning": ["torch", "tensorflow"],
            "automl_experiment_tracking": ["mlflow", "wandb"],
        }

        required_packages = package_requirements.get(feature_name, [])

        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                severity = (
                    ValidationSeverity.ERROR
                    if package in ["optuna", "scikit-learn"]
                    else ValidationSeverity.WARNING
                )
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.DEPENDENCY,
                        severity=severity,
                        feature_name=feature_name,
                        message=f"Missing required package: {package}",
                        details=f"Package '{package}' is required for feature '{feature_name}'",
                        recommendation="Install the required package",
                        fix_command=f"pip install {package}",
                    )
                )

        return issues

    def _validate_performance_impact(self, feature_name: str) -> list[ValidationIssue]:
        """Validate performance impact of features."""
        issues = []
        profile = self.performance_profiles.get(feature_name)

        if not profile:
            return issues

        # Check system resources
        try:
            import psutil

            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()

            # Memory validation
            if (
                profile["memory_usage"] in ["high", "very_high"]
                and memory.available < 4 * 1024**3
            ):
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.PERFORMANCE,
                        severity=ValidationSeverity.WARNING,
                        feature_name=feature_name,
                        message="High memory usage feature with limited system memory",
                        details=f"Feature requires {profile['memory_usage']} memory but only {memory.available / 1024**3:.1f}GB available",
                        recommendation="Consider increasing system memory or disabling memory-intensive features",
                    )
                )

            # CPU validation
            if profile["cpu_intensity"] in ["high", "very_high"] and cpu_count < 4:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.PERFORMANCE,
                        severity=ValidationSeverity.WARNING,
                        feature_name=feature_name,
                        message="CPU-intensive feature with limited processing power",
                        details=f"Feature requires {profile['cpu_intensity']} CPU usage but only {cpu_count} cores available",
                        recommendation="Consider using a system with more CPU cores for optimal performance",
                    )
                )

        except ImportError:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.RESOURCE,
                    severity=ValidationSeverity.INFO,
                    feature_name=feature_name,
                    message="Cannot validate system resources",
                    details="psutil package not available for resource validation",
                    recommendation="Install psutil for comprehensive resource validation",
                    fix_command="pip install psutil",
                )
            )

        return issues

    def _validate_security_requirements(
        self, feature_name: str
    ) -> list[ValidationIssue]:
        """Validate security requirements for features."""
        issues = []
        requirements = self.security_requirements.get(feature_name)

        if not requirements:
            return issues

        # Check data access level requirements
        if requirements.get("data_access_level") == "high":
            audit_enabled = (
                os.getenv("PYNOMALY_AUDIT_LOGGING", "false").lower() == "true"
            )
            if not audit_enabled:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.SECURITY,
                        severity=ValidationSeverity.WARNING,
                        feature_name=feature_name,
                        message="High data access feature without audit logging",
                        details="Feature accesses sensitive data but audit logging is disabled",
                        recommendation="Enable audit logging for security compliance",
                        fix_command="export PYNOMALY_AUDIT_LOGGING=true",
                    )
                )

        # Check network requirements
        network_reqs = requirements.get("network_requirements", [])
        if "outbound_https" in network_reqs:
            # In a real implementation, you'd check network policies
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.SECURITY,
                    severity=ValidationSeverity.INFO,
                    feature_name=feature_name,
                    message="Feature requires outbound HTTPS access",
                    details="Ensure firewall allows outbound HTTPS connections",
                    recommendation="Configure network policies to allow required connections",
                )
            )

        return issues

    def _validate_environment_compatibility(
        self, feature_name: str
    ) -> list[ValidationIssue]:
        """Validate feature compatibility with current environment."""
        issues = []
        current_env = automl_manager.get_environment()

        if not environment_configurator.is_feature_allowed_in_environment(
            feature_name, current_env
        ):
            fallback = environment_configurator.get_feature_fallback(
                feature_name, current_env
            )
            severity = (
                ValidationSeverity.ERROR if not fallback else ValidationSeverity.WARNING
            )

            message = f"Feature not allowed in {current_env.value} environment"
            recommendation = (
                f"Use fallback feature '{fallback}'"
                if fallback
                else f"Disable feature in {current_env.value}"
            )

            issues.append(
                ValidationIssue(
                    category=ValidationCategory.ENVIRONMENT,
                    severity=severity,
                    feature_name=feature_name,
                    message=message,
                    details=f"Feature '{feature_name}' is restricted in {current_env.value} environment",
                    recommendation=recommendation,
                    fix_command=f"export PYNOMALY_{feature_name.upper()}=false"
                    if not fallback
                    else None,
                )
            )

        return issues

    def _validate_configuration(self, feature_name: str) -> list[ValidationIssue]:
        """Validate feature-specific configuration."""
        issues = []

        # Check environment variables for feature configuration
        env_var = f"PYNOMALY_{feature_name.upper()}"
        env_value = os.getenv(env_var)

        if env_value is None:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.CONFIGURATION,
                    severity=ValidationSeverity.INFO,
                    feature_name=feature_name,
                    message="Feature configuration not explicitly set",
                    details=f"Environment variable {env_var} not set, using default",
                    recommendation="Explicitly set feature configuration for clarity",
                    fix_command=f"export {env_var}=true",
                )
            )

        return issues

    def _validate_feature_interactions(
        self, enabled_features: dict[str, bool]
    ) -> list[ValidationIssue]:
        """Validate interactions between enabled features."""
        issues = []

        enabled_feature_names = [
            name for name, enabled in enabled_features.items() if enabled
        ]

        # Check for conflicting features
        conflicts = [
            ("automl_distributed_search", "automl_neural_architecture_search"),
        ]

        for feature1, feature2 in conflicts:
            if feature1 in enabled_feature_names and feature2 in enabled_feature_names:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.COMPATIBILITY,
                        severity=ValidationSeverity.WARNING,
                        feature_name=f"{feature1}, {feature2}",
                        message="Conflicting features enabled simultaneously",
                        details=f"Features '{feature1}' and '{feature2}' may compete for resources",
                        recommendation="Consider disabling one of the conflicting features",
                    )
                )

        # Check resource combinations
        high_resource_features = [
            name
            for name in enabled_feature_names
            if self.performance_profiles.get(name, {}).get("memory_usage")
            in ["high", "very_high"]
        ]

        if len(high_resource_features) > 2:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.RESOURCE,
                    severity=ValidationSeverity.WARNING,
                    feature_name=", ".join(high_resource_features),
                    message="Multiple high-resource features enabled",
                    details=f"Features {high_resource_features} all require significant resources",
                    recommendation="Monitor system performance and consider limiting concurrent usage",
                )
            )

        return issues

    def _validate_environment_compliance(self) -> list[ValidationIssue]:
        """Validate overall environment compliance."""
        issues = []

        compliance = environment_configurator.validate_environment_compliance()

        for violation in compliance.get("violations", []):
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.ENVIRONMENT,
                    severity=ValidationSeverity.ERROR,
                    feature_name="environment",
                    message="Environment compliance violation",
                    details=violation,
                    recommendation="Address environment configuration issues",
                )
            )

        for warning in compliance.get("warnings", []):
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.ENVIRONMENT,
                    severity=ValidationSeverity.WARNING,
                    feature_name="environment",
                    message="Environment configuration warning",
                    details=warning,
                    recommendation="Review environment configuration",
                )
            )

        return issues

    def _generate_recommendations(self, issues: list[ValidationIssue]) -> list[str]:
        """Generate overall recommendations based on validation issues."""
        recommendations = []

        critical_count = len(
            [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        )
        error_count = len([i for i in issues if i.severity == ValidationSeverity.ERROR])
        warning_count = len(
            [i for i in issues if i.severity == ValidationSeverity.WARNING]
        )

        if critical_count > 0:
            recommendations.append(
                "🚨 CRITICAL: Address critical issues immediately before using AutoML features"
            )

        if error_count > 0:
            recommendations.append(
                "❌ Fix all error-level issues for reliable AutoML operation"
            )

        if warning_count > 0:
            recommendations.append("⚠️ Review warnings for optimal AutoML performance")

        if not issues:
            recommendations.append("✅ All AutoML features validated successfully")

        # Specific recommendations based on issue patterns
        dependency_issues = [
            i for i in issues if i.category == ValidationCategory.DEPENDENCY
        ]
        if dependency_issues:
            recommendations.append(
                "📦 Install missing packages and enable required dependencies"
            )

        performance_issues = [
            i for i in issues if i.category == ValidationCategory.PERFORMANCE
        ]
        if performance_issues:
            recommendations.append(
                "⚡ Consider system upgrades for better AutoML performance"
            )

        security_issues = [
            i for i in issues if i.category == ValidationCategory.SECURITY
        ]
        if security_issues:
            recommendations.append("🔒 Review and implement security requirements")

        return recommendations

    def validate_production_readiness(self) -> dict[str, Any]:
        """Validate production readiness of AutoML configuration."""
        report = self.validate_all_features()

        # Production-specific criteria
        production_criteria = {
            "no_critical_issues": report.summary["critical"] == 0,
            "no_error_issues": report.summary["errors"] == 0,
            "minimal_warnings": report.summary["warnings"] <= 2,
            "environment_compliant": automl_manager.get_environment().value
            in ["production", "staging"],
            "security_validated": len(
                [i for i in report.issues if i.category == ValidationCategory.SECURITY]
            )
            == 0,
            "dependencies_met": len(
                [
                    i
                    for i in report.issues
                    if i.category == ValidationCategory.DEPENDENCY
                ]
            )
            == 0,
        }

        readiness_score = sum(production_criteria.values()) / len(production_criteria)

        readiness_status = "ready" if readiness_score >= 0.8 else "not_ready"
        if readiness_score >= 0.6:
            readiness_status = "ready_with_warnings"

        return {
            "production_ready": readiness_status == "ready",
            "readiness_status": readiness_status,
            "readiness_score": readiness_score,
            "criteria": production_criteria,
            "validation_report": report,
            "blocking_issues": [
                i
                for i in report.issues
                if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]
            ],
            "recommendations": report.recommendations,
        }

    def generate_validation_summary(self, report: ValidationReport) -> str:
        """Generate a human-readable validation summary."""
        summary_lines = [
            "AutoML Feature Validation Report",
            f"Environment: {report.environment}",
            f"Features Checked: {report.total_features_checked}",
            f"Overall Status: {report.overall_status.upper()}",
            "",
            "Issues Summary:",
            f"  Critical: {report.summary['critical']}",
            f"  Errors: {report.summary['errors']}",
            f"  Warnings: {report.summary['warnings']}",
            f"  Info: {report.summary['info']}",
            "",
        ]

        if report.issues:
            summary_lines.append("Top Issues:")
            for issue in sorted(report.issues, key=lambda x: x.severity.value)[:5]:
                summary_lines.append(
                    f"  [{issue.severity.value.upper()}] {issue.feature_name}: {issue.message}"
                )
            summary_lines.append("")

        if report.recommendations:
            summary_lines.append("Recommendations:")
            for rec in report.recommendations:
                summary_lines.append(f"  • {rec}")

        return "\n".join(summary_lines)


# Global validator instance
automl_validator = AutoMLFeatureValidator()


# Convenience functions
def validate_automl_features() -> ValidationReport:
    """Run comprehensive AutoML feature validation."""
    return automl_validator.validate_all_features()


def check_production_readiness() -> dict[str, Any]:
    """Check if AutoML configuration is production-ready."""
    return automl_validator.validate_production_readiness()


def get_validation_summary() -> str:
    """Get a human-readable validation summary."""
    report = validate_automl_features()
    return automl_validator.generate_validation_summary(report)


def validate_feature(feature_name: str) -> list[ValidationIssue]:
    """Validate a specific AutoML feature."""
    return automl_validator._validate_single_feature(feature_name)
