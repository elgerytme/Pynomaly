"""Feature flag system for controlled feature rollout.

This module provides a centralized feature flag system that allows toggling
of advanced features without breaking core functionality. Essential for
maintaining simplicity while enabling controlled complexity reintroduction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

try:
    from pydantic import Field
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings, Field


class FeatureStage(Enum):
    """Development stages for feature maturity."""

    EXPERIMENTAL = "experimental"
    BETA = "beta"
    STABLE = "stable"
    DEPRECATED = "deprecated"


class FeatureCategory(Enum):
    """Categories for organizing features."""

    CORE = "core"
    ANALYTICS = "analytics"
    INTEGRATIONS = "integrations"
    ENTERPRISE = "enterprise"
    AUTOMATION = "automation"
    PERFORMANCE = "performance"


@dataclass
class FeatureDefinition:
    """Definition of a feature flag with metadata."""

    name: str
    description: str
    category: FeatureCategory
    stage: FeatureStage = FeatureStage.EXPERIMENTAL
    default_enabled: bool = False
    dependencies: set[str] = field(default_factory=set)
    conflicts: set[str] = field(default_factory=set)
    min_python_version: str | None = None
    required_packages: set[str] = field(default_factory=set)

    def __post_init__(self):
        """Validate feature definition."""
        if self.stage == FeatureStage.STABLE and not self.default_enabled:
            raise ValueError(f"Stable feature {self.name} should be enabled by default")


class FeatureFlags(BaseSettings):
    """Centralized feature flag configuration."""

    # Phase 2: Core Enhancement Features
    algorithm_optimization: bool = Field(
        default=True, description="Enable algorithm performance optimization"
    )

    memory_efficiency: bool = Field(
        default=True, description="Enable memory-efficient streaming processing"
    )

    performance_monitoring: bool = Field(
        default=True, description="Enable real-time performance monitoring"
    )

    ensemble_intelligence: bool = Field(
        default=False, description="Enable smart ensemble composition"
    )

    # Phase 2: User Experience Features
    cli_simplification: bool = Field(
        default=True, description="Enable simplified CLI workflows"
    )

    interactive_guidance: bool = Field(
        default=True, description="Enable step-by-step workflow guidance"
    )

    error_recovery: bool = Field(
        default=True, description="Enable advanced error recovery"
    )

    # Phase 2: Quality Infrastructure
    complexity_monitoring: bool = Field(
        default=True,  # Always enabled for maintainability
        description="Enable automated complexity monitoring",
    )

    quality_gates: bool = Field(
        default=True,  # Always enabled for code quality
        description="Enable quality gates for commits",
    )

    documentation_automation: bool = Field(
        default=False, description="Enable automated documentation generation"
    )

    # Phase 3: Enterprise Features (Disabled by default)
    jwt_authentication: bool = Field(
        default=False, description="Enable JWT-based authentication"
    )

    data_encryption: bool = Field(
        default=False, description="Enable data encryption at rest and in transit"
    )

    audit_logging: bool = Field(
        default=False, description="Enable comprehensive audit logging"
    )

    explainability_integration: bool = Field(
        default=False, description="Enable SHAP/LIME explainability features"
    )

    statistical_validation: bool = Field(
        default=False, description="Enable advanced statistical validation"
    )

    # Phase 3: Strategic Integrations (Disabled by default)
    database_connectivity: bool = Field(
        default=False, description="Enable PostgreSQL/MySQL connectivity"
    )

    cloud_storage: bool = Field(
        default=False, description="Enable S3/Azure Blob storage integration"
    )

    monitoring_integration: bool = Field(
        default=False, description="Enable Prometheus/Grafana integration"
    )

    # Phase 3A: Advanced ML/AI Features (Enabled for development)
    advanced_automl: bool = Field(
        default=True,
        description="Enable advanced AutoML with multi-objective optimization",
    )

    meta_learning: bool = Field(
        default=True, description="Enable meta-learning from optimization history"
    )

    ensemble_optimization: bool = Field(
        default=True, description="Enable advanced ensemble optimization"
    )

    deep_learning_adapters: bool = Field(
        default=False, description="Enable PyTorch/TensorFlow deep learning adapters"
    )

    deep_learning: bool = Field(
        default=True,
        description="Enable deep learning anomaly detection with PyTorch, TensorFlow, and JAX",
    )

    advanced_explainability: bool = Field(
        default=True, description="Enable advanced explainable AI features"
    )

    intelligent_selection: bool = Field(
        default=True,
        description="Enable intelligent algorithm selection with learning capabilities",
    )

    # Phase 3C: User Experience Features (Enabled for development)
    real_time_dashboards: bool = Field(
        default=True, description="Enable real-time analytics dashboards"
    )

    progressive_web_app: bool = Field(
        default=True, description="Enable PWA features with offline capabilities"
    )

    mobile_interface: bool = Field(
        default=True, description="Enable mobile-responsive interface"
    )

    business_intelligence: bool = Field(
        default=True, description="Enable advanced BI reporting and analytics"
    )

    # Phase 4: Advanced Capabilities (Disabled by default)
    automl_framework: bool = Field(
        default=False,
        description="Enable legacy AutoML framework (superseded by advanced_automl)",
    )

    streaming_analytics: bool = Field(
        default=False, description="Enable real-time streaming analytics"
    )

    business_intelligence: bool = Field(
        default=False, description="Enable BI integrations (PowerBI, Sheets)"
    )

    distributed_computing: bool = Field(
        default=False, description="Enable distributed computing capabilities"
    )

    graph_analytics: bool = Field(
        default=False, description="Enable graph anomaly detection"
    )

    multi_modal_detection: bool = Field(
        default=False, description="Enable text/image anomaly detection"
    )

    # Experimental ML Features
    ml_severity_classifier: bool = Field(
        default=False, description="Enable ML severity classifier with XGBoost/LightGBM models"
    )

    # Phase 5: Production Hardening (Selective enabling)
    health_monitoring: bool = Field(
        default=True,  # Essential for production
        description="Enable comprehensive health monitoring",
    )

    performance_optimization: bool = Field(
        default=False, description="Enable production performance optimization"
    )

    backup_recovery: bool = Field(
        default=False, description="Enable automated backup and recovery"
    )

    sso_integration: bool = Field(
        default=False, description="Enable SSO (SAML/OAuth) integration"
    )

    compliance_features: bool = Field(
        default=False, description="Enable GDPR/HIPAA compliance features"
    )

    multi_tenancy: bool = Field(
        default=False, description="Enable multi-tenant isolation"
    )

    class Config:
        """Pydantic configuration."""

        env_prefix = "PYNOMALY_"
        case_sensitive = False


class FeatureFlagManager:
    """Manager for feature flag operations and validation."""

    def __init__(self):
        """Initialize feature flag manager."""
        self.flags = FeatureFlags()
        self._feature_definitions = self._load_feature_definitions()

    def _load_feature_definitions(self) -> dict[str, FeatureDefinition]:
        """Load feature definitions with metadata."""
        return {
            # Phase 2: Core Enhancement
            "algorithm_optimization": FeatureDefinition(
                name="algorithm_optimization",
                description="Algorithm performance optimization",
                category=FeatureCategory.PERFORMANCE,
                stage=FeatureStage.BETA,
                required_packages={"pyod", "scikit-learn"},
            ),
            "memory_efficiency": FeatureDefinition(
                name="memory_efficiency",
                description="Memory-efficient data processing",
                category=FeatureCategory.PERFORMANCE,
                stage=FeatureStage.BETA,
                dependencies={"algorithm_optimization"},
            ),
            "performance_monitoring": FeatureDefinition(
                name="performance_monitoring",
                description="Real-time performance tracking",
                category=FeatureCategory.ANALYTICS,
                stage=FeatureStage.BETA,
                required_packages={"prometheus-client"},
            ),
            "ensemble_intelligence": FeatureDefinition(
                name="ensemble_intelligence",
                description="Smart ensemble composition",
                category=FeatureCategory.AUTOMATION,
                stage=FeatureStage.EXPERIMENTAL,
                dependencies={"algorithm_optimization"},
            ),
            # Phase 3: Enterprise Features
            "jwt_authentication": FeatureDefinition(
                name="jwt_authentication",
                description="JWT-based authentication",
                category=FeatureCategory.ENTERPRISE,
                stage=FeatureStage.EXPERIMENTAL,
                required_packages={"pyjwt", "passlib"},
            ),
            "explainability_integration": FeatureDefinition(
                name="explainability_integration",
                description="SHAP/LIME model explainability",
                category=FeatureCategory.ANALYTICS,
                stage=FeatureStage.EXPERIMENTAL,
                required_packages={"shap", "lime"},
            ),
            # Phase 3A: Advanced ML/AI Features
            "advanced_automl": FeatureDefinition(
                name="advanced_automl",
                description="Advanced AutoML with multi-objective optimization",
                category=FeatureCategory.AUTOMATION,
                stage=FeatureStage.BETA,
                dependencies={"algorithm_optimization", "performance_monitoring"},
                required_packages={"optuna"},
            ),
            "meta_learning": FeatureDefinition(
                name="meta_learning",
                description="Meta-learning from optimization history",
                category=FeatureCategory.AUTOMATION,
                stage=FeatureStage.BETA,
                dependencies={"advanced_automl"},
            ),
            "ensemble_optimization": FeatureDefinition(
                name="ensemble_optimization",
                description="Advanced ensemble optimization",
                category=FeatureCategory.AUTOMATION,
                stage=FeatureStage.BETA,
                dependencies={"algorithm_optimization"},
            ),
            "deep_learning_adapters": FeatureDefinition(
                name="deep_learning_adapters",
                description="PyTorch/TensorFlow deep learning adapters",
                category=FeatureCategory.INTEGRATIONS,
                stage=FeatureStage.EXPERIMENTAL,
                dependencies={"algorithm_optimization"},
                required_packages={"torch", "tensorflow"},
            ),
            "deep_learning": FeatureDefinition(
                name="deep_learning",
                description="Deep learning anomaly detection with PyTorch, TensorFlow, and JAX",
                category=FeatureCategory.INTEGRATIONS,
                stage=FeatureStage.BETA,
                dependencies={"algorithm_optimization"},
            ),
            "advanced_explainability": FeatureDefinition(
                name="advanced_explainability",
                description="Advanced explainable AI features",
                category=FeatureCategory.ANALYTICS,
                stage=FeatureStage.BETA,
                required_packages={"shap", "lime"},
            ),
            # Phase 3C: User Experience Features
            "real_time_dashboards": FeatureDefinition(
                name="real_time_dashboards",
                description="Real-time analytics dashboards",
                category=FeatureCategory.ANALYTICS,
                stage=FeatureStage.BETA,
                dependencies={"performance_monitoring"},
            ),
            "progressive_web_app": FeatureDefinition(
                name="progressive_web_app",
                description="PWA features with offline capabilities",
                category=FeatureCategory.INTEGRATIONS,
                stage=FeatureStage.BETA,
            ),
            "mobile_interface": FeatureDefinition(
                name="mobile_interface",
                description="Mobile-responsive interface",
                category=FeatureCategory.CORE,
                stage=FeatureStage.BETA,
                dependencies={"progressive_web_app"},
            ),
            "business_intelligence": FeatureDefinition(
                name="business_intelligence",
                description="Advanced BI reporting and analytics",
                category=FeatureCategory.ANALYTICS,
                stage=FeatureStage.BETA,
                dependencies={"real_time_dashboards"},
            ),
            # Phase 4: Advanced Capabilities
            "automl_framework": FeatureDefinition(
                name="automl_framework",
                description="Legacy AutoML framework (superseded by advanced_automl)",
                category=FeatureCategory.AUTOMATION,
                stage=FeatureStage.DEPRECATED,
                dependencies={"algorithm_optimization", "performance_monitoring"},
                required_packages={"optuna", "hyperopt"},
            ),
            "streaming_analytics": FeatureDefinition(
                name="streaming_analytics",
                description="Real-time streaming data processing",
                category=FeatureCategory.INTEGRATIONS,
                stage=FeatureStage.EXPERIMENTAL,
                dependencies={"memory_efficiency", "performance_monitoring"},
                conflicts={"distributed_computing"},  # Prevent both at once initially
                required_packages={"kafka-python", "redis"},
            ),
            "distributed_computing": FeatureDefinition(
                name="distributed_computing",
                description="Distributed processing capabilities",
                category=FeatureCategory.PERFORMANCE,
                stage=FeatureStage.EXPERIMENTAL,
                dependencies={"algorithm_optimization", "performance_monitoring"},
                conflicts={"streaming_analytics"},  # Prevent both at once initially
                required_packages={"dask", "ray"},
            ),
            "ml_severity_classifier": FeatureDefinition(
                name="ml_severity_classifier",
                description="ML severity classifier with XGBoost/LightGBM models",
                category=FeatureCategory.AUTOMATION,
                stage=FeatureStage.EXPERIMENTAL,
                dependencies={"algorithm_optimization"},
                required_packages={"xgboost", "lightgbm"},
            ),
        }

    def is_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled."""
        return getattr(self.flags, feature_name, False)

    def get_enabled_features(self) -> set[str]:
        """Get all currently enabled features."""
        enabled = set()
        for field_name in self.flags.__fields__:
            if getattr(self.flags, field_name):
                enabled.add(field_name)
        return enabled

    def validate_feature_compatibility(self) -> dict[str, list]:
        """Validate that enabled features are compatible."""
        issues = {"missing_dependencies": [], "conflicts": [], "missing_packages": []}
        enabled = self.get_enabled_features()

        for feature_name in enabled:
            if feature_name not in self._feature_definitions:
                continue

            definition = self._feature_definitions[feature_name]

            # Check dependencies
            for dependency in definition.dependencies:
                if dependency not in enabled:
                    issues["missing_dependencies"].append(
                        f"{feature_name} requires {dependency}"
                    )

            # Check conflicts
            for conflict in definition.conflicts:
                if conflict in enabled:
                    issues["conflicts"].append(
                        f"{feature_name} conflicts with {conflict}"
                    )

            # Check required packages (would need actual package checking in production)
            for package in definition.required_packages:
                try:
                    __import__(package.replace("-", "_"))
                except ImportError:
                    issues["missing_packages"].append(
                        f"{feature_name} requires package: {package}"
                    )

        return {k: v for k, v in issues.items() if v}

    def get_feature_info(self, feature_name: str) -> FeatureDefinition | None:
        """Get detailed information about a feature."""
        return self._feature_definitions.get(feature_name)

    def get_features_by_category(
        self, category: FeatureCategory
    ) -> dict[str, FeatureDefinition]:
        """Get all features in a specific category."""
        return {
            name: definition
            for name, definition in self._feature_definitions.items()
            if definition.category == category
        }

    def get_features_by_stage(
        self, stage: FeatureStage
    ) -> dict[str, FeatureDefinition]:
        """Get all features in a specific development stage."""
        return {
            name: definition
            for name, definition in self._feature_definitions.items()
            if definition.stage == stage
        }


# Global feature flag manager instance
feature_flags = FeatureFlagManager()


def require_feature(feature_name: str):
    """Decorator to require a feature flag for function execution."""
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not feature_flags.is_enabled(feature_name):
                raise RuntimeError(
                    f"Feature '{feature_name}' is not enabled. "
                    f"Enable it by setting PYNOMALY_{feature_name.upper()}=true"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def conditional_import(feature_name: str, module_name: str, fallback=None):
    """Conditionally import a module based on feature flag."""
    if feature_flags.is_enabled(feature_name):
        try:
            return __import__(module_name, fromlist=[""])
        except ImportError:
            if fallback is not None:
                return fallback
            raise
    return fallback


# Convenience functions for common feature checks
def is_algorithm_optimization_enabled() -> bool:
    """Check if algorithm optimization is enabled."""
    return feature_flags.is_enabled("algorithm_optimization")


def is_enterprise_features_enabled() -> bool:
    """Check if any enterprise features are enabled."""
    enterprise_features = feature_flags.get_features_by_category(
        FeatureCategory.ENTERPRISE
    )
    return any(feature_flags.is_enabled(name) for name in enterprise_features)


def is_advanced_analytics_enabled() -> bool:
    """Check if advanced analytics features are enabled."""
    return feature_flags.is_enabled(
        "explainability_integration"
    ) or feature_flags.is_enabled("statistical_validation")


def get_feature_flags() -> FeatureFlags:
    """Get the global feature flags instance."""
    global feature_flags
    return feature_flags
