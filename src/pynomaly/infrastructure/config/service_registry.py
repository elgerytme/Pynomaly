"""Service registry for managing optional dependencies and provider creation."""

from __future__ import annotations

import importlib
import logging
from typing import Any

from dependency_injector import providers

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """Registry for managing optional services and their availability."""

    def __init__(self):
        self._services: dict[str, tuple[type, dict[str, Any]]] = {}
        self._availability: dict[str, bool] = {}
        self._providers: dict[str, Any] = {}

    def register_service(self, name: str, import_path: str, **provider_kwargs) -> bool:
        """Register a service with optional import.

        Args:
            name: Service name for registry
            import_path: Full import path (e.g., 'package.module.ClassName')
            **provider_kwargs: Additional arguments for provider creation

        Returns:
            True if service is available, False otherwise
        """
        try:
            module_path, class_name = import_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            service_class = getattr(module, class_name)

            self._services[name] = (service_class, provider_kwargs)
            self._availability[name] = True
            logger.debug(f"Registered service: {name}")
            return True

        except (ImportError, AttributeError) as e:
            self._availability[name] = False
            logger.debug(f"Service {name} not available: {e}")
            return False

    def is_available(self, name: str) -> bool:
        """Check if a service is available."""
        return self._availability.get(name, False)

    def get_service_class(self, name: str) -> type | None:
        """Get the service class if available."""
        if not self.is_available(name):
            return None
        return self._services[name][0]

    def create_singleton_provider(
        self, name: str, **override_kwargs
    ) -> providers.Singleton | None:
        """Create a singleton provider for the service."""
        if not self.is_available(name):
            return None

        service_class, default_kwargs = self._services[name]
        kwargs = {**default_kwargs, **override_kwargs}
        return providers.Singleton(service_class, **kwargs)

    def create_factory_provider(
        self, name: str, **override_kwargs
    ) -> providers.Factory | None:
        """Create a factory provider for the service."""
        if not self.is_available(name):
            return None

        service_class, default_kwargs = self._services[name]
        kwargs = {**default_kwargs, **override_kwargs}
        return providers.Factory(service_class, **kwargs)

    def get_available_services(self) -> dict[str, bool]:
        """Get availability status of all registered services."""
        return self._availability.copy()


class RepositoryFactory:
    """Factory for creating repositories based on configuration."""

    def __init__(self, config):
        self.config = config

    def create_repository(
        self, repo_type: str, repository_classes: dict[str, type]
    ) -> Any:
        """Create repository based on configuration.

        Args:
            repo_type: Type of repository ('detector', 'dataset', 'result')
            repository_classes: Dict mapping class types to classes

        Returns:
            Repository instance
        """
        # Try database repository first if configured
        if (
            self.config.use_database_repositories
            and self.config.database_configured
            and f"database_{repo_type}" in repository_classes
        ):
            try:
                return self._create_database_repository(
                    repository_classes[f"database_{repo_type}"]
                )
            except Exception as e:
                logger.warning(
                    f"Database repository creation failed, falling back to file: {e}"
                )

        # Fall back to file repository
        file_repo_class = repository_classes.get(f"file_{repo_type}")
        if file_repo_class:
            return file_repo_class(self.config.storage_path)

        # Fall back to in-memory repository
        memory_repo_class = repository_classes.get(f"memory_{repo_type}")
        if memory_repo_class:
            return memory_repo_class()

        raise ValueError(f"No repository implementation found for type: {repo_type}")

    def _create_database_repository(self, repo_class: type) -> Any:
        """Create database repository with proper initialization."""
        from pynomaly.infrastructure.persistence import DatabaseManager

        db_manager = DatabaseManager(
            database_url=self.config.database_url,
            echo=self.config.database_echo or self.config.app.debug,
        )

        # Initialize database if needed
        try:
            from pynomaly.infrastructure.persistence.migrations import DatabaseMigrator

            migrator = DatabaseMigrator(db_manager)
            if not migrator.check_tables_exist():
                migrator.create_all_tables()
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}")

        return repo_class(db_manager.get_session)


class ServiceGroupFactory:
    """Factory for creating groups of related services."""

    def __init__(self, registry: ServiceRegistry, config):
        self.registry = registry
        self.config = config

    def create_auth_providers(self) -> dict[str, Any]:
        """Create authentication-related providers."""
        providers_dict = {}

        # JWT Auth Service
        jwt_provider = self.registry.create_singleton_provider(
            "jwt_auth_service",
            secret_key=self.config.secret_key,
            algorithm=self.config.jwt_algorithm,
            access_token_expire_minutes=self.config.jwt_expiration,
        )
        if jwt_provider:
            providers_dict["jwt_auth_service"] = jwt_provider

        # Permission Checker
        permission_provider = self.registry.create_singleton_provider(
            "permission_checker"
        )
        if permission_provider:
            providers_dict["permission_checker"] = permission_provider

        # Rate Limiter
        rate_limiter_provider = self.registry.create_singleton_provider(
            "rate_limiter", default_requests_per_minute=self.config.api_rate_limit
        )
        if rate_limiter_provider:
            providers_dict["rate_limiter"] = rate_limiter_provider

        return providers_dict

    def create_cache_providers(self) -> dict[str, Any]:
        """Create cache-related providers."""
        providers_dict = {}

        # Redis Cache
        redis_provider = self.registry.create_singleton_provider(
            "redis_cache",
            redis_url=self.config.redis_url,
            default_ttl=self.config.cache_ttl,
        )
        if redis_provider:
            providers_dict["redis_cache"] = redis_provider

        # Detector Cache Decorator
        if redis_provider and self.registry.is_available("detector_cache_decorator"):
            cache_decorator_provider = self.registry.create_singleton_provider(
                "detector_cache_decorator", cache=redis_provider
            )
            if cache_decorator_provider:
                providers_dict["detector_cache_decorator"] = cache_decorator_provider

        return providers_dict

    def create_monitoring_providers(self) -> dict[str, Any]:
        """Create monitoring-related providers."""
        providers_dict = {}

        # Telemetry Service
        telemetry_provider = self.registry.create_singleton_provider(
            "telemetry_service",
            service_name="pynomaly",
            environment=self.config.app.environment,
            otlp_endpoint=self.config.monitoring.otlp_endpoint,
        )
        if telemetry_provider:
            providers_dict["telemetry_service"] = telemetry_provider

        # Health Service
        health_provider = self.registry.create_singleton_provider(
            "health_service", max_history=100
        )
        if health_provider:
            providers_dict["health_service"] = health_provider

        return providers_dict

    def create_data_loader_providers(self) -> dict[str, Any]:
        """Create data loader providers."""
        providers_dict = {}

        # Standard loaders (always available)
        providers_dict["csv_loader"] = providers.Factory(
            self.registry.get_service_class("csv_loader"),
            delimiter=",",
            encoding="utf-8",
        )

        providers_dict["parquet_loader"] = providers.Factory(
            self.registry.get_service_class("parquet_loader"), use_pyarrow=True
        )

        # Optional high-performance loaders
        for loader_name, config in [
            ("polars_loader", {"lazy": True, "streaming": False}),
            ("arrow_loader", {"use_threads": True}),
            ("spark_loader", {"app_name": "Pynomaly", "master": "local[*]"}),
        ]:
            loader_provider = self.registry.create_factory_provider(
                loader_name, **config
            )
            if loader_provider:
                providers_dict[loader_name] = loader_provider

        return providers_dict

    def create_adapter_providers(self) -> dict[str, Any]:
        """Create algorithm adapter providers."""
        providers_dict = {}

        # Standard adapters (always available)
        for adapter_name in ["pyod_adapter", "sklearn_adapter"]:
            adapter_class = self.registry.get_service_class(adapter_name)
            if adapter_class:
                providers_dict[adapter_name] = providers.Factory(adapter_class)

        # Optional adapters
        for adapter_name in [
            "pygod_adapter",
            "pytorch_adapter",
            "tensorflow_adapter",
            "jax_adapter",
        ]:
            adapter_provider = self.registry.create_factory_provider(adapter_name)
            if adapter_provider:
                providers_dict[adapter_name] = adapter_provider

        return providers_dict


def register_all_services(registry: ServiceRegistry) -> ServiceRegistry:
    """Register all services with the registry."""

    # Core data loaders
    registry.register_service(
        "csv_loader", "pynomaly.infrastructure.data_loaders.CSVLoader"
    )
    registry.register_service(
        "parquet_loader", "pynomaly.infrastructure.data_loaders.ParquetLoader"
    )

    # Optional data loaders
    registry.register_service(
        "polars_loader", "pynomaly.infrastructure.data_loaders.PolarsLoader"
    )
    registry.register_service(
        "arrow_loader", "pynomaly.infrastructure.data_loaders.ArrowLoader"
    )
    registry.register_service(
        "spark_loader", "pynomaly.infrastructure.data_loaders.SparkLoader"
    )

    # Algorithm adapters
    registry.register_service(
        "pyod_adapter", "pynomaly.infrastructure.adapters.PyODAdapter"
    )
    registry.register_service(
        "sklearn_adapter", "pynomaly.infrastructure.adapters.SklearnAdapter"
    )
    registry.register_service(
        "pygod_adapter", "pynomaly.infrastructure.adapters.PyGODAdapter"
    )
    registry.register_service(
        "pytorch_adapter", "pynomaly.infrastructure.adapters.PyTorchAdapter"
    )
    registry.register_service(
        "tensorflow_adapter", "pynomaly.infrastructure.adapters.TensorFlowAdapter"
    )
    registry.register_service(
        "jax_adapter", "pynomaly.infrastructure.adapters.JAXAdapter"
    )

    # Authentication services
    registry.register_service(
        "jwt_auth_service", "pynomaly.infrastructure.auth.JWTAuthService"
    )
    registry.register_service(
        "permission_checker", "pynomaly.infrastructure.auth.PermissionChecker"
    )
    registry.register_service(
        "rate_limiter", "pynomaly.infrastructure.auth.RateLimiter"
    )

    # Cache services
    registry.register_service("redis_cache", "pynomaly.infrastructure.cache.RedisCache")
    registry.register_service(
        "detector_cache_decorator",
        "pynomaly.infrastructure.cache.DetectorCacheDecorator",
    )

    # Monitoring services
    registry.register_service(
        "telemetry_service", "pynomaly.infrastructure.monitoring.TelemetryService"
    )
    registry.register_service(
        "health_service", "pynomaly.infrastructure.monitoring.HealthService"
    )

    # Database services
    registry.register_service(
        "database_manager", "pynomaly.infrastructure.persistence.DatabaseManager"
    )

    # AutoML services
    registry.register_service(
        "automl_service", "pynomaly.application.services.AutoMLService"
    )

    # Explainability services
    registry.register_service(
        "explainability_service", "pynomaly.domain.services.ExplainabilityService"
    )
    registry.register_service(
        "shap_explainer", "pynomaly.infrastructure.explainers.SHAPExplainer"
    )
    registry.register_service(
        "lime_explainer", "pynomaly.infrastructure.explainers.LIMEExplainer"
    )

    return registry
