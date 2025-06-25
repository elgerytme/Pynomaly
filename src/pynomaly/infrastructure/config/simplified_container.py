"""Simplified dependency injection container using service registry pattern."""

from __future__ import annotations

from pathlib import Path

from dependency_injector import containers, providers

from pynomaly.application.services import (
    DetectionService,
    EnsembleService,
    ExperimentTrackingService,
    ModelPersistenceService,
)
from pynomaly.application.use_cases import (
    DetectAnomaliesUseCase,
    EvaluateModelUseCase,
    ExplainAnomalyUseCase,
    TrainDetectorUseCase,
)
from pynomaly.domain.services import (
    AnomalyScorer,
    EnsembleAggregator,
    FeatureValidator,
    ThresholdCalculator,
)
from pynomaly.infrastructure.config.feature_flags import FeatureFlagManager
from pynomaly.infrastructure.config.service_registry import (
    RepositoryFactory,
    ServiceGroupFactory,
    ServiceRegistry,
    register_all_services,
)
from pynomaly.infrastructure.config.settings import Settings
from pynomaly.infrastructure.repositories import (
    FileDatasetRepository,
    FileDetectorRepository,
    FileResultRepository,
    InMemoryDatasetRepository,
    InMemoryDetectorRepository,
    InMemoryResultRepository,
)
from pynomaly.infrastructure.repositories.async_wrappers import (
    AsyncDatasetRepositoryWrapper,
    AsyncDetectionResultRepositoryWrapper,
    AsyncDetectorRepositoryWrapper,
)


class SimplifiedContainer(containers.DeclarativeContainer):
    """Simplified dependency injection container using service registry pattern."""

    # Core configuration
    config = providers.Singleton(Settings)
    feature_flag_manager = providers.Singleton(FeatureFlagManager)

    # Initialize service registry and factories
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._service_registry = ServiceRegistry()
        register_all_services(self._service_registry)

        # Initialize factories
        self._setup_factories()
        self._setup_repositories()
        self._setup_optional_services()

    def _setup_factories(self):
        """Setup factory instances."""
        config_instance = Settings()
        self._repository_factory = RepositoryFactory(config_instance)
        self._service_group_factory = ServiceGroupFactory(
            self._service_registry, config_instance
        )

    def _setup_repositories(self):
        """Setup repositories using factory pattern."""
        Settings()

        # Repository class mappings
        repository_classes = {
            # Detector repositories
            "memory_detector": InMemoryDetectorRepository,
            "file_detector": FileDetectorRepository,
            # Dataset repositories
            "memory_dataset": InMemoryDatasetRepository,
            "file_dataset": FileDatasetRepository,
            # Result repositories
            "memory_result": InMemoryResultRepository,
            "file_result": FileResultRepository,
        }

        # Add database repositories if available
        try:
            from pynomaly.infrastructure.persistence import (
                DatabaseDatasetRepository,
                DatabaseDetectionResultRepository,
                DatabaseDetectorRepository,
            )

            repository_classes.update(
                {
                    "database_detector": DatabaseDetectorRepository,
                    "database_dataset": DatabaseDatasetRepository,
                    "database_result": DatabaseDetectionResultRepository,
                }
            )
        except ImportError:
            pass

        # Create repository providers
        self.detector_repository = providers.Singleton(
            lambda: self._repository_factory.create_repository(
                "detector", repository_classes
            )
        )
        self.dataset_repository = providers.Singleton(
            lambda: self._repository_factory.create_repository(
                "dataset", repository_classes
            )
        )
        self.result_repository = providers.Singleton(
            lambda: self._repository_factory.create_repository(
                "result", repository_classes
            )
        )

        # Async repository wrappers
        self.async_detector_repository = providers.Singleton(
            AsyncDetectorRepositoryWrapper, sync_repository=self.detector_repository
        )
        self.async_dataset_repository = providers.Singleton(
            AsyncDatasetRepositoryWrapper, sync_repository=self.dataset_repository
        )
        self.async_result_repository = providers.Singleton(
            AsyncDetectionResultRepositoryWrapper,
            sync_repository=self.result_repository,
        )

    def _setup_optional_services(self):
        """Setup optional services using service registry."""
        # Data loaders
        data_loader_providers = (
            self._service_group_factory.create_data_loader_providers()
        )
        for name, provider in data_loader_providers.items():
            setattr(self, name, provider)

        # Algorithm adapters
        adapter_providers = self._service_group_factory.create_adapter_providers()
        for name, provider in adapter_providers.items():
            setattr(self, name, provider)

        # Authentication services
        auth_providers = self._service_group_factory.create_auth_providers()
        for name, provider in auth_providers.items():
            setattr(self, name, provider)

        # Cache services
        cache_providers = self._service_group_factory.create_cache_providers()
        for name, provider in cache_providers.items():
            setattr(self, name, provider)

        # Monitoring services
        monitoring_providers = self._service_group_factory.create_monitoring_providers()
        for name, provider in monitoring_providers.items():
            setattr(self, name, provider)

        # AutoML service
        if self._service_registry.is_available("automl_service"):
            self.automl_service = self._service_registry.create_singleton_provider(
                "automl_service",
                detector_repository=self.async_detector_repository,
                dataset_repository=self.async_dataset_repository,
                adapter_registry=providers.Object("adapter_registry"),
                max_optimization_time=3600,
                n_trials=100,
                cv_folds=3,
                random_state=42,
            )

        # Explainability services
        if self._service_registry.is_available("explainability_service"):
            self.domain_explainability_service = (
                self._service_registry.create_singleton_provider(
                    "explainability_service"
                )
            )

            # Create explainer providers
            if self._service_registry.is_available("shap_explainer"):
                self.shap_explainer = self._service_registry.create_singleton_provider(
                    "shap_explainer"
                )

            if self._service_registry.is_available("lime_explainer"):
                self.lime_explainer = self._service_registry.create_singleton_provider(
                    "lime_explainer"
                )

    # Domain services
    anomaly_scorer = providers.Singleton(AnomalyScorer)
    threshold_calculator = providers.Singleton(ThresholdCalculator)
    feature_validator = providers.Singleton(FeatureValidator)
    ensemble_aggregator = providers.Singleton(EnsembleAggregator)

    # Application services
    detection_service = providers.Singleton(
        DetectionService,
        detector_repository=detector_repository,
        result_repository=result_repository,
        anomaly_scorer=anomaly_scorer,
        threshold_calculator=threshold_calculator,
    )

    ensemble_service = providers.Singleton(
        EnsembleService,
        detector_repository=detector_repository,
        ensemble_aggregator=ensemble_aggregator,
        anomaly_scorer=anomaly_scorer,
    )

    model_persistence_service = providers.Singleton(
        ModelPersistenceService,
        detector_repository=detector_repository,
        storage_path=config.provided.model_storage_path,
    )

    experiment_tracking_service = providers.Singleton(
        ExperimentTrackingService, tracking_path=config.provided.experiment_storage_path
    )

    # Use cases
    detect_anomalies_use_case = providers.Factory(
        DetectAnomaliesUseCase,
        detector_repository=detector_repository,
        feature_validator=feature_validator,
    )

    train_detector_use_case = providers.Factory(
        TrainDetectorUseCase,
        detector_repository=detector_repository,
        feature_validator=feature_validator,
        min_samples=10,
    )

    evaluate_model_use_case = providers.Factory(
        EvaluateModelUseCase, detector_repository=detector_repository
    )

    # Conditional use cases
    def _create_explainability_use_case(self):
        """Create explainability use case if service is available."""
        if hasattr(self, "domain_explainability_service"):
            return providers.Factory(
                ExplainAnomalyUseCase,
                explainability_service=self.domain_explainability_service,
            )
        return None

    @property
    def explain_anomaly_use_case(self):
        """Get explainability use case if available."""
        return self._create_explainability_use_case()

    def get_service_availability(self) -> dict:
        """Get availability status of all optional services."""
        return self._service_registry.get_available_services()

    def get_available_adapters(self) -> list:
        """Get list of available algorithm adapters."""
        adapters = []
        for adapter_name in [
            "pyod_adapter",
            "sklearn_adapter",
            "pygod_adapter",
            "pytorch_adapter",
            "tensorflow_adapter",
            "jax_adapter",
        ]:
            if hasattr(self, adapter_name):
                adapters.append(adapter_name)
        return adapters

    def get_available_data_loaders(self) -> list:
        """Get list of available data loaders."""
        loaders = []
        for loader_name in [
            "csv_loader",
            "parquet_loader",
            "polars_loader",
            "arrow_loader",
            "spark_loader",
        ]:
            if hasattr(self, loader_name):
                loaders.append(loader_name)
        return loaders


class TestContainer(SimplifiedContainer):
    """Container for testing with test-specific overrides."""

    # Override settings for testing
    config = providers.Singleton(
        Settings,
        storage_path=Path("./test_storage"),
        model_storage_path=Path("./test_storage/models"),
        experiment_storage_path=Path("./test_storage/experiments"),
        debug=True,
        auth_enabled=False,
    )


def create_simplified_container(testing: bool = False) -> SimplifiedContainer:
    """Create and configure the simplified DI container.

    Args:
        testing: Whether to create a test container

    Returns:
        Configured container
    """
    if testing:
        container = TestContainer()
    else:
        container = SimplifiedContainer()

    # Wire the container to modules that need it
    container.wire(
        modules=[
            "pynomaly.presentation.api",
            "pynomaly.presentation.cli",
            "pynomaly.presentation.web",
        ]
    )

    return container


# Maintain backward compatibility
Container = SimplifiedContainer
create_container = create_simplified_container
