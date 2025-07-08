"""Simplified dependency injection container using service registry pattern."""

from __future__ import annotations

import logging
from pathlib import Path

from dependency_injector import containers, providers

# Core imports (always available)
from pynomaly.application.services import (
    AnomalyClassificationService,
    DetectionService,
    EnsembleService,
    ExperimentTrackingService,
    ModelPersistenceService,
)
from pynomaly.application.use_cases import (
    DetectAnomaliesUseCase,
    EvaluateModelUseCase,
    TrainDetectorUseCase,
)
from pynomaly.domain.services import (
    AnomalyScorer,
    EnsembleAggregator,
    FeatureValidator,
    ThresholdCalculator,
    DefaultSeverityClassifier,
    DefaultTypeClassifier,
)
from pynomaly.infrastructure.config.feature_flags import FeatureFlagManager
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

logger = logging.getLogger(__name__)


class OptionalServiceManager:
    """Manages optional service creation with graceful degradation."""

    def __init__(self):
        self._services = {}
        self._availability = {}
        self._initialize_services()

    def _initialize_services(self):
        """Initialize all optional services."""
        # Data loaders
        self._register_service(
            "csv_loader", "pynomaly.infrastructure.data_loaders", "CSVLoader"
        )
        self._register_service(
            "parquet_loader", "pynomaly.infrastructure.data_loaders", "ParquetLoader"
        )
        self._register_service(
            "polars_loader", "pynomaly.infrastructure.data_loaders", "PolarsLoader"
        )
        self._register_service(
            "arrow_loader", "pynomaly.infrastructure.data_loaders", "ArrowLoader"
        )
        self._register_service(
            "spark_loader", "pynomaly.infrastructure.data_loaders", "SparkLoader"
        )

        # Algorithm adapters
        self._register_service(
            "pyod_adapter", "pynomaly.infrastructure.adapters", "PyODAdapter"
        )
        self._register_service(
            "sklearn_adapter", "pynomaly.infrastructure.adapters", "SklearnAdapter"
        )
        self._register_service(
            "pygod_adapter", "pynomaly.infrastructure.adapters", "PyGODAdapter"
        )
        self._register_service(
            "pytorch_adapter", "pynomaly.infrastructure.adapters", "PyTorchAdapter"
        )
        self._register_service(
            "tensorflow_adapter",
            "pynomaly.infrastructure.adapters",
            "TensorFlowAdapter",
        )
        self._register_service(
            "jax_adapter", "pynomaly.infrastructure.adapters", "JAXAdapter"
        )

        # Authentication services
        self._register_service(
            "jwt_auth_service",
            "pynomaly.infrastructure.auth.jwt_auth",
            "JWTAuthService",
        )
        self._register_service(
            "permission_checker",
            "pynomaly.infrastructure.auth.middleware",
            "PermissionChecker",
        )
        self._register_service(
            "rate_limiter", "pynomaly.infrastructure.auth.middleware", "RateLimiter"
        )

        # Security services
        self._register_service(
            "security_monitor",
            "pynomaly.infrastructure.security.security_monitor",
            "SecurityMonitor",
        )
        self._register_service(
            "audit_logger",
            "pynomaly.infrastructure.security.audit_logger",
            "AuditLogger",
        )
        self._register_service(
            "security_middleware_stack",
            "pynomaly.infrastructure.security.middleware_integration",
            "SecurityMiddlewareStack",
        )

        # Cache services
        self._register_service(
            "redis_cache", "pynomaly.infrastructure.cache", "RedisCache"
        )
        self._register_service(
            "detector_cache_decorator",
            "pynomaly.infrastructure.cache",
            "DetectorCacheDecorator",
        )

        # Monitoring services
        self._register_service(
            "telemetry_service",
            "pynomaly.infrastructure.monitoring",
            "TelemetryService",
        )
        self._register_service(
            "health_service", "pynomaly.infrastructure.monitoring", "HealthService"
        )

        # Logging and observability services
        self._register_service(
            "structured_logger",
            "pynomaly.infrastructure.logging.structured_logger",
            "StructuredLogger",
        )
        self._register_service(
            "metrics_collector",
            "pynomaly.infrastructure.logging.metrics_collector",
            "MetricsCollector",
        )
        self._register_service(
            "tracing_manager",
            "pynomaly.infrastructure.logging.tracing_manager",
            "TracingManager",
        )
        self._register_service(
            "log_aggregator",
            "pynomaly.infrastructure.logging.log_aggregator",
            "LogAggregator",
        )
        self._register_service(
            "log_analyzer",
            "pynomaly.infrastructure.logging.log_analysis",
            "LogAnalyzer",
        )
        self._register_service(
            "log_anomaly_detector",
            "pynomaly.infrastructure.logging.log_analysis",
            "AnomalyDetector",
        )
        self._register_service(
            "observability_service",
            "pynomaly.infrastructure.logging.observability_service",
            "ObservabilityService",
        )

        # Performance optimization services
        self._register_service(
            "cache_manager",
            "pynomaly.infrastructure.cache.cache_manager",
            "CacheManager",
        )
        self._register_service(
            "in_memory_cache",
            "pynomaly.infrastructure.cache.cache_manager",
            "InMemoryCache",
        )
        self._register_service(
            "redis_cache_backend",
            "pynomaly.infrastructure.cache.cache_manager",
            "RedisCache",
        )
        self._register_service(
            "performance_profiler",
            "pynomaly.infrastructure.performance.profiler",
            "PerformanceProfiler",
        )
        self._register_service(
            "system_monitor",
            "pynomaly.infrastructure.performance.profiler",
            "SystemMonitor",
        )
        self._register_service(
            "query_optimizer",
            "pynomaly.infrastructure.performance.query_optimizer",
            "QueryOptimizer",
        )
        self._register_service(
            "dataframe_optimizer",
            "pynomaly.infrastructure.performance.query_optimizer",
            "DataFrameOptimizer",
        )
        self._register_service(
            "query_cache",
            "pynomaly.infrastructure.performance.query_optimizer",
            "QueryCache",
        )

        # Distributed processing services
        self._register_service(
            "task_distributor",
            "pynomaly.infrastructure.distributed.task_distributor",
            "TaskDistributor",
        )
        self._register_service(
            "worker_manager",
            "pynomaly.infrastructure.distributed.worker_manager",
            "WorkerManager",
        )
        self._register_service(
            "data_partitioner",
            "pynomaly.infrastructure.distributed.data_partitioner",
            "DataPartitioner",
        )
        self._register_service(
            "distributed_detector",
            "pynomaly.infrastructure.distributed.distributed_detector",
            "DistributedDetector",
        )
        self._register_service(
            "cluster_coordinator",
            "pynomaly.infrastructure.distributed.cluster_coordinator",
            "ClusterCoordinator",
        )
        self._register_service(
            "result_aggregator",
            "pynomaly.infrastructure.distributed.result_aggregator",
            "ResultAggregator",
        )
        self._register_service(
            "load_balancer",
            "pynomaly.infrastructure.distributed.load_balancer",
            "LoadBalancer",
        )

        # Database services
        self._register_service(
            "database_manager", "pynomaly.infrastructure.persistence", "DatabaseManager"
        )
        self._register_service(
            "database_detector_repository",
            "pynomaly.infrastructure.persistence",
            "DatabaseDetectorRepository",
        )
        self._register_service(
            "database_dataset_repository",
            "pynomaly.infrastructure.persistence",
            "DatabaseDatasetRepository",
        )
        self._register_service(
            "database_result_repository",
            "pynomaly.infrastructure.persistence",
            "DatabaseDetectionResultRepository",
        )

        # Application services
        self._register_service(
            "automl_service",
            "pynomaly.application.services.automl_service",
            "AutoMLService",
        )
        self._register_service(
            "enhanced_automl_service",
            "pynomaly.application.services.enhanced_automl_service",
            "EnhancedAutoMLService",
        )
        self._register_service(
            "explainability_service",
            "pynomaly.domain.services.explainability_service",
            "ExplainabilityService",
        )
        self._register_service(
            "application_explainability_service",
            "pynomaly.application.services.explainability_service",
            "ApplicationExplainabilityService",
        )

        # Explainers
        self._register_service(
            "shap_explainer", "pynomaly.infrastructure.explainers", "SHAPExplainer"
        )
        self._register_service(
            "lime_explainer", "pynomaly.infrastructure.explainers", "LIMEExplainer"
        )

        # Use cases
        self._register_service(
            "explain_anomaly_use_case",
            "pynomaly.application.use_cases",
            "ExplainAnomalyUseCase",
        )
        self._register_service(
            "automl_use_case",
            "pynomaly.application.use_cases.automl_use_case",
            "AutoMLUseCase",
        )

    def _register_service(self, name: str, module_path: str, class_name: str):
        """Register a service with optional import."""
        try:
            import importlib

            module = importlib.import_module(module_path)
            service_class = getattr(module, class_name)
            self._services[name] = service_class
            self._availability[name] = True
            logger.debug(f"Registered optional service: {name}")
        except (ImportError, AttributeError) as e:
            self._availability[name] = False
            logger.debug(f"Optional service {name} not available: {e}")

    def is_available(self, name: str) -> bool:
        """Check if a service is available."""
        return self._availability.get(name, False)

    def get_service(self, name: str):
        """Get service class if available."""
        return self._services.get(name) if self.is_available(name) else None

    def create_provider(self, name: str, provider_type: str = "singleton", **kwargs):
        """Create a provider for the service if available."""
        service_class = self.get_service(name)
        if not service_class:
            return None

        if provider_type == "singleton":
            return providers.Singleton(service_class, **kwargs)
        elif provider_type == "factory":
            return providers.Factory(service_class, **kwargs)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")


def _create_repository(config, repo_type: str):
    """Unified repository creation logic."""
    service_manager = OptionalServiceManager()

    # Try database repository first if configured
    if config.use_database_repositories and config.database_configured:
        db_repo_class = service_manager.get_service(f"database_{repo_type}_repository")
        if db_repo_class and service_manager.is_available("database_manager"):
            try:
                db_manager_class = service_manager.get_service("database_manager")
                db_manager = db_manager_class(
                    database_url=config.database_url,
                    echo=config.database_echo or config.app.debug,
                )

                # Initialize database if needed
                try:
                    from pynomaly.infrastructure.persistence.migrations import (
                        DatabaseMigrator,
                    )

                    migrator = DatabaseMigrator(db_manager)
                    if not migrator.check_tables_exist():
                        migrator.create_all_tables()
                except Exception as e:
                    logger.warning(
                        f"Database initialization failed, falling back to file: {e}"
                    )
                    return _create_file_repository(config, repo_type)

                return db_repo_class(db_manager.get_session)
            except Exception as e:
                logger.warning(
                    f"Database repository creation failed, falling back to file: {e}"
                )

    # Fall back to file repository
    return _create_file_repository(config, repo_type)


def _create_file_repository(config, repo_type: str):
    """Create file-based repository."""
    repo_mapping = {
        "detector": FileDetectorRepository,
        "dataset": FileDatasetRepository,
        "result": FileResultRepository,
    }

    repo_class = repo_mapping.get(repo_type)
    if repo_class:
        return repo_class(config.storage_path)

    # Fall back to in-memory
    memory_mapping = {
        "detector": InMemoryDetectorRepository,
        "dataset": InMemoryDatasetRepository,
        "result": InMemoryResultRepository,
    }

    return memory_mapping[repo_type]()


class Container(containers.DeclarativeContainer):
    """Main dependency injection container with simplified architecture."""

    # Configuration
    config = providers.Singleton(Settings)
    feature_flag_manager = providers.Singleton(FeatureFlagManager)

    # Initialize service manager
    _service_manager = OptionalServiceManager()

    # Domain services
    anomaly_scorer = providers.Singleton(AnomalyScorer)
    threshold_calculator = providers.Singleton(ThresholdCalculator)
    feature_validator = providers.Singleton(FeatureValidator)
    ensemble_aggregator = providers.Singleton(EnsembleAggregator)
    
    # Classifier services
    default_severity_classifier = providers.Singleton(DefaultSeverityClassifier)
    default_type_classifier = providers.Singleton(DefaultTypeClassifier)

    # Repositories using unified creation logic
    detector_repository = providers.Singleton(
        lambda: _create_repository(Settings(), "detector")
    )
    dataset_repository = providers.Singleton(
        lambda: _create_repository(Settings(), "dataset")
    )
    result_repository = providers.Singleton(
        lambda: _create_repository(Settings(), "result")
    )

    # Async repository wrappers
    async_detector_repository = providers.Singleton(
        AsyncDetectorRepositoryWrapper, sync_repository=detector_repository
    )
    async_dataset_repository = providers.Singleton(
        AsyncDatasetRepositoryWrapper, sync_repository=dataset_repository
    )
    async_result_repository = providers.Singleton(
        AsyncDetectionResultRepositoryWrapper, sync_repository=result_repository
    )

    # Create optional service providers dynamically
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._setup_optional_providers()

    @classmethod
    def _setup_optional_providers(cls):
        """Setup optional service providers."""
        service_manager = OptionalServiceManager()

        # Data loaders
        for loader_name, config_params in [
            ("csv_loader", {"delimiter": ",", "encoding": "utf-8"}),
            ("parquet_loader", {"use_pyarrow": True}),
            ("polars_loader", {"lazy": True, "streaming": False}),
            ("arrow_loader", {"use_threads": True}),
            ("spark_loader", {"app_name": "Pynomaly", "master": "local[*]"}),
        ]:
            provider = service_manager.create_provider(
                loader_name, "factory", **config_params
            )
            if provider:
                setattr(cls, loader_name, provider)

        # Algorithm adapters
        for adapter_name in [
            "pyod_adapter",
            "sklearn_adapter",
            "pygod_adapter",
            "pytorch_adapter",
            "tensorflow_adapter",
            "jax_adapter",
        ]:
            provider = service_manager.create_provider(adapter_name, "factory")
            if provider:
                setattr(cls, adapter_name, provider)

        # Authentication services
        if service_manager.is_available("jwt_auth_service"):
            cls.jwt_auth_service = service_manager.create_provider(
                "jwt_auth_service",
                "singleton",
                secret_key=cls.config.provided.secret_key,
                algorithm=cls.config.provided.jwt_algorithm,
                access_token_expire_minutes=cls.config.provided.jwt_expiration,
            )

        if service_manager.is_available("permission_checker"):
            cls.permission_checker = service_manager.create_provider(
                "permission_checker", "singleton"
            )

        if service_manager.is_available("rate_limiter"):
            cls.rate_limiter = service_manager.create_provider(
                "rate_limiter",
                "singleton",
                default_requests_per_minute=cls.config.provided.api_rate_limit,
            )

        # Cache services
        if service_manager.is_available("redis_cache"):
            cls.redis_cache = service_manager.create_provider(
                "redis_cache",
                "singleton",
                redis_url=cls.config.provided.redis_url,
                default_ttl=cls.config.provided.cache_ttl,
            )

        # Monitoring services
        if service_manager.is_available("telemetry_service"):
            cls.telemetry_service = service_manager.create_provider(
                "telemetry_service",
                "singleton",
                service_name="pynomaly",
                environment=cls.config.provided.app.environment,
                otlp_endpoint=cls.config.provided.monitoring.otlp_endpoint,
            )

        if service_manager.is_available("health_service"):
            cls.health_service = service_manager.create_provider(
                "health_service", "singleton", max_history=100
            )

        # Observability services
        if service_manager.is_available("observability_service"):
            from pynomaly.infrastructure.logging.observability_service import (
                ObservabilityConfig,
            )

            cls.observability_service = service_manager.create_provider(
                "observability_service",
                "singleton",
                config=ObservabilityConfig(
                    logs_storage_path=None,  # Simplified for now
                    metrics_storage_path=None,  # Simplified for now
                    traces_storage_path=None,  # Simplified for now
                    log_level="INFO",
                    enable_console_logging=True,
                    enable_json_logging=True,
                    enable_system_metrics=True,
                    enable_tracing=False,
                    enable_log_analysis=True,
                    enable_anomaly_detection=True,
                    enable_alerts=False,
                ),
            )

        # Individual logging services for direct access
        if service_manager.is_available("structured_logger"):
            cls.structured_logger = service_manager.create_provider(
                "structured_logger",
                "factory",
                logger_name=providers.Object("pynomaly"),
                output_path=providers.Object(None),  # Simplified for now
            )

        if service_manager.is_available("metrics_collector"):
            cls.metrics_collector = service_manager.create_provider(
                "metrics_collector",
                "singleton",
                storage_path=providers.Object(None),  # Simplified for now
                enable_system_metrics=True,
            )

        if service_manager.is_available("log_aggregator"):
            cls.log_aggregator = service_manager.create_provider(
                "log_aggregator",
                "singleton",
                storage_path=providers.Object(None),  # Simplified for now
            )

        if service_manager.is_available("log_analyzer"):
            cls.log_analyzer = service_manager.create_provider(
                "log_analyzer",
                "singleton",
                enable_realtime=True,
                enable_background_analysis=True,
            )

        # Performance optimization services
        cls._register_performance_services(service_manager)

        # Security services
        if service_manager.is_available("audit_logger"):
            cls.audit_logger = service_manager.create_provider(
                "audit_logger",
                "singleton",
                enable_structured_logging=True,
                enable_compliance_logging=True,
            )

        if service_manager.is_available("security_monitor"):
            cls.security_monitor = service_manager.create_provider(
                "security_monitor",
                "singleton",
                audit_logger=cls.audit_logger if hasattr(cls, "audit_logger") else None,
            )

        # Distributed processing services
        if service_manager.is_available("task_distributor"):
            cls.task_distributor = service_manager.create_provider(
                "task_distributor", "singleton", config=cls.config.provided
            )

        if service_manager.is_available("worker_manager"):
            cls.worker_manager = service_manager.create_provider(
                "worker_manager", "singleton", config=cls.config.provided
            )

        if service_manager.is_available("data_partitioner"):
            cls.data_partitioner = service_manager.create_provider(
                "data_partitioner", "singleton"
            )

        if service_manager.is_available("distributed_detector"):
            cls.distributed_detector = service_manager.create_provider(
                "distributed_detector",
                "singleton",
                task_distributor=(
                    cls.task_distributor if hasattr(cls, "task_distributor") else None
                ),
                worker_manager=(
                    cls.worker_manager if hasattr(cls, "worker_manager") else None
                ),
                data_partitioner=(
                    cls.data_partitioner if hasattr(cls, "data_partitioner") else None
                ),
            )

        if service_manager.is_available("cluster_coordinator"):
            cls.cluster_coordinator = service_manager.create_provider(
                "cluster_coordinator", "singleton"
            )

        if service_manager.is_available("result_aggregator"):
            cls.result_aggregator = service_manager.create_provider(
                "result_aggregator", "singleton"
            )

        if service_manager.is_available("load_balancer"):
            cls.load_balancer = service_manager.create_provider(
                "load_balancer", "singleton"
            )

        # AutoML services
        if service_manager.is_available("automl_service"):
            cls.automl_service = service_manager.create_provider(
                "automl_service",
                "singleton",
                detector_repository=cls.async_detector_repository,
                dataset_repository=cls.async_dataset_repository,
                adapter_registry=providers.Object("adapter_registry"),
                max_optimization_time=3600,
                n_trials=100,
                cv_folds=3,
                random_state=42,
            )

        if service_manager.is_available("enhanced_automl_service"):
            from pynomaly.application.services.enhanced_automl_service import (
                EnhancedAutoMLConfig,
            )

            cls.enhanced_automl_service = service_manager.create_provider(
                "enhanced_automl_service",
                "singleton",
                detector_repository=cls.async_detector_repository,
                dataset_repository=cls.async_dataset_repository,
                adapter_registry=providers.Object("adapter_registry"),
                config=EnhancedAutoMLConfig(
                    max_optimization_time=(
                        cls.config.provided.automl_max_time
                        if hasattr(cls.config.provided, "automl_max_time")
                        else 3600
                    ),
                    n_trials=(
                        cls.config.provided.automl_n_trials
                        if hasattr(cls.config.provided, "automl_n_trials")
                        else 100
                    ),
                    enable_meta_learning=True,
                    enable_multi_objective=True,
                    enable_parallel=True,
                    random_state=42,
                ),
                storage_path=(
                    cls.config.provided.automl_storage_path
                    if hasattr(cls.config.provided, "automl_storage_path")
                    else Path("./automl_storage")
                ),
            )

        # Explainability services
        if service_manager.is_available("explainability_service"):
            cls.domain_explainability_service = service_manager.create_provider(
                "explainability_service", "singleton"
            )

            if service_manager.is_available("shap_explainer"):
                cls.shap_explainer = service_manager.create_provider(
                    "shap_explainer", "singleton"
                )

            if service_manager.is_available("lime_explainer"):
                cls.lime_explainer = service_manager.create_provider(
                    "lime_explainer", "singleton"
                )

            if service_manager.is_available("application_explainability_service"):
                cls.application_explainability_service = (
                    service_manager.create_provider(
                        "application_explainability_service",
                        "singleton",
                        domain_explainability_service=cls.domain_explainability_service,
                        detector_repository=cls.async_detector_repository,
                        dataset_repository=cls.async_dataset_repository,
                    )
                )

    @classmethod
    def _register_performance_services(cls, service_manager):
        """Register performance optimization services."""
        # Cache management services
        if service_manager.is_available("cache_manager"):
            # Create primary in-memory cache
            cls.primary_cache_backend = service_manager.create_provider(
                "in_memory_cache",
                "singleton",
                max_size=1000,
                max_memory_mb=100,
            )

            # Create Redis fallback cache if available
            redis_cache_backend = None
            if service_manager.is_available("redis_cache_backend"):
                redis_cache_backend = service_manager.create_provider(
                    "redis_cache_backend",
                    "singleton",
                    host="localhost",
                    port=6379,
                    default_ttl=3600,
                )

            # Create cache manager with fallback
            cls.cache_manager = service_manager.create_provider(
                "cache_manager",
                "singleton",
                primary_backend=cls.primary_cache_backend,
                fallback_backend=redis_cache_backend,
                enable_write_through=True,
                enable_metrics=True,
            )

        # Performance profiling services
        if service_manager.is_available("performance_profiler"):
            cls.performance_profiler = service_manager.create_provider(
                "performance_profiler",
                "singleton",
                enable_cpu_profiling=True,
                enable_memory_profiling=True,
                max_results=1000,
                output_dir=None,
            )

        if service_manager.is_available("system_monitor"):
            cls.system_monitor = service_manager.create_provider(
                "system_monitor",
                "singleton",
                monitoring_interval=5,
                history_size=1000,
                enable_alerts=False,  # Disable alerts by default
            )

        # Query optimization services
        if service_manager.is_available("query_optimizer"):
            cls.query_optimizer = service_manager.create_provider(
                "query_optimizer",
                "singleton",
                enable_caching=True,
                enable_optimization=True,
                cache_ttl=3600,
                max_cache_size=1000,
            )

        if service_manager.is_available("dataframe_optimizer"):
            cls.dataframe_optimizer = service_manager.create_provider(
                "dataframe_optimizer", "singleton"
            )

        if service_manager.is_available("query_cache"):
            cls.query_cache = service_manager.create_provider(
                "query_cache",
                "singleton",
                max_size=1000,
                default_ttl=3600,
            )

        # Legacy compatibility providers for connection pool manager
        # Provide a simple stub that returns None - endpoints handle this gracefully
        cls.connection_pool_manager = providers.Singleton(lambda: None)

    # Application services
    anomaly_classification_service = providers.Singleton(
        AnomalyClassificationService,
        severity_classifier=default_severity_classifier,
        type_classifier=default_type_classifier,
    )
    
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


# Initialize optional providers
Container._setup_optional_providers()


class TestContainer(Container):
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


def create_container(testing: bool = False) -> Container:
    """Create and configure the DI container.

    Args:
        testing: Whether to create a test container

    Returns:
        Configured container
    """
    if testing:
        container = TestContainer()
    else:
        container = Container()

    # Wire the container to modules that need it
    container.wire(
        modules=[
            "pynomaly.presentation.api",
            "pynomaly.presentation.cli",
            "pynomaly.presentation.web",
        ]
    )

    return container


def get_container_simple() -> Container:
    """Get a simple container instance for CLI usage.

    This is a simple wrapper around create_container() for CLI usage.
    It creates a new container if one doesn't exist.
    """
    return create_container(testing=False)
