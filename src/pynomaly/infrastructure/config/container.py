"""Dependency injection container using dependency-injector."""

from __future__ import annotations

from pathlib import Path

from dependency_injector import containers, providers

from pynomaly.application.services import (
    DetectionService,
    EnsembleService,
    ExperimentTrackingService,
    ModelPersistenceService
)
from pynomaly.application.use_cases import (
    DetectAnomaliesUseCase,
    EvaluateModelUseCase,
    ExplainAnomalyUseCase,
    TrainDetectorUseCase
)
from pynomaly.domain.services import (
    AnomalyScorer,
    EnsembleAggregator,
    FeatureValidator,
    ThresholdCalculator
)
from pynomaly.infrastructure.config.settings import Settings
from pynomaly.infrastructure.data_loaders import CSVLoader, ParquetLoader

# Preprocessing services
try:
    from pynomaly.infrastructure.preprocessing import DataCleaner, DataTransformer, PreprocessingPipeline
except ImportError:
    DataCleaner = None
    DataTransformer = None
    PreprocessingPipeline = None

# Authentication and security
try:
    from pynomaly.infrastructure.auth import JWTAuthService, PermissionChecker, RateLimiter
except ImportError:
    JWTAuthService = None
    PermissionChecker = None
    RateLimiter = None

# Cache services
try:
    from pynomaly.infrastructure.cache import RedisCache, DetectorCacheDecorator
except ImportError:
    RedisCache = None
    DetectorCacheDecorator = None

# Monitoring services
try:
    from pynomaly.infrastructure.monitoring import TelemetryService
except ImportError:
    TelemetryService = None

# Database repositories
try:
    from pynomaly.infrastructure.persistence import (
        DatabaseManager,
        DatabaseDetectorRepository,
        DatabaseDatasetRepository,
        DatabaseDetectionResultRepository
    )
except ImportError:
    DatabaseManager = None
    DatabaseDetectorRepository = None
    DatabaseDatasetRepository = None
    DatabaseDetectionResultRepository = None

# Resilience services
try:
    from pynomaly.infrastructure.resilience.service import ResilienceService
except ImportError:
    ResilienceService = None

# Optional high-performance data loaders - import only if available
try:
    from pynomaly.infrastructure.data_loaders import PolarsLoader
except ImportError:
    PolarsLoader = None

try:
    from pynomaly.infrastructure.data_loaders import ArrowLoader
except ImportError:
    ArrowLoader = None

try:
    from pynomaly.infrastructure.data_loaders import SparkLoader
except ImportError:
    SparkLoader = None
from pynomaly.infrastructure.repositories import (
    InMemoryDatasetRepository,
    InMemoryDetectorRepository,
    InMemoryResultRepository
)
from pynomaly.infrastructure.adapters import PyODAdapter, SklearnAdapter

# Optional adapters - import only if available
try:
    from pynomaly.infrastructure.adapters import TODSAdapter
except ImportError:
    TODSAdapter = None

try:
    from pynomaly.infrastructure.adapters import PyGODAdapter  
except ImportError:
    PyGODAdapter = None

try:
    from pynomaly.infrastructure.adapters import PyTorchAdapter
except ImportError:
    PyTorchAdapter = None


class Container(containers.DeclarativeContainer):
    """Main dependency injection container."""
    
    # Configuration
    config = providers.Singleton(Settings)
    
    # Domain services
    anomaly_scorer = providers.Singleton(AnomalyScorer)
    threshold_calculator = providers.Singleton(ThresholdCalculator)
    feature_validator = providers.Singleton(FeatureValidator)
    ensemble_aggregator = providers.Singleton(EnsembleAggregator)
    
    # Repositories
    detector_repository = providers.Singleton(InMemoryDetectorRepository)
    dataset_repository = providers.Singleton(InMemoryDatasetRepository)
    result_repository = providers.Singleton(InMemoryResultRepository)
    
    # Data loaders
    csv_loader = providers.Factory(
        CSVLoader,
        delimiter=",",
        encoding="utf-8"
    )
    
    parquet_loader = providers.Factory(
        ParquetLoader,
        use_pyarrow=True
    )
    
    # High-performance data loaders - only create providers if loaders are available
    if PolarsLoader is not None:
        polars_loader = providers.Factory(
            PolarsLoader,
            lazy=True,
            streaming=False
        )
    
    if ArrowLoader is not None:
        arrow_loader = providers.Factory(
            ArrowLoader,
            use_threads=True
        )
    
    if SparkLoader is not None:
        spark_loader = providers.Factory(
            SparkLoader,
            app_name="Pynomaly",
            master="local[*]"
        )
    
    # Algorithm adapters
    pyod_adapter = providers.Singleton(PyODAdapter)
    sklearn_adapter = providers.Singleton(SklearnAdapter)
    
    # Optional adapters - only create providers if adapters are available
    if TODSAdapter is not None:
        tods_adapter = providers.Singleton(TODSAdapter)
    
    if PyGODAdapter is not None:
        pygod_adapter = providers.Singleton(PyGODAdapter)
    
    if PyTorchAdapter is not None:
        pytorch_adapter = providers.Singleton(PyTorchAdapter)
    
    # Authentication services - only create if available
    if JWTAuthService is not None:
        jwt_auth_service = providers.Singleton(
            JWTAuthService,
            secret_key=config.provided.secret_key,
            algorithm=config.provided.jwt_algorithm,
            access_token_expire_minutes=config.provided.jwt_expiration
        )
    
    if PermissionChecker is not None:
        permission_checker = providers.Singleton(PermissionChecker)
    
    if RateLimiter is not None:
        rate_limiter = providers.Singleton(
            RateLimiter,
            default_requests_per_minute=config.provided.api_rate_limit
        )
    
    # Cache services - only create if available
    if RedisCache is not None:
        redis_cache = providers.Singleton(
            RedisCache,
            redis_url=config.provided.redis_url,
            default_ttl=config.provided.cache_ttl
        )
    
    if DetectorCacheDecorator is not None and RedisCache is not None:
        detector_cache_decorator = providers.Singleton(
            DetectorCacheDecorator,
            cache=redis_cache
        )
    
    # Monitoring services - only create if available
    if TelemetryService is not None:
        telemetry_service = providers.Singleton(
            TelemetryService,
            service_name="pynomaly",
            environment=config.provided.app.environment,
            otlp_endpoint=config.provided.monitoring.otlp_endpoint
        )
    
    # Database services - only create if available and configured
    if DatabaseManager is not None:
        database_manager = providers.Singleton(
            DatabaseManager,
            database_url=config.provided.database_url
        )
        
        # Alternative database repositories - only if database is configured
        if DatabaseDetectorRepository is not None:
            database_detector_repository = providers.Singleton(
                DatabaseDetectorRepository,
                session_factory=database_manager.provided.get_session
            )
        
        if DatabaseDatasetRepository is not None:
            database_dataset_repository = providers.Singleton(
                DatabaseDatasetRepository,
                session_factory=database_manager.provided.get_session
            )
        
        if DatabaseDetectionResultRepository is not None:
            database_result_repository = providers.Singleton(
                DatabaseDetectionResultRepository,
                session_factory=database_manager.provided.get_session
            )
    
    # Resilience services - only create if available
    if ResilienceService is not None:
        resilience_service = providers.Singleton(ResilienceService)
    
    # Preprocessing services - only create if available
    if DataCleaner is not None:
        data_cleaner = providers.Singleton(DataCleaner)
    
    if DataTransformer is not None:
        data_transformer = providers.Singleton(DataTransformer)
    
    if PreprocessingPipeline is not None:
        # Basic preprocessing pipeline
        basic_preprocessing_pipeline = providers.Factory(
            PreprocessingPipeline.create_basic_pipeline
        )
        
        # Anomaly detection optimized pipeline
        anomaly_preprocessing_pipeline = providers.Factory(
            PreprocessingPipeline.create_anomaly_detection_pipeline
        )
    
    # Application services
    detection_service = providers.Singleton(
        DetectionService,
        detector_repository=detector_repository,
        result_repository=result_repository,
        anomaly_scorer=anomaly_scorer,
        threshold_calculator=threshold_calculator
    )
    
    ensemble_service = providers.Singleton(
        EnsembleService,
        detector_repository=detector_repository,
        ensemble_aggregator=ensemble_aggregator,
        anomaly_scorer=anomaly_scorer
    )
    
    model_persistence_service = providers.Singleton(
        ModelPersistenceService,
        detector_repository=detector_repository,
        storage_path=config.provided.model_storage_path
    )
    
    experiment_tracking_service = providers.Singleton(
        ExperimentTrackingService,
        tracking_path=config.provided.experiment_storage_path
    )
    
    # Use cases
    detect_anomalies_use_case = providers.Factory(
        DetectAnomaliesUseCase,
        detector_repository=detector_repository,
        feature_validator=feature_validator
    )
    
    train_detector_use_case = providers.Factory(
        TrainDetectorUseCase,
        detector_repository=detector_repository,
        feature_validator=feature_validator,
        min_samples=10
    )
    
    evaluate_model_use_case = providers.Factory(
        EvaluateModelUseCase,
        detector_repository=detector_repository
    )
    
    explain_anomaly_use_case = providers.Factory(
        ExplainAnomalyUseCase,
        detector_repository=detector_repository
    )


class TestContainer(Container):
    """Container for testing with test-specific overrides."""
    
    # Override settings for testing
    config = providers.Singleton(
        Settings,
        storage_path=Path("./test_storage"),
        model_storage_path=Path("./test_storage/models"),
        experiment_storage_path=Path("./test_storage/experiments"),
        debug=True,
        auth_enabled=False
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