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

# Phase 2 services (conditionally imported)
try:
    from pynomaly.application.services.algorithm_benchmark import AlgorithmBenchmarkService
    from pynomaly.infrastructure.monitoring.complexity_monitor import ComplexityMonitor
    PHASE2_SERVICES_AVAILABLE = True
except ImportError:
    AlgorithmBenchmarkService = None
    ComplexityMonitor = None
    PHASE2_SERVICES_AVAILABLE = False

# Optional AutoML service
try:
    from pynomaly.application.services.automl_service import AutoMLService
except ImportError:
    AutoMLService = None

# Optional explainability services
try:
    from pynomaly.domain.services.explainability_service import ExplainabilityService
    from pynomaly.application.services.explainability_service import ApplicationExplainabilityService
    from pynomaly.infrastructure.explainers import SHAPExplainer, LIMEExplainer, SHAP_AVAILABLE, LIME_AVAILABLE
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    ExplainabilityService = None
    ApplicationExplainabilityService = None
    SHAPExplainer = None
    LIMEExplainer = None
    SHAP_AVAILABLE = False
    LIME_AVAILABLE = False
    EXPLAINABILITY_AVAILABLE = False

# Streaming infrastructure completely removed for simplification

# Distributed processing removed for simplification
DistributedProcessingManager = None
DetectionCoordinator = None
LoadBalancer = None
DISTRIBUTED_AVAILABLE = False
from pynomaly.application.use_cases import (
    DetectAnomaliesUseCase,
    EvaluateModelUseCase,
    ExplainAnomalyUseCase,
    TrainDetectorUseCase
)

# Optional AutoML use case
try:
    from pynomaly.application.use_cases.automl_use_case import AutoMLUseCase
except ImportError:
    AutoMLUseCase = None
from pynomaly.domain.services import (
    AnomalyScorer,
    EnsembleAggregator,
    FeatureValidator,
    ThresholdCalculator
)
from pynomaly.infrastructure.config.settings import Settings
from pynomaly.infrastructure.config.feature_flags import FeatureFlagManager, feature_flags
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
    from pynomaly.infrastructure.monitoring import TelemetryService, HealthService
except ImportError:
    TelemetryService = None
    HealthService = None

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

# Lifecycle services
try:
    from pynomaly.infrastructure.lifecycle import ShutdownService
except ImportError:
    ShutdownService = None

# Security services
try:
    from pynomaly.infrastructure.security import (
        InputSanitizer, SanitizationConfig,
        SQLInjectionProtector, QuerySanitizer, SafeQueryBuilder,
        EncryptionService, DataEncryption, FieldEncryption, EncryptionConfig,
        SecurityHeadersMiddleware, SecurityHeaders, CSPConfig,
        AuditLogger, SecurityMonitor, UserActionTracker,
        create_development_headers, create_production_headers
    )
    SECURITY_AVAILABLE = True
except ImportError:
    InputSanitizer = None
    SanitizationConfig = None
    SQLInjectionProtector = None
    QuerySanitizer = None
    SafeQueryBuilder = None
    EncryptionService = None
    DataEncryption = None
    FieldEncryption = None
    EncryptionConfig = None
    SecurityHeadersMiddleware = None
    SecurityHeaders = None
    CSPConfig = None
    AuditLogger = None
    SecurityMonitor = None
    UserActionTracker = None
    create_development_headers = None
    create_production_headers = None
    SECURITY_AVAILABLE = False

# Performance optimization services
try:
    from pynomaly.infrastructure.performance import (
        ConnectionPoolManager,
        QueryOptimizer,
        QueryCache,
        PoolConfiguration,
        PerformanceService
    )
except ImportError:
    ConnectionPoolManager = None
    QueryOptimizer = None
    QueryCache = None
    PoolConfiguration = None
    PerformanceService = None

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

try:
    from pynomaly.infrastructure.adapters import TensorFlowAdapter
except ImportError:
    TensorFlowAdapter = None

try:
    from pynomaly.infrastructure.adapters import JAXAdapter
except ImportError:
    JAXAdapter = None


class Container(containers.DeclarativeContainer):
    """Main dependency injection container."""
    
    # Configuration
    config = providers.Singleton(Settings)
    
    # Feature flag management
    feature_flag_manager = providers.Singleton(FeatureFlagManager)
    
    # Domain services
    anomaly_scorer = providers.Singleton(AnomalyScorer)
    threshold_calculator = providers.Singleton(ThresholdCalculator)
    feature_validator = providers.Singleton(FeatureValidator)
    ensemble_aggregator = providers.Singleton(EnsembleAggregator)
    
    # Repositories - Select based on configuration
    detector_repository = providers.Singleton(
        lambda: _create_detector_repository(Settings())
    )
    dataset_repository = providers.Singleton(
        lambda: _create_dataset_repository(Settings())
    )
    result_repository = providers.Singleton(
        lambda: _create_result_repository(Settings())
    )
    
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
    
    if TensorFlowAdapter is not None:
        tensorflow_adapter = providers.Singleton(TensorFlowAdapter)
    
    if JAXAdapter is not None:
        jax_adapter = providers.Singleton(JAXAdapter)
    
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
    
    if HealthService is not None:
        health_service = providers.Singleton(
            HealthService,
            max_history=100
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
    
    # Lifecycle services - only create if available
    if ShutdownService is not None:
        shutdown_service = providers.Singleton(
            ShutdownService,
            shutdown_timeout=60.0
        )
    
    # Performance optimization services - only create if available
    if ConnectionPoolManager is not None:
        connection_pool_manager = providers.Singleton(ConnectionPoolManager)
        
        # Pool configuration for different environments
        pool_config = providers.Singleton(
            PoolConfiguration,
            min_size=5,
            max_size=20,
            timeout=30.0,
            max_overflow=10,
            recycle_time=3600,
            health_check_interval=60
        )
    
    # Initialize query_optimizer as None
    query_optimizer = None
    if QueryOptimizer is not None and DatabaseManager is not None:
        query_optimizer = providers.Singleton(
            QueryOptimizer,
            engine=database_manager.provided.engine,
            cache_size=1000,
            cache_ttl=3600
        )
    
    if PerformanceService is not None and ConnectionPoolManager is not None:
        performance_service = providers.Singleton(
            PerformanceService,
            pool_manager=connection_pool_manager,
            query_optimizer=query_optimizer,
            monitoring_interval=300.0
        )
    
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
    
    # AutoML service - only create if available
    if AutoMLService is not None:
        automl_service = providers.Singleton(
            AutoMLService,
            detector_repository=detector_repository,
            dataset_repository=dataset_repository,
            adapter_registry=providers.Object("adapter_registry"),  # Will be injected
            max_optimization_time=3600,
            n_trials=100,
            cv_folds=3,
            random_state=42
        )
    
    # Explainability services - only create if available
    if EXPLAINABILITY_AVAILABLE:
        # Domain explainability service
        domain_explainability_service = providers.Singleton(ExplainabilityService)
        
        # Register explainers if available
        if SHAP_AVAILABLE and SHAPExplainer is not None:
            shap_explainer = providers.Singleton(SHAPExplainer)
        
        if LIME_AVAILABLE and LIMEExplainer is not None:
            lime_explainer = providers.Singleton(LIMEExplainer)
        
        # Application explainability service
        if ApplicationExplainabilityService is not None:
            application_explainability_service = providers.Singleton(
                ApplicationExplainabilityService,
                domain_explainability_service=domain_explainability_service,
                detector_repository=detector_repository,
                dataset_repository=dataset_repository
            )
    
    # Phase 2 services - only create if available and feature flags enabled
    if PHASE2_SERVICES_AVAILABLE and feature_flags.is_enabled("algorithm_optimization"):
        algorithm_benchmark_service = providers.Singleton(AlgorithmBenchmarkService)
    
    if PHASE2_SERVICES_AVAILABLE and feature_flags.is_enabled("complexity_monitoring"):
        complexity_monitor = providers.Singleton(ComplexityMonitor)
    
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
    
    # Explainability use case - only create if available
    if EXPLAINABILITY_AVAILABLE and ApplicationExplainabilityService is not None:
        explain_anomaly_use_case = providers.Factory(
            ExplainAnomalyUseCase,
            explainability_service=application_explainability_service
        )
    
    # AutoML use case - only create if available
    if AutoMLUseCase is not None and AutoMLService is not None:
        automl_use_case = providers.Factory(
            AutoMLUseCase,
            automl_service=automl_service
        )
    
    
    # Security services - only create if available
    if SECURITY_AVAILABLE:
        # Input sanitization
        sanitization_config = providers.Singleton(
            SanitizationConfig,
            level=config.provided.security.sanitization_level,
            max_length=config.provided.security.max_input_length,
            allow_html=config.provided.security.allow_html
        )
        
        input_sanitizer = providers.Singleton(
            InputSanitizer,
            config=sanitization_config
        )
        
        # SQL injection protection
        sql_injection_protector = providers.Singleton(SQLInjectionProtector)
        query_sanitizer = providers.Singleton(QuerySanitizer)
        
        if DatabaseManager is not None:
            safe_query_builder = providers.Singleton(
                SafeQueryBuilder,
                metadata=database_manager.provided.metadata
            )
        
        # Encryption services
        encryption_config = providers.Singleton(
            EncryptionConfig,
            algorithm=config.provided.security.encryption_algorithm,
            key_length=config.provided.security.encryption_key_length,
            enable_key_rotation=config.provided.security.enable_key_rotation
        )
        
        encryption_service = providers.Singleton(
            EncryptionService,
            config=encryption_config
        )
        
        data_encryption = providers.Singleton(
            DataEncryption,
            config=encryption_config
        )
        
        field_encryption = providers.Singleton(
            FieldEncryption,
            encryption_service=encryption_service
        )
        
        # Security headers
        security_headers_config = providers.Factory(
            lambda config=config: create_development_headers() if config.app.environment == "development" 
            else create_production_headers()
        )
        
        # Audit logging
        audit_logger = providers.Singleton(
            AuditLogger,
            logger_name="pynomaly.audit",
            enable_structured_logging=True,
            enable_compliance_logging=config.provided.security.enable_compliance_logging
        )
        
        # Security monitoring
        security_monitor = providers.Singleton(
            SecurityMonitor,
            audit_logger=audit_logger
        )
        
        # User action tracking
        user_action_tracker = providers.Singleton(
            UserActionTracker,
            audit_logger=audit_logger,
            security_monitor=security_monitor
        )


def _create_detector_repository(config):
    """Create appropriate detector repository based on configuration."""
    if config.use_database_repositories and config.database_configured:
        if DatabaseDetectorRepository is not None and DatabaseManager is not None:
            # Create database manager and repository
            db_manager = DatabaseManager(
                database_url=config.database_url,
                echo=config.database_echo or config.app.debug
            )
            # Initialize database if needed
            try:
                from pynomaly.infrastructure.persistence.migrations import DatabaseMigrator
                migrator = DatabaseMigrator(db_manager)
                if not migrator.check_tables_exist():
                    migrator.create_all_tables()
            except Exception as e:
                import logging
                logging.warning(f"Database initialization failed, falling back to in-memory: {e}")
                return InMemoryDetectorRepository()
            
            return DatabaseDetectorRepository(db_manager.get_session)
    
    # Default to in-memory repository
    return InMemoryDetectorRepository()


def _create_dataset_repository(config):
    """Create appropriate dataset repository based on configuration."""
    if config.use_database_repositories and config.database_configured:
        if DatabaseDatasetRepository is not None and DatabaseManager is not None:
            # Create database manager and repository
            db_manager = DatabaseManager(
                database_url=config.database_url,
                echo=config.database_echo or config.app.debug
            )
            return DatabaseDatasetRepository(db_manager.get_session)
    
    # Default to in-memory repository
    return InMemoryDatasetRepository()


def _create_result_repository(config):
    """Create appropriate result repository based on configuration."""
    if config.use_database_repositories and config.database_configured:
        if DatabaseDetectionResultRepository is not None and DatabaseManager is not None:
            # Create database manager and repository
            db_manager = DatabaseManager(
                database_url=config.database_url,
                echo=config.database_echo or config.app.debug
            )
            return DatabaseDetectionResultRepository(db_manager.get_session)
    
    # Default to in-memory repository
    return InMemoryResultRepository()


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