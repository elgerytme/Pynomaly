"""Data Profiling package dependency injection container."""

import importlib
from typing import Dict, Any, Optional, Type
import structlog
from datetime import datetime

from dependency_injector import containers, providers

# Import application services
from ...application.services.profiling_engine import ProfilingEngine, ProfilingConfig
from ...application.services.schema_analysis_service import SchemaAnalysisService
from ...application.services.statistical_profiling_service import StatisticalProfilingService
from ...application.services.pattern_discovery_service import PatternDiscoveryService
from ...application.services.quality_assessment_service import QualityAssessmentService
from ...application.services.performance_optimizer import PerformanceOptimizer

# Import use cases
from ...application.use_cases.profile_dataset import ProfileDatasetUseCase
from ...application.use_cases.execute_data_profiling import ExecuteDataProfilingUseCase

# Import infrastructure adapters
from ..adapters.in_memory_data_profile_repository import InMemoryDataProfileRepository

logger = structlog.get_logger(__name__)


class OptionalServiceManager:
    """Manages optional services with graceful degradation."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._availability: Dict[str, bool] = {}
    
    def register_service(self, name: str, module_path: str, class_name: str) -> None:
        """Register a service with graceful degradation."""
        try:
            module = importlib.import_module(module_path)
            service_class = getattr(module, class_name)
            self._services[name] = service_class
            self._availability[name] = True
            logger.info("Service registered successfully", service=name)
        except (ImportError, AttributeError) as e:
            self._availability[name] = False
            logger.warning("Service unavailable", service=name, error=str(e))
    
    def get_service(self, name: str) -> Optional[Any]:
        """Get a service if available."""
        return self._services.get(name) if self._availability.get(name, False) else None
    
    def is_available(self, name: str) -> bool:
        """Check if a service is available."""
        return self._availability.get(name, False)


class DataProfilingContainer(containers.DeclarativeContainer):
    """Data Profiling package container with protocol-based dependency injection."""
    
    # Configuration
    config = providers.Configuration()
    
    # Logging
    logger = providers.Singleton(structlog.get_logger, name="data_profiling")
    
    # Optional service manager
    service_manager = providers.Singleton(OptionalServiceManager)
    
    # Core infrastructure - will be wired by specific implementations
    
    # Repositories (protocol-based)
    data_profile_repository = providers.Singleton(
        lambda: None  # Will be overridden by infrastructure layer
    )
    
    dataset_repository = providers.Singleton(
        lambda: None  # Will be overridden by infrastructure layer
    )
    
    schema_repository = providers.Singleton(
        lambda: None  # Will be overridden by infrastructure layer
    )
    
    # External service adapters
    database_connection_service = providers.Singleton(
        lambda: None  # Will be overridden by infrastructure layer
    )
    
    file_system_service = providers.Singleton(
        lambda: None  # Will be overridden by infrastructure layer
    )
    
    streaming_service = providers.Singleton(
        lambda: None  # Will be overridden by infrastructure layer
    )
    
    visualization_service = providers.Singleton(
        lambda: None  # Will be overridden by infrastructure layer
    )
    
    # Processing engines
    statistical_engine = providers.Singleton(
        lambda: None  # Will be overridden by infrastructure layer
    )
    
    pattern_recognition_engine = providers.Singleton(
        lambda: None  # Will be overridden by infrastructure layer
    )
    
    # Repository implementations
    data_profile_repository_impl = providers.Singleton(
        InMemoryDataProfileRepository
    )
    
    # Application services
    performance_optimizer = providers.Singleton(
        PerformanceOptimizer
    )
    
    schema_analysis_service = providers.Singleton(
        SchemaAnalysisService
    )
    
    statistical_profiling_service = providers.Singleton(
        StatisticalProfilingService
    )
    
    pattern_discovery_service = providers.Singleton(
        PatternDiscoveryService
    )
    
    quality_assessment_service = providers.Singleton(
        QualityAssessmentService
    )
    
    # Central profiling engine
    profiling_config = providers.Singleton(
        ProfilingConfig
    )
    
    profiling_engine = providers.Singleton(
        ProfilingEngine,
        config=profiling_config
    )
    
    # Use cases
    profile_dataset_use_case = providers.Factory(
        ProfileDatasetUseCase
    )
    
    execute_data_profiling_use_case = providers.Factory(
        ExecuteDataProfilingUseCase,
        repository=data_profile_repository_impl
    )


class DataProfilingInfrastructureContainer:
    """Infrastructure container for data profiling services."""
    
    def __init__(self, settings: Optional[Any] = None):
        self.settings = settings
        self._service_cache: Dict[str, Any] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize infrastructure services."""
        if self._initialized:
            return
            
        logger.info("Initializing data profiling infrastructure")
        
        # Initialize database connections, external services, etc.
        await self._setup_repositories()
        await self._setup_data_connectors()
        await self._setup_processing_engines()
        
        self._initialized = True
        logger.info("Data profiling infrastructure initialized successfully")
    
    async def _setup_repositories(self) -> None:
        """Setup repository implementations."""
        logger.info("Setting up data profiling repositories")
        
        # Initialize in-memory repository for now
        repository = InMemoryDataProfileRepository()
        self._service_cache["data_profile_repository"] = repository
        
        logger.info("Data profiling repositories initialized")
    
    async def _setup_data_connectors(self) -> None:
        """Setup data source connectors."""
        logger.info("Setting up data source connectors")
        
        # Setup available data connectors
        try:
            # File system connector (basic implementation)
            from ..adapters.file_adapter import FileAdapter
            self._service_cache["file_adapter"] = FileAdapter()
            logger.info("File adapter initialized")
        except ImportError:
            logger.warning("File adapter not available")
        
        try:
            # Database connector (if available)
            from ..adapters.database_adapter import DatabaseAdapter
            self._service_cache["database_adapter"] = DatabaseAdapter()
            logger.info("Database adapter initialized")
        except ImportError:
            logger.warning("Database adapter not available")
        
        try:
            # Cloud storage connector (if available)
            from ..adapters.cloud_storage_adapter import CloudStorageAdapter
            self._service_cache["cloud_storage_adapter"] = CloudStorageAdapter()
            logger.info("Cloud storage adapter initialized")
        except ImportError:
            logger.warning("Cloud storage adapter not available")
        
        logger.info("Data source connectors setup completed")
    
    async def _setup_processing_engines(self) -> None:
        """Setup processing engines for profiling."""
        logger.info("Setting up profiling processing engines")
        
        # Setup core processing services
        self._service_cache["schema_analysis_service"] = SchemaAnalysisService()
        self._service_cache["statistical_profiling_service"] = StatisticalProfilingService()
        self._service_cache["pattern_discovery_service"] = PatternDiscoveryService()
        self._service_cache["quality_assessment_service"] = QualityAssessmentService()
        self._service_cache["performance_optimizer"] = PerformanceOptimizer()
        
        # Setup central profiling engine
        profiling_config = ProfilingConfig()
        self._service_cache["profiling_engine"] = ProfilingEngine(profiling_config)
        
        logger.info("Profiling processing engines initialized")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all services."""
        health_status = {
            "status": "healthy",
            "services": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check each service
        for service_name, service in self._service_cache.items():
            try:
                if hasattr(service, 'health_check'):
                    health_status["services"][service_name] = await service.health_check()
                else:
                    health_status["services"][service_name] = "healthy"
            except Exception as e:
                health_status["services"][service_name] = f"unhealthy: {str(e)}"
                health_status["status"] = "degraded"
        
        return health_status
    
    async def cleanup(self) -> None:
        """Cleanup infrastructure resources."""
        if not self._initialized:
            return
            
        logger.info("Cleaning up data profiling infrastructure")
        
        # Cleanup database connections, external services, etc.
        self._service_cache.clear()
        self._initialized = False
        
        logger.info("Data profiling infrastructure cleanup completed")
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """Get a service from the cache."""
        return self._service_cache.get(service_name)


def create_data_profiling_container(testing: bool = False) -> DataProfilingContainer:
    """Factory method to create data profiling container."""
    container = DataProfilingContainer()
    
    if testing:
        # Override with test implementations
        logger.info("Creating test data profiling container")
        # Override repositories with in-memory implementations for testing
        container.data_profile_repository.override(
            providers.Singleton(InMemoryDataProfileRepository)
        )
    else:
        # Use production implementations
        logger.info("Creating production data profiling container")
        # Use the configured repository implementation
        container.data_profile_repository.override(
            container.data_profile_repository_impl
        )
    
    # Configure services with proper dependencies
    try:
        # Wire the container to the application modules (using relative imports)
        logger.info("Wiring data profiling container")
        
        # Manually wire key services for now since auto-wiring can be complex
        container.profiling_engine.override(
            providers.Singleton(
                ProfilingEngine,
                config=container.profiling_config
            )
        )
        
        logger.info("Data profiling container created successfully")
    except Exception as e:
        logger.error("Failed to wire data profiling container", error=str(e))
        # Continue without wiring - services can still be accessed manually
    
    return container


async def setup_infrastructure_integration(
    container: DataProfilingContainer,
    infrastructure: DataProfilingInfrastructureContainer
) -> None:
    """Setup integration between application container and infrastructure."""
    
    logger.info("Setting up data profiling infrastructure integration")
    
    # Initialize infrastructure
    await infrastructure.initialize()
    
    # Override container providers with infrastructure implementations
    if infrastructure.get_service("data_profile_repository"):
        container.data_profile_repository.override(
            providers.Singleton(lambda: infrastructure.get_service("data_profile_repository"))
        )
    
    if infrastructure.get_service("profiling_engine"):
        container.profiling_engine.override(
            providers.Singleton(lambda: infrastructure.get_service("profiling_engine"))
        )
    
    logger.info("Data profiling infrastructure integration completed")


def get_container() -> DataProfilingContainer:
    """Get the global container instance."""
    if not hasattr(get_container, '_container'):
        get_container._container = create_data_profiling_container()
    return get_container._container


def get_infrastructure() -> DataProfilingInfrastructureContainer:
    """Get the global infrastructure instance."""
    if not hasattr(get_infrastructure, '_infrastructure'):
        get_infrastructure._infrastructure = DataProfilingInfrastructureContainer()
    return get_infrastructure._infrastructure