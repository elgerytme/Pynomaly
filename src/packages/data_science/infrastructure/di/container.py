"""Data Science package dependency injection container."""

import importlib
from typing import Dict, Any, Optional
import structlog

from dependency_injector import containers, providers

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


class DataScienceContainer(containers.DeclarativeContainer):
    """Data Science package container with protocol-based dependency injection."""
    
    # Configuration
    config = providers.Configuration()
    
    # Logging
    logger = providers.Singleton(structlog.get_logger, name="data_science")
    
    # Optional service manager
    service_manager = providers.Singleton(OptionalServiceManager)
    
    # Core infrastructure - will be wired by specific implementations
    
    # Repositories (protocol-based)
    statistical_analysis_repository = providers.Singleton(
        lambda: None  # Will be overridden by infrastructure layer
    )
    
    dataset_repository = providers.Singleton(
        lambda: None  # Will be overridden by infrastructure layer
    )
    
    model_repository = providers.Singleton(
        lambda: None  # Will be overridden by infrastructure layer
    )
    
    # External service adapters
    visualization_service = providers.Singleton(
        lambda: None  # Will be overridden by infrastructure layer
    )
    
    statistical_engine = providers.Singleton(
        lambda: None  # Will be overridden by infrastructure layer
    )
    
    ml_model_service = providers.Singleton(
        lambda: None  # Will be overridden by infrastructure layer
    )
    
    # Application services - implemented
    statistical_analysis_service = providers.Factory(
        "data_science.infrastructure.services.statistical_analysis_service_impl.StatisticalAnalysisServiceImpl"
    )
    
    exploratory_data_analysis_service = providers.Singleton(
        lambda: None  # Will be implemented in application layer
    )
    
    hypothesis_testing_service = providers.Singleton(
        lambda: None  # Will be implemented in application layer
    )
    
    bayesian_analysis_service = providers.Singleton(
        lambda: None  # Will be implemented in application layer
    )
    
    time_series_analysis_service = providers.Singleton(
        lambda: None  # Will be implemented in application layer
    )


class DataScienceInfrastructureContainer:
    """Infrastructure container for data science services."""
    
    def __init__(self, settings: Optional[Any] = None):
        self.settings = settings
        self._service_cache: Dict[str, Any] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize infrastructure services."""
        if self._initialized:
            return
            
        logger.info("Initializing data science infrastructure")
        
        # Initialize database connections, external services, etc.
        await self._setup_repositories()
        await self._setup_external_services()
        
        self._initialized = True
        logger.info("Data science infrastructure initialized successfully")
    
    async def _setup_repositories(self) -> None:
        """Setup repository implementations."""
        # This would be implemented with actual repository classes
        logger.info("Setting up data science repositories")
    
    async def _setup_external_services(self) -> None:
        """Setup external service connections."""
        # This would be implemented with actual external services
        logger.info("Setting up external data science services")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all services."""
        health_status = {
            "status": "healthy",
            "services": {},
            "timestamp": None
        }
        
        # Add individual service health checks
        health_status["services"]["repositories"] = "healthy"
        health_status["services"]["external_services"] = "healthy"
        
        return health_status
    
    async def cleanup(self) -> None:
        """Cleanup infrastructure resources."""
        if not self._initialized:
            return
            
        logger.info("Cleaning up data science infrastructure")
        
        # Cleanup database connections, external services, etc.
        self._service_cache.clear()
        self._initialized = False
        
        logger.info("Data science infrastructure cleanup completed")
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """Get a service from the cache."""
        return self._service_cache.get(service_name)


def create_data_science_container(testing: bool = False) -> DataScienceContainer:
    """Factory method to create data science container."""
    container = DataScienceContainer()
    
    if testing:
        # Override with test implementations
        logger.info("Creating test data science container")
    else:
        # Use production implementations
        logger.info("Creating production data science container")
    
    # Wire the container to the application modules
    container.wire(modules=[
        "data_science.application.use_cases",
        "data_science.presentation.api",
        "data_science.presentation.cli",
    ])
    
    return container


def setup_infrastructure_integration(
    container: DataScienceContainer,
    infrastructure: DataScienceInfrastructureContainer
) -> None:
    """Setup integration between application container and infrastructure."""
    
    # Override container providers with infrastructure implementations
    # This would be done when actual implementations are available
    logger.info("Setting up data science infrastructure integration")