"""
Dependency injection setup for the FastAPI application.

This module handles the registration of all dependencies during application startup,
ensuring that all services are properly wired without circular imports.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pynomaly.infrastructure.dependencies import (
    register_dependency,
    register_dependency_provider,
    initialize_dependencies,
)
from pynomaly.infrastructure.config import Container

logger = logging.getLogger(__name__)


class DependencySetup:
    """Handles the setup and registration of all application dependencies."""
    
    def __init__(self, container: Container):
        """Initialize the dependency setup.
        
        Args:
            container: The dependency injection container
        """
        self.container = container
        self._registered_services: Dict[str, Any] = {}
    
    def register_core_services(self) -> None:
        """Register core application services."""
        logger.info("Registering core services...")
        
        # Authentication services
        try:
            auth_service = self.container.auth_service()
            register_dependency("auth_service", auth_service)
            self._registered_services["auth_service"] = auth_service
            logger.debug("Registered auth_service")
        except Exception as e:
            logger.warning(f"Failed to register auth_service: {e}")
            register_dependency("auth_service", None)
        
        # User services
        try:
            user_service = self.container.user_service()
            register_dependency("user_service", user_service)
            self._registered_services["user_service"] = user_service
            logger.debug("Registered user_service")
        except Exception as e:
            logger.warning(f"Failed to register user_service: {e}")
            register_dependency("user_service", None)
        
        # Detection services
        try:
            detection_service = self.container.detection_service()
            register_dependency("detection_service", detection_service)
            self._registered_services["detection_service"] = detection_service
            logger.debug("Registered detection_service")
        except Exception as e:
            logger.error(f"Failed to register detection_service: {e}")
            # Detection service is required, so we'll create a fallback
            from pynomaly.application.services import DetectionService
            fallback_service = DetectionService(
                dataset_repo=self.container.dataset_repository(),
                detector_repo=self.container.detector_repository(),
                result_repo=self.container.result_repository(),
            )
            register_dependency("detection_service", fallback_service)
            self._registered_services["detection_service"] = fallback_service
            logger.info("Registered fallback detection_service")
        
        # Model services
        try:
            model_service = self.container.model_service()
            register_dependency("model_service", model_service)
            self._registered_services["model_service"] = model_service
            logger.debug("Registered model_service")
        except Exception as e:
            logger.warning(f"Failed to register model_service: {e}")
            # Create a fallback model service
            from pynomaly.application.services import ModelPersistenceService
            fallback_service = ModelPersistenceService()
            register_dependency("model_service", fallback_service)
            self._registered_services["model_service"] = fallback_service
            logger.info("Registered fallback model_service")
    
    def register_infrastructure_services(self) -> None:
        """Register infrastructure services."""
        logger.info("Registering infrastructure services...")
        
        # Database services
        try:
            database_service = self.container.database_service()
            register_dependency("database_service", database_service)
            self._registered_services["database_service"] = database_service
            logger.debug("Registered database_service")
        except Exception as e:
            logger.warning(f"Failed to register database_service: {e}")
            register_dependency("database_service", None)
        
        # Cache services
        try:
            cache_service = self.container.cache_service()
            register_dependency("cache_service", cache_service)
            self._registered_services["cache_service"] = cache_service
            logger.debug("Registered cache_service")
        except Exception as e:
            logger.warning(f"Failed to register cache_service: {e}")
            register_dependency("cache_service", None)
        
        # Metrics services
        try:
            metrics_service = self.container.metrics_service()
            register_dependency("metrics_service", metrics_service)
            self._registered_services["metrics_service"] = metrics_service
            logger.debug("Registered metrics_service")
        except Exception as e:
            logger.warning(f"Failed to register metrics_service: {e}")
            register_dependency("metrics_service", None)
    
    def register_use_cases(self) -> None:
        """Register use cases as dependencies."""
        logger.info("Registering use cases...")
        
        # Core use cases
        try:
            detect_anomalies_use_case = self.container.detect_anomalies_use_case()
            register_dependency("detect_anomalies_use_case", detect_anomalies_use_case)
            self._registered_services["detect_anomalies_use_case"] = detect_anomalies_use_case
            logger.debug("Registered detect_anomalies_use_case")
        except Exception as e:
            logger.error(f"Failed to register detect_anomalies_use_case: {e}")
        
        try:
            train_detector_use_case = self.container.train_detector_use_case()
            register_dependency("train_detector_use_case", train_detector_use_case)
            self._registered_services["train_detector_use_case"] = train_detector_use_case
            logger.debug("Registered train_detector_use_case")
        except Exception as e:
            logger.error(f"Failed to register train_detector_use_case: {e}")
        
        try:
            evaluate_model_use_case = self.container.evaluate_model_use_case()
            register_dependency("evaluate_model_use_case", evaluate_model_use_case)
            self._registered_services["evaluate_model_use_case"] = evaluate_model_use_case
            logger.debug("Registered evaluate_model_use_case")
        except Exception as e:
            logger.error(f"Failed to register evaluate_model_use_case: {e}")
    
    def register_repositories(self) -> None:
        """Register repository dependencies."""
        logger.info("Registering repositories...")
        
        # Dataset repository
        try:
            dataset_repo = self.container.dataset_repository()
            register_dependency("dataset_repository", dataset_repo)
            self._registered_services["dataset_repository"] = dataset_repo
            logger.debug("Registered dataset_repository")
        except Exception as e:
            logger.error(f"Failed to register dataset_repository: {e}")
        
        # Detector repository
        try:
            detector_repo = self.container.detector_repository()
            register_dependency("detector_repository", detector_repo)
            self._registered_services["detector_repository"] = detector_repo
            logger.debug("Registered detector_repository")
        except Exception as e:
            logger.error(f"Failed to register detector_repository: {e}")
        
        # Result repository
        try:
            result_repo = self.container.result_repository()
            register_dependency("result_repository", result_repo)
            self._registered_services["result_repository"] = result_repo
            logger.debug("Registered result_repository")
        except Exception as e:
            logger.error(f"Failed to register result_repository: {e}")
    
    def setup_all(self) -> None:
        """Setup all dependencies."""
        logger.info("Setting up all dependencies...")
        
        try:
            self.register_core_services()
            self.register_infrastructure_services()
            self.register_use_cases()
            self.register_repositories()
            
            # Mark the dependency system as initialized
            initialize_dependencies()
            
            logger.info(f"Successfully registered {len(self._registered_services)} dependencies")
            
        except Exception as e:
            logger.error(f"Failed to setup dependencies: {e}")
            raise
    
    def get_registered_services(self) -> Dict[str, Any]:
        """Get all registered services.
        
        Returns:
            Dictionary of registered services
        """
        return self._registered_services.copy()


def setup_dependencies(container: Container) -> DependencySetup:
    """Setup all application dependencies.
    
    Args:
        container: The dependency injection container
        
    Returns:
        DependencySetup instance with all dependencies registered
    """
    setup = DependencySetup(container)
    setup.setup_all()
    return setup
