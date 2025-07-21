"""Dependency injection container for data science package."""

from dependency_injector import containers, providers

# TODO: Implement within data platform science domain - from packages.data_science.domain.services.statistical_analysis_service import StatisticalAnalysisService
# TODO: Implement within data platform science domain - from packages.data_science.domain.services.model_training_service import ModelTrainingService
# TODO: Implement within data platform science domain - from packages.data_science.domain.services.model_lifecycle_service import ModelLifecycleService
# TODO: Implement within data platform science domain - from packages.data_science.domain.services.performance_baseline_service import PerformanceBaselineService
# TODO: Implement within data platform science domain - from packages.data_science.domain.services.performance_history_service import PerformanceHistoryService


class DataScienceContainer(containers.DeclarativeContainer):
    """Dependency injection container for data science package."""
    
    # Configuration
    config = providers.Configuration()
    
    # Domain Services
    statistical_analysis_service = providers.Singleton(
        StatisticalAnalysisService
    )
    
    model_training_service = providers.Singleton(
        ModelTrainingService
    )
    
    model_lifecycle_service = providers.Singleton(
        ModelLifecycleService
    )
    
    performance_baseline_service = providers.Singleton(
        PerformanceBaselineService
    )
    
    performance_history_service = providers.Singleton(
        PerformanceHistoryService
    )
    
    # Repository interfaces would be configured here when implementations are available
    # data_science_model_repository = providers.Singleton(...)
    # experiment_repository = providers.Singleton(...)
    # feature_store_repository = providers.Singleton(...)
    # dataset_profile_repository = providers.Singleton(...)
    
    # Application Services would be configured here
    # model_training_application_service = providers.Singleton(...)
    # experiment_management_service = providers.Singleton(...)
    # feature_store_management_service = providers.Singleton(...)