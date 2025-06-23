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
from pynomaly.infrastructure.repositories import (
    InMemoryDatasetRepository,
    InMemoryDetectorRepository,
    InMemoryResultRepository
)


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
        ]
    )
    
    return container