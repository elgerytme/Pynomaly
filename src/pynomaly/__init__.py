"""Pynomaly - State-of-the-art anomaly detection platform.

This package provides a comprehensive anomaly detection platform with:
- 40+ algorithms from PyOD, scikit-learn, and deep learning frameworks
- Clean architecture with Domain-Driven Design
- Production-ready features including monitoring and caching
- Multiple interfaces: REST API, CLI, and Progressive Web App
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"
__author__ = "Pynomaly Team"
__email__ = "team@pynomaly.io"
__license__ = "MIT"

# Core imports for convenience
from pynomaly.domain.entities import Anomaly, Dataset, DetectionResult, Detector
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
from pynomaly.infrastructure.config import Settings, create_container


# High-level API
def create_detector(
    name: str,
    algorithm: str = "IsolationForest",
    contamination: float = 0.1,
    **parameters: dict[str, Any],
) -> Detector:
    """Create a new anomaly detector.

    Args:
        name: Detector name
        algorithm: Algorithm to use (default: IsolationForest)
        contamination: Expected contamination rate (default: 0.1)
        **parameters: Additional algorithm parameters

    Returns:
        Detector instance

    Example:
        >>> detector = create_detector(
        ...     "My Detector",
        ...     algorithm="LOF",
        ...     contamination=0.05,
        ...     n_neighbors=20
        ... )
    """
    params = {"contamination": contamination}
    params.update(parameters)

    return Detector(name=name, algorithm=algorithm, parameters=params)


def load_dataset(
    data, name: str = "Dataset", target_column: str = None, **kwargs: dict[str, Any]
) -> Dataset:
    """Load a dataset for anomaly detection.

    Args:
        data: Data as DataFrame or file path
        name: Dataset name
        target_column: Column with labels (optional)
        **kwargs: Additional dataset parameters

    Returns:
        Dataset instance

    Example:
        >>> dataset = load_dataset(
        ...     "data.csv",
        ...     name="Sales Data",
        ...     target_column="is_fraud"
        ... )
    """
    from pathlib import Path

    import pandas as pd

    # Load data if path provided
    if isinstance(data, str | Path):
        path = Path(data)
        if path.suffix.lower() == ".csv":
            data = pd.read_csv(path)
        elif path.suffix.lower() in [".parquet", ".pq"]:
            data = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    return Dataset(name=name, data=data, target_column=target_column, **kwargs)


async def detect_anomalies(
    detector: Detector, dataset: Dataset, container=None
) -> DetectionResult:
    """Detect anomalies using a trained detector.

    Args:
        detector: Trained detector instance
        dataset: Dataset to analyze
        container: DI container (optional)

    Returns:
        Detection results

    Example:
        >>> results = await detect_anomalies(detector, dataset)
        >>> print(f"Found {results.n_anomalies} anomalies")
    """
    if container is None:
        container = create_container()

    # Ensure detector is trained
    if not detector.is_fitted:
        train_use_case = container.train_detector_use_case()
        from pynomaly.application.use_cases import TrainDetectorRequest

        train_request = TrainDetectorRequest(
            detector_id=detector.id,
            dataset=dataset,
            validate_data=True,
            save_model=True,
        )
        await train_use_case.execute(train_request)

    # Run detection
    detect_use_case = container.detect_anomalies_use_case()
    from pynomaly.application.use_cases import DetectAnomaliesRequest

    detect_request = DetectAnomaliesRequest(
        detector_id=detector.id,
        dataset=dataset,
        validate_features=True,
        save_results=True,
    )

    response = await detect_use_case.execute(detect_request)
    return response.result


# Convenience exports
__all__ = [
    # Version
    "__version__",
    # Entities
    "Anomaly",
    "Dataset",
    "Detector",
    "DetectionResult",
    # Value Objects
    "AnomalyScore",
    "ContaminationRate",
    # Infrastructure
    "Settings",
    "create_container",
    # High-level API
    "create_detector",
    "load_dataset",
    "detect_anomalies",
]
