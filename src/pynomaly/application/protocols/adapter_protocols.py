"""Application layer adapter protocols."""

from typing import Any, Protocol, runtime_checkable

import numpy as np

from ...domain.entities.dataset import Dataset
from ...domain.protocols.detection_protocols import (
    AlgorithmAdapterProtocol,
    DetectionAlgorithm,
    DetectionConfig,
)


@runtime_checkable
class ApplicationAlgorithmFactoryProtocol(Protocol):
    """Protocol for algorithm factory at application layer."""

    def create_algorithm(self, config: DetectionConfig) -> AlgorithmAdapterProtocol:
        """Create an algorithm adapter."""
        ...

    def get_available_algorithms(self) -> list[DetectionAlgorithm]:
        """Get list of available algorithms."""
        ...

    def validate_config(self, config: DetectionConfig) -> bool:
        """Validate algorithm configuration."""
        ...

    def get_default_config(self, algorithm: DetectionAlgorithm) -> DetectionConfig:
        """Get default configuration for algorithm."""
        ...


@runtime_checkable
class ApplicationDataLoaderProtocol(Protocol):
    """Protocol for data loading at application layer."""

    def load_csv(self, file_path: str, **kwargs: Any) -> Dataset:
        """Load dataset from CSV file."""
        ...

    def load_parquet(self, file_path: str, **kwargs: Any) -> Dataset:
        """Load dataset from Parquet file."""
        ...

    def load_json(self, file_path: str, **kwargs: Any) -> Dataset:
        """Load dataset from JSON file."""
        ...

    def load_from_url(self, url: str, format_type: str, **kwargs: Any) -> Dataset:
        """Load dataset from URL."""
        ...

    def validate_format(self, file_path: str) -> str:
        """Validate and detect file format."""
        ...

    def get_supported_formats(self) -> list[str]:
        """Get list of supported file formats."""
        ...


@runtime_checkable
class ApplicationModelSerializerProtocol(Protocol):
    """Protocol for model serialization at application layer."""

    def serialize_model(self, model: Any, format_type: str = "pickle") -> bytes:
        """Serialize a model to bytes."""
        ...

    def deserialize_model(self, data: bytes, format_type: str = "pickle") -> Any:
        """Deserialize a model from bytes."""
        ...

    def serialize_metadata(self, metadata: dict[str, Any]) -> bytes:
        """Serialize model metadata."""
        ...

    def deserialize_metadata(self, data: bytes) -> dict[str, Any]:
        """Deserialize model metadata."""
        ...

    def get_supported_formats(self) -> list[str]:
        """Get list of supported serialization formats."""
        ...

    def validate_model(self, model: Any) -> bool:
        """Validate that model can be serialized."""
        ...


@runtime_checkable
class ApplicationStreamingProtocol(Protocol):
    """Protocol for streaming operations at application layer."""

    def create_producer(self, config: dict[str, Any]) -> Any:
        """Create a streaming producer."""
        ...

    def create_consumer(self, config: dict[str, Any]) -> Any:
        """Create a streaming consumer."""
        ...

    def publish_message(self, topic: str, message: dict[str, Any]) -> bool:
        """Publish a message to a topic."""
        ...

    def consume_messages(self, topic: str, callback: Any) -> None:
        """Consume messages from a topic."""
        ...


@runtime_checkable
class ApplicationMLOpsProtocol(Protocol):
    """Protocol for MLOps operations at application layer."""

    def register_model(
        self,
        name: str,
        version: str,
        model_data: bytes,
        metadata: dict[str, Any]
    ) -> str:
        """Register a model in the model registry."""
        ...

    def deploy_model(self, model_id: str, deployment_config: dict[str, Any]) -> str:
        """Deploy a model to production."""
        ...

    def track_experiment(self, experiment_data: dict[str, Any]) -> str:
        """Track an ML experiment."""
        ...

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        """Log metrics for a run."""
        ...

    def log_artifacts(self, run_id: str, artifacts: dict[str, bytes]) -> None:
        """Log artifacts for a run."""
        ...


@runtime_checkable
class ApplicationAutoMLProtocol(Protocol):
    """Protocol for AutoML operations at application layer."""

    def search_hyperparameters(
        self,
        dataset: Dataset,
        algorithm: DetectionAlgorithm,
        search_space: dict[str, Any],
        max_trials: int = 100
    ) -> dict[str, Any]:
        """Search for optimal hyperparameters."""
        ...

    def auto_select_algorithm(
        self,
        dataset: Dataset,
        criteria: dict[str, Any] | None = None
    ) -> DetectionAlgorithm:
        """Automatically select best algorithm for dataset."""
        ...

    def optimize_ensemble(
        self,
        dataset: Dataset,
        algorithms: list[DetectionAlgorithm],
        optimization_criteria: str = "f1_score"
    ) -> dict[str, Any]:
        """Optimize ensemble configuration."""
        ...


@runtime_checkable
class ApplicationExplainabilityProtocol(Protocol):
    """Protocol for explainability operations at application layer."""

    def explain_prediction(
        self,
        model: Any,
        data_point: np.ndarray,
        method: str = "shap"
    ) -> dict[str, Any]:
        """Explain a single prediction."""
        ...

    def explain_global(
        self,
        model: Any,
        dataset: Dataset,
        method: str = "shap"
    ) -> dict[str, Any]:
        """Generate global explanations."""
        ...

    def feature_importance(
        self,
        model: Any,
        dataset: Dataset
    ) -> dict[str, float]:
        """Calculate feature importance."""
        ...

    def get_supported_methods(self) -> list[str]:
        """Get list of supported explanation methods."""
        ...
