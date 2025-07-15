"""Domain protocols for detection services."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import numpy as np

from ..entities.anomaly import Anomaly
from ..entities.dataset import Dataset
from ..entities.detection_result import DetectionResult
from ..value_objects.anomaly_score import AnomalyScore
from ..value_objects.confidence_interval import ConfidenceInterval
from ..value_objects.contamination_rate import ContaminationRate


class DetectionAlgorithm(Enum):
    """Available detection algorithms."""
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    AUTOENCODER = "autoencoder"
    DBSCAN = "dbscan"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"
    PCA_BASED = "pca_based"
    KNN = "knn"
    ABOD = "abod"
    COPOD = "copod"
    ECOD = "ecod"
    LODA = "loda"
    MCD = "mcd"
    VAE = "vae"


class EnsembleMethod(Enum):
    """Ensemble methods for combining algorithms."""
    MAJORITY_VOTE = "majority_vote"
    AVERAGE = "average"
    WEIGHTED_AVERAGE = "weighted_average"
    MAX = "max"
    MIN = "min"
    MEDIAN = "median"
    DYNAMIC_SELECTION = "dynamic_selection"


class PreprocessingMethod(Enum):
    """Data preprocessing methods."""
    NONE = "none"
    STANDARD_SCALER = "standard_scaler"
    ROBUST_SCALER = "robust_scaler"
    NORMALIZATION = "normalization"
    PCA = "pca"


@dataclass
class DetectionConfig:
    """Configuration for detection algorithms."""
    algorithm: DetectionAlgorithm
    contamination_rate: float = 0.1
    n_estimators: int | None = None
    max_samples: int | float | None = None
    random_state: int | None = None
    preprocessing: PreprocessingMethod = PreprocessingMethod.NONE
    extra_params: dict[str, Any] | None = None


@dataclass
class EnsembleConfig:
    """Configuration for ensemble detection."""
    algorithms: list[DetectionConfig]
    ensemble_method: EnsembleMethod = EnsembleMethod.AVERAGE
    weights: list[float] | None = None
    voting_threshold: float = 0.5


@dataclass
class DetectionMetrics:
    """Metrics for detection performance."""
    total_samples: int
    total_anomalies: int
    anomaly_rate: float
    processing_time_seconds: float
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    auc_score: float | None = None
    confidence_scores: list[float] | None = None


class AlgorithmAdapterProtocol(Protocol):
    """Protocol for algorithm adapters."""

    def fit(self, data: np.ndarray) -> None:
        """Fit the algorithm to training data."""
        ...

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies in data."""
        ...

    def decision_function(self, data: np.ndarray) -> np.ndarray:
        """Get anomaly scores for data."""
        ...

    def get_params(self) -> dict[str, Any]:
        """Get algorithm parameters."""
        ...

    def set_params(self, **params: Any) -> None:
        """Set algorithm parameters."""
        ...


class PreprocessorProtocol(Protocol):
    """Protocol for data preprocessors."""

    def fit(self, data: np.ndarray) -> None:
        """Fit the preprocessor to data."""
        ...

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data."""
        ...

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        ...

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform data."""
        ...


class AlgorithmFactoryProtocol(Protocol):
    """Protocol for algorithm factories."""

    def create_algorithm(self, config: DetectionConfig) -> AlgorithmAdapterProtocol:
        """Create an algorithm adapter."""
        ...

    def create_preprocessor(self, method: PreprocessingMethod) -> PreprocessorProtocol:
        """Create a preprocessor."""
        ...

    def get_available_algorithms(self) -> list[DetectionAlgorithm]:
        """Get list of available algorithms."""
        ...


class EnsembleAggregatorProtocol(Protocol):
    """Protocol for ensemble aggregation."""

    def aggregate_predictions(
        self,
        predictions: list[np.ndarray],
        method: EnsembleMethod = EnsembleMethod.AVERAGE,
        weights: list[float] | None = None
    ) -> np.ndarray:
        """Aggregate predictions from multiple algorithms."""
        ...

    def aggregate_scores(
        self,
        scores: list[np.ndarray],
        method: EnsembleMethod = EnsembleMethod.AVERAGE,
        weights: list[float] | None = None
    ) -> np.ndarray:
        """Aggregate anomaly scores from multiple algorithms."""
        ...


class AdvancedDetectionServiceProtocol(Protocol):
    """Protocol for advanced detection services."""

    async def detect_anomalies_single(
        self,
        dataset: Dataset,
        config: DetectionConfig
    ) -> DetectionResult:
        """Detect anomalies using a single algorithm."""
        ...

    async def detect_anomalies_ensemble(
        self,
        dataset: Dataset,
        ensemble_config: EnsembleConfig
    ) -> DetectionResult:
        """Detect anomalies using ensemble methods."""
        ...

    async def evaluate_algorithm_performance(
        self,
        dataset: Dataset,
        ground_truth: np.ndarray | None,
        config: DetectionConfig
    ) -> DetectionMetrics:
        """Evaluate algorithm performance against ground truth."""
        ...

    async def compare_algorithms(
        self,
        dataset: Dataset,
        configs: list[DetectionConfig],
        ground_truth: np.ndarray | None = None
    ) -> dict[str, DetectionMetrics]:
        """Compare multiple algorithms on the same dataset."""
        ...

    async def auto_tune_parameters(
        self,
        dataset: Dataset,
        algorithm: DetectionAlgorithm,
        ground_truth: np.ndarray | None = None
    ) -> DetectionConfig:
        """Automatically tune algorithm parameters."""
        ...


class AdvancedDetectionService(ABC):
    """Abstract base class for advanced detection services."""

    def __init__(
        self,
        algorithm_factory: AlgorithmFactoryProtocol,
        ensemble_aggregator: EnsembleAggregatorProtocol
    ):
        self.algorithm_factory = algorithm_factory
        self.ensemble_aggregator = ensemble_aggregator

    @abstractmethod
    async def detect_anomalies_single(
        self,
        dataset: Dataset,
        config: DetectionConfig
    ) -> DetectionResult:
        """Detect anomalies using a single algorithm."""
        ...

    @abstractmethod
    async def detect_anomalies_ensemble(
        self,
        dataset: Dataset,
        ensemble_config: EnsembleConfig
    ) -> DetectionResult:
        """Detect anomalies using ensemble methods."""
        ...

    def _prepare_data(
        self,
        dataset: Dataset,
        preprocessing: PreprocessingMethod = PreprocessingMethod.NONE
    ) -> tuple[np.ndarray, PreprocessorProtocol | None]:
        """Prepare data for detection."""
        data = dataset.data

        if preprocessing == PreprocessingMethod.NONE:
            return data, None

        preprocessor = self.algorithm_factory.create_preprocessor(preprocessing)
        processed_data = preprocessor.fit_transform(data)
        return processed_data, preprocessor

    def _create_detection_result(
        self,
        dataset: Dataset,
        predictions: np.ndarray,
        scores: np.ndarray,
        config: DetectionConfig,
        processing_time: float
    ) -> DetectionResult:
        """Create detection result from predictions and scores."""
        anomaly_indices = np.where(predictions == 1)[0]
        anomalies = []

        for idx in anomaly_indices:
            anomaly = Anomaly(
                data_point=dataset.data[idx],
                score=AnomalyScore(scores[idx]),
                feature_contributions=None,
                context={"algorithm": config.algorithm.value, "index": int(idx)}
            )
            anomalies.append(anomaly)

        return DetectionResult(
            dataset_id=dataset.id,
            anomalies=anomalies,
            contamination_rate=ContaminationRate(config.contamination_rate),
            confidence_interval=ConfidenceInterval(
                lower_bound=float(np.min(scores)),
                upper_bound=float(np.max(scores)),
                confidence_level=0.95
            ),
            processing_time_seconds=processing_time,
            algorithm_used=config.algorithm.value,
            metadata={
                "total_samples": len(dataset.data),
                "total_anomalies": len(anomalies),
                "anomaly_rate": len(anomalies) / len(dataset.data),
                "config": {
                    "algorithm": config.algorithm.value,
                    "contamination_rate": config.contamination_rate,
                    "preprocessing": config.preprocessing.value
                }
            }
        )
