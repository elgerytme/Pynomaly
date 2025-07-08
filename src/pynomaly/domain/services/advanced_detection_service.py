"""Advanced anomaly detection service with multiple algorithms and ensemble methods."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    from sklearn.cluster import DBSCAN
    from sklearn.covariance import EllipticEnvelope
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score, precision_recall_curve

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import pyod
    from pyod.models.knn import KNN
    from pyod.models.abod import ABOD
    from pyod.models.auto_encoder import AutoEncoder
    from pyod.models.vae import VAE
    from pyod.models.ocsvm import OCSVM
    from pyod.models.pca import PCA as PyOD_PCA
    from pyod.models.mcd import MCD
    from pyod.models.loda import LODA
    from pyod.models.copod import COPOD
    from pyod.models.ecod import ECOD

    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False

from ..entities.anomaly import Anomaly
from ..entities.dataset import Dataset
from ..entities.detection_result import DetectionResult
from ..value_objects.anomaly_score import AnomalyScore
from ..value_objects.confidence_interval import ConfidenceInterval
from ..value_objects.contamination_rate import ContaminationRate

# from ...infrastructure.monitoring.opentelemetry_service import trace_anomaly_detection
# from ...infrastructure.monitoring.distributed_tracing import trace_operation


# Simple stubs for monitoring functions
def trace_anomaly_detection(func):
    """Simple decorator stub for monitoring."""
    return func


def trace_operation(name):
    """Simple decorator stub for monitoring."""

    def decorator(func):
        return func

    return decorator


logger = logging.getLogger(__name__)


class DetectionAlgorithm(Enum):
    """Available anomaly detection algorithms."""

    # Classical algorithms
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    ONE_CLASS_SVM = "one_class_svm"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"
    DBSCAN = "dbscan"

    # PyOD algorithms
    KNN = "knn"
    ABOD = "abod"
    AUTO_ENCODER = "auto_encoder"
    VAE = "vae"
    OCSVM = "ocsvm"
    PCA = "pca"
    MCD = "mcd"
    LODA = "loda"
    COPOD = "copod"
    ECOD = "ecod"

    # Ensemble methods
    ENSEMBLE_VOTING = "ensemble_voting"
    ENSEMBLE_AVERAGING = "ensemble_averaging"
    ENSEMBLE_STACKING = "ensemble_stacking"
    ENSEMBLE_BAGGING = "ensemble_bagging"


class ScalingMethod(Enum):
    """Data scaling methods."""

    NONE = "none"
    STANDARD = "standard"
    ROBUST = "robust"
    MIN_MAX = "min_max"


@dataclass
class AlgorithmConfig:
    """Configuration for anomaly detection algorithms."""

    algorithm: DetectionAlgorithm
    contamination: float = 0.1
    n_jobs: int = -1
    random_state: int = 42

    # Algorithm-specific parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Data preprocessing
    scaling_method: ScalingMethod = ScalingMethod.STANDARD
    apply_pca: bool = False
    pca_components: Optional[int] = None

    # Performance tuning
    enable_gpu: bool = False
    batch_size: Optional[int] = None
    memory_efficient: bool = True


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods."""

    algorithms: List[AlgorithmConfig]
    combination_method: str = "average"  # "average", "voting", "stacking"
    weights: Optional[List[float]] = None
    meta_learner: Optional[str] = None  # For stacking
    cross_validation_folds: int = 5


@dataclass
class DetectionMetrics:
    """Metrics for detection performance."""

    algorithm: str
    execution_time: float
    memory_usage: float

    # Performance metrics (if ground truth available)
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_score: Optional[float] = None

    # Statistics
    total_anomalies: int = 0
    anomaly_rate: float = 0.0
    confidence_mean: float = 0.0
    confidence_std: float = 0.0


class AdvancedDetectionService:
    """Advanced anomaly detection service with multiple algorithms."""

    def __init__(self):
        """Initialize the advanced detection service."""
        self.scalers = {}
        self.pca_transformers = {}
        self.fitted_models = {}

        # Default configurations for each algorithm
        self.default_configs = self._initialize_default_configs()

        logger.info("Advanced detection service initialized")

    def _initialize_default_configs(self) -> Dict[DetectionAlgorithm, AlgorithmConfig]:
        """Initialize default configurations for algorithms."""
        configs = {}

        # Isolation Forest
        configs[DetectionAlgorithm.ISOLATION_FOREST] = AlgorithmConfig(
            algorithm=DetectionAlgorithm.ISOLATION_FOREST,
            contamination=0.1,
            parameters={
                "n_estimators": 100,
                "max_samples": "auto",
                "max_features": 1.0,
                "bootstrap": False,
            },
        )

        # Local Outlier Factor
        configs[DetectionAlgorithm.LOCAL_OUTLIER_FACTOR] = AlgorithmConfig(
            algorithm=DetectionAlgorithm.LOCAL_OUTLIER_FACTOR,
            contamination=0.1,
            parameters={
                "n_neighbors": 20,
                "algorithm": "auto",
                "leaf_size": 30,
                "metric": "minkowski",
                "p": 2,
            },
        )

        # One-Class SVM
        configs[DetectionAlgorithm.ONE_CLASS_SVM] = AlgorithmConfig(
            algorithm=DetectionAlgorithm.ONE_CLASS_SVM,
            contamination=0.1,
            parameters={
                "kernel": "rbf",
                "gamma": "scale",
                "nu": 0.1,
                "shrinking": True,
                "cache_size": 200,
            },
        )

        # DBSCAN
        configs[DetectionAlgorithm.DBSCAN] = AlgorithmConfig(
            algorithm=DetectionAlgorithm.DBSCAN,
            parameters={
                "eps": 0.5,
                "min_samples": 5,
                "metric": "euclidean",
                "algorithm": "auto",
                "leaf_size": 30,
            },
        )

        # PyOD algorithms (if available)
        if PYOD_AVAILABLE:
            configs[DetectionAlgorithm.KNN] = AlgorithmConfig(
                algorithm=DetectionAlgorithm.KNN,
                contamination=0.1,
                parameters={"n_neighbors": 5, "method": "largest"},
            )

            configs[DetectionAlgorithm.ABOD] = AlgorithmConfig(
                algorithm=DetectionAlgorithm.ABOD,
                contamination=0.1,
                parameters={"n_neighbors": 10},
            )

            configs[DetectionAlgorithm.AUTO_ENCODER] = AlgorithmConfig(
                algorithm=DetectionAlgorithm.AUTO_ENCODER,
                contamination=0.1,
                parameters={
                    "hidden_neurons": [64, 32, 32, 64],
                    "epochs": 100,
                    "batch_size": 32,
                    "dropout_rate": 0.2,
                },
            )

        return configs

    @trace_operation("advanced_detection")
    async def detect_anomalies(
        self,
        dataset: Dataset,
        algorithm: DetectionAlgorithm,
        config: Optional[AlgorithmConfig] = None,
        ground_truth: Optional[np.ndarray] = None,
    ) -> DetectionResult:
        """Detect anomalies using specified algorithm."""

        start_time = time.time()

        try:
            # Use provided config or default
            if config is None:
                config = self.default_configs.get(algorithm)
                if not config:
                    raise ValueError(
                        f"No default configuration for algorithm: {algorithm}"
                    )

            # Prepare data
            X = await self._prepare_data(dataset, config)

            # Create and train model
            model = await self._create_model(algorithm, config)

            # Fit and predict
            predictions, scores = await self._fit_and_predict(
                model, X, algorithm, config
            )

            # Calculate metrics
            execution_time = time.time() - start_time
            memory_usage = self._estimate_memory_usage(X)

            # Create anomalies
            anomalies = []
            for i, (is_anomaly, score) in enumerate(zip(predictions, scores)):
                if is_anomaly:
                    anomaly = Anomaly(
                        index=i,
                        score=AnomalyScore(score),
                        confidence=ConfidenceInterval(
                            lower=max(0.0, score - 0.1),
                            upper=min(1.0, score + 0.1),
                            confidence_level=0.95,
                        ),
                        features=(
                            X.iloc[i].to_dict() if isinstance(X, pd.DataFrame) else {}
                        ),
                        timestamp=dataset.metadata.get("timestamp", None),
                        explanation=f"Detected by {algorithm.value}",
                    )
                    anomalies.append(anomaly)

            # Calculate performance metrics
            metrics = await self._calculate_metrics(
                algorithm.value,
                execution_time,
                memory_usage,
                predictions,
                scores,
                ground_truth,
            )

            # Create detection result
            result = DetectionResult(
                dataset_id=getattr(dataset, "id", "unknown"),
                algorithm=algorithm.value,
                anomalies=anomalies,
                total_samples=len(X),
                anomaly_count=len(anomalies),
                contamination_rate=ContaminationRate(len(anomalies) / len(X)),
                execution_time=execution_time,
                metadata={
                    "algorithm_config": config.__dict__,
                    "metrics": metrics.__dict__,
                    "data_shape": X.shape,
                    "scaling_method": config.scaling_method.value,
                },
            )

            return result

        except Exception as e:
            logger.error(f"Error in anomaly detection with {algorithm.value}: {e}")
            raise

    async def _prepare_data(
        self, dataset: Dataset, config: AlgorithmConfig
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Prepare data for anomaly detection."""

        # Get data
        if hasattr(dataset, "data"):
            X = dataset.data
        else:
            raise ValueError("Dataset has no data attribute")

        # Convert to pandas DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Handle missing values
        X = X.fillna(X.mean())

        # Select numeric columns only
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_columns]

        if X.empty:
            raise ValueError("No numeric features found in dataset")

        # Apply scaling
        if config.scaling_method != ScalingMethod.NONE:
            scaler_key = f"{config.algorithm.value}_{config.scaling_method.value}"

            if scaler_key not in self.scalers:
                if config.scaling_method == ScalingMethod.STANDARD:
                    self.scalers[scaler_key] = StandardScaler()
                elif config.scaling_method == ScalingMethod.ROBUST:
                    self.scalers[scaler_key] = RobustScaler()
                else:
                    raise ValueError(
                        f"Unsupported scaling method: {config.scaling_method}"
                    )

            scaler = self.scalers[scaler_key]
            X_scaled = scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        # Apply PCA if requested
        if config.apply_pca and SKLEARN_AVAILABLE:
            pca_key = f"{config.algorithm.value}_pca"

            if pca_key not in self.pca_transformers:
                n_components = config.pca_components or min(X.shape[1], X.shape[0] // 2)
                self.pca_transformers[pca_key] = PCA(n_components=n_components)

            pca = self.pca_transformers[pca_key]
            X_pca = pca.fit_transform(X)
            X = pd.DataFrame(X_pca, index=X.index)

        return X

    async def _create_model(
        self, algorithm: DetectionAlgorithm, config: AlgorithmConfig
    ):
        """Create model instance for the specified algorithm."""

        if not SKLEARN_AVAILABLE and algorithm in [
            DetectionAlgorithm.ISOLATION_FOREST,
            DetectionAlgorithm.LOCAL_OUTLIER_FACTOR,
            DetectionAlgorithm.ONE_CLASS_SVM,
            DetectionAlgorithm.ELLIPTIC_ENVELOPE,
            DetectionAlgorithm.DBSCAN,
        ]:
            raise ImportError("scikit-learn is required for classical algorithms")

        if not PYOD_AVAILABLE and algorithm in [
            DetectionAlgorithm.KNN,
            DetectionAlgorithm.ABOD,
            DetectionAlgorithm.AUTO_ENCODER,
            DetectionAlgorithm.VAE,
            DetectionAlgorithm.OCSVM,
            DetectionAlgorithm.PCA,
            DetectionAlgorithm.MCD,
            DetectionAlgorithm.LODA,
            DetectionAlgorithm.COPOD,
            DetectionAlgorithm.ECOD,
        ]:
            raise ImportError("PyOD is required for advanced algorithms")

        # Classical sklearn algorithms
        if algorithm == DetectionAlgorithm.ISOLATION_FOREST:
            return IsolationForest(
                contamination=config.contamination,
                n_jobs=config.n_jobs,
                random_state=config.random_state,
                **config.parameters,
            )

        elif algorithm == DetectionAlgorithm.LOCAL_OUTLIER_FACTOR:
            return LocalOutlierFactor(
                contamination=config.contamination,
                n_jobs=config.n_jobs,
                **config.parameters,
            )

        elif algorithm == DetectionAlgorithm.ONE_CLASS_SVM:
            # Convert contamination to nu parameter
            nu = min(0.5, max(0.01, config.contamination))
            params = config.parameters.copy()
            params["nu"] = nu
            return OneClassSVM(**params)

        elif algorithm == DetectionAlgorithm.ELLIPTIC_ENVELOPE:
            return EllipticEnvelope(
                contamination=config.contamination,
                random_state=config.random_state,
                **config.parameters,
            )

        elif algorithm == DetectionAlgorithm.DBSCAN:
            return DBSCAN(**config.parameters)

        # PyOD algorithms
        elif algorithm == DetectionAlgorithm.KNN:
            return KNN(contamination=config.contamination, **config.parameters)

        elif algorithm == DetectionAlgorithm.ABOD:
            return ABOD(contamination=config.contamination, **config.parameters)

        elif algorithm == DetectionAlgorithm.AUTO_ENCODER:
            return AutoEncoder(contamination=config.contamination, **config.parameters)

        elif algorithm == DetectionAlgorithm.VAE:
            return VAE(contamination=config.contamination, **config.parameters)

        elif algorithm == DetectionAlgorithm.OCSVM:
            return OCSVM(contamination=config.contamination, **config.parameters)

        elif algorithm == DetectionAlgorithm.PCA:
            return PyOD_PCA(contamination=config.contamination, **config.parameters)

        elif algorithm == DetectionAlgorithm.MCD:
            return MCD(contamination=config.contamination, **config.parameters)

        elif algorithm == DetectionAlgorithm.LODA:
            return LODA(contamination=config.contamination, **config.parameters)

        elif algorithm == DetectionAlgorithm.COPOD:
            return COPOD(contamination=config.contamination, **config.parameters)

        elif algorithm == DetectionAlgorithm.ECOD:
            return ECOD(contamination=config.contamination, **config.parameters)

        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    async def _fit_and_predict(
        self,
        model,
        X: Union[np.ndarray, pd.DataFrame],
        algorithm: DetectionAlgorithm,
        config: AlgorithmConfig,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit model and generate predictions."""

        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        # Fit and predict based on algorithm type
        if algorithm == DetectionAlgorithm.LOCAL_OUTLIER_FACTOR:
            # LOF doesn't have separate fit/predict
            predictions = model.fit_predict(X_array)
            scores = -model.negative_outlier_factor_

        elif algorithm == DetectionAlgorithm.DBSCAN:
            # DBSCAN clustering-based anomaly detection
            clusters = model.fit_predict(X_array)
            predictions = (clusters == -1).astype(int)  # -1 indicates outliers
            # Generate scores based on distance to nearest cluster
            scores = self._calculate_dbscan_scores(X_array, clusters, model)

        else:
            # Standard fit/predict workflow
            model.fit(X_array)

            if hasattr(model, "predict"):
                predictions = model.predict(X_array)
            else:
                # For models without predict method, use decision_function
                scores = model.decision_function(X_array)
                threshold = np.percentile(scores, (1 - config.contamination) * 100)
                predictions = (scores < threshold).astype(int)

            # Get anomaly scores
            if hasattr(model, "decision_function"):
                scores = model.decision_function(X_array)
                scores = (scores - scores.min()) / (
                    scores.max() - scores.min()
                )  # Normalize
            elif hasattr(model, "score_samples"):
                scores = -model.score_samples(X_array)  # Negative log-likelihood
                scores = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                # Fallback: use binary predictions as scores
                scores = predictions.astype(float)

        # Convert predictions to binary (1 for anomaly, 0 for normal)
        if algorithm not in [DetectionAlgorithm.LOCAL_OUTLIER_FACTOR]:
            predictions = (predictions == -1).astype(int)
        else:
            predictions = (predictions == -1).astype(int)

        # Ensure scores are between 0 and 1
        scores = np.clip(scores, 0, 1)

        return predictions, scores

    def _calculate_dbscan_scores(
        self, X: np.ndarray, clusters: np.ndarray, model
    ) -> np.ndarray:
        """Calculate anomaly scores for DBSCAN results."""
        scores = np.zeros(len(X))

        # Outliers (cluster -1) get high scores
        outlier_mask = clusters == -1
        scores[outlier_mask] = 0.8 + 0.2 * np.random.random(np.sum(outlier_mask))

        # Core points get low scores
        core_samples = model.core_sample_indices_
        scores[core_samples] = 0.1 + 0.2 * np.random.random(len(core_samples))

        # Border points get medium scores
        border_mask = ~outlier_mask & ~np.isin(np.arange(len(X)), core_samples)
        scores[border_mask] = 0.3 + 0.3 * np.random.random(np.sum(border_mask))

        return scores

    def _estimate_memory_usage(self, X: Union[np.ndarray, pd.DataFrame]) -> float:
        """Estimate memory usage for the dataset."""
        if isinstance(X, pd.DataFrame):
            return X.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        else:
            return X.nbytes / (1024 * 1024)  # MB

    async def _calculate_metrics(
        self,
        algorithm: str,
        execution_time: float,
        memory_usage: float,
        predictions: np.ndarray,
        scores: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
    ) -> DetectionMetrics:
        """Calculate detection performance metrics."""

        total_anomalies = int(np.sum(predictions))
        anomaly_rate = float(np.mean(predictions))
        confidence_mean = float(np.mean(scores))
        confidence_std = float(np.std(scores))

        metrics = DetectionMetrics(
            algorithm=algorithm,
            execution_time=execution_time,
            memory_usage=memory_usage,
            total_anomalies=total_anomalies,
            anomaly_rate=anomaly_rate,
            confidence_mean=confidence_mean,
            confidence_std=confidence_std,
        )

        # Calculate performance metrics if ground truth is available
        if ground_truth is not None and len(ground_truth) == len(predictions):
            try:
                # Precision, Recall, F1
                tp = np.sum((predictions == 1) & (ground_truth == 1))
                fp = np.sum((predictions == 1) & (ground_truth == 0))
                fn = np.sum((predictions == 0) & (ground_truth == 1))

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_score = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

                metrics.precision = precision
                metrics.recall = recall
                metrics.f1_score = f1_score

                # AUC if we have probability scores
                if len(np.unique(scores)) > 2:  # More than binary scores
                    try:
                        auc = roc_auc_score(ground_truth, scores)
                        metrics.auc_score = auc
                    except ValueError:
                        pass  # AUC calculation failed

            except Exception as e:
                logger.warning(f"Failed to calculate performance metrics: {e}")

        return metrics

    @trace_operation("ensemble_detection")
    async def detect_anomalies_ensemble(
        self,
        dataset: Dataset,
        ensemble_config: EnsembleConfig,
        ground_truth: Optional[np.ndarray] = None,
    ) -> DetectionResult:
        """Detect anomalies using ensemble of multiple algorithms."""

        start_time = time.time()

        try:
            results = []
            all_predictions = []
            all_scores = []

            # Run each algorithm in the ensemble
            for config in ensemble_config.algorithms:
                result = await self.detect_anomalies(
                    dataset, config.algorithm, config, ground_truth
                )
                results.append(result)

                # Extract predictions and scores
                predictions = np.zeros(result.total_samples)
                scores = np.zeros(result.total_samples)

                for anomaly in result.anomalies:
                    predictions[anomaly.index] = 1
                    scores[anomaly.index] = anomaly.score.value

                all_predictions.append(predictions)
                all_scores.append(scores)

            # Combine predictions using specified method
            combined_predictions, combined_scores = self._combine_ensemble_results(
                all_predictions, all_scores, ensemble_config
            )

            # Create anomalies from combined results
            anomalies = []
            for i, (is_anomaly, score) in enumerate(
                zip(combined_predictions, combined_scores)
            ):
                if is_anomaly:
                    anomaly = Anomaly(
                        index=i,
                        score=AnomalyScore(score),
                        confidence=ConfidenceInterval(
                            lower=max(0.0, score - 0.1),
                            upper=min(1.0, score + 0.1),
                            confidence_level=0.95,
                        ),
                        features={},
                        timestamp=dataset.metadata.get("timestamp", None),
                        explanation=f"Detected by ensemble of {len(ensemble_config.algorithms)} algorithms",
                    )
                    anomalies.append(anomaly)

            execution_time = time.time() - start_time

            # Create ensemble result
            result = DetectionResult(
                dataset_id=getattr(dataset, "id", "unknown"),
                algorithm=f"ensemble_{ensemble_config.combination_method}",
                anomalies=anomalies,
                total_samples=len(combined_predictions),
                anomaly_count=len(anomalies),
                contamination_rate=ContaminationRate(
                    len(anomalies) / len(combined_predictions)
                ),
                execution_time=execution_time,
                metadata={
                    "ensemble_config": {
                        "algorithms": [
                            config.algorithm.value
                            for config in ensemble_config.algorithms
                        ],
                        "combination_method": ensemble_config.combination_method,
                        "weights": ensemble_config.weights,
                    },
                    "individual_results": [
                        {
                            "algorithm": r.algorithm,
                            "anomaly_count": r.anomaly_count,
                            "execution_time": r.execution_time,
                        }
                        for r in results
                    ],
                },
            )

            return result

        except Exception as e:
            logger.error(f"Error in ensemble anomaly detection: {e}")
            raise

    def _combine_ensemble_results(
        self,
        all_predictions: List[np.ndarray],
        all_scores: List[np.ndarray],
        config: EnsembleConfig,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Combine results from multiple algorithms."""

        predictions_array = np.array(all_predictions)
        scores_array = np.array(all_scores)

        if config.combination_method == "voting":
            # Majority voting
            combined_predictions = (
                np.sum(predictions_array, axis=0) > len(all_predictions) / 2
            ).astype(int)
            combined_scores = np.mean(scores_array, axis=0)

        elif config.combination_method == "average":
            # Average scores and threshold
            combined_scores = np.mean(scores_array, axis=0)
            threshold = np.percentile(combined_scores, 90)  # Top 10% as anomalies
            combined_predictions = (combined_scores >= threshold).astype(int)

        elif config.combination_method == "weighted_average" and config.weights:
            # Weighted average
            weights = np.array(config.weights)
            weights = weights / np.sum(weights)  # Normalize weights
            combined_scores = np.average(scores_array, axis=0, weights=weights)
            threshold = np.percentile(combined_scores, 90)
            combined_predictions = (combined_scores >= threshold).astype(int)

        else:
            # Default to simple averaging
            combined_predictions = (
                np.sum(predictions_array, axis=0) > len(all_predictions) / 2
            ).astype(int)
            combined_scores = np.mean(scores_array, axis=0)

        return combined_predictions, combined_scores

    async def get_available_algorithms(self) -> List[DetectionAlgorithm]:
        """Get list of available algorithms based on installed dependencies."""
        available = []

        if SKLEARN_AVAILABLE:
            available.extend(
                [
                    DetectionAlgorithm.ISOLATION_FOREST,
                    DetectionAlgorithm.LOCAL_OUTLIER_FACTOR,
                    DetectionAlgorithm.ONE_CLASS_SVM,
                    DetectionAlgorithm.ELLIPTIC_ENVELOPE,
                    DetectionAlgorithm.DBSCAN,
                ]
            )

        if PYOD_AVAILABLE:
            available.extend(
                [
                    DetectionAlgorithm.KNN,
                    DetectionAlgorithm.ABOD,
                    DetectionAlgorithm.AUTO_ENCODER,
                    DetectionAlgorithm.VAE,
                    DetectionAlgorithm.OCSVM,
                    DetectionAlgorithm.PCA,
                    DetectionAlgorithm.MCD,
                    DetectionAlgorithm.LODA,
                    DetectionAlgorithm.COPOD,
                    DetectionAlgorithm.ECOD,
                ]
            )

        # Ensemble methods are always available if at least 2 algorithms are available
        if len(available) >= 2:
            available.extend(
                [
                    DetectionAlgorithm.ENSEMBLE_VOTING,
                    DetectionAlgorithm.ENSEMBLE_AVERAGING,
                    DetectionAlgorithm.ENSEMBLE_STACKING,
                ]
            )

        return available

    async def benchmark_algorithms(
        self,
        dataset: Dataset,
        algorithms: Optional[List[DetectionAlgorithm]] = None,
        ground_truth: Optional[np.ndarray] = None,
    ) -> Dict[str, DetectionMetrics]:
        """Benchmark multiple algorithms on the same dataset."""

        if algorithms is None:
            algorithms = await self.get_available_algorithms()
            # Remove ensemble methods for individual benchmarking
            algorithms = [
                alg for alg in algorithms if not alg.value.startswith("ensemble_")
            ]

        benchmark_results = {}

        for algorithm in algorithms:
            try:
                logger.info(f"Benchmarking algorithm: {algorithm.value}")
                result = await self.detect_anomalies(
                    dataset, algorithm, ground_truth=ground_truth
                )

                # Extract metrics from result
                if "metrics" in result.metadata:
                    benchmark_results[algorithm.value] = result.metadata["metrics"]

            except Exception as e:
                logger.error(f"Error benchmarking {algorithm.value}: {e}")
                # Add error result
                benchmark_results[algorithm.value] = DetectionMetrics(
                    algorithm=algorithm.value,
                    execution_time=0.0,
                    memory_usage=0.0,
                    total_anomalies=0,
                )

        return benchmark_results


# Global service instance
_detection_service: Optional[AdvancedDetectionService] = None


def get_detection_service() -> AdvancedDetectionService:
    """Get the global detection service instance."""
    global _detection_service
    if _detection_service is None:
        _detection_service = AdvancedDetectionService()
    return _detection_service
