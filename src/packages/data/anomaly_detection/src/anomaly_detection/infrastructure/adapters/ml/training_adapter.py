"""Machine Learning Training Adapter.

This adapter implements the MLModelTrainingPort interface by integrating
with the machine_learning package. It translates between anomaly detection
domain concepts and the machine learning package's APIs.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from anomaly_detection.domain.interfaces.ml_operations import (
    MLModelTrainingPort,
    TrainingRequest,
    TrainingResult,
    HyperparameterOptimizationRequest,
    OptimizationResult,
    ModelEvaluationRequest,
    EvaluationResult,
    ModelStatus,
    OptimizationObjective,
    TrainingError,
    OptimizationError,
    EvaluationError,
    PredictionError,
    UnsupportedAlgorithmError,
)
try:
    from data.processing.domain.entities.model import Model
except ImportError:
    from anomaly_detection.domain.entities.model import Model
try:
    from data.processing.domain.entities.dataset import Dataset
except ImportError:
    from anomaly_detection.domain.entities.dataset import Dataset

try:
    from data.processing.domain.entities.detection_result import DetectionResult
except ImportError:
    from anomaly_detection.domain.entities.detection_result import DetectionResult

# Machine learning package imports
try:
    from machine_learning.domain.services.automl_service import AutoMLService
    from machine_learning.application.use_cases.train_model import TrainModelUseCase
    from machine_learning.application.use_cases.evaluate_model import EvaluateModelUseCase
    from machine_learning.application.use_cases.automl_optimization import AutoMLOptimizationUseCase
    from machine_learning.domain.dto.training_dto import TrainingRequestDTO
    from machine_learning.domain.dto.optimization_dto import OptimizationRequestDTO
    from machine_learning.domain.entities.model import Model as MLModel
    MACHINE_LEARNING_AVAILABLE = True
except ImportError:
    MACHINE_LEARNING_AVAILABLE = False
    # Create type stubs when machine learning package is not available
    AutoMLService = Any
    TrainModelUseCase = Any
    EvaluateModelUseCase = Any
    AutoMLOptimizationUseCase = Any
    TrainingRequestDTO = Any
    OptimizationRequestDTO = Any
    MLModel = Any


class MachinelearningTrainingAdapter(MLModelTrainingPort):
    """Adapter for machine learning model training operations.
    
    This adapter integrates the anomaly detection domain with the
    machine_learning package, providing model training, optimization,
    and evaluation capabilities.
    """

    def __init__(
        self,
        automl_service: AutoMLService,
        train_use_case: TrainModelUseCase,
        evaluate_use_case: EvaluateModelUseCase,
        optimization_use_case: AutoMLOptimizationUseCase,
    ):
        """Initialize the machine learning training adapter.
        
        Args:
            automl_service: AutoML service from machine_learning package
            train_use_case: Training use case implementation
            evaluate_use_case: Evaluation use case implementation
            optimization_use_case: Optimization use case implementation
        """
        if not MACHINE_LEARNING_AVAILABLE:
            raise ImportError(
                "machine_learning package is not available. "
                "Please install it to use this adapter."
            )
        
        self._automl_service = automl_service
        self._train_use_case = train_use_case
        self._evaluate_use_case = evaluate_use_case
        self._optimization_use_case = optimization_use_case
        self._logger = logging.getLogger(__name__)
        
        # Algorithm mapping from anomaly detection to machine learning package
        self._algorithm_mapping = {
            "isolation_forest": "IsolationForest",
            "one_class_svm": "OneClassSVM",
            "local_outlier_factor": "LocalOutlierFactor",
            "elliptic_envelope": "EllipticEnvelope",
            "autoencoder": "Autoencoder",
            "variational_autoencoder": "VariationalAutoencoder",
            "lstm_autoencoder": "LSTMAutoencoder",
        }
        
        self._logger.info("MachinelearningTrainingAdapter initialized successfully")

    async def train_model(self, request: TrainingRequest) -> TrainingResult:
        """Train a machine learning model for anomaly detection.
        
        Args:
            request: Training request with algorithm, data, and parameters
            
        Returns:
            Training result with trained model and metrics
            
        Raises:
            TrainingError: If training fails
            UnsupportedAlgorithmError: If algorithm is not supported
        """
        start_time = time.time()
        
        try:
            # Validate algorithm
            if request.algorithm_name not in self._algorithm_mapping:
                raise UnsupportedAlgorithmError(
                    f"Algorithm '{request.algorithm_name}' is not supported. "
                    f"Supported algorithms: {list(self._algorithm_mapping.keys())}"
                )
            
            # Convert to machine learning package format
            ml_algorithm = self._algorithm_mapping[request.algorithm_name]
            ml_training_request = self._create_ml_training_request(request, ml_algorithm)
            
            self._logger.info(f"Starting training for algorithm: {request.algorithm_name}")
            
            # Execute training through machine learning package
            ml_result = await self._train_use_case.execute(ml_training_request)
            
            # Convert result back to anomaly detection format
            training_result = self._convert_training_result(
                ml_result, request, start_time
            )
            
            self._logger.info(
                f"Training completed successfully in {training_result.training_duration_seconds:.2f}s"
            )
            
            return training_result
            
        except UnsupportedAlgorithmError:
            raise
        except Exception as e:
            self._logger.error(f"Training failed: {str(e)}")
            raise TrainingError(f"Model training failed: {str(e)}") from e

    async def optimize_hyperparameters(
        self, 
        request: HyperparameterOptimizationRequest
    ) -> OptimizationResult:
        """Optimize hyperparameters for a given algorithm.
        
        Args:
            request: Optimization request with search space and objective
            
        Returns:
            Optimization result with best parameters and history
            
        Raises:
            OptimizationError: If optimization fails
        """
        start_time = time.time()
        
        try:
            # Validate algorithm
            if request.algorithm_name not in self._algorithm_mapping:
                raise UnsupportedAlgorithmError(
                    f"Algorithm '{request.algorithm_name}' is not supported"
                )
            
            # Convert to machine learning package format
            ml_algorithm = self._algorithm_mapping[request.algorithm_name]
            ml_optimization_request = self._create_ml_optimization_request(
                request, ml_algorithm
            )
            
            self._logger.info(
                f"Starting hyperparameter optimization for: {request.algorithm_name}"
            )
            
            # Execute optimization through machine learning package
            ml_result = await self._optimization_use_case.execute(ml_optimization_request)
            
            # Convert result back to anomaly detection format
            optimization_result = self._convert_optimization_result(
                ml_result, start_time
            )
            
            self._logger.info(
                f"Optimization completed with best score: {optimization_result.best_score:.4f}"
            )
            
            return optimization_result
            
        except UnsupportedAlgorithmError:
            raise
        except Exception as e:
            self._logger.error(f"Optimization failed: {str(e)}")
            raise OptimizationError(f"Hyperparameter optimization failed: {str(e)}") from e

    async def evaluate_model(self, request: ModelEvaluationRequest) -> EvaluationResult:
        """Evaluate a trained model on test data.
        
        Args:
            request: Evaluation request with model and test data
            
        Returns:
            Evaluation result with metrics and optional explanations
            
        Raises:
            EvaluationError: If evaluation fails
        """
        start_time = time.time()
        
        try:
            # Convert to machine learning package format
            ml_evaluation_request = self._create_ml_evaluation_request(request)
            
            self._logger.info(f"Starting model evaluation for model: {request.model.id}")
            
            # Execute evaluation through machine learning package
            ml_result = await self._evaluate_use_case.execute(ml_evaluation_request)
            
            # Convert result back to anomaly detection format
            evaluation_result = self._convert_evaluation_result(ml_result, start_time)
            
            self._logger.info("Model evaluation completed successfully")
            
            return evaluation_result
            
        except Exception as e:
            self._logger.error(f"Evaluation failed: {str(e)}")
            raise EvaluationError(f"Model evaluation failed: {str(e)}") from e

    async def predict_batch(self, model: Model, data: Dataset) -> DetectionResult:
        """Make batch predictions using a trained model.
        
        Args:
            model: Trained model to use for predictions
            data: Input data for predictions
            
        Returns:
            Detection result with predictions and scores
            
        Raises:
            PredictionError: If prediction fails
        """
        try:
            # Convert model to machine learning package format
            ml_model = self._convert_model_to_ml_format(model)
            
            # Prepare data
            input_data = self._prepare_prediction_data(data)
            
            self._logger.info(f"Making batch predictions for {len(input_data)} samples")
            
            # Make predictions through machine learning package
            predictions = ml_model.predict(input_data)
            scores = ml_model.decision_function(input_data) if hasattr(ml_model, 'decision_function') else predictions
            
            # Convert to detection result
            detection_result = self._create_detection_result(
                predictions, scores, data, model
            )
            
            self._logger.info(f"Batch prediction completed: {len(predictions)} predictions made")
            
            return detection_result
            
        except Exception as e:
            self._logger.error(f"Batch prediction failed: {str(e)}")
            raise PredictionError(f"Batch prediction failed: {str(e)}") from e

    async def get_supported_algorithms(self) -> List[str]:
        """Get list of supported algorithms.
        
        Returns:
            List of algorithm names supported by this implementation
        """
        return list(self._algorithm_mapping.keys())

    async def get_algorithm_parameters(self, algorithm_name: str) -> Dict[str, Any]:
        """Get parameter schema for a specific algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Parameter schema with types, defaults, and constraints
            
        Raises:
            UnsupportedAlgorithmError: If algorithm is not supported
        """
        if algorithm_name not in self._algorithm_mapping:
            raise UnsupportedAlgorithmError(
                f"Algorithm '{algorithm_name}' is not supported"
            )
        
        # Algorithm parameter schemas
        parameter_schemas = {
            "isolation_forest": {
                "n_estimators": {"type": "int", "default": 100, "range": [10, 1000]},
                "contamination": {"type": "float", "default": 0.1, "range": [0.0, 0.5]},
                "max_samples": {"type": "str", "default": "auto", "options": ["auto", "int", "float"]},
                "max_features": {"type": "float", "default": 1.0, "range": [0.1, 1.0]},
                "bootstrap": {"type": "bool", "default": False},
                "random_state": {"type": "int", "default": None, "optional": True},
            },
            "one_class_svm": {
                "kernel": {"type": "str", "default": "rbf", "options": ["linear", "poly", "rbf", "sigmoid"]},
                "degree": {"type": "int", "default": 3, "range": [1, 10]},
                "gamma": {"type": "str", "default": "scale", "options": ["scale", "auto", "float"]},
                "coef0": {"type": "float", "default": 0.0, "range": [-10.0, 10.0]},
                "tol": {"type": "float", "default": 1e-3, "range": [1e-6, 1e-1]},
                "nu": {"type": "float", "default": 0.5, "range": [0.01, 1.0]},
                "shrinking": {"type": "bool", "default": True},
                "cache_size": {"type": "int", "default": 200, "range": [50, 2000]},
            },
            "local_outlier_factor": {
                "n_neighbors": {"type": "int", "default": 20, "range": [1, 100]},
                "algorithm": {"type": "str", "default": "auto", "options": ["auto", "ball_tree", "kd_tree", "brute"]},
                "leaf_size": {"type": "int", "default": 30, "range": [10, 100]},
                "metric": {"type": "str", "default": "minkowski", "options": ["euclidean", "manhattan", "minkowski"]},
                "p": {"type": "int", "default": 2, "range": [1, 5]},
                "contamination": {"type": "float", "default": 0.1, "range": [0.0, 0.5]},
            },
            "elliptic_envelope": {
                "store_precision": {"type": "bool", "default": True},
                "assume_centered": {"type": "bool", "default": False},
                "support_fraction": {"type": "float", "default": None, "optional": True, "range": [0.1, 1.0]},
                "contamination": {"type": "float", "default": 0.1, "range": [0.0, 0.5]},
                "random_state": {"type": "int", "default": None, "optional": True},
            },
            "autoencoder": {
                "encoding_dim": {"type": "int", "default": 32, "range": [8, 256]},
                "hidden_layers": {"type": "list", "default": [64, 32], "element_type": "int"},
                "activation": {"type": "str", "default": "relu", "options": ["relu", "tanh", "sigmoid"]},
                "optimizer": {"type": "str", "default": "adam", "options": ["adam", "sgd", "rmsprop"]},
                "loss": {"type": "str", "default": "mse", "options": ["mse", "mae", "huber"]},
                "epochs": {"type": "int", "default": 100, "range": [10, 1000]},
                "batch_size": {"type": "int", "default": 32, "range": [8, 256]},
                "learning_rate": {"type": "float", "default": 0.001, "range": [1e-5, 1e-1]},
                "dropout_rate": {"type": "float", "default": 0.2, "range": [0.0, 0.8]},
            },
            "variational_autoencoder": {
                "latent_dim": {"type": "int", "default": 16, "range": [4, 128]},
                "encoder_layers": {"type": "list", "default": [64, 32], "element_type": "int"},
                "decoder_layers": {"type": "list", "default": [32, 64], "element_type": "int"},
                "activation": {"type": "str", "default": "relu", "options": ["relu", "tanh", "sigmoid"]},
                "beta": {"type": "float", "default": 1.0, "range": [0.1, 10.0]},
                "epochs": {"type": "int", "default": 100, "range": [10, 1000]},
                "batch_size": {"type": "int", "default": 32, "range": [8, 256]},
                "learning_rate": {"type": "float", "default": 0.001, "range": [1e-5, 1e-1]},
            },
            "lstm_autoencoder": {
                "sequence_length": {"type": "int", "default": 10, "range": [5, 100]},
                "lstm_units": {"type": "int", "default": 50, "range": [10, 200]},
                "num_layers": {"type": "int", "default": 1, "range": [1, 5]},
                "dropout_rate": {"type": "float", "default": 0.2, "range": [0.0, 0.8]},
                "epochs": {"type": "int", "default": 100, "range": [10, 1000]},
                "batch_size": {"type": "int", "default": 32, "range": [8, 256]},
                "learning_rate": {"type": "float", "default": 0.001, "range": [1e-5, 1e-1]},
            },
        }
        
        return parameter_schemas.get(algorithm_name, {})

    def _create_ml_training_request(
        self, 
        request: TrainingRequest, 
        ml_algorithm: str
    ) -> TrainingRequestDTO:
        """Create machine learning package training request."""
        return TrainingRequestDTO(
            algorithm_name=ml_algorithm,
            training_data=self._convert_dataset_to_ml_format(request.training_data),
            validation_data=self._convert_dataset_to_ml_format(request.validation_data) if request.validation_data else None,
            parameters=request.parameters,
            optimization_objective=self._convert_optimization_objective(request.optimization_objective),
            max_training_time_seconds=request.max_training_time_seconds,
            created_by=request.created_by,
        )

    def _create_ml_optimization_request(
        self, 
        request: HyperparameterOptimizationRequest, 
        ml_algorithm: str
    ) -> OptimizationRequestDTO:
        """Create machine learning package optimization request."""
        return OptimizationRequestDTO(
            algorithm_name=ml_algorithm,
            training_data=self._convert_dataset_to_ml_format(request.training_data),
            validation_data=self._convert_dataset_to_ml_format(request.validation_data),
            parameter_space=request.parameter_space,
            optimization_objective=self._convert_optimization_objective(request.optimization_objective),
            max_trials=request.max_trials,
            max_time_seconds=request.max_time_seconds,
            created_by=request.created_by,
        )

    def _create_ml_evaluation_request(self, request: ModelEvaluationRequest) -> Dict[str, Any]:
        """Create machine learning package evaluation request."""
        return {
            "model": self._convert_model_to_ml_format(request.model),
            "test_data": self._convert_dataset_to_ml_format(request.test_data),
            "evaluation_metrics": request.evaluation_metrics,
            "generate_explanations": request.generate_explanations,
        }

    def _convert_dataset_to_ml_format(self, dataset: Optional[Dataset]) -> Optional[Any]:
        """Convert anomaly detection dataset to machine learning package format."""
        if dataset is None:
            return None
        
        # Convert dataset to format expected by machine learning package
        return {
            "data": dataset.data,
            "feature_names": dataset.feature_names,
            "metadata": dataset.metadata,
        }

    def _convert_model_to_ml_format(self, model: Model) -> Any:
        """Convert anomaly detection model to machine learning package format."""
        # This would need to be implemented based on the actual
        # machine learning package model format
        return model.model_object

    def _convert_optimization_objective(
        self, 
        objective: Optional[OptimizationObjective]
    ) -> Optional[str]:
        """Convert optimization objective to machine learning package format."""
        if objective is None:
            return None
        
        objective_mapping = {
            OptimizationObjective.MAXIMIZE_F1: "f1_score",
            OptimizationObjective.MAXIMIZE_PRECISION: "precision",
            OptimizationObjective.MAXIMIZE_RECALL: "recall",
            OptimizationObjective.MINIMIZE_FALSE_POSITIVES: "fpr",
            OptimizationObjective.MINIMIZE_TRAINING_TIME: "training_time",
        }
        
        return objective_mapping.get(objective, "f1_score")

    def _convert_training_result(
        self, 
        ml_result: Any, 
        request: TrainingRequest, 
        start_time: float
    ) -> TrainingResult:
        """Convert machine learning package training result to domain format."""
        training_duration = time.time() - start_time
        
        # Create anomaly detection model from ML result
        model = Model(
            id=str(uuid.uuid4()),
            name=f"{request.algorithm_name}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            algorithm=request.algorithm_name,
            parameters=request.parameters,
            model_object=ml_result.model,
            created_at=datetime.now(),
            version="1.0.0",
            metrics=ml_result.training_metrics,
            metadata={
                "training_duration": training_duration,
                "algorithm": request.algorithm_name,
                "created_by": request.created_by,
            }
        )
        
        return TrainingResult(
            model=model,
            training_metrics=ml_result.training_metrics,
            validation_metrics=getattr(ml_result, 'validation_metrics', None),
            training_duration_seconds=training_duration,
            feature_importance=getattr(ml_result, 'feature_importance', None),
            model_artifacts=getattr(ml_result, 'artifacts', {}),
            status=ModelStatus.TRAINED,
        )

    def _convert_optimization_result(
        self, 
        ml_result: Any, 
        start_time: float
    ) -> OptimizationResult:
        """Convert machine learning package optimization result to domain format."""
        optimization_duration = time.time() - start_time
        
        return OptimizationResult(
            best_parameters=ml_result.best_parameters,
            best_score=ml_result.best_score,
            optimization_history=getattr(ml_result, 'optimization_history', []),
            total_trials=getattr(ml_result, 'total_trials', 0),
            optimization_duration_seconds=optimization_duration,
        )

    def _convert_evaluation_result(
        self, 
        ml_result: Any, 
        start_time: float
    ) -> EvaluationResult:
        """Convert machine learning package evaluation result to domain format."""
        evaluation_duration = time.time() - start_time
        
        return EvaluationResult(
            metrics=ml_result.metrics,
            confusion_matrix=getattr(ml_result, 'confusion_matrix', None),
            feature_importance=getattr(ml_result, 'feature_importance', None),
            prediction_explanations=getattr(ml_result, 'explanations', None),
            evaluation_duration_seconds=evaluation_duration,
        )

    def _prepare_prediction_data(self, data: Dataset) -> Any:
        """Prepare data for prediction."""
        return data.data

    def _create_detection_result(
        self, 
        predictions: Any, 
        scores: Any, 
        data: Dataset, 
        model: Model
    ) -> DetectionResult:
        """Create detection result from predictions."""
        from anomaly_detection.domain.entities.anomaly import Anomaly
        
        # Convert predictions to anomalies
        anomalies = []
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            if pred == -1:  # Anomaly detected
                anomaly = Anomaly(
                    index=i,
                    score=float(score),
                    confidence=abs(float(score)),  # Use absolute score as confidence
                    features={name: data.data[i][j] for j, name in enumerate(data.feature_names)},
                    explanation={}
                )
                anomalies.append(anomaly)
        
        return DetectionResult(
            id=str(uuid.uuid4()),
            predictions=predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
            scores=scores.tolist() if hasattr(scores, 'tolist') else list(scores),
            anomalies=anomalies,
            algorithm=model.algorithm,
            parameters=model.parameters,
            metadata={
                "model_id": model.id,
                "prediction_timestamp": datetime.now().isoformat(),
                "total_samples": len(predictions),
                "anomaly_count": len(anomalies),
            }
        )