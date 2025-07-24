"""Machine Learning Operations Interfaces (Ports).

This module defines the abstract interfaces for machine learning operations
that the anomaly detection domain requires. These interfaces represent the
"ports" in hexagonal architecture, defining contracts for external ML services
without coupling to specific implementations.

Following DDD principles, these interfaces belong to the domain layer and
define what the domain needs from external ML capabilities.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from ..entities.model import Model
from ..entities.dataset import Dataset
from ..entities.detection_result import DetectionResult


class ModelStatus(Enum):
    """Status of a machine learning model."""
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class OptimizationObjective(Enum):
    """Objective for hyperparameter optimization."""
    MAXIMIZE_F1 = "maximize_f1"
    MAXIMIZE_PRECISION = "maximize_precision"
    MAXIMIZE_RECALL = "maximize_recall"
    MINIMIZE_FALSE_POSITIVES = "minimize_false_positives"
    MINIMIZE_TRAINING_TIME = "minimize_training_time"


@dataclass
class TrainingRequest:
    """Request for training a machine learning model."""
    algorithm_name: str
    training_data: Dataset
    parameters: Dict[str, Any]
    validation_data: Optional[Dataset] = None
    optimization_objective: Optional[OptimizationObjective] = None
    max_training_time_seconds: Optional[int] = None
    created_by: str = "system"


@dataclass
class TrainingResult:
    """Result of model training operation."""
    model: Model
    training_metrics: Dict[str, float]
    validation_metrics: Optional[Dict[str, float]]
    training_duration_seconds: float
    feature_importance: Optional[Dict[str, float]]
    model_artifacts: Dict[str, Any]
    status: ModelStatus


@dataclass
class HyperparameterOptimizationRequest:
    """Request for hyperparameter optimization."""
    algorithm_name: str
    training_data: Dataset
    validation_data: Dataset
    parameter_space: Dict[str, Any]
    optimization_objective: OptimizationObjective
    max_trials: int = 100
    max_time_seconds: Optional[int] = None
    created_by: str = "system"


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    best_parameters: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    total_trials: int
    optimization_duration_seconds: float


@dataclass
class ModelEvaluationRequest:
    """Request for model evaluation."""
    model: Model
    test_data: Dataset
    evaluation_metrics: List[str]
    generate_explanations: bool = False


@dataclass
class EvaluationResult:
    """Result of model evaluation."""
    metrics: Dict[str, float]
    confusion_matrix: Optional[List[List[int]]]
    feature_importance: Optional[Dict[str, float]]
    prediction_explanations: Optional[List[Dict[str, Any]]]
    evaluation_duration_seconds: float


class MLModelTrainingPort(ABC):
    """Port for machine learning model training operations.
    
    This interface defines the contract for training anomaly detection models
    using external ML libraries and frameworks. Implementations should handle
    the translation between domain concepts and specific ML library APIs.
    """

    @abstractmethod
    async def train_model(self, request: TrainingRequest) -> TrainingResult:
        """Train a machine learning model for anomaly detection.
        
        Args:
            request: Training request with algorithm, data, and parameters
            
        Returns:
            Training result with trained model and metrics
            
        Raises:
            TrainingError: If training fails
            ValidationError: If request parameters are invalid
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def evaluate_model(self, request: ModelEvaluationRequest) -> EvaluationResult:
        """Evaluate a trained model on test data.
        
        Args:
            request: Evaluation request with model and test data
            
        Returns:
            Evaluation result with metrics and optional explanations
            
        Raises:
            EvaluationError: If evaluation fails
        """
        pass

    @abstractmethod
    async def predict_batch(
        self, 
        model: Model, 
        data: Dataset
    ) -> DetectionResult:
        """Make batch predictions using a trained model.
        
        Args:
            model: Trained model to use for predictions
            data: Input data for predictions
            
        Returns:
            Detection result with predictions and scores
            
        Raises:
            PredictionError: If prediction fails
        """
        pass

    @abstractmethod
    async def get_supported_algorithms(self) -> List[str]:
        """Get list of supported algorithms.
        
        Returns:
            List of algorithm names supported by this implementation
        """
        pass

    @abstractmethod
    async def get_algorithm_parameters(self, algorithm_name: str) -> Dict[str, Any]:
        """Get parameter schema for a specific algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Parameter schema with types, defaults, and constraints
            
        Raises:
            UnsupportedAlgorithmError: If algorithm is not supported
        """
        pass


class MLModelRegistryPort(ABC):
    """Port for machine learning model registry operations.
    
    This interface defines the contract for storing, retrieving, and managing
    trained models. It provides model versioning and metadata management
    capabilities.
    """

    @abstractmethod
    async def register_model(
        self, 
        model: Model, 
        metadata: Dict[str, Any]
    ) -> str:
        """Register a new model in the registry.
        
        Args:
            model: Trained model to register
            metadata: Additional metadata for the model
            
        Returns:
            Unique model ID assigned by the registry
            
        Raises:
            RegistrationError: If model registration fails
        """
        pass

    @abstractmethod
    async def get_model(self, model_id: str) -> Optional[Model]:
        """Retrieve a model by its ID.
        
        Args:
            model_id: Unique identifier of the model
            
        Returns:
            Model if found, None otherwise
            
        Raises:
            RetrievalError: If model retrieval fails
        """
        pass

    @abstractmethod
    async def list_models(
        self, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Model]:
        """List models with optional filtering.
        
        Args:
            filters: Optional filters (algorithm, status, created_by, etc.)
            
        Returns:
            List of models matching the filters
            
        Raises:
            QueryError: If listing operation fails
        """
        pass

    @abstractmethod
    async def update_model_status(self, model_id: str, status: ModelStatus) -> None:
        """Update the status of a model.
        
        Args:
            model_id: Unique identifier of the model
            status: New status to set
            
        Raises:
            UpdateError: If status update fails
        """
        pass

    @abstractmethod
    async def delete_model(self, model_id: str) -> None:
        """Delete a model from the registry.
        
        Args:
            model_id: Unique identifier of the model to delete
            
        Raises:
            DeletionError: If model deletion fails
        """
        pass

    @abstractmethod
    async def create_model_version(
        self, 
        model_id: str, 
        version_metadata: Dict[str, Any]
    ) -> str:
        """Create a new version of an existing model.
        
        Args:
            model_id: Base model identifier
            version_metadata: Metadata for the new version
            
        Returns:
            Version identifier
            
        Raises:
            VersionCreationError: If version creation fails
        """
        pass


class MLFeatureEngineeringPort(ABC):
    """Port for feature engineering and data preprocessing operations.
    
    This interface defines the contract for data preprocessing, feature
    extraction, and transformation operations required for anomaly detection.
    """

    @abstractmethod
    async def preprocess_data(
        self, 
        data: Dataset, 
        preprocessing_config: Dict[str, Any]
    ) -> Dataset:
        """Preprocess raw data for anomaly detection.
        
        Args:
            data: Raw input data
            preprocessing_config: Configuration for preprocessing steps
            
        Returns:
            Preprocessed dataset ready for training/inference
            
        Raises:
            PreprocessingError: If preprocessing fails
        """
        pass

    @abstractmethod
    async def extract_features(
        self, 
        data: Dataset, 
        feature_config: Dict[str, Any]
    ) -> Dataset:
        """Extract features from raw data.
        
        Args:
            data: Input data for feature extraction
            feature_config: Configuration for feature extraction
            
        Returns:
            Dataset with extracted features
            
        Raises:
            FeatureExtractionError: If feature extraction fails
        """
        pass

    @abstractmethod
    async def validate_data_quality(
        self, 
        data: Dataset
    ) -> Dict[str, Any]:
        """Validate the quality of input data.
        
        Args:
            data: Dataset to validate
            
        Returns:
            Data quality report with metrics and recommendations
            
        Raises:
            ValidationError: If validation fails
        """
        pass

    @abstractmethod
    async def detect_data_drift(
        self, 
        reference_data: Dataset, 
        current_data: Dataset
    ) -> Dict[str, Any]:
        """Detect data drift between reference and current data.
        
        Args:
            reference_data: Reference dataset (e.g., training data)
            current_data: Current dataset to compare
            
        Returns:
            Drift detection results with metrics and recommendations
            
        Raises:
            DriftDetectionError: If drift detection fails
        """
        pass


class MLModelExplainabilityPort(ABC):
    """Port for model explainability and interpretability operations.
    
    This interface defines the contract for generating explanations of
    anomaly detection model predictions and behaviors.
    """

    @abstractmethod
    async def explain_prediction(
        self, 
        model: Model, 
        instance: Dict[str, Any],
        explanation_type: str = "shap"
    ) -> Dict[str, Any]:
        """Generate explanation for a single prediction.
        
        Args:
            model: Trained model to explain
            instance: Input instance to explain
            explanation_type: Type of explanation (shap, lime, etc.)
            
        Returns:
            Explanation with feature contributions and metadata
            
        Raises:
            ExplanationError: If explanation generation fails
        """
        pass

    @abstractmethod
    async def explain_model_behavior(
        self, 
        model: Model, 
        sample_data: Dataset
    ) -> Dict[str, Any]:
        """Generate global explanation of model behavior.
        
        Args:
            model: Trained model to explain
            sample_data: Representative sample for global explanation
            
        Returns:
            Global explanation with feature importance and patterns
            
        Raises:
            ExplanationError: If explanation generation fails
        """
        pass

    @abstractmethod
    async def generate_model_report(
        self, 
        model: Model, 
        test_data: Dataset
    ) -> Dict[str, Any]:
        """Generate comprehensive model analysis report.
        
        Args:
            model: Trained model to analyze
            test_data: Test data for analysis
            
        Returns:
            Comprehensive report with performance, fairness, and interpretability
            
        Raises:
            ReportGenerationError: If report generation fails
        """
        pass


# Custom exceptions for ML operations
class MLOperationError(Exception):
    """Base exception for ML operation errors."""
    pass


class TrainingError(MLOperationError):
    """Exception raised during model training."""
    pass


class OptimizationError(MLOperationError):
    """Exception raised during hyperparameter optimization."""
    pass


class EvaluationError(MLOperationError):
    """Exception raised during model evaluation."""
    pass


class PredictionError(MLOperationError):
    """Exception raised during model prediction."""
    pass


class RegistrationError(MLOperationError):
    """Exception raised during model registration."""
    pass


class RetrievalError(MLOperationError):
    """Exception raised during model retrieval."""
    pass


class QueryError(MLOperationError):
    """Exception raised during query operations."""
    pass


class UpdateError(MLOperationError):
    """Exception raised during update operations."""
    pass


class DeletionError(MLOperationError):
    """Exception raised during deletion operations."""
    pass


class VersionCreationError(MLOperationError):
    """Exception raised during version creation."""
    pass


class PreprocessingError(MLOperationError):
    """Exception raised during data preprocessing."""
    pass


class FeatureExtractionError(MLOperationError):
    """Exception raised during feature extraction."""
    pass


class ValidationError(MLOperationError):
    """Exception raised during data validation."""
    pass


class DriftDetectionError(MLOperationError):
    """Exception raised during drift detection."""
    pass


class ExplanationError(MLOperationError):
    """Exception raised during explanation generation."""
    pass


class ReportGenerationError(MLOperationError):
    """Exception raised during report generation."""
    pass


class UnsupportedAlgorithmError(MLOperationError):
    """Exception raised for unsupported algorithms."""
    pass