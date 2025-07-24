"""ML Operations interfaces for the anomaly detection domain.

This module defines the interfaces (ports) that the domain layer uses to
interact with machine learning operations like training, evaluation, and
model management. These interfaces decouple the domain from specific
ML library implementations.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

try:
    from data.processing.domain.entities.dataset import Dataset
except ImportError:
    from anomaly_detection.domain.entities.dataset import Dataset

try:
    from data.processing.domain.entities.model import Model
except ImportError:
    from anomaly_detection.domain.entities.model import Model


class ModelStatus(Enum):
    """Status of a model operation."""
    PENDING = "pending"
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATING = "evaluating"
    EVALUATED = "evaluated"
    FAILED = "failed"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"


class OptimizationObjective(Enum):
    """Optimization objectives for hyperparameter tuning."""
    MAXIMIZE_ACCURACY = "maximize_accuracy"
    MAXIMIZE_PRECISION = "maximize_precision"
    MAXIMIZE_RECALL = "maximize_recall"
    MAXIMIZE_F1 = "maximize_f1"
    MINIMIZE_FALSE_POSITIVES = "minimize_false_positives"
    MINIMIZE_FALSE_NEGATIVES = "minimize_false_negatives"
    CUSTOM = "custom"


@dataclass
class TrainingRequest:
    """Request for model training."""
    algorithm_name: str
    training_data: Dataset
    validation_data: Optional[Dataset] = None
    parameters: Dict[str, Any] = None
    optimization_objective: Optional[OptimizationObjective] = None
    max_training_time_seconds: Optional[int] = None
    created_by: str = "system"
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class TrainingResult:
    """Result of model training."""
    model: Model
    training_metrics: Dict[str, float]
    validation_metrics: Optional[Dict[str, float]] = None
    training_duration_seconds: float = 0.0
    feature_importance: Optional[Dict[str, float]] = None
    model_artifacts: Dict[str, Any] = None
    status: ModelStatus = ModelStatus.TRAINED
    
    def __post_init__(self):
        if self.model_artifacts is None:
            self.model_artifacts = {}


@dataclass
class HyperparameterOptimizationRequest:
    """Request for hyperparameter optimization."""
    algorithm_name: str
    training_data: Dataset
    validation_data: Optional[Dataset] = None
    parameter_space: Dict[str, Any] = None
    optimization_objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_F1
    max_evaluations: int = 50
    max_time_seconds: Optional[int] = None
    created_by: str = "system"
    
    def __post_init__(self):
        if self.parameter_space is None:
            self.parameter_space = {}


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    best_parameters: Dict[str, Any]
    best_score: float
    best_model: Model
    optimization_history: List[Dict[str, Any]]
    total_evaluations: int
    optimization_duration_seconds: float
    status: ModelStatus = ModelStatus.TRAINED


@dataclass
class ModelEvaluationRequest:
    """Request for model evaluation."""
    model: Model
    test_data: Dataset
    evaluation_metrics: List[str] = None
    created_by: str = "system"
    
    def __post_init__(self):
        if self.evaluation_metrics is None:
            self.evaluation_metrics = ["accuracy", "precision", "recall", "f1_score"]


@dataclass
class EvaluationResult:
    """Result of model evaluation."""
    metrics: Dict[str, float]
    confusion_matrix: Optional[List[List[int]]] = None
    roc_curve: Optional[Dict[str, List[float]]] = None
    precision_recall_curve: Optional[Dict[str, List[float]]] = None
    feature_importance: Optional[Dict[str, float]] = None
    evaluation_duration_seconds: float = 0.0
    status: ModelStatus = ModelStatus.EVALUATED


# ML Operations Interfaces (Ports)

class MLModelTrainingPort(ABC):
    """Interface for ML model training operations."""
    
    @abstractmethod
    async def train_model(self, request: TrainingRequest) -> TrainingResult:
        """Train a model with the given parameters.
        
        Args:
            request: Training request with algorithm, data, and parameters
            
        Returns:
            Training result with trained model and metrics
            
        Raises:
            TrainingError: If training fails
        """
        pass
    
    @abstractmethod
    async def optimize_hyperparameters(
        self, 
        request: HyperparameterOptimizationRequest
    ) -> OptimizationResult:
        """Optimize hyperparameters for the given algorithm.
        
        Args:
            request: Optimization request with search space and objective
            
        Returns:
            Optimization result with best parameters and model
            
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
            Evaluation result with metrics and analysis
            
        Raises:
            EvaluationError: If evaluation fails
        """
        pass
    
    @abstractmethod
    async def get_supported_algorithms(self) -> List[str]:
        """Get list of supported algorithms.
        
        Returns:
            List of algorithm names
        """
        pass
    
    @abstractmethod
    async def get_algorithm_parameters(self, algorithm_name: str) -> Dict[str, Any]:
        """Get parameter schema for an algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Parameter schema with types, defaults, and constraints
            
        Raises:
            UnsupportedAlgorithmError: If algorithm is not supported
        """
        pass


class MLModelRegistryPort(ABC):
    """Interface for ML model registry operations."""
    
    @abstractmethod
    async def register_model(
        self, 
        model: Model, 
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Register a trained model in the registry.
        
        Args:
            model: Trained model to register
            tags: Optional tags for the model
            
        Returns:
            Model ID in the registry
            
        Raises:
            RegistrationError: If registration fails
        """
        pass
    
    @abstractmethod
    async def retrieve_model(self, model_id: str) -> Model:
        """Retrieve a model from the registry.
        
        Args:
            model_id: ID of the model to retrieve
            
        Returns:
            Retrieved model
            
        Raises:
            RetrievalError: If model not found or retrieval fails
        """
        pass
    
    @abstractmethod
    async def query_models(
        self, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query models in the registry.
        
        Args:
            filters: Optional filters for the query
            
        Returns:
            List of model metadata
            
        Raises:
            QueryError: If query fails
        """
        pass
    
    @abstractmethod
    async def update_model(
        self, 
        model_id: str, 
        updates: Dict[str, Any]
    ) -> None:
        """Update model metadata.
        
        Args:
            model_id: ID of the model to update
            updates: Updates to apply
            
        Raises:
            UpdateError: If update fails
        """
        pass
    
    @abstractmethod
    async def delete_model(self, model_id: str) -> None:
        """Delete a model from the registry.
        
        Args:
            model_id: ID of the model to delete
            
        Raises:
            DeletionError: If deletion fails
        """
        pass


class MLFeatureEngineeringPort(ABC):
    """Interface for feature engineering operations."""
    
    @abstractmethod
    async def extract_features(
        self, 
        data: Dataset, 
        feature_config: Dict[str, Any]
    ) -> Dataset:
        """Extract features from raw data.
        
        Args:
            data: Input dataset
            feature_config: Feature extraction configuration
            
        Returns:
            Dataset with extracted features
            
        Raises:
            FeatureExtractionError: If feature extraction fails
        """
        pass
    
    @abstractmethod
    async def validate_features(
        self, 
        data: Dataset, 
        reference_data: Dataset
    ) -> Dict[str, Any]:
        """Validate features against reference data.
        
        Args:
            data: Data to validate
            reference_data: Reference data for validation
            
        Returns:
            Validation results
            
        Raises:
            ValidationError: If validation fails
        """
        pass
    
    @abstractmethod
    async def detect_feature_drift(
        self, 
        current_data: Dataset, 
        reference_data: Dataset
    ) -> Dict[str, Any]:
        """Detect feature drift between datasets.
        
        Args:
            current_data: Current dataset
            reference_data: Reference dataset
            
        Returns:
            Drift detection results
            
        Raises:
            DriftDetectionError: If drift detection fails
        """
        pass


class MLModelExplainabilityPort(ABC):
    """Interface for model explainability operations."""
    
    @abstractmethod
    async def explain_prediction(
        self, 
        model: Model, 
        data: Dataset, 
        explanation_type: str = "shap"
    ) -> Dict[str, Any]:
        """Explain model predictions.
        
        Args:
            model: Model to explain
            data: Data for explanation
            explanation_type: Type of explanation to generate
            
        Returns:
            Explanation results
            
        Raises:
            ExplanationError: If explanation generation fails
        """
        pass
    
    @abstractmethod
    async def generate_feature_importance(
        self, 
        model: Model, 
        data: Dataset
    ) -> Dict[str, float]:
        """Generate feature importance for the model.
        
        Args:
            model: Model to analyze
            data: Data for analysis
            
        Returns:
            Feature importance scores
            
        Raises:
            ExplanationError: If importance generation fails
        """
        pass
    
    @abstractmethod
    async def generate_explanation_report(
        self, 
        model: Model, 
        data: Dataset, 
        report_format: str = "html"
    ) -> str:
        """Generate an explanation report.
        
        Args:
            model: Model to explain
            data: Data for explanation
            report_format: Format of the report (html, pdf, json)
            
        Returns:
            Report content or path
            
        Raises:
            ReportGenerationError: If report generation fails
        """
        pass


# Exceptions
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
    """Exception raised during model query."""
    pass


class UpdateError(MLOperationError):
    """Exception raised during model update."""
    pass


class DeletionError(MLOperationError):
    """Exception raised during model deletion."""
    pass


class VersionCreationError(MLOperationError):
    """Exception raised during model version creation."""
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
    """Exception raised during model explanation."""
    pass


class ReportGenerationError(MLOperationError):
    """Exception raised during report generation."""
    pass


class UnsupportedAlgorithmError(MLOperationError):
    """Exception raised when algorithm is not supported."""
    pass