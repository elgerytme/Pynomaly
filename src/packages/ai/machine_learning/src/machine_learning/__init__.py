"""Machine Learning Package.

A comprehensive machine learning package providing AutoML, ensemble methods, 
explainable AI, active learning, and model optimization capabilities.
"""

__version__ = "0.1.0"
__author__ = "Machine Learning Team"
__email__ = "ml-team@domain.com"

# Domain layer exports
from .domain.entities.model import Model
from .domain.entities.automl import AutoML
from .domain.entities.training_job import TrainingJob
from .domain.entities.model_registry import ModelRegistry

from .domain.services.automl_service import AutoMLService
from .domain.services.ensemble_aggregator import EnsembleAggregator
from .domain.services.explainability_service import ExplainabilityService
from .domain.services.active_learning_service import ActiveLearningService

from .domain.value_objects.model_metrics import ModelMetrics
from .domain.value_objects.hyperparameters import Hyperparameters

# Application layer exports  
from .application.use_cases.automl_use_case import AutoMLUseCase
from .application.use_cases.evaluate_model import EvaluateModel
from .application.use_cases.manage_active_learning import ManageActiveLearning

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    # Domain entities
    "Model",
    "AutoML",
    "TrainingJob", 
    "ModelRegistry",
    # Domain services
    "AutoMLService",
    "EnsembleAggregator",
    "ExplainabilityService",
    "ActiveLearningService", 
    # Domain value objects
    "ModelMetrics",
    "Hyperparameters",
    # Application use cases
    "AutoMLUseCase",
    "EvaluateModel",
    "ManageActiveLearning",
]