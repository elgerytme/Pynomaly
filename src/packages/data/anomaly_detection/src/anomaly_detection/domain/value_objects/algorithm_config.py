"""Algorithm configuration value objects."""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum


class AlgorithmType(str, Enum):
    """Available algorithm types."""
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    LOCAL_OUTLIER_FACTOR = "lof"
    ENSEMBLE = "ensemble"


@dataclass(frozen=True)
class AlgorithmConfig:
    """Configuration for anomaly detection algorithms."""
    
    algorithm_type: AlgorithmType
    contamination: float = 0.1
    random_state: Optional[int] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.hyperparameters is None:
            object.__setattr__(self, 'hyperparameters', {})
        
        if not 0.0 <= self.contamination <= 0.5:
            raise ValueError("Contamination must be between 0.0 and 0.5")
    
    @property
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return (
            self.algorithm_type in AlgorithmType and
            0.0 <= self.contamination <= 0.5
        )
    
    def get_sklearn_params(self) -> Dict[str, Any]:
        """Get parameters in sklearn format."""
        params = {
            "contamination": self.contamination,
            **(self.hyperparameters or {})
        }
        
        if self.random_state is not None:
            params["random_state"] = self.random_state
            
        return params
    
    @classmethod
    def create_default(cls, algorithm_type: AlgorithmType) -> "AlgorithmConfig":
        """Create default configuration for algorithm type."""
        defaults = {
            AlgorithmType.ISOLATION_FOREST: {"n_estimators": 100},
            AlgorithmType.ONE_CLASS_SVM: {"nu": 0.05, "kernel": "rbf", "gamma": "scale"},
            AlgorithmType.LOCAL_OUTLIER_FACTOR: {"n_neighbors": 20},
            AlgorithmType.ENSEMBLE: {}
        }
        
        return cls(
            algorithm_type=algorithm_type,
            hyperparameters=defaults.get(algorithm_type, {})
        )