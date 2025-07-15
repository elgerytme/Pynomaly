"""
Algorithm Configuration Value Object

Represents immutable configuration for anomaly detection algorithms.
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class AlgorithmType(Enum):
    """Supported anomaly detection algorithms."""
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    ONE_CLASS_SVM = "one_class_svm"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"
    AUTOENCODER = "autoencoder"
    ENSEMBLE = "ensemble"


@dataclass(frozen=True)
class AlgorithmConfig:
    """
    Immutable configuration for anomaly detection algorithms.
    
    This value object encapsulates all configuration parameters
    needed to configure and run an anomaly detection algorithm.
    """
    
    algorithm_type: AlgorithmType
    parameters: Dict[str, Any]
    contamination: float = 0.1
    random_state: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate the configuration after initialization."""
        if not 0.0 < self.contamination < 0.5:
            raise ValueError("Contamination must be between 0.0 and 0.5")
            
        if self.random_state is not None and self.random_state < 0:
            raise ValueError("Random state must be non-negative")
            
        # Validate algorithm-specific parameters
        self._validate_algorithm_parameters()
    
    def _validate_algorithm_parameters(self) -> None:
        """Validate algorithm-specific parameters."""
        if self.algorithm_type == AlgorithmType.ISOLATION_FOREST:
            self._validate_isolation_forest_params()
        elif self.algorithm_type == AlgorithmType.LOCAL_OUTLIER_FACTOR:
            self._validate_lof_params()
        elif self.algorithm_type == AlgorithmType.ONE_CLASS_SVM:
            self._validate_svm_params()
        elif self.algorithm_type == AlgorithmType.ELLIPTIC_ENVELOPE:
            self._validate_elliptic_envelope_params()
        elif self.algorithm_type == AlgorithmType.AUTOENCODER:
            self._validate_autoencoder_params()
        elif self.algorithm_type == AlgorithmType.ENSEMBLE:
            self._validate_ensemble_params()
    
    def _validate_isolation_forest_params(self) -> None:
        """Validate Isolation Forest parameters."""
        valid_params = {
            'n_estimators', 'max_samples', 'max_features', 
            'bootstrap', 'n_jobs', 'warm_start'
        }
        
        for param in self.parameters:
            if param not in valid_params:
                raise ValueError(f"Invalid parameter for Isolation Forest: {param}")
                
        n_estimators = self.parameters.get('n_estimators', 100)
        if not isinstance(n_estimators, int) or n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer")
    
    def _validate_lof_params(self) -> None:
        """Validate Local Outlier Factor parameters."""
        valid_params = {
            'n_neighbors', 'algorithm', 'leaf_size', 
            'metric', 'p', 'metric_params', 'n_jobs'
        }
        
        for param in self.parameters:
            if param not in valid_params:
                raise ValueError(f"Invalid parameter for LOF: {param}")
                
        n_neighbors = self.parameters.get('n_neighbors', 20)
        if not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer")
    
    def _validate_svm_params(self) -> None:
        """Validate One-Class SVM parameters."""
        valid_params = {
            'kernel', 'degree', 'gamma', 'coef0', 
            'tol', 'nu', 'shrinking', 'cache_size', 'max_iter'
        }
        
        for param in self.parameters:
            if param not in valid_params:
                raise ValueError(f"Invalid parameter for One-Class SVM: {param}")
                
        nu = self.parameters.get('nu', 0.5)
        if not isinstance(nu, (int, float)) or not 0.0 < nu <= 1.0:
            raise ValueError("nu must be between 0.0 and 1.0")
    
    def _validate_elliptic_envelope_params(self) -> None:
        """Validate Elliptic Envelope parameters."""
        valid_params = {
            'store_precision', 'assume_centered', 'support_fraction',
            'random_state'
        }
        
        for param in self.parameters:
            if param not in valid_params:
                raise ValueError(f"Invalid parameter for Elliptic Envelope: {param}")
    
    def _validate_autoencoder_params(self) -> None:
        """Validate Autoencoder parameters."""
        valid_params = {
            'hidden_neurons', 'hidden_activation', 'output_activation',
            'loss', 'optimizer', 'epochs', 'batch_size', 'dropout_rate',
            'l2_regularizer', 'validation_size', 'preprocessing'
        }
        
        for param in self.parameters:
            if param not in valid_params:
                raise ValueError(f"Invalid parameter for Autoencoder: {param}")
                
        epochs = self.parameters.get('epochs', 100)
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
    
    def _validate_ensemble_params(self) -> None:
        """Validate Ensemble parameters."""
        valid_params = {
            'base_estimators', 'method', 'standardization'
        }
        
        for param in self.parameters:
            if param not in valid_params:
                raise ValueError(f"Invalid parameter for Ensemble: {param}")
                
        base_estimators = self.parameters.get('base_estimators', [])
        if not isinstance(base_estimators, list) or len(base_estimators) < 2:
            raise ValueError("base_estimators must be a list with at least 2 estimators")
    
    def is_valid(self) -> bool:
        """
        Check if the configuration is valid.
        
        Returns:
            bool: True if the configuration is valid.
        """
        try:
            self._validate_algorithm_parameters()
            return True
        except ValueError:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the value object to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation.
        """
        return {
            "algorithm_type": self.algorithm_type.value,
            "parameters": self.parameters.copy(),
            "contamination": self.contamination,
            "random_state": self.random_state
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlgorithmConfig':
        """
        Create an AlgorithmConfig from a dictionary.
        
        Args:
            data: Dictionary containing configuration data.
            
        Returns:
            AlgorithmConfig: New instance from the dictionary.
        """
        algorithm_type = AlgorithmType(data["algorithm_type"])
        return cls(
            algorithm_type=algorithm_type,
            parameters=data.get("parameters", {}),
            contamination=data.get("contamination", 0.1),
            random_state=data.get("random_state")
        )
    
    def with_contamination(self, contamination: float) -> 'AlgorithmConfig':
        """
        Create a new configuration with different contamination rate.
        
        Args:
            contamination: New contamination rate.
            
        Returns:
            AlgorithmConfig: New instance with updated contamination.
        """
        return AlgorithmConfig(
            algorithm_type=self.algorithm_type,
            parameters=self.parameters,
            contamination=contamination,
            random_state=self.random_state
        )
    
    def with_parameters(self, **new_parameters: Any) -> 'AlgorithmConfig':
        """
        Create a new configuration with updated parameters.
        
        Args:
            **new_parameters: New parameter values.
            
        Returns:
            AlgorithmConfig: New instance with updated parameters.
        """
        updated_params = self.parameters.copy()
        updated_params.update(new_parameters)
        
        return AlgorithmConfig(
            algorithm_type=self.algorithm_type,
            parameters=updated_params,
            contamination=self.contamination,
            random_state=self.random_state
        )