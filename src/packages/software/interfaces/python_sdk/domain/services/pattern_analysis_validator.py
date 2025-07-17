"""
Processing Validator Domain Service

Provides validation logic for processing-related domain objects.
"""

from typing import List, Tuple
import numpy as np

from ..entities.pattern_analysis_request import PatternAnalysisRequest
from ..value_objects.algorithm_config import AlgorithmConfig, AlgorithmType
from ..exceptions.validation_exceptions import ValidationError


class PatternAnalysisValidator:
    """
    Domain service for validating processing requests and configurations.
    
    This service encapsulates domain logic for ensuring that processing
    requests and configurations are valid and meet business requirements.
    """
    
    @staticmethod
    def validate_pattern_analysis_request(request: PatternAnalysisRequest) -> Tuple[bool, List[str]]:
        """
        Validate a complete processing request.
        
        Args:
            request: The processing request to validate.
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate basic request structure
        if not request.validate():
            errors.append("Processing request basic validation failed")
        
        # Validate data
        data_errors = PatternAnalysisValidator._validate_data(request.data)
        errors.extend(data_errors)
        
        # Validate algorithm configuration
        config_errors = PatternAnalysisValidator._validate_algorithm_config(
            request.algorithm_config, 
            len(request.data)
        )
        errors.extend(config_errors)
        
        # Validate data compatibility with algorithm
        compatibility_errors = PatternAnalysisValidator._validate_data_algorithm_compatibility(
            request.data, 
            request.algorithm_config
        )
        errors.extend(compatibility_errors)
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _validate_data(data: List[float]) -> List[str]:
        """
        Validate input data for pattern analysis processing.
        
        Args:
            data: The input data to validate.
            
        Returns:
            List[str]: List of validation errors.
        """
        errors = []
        
        if not data:
            errors.append("Data cannot be empty")
            return errors
        
        if len(data) < 10:
            errors.append("Data must contain at least 10 data points for reliable processing")
        
        if len(data) > 1000000:
            errors.append("Data size exceeds maximum limit of 1,000,000 points")
        
        # Check for invalid values
        try:
            data_array = np.array(data)
            
            if np.any(np.isnan(data_array)):
                errors.append("Data contains NaN values")
            
            if np.any(np.isinf(data_array)):
                errors.append("Data contains infinite values")
                
            # Check for constant data
            if np.var(data_array) == 0:
                errors.append("Data has zero variance (all values are identical)")
                
        except Exception as e:
            errors.append(f"Error processing data: {str(e)}")
        
        return errors
    
    @staticmethod
    def _validate_algorithm_config(config: AlgorithmConfig, data_size: int) -> List[str]:
        """
        Validate algorithm configuration.
        
        Args:
            config: The algorithm configuration to validate.
            data_size: Size of the data to be processed.
            
        Returns:
            List[str]: List of validation errors.
        """
        errors = []
        
        if not config.is_valid():
            errors.append("Algorithm configuration is invalid")
        
        # Algorithm-specific validations based on data size
        if config.algorithm_type == AlgorithmType.LOCAL_OUTLIER_FACTOR:
            n_neighbors = config.parameters.get('n_neighbors', 20)
            if n_neighbors >= data_size:
                errors.append(f"LOF n_neighbors ({n_neighbors}) must be less than data size ({data_size})")
        
        if config.algorithm_type == AlgorithmType.ISOLATION_FOREST:
            max_samples = config.parameters.get('max_samples', 'auto')
            if isinstance(max_samples, int) and max_samples > data_size:
                errors.append(f"Isolation Forest max_samples ({max_samples}) cannot exceed data size ({data_size})")
        
        # Validate contamination rate against data size
        min_outliers = int(config.contamination * data_size)
        if min_outliers < 1:
            errors.append(f"Contamination rate ({config.contamination}) would result in less than 1 outlier for data size {data_size}")
        
        return errors
    
    @staticmethod
    def _validate_data_algorithm_compatibility(data: List[float], config: AlgorithmConfig) -> List[str]:
        """
        Validate compatibility between data and algorithm.
        
        Args:
            data: The input data.
            config: The algorithm configuration.
            
        Returns:
            List[str]: List of compatibility errors.
        """
        errors = []
        data_size = len(data)
        
        # Memory and performance considerations
        if config.algorithm_type == AlgorithmType.LOCAL_OUTLIER_FACTOR and data_size > 50000:
            errors.append("LOF algorithm may be slow with large datasets (>50,000 points)")
        
        if config.algorithm_type == AlgorithmType.ONE_CLASS_SVM and data_size > 10000:
            errors.append("One-Class SVM may be slow with large datasets (>10,000 points)")
        
        # Autoencoder specific validations
        if config.algorithm_type == AlgorithmType.AUTOENCODER:
            if data_size < 100:
                errors.append("Autoencoder requires at least 100 data points for training")
            
            hidden_neurons = config.parameters.get('hidden_neurons', [])
            if hidden_neurons and max(hidden_neurons) > data_size:
                errors.append("Autoencoder hidden layer size cannot exceed data size")
        
        # Ensemble validations
        if config.algorithm_type == AlgorithmType.ENSEMBLE:
            base_estimators = config.parameters.get('base_estimators', [])
            for estimator_config in base_estimators:
                if isinstance(estimator_config, dict):
                    # Recursively validate base estimators
                    try:
                        base_config = AlgorithmConfig.from_dict(estimator_config)
                        base_errors = PatternAnalysisValidator._validate_data_algorithm_compatibility(data, base_config)
                        errors.extend([f"Base estimator: {error}" for error in base_errors])
                    except Exception as e:
                        errors.append(f"Invalid base estimator configuration: {str(e)}")
        
        return errors
    
    @staticmethod
    def validate_and_raise(request: PatternAnalysisRequest) -> None:
        """
        Validate a processing request and raise an exception if invalid.
        
        Args:
            request: The processing request to validate.
            
        Raises:
            ValidationError: If the request is invalid.
        """
        is_valid, errors = PatternAnalysisValidator.validate_pattern_analysis_request(request)
        
        if not is_valid:
            raise ValidationError(f"Processing request validation failed: {'; '.join(errors)}")
    
    @staticmethod
    def can_handle_data_size(algorithm_type: AlgorithmType, data_size: int) -> bool:
        """
        Check if an algorithm can handle a given data size efficiently.
        
        Args:
            algorithm_type: The algorithm type.
            data_size: The size of the data.
            
        Returns:
            bool: True if the algorithm can handle the data size efficiently.
        """
        efficiency_limits = {
            AlgorithmType.ISOLATION_FOREST: 100000,
            AlgorithmType.LOCAL_OUTLIER_FACTOR: 50000,
            AlgorithmType.ONE_CLASS_SVM: 10000,
            AlgorithmType.ELLIPTIC_ENVELOPE: 50000,
            AlgorithmType.AUTOENCODER: 1000000,
            AlgorithmType.ENSEMBLE: 50000
        }
        
        return data_size <= efficiency_limits.get(algorithm_type, 100000)