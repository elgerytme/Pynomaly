"""
Algorithm Adapter Interface

Defines the interface for algorithm execution adapters.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np

from ...domain.value_objects.algorithm_config import AlgorithmConfig


class AlgorithmResult:
    """
    Result of an anomaly detection algorithm execution.
    
    Encapsulates the output of algorithm execution including
    anomaly predictions, scores, and metadata.
    """
    
    def __init__(
        self,
        anomalies: List[int],
        scores: List[float],
        algorithm_type: str,
        execution_time_ms: int,
        metadata: Dict[str, Any] = None
    ):
        self.anomalies = anomalies
        self.scores = scores
        self.algorithm_type = algorithm_type
        self.execution_time_ms = execution_time_ms
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "anomalies": self.anomalies,
            "scores": self.scores,
            "algorithm_type": self.algorithm_type,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata
        }


class AlgorithmAdapter(ABC):
    """
    Abstract adapter for anomaly detection algorithm execution.
    
    This interface defines the contract that infrastructure
    implementations must follow to execute anomaly detection algorithms.
    """
    
    @abstractmethod
    async def detect_anomalies(
        self, 
        data: List[float], 
        algorithm_config: AlgorithmConfig
    ) -> AlgorithmResult:
        """
        Execute anomaly detection on the provided data.
        
        Args:
            data: Input data for anomaly detection.
            algorithm_config: Configuration for the algorithm.
            
        Returns:
            AlgorithmResult: Results of the anomaly detection.
            
        Raises:
            AlgorithmExecutionError: If algorithm execution fails.
        """
        pass
    
    @abstractmethod
    async def validate_algorithm_support(self, algorithm_config: AlgorithmConfig) -> bool:
        """
        Check if the adapter supports the specified algorithm configuration.
        
        Args:
            algorithm_config: Configuration to validate.
            
        Returns:
            bool: True if the algorithm is supported.
        """
        pass
    
    @abstractmethod
    async def get_supported_algorithms(self) -> List[str]:
        """
        Get list of supported algorithm types.
        
        Returns:
            List[str]: List of supported algorithm type names.
        """
        pass
    
    @abstractmethod
    async def estimate_execution_time(
        self, 
        data_size: int, 
        algorithm_config: AlgorithmConfig
    ) -> int:
        """
        Estimate execution time for the given data size and algorithm.
        
        Args:
            data_size: Size of the input data.
            algorithm_config: Algorithm configuration.
            
        Returns:
            int: Estimated execution time in milliseconds.
        """
        pass


class AlgorithmExecutionError(Exception):
    """Exception raised when algorithm execution fails."""
    
    def __init__(self, algorithm_type: str, message: str, original_error: Exception = None):
        super().__init__(f"Algorithm {algorithm_type} execution failed: {message}")
        self.algorithm_type = algorithm_type
        self.original_error = original_error