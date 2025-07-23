"""Shared interfaces for data profiling services."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd


class DataProfilingInterface(ABC):
    """Interface for data profiling services across packages."""
    
    @abstractmethod
    def profile_dataset(
        self, 
        data: pd.DataFrame, 
        dataset_id: str,
        use_advanced_orchestrator: bool = False
    ) -> Dict[str, Any]:
        """Profile a dataset and return comprehensive profiling results."""
        pass
    
    @abstractmethod
    def get_profiling_summary(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of profiling results."""
        pass


class PatternDiscoveryInterface(ABC):
    """Interface for pattern discovery services."""
    
    @abstractmethod
    def discover(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Discover patterns in data."""
        pass


class StatisticalProfilingInterface(ABC):
    """Interface for statistical profiling services."""
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical analysis on data."""
        pass