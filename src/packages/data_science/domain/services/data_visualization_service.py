"""Data Visualization domain service interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from ..entities.statistical_profile import StatisticalProfile
from ..value_objects.correlation_matrix import CorrelationMatrix


class IDataVisualizationService(ABC):
    """Domain service for data visualization operations."""
    
    @abstractmethod
    async def generate_statistical_plots(self, profile: StatisticalProfile,
                                        plot_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate statistical plots from profile data."""
        pass
    
    @abstractmethod
    async def create_correlation_heatmap(self, correlation_matrix: CorrelationMatrix,
                                       title: Optional[str] = None) -> Dict[str, Any]:
        """Create correlation matrix heatmap visualization."""
        pass
    
    @abstractmethod
    async def generate_interactive_dashboard(self, dataset: Any,
                                           dashboard_config: Dict[str, Any]) -> str:
        """Generate interactive dashboard for dataset exploration."""
        pass