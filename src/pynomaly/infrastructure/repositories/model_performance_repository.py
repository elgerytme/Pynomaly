"""Model performance repository implementation."""

from typing import Optional, List, Dict, Any


class ModelPerformanceRepository:
    """Repository for model performance data."""
    
    def __init__(self):
        self._performance_data = {}
    
    def get_performance_data(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get performance data for a model."""
        return self._performance_data.get(model_id)
    
    def save_performance_data(self, model_id: str, data: Dict[str, Any]) -> None:
        """Save performance data for a model."""
        self._performance_data[model_id] = data
    
    def list_performance_data(self) -> List[Dict[str, Any]]:
        """List all performance data."""
        return list(self._performance_data.values())


class PerformanceBaselineRepository:
    """Repository for performance baselines."""
    
    def __init__(self):
        self._baselines = {}
    
    def get_baseline(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get baseline for a model."""
        return self._baselines.get(model_id)
    
    def save_baseline(self, model_id: str, baseline: Dict[str, Any]) -> None:
        """Save baseline for a model."""
        self._baselines[model_id] = baseline
    
    def list_baselines(self) -> List[Dict[str, Any]]:
        """List all baselines."""
        return list(self._baselines.values())
