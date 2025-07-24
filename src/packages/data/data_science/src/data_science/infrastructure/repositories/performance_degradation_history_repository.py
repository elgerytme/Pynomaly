"""Performance degradation history repository for tracking model performance over time."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID

# TODO: Implement within data monorepo science domain - from packages.data_science.domain.entities.model_performance_degradation import (
    ModelPerformanceDegradation,
    DegradationStatus,
)
# TODO: Implement within data monorepo science domain - from packages.data_science.domain.value_objects.model_performance_metrics import (
    ModelPerformanceMetrics,
)
# TODO: Implement within data monorepo science domain - from packages.data_science.domain.value_objects.performance_degradation_metrics import (
    DegradationMetricType,
    DegradationSeverity,
)


class PerformanceDegradationHistoryRepository(ABC):
    """Abstract repository for performance degradation history tracking."""
    
    @abstractmethod
    async def save_degradation_event(
        self,
        model_id: str,
        degradation_data: Dict[str, Any],
        metrics: ModelPerformanceMetrics,
        timestamp: datetime = None,
    ) -> None:
        """Save a performance degradation event."""
        ...
    
    @abstractmethod
    async def get_degradation_history(
        self,
        model_id: str,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get degradation history for a model."""
        ...
    
    @abstractmethod
    async def get_degradation_summary(
        self,
        model_id: str,
        days_back: int = 30,
    ) -> Dict[str, Any]:
        """Get degradation summary statistics."""
        ...
    
    @abstractmethod
    async def get_degradation_trends(
        self,
        model_id: str,
        metric_type: DegradationMetricType,
        days_back: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get degradation trends for a specific metric."""
        ...
    
    @abstractmethod
    async def get_recovery_history(
        self,
        model_id: str,
        days_back: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get recovery action history."""
        ...
    
    @abstractmethod
    async def get_alert_history(
        self,
        model_id: str,
        days_back: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get alert history for a model."""
        ...
    
    @abstractmethod
    async def cleanup_old_history(
        self,
        retention_days: int = 90,
    ) -> int:
        """Clean up old history records."""
        ...
    
    @abstractmethod
    async def get_models_with_degradation_history(
        self,
        status_filter: Optional[DegradationStatus] = None,
    ) -> List[str]:
        """Get list of models with degradation history."""
        ...
    
    @abstractmethod
    async def get_global_degradation_statistics(
        self,
        days_back: int = 30,
    ) -> Dict[str, Any]:
        """Get global degradation statistics across all models."""
        ...