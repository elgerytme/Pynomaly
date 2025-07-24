"""Repository interface for pipeline health."""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from ..entities.pipeline_health import PipelineHealth, PipelineMetric, PipelineAlert


class PipelineHealthRepository(ABC):
    """Abstract repository for pipeline health."""
    
    @abstractmethod
    async def save_pipeline_health(self, health: PipelineHealth) -> PipelineHealth:
        """Save pipeline health status."""
        pass
    
    @abstractmethod
    async def get_pipeline_health(self, pipeline_id: UUID) -> Optional[PipelineHealth]:
        """Get pipeline health by ID."""
        pass
    
    @abstractmethod
    async def get_all_pipeline_health(self) -> List[PipelineHealth]:
        """Get all pipeline health statuses."""
        pass
    
    @abstractmethod
    async def add_metric(self, pipeline_id: UUID, metric: PipelineMetric) -> None:
        """Add a metric to a pipeline's health."""
        pass
    
    @abstractmethod
    async def add_alert(self, pipeline_id: UUID, alert: PipelineAlert) -> None:
        """Add an alert to a pipeline's health."""
        pass
    
    @abstractmethod
    async def resolve_alert(self, pipeline_id: UUID, alert_id: UUID) -> bool:
        """Resolve an alert for a pipeline."""
        pass
    
    @abstractmethod
    async def get_metric_history(self, pipeline_id: UUID, metric_type: str = None, hours: int = 24) -> List[PipelineMetric]:
        """Get metric history for a pipeline."""
        pass
    
    @abstractmethod
    async def get_active_alerts(self, pipeline_id: UUID = None) -> List[PipelineAlert]:
        """Get active alerts for a pipeline, or all active alerts."""
        pass
    
    @abstractmethod
    async def delete_pipeline_health(self, pipeline_id: UUID) -> bool:
        """Delete pipeline health status by ID."""
        pass
