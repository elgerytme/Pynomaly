"""Repository interface for quality prediction."""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from ..entities.quality_prediction import QualityPrediction, QualityForecast, QualityTrend, QualityAlert, QualityMetricPoint


class QualityPredictionRepository(ABC):
    """Abstract repository for quality prediction."""
    
    @abstractmethod
    async def save_prediction(self, prediction: QualityPrediction) -> QualityPrediction:
        """Save a quality prediction."""
        pass
    
    @abstractmethod
    async def get_prediction_by_id(self, prediction_id: UUID) -> Optional[QualityPrediction]:
        """Get a quality prediction by ID."""
        pass
    
    @abstractmethod
    async def save_forecast(self, forecast: QualityForecast) -> QualityForecast:
        """Save a quality forecast."""
        pass
    
    @abstractmethod
    async def get_forecast_by_id(self, forecast_id: UUID) -> Optional[QualityForecast]:
        """Get a quality forecast by ID."""
        pass
    
    @abstractmethod
    async def save_trend(self, trend: QualityTrend) -> QualityTrend:
        """Save a quality trend."""
        pass
    
    @abstractmethod
    async def get_trend_by_id(self, trend_id: UUID) -> Optional[QualityTrend]:
        """Get a quality trend by ID."""
        pass
    
    @abstractmethod
    async def add_metric_point(self, metric_point: QualityMetricPoint) -> QualityMetricPoint:
        """Add a quality metric data point."""
        pass
    
    @abstractmethod
    async def get_metric_history(self, asset_id: UUID, metric_type: str, days: int) -> List[QualityMetricPoint]:
        """Get quality metric history for an asset."""
        pass
    
    @abstractmethod
    async def save_alert(self, alert: QualityAlert) -> QualityAlert:
        """Save a quality alert."""
        pass
    
    @abstractmethod
    async def get_alert_by_id(self, alert_id: UUID) -> Optional[QualityAlert]:
        """Get a quality alert by ID."""
        pass
    
    @abstractmethod
    async def get_active_alerts(self, asset_id: UUID = None) -> List[QualityAlert]:
        """Get active quality alerts for an asset, or all active alerts."""
        pass

    @abstractmethod
    async def get_all_predictions(self) -> List[QualityPrediction]:
        """Get all quality predictions."""
        pass
    
    @abstractmethod
    async def update_prediction(self, prediction: QualityPrediction) -> QualityPrediction:
        """Update an existing quality prediction."""
        pass
    
    @abstractmethod
    async def update_alert(self, alert: QualityAlert) -> QualityAlert:
        """Update an existing quality alert."""
        pass
    
    @abstractmethod
    async def delete_prediction(self, prediction_id: UUID) -> bool:
        """Delete a quality prediction by ID."""
        pass
    
    @abstractmethod
    async def delete_alert(self, alert_id: UUID) -> bool:
        """Delete a quality alert by ID."""
        pass
