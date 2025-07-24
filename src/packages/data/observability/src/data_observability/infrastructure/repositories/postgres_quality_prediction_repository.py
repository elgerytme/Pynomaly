"""PostgreSQL repository implementation for quality prediction."""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy import select, insert, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from ...domain.entities.quality_prediction import QualityPrediction, QualityForecast, QualityTrend, QualityAlert, QualityMetricPoint, PredictionType, PredictionConfidence, TrendDirection, SeasonalPattern
from ...domain.repositories.quality_prediction_repository import QualityPredictionRepository
from ...infrastructure.persistence.models import QualityPredictionModel, QualityAlertModel


class PostgresQualityPredictionRepository(QualityPredictionRepository):
    """PostgreSQL implementation of quality prediction repository."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with database session."""
        self._session = session

    def _prediction_to_model(self, entity: QualityPrediction) -> QualityPredictionModel:
        """Convert QualityPrediction entity to QualityPredictionModel."""
        return QualityPredictionModel(
            id=entity.id,
            asset_id=entity.asset_id,
            predicted_value=entity.predicted_value,
            current_value=entity.current_value,
            threshold_value=entity.threshold_value,
            prediction_type=entity.prediction_type.value,
            prediction_time=entity.prediction_time,
            target_time=entity.target_time,
            time_horizon_seconds=entity.time_horizon.total_seconds(),
            confidence=entity.confidence.value,
            confidence_score=entity.confidence_score,
            prediction_interval=entity.prediction_interval,
            model_name=entity.model_name,
            model_version=entity.model_version,
            features_used=entity.features_used,
            impact_score=entity.impact_score,
            affected_systems=entity.affected_systems,
            recommended_actions=entity.recommended_actions,
            is_validated=entity.is_validated,
            actual_value=entity.actual_value,
            validation_time=entity.validation_time,
            prediction_error=entity.prediction_error
        )

    def _model_to_prediction(self, model: QualityPredictionModel) -> QualityPrediction:
        """Convert QualityPredictionModel to QualityPrediction entity."""
        return QualityPrediction(
            id=model.id,
            asset_id=model.asset_id,
            prediction_type=PredictionType(model.prediction_type),
            predicted_value=model.predicted_value,
            current_value=model.current_value,
            threshold_value=model.threshold_value,
            prediction_time=model.prediction_time,
            target_time=model.target_time,
            time_horizon=timedelta(seconds=model.time_horizon_seconds),
            confidence=PredictionConfidence(model.confidence),
            confidence_score=model.confidence_score,
            prediction_interval=model.prediction_interval,
            model_name=model.model_name,
            model_version=model.model_version,
            features_used=model.features_used if model.features_used else [],
            impact_score=model.impact_score,
            affected_systems=model.affected_systems if model.affected_systems else [],
            recommended_actions=model.recommended_actions if model.recommended_actions else [],
            is_validated=model.is_validated,
            actual_value=model.actual_value,
            validation_time=model.validation_time,
            prediction_error=model.prediction_error
        )

    def _alert_to_model(self, entity: QualityAlert) -> QualityAlertModel:
        """Convert QualityAlert entity to QualityAlertModel."""
        return QualityAlertModel(
            id=entity.id,
            asset_id=entity.asset_id,
            prediction_id=entity.prediction_id,
            alert_type=entity.alert_type,
            severity=entity.severity,
            title=entity.title,
            description=entity.description,
            created_at=entity.created_at,
            expected_time=entity.expected_time,
            recommended_actions=entity.recommended_actions,
            assigned_to=entity.assigned_to,
            status=entity.status,
            acknowledged_at=entity.acknowledged_at,
            resolved_at=entity.resolved_at
        )

    def _model_to_alert(self, model: QualityAlertModel) -> QualityAlert:
        """Convert QualityAlertModel to QualityAlert entity."""
        return QualityAlert(
            id=model.id,
            prediction_id=model.prediction_id,
            asset_id=model.asset_id,
            alert_type=model.alert_type,
            severity=model.severity,
            title=model.title,
            description=model.description,
            created_at=model.created_at,
            expected_time=model.expected_time,
            recommended_actions=model.recommended_actions if model.recommended_actions else [],
            assigned_to=model.assigned_to,
            status=model.status,
            acknowledged_at=model.acknowledged_at,
            resolved_at=model.resolved_at
        )

    async def save_prediction(self, prediction: QualityPrediction) -> QualityPrediction:
        """Save a quality prediction."""
        model = self._prediction_to_model(prediction)
        self._session.add(model)
        await self._session.flush()
        await self._session.commit()
        return prediction

    async def get_prediction_by_id(self, prediction_id: UUID) -> Optional[QualityPrediction]:
        """Get a quality prediction by ID."""
        result = await self._session.execute(
            select(QualityPredictionModel).filter(QualityPredictionModel.id == prediction_id)
        )
        model = result.scalar_one_or_none()
        return self._model_to_prediction(model) if model else None

    async def save_forecast(self, forecast: QualityForecast) -> QualityForecast:
        """Save a quality forecast."""
        # Forecasts are not directly mapped to a model, they are derived from predictions/metrics.
        # For now, we'll just return the forecast without persistence.
        # In a real system, you might store forecast parameters or a summary.
        return forecast

    async def get_forecast_by_id(self, forecast_id: UUID) -> Optional[QualityForecast]:
        """Get a quality forecast by ID."""
        # Forecasts are not directly mapped to a model.
        return None

    async def save_trend(self, trend: QualityTrend) -> QualityTrend:
        """Save a quality trend."""
        # Trends are not directly mapped to a model, they are derived from metrics.
        # For now, we'll just return the trend without persistence.
        return trend

    async def get_trend_by_id(self, trend_id: UUID) -> Optional[QualityTrend]:
        """Get a quality trend by ID."""
        # Trends are not directly mapped to a model.
        return None

    async def add_metric_point(self, metric_point: QualityMetricPoint) -> QualityMetricPoint:
        """Add a quality metric data point."""
        # Metric points are not directly mapped to a model.
        # They are usually part of a time-series database or a dedicated metrics store.
        # For now, we'll just return the metric point without persistence.
        return metric_point

    async def get_metric_history(self, asset_id: UUID, metric_type: str, days: int) -> List[QualityMetricPoint]:
        """Get quality metric history for an asset."""
        # Metric history is not directly mapped to a model.
        # This would typically query a time-series database.
        return []

    async def save_alert(self, alert: QualityAlert) -> QualityAlert:
        """Save a quality alert."""
        model = self._alert_to_model(alert)
        self._session.add(model)
        await self._session.flush()
        await self._session.commit()
        return alert

    async def get_alert_by_id(self, alert_id: UUID) -> Optional[QualityAlert]:
        """Get a quality alert by ID."""
        result = await self._session.execute(
            select(QualityAlertModel).filter(QualityAlertModel.id == alert_id)
        )
        model = result.scalar_one_or_none()
        return self._model_to_alert(model) if model else None

    async def get_active_alerts(self, asset_id: UUID = None) -> List[QualityAlert]:
        """Get active quality alerts for an asset, or all active alerts."""
        query = select(QualityAlertModel).filter(QualityAlertModel.status.in_(["open", "acknowledged"]))
        if asset_id:
            query = query.filter(QualityAlertModel.asset_id == asset_id)
        
        result = await self._session.execute(query)
        return [self._model_to_alert(model) for model in result.scalars().all()]

    async def update_prediction(self, prediction: QualityPrediction) -> QualityPrediction:
        """Update an existing quality prediction."""
        model_data = self._prediction_to_model(prediction).dict(exclude_unset=True)
        model_data.pop("id", None)
        model_data.pop("created_at", None)
        model_data.pop("prediction_date", None)

        result = await self._session.execute(
            update(QualityPredictionModel)
            .filter(QualityPredictionModel.id == prediction.id)
            .values(**model_data)
        )
        await self._session.flush()
        await self._session.commit()
        if result.rowcount > 0:
            return await self.get_prediction_by_id(prediction.id)
        else:
            raise ValueError(f"Prediction with ID {prediction.id} not found for update")

    async def update_alert(self, alert: QualityAlert) -> QualityAlert:
        """Update an existing quality alert."""
        model_data = self._alert_to_model(alert).dict(exclude_unset=True)
        model_data.pop("id", None)
        model_data.pop("created_at", None)
        model_data.pop("updated_at", None)

        result = await self._session.execute(
            update(QualityAlertModel)
            .filter(QualityAlertModel.id == alert.id)
            .values(**model_data)
        )
        await self._session.flush()
        await self._session.commit()
        if result.rowcount > 0:
            return await self.get_alert_by_id(alert.id)
        else:
            raise ValueError(f"Alert with ID {alert.id} not found for update")

    async def delete_prediction(self, prediction_id: UUID) -> bool:
        """Delete a quality prediction by ID."""
        result = await self._session.execute(
            delete(QualityPredictionModel).filter(QualityPredictionModel.id == prediction_id)
        )
        await self._session.flush()
        await self._session.commit()
        return result.rowcount > 0

    async def delete_alert(self, alert_id: UUID) -> bool:
        """Delete a quality alert by ID."""
        result = await self._session.execute(
            delete(QualityAlertModel).filter(QualityAlertModel.id == alert_id)
        )
        await self._session.flush()
        await self._session.commit()
        return result.rowcount > 0
