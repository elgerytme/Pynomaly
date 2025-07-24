"""PostgreSQL repository implementation for pipeline health."""

from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, insert, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ...domain.entities.pipeline_health import PipelineHealth, PipelineMetric, PipelineAlert, PipelineStatus, MetricType, AlertSeverity
from ...domain.repositories.pipeline_health_repository import PipelineHealthRepository
from ...infrastructure.persistence.models import PipelineHealthModel, PipelineMetricModel, PipelineAlertModel


class PostgresPipelineHealthRepository(PipelineHealthRepository):
    """PostgreSQL implementation of pipeline health repository."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with database session."""
        self._session = session

    def _metric_to_model(self, metric: PipelineMetric, pipeline_health_id: UUID) -> PipelineMetricModel:
        """Convert PipelineMetric entity to PipelineMetricModel."""
        return PipelineMetricModel(
            id=metric.id,
            pipeline_health_id=pipeline_health_id,
            metric_type=metric.metric_type.value,
            name=metric.name,
            value=metric.value,
            unit=metric.unit,
            timestamp=metric.timestamp,
            labels=metric.labels,
            source=metric.source
        )

    def _model_to_metric(self, model: PipelineMetricModel) -> PipelineMetric:
        """Convert PipelineMetricModel to PipelineMetric entity."""
        # Note: MetricThreshold is not stored in the model, so it will be None
        return PipelineMetric(
            id=model.id,
            pipeline_id=model.pipeline_health_id, # This is actually pipeline_health_id, not pipeline_id
            metric_type=MetricType(model.metric_type),
            name=model.name,
            value=model.value,
            unit=model.unit,
            timestamp=model.timestamp,
            labels=model.labels,
            source=model.source,
            threshold=None # Threshold is not persisted in the model
        )

    def _alert_to_model(self, alert: PipelineAlert, pipeline_health_id: UUID) -> PipelineAlertModel:
        """Convert PipelineAlert entity to PipelineAlertModel."""
        return PipelineAlertModel(
            id=alert.id,
            pipeline_health_id=pipeline_health_id,
            metric_id=alert.metric_id,
            severity=alert.severity.value,
            title=alert.title,
            description=alert.description,
            created_at=alert.created_at,
            resolved_at=alert.resolved_at,
            triggered_by=alert.triggered_by,
            current_value=alert.current_value,
            threshold_value=alert.threshold_value,
            acknowledged=alert.acknowledged,
            acknowledged_by=alert.acknowledged_by,
            acknowledged_at=alert.acknowledged_at
        )

    def _model_to_alert(self, model: PipelineAlertModel) -> PipelineAlert:
        """Convert PipelineAlertModel to PipelineAlert entity."""
        return PipelineAlert(
            id=model.id,
            pipeline_id=model.pipeline_health_id, # This is actually pipeline_health_id, not pipeline_id
            metric_id=model.metric_id,
            severity=AlertSeverity(model.severity),
            title=model.title,
            description=model.description,
            created_at=model.created_at,
            resolved_at=model.resolved_at,
            triggered_by=model.triggered_by,
            current_value=model.current_value,
            threshold_value=model.threshold_value,
            acknowledged=model.acknowledged,
            acknowledged_by=model.acknowledged_by,
            acknowledged_at=model.acknowledged_at
        )

    def _to_model(self, entity: PipelineHealth) -> PipelineHealthModel:
        """Convert PipelineHealth entity to PipelineHealthModel."""
        return PipelineHealthModel(
            id=entity.id,
            pipeline_id=entity.pipeline_id,
            pipeline_name=entity.pipeline_name,
            overall_health_score=entity.get_health_score(),
            execution_success_rate=1.0 - entity.error_rate, # Assuming error_rate is 0-1
            data_quality_score=entity.data_quality_score if entity.data_quality_score is not None else 0.0,
            performance_score=entity.performance_score,
            availability_percentage=entity.availability_percentage,
            status=entity.status.value,
            last_run_status=None, # Not directly mapped from entity
            last_successful_run=None, # Not directly mapped from entity
            last_failed_run=None, # Not directly mapped from entity
            last_execution=entity.last_execution,
            execution_duration=entity.execution_duration.total_seconds() if entity.execution_duration else None,
            total_runs=entity.successful_executions + entity.failed_executions,
            successful_runs=entity.successful_executions,
            failed_runs=entity.failed_executions,
            avg_execution_time=entity.execution_duration.total_seconds() if entity.execution_duration else None,
            metadata=entity.health_history, # Store health history in metadata for now
            created_at=entity.created_at, # Using entity's created_at
            updated_at=entity.updated_at
        )

    def _to_entity(self, model: PipelineHealthModel) -> PipelineHealth:
        """Convert PipelineHealthModel to PipelineHealth entity."""
        pipeline_health = PipelineHealth(
            id=model.id,
            pipeline_id=model.pipeline_id,
            pipeline_name=model.pipeline_name,
            status=PipelineStatus(model.status),
            last_updated=model.updated_at,
            created_at=model.created_at
        )
        pipeline_health.availability_percentage = model.availability_percentage
        pipeline_health.performance_score = model.performance_score
        pipeline_health.error_rate = 1.0 - model.execution_success_rate
        pipeline_health.data_quality_score = model.data_quality_score
        pipeline_health.last_execution = model.last_execution
        pipeline_health.execution_duration = timedelta(seconds=model.execution_duration) if model.execution_duration else None
        pipeline_health.successful_executions = model.successful_runs
        pipeline_health.failed_executions = model.failed_runs
        
        # Metrics and alerts will be loaded separately
        pipeline_health.current_metrics = []
        pipeline_health.active_alerts = []
        pipeline_health.health_history = model.metadata if model.metadata else []

        return pipeline_health

    async def save_pipeline_health(self, health: PipelineHealth) -> PipelineHealth:
        """Save pipeline health status."""
        model = self._to_model(health)
        # Check if pipeline already exists
        existing_model = await self._session.execute(
            select(PipelineHealthModel).filter(PipelineHealthModel.pipeline_id == health.pipeline_id)
        )
        if existing_model.scalar_one_or_none():
            await self._session.execute(
                update(PipelineHealthModel)
                .filter(PipelineHealthModel.pipeline_id == health.pipeline_id)
                .values(**model.dict(exclude_unset=True, exclude={"id", "created_at"}))
            )
        else:
            self._session.add(model)
        await self._session.flush()
        await self._session.commit()
        return health

    async def get_pipeline_health(self, pipeline_id: UUID) -> Optional[PipelineHealth]:
        """Get pipeline health by ID."""
        result = await self._session.execute(
            select(PipelineHealthModel)
            .filter(PipelineHealthModel.pipeline_id == pipeline_id)
            .options(selectinload(PipelineHealthModel.metrics))
            .options(selectinload(PipelineHealthModel.alerts))
        )
        model = result.scalar_one_or_none()
        if model:
            health = self._to_entity(model)
            # Manually load metrics and alerts since they are not part of the base _to_entity conversion
            health.current_metrics = [self._model_to_metric(m) for m in model.metrics]
            health.active_alerts = [self._model_to_alert(a) for a in model.alerts]
            return health
        return None

    async def get_all_pipeline_health(self) -> List[PipelineHealth]:
        """Get all pipeline health statuses."""
        result = await self._session.execute(
            select(PipelineHealthModel)
        )
        return [self._to_entity(model) for model in result.scalars().all()]

    async def add_metric(self, pipeline_id: UUID, metric: PipelineMetric) -> None:
        """Add a metric to a pipeline's health."""
        pipeline_health_model = await self._session.execute(
            select(PipelineHealthModel).filter(PipelineHealthModel.pipeline_id == pipeline_id)
        )
        pipeline_health_model = pipeline_health_model.scalar_one_or_none()

        if not pipeline_health_model:
            raise ValueError(f"Pipeline {pipeline_id} not found")

        metric_model = self._metric_to_model(metric, pipeline_health_model.id)
        self._session.add(metric_model)
        await self._session.flush()
        await self._session.commit()

    async def add_alert(self, pipeline_id: UUID, alert: PipelineAlert) -> None:
        """Add an alert to a pipeline's health."""
        pipeline_health_model = await self._session.execute(
            select(PipelineHealthModel).filter(PipelineHealthModel.pipeline_id == pipeline_id)
        )
        pipeline_health_model = pipeline_health_model.scalar_one_or_none()

        if not pipeline_health_model:
            raise ValueError(f"Pipeline {pipeline_id} not found")

        alert_model = self._alert_to_model(alert, pipeline_health_model.id)
        self._session.add(alert_model)
        await self._session.flush()
        await self._session.commit()

    async def resolve_alert(self, pipeline_id: UUID, alert_id: UUID) -> bool:
        """Resolve an alert for a pipeline."""
        result = await self._session.execute(
            update(PipelineAlertModel)
            .filter(PipelineAlertModel.id == alert_id)
            .values(resolved_at=datetime.utcnow(), resolved=True, status="resolved")
        )
        await self._session.commit()
        return result.rowcount > 0

    async def get_metric_history(self, pipeline_id: UUID, metric_type: MetricType = None, hours: int = 24) -> List[PipelineMetric]:
        """Get metric history for a pipeline."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        query = select(PipelineMetricModel).filter(
            PipelineMetricModel.pipeline_health_id == pipeline_id,
            PipelineMetricModel.timestamp >= cutoff_time
        )
        if metric_type:
            query = query.filter(PipelineMetricModel.metric_type == metric_type.value)
        
        result = await self._session.execute(query.order_by(PipelineMetricModel.timestamp))
        return [self._model_to_metric(model) for model in result.scalars().all()]

    async def get_active_alerts(self, pipeline_id: UUID = None) -> List[PipelineAlert]:
        """Get active alerts for a pipeline, or all active alerts."""
        query = select(PipelineAlertModel).filter(PipelineAlertModel.resolved_at == None) # noqa
        if pipeline_id:
            query = query.filter(PipelineAlertModel.pipeline_health_id == pipeline_id)
        
        result = await self._session.execute(query)
        return [self._model_to_alert(model) for model in result.scalars().all()]

    async def delete_pipeline_health(self, pipeline_id: UUID) -> bool:
        """Delete pipeline health status by ID."""
        result = await self._session.execute(
            delete(PipelineHealthModel).filter(PipelineHealthModel.pipeline_id == str(pipeline_id))
        )
        await self._session.commit()
        return result.rowcount > 0
