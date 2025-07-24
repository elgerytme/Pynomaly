from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from ...application.ports.data_quality_check_repository import DataQualityCheckRepository
from ...domain.entities.data_quality_check import DataQualityCheck, CheckStatus, CheckSeverity, CheckResult
from ..database.models import DataQualityCheckModel


class SQLAlchemyDataQualityCheckRepository(DataQualityCheckRepository):
    """SQLAlchemy implementation of DataQualityCheckRepository."""

    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, id: UUID) -> Optional[DataQualityCheck]:
        """Get a data quality check by its ID."""
        model = self.session.query(DataQualityCheckModel).filter_by(id=id).first()
        if model:
            return self._to_entity(model)
        return None

    def save(self, data_quality_check: DataQualityCheck) -> None:
        """Save a data quality check."""
        model = self.session.query(DataQualityCheckModel).filter_by(id=data_quality_check.id).first()
        if model:
            self._update_model_from_entity(model, data_quality_check)
        else:
            model = self._to_model(data_quality_check)
            self.session.add(model)
        self.session.commit()

    def _to_entity(self, model: DataQualityCheckModel) -> DataQualityCheck:
        """Convert a SQLAlchemy model to a DataQualityCheck entity."""
        last_result = None
        if model.last_result:
            last_result = CheckResult(
                id=model.last_result.get("id"),
                check_id=model.last_result.get("check_id"),
                dataset_name=model.last_result.get("dataset_name"),
                column_name=model.last_result.get("column_name"),
                passed=model.last_result.get("passed"),
                score=model.last_result.get("score"),
                total_records=model.last_result.get("total_records"),
                passed_records=model.last_result.get("passed_records"),
                failed_records=model.last_result.get("failed_records"),
                executed_at=model.last_result.get("executed_at"),
                execution_time_ms=model.last_result.get("execution_time_ms"),
                severity=CheckSeverity(model.last_result.get("severity")),
                message=model.last_result.get("message"),
                details=model.last_result.get("details"),
                failed_values=model.last_result.get("failed_values"),
                sample_failures=model.last_result.get("sample_failures"),
                metadata=model.last_result.get("metadata"),
                tags=model.last_result.get("tags"),
            )

        return DataQualityCheck(
            id=model.id,
            name=model.name,
            description=model.description,
            check_type=model.check_type,
            rule_id=model.rule_id,
            dataset_name=model.dataset_name,
            column_name=model.column_name,
            schema_name=model.schema_name,
            table_name=model.table_name,
            query=model.query,
            expression=model.expression,
            expected_value=model.expected_value,
            threshold=model.threshold,
            tolerance=model.tolerance,
            is_active=model.is_active,
            schedule_cron=model.schedule_cron,
            timeout_seconds=model.timeout_seconds,
            retry_attempts=model.retry_attempts,
            status=CheckStatus(model.status),
            last_executed_at=model.last_executed_at,
            next_execution_at=model.next_execution_at,
            execution_count=model.execution_count,
            last_result=last_result,
            consecutive_failures=model.consecutive_failures,
            success_rate=model.success_rate,
            created_at=model.created_at,
            created_by=model.created_by,
            updated_at=model.updated_at,
            updated_by=model.updated_by,
            config=model.config,
            environment_vars=model.environment_vars,
            tags=model.tags,
            depends_on=model.depends_on,
            blocks=model.blocks,
        )

    def _to_model(self, entity: DataQualityCheck) -> DataQualityCheckModel:
        """Convert a DataQualityCheck entity to a SQLAlchemy model."""
        return DataQualityCheckModel(
            id=entity.id,
            name=entity.name,
            description=entity.description,
            check_type=entity.check_type.value,
            rule_id=entity.rule_id,
            dataset_name=entity.dataset_name,
            column_name=entity.column_name,
            schema_name=entity.schema_name,
            table_name=entity.table_name,
            query=entity.query,
            expression=entity.expression,
            expected_value=entity.expected_value,
            threshold=entity.threshold,
            tolerance=entity.tolerance,
            is_active=entity.is_active,
            schedule_cron=entity.schedule_cron,
            timeout_seconds=entity.timeout_seconds,
            retry_attempts=entity.retry_attempts,
            status=entity.status.value,
            last_executed_at=entity.last_executed_at,
            next_execution_at=entity.next_execution_at,
            execution_count=entity.execution_count,
            last_result=entity.last_result.to_dict() if entity.last_result else None,
            consecutive_failures=entity.consecutive_failures,
            success_rate=entity.success_rate,
            created_at=entity.created_at,
            created_by=entity.created_by,
            updated_at=entity.updated_at,
            updated_by=entity.updated_by,
            config=entity.config,
            environment_vars=entity.environment_vars,
            tags=entity.tags,
            depends_on=entity.depends_on,
            blocks=entity.blocks,
        )

    def _update_model_from_entity(self, model: DataQualityCheckModel, entity: DataQualityCheck) -> None:
        """Update a SQLAlchemy model from a DataQualityCheck entity."""
        model.name = entity.name
        model.description = entity.description
        model.check_type = entity.check_type.value
        model.rule_id = entity.rule_id
        model.dataset_name = entity.dataset_name
        model.column_name = entity.column_name
        model.schema_name = entity.schema_name
        model.table_name = entity.table_name
        model.query = entity.query
        model.expression = entity.expression
        model.expected_value = entity.expected_value
        model.threshold = entity.threshold
        model.tolerance = entity.tolerance
        model.is_active = entity.is_active
        model.schedule_cron = entity.schedule_cron
        model.timeout_seconds = entity.timeout_seconds
        model.retry_attempts = entity.retry_attempts
        model.status = entity.status.value
        model.last_executed_at = entity.last_executed_at
        model.next_execution_at = entity.next_execution_at
        model.execution_count = entity.execution_count
        model.last_result = entity.last_result.to_dict() if entity.last_result else None
        model.consecutive_failures = entity.consecutive_failures
        model.success_rate = entity.success_rate
        model.created_at = entity.created_at
        model.created_by = entity.created_by
        model.updated_at = entity.updated_at
        model.updated_by = entity.updated_by
        model.config = entity.config
        model.environment_vars = entity.environment_vars
        model.tags = entity.tags
        model.depends_on = entity.depends_on
        model.blocks = entity.blocks