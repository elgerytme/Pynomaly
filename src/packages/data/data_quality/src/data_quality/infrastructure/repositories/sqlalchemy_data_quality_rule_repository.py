from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from ...application.ports.data_quality_rule_repository import DataQualityRuleRepository
from ...domain.entities.data_quality_rule import DataQualityRule, RuleCondition, RuleOperator, RuleType, RuleSeverity
from ..database.models import DataQualityRuleModel


class SQLAlchemyDataQualityRuleRepository(DataQualityRuleRepository):
    """SQLAlchemy implementation of DataQualityRuleRepository."""

    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, id: UUID) -> Optional[DataQualityRule]:
        """Get a data quality rule by its ID."""
        model = self.session.query(DataQualityRuleModel).filter_by(id=id).first()
        if model:
            return self._to_entity(model)
        return None

    def save(self, data_quality_rule: DataQualityRule) -> None:
        """Save a data quality rule."""
        model = self.session.query(DataQualityRuleModel).filter_by(id=data_quality_rule.id).first()
        if model:
            self._update_model_from_entity(model, data_quality_rule)
        else:
            model = self._to_model(data_quality_rule)
            self.session.add(model)
        self.session.commit()

    def _to_entity(self, model: DataQualityRuleModel) -> DataQualityRule:
        """Convert a SQLAlchemy model to a DataQualityRule entity."""
        conditions = []
        if model.conditions:
            for cond_dict in model.conditions:
                conditions.append(RuleCondition(
                    id=UUID(cond_dict["id"]),
                    column_name=cond_dict["column_name"],
                    operator=RuleOperator(cond_dict["operator"]),
                    value=cond_dict["value"],
                    case_sensitive=cond_dict["case_sensitive"],
                    min_value=cond_dict["min_value"],
                    max_value=cond_dict["max_value"],
                    pattern=cond_dict["pattern"],
                    reference_table=cond_dict["reference_table"],
                    reference_column=cond_dict["reference_column"],
                    description=cond_dict["description"],
                    created_at=cond_dict["created_at"],
                ))

        return DataQualityRule(
            id=model.id,
            name=model.name,
            description=model.description,
            rule_type=RuleType(model.rule_type),
            severity=RuleSeverity(model.severity),
            dataset_name=model.dataset_name,
            table_name=model.table_name,
            schema_name=model.schema_name,
            conditions=conditions,
            logical_operator=model.logical_operator,
            expression=model.expression,
            is_active=model.is_active,
            is_blocking=model.is_blocking,
            auto_fix=model.auto_fix,
            fix_action=model.fix_action,
            violation_threshold=model.violation_threshold,
            sample_size=model.sample_size,
            created_at=model.created_at,
            created_by=model.created_by,
            updated_at=model.updated_at,
            updated_by=model.updated_by,
            last_evaluated_at=model.last_evaluated_at,
            evaluation_count=model.evaluation_count,
            violation_count=model.violation_count,
            last_violation_at=model.last_violation_at,
            config=model.config,
            tags=model.tags,
            depends_on=model.depends_on,
        )

    def _to_model(self, entity: DataQualityRule) -> DataQualityRuleModel:
        """Convert a DataQualityRule entity to a SQLAlchemy model."""
        return DataQualityRuleModel(
            id=entity.id,
            name=entity.name,
            description=entity.description,
            rule_type=entity.rule_type.value,
            severity=entity.severity.value,
            dataset_name=entity.dataset_name,
            table_name=entity.table_name,
            schema_name=entity.schema_name,
            conditions=[cond.to_dict() for cond in entity.conditions],
            logical_operator=entity.logical_operator,
            expression=entity.expression,
            is_active=entity.is_active,
            is_blocking=entity.is_blocking,
            auto_fix=entity.auto_fix,
            fix_action=entity.fix_action,
            violation_threshold=entity.violation_threshold,
            sample_size=entity.sample_size,
            created_at=entity.created_at,
            created_by=entity.created_by,
            updated_at=entity.updated_at,
            updated_by=entity.updated_by,
            last_evaluated_at=entity.last_evaluated_at,
            evaluation_count=entity.evaluation_count,
            violation_count=entity.violation_count,
            last_violation_at=entity.last_violation_at,
            config=entity.config,
            tags=entity.tags,
            depends_on=entity.depends_on,
        )

    def _update_model_from_entity(self, model: DataQualityRuleModel, entity: DataQualityRule) -> None:
        """Update a SQLAlchemy model from a DataQualityRule entity."""
        model.name = entity.name
        model.description = entity.description
        model.rule_type = entity.rule_type.value
        model.severity = entity.severity.value
        model.dataset_name = entity.dataset_name
        model.table_name = entity.table_name
        model.schema_name = entity.schema_name
        model.conditions = [cond.to_dict() for cond in entity.conditions]
        model.logical_operator = entity.logical_operator
        model.expression = entity.expression
        model.is_active = entity.is_active
        model.is_blocking = entity.is_blocking
        model.auto_fix = entity.auto_fix
        model.fix_action = entity.fix_action
        model.violation_threshold = entity.violation_threshold
        model.sample_size = entity.sample_size
        model.created_at = entity.created_at
        model.created_by = entity.created_by
        model.updated_at = entity.updated_at
        model.updated_by = entity.updated_by
        model.last_evaluated_at = entity.last_evaluated_at
        model.evaluation_count = entity.evaluation_count
        model.violation_count = entity.violation_count
        model.last_violation_at = entity.last_violation_at
        model.config = entity.config
        model.tags = entity.tags
        model.depends_on = entity.depends_on