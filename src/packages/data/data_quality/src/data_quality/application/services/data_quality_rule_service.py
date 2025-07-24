
from uuid import UUID

from ..ports.data_quality_rule_repository import DataQualityRuleRepository
from ...domain.entities.data_quality_rule import DataQualityRule


class DataQualityRuleService:
    """Service for data quality rules."""

    def __init__(self, data_quality_rule_repository: DataQualityRuleRepository):
        self.data_quality_rule_repository = data_quality_rule_repository

    def create_rule(self, rule: DataQualityRule) -> DataQualityRule:
        """Create a new data quality rule."""
        self.data_quality_rule_repository.save(rule)
        return rule

    def get_rule(self, rule_id: UUID) -> DataQualityRule:
        """Get a data quality rule by its ID."""
        return self.data_quality_rule_repository.get_by_id(rule_id)
