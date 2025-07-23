
from uuid import UUID

from src.packages.data.data_quality.src.data_quality.application.services.data_quality_rule_service import DataQualityRuleService
from src.packages.data.data_quality.src.data_quality.domain.entities.data_quality_rule import DataQualityRule


class ManageDataQualityRulesUseCase:
    """Use case for managing data quality rules."""

    def __init__(self, data_quality_rule_service: DataQualityRuleService):
        self.data_quality_rule_service = data_quality_rule_service

    def create_rule(self, rule: DataQualityRule) -> DataQualityRule:
        """Create a new data quality rule."""
        return self.data_quality_rule_service.create_rule(rule)

    def get_rule(self, rule_id: UUID) -> DataQualityRule:
        """Get a data quality rule by its ID."""
        return self.data_quality_rule_service.get_rule(rule_id)
