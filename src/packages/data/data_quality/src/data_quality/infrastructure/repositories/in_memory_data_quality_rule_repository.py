
from typing import Dict, Optional
from uuid import UUID

from ...application.ports.data_quality_rule_repository import DataQualityRuleRepository
from ...domain.entities.data_quality_rule import DataQualityRule


class InMemoryDataQualityRuleRepository(DataQualityRuleRepository):
    """In-memory implementation of DataQualityRuleRepository."""

    def __init__(self):
        self.rules: Dict[UUID, DataQualityRule] = {}

    def get_by_id(self, id: UUID) -> Optional[DataQualityRule]:
        """Get a data quality rule by its ID."""
        return self.rules.get(id)

    def save(self, data_quality_rule: DataQualityRule) -> None:
        """Save a data quality rule."""
        self.rules[data_quality_rule.id] = data_quality_rule
