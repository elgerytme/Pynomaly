"""In-memory implementation of Quality Rule Repository for testing and development."""

from typing import List, Optional, Dict
from uuid import UUID

from ...domain.entities.quality_rule import (
    QualityRule, RuleId, DatasetId, UserId, RuleType, 
    RuleStatus, QualityCategory
)
from ...domain.repositories.quality_rule_repository import QualityRuleRepository


class InMemoryQualityRuleRepository(QualityRuleRepository):
    """In-memory implementation of Quality Rule Repository."""
    
    def __init__(self):
        self._rules: Dict[UUID, QualityRule] = {}
    
    async def save(self, rule: QualityRule) -> None:
        """Save a quality rule."""
        self._rules[rule.rule_id.value] = rule
    
    async def get_by_id(self, rule_id: RuleId) -> Optional[QualityRule]:
        """Get quality rule by ID."""
        return self._rules.get(rule_id.value)
    
    async def get_by_name(self, rule_name: str) -> Optional[QualityRule]:
        """Get quality rule by name."""
        for rule in self._rules.values():
            if rule.rule_name == rule_name:
                return rule
        return None
    
    async def get_by_dataset_id(self, dataset_id: DatasetId) -> List[QualityRule]:
        """Get all rules for a dataset."""
        result = []
        for rule in self._rules.values():
            if dataset_id in rule.target_datasets or not rule.target_datasets:
                result.append(rule)
        return result
    
    async def get_active_rules(self) -> List[QualityRule]:
        """Get all active rules."""
        return [rule for rule in self._rules.values() if rule.status == RuleStatus.ACTIVE]
    
    async def get_by_status(self, status: RuleStatus) -> List[QualityRule]:
        """Get rules by status."""
        return [rule for rule in self._rules.values() if rule.status == status]
    
    async def get_by_type(self, rule_type: RuleType) -> List[QualityRule]:
        """Get rules by type."""
        return [rule for rule in self._rules.values() if rule.rule_type == rule_type]
    
    async def get_by_category(self, category: QualityCategory) -> List[QualityRule]:
        """Get rules by category."""
        return [rule for rule in self._rules.values() if rule.category == category]
    
    async def get_by_created_by(self, user_id: UserId) -> List[QualityRule]:
        """Get rules created by a user."""
        return [rule for rule in self._rules.values() if rule.created_by == user_id]
    
    async def get_scheduled_rules(self) -> List[QualityRule]:
        """Get rules that have scheduling enabled."""
        return [rule for rule in self._rules.values() if rule.is_scheduled()]
    
    async def get_rules_due_for_execution(self) -> List[QualityRule]:
        """Get rules that are due for execution."""
        return [rule for rule in self._rules.values() if rule.should_execute_now()]
    
    async def delete(self, rule_id: RuleId) -> None:
        """Delete a quality rule."""
        if rule_id.value in self._rules:
            del self._rules[rule_id.value]
    
    async def list_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[QualityRule]:
        """List all quality rules with pagination."""
        rules = list(self._rules.values())
        
        if offset:
            rules = rules[offset:]
        
        if limit:
            rules = rules[:limit]
        
        return rules
    
    async def count(self) -> int:
        """Count total number of rules."""
        return len(self._rules)
    
    async def search_by_name_or_description(self, query: str) -> List[QualityRule]:
        """Search rules by name or description."""
        query_lower = query.lower()
        result = []
        
        for rule in self._rules.values():
            if (query_lower in rule.rule_name.lower() or 
                query_lower in rule.metadata.description.lower()):
                result.append(rule)
        
        return result
    
    async def get_rules_with_violations(self) -> List[QualityRule]:
        """Get rules that currently have threshold violations."""
        return [rule for rule in self._rules.values() if rule.is_threshold_violated()]
    
    def clear(self) -> None:
        """Clear all rules (for testing)."""
        self._rules.clear()
    
    def get_rule_count(self) -> int:
        """Get total number of stored rules."""
        return len(self._rules)