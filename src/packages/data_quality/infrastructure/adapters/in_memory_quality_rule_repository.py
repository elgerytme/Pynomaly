"""In-memory implementation of Quality Rule Repository."""

from typing import Dict, List, Optional
from uuid import UUID
from datetime import datetime
import structlog

from ...domain.entities.quality_rule import (
    QualityRule, RuleId, DatasetId, UserId, RuleType, 
    RuleStatus, QualityCategory
)
from ...domain.repositories.quality_rule_repository import QualityRuleRepository

logger = structlog.get_logger(__name__)


class InMemoryQualityRuleRepository(QualityRuleRepository):
    """In-memory implementation for testing and development."""
    
    def __init__(self):
        self._rules: Dict[UUID, QualityRule] = {}
        logger.info("Initialized in-memory quality rule repository")
    
    async def save(self, rule: QualityRule) -> None:
        """Save a quality rule."""
        self._rules[rule.rule_id.value] = rule
        logger.debug(
            "Saved quality rule",
            rule_id=str(rule.rule_id.value),
            rule_name=rule.rule_name,
            status=rule.status.value
        )
    
    async def get_by_id(self, rule_id: RuleId) -> Optional[QualityRule]:
        """Get quality rule by ID."""
        rule = self._rules.get(rule_id.value)
        if rule:
            logger.debug(
                "Retrieved quality rule",
                rule_id=str(rule_id.value)
            )
        return rule
    
    async def get_by_name(self, rule_name: str) -> Optional[QualityRule]:
        """Get quality rule by name."""
        for rule in self._rules.values():
            if rule.rule_name == rule_name:
                logger.debug(
                    "Retrieved quality rule by name",
                    rule_name=rule_name
                )
                return rule
        return None
    
    async def get_by_dataset_id(self, dataset_id: DatasetId) -> List[QualityRule]:
        """Get all rules for a dataset."""
        rules = [
            rule for rule in self._rules.values()
            if dataset_id in rule.target_datasets
        ]
        logger.debug(
            "Retrieved rules by dataset",
            dataset_id=str(dataset_id.value),
            count=len(rules)
        )
        return rules
    
    async def get_active_rules(self) -> List[QualityRule]:
        """Get all active rules."""
        rules = [
            rule for rule in self._rules.values()
            if rule.status == RuleStatus.ACTIVE
        ]
        logger.debug(
            "Retrieved active rules",
            count=len(rules)
        )
        return rules
    
    async def get_by_status(self, status: RuleStatus) -> List[QualityRule]:
        """Get rules by status."""
        rules = [
            rule for rule in self._rules.values()
            if rule.status == status
        ]
        logger.debug(
            "Retrieved rules by status",
            status=status.value,
            count=len(rules)
        )
        return rules
    
    async def get_by_type(self, rule_type: RuleType) -> List[QualityRule]:
        """Get rules by type."""
        rules = [
            rule for rule in self._rules.values()
            if rule.rule_type == rule_type
        ]
        logger.debug(
            "Retrieved rules by type",
            rule_type=rule_type.value,
            count=len(rules)
        )
        return rules
    
    async def get_by_category(self, category: QualityCategory) -> List[QualityRule]:
        """Get rules by category."""
        rules = [
            rule for rule in self._rules.values()
            if rule.category == category
        ]
        logger.debug(
            "Retrieved rules by category",
            category=category.value,
            count=len(rules)
        )
        return rules
    
    async def get_by_created_by(self, user_id: UserId) -> List[QualityRule]:
        """Get rules created by a user."""
        rules = [
            rule for rule in self._rules.values()
            if rule.created_by.value == user_id.value
        ]
        logger.debug(
            "Retrieved rules by creator",
            user_id=str(user_id.value),
            count=len(rules)
        )
        return rules
    
    async def get_scheduled_rules(self) -> List[QualityRule]:
        """Get rules that have scheduling enabled."""
        rules = [
            rule for rule in self._rules.values()
            if rule.is_scheduled()
        ]
        logger.debug(
            "Retrieved scheduled rules",
            count=len(rules)
        )
        return rules
    
    async def get_rules_due_for_execution(self) -> List[QualityRule]:
        """Get rules that are due for execution."""
        rules = [
            rule for rule in self._rules.values()
            if rule.should_execute_now()
        ]
        logger.debug(
            "Retrieved rules due for execution",
            count=len(rules)
        )
        return rules
    
    async def delete(self, rule_id: RuleId) -> None:
        """Delete a quality rule."""
        if rule_id.value in self._rules:
            del self._rules[rule_id.value]
            logger.debug(
                "Deleted quality rule",
                rule_id=str(rule_id.value)
            )
    
    async def list_all(
        self, 
        limit: Optional[int] = None, 
        offset: Optional[int] = None
    ) -> List[QualityRule]:
        """List all quality rules with pagination."""
        rules = list(self._rules.values())
        
        # Sort by creation time (most recent first)
        rules.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        if offset:
            rules = rules[offset:]
        if limit:
            rules = rules[:limit]
        
        logger.debug(
            "Listed all rules",
            total_count=len(self._rules),
            returned_count=len(rules)
        )
        return rules
    
    async def count(self) -> int:
        """Count total number of rules."""
        count = len(self._rules)
        logger.debug("Counted rules", count=count)
        return count
    
    async def search_by_name_or_description(self, query: str) -> List[QualityRule]:
        """Search rules by name or description."""
        query_lower = query.lower()
        matching_rules = []
        
        for rule in self._rules.values():
            if (query_lower in rule.rule_name.lower() or 
                query_lower in rule.rule_metadata.description.lower()):
                matching_rules.append(rule)
        
        logger.debug(
            "Searched rules by query",
            query=query,
            count=len(matching_rules)
        )
        return matching_rules
    
    async def get_rules_with_violations(self) -> List[QualityRule]:
        """Get rules that currently have threshold violations."""
        rules_with_violations = []
        
        for rule in self._rules.values():
            if rule.is_threshold_violated():
                rules_with_violations.append(rule)
        
        logger.debug(
            "Retrieved rules with violations",
            count=len(rules_with_violations)
        )
        return rules_with_violations
    
    # Additional utility methods for testing
    
    def clear(self) -> None:
        """Clear all rules (for testing)."""
        self._rules.clear()
        logger.debug("Cleared all quality rules")
    
    def get_all_sync(self) -> List[QualityRule]:
        """Get all rules synchronously (for testing)."""
        return list(self._rules.values())