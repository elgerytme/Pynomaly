"""Quality Rule repository interface."""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from ..entities.quality_rule import QualityRule, RuleId, DatasetId, UserId, RuleType, RuleStatus, QualityCategory


class QualityRuleRepository(ABC):
    """Repository interface for quality rule persistence."""
    
    @abstractmethod
    async def save(self, rule: QualityRule) -> None:
        """Save a quality rule."""
        pass
    
    @abstractmethod
    async def get_by_id(self, rule_id: RuleId) -> Optional[QualityRule]:
        """Get quality rule by ID."""
        pass
    
    @abstractmethod
    async def get_by_name(self, rule_name: str) -> Optional[QualityRule]:
        """Get quality rule by name."""
        pass
    
    @abstractmethod
    async def get_by_dataset_id(self, dataset_id: DatasetId) -> List[QualityRule]:
        """Get all rules for a dataset."""
        pass
    
    @abstractmethod
    async def get_active_rules(self) -> List[QualityRule]:
        """Get all active rules."""
        pass
    
    @abstractmethod
    async def get_by_status(self, status: RuleStatus) -> List[QualityRule]:
        """Get rules by status."""
        pass
    
    @abstractmethod
    async def get_by_type(self, rule_type: RuleType) -> List[QualityRule]:
        """Get rules by type."""
        pass
    
    @abstractmethod
    async def get_by_category(self, category: QualityCategory) -> List[QualityRule]:
        """Get rules by category."""
        pass
    
    @abstractmethod
    async def get_by_created_by(self, user_id: UserId) -> List[QualityRule]:
        """Get rules created by a user."""
        pass
    
    @abstractmethod
    async def get_scheduled_rules(self) -> List[QualityRule]:
        """Get rules that have scheduling enabled."""
        pass
    
    @abstractmethod
    async def get_rules_due_for_execution(self) -> List[QualityRule]:
        """Get rules that are due for execution."""
        pass
    
    @abstractmethod
    async def delete(self, rule_id: RuleId) -> None:
        """Delete a quality rule."""
        pass
    
    @abstractmethod
    async def list_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[QualityRule]:
        """List all quality rules with pagination."""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Count total number of rules."""
        pass
    
    @abstractmethod
    async def search_by_name_or_description(self, query: str) -> List[QualityRule]:
        """Search rules by name or description."""
        pass
    
    @abstractmethod
    async def get_rules_with_violations(self) -> List[QualityRule]:
        """Get rules that currently have threshold violations."""
        pass