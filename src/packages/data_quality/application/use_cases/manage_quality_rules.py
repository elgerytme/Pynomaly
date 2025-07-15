"""Manage Quality Rules Use Case."""

from typing import Dict, Any, Optional, List
import structlog

from ...domain.entities.quality_rule import (
    QualityRule, RuleId, DatasetId, UserId, RuleType, 
    Severity, QualityCategory, ValidationLogic, RuleMetadata
)
from ...domain.repositories.quality_rule_repository import QualityRuleRepository

logger = structlog.get_logger(__name__)


class ManageQualityRulesUseCase:
    """Use case for managing quality rules."""
    
    def __init__(self, repository: QualityRuleRepository):
        self.repository = repository
    
    async def create_rule(
        self,
        rule_name: str,
        rule_type: RuleType,
        category: QualityCategory,
        severity: Severity,
        validation_logic: ValidationLogic,
        metadata: RuleMetadata,
        created_by: UserId,
        target_datasets: Optional[List[DatasetId]] = None,
        target_columns: Optional[List[str]] = None
    ) -> QualityRule:
        """Create a new quality rule."""
        
        # Check if rule name already exists
        existing_rule = await self.repository.get_by_name(rule_name)
        if existing_rule:
            raise ValueError(f"Rule with name '{rule_name}' already exists")
        
        # Create new rule
        rule = QualityRule(
            rule_id=RuleId(),
            rule_name=rule_name,
            rule_type=rule_type,
            category=category,
            severity=severity,
            validation_logic=validation_logic,
            target_datasets=target_datasets or [],
            target_columns=target_columns or [],
            metadata=metadata,
            created_by=created_by
        )
        
        await self.repository.save(rule)
        
        logger.info(
            "Quality rule created",
            rule_id=str(rule.rule_id.value),
            rule_name=rule_name,
            rule_type=rule_type.value
        )
        
        return rule
    
    async def activate_rule(
        self,
        rule_id: RuleId,
        approved_by: UserId
    ) -> QualityRule:
        """Activate a draft rule after approval."""
        
        rule = await self.repository.get_by_id(rule_id)
        if not rule:
            raise ValueError(f"Rule with ID {rule_id.value} not found")
        
        rule.activate(approved_by)
        await self.repository.save(rule)
        
        logger.info(
            "Quality rule activated",
            rule_id=str(rule.rule_id.value),
            rule_name=rule.rule_name,
            approved_by=str(approved_by.value)
        )
        
        return rule
    
    async def deactivate_rule(
        self,
        rule_id: RuleId
    ) -> QualityRule:
        """Deactivate a rule."""
        
        rule = await self.repository.get_by_id(rule_id)
        if not rule:
            raise ValueError(f"Rule with ID {rule_id.value} not found")
        
        rule.deactivate()
        await self.repository.save(rule)
        
        logger.info(
            "Quality rule deactivated",
            rule_id=str(rule.rule_id.value),
            rule_name=rule.rule_name
        )
        
        return rule
    
    async def update_rule_logic(
        self,
        rule_id: RuleId,
        new_logic: ValidationLogic
    ) -> QualityRule:
        """Update the validation logic of a rule."""
        
        rule = await self.repository.get_by_id(rule_id)
        if not rule:
            raise ValueError(f"Rule with ID {rule_id.value} not found")
        
        rule.update_validation_logic(new_logic)
        await self.repository.save(rule)
        
        logger.info(
            "Quality rule logic updated",
            rule_id=str(rule.rule_id.value),
            rule_name=rule.rule_name
        )
        
        return rule
    
    async def delete_rule(
        self,
        rule_id: RuleId
    ) -> None:
        """Delete a quality rule."""
        
        rule = await self.repository.get_by_id(rule_id)
        if not rule:
            raise ValueError(f"Rule with ID {rule_id.value} not found")
        
        await self.repository.delete(rule_id)
        
        logger.info(
            "Quality rule deleted",
            rule_id=str(rule.rule_id.value),
            rule_name=rule.rule_name
        )
    
    async def get_rule_by_id(
        self,
        rule_id: RuleId
    ) -> Optional[QualityRule]:
        """Get a rule by its ID."""
        return await self.repository.get_by_id(rule_id)
    
    async def get_rule_by_name(
        self,
        rule_name: str
    ) -> Optional[QualityRule]:
        """Get a rule by its name."""
        return await self.repository.get_by_name(rule_name)
    
    async def get_rules_by_type(
        self,
        rule_type: RuleType
    ) -> List[QualityRule]:
        """Get all rules of a specific type."""
        return await self.repository.get_by_type(rule_type)
    
    async def get_rules_by_category(
        self,
        category: QualityCategory
    ) -> List[QualityRule]:
        """Get all rules in a specific category."""
        return await self.repository.get_by_category(category)
    
    async def get_rules_by_dataset(
        self,
        dataset_id: DatasetId
    ) -> List[QualityRule]:
        """Get all rules applicable to a dataset."""
        return await self.repository.get_by_dataset_id(dataset_id)
    
    async def get_rules_by_creator(
        self,
        user_id: UserId
    ) -> List[QualityRule]:
        """Get all rules created by a user."""
        return await self.repository.get_by_created_by(user_id)
    
    async def search_rules(
        self,
        query: str
    ) -> List[QualityRule]:
        """Search rules by name or description."""
        return await self.repository.search_by_name_or_description(query)
    
    async def get_scheduled_rules(self) -> List[QualityRule]:
        """Get all rules that have scheduling enabled."""
        return await self.repository.get_scheduled_rules()
    
    async def get_active_rules(self) -> List[QualityRule]:
        """Get all active rules."""
        return await self.repository.get_active_rules()
    
    async def get_rules_with_violations(self) -> List[QualityRule]:
        """Get rules that currently have threshold violations."""
        return await self.repository.get_rules_with_violations()
    
    async def bulk_activate_rules(
        self,
        rule_ids: List[RuleId],
        approved_by: UserId
    ) -> List[QualityRule]:
        """Activate multiple rules in bulk."""
        
        activated_rules = []
        for rule_id in rule_ids:
            try:
                rule = await self.activate_rule(rule_id, approved_by)
                activated_rules.append(rule)
            except Exception as e:
                logger.warning(
                    "Failed to activate rule in bulk operation",
                    rule_id=str(rule_id.value),
                    error=str(e)
                )
        
        logger.info(
            "Bulk rule activation completed",
            requested_count=len(rule_ids),
            activated_count=len(activated_rules)
        )
        
        return activated_rules
    
    async def bulk_deactivate_rules(
        self,
        rule_ids: List[RuleId]
    ) -> List[QualityRule]:
        """Deactivate multiple rules in bulk."""
        
        deactivated_rules = []
        for rule_id in rule_ids:
            try:
                rule = await self.deactivate_rule(rule_id)
                deactivated_rules.append(rule)
            except Exception as e:
                logger.warning(
                    "Failed to deactivate rule in bulk operation",
                    rule_id=str(rule_id.value),
                    error=str(e)
                )
        
        logger.info(
            "Bulk rule deactivation completed",
            requested_count=len(rule_ids),
            deactivated_count=len(deactivated_rules)
        )
        
        return deactivated_rules