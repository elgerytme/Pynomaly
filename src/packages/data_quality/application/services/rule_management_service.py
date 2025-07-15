"""Rule Management Service for managing quality rule lifecycle."""

from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

from ...domain.entities.quality_rule import (
    QualityRule, RuleId, RuleType, RuleStatus, QualityCategory, 
    Severity, ValidationLogic, RuleMetadata, RuleSchedule, 
    QualityThreshold, DatasetId, UserId
)
from ...domain.repositories.quality_rule_repository import QualityRuleRepository


logger = logging.getLogger(__name__)


class RuleManagementService:
    """Service for managing quality rule lifecycle and operations."""
    
    def __init__(self, rule_repository: QualityRuleRepository):
        self.rule_repository = rule_repository
    
    async def create_rule(
        self,
        rule_name: str,
        rule_type: RuleType,
        category: QualityCategory,
        validation_logic: ValidationLogic,
        metadata: RuleMetadata,
        severity: Severity = Severity.MEDIUM,
        target_datasets: Optional[List[DatasetId]] = None,
        target_columns: Optional[List[str]] = None,
        thresholds: Optional[QualityThreshold] = None,
        schedule: Optional[RuleSchedule] = None,
        created_by: Optional[UserId] = None
    ) -> QualityRule:
        """Create a new quality rule."""
        logger.info(f"Creating new quality rule: {rule_name}")
        
        # Check if rule with same name already exists
        existing_rule = await self.rule_repository.get_by_name(rule_name)
        if existing_rule:
            raise ValueError(f"Rule with name '{rule_name}' already exists")
        
        # Validate rule configuration
        self._validate_rule_configuration(validation_logic, target_columns)
        
        # Create rule
        rule = QualityRule(
            rule_id=RuleId(),
            rule_name=rule_name,
            rule_type=rule_type,
            category=category,
            severity=severity,
            status=RuleStatus.DRAFT,
            validation_logic=validation_logic,
            target_datasets=target_datasets or [],
            target_columns=target_columns or [],
            thresholds=thresholds or QualityThreshold(),
            schedule=schedule,
            metadata=metadata,
            created_by=created_by or UserId()
        )
        
        await self.rule_repository.save(rule)
        logger.info(f"Created rule {rule.rule_id.value} with status {rule.status}")
        
        return rule
    
    async def update_rule(
        self,
        rule_id: RuleId,
        updates: Dict[str, Any]
    ) -> QualityRule:
        """Update an existing quality rule."""
        logger.info(f"Updating rule {rule_id.value}")
        
        rule = await self.rule_repository.get_by_id(rule_id)
        if not rule:
            raise ValueError(f"Rule {rule_id.value} not found")
        
        # Handle validation logic updates
        if "validation_logic" in updates:
            new_logic = updates["validation_logic"]
            self._validate_rule_configuration(new_logic, rule.target_columns)
            rule.update_validation_logic(new_logic)
        
        # Handle other updates
        updatable_fields = [
            "rule_name", "severity", "target_datasets", "target_columns",
            "thresholds", "schedule", "metadata"
        ]
        
        for field, value in updates.items():
            if field in updatable_fields and hasattr(rule, field):
                setattr(rule, field, value)
                rule.updated_at = datetime.utcnow()
                rule.version += 1
        
        await self.rule_repository.save(rule)
        logger.info(f"Updated rule {rule_id.value}")
        
        return rule
    
    async def activate_rule(self, rule_id: RuleId, approved_by: UserId) -> QualityRule:
        """Activate a quality rule."""
        logger.info(f"Activating rule {rule_id.value}")
        
        rule = await self.rule_repository.get_by_id(rule_id)
        if not rule:
            raise ValueError(f"Rule {rule_id.value} not found")
        
        # Validate rule before activation
        await self._validate_rule_for_activation(rule)
        
        rule.activate(approved_by)
        await self.rule_repository.save(rule)
        
        logger.info(f"Activated rule {rule_id.value}")
        return rule
    
    async def deactivate_rule(self, rule_id: RuleId) -> QualityRule:
        """Deactivate a quality rule."""
        logger.info(f"Deactivating rule {rule_id.value}")
        
        rule = await self.rule_repository.get_by_id(rule_id)
        if not rule:
            raise ValueError(f"Rule {rule_id.value} not found")
        
        rule.deactivate()
        await self.rule_repository.save(rule)
        
        logger.info(f"Deactivated rule {rule_id.value}")
        return rule
    
    async def deprecate_rule(self, rule_id: RuleId, reason: str) -> QualityRule:
        """Deprecate a quality rule."""
        logger.info(f"Deprecating rule {rule_id.value}: {reason}")
        
        rule = await self.rule_repository.get_by_id(rule_id)
        if not rule:
            raise ValueError(f"Rule {rule_id.value} not found")
        
        rule.deprecate()
        
        # Add deprecation reason to metadata
        rule.metadata.description += f"\n[DEPRECATED: {reason}]"
        
        await self.rule_repository.save(rule)
        
        logger.info(f"Deprecated rule {rule_id.value}")
        return rule
    
    async def delete_rule(self, rule_id: RuleId, force: bool = False) -> None:
        """Delete a quality rule."""
        logger.info(f"Deleting rule {rule_id.value}")
        
        rule = await self.rule_repository.get_by_id(rule_id)
        if not rule:
            raise ValueError(f"Rule {rule_id.value} not found")
        
        # Check if rule can be deleted
        if rule.status == RuleStatus.ACTIVE and not force:
            raise ValueError("Cannot delete active rule. Deactivate first or use force=True")
        
        await self.rule_repository.delete(rule_id)
        logger.info(f"Deleted rule {rule_id.value}")
    
    async def get_rules_by_dataset(self, dataset_id: DatasetId) -> List[QualityRule]:
        """Get all rules for a specific dataset."""
        return await self.rule_repository.get_by_dataset_id(dataset_id)
    
    async def get_rules_by_type(self, rule_type: RuleType) -> List[QualityRule]:
        """Get all rules of a specific type."""
        return await self.rule_repository.get_by_type(rule_type)
    
    async def get_rules_by_category(self, category: QualityCategory) -> List[QualityRule]:
        """Get all rules in a specific category."""
        return await self.rule_repository.get_by_category(category)
    
    async def get_active_rules(self) -> List[QualityRule]:
        """Get all active rules."""
        return await self.rule_repository.get_active_rules()
    
    async def get_scheduled_rules(self) -> List[QualityRule]:
        """Get all rules with scheduling enabled."""
        return await self.rule_repository.get_scheduled_rules()
    
    async def get_rules_due_for_execution(self) -> List[QualityRule]:
        """Get rules that are due for execution."""
        return await self.rule_repository.get_rules_due_for_execution()
    
    async def search_rules(self, query: str) -> List[QualityRule]:
        """Search rules by name or description."""
        return await self.rule_repository.search_by_name_or_description(query)
    
    async def get_rules_with_violations(self) -> List[QualityRule]:
        """Get rules that currently have threshold violations."""
        return await self.rule_repository.get_rules_with_violations()
    
    async def clone_rule(
        self,
        source_rule_id: RuleId,
        new_name: str,
        created_by: UserId
    ) -> QualityRule:
        """Clone an existing rule with a new name."""
        logger.info(f"Cloning rule {source_rule_id.value} as {new_name}")
        
        source_rule = await self.rule_repository.get_by_id(source_rule_id)
        if not source_rule:
            raise ValueError(f"Source rule {source_rule_id.value} not found")
        
        # Check if new name is available
        existing_rule = await self.rule_repository.get_by_name(new_name)
        if existing_rule:
            raise ValueError(f"Rule with name '{new_name}' already exists")
        
        # Create cloned rule
        cloned_rule = QualityRule(
            rule_id=RuleId(),
            rule_name=new_name,
            rule_type=source_rule.rule_type,
            category=source_rule.category,
            severity=source_rule.severity,
            status=RuleStatus.DRAFT,
            validation_logic=source_rule.validation_logic,
            target_datasets=source_rule.target_datasets.copy(),
            target_columns=source_rule.target_columns.copy(),
            thresholds=source_rule.thresholds,
            schedule=source_rule.schedule,
            metadata=RuleMetadata(
                description=f"Cloned from {source_rule.rule_name}: {source_rule.metadata.description}",
                business_justification=source_rule.metadata.business_justification,
                data_owner=source_rule.metadata.data_owner,
                business_glossary_terms=source_rule.metadata.business_glossary_terms.copy(),
                related_regulations=source_rule.metadata.related_regulations.copy(),
                documentation_url=source_rule.metadata.documentation_url
            ),
            created_by=created_by
        )
        
        await self.rule_repository.save(cloned_rule)
        logger.info(f"Cloned rule {source_rule_id.value} to {cloned_rule.rule_id.value}")
        
        return cloned_rule
    
    async def bulk_update_status(
        self,
        rule_ids: List[RuleId],
        new_status: RuleStatus,
        updated_by: UserId
    ) -> List[QualityRule]:
        """Bulk update status for multiple rules."""
        logger.info(f"Bulk updating {len(rule_ids)} rules to status {new_status}")
        
        updated_rules = []
        
        for rule_id in rule_ids:
            try:
                rule = await self.rule_repository.get_by_id(rule_id)
                if rule:
                    if new_status == RuleStatus.ACTIVE:
                        await self.activate_rule(rule_id, updated_by)
                    elif new_status == RuleStatus.INACTIVE:
                        await self.deactivate_rule(rule_id)
                    elif new_status == RuleStatus.DEPRECATED:
                        await self.deprecate_rule(rule_id, "Bulk deprecation")
                    
                    updated_rule = await self.rule_repository.get_by_id(rule_id)
                    if updated_rule:
                        updated_rules.append(updated_rule)
                        
            except Exception as e:
                logger.error(f"Failed to update rule {rule_id.value}: {str(e)}")
                continue
        
        logger.info(f"Successfully updated {len(updated_rules)} rules")
        return updated_rules
    
    async def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about rules in the system."""
        all_rules = await self.rule_repository.list_all()
        
        stats = {
            "total_rules": len(all_rules),
            "by_status": {},
            "by_type": {},
            "by_category": {},
            "by_severity": {},
            "active_rules": 0,
            "scheduled_rules": 0,
            "rules_with_violations": 0
        }
        
        for rule in all_rules:
            # Status distribution
            status = rule.status.value
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
            
            # Type distribution
            rule_type = rule.rule_type.value
            stats["by_type"][rule_type] = stats["by_type"].get(rule_type, 0) + 1
            
            # Category distribution
            category = rule.category.value
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
            
            # Severity distribution
            severity = rule.severity.value
            stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1
            
            # Active rules
            if rule.is_active():
                stats["active_rules"] += 1
            
            # Scheduled rules
            if rule.is_scheduled():
                stats["scheduled_rules"] += 1
            
            # Rules with violations
            if rule.is_threshold_violated():
                stats["rules_with_violations"] += 1
        
        return stats
    
    def _validate_rule_configuration(
        self,
        validation_logic: ValidationLogic,
        target_columns: Optional[List[str]]
    ) -> None:
        """Validate rule configuration for correctness."""
        if not validation_logic.expression.strip():
            raise ValueError("Validation logic expression cannot be empty")
        
        # Validate column requirements
        column_dependent_types = ["regex", "range", "list"]
        if validation_logic.logic_type.value in column_dependent_types:
            if not target_columns:
                raise ValueError(f"Target columns required for {validation_logic.logic_type.value} validation")
        
        # Validate parameters
        if validation_logic.logic_type.value == "range":
            params = validation_logic.parameters
            if "min_value" in params and "max_value" in params:
                if params["min_value"] >= params["max_value"]:
                    raise ValueError("min_value must be less than max_value")
    
    async def _validate_rule_for_activation(self, rule: QualityRule) -> None:
        """Validate that a rule is ready for activation."""
        if not rule.metadata.description.strip():
            raise ValueError("Rule description is required for activation")
        
        if not rule.metadata.business_justification.strip():
            raise ValueError("Business justification is required for activation")
        
        if not rule.target_datasets and not rule.target_columns:
            logger.warning(f"Rule {rule.rule_id.value} has no target datasets or columns (will apply globally)")
        
        # Additional validation can be added here
        # e.g., test the rule against sample data