"""Use case for defining and managing quality rules."""

from typing import List, Optional
import logging

from ...domain.entities.quality_rule import (
    QualityRule, RuleType, QualityCategory, Severity, 
    ValidationLogic, RuleMetadata, RuleSchedule, 
    QualityThreshold, DatasetId, UserId
)
from ..services.rule_management_service import RuleManagementService


logger = logging.getLogger(__name__)


class DefineQualityRulesUseCase:
    """Use case for defining quality rules by data quality engineers."""
    
    def __init__(self, rule_management_service: RuleManagementService):
        self.rule_management_service = rule_management_service
    
    async def create_completeness_rule(
        self,
        rule_name: str,
        target_columns: List[str],
        target_datasets: Optional[List[DatasetId]] = None,
        description: str = "",
        business_justification: str = "",
        severity: Severity = Severity.MEDIUM,
        pass_threshold: float = 0.95,
        created_by: Optional[UserId] = None
    ) -> QualityRule:
        """Create a completeness validation rule."""
        
        validation_logic = ValidationLogic(
            logic_type="completeness",
            expression="NOT NULL AND LENGTH(TRIM(value)) > 0",
            parameters={},
            error_message_template="Field {column} is empty or null",
            success_criteria="All specified fields must have non-null, non-empty values"
        )
        
        metadata = RuleMetadata(
            description=description or f"Completeness check for columns: {', '.join(target_columns)}",
            business_justification=business_justification or "Ensure data completeness for critical business fields",
            business_glossary_terms=["completeness", "data_quality", "required_fields"]
        )
        
        thresholds = QualityThreshold(
            pass_rate_threshold=pass_threshold,
            warning_threshold=max(0.8, pass_threshold - 0.1),
            critical_threshold=max(0.7, pass_threshold - 0.2)
        )
        
        return await self.rule_management_service.create_rule(
            rule_name=rule_name,
            rule_type=RuleType.COMPLETENESS,
            category=QualityCategory.DATA_INTEGRITY,
            validation_logic=validation_logic,
            metadata=metadata,
            severity=severity,
            target_datasets=target_datasets,
            target_columns=target_columns,
            thresholds=thresholds,
            created_by=created_by
        )
    
    async def create_format_validation_rule(
        self,
        rule_name: str,
        target_column: str,
        regex_pattern: str,
        target_datasets: Optional[List[DatasetId]] = None,
        description: str = "",
        business_justification: str = "",
        severity: Severity = Severity.MEDIUM,
        pass_threshold: float = 0.95,
        created_by: Optional[UserId] = None
    ) -> QualityRule:
        """Create a format validation rule using regex."""
        
        validation_logic = ValidationLogic(
            logic_type="regex",
            expression=regex_pattern,
            parameters={"pattern": regex_pattern},
            error_message_template=f"Field {target_column} does not match required format",
            success_criteria=f"Values in {target_column} must match pattern: {regex_pattern}"
        )
        
        metadata = RuleMetadata(
            description=description or f"Format validation for {target_column}",
            business_justification=business_justification or f"Ensure {target_column} follows required format standards",
            business_glossary_terms=["format_validation", "regex", "data_standards"]
        )
        
        thresholds = QualityThreshold(
            pass_rate_threshold=pass_threshold,
            warning_threshold=max(0.8, pass_threshold - 0.1),
            critical_threshold=max(0.7, pass_threshold - 0.2)
        )
        
        return await self.rule_management_service.create_rule(
            rule_name=rule_name,
            rule_type=RuleType.VALIDITY,
            category=QualityCategory.FORMAT_VALIDATION,
            validation_logic=validation_logic,
            metadata=metadata,
            severity=severity,
            target_datasets=target_datasets,
            target_columns=[target_column],
            thresholds=thresholds,
            created_by=created_by
        )
    
    async def create_range_validation_rule(
        self,
        rule_name: str,
        target_column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        target_datasets: Optional[List[DatasetId]] = None,
        description: str = "",
        business_justification: str = "",
        severity: Severity = Severity.MEDIUM,
        pass_threshold: float = 0.95,
        created_by: Optional[UserId] = None
    ) -> QualityRule:
        """Create a numeric range validation rule."""
        
        if min_value is None and max_value is None:
            raise ValueError("Either min_value or max_value must be specified")
        
        # Build range expression
        conditions = []
        if min_value is not None:
            conditions.append(f"value >= {min_value}")
        if max_value is not None:
            conditions.append(f"value <= {max_value}")
        
        range_expression = " AND ".join(conditions)
        
        validation_logic = ValidationLogic(
            logic_type="range",
            expression=range_expression,
            parameters={
                "min_value": min_value,
                "max_value": max_value
            },
            error_message_template=f"Field {target_column} is outside acceptable range",
            success_criteria=f"Values in {target_column} must be within specified range"
        )
        
        range_desc = ""
        if min_value is not None and max_value is not None:
            range_desc = f"between {min_value} and {max_value}"
        elif min_value is not None:
            range_desc = f"greater than or equal to {min_value}"
        else:
            range_desc = f"less than or equal to {max_value}"
        
        metadata = RuleMetadata(
            description=description or f"Range validation for {target_column} ({range_desc})",
            business_justification=business_justification or f"Ensure {target_column} values are within business-acceptable ranges",
            business_glossary_terms=["range_validation", "numerical_bounds", "data_constraints"]
        )
        
        thresholds = QualityThreshold(
            pass_rate_threshold=pass_threshold,
            warning_threshold=max(0.8, pass_threshold - 0.1),
            critical_threshold=max(0.7, pass_threshold - 0.2)
        )
        
        return await self.rule_management_service.create_rule(
            rule_name=rule_name,
            rule_type=RuleType.VALIDITY,
            category=QualityCategory.BUSINESS_RULES,
            validation_logic=validation_logic,
            metadata=metadata,
            severity=severity,
            target_datasets=target_datasets,
            target_columns=[target_column],
            thresholds=thresholds,
            created_by=created_by
        )
    
    async def create_uniqueness_rule(
        self,
        rule_name: str,
        target_columns: List[str],
        target_datasets: Optional[List[DatasetId]] = None,
        description: str = "",
        business_justification: str = "",
        severity: Severity = Severity.HIGH,
        pass_threshold: float = 0.99,
        created_by: Optional[UserId] = None
    ) -> QualityRule:
        """Create a uniqueness validation rule."""
        
        validation_logic = ValidationLogic(
            logic_type="uniqueness",
            expression=f"UNIQUE({', '.join(target_columns)})",
            parameters={"columns": target_columns},
            error_message_template=f"Duplicate values found in {', '.join(target_columns)}",
            success_criteria=f"Values in {', '.join(target_columns)} must be unique"
        )
        
        metadata = RuleMetadata(
            description=description or f"Uniqueness check for {', '.join(target_columns)}",
            business_justification=business_justification or f"Ensure {', '.join(target_columns)} maintain uniqueness constraints",
            business_glossary_terms=["uniqueness", "duplicate_detection", "data_integrity"]
        )
        
        thresholds = QualityThreshold(
            pass_rate_threshold=pass_threshold,
            warning_threshold=max(0.95, pass_threshold - 0.05),
            critical_threshold=max(0.90, pass_threshold - 0.1)
        )
        
        return await self.rule_management_service.create_rule(
            rule_name=rule_name,
            rule_type=RuleType.UNIQUENESS,
            category=QualityCategory.DATA_INTEGRITY,
            validation_logic=validation_logic,
            metadata=metadata,
            severity=severity,
            target_datasets=target_datasets,
            target_columns=target_columns,
            thresholds=thresholds,
            created_by=created_by
        )
    
    async def create_custom_sql_rule(
        self,
        rule_name: str,
        sql_expression: str,
        target_datasets: Optional[List[DatasetId]] = None,
        target_columns: Optional[List[str]] = None,
        description: str = "",
        business_justification: str = "",
        rule_type: RuleType = RuleType.CUSTOM,
        category: QualityCategory = QualityCategory.BUSINESS_RULES,
        severity: Severity = Severity.MEDIUM,
        pass_threshold: float = 0.95,
        created_by: Optional[UserId] = None
    ) -> QualityRule:
        """Create a custom SQL-based validation rule."""
        
        validation_logic = ValidationLogic(
            logic_type="sql",
            expression=sql_expression,
            parameters={},
            error_message_template="Custom validation rule failed",
            success_criteria="Records must satisfy the custom SQL condition"
        )
        
        metadata = RuleMetadata(
            description=description or f"Custom SQL validation: {sql_expression[:100]}...",
            business_justification=business_justification or "Custom business logic validation",
            business_glossary_terms=["custom_validation", "sql_rule", "business_logic"]
        )
        
        thresholds = QualityThreshold(
            pass_rate_threshold=pass_threshold,
            warning_threshold=max(0.8, pass_threshold - 0.1),
            critical_threshold=max(0.7, pass_threshold - 0.2)
        )
        
        return await self.rule_management_service.create_rule(
            rule_name=rule_name,
            rule_type=rule_type,
            category=category,
            validation_logic=validation_logic,
            metadata=metadata,
            severity=severity,
            target_datasets=target_datasets,
            target_columns=target_columns or [],
            thresholds=thresholds,
            created_by=created_by
        )
    
    async def create_scheduled_rule(
        self,
        rule_name: str,
        rule_type: RuleType,
        validation_logic: ValidationLogic,
        metadata: RuleMetadata,
        schedule_frequency: str,  # "hourly", "daily", "weekly", "monthly"
        cron_expression: Optional[str] = None,
        target_datasets: Optional[List[DatasetId]] = None,
        target_columns: Optional[List[str]] = None,
        severity: Severity = Severity.MEDIUM,
        pass_threshold: float = 0.95,
        created_by: Optional[UserId] = None
    ) -> QualityRule:
        """Create a scheduled validation rule."""
        
        schedule = RuleSchedule(
            enabled=True,
            frequency=schedule_frequency,
            cron_expression=cron_expression
        )
        
        thresholds = QualityThreshold(
            pass_rate_threshold=pass_threshold,
            warning_threshold=max(0.8, pass_threshold - 0.1),
            critical_threshold=max(0.7, pass_threshold - 0.2)
        )
        
        return await self.rule_management_service.create_rule(
            rule_name=rule_name,
            rule_type=rule_type,
            category=QualityCategory.BUSINESS_RULES,
            validation_logic=validation_logic,
            metadata=metadata,
            severity=severity,
            target_datasets=target_datasets,
            target_columns=target_columns or [],
            thresholds=thresholds,
            schedule=schedule,
            created_by=created_by
        )
    
    async def create_email_validation_rule(
        self,
        rule_name: str,
        target_column: str,
        target_datasets: Optional[List[DatasetId]] = None,
        description: str = "",
        business_justification: str = "",
        severity: Severity = Severity.MEDIUM,
        pass_threshold: float = 0.98,
        created_by: Optional[UserId] = None
    ) -> QualityRule:
        """Create an email format validation rule."""
        
        # Standard email regex pattern
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        return await self.create_format_validation_rule(
            rule_name=rule_name,
            target_column=target_column,
            regex_pattern=email_pattern,
            target_datasets=target_datasets,
            description=description or f"Email format validation for {target_column}",
            business_justification=business_justification or f"Ensure {target_column} contains valid email addresses",
            severity=severity,
            pass_threshold=pass_threshold,
            created_by=created_by
        )
    
    async def create_phone_validation_rule(
        self,
        rule_name: str,
        target_column: str,
        target_datasets: Optional[List[DatasetId]] = None,
        description: str = "",
        business_justification: str = "",
        severity: Severity = Severity.MEDIUM,
        pass_threshold: float = 0.95,
        created_by: Optional[UserId] = None
    ) -> QualityRule:
        """Create a phone number format validation rule."""
        
        # Standard phone number pattern (US format)
        phone_pattern = r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$'
        
        return await self.create_format_validation_rule(
            rule_name=rule_name,
            target_column=target_column,
            regex_pattern=phone_pattern,
            target_datasets=target_datasets,
            description=description or f"Phone number format validation for {target_column}",
            business_justification=business_justification or f"Ensure {target_column} contains valid phone numbers",
            severity=severity,
            pass_threshold=pass_threshold,
            created_by=created_by
        )
    
    async def get_rule_templates(self) -> List[dict]:
        """Get predefined rule templates for common validation patterns."""
        return [
            {
                "name": "Email Validation",
                "type": "format",
                "pattern": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                "description": "Validates email address format"
            },
            {
                "name": "Phone Number Validation (US)",
                "type": "format",
                "pattern": r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$',
                "description": "Validates US phone number format"
            },
            {
                "name": "ZIP Code Validation (US)",
                "type": "format",
                "pattern": r'^\d{5}(-\d{4})?$',
                "description": "Validates US ZIP code format"
            },
            {
                "name": "Date Format Validation (YYYY-MM-DD)",
                "type": "format",
                "pattern": r'^\d{4}-\d{2}-\d{2}$',
                "description": "Validates ISO date format"
            },
            {
                "name": "Percentage Range (0-100)",
                "type": "range",
                "min_value": 0,
                "max_value": 100,
                "description": "Validates percentage values between 0 and 100"
            },
            {
                "name": "Age Range (0-120)",
                "type": "range",
                "min_value": 0,
                "max_value": 120,
                "description": "Validates human age values"
            },
            {
                "name": "Required Field Check",
                "type": "completeness",
                "description": "Ensures field is not null or empty"
            },
            {
                "name": "Unique Identifier Check",
                "type": "uniqueness",
                "description": "Ensures field values are unique"
            }
        ]