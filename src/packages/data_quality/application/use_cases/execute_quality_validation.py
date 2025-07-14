"""Execute Quality Validation Use Case."""

from typing import Dict, Any, Optional, List
import pandas as pd
import structlog

from ...domain.entities.quality_rule import (
    QualityRule, RuleId, DatasetId, UserId, ValidationResult, 
    ValidationStatus, ValidationError
)
from ...domain.repositories.quality_rule_repository import QualityRuleRepository

logger = structlog.get_logger(__name__)


class ExecuteQualityValidationUseCase:
    """Use case for executing quality validation."""
    
    def __init__(
        self,
        repository: QualityRuleRepository,
        validation_service: Optional[Any] = None
    ):
        self.repository = repository
        self.validation_service = validation_service
    
    async def execute_rule_validation(
        self,
        rule: QualityRule,
        data: pd.DataFrame,
        dataset_id: DatasetId
    ) -> ValidationResult:
        """Execute validation for a specific rule."""
        
        logger.info(
            "Executing quality validation",
            rule_id=str(rule.rule_id.value),
            rule_name=rule.rule_name,
            dataset_id=str(dataset_id.value)
        )
        
        try:
            # Use validation service if available
            if self.validation_service:
                result = await self.validation_service.validate_rule(rule, data, dataset_id)
            else:
                # Basic fallback validation
                result = self._execute_basic_validation(rule, data, dataset_id)
            
            # Add result to rule history
            rule.add_validation_result(result)
            await self.repository.save(rule)
            
            return result
            
        except Exception as e:
            logger.error(
                "Quality validation failed",
                rule_id=str(rule.rule_id.value),
                error=str(e)
            )
            
            # Create error result
            error_result = ValidationResult(
                rule_id=rule.rule_id,
                dataset_id=dataset_id,
                status=ValidationStatus.ERROR,
                total_records=len(data),
                records_passed=0,
                records_failed=len(data),
                pass_rate=0.0,
                validation_errors=[
                    ValidationError(
                        error_message=f"Validation execution failed: {str(e)}"
                    )
                ],
                execution_time_seconds=0.0
            )
            
            rule.add_validation_result(error_result)
            await self.repository.save(rule)
            
            raise
    
    def _execute_basic_validation(
        self,
        rule: QualityRule,
        data: pd.DataFrame,
        dataset_id: DatasetId
    ) -> ValidationResult:
        """Execute basic validation logic."""
        from ...domain.entities.quality_rule import LogicType
        import time
        
        start_time = time.time()
        total_records = len(data)
        validation_errors = []
        
        # Basic validation based on rule type
        if rule.validation_logic.logic_type == LogicType.REGEX and rule.target_columns:
            # Regex validation
            import re
            pattern = rule.validation_logic.expression
            column = rule.target_columns[0]
            
            if column in data.columns:
                col_data = data[column].astype(str)
                failed_indices = []
                
                for idx, value in col_data.items():
                    if not re.match(pattern, str(value)):
                        failed_indices.append(idx)
                        validation_errors.append(
                            ValidationError(
                                row_identifier=str(idx),
                                column_name=column,
                                invalid_value=str(value),
                                error_message=f"Value does not match pattern: {pattern}"
                            )
                        )
                
                records_failed = len(failed_indices)
                records_passed = total_records - records_failed
            else:
                records_failed = total_records
                records_passed = 0
                validation_errors.append(
                    ValidationError(
                        error_message=f"Column '{column}' not found in dataset"
                    )
                )
        
        elif rule.validation_logic.logic_type == LogicType.RANGE and rule.target_columns:
            # Range validation
            column = rule.target_columns[0]
            
            if column in data.columns and pd.api.types.is_numeric_dtype(data[column]):
                params = rule.validation_logic.parameters
                min_val = params.get('min_value')
                max_val = params.get('max_value')
                
                col_data = data[column]
                failed_mask = (col_data < min_val) | (col_data > max_val) | col_data.isnull()
                failed_indices = data[failed_mask].index.tolist()
                
                for idx in failed_indices:
                    value = col_data.loc[idx]
                    validation_errors.append(
                        ValidationError(
                            row_identifier=str(idx),
                            column_name=column,
                            invalid_value=str(value),
                            error_message=f"Value {value} is outside valid range ({min_val}, {max_val})"
                        )
                    )
                
                records_failed = len(failed_indices)
                records_passed = total_records - records_failed
            else:
                records_failed = total_records
                records_passed = 0
                validation_errors.append(
                    ValidationError(
                        error_message=f"Column '{column}' not found or not numeric"
                    )
                )
        
        else:
            # Default: no validation errors (pass-through)
            records_passed = total_records
            records_failed = 0
        
        pass_rate = records_passed / total_records if total_records > 0 else 0.0
        status = ValidationStatus.PASSED if records_failed == 0 else ValidationStatus.FAILED
        
        execution_time = time.time() - start_time
        
        return ValidationResult(
            rule_id=rule.rule_id,
            dataset_id=dataset_id,
            status=status,
            total_records=total_records,
            records_passed=records_passed,
            records_failed=records_failed,
            pass_rate=pass_rate,
            validation_errors=validation_errors[:100],  # Limit to first 100 errors
            execution_time_seconds=execution_time
        )
    
    async def execute_dataset_validation(
        self,
        dataset_id: DatasetId,
        data: pd.DataFrame
    ) -> List[ValidationResult]:
        """Execute validation for all active rules against a dataset."""
        
        # Get all active rules for this dataset
        rules = await self.repository.get_by_dataset_id(dataset_id)
        active_rules = [rule for rule in rules if rule.is_active()]
        
        logger.info(
            "Executing dataset validation",
            dataset_id=str(dataset_id.value),
            active_rules_count=len(active_rules)
        )
        
        results = []
        for rule in active_rules:
            try:
                result = await self.execute_rule_validation(rule, data, dataset_id)
                results.append(result)
            except Exception as e:
                logger.warning(
                    "Rule validation failed",
                    rule_id=str(rule.rule_id.value),
                    error=str(e)
                )
        
        return results
    
    async def get_rule_by_id(
        self, 
        rule_id: RuleId
    ) -> Optional[QualityRule]:
        """Get rule by ID."""
        return await self.repository.get_by_id(rule_id)
    
    async def get_active_rules(self) -> List[QualityRule]:
        """Get all active rules."""
        return await self.repository.get_active_rules()
    
    async def get_rules_with_violations(self) -> List[QualityRule]:
        """Get rules that currently have threshold violations."""
        return await self.repository.get_rules_with_violations()
    
    async def get_rules_due_for_execution(self) -> List[QualityRule]:
        """Get rules that are due for execution based on schedule."""
        return await self.repository.get_rules_due_for_execution()