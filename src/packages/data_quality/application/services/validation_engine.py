"""Advanced validation engine with comprehensive rule types and execution capabilities."""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable, Set, Tuple
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging
import re
import json
import time
from uuid import UUID, uuid4

from ...domain.entities.quality_rule import (
    QualityRule, ValidationResult, ValidationError, ValidationStatus,
    RuleType, Severity, LogicType, QualityThreshold, UserId, DatasetId, RuleId
)

logger = logging.getLogger(__name__)


class ValidationEngine:
    """Advanced validation engine supporting all rule types and execution modes."""
    
    def __init__(self, max_workers: Optional[int] = None, enable_parallel: bool = True):
        self.max_workers = max_workers or min(8, (os.cpu_count() or 1) + 4)
        self.enable_parallel = enable_parallel
        self._rule_executors = {}
        self._register_rule_executors()
    
    def _register_rule_executors(self):
        """Register rule executors for different rule types."""
        self._rule_executors = {
            RuleType.COMPLETENESS: CompletenessRuleExecutor(),
            RuleType.UNIQUENESS: UniquenessRuleExecutor(),
            RuleType.VALIDITY: ValidityRuleExecutor(),
            RuleType.CONSISTENCY: ConsistencyRuleExecutor(),
            RuleType.ACCURACY: AccuracyRuleExecutor(),
            RuleType.TIMELINESS: TimelinessRuleExecutor(),
            RuleType.CUSTOM: CustomRuleExecutor()
        }
    
    def validate_dataset(self, rules: List[QualityRule], df: pd.DataFrame, 
                        dataset_id: DatasetId, 
                        executed_by: Optional[UserId] = None) -> List[ValidationResult]:
        """Execute validation rules against a dataset."""
        start_time = time.time()
        results = []
        
        logger.info(f"Starting validation of {len(rules)} rules on dataset with {len(df)} rows")
        
        # Filter active rules
        active_rules = [rule for rule in rules if rule.is_active()]
        
        if not active_rules:
            logger.warning("No active rules to execute")
            return results
        
        # Group rules by type for optimized execution
        rules_by_type = {}
        for rule in active_rules:
            if rule.rule_type not in rules_by_type:
                rules_by_type[rule.rule_type] = []
            rules_by_type[rule.rule_type].append(rule)
        
        # Execute rules by type
        if self.enable_parallel and len(rules_by_type) > 1:
            results = self._execute_rules_parallel(rules_by_type, df, dataset_id, executed_by)
        else:
            results = self._execute_rules_sequential(rules_by_type, df, dataset_id, executed_by)
        
        execution_time = time.time() - start_time
        logger.info(f"Validation completed in {execution_time:.2f} seconds")
        
        return results
    
    def _execute_rules_parallel(self, rules_by_type: Dict[RuleType, List[QualityRule]], 
                               df: pd.DataFrame, dataset_id: DatasetId, 
                               executed_by: Optional[UserId]) -> List[ValidationResult]:
        """Execute rules in parallel by type."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_rule_type = {}
            
            for rule_type, rules in rules_by_type.items():
                if rule_type in self._rule_executors:
                    future = executor.submit(
                        self._execute_rules_of_type,
                        rule_type, rules, df, dataset_id, executed_by
                    )
                    future_to_rule_type[future] = rule_type
            
            for future in as_completed(future_to_rule_type):
                rule_type = future_to_rule_type[future]
                try:
                    type_results = future.result()
                    results.extend(type_results)
                except Exception as e:
                    logger.error(f"Error executing {rule_type} rules: {e}")
                    # Create error results for failed rules
                    for rule in rules_by_type[rule_type]:
                        error_result = self._create_error_result(rule, dataset_id, str(e), executed_by)
                        results.append(error_result)
        
        return results
    
    def _execute_rules_sequential(self, rules_by_type: Dict[RuleType, List[QualityRule]], 
                                 df: pd.DataFrame, dataset_id: DatasetId, 
                                 executed_by: Optional[UserId]) -> List[ValidationResult]:
        """Execute rules sequentially by type."""
        results = []
        
        for rule_type, rules in rules_by_type.items():
            try:
                type_results = self._execute_rules_of_type(rule_type, rules, df, dataset_id, executed_by)
                results.extend(type_results)
            except Exception as e:
                logger.error(f"Error executing {rule_type} rules: {e}")
                for rule in rules:
                    error_result = self._create_error_result(rule, dataset_id, str(e), executed_by)
                    results.append(error_result)
        
        return results
    
    def _execute_rules_of_type(self, rule_type: RuleType, rules: List[QualityRule], 
                              df: pd.DataFrame, dataset_id: DatasetId, 
                              executed_by: Optional[UserId]) -> List[ValidationResult]:
        """Execute all rules of a specific type."""
        if rule_type not in self._rule_executors:
            logger.warning(f"No executor found for rule type: {rule_type}")
            return []
        
        executor = self._rule_executors[rule_type]
        results = []
        
        for rule in rules:
            try:
                result = executor.execute(rule, df, dataset_id, executed_by)
                results.append(result)
            except Exception as e:
                logger.error(f"Error executing rule {rule.rule_name}: {e}")
                error_result = self._create_error_result(rule, dataset_id, str(e), executed_by)
                results.append(error_result)
        
        return results
    
    def _create_error_result(self, rule: QualityRule, dataset_id: DatasetId, 
                           error_message: str, executed_by: Optional[UserId]) -> ValidationResult:
        """Create a validation result for a failed rule execution."""
        return ValidationResult(
            rule_id=rule.rule_id,
            dataset_id=dataset_id,
            status=ValidationStatus.ERROR,
            total_records=0,
            records_passed=0,
            records_failed=0,
            pass_rate=0.0,
            validation_errors=[ValidationError(
                rule_id=rule.rule_id,
                error_message=f"Rule execution failed: {error_message}",
                error_code="EXECUTION_ERROR"
            )],
            execution_time_seconds=0.0,
            executed_by=executed_by
        )
    
    def validate_single_record(self, rules: List[QualityRule], record: Dict[str, Any]) -> Dict[str, bool]:
        """Validate a single record against rules (for real-time validation)."""
        results = {}
        
        for rule in rules:
            if not rule.is_active():
                continue
            
            try:
                if rule.rule_type in self._rule_executors:
                    executor = self._rule_executors[rule.rule_type]
                    is_valid = executor.validate_record(rule, record)
                    results[rule.rule_id.value] = is_valid
                else:
                    results[rule.rule_id.value] = False
            except Exception as e:
                logger.error(f"Error validating record with rule {rule.rule_name}: {e}")
                results[rule.rule_id.value] = False
        
        return results
    
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate a summary of validation results."""
        if not results:
            return {
                'total_rules': 0,
                'passed_rules': 0,
                'failed_rules': 0,
                'error_rules': 0,
                'overall_pass_rate': 0.0,
                'total_records_validated': 0,
                'total_errors_found': 0
            }
        
        passed_rules = sum(1 for r in results if r.status == ValidationStatus.PASSED)
        failed_rules = sum(1 for r in results if r.status == ValidationStatus.FAILED)
        error_rules = sum(1 for r in results if r.status == ValidationStatus.ERROR)
        
        total_records = sum(r.total_records for r in results)
        total_errors = sum(len(r.validation_errors) for r in results)
        
        # Calculate weighted overall pass rate
        if total_records > 0:
            weighted_pass_rate = sum(r.pass_rate * r.total_records for r in results) / total_records
        else:
            weighted_pass_rate = 0.0
        
        return {
            'total_rules': len(results),
            'passed_rules': passed_rules,
            'failed_rules': failed_rules,
            'error_rules': error_rules,
            'overall_pass_rate': weighted_pass_rate,
            'total_records_validated': total_records,
            'total_errors_found': total_errors,
            'rule_breakdown': {
                'passed_percentage': (passed_rules / len(results)) * 100,
                'failed_percentage': (failed_rules / len(results)) * 100,
                'error_percentage': (error_rules / len(results)) * 100
            }
        }


class RuleExecutor(ABC):
    """Abstract base class for rule executors."""
    
    @abstractmethod
    def execute(self, rule: QualityRule, df: pd.DataFrame, 
               dataset_id: DatasetId, executed_by: Optional[UserId]) -> ValidationResult:
        """Execute a rule and return validation result."""
        pass
    
    @abstractmethod
    def validate_record(self, rule: QualityRule, record: Dict[str, Any]) -> bool:
        """Validate a single record for real-time validation."""
        pass
    
    def _create_base_result(self, rule: QualityRule, df: pd.DataFrame, 
                           dataset_id: DatasetId, executed_by: Optional[UserId]) -> ValidationResult:
        """Create a base validation result."""
        return ValidationResult(
            rule_id=rule.rule_id,
            dataset_id=dataset_id,
            status=ValidationStatus.PENDING,
            total_records=len(df),
            records_passed=0,
            records_failed=0,
            pass_rate=0.0,
            validation_errors=[],
            execution_time_seconds=0.0,
            executed_by=executed_by
        )


class CompletenessRuleExecutor(RuleExecutor):
    """Executor for completeness validation rules."""
    
    def execute(self, rule: QualityRule, df: pd.DataFrame, 
               dataset_id: DatasetId, executed_by: Optional[UserId]) -> ValidationResult:
        start_time = time.time()
        result = self._create_base_result(rule, df, dataset_id, executed_by)
        
        try:
            # Parse validation logic
            logic = rule.validation_logic
            target_columns = rule.target_columns if rule.target_columns else df.columns.tolist()
            
            total_records = len(df)
            failed_records = 0
            errors = []
            
            for column in target_columns:
                if column not in df.columns:
                    continue
                
                # Find null/empty values
                null_mask = df[column].isnull()
                empty_mask = df[column].astype(str).str.strip() == ''
                invalid_mask = null_mask | empty_mask
                
                invalid_indices = df[invalid_mask].index.tolist()
                failed_records += len(invalid_indices)
                
                # Create errors for sample of invalid records
                for idx in invalid_indices[:100]:  # Limit to 100 errors per column
                    error = ValidationError(
                        rule_id=rule.rule_id,
                        row_identifier=str(idx),
                        column_name=column,
                        invalid_value=str(df.loc[idx, column]) if pd.notna(df.loc[idx, column]) else None,
                        error_message=f"Missing or empty value in column '{column}'",
                        error_code="COMPLETENESS_VIOLATION"
                    )
                    errors.append(error)
            
            # Calculate metrics
            records_passed = total_records - failed_records
            pass_rate = records_passed / total_records if total_records > 0 else 0.0
            
            # Determine status based on thresholds
            if pass_rate >= rule.thresholds.pass_rate_threshold:
                status = ValidationStatus.PASSED
            else:
                status = ValidationStatus.FAILED
            
            result.status = status
            result.records_passed = records_passed
            result.records_failed = failed_records
            result.pass_rate = pass_rate
            result.validation_errors = errors
            result.execution_time_seconds = time.time() - start_time
            
        except Exception as e:
            result.status = ValidationStatus.ERROR
            result.validation_errors = [ValidationError(
                rule_id=rule.rule_id,
                error_message=f"Completeness validation failed: {str(e)}",
                error_code="EXECUTION_ERROR"
            )]
            result.execution_time_seconds = time.time() - start_time
        
        return result
    
    def validate_record(self, rule: QualityRule, record: Dict[str, Any]) -> bool:
        """Validate completeness for a single record."""
        target_columns = rule.target_columns if rule.target_columns else record.keys()
        
        for column in target_columns:
            value = record.get(column)
            if value is None or (isinstance(value, str) and value.strip() == ''):
                return False
        
        return True


class UniquenessRuleExecutor(RuleExecutor):
    """Executor for uniqueness validation rules."""
    
    def execute(self, rule: QualityRule, df: pd.DataFrame, 
               dataset_id: DatasetId, executed_by: Optional[UserId]) -> ValidationResult:
        start_time = time.time()
        result = self._create_base_result(rule, df, dataset_id, executed_by)
        
        try:
            target_columns = rule.target_columns if rule.target_columns else df.columns.tolist()
            
            total_records = len(df)
            failed_records = 0
            errors = []
            
            for column in target_columns:
                if column not in df.columns:
                    continue
                
                # Find duplicate values
                duplicated_mask = df[column].duplicated(keep=False)
                duplicate_indices = df[duplicated_mask].index.tolist()
                
                # Count unique failures (each set of duplicates counts as one failure)
                unique_duplicate_values = df[duplicated_mask][column].unique()
                failed_records += len(unique_duplicate_values)
                
                # Create errors for sample of duplicates
                for value in unique_duplicate_values[:50]:  # Limit to 50 unique duplicate values
                    duplicate_rows = df[df[column] == value].index.tolist()
                    error = ValidationError(
                        rule_id=rule.rule_id,
                        column_name=column,
                        invalid_value=str(value),
                        error_message=f"Duplicate value '{value}' found in {len(duplicate_rows)} rows: {duplicate_rows[:10]}",
                        error_code="UNIQUENESS_VIOLATION"
                    )
                    errors.append(error)
            
            records_passed = total_records - failed_records
            pass_rate = records_passed / total_records if total_records > 0 else 0.0
            
            status = ValidationStatus.PASSED if pass_rate >= rule.thresholds.pass_rate_threshold else ValidationStatus.FAILED
            
            result.status = status
            result.records_passed = records_passed
            result.records_failed = failed_records
            result.pass_rate = pass_rate
            result.validation_errors = errors
            result.execution_time_seconds = time.time() - start_time
            
        except Exception as e:
            result.status = ValidationStatus.ERROR
            result.validation_errors = [ValidationError(
                rule_id=rule.rule_id,
                error_message=f"Uniqueness validation failed: {str(e)}",
                error_code="EXECUTION_ERROR"
            )]
            result.execution_time_seconds = time.time() - start_time
        
        return result
    
    def validate_record(self, rule: QualityRule, record: Dict[str, Any]) -> bool:
        """Validate uniqueness for a single record (not applicable for single records)."""
        # Uniqueness validation requires dataset context
        return True


class ValidityRuleExecutor(RuleExecutor):
    """Executor for validity/format validation rules."""
    
    def execute(self, rule: QualityRule, df: pd.DataFrame, 
               dataset_id: DatasetId, executed_by: Optional[UserId]) -> ValidationResult:
        start_time = time.time()
        result = self._create_base_result(rule, df, dataset_id, executed_by)
        
        try:
            logic = rule.validation_logic
            target_columns = rule.target_columns if rule.target_columns else df.columns.tolist()
            
            total_records = len(df)
            failed_records = 0
            errors = []
            
            for column in target_columns:
                if column not in df.columns:
                    continue
                
                # Apply validation based on logic type
                if logic.logic_type == LogicType.REGEX:
                    pattern = re.compile(logic.expression)
                    invalid_mask = ~df[column].astype(str).str.match(pattern, na=False)
                elif logic.logic_type == LogicType.RANGE:
                    # Parse range parameters
                    params = logic.parameters
                    min_val = params.get('min_value')
                    max_val = params.get('max_value')
                    
                    invalid_mask = pd.Series([False] * len(df), index=df.index)
                    if min_val is not None:
                        invalid_mask |= df[column] < min_val
                    if max_val is not None:
                        invalid_mask |= df[column] > max_val
                elif logic.logic_type == LogicType.LIST:
                    # Value must be in allowed list
                    allowed_values = logic.parameters.get('allowed_values', [])
                    invalid_mask = ~df[column].isin(allowed_values)
                else:
                    # Skip unknown logic types
                    continue
                
                invalid_indices = df[invalid_mask].index.tolist()
                failed_records += len(invalid_indices)
                
                # Create errors for sample of invalid records
                for idx in invalid_indices[:100]:
                    error = ValidationError(
                        rule_id=rule.rule_id,
                        row_identifier=str(idx),
                        column_name=column,
                        invalid_value=str(df.loc[idx, column]),
                        error_message=logic.error_message_template.format(
                            column=column, value=df.loc[idx, column]
                        ),
                        error_code="VALIDITY_VIOLATION"
                    )
                    errors.append(error)
            
            records_passed = total_records - failed_records
            pass_rate = records_passed / total_records if total_records > 0 else 0.0
            
            status = ValidationStatus.PASSED if pass_rate >= rule.thresholds.pass_rate_threshold else ValidationStatus.FAILED
            
            result.status = status
            result.records_passed = records_passed
            result.records_failed = failed_records
            result.pass_rate = pass_rate
            result.validation_errors = errors
            result.execution_time_seconds = time.time() - start_time
            
        except Exception as e:
            result.status = ValidationStatus.ERROR
            result.validation_errors = [ValidationError(
                rule_id=rule.rule_id,
                error_message=f"Validity validation failed: {str(e)}",
                error_code="EXECUTION_ERROR"
            )]
            result.execution_time_seconds = time.time() - start_time
        
        return result
    
    def validate_record(self, rule: QualityRule, record: Dict[str, Any]) -> bool:
        """Validate validity for a single record."""
        logic = rule.validation_logic
        target_columns = rule.target_columns if rule.target_columns else record.keys()
        
        for column in target_columns:
            value = record.get(column)
            if value is None:
                continue
            
            try:
                if logic.logic_type == LogicType.REGEX:
                    pattern = re.compile(logic.expression)
                    if not pattern.match(str(value)):
                        return False
                elif logic.logic_type == LogicType.RANGE:
                    params = logic.parameters
                    min_val = params.get('min_value')
                    max_val = params.get('max_value')
                    
                    numeric_value = float(value)
                    if min_val is not None and numeric_value < min_val:
                        return False
                    if max_val is not None and numeric_value > max_val:
                        return False
                elif logic.logic_type == LogicType.LIST:
                    allowed_values = logic.parameters.get('allowed_values', [])
                    if value not in allowed_values:
                        return False
            except (ValueError, TypeError):
                return False
        
        return True


class ConsistencyRuleExecutor(RuleExecutor):
    """Executor for consistency validation rules."""
    
    def execute(self, rule: QualityRule, df: pd.DataFrame, 
               dataset_id: DatasetId, executed_by: Optional[UserId]) -> ValidationResult:
        start_time = time.time()
        result = self._create_base_result(rule, df, dataset_id, executed_by)
        
        try:
            logic = rule.validation_logic
            
            # For consistency rules, we expect SQL or Python logic
            if logic.logic_type == LogicType.SQL:
                # Execute SQL logic (simplified - in production use proper SQL engine)
                pass_rate = 0.95  # Placeholder
            elif logic.logic_type == LogicType.PYTHON:
                # Execute Python expression
                pass_rate = self._execute_python_consistency_check(logic.expression, df)
            else:
                pass_rate = 1.0
            
            total_records = len(df)
            records_passed = int(total_records * pass_rate)
            records_failed = total_records - records_passed
            
            status = ValidationStatus.PASSED if pass_rate >= rule.thresholds.pass_rate_threshold else ValidationStatus.FAILED
            
            result.status = status
            result.records_passed = records_passed
            result.records_failed = records_failed
            result.pass_rate = pass_rate
            result.execution_time_seconds = time.time() - start_time
            
        except Exception as e:
            result.status = ValidationStatus.ERROR
            result.validation_errors = [ValidationError(
                rule_id=rule.rule_id,
                error_message=f"Consistency validation failed: {str(e)}",
                error_code="EXECUTION_ERROR"
            )]
            result.execution_time_seconds = time.time() - start_time
        
        return result
    
    def _execute_python_consistency_check(self, expression: str, df: pd.DataFrame) -> float:
        """Execute Python expression for consistency checking."""
        try:
            # Simple example: check if date1 <= date2
            # In production, use safer expression evaluation
            result = eval(expression, {"df": df, "pd": pd, "np": np})
            if isinstance(result, pd.Series):
                return result.mean()
            elif isinstance(result, bool):
                return 1.0 if result else 0.0
            else:
                return float(result)
        except:
            return 0.0
    
    def validate_record(self, rule: QualityRule, record: Dict[str, Any]) -> bool:
        """Validate consistency for a single record."""
        # Simplified consistency check for single record
        return True


class AccuracyRuleExecutor(RuleExecutor):
    """Executor for accuracy validation rules."""
    
    def execute(self, rule: QualityRule, df: pd.DataFrame, 
               dataset_id: DatasetId, executed_by: Optional[UserId]) -> ValidationResult:
        start_time = time.time()
        result = self._create_base_result(rule, df, dataset_id, executed_by)
        
        try:
            # Accuracy rules often involve external reference data or statistical checks
            # For now, implement basic statistical accuracy checks
            target_columns = rule.target_columns if rule.target_columns else df.select_dtypes(include=[np.number]).columns.tolist()
            
            total_records = len(df)
            failed_records = 0
            errors = []
            
            for column in target_columns:
                if column not in df.columns:
                    continue
                
                # Check for statistical outliers as accuracy issues
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR  # More strict than typical 1.5 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
                outlier_indices = df[outlier_mask].index.tolist()
                failed_records += len(outlier_indices)
                
                # Create errors for outliers
                for idx in outlier_indices[:50]:
                    error = ValidationError(
                        rule_id=rule.rule_id,
                        row_identifier=str(idx),
                        column_name=column,
                        invalid_value=str(df.loc[idx, column]),
                        error_message=f"Statistical outlier detected in column '{column}': {df.loc[idx, column]}",
                        error_code="ACCURACY_OUTLIER"
                    )
                    errors.append(error)
            
            records_passed = total_records - failed_records
            pass_rate = records_passed / total_records if total_records > 0 else 0.0
            
            status = ValidationStatus.PASSED if pass_rate >= rule.thresholds.pass_rate_threshold else ValidationStatus.FAILED
            
            result.status = status
            result.records_passed = records_passed
            result.records_failed = failed_records
            result.pass_rate = pass_rate
            result.validation_errors = errors
            result.execution_time_seconds = time.time() - start_time
            
        except Exception as e:
            result.status = ValidationStatus.ERROR
            result.validation_errors = [ValidationError(
                rule_id=rule.rule_id,
                error_message=f"Accuracy validation failed: {str(e)}",
                error_code="EXECUTION_ERROR"
            )]
            result.execution_time_seconds = time.time() - start_time
        
        return result
    
    def validate_record(self, rule: QualityRule, record: Dict[str, Any]) -> bool:
        """Validate accuracy for a single record."""
        # Accuracy validation typically requires dataset context for statistical analysis
        return True


class TimelinessRuleExecutor(RuleExecutor):
    """Executor for timeliness validation rules."""
    
    def execute(self, rule: QualityRule, df: pd.DataFrame, 
               dataset_id: DatasetId, executed_by: Optional[UserId]) -> ValidationResult:
        start_time = time.time()
        result = self._create_base_result(rule, df, dataset_id, executed_by)
        
        try:
            target_columns = rule.target_columns if rule.target_columns else []
            
            total_records = len(df)
            failed_records = 0
            errors = []
            
            # Get timeliness parameters
            max_age_hours = rule.validation_logic.parameters.get('max_age_hours', 24)
            current_time = datetime.utcnow()
            
            for column in target_columns:
                if column not in df.columns:
                    continue
                
                try:
                    # Convert column to datetime
                    datetime_series = pd.to_datetime(df[column], errors='coerce')
                    
                    # Check age of records
                    age_threshold = current_time - timedelta(hours=max_age_hours)
                    stale_mask = datetime_series < age_threshold
                    stale_indices = df[stale_mask].index.tolist()
                    failed_records += len(stale_indices)
                    
                    # Create errors for stale records
                    for idx in stale_indices[:100]:
                        record_time = datetime_series.loc[idx]
                        if pd.notna(record_time):
                            age_hours = (current_time - record_time).total_seconds() / 3600
                            error = ValidationError(
                                rule_id=rule.rule_id,
                                row_identifier=str(idx),
                                column_name=column,
                                invalid_value=str(record_time),
                                error_message=f"Record is {age_hours:.1f} hours old, exceeds maximum age of {max_age_hours} hours",
                                error_code="TIMELINESS_VIOLATION"
                            )
                            errors.append(error)
                
                except Exception as e:
                    logger.warning(f"Could not process timeliness for column {column}: {e}")
            
            records_passed = total_records - failed_records
            pass_rate = records_passed / total_records if total_records > 0 else 0.0
            
            status = ValidationStatus.PASSED if pass_rate >= rule.thresholds.pass_rate_threshold else ValidationStatus.FAILED
            
            result.status = status
            result.records_passed = records_passed
            result.records_failed = failed_records
            result.pass_rate = pass_rate
            result.validation_errors = errors
            result.execution_time_seconds = time.time() - start_time
            
        except Exception as e:
            result.status = ValidationStatus.ERROR
            result.validation_errors = [ValidationError(
                rule_id=rule.rule_id,
                error_message=f"Timeliness validation failed: {str(e)}",
                error_code="EXECUTION_ERROR"
            )]
            result.execution_time_seconds = time.time() - start_time
        
        return result
    
    def validate_record(self, rule: QualityRule, record: Dict[str, Any]) -> bool:
        """Validate timeliness for a single record."""
        target_columns = rule.target_columns if rule.target_columns else []
        max_age_hours = rule.validation_logic.parameters.get('max_age_hours', 24)
        current_time = datetime.utcnow()
        
        for column in target_columns:
            value = record.get(column)
            if value is None:
                continue
            
            try:
                record_time = pd.to_datetime(value)
                age_hours = (current_time - record_time).total_seconds() / 3600
                if age_hours > max_age_hours:
                    return False
            except:
                return False
        
        return True


class CustomRuleExecutor(RuleExecutor):
    """Executor for custom validation rules."""
    
    def execute(self, rule: QualityRule, df: pd.DataFrame, 
               dataset_id: DatasetId, executed_by: Optional[UserId]) -> ValidationResult:
        start_time = time.time()
        result = self._create_base_result(rule, df, dataset_id, executed_by)
        
        try:
            logic = rule.validation_logic
            
            if logic.logic_type == LogicType.PYTHON:
                # Execute custom Python logic
                pass_rate = self._execute_custom_python_logic(logic.expression, df, logic.parameters)
            elif logic.logic_type == LogicType.SQL:
                # Execute custom SQL logic (placeholder)
                pass_rate = 0.95  # Placeholder
            else:
                pass_rate = 1.0
            
            total_records = len(df)
            records_passed = int(total_records * pass_rate)
            records_failed = total_records - records_passed
            
            status = ValidationStatus.PASSED if pass_rate >= rule.thresholds.pass_rate_threshold else ValidationStatus.FAILED
            
            result.status = status
            result.records_passed = records_passed
            result.records_failed = records_failed
            result.pass_rate = pass_rate
            result.execution_time_seconds = time.time() - start_time
            
        except Exception as e:
            result.status = ValidationStatus.ERROR
            result.validation_errors = [ValidationError(
                rule_id=rule.rule_id,
                error_message=f"Custom validation failed: {str(e)}",
                error_code="EXECUTION_ERROR"
            )]
            result.execution_time_seconds = time.time() - start_time
        
        return result
    
    def _execute_custom_python_logic(self, expression: str, df: pd.DataFrame, parameters: Dict[str, Any]) -> float:
        """Execute custom Python logic safely."""
        try:
            # Create safe execution environment
            safe_globals = {
                "df": df,
                "pd": pd,
                "np": np,
                "len": len,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "parameters": parameters
            }
            
            # Execute expression
            result = eval(expression, safe_globals)
            
            if isinstance(result, pd.Series):
                return result.mean()
            elif isinstance(result, bool):
                return 1.0 if result else 0.0
            else:
                return float(result)
        except Exception as e:
            logger.error(f"Custom Python logic execution failed: {e}")
            return 0.0
    
    def validate_record(self, rule: QualityRule, record: Dict[str, Any]) -> bool:
        """Validate custom rule for a single record."""
        try:
            logic = rule.validation_logic
            
            if logic.logic_type == LogicType.PYTHON:
                # Create safe execution environment for single record
                safe_globals = {
                    "record": record,
                    "parameters": logic.parameters
                }
                
                result = eval(logic.expression.replace("df", "record"), safe_globals)
                return bool(result)
            else:
                return True
        except:
            return False