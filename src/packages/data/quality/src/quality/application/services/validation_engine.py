"""Comprehensive validation engine for data quality rules."""

import re
import time
import traceback
import ast
import operator
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ...domain.entities.quality_entity import (
    QualityRule, ValidationResult, ValidationStatus, ValidationError,
    LogicType, Severity, DatasetId, ValidationId, ValidationLogic
)
from ...infrastructure.config.quality_config import ValidationEngineConfig
from ...infrastructure.logging.quality_logger import get_logger

logger = get_logger(__name__)


class ValidationEngine:
    """High-performance validation engine with caching and parallel processing."""
    
    def __init__(self, config: ValidationEngineConfig):
        """Initialize validation engine."""
        self.config = config
        
        # Caching
        self._cache: Optional[Dict[str, ValidationResult]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._compiled_rules_cache: Dict[int, Any] = {}
        
        # Metrics
        self._execution_metrics = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_execution_time': 0.0
        }
        
        # Rule executors
        self._rule_executors = {
            LogicType.PYTHON: self._execute_python_rule,
            LogicType.SQL: self._execute_sql_rule,
            LogicType.REGEX: self._execute_regex_rule,
            LogicType.STATISTICAL: self._execute_statistical_rule,
            LogicType.COMPARISON: self._execute_comparison_rule,
            LogicType.AGGREGATION: self._execute_aggregation_rule,
            LogicType.LOOKUP: self._execute_lookup_rule,
            LogicType.CONDITIONAL: self._execute_conditional_rule,
            LogicType.EXPRESSION: self._execute_expression_rule
        }
    
    def validate_dataset(self, 
                        df: pd.DataFrame,
                        rules: List[QualityRule],
                        dataset_id: DatasetId) -> List[ValidationResult]:
        """Validate dataset against quality rules."""
        start_time = time.time()
        
        try:
            # Apply sampling if enabled
            if self.config.enable_sampling and len(df) > self.config.sample_size:
                df = self._apply_sampling(df)
                logger.info(f"Applied sampling: {len(df)} rows")
            
            # Filter applicable rules
            applicable_rules = self._filter_applicable_rules(df, rules)
            logger.info(f"Executing {len(applicable_rules)} applicable rules")
            
            # Execute validations
            if self.config.enable_parallel_processing and len(applicable_rules) > 1:
                results = self._execute_parallel_validation(df, applicable_rules, dataset_id)
            else:
                results = self._execute_sequential_validation(df, applicable_rules, dataset_id)
            
            # Update metrics
            execution_time = time.time() - start_time
            self._update_execution_metrics(results, execution_time)
            
            logger.info(f"Validation completed in {execution_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def validate_single_rule(self, 
                           df: pd.DataFrame,
                           rule: QualityRule,
                           dataset_id: DatasetId) -> ValidationResult:
        """Validate dataset against a single rule."""
        start_time = time.time()
        
        try:
            # Check cache first
            if self._cache and self._should_use_cache(df, rule):
                cached_result = self._get_from_cache(df, rule)
                if cached_result:
                    self._execution_metrics['cache_hits'] += 1
                    return cached_result
            
            self._execution_metrics['cache_misses'] += 1
            
            # Execute validation
            result = self._execute_single_rule(df, rule, dataset_id)
            
            # Cache result
            if self._cache:
                self._cache_result(df, rule, result)
            
            execution_time = time.time() - start_time
            logger.debug(f"Rule {rule.rule_name} executed in {execution_time:.3f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"Rule validation failed for {rule.rule_name}: {str(e)}")
            return self._create_error_result(rule, dataset_id, str(e))
    
    def _execute_single_rule(self, 
                           df: pd.DataFrame,
                           rule: QualityRule,
                           dataset_id: DatasetId) -> ValidationResult:
        """Execute a single validation rule."""
        validation_id = ValidationId()
        start_time = datetime.now()
        
        try:
            # Get rule executor
            executor = self._rule_executors.get(rule.validation_logic.logic_type)
            if not executor:
                raise ValueError(f"No executor for logic type: {rule.validation_logic.logic_type}")
            
            # Execute rule
            passed_records, failed_records, error_details = executor(df, rule.validation_logic)
            
            # Calculate metrics
            total_records = len(df)
            failure_rate = failed_records / total_records if total_records > 0 else 0.0
            
            # Determine status
            success_criteria = rule.validation_logic.success_criteria
            pass_rate = passed_records / total_records if total_records > 0 else 0.0
            
            if pass_rate >= success_criteria.min_pass_rate:
                if success_criteria.max_failure_count is None or failed_records <= success_criteria.max_failure_count:
                    status = ValidationStatus.PASSED
                else:
                    status = ValidationStatus.FAILED
            elif pass_rate >= success_criteria.warning_threshold:
                status = ValidationStatus.WARNING
            else:
                status = ValidationStatus.FAILED
            
            execution_time = datetime.now() - start_time
            
            return ValidationResult(
                validation_id=validation_id,
                rule_id=rule.rule_id,
                dataset_id=dataset_id,
                status=status,
                passed_records=passed_records,
                failed_records=failed_records,
                failure_rate=failure_rate,
                error_details=error_details,
                execution_time=execution_time,
                validated_at=datetime.now(),
                total_records=total_records
            )
            
        except Exception as e:
            logger.error(f"Rule execution failed: {str(e)}")
            return self._create_error_result(rule, dataset_id, str(e))
    
    def _execute_python_rule(self, 
                           df: pd.DataFrame,
                           logic: ValidationLogic) -> tuple[int, int, List[ValidationError]]:
        """Execute Python-based validation rule."""
        try:
            # Compile Python expression
            compiled_expr = self._compile_python_rule(logic)
            
            # Execute validation
            if logic.get_parameter('row_wise', False):
                # Row-wise validation
                return self._execute_row_wise_python(df, compiled_expr, logic)
            else:
                # Column-wise or aggregate validation
                return self._execute_aggregate_python(df, compiled_expr, logic)
                
        except Exception as e:
            logger.error(f"Python rule execution failed: {str(e)}")
            return 0, len(df), [ValidationError(error_message=str(e))]
    
    def _execute_sql_rule(self, 
                        df: pd.DataFrame,
                        logic: ValidationLogic) -> tuple[int, int, List[ValidationError]]:
        """Execute SQL-based validation rule."""
        try:
            # Convert SQL to pandas operations
            # This is a simplified implementation
            sql_expr = logic.expression.lower()
            
            if 'select' in sql_expr and 'where' in sql_expr:
                # Extract WHERE clause for filtering
                where_clause = self._extract_where_clause(sql_expr)
                valid_mask = self._evaluate_sql_condition(df, where_clause)
                
                passed_records = valid_mask.sum()
                failed_records = len(df) - passed_records
                
                # Create error details for failed records
                error_details = []
                if failed_records > 0:
                    failed_indices = df[~valid_mask].index.tolist()
                    for idx in failed_indices[:100]:  # Limit error details
                        error_details.append(ValidationError(
                            row_index=idx,
                            error_message=logic.error_message,
                            severity=Severity.MEDIUM
                        ))
                
                return passed_records, failed_records, error_details
            else:
                raise ValueError("Invalid SQL expression format")
                
        except Exception as e:
            logger.error(f"SQL rule execution failed: {str(e)}")
            return 0, len(df), [ValidationError(error_message=str(e))]
    
    def _execute_regex_rule(self, 
                          df: pd.DataFrame,
                          logic: ValidationLogic) -> tuple[int, int, List[ValidationError]]:
        """Execute regex-based validation rule."""
        try:
            pattern = logic.expression
            column_name = logic.get_parameter('column_name')
            
            if not column_name or column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found in dataset")
            
            # Compile regex pattern
            compiled_pattern = re.compile(pattern)
            
            # Apply regex validation
            series = df[column_name].astype(str)
            valid_mask = series.str.match(compiled_pattern, na=False)
            
            passed_records = valid_mask.sum()
            failed_records = len(df) - passed_records
            
            # Create error details for failed records
            error_details = []
            if failed_records > 0:
                failed_data = df[~valid_mask]
                for idx, row in failed_data.head(100).iterrows():
                    error_details.append(ValidationError(
                        row_index=idx,
                        column_name=column_name,
                        field_value=str(row[column_name]),
                        error_message=f"Value does not match pattern: {pattern}",
                        severity=Severity.MEDIUM
                    ))
            
            return passed_records, failed_records, error_details
            
        except Exception as e:
            logger.error(f"Regex rule execution failed: {str(e)}")
            return 0, len(df), [ValidationError(error_message=str(e))]
    
    def _execute_statistical_rule(self, 
                                df: pd.DataFrame,
                                logic: ValidationLogic) -> tuple[int, int, List[ValidationError]]:
        """Execute statistical validation rule."""
        try:
            column_name = logic.get_parameter('column_name')
            stat_type = logic.get_parameter('stat_type', 'outlier')
            
            if not column_name or column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found in dataset")
            
            series = df[column_name]
            
            if stat_type == 'outlier':
                # IQR-based outlier detection
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                valid_mask = (series >= lower_bound) & (series <= upper_bound)
                
            elif stat_type == 'z_score':
                # Z-score based outlier detection
                threshold = logic.get_parameter('threshold', 3.0)
                z_scores = np.abs((series - series.mean()) / series.std())
                valid_mask = z_scores <= threshold
                
            elif stat_type == 'range':
                # Range validation
                min_val = logic.get_parameter('min_value')
                max_val = logic.get_parameter('max_value')
                valid_mask = (series >= min_val) & (series <= max_val)
                
            else:
                raise ValueError(f"Unknown statistical validation type: {stat_type}")
            
            passed_records = valid_mask.sum()
            failed_records = len(df) - passed_records
            
            # Create error details for failed records
            error_details = []
            if failed_records > 0:
                failed_data = df[~valid_mask]
                for idx, row in failed_data.head(100).iterrows():
                    error_details.append(ValidationError(
                        row_index=idx,
                        column_name=column_name,
                        field_value=str(row[column_name]),
                        error_message=f"Statistical validation failed: {stat_type}",
                        severity=Severity.MEDIUM
                    ))
            
            return passed_records, failed_records, error_details
            
        except Exception as e:
            logger.error(f"Statistical rule execution failed: {str(e)}")
            return 0, len(df), [ValidationError(error_message=str(e))]
    
    def _execute_comparison_rule(self, 
                               df: pd.DataFrame,
                               logic: ValidationLogic) -> tuple[int, int, List[ValidationError]]:
        """Execute comparison validation rule."""
        try:
            column1 = logic.get_parameter('column1')
            column2 = logic.get_parameter('column2')
            operator = logic.get_parameter('operator', '==')
            
            if not column1 or column1 not in df.columns:
                raise ValueError(f"Column '{column1}' not found in dataset")
            if not column2 or column2 not in df.columns:
                raise ValueError(f"Column '{column2}' not found in dataset")
            
            # Perform comparison
            series1 = df[column1]
            series2 = df[column2]
            
            if operator == '==':
                valid_mask = series1 == series2
            elif operator == '!=':
                valid_mask = series1 != series2
            elif operator == '<':
                valid_mask = series1 < series2
            elif operator == '<=':
                valid_mask = series1 <= series2
            elif operator == '>':
                valid_mask = series1 > series2
            elif operator == '>=':
                valid_mask = series1 >= series2
            else:
                raise ValueError(f"Unknown comparison operator: {operator}")
            
            passed_records = valid_mask.sum()
            failed_records = len(df) - passed_records
            
            # Create error details for failed records
            error_details = []
            if failed_records > 0:
                failed_data = df[~valid_mask]
                for idx, row in failed_data.head(100).iterrows():
                    error_details.append(ValidationError(
                        row_index=idx,
                        column_name=f"{column1}, {column2}",
                        field_value=f"{row[column1]} {operator} {row[column2]}",
                        error_message=f"Comparison validation failed: {column1} {operator} {column2}",
                        severity=Severity.MEDIUM
                    ))
            
            return passed_records, failed_records, error_details
            
        except Exception as e:
            logger.error(f"Comparison rule execution failed: {str(e)}")
            return 0, len(df), [ValidationError(error_message=str(e))]
    
    def _execute_aggregation_rule(self, 
                                df: pd.DataFrame,
                                logic: ValidationLogic) -> tuple[int, int, List[ValidationError]]:
        """Execute aggregation validation rule."""
        try:
            column_name = logic.get_parameter('column_name')
            agg_function = logic.get_parameter('function', 'sum')
            expected_value = logic.get_parameter('expected_value')
            tolerance = logic.get_parameter('tolerance', 0.0)
            
            if not column_name or column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found in dataset")
            
            # Calculate aggregation
            series = df[column_name]
            
            if agg_function == 'sum':
                actual_value = series.sum()
            elif agg_function == 'mean':
                actual_value = series.mean()
            elif agg_function == 'count':
                actual_value = series.count()
            elif agg_function == 'max':
                actual_value = series.max()
            elif agg_function == 'min':
                actual_value = series.min()
            else:
                raise ValueError(f"Unknown aggregation function: {agg_function}")
            
            # Check if within tolerance
            if abs(actual_value - expected_value) <= tolerance:
                return len(df), 0, []
            else:
                error_details = [ValidationError(
                    error_message=f"Aggregation validation failed: {agg_function}({column_name}) = {actual_value}, expected {expected_value} ± {tolerance}",
                    severity=Severity.HIGH
                )]
                return 0, len(df), error_details
            
        except Exception as e:
            logger.error(f"Aggregation rule execution failed: {str(e)}")
            return 0, len(df), [ValidationError(error_message=str(e))]
    
    def _execute_lookup_rule(self, 
                           df: pd.DataFrame,
                           logic: ValidationLogic) -> tuple[int, int, List[ValidationError]]:
        """Execute lookup validation rule."""
        try:
            column_name = logic.get_parameter('column_name')
            lookup_values = logic.get_parameter('lookup_values', [])
            
            if not column_name or column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found in dataset")
            
            if not lookup_values:
                raise ValueError("Lookup values not provided")
            
            # Perform lookup validation
            series = df[column_name]
            valid_mask = series.isin(lookup_values)
            
            passed_records = valid_mask.sum()
            failed_records = len(df) - passed_records
            
            # Create error details for failed records
            error_details = []
            if failed_records > 0:
                failed_data = df[~valid_mask]
                for idx, row in failed_data.head(100).iterrows():
                    error_details.append(ValidationError(
                        row_index=idx,
                        column_name=column_name,
                        field_value=str(row[column_name]),
                        error_message=f"Value not found in lookup list: {lookup_values}",
                        severity=Severity.MEDIUM
                    ))
            
            return passed_records, failed_records, error_details
            
        except Exception as e:
            logger.error(f"Lookup rule execution failed: {str(e)}")
            return 0, len(df), [ValidationError(error_message=str(e))]
    
    def _execute_conditional_rule(self, 
                                df: pd.DataFrame,
                                logic: ValidationLogic) -> tuple[int, int, List[ValidationError]]:
        """Execute conditional validation rule."""
        try:
            condition = logic.get_parameter('condition')
            validation_expr = logic.get_parameter('validation_expression')
            
            if not condition or not validation_expr:
                raise ValueError("Condition and validation expression are required")
            
            # Evaluate condition
            condition_mask = self._evaluate_expression(df, condition)
            
            # Apply validation only to records that meet the condition
            filtered_df = df[condition_mask]
            
            if len(filtered_df) == 0:
                return len(df), 0, []  # No records match condition
            
            # Execute validation on filtered data
            validation_mask = self._evaluate_expression(filtered_df, validation_expr)
            
            passed_records = validation_mask.sum()
            failed_records = len(filtered_df) - passed_records
            
            # Create error details for failed records
            error_details = []
            if failed_records > 0:
                failed_data = filtered_df[~validation_mask]
                for idx, row in failed_data.head(100).iterrows():
                    error_details.append(ValidationError(
                        row_index=idx,
                        error_message=f"Conditional validation failed: {validation_expr}",
                        severity=Severity.MEDIUM
                    ))
            
            # Return results relative to full dataset
            total_passed = len(df) - len(filtered_df) + passed_records
            total_failed = failed_records
            
            return total_passed, total_failed, error_details
            
        except Exception as e:
            logger.error(f"Conditional rule execution failed: {str(e)}")
            return 0, len(df), [ValidationError(error_message=str(e))]
    
    def _execute_expression_rule(self, 
                               df: pd.DataFrame,
                               logic: ValidationLogic) -> tuple[int, int, List[ValidationError]]:
        """Execute expression-based validation rule."""
        try:
            expression = logic.expression
            
            # Evaluate expression
            valid_mask = self._evaluate_expression(df, expression)
            
            passed_records = valid_mask.sum()
            failed_records = len(df) - passed_records
            
            # Create error details for failed records
            error_details = []
            if failed_records > 0:
                failed_data = df[~valid_mask]
                for idx, row in failed_data.head(100).iterrows():
                    error_details.append(ValidationError(
                        row_index=idx,
                        error_message=f"Expression validation failed: {expression}",
                        severity=Severity.MEDIUM
                    ))
            
            return passed_records, failed_records, error_details
            
        except Exception as e:
            logger.error(f"Expression rule execution failed: {str(e)}")
            return 0, len(df), [ValidationError(error_message=str(e))]
    
    def _evaluate_expression(self, df: pd.DataFrame, expression: str) -> pd.Series:
        """Safely evaluate pandas expression using pandas.eval()."""
        try:
            # Use pandas.eval() which is safer than builtin eval()
            # It only allows mathematical/logical operations on DataFrame columns
            return pd.eval(expression, local_dict={'df': df}, global_dict=df.to_dict('series'))
            
        except Exception as e:
            logger.error(f"Expression evaluation failed: {str(e)}")
            return pd.Series([False] * len(df))
    
    def _apply_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply sampling to large datasets."""
        return df.sample(
            n=min(self.config.sample_size, len(df)),
            random_state=self.config.sample_random_state
        )
    
    def _filter_applicable_rules(self, df: pd.DataFrame, rules: List[QualityRule]) -> List[QualityRule]:
        """Filter rules that are applicable to the dataset."""
        applicable_rules = []
        
        for rule in rules:
            if not rule.is_active:
                continue
            
            # Check if rule applies to any columns in the dataset
            if rule.target_columns:
                if not any(col in df.columns for col in rule.target_columns):
                    continue
            
            applicable_rules.append(rule)
        
        return applicable_rules
    
    def _execute_parallel_validation(self, 
                                   df: pd.DataFrame,
                                   rules: List[QualityRule],
                                   dataset_id: DatasetId) -> List[ValidationResult]:
        """Execute validation rules in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_rule = {}
            
            for rule in rules:
                future = executor.submit(self.validate_single_rule, df, rule, dataset_id)
                future_to_rule[future] = rule
            
            for future in as_completed(future_to_rule, timeout=self.config.timeout_seconds):
                rule = future_to_rule[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Parallel validation failed for rule {rule.rule_name}: {str(e)}")
                    results.append(self._create_error_result(rule, dataset_id, str(e)))
        
        return results
    
    def _execute_sequential_validation(self, 
                                     df: pd.DataFrame,
                                     rules: List[QualityRule],
                                     dataset_id: DatasetId) -> List[ValidationResult]:
        """Execute validation rules sequentially."""
        results = []
        
        for rule in rules:
            try:
                result = self.validate_single_rule(df, rule, dataset_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Sequential validation failed for rule {rule.rule_name}: {str(e)}")
                results.append(self._create_error_result(rule, dataset_id, str(e)))
        
        return results
    
    def _create_error_result(self, rule: QualityRule, dataset_id: DatasetId, error_message: str) -> ValidationResult:
        """Create a validation result for errors."""
        return ValidationResult(
            validation_id=ValidationId(),
            rule_id=rule.rule_id,
            dataset_id=dataset_id,
            status=ValidationStatus.ERROR,
            passed_records=0,
            failed_records=0,
            failure_rate=0.0,
            error_details=[ValidationError(error_message=error_message, severity=Severity.HIGH)],
            execution_time=timedelta(seconds=0),
            validated_at=datetime.now()
        )
    
    def _should_use_cache(self, df: pd.DataFrame, rule: QualityRule) -> bool:
        """Check if cached result should be used."""
        if not self._cache:
            return False
        
        cache_key = self._get_cache_key(df, rule)
        
        if cache_key not in self._cache:
            return False
        
        # Check if cache is still valid (simple time-based invalidation)
        cache_time = self._cache_timestamps.get(cache_key, datetime.min)
        if datetime.now() - cache_time > timedelta(minutes=30):
            return False
        
        return True
    
    def _get_from_cache(self, df: pd.DataFrame, rule: QualityRule) -> Optional[ValidationResult]:
        """Get validation result from cache."""
        cache_key = self._get_cache_key(df, rule)
        return self._cache.get(cache_key)
    
    def _cache_result(self, df: pd.DataFrame, rule: QualityRule, result: ValidationResult) -> None:
        """Cache validation result."""
        if not self._cache:
            return
        
        cache_key = self._get_cache_key(df, rule)
        self._cache[cache_key] = result
        self._cache_timestamps[cache_key] = datetime.now()
        
        # Simple cache size management
        if len(self._cache) > self.config.cache_size:
            # Remove oldest entries
            oldest_key = min(self._cache_timestamps.keys(), key=lambda k: self._cache_timestamps[k])
            del self._cache[oldest_key]
            del self._cache_timestamps[oldest_key]
    
    def _get_cache_key(self, df: pd.DataFrame, rule: QualityRule) -> str:
        """Generate cache key for dataset and rule."""
        # Simple hash-based key (in production, use more sophisticated hashing)
        df_hash = hash(str(df.dtypes.to_dict()) + str(len(df)))
        rule_hash = hash(str(rule.rule_id) + rule.validation_logic.expression)
        return f"{df_hash}_{rule_hash}"
    
    def _compile_python_rule(self, logic: ValidationLogic) -> str:
        """Compile Python rule for execution."""
        rule_key = hash(logic.expression)
        
        if rule_key not in self._compiled_rules_cache:
            # Basic compilation (in production, use AST parsing and validation)
            self._compiled_rules_cache[rule_key] = compile(logic.expression, '<string>', 'eval')
        
        return self._compiled_rules_cache[rule_key]
    
    def _execute_row_wise_python(self, 
                                df: pd.DataFrame,
                                compiled_expr: str,
                                logic: ValidationLogic) -> tuple[int, int, List[ValidationError]]:
        """Execute row-wise Python validation."""
        passed_records = 0
        failed_records = 0
        error_details = []
        
        for idx, row in df.iterrows():
            try:
                # Create namespace for row
                namespace = dict(row)
                namespace['row'] = row
                
                # Evaluate expression
                result = eval(compiled_expr, {"__builtins__": {}}, namespace)
                
                if result:
                    passed_records += 1
                else:
                    failed_records += 1
                    if len(error_details) < 100:  # Limit error details
                        error_details.append(ValidationError(
                            row_index=idx,
                            error_message=logic.error_message,
                            severity=Severity.MEDIUM
                        ))
                        
            except Exception as e:
                failed_records += 1
                if len(error_details) < 100:
                    error_details.append(ValidationError(
                        row_index=idx,
                        error_message=f"Python evaluation error: {str(e)}",
                        severity=Severity.HIGH
                    ))
        
        return passed_records, failed_records, error_details
    
    def _execute_aggregate_python(self, 
                                 df: pd.DataFrame,
                                 compiled_expr: str,
                                 logic: ValidationLogic) -> tuple[int, int, List[ValidationError]]:
        """Execute aggregate Python validation."""
        try:
            # Create namespace for DataFrame
            namespace = {'df': df}
            for col in df.columns:
                namespace[col] = df[col]
            
            # Evaluate expression
            result = eval(compiled_expr, {"__builtins__": {}}, namespace)
            
            if isinstance(result, bool):
                if result:
                    return len(df), 0, []
                else:
                    return 0, len(df), [ValidationError(
                        error_message=logic.error_message,
                        severity=Severity.HIGH
                    )]
            elif isinstance(result, pd.Series):
                # Series result - row-wise validation
                passed = result.sum()
                failed = len(result) - passed
                
                error_details = []
                if failed > 0:
                    failed_indices = result[~result].index.tolist()
                    for idx in failed_indices[:100]:
                        error_details.append(ValidationError(
                            row_index=idx,
                            error_message=logic.error_message,
                            severity=Severity.MEDIUM
                        ))
                
                return passed, failed, error_details
            else:
                raise ValueError(f"Unexpected result type: {type(result)}")
                
        except Exception as e:
            return 0, len(df), [ValidationError(
                error_message=f"Python evaluation error: {str(e)}",
                severity=Severity.HIGH
            )]
    
    def _extract_where_clause(self, sql: str) -> str:
        """Extract WHERE clause from SQL statement."""
        # Simplified SQL parsing - in production, use proper SQL parser
        where_idx = sql.find('where')
        if where_idx == -1:
            return "1=1"  # Always true
        
        where_clause = sql[where_idx + 5:].strip()
        
        # Remove trailing semicolon if present
        if where_clause.endswith(';'):
            where_clause = where_clause[:-1]
        
        return where_clause
    
    def _evaluate_sql_condition(self, df: pd.DataFrame, condition: str) -> pd.Series:
        """Evaluate SQL condition as pandas expression."""
        # Convert basic SQL to pandas (simplified)
        pandas_expr = condition.replace('=', '==')
        return self._evaluate_expression(df, pandas_expr)
    
    def _update_execution_metrics(self, results: List[ValidationResult], execution_time: float) -> None:
        """Update execution metrics."""
        self._execution_metrics['total_validations'] += len(results)
        self._execution_metrics['total_execution_time'] += execution_time
        
        for result in results:
            if result.status == ValidationStatus.PASSED:
                self._execution_metrics['successful_validations'] += 1
            else:
                self._execution_metrics['failed_validations'] += 1
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get engine execution metrics."""
        metrics = self._execution_metrics.copy()
        
        # Calculate derived metrics
        total_validations = metrics['total_validations']
        if total_validations > 0:
            metrics['success_rate'] = metrics['successful_validations'] / total_validations
            metrics['failure_rate'] = metrics['failed_validations'] / total_validations
            metrics['average_execution_time'] = metrics['total_execution_time'] / total_validations
        else:
            metrics['success_rate'] = 0.0
            metrics['failure_rate'] = 0.0
            metrics['average_execution_time'] = 0.0
        
        # Cache metrics
        if self._cache:
            metrics['cache_size'] = len(self._cache)
            total_cache_requests = metrics['cache_hits'] + metrics['cache_misses']
            if total_cache_requests > 0:
                metrics['cache_hit_rate'] = metrics['cache_hits'] / total_cache_requests
            else:
                metrics['cache_hit_rate'] = 0.0
        
        return metrics
    
    def clear_cache(self) -> None:
        """Clear validation cache."""
        if self._cache:
            self._cache.clear()
            self._cache_timestamps.clear()
            self._compiled_rules_cache.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information."""
        if not self._cache:
            return {'cache_enabled': False}
        
        return {
            'cache_enabled': True,
            'cache_size': len(self._cache),
            'cache_capacity': self.config.cache_size,
            'compiled_rules_cached': len(self._compiled_rules_cache)
        }