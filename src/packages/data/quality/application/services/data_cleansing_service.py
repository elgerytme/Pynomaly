"""Data Cleansing Service.

Automated data cleansing service that provides comprehensive data cleaning
and transformation capabilities with configurable cleansing rules and strategies.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import re
from enum import Enum
from abc import ABC, abstractmethod
import warnings

from ...domain.entities.validation_rule import QualityRule, ValidationResult, Severity
from ...domain.entities.quality_profile import DatasetId
from ...domain.entities.quality_scores import QualityScores

logger = logging.getLogger(__name__)


class CleansingStrategy(Enum):
    """Data cleansing strategies."""
    REMOVE = "remove"              # Remove problematic records/values
    REPLACE = "replace"            # Replace with specified values
    IMPUTE = "impute"             # Statistical imputation
    STANDARDIZE = "standardize"    # Standardize formats
    VALIDATE = "validate"          # Validate and flag
    TRANSFORM = "transform"        # Transform using functions
    INTERPOLATE = "interpolate"    # Interpolate missing values
    AGGREGATE = "aggregate"        # Aggregate duplicate records


class CleansingAction(Enum):
    """Types of cleansing actions."""
    MISSING_VALUES = "missing_values"
    DUPLICATES = "duplicates"
    OUTLIERS = "outliers"
    FORMAT_ISSUES = "format_issues"
    INVALID_VALUES = "invalid_values"
    INCONSISTENT_CASING = "inconsistent_casing"
    WHITESPACE = "whitespace"
    DATA_TYPE_ISSUES = "data_type_issues"
    PATTERN_VIOLATIONS = "pattern_violations"
    RANGE_VIOLATIONS = "range_violations"


@dataclass(frozen=True)
class CleansingRule:
    """Configuration for a data cleansing rule."""
    action: CleansingAction
    strategy: CleansingStrategy
    target_columns: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[str] = None  # Optional condition for when to apply
    priority: int = 1  # Higher priority rules are applied first
    
    def applies_to_column(self, column_name: str) -> bool:
        """Check if rule applies to a specific column."""
        return not self.target_columns or column_name in self.target_columns
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get parameter value with default."""
        return self.parameters.get(key, default)


@dataclass
class CleansingResult:
    """Result of data cleansing operation."""
    original_records: int
    cleaned_records: int
    removed_records: int
    modified_records: int
    actions_applied: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    quality_improvement: Optional[float] = None
    
    @property
    def records_retained(self) -> int:
        """Calculate number of records retained."""
        return self.original_records - self.removed_records
    
    @property
    def retention_rate(self) -> float:
        """Calculate retention rate."""
        if self.original_records == 0:
            return 1.0
        return self.records_retained / self.original_records
    
    @property
    def modification_rate(self) -> float:
        """Calculate modification rate."""
        if self.original_records == 0:
            return 0.0
        return self.modified_records / self.original_records
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cleansing result summary."""
        return {
            'original_records': self.original_records,
            'cleaned_records': self.cleaned_records,
            'removed_records': self.removed_records,
            'modified_records': self.modified_records,
            'retention_rate': round(self.retention_rate, 4),
            'modification_rate': round(self.modification_rate, 4),
            'actions_applied_count': len(self.actions_applied),
            'warnings_count': len(self.warnings),
            'errors_count': len(self.errors),
            'execution_time_seconds': self.execution_time_seconds,
            'quality_improvement': self.quality_improvement
        }


class CleansingProcessor(ABC):
    """Abstract base class for cleansing processors."""
    
    @abstractmethod
    def process(self, df: pd.DataFrame, rule: CleansingRule) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process data according to cleansing rule."""
        pass
    
    @abstractmethod
    def can_handle(self, action: CleansingAction, strategy: CleansingStrategy) -> bool:
        """Check if processor can handle the action/strategy combination."""
        pass


class MissingValueProcessor(CleansingProcessor):
    """Processor for handling missing values."""
    
    def can_handle(self, action: CleansingAction, strategy: CleansingStrategy) -> bool:
        """Check if can handle missing value actions."""
        return action == CleansingAction.MISSING_VALUES
    
    def process(self, df: pd.DataFrame, rule: CleansingRule) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process missing values according to strategy."""
        result_df = df.copy()
        action_info = {
            'action': rule.action.value,
            'strategy': rule.strategy.value,
            'columns_processed': [],
            'records_affected': 0
        }
        
        target_columns = rule.target_columns if rule.target_columns else df.columns.tolist()
        
        for column in target_columns:
            if column not in df.columns:
                continue
                
            missing_before = df[column].isna().sum()
            if missing_before == 0:
                continue
            
            if rule.strategy == CleansingStrategy.REMOVE:
                # Remove rows with missing values in this column
                result_df = result_df.dropna(subset=[column])
                
            elif rule.strategy == CleansingStrategy.REPLACE:
                # Replace with specified value
                replace_value = rule.get_parameter('replace_value', 0)
                result_df[column] = result_df[column].fillna(replace_value)
                
            elif rule.strategy == CleansingStrategy.IMPUTE:
                # Statistical imputation
                impute_method = rule.get_parameter('impute_method', 'mean')
                
                if df[column].dtype in ['int64', 'float64']:
                    if impute_method == 'mean':
                        fill_value = df[column].mean()
                    elif impute_method == 'median':
                        fill_value = df[column].median()
                    elif impute_method == 'mode':
                        fill_value = df[column].mode().iloc[0] if not df[column].mode().empty else 0
                    else:
                        fill_value = 0
                else:
                    # For non-numeric columns, use mode or specified value
                    if impute_method == 'mode':
                        fill_value = df[column].mode().iloc[0] if not df[column].mode().empty else 'Unknown'
                    else:
                        fill_value = rule.get_parameter('default_value', 'Unknown')
                
                result_df[column] = result_df[column].fillna(fill_value)
                
            elif rule.strategy == CleansingStrategy.INTERPOLATE:
                # Interpolation for numeric columns
                if df[column].dtype in ['int64', 'float64']:
                    method = rule.get_parameter('interpolation_method', 'linear')
                    result_df[column] = result_df[column].interpolate(method=method)
                
            action_info['columns_processed'].append(column)
            action_info['records_affected'] += missing_before
        
        return result_df, action_info


class DuplicateProcessor(CleansingProcessor):
    """Processor for handling duplicate records."""
    
    def can_handle(self, action: CleansingAction, strategy: CleansingStrategy) -> bool:
        """Check if can handle duplicate actions."""
        return action == CleansingAction.DUPLICATES
    
    def process(self, df: pd.DataFrame, rule: CleansingRule) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process duplicates according to strategy."""
        result_df = df.copy()
        action_info = {
            'action': rule.action.value,
            'strategy': rule.strategy.value,
            'columns_processed': rule.target_columns or ['all'],
            'records_affected': 0
        }
        
        # Identify duplicates
        subset_columns = rule.target_columns if rule.target_columns else None
        duplicates_before = df.duplicated(subset=subset_columns).sum()
        
        if duplicates_before == 0:
            return result_df, action_info
        
        if rule.strategy == CleansingStrategy.REMOVE:
            # Remove duplicate rows
            keep_strategy = rule.get_parameter('keep', 'first')  # 'first', 'last', False
            result_df = result_df.drop_duplicates(subset=subset_columns, keep=keep_strategy)
            
        elif rule.strategy == CleansingStrategy.AGGREGATE:
            # Aggregate duplicate rows
            agg_functions = rule.get_parameter('aggregation_functions', {})
            if subset_columns and agg_functions:
                # Group by subset columns and aggregate others
                result_df = result_df.groupby(subset_columns).agg(agg_functions).reset_index()
        
        action_info['records_affected'] = duplicates_before
        return result_df, action_info


class OutlierProcessor(CleansingProcessor):
    """Processor for handling outliers."""
    
    def can_handle(self, action: CleansingAction, strategy: CleansingStrategy) -> bool:
        """Check if can handle outlier actions."""
        return action == CleansingAction.OUTLIERS
    
    def process(self, df: pd.DataFrame, rule: CleansingRule) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process outliers according to strategy."""
        result_df = df.copy()
        action_info = {
            'action': rule.action.value,
            'strategy': rule.strategy.value,
            'columns_processed': [],
            'records_affected': 0
        }
        
        target_columns = rule.target_columns if rule.target_columns else df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_method = rule.get_parameter('outlier_method', 'iqr')
        
        for column in target_columns:
            if column not in df.columns or df[column].dtype not in ['int64', 'float64']:
                continue
            
            # Identify outliers
            if outlier_method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
                
            elif outlier_method == 'z_score':
                z_threshold = rule.get_parameter('z_threshold', 3.0)
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                outlier_mask = z_scores > z_threshold
                
            elif outlier_method == 'percentile':
                lower_percentile = rule.get_parameter('lower_percentile', 1)
                upper_percentile = rule.get_parameter('upper_percentile', 99)
                lower_bound = df[column].quantile(lower_percentile / 100)
                upper_bound = df[column].quantile(upper_percentile / 100)
                outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
                
            else:
                continue
            
            outliers_count = outlier_mask.sum()
            if outliers_count == 0:
                continue
            
            if rule.strategy == CleansingStrategy.REMOVE:
                # Remove outlier rows
                result_df = result_df[~outlier_mask]
                
            elif rule.strategy == CleansingStrategy.REPLACE:
                # Replace outliers with specified values
                replace_value = rule.get_parameter('replace_value')
                if replace_value is not None:
                    result_df.loc[outlier_mask, column] = replace_value
                else:
                    # Replace with median
                    result_df.loc[outlier_mask, column] = df[column].median()
            
            elif rule.strategy == CleansingStrategy.TRANSFORM:
                # Cap outliers at bounds
                if outlier_method == 'iqr':
                    result_df.loc[result_df[column] < lower_bound, column] = lower_bound
                    result_df.loc[result_df[column] > upper_bound, column] = upper_bound
                elif outlier_method == 'percentile':
                    result_df.loc[result_df[column] < lower_bound, column] = lower_bound
                    result_df.loc[result_df[column] > upper_bound, column] = upper_bound
            
            action_info['columns_processed'].append(column)
            action_info['records_affected'] += outliers_count
        
        return result_df, action_info


class FormatProcessor(CleansingProcessor):
    """Processor for handling format issues."""
    
    def can_handle(self, action: CleansingAction, strategy: CleansingStrategy) -> bool:
        """Check if can handle format actions."""
        return action in [CleansingAction.FORMAT_ISSUES, CleansingAction.WHITESPACE, 
                         CleansingAction.INCONSISTENT_CASING]
    
    def process(self, df: pd.DataFrame, rule: CleansingRule) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process format issues according to strategy."""
        result_df = df.copy()
        action_info = {
            'action': rule.action.value,
            'strategy': rule.strategy.value,
            'columns_processed': [],
            'records_affected': 0
        }
        
        target_columns = rule.target_columns if rule.target_columns else df.select_dtypes(include=['object']).columns.tolist()
        
        for column in target_columns:
            if column not in df.columns:
                continue
            
            if rule.action == CleansingAction.WHITESPACE:
                # Handle whitespace issues
                if rule.strategy == CleansingStrategy.STANDARDIZE:
                    # Strip whitespace and normalize spaces
                    original_values = result_df[column].astype(str)
                    result_df[column] = original_values.str.strip().str.replace(r'\s+', ' ', regex=True)
                    
                    # Count affected records
                    affected = (original_values != result_df[column]).sum()
                    action_info['records_affected'] += affected
                    
            elif rule.action == CleansingAction.INCONSISTENT_CASING:
                # Handle casing issues
                if rule.strategy == CleansingStrategy.STANDARDIZE:
                    case_method = rule.get_parameter('case_method', 'lower')  # 'lower', 'upper', 'title', 'proper'
                    
                    if case_method == 'lower':
                        result_df[column] = result_df[column].astype(str).str.lower()
                    elif case_method == 'upper':
                        result_df[column] = result_df[column].astype(str).str.upper()
                    elif case_method == 'title':
                        result_df[column] = result_df[column].astype(str).str.title()
                    elif case_method == 'proper':
                        result_df[column] = result_df[column].astype(str).str.capitalize()
                    
                    action_info['records_affected'] += len(df)
                    
            elif rule.action == CleansingAction.FORMAT_ISSUES:
                # Handle general format issues
                if rule.strategy == CleansingStrategy.STANDARDIZE:
                    format_patterns = rule.get_parameter('format_patterns', {})
                    
                    for pattern, replacement in format_patterns.items():
                        before_count = result_df[column].astype(str).str.contains(pattern, regex=True).sum()
                        result_df[column] = result_df[column].astype(str).str.replace(pattern, replacement, regex=True)
                        action_info['records_affected'] += before_count
            
            action_info['columns_processed'].append(column)
        
        return result_df, action_info


class ValidationProcessor(CleansingProcessor):
    """Processor for handling validation issues."""
    
    def can_handle(self, action: CleansingAction, strategy: CleansingStrategy) -> bool:
        """Check if can handle validation actions."""
        return action in [CleansingAction.INVALID_VALUES, CleansingAction.PATTERN_VIOLATIONS,
                         CleansingAction.RANGE_VIOLATIONS, CleansingAction.DATA_TYPE_ISSUES]
    
    def process(self, df: pd.DataFrame, rule: CleansingRule) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process validation issues according to strategy."""
        result_df = df.copy()
        action_info = {
            'action': rule.action.value,
            'strategy': rule.strategy.value,
            'columns_processed': [],
            'records_affected': 0
        }
        
        target_columns = rule.target_columns if rule.target_columns else df.columns.tolist()
        
        for column in target_columns:
            if column not in df.columns:
                continue
            
            if rule.action == CleansingAction.PATTERN_VIOLATIONS:
                # Handle pattern violations
                pattern = rule.get_parameter('pattern')
                if pattern:
                    pattern_regex = re.compile(pattern)
                    valid_mask = df[column].astype(str).str.match(pattern_regex, na=False)
                    invalid_count = (~valid_mask).sum()
                    
                    if invalid_count > 0:
                        if rule.strategy == CleansingStrategy.REMOVE:
                            result_df = result_df[valid_mask]
                        elif rule.strategy == CleansingStrategy.REPLACE:
                            replace_value = rule.get_parameter('replace_value', 'INVALID')
                            result_df.loc[~valid_mask, column] = replace_value
                        
                        action_info['records_affected'] += invalid_count
                        
            elif rule.action == CleansingAction.RANGE_VIOLATIONS:
                # Handle range violations
                min_value = rule.get_parameter('min_value')
                max_value = rule.get_parameter('max_value')
                
                if df[column].dtype in ['int64', 'float64'] and (min_value is not None or max_value is not None):
                    if min_value is not None:
                        min_violations = df[column] < min_value
                        violations_count = min_violations.sum()
                        
                        if violations_count > 0:
                            if rule.strategy == CleansingStrategy.REMOVE:
                                result_df = result_df[~min_violations]
                            elif rule.strategy == CleansingStrategy.TRANSFORM:
                                result_df.loc[min_violations, column] = min_value
                            
                            action_info['records_affected'] += violations_count
                    
                    if max_value is not None:
                        max_violations = df[column] > max_value
                        violations_count = max_violations.sum()
                        
                        if violations_count > 0:
                            if rule.strategy == CleansingStrategy.REMOVE:
                                result_df = result_df[~max_violations]
                            elif rule.strategy == CleansingStrategy.TRANSFORM:
                                result_df.loc[max_violations, column] = max_value
                            
                            action_info['records_affected'] += violations_count
            
            elif rule.action == CleansingAction.DATA_TYPE_ISSUES:
                # Handle data type conversion issues
                target_type = rule.get_parameter('target_type')
                if target_type:
                    try:
                        if target_type == 'numeric':
                            # Convert to numeric, coercing errors
                            original_values = result_df[column].copy()
                            result_df[column] = pd.to_numeric(result_df[column], errors='coerce')
                            
                            # Count conversion failures
                            failures = result_df[column].isna() & original_values.notna()
                            failures_count = failures.sum()
                            
                            if failures_count > 0:
                                if rule.strategy == CleansingStrategy.REMOVE:
                                    result_df = result_df[~failures]
                                elif rule.strategy == CleansingStrategy.REPLACE:
                                    replace_value = rule.get_parameter('replace_value', 0)
                                    result_df.loc[failures, column] = replace_value
                                
                                action_info['records_affected'] += failures_count
                        
                        elif target_type == 'datetime':
                            # Convert to datetime
                            original_values = result_df[column].copy()
                            result_df[column] = pd.to_datetime(result_df[column], errors='coerce')
                            
                            failures = result_df[column].isna() & original_values.notna()
                            failures_count = failures.sum()
                            
                            if failures_count > 0:
                                if rule.strategy == CleansingStrategy.REMOVE:
                                    result_df = result_df[~failures]
                                elif rule.strategy == CleansingStrategy.REPLACE:
                                    replace_value = rule.get_parameter('replace_value', pd.Timestamp('1900-01-01'))
                                    result_df.loc[failures, column] = replace_value
                                
                                action_info['records_affected'] += failures_count
                                
                    except Exception as e:
                        logger.warning(f"Data type conversion failed for column {column}: {e}")
            
            action_info['columns_processed'].append(column)
        
        return result_df, action_info


@dataclass(frozen=True)
class DataCleansingConfig:
    """Configuration for data cleansing service."""
    # Processing options
    enable_parallel_processing: bool = False  # For future parallel processing
    max_memory_usage_mb: int = 1024
    chunk_size: int = 10000
    
    # Quality assessment
    assess_quality_before: bool = True
    assess_quality_after: bool = True
    
    # Backup and recovery
    create_backup: bool = True
    backup_sample_size: int = 1000
    
    # Validation
    validate_after_cleansing: bool = True
    strict_mode: bool = False  # Fail on any error
    
    # Reporting
    detailed_reporting: bool = True
    max_warnings: int = 100
    max_errors: int = 50


class DataCleansingService:
    """Service for automated data cleansing operations."""
    
    def __init__(self, config: DataCleansingConfig = None):
        """Initialize data cleansing service.
        
        Args:
            config: Service configuration
        """
        self.config = config or DataCleansingConfig()
        
        # Initialize processors
        self._processors = [
            MissingValueProcessor(),
            DuplicateProcessor(),
            OutlierProcessor(),
            FormatProcessor(),
            ValidationProcessor()
        ]
        
        # Create processor lookup
        self._processor_map = {}
        for processor in self._processors:
            for action in CleansingAction:
                for strategy in CleansingStrategy:
                    if processor.can_handle(action, strategy):
                        self._processor_map[(action, strategy)] = processor
        
        logger.info("Data Cleansing Service initialized")
    
    def cleanse_dataset(self, 
                       df: pd.DataFrame,
                       cleansing_rules: List[CleansingRule],
                       dataset_id: Optional[DatasetId] = None) -> Tuple[pd.DataFrame, CleansingResult]:
        """Cleanse dataset using specified rules.
        
        Args:
            df: Input DataFrame
            cleansing_rules: List of cleansing rules to apply
            dataset_id: Optional dataset identifier
            
        Returns:
            Tuple of (cleaned_dataframe, cleansing_result)
        """
        start_time = datetime.now()
        
        # Initialize result
        result = CleansingResult(
            original_records=len(df),
            cleaned_records=0,
            removed_records=0,
            modified_records=0
        )
        
        # Create backup if enabled
        backup_df = None
        if self.config.create_backup:
            backup_df = df.copy()
            if len(df) > self.config.backup_sample_size:
                backup_df = df.sample(self.config.backup_sample_size)
        
        # Assess quality before cleansing
        quality_before = None
        if self.config.assess_quality_before:
            quality_before = self._assess_quality(df)
        
        try:
            # Apply cleansing rules
            cleaned_df = df.copy()
            
            # Sort rules by priority (higher priority first)
            sorted_rules = sorted(cleansing_rules, key=lambda r: r.priority, reverse=True)
            
            for rule in sorted_rules:
                try:
                    # Check if rule applies
                    if rule.condition:
                        # Evaluate condition (simplified)
                        if not self._evaluate_condition(cleaned_df, rule.condition):
                            continue
                    
                    # Find appropriate processor
                    processor = self._processor_map.get((rule.action, rule.strategy))
                    if not processor:
                        result.warnings.append(f"No processor found for {rule.action.value} with {rule.strategy.value}")
                        continue
                    
                    # Apply cleansing
                    records_before = len(cleaned_df)
                    cleaned_df, action_info = processor.process(cleaned_df, rule)
                    records_after = len(cleaned_df)
                    
                    # Update result
                    result.actions_applied.append(action_info)
                    result.removed_records += (records_before - records_after)
                    result.modified_records += action_info.get('records_affected', 0)
                    
                    logger.debug(f"Applied rule {rule.action.value}: {action_info}")
                    
                except Exception as e:
                    error_msg = f"Error applying rule {rule.action.value}: {str(e)}"
                    result.errors.append(error_msg)
                    logger.error(error_msg)
                    
                    if self.config.strict_mode:
                        raise
            
            # Finalize result
            result.cleaned_records = len(cleaned_df)
            
            # Assess quality after cleansing
            quality_after = None
            if self.config.assess_quality_after:
                quality_after = self._assess_quality(cleaned_df)
                
                # Calculate quality improvement
                if quality_before and quality_after:
                    result.quality_improvement = quality_after - quality_before
            
            # Calculate execution time
            result.execution_time_seconds = (datetime.now() - start_time).total_seconds()
            
            # Validate after cleansing
            if self.config.validate_after_cleansing:
                validation_warnings = self._validate_cleaned_data(cleaned_df, df)
                result.warnings.extend(validation_warnings)
            
            logger.info(f"Data cleansing completed: {result.get_summary()}")
            
            return cleaned_df, result
            
        except Exception as e:
            error_msg = f"Data cleansing failed: {str(e)}"
            result.errors.append(error_msg)
            logger.error(error_msg)
            
            if self.config.strict_mode:
                raise
            
            # Return original data if cleansing fails
            result.cleaned_records = len(df)
            result.execution_time_seconds = (datetime.now() - start_time).total_seconds()
            
            return df, result
    
    def create_default_cleansing_rules(self, df: pd.DataFrame) -> List[CleansingRule]:
        """Create default cleansing rules based on data analysis.
        
        Args:
            df: Input DataFrame to analyze
            
        Returns:
            List of default cleansing rules
        """
        rules = []
        
        # Rule 1: Handle missing values
        rules.append(CleansingRule(
            action=CleansingAction.MISSING_VALUES,
            strategy=CleansingStrategy.IMPUTE,
            target_columns=[],  # Apply to all columns
            parameters={'impute_method': 'median'},
            priority=3
        ))
        
        # Rule 2: Remove duplicates
        rules.append(CleansingRule(
            action=CleansingAction.DUPLICATES,
            strategy=CleansingStrategy.REMOVE,
            target_columns=[],  # All columns
            parameters={'keep': 'first'},
            priority=2
        ))
        
        # Rule 3: Handle outliers in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            rules.append(CleansingRule(
                action=CleansingAction.OUTLIERS,
                strategy=CleansingStrategy.TRANSFORM,
                target_columns=numeric_columns,
                parameters={'outlier_method': 'iqr'},
                priority=1
            ))
        
        # Rule 4: Standardize whitespace
        string_columns = df.select_dtypes(include=['object']).columns.tolist()
        if string_columns:
            rules.append(CleansingRule(
                action=CleansingAction.WHITESPACE,
                strategy=CleansingStrategy.STANDARDIZE,
                target_columns=string_columns,
                priority=4
            ))
        
        # Rule 5: Standardize casing
        if string_columns:
            rules.append(CleansingRule(
                action=CleansingAction.INCONSISTENT_CASING,
                strategy=CleansingStrategy.STANDARDIZE,
                target_columns=string_columns,
                parameters={'case_method': 'lower'},
                priority=5
            ))
        
        return rules
    
    def _assess_quality(self, df: pd.DataFrame) -> float:
        """Assess data quality (simplified scoring).
        
        Args:
            df: DataFrame to assess
            
        Returns:
            Quality score between 0 and 1
        """
        if len(df) == 0:
            return 0.0
        
        # Calculate basic quality metrics
        total_cells = len(df) * len(df.columns)
        
        # Completeness
        null_cells = df.isnull().sum().sum()
        completeness = 1 - (null_cells / total_cells) if total_cells > 0 else 0
        
        # Uniqueness (for non-numeric columns)
        uniqueness_scores = []
        for col in df.select_dtypes(include=['object']).columns:
            if len(df[col]) > 0:
                unique_ratio = df[col].nunique() / len(df[col])
                uniqueness_scores.append(unique_ratio)
        
        uniqueness = np.mean(uniqueness_scores) if uniqueness_scores else 1.0
        
        # Consistency (simplified - no extreme outliers)
        consistency_scores = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if len(df[col]) > 0:
                # Use IQR method to detect outliers
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    consistency_scores.append(1 - (outliers / len(df[col])))
                else:
                    consistency_scores.append(1.0)
        
        consistency = np.mean(consistency_scores) if consistency_scores else 1.0
        
        # Overall quality score (weighted average)
        quality_score = (completeness * 0.4) + (uniqueness * 0.3) + (consistency * 0.3)
        
        return max(0.0, min(1.0, quality_score))
    
    def _evaluate_condition(self, df: pd.DataFrame, condition: str) -> bool:
        """Evaluate condition for rule application.
        
        Args:
            df: DataFrame to evaluate condition on
            condition: Condition string
            
        Returns:
            Whether condition is met
        """
        try:
            # Simple condition evaluation
            # In production, use proper expression parser
            if 'len(' in condition:
                return eval(condition.replace('df', f'len(df)'), {'len': len, 'df': df})
            else:
                return True  # Default to applying rule
        except:
            return True  # Default to applying rule if evaluation fails
    
    def _validate_cleaned_data(self, cleaned_df: pd.DataFrame, original_df: pd.DataFrame) -> List[str]:
        """Validate cleaned data against original.
        
        Args:
            cleaned_df: Cleaned DataFrame
            original_df: Original DataFrame
            
        Returns:
            List of validation warnings
        """
        warnings = []
        
        # Check for excessive data loss
        retention_rate = len(cleaned_df) / len(original_df) if len(original_df) > 0 else 0
        if retention_rate < 0.5:
            warnings.append(f"High data loss: {(1-retention_rate)*100:.1f}% of records removed")
        
        # Check for column changes
        if list(cleaned_df.columns) != list(original_df.columns):
            warnings.append("Column structure changed during cleansing")
        
        # Check for data type changes
        for col in cleaned_df.columns:
            if col in original_df.columns:
                if cleaned_df[col].dtype != original_df[col].dtype:
                    warnings.append(f"Data type changed for column {col}: {original_df[col].dtype} -> {cleaned_df[col].dtype}")
        
        return warnings
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get information about available processors.
        
        Returns:
            Dictionary with processor information
        """
        processor_info = {
            'total_processors': len(self._processors),
            'supported_actions': list(CleansingAction),
            'supported_strategies': list(CleansingStrategy),
            'processor_capabilities': {}
        }
        
        for processor in self._processors:
            processor_name = processor.__class__.__name__
            capabilities = []
            
            for action in CleansingAction:
                for strategy in CleansingStrategy:
                    if processor.can_handle(action, strategy):
                        capabilities.append(f"{action.value}:{strategy.value}")
            
            processor_info['processor_capabilities'][processor_name] = capabilities
        
        return processor_info
    
    def recommend_cleansing_rules(self, df: pd.DataFrame, 
                                 quality_threshold: float = 0.8) -> List[CleansingRule]:
        """Recommend cleansing rules based on data quality assessment.
        
        Args:
            df: DataFrame to analyze
            quality_threshold: Minimum quality threshold
            
        Returns:
            List of recommended cleansing rules
        """
        rules = []
        
        # Assess current quality
        current_quality = self._assess_quality(df)
        
        if current_quality >= quality_threshold:
            logger.info(f"Data quality {current_quality:.3f} meets threshold {quality_threshold}")
            return rules
        
        # Analyze specific quality issues
        
        # 1. Missing values
        missing_percentages = df.isnull().sum() / len(df)
        high_missing_cols = missing_percentages[missing_percentages > 0.1].index.tolist()
        
        if high_missing_cols:
            rules.append(CleansingRule(
                action=CleansingAction.MISSING_VALUES,
                strategy=CleansingStrategy.IMPUTE,
                target_columns=high_missing_cols,
                parameters={'impute_method': 'median'},
                priority=3
            ))
        
        # 2. Duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            rules.append(CleansingRule(
                action=CleansingAction.DUPLICATES,
                strategy=CleansingStrategy.REMOVE,
                target_columns=[],
                parameters={'keep': 'first'},
                priority=2
            ))
        
        # 3. Outliers in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_columns = []
        
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                
                if outliers / len(df) > 0.05:  # More than 5% outliers
                    outlier_columns.append(col)
        
        if outlier_columns:
            rules.append(CleansingRule(
                action=CleansingAction.OUTLIERS,
                strategy=CleansingStrategy.TRANSFORM,
                target_columns=outlier_columns,
                parameters={'outlier_method': 'iqr'},
                priority=1
            ))
        
        # 4. String formatting issues
        string_columns = df.select_dtypes(include=['object']).columns
        if len(string_columns) > 0:
            rules.append(CleansingRule(
                action=CleansingAction.WHITESPACE,
                strategy=CleansingStrategy.STANDARDIZE,
                target_columns=string_columns.tolist(),
                priority=4
            ))
        
        logger.info(f"Recommended {len(rules)} cleansing rules to improve quality from {current_quality:.3f}")
        return rules