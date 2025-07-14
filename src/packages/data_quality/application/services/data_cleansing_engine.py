"""Advanced data cleansing engine with automated cleaning and standardization."""

import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging
import phonenumbers
from email_validator import validate_email, EmailNotValidError

logger = logging.getLogger(__name__)


class CleansingStrategy(str, Enum):
    """Data cleansing strategy options."""
    REMOVE = "remove"
    REPLACE = "replace"
    STANDARDIZE = "standardize"
    IMPUTE = "impute"
    FLAG = "flag"


class CleansingAction(str, Enum):
    """Types of cleansing actions."""
    DUPLICATE_REMOVAL = "duplicate_removal"
    FORMAT_STANDARDIZATION = "format_standardization"
    VALUE_NORMALIZATION = "value_normalization"
    MISSING_VALUE_IMPUTATION = "missing_value_imputation"
    OUTLIER_TREATMENT = "outlier_treatment"
    DATA_TYPE_CONVERSION = "data_type_conversion"
    TEXT_CLEANING = "text_cleaning"
    DATE_STANDARDIZATION = "date_standardization"


@dataclass
class CleansingResult:
    """Result of a cleansing operation."""
    action: CleansingAction
    column_name: str
    records_affected: int
    original_values: List[Any]
    cleaned_values: List[Any]
    strategy_used: CleansingStrategy
    success: bool
    error_message: Optional[str] = None
    statistics: Optional[Dict[str, Any]] = None


@dataclass
class CleansingReport:
    """Comprehensive report of all cleansing operations."""
    dataset_name: str
    total_records: int
    total_columns: int
    cleansing_results: List[CleansingResult]
    execution_time_seconds: float
    overall_success: bool
    quality_improvement: Dict[str, float]  # Before/after quality scores


class DataCleansingEngine:
    """Advanced data cleansing engine with automated cleaning capabilities."""
    
    def __init__(self, 
                 enable_backup: bool = True,
                 max_memory_usage_mb: float = 1000.0,
                 parallel_processing: bool = True):
        self.enable_backup = enable_backup
        self.max_memory_usage_mb = max_memory_usage_mb
        self.parallel_processing = parallel_processing
        self._backup_data = {}
        self._cleansing_history = []
        
        # Initialize cleaners
        self._cleaners = {
            CleansingAction.DUPLICATE_REMOVAL: DuplicateRemovalCleaner(),
            CleansingAction.FORMAT_STANDARDIZATION: FormatStandardizationCleaner(),
            CleansingAction.VALUE_NORMALIZATION: ValueNormalizationCleaner(),
            CleansingAction.MISSING_VALUE_IMPUTATION: MissingValueImputationCleaner(),
            CleansingAction.OUTLIER_TREATMENT: OutlierTreatmentCleaner(),
            CleansingAction.DATA_TYPE_CONVERSION: DataTypeConversionCleaner(),
            CleansingAction.TEXT_CLEANING: TextCleaningCleaner(),
            CleansingAction.DATE_STANDARDIZATION: DateStandardizationCleaner()
        }
    
    def clean_dataset(self, df: pd.DataFrame, 
                     cleansing_config: Dict[str, Any],
                     dataset_name: str = "dataset") -> Tuple[pd.DataFrame, CleansingReport]:
        """Clean entire dataset according to configuration."""
        start_time = datetime.utcnow()
        
        # Create backup if enabled
        if self.enable_backup:
            self._backup_data[dataset_name] = df.copy()
        
        cleaned_df = df.copy()
        cleansing_results = []
        
        logger.info(f"Starting data cleansing for dataset '{dataset_name}' with {len(df)} rows")
        
        # Execute cleansing actions
        for action_name, action_config in cleansing_config.items():
            try:
                action = CleansingAction(action_name)
                if action in self._cleaners:
                    cleaner = self._cleaners[action]
                    
                    # Apply cleansing
                    cleaned_df, results = cleaner.clean(cleaned_df, action_config)
                    cleansing_results.extend(results)
                    
                    logger.info(f"Completed {action_name}: {len(results)} operations")
                else:
                    logger.warning(f"Unknown cleansing action: {action_name}")
            
            except Exception as e:
                logger.error(f"Error during {action_name}: {e}")
                error_result = CleansingResult(
                    action=CleansingAction(action_name),
                    column_name="N/A",
                    records_affected=0,
                    original_values=[],
                    cleaned_values=[],
                    strategy_used=CleansingStrategy.FLAG,
                    success=False,
                    error_message=str(e)
                )
                cleansing_results.append(error_result)
        
        # Calculate execution time
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Calculate quality improvement
        quality_improvement = self._calculate_quality_improvement(df, cleaned_df)
        
        # Create report
        report = CleansingReport(
            dataset_name=dataset_name,
            total_records=len(df),
            total_columns=len(df.columns),
            cleansing_results=cleansing_results,
            execution_time_seconds=execution_time,
            overall_success=all(result.success for result in cleansing_results),
            quality_improvement=quality_improvement
        )
        
        # Store in history
        self._cleansing_history.append(report)
        
        logger.info(f"Data cleansing completed in {execution_time:.2f} seconds")
        
        return cleaned_df, report
    
    def rollback_cleansing(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """Rollback to original dataset before cleansing."""
        if dataset_name in self._backup_data:
            return self._backup_data[dataset_name].copy()
        else:
            logger.warning(f"No backup found for dataset '{dataset_name}'")
            return None
    
    def get_cleansing_history(self) -> List[CleansingReport]:
        """Get history of all cleansing operations."""
        return self._cleansing_history.copy()
    
    def _calculate_quality_improvement(self, original_df: pd.DataFrame, 
                                     cleaned_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate quality improvement metrics."""
        try:
            original_stats = self._calculate_quality_stats(original_df)
            cleaned_stats = self._calculate_quality_stats(cleaned_df)
            
            improvement = {}
            for metric, original_value in original_stats.items():
                cleaned_value = cleaned_stats.get(metric, original_value)
                if original_value > 0:
                    improvement[metric] = ((cleaned_value - original_value) / original_value) * 100
                else:
                    improvement[metric] = 0.0
            
            return improvement
        except Exception as e:
            logger.error(f"Error calculating quality improvement: {e}")
            return {}
    
    def _calculate_quality_stats(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic quality statistics."""
        return {
            'completeness': 1.0 - (df.isnull().sum().sum() / (len(df) * len(df.columns))),
            'uniqueness': df.nunique().sum() / len(df) if len(df) > 0 else 0,
            'consistency': 1.0 - (df.duplicated().sum() / len(df)) if len(df) > 0 else 0
        }


class BaseCleaner:
    """Base class for data cleaners."""
    
    def clean(self, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, List[CleansingResult]]:
        """Clean data according to configuration."""
        raise NotImplementedError


class DuplicateRemovalCleaner(BaseCleaner):
    """Remove duplicate records from dataset."""
    
    def clean(self, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, List[CleansingResult]]:
        results = []
        cleaned_df = df.copy()
        
        # Get configuration
        subset_columns = config.get('subset_columns', None)
        keep_strategy = config.get('keep', 'first')  # 'first', 'last', 'none'
        
        # Find duplicates
        if subset_columns:
            duplicate_mask = cleaned_df.duplicated(subset=subset_columns, keep=False)
        else:
            duplicate_mask = cleaned_df.duplicated(keep=False)
        
        duplicate_count = duplicate_mask.sum()
        
        if duplicate_count > 0:
            # Get duplicate records before removal
            duplicate_records = cleaned_df[duplicate_mask]
            
            # Remove duplicates
            if subset_columns:
                cleaned_df = cleaned_df.drop_duplicates(subset=subset_columns, keep=keep_strategy)
            else:
                cleaned_df = cleaned_df.drop_duplicates(keep=keep_strategy)
            
            records_removed = len(df) - len(cleaned_df)
            
            result = CleansingResult(
                action=CleansingAction.DUPLICATE_REMOVAL,
                column_name=str(subset_columns) if subset_columns else "all_columns",
                records_affected=records_removed,
                original_values=duplicate_records.to_dict('records'),
                cleaned_values=[],  # Records were removed
                strategy_used=CleansingStrategy.REMOVE,
                success=True,
                statistics={
                    'duplicates_found': duplicate_count,
                    'records_removed': records_removed,
                    'keep_strategy': keep_strategy
                }
            )
            results.append(result)
        
        return cleaned_df, results


class FormatStandardizationCleaner(BaseCleaner):
    """Standardize data formats (phone numbers, emails, dates, etc.)."""
    
    def clean(self, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, List[CleansingResult]]:
        results = []
        cleaned_df = df.copy()
        
        # Get format standardization configurations
        for column, format_config in config.items():
            if column not in df.columns:
                continue
            
            format_type = format_config.get('type')
            
            if format_type == 'phone':
                cleaned_df, result = self._standardize_phone_numbers(cleaned_df, column, format_config)
                results.append(result)
            elif format_type == 'email':
                cleaned_df, result = self._standardize_emails(cleaned_df, column, format_config)
                results.append(result)
            elif format_type == 'date':
                cleaned_df, result = self._standardize_dates(cleaned_df, column, format_config)
                results.append(result)
            elif format_type == 'currency':
                cleaned_df, result = self._standardize_currency(cleaned_df, column, format_config)
                results.append(result)
        
        return cleaned_df, results
    
    def _standardize_phone_numbers(self, df: pd.DataFrame, column: str, 
                                  config: Dict[str, Any]) -> Tuple[pd.DataFrame, CleansingResult]:
        """Standardize phone number formats."""
        original_values = []
        cleaned_values = []
        records_affected = 0
        
        default_region = config.get('default_region', 'US')
        output_format = config.get('output_format', 'E164')  # E164, NATIONAL, INTERNATIONAL
        
        for idx, value in df[column].items():
            if pd.isna(value) or value == '':
                continue
            
            try:
                # Parse phone number
                parsed_number = phonenumbers.parse(str(value), default_region)
                
                if phonenumbers.is_valid_number(parsed_number):
                    # Format according to specification
                    if output_format == 'E164':
                        formatted = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.E164)
                    elif output_format == 'NATIONAL':
                        formatted = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.NATIONAL)
                    else:  # INTERNATIONAL
                        formatted = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
                    
                    if str(value) != formatted:
                        original_values.append(str(value))
                        cleaned_values.append(formatted)
                        df.loc[idx, column] = formatted
                        records_affected += 1
                
            except phonenumbers.NumberParseException:
                # Invalid phone number - optionally flag or remove
                if config.get('remove_invalid', False):
                    df.loc[idx, column] = None
                    records_affected += 1
        
        return df, CleansingResult(
            action=CleansingAction.FORMAT_STANDARDIZATION,
            column_name=column,
            records_affected=records_affected,
            original_values=original_values,
            cleaned_values=cleaned_values,
            strategy_used=CleansingStrategy.STANDARDIZE,
            success=True,
            statistics={'format_type': 'phone', 'output_format': output_format}
        )
    
    def _standardize_emails(self, df: pd.DataFrame, column: str, 
                           config: Dict[str, Any]) -> Tuple[pd.DataFrame, CleansingResult]:
        """Standardize email formats."""
        original_values = []
        cleaned_values = []
        records_affected = 0
        
        normalize_case = config.get('normalize_case', True)
        remove_invalid = config.get('remove_invalid', False)
        
        for idx, value in df[column].items():
            if pd.isna(value) or value == '':
                continue
            
            try:
                # Validate and normalize email
                validated_email = validate_email(str(value))
                normalized_email = validated_email.email
                
                if normalize_case:
                    normalized_email = normalized_email.lower()
                
                if str(value) != normalized_email:
                    original_values.append(str(value))
                    cleaned_values.append(normalized_email)
                    df.loc[idx, column] = normalized_email
                    records_affected += 1
                    
            except EmailNotValidError:
                if remove_invalid:
                    df.loc[idx, column] = None
                    records_affected += 1
        
        return df, CleansingResult(
            action=CleansingAction.FORMAT_STANDARDIZATION,
            column_name=column,
            records_affected=records_affected,
            original_values=original_values,
            cleaned_values=cleaned_values,
            strategy_used=CleansingStrategy.STANDARDIZE,
            success=True,
            statistics={'format_type': 'email', 'normalize_case': normalize_case}
        )
    
    def _standardize_dates(self, df: pd.DataFrame, column: str, 
                          config: Dict[str, Any]) -> Tuple[pd.DataFrame, CleansingResult]:
        """Standardize date formats."""
        original_values = []
        cleaned_values = []
        records_affected = 0
        
        output_format = config.get('output_format', '%Y-%m-%d')
        
        for idx, value in df[column].items():
            if pd.isna(value) or value == '':
                continue
            
            try:
                # Parse date with pandas (handles multiple formats)
                parsed_date = pd.to_datetime(value)
                formatted_date = parsed_date.strftime(output_format)
                
                if str(value) != formatted_date:
                    original_values.append(str(value))
                    cleaned_values.append(formatted_date)
                    df.loc[idx, column] = formatted_date
                    records_affected += 1
                    
            except (ValueError, TypeError):
                if config.get('remove_invalid', False):
                    df.loc[idx, column] = None
                    records_affected += 1
        
        return df, CleansingResult(
            action=CleansingAction.FORMAT_STANDARDIZATION,
            column_name=column,
            records_affected=records_affected,
            original_values=original_values,
            cleaned_values=cleaned_values,
            strategy_used=CleansingStrategy.STANDARDIZE,
            success=True,
            statistics={'format_type': 'date', 'output_format': output_format}
        )
    
    def _standardize_currency(self, df: pd.DataFrame, column: str, 
                             config: Dict[str, Any]) -> Tuple[pd.DataFrame, CleansingResult]:
        """Standardize currency formats."""
        original_values = []
        cleaned_values = []
        records_affected = 0
        
        currency_symbol = config.get('currency_symbol', '$')
        decimal_places = config.get('decimal_places', 2)
        
        for idx, value in df[column].items():
            if pd.isna(value) or value == '':
                continue
            
            try:
                # Extract numeric value from currency string
                numeric_str = re.sub(r'[^\d.-]', '', str(value))
                numeric_value = float(numeric_str)
                
                # Format as currency
                formatted_currency = f"{currency_symbol}{numeric_value:.{decimal_places}f}"
                
                if str(value) != formatted_currency:
                    original_values.append(str(value))
                    cleaned_values.append(formatted_currency)
                    df.loc[idx, column] = formatted_currency
                    records_affected += 1
                    
            except (ValueError, TypeError):
                if config.get('remove_invalid', False):
                    df.loc[idx, column] = None
                    records_affected += 1
        
        return df, CleansingResult(
            action=CleansingAction.FORMAT_STANDARDIZATION,
            column_name=column,
            records_affected=records_affected,
            original_values=original_values,
            cleaned_values=cleaned_values,
            strategy_used=CleansingStrategy.STANDARDIZE,
            success=True,
            statistics={'format_type': 'currency', 'currency_symbol': currency_symbol}
        )


class ValueNormalizationCleaner(BaseCleaner):
    """Normalize values for consistency."""
    
    def clean(self, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, List[CleansingResult]]:
        results = []
        cleaned_df = df.copy()
        
        for column, norm_config in config.items():
            if column not in df.columns:
                continue
            
            norm_type = norm_config.get('type')
            
            if norm_type == 'case':
                cleaned_df, result = self._normalize_case(cleaned_df, column, norm_config)
                results.append(result)
            elif norm_type == 'whitespace':
                cleaned_df, result = self._normalize_whitespace(cleaned_df, column, norm_config)
                results.append(result)
            elif norm_type == 'categorical':
                cleaned_df, result = self._normalize_categorical(cleaned_df, column, norm_config)
                results.append(result)
        
        return cleaned_df, results
    
    def _normalize_case(self, df: pd.DataFrame, column: str, 
                       config: Dict[str, Any]) -> Tuple[pd.DataFrame, CleansingResult]:
        """Normalize text case."""
        original_values = []
        cleaned_values = []
        records_affected = 0
        
        case_type = config.get('case_type', 'lower')  # 'lower', 'upper', 'title', 'sentence'
        
        for idx, value in df[column].items():
            if pd.isna(value) or not isinstance(value, str):
                continue
            
            if case_type == 'lower':
                normalized = value.lower()
            elif case_type == 'upper':
                normalized = value.upper()
            elif case_type == 'title':
                normalized = value.title()
            elif case_type == 'sentence':
                normalized = value.capitalize()
            else:
                normalized = value
            
            if value != normalized:
                original_values.append(value)
                cleaned_values.append(normalized)
                df.loc[idx, column] = normalized
                records_affected += 1
        
        return df, CleansingResult(
            action=CleansingAction.VALUE_NORMALIZATION,
            column_name=column,
            records_affected=records_affected,
            original_values=original_values,
            cleaned_values=cleaned_values,
            strategy_used=CleansingStrategy.STANDARDIZE,
            success=True,
            statistics={'normalization_type': 'case', 'case_type': case_type}
        )
    
    def _normalize_whitespace(self, df: pd.DataFrame, column: str, 
                             config: Dict[str, Any]) -> Tuple[pd.DataFrame, CleansingResult]:
        """Normalize whitespace in text."""
        original_values = []
        cleaned_values = []
        records_affected = 0
        
        for idx, value in df[column].items():
            if pd.isna(value) or not isinstance(value, str):
                continue
            
            # Remove leading/trailing whitespace and normalize internal whitespace
            normalized = re.sub(r'\s+', ' ', value.strip())
            
            if value != normalized:
                original_values.append(value)
                cleaned_values.append(normalized)
                df.loc[idx, column] = normalized
                records_affected += 1
        
        return df, CleansingResult(
            action=CleansingAction.VALUE_NORMALIZATION,
            column_name=column,
            records_affected=records_affected,
            original_values=original_values,
            cleaned_values=cleaned_values,
            strategy_used=CleansingStrategy.STANDARDIZE,
            success=True,
            statistics={'normalization_type': 'whitespace'}
        )
    
    def _normalize_categorical(self, df: pd.DataFrame, column: str, 
                              config: Dict[str, Any]) -> Tuple[pd.DataFrame, CleansingResult]:
        """Normalize categorical values using mapping."""
        original_values = []
        cleaned_values = []
        records_affected = 0
        
        # Get value mapping
        value_mapping = config.get('value_mapping', {})
        
        for idx, value in df[column].items():
            if pd.isna(value):
                continue
            
            str_value = str(value)
            if str_value in value_mapping:
                normalized = value_mapping[str_value]
                original_values.append(value)
                cleaned_values.append(normalized)
                df.loc[idx, column] = normalized
                records_affected += 1
        
        return df, CleansingResult(
            action=CleansingAction.VALUE_NORMALIZATION,
            column_name=column,
            records_affected=records_affected,
            original_values=original_values,
            cleaned_values=cleaned_values,
            strategy_used=CleansingStrategy.REPLACE,
            success=True,
            statistics={'normalization_type': 'categorical', 'mappings_applied': len(value_mapping)}
        )


class MissingValueImputationCleaner(BaseCleaner):
    """Impute missing values using various strategies."""
    
    def clean(self, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, List[CleansingResult]]:
        results = []
        cleaned_df = df.copy()
        
        for column, impute_config in config.items():
            if column not in df.columns:
                continue
            
            strategy = impute_config.get('strategy', 'mean')
            cleaned_df, result = self._impute_column(cleaned_df, column, strategy, impute_config)
            results.append(result)
        
        return cleaned_df, results
    
    def _impute_column(self, df: pd.DataFrame, column: str, strategy: str, 
                      config: Dict[str, Any]) -> Tuple[pd.DataFrame, CleansingResult]:
        """Impute missing values in a column."""
        original_missing_count = df[column].isnull().sum()
        
        if original_missing_count == 0:
            return df, CleansingResult(
                action=CleansingAction.MISSING_VALUE_IMPUTATION,
                column_name=column,
                records_affected=0,
                original_values=[],
                cleaned_values=[],
                strategy_used=CleansingStrategy.IMPUTE,
                success=True,
                statistics={'strategy': strategy, 'missing_count': 0}
            )
        
        missing_indices = df[df[column].isnull()].index.tolist()
        
        if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[column]):
            fill_value = df[column].mean()
        elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[column]):
            fill_value = df[column].median()
        elif strategy == 'mode':
            mode_values = df[column].mode()
            fill_value = mode_values[0] if len(mode_values) > 0 else None
        elif strategy == 'constant':
            fill_value = config.get('fill_value', 'Unknown')
        elif strategy == 'forward_fill':
            df[column] = df[column].fillna(method='ffill')
            fill_value = None
        elif strategy == 'backward_fill':
            df[column] = df[column].fillna(method='bfill')
            fill_value = None
        else:
            fill_value = 'Unknown'
        
        if fill_value is not None:
            df[column] = df[column].fillna(fill_value)
        
        return df, CleansingResult(
            action=CleansingAction.MISSING_VALUE_IMPUTATION,
            column_name=column,
            records_affected=original_missing_count,
            original_values=[None] * original_missing_count,
            cleaned_values=[fill_value] * original_missing_count if fill_value is not None else [],
            strategy_used=CleansingStrategy.IMPUTE,
            success=True,
            statistics={'strategy': strategy, 'fill_value': str(fill_value), 'missing_count': original_missing_count}
        )


class OutlierTreatmentCleaner(BaseCleaner):
    """Treat outliers in numerical data."""
    
    def clean(self, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, List[CleansingResult]]:
        results = []
        cleaned_df = df.copy()
        
        for column, outlier_config in config.items():
            if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
                continue
            
            method = outlier_config.get('method', 'iqr')
            action = outlier_config.get('action', 'cap')  # 'remove', 'cap', 'flag'
            
            cleaned_df, result = self._treat_outliers(cleaned_df, column, method, action, outlier_config)
            results.append(result)
        
        return cleaned_df, results
    
    def _treat_outliers(self, df: pd.DataFrame, column: str, method: str, action: str, 
                       config: Dict[str, Any]) -> Tuple[pd.DataFrame, CleansingResult]:
        """Treat outliers in a numerical column."""
        original_values = []
        cleaned_values = []
        records_affected = 0
        
        # Detect outliers
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            multiplier = config.get('iqr_multiplier', 1.5)
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            threshold = config.get('zscore_threshold', 3)
            outlier_mask = z_scores > threshold
        
        else:
            # No outliers detected for unknown method
            outlier_mask = pd.Series([False] * len(df), index=df.index)
        
        outlier_indices = df[outlier_mask].index.tolist()
        
        if len(outlier_indices) > 0:
            original_values = df.loc[outlier_indices, column].tolist()
            
            if action == 'remove':
                df = df[~outlier_mask]
                cleaned_values = []  # Values were removed
                records_affected = len(outlier_indices)
            
            elif action == 'cap':
                if method == 'iqr':
                    df.loc[df[column] < lower_bound, column] = lower_bound
                    df.loc[df[column] > upper_bound, column] = upper_bound
                elif method == 'zscore':
                    # Cap to mean ± 3 standard deviations
                    mean = df[column].mean()
                    std = df[column].std()
                    threshold = config.get('zscore_threshold', 3)
                    df.loc[df[column] < mean - threshold * std, column] = mean - threshold * std
                    df.loc[df[column] > mean + threshold * std, column] = mean + threshold * std
                
                cleaned_values = df.loc[outlier_indices, column].tolist()
                records_affected = len(outlier_indices)
            
            elif action == 'flag':
                # Add flag column
                flag_column = f"{column}_outlier_flag"
                df[flag_column] = outlier_mask
                cleaned_values = [True] * len(outlier_indices)
                records_affected = len(outlier_indices)
        
        return df, CleansingResult(
            action=CleansingAction.OUTLIER_TREATMENT,
            column_name=column,
            records_affected=records_affected,
            original_values=original_values,
            cleaned_values=cleaned_values,
            strategy_used=CleansingStrategy.REPLACE if action == 'cap' else CleansingStrategy.REMOVE if action == 'remove' else CleansingStrategy.FLAG,
            success=True,
            statistics={'method': method, 'action': action, 'outliers_found': len(outlier_indices)}
        )


class DataTypeConversionCleaner(BaseCleaner):
    """Convert data types for consistency."""
    
    def clean(self, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, List[CleansingResult]]:
        results = []
        cleaned_df = df.copy()
        
        for column, conversion_config in config.items():
            if column not in df.columns:
                continue
            
            target_type = conversion_config.get('target_type')
            cleaned_df, result = self._convert_column_type(cleaned_df, column, target_type, conversion_config)
            results.append(result)
        
        return cleaned_df, results
    
    def _convert_column_type(self, df: pd.DataFrame, column: str, target_type: str, 
                            config: Dict[str, Any]) -> Tuple[pd.DataFrame, CleansingResult]:
        """Convert column to target data type."""
        original_type = str(df[column].dtype)
        records_affected = 0
        errors_count = 0
        
        try:
            if target_type == 'int':
                df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
            elif target_type == 'float':
                df[column] = pd.to_numeric(df[column], errors='coerce')
            elif target_type == 'string':
                df[column] = df[column].astype(str)
            elif target_type == 'datetime':
                df[column] = pd.to_datetime(df[column], errors='coerce')
            elif target_type == 'category':
                df[column] = df[column].astype('category')
            elif target_type == 'boolean':
                df[column] = df[column].astype(bool)
            
            # Count conversion errors (NaN values introduced)
            if target_type in ['int', 'float', 'datetime']:
                errors_count = df[column].isnull().sum()
            
            records_affected = len(df)
            
            return df, CleansingResult(
                action=CleansingAction.DATA_TYPE_CONVERSION,
                column_name=column,
                records_affected=records_affected,
                original_values=[original_type],
                cleaned_values=[str(df[column].dtype)],
                strategy_used=CleansingStrategy.REPLACE,
                success=True,
                statistics={
                    'original_type': original_type,
                    'target_type': target_type,
                    'conversion_errors': errors_count
                }
            )
        
        except Exception as e:
            return df, CleansingResult(
                action=CleansingAction.DATA_TYPE_CONVERSION,
                column_name=column,
                records_affected=0,
                original_values=[original_type],
                cleaned_values=[original_type],
                strategy_used=CleansingStrategy.FLAG,
                success=False,
                error_message=str(e),
                statistics={'original_type': original_type, 'target_type': target_type}
            )


class TextCleaningCleaner(BaseCleaner):
    """Clean text data."""
    
    def clean(self, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, List[CleansingResult]]:
        results = []
        cleaned_df = df.copy()
        
        for column, text_config in config.items():
            if column not in df.columns:
                continue
            
            cleaned_df, result = self._clean_text_column(cleaned_df, column, text_config)
            results.append(result)
        
        return cleaned_df, results
    
    def _clean_text_column(self, df: pd.DataFrame, column: str, 
                          config: Dict[str, Any]) -> Tuple[pd.DataFrame, CleansingResult]:
        """Clean text in a column."""
        original_values = []
        cleaned_values = []
        records_affected = 0
        
        remove_html = config.get('remove_html', False)
        remove_urls = config.get('remove_urls', False)
        remove_emails = config.get('remove_emails', False)
        remove_special_chars = config.get('remove_special_chars', False)
        normalize_unicode = config.get('normalize_unicode', False)
        
        for idx, value in df[column].items():
            if pd.isna(value) or not isinstance(value, str):
                continue
            
            original_value = value
            cleaned_value = value
            
            if remove_html:
                cleaned_value = re.sub(r'<[^>]+>', '', cleaned_value)
            
            if remove_urls:
                cleaned_value = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned_value)
            
            if remove_emails:
                cleaned_value = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', cleaned_value)
            
            if remove_special_chars:
                cleaned_value = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned_value)
            
            if normalize_unicode:
                import unicodedata
                cleaned_value = unicodedata.normalize('NFKD', cleaned_value)
            
            # Clean up extra whitespace
            cleaned_value = re.sub(r'\s+', ' ', cleaned_value.strip())
            
            if original_value != cleaned_value:
                original_values.append(original_value)
                cleaned_values.append(cleaned_value)
                df.loc[idx, column] = cleaned_value
                records_affected += 1
        
        return df, CleansingResult(
            action=CleansingAction.TEXT_CLEANING,
            column_name=column,
            records_affected=records_affected,
            original_values=original_values,
            cleaned_values=cleaned_values,
            strategy_used=CleansingStrategy.STANDARDIZE,
            success=True,
            statistics={
                'remove_html': remove_html,
                'remove_urls': remove_urls,
                'remove_emails': remove_emails,
                'remove_special_chars': remove_special_chars,
                'normalize_unicode': normalize_unicode
            }
        )


class DateStandardizationCleaner(BaseCleaner):
    """Standardize date formats and handle date parsing."""
    
    def clean(self, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, List[CleansingResult]]:
        results = []
        cleaned_df = df.copy()
        
        for column, date_config in config.items():
            if column not in df.columns:
                continue
            
            cleaned_df, result = self._standardize_dates_column(cleaned_df, column, date_config)
            results.append(result)
        
        return cleaned_df, results
    
    def _standardize_dates_column(self, df: pd.DataFrame, column: str, 
                                 config: Dict[str, Any]) -> Tuple[pd.DataFrame, CleansingResult]:
        """Standardize dates in a column."""
        original_values = []
        cleaned_values = []
        records_affected = 0
        errors_count = 0
        
        target_format = config.get('target_format', '%Y-%m-%d')
        infer_format = config.get('infer_format', True)
        
        for idx, value in df[column].items():
            if pd.isna(value) or value == '':
                continue
            
            try:
                # Parse date
                if infer_format:
                    parsed_date = pd.to_datetime(value, infer_datetime_format=True)
                else:
                    source_format = config.get('source_format')
                    if source_format:
                        parsed_date = pd.to_datetime(value, format=source_format)
                    else:
                        parsed_date = pd.to_datetime(value)
                
                # Format date
                formatted_date = parsed_date.strftime(target_format)
                
                if str(value) != formatted_date:
                    original_values.append(str(value))
                    cleaned_values.append(formatted_date)
                    df.loc[idx, column] = formatted_date
                    records_affected += 1
                    
            except (ValueError, TypeError):
                errors_count += 1
                if config.get('remove_invalid', False):
                    df.loc[idx, column] = None
        
        return df, CleansingResult(
            action=CleansingAction.DATE_STANDARDIZATION,
            column_name=column,
            records_affected=records_affected,
            original_values=original_values,
            cleaned_values=cleaned_values,
            strategy_used=CleansingStrategy.STANDARDIZE,
            success=True,
            statistics={
                'target_format': target_format,
                'infer_format': infer_format,
                'parsing_errors': errors_count
            }
        )