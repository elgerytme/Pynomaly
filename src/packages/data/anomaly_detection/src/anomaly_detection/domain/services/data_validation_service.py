"""Data validation service for quality checks and schema validation."""

import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import jsonschema
import structlog

logger = structlog.get_logger(__name__)


class DataValidationService:
    """Service for comprehensive data validation and quality checks."""
    
    def __init__(self):
        self.validation_rules = self._load_default_rules()
    
    def _load_default_rules(self) -> Dict[str, Any]:
        """Load default validation rules."""
        return {
            'max_missing_percentage': 50.0,
            'max_duplicate_percentage': 20.0,
            'min_numeric_columns': 1,
            'max_outlier_percentage': 10.0,
            'required_row_count': 10
        }
    
    async def validate_file(
        self,
        file_path: Path,
        schema_file: Optional[Path] = None,
        check_types: bool = True,
        check_missing: bool = True,
        check_outliers: bool = True,
        check_duplicates: bool = True,
        custom_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate a single data file."""
        
        validation_result = {
            'file_path': str(file_path),
            'timestamp': datetime.utcnow().isoformat(),
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {},
            'checks_performed': []
        }
        
        try:
            logger.info("Starting file validation", file=str(file_path))
            
            # Load data
            data = await self._load_data(file_path)
            
            # Basic statistics
            validation_result['statistics'] = await self._generate_statistics(data)
            
            # Schema validation
            if schema_file:
                schema_result = await self._validate_schema(data, schema_file)
                validation_result['errors'].extend(schema_result['errors'])
                validation_result['warnings'].extend(schema_result['warnings'])
                validation_result['checks_performed'].append('schema_validation')
            
            # Data type validation
            if check_types:
                type_result = await self._validate_data_types(data)
                validation_result['errors'].extend(type_result['errors'])
                validation_result['warnings'].extend(type_result['warnings'])
                validation_result['checks_performed'].append('data_types')
            
            # Missing values check
            if check_missing:
                missing_result = await self._check_missing_values(data, custom_rules)
                validation_result['errors'].extend(missing_result['errors'])
                validation_result['warnings'].extend(missing_result['warnings'])
                validation_result['checks_performed'].append('missing_values')
            
            # Duplicate check
            if check_duplicates:
                duplicate_result = await self._check_duplicates(data, custom_rules)
                validation_result['errors'].extend(duplicate_result['errors'])
                validation_result['warnings'].extend(duplicate_result['warnings'])
                validation_result['checks_performed'].append('duplicates')
            
            # Outlier detection
            if check_outliers:
                outlier_result = await self._check_outliers(data, custom_rules)
                validation_result['errors'].extend(outlier_result['errors'])
                validation_result['warnings'].extend(outlier_result['warnings'])
                validation_result['checks_performed'].append('outliers')
            
            # Data consistency checks
            consistency_result = await self._check_data_consistency(data)
            validation_result['errors'].extend(consistency_result['errors'])
            validation_result['warnings'].extend(consistency_result['warnings'])
            validation_result['checks_performed'].append('consistency')
            
            # Overall validation status
            validation_result['is_valid'] = len(validation_result['errors']) == 0
            
            logger.info("File validation completed", 
                       file=str(file_path),
                       is_valid=validation_result['is_valid'],
                       errors=len(validation_result['errors']),
                       warnings=len(validation_result['warnings']))
            
            return validation_result
            
        except Exception as e:
            logger.error("File validation failed", file=str(file_path), error=str(e))
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Validation failed: {str(e)}")
            return validation_result
    
    async def _load_data(self, file_path: Path) -> pd.DataFrame:
        """Load data from file."""
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.csv':
                return pd.read_csv(file_path)
            elif file_extension == '.json':
                return pd.read_json(file_path)
            elif file_extension == '.parquet':
                return pd.read_parquet(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            raise ValueError(f"Failed to load data: {str(e)}")
    
    async def _generate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic data statistics."""
        return {
            'row_count': len(data),
            'column_count': len(data.columns),
            'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(data.select_dtypes(include=['object']).columns),
            'datetime_columns': len(data.select_dtypes(include=['datetime64']).columns),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_values_total': data.isnull().sum().sum(),
            'duplicate_rows': data.duplicated().sum()
        }
    
    async def _validate_schema(
        self, 
        data: pd.DataFrame, 
        schema_file: Path
    ) -> Dict[str, List[str]]:
        """Validate data against JSON schema."""
        result = {'errors': [], 'warnings': []}
        
        try:
            # Load schema
            with open(schema_file, 'r') as f:
                schema = json.load(f)
            
            # Convert DataFrame to records for validation
            records = data.to_dict('records')
            
            # Validate each record (sample first few for performance)
            sample_size = min(100, len(records))
            validation_errors = []
            
            for i, record in enumerate(records[:sample_size]):
                try:
                    jsonschema.validate(record, schema)
                except jsonschema.ValidationError as e:
                    validation_errors.append(f"Row {i}: {e.message}")
                    if len(validation_errors) >= 10:  # Limit error count
                        break
            
            if validation_errors:
                result['errors'].extend(validation_errors)
                if len(records) > sample_size:
                    result['warnings'].append(
                        f"Schema validation performed on sample of {sample_size} rows"
                    )
            
        except Exception as e:
            result['errors'].append(f"Schema validation failed: {str(e)}")
        
        return result
    
    async def _validate_data_types(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate data types and detect type inconsistencies."""
        result = {'errors': [], 'warnings': []}
        
        # Check for mixed types in columns
        for column in data.columns:
            try:
                # Try to infer the most appropriate type
                series = data[column].dropna()
                if len(series) == 0:
                    continue
                
                # Check for mixed numeric/string types
                numeric_count = 0
                string_count = 0
                
                for value in series.head(100):  # Sample for performance
                    try:
                        float(value)
                        numeric_count += 1
                    except (ValueError, TypeError):
                        string_count += 1
                
                if numeric_count > 0 and string_count > 0:
                    ratio = min(numeric_count, string_count) / len(series.head(100))
                    if ratio > 0.1:  # More than 10% mixed types
                        result['warnings'].append(
                            f"Column '{column}' has mixed numeric/string types ({ratio:.1%})"
                        )
                
                # Check for potential datetime columns stored as strings
                if data[column].dtype == 'object':
                    sample_values = series.head(10).astype(str)
                    datetime_like = 0
                    
                    for value in sample_values:
                        if self._looks_like_datetime(value):
                            datetime_like += 1
                    
                    if datetime_like > len(sample_values) * 0.8:
                        result['warnings'].append(
                            f"Column '{column}' appears to contain datetime values but is stored as text"
                        )
                
            except Exception as e:
                result['warnings'].append(f"Type validation failed for column '{column}': {str(e)}")
        
        # Check for minimum numeric columns requirement
        numeric_columns = len(data.select_dtypes(include=[np.number]).columns)
        min_required = self.validation_rules.get('min_numeric_columns', 1)
        
        if numeric_columns < min_required:
            result['errors'].append(
                f"Insufficient numeric columns: found {numeric_columns}, required {min_required}"
            )
        
        return result
    
    async def _check_missing_values(
        self, 
        data: pd.DataFrame, 
        custom_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[str]]:
        """Check for missing values and validate against thresholds."""
        result = {'errors': [], 'warnings': []}
        
        rules = {**self.validation_rules, **(custom_rules or {})}
        max_missing_pct = rules.get('max_missing_percentage', 50.0)
        
        total_cells = len(data) * len(data.columns)
        missing_cells = data.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        # Overall missing data check
        if missing_percentage > max_missing_pct:
            result['errors'].append(
                f"High missing data percentage: {missing_percentage:.1f}% (threshold: {max_missing_pct}%)"
            )
        elif missing_percentage > max_missing_pct * 0.7:  # Warning at 70% of threshold
            result['warnings'].append(
                f"Moderate missing data percentage: {missing_percentage:.1f}%"
            )
        
        # Per-column missing data check
        for column in data.columns:
            missing_count = data[column].isnull().sum()
            column_missing_pct = (missing_count / len(data)) * 100 if len(data) > 0 else 0
            
            if column_missing_pct > max_missing_pct:
                result['errors'].append(
                    f"Column '{column}' has {column_missing_pct:.1f}% missing values (threshold: {max_missing_pct}%)"
                )
            elif column_missing_pct > max_missing_pct * 0.7:
                result['warnings'].append(
                    f"Column '{column}' has {column_missing_pct:.1f}% missing values"
                )
        
        return result
    
    async def _check_duplicates(
        self, 
        data: pd.DataFrame, 
        custom_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[str]]:
        """Check for duplicate rows."""
        result = {'errors': [], 'warnings': []}
        
        rules = {**self.validation_rules, **(custom_rules or {})}
        max_duplicate_pct = rules.get('max_duplicate_percentage', 20.0)
        
        duplicate_count = data.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(data)) * 100 if len(data) > 0 else 0
        
        if duplicate_percentage > max_duplicate_pct:
            result['errors'].append(
                f"High duplicate row percentage: {duplicate_percentage:.1f}% ({duplicate_count} rows, threshold: {max_duplicate_pct}%)"
            )
        elif duplicate_percentage > max_duplicate_pct * 0.5:
            result['warnings'].append(
                f"Moderate duplicate row percentage: {duplicate_percentage:.1f}% ({duplicate_count} rows)"
            )
        
        # Check for columns with high duplicate rates (potential ID columns)
        for column in data.columns:
            unique_count = data[column].nunique()
            duplicate_col_pct = (1 - unique_count / len(data)) * 100 if len(data) > 0 else 0
            
            if duplicate_col_pct > 90 and unique_count > 1:  # Very high duplication but not constant
                result['warnings'].append(
                    f"Column '{column}' has very low uniqueness: {unique_count} unique values ({duplicate_col_pct:.1f}% duplicates)"
                )
        
        return result
    
    async def _check_outliers(
        self, 
        data: pd.DataFrame, 
        custom_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[str]]:
        """Check for statistical outliers in numeric columns."""
        result = {'errors': [], 'warnings': []}
        
        rules = {**self.validation_rules, **(custom_rules or {})}
        max_outlier_pct = rules.get('max_outlier_percentage', 10.0)
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            try:
                series = data[column].dropna()
                if len(series) < 10:  # Need minimum data for outlier detection
                    continue
                
                # Use IQR method for outlier detection
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR == 0:  # No variation in data
                    result['warnings'].append(f"Column '{column}' has no variation (constant values)")
                    continue
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                outlier_percentage = (len(outliers) / len(series)) * 100
                
                if outlier_percentage > max_outlier_pct:
                    result['warnings'].append(
                        f"Column '{column}' has {outlier_percentage:.1f}% outliers ({len(outliers)} values, threshold: {max_outlier_pct}%)"
                    )
                
                # Check for extreme outliers (beyond 3 * IQR)
                extreme_lower = Q1 - 3 * IQR
                extreme_upper = Q3 + 3 * IQR
                extreme_outliers = series[(series < extreme_lower) | (series > extreme_upper)]
                
                if len(extreme_outliers) > 0:
                    extreme_pct = (len(extreme_outliers) / len(series)) * 100
                    result['warnings'].append(
                        f"Column '{column}' has {extreme_pct:.1f}% extreme outliers ({len(extreme_outliers)} values)"
                    )
                
            except Exception as e:
                result['warnings'].append(f"Outlier detection failed for column '{column}': {str(e)}")
        
        return result
    
    async def _check_data_consistency(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Check for data consistency issues."""
        result = {'errors': [], 'warnings': []}
        
        # Check minimum row count
        min_rows = self.validation_rules.get('required_row_count', 10)
        if len(data) < min_rows:
            result['errors'].append(f"Insufficient data: {len(data)} rows (minimum: {min_rows})")
        
        # Check for completely empty columns
        empty_columns = [col for col in data.columns if data[col].isnull().all()]
        if empty_columns:
            result['errors'].append(f"Completely empty columns found: {', '.join(empty_columns)}")
        
        # Check for single-value columns (no variation)
        constant_columns = []
        for column in data.columns:
            unique_values = data[column].dropna().nunique()
            if unique_values <= 1 and not data[column].isnull().all():
                constant_columns.append(column)
        
        if constant_columns:
            result['warnings'].append(f"Constant value columns (no variation): {', '.join(constant_columns)}")
        
        # Check for unrealistic date ranges if datetime columns exist
        datetime_columns = data.select_dtypes(include=['datetime64']).columns
        for col in datetime_columns:
            try:
                min_date = data[col].min()
                max_date = data[col].max()
                
                # Check for dates far in the future or past
                current_year = datetime.now().year
                if min_date and min_date.year < 1900:
                    result['warnings'].append(f"Column '{col}' has very old dates (before 1900)")
                if max_date and max_date.year > current_year + 10:
                    result['warnings'].append(f"Column '{col}' has dates far in the future")
                    
            except Exception:
                pass  # Skip datetime validation if conversion fails
        
        return result
    
    def _looks_like_datetime(self, value: str) -> bool:
        """Check if a string value looks like a datetime."""
        datetime_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
        ]
        
        import re
        for pattern in datetime_patterns:
            if re.search(pattern, str(value)):
                return True
        return False
    
    async def validate_multiple_files(
        self,
        file_paths: List[Path],
        **validation_options
    ) -> Dict[str, Any]:
        """Validate multiple files in parallel."""
        
        logger.info("Starting batch file validation", files=len(file_paths))
        
        # Create validation tasks
        tasks = []
        for file_path in file_paths:
            task = asyncio.create_task(
                self.validate_file(file_path, **validation_options)
            )
            tasks.append(task)
        
        # Wait for all validations to complete
        results = []
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
            except Exception as e:
                logger.error("File validation task failed", error=str(e))
                results.append({
                    'is_valid': False,
                    'errors': [f"Validation task failed: {str(e)}"],
                    'warnings': [],
                    'file_path': 'unknown'
                })
        
        # Generate summary
        total_files = len(results)
        valid_files = sum(1 for r in results if r['is_valid'])
        total_errors = sum(len(r['errors']) for r in results)
        total_warnings = sum(len(r['warnings']) for r in results)
        
        summary = {
            'batch_validation_summary': {
                'total_files': total_files,
                'valid_files': valid_files,
                'invalid_files': total_files - valid_files,
                'total_errors': total_errors,
                'total_warnings': total_warnings,
                'success_rate': (valid_files / total_files) * 100 if total_files > 0 else 0,
                'timestamp': datetime.utcnow().isoformat()
            },
            'file_results': results
        }
        
        logger.info("Batch validation completed", 
                   total_files=total_files,
                   valid_files=valid_files,
                   success_rate=summary['batch_validation_summary']['success_rate'])
        
        return summary