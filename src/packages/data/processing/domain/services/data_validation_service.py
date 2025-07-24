"""Data validation service for ensuring data quality and integrity."""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import structlog

logger = structlog.get_logger()


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Validation category types."""
    MISSING_DATA = "missing_data"
    DATA_TYPE = "data_type"
    RANGE = "range"
    DISTRIBUTION = "distribution"
    SCHEMA = "schema"
    QUALITY = "quality"
    CONSISTENCY = "consistency"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    column: Optional[str] = None
    row_indices: Optional[List[int]] = None
    value_count: Optional[int] = None
    percentage: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationRule:
    """Data validation rule."""
    name: str
    category: ValidationCategory
    severity: ValidationSeverity
    description: str
    enabled: bool = True
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    dataset_name: str
    total_rows: int
    total_columns: int
    validation_timestamp: datetime
    overall_score: float  # 0-100
    issues: List[ValidationIssue]
    summary_stats: Dict[str, Any]
    passed_rules: List[str]
    failed_rules: List[str]
    recommendations: List[str]


class DataValidationService:
    """Service for comprehensive data validation."""
    
    def __init__(self):
        self.default_rules = self._create_default_rules()
        self.custom_rules: List[ValidationRule] = []
    
    def _create_default_rules(self) -> List[ValidationRule]:
        """Create default validation rules."""
        return [
            ValidationRule(
                name="missing_data_check",
                category=ValidationCategory.MISSING_DATA,
                severity=ValidationSeverity.WARNING,
                description="Check for missing values in dataset",
                parameters={"max_missing_percentage": 20.0}
            ),
            ValidationRule(
                name="data_type_consistency",
                category=ValidationCategory.DATA_TYPE,
                severity=ValidationSeverity.ERROR,
                description="Ensure consistent data types within columns"
            ),
            ValidationRule(
                name="numeric_range_check",
                category=ValidationCategory.RANGE,
                severity=ValidationSeverity.WARNING,
                description="Check for outliers and range violations in numeric columns"
            ),
            ValidationRule(
                name="distribution_check",
                category=ValidationCategory.DISTRIBUTION,
                severity=ValidationSeverity.INFO,
                description="Analyze data distributions for anomalies"
            ),
            ValidationRule(
                name="schema_validation",
                category=ValidationCategory.SCHEMA,
                severity=ValidationSeverity.CRITICAL,
                description="Validate data schema and structure"
            ),
            ValidationRule(
                name="duplicate_check",
                category=ValidationCategory.QUALITY,
                severity=ValidationSeverity.WARNING,
                description="Check for duplicate rows",
                parameters={"max_duplicate_percentage": 10.0}
            ),
            ValidationRule(
                name="column_correlation",
                category=ValidationCategory.CONSISTENCY,
                severity=ValidationSeverity.INFO,
                description="Check for highly correlated columns"
            )
        ]
    
    async def validate_dataset(
        self,
        data: Union[pd.DataFrame, str, Dict[str, Any]],
        dataset_name: str = "dataset",
        schema: Optional[Dict[str, Any]] = None,
        custom_rules: Optional[List[ValidationRule]] = None
    ) -> ValidationReport:
        """Perform comprehensive dataset validation."""
        
        # Convert input to DataFrame
        df = self._prepare_dataframe(data)
        
        # Combine default and custom rules
        all_rules = self.default_rules + (custom_rules or []) + self.custom_rules
        enabled_rules = [rule for rule in all_rules if rule.enabled]
        
        issues: List[ValidationIssue] = []
        passed_rules: List[str] = []
        failed_rules: List[str] = []
        
        logger.info("Starting dataset validation", 
                   dataset_name=dataset_name,
                   rows=len(df),
                   columns=len(df.columns))
        
        # Run validation rules
        for rule in enabled_rules:
            try:
                rule_issues = await self._apply_validation_rule(df, rule, schema)
                
                if rule_issues:
                    issues.extend(rule_issues)
                    failed_rules.append(rule.name)
                else:
                    passed_rules.append(rule.name)
                    
            except Exception as e:
                logger.error("Validation rule failed", 
                           rule=rule.name, 
                           error=str(e))
                issues.append(ValidationIssue(
                    category=ValidationCategory.QUALITY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Validation rule '{rule.name}' failed: {str(e)}",
                    details={"rule": rule.name, "error": str(e)}
                ))
                failed_rules.append(rule.name)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(issues, len(df), len(df.columns))
        
        # Generate summary statistics
        summary_stats = self._generate_summary_stats(df, issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, df)
        
        report = ValidationReport(
            dataset_name=dataset_name,
            total_rows=len(df),
            total_columns=len(df.columns),
            validation_timestamp=datetime.now(),
            overall_score=overall_score,
            issues=issues,
            summary_stats=summary_stats,
            passed_rules=passed_rules,
            failed_rules=failed_rules,
            recommendations=recommendations
        )
        
        logger.info("Dataset validation completed", 
                   dataset_name=dataset_name,
                   overall_score=overall_score,
                   total_issues=len(issues))
        
        return report
    
    async def _apply_validation_rule(
        self,
        df: pd.DataFrame,
        rule: ValidationRule,
        schema: Optional[Dict[str, Any]]
    ) -> List[ValidationIssue]:
        """Apply a single validation rule."""
        
        if rule.name == "missing_data_check":
            return self._check_missing_data(df, rule)
        elif rule.name == "data_type_consistency":
            return self._check_data_type_consistency(df, rule)
        elif rule.name == "numeric_range_check":
            return self._check_numeric_ranges(df, rule)
        elif rule.name == "distribution_check":
            return self._check_distributions(df, rule)
        elif rule.name == "schema_validation":
            return self._validate_schema(df, rule, schema)
        elif rule.name == "duplicate_check":
            return self._check_duplicates(df, rule)
        elif rule.name == "column_correlation":
            return self._check_correlations(df, rule)
        else:
            logger.warning("Unknown validation rule", rule_name=rule.name)
            return []
    
    def _check_missing_data(self, df: pd.DataFrame, rule: ValidationRule) -> List[ValidationIssue]:
        """Check for missing data issues."""
        issues = []
        max_missing_pct = rule.parameters.get("max_missing_percentage", 20.0) if rule.parameters else 20.0
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            if missing_count > 0:
                severity = rule.severity
                if missing_pct > max_missing_pct:
                    severity = ValidationSeverity.ERROR
                
                issues.append(ValidationIssue(
                    category=rule.category,
                    severity=severity,
                    message=f"Column '{column}' has {missing_count} missing values ({missing_pct:.1f}%)",
                    column=column,
                    value_count=missing_count,
                    percentage=missing_pct,
                    details={"threshold": max_missing_pct}
                ))
        
        return issues
    
    def _check_data_type_consistency(self, df: pd.DataFrame, rule: ValidationRule) -> List[ValidationIssue]:
        """Check for data type consistency issues."""
        issues = []
        
        for column in df.columns:
            col_data = df[column].dropna()
            if len(col_data) == 0:
                continue
            
            # Check for mixed types
            types_found = set(type(val).__name__ for val in col_data.head(1000))  # Sample for performance
            
            if len(types_found) > 1:
                issues.append(ValidationIssue(
                    category=rule.category,
                    severity=rule.severity,
                    message=f"Column '{column}' has mixed data types: {', '.join(types_found)}",
                    column=column,
                    details={"types_found": list(types_found)}
                ))
            
            # Check for potential type mismatches
            if df[column].dtype == 'object':
                numeric_convertible = pd.to_numeric(col_data, errors='coerce').notna().sum()
                if numeric_convertible > len(col_data) * 0.8:  # 80% convertible
                    issues.append(ValidationIssue(
                        category=rule.category,
                        severity=ValidationSeverity.INFO,
                        message=f"Column '{column}' appears to be numeric but stored as text",
                        column=column,
                        details={"convertible_percentage": (numeric_convertible / len(col_data)) * 100}
                    ))
        
        return issues
    
    def _check_numeric_ranges(self, df: pd.DataFrame, rule: ValidationRule) -> List[ValidationIssue]:
        """Check numeric columns for range issues and outliers."""
        issues = []
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            col_data = df[column].dropna()
            if len(col_data) == 0:
                continue
            
            # Check for infinite values
            inf_count = np.isinf(col_data).sum()
            if inf_count > 0:
                issues.append(ValidationIssue(
                    category=rule.category,
                    severity=ValidationSeverity.ERROR,
                    message=f"Column '{column}' contains {inf_count} infinite values",
                    column=column,
                    value_count=inf_count
                ))
            
            # Check for outliers using IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
            outlier_pct = (outliers / len(col_data)) * 100
            
            if outlier_pct > 5.0:  # More than 5% outliers
                issues.append(ValidationIssue(
                    category=rule.category,
                    severity=rule.severity,
                    message=f"Column '{column}' has {outliers} potential outliers ({outlier_pct:.1f}%)",
                    column=column,
                    value_count=outliers,
                    percentage=outlier_pct,
                    details={
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                        "IQR": IQR
                    }
                ))
        
        return issues
    
    def _check_distributions(self, df: pd.DataFrame, rule: ValidationRule) -> List[ValidationIssue]:
        """Check data distributions for anomalies."""
        issues = []
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            col_data = df[column].dropna()
            if len(col_data) < 10:  # Need sufficient data
                continue
            
            # Check for constant values
            if col_data.nunique() == 1:
                issues.append(ValidationIssue(
                    category=rule.category,
                    severity=ValidationSeverity.WARNING,
                    message=f"Column '{column}' has constant values",
                    column=column,
                    details={"unique_value": col_data.iloc[0]}
                ))
            
            # Check for highly skewed distributions
            try:
                from scipy import stats
                skewness = stats.skew(col_data)
                if abs(skewness) > 2:  # Highly skewed
                    issues.append(ValidationIssue(
                        category=rule.category,
                        severity=ValidationSeverity.INFO,
                        message=f"Column '{column}' is highly skewed (skewness: {skewness:.2f})",
                        column=column,
                        details={"skewness": skewness}
                    ))
            except ImportError:
                pass  # Skip if scipy not available
        
        return issues
    
    def _validate_schema(
        self,
        df: pd.DataFrame,
        rule: ValidationRule,
        schema: Optional[Dict[str, Any]]
    ) -> List[ValidationIssue]:
        """Validate data against expected schema."""
        issues = []
        
        if not schema:
            return issues
        
        expected_columns = schema.get("columns", {})
        required_columns = schema.get("required_columns", [])
        
        # Check for missing required columns
        missing_required = set(required_columns) - set(df.columns)
        for col in missing_required:
            issues.append(ValidationIssue(
                category=rule.category,
                severity=ValidationSeverity.CRITICAL,
                message=f"Required column '{col}' is missing",
                column=col
            ))
        
        # Check for unexpected columns
        unexpected_columns = set(df.columns) - set(expected_columns.keys())
        for col in unexpected_columns:
            issues.append(ValidationIssue(
                category=rule.category,
                severity=ValidationSeverity.WARNING,
                message=f"Unexpected column '{col}' found",
                column=col
            ))
        
        # Check column types
        for col, expected_info in expected_columns.items():
            if col not in df.columns:
                continue
            
            expected_type = expected_info.get("type")
            if expected_type:
                actual_type = str(df[col].dtype)
                if not self._types_compatible(actual_type, expected_type):
                    issues.append(ValidationIssue(
                        category=rule.category,
                        severity=ValidationSeverity.ERROR,
                        message=f"Column '{col}' has type '{actual_type}', expected '{expected_type}'",
                        column=col,
                        details={"expected_type": expected_type, "actual_type": actual_type}
                    ))
        
        return issues
    
    def _check_duplicates(self, df: pd.DataFrame, rule: ValidationRule) -> List[ValidationIssue]:
        """Check for duplicate rows."""
        issues = []
        max_duplicate_pct = rule.parameters.get("max_duplicate_percentage", 10.0) if rule.parameters else 10.0
        
        duplicate_count = df.duplicated().sum()
        duplicate_pct = (duplicate_count / len(df)) * 100
        
        if duplicate_count > 0:
            severity = rule.severity
            if duplicate_pct > max_duplicate_pct:
                severity = ValidationSeverity.ERROR
            
            issues.append(ValidationIssue(
                category=rule.category,
                severity=severity,
                message=f"Found {duplicate_count} duplicate rows ({duplicate_pct:.1f}%)",
                value_count=duplicate_count,
                percentage=duplicate_pct,
                details={"threshold": max_duplicate_pct}
            ))
        
        return issues
    
    def _check_correlations(self, df: pd.DataFrame, rule: ValidationRule) -> List[ValidationIssue]:
        """Check for highly correlated columns."""
        issues = []
        
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return issues
        
        try:
            corr_matrix = numeric_df.corr()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.9:  # High correlation threshold
                        high_corr_pairs.append((
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_val
                        ))
            
            for col1, col2, corr_val in high_corr_pairs:
                issues.append(ValidationIssue(
                    category=rule.category,
                    severity=rule.severity,
                    message=f"Columns '{col1}' and '{col2}' are highly correlated ({corr_val:.3f})",
                    details={"column1": col1, "column2": col2, "correlation": corr_val}
                ))
        
        except Exception as e:
            logger.warning("Correlation check failed", error=str(e))
        
        return issues
    
    def _prepare_dataframe(self, data: Union[pd.DataFrame, str, Dict[str, Any]]) -> pd.DataFrame:
        """Convert input data to DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, str):
            # Assume it's a file path
            if data.endswith('.csv'):
                return pd.read_csv(data)
            elif data.endswith('.json'):
                return pd.read_json(data)
            elif data.endswith('.parquet'):
                return pd.read_parquet(data)
            else:
                raise ValueError(f"Unsupported file format: {data}")
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _types_compatible(self, actual_type: str, expected_type: str) -> bool:
        """Check if data types are compatible."""
        type_mapping = {
            "int64": ["integer", "int", "numeric"],
            "float64": ["float", "numeric"],
            "object": ["string", "text", "object"],
            "bool": ["boolean", "bool"],
            "datetime64": ["datetime", "timestamp"]
        }
        
        compatible_types = type_mapping.get(actual_type, [actual_type])
        return expected_type.lower() in compatible_types
    
    def _calculate_overall_score(self, issues: List[ValidationIssue], total_rows: int, total_columns: int) -> float:
        """Calculate overall data quality score (0-100)."""
        if not issues:
            return 100.0
        
        # Weight issues by severity
        severity_weights = {
            ValidationSeverity.INFO: 1,
            ValidationSeverity.WARNING: 3,
            ValidationSeverity.ERROR: 5,
            ValidationSeverity.CRITICAL: 10
        }
        
        total_weight = sum(severity_weights[issue.severity] for issue in issues)
        max_possible_weight = total_columns * severity_weights[ValidationSeverity.CRITICAL]
        
        # Calculate score (higher weight = lower score)
        score = max(0, 100 - (total_weight / max_possible_weight * 100))
        
        return round(score, 1)
    
    def _generate_summary_stats(self, df: pd.DataFrame, issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            "total_cells": len(df) * len(df.columns),
            "missing_cells": df.isnull().sum().sum(),
            "missing_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=['object']).columns),
            "issues_by_severity": {
                severity.value: len([i for i in issues if i.severity == severity])
                for severity in ValidationSeverity
            },
            "issues_by_category": {
                category.value: len([i for i in issues if i.category == category])
                for category in ValidationCategory
            }
        }
    
    def _generate_recommendations(self, issues: List[ValidationIssue], df: pd.DataFrame) -> List[str]:
        """Generate data quality recommendations."""
        recommendations = []
        
        # Missing data recommendations
        missing_issues = [i for i in issues if i.category == ValidationCategory.MISSING_DATA]
        if missing_issues:
            recommendations.append("Consider imputation strategies for missing data or remove columns with excessive missing values")
        
        # Data type recommendations
        type_issues = [i for i in issues if i.category == ValidationCategory.DATA_TYPE]
        if type_issues:
            recommendations.append("Convert columns to appropriate data types for better performance and accuracy")
        
        # Outlier recommendations
        range_issues = [i for i in issues if i.category == ValidationCategory.RANGE]
        if range_issues:
            recommendations.append("Investigate outliers - they may be data errors or important anomalies")
        
        # Duplicate recommendations
        duplicate_issues = [i for i in issues if i.category == ValidationCategory.QUALITY and "duplicate" in i.message.lower()]
        if duplicate_issues:
            recommendations.append("Remove duplicate rows to improve data quality and model performance")
        
        # Schema recommendations
        schema_issues = [i for i in issues if i.category == ValidationCategory.SCHEMA]
        if schema_issues:
            recommendations.append("Update data schema or fix data structure issues before processing")
        
        # Correlation recommendations
        corr_issues = [i for i in issues if i.category == ValidationCategory.CONSISTENCY]
        if corr_issues:
            recommendations.append("Consider removing highly correlated features to reduce multicollinearity")
        
        if not recommendations:
            recommendations.append("Data quality appears good - ready for further processing")
        
        return recommendations
    
    def add_custom_rule(self, rule: ValidationRule):
        """Add a custom validation rule."""
        self.custom_rules.append(rule)
        logger.info("Custom validation rule added", rule_name=rule.name)
    
    def remove_custom_rule(self, rule_name: str) -> bool:
        """Remove a custom validation rule."""
        for i, rule in enumerate(self.custom_rules):
            if rule.name == rule_name:
                del self.custom_rules[i]
                logger.info("Custom validation rule removed", rule_name=rule_name)
                return True
        return False