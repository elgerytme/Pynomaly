"""Comprehensive data validation infrastructure.

This module provides robust data validation capabilities for ensuring data quality
and preventing errors in the anomaly detection pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import ValidationError
from pynomaly.infrastructure.monitoring import get_monitor, monitor_operation


class ValidationSeverity(Enum):
    """Validation issue severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Categories of validation checks."""

    STRUCTURE = "structure"
    DATA_TYPE = "data_type"
    RANGE = "range"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    QUALITY = "quality"
    SECURITY = "security"


@dataclass
class ValidationIssue:
    """Individual validation issue."""

    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    column: str | None = None
    row_indices: list[int] | None = None
    value_sample: list[Any] | None = None
    suggestion: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "column": self.column,
            "row_indices": (
                self.row_indices[:10] if self.row_indices else None
            ),  # Limit sample
            "value_sample": (
                self.value_sample[:5] if self.value_sample else None
            ),  # Limit sample
            "suggestion": self.suggestion,
            "metadata": self.metadata,
        }


@dataclass
class ValidationResult:
    """Complete validation results."""

    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def has_critical_issues(self) -> bool:
        """Check if there are critical issues."""
        return any(
            issue.severity == ValidationSeverity.CRITICAL for issue in self.issues
        )

    @property
    def has_errors(self) -> bool:
        """Check if there are error-level issues."""
        return any(
            issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
            for issue in self.issues
        )

    def get_issues_by_severity(
        self, severity: ValidationSeverity
    ) -> list[ValidationIssue]:
        """Get issues by severity level."""
        return [issue for issue in self.issues if issue.severity == severity]

    def get_issues_by_category(
        self, category: ValidationCategory
    ) -> list[ValidationIssue]:
        """Get issues by category."""
        return [issue for issue in self.issues if issue.category == category]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "has_critical_issues": self.has_critical_issues,
            "has_errors": self.has_errors,
            "total_issues": len(self.issues),
            "issues_by_severity": {
                severity.value: len(self.get_issues_by_severity(severity))
                for severity in ValidationSeverity
            },
            "issues_by_category": {
                category.value: len(self.get_issues_by_category(category))
                for category in ValidationCategory
            },
            "issues": [issue.to_dict() for issue in self.issues],
            "summary": self.summary,
            "timestamp": self.timestamp.isoformat(),
        }


class DataValidator:
    """Comprehensive data validator with configurable rules."""

    def __init__(
        self,
        fail_on_critical: bool = True,
        fail_on_errors: bool = False,
        max_issues_per_check: int = 100,
    ):
        """Initialize data validator.

        Args:
            fail_on_critical: Raise exception on critical issues
            fail_on_errors: Raise exception on error-level issues
            max_issues_per_check: Maximum issues to collect per validation check
        """
        self.fail_on_critical = fail_on_critical
        self.fail_on_errors = fail_on_errors
        self.max_issues_per_check = max_issues_per_check

        # Default validation rules
        self.validation_rules: dict[str, Callable] = {
            "structure": self._validate_structure,
            "data_types": self._validate_data_types,
            "missing_values": self._validate_missing_values,
            "numeric_ranges": self._validate_numeric_ranges,
            "duplicates": self._validate_duplicates,
            "outliers": self._validate_outliers,
            "consistency": self._validate_consistency,
            "security": self._validate_security,
        }

    def validate_dataset(
        self,
        dataset: Dataset,
        rules: list[str] | None = None,
        custom_rules: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """Validate dataset comprehensively.

        Args:
            dataset: Dataset to validate
            rules: Specific rules to run (None for all)
            custom_rules: Custom validation rules

        Returns:
            ValidationResult with all issues found
        """
        if dataset.data is None:
            raise ValidationError("Dataset has no data to validate")

        with monitor_operation("dataset_validation", "data_validator"):
            issues: list[ValidationIssue] = []

            # Run standard validation rules
            rules_to_run = rules or list(self.validation_rules.keys())

            for rule_name in rules_to_run:
                if rule_name in self.validation_rules:
                    try:
                        rule_issues = self.validation_rules[rule_name](dataset.data)
                        issues.extend(rule_issues[: self.max_issues_per_check])
                    except Exception as e:
                        issues.append(
                            ValidationIssue(
                                category=ValidationCategory.QUALITY,
                                severity=ValidationSeverity.ERROR,
                                message=f"Validation rule '{rule_name}' failed: {e}",
                                suggestion="Check validation rule implementation",
                            )
                        )

            # Run custom rules if provided
            if custom_rules:
                for rule_name, rule_config in custom_rules.items():
                    try:
                        custom_issues = self._run_custom_rule(
                            dataset.data, rule_name, rule_config
                        )
                        issues.extend(custom_issues[: self.max_issues_per_check])
                    except Exception as e:
                        issues.append(
                            ValidationIssue(
                                category=ValidationCategory.QUALITY,
                                severity=ValidationSeverity.ERROR,
                                message=f"Custom rule '{rule_name}' failed: {e}",
                                suggestion="Check custom rule configuration",
                            )
                        )

            # Create validation result
            result = ValidationResult(
                is_valid=not any(
                    issue.severity
                    in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
                    for issue in issues
                ),
                issues=issues,
                summary=self._create_summary(dataset.data, issues),
            )

            # Log validation results
            get_monitor().info(
                f"Dataset validation complete: {len(issues)} issues found",
                operation="dataset_validation",
                component="data_validator",
                dataset_id=dataset.id,
                total_issues=len(issues),
                critical_issues=len(
                    result.get_issues_by_severity(ValidationSeverity.CRITICAL)
                ),
                error_issues=len(
                    result.get_issues_by_severity(ValidationSeverity.ERROR)
                ),
                is_valid=result.is_valid,
            )

            # Handle validation failures
            if self.fail_on_critical and result.has_critical_issues:
                critical_messages = [
                    issue.message
                    for issue in result.get_issues_by_severity(
                        ValidationSeverity.CRITICAL
                    )
                ]
                raise ValidationError(
                    f"Critical validation issues: {'; '.join(critical_messages)}"
                )

            if self.fail_on_errors and result.has_errors:
                error_messages = [
                    issue.message
                    for issue in issues
                    if issue.severity
                    in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
                ]
                raise ValidationError(f"Validation errors: {'; '.join(error_messages)}")

            return result

    def _validate_structure(self, df: pd.DataFrame) -> list[ValidationIssue]:
        """Validate basic data structure."""
        issues = []

        # Check if DataFrame is empty
        if df.empty:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.STRUCTURE,
                    severity=ValidationSeverity.CRITICAL,
                    message="Dataset is empty",
                    suggestion="Ensure dataset contains data before processing",
                )
            )
            return issues

        # Check for minimum rows
        if len(df) < 10:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.STRUCTURE,
                    severity=ValidationSeverity.WARNING,
                    message=f"Dataset has only {len(df)} rows, which may be insufficient for analysis",
                    suggestion="Consider collecting more data for reliable analysis",
                )
            )

        # Check for duplicate column names
        if df.columns.duplicated().any():
            duplicate_cols = df.columns[df.columns.duplicated()].tolist()
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.STRUCTURE,
                    severity=ValidationSeverity.ERROR,
                    message=f"Duplicate column names found: {duplicate_cols}",
                    suggestion="Rename duplicate columns to unique names",
                )
            )

        # Check for unnamed columns
        unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
        if unnamed_cols:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.STRUCTURE,
                    severity=ValidationSeverity.WARNING,
                    message=f"Unnamed columns found: {unnamed_cols}",
                    suggestion="Provide meaningful names for all columns",
                )
            )

        return issues

    def _validate_data_types(self, df: pd.DataFrame) -> list[ValidationIssue]:
        """Validate data types and detect type inconsistencies."""
        issues = []

        for col in df.columns:
            col_data = df[col]

            # Check for mixed types in object columns
            if col_data.dtype == "object":
                # Sample non-null values to check type consistency
                non_null_sample = col_data.dropna().head(1000)
                if len(non_null_sample) > 0:
                    types_found = {type(val).__name__ for val in non_null_sample}

                    if len(types_found) > 1:
                        issues.append(
                            ValidationIssue(
                                category=ValidationCategory.DATA_TYPE,
                                severity=ValidationSeverity.WARNING,
                                message=f"Column '{col}' contains mixed data types: {types_found}",
                                column=col,
                                suggestion="Consider data cleaning or type conversion",
                            )
                        )

            # Check for potential numeric columns stored as strings
            if col_data.dtype == "object":
                # Try to convert to numeric and see if most values succeed
                try:
                    numeric_converted = pd.to_numeric(col_data, errors="coerce")
                    non_null_original = col_data.dropna()
                    non_null_converted = numeric_converted.dropna()

                    if len(non_null_converted) > 0.8 * len(non_null_original):
                        issues.append(
                            ValidationIssue(
                                category=ValidationCategory.DATA_TYPE,
                                severity=ValidationSeverity.INFO,
                                message=f"Column '{col}' appears to contain numeric data but is stored as text",
                                column=col,
                                suggestion="Consider converting to numeric type for better performance",
                            )
                        )
                except Exception:
                    pass

        return issues

    def _validate_missing_values(self, df: pd.DataFrame) -> list[ValidationIssue]:
        """Validate missing values patterns."""
        issues = []

        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100

            if missing_pct > 50:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.COMPLETENESS,
                        severity=ValidationSeverity.ERROR,
                        message=f"Column '{col}' has {missing_pct:.1f}% missing values",
                        column=col,
                        metadata={
                            "missing_count": missing_count,
                            "missing_percentage": missing_pct,
                        },
                        suggestion="Consider removing column or imputing missing values",
                    )
                )
            elif missing_pct > 20:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.COMPLETENESS,
                        severity=ValidationSeverity.WARNING,
                        message=f"Column '{col}' has {missing_pct:.1f}% missing values",
                        column=col,
                        metadata={
                            "missing_count": missing_count,
                            "missing_percentage": missing_pct,
                        },
                        suggestion="Consider imputation strategy for missing values",
                    )
                )

        # Check for rows with all missing values
        all_missing_rows = df.isna().all(axis=1).sum()
        if all_missing_rows > 0:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.COMPLETENESS,
                    severity=ValidationSeverity.WARNING,
                    message=f"{all_missing_rows} rows have all missing values",
                    suggestion="Consider removing completely empty rows",
                )
            )

        return issues

    def _validate_numeric_ranges(self, df: pd.DataFrame) -> list[ValidationIssue]:
        """Validate numeric ranges and detect anomalous values."""
        issues = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_data = df[col].dropna()

            if len(col_data) == 0:
                continue

            # Check for infinite values
            inf_count = np.isinf(col_data).sum()
            if inf_count > 0:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.RANGE,
                        severity=ValidationSeverity.ERROR,
                        message=f"Column '{col}' contains {inf_count} infinite values",
                        column=col,
                        suggestion="Replace infinite values with appropriate finite values",
                    )
                )

            # Check for extremely large values (potential data entry errors)
            col_data.quantile(0.99)
            col_data.quantile(0.01)
            IQR = col_data.quantile(0.75) - col_data.quantile(0.25)

            # Values more than 10 IQRs from median might be errors
            median = col_data.median()
            extreme_threshold = 10 * IQR

            if extreme_threshold > 0:
                extreme_values = col_data[
                    (col_data < median - extreme_threshold)
                    | (col_data > median + extreme_threshold)
                ]

                if len(extreme_values) > 0:
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.RANGE,
                            severity=ValidationSeverity.WARNING,
                            message=f"Column '{col}' has {len(extreme_values)} potentially extreme values",
                            column=col,
                            value_sample=extreme_values.head(5).tolist(),
                            suggestion="Review extreme values for data entry errors",
                        )
                    )

            # Check for constant values
            if col_data.nunique() == 1:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.RANGE,
                        severity=ValidationSeverity.WARNING,
                        message=f"Column '{col}' has constant value: {col_data.iloc[0]}",
                        column=col,
                        suggestion="Constant columns provide no information for analysis",
                    )
                )

        return issues

    def _validate_duplicates(self, df: pd.DataFrame) -> list[ValidationIssue]:
        """Validate duplicate records."""
        issues = []

        # Check for exact duplicate rows
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            duplicate_pct = (duplicate_rows / len(df)) * 100
            severity = (
                ValidationSeverity.ERROR
                if duplicate_pct > 10
                else ValidationSeverity.WARNING
            )

            issues.append(
                ValidationIssue(
                    category=ValidationCategory.CONSISTENCY,
                    severity=severity,
                    message=f"{duplicate_rows} ({duplicate_pct:.1f}%) duplicate rows found",
                    metadata={
                        "duplicate_count": duplicate_rows,
                        "duplicate_percentage": duplicate_pct,
                    },
                    suggestion="Consider removing duplicate rows or investigating data collection process",
                )
            )

        return issues

    def _validate_outliers(self, df: pd.DataFrame) -> list[ValidationIssue]:
        """Validate outliers using statistical methods."""
        issues = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_data = df[col].dropna()

            if len(col_data) < 10:  # Need sufficient data for outlier detection
                continue

            # IQR method for outlier detection
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1

            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                outlier_pct = (len(outliers) / len(col_data)) * 100

                if (
                    outlier_pct > 5
                ):  # More than 5% outliers might indicate data quality issues
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.QUALITY,
                            severity=ValidationSeverity.WARNING,
                            message=f"Column '{col}' has {len(outliers)} ({outlier_pct:.1f}%) statistical outliers",
                            column=col,
                            value_sample=outliers.head(5).tolist(),
                            metadata={
                                "outlier_count": len(outliers),
                                "outlier_percentage": outlier_pct,
                            },
                            suggestion="Review outliers to determine if they are valid or errors",
                        )
                    )

        return issues

    def _validate_consistency(self, df: pd.DataFrame) -> list[ValidationIssue]:
        """Validate data consistency and relationships."""
        issues = []

        # Check for inconsistent categorical values (case sensitivity, whitespace)
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns

        for col in categorical_cols:
            col_data = df[col].dropna().astype(str)

            if len(col_data) == 0:
                continue

            # Check for values that differ only by case or whitespace
            normalized_values = col_data.str.strip().str.lower()
            original_unique = set(col_data.unique())
            normalized_unique = set(normalized_values.unique())

            if len(original_unique) > len(normalized_unique):
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.CONSISTENCY,
                        severity=ValidationSeverity.WARNING,
                        message=f"Column '{col}' has inconsistent categorical values (case/whitespace)",
                        column=col,
                        suggestion="Standardize categorical values for consistency",
                    )
                )

        return issues

    def _validate_security(self, df: pd.DataFrame) -> list[ValidationIssue]:
        """Validate for potential security issues."""
        issues = []

        # Check for potential PII patterns
        pii_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        }

        text_cols = df.select_dtypes(include=["object"]).columns

        for col in text_cols:
            col_data = df[col].dropna().astype(str)

            for pii_type, pattern in pii_patterns.items():
                matches = col_data.str.contains(pattern, pattern=True, na=False).sum()

                if matches > 0:
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.SECURITY,
                            severity=ValidationSeverity.WARNING,
                            message=f"Column '{col}' may contain {pii_type} data ({matches} potential matches)",
                            column=col,
                            suggestion=f"Review and anonymize potential {pii_type} data",
                        )
                    )

        return issues

    def _run_custom_rule(
        self, df: pd.DataFrame, rule_name: str, rule_config: dict[str, Any]
    ) -> list[ValidationIssue]:
        """Run a custom validation rule."""
        issues = []

        # Custom rule should specify function and parameters
        if "function" not in rule_config:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.QUALITY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Custom rule '{rule_name}' missing 'function' specification",
                )
            )
            return issues

        try:
            rule_func = rule_config["function"]
            rule_params = rule_config.get("parameters", {})

            # Execute custom rule
            if callable(rule_func):
                custom_issues = rule_func(df, **rule_params)
                if isinstance(custom_issues, list):
                    issues.extend(custom_issues)
                else:
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.QUALITY,
                            severity=ValidationSeverity.WARNING,
                            message=f"Custom rule '{rule_name}' returned non-list result",
                        )
                    )
            else:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.QUALITY,
                        severity=ValidationSeverity.ERROR,
                        message=f"Custom rule '{rule_name}' function is not callable",
                    )
                )

        except Exception as e:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.QUALITY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Custom rule '{rule_name}' execution failed: {e}",
                )
            )

        return issues

    def _create_summary(
        self, df: pd.DataFrame, issues: list[ValidationIssue]
    ) -> dict[str, Any]:
        """Create validation summary."""
        return {
            "dataset_shape": df.shape,
            "total_issues": len(issues),
            "issues_by_severity": {
                severity.value: len([i for i in issues if i.severity == severity])
                for severity in ValidationSeverity
            },
            "issues_by_category": {
                category.value: len([i for i in issues if i.category == category])
                for category in ValidationCategory
            },
            "columns_with_issues": len(
                {issue.column for issue in issues if issue.column}
            ),
            "data_types": df.dtypes.value_counts().to_dict(),
            "missing_data_summary": {
                "columns_with_missing": df.isna().any().sum(),
                "total_missing_values": df.isna().sum().sum(),
                "missing_percentage": (
                    df.isna().sum().sum() / (df.shape[0] * df.shape[1])
                )
                * 100,
            },
        }


def validate_file_format(file_path: str | Path) -> ValidationResult:
    """Validate file format and basic accessibility.

    Args:
        file_path: Path to file to validate

    Returns:
        ValidationResult
    """
    file_path = Path(file_path)
    issues = []

    # Check if file exists
    if not file_path.exists():
        issues.append(
            ValidationIssue(
                category=ValidationCategory.STRUCTURE,
                severity=ValidationSeverity.CRITICAL,
                message=f"File does not exist: {file_path}",
            )
        )
        return ValidationResult(is_valid=False, issues=issues)

    # Check file size
    file_size_mb = file_path.stat().st_size / 1024 / 1024
    if file_size_mb > 1000:  # > 1GB
        issues.append(
            ValidationIssue(
                category=ValidationCategory.STRUCTURE,
                severity=ValidationSeverity.WARNING,
                message=f"Large file size: {file_size_mb:.1f}MB",
                suggestion="Consider using streaming processing for large files",
            )
        )

    # Check file extension
    supported_extensions = {".csv", ".parquet", ".pq", ".xlsx", ".xls", ".json"}
    if file_path.suffix.lower() not in supported_extensions:
        issues.append(
            ValidationIssue(
                category=ValidationCategory.STRUCTURE,
                severity=ValidationSeverity.WARNING,
                message=f"Unsupported file extension: {file_path.suffix}",
                suggestion=f"Supported formats: {', '.join(supported_extensions)}",
            )
        )

    # Try to peek at file content for basic validation
    try:
        if file_path.suffix.lower() == ".csv":
            # Read first few lines to check format
            with open(file_path, encoding="utf-8") as f:
                first_line = f.readline().strip()
                if not first_line:
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.STRUCTURE,
                            severity=ValidationSeverity.ERROR,
                            message="CSV file appears to be empty",
                        )
                    )
    except UnicodeDecodeError:
        issues.append(
            ValidationIssue(
                category=ValidationCategory.STRUCTURE,
                severity=ValidationSeverity.ERROR,
                message="File encoding issues detected",
                suggestion="Ensure file is in UTF-8 encoding",
            )
        )
    except Exception as e:
        issues.append(
            ValidationIssue(
                category=ValidationCategory.STRUCTURE,
                severity=ValidationSeverity.WARNING,
                message=f"Could not validate file content: {e}",
            )
        )

    is_valid = not any(
        issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
        for issue in issues
    )

    return ValidationResult(is_valid=is_valid, issues=issues)
