#!/usr/bin/env python3
"""
Data Quality and Validation Pipeline for Pynomaly.
Provides comprehensive data validation, quality assessment, and pipeline monitoring.
"""

import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataQualityStatus(Enum):
    """Data quality assessment status."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class ValidationRuleType(Enum):
    """Types of validation rules."""

    SCHEMA = "schema"
    RANGE = "range"
    PATTERN = "pattern"
    UNIQUENESS = "uniqueness"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    FRESHNESS = "freshness"
    DISTRIBUTION = "distribution"
    CUSTOM = "custom"


@dataclass
class ValidationIssue:
    """Data validation issue."""

    issue_id: str
    rule_name: str
    rule_type: ValidationRuleType
    severity: ValidationSeverity
    message: str
    affected_rows: list[int]
    affected_columns: list[str]
    details: dict[str, Any]
    timestamp: datetime


@dataclass
class DataQualityMetrics:
    """Data quality metrics."""

    completeness_score: float  # 0-1, percentage of non-null values
    validity_score: float  # 0-1, percentage of valid values
    uniqueness_score: float  # 0-1, percentage of unique values where expected
    consistency_score: float  # 0-1, internal consistency
    accuracy_score: float  # 0-1, estimated accuracy
    timeliness_score: float  # 0-1, data freshness
    overall_score: float  # 0-1, weighted average
    total_rows: int
    total_columns: int
    issues_count: int
    critical_issues_count: int


@dataclass
class ValidationReport:
    """Data validation report."""

    report_id: str
    dataset_name: str
    validation_timestamp: datetime
    data_quality_metrics: DataQualityMetrics
    validation_issues: list[ValidationIssue]
    data_summary: dict[str, Any]
    recommendations: list[str]
    passed_rules: int
    failed_rules: int
    total_rules: int
    execution_time_seconds: float


@dataclass
class ValidationRule:
    """Data validation rule."""

    rule_id: str
    name: str
    description: str
    rule_type: ValidationRuleType
    severity: ValidationSeverity
    enabled: bool
    parameters: dict[str, Any]
    custom_validator: Callable | None = None


class SchemaValidator:
    """Schema validation for data consistency."""

    def __init__(self):
        self.expected_schemas: dict[str, dict[str, Any]] = {}

    def register_schema(self, dataset_name: str, schema: dict[str, Any]):
        """Register expected schema for a dataset."""
        self.expected_schemas[dataset_name] = schema
        logger.info(f"Registered schema for dataset: {dataset_name}")

    def validate_schema(
        self, df: pd.DataFrame, dataset_name: str
    ) -> list[ValidationIssue]:
        """Validate DataFrame against expected schema."""
        issues = []

        if dataset_name not in self.expected_schemas:
            issues.append(
                ValidationIssue(
                    issue_id=str(uuid.uuid4()),
                    rule_name="schema_exists",
                    rule_type=ValidationRuleType.SCHEMA,
                    severity=ValidationSeverity.WARNING,
                    message=f"No schema registered for dataset: {dataset_name}",
                    affected_rows=[],
                    affected_columns=[],
                    details={"dataset_name": dataset_name},
                    timestamp=datetime.now(),
                )
            )
            return issues

        expected_schema = self.expected_schemas[dataset_name]

        # Check column existence
        expected_columns = set(expected_schema.get("columns", {}))
        actual_columns = set(df.columns)

        missing_columns = expected_columns - actual_columns
        extra_columns = actual_columns - expected_columns

        if missing_columns:
            issues.append(
                ValidationIssue(
                    issue_id=str(uuid.uuid4()),
                    rule_name="missing_columns",
                    rule_type=ValidationRuleType.SCHEMA,
                    severity=ValidationSeverity.ERROR,
                    message=f"Missing required columns: {list(missing_columns)}",
                    affected_rows=[],
                    affected_columns=list(missing_columns),
                    details={"missing_columns": list(missing_columns)},
                    timestamp=datetime.now(),
                )
            )

        if extra_columns:
            issues.append(
                ValidationIssue(
                    issue_id=str(uuid.uuid4()),
                    rule_name="extra_columns",
                    rule_type=ValidationRuleType.SCHEMA,
                    severity=ValidationSeverity.WARNING,
                    message=f"Unexpected columns found: {list(extra_columns)}",
                    affected_rows=[],
                    affected_columns=list(extra_columns),
                    details={"extra_columns": list(extra_columns)},
                    timestamp=datetime.now(),
                )
            )

        # Check data types
        for column, expected_type in expected_schema.get("columns", {}).items():
            if column in df.columns:
                actual_type = str(df[column].dtype)
                if not self._types_compatible(actual_type, expected_type):
                    issues.append(
                        ValidationIssue(
                            issue_id=str(uuid.uuid4()),
                            rule_name="data_type_mismatch",
                            rule_type=ValidationRuleType.SCHEMA,
                            severity=ValidationSeverity.ERROR,
                            message=f"Column {column}: expected {expected_type}, got {actual_type}",
                            affected_rows=[],
                            affected_columns=[column],
                            details={
                                "column": column,
                                "expected_type": expected_type,
                                "actual_type": actual_type,
                            },
                            timestamp=datetime.now(),
                        )
                    )

        return issues

    def _types_compatible(self, actual_type: str, expected_type: str) -> bool:
        """Check if data types are compatible."""
        type_mappings = {
            "int64": ["int", "integer", "int64"],
            "float64": ["float", "double", "float64", "numeric"],
            "object": ["string", "text", "object", "str"],
            "bool": ["boolean", "bool"],
            "datetime64[ns]": ["datetime", "timestamp", "date"],
        }

        if actual_type in type_mappings:
            return expected_type.lower() in type_mappings[actual_type]
        return actual_type.lower() == expected_type.lower()


class DataQualityValidator:
    """Comprehensive data quality validation."""

    def __init__(self):
        self.validation_rules: dict[str, ValidationRule] = {}
        self.schema_validator = SchemaValidator()

    def add_validation_rule(self, rule: ValidationRule):
        """Add a validation rule."""
        self.validation_rules[rule.rule_id] = rule
        logger.info(f"Added validation rule: {rule.name}")

    def remove_validation_rule(self, rule_id: str):
        """Remove a validation rule."""
        if rule_id in self.validation_rules:
            del self.validation_rules[rule_id]
            logger.info(f"Removed validation rule: {rule_id}")

    def validate_dataset(
        self, df: pd.DataFrame, dataset_name: str = "unknown", rules: list[str] = None
    ) -> ValidationReport:
        """Validate a dataset against all applicable rules."""
        start_time = datetime.now()
        report_id = str(uuid.uuid4())

        # Determine which rules to run
        if rules:
            active_rules = {
                rid: rule
                for rid, rule in self.validation_rules.items()
                if rid in rules and rule.enabled
            }
        else:
            active_rules = {
                rid: rule for rid, rule in self.validation_rules.items() if rule.enabled
            }

        validation_issues = []
        passed_rules = 0
        failed_rules = 0

        # Schema validation
        schema_issues = self.schema_validator.validate_schema(df, dataset_name)
        validation_issues.extend(schema_issues)

        # Run validation rules
        for rule_id, rule in active_rules.items():
            try:
                rule_issues = self._execute_validation_rule(df, rule)
                validation_issues.extend(rule_issues)

                if rule_issues:
                    failed_rules += 1
                else:
                    passed_rules += 1

            except Exception as e:
                logger.error(f"Validation rule {rule.name} failed: {e}")
                validation_issues.append(
                    ValidationIssue(
                        issue_id=str(uuid.uuid4()),
                        rule_name=rule.name,
                        rule_type=rule.rule_type,
                        severity=ValidationSeverity.ERROR,
                        message=f"Rule execution failed: {str(e)}",
                        affected_rows=[],
                        affected_columns=[],
                        details={"error": str(e)},
                        timestamp=datetime.now(),
                    )
                )
                failed_rules += 1

        # Calculate data quality metrics
        quality_metrics = self._calculate_quality_metrics(df, validation_issues)

        # Generate data summary
        data_summary = self._generate_data_summary(df)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            validation_issues, quality_metrics
        )

        # Create report
        execution_time = (datetime.now() - start_time).total_seconds()

        report = ValidationReport(
            report_id=report_id,
            dataset_name=dataset_name,
            validation_timestamp=start_time,
            data_quality_metrics=quality_metrics,
            validation_issues=validation_issues,
            data_summary=data_summary,
            recommendations=recommendations,
            passed_rules=passed_rules,
            failed_rules=failed_rules,
            total_rules=len(active_rules),
            execution_time_seconds=execution_time,
        )

        logger.info(
            f"Dataset validation completed: {dataset_name} (score: {quality_metrics.overall_score:.3f})"
        )
        return report

    def _execute_validation_rule(
        self, df: pd.DataFrame, rule: ValidationRule
    ) -> list[ValidationIssue]:
        """Execute a single validation rule."""
        issues = []

        if rule.rule_type == ValidationRuleType.COMPLETENESS:
            issues.extend(self._validate_completeness(df, rule))
        elif rule.rule_type == ValidationRuleType.RANGE:
            issues.extend(self._validate_range(df, rule))
        elif rule.rule_type == ValidationRuleType.PATTERN:
            issues.extend(self._validate_pattern(df, rule))
        elif rule.rule_type == ValidationRuleType.UNIQUENESS:
            issues.extend(self._validate_uniqueness(df, rule))
        elif rule.rule_type == ValidationRuleType.CONSISTENCY:
            issues.extend(self._validate_consistency(df, rule))
        elif rule.rule_type == ValidationRuleType.FRESHNESS:
            issues.extend(self._validate_freshness(df, rule))
        elif rule.rule_type == ValidationRuleType.DISTRIBUTION:
            issues.extend(self._validate_distribution(df, rule))
        elif rule.rule_type == ValidationRuleType.CUSTOM and rule.custom_validator:
            issues.extend(self._validate_custom(df, rule))

        return issues

    def _validate_completeness(
        self, df: pd.DataFrame, rule: ValidationRule
    ) -> list[ValidationIssue]:
        """Validate data completeness."""
        issues = []
        columns = rule.parameters.get("columns", df.columns)
        threshold = rule.parameters.get("threshold", 0.95)

        for column in columns:
            if column in df.columns:
                completeness = df[column].notna().sum() / len(df)
                if completeness < threshold:
                    null_rows = df[df[column].isna()].index.tolist()
                    issues.append(
                        ValidationIssue(
                            issue_id=str(uuid.uuid4()),
                            rule_name=rule.name,
                            rule_type=rule.rule_type,
                            severity=rule.severity,
                            message=f"Column {column} completeness {completeness:.3f} below threshold {threshold}",
                            affected_rows=null_rows[:100],  # Limit to first 100
                            affected_columns=[column],
                            details={
                                "column": column,
                                "completeness": completeness,
                                "threshold": threshold,
                                "null_count": df[column].isna().sum(),
                            },
                            timestamp=datetime.now(),
                        )
                    )

        return issues

    def _validate_range(
        self, df: pd.DataFrame, rule: ValidationRule
    ) -> list[ValidationIssue]:
        """Validate numeric ranges."""
        issues = []
        columns = rule.parameters.get("columns", [])

        for column_config in columns:
            column = column_config["name"]
            min_val = column_config.get("min")
            max_val = column_config.get("max")

            if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                out_of_range = pd.Series([False] * len(df))

                if min_val is not None:
                    out_of_range |= df[column] < min_val
                if max_val is not None:
                    out_of_range |= df[column] > max_val

                if out_of_range.any():
                    invalid_rows = df[out_of_range].index.tolist()
                    issues.append(
                        ValidationIssue(
                            issue_id=str(uuid.uuid4()),
                            rule_name=rule.name,
                            rule_type=rule.rule_type,
                            severity=rule.severity,
                            message=f"Column {column} has {out_of_range.sum()} values outside range [{min_val}, {max_val}]",
                            affected_rows=invalid_rows[:100],
                            affected_columns=[column],
                            details={
                                "column": column,
                                "min_allowed": min_val,
                                "max_allowed": max_val,
                                "violations": out_of_range.sum(),
                                "actual_min": df[column].min(),
                                "actual_max": df[column].max(),
                            },
                            timestamp=datetime.now(),
                        )
                    )

        return issues

    def _validate_pattern(
        self, df: pd.DataFrame, rule: ValidationRule
    ) -> list[ValidationIssue]:
        """Validate string patterns."""
        issues = []
        columns = rule.parameters.get("columns", [])

        for column_config in columns:
            column = column_config["name"]
            pattern = column_config["pattern"]

            if column in df.columns:
                # Convert to string and check pattern
                string_series = df[column].astype(str)
                matches = string_series.str.match(pattern, na=False)

                if not matches.all():
                    invalid_rows = df[~matches].index.tolist()
                    issues.append(
                        ValidationIssue(
                            issue_id=str(uuid.uuid4()),
                            rule_name=rule.name,
                            rule_type=rule.rule_type,
                            severity=rule.severity,
                            message=f"Column {column} has {(~matches).sum()} values not matching pattern {pattern}",
                            affected_rows=invalid_rows[:100],
                            affected_columns=[column],
                            details={
                                "column": column,
                                "pattern": pattern,
                                "violations": (~matches).sum(),
                            },
                            timestamp=datetime.now(),
                        )
                    )

        return issues

    def _validate_uniqueness(
        self, df: pd.DataFrame, rule: ValidationRule
    ) -> list[ValidationIssue]:
        """Validate data uniqueness."""
        issues = []
        columns = rule.parameters.get("columns", [])

        for column in columns:
            if column in df.columns:
                duplicates = df.duplicated(subset=[column], keep=False)
                if duplicates.any():
                    duplicate_rows = df[duplicates].index.tolist()
                    issues.append(
                        ValidationIssue(
                            issue_id=str(uuid.uuid4()),
                            rule_name=rule.name,
                            rule_type=rule.rule_type,
                            severity=rule.severity,
                            message=f"Column {column} has {duplicates.sum()} duplicate values",
                            affected_rows=duplicate_rows[:100],
                            affected_columns=[column],
                            details={
                                "column": column,
                                "duplicate_count": duplicates.sum(),
                                "unique_count": df[column].nunique(),
                            },
                            timestamp=datetime.now(),
                        )
                    )

        return issues

    def _validate_consistency(
        self, df: pd.DataFrame, rule: ValidationRule
    ) -> list[ValidationIssue]:
        """Validate internal data consistency."""
        issues = []
        checks = rule.parameters.get("checks", [])

        for check in checks:
            check_type = check["type"]

            if check_type == "conditional":
                condition = check["condition"]
                expected = check["expected"]

                try:
                    # Evaluate condition
                    mask = df.eval(condition)
                    expected_mask = df.eval(expected)

                    violations = mask & ~expected_mask
                    if violations.any():
                        invalid_rows = df[violations].index.tolist()
                        issues.append(
                            ValidationIssue(
                                issue_id=str(uuid.uuid4()),
                                rule_name=rule.name,
                                rule_type=rule.rule_type,
                                severity=rule.severity,
                                message=f"Consistency check failed: {condition} -> {expected}",
                                affected_rows=invalid_rows[:100],
                                affected_columns=[],
                                details={
                                    "condition": condition,
                                    "expected": expected,
                                    "violations": violations.sum(),
                                },
                                timestamp=datetime.now(),
                            )
                        )

                except Exception as e:
                    logger.error(f"Consistency check failed: {e}")

        return issues

    def _validate_freshness(
        self, df: pd.DataFrame, rule: ValidationRule
    ) -> list[ValidationIssue]:
        """Validate data freshness."""
        issues = []
        timestamp_column = rule.parameters.get("timestamp_column")
        max_age_hours = rule.parameters.get("max_age_hours", 24)

        if timestamp_column and timestamp_column in df.columns:
            try:
                timestamps = pd.to_datetime(df[timestamp_column])
                current_time = datetime.now()
                age_threshold = current_time - timedelta(hours=max_age_hours)

                old_data = timestamps < age_threshold
                if old_data.any():
                    old_rows = df[old_data].index.tolist()
                    issues.append(
                        ValidationIssue(
                            issue_id=str(uuid.uuid4()),
                            rule_name=rule.name,
                            rule_type=rule.rule_type,
                            severity=rule.severity,
                            message=f"{old_data.sum()} records older than {max_age_hours} hours",
                            affected_rows=old_rows[:100],
                            affected_columns=[timestamp_column],
                            details={
                                "timestamp_column": timestamp_column,
                                "max_age_hours": max_age_hours,
                                "old_records": old_data.sum(),
                                "oldest_record": timestamps.min(),
                                "newest_record": timestamps.max(),
                            },
                            timestamp=datetime.now(),
                        )
                    )
            except Exception as e:
                issues.append(
                    ValidationIssue(
                        issue_id=str(uuid.uuid4()),
                        rule_name=rule.name,
                        rule_type=rule.rule_type,
                        severity=ValidationSeverity.ERROR,
                        message=f"Freshness validation failed: {str(e)}",
                        affected_rows=[],
                        affected_columns=[timestamp_column],
                        details={"error": str(e)},
                        timestamp=datetime.now(),
                    )
                )

        return issues

    def _validate_distribution(
        self, df: pd.DataFrame, rule: ValidationRule
    ) -> list[ValidationIssue]:
        """Validate data distribution."""
        issues = []
        columns = rule.parameters.get("columns", [])

        for column_config in columns:
            column = column_config["name"]
            expected_distribution = column_config.get("distribution", {})

            if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                actual_stats = {
                    "mean": df[column].mean(),
                    "std": df[column].std(),
                    "min": df[column].min(),
                    "max": df[column].max(),
                    "skewness": df[column].skew(),
                    "kurtosis": df[column].kurtosis(),
                }

                for stat_name, expected_value in expected_distribution.items():
                    if stat_name in actual_stats:
                        actual_value = actual_stats[stat_name]
                        tolerance = column_config.get("tolerance", 0.1)

                        if abs(actual_value - expected_value) > tolerance:
                            issues.append(
                                ValidationIssue(
                                    issue_id=str(uuid.uuid4()),
                                    rule_name=rule.name,
                                    rule_type=rule.rule_type,
                                    severity=rule.severity,
                                    message=f"Column {column} {stat_name} ({actual_value:.3f}) deviates from expected ({expected_value:.3f})",
                                    affected_rows=[],
                                    affected_columns=[column],
                                    details={
                                        "column": column,
                                        "statistic": stat_name,
                                        "expected": expected_value,
                                        "actual": actual_value,
                                        "tolerance": tolerance,
                                    },
                                    timestamp=datetime.now(),
                                )
                            )

        return issues

    def _validate_custom(
        self, df: pd.DataFrame, rule: ValidationRule
    ) -> list[ValidationIssue]:
        """Execute custom validation function."""
        try:
            if rule.custom_validator:
                return rule.custom_validator(df, rule)
        except Exception as e:
            logger.error(f"Custom validation failed: {e}")
            return [
                ValidationIssue(
                    issue_id=str(uuid.uuid4()),
                    rule_name=rule.name,
                    rule_type=rule.rule_type,
                    severity=ValidationSeverity.ERROR,
                    message=f"Custom validation failed: {str(e)}",
                    affected_rows=[],
                    affected_columns=[],
                    details={"error": str(e)},
                    timestamp=datetime.now(),
                )
            ]
        return []

    def _calculate_quality_metrics(
        self, df: pd.DataFrame, issues: list[ValidationIssue]
    ) -> DataQualityMetrics:
        """Calculate comprehensive data quality metrics."""
        total_rows = len(df)
        total_columns = len(df.columns)

        # Completeness: percentage of non-null values
        completeness_score = (df.notna().sum().sum()) / (total_rows * total_columns)

        # Count issues by severity
        critical_issues = sum(
            1 for issue in issues if issue.severity == ValidationSeverity.CRITICAL
        )
        error_issues = sum(
            1 for issue in issues if issue.severity == ValidationSeverity.ERROR
        )

        # Validity: affected by errors and critical issues
        validity_score = max(
            0, 1 - (error_issues + critical_issues * 2) / max(1, total_columns)
        )

        # Uniqueness: estimate based on uniqueness issues
        uniqueness_issues = sum(
            1 for issue in issues if issue.rule_type == ValidationRuleType.UNIQUENESS
        )
        uniqueness_score = max(0, 1 - uniqueness_issues / max(1, total_columns))

        # Consistency: affected by consistency issues
        consistency_issues = sum(
            1 for issue in issues if issue.rule_type == ValidationRuleType.CONSISTENCY
        )
        consistency_score = max(0, 1 - consistency_issues / max(1, total_columns))

        # Timeliness: affected by freshness issues
        freshness_issues = sum(
            1 for issue in issues if issue.rule_type == ValidationRuleType.FRESHNESS
        )
        timeliness_score = max(0, 1 - freshness_issues / max(1, total_columns))

        # Accuracy: estimated based on range and pattern violations
        accuracy_issues = sum(
            1
            for issue in issues
            if issue.rule_type in [ValidationRuleType.RANGE, ValidationRuleType.PATTERN]
        )
        accuracy_score = max(0, 1 - accuracy_issues / max(1, total_columns))

        # Overall score: weighted average
        overall_score = (
            completeness_score * 0.25
            + validity_score * 0.25
            + uniqueness_score * 0.15
            + consistency_score * 0.15
            + accuracy_score * 0.15
            + timeliness_score * 0.05
        )

        return DataQualityMetrics(
            completeness_score=completeness_score,
            validity_score=validity_score,
            uniqueness_score=uniqueness_score,
            consistency_score=consistency_score,
            accuracy_score=accuracy_score,
            timeliness_score=timeliness_score,
            overall_score=overall_score,
            total_rows=total_rows,
            total_columns=total_columns,
            issues_count=len(issues),
            critical_issues_count=critical_issues,
        )

    def _generate_data_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """Generate comprehensive data summary."""
        summary = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": {},
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        }

        # Column-level statistics
        for column in df.columns:
            col_summary = {"type": str(df[column].dtype)}

            if pd.api.types.is_numeric_dtype(df[column]):
                col_summary.update(
                    {
                        "min": df[column].min(),
                        "max": df[column].max(),
                        "mean": df[column].mean(),
                        "std": df[column].std(),
                        "median": df[column].median(),
                        "unique_count": df[column].nunique(),
                    }
                )
            else:
                col_summary.update(
                    {
                        "unique_count": df[column].nunique(),
                        "most_common": df[column].mode().iloc[0]
                        if not df[column].mode().empty
                        else None,
                    }
                )

            summary["columns"][column] = col_summary

        return summary

    def _generate_recommendations(
        self, issues: list[ValidationIssue], metrics: DataQualityMetrics
    ) -> list[str]:
        """Generate data quality improvement recommendations."""
        recommendations = []

        # Completeness recommendations
        if metrics.completeness_score < 0.9:
            recommendations.append(
                "Review data collection processes to reduce missing values"
            )

        # Validity recommendations
        if metrics.validity_score < 0.9:
            recommendations.append(
                "Implement data validation at source to prevent invalid values"
            )

        # Pattern-based recommendations
        pattern_issues = [
            i for i in issues if i.rule_type == ValidationRuleType.PATTERN
        ]
        if pattern_issues:
            recommendations.append(
                "Standardize data formats and implement input validation"
            )

        # Range-based recommendations
        range_issues = [i for i in issues if i.rule_type == ValidationRuleType.RANGE]
        if range_issues:
            recommendations.append(
                "Review business rules and implement range constraints"
            )

        # Uniqueness recommendations
        uniqueness_issues = [
            i for i in issues if i.rule_type == ValidationRuleType.UNIQUENESS
        ]
        if uniqueness_issues:
            recommendations.append("Investigate and resolve duplicate data entries")

        # Critical issues
        critical_issues = [
            i for i in issues if i.severity == ValidationSeverity.CRITICAL
        ]
        if critical_issues:
            recommendations.append("Address critical data quality issues immediately")

        # Overall score recommendations
        if metrics.overall_score < 0.7:
            recommendations.append(
                "Consider implementing comprehensive data quality monitoring"
            )
            recommendations.append(
                "Review and update data collection and processing pipelines"
            )

        return recommendations


class DataPipelineMonitor:
    """Monitor data pipeline quality over time."""

    def __init__(self, storage_path: str = "mlops/data_quality"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.validator = DataQualityValidator()
        self.report_history: list[ValidationReport] = []

        # Load existing reports
        self._load_report_history()

    def _load_report_history(self):
        """Load validation report history."""
        reports_dir = self.storage_path / "reports"
        if reports_dir.exists():
            for report_file in reports_dir.glob("*.json"):
                try:
                    with open(report_file) as f:
                        report_data = json.load(f)
                    # Convert back to ValidationReport (simplified)
                    # In practice, you'd implement proper deserialization
                    logger.info(f"Loaded report: {report_file.name}")
                except Exception as e:
                    logger.error(f"Failed to load report {report_file}: {e}")

    def validate_and_monitor(
        self, df: pd.DataFrame, dataset_name: str, save_report: bool = True
    ) -> ValidationReport:
        """Validate dataset and store monitoring information."""
        # Run validation
        report = self.validator.validate_dataset(df, dataset_name)

        # Store report
        if save_report:
            self._save_report(report)

        # Add to history
        self.report_history.append(report)

        # Keep only last 100 reports in memory
        if len(self.report_history) > 100:
            self.report_history = self.report_history[-100:]

        # Alert on quality degradation
        self._check_quality_alerts(report)

        return report

    def _save_report(self, report: ValidationReport):
        """Save validation report to storage."""
        reports_dir = self.storage_path / "reports"
        reports_dir.mkdir(exist_ok=True)

        timestamp = report.validation_timestamp.strftime("%Y%m%d_%H%M%S")
        report_file = (
            reports_dir / f"{report.dataset_name}_{timestamp}_{report.report_id}.json"
        )

        try:
            with open(report_file, "w") as f:
                json.dump(asdict(report), f, indent=2, default=str)
            logger.info(f"Saved validation report: {report_file.name}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    def _check_quality_alerts(self, report: ValidationReport):
        """Check for quality degradation and trigger alerts."""
        # Get recent reports for same dataset
        recent_reports = [
            r
            for r in self.report_history[-10:]  # Last 10 reports
            if r.dataset_name == report.dataset_name
        ]

        if len(recent_reports) >= 2:
            # Compare with previous report
            prev_report = recent_reports[-2]
            current_score = report.data_quality_metrics.overall_score
            prev_score = prev_report.data_quality_metrics.overall_score

            # Alert on significant degradation
            if current_score < prev_score - 0.1:
                logger.warning(
                    f"Data quality degradation detected for {report.dataset_name}: "
                    f"{prev_score:.3f} -> {current_score:.3f}"
                )

    def get_quality_trends(self, dataset_name: str, days: int = 30) -> dict[str, Any]:
        """Get quality trends for a dataset."""
        cutoff_date = datetime.now() - timedelta(days=days)

        relevant_reports = [
            r
            for r in self.report_history
            if r.dataset_name == dataset_name and r.validation_timestamp >= cutoff_date
        ]

        if not relevant_reports:
            return {"message": "No reports found for the specified period"}

        # Extract trend data
        timestamps = [r.validation_timestamp for r in relevant_reports]
        overall_scores = [
            r.data_quality_metrics.overall_score for r in relevant_reports
        ]
        completeness_scores = [
            r.data_quality_metrics.completeness_score for r in relevant_reports
        ]
        validity_scores = [
            r.data_quality_metrics.validity_score for r in relevant_reports
        ]

        return {
            "dataset_name": dataset_name,
            "period_days": days,
            "report_count": len(relevant_reports),
            "trends": {
                "timestamps": [ts.isoformat() for ts in timestamps],
                "overall_score": overall_scores,
                "completeness_score": completeness_scores,
                "validity_score": validity_scores,
            },
            "summary": {
                "avg_overall_score": np.mean(overall_scores),
                "min_overall_score": np.min(overall_scores),
                "max_overall_score": np.max(overall_scores),
                "current_score": overall_scores[-1] if overall_scores else 0,
            },
        }


# Global data pipeline monitor
data_pipeline_monitor = DataPipelineMonitor()

# Export for use
__all__ = [
    "DataQualityValidator",
    "DataPipelineMonitor",
    "ValidationRule",
    "ValidationReport",
    "DataQualityMetrics",
    "ValidationRuleType",
    "ValidationSeverity",
    "data_pipeline_monitor",
]
