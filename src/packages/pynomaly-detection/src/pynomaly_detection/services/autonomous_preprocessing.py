"""Autonomous preprocessing components for intelligent data preparation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from pynomaly.domain.entities import Dataset
from pynomaly.infrastructure.preprocessing.data_cleaner import DataCleaner
from pynomaly.infrastructure.preprocessing.data_transformer import DataTransformer


class DataQualityIssue(Enum):
    """Types of data quality issues."""

    MISSING_VALUES = "missing_values"
    OUTLIERS = "outliers"
    DUPLICATES = "duplicates"
    CONSTANT_FEATURES = "constant_features"
    LOW_VARIANCE = "low_variance"
    INFINITE_VALUES = "infinite_values"
    ZERO_VALUES = "zero_values"
    HIGH_CARDINALITY = "high_cardinality"
    IMBALANCED_CATEGORIES = "imbalanced_categories"
    POOR_SCALING = "poor_scaling"


@dataclass
class QualityIssue:
    """Represents a specific data quality issue."""

    issue_type: DataQualityIssue
    severity: float  # 0.0 to 1.0
    affected_columns: list[str]
    description: str
    impact_on_detection: str
    recommended_strategies: list[str]
    metadata: dict[str, Any]


@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment."""

    overall_score: float  # 0.0 to 1.0
    issues: list[QualityIssue]
    preprocessing_required: bool
    estimated_improvement: float
    recommended_pipeline: dict[str, Any] | None
    processing_time_estimate: float
    memory_impact_estimate: str


@dataclass
class PreprocessingStrategy:
    """Recommended preprocessing strategy."""

    strategy_name: str
    confidence: float  # 0.0 to 1.0
    expected_improvement: float
    processing_time: float
    memory_impact: str
    steps: list[dict[str, Any]]
    reasoning: str


class AutonomousQualityAnalyzer:
    """Analyzes data quality and recommends preprocessing strategies."""

    def __init__(self):
        """Initialize the quality analyzer."""
        self.logger = logging.getLogger(__name__)

    def analyze_data_quality(self, dataset: Dataset) -> DataQualityReport:
        """Perform comprehensive data quality analysis.

        Args:
            dataset: Dataset to analyze

        Returns:
            Comprehensive quality report with issues and recommendations
        """
        df = dataset.data
        issues = []

        # Check for missing values
        missing_issue = self._check_missing_values(df)
        if missing_issue:
            issues.append(missing_issue)

        # Check for outliers
        outlier_issue = self._check_outliers(df)
        if outlier_issue:
            issues.append(outlier_issue)

        # Check for duplicates
        duplicate_issue = self._check_duplicates(df)
        if duplicate_issue:
            issues.append(duplicate_issue)

        # Check for constant features
        constant_issue = self._check_constant_features(df)
        if constant_issue:
            issues.append(constant_issue)

        # Check for low variance features
        variance_issue = self._check_low_variance(df)
        if variance_issue:
            issues.append(variance_issue)

        # Check for infinite values
        infinite_issue = self._check_infinite_values(df)
        if infinite_issue:
            issues.append(infinite_issue)

        # Check for scaling issues
        scaling_issue = self._check_scaling_issues(df)
        if scaling_issue:
            issues.append(scaling_issue)

        # Check categorical issues
        categorical_issues = self._check_categorical_issues(df)
        issues.extend(categorical_issues)

        # Calculate overall quality score
        overall_score = self._calculate_quality_score(issues, df)

        # Determine if preprocessing is required
        preprocessing_required = overall_score < 0.8 or len(issues) > 0

        # Estimate improvement potential
        estimated_improvement = self._estimate_improvement_potential(issues, df)

        # Generate recommended pipeline
        recommended_pipeline = None
        if preprocessing_required:
            recommended_pipeline = self._generate_pipeline_recommendation(issues, df)

        # Estimate processing requirements
        processing_time = self._estimate_processing_time(df, issues)
        memory_impact = self._estimate_memory_impact(df, issues)

        return DataQualityReport(
            overall_score=overall_score,
            issues=issues,
            preprocessing_required=preprocessing_required,
            estimated_improvement=estimated_improvement,
            recommended_pipeline=recommended_pipeline,
            processing_time_estimate=processing_time,
            memory_impact_estimate=memory_impact,
        )

    def _check_missing_values(self, df: pd.DataFrame) -> QualityIssue | None:
        """Check for missing values."""
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0].index.tolist()

        if not missing_cols:
            return None

        total_missing = missing_counts.sum()
        missing_ratio = total_missing / (df.shape[0] * df.shape[1])

        # Severity based on percentage of missing data
        severity = min(missing_ratio * 3, 1.0)  # Cap at 1.0

        # Recommend strategies based on missing pattern
        strategies = []
        for col in missing_cols:
            col_missing_ratio = missing_counts[col] / len(df)
            if col_missing_ratio > 0.5:
                strategies.append("drop_columns")
            elif df[col].dtype in ["int64", "float64"]:
                if col_missing_ratio < 0.1:
                    strategies.append("fill_median")
                else:
                    strategies.append("knn_impute")
            else:
                strategies.append("fill_mode")

        return QualityIssue(
            issue_type=DataQualityIssue.MISSING_VALUES,
            severity=severity,
            affected_columns=missing_cols,
            description=f"Found {total_missing:,} missing values across {len(missing_cols)} columns",
            impact_on_detection="Missing values can cause algorithm failures or biased results",
            recommended_strategies=list(set(strategies)),
            metadata={
                "total_missing": total_missing,
                "missing_ratio": missing_ratio,
                "columns_affected": len(missing_cols),
            },
        )

    def _check_outliers(self, df: pd.DataFrame) -> QualityIssue | None:
        """Check for outliers using IQR method."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return None

        outlier_counts = {}
        total_outliers = 0

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR > 0:  # Avoid division by zero
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_counts[col] = outliers
                total_outliers += outliers

        if total_outliers == 0:
            return None

        outlier_ratio = total_outliers / (len(df) * len(numeric_cols))
        severity = min(outlier_ratio * 10, 1.0)  # More sensitive to outliers

        # Recommend strategy based on outlier density
        if outlier_ratio > 0.1:
            strategies = ["winsorize", "clip"]
        elif outlier_ratio > 0.05:
            strategies = ["clip", "transform_log"]
        else:
            strategies = ["clip"]

        affected_cols = [col for col, count in outlier_counts.items() if count > 0]

        return QualityIssue(
            issue_type=DataQualityIssue.OUTLIERS,
            severity=severity,
            affected_columns=affected_cols,
            description=f"Found {total_outliers:,} outliers across {len(affected_cols)} numeric columns",
            impact_on_detection="Outliers can dominate distance-based algorithms and skew results",
            recommended_strategies=strategies,
            metadata={
                "total_outliers": total_outliers,
                "outlier_ratio": outlier_ratio,
                "outlier_counts": outlier_counts,
            },
        )

    def _check_duplicates(self, df: pd.DataFrame) -> QualityIssue | None:
        """Check for duplicate rows."""
        duplicate_count = df.duplicated().sum()

        if duplicate_count == 0:
            return None

        duplicate_ratio = duplicate_count / len(df)
        severity = min(duplicate_ratio * 5, 1.0)

        return QualityIssue(
            issue_type=DataQualityIssue.DUPLICATES,
            severity=severity,
            affected_columns=list(df.columns),
            description=f"Found {duplicate_count:,} duplicate rows ({duplicate_ratio:.1%} of data)",
            impact_on_detection="Duplicates can bias algorithms toward repeated patterns",
            recommended_strategies=["remove"],
            metadata={
                "duplicate_count": duplicate_count,
                "duplicate_ratio": duplicate_ratio,
            },
        )

    def _check_constant_features(self, df: pd.DataFrame) -> QualityIssue | None:
        """Check for constant features."""
        constant_cols = []

        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)

        if not constant_cols:
            return None

        severity = len(constant_cols) / len(df.columns)

        return QualityIssue(
            issue_type=DataQualityIssue.CONSTANT_FEATURES,
            severity=severity,
            affected_columns=constant_cols,
            description=f"Found {len(constant_cols)} constant features with no variation",
            impact_on_detection="Constant features provide no information for anomaly detection",
            recommended_strategies=["remove"],
            metadata={"constant_count": len(constant_cols)},
        )

    def _check_low_variance(self, df: pd.DataFrame) -> QualityIssue | None:
        """Check for low variance features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        low_variance_cols = []

        for col in numeric_cols:
            if df[col].var() < 0.01:  # Very low variance threshold
                low_variance_cols.append(col)

        if not low_variance_cols:
            return None

        severity = (
            len(low_variance_cols) / len(numeric_cols) if len(numeric_cols) > 0 else 0
        )

        return QualityIssue(
            issue_type=DataQualityIssue.LOW_VARIANCE,
            severity=severity,
            affected_columns=low_variance_cols,
            description=f"Found {len(low_variance_cols)} low variance features",
            impact_on_detection="Low variance features contribute little to anomaly detection",
            recommended_strategies=["variance_threshold"],
            metadata={"low_variance_count": len(low_variance_cols)},
        )

    def _check_infinite_values(self, df: pd.DataFrame) -> QualityIssue | None:
        """Check for infinite values."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        infinite_cols = []
        total_infinite = 0

        for col in numeric_cols:
            infinite_count = np.isinf(df[col]).sum()
            if infinite_count > 0:
                infinite_cols.append(col)
                total_infinite += infinite_count

        if not infinite_cols:
            return None

        infinite_ratio = total_infinite / (len(df) * len(numeric_cols))
        severity = min(infinite_ratio * 10, 1.0)

        return QualityIssue(
            issue_type=DataQualityIssue.INFINITE_VALUES,
            severity=severity,
            affected_columns=infinite_cols,
            description=f"Found {total_infinite:,} infinite values across {len(infinite_cols)} columns",
            impact_on_detection="Infinite values cause numerical instability in algorithms",
            recommended_strategies=["remove", "clip"],
            metadata={
                "total_infinite": total_infinite,
                "infinite_ratio": infinite_ratio,
            },
        )

    def _check_scaling_issues(self, df: pd.DataFrame) -> QualityIssue | None:
        """Check for scaling issues between numeric features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return None

        # Calculate scale differences
        scales = []
        for col in numeric_cols:
            scale = abs(df[col].mean()) + df[col].std()
            scales.append(scale)

        if len(scales) < 2:
            return None

        max_scale = max(scales)
        min_scale = min(scales)

        # If scale difference is more than 100x, recommend scaling
        if max_scale / min_scale > 100:
            severity = min(np.log10(max_scale / min_scale) / 3, 1.0)

            return QualityIssue(
                issue_type=DataQualityIssue.POOR_SCALING,
                severity=severity,
                affected_columns=list(numeric_cols),
                description=f"Large scale differences detected (max/min ratio: {max_scale / min_scale:.1f})",
                impact_on_detection="Poor scaling can cause distance-based algorithms to focus on large-scale features",
                recommended_strategies=["standard", "minmax", "robust"],
                metadata={
                    "scale_ratio": max_scale / min_scale,
                    "max_scale": max_scale,
                    "min_scale": min_scale,
                },
            )

        return None

    def _check_categorical_issues(self, df: pd.DataFrame) -> list[QualityIssue]:
        """Check for categorical data issues."""
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        issues = []

        for col in categorical_cols:
            unique_values = df[col].nunique()
            total_values = len(df[col].dropna())

            # High cardinality check
            if unique_values > 50 and unique_values / total_values > 0.5:
                issues.append(
                    QualityIssue(
                        issue_type=DataQualityIssue.HIGH_CARDINALITY,
                        severity=min(unique_values / total_values, 1.0),
                        affected_columns=[col],
                        description=f"High cardinality feature '{col}' with {unique_values} unique values",
                        impact_on_detection="High cardinality can create sparse, high-dimensional representations",
                        recommended_strategies=["frequency", "target", "binary"],
                        metadata={
                            "unique_values": unique_values,
                            "cardinality_ratio": unique_values / total_values,
                        },
                    )
                )

            # Imbalanced categories check
            value_counts = df[col].value_counts()
            if len(value_counts) > 1:
                max_freq = value_counts.iloc[0]
                min_freq = value_counts.iloc[-1]

                if max_freq / min_freq > 100:  # Very imbalanced
                    issues.append(
                        QualityIssue(
                            issue_type=DataQualityIssue.IMBALANCED_CATEGORIES,
                            severity=min(np.log10(max_freq / min_freq) / 3, 1.0),
                            affected_columns=[col],
                            description=f"Imbalanced categories in '{col}' (max/min ratio: {max_freq / min_freq:.1f})",
                            impact_on_detection="Imbalanced categories can bias algorithms toward frequent categories",
                            recommended_strategies=["frequency", "target"],
                            metadata={
                                "imbalance_ratio": max_freq / min_freq,
                                "max_frequency": max_freq,
                                "min_frequency": min_freq,
                            },
                        )
                    )

        return issues

    def _calculate_quality_score(
        self, issues: list[QualityIssue], df: pd.DataFrame
    ) -> float:
        """Calculate overall data quality score."""
        if not issues:
            return 1.0

        # Weighted score based on issue severity and type
        weights = {
            DataQualityIssue.MISSING_VALUES: 0.3,
            DataQualityIssue.OUTLIERS: 0.2,
            DataQualityIssue.DUPLICATES: 0.1,
            DataQualityIssue.CONSTANT_FEATURES: 0.15,
            DataQualityIssue.LOW_VARIANCE: 0.1,
            DataQualityIssue.INFINITE_VALUES: 0.25,
            DataQualityIssue.ZERO_VALUES: 0.05,
            DataQualityIssue.HIGH_CARDINALITY: 0.1,
            DataQualityIssue.IMBALANCED_CATEGORIES: 0.05,
            DataQualityIssue.POOR_SCALING: 0.2,
        }

        total_penalty = 0.0
        total_weight = 0.0

        for issue in issues:
            weight = weights.get(issue.issue_type, 0.1)
            penalty = issue.severity * weight
            total_penalty += penalty
            total_weight += weight

        # Normalize and convert to quality score
        if total_weight > 0:
            average_penalty = total_penalty / total_weight
            quality_score = max(0.0, 1.0 - average_penalty)
        else:
            quality_score = 1.0

        return quality_score

    def _estimate_improvement_potential(
        self, issues: list[QualityIssue], df: pd.DataFrame
    ) -> float:
        """Estimate potential improvement from preprocessing."""
        if not issues:
            return 0.0

        # Base improvement on issue severity and detectability impact
        total_improvement = 0.0

        for issue in issues:
            # Different issues have different improvement potential
            if issue.issue_type == DataQualityIssue.MISSING_VALUES:
                improvement = issue.severity * 0.3
            elif issue.issue_type == DataQualityIssue.OUTLIERS:
                improvement = issue.severity * 0.4
            elif issue.issue_type == DataQualityIssue.POOR_SCALING:
                improvement = issue.severity * 0.5
            elif issue.issue_type == DataQualityIssue.CONSTANT_FEATURES:
                improvement = issue.severity * 0.2
            else:
                improvement = issue.severity * 0.2

            total_improvement += improvement

        return min(total_improvement, 0.8)  # Cap at 80% improvement

    def _generate_pipeline_recommendation(
        self, issues: list[QualityIssue], df: pd.DataFrame
    ) -> dict[str, Any]:
        """Generate recommended preprocessing pipeline."""
        steps = []
        step_counter = 1

        # Priority order for preprocessing steps
        issue_priority = [
            DataQualityIssue.INFINITE_VALUES,
            DataQualityIssue.MISSING_VALUES,
            DataQualityIssue.DUPLICATES,
            DataQualityIssue.CONSTANT_FEATURES,
            DataQualityIssue.OUTLIERS,
            DataQualityIssue.LOW_VARIANCE,
            DataQualityIssue.POOR_SCALING,
            DataQualityIssue.HIGH_CARDINALITY,
            DataQualityIssue.IMBALANCED_CATEGORIES,
        ]

        # Group issues by type
        issues_by_type = {issue.issue_type: issue for issue in issues}

        # Generate steps in priority order
        for issue_type in issue_priority:
            if issue_type not in issues_by_type:
                continue

            issue = issues_by_type[issue_type]

            if issue_type == DataQualityIssue.INFINITE_VALUES:
                steps.append(
                    {
                        "name": f"handle_infinite_values_{step_counter}",
                        "operation": "handle_infinite_values",
                        "parameters": {"strategy": "remove"},
                        "enabled": True,
                        "description": "Remove infinite values that cause numerical instability",
                    }
                )
                step_counter += 1

            elif issue_type == DataQualityIssue.MISSING_VALUES:
                # Choose strategy based on data characteristics
                if issue.metadata.get("missing_ratio", 0) > 0.3:
                    strategy = "drop_columns"
                else:
                    strategy = "fill_median"

                steps.append(
                    {
                        "name": f"handle_missing_values_{step_counter}",
                        "operation": "handle_missing_values",
                        "parameters": {"strategy": strategy},
                        "enabled": True,
                        "description": f"Handle missing values using {strategy} strategy",
                    }
                )
                step_counter += 1

            elif issue_type == DataQualityIssue.DUPLICATES:
                steps.append(
                    {
                        "name": f"remove_duplicates_{step_counter}",
                        "operation": "remove_duplicates",
                        "parameters": {},
                        "enabled": True,
                        "description": "Remove duplicate rows to avoid bias",
                    }
                )
                step_counter += 1

            elif issue_type == DataQualityIssue.CONSTANT_FEATURES:
                steps.append(
                    {
                        "name": f"remove_constant_features_{step_counter}",
                        "operation": "remove_constant_features",
                        "parameters": {},
                        "enabled": True,
                        "description": "Remove constant features with no variation",
                    }
                )
                step_counter += 1

            elif issue_type == DataQualityIssue.OUTLIERS:
                # Choose strategy based on outlier ratio
                outlier_ratio = issue.metadata.get("outlier_ratio", 0)
                if outlier_ratio > 0.1:
                    strategy = "winsorize"
                else:
                    strategy = "clip"

                steps.append(
                    {
                        "name": f"handle_outliers_{step_counter}",
                        "operation": "handle_outliers",
                        "parameters": {"strategy": strategy, "threshold": 3.0},
                        "enabled": True,
                        "description": f"Handle outliers using {strategy} method",
                    }
                )
                step_counter += 1

            elif issue_type == DataQualityIssue.POOR_SCALING:
                steps.append(
                    {
                        "name": f"scale_features_{step_counter}",
                        "operation": "scale_features",
                        "parameters": {"strategy": "standard"},
                        "enabled": True,
                        "description": "Apply standard scaling to normalize feature scales",
                    }
                )
                step_counter += 1

            elif issue_type == DataQualityIssue.HIGH_CARDINALITY:
                steps.append(
                    {
                        "name": f"encode_categorical_{step_counter}",
                        "operation": "encode_categorical",
                        "parameters": {"strategy": "frequency"},
                        "enabled": True,
                        "description": "Encode high cardinality features using frequency encoding",
                    }
                )
                step_counter += 1

        # Add feature selection if many features
        if len(df.columns) > 50:
            steps.append(
                {
                    "name": f"select_features_{step_counter}",
                    "operation": "select_features",
                    "parameters": {"strategy": "variance_threshold"},
                    "enabled": True,
                    "description": "Remove low variance features to reduce dimensionality",
                }
            )

        return {
            "name": "autonomous_preprocessing_pipeline",
            "description": "Auto-generated preprocessing pipeline for anomaly detection",
            "steps": steps,
        }

    def _estimate_processing_time(
        self, df: pd.DataFrame, issues: list[QualityIssue]
    ) -> float:
        """Estimate preprocessing time in seconds."""
        base_time = len(df) * len(df.columns) / 100000  # Base time per operation

        time_multipliers = {
            DataQualityIssue.MISSING_VALUES: 2.0,
            DataQualityIssue.OUTLIERS: 1.5,
            DataQualityIssue.DUPLICATES: 1.2,
            DataQualityIssue.CONSTANT_FEATURES: 1.0,
            DataQualityIssue.LOW_VARIANCE: 1.0,
            DataQualityIssue.INFINITE_VALUES: 1.0,
            DataQualityIssue.POOR_SCALING: 1.5,
            DataQualityIssue.HIGH_CARDINALITY: 3.0,
        }

        total_time = base_time
        for issue in issues:
            multiplier = time_multipliers.get(issue.issue_type, 1.0)
            total_time += base_time * multiplier * issue.severity

        return total_time

    def _estimate_memory_impact(
        self, df: pd.DataFrame, issues: list[QualityIssue]
    ) -> str:
        """Estimate memory impact of preprocessing."""
        df.memory_usage(deep=True).sum()

        # Estimate memory changes based on operations
        memory_change = 1.0

        for issue in issues:
            if issue.issue_type == DataQualityIssue.HIGH_CARDINALITY:
                memory_change *= 2.0  # Encoding can increase memory
            elif issue.issue_type == DataQualityIssue.CONSTANT_FEATURES:
                memory_change *= 0.9  # Removing features reduces memory
            elif issue.issue_type == DataQualityIssue.DUPLICATES:
                duplicate_ratio = issue.metadata.get("duplicate_ratio", 0)
                memory_change *= 1.0 - duplicate_ratio

        if memory_change > 1.5:
            return "High (50%+ increase)"
        elif memory_change > 1.2:
            return "Moderate (20-50% increase)"
        elif memory_change < 0.8:
            return "Reduction (20%+ decrease)"
        else:
            return "Low (minimal change)"


class AutonomousPreprocessingOrchestrator:
    """Orchestrates preprocessing steps within autonomous detection workflow."""

    def __init__(self):
        """Initialize the orchestrator."""
        self.quality_analyzer = AutonomousQualityAnalyzer()
        self.cleaner = DataCleaner()
        self.transformer = DataTransformer()
        self.logger = logging.getLogger(__name__)

    def should_preprocess(
        self, dataset: Dataset, quality_threshold: float = 0.8
    ) -> tuple[bool, DataQualityReport]:
        """Determine if preprocessing is needed.

        Args:
            dataset: Dataset to analyze
            quality_threshold: Minimum quality score to skip preprocessing

        Returns:
            Tuple of (should_preprocess, quality_report)
        """
        quality_report = self.quality_analyzer.analyze_data_quality(dataset)
        should_process = (
            quality_report.overall_score < quality_threshold
            or quality_report.preprocessing_required
        )

        return should_process, quality_report

    def preprocess_for_autonomous_detection(
        self,
        dataset: Dataset,
        quality_report: DataQualityReport,
        max_processing_time: float = 300.0,  # 5 minutes max
    ) -> tuple[Dataset, dict[str, Any]]:
        """Apply preprocessing optimized for autonomous anomaly detection.

        Args:
            dataset: Dataset to preprocess
            quality_report: Quality analysis results
            max_processing_time: Maximum allowed processing time

        Returns:
            Tuple of (preprocessed_dataset, preprocessing_metadata)
        """
        if not quality_report.preprocessing_required:
            return dataset, {
                "preprocessing_applied": False,
                "reason": "No preprocessing needed",
            }

        # Skip if estimated processing time is too long
        if quality_report.processing_time_estimate > max_processing_time:
            self.logger.warning(
                f"Skipping preprocessing: estimated time {quality_report.processing_time_estimate:.1f}s "
                f"exceeds limit {max_processing_time:.1f}s"
            )
            return dataset, {
                "preprocessing_applied": False,
                "reason": "Processing time too long",
                "estimated_time": quality_report.processing_time_estimate,
            }

        # Apply preprocessing pipeline
        df = dataset.data.copy()
        applied_steps = []

        try:
            # Apply high-priority fixes first
            for issue in sorted(
                quality_report.issues, key=lambda x: x.severity, reverse=True
            ):
                step_applied = self._apply_issue_fix(df, issue)
                if step_applied:
                    applied_steps.append(step_applied)

            # Create new dataset with preprocessed data
            preprocessed_dataset = Dataset(
                id=dataset.id,
                name=f"{dataset.name}_preprocessed",
                description=f"Preprocessed version of {dataset.name}",
                data=df,
                target_column=(
                    dataset.target_column if hasattr(dataset, "target_column") else None
                ),
                metadata={
                    **dataset.metadata,
                    "preprocessing_applied": True,
                    "original_shape": dataset.data.shape,
                    "preprocessed_shape": df.shape,
                },
            )

            metadata = {
                "preprocessing_applied": True,
                "applied_steps": applied_steps,
                "original_shape": dataset.data.shape,
                "final_shape": df.shape,
                "quality_improvement": quality_report.estimated_improvement,
                "processing_successful": True,
            }

            return preprocessed_dataset, metadata

        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            return dataset, {
                "preprocessing_applied": False,
                "reason": f"Processing failed: {str(e)}",
                "error": str(e),
            }

    def _apply_issue_fix(
        self, df: pd.DataFrame, issue: QualityIssue
    ) -> dict[str, Any] | None:
        """Apply fix for a specific data quality issue.

        Args:
            df: DataFrame to modify (in-place)
            issue: Quality issue to fix

        Returns:
            Metadata about applied fix, or None if not applied
        """
        try:
            if issue.issue_type == DataQualityIssue.INFINITE_VALUES:
                # Remove infinite values
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    df = df.dropna(subset=[col])

                return {
                    "type": "infinite_values",
                    "action": "removed",
                    "columns": list(numeric_cols),
                }

            elif issue.issue_type == DataQualityIssue.MISSING_VALUES:
                # Handle missing values with appropriate strategy
                for col in issue.affected_columns:
                    if df[col].dtype in ["int64", "float64"]:
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna(
                            df[col].mode().iloc[0]
                            if len(df[col].mode()) > 0
                            else "unknown"
                        )

                return {
                    "type": "missing_values",
                    "action": "filled",
                    "columns": issue.affected_columns,
                    "strategy": "median/mode",
                }

            elif issue.issue_type == DataQualityIssue.DUPLICATES:
                original_len = len(df)
                df.drop_duplicates(inplace=True)
                removed = original_len - len(df)

                return {"type": "duplicates", "action": "removed", "count": removed}

            elif issue.issue_type == DataQualityIssue.CONSTANT_FEATURES:
                df.drop(columns=issue.affected_columns, inplace=True)

                return {
                    "type": "constant_features",
                    "action": "removed",
                    "columns": issue.affected_columns,
                }

            elif issue.issue_type == DataQualityIssue.OUTLIERS:
                # Apply clipping to outliers
                for col in issue.affected_columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

                return {
                    "type": "outliers",
                    "action": "clipped",
                    "columns": issue.affected_columns,
                }

            elif issue.issue_type == DataQualityIssue.POOR_SCALING:
                # Apply standard scaling
                from sklearn.preprocessing import StandardScaler

                numeric_cols = df.select_dtypes(include=[np.number]).columns
                scaler = StandardScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

                return {
                    "type": "scaling",
                    "action": "standardized",
                    "columns": list(numeric_cols),
                }

            return None

        except Exception as e:
            self.logger.error(f"Failed to apply fix for {issue.issue_type}: {str(e)}")
            return None
