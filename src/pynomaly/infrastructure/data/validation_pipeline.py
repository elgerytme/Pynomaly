"""Comprehensive data validation and preprocessing pipeline for Pynomaly."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class DataQuality(Enum):
    """Data quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNUSABLE = "unusable"


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a data validation issue."""
    severity: ValidationSeverity
    message: str
    column: Optional[str] = None
    row_indices: Optional[List[int]] = None
    recommendation: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment report."""
    overall_quality: DataQuality
    score: float  # 0-100
    issues: List[ValidationIssue]
    statistics: Dict[str, Any]
    recommendations: List[str]
    
    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues."""
        return any(issue.severity == ValidationSeverity.CRITICAL for issue in self.issues)
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues by severity level."""
        return [issue for issue in self.issues if issue.severity == severity]


class DataValidationPipeline:
    """Comprehensive data validation and preprocessing pipeline."""
    
    def __init__(self, 
                 min_samples: int = 10,
                 min_features: int = 1,
                 max_missing_ratio: float = 0.3,
                 max_constant_features_ratio: float = 0.8,
                 outlier_detection_threshold: float = 3.0):
        """Initialize validation pipeline.
        
        Args:
            min_samples: Minimum number of samples required
            min_features: Minimum number of features required
            max_missing_ratio: Maximum allowed ratio of missing values
            max_constant_features_ratio: Maximum allowed ratio of constant features
            outlier_detection_threshold: Z-score threshold for outlier detection
        """
        self.min_samples = min_samples
        self.min_features = min_features
        self.max_missing_ratio = max_missing_ratio
        self.max_constant_features_ratio = max_constant_features_ratio
        self.outlier_detection_threshold = outlier_detection_threshold
    
    def validate(self, data: pd.DataFrame) -> DataQualityReport:
        """Perform comprehensive data validation.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Comprehensive data quality report
        """
        issues = []
        statistics = {}
        
        # Basic structure validation
        issues.extend(self._validate_structure(data, statistics))
        
        # Data type validation
        issues.extend(self._validate_data_types(data, statistics))
        
        # Missing values validation
        issues.extend(self._validate_missing_values(data, statistics))
        
        # Feature quality validation
        issues.extend(self._validate_feature_quality(data, statistics))
        
        # Outlier analysis
        issues.extend(self._analyze_outliers(data, statistics))
        
        # Data distribution analysis
        issues.extend(self._analyze_distributions(data, statistics))
        
        # Calculate overall quality score
        score = self._calculate_quality_score(issues, statistics)
        overall_quality = self._determine_overall_quality(score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, statistics)
        
        return DataQualityReport(
            overall_quality=overall_quality,
            score=score,
            issues=issues,
            statistics=statistics,
            recommendations=recommendations
        )
    
    def _validate_structure(self, data: pd.DataFrame, statistics: Dict) -> List[ValidationIssue]:
        """Validate basic data structure."""
        issues = []
        
        n_samples, n_features = data.shape
        statistics['n_samples'] = n_samples
        statistics['n_features'] = n_features
        
        # Check minimum samples
        if n_samples < self.min_samples:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                message=f"Insufficient samples: {n_samples} < {self.min_samples}",
                recommendation="Collect more data samples for reliable anomaly detection",
                details={"current": n_samples, "required": self.min_samples}
            ))
        
        # Check minimum features
        if n_features < self.min_features:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                message=f"Insufficient features: {n_features} < {self.min_features}",
                recommendation="Ensure dataset has at least one numeric feature",
                details={"current": n_features, "required": self.min_features}
            ))
        
        # Check for empty dataset
        if data.empty:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                message="Dataset is empty",
                recommendation="Provide a non-empty dataset"
            ))
        
        return issues
    
    def _validate_data_types(self, data: pd.DataFrame, statistics: Dict) -> List[ValidationIssue]:
        """Validate data types and detect numeric features."""
        issues = []
        
        # Identify numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()
        
        statistics['numeric_columns'] = numeric_columns
        statistics['non_numeric_columns'] = non_numeric_columns
        statistics['numeric_feature_ratio'] = len(numeric_columns) / len(data.columns) if len(data.columns) > 0 else 0
        
        # Check if we have any numeric features
        if not numeric_columns:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                message="No numeric features found",
                recommendation="Ensure dataset contains numeric columns for anomaly detection",
                details={"all_columns": data.columns.tolist()}
            ))
        
        # Warn about non-numeric columns that will be ignored
        if non_numeric_columns:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"Non-numeric columns will be ignored: {non_numeric_columns}",
                recommendation="Convert categorical features to numeric if needed for analysis",
                details={"non_numeric_columns": non_numeric_columns}
            ))
        
        return issues
    
    def _validate_missing_values(self, data: pd.DataFrame, statistics: Dict) -> List[ValidationIssue]:
        """Validate missing values."""
        issues = []
        
        missing_counts = data.isnull().sum()
        missing_ratios = missing_counts / len(data)
        total_missing_ratio = data.isnull().sum().sum() / data.size
        
        statistics['missing_counts'] = missing_counts.to_dict()
        statistics['missing_ratios'] = missing_ratios.to_dict()
        statistics['total_missing_ratio'] = total_missing_ratio
        
        # Check overall missing ratio
        if total_missing_ratio > self.max_missing_ratio:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"High missing value ratio: {total_missing_ratio:.2%} > {self.max_missing_ratio:.2%}",
                recommendation="Consider data imputation or collecting more complete data",
                details={"ratio": total_missing_ratio, "threshold": self.max_missing_ratio}
            ))
        
        # Check individual columns
        for column, ratio in missing_ratios.items():
            if ratio > 0.5:  # More than 50% missing
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Column '{column}' has {ratio:.2%} missing values",
                    column=column,
                    recommendation=f"Consider removing column '{column}' or imputing missing values",
                    details={"ratio": ratio, "count": missing_counts[column]}
                ))
        
        return issues
    
    def _validate_feature_quality(self, data: pd.DataFrame, statistics: Dict) -> List[ValidationIssue]:
        """Validate feature quality (constant features, low variance, etc.)."""
        issues = []
        
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return issues
        
        # Check for constant features
        constant_features = []
        low_variance_features = []
        
        for column in numeric_data.columns:
            col_data = numeric_data[column].dropna()
            if len(col_data) == 0:
                continue
                
            # Check if constant
            if col_data.nunique() <= 1:
                constant_features.append(column)
            else:
                # Check for low variance
                variance = col_data.var()
                if variance < 1e-10:  # Very low variance
                    low_variance_features.append(column)
        
        statistics['constant_features'] = constant_features
        statistics['low_variance_features'] = low_variance_features
        statistics['constant_feature_ratio'] = len(constant_features) / len(numeric_data.columns)
        
        # Report constant features
        if constant_features:
            if len(constant_features) / len(numeric_data.columns) > self.max_constant_features_ratio:
                severity = ValidationSeverity.ERROR
            else:
                severity = ValidationSeverity.WARNING
                
            issues.append(ValidationIssue(
                severity=severity,
                message=f"Constant features detected: {constant_features}",
                recommendation="Remove constant features as they don't contribute to anomaly detection",
                details={"constant_features": constant_features}
            ))
        
        # Report low variance features
        if low_variance_features:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"Low variance features detected: {low_variance_features}",
                recommendation="Consider feature scaling or transformation",
                details={"low_variance_features": low_variance_features}
            ))
        
        return issues
    
    def _analyze_outliers(self, data: pd.DataFrame, statistics: Dict) -> List[ValidationIssue]:
        """Analyze potential outliers in the data."""
        issues = []
        
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return issues
        
        outlier_info = {}
        total_outliers = 0
        
        for column in numeric_data.columns:
            col_data = numeric_data[column].dropna()
            if len(col_data) < 3:  # Need at least 3 points for outlier detection
                continue
            
            # Z-score based outlier detection
            z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
            outlier_mask = z_scores > self.outlier_detection_threshold
            outlier_count = outlier_mask.sum()
            outlier_indices = col_data[outlier_mask].index.tolist()
            
            outlier_info[column] = {
                'count': outlier_count,
                'ratio': outlier_count / len(col_data),
                'indices': outlier_indices
            }
            total_outliers += outlier_count
        
        statistics['outlier_analysis'] = outlier_info
        statistics['total_outlier_ratio'] = total_outliers / len(data) if len(data) > 0 else 0
        
        # Report outlier findings
        high_outlier_columns = [col for col, info in outlier_info.items() 
                               if info['ratio'] > 0.1]  # More than 10% outliers
        
        if high_outlier_columns:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"High outlier ratio in columns: {high_outlier_columns}",
                recommendation="This is normal for anomaly detection datasets, but verify data quality",
                details={"high_outlier_columns": {col: outlier_info[col] for col in high_outlier_columns}}
            ))
        
        return issues
    
    def _analyze_distributions(self, data: pd.DataFrame, statistics: Dict) -> List[ValidationIssue]:
        """Analyze data distributions."""
        issues = []
        
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return issues
        
        distribution_info = {}
        
        for column in numeric_data.columns:
            col_data = numeric_data[column].dropna()
            if len(col_data) < 3:
                continue
            
            # Basic distribution statistics
            skewness = col_data.skew()
            kurtosis = col_data.kurtosis()
            
            distribution_info[column] = {
                'mean': col_data.mean(),
                'std': col_data.std(),
                'skewness': skewness,
                'kurtosis': kurtosis,
                'min': col_data.min(),
                'max': col_data.max()
            }
        
        statistics['distribution_analysis'] = distribution_info
        
        # Check for highly skewed distributions
        highly_skewed = [col for col, info in distribution_info.items() 
                        if abs(info['skewness']) > 2]
        
        if highly_skewed:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"Highly skewed distributions in: {highly_skewed}",
                recommendation="Consider data transformation (log, sqrt) for better algorithm performance",
                details={"skewed_columns": {col: distribution_info[col]['skewness'] 
                                          for col in highly_skewed}}
            ))
        
        return issues
    
    def _calculate_quality_score(self, issues: List[ValidationIssue], statistics: Dict) -> float:
        """Calculate overall data quality score (0-100)."""
        base_score = 100.0
        
        # Deduct points based on issue severity
        for issue in issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                base_score -= 40
            elif issue.severity == ValidationSeverity.ERROR:
                base_score -= 20
            elif issue.severity == ValidationSeverity.WARNING:
                base_score -= 10
            elif issue.severity == ValidationSeverity.INFO:
                base_score -= 2
        
        # Adjust based on data characteristics
        if 'total_missing_ratio' in statistics:
            base_score -= statistics['total_missing_ratio'] * 30
        
        if 'constant_feature_ratio' in statistics:
            base_score -= statistics['constant_feature_ratio'] * 25
        
        return max(0.0, min(100.0, base_score))
    
    def _determine_overall_quality(self, score: float) -> DataQuality:
        """Determine overall quality level based on score."""
        if score >= 90:
            return DataQuality.EXCELLENT
        elif score >= 75:
            return DataQuality.GOOD
        elif score >= 60:
            return DataQuality.FAIR
        elif score >= 40:
            return DataQuality.POOR
        else:
            return DataQuality.UNUSABLE
    
    def _generate_recommendations(self, issues: List[ValidationIssue], statistics: Dict) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Collect unique recommendations from issues
        issue_recommendations = set()
        for issue in issues:
            if issue.recommendation:
                issue_recommendations.add(issue.recommendation)
        
        recommendations.extend(sorted(issue_recommendations))
        
        # Add general recommendations based on statistics
        if statistics.get('total_missing_ratio', 0) > 0.1:
            recommendations.append("Consider implementing data imputation strategies for missing values")
        
        if statistics.get('numeric_feature_ratio', 1) < 0.5:
            recommendations.append("Increase the proportion of numeric features for better anomaly detection")
        
        if len(recommendations) == 0:
            recommendations.append("Data quality is good - proceed with anomaly detection")
        
        return recommendations


def preprocess_for_anomaly_detection(data: pd.DataFrame, 
                                    handle_missing: str = 'drop',
                                    scale_features: bool = True,
                                    remove_constant: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Preprocess data for anomaly detection.
    
    Args:
        data: Input DataFrame
        handle_missing: How to handle missing values ('drop', 'mean', 'median', 'mode')
        scale_features: Whether to scale features to unit variance
        remove_constant: Whether to remove constant features
        
    Returns:
        Tuple of (processed_data, preprocessing_info)
    """
    processed_data = data.copy()
    preprocessing_info = {
        'original_shape': data.shape,
        'steps_applied': [],
        'removed_columns': [],
        'transformations': {}
    }
    
    # Get numeric columns only
    numeric_columns = processed_data.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_columns = processed_data.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if non_numeric_columns:
        processed_data = processed_data[numeric_columns]
        preprocessing_info['removed_columns'].extend(non_numeric_columns)
        preprocessing_info['steps_applied'].append('removed_non_numeric_columns')
    
    # Handle missing values
    if handle_missing == 'drop':
        original_len = len(processed_data)
        processed_data = processed_data.dropna()
        dropped_rows = original_len - len(processed_data)
        if dropped_rows > 0:
            preprocessing_info['steps_applied'].append(f'dropped_{dropped_rows}_rows_with_missing_values')
    elif handle_missing in ['mean', 'median']:
        fill_values = {}
        for column in processed_data.columns:
            if processed_data[column].isnull().any():
                if handle_missing == 'mean':
                    fill_value = processed_data[column].mean()
                else:  # median
                    fill_value = processed_data[column].median()
                processed_data[column].fillna(fill_value, inplace=True)
                fill_values[column] = fill_value
        if fill_values:
            preprocessing_info['transformations']['missing_value_imputation'] = fill_values
            preprocessing_info['steps_applied'].append(f'imputed_missing_values_with_{handle_missing}')
    
    # Remove constant features
    if remove_constant:
        constant_columns = []
        for column in processed_data.columns:
            if processed_data[column].nunique() <= 1:
                constant_columns.append(column)
        
        if constant_columns:
            processed_data = processed_data.drop(columns=constant_columns)
            preprocessing_info['removed_columns'].extend(constant_columns)
            preprocessing_info['steps_applied'].append(f'removed_{len(constant_columns)}_constant_features')
    
    # Scale features
    if scale_features and not processed_data.empty:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        scaled_data = scaler.fit_transform(processed_data)
        processed_data = pd.DataFrame(scaled_data, columns=processed_data.columns, index=processed_data.index)
        
        preprocessing_info['transformations']['feature_scaling'] = {
            'method': 'StandardScaler',
            'means': scaler.mean_.tolist(),
            'scales': scaler.scale_.tolist()
        }
        preprocessing_info['steps_applied'].append('applied_standard_scaling')
    
    preprocessing_info['final_shape'] = processed_data.shape
    
    return processed_data, preprocessing_info