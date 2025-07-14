import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from ...domain.entities.data_profile import (
    QualityAssessment, QualityIssue, QualityIssueType, 
    ColumnProfile, SchemaProfile
)


class QualityAssessmentService:
    """Advanced service to perform comprehensive data quality assessment."""
    
    def __init__(self):
        # Quality dimension weights (can be customized)
        self.default_weights = {
            'completeness': 0.25,
            'consistency': 0.20,
            'accuracy': 0.20,
            'validity': 0.20,
            'uniqueness': 0.15
        }
    
    def assess_quality(self, 
                      schema_profile: SchemaProfile,
                      df: pd.DataFrame,
                      dimension_weights: Dict[str, float] = None) -> QualityAssessment:
        """Perform comprehensive quality assessment."""
        if dimension_weights is None:
            dimension_weights = self.default_weights.copy()
        
        # Calculate individual quality dimensions
        completeness_score = self._calculate_completeness_score(schema_profile, df)
        consistency_score = self._calculate_consistency_score(schema_profile, df)
        accuracy_score = self._calculate_accuracy_score(schema_profile, df)
        validity_score = self._calculate_validity_score(schema_profile, df)
        uniqueness_score = self._calculate_uniqueness_score(schema_profile, df)
        
        # Calculate overall weighted score
        overall_score = (
            completeness_score * dimension_weights['completeness'] +
            consistency_score * dimension_weights['consistency'] +
            accuracy_score * dimension_weights['accuracy'] +
            validity_score * dimension_weights['validity'] +
            uniqueness_score * dimension_weights['uniqueness']
        )
        
        # Count issues by severity
        issue_counts = self._count_issues_by_severity(schema_profile)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(schema_profile, df)
        
        return QualityAssessment(
            overall_score=round(overall_score, 3),
            completeness_score=round(completeness_score, 3),
            consistency_score=round(consistency_score, 3),
            accuracy_score=round(accuracy_score, 3),
            validity_score=round(validity_score, 3),
            uniqueness_score=round(uniqueness_score, 3),
            dimension_weights=dimension_weights,
            critical_issues=issue_counts['critical'],
            high_issues=issue_counts['high'],
            medium_issues=issue_counts['medium'],
            low_issues=issue_counts['low'],
            recommendations=recommendations
        )
    
    def _calculate_completeness_score(self, schema_profile: SchemaProfile, df: pd.DataFrame) -> float:
        """Calculate completeness score based on missing values."""
        if not schema_profile.columns:
            return 0.0
        
        total_completeness = 0.0
        for column in schema_profile.columns:
            completeness_ratio = column.distribution.completeness_ratio
            total_completeness += completeness_ratio
        
        return total_completeness / len(schema_profile.columns)
    
    def _calculate_consistency_score(self, schema_profile: SchemaProfile, df: pd.DataFrame) -> float:
        """Calculate consistency score based on data type consistency and patterns."""
        if df.empty:
            return 1.0
        
        consistency_issues = 0
        total_checks = 0
        
        for column in schema_profile.columns:
            series = df[column.column_name]
            total_checks += 1
            
            # Check data type consistency
            if pd.api.types.is_numeric_dtype(series):
                # For numeric columns, check for non-numeric values in object dtype
                if series.dtype == 'object':
                    non_numeric = 0
                    for val in series.dropna():
                        try:
                            float(val)
                        except (ValueError, TypeError):
                            non_numeric += 1
                    if non_numeric > 0:
                        consistency_issues += 1
            
            # Check for inconsistent formats in string columns
            elif series.dtype == 'object':
                # Check for inconsistent casing, spacing, etc.
                string_values = series.dropna().astype(str)
                if len(string_values) > 0:
                    # Check for mixed case patterns
                    has_mixed_case = any(
                        val.islower() and any(v.isupper() for v in string_values if v != val)
                        for val in string_values.head(100)
                    )
                    if has_mixed_case:
                        consistency_issues += 0.5  # Partial issue
        
        if total_checks == 0:
            return 1.0
        
        return max(0.0, 1.0 - (consistency_issues / total_checks))
    
    def _calculate_accuracy_score(self, schema_profile: SchemaProfile, df: pd.DataFrame) -> float:
        """Calculate accuracy score based on outliers and format violations."""
        if df.empty:
            return 1.0
        
        total_accuracy = 0.0
        valid_columns = 0
        
        for column in schema_profile.columns:
            series = df[column.column_name]
            column_accuracy = 1.0
            
            # Check for outliers in numeric columns
            if pd.api.types.is_numeric_dtype(series):
                clean_series = series.dropna()
                if len(clean_series) > 10:
                    Q1 = clean_series.quantile(0.25)
                    Q3 = clean_series.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    if IQR > 0:
                        lower_bound = Q1 - 3 * IQR  # More strict outlier detection for accuracy
                        upper_bound = Q3 + 3 * IQR
                        
                        outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
                        outlier_ratio = len(outliers) / len(clean_series)
                        column_accuracy -= outlier_ratio * 0.5  # Penalize based on outlier ratio
            
            # Check for format violations in string columns
            elif series.dtype == 'object':
                # Check against detected patterns
                for pattern in column.patterns:
                    if pattern.percentage < 0.8:  # If pattern coverage is low, accuracy is affected
                        column_accuracy -= (1 - pattern.percentage) * 0.3
            
            total_accuracy += max(0.0, column_accuracy)
            valid_columns += 1
        
        return total_accuracy / valid_columns if valid_columns > 0 else 1.0
    
    def _calculate_validity_score(self, schema_profile: SchemaProfile, df: pd.DataFrame) -> float:
        """Calculate validity score based on data format and constraint violations."""
        if df.empty:
            return 1.0
        
        total_validity = 0.0
        valid_columns = 0
        
        for column in schema_profile.columns:
            series = df[column.column_name]
            column_validity = 1.0
            
            # Check data type validity
            if column.data_type.value == 'integer':
                non_integer_count = 0
                for val in series.dropna():
                    if not isinstance(val, (int, np.integer)):
                        try:
                            float(val)
                            if float(val) != int(float(val)):
                                non_integer_count += 1
                        except (ValueError, TypeError):
                            non_integer_count += 1
                
                if len(series.dropna()) > 0:
                    validity_ratio = 1.0 - (non_integer_count / len(series.dropna()))
                    column_validity *= validity_ratio
            
            # Check pattern validity for string columns
            elif series.dtype == 'object' and column.patterns:
                total_pattern_coverage = sum(pattern.percentage for pattern in column.patterns)
                if total_pattern_coverage < 0.7:  # Low pattern coverage indicates validity issues
                    column_validity *= total_pattern_coverage
            
            # Check for negative values where they shouldn't exist (e.g., age, count fields)
            if pd.api.types.is_numeric_dtype(series):
                if 'count' in column.column_name.lower() or 'age' in column.column_name.lower():
                    negative_count = (series < 0).sum()
                    if negative_count > 0:
                        validity_ratio = 1.0 - (negative_count / len(series))
                        column_validity *= validity_ratio
            
            total_validity += max(0.0, column_validity)
            valid_columns += 1
        
        return total_validity / valid_columns if valid_columns > 0 else 1.0
    
    def _calculate_uniqueness_score(self, schema_profile: SchemaProfile, df: pd.DataFrame) -> float:
        """Calculate uniqueness score based on duplicate detection."""
        if df.empty:
            return 1.0
        
        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        row_uniqueness = 1.0 - (duplicate_rows / len(df)) if len(df) > 0 else 1.0
        
        # Check for columns that should be unique but have duplicates
        column_uniqueness_scores = []
        
        for column in schema_profile.columns:
            series = df[column.column_name]
            
            # If column looks like an ID field, it should be unique
            if ('id' in column.column_name.lower() or 
                column.column_name.lower().endswith('_id') or
                'key' in column.column_name.lower()):
                
                duplicate_values = series.duplicated().sum()
                column_uniqueness = 1.0 - (duplicate_values / len(series)) if len(series) > 0 else 1.0
                column_uniqueness_scores.append(column_uniqueness)
        
        # Combine row and column uniqueness
        if column_uniqueness_scores:
            avg_column_uniqueness = sum(column_uniqueness_scores) / len(column_uniqueness_scores)
            return (row_uniqueness + avg_column_uniqueness) / 2
        else:
            return row_uniqueness
    
    def _count_issues_by_severity(self, schema_profile: SchemaProfile) -> Dict[str, int]:
        """Count quality issues by severity level."""
        counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for column in schema_profile.columns:
            for issue in column.quality_issues:
                if issue.severity in counts:
                    counts[issue.severity] += 1
        
        return counts
    
    def _generate_recommendations(self, schema_profile: SchemaProfile, df: pd.DataFrame) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        # Analyze overall issues
        total_null_percentage = sum(
            (col.distribution.null_count / col.distribution.total_count) * 100 
            for col in schema_profile.columns 
            if col.distribution.total_count > 0
        ) / len(schema_profile.columns) if schema_profile.columns else 0
        
        if total_null_percentage > 10:
            recommendations.append(
                f"High missing data rate ({total_null_percentage:.1f}%) - Consider data collection improvement or imputation strategies"
            )
        
        # Check for columns with many quality issues
        problematic_columns = []
        for column in schema_profile.columns:
            high_severity_issues = [issue for issue in column.quality_issues if issue.severity in ['critical', 'high']]
            if len(high_severity_issues) >= 2:
                problematic_columns.append(column.column_name)
        
        if problematic_columns:
            recommendations.append(
                f"Columns with multiple quality issues require attention: {', '.join(problematic_columns[:5])}"
            )
        
        # Check for potential PII that needs protection
        potential_pii_columns = []
        for column in schema_profile.columns:
            col_name = column.column_name.lower()
            if any(pii_term in col_name for pii_term in ['email', 'phone', 'ssn', 'name', 'address']):
                potential_pii_columns.append(column.column_name)
        
        if potential_pii_columns:
            recommendations.append(
                f"Potential PII detected in columns: {', '.join(potential_pii_columns)} - Consider data anonymization or encryption"
            )
        
        # Check for normalization opportunities
        string_columns_with_inconsistencies = []
        for column in schema_profile.columns:
            if df[column.column_name].dtype == 'object':
                series = df[column.column_name].dropna()
                if len(series) > 0:
                    unique_values = series.unique()
                    # Check for case inconsistencies
                    lower_values = [str(val).lower() for val in unique_values]
                    if len(set(lower_values)) < len(unique_values):
                        string_columns_with_inconsistencies.append(column.column_name)
        
        if string_columns_with_inconsistencies:
            recommendations.append(
                f"String normalization needed for: {', '.join(string_columns_with_inconsistencies[:3])}"
            )
        
        # Check for indexing opportunities
        potential_index_columns = []
        for column in schema_profile.columns:
            if (column.distribution.unique_count / column.distribution.total_count > 0.95 and
                column.distribution.total_count > 1000):
                potential_index_columns.append(column.column_name)
        
        if potential_index_columns:
            recommendations.append(
                f"Consider indexing high-cardinality columns: {', '.join(potential_index_columns[:3])}"
            )
        
        return recommendations
    
    def assess_column_quality_detailed(self, 
                                     column_profile: ColumnProfile, 
                                     series: pd.Series) -> Tuple[float, List[QualityIssue]]:
        """Perform detailed quality assessment for a single column."""
        issues = []
        quality_score = 1.0
        
        # Completeness assessment
        null_percentage = (column_profile.distribution.null_count / column_profile.distribution.total_count) * 100
        if null_percentage > 0:
            severity = self._determine_severity(null_percentage, [5, 15, 30])
            issues.append(QualityIssue(
                issue_type=QualityIssueType.MISSING_VALUES,
                severity=severity,
                description=f"Column has {null_percentage:.1f}% missing values",
                affected_rows=column_profile.distribution.null_count,
                affected_percentage=null_percentage,
                examples=[],
                suggested_action=self._suggest_missing_value_action(null_percentage)
            ))
            quality_score -= (null_percentage / 100) * 0.4
        
        # Consistency assessment for string columns
        if series.dtype == 'object':
            consistency_issues = self._assess_string_consistency(series)
            for issue in consistency_issues:
                issues.append(issue)
                quality_score -= 0.1
        
        # Validity assessment
        validity_issues = self._assess_column_validity(column_profile, series)
        for issue in validity_issues:
            issues.append(issue)
            quality_score -= 0.15
        
        return max(0.0, quality_score), issues
    
    def _determine_severity(self, percentage: float, thresholds: List[float]) -> str:
        """Determine severity based on percentage and thresholds."""
        if percentage <= thresholds[0]:
            return "low"
        elif percentage <= thresholds[1]:
            return "medium"
        elif percentage <= thresholds[2]:
            return "high"
        else:
            return "critical"
    
    def _suggest_missing_value_action(self, null_percentage: float) -> str:
        """Suggest action for handling missing values."""
        if null_percentage < 5:
            return "Consider removing rows with missing values"
        elif null_percentage < 20:
            return "Consider imputation using mean/median/mode"
        else:
            return "Investigate data collection process; consider dropping column if not critical"
    
    def _assess_string_consistency(self, series: pd.Series) -> List[QualityIssue]:
        """Assess consistency issues in string columns."""
        issues = []
        string_values = series.dropna().astype(str)
        
        if len(string_values) == 0:
            return issues
        
        # Check for mixed case
        has_upper = any(val.isupper() for val in string_values.head(100))
        has_lower = any(val.islower() for val in string_values.head(100))
        has_mixed = any(val != val.upper() and val != val.lower() for val in string_values.head(100))
        
        if sum([has_upper, has_lower, has_mixed]) > 1:
            inconsistent_count = sum(
                1 for val in string_values.head(100) 
                if not (val.isupper() or val.islower())
            )
            if inconsistent_count > 0:
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.INCONSISTENT_VALUES,
                    severity="medium",
                    description="Mixed case patterns detected",
                    affected_rows=inconsistent_count,
                    affected_percentage=(inconsistent_count / len(string_values)) * 100,
                    examples=string_values[
                        ~(string_values.str.isupper() | string_values.str.islower())
                    ].head(3).tolist(),
                    suggested_action="Standardize case formatting"
                ))
        
        return issues
    
    def _assess_column_validity(self, column_profile: ColumnProfile, series: pd.Series) -> List[QualityIssue]:
        """Assess validity issues in a column."""
        issues = []
        
        # Check for invalid formats based on column name patterns
        col_name = column_profile.column_name.lower()
        
        if 'email' in col_name:
            invalid_emails = self._check_email_validity(series)
            if invalid_emails['count'] > 0:
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.INVALID_FORMAT,
                    severity="high",
                    description=f"Invalid email formats detected",
                    affected_rows=invalid_emails['count'],
                    affected_percentage=invalid_emails['percentage'],
                    examples=invalid_emails['examples'],
                    suggested_action="Validate and correct email formats"
                ))
        
        elif 'phone' in col_name:
            invalid_phones = self._check_phone_validity(series)
            if invalid_phones['count'] > 0:
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.INVALID_FORMAT,
                    severity="medium",
                    description=f"Invalid phone number formats detected",
                    affected_rows=invalid_phones['count'],
                    affected_percentage=invalid_phones['percentage'],
                    examples=invalid_phones['examples'],
                    suggested_action="Standardize phone number formats"
                ))
        
        return issues
    
    def _check_email_validity(self, series: pd.Series) -> Dict[str, Any]:
        """Check email validity in a series."""
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        string_values = series.dropna().astype(str)
        invalid_emails = []
        
        for val in string_values:
            if not re.match(email_pattern, val):
                invalid_emails.append(val)
        
        return {
            'count': len(invalid_emails),
            'percentage': (len(invalid_emails) / len(string_values)) * 100 if len(string_values) > 0 else 0,
            'examples': invalid_emails[:3]
        }
    
    def _check_phone_validity(self, series: pd.Series) -> Dict[str, Any]:
        """Check phone number validity in a series."""
        import re
        # Simple phone pattern - can be made more sophisticated
        phone_pattern = r'^[\+]?[\d\s\-\(\)]{10,}$'
        
        string_values = series.dropna().astype(str)
        invalid_phones = []
        
        for val in string_values:
            if not re.match(phone_pattern, val):
                invalid_phones.append(val)
        
        return {
            'count': len(invalid_phones),
            'percentage': (len(invalid_phones) / len(string_values)) * 100 if len(string_values) > 0 else 0,
            'examples': invalid_phones[:3]
        }