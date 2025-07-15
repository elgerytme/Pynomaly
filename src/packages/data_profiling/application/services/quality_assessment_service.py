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
                    
                    if non_numeric > len(series) * 0.05:  # More than 5% non-numeric
                        consistency_issues += 1
            
            # Check for inconsistent formats in string columns
            elif series.dtype == 'object':
                # Check for mixed formats (e.g., dates in different formats)
                if self._has_mixed_formats(series):
                    consistency_issues += 1
        
        if total_checks == 0:
            return 1.0
        
        return max(0.0, 1.0 - (consistency_issues / total_checks))
    
    def _calculate_accuracy_score(self, schema_profile: SchemaProfile, df: pd.DataFrame) -> float:
        """Calculate accuracy score based on data validation rules."""
        if df.empty:
            return 1.0
        
        accuracy_issues = 0
        total_checks = 0
        
        for column in schema_profile.columns:
            series = df[column.column_name]
            total_checks += 1
            
            # Check for values outside expected ranges
            if pd.api.types.is_numeric_dtype(series):
                # Check for impossible values (e.g., negative ages, dates in future)
                if self._has_impossible_values(series, column.column_name):
                    accuracy_issues += 1
            
            # Check for invalid formats
            elif series.dtype == 'object':
                # Check for format violations
                if self._has_format_violations(series, column.column_name):
                    accuracy_issues += 1
        
        if total_checks == 0:
            return 1.0
        
        return max(0.0, 1.0 - (accuracy_issues / total_checks))
    
    def _calculate_validity_score(self, schema_profile: SchemaProfile, df: pd.DataFrame) -> float:
        """Calculate validity score based on business rules and constraints."""
        if df.empty:
            return 1.0
        
        validity_issues = 0
        total_checks = 0
        
        for column in schema_profile.columns:
            series = df[column.column_name]
            total_checks += 1
            
            # Check constraint violations
            if not column.nullable and series.isnull().any():
                validity_issues += 1
            
            # Check for referential integrity violations
            if column.column_name in schema_profile.foreign_keys:
                # This would require access to referenced tables
                # For now, just check for null values in FK columns
                if series.isnull().sum() > len(series) * 0.1:  # More than 10% nulls
                    validity_issues += 1
        
        if total_checks == 0:
            return 1.0
        
        return max(0.0, 1.0 - (validity_issues / total_checks))
    
    def _calculate_uniqueness_score(self, schema_profile: SchemaProfile, df: pd.DataFrame) -> float:
        """Calculate uniqueness score based on duplicate detection."""
        if df.empty:
            return 1.0
        
        uniqueness_issues = 0
        total_checks = 0
        
        for column in schema_profile.columns:
            series = df[column.column_name]
            total_checks += 1
            
            # Check for excessive duplicates
            if column.column_name in schema_profile.primary_keys:
                # Primary keys should be unique
                if series.duplicated().any():
                    uniqueness_issues += 1
            else:
                # For other columns, check for unusual duplicate patterns
                duplicate_ratio = series.duplicated().sum() / len(series)
                if duplicate_ratio > 0.8:  # More than 80% duplicates
                    uniqueness_issues += 1
        
        if total_checks == 0:
            return 1.0
        
        return max(0.0, 1.0 - (uniqueness_issues / total_checks))
    
    def _has_mixed_formats(self, series: pd.Series) -> bool:
        """Check if a series has mixed formats."""
        try:
            # Sample some values to check for format consistency
            sample = series.dropna().head(100)
            if len(sample) < 10:
                return False
            
            # Check for date format mixing
            date_formats = []
            for val in sample:
                val_str = str(val)
                if '-' in val_str and '/' in val_str:
                    return True  # Mixed date separators
                
                # Check for different date patterns
                if len(val_str) > 8:
                    if val_str.count('-') == 2:
                        date_formats.append('dash')
                    elif val_str.count('/') == 2:
                        date_formats.append('slash')
            
            return len(set(date_formats)) > 1
            
        except Exception:
            return False
    
    def _has_impossible_values(self, series: pd.Series, column_name: str) -> bool:
        """Check for impossible values based on column semantics."""
        try:
            col_lower = column_name.lower()
            
            # Age-related checks
            if 'age' in col_lower:
                return (series < 0).any() or (series > 150).any()
            
            # Percentage checks
            if 'percent' in col_lower or 'rate' in col_lower:
                return (series < 0).any() or (series > 100).any()
            
            # Price/amount checks
            if 'price' in col_lower or 'amount' in col_lower or 'cost' in col_lower:
                return (series < 0).any()
            
            # Year checks
            if 'year' in col_lower:
                current_year = pd.Timestamp.now().year
                return (series < 1900).any() or (series > current_year + 10).any()
            
            return False
            
        except Exception:
            return False
    
    def _has_format_violations(self, series: pd.Series, column_name: str) -> bool:
        """Check for format violations in text columns."""
        try:
            col_lower = column_name.lower()
            sample = series.dropna().head(100)
            
            if len(sample) == 0:
                return False
            
            # Email format checks
            if 'email' in col_lower:
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                return not sample.str.match(email_pattern).all()
            
            # Phone format checks
            if 'phone' in col_lower:
                # Simple check for phone-like patterns
                phone_chars = sample.str.replace(r'[^0-9]', '', regex=True)
                return not phone_chars.str.len().between(10, 15).all()
            
            # URL format checks
            if 'url' in col_lower or 'website' in col_lower:
                return not sample.str.contains(r'https?://', case=False).all()
            
            return False
            
        except Exception:
            return False
    
    def _count_issues_by_severity(self, schema_profile: SchemaProfile) -> Dict[str, int]:
        """Count quality issues by severity level."""
        issue_counts = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0
        }
        
        for column in schema_profile.columns:
            for issue in column.quality_issues:
                severity = issue.severity.lower()
                if severity in issue_counts:
                    issue_counts[severity] += 1
        
        return issue_counts
    
    def _generate_recommendations(self, schema_profile: SchemaProfile, df: pd.DataFrame) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        # Completeness recommendations
        high_null_columns = []
        for column in schema_profile.columns:
            if column.distribution.completeness_ratio < 0.8:
                high_null_columns.append(column.column_name)
        
        if high_null_columns:
            recommendations.append(
                f"Consider imputation or removal for columns with high missing values: {', '.join(high_null_columns[:5])}"
            )
        
        # Uniqueness recommendations
        if len(schema_profile.primary_keys) == 0:
            recommendations.append("Consider defining primary keys for better data integrity")
        
        # Data type recommendations
        object_columns = [col.column_name for col in schema_profile.columns if col.data_type.value == 'string']
        if len(object_columns) > 10:
            recommendations.append("Consider converting some string columns to categorical for better performance")
        
        # Size recommendations
        if schema_profile.estimated_size_bytes and schema_profile.estimated_size_bytes > 100 * 1024 * 1024:
            recommendations.append("Consider data compression or archiving for large datasets")
        
        # Quality-specific recommendations
        total_issues = sum(len(col.quality_issues) for col in schema_profile.columns)
        if total_issues > 0:
            recommendations.append(f"Address {total_issues} identified quality issues for better data reliability")
        
        # Constraint recommendations
        if not schema_profile.foreign_keys:
            recommendations.append("Consider defining foreign key relationships for better data integrity")
        
        return recommendations
    
    def assess_column_quality(self, column_profile: ColumnProfile, series: pd.Series) -> Dict[str, Any]:
        """Assess quality for a specific column."""
        quality_metrics = {}
        
        # Completeness
        quality_metrics['completeness'] = column_profile.distribution.completeness_ratio
        
        # Uniqueness
        total_count = column_profile.distribution.total_count
        unique_count = column_profile.distribution.unique_count
        quality_metrics['uniqueness'] = unique_count / total_count if total_count > 0 else 0
        
        # Consistency (based on data type inference confidence)
        quality_metrics['consistency'] = self._calculate_type_consistency(series)
        
        # Validity (based on format compliance)
        quality_metrics['validity'] = self._calculate_format_validity(series, column_profile.column_name)
        
        # Overall column quality
        quality_metrics['overall'] = np.mean([
            quality_metrics['completeness'],
            quality_metrics['uniqueness'],
            quality_metrics['consistency'],
            quality_metrics['validity']
        ])
        
        return quality_metrics
    
    def _calculate_type_consistency(self, series: pd.Series) -> float:
        """Calculate type consistency score for a series."""
        try:
            if pd.api.types.is_numeric_dtype(series):
                # Check for mixed numeric types
                return 1.0  # Numeric types are generally consistent
            
            elif series.dtype == 'object':
                # Check for mixed object types
                sample = series.dropna().head(100)
                if len(sample) == 0:
                    return 1.0
                
                # Check if all values can be converted to same type
                types = []
                for val in sample:
                    val_type = type(val).__name__
                    types.append(val_type)
                
                # If all values are same type, consistency is high
                unique_types = set(types)
                return 1.0 / len(unique_types)
            
            else:
                return 1.0
                
        except Exception:
            return 0.5  # Default to medium consistency if calculation fails
    
    def _calculate_format_validity(self, series: pd.Series, column_name: str) -> float:
        """Calculate format validity score for a series."""
        try:
            col_lower = column_name.lower()
            sample = series.dropna().head(100)
            
            if len(sample) == 0:
                return 1.0
            
            # Email validation
            if 'email' in col_lower:
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                valid_emails = sample.str.match(email_pattern).sum()
                return valid_emails / len(sample)
            
            # Phone validation
            if 'phone' in col_lower:
                # Simple phone validation
                phone_chars = sample.str.replace(r'[^0-9]', '', regex=True)
                valid_phones = phone_chars.str.len().between(10, 15).sum()
                return valid_phones / len(sample)
            
            # Date validation
            if 'date' in col_lower:
                try:
                    pd.to_datetime(sample, errors='coerce')
                    valid_dates = pd.to_datetime(sample, errors='coerce').notna().sum()
                    return valid_dates / len(sample)
                except:
                    return 0.5
            
            # Default to high validity for other columns
            return 1.0
            
        except Exception:
            return 0.5  # Default to medium validity if calculation fails
    
    def generate_quality_report(self, quality_assessment: QualityAssessment, 
                              schema_profile: SchemaProfile) -> Dict[str, Any]:
        """Generate a comprehensive quality report."""
        report = {
            'overall_quality': {
                'score': quality_assessment.overall_score,
                'grade': self._get_quality_grade(quality_assessment.overall_score),
                'interpretation': self._interpret_quality_score(quality_assessment.overall_score)
            },
            'dimension_scores': {
                'completeness': quality_assessment.completeness_score,
                'consistency': quality_assessment.consistency_score,
                'accuracy': quality_assessment.accuracy_score,
                'validity': quality_assessment.validity_score,
                'uniqueness': quality_assessment.uniqueness_score
            },
            'issue_summary': {
                'total_issues': (quality_assessment.critical_issues + 
                               quality_assessment.high_issues + 
                               quality_assessment.medium_issues + 
                               quality_assessment.low_issues),
                'critical_issues': quality_assessment.critical_issues,
                'high_issues': quality_assessment.high_issues,
                'medium_issues': quality_assessment.medium_issues,
                'low_issues': quality_assessment.low_issues
            },
            'recommendations': quality_assessment.recommendations,
            'column_quality': self._get_column_quality_summary(schema_profile)
        }
        
        return report
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade."""
        if score >= 0.95:
            return 'A+'
        elif score >= 0.90:
            return 'A'
        elif score >= 0.85:
            return 'B+'
        elif score >= 0.80:
            return 'B'
        elif score >= 0.75:
            return 'C+'
        elif score >= 0.70:
            return 'C'
        elif score >= 0.65:
            return 'D+'
        elif score >= 0.60:
            return 'D'
        else:
            return 'F'
    
    def _interpret_quality_score(self, score: float) -> str:
        """Provide interpretation of quality score."""
        if score >= 0.95:
            return "Excellent data quality - suitable for critical applications"
        elif score >= 0.85:
            return "Good data quality - minor issues that can be addressed"
        elif score >= 0.75:
            return "Moderate data quality - requires attention before use"
        elif score >= 0.60:
            return "Poor data quality - significant issues need resolution"
        else:
            return "Very poor data quality - extensive cleanup required"
    
    def _get_column_quality_summary(self, schema_profile: SchemaProfile) -> List[Dict[str, Any]]:
        """Get quality summary for each column."""
        column_summaries = []
        
        for column in schema_profile.columns:
            summary = {
                'column_name': column.column_name,
                'data_type': column.data_type.value,
                'quality_score': column.quality_score,
                'completeness': column.distribution.completeness_ratio,
                'unique_values': column.distribution.unique_count,
                'null_count': column.distribution.null_count,
                'issues': [
                    {
                        'type': issue.issue_type.value,
                        'severity': issue.severity,
                        'description': issue.description
                    }
                    for issue in column.quality_issues
                ]
            }
            column_summaries.append(summary)
        
        return column_summaries