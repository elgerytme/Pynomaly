import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from ...domain.entities.data_profile import (
    SchemaProfile, ColumnProfile, ValueDistribution, StatisticalSummary,
    DataType, CardinalityLevel, QualityIssue, QualityIssueType
)

class SchemaAnalysisService:
    """Advanced service to perform comprehensive schema analysis on a pandas DataFrame."""
    
    def __init__(self):
        self.type_inference_mapping = {
            'int64': DataType.INTEGER,
            'int32': DataType.INTEGER,
            'float64': DataType.FLOAT,
            'float32': DataType.FLOAT,
            'bool': DataType.BOOLEAN,
            'datetime64[ns]': DataType.DATETIME,
            'object': DataType.STRING
        }
    
    def infer(self, df: pd.DataFrame) -> SchemaProfile:
        """Infer comprehensive schema profile from DataFrame."""
        columns = []
        total_rows = len(df)
        
        for col in df.columns:
            column_profile = self._analyze_column(df[col], col, total_rows)
            columns.append(column_profile)
        
        # Detect relationships and constraints
        primary_keys = self._detect_primary_keys(df, columns)
        foreign_keys = self._detect_foreign_keys(df, columns)
        unique_constraints = self._detect_unique_constraints(df, columns)
        
        # Calculate size metrics
        estimated_size_bytes = int(df.memory_usage(deep=True).sum())
        
        schema_profile = SchemaProfile(
            table_name="dataset",
            total_columns=len(columns),
            total_rows=total_rows,
            columns=columns,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            unique_constraints=unique_constraints,
            check_constraints=[],
            estimated_size_bytes=estimated_size_bytes,
            compression_ratio=None
        )
        return schema_profile
    
    def _analyze_column(self, series: pd.Series, column_name: str, total_rows: int) -> ColumnProfile:
        """Analyze a single column comprehensively."""
        null_count = int(series.isnull().sum())
        unique_count = int(series.nunique(dropna=True))
        completeness_ratio = (total_rows - null_count) / total_rows if total_rows > 0 else 0.0
        
        # Infer data types
        pandas_dtype = str(series.dtype)
        data_type = self._infer_data_type(series, pandas_dtype)
        inferred_type = self._infer_semantic_type(series)
        
        # Calculate cardinality
        cardinality = self._calculate_cardinality(unique_count, total_rows)
        
        # Create value distribution
        distribution = self._create_value_distribution(series, null_count, total_rows)
        
        # Statistical summary for numeric columns
        statistical_summary = None
        if pd.api.types.is_numeric_dtype(series):
            statistical_summary = self._create_statistical_summary(series)
        
        # Quality assessment
        quality_score, quality_issues = self._assess_column_quality(series, column_name)
        
        return ColumnProfile(
            column_name=column_name,
            data_type=data_type,
            inferred_type=inferred_type,
            nullable=null_count > 0,
            distribution=distribution,
            cardinality=cardinality,
            statistical_summary=statistical_summary,
            patterns=[],
            quality_score=quality_score,
            quality_issues=quality_issues,
            semantic_type=None,
            business_meaning=None
        )
    
    def _infer_data_type(self, series: pd.Series, pandas_dtype: str) -> DataType:
        """Infer the appropriate DataType enum value."""
        if pandas_dtype in self.type_inference_mapping:
            return self.type_inference_mapping[pandas_dtype]
        
        # Additional inference for object columns
        if pandas_dtype == 'object':
            sample_values = series.dropna().head(100)
            if sample_values.empty:
                return DataType.STRING
            
            # Try to infer if it's actually numeric
            try:
                pd.to_numeric(sample_values)
                return DataType.FLOAT
            except (ValueError, TypeError):
                pass
            
            # Try to infer if it's datetime
            try:
                pd.to_datetime(sample_values)
                return DataType.DATETIME
            except (ValueError, TypeError):
                pass
        
        return DataType.UNKNOWN
    
    def _infer_semantic_type(self, series: pd.Series) -> Optional[DataType]:
        """Infer semantic type for better understanding."""
        if pd.api.types.is_categorical_dtype(series):
            return DataType.CATEGORICAL
        return None
    
    def _calculate_cardinality(self, unique_count: int, total_count: int) -> CardinalityLevel:
        """Calculate cardinality level based on unique/total ratio."""
        if total_count == 0:
            return CardinalityLevel.LOW
        
        ratio = unique_count / total_count
        if unique_count < 10:
            return CardinalityLevel.LOW
        elif unique_count < 100:
            return CardinalityLevel.MEDIUM
        elif unique_count < 1000:
            return CardinalityLevel.HIGH
        else:
            return CardinalityLevel.VERY_HIGH
    
    def _create_value_distribution(self, series: pd.Series, null_count: int, total_count: int) -> ValueDistribution:
        """Create comprehensive value distribution."""
        unique_count = int(series.nunique(dropna=True))
        completeness_ratio = (total_count - null_count) / total_count if total_count > 0 else 0.0
        
        # Get top values
        value_counts = series.value_counts(dropna=True)
        top_values = {str(k): int(v) for k, v in value_counts.head(10).items()}
        
        return ValueDistribution(
            unique_count=unique_count,
            null_count=null_count,
            total_count=total_count,
            completeness_ratio=completeness_ratio,
            top_values=top_values
        )
    
    def _create_statistical_summary(self, series: pd.Series) -> StatisticalSummary:
        """Create statistical summary for numeric columns."""
        clean_series = series.dropna()
        if clean_series.empty:
            return StatisticalSummary()
        
        quartiles = clean_series.quantile([0.25, 0.5, 0.75]).tolist()
        
        return StatisticalSummary(
            min_value=float(clean_series.min()),
            max_value=float(clean_series.max()),
            mean=float(clean_series.mean()),
            median=float(clean_series.median()),
            std_dev=float(clean_series.std()),
            quartiles=quartiles
        )
    
    def _assess_column_quality(self, series: pd.Series, column_name: str) -> Tuple[float, List[QualityIssue]]:
        """Assess column quality and identify issues."""
        issues = []
        quality_score = 1.0
        total_count = len(series)
        
        # Check for missing values
        null_count = series.isnull().sum()
        if null_count > 0:
            null_percentage = (null_count / total_count) * 100
            severity = "high" if null_percentage > 20 else "medium" if null_percentage > 5 else "low"
            issues.append(QualityIssue(
                issue_type=QualityIssueType.MISSING_VALUES,
                severity=severity,
                description=f"Column has {null_count} missing values ({null_percentage:.1f}%)",
                affected_rows=int(null_count),
                affected_percentage=null_percentage,
                examples=[],
                suggested_action="Consider imputation or removal of missing values"
            ))
            quality_score -= (null_percentage / 100) * 0.5
        
        # Check for duplicate values (if column should be unique)
        non_null_series = series.dropna()
        if not non_null_series.empty:
            duplicate_count = non_null_series.duplicated().sum()
            if duplicate_count > 0:
                duplicate_percentage = (duplicate_count / len(non_null_series)) * 100
                if duplicate_percentage > 10:  # Only flag if significant duplication
                    issues.append(QualityIssue(
                        issue_type=QualityIssueType.DUPLICATE_VALUES,
                        severity="medium",
                        description=f"Column has {duplicate_count} duplicate values ({duplicate_percentage:.1f}%)",
                        affected_rows=int(duplicate_count),
                        affected_percentage=duplicate_percentage,
                        examples=non_null_series[non_null_series.duplicated()].head(3).tolist(),
                        suggested_action="Review if duplicates are expected or need deduplication"
                    ))
                    quality_score -= (duplicate_percentage / 100) * 0.2
        
        # Check for outliers in numeric data
        if pd.api.types.is_numeric_dtype(series):
            clean_series = series.dropna()
            if len(clean_series) > 10:  # Need sufficient data for outlier detection
                Q1 = clean_series.quantile(0.25)
                Q3 = clean_series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
                if len(outliers) > 0:
                    outlier_percentage = (len(outliers) / len(clean_series)) * 100
                    severity = "high" if outlier_percentage > 10 else "medium" if outlier_percentage > 5 else "low"
                    issues.append(QualityIssue(
                        issue_type=QualityIssueType.OUTLIERS,
                        severity=severity,
                        description=f"Column has {len(outliers)} outliers ({outlier_percentage:.1f}%)",
                        affected_rows=len(outliers),
                        affected_percentage=outlier_percentage,
                        examples=outliers.head(3).tolist(),
                        suggested_action="Review outliers for data entry errors or legitimate extreme values"
                    ))
                    quality_score -= (outlier_percentage / 100) * 0.1
        
        return max(0.0, quality_score), issues
    
    def _detect_primary_keys(self, df: pd.DataFrame, columns: List[ColumnProfile]) -> List[str]:
        """Detect potential primary key columns."""
        primary_keys = []
        for column in columns:
            # A primary key candidate should be unique and not null
            if (column.distribution.unique_count == column.distribution.total_count and 
                column.distribution.null_count == 0):
                primary_keys.append(column.column_name)
        return primary_keys
    
    def _detect_foreign_keys(self, df: pd.DataFrame, columns: List[ColumnProfile]) -> Dict[str, str]:
        """Detect potential foreign key relationships."""
        # This is a simplified implementation - in practice, you'd need reference to other tables
        foreign_keys = {}
        
        for column in columns:
            col_name = column.column_name
            # Simple heuristic: columns ending with '_id' or 'Id' might be foreign keys
            if col_name.lower().endswith('_id') or col_name.lower().endswith('id'):
                # Would need actual reference table information to complete this
                pass
        
        return foreign_keys
    
    def _detect_unique_constraints(self, df: pd.DataFrame, columns: List[ColumnProfile]) -> List[List[str]]:
        """Detect unique constraints on single or multiple columns."""
        unique_constraints = []
        
        # Single column unique constraints
        for column in columns:
            if (column.distribution.unique_count == column.distribution.total_count - column.distribution.null_count
                and column.distribution.total_count > 0):
                unique_constraints.append([column.column_name])
        
        # Could extend to detect multi-column unique constraints
        
        return unique_constraints