"""Data profiling service for analyzing dataset characteristics and quality."""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import structlog

from ..entities.dataset import Dataset

logger = structlog.get_logger()


class ProfileLevel(Enum):
    """Data profiling depth levels."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class DataType(Enum):
    """Enhanced data type categories."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class ColumnProfile:
    """Detailed profile of a single column."""
    name: str
    data_type: DataType
    pandas_dtype: str
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    most_frequent_value: Any
    most_frequent_count: int
    memory_usage: int
    
    # Numeric-specific
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    median_value: Optional[float] = None
    std_deviation: Optional[float] = None
    quartiles: Optional[Dict[str, float]] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    outlier_count: Optional[int] = None
    
    # Categorical-specific
    top_categories: Optional[List[Tuple[Any, int]]] = None
    category_count: Optional[int] = None
    
    # Text-specific
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    avg_length: Optional[float] = None
    contains_special_chars: Optional[bool] = None
    
    # DateTime-specific
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    date_range_days: Optional[int] = None


@dataclass
class DatasetProfile:
    """Comprehensive dataset profile."""
    name: str
    total_rows: int
    total_columns: int
    total_cells: int
    memory_usage_mb: float
    profile_timestamp: datetime
    profile_level: ProfileLevel
    
    # Data quality metrics
    total_missing_values: int
    missing_value_percentage: float
    total_duplicates: int
    duplicate_percentage: float
    
    # Column profiles
    columns: List[ColumnProfile]
    
    # Data type distribution
    data_type_counts: Dict[str, int]
    
    # Correlation analysis
    correlations: Optional[Dict[str, Any]] = None
    
    # Sample data
    sample_rows: Optional[List[Dict[str, Any]]] = None
    
    # Additional statistics
    statistics: Dict[str, Any] = None


class DataProfilingService:
    """Service for comprehensive data profiling and analysis."""
    
    def __init__(self):
        self.profile_cache: Dict[str, DatasetProfile] = {}
    
    async def profile_dataset(
        self,
        dataset: Union[Dataset, pd.DataFrame, str],
        dataset_name: str = "dataset",
        level: ProfileLevel = ProfileLevel.STANDARD,
        sample_size: Optional[int] = None
    ) -> DatasetProfile:
        """Create comprehensive profile of a dataset."""
        
        start_time = datetime.now()
        
        try:
            # Convert input to DataFrame
            df = self._prepare_dataframe(dataset)
            
            # Apply sampling if requested
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
                logger.info("Applied sampling", 
                           original_size=len(df),
                           sample_size=sample_size)
            
            logger.info("Starting dataset profiling",
                       dataset_name=dataset_name,
                       rows=len(df),
                       columns=len(df.columns),
                       level=level.value)
            
            # Basic dataset metrics
            total_rows = len(df)
            total_columns = len(df.columns)
            total_cells = total_rows * total_columns
            memory_usage_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            
            # Quality metrics
            total_missing = df.isnull().sum().sum()
            missing_percentage = (total_missing / total_cells) * 100 if total_cells > 0 else 0
            
            total_duplicates = df.duplicated().sum()
            duplicate_percentage = (total_duplicates / total_rows) * 100 if total_rows > 0 else 0
            
            # Profile each column
            column_profiles = []
            for column in df.columns:
                profile = await self._profile_column(df[column], level)
                column_profiles.append(profile)
            
            # Data type distribution
            data_type_counts = {}
            for profile in column_profiles:
                type_name = profile.data_type.value
                data_type_counts[type_name] = data_type_counts.get(type_name, 0) + 1
            
            # Correlation analysis (for standard+ levels)
            correlations = None
            if level in [ProfileLevel.STANDARD, ProfileLevel.COMPREHENSIVE]:
                correlations = await self._analyze_correlations(df)
            
            # Sample data
            sample_rows = None
            if level == ProfileLevel.COMPREHENSIVE:
                sample_rows = df.head(10).to_dict('records')
            
            # Additional statistics
            statistics = await self._calculate_dataset_statistics(df, level)
            
            profile = DatasetProfile(
                name=dataset_name,
                total_rows=total_rows,
                total_columns=total_columns,
                total_cells=total_cells,
                memory_usage_mb=memory_usage_mb,
                profile_timestamp=datetime.now(),
                profile_level=level,
                total_missing_values=total_missing,
                missing_value_percentage=missing_percentage,
                total_duplicates=total_duplicates,
                duplicate_percentage=duplicate_percentage,
                columns=column_profiles,
                data_type_counts=data_type_counts,
                correlations=correlations,
                sample_rows=sample_rows,
                statistics=statistics
            )
            
            # Cache the profile
            self.profile_cache[dataset_name] = profile
            
            profiling_time = (datetime.now() - start_time).total_seconds()
            logger.info("Dataset profiling completed",
                       dataset_name=dataset_name,
                       profiling_time=profiling_time)
            
            return profile
            
        except Exception as e:
            logger.error("Dataset profiling failed",
                        dataset_name=dataset_name,
                        error=str(e))
            raise
    
    async def _profile_column(self, series: pd.Series, level: ProfileLevel) -> ColumnProfile:
        """Profile a single column."""
        
        name = series.name
        pandas_dtype = str(series.dtype)
        total_count = len(series)
        
        # Basic statistics
        null_count = series.isnull().sum()
        null_percentage = (null_count / total_count) * 100 if total_count > 0 else 0
        
        non_null_series = series.dropna()
        unique_count = non_null_series.nunique()
        unique_percentage = (unique_count / total_count) * 100 if total_count > 0 else 0
        
        # Most frequent value
        if len(non_null_series) > 0:
            value_counts = non_null_series.value_counts()
            most_frequent_value = value_counts.index[0]
            most_frequent_count = value_counts.iloc[0]
        else:
            most_frequent_value = None
            most_frequent_count = 0
        
        memory_usage = series.memory_usage(deep=True)
        
        # Determine data type
        data_type = self._infer_data_type(series)
        
        # Initialize profile
        profile = ColumnProfile(
            name=name,
            data_type=data_type,
            pandas_dtype=pandas_dtype,
            null_count=null_count,
            null_percentage=null_percentage,
            unique_count=unique_count,
            unique_percentage=unique_percentage,
            most_frequent_value=most_frequent_value,
            most_frequent_count=most_frequent_count,
            memory_usage=memory_usage
        )
        
        # Add type-specific analysis
        if data_type == DataType.NUMERIC and len(non_null_series) > 0:
            await self._add_numeric_profile(profile, non_null_series, level)
        
        elif data_type == DataType.CATEGORICAL and len(non_null_series) > 0:
            await self._add_categorical_profile(profile, non_null_series, level)
        
        elif data_type == DataType.TEXT and len(non_null_series) > 0:
            await self._add_text_profile(profile, non_null_series, level)
        
        elif data_type == DataType.DATETIME and len(non_null_series) > 0:
            await self._add_datetime_profile(profile, non_null_series, level)
        
        return profile
    
    def _infer_data_type(self, series: pd.Series) -> DataType:
        """Infer the semantic data type of a column."""
        
        # Handle empty series
        if len(series.dropna()) == 0:
            return DataType.UNKNOWN
        
        pandas_dtype = series.dtype
        
        # Numeric types
        if pd.api.types.is_numeric_dtype(pandas_dtype):
            return DataType.NUMERIC
        
        # Boolean types
        if pd.api.types.is_bool_dtype(pandas_dtype):
            return DataType.BOOLEAN
        
        # DateTime types
        if pd.api.types.is_datetime64_any_dtype(pandas_dtype):
            return DataType.DATETIME
        
        # For object types, need deeper analysis
        if pandas_dtype == 'object':
            non_null = series.dropna()
            
            # Try to detect datetime strings
            if self._is_datetime_column(non_null):
                return DataType.DATETIME
            
            # Check if it's categorical (low cardinality)
            unique_ratio = non_null.nunique() / len(non_null)
            if unique_ratio < 0.5 and non_null.nunique() < 1000:
                return DataType.CATEGORICAL
            
            # Check if it's mostly text
            if self._is_text_column(non_null):
                return DataType.TEXT
            
            # Mixed types
            return DataType.MIXED
        
        return DataType.UNKNOWN
    
    def _is_datetime_column(self, series: pd.Series) -> bool:
        """Check if a series contains datetime strings."""
        try:
            # Try to parse a sample as datetime
            sample = series.head(min(100, len(series)))
            pd.to_datetime(sample, errors='raise')
            return True
        except:
            return False
    
    def _is_text_column(self, series: pd.Series) -> bool:
        """Check if a series contains text data."""
        # Check if values are strings with reasonable length
        sample = series.head(min(100, len(series)))
        string_count = sum(1 for val in sample if isinstance(val, str) and len(val) > 2)
        return string_count > len(sample) * 0.8
    
    async def _add_numeric_profile(self, profile: ColumnProfile, series: pd.Series, level: ProfileLevel):
        """Add numeric-specific profiling information."""
        
        try:
            profile.min_value = float(series.min())
            profile.max_value = float(series.max())
            profile.mean_value = float(series.mean())
            profile.median_value = float(series.median())
            profile.std_deviation = float(series.std())
            
            # Quartiles
            profile.quartiles = {
                "q1": float(series.quantile(0.25)),
                "q2": float(series.quantile(0.5)),
                "q3": float(series.quantile(0.75))
            }
            
            if level in [ProfileLevel.STANDARD, ProfileLevel.COMPREHENSIVE]:
                # Skewness and kurtosis
                try:
                    profile.skewness = float(series.skew())
                    profile.kurtosis = float(series.kurtosis())
                except:
                    pass
                
                # Outlier detection using IQR method
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                profile.outlier_count = ((series < lower_bound) | (series > upper_bound)).sum()
                
        except Exception as e:
            logger.warning("Numeric profiling failed", column=profile.name, error=str(e))
    
    async def _add_categorical_profile(self, profile: ColumnProfile, series: pd.Series, level: ProfileLevel):
        """Add categorical-specific profiling information."""
        
        try:
            value_counts = series.value_counts()
            
            profile.category_count = len(value_counts)
            
            if level in [ProfileLevel.STANDARD, ProfileLevel.COMPREHENSIVE]:
                # Top categories (up to 10)
                top_n = min(10, len(value_counts))
                profile.top_categories = [
                    (category, count) for category, count in value_counts.head(top_n).items()
                ]
                
        except Exception as e:
            logger.warning("Categorical profiling failed", column=profile.name, error=str(e))
    
    async def _add_text_profile(self, profile: ColumnProfile, series: pd.Series, level: ProfileLevel):
        """Add text-specific profiling information."""
        
        try:
            # Convert to string and calculate lengths
            str_series = series.astype(str)
            lengths = str_series.str.len()
            
            profile.min_length = int(lengths.min())
            profile.max_length = int(lengths.max())
            profile.avg_length = float(lengths.mean())
            
            if level in [ProfileLevel.STANDARD, ProfileLevel.COMPREHENSIVE]:
                # Check for special characters
                special_char_pattern = r'[^a-zA-Z0-9\s]'
                has_special = str_series.str.contains(special_char_pattern, regex=True, na=False)
                profile.contains_special_chars = has_special.any()
                
        except Exception as e:
            logger.warning("Text profiling failed", column=profile.name, error=str(e))
    
    async def _add_datetime_profile(self, profile: ColumnProfile, series: pd.Series, level: ProfileLevel):
        """Add datetime-specific profiling information."""
        
        try:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(series.dtype):
                dt_series = pd.to_datetime(series, errors='coerce').dropna()
            else:
                dt_series = series
            
            if len(dt_series) > 0:
                min_date = dt_series.min()
                max_date = dt_series.max()
                
                profile.min_date = min_date.isoformat() if pd.notna(min_date) else None
                profile.max_date = max_date.isoformat() if pd.notna(max_date) else None
                
                if pd.notna(min_date) and pd.notna(max_date):
                    profile.date_range_days = (max_date - min_date).days
                
        except Exception as e:
            logger.warning("DateTime profiling failed", column=profile.name, error=str(e))
    
    async def _analyze_correlations(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Analyze correlations between numeric columns."""
        
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) < 2:
                return None
            
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()
            
            # Find high correlations (>0.7 or <-0.7)
            high_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_correlations.append({
                            "column1": corr_matrix.columns[i],
                            "column2": corr_matrix.columns[j],
                            "correlation": float(corr_val)
                        })
            
            return {
                "correlation_matrix": corr_matrix.to_dict(),
                "high_correlations": high_correlations,
                "average_correlation": float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean())
            }
            
        except Exception as e:
            logger.warning("Correlation analysis failed", error=str(e))
            return None
    
    async def _calculate_dataset_statistics(self, df: pd.DataFrame, level: ProfileLevel) -> Dict[str, Any]:
        """Calculate additional dataset statistics."""
        
        stats = {
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=['object']).columns),
            "datetime_columns": len(df.select_dtypes(include=['datetime64']).columns),
            "boolean_columns": len(df.select_dtypes(include=['bool']).columns)
        }
        
        if level == ProfileLevel.COMPREHENSIVE:
            try:
                # Data sparsity
                total_cells = df.shape[0] * df.shape[1]
                non_null_cells = total_cells - df.isnull().sum().sum()
                stats["data_density"] = float(non_null_cells / total_cells) if total_cells > 0 else 0
                
                # Column completeness distribution
                completeness = []
                for col in df.columns:
                    completeness.append(1 - (df[col].isnull().sum() / len(df)))
                
                stats["avg_column_completeness"] = float(np.mean(completeness))
                stats["min_column_completeness"] = float(np.min(completeness))
                stats["max_column_completeness"] = float(np.max(completeness))
                
            except Exception as e:
                logger.warning("Advanced statistics calculation failed", error=str(e))
        
        return stats
    
    def _prepare_dataframe(self, dataset: Union[Dataset, pd.DataFrame, str]) -> pd.DataFrame:
        """Convert input to DataFrame."""
        if isinstance(dataset, pd.DataFrame):
            return dataset
        elif isinstance(dataset, Dataset):
            return dataset.to_dataframe()
        elif isinstance(dataset, str):
            # Assume it's a file path
            if dataset.endswith('.csv'):
                return pd.read_csv(dataset)
            elif dataset.endswith('.json'):
                return pd.read_json(dataset)
            elif dataset.endswith('.parquet'):
                return pd.read_parquet(dataset)
            else:
                raise ValueError(f"Unsupported file format: {dataset}")
        else:
            raise ValueError(f"Unsupported data type: {type(dataset)}")
    
    async def compare_profiles(self, profile1: DatasetProfile, profile2: DatasetProfile) -> Dict[str, Any]:
        """Compare two dataset profiles."""
        
        comparison = {
            "dataset1": profile1.name,
            "dataset2": profile2.name,
            "comparison_timestamp": datetime.now().isoformat(),
            "shape_comparison": {
                "rows_diff": profile2.total_rows - profile1.total_rows,
                "columns_diff": profile2.total_columns - profile1.total_columns,
                "memory_diff_mb": profile2.memory_usage_mb - profile1.memory_usage_mb
            },
            "quality_comparison": {
                "missing_percentage_diff": profile2.missing_value_percentage - profile1.missing_value_percentage,
                "duplicate_percentage_diff": profile2.duplicate_percentage - profile1.duplicate_percentage
            },
            "data_type_changes": {},
            "column_changes": {
                "added_columns": [],
                "removed_columns": [],
                "type_changes": []
            }
        }
        
        # Compare data type distributions
        for data_type in set(list(profile1.data_type_counts.keys()) + list(profile2.data_type_counts.keys())):
            count1 = profile1.data_type_counts.get(data_type, 0)
            count2 = profile2.data_type_counts.get(data_type, 0)
            comparison["data_type_changes"][data_type] = count2 - count1
        
        # Compare columns
        columns1 = {col.name: col for col in profile1.columns}
        columns2 = {col.name: col for col in profile2.columns}
        
        comparison["column_changes"]["added_columns"] = list(set(columns2.keys()) - set(columns1.keys()))
        comparison["column_changes"]["removed_columns"] = list(set(columns1.keys()) - set(columns2.keys()))
        
        # Check for type changes in common columns
        common_columns = set(columns1.keys()) & set(columns2.keys())
        for col_name in common_columns:
            if columns1[col_name].data_type != columns2[col_name].data_type:
                comparison["column_changes"]["type_changes"].append({
                    "column": col_name,
                    "old_type": columns1[col_name].data_type.value,
                    "new_type": columns2[col_name].data_type.value
                })
        
        return comparison
    
    def get_profile_summary(self, profile: DatasetProfile) -> Dict[str, Any]:
        """Get a summary of the dataset profile."""
        
        return {
            "dataset_name": profile.name,
            "basic_info": {
                "rows": profile.total_rows,
                "columns": profile.total_columns,
                "memory_mb": round(profile.memory_usage_mb, 2),
                "profile_level": profile.profile_level.value
            },
            "data_quality": {
                "missing_percentage": round(profile.missing_value_percentage, 2),
                "duplicate_percentage": round(profile.duplicate_percentage, 2),
                "completeness_score": round(100 - profile.missing_value_percentage, 1)
            },
            "data_types": profile.data_type_counts,
            "top_issues": self._identify_data_issues(profile)
        }
    
    def _identify_data_issues(self, profile: DatasetProfile) -> List[str]:
        """Identify potential data quality issues."""
        
        issues = []
        
        # High missing data
        if profile.missing_value_percentage > 20:
            issues.append(f"High missing data: {profile.missing_value_percentage:.1f}%")
        
        # High duplicate rate
        if profile.duplicate_percentage > 10:
            issues.append(f"High duplicate rate: {profile.duplicate_percentage:.1f}%")
        
        # Columns with all missing values
        null_columns = [col.name for col in profile.columns if col.null_percentage == 100]
        if null_columns:
            issues.append(f"Columns with no data: {', '.join(null_columns[:3])}{'...' if len(null_columns) > 3 else ''}")
        
        # High cardinality categorical columns
        high_cardinality = [
            col.name for col in profile.columns 
            if col.data_type == DataType.CATEGORICAL and col.unique_count > 1000
        ]
        if high_cardinality:
            issues.append(f"High cardinality categories: {', '.join(high_cardinality[:3])}")
        
        # Columns with outliers
        outlier_columns = [
            col.name for col in profile.columns 
            if col.outlier_count and col.outlier_count > profile.total_rows * 0.05
        ]
        if outlier_columns:
            issues.append(f"Columns with many outliers: {', '.join(outlier_columns[:3])}")
        
        return issues[:5]  # Return top 5 issues
    
    def export_profile_report(self, profile: DatasetProfile, output_path: str, format: str = "json"):
        """Export profile report to file."""
        
        try:
            if format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(asdict(profile), f, indent=2, default=str)
            
            elif format.lower() == "html":
                self._export_html_report(profile, output_path)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
            logger.info("Profile report exported", 
                       output_path=output_path,
                       format=format)
                       
        except Exception as e:
            logger.error("Failed to export profile report",
                        output_path=output_path,
                        error=str(e))
            raise
    
    def _export_html_report(self, profile: DatasetProfile, output_path: str):
        """Export profile as HTML report."""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Profile Report - {profile.name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9ecef; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .issue {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Profile Report</h1>
                <h2>{profile.name}</h2>
                <p>Generated on: {profile.profile_timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h3>Dataset Overview</h3>
                <div class="metric">Rows: {profile.total_rows:,}</div>
                <div class="metric">Columns: {profile.total_columns}</div>
                <div class="metric">Memory: {profile.memory_usage_mb:.2f} MB</div>
                <div class="metric">Missing: {profile.missing_value_percentage:.1f}%</div>
                <div class="metric">Duplicates: {profile.duplicate_percentage:.1f}%</div>
            </div>
            
            <div class="section">
                <h3>Data Quality Issues</h3>
                {"".join(f'<div class="issue">{issue}</div>' for issue in self._identify_data_issues(profile))}
            </div>
            
            <div class="section">
                <h3>Column Details</h3>
                <table>
                    <tr>
                        <th>Column</th>
                        <th>Type</th>
                        <th>Missing %</th>
                        <th>Unique</th>
                        <th>Most Frequent</th>
                    </tr>
                    {"".join(f'''
                    <tr>
                        <td>{col.name}</td>
                        <td>{col.data_type.value}</td>
                        <td>{col.null_percentage:.1f}%</td>
                        <td>{col.unique_count}</td>
                        <td>{col.most_frequent_value}</td>
                    </tr>
                    ''' for col in profile.columns)}
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)