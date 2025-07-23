"""Data profiling service for comprehensive data analysis."""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import structlog

logger = structlog.get_logger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class DataProfilingService:
    """Service for comprehensive data profiling and statistical analysis."""
    
    def __init__(self):
        self.profile_sections = {
            'dataset_info': self._profile_dataset_info,
            'column_info': self._profile_columns,
            'data_quality': self._profile_data_quality,
            'statistical_summary': self._profile_statistics,
            'correlations': self._profile_correlations,
            'distributions': self._profile_distributions,
            'patterns': self._profile_patterns
        }
    
    async def profile_file(
        self,
        file_path: Path,
        include_correlations: bool = True,
        include_distributions: bool = True,
        generate_plots: bool = False,
        sample_size: Optional[int] = None,
        sections: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive profile for a data file."""
        
        try:
            logger.info("Starting data profiling", file=str(file_path))
            
            # Load data
            data = await self._load_data(file_path)
            
            # Apply sampling if specified
            if sample_size and len(data) > sample_size:
                data = data.sample(n=sample_size, random_state=42)
                logger.info("Applied sampling", original_size=len(data), sample_size=sample_size)
            
            # Initialize profile
            profile = {
                'profiling_metadata': {
                    'file_path': str(file_path),
                    'profiling_timestamp': datetime.utcnow().isoformat(),
                    'sampled': sample_size is not None and len(data) < sample_size,
                    'sample_size': len(data)
                }
            }
            
            # Determine which sections to include
            sections_to_run = sections or list(self.profile_sections.keys())
            
            # Generate each profile section
            for section_name in sections_to_run:
                if section_name in self.profile_sections:
                    try:
                        logger.debug("Generating profile section", section=section_name)
                        
                        # Special handling for optional sections
                        if section_name == 'correlations' and not include_correlations:
                            continue
                        if section_name == 'distributions' and not include_distributions:
                            continue
                        
                        section_data = await self.profile_sections[section_name](data)
                        profile[section_name] = section_data
                        
                    except Exception as e:
                        logger.warning("Profile section failed", 
                                     section=section_name, 
                                     error=str(e))
                        profile[section_name] = {
                            'error': f"Section generation failed: {str(e)}"
                        }
            
            # Generate plots if requested
            if generate_plots:
                try:
                    plots_info = await self._generate_plots(data, file_path.parent / f"{file_path.stem}_plots")
                    profile['plots'] = plots_info
                except Exception as e:
                    logger.warning("Plot generation failed", error=str(e))
                    profile['plots'] = {'error': f"Plot generation failed: {str(e)}"}
            
            logger.info("Data profiling completed", 
                       file=str(file_path),
                       sections=len([s for s in profile.keys() if not s.startswith('profiling_')]))
            
            return profile
            
        except Exception as e:
            logger.error("Data profiling failed", file=str(file_path), error=str(e))
            raise
    
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
    
    async def _profile_dataset_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic dataset information."""
        memory_usage = data.memory_usage(deep=True)
        
        return {
            'row_count': len(data),
            'column_count': len(data.columns),
            'total_cells': len(data) * len(data.columns),
            'memory_usage_bytes': memory_usage.sum(),
            'memory_usage_mb': memory_usage.sum() / (1024 * 1024),
            'average_row_size_bytes': memory_usage.sum() / len(data) if len(data) > 0 else 0,
            'column_names': list(data.columns),
            'duplicate_rows': data.duplicated().sum(),
            'duplicate_row_percentage': (data.duplicated().sum() / len(data)) * 100 if len(data) > 0 else 0
        }
    
    async def _profile_columns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate detailed column information."""
        column_profiles = {}
        
        for column in data.columns:
            try:
                series = data[column]
                column_profile = {
                    'dtype': str(series.dtype),
                    'non_null_count': series.count(),
                    'null_count': series.isnull().sum(),
                    'null_percentage': (series.isnull().sum() / len(series)) * 100 if len(series) > 0 else 0,
                    'unique_count': series.nunique(),
                    'unique_percentage': (series.nunique() / len(series)) * 100 if len(series) > 0 else 0,
                    'is_unique': series.nunique() == len(series),
                    'memory_usage_bytes': series.memory_usage(deep=True)
                }
                
                # Type-specific profiling
                if pd.api.types.is_numeric_dtype(series):
                    column_profile.update(await self._profile_numeric_column(series))
                elif pd.api.types.is_string_dtype(series) or series.dtype == 'object':
                    column_profile.update(await self._profile_text_column(series))
                elif pd.api.types.is_datetime64_any_dtype(series):
                    column_profile.update(await self._profile_datetime_column(series))
                elif pd.api.types.is_bool_dtype(series):
                    column_profile.update(await self._profile_boolean_column(series))
                
                column_profiles[column] = column_profile
                
            except Exception as e:
                logger.warning("Column profiling failed", column=column, error=str(e))
                column_profiles[column] = {
                    'error': f"Column profiling failed: {str(e)}",
                    'dtype': str(data[column].dtype)
                }
        
        return column_profiles
    
    async def _profile_numeric_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile numeric column."""
        series_clean = series.dropna()
        
        if len(series_clean) == 0:
            return {'error': 'No non-null values for statistics'}
        
        try:
            # Basic statistics
            stats = {
                'min': float(series_clean.min()),
                'max': float(series_clean.max()),
                'mean': float(series_clean.mean()),
                'median': float(series_clean.median()),
                'mode': float(series_clean.mode().iloc[0]) if not series_clean.mode().empty else None,
                'std': float(series_clean.std()),
                'var': float(series_clean.var()),
                'skewness': float(series_clean.skew()),
                'kurtosis': float(series_clean.kurtosis())
            }
            
            # Percentiles
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            for p in percentiles:
                stats[f'p{p}'] = float(series_clean.quantile(p / 100))
            
            # Outlier detection using IQR
            Q1 = series_clean.quantile(0.25)
            Q3 = series_clean.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = series_clean[(series_clean < lower_bound) | (series_clean > upper_bound)]
                
                stats.update({
                    'outlier_count': len(outliers),
                    'outlier_percentage': (len(outliers) / len(series_clean)) * 100,
                    'iqr': float(IQR),
                    'lower_fence': float(lower_bound),
                    'upper_fence': float(upper_bound)
                })
            
            # Value distribution
            stats.update({
                'zeros_count': int((series_clean == 0).sum()),
                'zeros_percentage': float(((series_clean == 0).sum() / len(series_clean)) * 100),
                'negative_count': int((series_clean < 0).sum()),
                'negative_percentage': float(((series_clean < 0).sum() / len(series_clean)) * 100),
                'positive_count': int((series_clean > 0).sum()),
                'positive_percentage': float(((series_clean > 0).sum() / len(series_clean)) * 100)
            })
            
            return stats
            
        except Exception as e:
            return {'error': f'Numeric profiling failed: {str(e)}'}
    
    async def _profile_text_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile text/object column."""
        series_clean = series.dropna().astype(str)
        
        if len(series_clean) == 0:
            return {'error': 'No non-null values for statistics'}
        
        try:
            # Length statistics
            lengths = series_clean.str.len()
            
            stats = {
                'min_length': int(lengths.min()),
                'max_length': int(lengths.max()),
                'mean_length': float(lengths.mean()),
                'median_length': float(lengths.median()),
                'std_length': float(lengths.std())
            }
            
            # Character statistics
            stats.update({
                'contains_numeric': int(series_clean.str.contains(r'\\d', regex=True, na=False).sum()),
                'contains_alpha': int(series_clean.str.contains(r'[a-zA-Z]', regex=True, na=False).sum()),
                'contains_special': int(series_clean.str.contains(r'[^a-zA-Z0-9\\s]', regex=True, na=False).sum()),
                'all_uppercase': int(series_clean.str.isupper().sum()),
                'all_lowercase': int(series_clean.str.islower().sum()),
                'empty_strings': int((series_clean.str.len() == 0).sum())
            })
            
            # Pattern detection
            patterns = {
                'email_like': r'^[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}$',
                'phone_like': r'^[\\+]?[1-9]?[0-9]{7,15}$',
                'url_like': r'^https?://[\\w\\.-]+',
                'date_like': r'^\\d{4}-\\d{2}-\\d{2}$|^\\d{2}/\\d{2}/\\d{4}$',
                'numeric_like': r'^-?\\d+\\.?\\d*$'
            }
            
            for pattern_name, pattern in patterns.items():
                try:
                    matches = series_clean.str.contains(pattern, regex=True, na=False).sum()
                    stats[f'{pattern_name}_count'] = int(matches)
                    stats[f'{pattern_name}_percentage'] = float((matches / len(series_clean)) * 100)
                except Exception:
                    stats[f'{pattern_name}_count'] = 0
                    stats[f'{pattern_name}_percentage'] = 0.0
            
            # Most common values
            value_counts = series_clean.value_counts().head(10)
            stats['most_common_values'] = [
                {'value': str(val), 'count': int(count), 'percentage': float((count / len(series_clean)) * 100)}
                for val, count in value_counts.items()
            ]
            
            return stats
            
        except Exception as e:
            return {'error': f'Text profiling failed: {str(e)}'}
    
    async def _profile_datetime_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile datetime column."""
        series_clean = series.dropna()
        
        if len(series_clean) == 0:
            return {'error': 'No non-null values for statistics'}
        
        try:
            stats = {
                'min_date': series_clean.min().isoformat() if hasattr(series_clean.min(), 'isoformat') else str(series_clean.min()),
                'max_date': series_clean.max().isoformat() if hasattr(series_clean.max(), 'isoformat') else str(series_clean.max()),
                'date_range_days': (series_clean.max() - series_clean.min()).days if hasattr(series_clean.max() - series_clean.min(), 'days') else None
            }
            
            # Extract time components
            if hasattr(series_clean.dt, 'year'):
                stats.update({
                    'year_range': [int(series_clean.dt.year.min()), int(series_clean.dt.year.max())],
                    'month_distribution': series_clean.dt.month.value_counts().to_dict(),
                    'weekday_distribution': series_clean.dt.dayofweek.value_counts().to_dict(),
                    'hour_distribution': series_clean.dt.hour.value_counts().to_dict() if hasattr(series_clean.dt, 'hour') else {}
                })
            
            return stats
            
        except Exception as e:
            return {'error': f'Datetime profiling failed: {str(e)}'}
    
    async def _profile_boolean_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile boolean column."""
        series_clean = series.dropna()
        
        if len(series_clean) == 0:
            return {'error': 'No non-null values for statistics'}
        
        try:
            true_count = int(series_clean.sum())
            false_count = len(series_clean) - true_count
            
            return {
                'true_count': true_count,
                'false_count': false_count,
                'true_percentage': float((true_count / len(series_clean)) * 100),
                'false_percentage': float((false_count / len(series_clean)) * 100)
            }
            
        except Exception as e:
            return {'error': f'Boolean profiling failed: {str(e)}'}
    
    async def _profile_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Profile data quality metrics."""
        try:
            # Missing values analysis
            missing_data = data.isnull().sum()
            total_cells = len(data) * len(data.columns)
            total_missing = missing_data.sum()
            
            quality_metrics = {
                'missing_values_total': int(total_missing),
                'missing_values_percentage': float((total_missing / total_cells) * 100) if total_cells > 0 else 0,
                'columns_with_missing': int((missing_data > 0).sum()),
                'complete_rows': int(data.dropna().shape[0]),
                'complete_rows_percentage': float((data.dropna().shape[0] / len(data)) * 100) if len(data) > 0 else 0
            }
            
            # Missing data by column
            missing_by_column = {}
            for col in data.columns:
                missing_count = int(missing_data[col])
                missing_by_column[col] = {
                    'count': missing_count,
                    'percentage': float((missing_count / len(data)) * 100) if len(data) > 0 else 0
                }
            
            quality_metrics['missing_by_column'] = missing_by_column
            
            # Duplicate analysis
            duplicates = data.duplicated()
            quality_metrics.update({
                'duplicate_rows': int(duplicates.sum()),
                'duplicate_percentage': float((duplicates.sum() / len(data)) * 100) if len(data) > 0 else 0,
                'unique_rows': int(len(data) - duplicates.sum())
            })
            
            # Data type consistency
            type_issues = []
            for col in data.columns:
                if data[col].dtype == 'object':
                    # Check for mixed types in object columns
                    non_null_values = data[col].dropna()
                    if len(non_null_values) > 0:
                        sample_types = [type(val).__name__ for val in non_null_values.head(100)]
                        unique_types = set(sample_types)
                        if len(unique_types) > 1:
                            type_issues.append({
                                'column': col,
                                'issue': 'mixed_types',
                                'types_found': list(unique_types)
                            })
            
            quality_metrics['type_issues'] = type_issues
            
            return quality_metrics
            
        except Exception as e:
            return {'error': f'Data quality profiling failed: {str(e)}'}
    
    async def _profile_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical summary."""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return {'error': 'No numeric columns for statistical analysis'}
            
            # Basic descriptive statistics
            desc_stats = numeric_data.describe()
            
            # Convert to dictionary with proper data types
            stats_dict = {}
            for col in desc_stats.columns:
                stats_dict[col] = {
                    'count': int(desc_stats.loc['count', col]),
                    'mean': float(desc_stats.loc['mean', col]),
                    'std': float(desc_stats.loc['std', col]),
                    'min': float(desc_stats.loc['min', col]),
                    'p25': float(desc_stats.loc['25%', col]),
                    'p50': float(desc_stats.loc['50%', col]),
                    'p75': float(desc_stats.loc['75%', col]),
                    'max': float(desc_stats.loc['max', col])
                }
            
            return {
                'descriptive_statistics': stats_dict,
                'numeric_columns_count': len(numeric_data.columns),
                'total_numeric_values': int(numeric_data.count().sum())
            }
            
        except Exception as e:
            return {'error': f'Statistical profiling failed: {str(e)}'}
    
    async def _profile_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate correlation analysis."""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if len(numeric_data.columns) < 2:
                return {'error': 'Need at least 2 numeric columns for correlation analysis'}
            
            # Calculate correlation matrix
            corr_matrix = numeric_data.corr()
            
            # Convert to serializable format
            correlation_dict = {}
            for i, col1 in enumerate(corr_matrix.columns):
                correlation_dict[col1] = {}
                for j, col2 in enumerate(corr_matrix.columns):
                    if i <= j:  # Only store upper triangle to avoid duplication
                        correlation_dict[col1][col2] = float(corr_matrix.iloc[i, j]) if not pd.isna(corr_matrix.iloc[i, j]) else None
            
            # Find strong correlations
            strong_correlations = []
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j:  # Avoid diagonal and duplicates
                        corr_value = corr_matrix.iloc[i, j]
                        if not pd.isna(corr_value) and abs(corr_value) > 0.7:
                            strong_correlations.append({
                                'column1': col1,
                                'column2': col2,
                                'correlation': float(corr_value),
                                'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                            })
            
            return {
                'correlation_matrix': correlation_dict,
                'strong_correlations': strong_correlations,
                'correlation_pairs_analyzed': len(corr_matrix.columns) * (len(corr_matrix.columns) - 1) // 2
            }
            
        except Exception as e:
            return {'error': f'Correlation analysis failed: {str(e)}'}
    
    async def _profile_distributions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Profile data distributions."""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return {'error': 'No numeric columns for distribution analysis'}
            
            distributions = {}
            
            for col in numeric_data.columns:
                series = numeric_data[col].dropna()
                if len(series) < 10:  # Need minimum data for distribution analysis
                    continue
                
                try:
                    # Basic distribution metrics
                    distribution_info = {
                        'skewness': float(series.skew()),
                        'kurtosis': float(series.kurtosis()),
                        'normality_test_p_value': None,  # Would need scipy for actual test
                        'distribution_type': self._guess_distribution_type(series)
                    }
                    
                    # Create histogram data
                    hist_data, bin_edges = np.histogram(series, bins=20)
                    distribution_info['histogram'] = {
                        'counts': hist_data.tolist(),
                        'bin_edges': bin_edges.tolist()
                    }
                    
                    distributions[col] = distribution_info
                    
                except Exception as e:
                    distributions[col] = {'error': f'Distribution analysis failed: {str(e)}'}
            
            return {
                'distributions': distributions,
                'columns_analyzed': len(distributions)
            }
            
        except Exception as e:
            return {'error': f'Distribution profiling failed: {str(e)}'}
    
    def _guess_distribution_type(self, series: pd.Series) -> str:
        """Guess the distribution type based on basic statistics."""
        skewness = abs(series.skew())
        kurtosis = series.kurtosis()
        
        if skewness < 0.5 and -1 < kurtosis < 1:
            return 'normal'
        elif skewness > 1:
            return 'right_skewed'
        elif skewness < -1:
            return 'left_skewed'
        elif kurtosis > 3:
            return 'heavy_tailed'
        elif kurtosis < -1:
            return 'light_tailed'
        else:
            return 'unknown'
    
    async def _profile_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify data patterns and anomalies."""
        try:
            patterns = {}
            
            # Check for constant columns
            constant_columns = []
            for col in data.columns:
                if data[col].nunique() <= 1:
                    constant_columns.append(col)
            
            patterns['constant_columns'] = constant_columns
            
            # Check for potential ID columns
            id_columns = []
            for col in data.columns:
                unique_ratio = data[col].nunique() / len(data) if len(data) > 0 else 0
                if unique_ratio > 0.95:  # More than 95% unique values
                    id_columns.append({
                        'column': col,
                        'unique_ratio': float(unique_ratio),
                        'likely_id': True
                    })
            
            patterns['potential_id_columns'] = id_columns
            
            # Check for seasonal patterns in numeric data
            numeric_data = data.select_dtypes(include=[np.number])
            seasonal_patterns = []
            
            for col in numeric_data.columns:
                series = numeric_data[col].dropna()
                if len(series) > 50:  # Need sufficient data
                    # Simple seasonality check using autocorrelation at lag 12 (monthly) and 4 (quarterly)
                    try:
                        # This is a simplified check - would need statsmodels for proper analysis
                        seasonal_patterns.append({
                            'column': col,
                            'potential_seasonality': 'unknown',  # Placeholder
                            'trend': 'increasing' if series.iloc[-10:].mean() > series.iloc[:10].mean() else 'decreasing'
                        })
                    except Exception:
                        pass
            
            patterns['seasonal_patterns'] = seasonal_patterns
            
            return patterns
            
        except Exception as e:
            return {'error': f'Pattern analysis failed: {str(e)}'}
    
    async def _generate_plots(self, data: pd.DataFrame, plots_dir: Path) -> Dict[str, Any]:
        """Generate visualization plots (placeholder for actual plotting)."""
        # This would require matplotlib/seaborn for actual implementation
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        plots_info = {
            'plots_directory': str(plots_dir),
            'plots_generated': [],
            'note': 'Plot generation requires matplotlib/seaborn dependencies'
        }
        
        # Would generate:
        # - Histograms for numeric columns
        # - Box plots for outlier detection
        # - Correlation heatmap
        # - Missing data heatmap
        # - Distribution plots
        
        return plots_info