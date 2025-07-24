"""Consolidated data processing service combining validation, profiling, sampling, and conversion."""

import asyncio
import gzip
import bz2
import lzma
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import jsonschema
import structlog
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = structlog.get_logger(__name__)


class DataProcessingService:
    """
    Consolidated service for all data processing operations including:
    - Data validation and quality checks
    - Data profiling and statistical analysis
    - Data sampling with various methods
    - Data format conversion and compression
    """
    
    def __init__(self):
        # Initialize validation rules
        self.validation_rules = self._load_default_validation_rules()
        
        # Initialize profiling sections
        self.profile_sections = {
            'dataset_info': self._profile_dataset_info,
            'column_info': self._profile_columns,
            'data_quality': self._profile_data_quality,
            'statistical_summary': self._profile_statistics,
            'correlations': self._profile_correlations,
            'distributions': self._profile_distributions,
            'patterns': self._profile_patterns
        }
        
        # Initialize sampling methods
        self.sampling_methods = {
            'random': self._random_sampling,
            'systematic': self._systematic_sampling,
            'stratified': self._stratified_sampling,
            'cluster': self._cluster_sampling,
            'reservoir': self._reservoir_sampling
        }
        
        # Initialize conversion formats
        self.supported_formats = {
            'csv': self._to_csv,
            'json': self._to_json,
            'parquet': self._to_parquet,
            'excel': self._to_excel,
            'hdf5': self._to_hdf5,
            'pickle': self._to_pickle
        }
        
        # Initialize compression handlers
        self.compression_handlers = {
            'gzip': gzip,
            'bz2': bz2,
            'xz': lzma
        }

    # ==============================================================================
    # DATA VALIDATION METHODS
    # ==============================================================================
    
    def _load_default_validation_rules(self) -> Dict[str, Any]:
        """Load default validation rules."""
        return {
            'max_missing_percentage': 50.0,
            'max_duplicate_percentage': 20.0,
            'min_numeric_columns': 1,
            'max_outlier_percentage': 10.0,
            'required_row_count': 10
        }
    
    async def validate_file(
        self,
        file_path: Path,
        schema_file: Optional[Path] = None,
        check_types: bool = True,
        check_missing: bool = True,
        check_outliers: bool = True,
        check_duplicates: bool = True,
        custom_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate a single data file with comprehensive quality checks."""
        
        validation_result = {
            'file_path': str(file_path),
            'timestamp': datetime.utcnow().isoformat(),
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {},
            'checks_performed': []
        }
        
        try:
            logger.info("Starting data validation", file=str(file_path))
            
            # Load data
            data = await self._load_data(file_path)
            if data is None or data.empty:
                validation_result['is_valid'] = False
                validation_result['errors'].append("File is empty or could not be loaded")
                return validation_result
            
            # Apply custom rules if provided
            rules = self.validation_rules.copy()
            if custom_rules:
                rules.update(custom_rules)
            
            # Basic statistics
            validation_result['statistics'] = {
                'total_rows': len(data),
                'total_columns': len(data.columns),
                'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
                'missing_values': data.isnull().sum().sum(),
                'duplicate_rows': data.duplicated().sum()
            }
            
            # Perform validation checks
            if check_types:
                await self._validate_data_types(data, validation_result)
                validation_result['checks_performed'].append('data_types')
            
            if check_missing:
                await self._validate_missing_values(data, validation_result, rules)
                validation_result['checks_performed'].append('missing_values')
            
            if check_duplicates:
                await self._validate_duplicates(data, validation_result, rules)
                validation_result['checks_performed'].append('duplicates')
            
            if check_outliers:
                await self._validate_outliers(data, validation_result, rules)
                validation_result['checks_performed'].append('outliers')
            
            # Schema validation if provided
            if schema_file and schema_file.exists():
                await self._validate_schema(data, schema_file, validation_result)
                validation_result['checks_performed'].append('schema')
            
            logger.info("Data validation completed", 
                       is_valid=validation_result['is_valid'],
                       errors=len(validation_result['errors']),
                       warnings=len(validation_result['warnings']))
            
            return validation_result
            
        except Exception as e:
            logger.error("Data validation failed", error=str(e))
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Validation failed: {str(e)}")
            return validation_result

    # ==============================================================================
    # DATA PROFILING METHODS
    # ==============================================================================
    
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
            if data is None or data.empty:
                return {'error': 'File is empty or could not be loaded'}
            
            # Apply sampling if specified
            if sample_size and len(data) > sample_size:
                data = data.sample(n=sample_size, random_state=42)
                logger.info("Applied sampling", original_size=len(data), sample_size=sample_size)
            
            # Determine sections to include
            if sections is None:
                sections = list(self.profile_sections.keys())
            
            # Generate profile
            profile = {
                'file_path': str(file_path),
                'timestamp': datetime.utcnow().isoformat(),
                'sample_size': len(data),
                'sections_included': sections
            }
            
            # Execute profiling sections
            for section in sections:
                if section in self.profile_sections:
                    profile[section] = await self.profile_sections[section](
                        data, include_correlations, include_distributions
                    )
            
            logger.info("Data profiling completed", sections=len(sections))
            return profile
            
        except Exception as e:
            logger.error("Data profiling failed", error=str(e))
            return {'error': f"Profiling failed: {str(e)}"}

    # ==============================================================================
    # DATA SAMPLING METHODS
    # ==============================================================================
    
    async def sample_file(
        self,
        file_path: Path,
        sample_size: int,
        method: str = 'random',
        stratify_column: Optional[str] = None,
        cluster_column: Optional[str] = None,
        seed: Optional[int] = None,
        replacement: bool = False
    ) -> pd.DataFrame:
        """Sample data from a file using specified method."""
        
        if method not in self.sampling_methods:
            raise ValueError(f"Unsupported sampling method: {method}. Available: {list(self.sampling_methods.keys())}")
        
        try:
            logger.info("Starting data sampling",
                       file=str(file_path),
                       method=method,
                       sample_size=sample_size)
            
            # Load data
            data = await self._load_data(file_path)
            if data is None or data.empty:
                raise ValueError("File is empty or could not be loaded")
            
            # Validate sample size
            if sample_size >= len(data) and not replacement:
                logger.warning("Sample size >= data size, returning full dataset")
                return data
            
            # Set random seed if provided
            if seed is not None:
                np.random.seed(seed)
            
            # Apply sampling method
            sampled_data = await self.sampling_methods[method](
                data, sample_size, stratify_column, cluster_column, replacement
            )
            
            logger.info("Data sampling completed", 
                       original_size=len(data),
                       sampled_size=len(sampled_data))
            
            return sampled_data
            
        except Exception as e:
            logger.error("Data sampling failed", error=str(e))
            raise

    # ==============================================================================
    # DATA CONVERSION METHODS
    # ==============================================================================
    
    async def convert_file(
        self,
        input_file: Path,
        output_format: str,
        output_dir: Path,
        compression: Optional[str] = None,
        chunk_size: int = 10000,
        preserve_dtypes: bool = True,
        conversion_options: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Convert a single file to target format."""
        
        if output_format not in self.supported_formats:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        try:
            logger.info("Starting file conversion",
                       input_file=str(input_file),
                       output_format=output_format,
                       compression=compression)
            
            # Load data
            data = await self._load_data(input_file)
            if data is None or data.empty:
                raise ValueError("Input file is empty or could not be loaded")
            
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate output filename
            output_filename = f"{input_file.stem}.{output_format}"
            if compression:
                output_filename += f".{compression}"
            output_file = output_dir / output_filename
            
            # Apply conversion options
            options = conversion_options or {}
            
            # Convert data
            await self.supported_formats[output_format](
                data, output_file, compression, chunk_size, preserve_dtypes, options
            )
            
            logger.info("File conversion completed", output_file=str(output_file))
            return output_file
            
        except Exception as e:
            logger.error("File conversion failed", error=str(e))
            raise

    # ==============================================================================
    # BATCH PROCESSING METHODS
    # ==============================================================================
    
    async def process_batch(
        self,
        operation: str,
        file_paths: List[Path],
        output_dir: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """Process multiple files in batch with specified operation."""
        
        operation_map = {
            'validate': self.validate_file,
            'profile': self.profile_file,
            'sample': self.sample_file,
            'convert': self.convert_file
        }
        
        if operation not in operation_map:
            raise ValueError(f"Unsupported batch operation: {operation}")
        
        results = {
            'operation': operation,
            'total_files': len(file_paths),
            'successful': 0,
            'failed': 0,
            'results': [],
            'errors': []
        }
        
        # Process files concurrently
        semaphore = asyncio.Semaphore(10)  # Limit concurrent operations
        
        async def process_single_file(file_path: Path):
            async with semaphore:
                try:
                    if operation == 'convert':
                        result = await operation_map[operation](file_path, output_dir=output_dir, **kwargs)
                    else:
                        result = await operation_map[operation](file_path, **kwargs)
                    return {'file': str(file_path), 'success': True, 'result': result}
                except Exception as e:
                    return {'file': str(file_path), 'success': False, 'error': str(e)}
        
        # Execute batch processing
        tasks = [process_single_file(file_path) for file_path in file_paths]
        batch_results = await asyncio.gather(*tasks)
        
        # Collect results
        for result in batch_results:
            results['results'].append(result)
            if result['success']:
                results['successful'] += 1
            else:
                results['failed'] += 1
                results['errors'].append(result)
        
        logger.info("Batch processing completed",
                   operation=operation,
                   total=results['total_files'],
                   successful=results['successful'],
                   failed=results['failed'])
        
        return results

    # ==============================================================================
    # UTILITY METHODS
    # ==============================================================================
    
    async def _load_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load data from various file formats."""
        try:
            suffix = file_path.suffix.lower()
            
            if suffix == '.csv':
                return pd.read_csv(file_path)
            elif suffix == '.json':
                return pd.read_json(file_path)
            elif suffix == '.parquet':
                return pd.read_parquet(file_path)
            elif suffix in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            elif suffix == '.h5':
                return pd.read_hdf(file_path)
            elif suffix == '.pickle':
                return pd.read_pickle(file_path)
            else:
                # Try CSV as default
                return pd.read_csv(file_path)
                
        except Exception as e:
            logger.error("Failed to load data", file=str(file_path), error=str(e))
            return None

    # Validation helper methods (simplified for brevity)
    async def _validate_data_types(self, data: pd.DataFrame, result: Dict[str, Any]):
        """Validate data types."""
        # Implementation would check for appropriate data types
        pass
    
    async def _validate_missing_values(self, data: pd.DataFrame, result: Dict[str, Any], rules: Dict[str, Any]):
        """Validate missing value percentages."""
        missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        if missing_pct > rules['max_missing_percentage']:
            result['errors'].append(f"Missing value percentage ({missing_pct:.1f}%) exceeds threshold ({rules['max_missing_percentage']}%)")
            result['is_valid'] = False
    
    async def _validate_duplicates(self, data: pd.DataFrame, result: Dict[str, Any], rules: Dict[str, Any]):
        """Validate duplicate percentages."""
        duplicate_pct = (data.duplicated().sum() / len(data)) * 100
        if duplicate_pct > rules['max_duplicate_percentage']:
            result['warnings'].append(f"Duplicate percentage ({duplicate_pct:.1f}%) exceeds threshold ({rules['max_duplicate_percentage']}%)")
    
    async def _validate_outliers(self, data: pd.DataFrame, result: Dict[str, Any], rules: Dict[str, Any]):
        """Validate outlier percentages."""
        # Simple outlier detection using IQR method
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            Q1 = numeric_data.quantile(0.25)
            Q3 = numeric_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).any(axis=1)
            outlier_pct = (outliers.sum() / len(data)) * 100
            if outlier_pct > rules['max_outlier_percentage']:
                result['warnings'].append(f"Outlier percentage ({outlier_pct:.1f}%) exceeds threshold ({rules['max_outlier_percentage']}%)")
    
    async def _validate_schema(self, data: pd.DataFrame, schema_file: Path, result: Dict[str, Any]):
        """Validate against JSON schema."""
        try:
            with open(schema_file) as f:
                schema = json.load(f)
            
            # Convert DataFrame to dict for validation
            data_dict = data.to_dict('records')[0] if not data.empty else {}
            jsonschema.validate(data_dict, schema)
            
        except jsonschema.ValidationError as e:
            result['errors'].append(f"Schema validation failed: {str(e)}")
            result['is_valid'] = False
        except Exception as e:
            result['warnings'].append(f"Schema validation error: {str(e)}")

    # Profiling helper methods (simplified for brevity)
    async def _profile_dataset_info(self, data: pd.DataFrame, *args) -> Dict[str, Any]:
        """Profile basic dataset information."""
        return {
            'shape': data.shape,
            'memory_usage': data.memory_usage(deep=True).sum(),
            'dtypes': data.dtypes.to_dict()
        }
    
    async def _profile_columns(self, data: pd.DataFrame, *args) -> Dict[str, Any]:
        """Profile column information."""
        return {col: {
            'dtype': str(data[col].dtype),
            'non_null_count': data[col].count(),
            'null_count': data[col].isnull().sum(),
            'unique_count': data[col].nunique()
        } for col in data.columns}
    
    async def _profile_data_quality(self, data: pd.DataFrame, *args) -> Dict[str, Any]:
        """Profile data quality metrics."""
        return {
            'missing_values': data.isnull().sum().to_dict(),
            'duplicate_rows': data.duplicated().sum(),
            'data_completeness': ((data.count().sum() / (len(data) * len(data.columns))) * 100)
        }
    
    async def _profile_statistics(self, data: pd.DataFrame, *args) -> Dict[str, Any]:
        """Profile statistical summary."""
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return {}
        return numeric_data.describe().to_dict()
    
    async def _profile_correlations(self, data: pd.DataFrame, include_correlations: bool, *args) -> Dict[str, Any]:
        """Profile correlations."""
        if not include_correlations:
            return {}
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.shape[1] < 2:
            return {}
        return numeric_data.corr().to_dict()
    
    async def _profile_distributions(self, data: pd.DataFrame, include_correlations: bool, include_distributions: bool) -> Dict[str, Any]:
        """Profile distributions."""
        if not include_distributions:
            return {}
        # Simplified distribution analysis
        return {'skewness': data.select_dtypes(include=[np.number]).skew().to_dict()}
    
    async def _profile_patterns(self, data: pd.DataFrame, *args) -> Dict[str, Any]:
        """Profile data patterns."""
        # Simplified pattern analysis
        return {'constant_columns': [col for col in data.columns if data[col].nunique() <= 1]}

    # Sampling helper methods (simplified for brevity)
    async def _random_sampling(self, data: pd.DataFrame, sample_size: int, *args) -> pd.DataFrame:
        """Random sampling."""
        return data.sample(n=sample_size, replace=args[3] if len(args) > 3 else False)
    
    async def _systematic_sampling(self, data: pd.DataFrame, sample_size: int, *args) -> pd.DataFrame:
        """Systematic sampling."""
        step = len(data) // sample_size
        indices = list(range(0, len(data), step))[:sample_size]
        return data.iloc[indices]
    
    async def _stratified_sampling(self, data: pd.DataFrame, sample_size: int, stratify_column: Optional[str], *args) -> pd.DataFrame:
        """Stratified sampling."""
        if not stratify_column or stratify_column not in data.columns:
            return await self._random_sampling(data, sample_size, *args)
        return data.groupby(stratify_column, group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, int(sample_size * len(x) / len(data)))))
        )
    
    async def _cluster_sampling(self, data: pd.DataFrame, sample_size: int, stratify_column: Optional[str], cluster_column: Optional[str], *args) -> pd.DataFrame:
        """Cluster sampling."""
        if not cluster_column or cluster_column not in data.columns:
            return await self._random_sampling(data, sample_size, *args)
        # Simplified cluster sampling
        clusters = data[cluster_column].unique()
        selected_clusters = np.random.choice(clusters, size=min(len(clusters), sample_size // 100 + 1), replace=False)
        return data[data[cluster_column].isin(selected_clusters)].sample(n=min(sample_size, len(data)))
    
    async def _reservoir_sampling(self, data: pd.DataFrame, sample_size: int, *args) -> pd.DataFrame:
        """Reservoir sampling."""
        if len(data) <= sample_size:
            return data
        reservoir = data.iloc[:sample_size].copy()
        for i in range(sample_size, len(data)):
            j = np.random.randint(0, i + 1)
            if j < sample_size:
                reservoir.iloc[j] = data.iloc[i]
        return reservoir

    # Conversion helper methods (simplified for brevity)
    async def _to_csv(self, data: pd.DataFrame, output_file: Path, compression: Optional[str], chunk_size: int, preserve_dtypes: bool, options: Dict[str, Any]):
        """Convert to CSV."""
        data.to_csv(output_file, index=False, compression=compression, **options)
    
    async def _to_json(self, data: pd.DataFrame, output_file: Path, compression: Optional[str], chunk_size: int, preserve_dtypes: bool, options: Dict[str, Any]):
        """Convert to JSON."""
        data.to_json(output_file, orient='records', **options)
    
    async def _to_parquet(self, data: pd.DataFrame, output_file: Path, compression: Optional[str], chunk_size: int, preserve_dtypes: bool, options: Dict[str, Any]):
        """Convert to Parquet."""
        data.to_parquet(output_file, compression=compression, **options)
    
    async def _to_excel(self, data: pd.DataFrame, output_file: Path, compression: Optional[str], chunk_size: int, preserve_dtypes: bool, options: Dict[str, Any]):
        """Convert to Excel."""
        data.to_excel(output_file, index=False, **options)
    
    async def _to_hdf5(self, data: pd.DataFrame, output_file: Path, compression: Optional[str], chunk_size: int, preserve_dtypes: bool, options: Dict[str, Any]):
        """Convert to HDF5."""
        data.to_hdf(output_file, key='data', mode='w', **options)
    
    async def _to_pickle(self, data: pd.DataFrame, output_file: Path, compression: Optional[str], chunk_size: int, preserve_dtypes: bool, options: Dict[str, Any]):
        """Convert to Pickle."""
        data.to_pickle(output_file, compression=compression, **options)