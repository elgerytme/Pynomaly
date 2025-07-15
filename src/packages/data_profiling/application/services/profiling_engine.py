import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

from ...domain.entities.data_profile import (
    DataProfile, ProfilingStatus, ProfileId, DatasetId, 
    ProfilingMetadata, QualityAssessment
)
from .schema_analysis_service import SchemaAnalysisService
from .statistical_profiling_service import StatisticalProfilingService
from .pattern_discovery_service import PatternDiscoveryService
from .quality_assessment_service import QualityAssessmentService
from .performance_optimizer import PerformanceOptimizer

logger = logging.getLogger(__name__)


@dataclass
class ProfilingConfig:
    """Configuration for profiling operations."""
    
    # Sampling configuration
    enable_sampling: bool = True
    sample_size: int = 10000
    sample_percentage: Optional[float] = None
    
    # Analysis configuration
    include_schema_analysis: bool = True
    include_statistical_analysis: bool = True
    include_pattern_discovery: bool = True
    include_quality_assessment: bool = True
    include_relationship_analysis: bool = True
    
    # Performance configuration
    enable_parallel_processing: bool = True
    max_workers: int = 4
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    
    # Advanced features
    enable_advanced_patterns: bool = True
    enable_ml_clustering: bool = True
    enable_pii_detection: bool = True
    enable_time_series_analysis: bool = True
    
    # Quality thresholds
    completeness_threshold: float = 0.95
    consistency_threshold: float = 0.90
    accuracy_threshold: float = 0.85
    
    # Memory management
    max_memory_mb: int = 2048
    chunk_size: int = 1000
    
    # Output configuration
    include_examples: bool = True
    max_examples_per_pattern: int = 5


class ProfilingEngine:
    """Main orchestrator for comprehensive data profiling operations."""
    
    def __init__(self, config: ProfilingConfig = None):
        self.config = config or ProfilingConfig()
        
        # Initialize services
        self.schema_service = SchemaAnalysisService()
        self.statistical_service = StatisticalProfilingService()
        self.pattern_service = PatternDiscoveryService()
        self.quality_service = QualityAssessmentService()
        self.performance_optimizer = PerformanceOptimizer()
        
        # Initialize cache
        self.cache = {} if self.config.enable_caching else None
        
        logger.info(f"Initialized ProfilingEngine with config: {self.config}")
    
    def profile_dataset(self, df: pd.DataFrame, 
                       dataset_id: str = None,
                       source_type: str = "dataframe",
                       source_connection: Dict[str, Any] = None) -> DataProfile:
        """Profile a complete dataset."""
        
        # Generate IDs
        profile_id = ProfileId()
        dataset_id = DatasetId() if dataset_id is None else DatasetId(dataset_id)
        
        # Create initial profile
        profile = DataProfile(
            profile_id=profile_id,
            dataset_id=dataset_id,
            source_type=source_type,
            source_connection=source_connection or {}
        )
        
        try:
            # Start profiling
            profile.start_profiling()
            start_time = time.time()
            
            # Optimize dataset for profiling
            optimized_df = self._optimize_dataset(df)
            
            # Run comprehensive profiling
            results = self._run_comprehensive_profiling(optimized_df)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create metadata
            metadata = ProfilingMetadata(
                profiling_strategy=self._get_profiling_strategy(df),
                sample_size=len(optimized_df) if self.config.enable_sampling else None,
                sample_percentage=self._calculate_sample_percentage(df, optimized_df),
                execution_time_seconds=execution_time,
                memory_usage_mb=self._estimate_memory_usage(optimized_df),
                include_patterns=self.config.include_pattern_discovery,
                include_statistical_analysis=self.config.include_statistical_analysis,
                include_quality_assessment=self.config.include_quality_assessment
            )
            
            # Complete profiling
            profile.complete_profiling(
                schema_profile=results['schema_profile'],
                quality_assessment=results['quality_assessment'],
                metadata=metadata
            )
            
            logger.info(f"Successfully profiled dataset: {len(df)} rows, {len(df.columns)} columns in {execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Profiling failed: {str(e)}")
            profile.fail_profiling(str(e))
            raise
        
        return profile
    
    def _optimize_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize dataset for profiling performance."""
        try:
            # Apply performance optimizations
            if self.config.enable_sampling:
                df = self.performance_optimizer.apply_intelligent_sampling(
                    df, 
                    target_size=self.config.sample_size,
                    target_percentage=self.config.sample_percentage
                )
            
            # Memory optimization
            df = self.performance_optimizer.optimize_memory_usage(df)
            
            # Data type optimization
            df = self.performance_optimizer.optimize_data_types(df)
            
            return df
            
        except Exception as e:
            logger.warning(f"Dataset optimization failed: {e}")
            return df
    
    def _run_comprehensive_profiling(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive profiling analysis."""
        results = {}
        
        if self.config.enable_parallel_processing:
            # Run analyses in parallel
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {}
                
                # Submit profiling tasks
                if self.config.include_schema_analysis:
                    futures['schema'] = executor.submit(self._run_schema_analysis, df)
                
                if self.config.include_statistical_analysis:
                    futures['statistical'] = executor.submit(self._run_statistical_analysis, df)
                
                if self.config.include_pattern_discovery:
                    futures['patterns'] = executor.submit(self._run_pattern_discovery, df)
                
                if self.config.include_relationship_analysis:
                    futures['relationships'] = executor.submit(self._run_relationship_analysis, df)
                
                # Collect results
                for key, future in futures.items():
                    try:
                        results[key] = future.result(timeout=300)  # 5 minute timeout
                    except Exception as e:
                        logger.error(f"Error in {key} analysis: {e}")
                        results[key] = None
        else:
            # Run analyses sequentially
            if self.config.include_schema_analysis:
                results['schema'] = self._run_schema_analysis(df)
            
            if self.config.include_statistical_analysis:
                results['statistical'] = self._run_statistical_analysis(df)
            
            if self.config.include_pattern_discovery:
                results['patterns'] = self._run_pattern_discovery(df)
            
            if self.config.include_relationship_analysis:
                results['relationships'] = self._run_relationship_analysis(df)
        
        # Integrate patterns into schema profile
        schema_profile = results.get('schema')
        if schema_profile and results.get('patterns'):
            schema_profile = self._integrate_patterns_into_schema(schema_profile, results['patterns'])
        
        # Run quality assessment (depends on schema analysis)
        quality_assessment = None
        if self.config.include_quality_assessment and schema_profile:
            quality_assessment = self._run_quality_assessment(schema_profile, df)
        
        return {
            'schema_profile': schema_profile,
            'quality_assessment': quality_assessment,
            'statistical_results': results.get('statistical'),
            'pattern_results': results.get('patterns'),
            'relationship_results': results.get('relationships')
        }
    
    def _run_schema_analysis(self, df: pd.DataFrame):
        """Run schema analysis."""
        try:
            logger.info("Running schema analysis...")
            
            # Check cache
            cache_key = f"schema_{self._get_dataframe_hash(df)}"
            if self.cache and cache_key in self.cache:
                return self.cache[cache_key]
            
            # Run analysis
            schema_profile = self.schema_service.infer(df)
            
            # Cache result
            if self.cache:
                self.cache[cache_key] = schema_profile
            
            return schema_profile
            
        except Exception as e:
            logger.error(f"Schema analysis failed: {e}")
            raise
    
    def _run_statistical_analysis(self, df: pd.DataFrame):
        """Run statistical analysis."""
        try:
            logger.info("Running statistical analysis...")
            
            # Check cache
            cache_key = f"statistical_{self._get_dataframe_hash(df)}"
            if self.cache and cache_key in self.cache:
                return self.cache[cache_key]
            
            # Run analysis
            statistical_results = {}
            
            # Basic statistical analysis
            statistical_results['column_stats'] = self.statistical_service.analyze(df)
            
            # Distribution analysis
            statistical_results['distributions'] = {}
            for col in df.select_dtypes(include='number').columns:
                statistical_results['distributions'][col] = self.statistical_service.analyze_distribution(df[col])
            
            # Correlation analysis
            statistical_results['correlations'] = self.statistical_service.correlation_analysis(df)
            
            # Outlier detection
            statistical_results['outliers'] = {}
            for col in df.select_dtypes(include='number').columns:
                statistical_results['outliers'][col] = self.statistical_service.detect_outliers(df[col])
            
            # Time series analysis (if applicable)
            if self.config.enable_time_series_analysis:
                date_columns = df.select_dtypes(include=['datetime64']).columns
                if len(date_columns) > 0:
                    for date_col in date_columns:
                        for value_col in df.select_dtypes(include='number').columns:
                            ts_key = f"{date_col}_{value_col}"
                            statistical_results[f'time_series_{ts_key}'] = self.statistical_service.analyze_time_series(
                                df[value_col], df[date_col]
                            )
            
            # Cache result
            if self.cache:
                self.cache[cache_key] = statistical_results
            
            return statistical_results
            
        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            raise
    
    def _run_pattern_discovery(self, df: pd.DataFrame):
        """Run pattern discovery."""
        try:
            logger.info("Running pattern discovery...")
            
            # Check cache
            cache_key = f"patterns_{self._get_dataframe_hash(df)}"
            if self.cache and cache_key in self.cache:
                return self.cache[cache_key]
            
            # Run analysis
            pattern_results = {}
            
            # Basic pattern discovery
            pattern_results['column_patterns'] = self.pattern_service.discover(df)
            
            # Data relationships
            pattern_results['relationships'] = self.pattern_service.analyze_data_relationships(df)
            
            # Cache result
            if self.cache:
                self.cache[cache_key] = pattern_results
            
            return pattern_results
            
        except Exception as e:
            logger.error(f"Pattern discovery failed: {e}")
            raise
    
    def _run_relationship_analysis(self, df: pd.DataFrame):
        """Run relationship analysis."""
        try:
            logger.info("Running relationship analysis...")
            
            # Check cache
            cache_key = f"relationships_{self._get_dataframe_hash(df)}"
            if self.cache and cache_key in self.cache:
                return self.cache[cache_key]
            
            # Run analysis
            relationship_results = self.schema_service.detect_advanced_relationships(df)
            
            # Cache result
            if self.cache:
                self.cache[cache_key] = relationship_results
            
            return relationship_results
            
        except Exception as e:
            logger.error(f"Relationship analysis failed: {e}")
            raise
    
    def _run_quality_assessment(self, schema_profile, df: pd.DataFrame):
        """Run quality assessment."""
        try:
            logger.info("Running quality assessment...")
            
            # Check cache
            cache_key = f"quality_{self._get_dataframe_hash(df)}"
            if self.cache and cache_key in self.cache:
                return self.cache[cache_key]
            
            # Run assessment
            quality_assessment = self.quality_service.assess_quality(schema_profile, df)
            
            # Cache result
            if self.cache:
                self.cache[cache_key] = quality_assessment
            
            return quality_assessment
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            raise
    
    def _integrate_patterns_into_schema(self, schema_profile, pattern_results):
        """Integrate discovered patterns into schema profile."""
        try:
            if not pattern_results or 'column_patterns' not in pattern_results:
                return schema_profile
            
            column_patterns = pattern_results['column_patterns']
            
            # Update column profiles with patterns
            for column in schema_profile.columns:
                if column.column_name in column_patterns:
                    # Update patterns (this would require making ColumnProfile mutable or creating a new one)
                    # For now, we'll log the integration
                    patterns = column_patterns[column.column_name]
                    logger.info(f"Found {len(patterns)} patterns for column {column.column_name}")
            
            return schema_profile
            
        except Exception as e:
            logger.error(f"Pattern integration failed: {e}")
            return schema_profile
    
    def _get_profiling_strategy(self, df: pd.DataFrame) -> str:
        """Determine profiling strategy based on data size."""
        total_size = df.memory_usage(deep=True).sum()
        
        if total_size > 100 * 1024 * 1024:  # > 100MB
            return "sample"
        elif len(df) > 50000:  # > 50k rows
            return "incremental"
        else:
            return "full"
    
    def _calculate_sample_percentage(self, original_df: pd.DataFrame, sample_df: pd.DataFrame) -> Optional[float]:
        """Calculate sampling percentage."""
        if len(original_df) == 0:
            return None
        
        return (len(sample_df) / len(original_df)) * 100
    
    def _estimate_memory_usage(self, df: pd.DataFrame) -> float:
        """Estimate memory usage in MB."""
        return df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    def _get_dataframe_hash(self, df: pd.DataFrame) -> str:
        """Generate hash for DataFrame caching."""
        try:
            # Create a hash based on DataFrame shape and column names
            import hashlib
            
            hash_input = f"{df.shape}_{list(df.columns)}_{df.dtypes.to_dict()}"
            return hashlib.md5(hash_input.encode()).hexdigest()
            
        except Exception:
            return str(time.time())
    
    def profile_incremental(self, df: pd.DataFrame, 
                           previous_profile: DataProfile) -> DataProfile:
        """Perform incremental profiling on new data."""
        try:
            logger.info("Running incremental profiling...")
            
            # For now, run full profiling
            # TODO: Implement true incremental profiling
            return self.profile_dataset(df)
            
        except Exception as e:
            logger.error(f"Incremental profiling failed: {e}")
            raise
    
    def profile_streaming(self, df: pd.DataFrame, 
                         window_size: int = 1000) -> DataProfile:
        """Profile streaming data with sliding window."""
        try:
            logger.info("Running streaming profiling...")
            
            # Take a window of recent data
            if len(df) > window_size:
                windowed_df = df.tail(window_size)
            else:
                windowed_df = df
            
            # Run profiling on windowed data
            profile = self.profile_dataset(windowed_df, source_type="streaming")
            
            return profile
            
        except Exception as e:
            logger.error(f"Streaming profiling failed: {e}")
            raise
    
    def get_profiling_summary(self, profile: DataProfile) -> Dict[str, Any]:
        """Get a summary of profiling results."""
        try:
            summary = {
                'profile_id': str(profile.profile_id.value),
                'dataset_id': str(profile.dataset_id.value),
                'status': profile.status,
                'source_type': profile.source_type,
                'created_at': profile.created_at,
                'execution_time': profile.profiling_metadata.execution_time_seconds if profile.profiling_metadata else None
            }
            
            if profile.schema_profile:
                summary['schema'] = {
                    'total_columns': profile.schema_profile.total_columns,
                    'total_rows': profile.schema_profile.total_rows,
                    'primary_keys': profile.schema_profile.primary_keys,
                    'foreign_keys': profile.schema_profile.foreign_keys,
                    'estimated_size_mb': profile.schema_profile.estimated_size_bytes / (1024 * 1024) if profile.schema_profile.estimated_size_bytes else None
                }
            
            if profile.quality_assessment:
                summary['quality'] = {
                    'overall_score': profile.quality_assessment.overall_score,
                    'completeness_score': profile.quality_assessment.completeness_score,
                    'consistency_score': profile.quality_assessment.consistency_score,
                    'accuracy_score': profile.quality_assessment.accuracy_score,
                    'total_issues': (profile.quality_assessment.critical_issues + 
                                   profile.quality_assessment.high_issues + 
                                   profile.quality_assessment.medium_issues + 
                                   profile.quality_assessment.low_issues)
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate profiling summary: {e}")
            return {'error': str(e)}
    
    def clear_cache(self) -> None:
        """Clear profiling cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Profiling cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cache usage."""
        if not self.cache:
            return {'cache_enabled': False}
        
        return {
            'cache_enabled': True,
            'cache_size': len(self.cache),
            'cache_keys': list(self.cache.keys())[:10]  # Show first 10 keys
        }