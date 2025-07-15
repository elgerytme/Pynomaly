"""Core profiling engine for orchestrating comprehensive data profiling operations."""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from uuid import uuid4
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum

from ...domain.entities.data_profile import (
    DataProfile, ProfileId, DatasetId, ProfilingStatus, ProfilingMetadata
)
from ...infrastructure.adapters.file_adapter import get_file_adapter
from .schema_analysis_service import SchemaAnalysisService
from .statistical_profiling_service import StatisticalProfilingService
from .pattern_discovery_service import PatternDiscoveryService
from .quality_assessment_service import QualityAssessmentService
from .performance_optimizer import PerformanceOptimizer


class ProfilingStrategy(Enum):
    """Profiling strategy options."""
    FULL = "full"
    SAMPLE = "sample"
    FAST = "fast"
    COMPREHENSIVE = "comprehensive"
    INCREMENTAL = "incremental"


@dataclass
class ProfilingConfig:
    """Configuration for profiling operations."""
    strategy: ProfilingStrategy = ProfilingStrategy.FULL
    sample_size: Optional[int] = None
    sample_percentage: Optional[float] = None
    enable_parallel: bool = True
    max_workers: int = 4
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    quality_threshold: float = 0.8
    enable_pattern_discovery: bool = True
    enable_relationship_analysis: bool = True
    timeout_seconds: Optional[int] = None


class ProfilingEngine:
    """Central orchestration engine for comprehensive data profiling operations."""
    
    def __init__(self, config: Optional[ProfilingConfig] = None):
        """Initialize the profiling engine with configuration."""
        self.config = config or ProfilingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        self.schema_service = SchemaAnalysisService()
        self.stats_service = StatisticalProfilingService()
        self.pattern_service = PatternDiscoveryService()
        self.quality_service = QualityAssessmentService()
        self.performance_optimizer = PerformanceOptimizer()
        
        # Internal state
        self._active_profiles: Dict[str, DataProfile] = {}
        self._cache: Dict[str, Any] = {}
    
    async def profile_dataset(
        self,
        source: Union[str, Dict[str, Any]],
        source_type: str = "file",
        config_override: Optional[ProfilingConfig] = None
    ) -> DataProfile:
        """
        Profile a dataset from various sources.
        
        Args:
            source: Data source (file path, database connection, etc.)
            source_type: Type of data source (file, database, stream, etc.)
            config_override: Override default configuration
            
        Returns:
            Comprehensive DataProfile with all analysis results
        """
        effective_config = config_override or self.config
        start_time = datetime.utcnow()
        
        # Generate unique identifiers
        profile_id = ProfileId(value=uuid4())
        dataset_id = DatasetId(value=uuid4())
        
        # Initialize profile
        profile = DataProfile(
            profile_id=profile_id,
            dataset_id=dataset_id,
            status=ProfilingStatus.PENDING,
            source_type=source_type,
            source_connection=source if isinstance(source, dict) else {"path": source},
            created_at=start_time,
            metadata=ProfilingMetadata(
                profiling_strategy=effective_config.strategy.value,
                sample_size=effective_config.sample_size,
                enable_parallel=effective_config.enable_parallel
            )
        )
        
        # Store active profile
        self._active_profiles[str(profile_id.value)] = profile
        
        try:
            # Start profiling
            profile.start_profiling()
            self.logger.info(f"Started profiling dataset {dataset_id.value} with strategy {effective_config.strategy.value}")
            
            # Load and prepare data
            df = await self._load_data(source, source_type, effective_config)
            
            if df.empty:
                raise ValueError("Dataset is empty")
            
            # Apply performance optimizations
            df_optimized = await self._optimize_data_for_profiling(df, effective_config)
            
            # Execute profiling pipeline
            await self._execute_profiling_pipeline(profile, df_optimized, effective_config)
            
            # Generate summary and insights
            await self._generate_profile_summary(profile)
            
            # Complete profiling
            profile.complete_profiling()
            self.logger.info(f"Completed profiling dataset {dataset_id.value}")
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Profiling failed for dataset {dataset_id.value}: {str(e)}")
            profile.fail_profiling(str(e))
            raise
        finally:
            # Cleanup
            if str(profile_id.value) in self._active_profiles:
                del self._active_profiles[str(profile_id.value)]
    
    async def profile_incremental(
        self,
        source: Union[str, Dict[str, Any]],
        baseline_profile: DataProfile,
        source_type: str = "file"
    ) -> DataProfile:
        """
        Perform incremental profiling against a baseline profile.
        
        Args:
            source: New data source to profile
            baseline_profile: Previous profile for comparison
            source_type: Type of data source
            
        Returns:
            Updated DataProfile with delta analysis
        """
        self.logger.info(f"Starting incremental profiling against baseline {baseline_profile.profile_id.value}")
        
        # Configure for incremental strategy
        incremental_config = ProfilingConfig(
            strategy=ProfilingStrategy.INCREMENTAL,
            enable_parallel=self.config.enable_parallel,
            max_workers=self.config.max_workers
        )
        
        # Profile new data
        new_profile = await self.profile_dataset(source, source_type, incremental_config)
        
        # Perform delta analysis
        await self._analyze_profile_delta(baseline_profile, new_profile)
        
        return new_profile
    
    async def profile_distributed(
        self,
        sources: List[Union[str, Dict[str, Any]]],
        source_type: str = "file",
        merge_results: bool = True
    ) -> Union[DataProfile, List[DataProfile]]:
        """
        Profile multiple datasets in parallel with optional result merging.
        
        Args:
            sources: List of data sources to profile
            source_type: Type of data sources
            merge_results: Whether to merge results into single profile
            
        Returns:
            Single merged DataProfile or list of individual profiles
        """
        self.logger.info(f"Starting distributed profiling of {len(sources)} sources")
        
        # Create profiling tasks
        tasks = [
            self.profile_dataset(source, source_type)
            for source in sources
        ]
        
        # Execute in parallel
        profiles = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        successful_profiles = []
        for i, result in enumerate(profiles):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to profile source {i}: {str(result)}")
            else:
                successful_profiles.append(result)
        
        if not successful_profiles:
            raise RuntimeError("All distributed profiling tasks failed")
        
        if merge_results and len(successful_profiles) > 1:
            return await self._merge_profiles(successful_profiles)
        else:
            return successful_profiles if not merge_results else successful_profiles[0]
    
    async def get_profiling_status(self, profile_id: str) -> Optional[ProfilingStatus]:
        """Get the current status of an active profiling operation."""
        if profile_id in self._active_profiles:
            return self._active_profiles[profile_id].status
        return None
    
    async def cancel_profiling(self, profile_id: str) -> bool:
        """Cancel an active profiling operation."""
        if profile_id in self._active_profiles:
            profile = self._active_profiles[profile_id]
            profile.fail_profiling("Cancelled by user")
            del self._active_profiles[profile_id]
            self.logger.info(f"Cancelled profiling operation {profile_id}")
            return True
        return False
    
    async def _load_data(self, source: Union[str, Dict[str, Any]], source_type: str, config: ProfilingConfig) -> Any:
        """Load data from various source types."""
        if source_type == "file":
            if isinstance(source, str):
                adapter = get_file_adapter(source)
                return adapter.load(source)
            else:
                raise ValueError("File source must be a string path")
        
        elif source_type == "database":
            # TODO: Implement database adapter
            raise NotImplementedError("Database profiling not yet implemented")
        
        elif source_type == "stream":
            # TODO: Implement stream adapter
            raise NotImplementedError("Stream profiling not yet implemented")
        
        elif source_type == "cloud":
            # TODO: Implement cloud storage adapter
            raise NotImplementedError("Cloud storage profiling not yet implemented")
        
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    async def _optimize_data_for_profiling(self, df: Any, config: ProfilingConfig) -> Any:
        """Apply performance optimizations to data before profiling."""
        if config.strategy == ProfilingStrategy.SAMPLE:
            if config.sample_size:
                return self.performance_optimizer.optimize_sample_size(df, config.sample_size)
            elif config.sample_percentage:
                sample_size = int(len(df) * config.sample_percentage)
                return self.performance_optimizer.optimize_sample_size(df, sample_size)
        
        elif config.strategy == ProfilingStrategy.FAST:
            # Use smaller sample for fast profiling
            fast_sample_size = min(10000, len(df))
            return self.performance_optimizer.optimize_sample_size(df, fast_sample_size)
        
        # Apply memory optimizations
        return self.performance_optimizer.optimize_memory_usage(df)
    
    async def _execute_profiling_pipeline(self, profile: DataProfile, df: Any, config: ProfilingConfig) -> None:
        """Execute the complete profiling pipeline."""
        
        if config.enable_parallel:
            # Execute profiling steps in parallel where possible
            await self._execute_parallel_profiling(profile, df, config)
        else:
            # Execute profiling steps sequentially
            await self._execute_sequential_profiling(profile, df, config)
    
    async def _execute_parallel_profiling(self, profile: DataProfile, df: Any, config: ProfilingConfig) -> None:
        """Execute profiling steps in parallel for better performance."""
        
        # Create concurrent tasks for independent profiling steps
        tasks = []
        
        # Schema analysis (always first, as others may depend on it)
        schema_task = asyncio.create_task(self._run_schema_analysis(profile, df))
        tasks.append(("schema", schema_task))
        
        # Wait for schema analysis to complete first
        schema_profile = await schema_task
        profile.schema_profile = schema_profile
        
        # Now run other analyses in parallel
        parallel_tasks = [
            ("statistics", asyncio.create_task(self._run_statistical_analysis(profile, df))),
            ("quality", asyncio.create_task(self._run_quality_assessment(profile, df))),
        ]
        
        if config.enable_pattern_discovery:
            parallel_tasks.append(("patterns", asyncio.create_task(self._run_pattern_discovery(profile, df))))
        
        # Execute parallel tasks
        for task_name, task in parallel_tasks:
            try:
                result = await task
                setattr(profile, f"{task_name}_profile", result)
            except Exception as e:
                self.logger.error(f"Failed to execute {task_name} analysis: {str(e)}")
                # Continue with other analyses
        
        # Relationship analysis (depends on schema and patterns)
        if config.enable_relationship_analysis:
            relationships = await self._run_relationship_analysis(profile, df)
            profile.relationships = relationships
    
    async def _execute_sequential_profiling(self, profile: DataProfile, df: Any, config: ProfilingConfig) -> None:
        """Execute profiling steps sequentially."""
        
        # Schema analysis
        schema_profile = await self._run_schema_analysis(profile, df)
        profile.schema_profile = schema_profile
        
        # Statistical analysis
        stats_profile = await self._run_statistical_analysis(profile, df)
        profile.statistics_profile = stats_profile
        
        # Quality assessment
        quality_profile = await self._run_quality_assessment(profile, df)
        profile.quality_profile = quality_profile
        
        # Pattern discovery
        if config.enable_pattern_discovery:
            patterns_profile = await self._run_pattern_discovery(profile, df)
            profile.patterns_profile = patterns_profile
        
        # Relationship analysis
        if config.enable_relationship_analysis:
            relationships = await self._run_relationship_analysis(profile, df)
            profile.relationships = relationships
    
    async def _run_schema_analysis(self, profile: DataProfile, df: Any) -> Any:
        """Run schema analysis step."""
        return self.schema_service.infer(df)
    
    async def _run_statistical_analysis(self, profile: DataProfile, df: Any) -> Any:
        """Run statistical analysis step."""
        return self.stats_service.analyze(df)
    
    async def _run_quality_assessment(self, profile: DataProfile, df: Any) -> Any:
        """Run quality assessment step."""
        return self.quality_service.assess(df)
    
    async def _run_pattern_discovery(self, profile: DataProfile, df: Any) -> Any:
        """Run pattern discovery step."""
        return self.pattern_service.discover(df)
    
    async def _run_relationship_analysis(self, profile: DataProfile, df: Any) -> Any:
        """Run relationship analysis step."""
        # TODO: Implement comprehensive relationship analysis
        return []
    
    async def _generate_profile_summary(self, profile: DataProfile) -> None:
        """Generate comprehensive summary and insights for the profile."""
        
        # Calculate overall data quality score
        if hasattr(profile, 'quality_profile') and profile.quality_profile:
            overall_score = self.quality_service.calculate_overall_score(profile.quality_profile)
            profile.summary = {
                "overall_quality_score": overall_score,
                "total_rows": getattr(profile.schema_profile, 'row_count', 0) if hasattr(profile, 'schema_profile') else 0,
                "total_columns": getattr(profile.schema_profile, 'column_count', 0) if hasattr(profile, 'schema_profile') else 0,
                "data_quality_issues": getattr(profile.quality_profile, 'quality_issues', []) if hasattr(profile, 'quality_profile') else [],
                "key_insights": self._generate_key_insights(profile)
            }
    
    def _generate_key_insights(self, profile: DataProfile) -> List[str]:
        """Generate key insights from the profiling results."""
        insights = []
        
        # Schema insights
        if hasattr(profile, 'schema_profile') and profile.schema_profile:
            schema = profile.schema_profile
            if hasattr(schema, 'columns'):
                numerical_cols = len([col for col in schema.columns if getattr(col, 'inferred_type', '') in ['integer', 'float']])
                categorical_cols = len([col for col in schema.columns if getattr(col, 'inferred_type', '') == 'string'])
                
                if numerical_cols > categorical_cols:
                    insights.append(f"Dataset is primarily numerical ({numerical_cols} numerical vs {categorical_cols} categorical columns)")
                elif categorical_cols > numerical_cols:
                    insights.append(f"Dataset is primarily categorical ({categorical_cols} categorical vs {numerical_cols} numerical columns)")
        
        # Quality insights
        if hasattr(profile, 'quality_profile') and profile.quality_profile:
            quality = profile.quality_profile
            if hasattr(quality, 'overall_score'):
                if quality.overall_score >= 0.9:
                    insights.append("Excellent data quality with minimal issues")
                elif quality.overall_score >= 0.7:
                    insights.append("Good data quality with some minor issues")
                elif quality.overall_score >= 0.5:
                    insights.append("Moderate data quality requiring attention")
                else:
                    insights.append("Poor data quality requiring significant cleanup")
        
        return insights
    
    async def _analyze_profile_delta(self, baseline: DataProfile, new_profile: DataProfile) -> None:
        """Analyze differences between baseline and new profile."""
        # TODO: Implement comprehensive delta analysis
        pass
    
    async def _merge_profiles(self, profiles: List[DataProfile]) -> DataProfile:
        """Merge multiple profiles into a single consolidated profile."""
        # TODO: Implement profile merging logic
        # For now, return the first profile
        return profiles[0]