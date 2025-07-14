from datetime import datetime
from uuid import uuid4
from ...infrastructure.adapters.file_adapter import get_file_adapter
from ..services.schema_analysis_service import SchemaAnalysisService
from ..services.statistical_profiling_service import StatisticalProfilingService
from ..services.pattern_discovery_service import PatternDiscoveryService
from ..services.quality_assessment_service import QualityAssessmentService
from ...domain.entities.data_profile import (
    DataProfile, ProfileId, DatasetId, ProfilingStatus,
    ProfilingMetadata
)


class ProfileDatasetUseCase:
    """Use case to orchestrate comprehensive data profiling for various data sources."""
    
    def __init__(self) -> None:
        self.schema_service = SchemaAnalysisService()
        self.stats_service = StatisticalProfilingService()
        self.pattern_service = PatternDiscoveryService()
        self.quality_service = QualityAssessmentService()

    def execute(self, path: str, profiling_strategy: str = "full") -> DataProfile:
        """Profile the dataset at the given path and return a comprehensive DataProfile."""
        start_time = datetime.utcnow()
        
        # Initialize profile
        profile = DataProfile(
            profile_id=ProfileId(value=uuid4()),
            dataset_id=DatasetId(value=uuid4()),
            status=ProfilingStatus.PENDING,
            source_type="file",
            source_connection={"path": path},
            created_at=start_time
        )
        
        try:
            # Start profiling
            profile.start_profiling()
            
            # Load data
            adapter = get_file_adapter(path)
            df = adapter.load(path)
            
            if df.empty:
                raise ValueError("Dataset is empty")
            
            # Schema analysis
            schema_profile = self.schema_service.infer(df)
            
            # Pattern discovery
            patterns = self.pattern_service.discover(df)
            
            # Attach patterns to column profiles
            for col_profile in schema_profile.columns:
                if col_profile.column_name in patterns:
                    # Since Pydantic models are immutable, we need to update the patterns
                    col_profile.patterns.extend(patterns[col_profile.column_name])
            
            # Statistical analysis (for the schema profile's statistical summaries)
            statistical_summaries = self.stats_service.analyze(df)
            
            # Update statistical summaries in column profiles
            for col_profile in schema_profile.columns:
                if col_profile.column_name in statistical_summaries:
                    # Update the statistical summary if it exists
                    if col_profile.statistical_summary is None:
                        col_profile.statistical_summary = statistical_summaries[col_profile.column_name]
            
            # Quality assessment
            quality_assessment = self.quality_service.assess_quality(schema_profile, df)
            
            # Create profiling metadata
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            metadata = ProfilingMetadata(
                profiling_strategy=profiling_strategy,
                sample_size=len(df) if profiling_strategy == "full" else None,
                sample_percentage=100.0 if profiling_strategy == "full" else None,
                execution_time_seconds=execution_time,
                memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
                include_patterns=True,
                include_statistical_analysis=True,
                include_quality_assessment=True
            )
            
            # Complete profiling
            profile.complete_profiling(
                schema_profile=schema_profile,
                quality_assessment=quality_assessment,
                metadata=metadata
            )
            
        except Exception as e:
            # Handle profiling failure
            profile.fail_profiling(str(e))
            raise
        
        return profile
    
    def execute_sample(self, path: str, sample_size: int = 10000) -> DataProfile:
        """Profile a sample of the dataset for performance."""
        return self._execute_with_sampling(path, sample_size=sample_size)
    
    def execute_percentage_sample(self, path: str, sample_percentage: float = 10.0) -> DataProfile:
        """Profile a percentage sample of the dataset."""
        return self._execute_with_sampling(path, sample_percentage=sample_percentage)
    
    def _execute_with_sampling(self, path: str, sample_size: int = None, sample_percentage: float = None) -> DataProfile:
        """Execute profiling with sampling."""
        start_time = datetime.utcnow()
        
        # Initialize profile
        profile = DataProfile(
            profile_id=ProfileId(value=uuid4()),
            dataset_id=DatasetId(value=uuid4()),
            status=ProfilingStatus.PENDING,
            source_type="file",
            source_connection={"path": path},
            created_at=start_time
        )
        
        try:
            profile.start_profiling()
            
            # Load data
            adapter = get_file_adapter(path)
            df = adapter.load(path)
            
            if df.empty:
                raise ValueError("Dataset is empty")
            
            # Apply sampling
            original_size = len(df)
            if sample_size and sample_size < original_size:
                df = df.sample(n=sample_size, random_state=42)
                actual_sample_size = sample_size
                actual_sample_percentage = (sample_size / original_size) * 100
            elif sample_percentage and sample_percentage < 100:
                sample_count = int((sample_percentage / 100) * original_size)
                df = df.sample(n=sample_count, random_state=42)
                actual_sample_size = sample_count
                actual_sample_percentage = sample_percentage
            else:
                actual_sample_size = original_size
                actual_sample_percentage = 100.0
            
            # Perform analysis on sampled data
            schema_profile = self.schema_service.infer(df)
            patterns = self.pattern_service.discover(df)
            statistical_summaries = self.stats_service.analyze(df)
            
            # Attach patterns and statistical summaries
            for col_profile in schema_profile.columns:
                if col_profile.column_name in patterns:
                    col_profile.patterns.extend(patterns[col_profile.column_name])
                if col_profile.column_name in statistical_summaries and col_profile.statistical_summary is None:
                    col_profile.statistical_summary = statistical_summaries[col_profile.column_name]
            
            # Quality assessment
            quality_assessment = self.quality_service.assess_quality(schema_profile, df)
            
            # Create profiling metadata
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            metadata = ProfilingMetadata(
                profiling_strategy="sample",
                sample_size=actual_sample_size,
                sample_percentage=actual_sample_percentage,
                execution_time_seconds=execution_time,
                memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
                include_patterns=True,
                include_statistical_analysis=True,
                include_quality_assessment=True
            )
            
            # Update schema profile to reflect original dataset size
            schema_profile.total_rows = original_size
            
            # Complete profiling
            profile.complete_profiling(
                schema_profile=schema_profile,
                quality_assessment=quality_assessment,
                metadata=metadata
            )
            
        except Exception as e:
            profile.fail_profiling(str(e))
            raise
        
        return profile
    
    def analyze_relationships(self, path: str) -> dict:
        """Analyze data relationships and patterns across columns."""
        adapter = get_file_adapter(path)
        df = adapter.load(path)
        
        return self.pattern_service.analyze_data_relationships(df)
    
    def get_profiling_summary(self, profile: DataProfile) -> dict:
        """Get a summary of the profiling results."""
        if not profile.is_completed():
            return {"status": "incomplete", "error": profile.error_message}
        
        summary = {
            "profile_id": str(profile.profile_id.value),
            "dataset_id": str(profile.dataset_id.value),
            "status": profile.status.value,
            "execution_time_seconds": profile.profiling_metadata.execution_time_seconds,
            "memory_usage_mb": profile.profiling_metadata.memory_usage_mb,
            "total_rows": profile.schema_profile.total_rows,
            "total_columns": profile.schema_profile.total_columns,
            "overall_quality_score": profile.quality_assessment.overall_score,
            "quality_issues": {
                "critical": profile.quality_assessment.critical_issues,
                "high": profile.quality_assessment.high_issues,
                "medium": profile.quality_assessment.medium_issues,
                "low": profile.quality_assessment.low_issues
            },
            "data_types": {},
            "patterns_discovered": 0,
            "primary_keys": profile.schema_profile.primary_keys,
            "recommendations": profile.quality_assessment.recommendations
        }
        
        # Analyze data types
        for column in profile.schema_profile.columns:
            data_type = column.data_type.value
            if data_type not in summary["data_types"]:
                summary["data_types"][data_type] = 0
            summary["data_types"][data_type] += 1
            summary["patterns_discovered"] += len(column.patterns)
        
        return summary