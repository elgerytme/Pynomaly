"""Profiling metadata value objects for data profiling operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4


class ProfilingStrategy(str, Enum):
    """Data profiling strategies."""
    FULL = "full"
    SAMPLE = "sample"
    INCREMENTAL = "incremental"
    STREAMING = "streaming"
    LIGHTWEIGHT = "lightweight"
    COMPREHENSIVE = "comprehensive"


class ProfilingStatus(str, Enum):
    """Profiling operation status."""
    PENDING = "pending"
    INITIALIZING = "initializing"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    RETRYING = "retrying"


class ExecutionPhase(str, Enum):
    """Profiling execution phases."""
    INITIALIZATION = "initialization"
    DATA_LOADING = "data_loading"
    SCHEMA_ANALYSIS = "schema_analysis"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    PATTERN_DISCOVERY = "pattern_discovery"
    QUALITY_ASSESSMENT = "quality_assessment"
    RELATIONSHIP_ANALYSIS = "relationship_analysis"
    FINALIZATION = "finalization"


class ResourceType(str, Enum):
    """Resource types for monitoring."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    DATABASE_CONNECTIONS = "database_connections"


@dataclass(frozen=True)
class SamplingConfiguration:
    """Configuration for data sampling during profiling."""
    strategy: str  # "random", "systematic", "stratified", "cluster"
    sample_size: Optional[int] = None
    sample_percentage: Optional[float] = None
    seed: Optional[int] = None
    stratify_column: Optional[str] = None
    preserve_distribution: bool = True
    minimum_sample_size: int = 1000
    maximum_sample_size: int = 100000
    adaptive_sampling: bool = True
    confidence_level: float = 0.95
    margin_of_error: float = 0.05
    
    def __post_init__(self):
        if self.sample_percentage is not None:
            if not 0.0 < self.sample_percentage <= 100.0:
                raise ValueError("Sample percentage must be between 0 and 100")
        
        if self.sample_size is not None:
            if self.sample_size <= 0:
                raise ValueError("Sample size must be positive")
        
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("Confidence level must be between 0 and 1")
        
        if not 0.0 < self.margin_of_error < 1.0:
            raise ValueError("Margin of error must be between 0 and 1")
    
    @property
    def requires_stratification(self) -> bool:
        """Check if stratified sampling is required."""
        return self.strategy == "stratified" and self.stratify_column is not None
    
    @classmethod
    def create_automatic(cls, total_rows: int) -> SamplingConfiguration:
        """Create automatic sampling configuration based on data size."""
        if total_rows <= 10000:
            return cls(
                strategy="random",
                sample_percentage=100.0,
                adaptive_sampling=False
            )
        elif total_rows <= 100000:
            return cls(
                strategy="random",
                sample_size=min(50000, total_rows),
                adaptive_sampling=True
            )
        else:
            return cls(
                strategy="systematic",
                sample_size=100000,
                adaptive_sampling=True,
                preserve_distribution=True
            )


@dataclass(frozen=True)
class ResourceMetrics:
    """Resource usage metrics during profiling."""
    resource_type: ResourceType
    current_usage: float
    peak_usage: float
    average_usage: float
    unit: str  # "MB", "percentage", "count", etc.
    measurement_window_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def usage_efficiency(self) -> float:
        """Calculate usage efficiency ratio."""
        if self.peak_usage == 0:
            return 1.0
        return self.average_usage / self.peak_usage
    
    @classmethod
    def create_memory_metrics(
        cls, 
        current_mb: float, 
        peak_mb: float, 
        average_mb: float,
        window_seconds: float = 60.0
    ) -> ResourceMetrics:
        """Create memory usage metrics."""
        return cls(
            resource_type=ResourceType.MEMORY,
            current_usage=current_mb,
            peak_usage=peak_mb,
            average_usage=average_mb,
            unit="MB",
            measurement_window_seconds=window_seconds
        )
    
    @classmethod
    def create_cpu_metrics(
        cls, 
        current_percent: float, 
        peak_percent: float, 
        average_percent: float,
        window_seconds: float = 60.0
    ) -> ResourceMetrics:
        """Create CPU usage metrics."""
        return cls(
            resource_type=ResourceType.CPU,
            current_usage=current_percent,
            peak_usage=peak_percent,
            average_usage=average_percent,
            unit="percentage",
            measurement_window_seconds=window_seconds
        )


@dataclass(frozen=True)
class ExecutionTimeline:
    """Timeline of profiling execution phases."""
    phase: ExecutionPhase
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    status: ProfilingStatus = ProfilingStatus.IN_PROGRESS
    progress_percentage: float = 0.0
    items_processed: int = 0
    total_items: Optional[int] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.end_time and not self.duration_seconds:
            duration = (self.end_time - self.start_time).total_seconds()
            object.__setattr__(self, 'duration_seconds', duration)
    
    @property
    def is_completed(self) -> bool:
        """Check if phase is completed."""
        return self.status == ProfilingStatus.COMPLETED
    
    @property
    def is_running(self) -> bool:
        """Check if phase is currently running."""
        return self.status == ProfilingStatus.IN_PROGRESS
    
    @property
    def has_failed(self) -> bool:
        """Check if phase has failed."""
        return self.status == ProfilingStatus.FAILED
    
    @property
    def throughput_per_second(self) -> Optional[float]:
        """Calculate processing throughput."""
        if self.duration_seconds and self.duration_seconds > 0:
            return self.items_processed / self.duration_seconds
        return None
    
    def complete(self, items_processed: int = None) -> ExecutionTimeline:
        """Mark phase as completed."""
        return dataclass.replace(
            self,
            end_time=datetime.now(),
            status=ProfilingStatus.COMPLETED,
            progress_percentage=100.0,
            items_processed=items_processed or self.items_processed
        )
    
    def fail(self, error_message: str) -> ExecutionTimeline:
        """Mark phase as failed."""
        return dataclass.replace(
            self,
            end_time=datetime.now(),
            status=ProfilingStatus.FAILED,
            error_message=error_message
        )


@dataclass(frozen=True)
class ProfilingConfiguration:
    """Comprehensive profiling configuration."""
    config_id: str
    strategy: ProfilingStrategy
    sampling: Optional[SamplingConfiguration] = None
    
    # Analysis toggles
    enable_schema_analysis: bool = True
    enable_statistical_analysis: bool = True
    enable_pattern_discovery: bool = True
    enable_quality_assessment: bool = True
    enable_relationship_analysis: bool = True
    enable_data_lineage: bool = False
    enable_semantic_analysis: bool = True
    
    # Performance settings
    parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 10000
    memory_limit_mb: int = 2048
    timeout_minutes: int = 60
    
    # Output settings
    include_examples: bool = True
    max_examples_per_pattern: int = 10
    include_histograms: bool = True
    histogram_bins: int = 50
    
    # Quality settings
    quality_rules_enabled: bool = True
    quality_threshold: float = 0.8
    alert_on_quality_issues: bool = True
    
    # Advanced settings
    machine_learning_features: bool = False
    clustering_enabled: bool = False
    anomaly_detection: bool = True
    text_analysis_enabled: bool = True
    
    # Security and privacy
    pii_detection_enabled: bool = True
    data_masking_enabled: bool = False
    audit_logging_enabled: bool = True
    
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    
    @property
    def analysis_modules(self) -> List[str]:
        """Get list of enabled analysis modules."""
        modules = []
        if self.enable_schema_analysis:
            modules.append("schema_analysis")
        if self.enable_statistical_analysis:
            modules.append("statistical_analysis")
        if self.enable_pattern_discovery:
            modules.append("pattern_discovery")
        if self.enable_quality_assessment:
            modules.append("quality_assessment")
        if self.enable_relationship_analysis:
            modules.append("relationship_analysis")
        if self.enable_semantic_analysis:
            modules.append("semantic_analysis")
        return modules
    
    @property
    def estimated_complexity_score(self) -> float:
        """Estimate configuration complexity (0-1 scale)."""
        complexity_factors = [
            self.enable_schema_analysis,
            self.enable_statistical_analysis,
            self.enable_pattern_discovery,
            self.enable_quality_assessment,
            self.enable_relationship_analysis,
            self.enable_semantic_analysis,
            self.machine_learning_features,
            self.clustering_enabled,
            self.pii_detection_enabled
        ]
        return sum(complexity_factors) / len(complexity_factors)
    
    @classmethod
    def create_lightweight(cls, config_id: str) -> ProfilingConfiguration:
        """Create lightweight profiling configuration."""
        return cls(
            config_id=config_id,
            strategy=ProfilingStrategy.LIGHTWEIGHT,
            enable_statistical_analysis=False,
            enable_pattern_discovery=False,
            enable_relationship_analysis=False,
            enable_semantic_analysis=False,
            machine_learning_features=False,
            include_histograms=False,
            max_workers=2,
            timeout_minutes=15
        )
    
    @classmethod
    def create_comprehensive(cls, config_id: str) -> ProfilingConfiguration:
        """Create comprehensive profiling configuration."""
        return cls(
            config_id=config_id,
            strategy=ProfilingStrategy.COMPREHENSIVE,
            enable_data_lineage=True,
            machine_learning_features=True,
            clustering_enabled=True,
            text_analysis_enabled=True,
            max_workers=8,
            memory_limit_mb=4096,
            timeout_minutes=180
        )


@dataclass(frozen=True)
class ProfilingResult:
    """Results and metadata from a profiling operation."""
    result_id: str
    profile_id: str
    dataset_id: str
    configuration: ProfilingConfiguration
    
    # Execution details
    status: ProfilingStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_seconds: Optional[float] = None
    
    # Timeline and progress
    execution_timeline: List[ExecutionTimeline] = field(default_factory=list)
    current_phase: Optional[ExecutionPhase] = None
    overall_progress_percentage: float = 0.0
    
    # Resource usage
    resource_metrics: List[ResourceMetrics] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    average_cpu_percentage: float = 0.0
    
    # Data processing metrics
    total_rows_processed: int = 0
    total_columns_analyzed: int = 0
    sampling_applied: bool = False
    sample_size: Optional[int] = None
    
    # Results summary
    schema_elements_discovered: int = 0
    patterns_discovered: int = 0
    quality_issues_found: int = 0
    relationships_identified: int = 0
    
    # Error handling
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    error_details: Optional[str] = None
    
    # Output artifacts
    report_paths: List[str] = field(default_factory=list)
    export_formats: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if self.end_time and self.start_time and not self.total_duration_seconds:
            duration = (self.end_time - self.start_time).total_seconds()
            object.__setattr__(self, 'total_duration_seconds', duration)
    
    @property
    def is_completed(self) -> bool:
        """Check if profiling is completed."""
        return self.status == ProfilingStatus.COMPLETED
    
    @property
    def has_errors(self) -> bool:
        """Check if profiling encountered errors."""
        return len(self.errors) > 0 or self.status == ProfilingStatus.FAILED
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate based on phases."""
        if not self.execution_timeline:
            return 0.0
        
        completed_phases = sum(1 for phase in self.execution_timeline if phase.is_completed)
        return completed_phases / len(self.execution_timeline)
    
    @property
    def processing_rate_rows_per_second(self) -> Optional[float]:
        """Calculate row processing rate."""
        if self.total_duration_seconds and self.total_duration_seconds > 0:
            return self.total_rows_processed / self.total_duration_seconds
        return None
    
    def get_phase_duration(self, phase: ExecutionPhase) -> Optional[float]:
        """Get duration for specific phase."""
        for timeline in self.execution_timeline:
            if timeline.phase == phase:
                return timeline.duration_seconds
        return None
    
    def get_resource_peak_usage(self, resource_type: ResourceType) -> Optional[float]:
        """Get peak usage for specific resource type."""
        matching_metrics = [m for m in self.resource_metrics if m.resource_type == resource_type]
        if matching_metrics:
            return max(m.peak_usage for m in matching_metrics)
        return None
    
    def add_warning(self, message: str) -> ProfilingResult:
        """Add warning message."""
        new_warnings = list(self.warnings) + [message]
        return dataclass.replace(self, warnings=new_warnings)
    
    def add_error(self, message: str) -> ProfilingResult:
        """Add error message."""
        new_errors = list(self.errors) + [message]
        return dataclass.replace(self, errors=new_errors)
    
    @classmethod
    def create_initial(
        cls,
        result_id: str,
        profile_id: str,
        dataset_id: str,
        configuration: ProfilingConfiguration
    ) -> ProfilingResult:
        """Create initial profiling result."""
        return cls(
            result_id=result_id,
            profile_id=profile_id,
            dataset_id=dataset_id,
            configuration=configuration,
            status=ProfilingStatus.PENDING,
            start_time=datetime.now()
        )