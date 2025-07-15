"""Data profiling package for automated data analysis and quality assessment.

This package provides comprehensive data profiling capabilities including:
- Schema discovery and inference
- Statistical analysis and distribution fitting
- Pattern recognition and format validation
- Data quality assessment
- Performance optimization with intelligent sampling
- Support for multiple data sources (files, databases, cloud storage, streaming)
"""

from .application.services.profiling_engine import ProfilingEngine, ProfilingConfig
from .application.services.schema_analysis_service import SchemaAnalysisService
from .application.services.statistical_profiling_service import StatisticalProfilingService
from .application.services.pattern_discovery_service import PatternDiscoveryService
from .application.services.quality_assessment_service import QualityAssessmentService
from .application.services.performance_optimizer import PerformanceOptimizer

from .domain.entities.data_profile import (
    DataProfile, SchemaProfile, ColumnProfile, QualityAssessment,
    ProfilingStatus, DataType, CardinalityLevel, PatternType, QualityIssueType,
    ProfileId, DatasetId, ProfilingMetadata, ValueDistribution, StatisticalSummary,
    Pattern, QualityIssue
)

from .infrastructure.adapters.database_adapter import (
    DatabaseAdapter, PostgreSQLAdapter, MySQLAdapter, SQLiteAdapter,
    get_database_adapter, DatabaseProfiler
)

from .infrastructure.adapters.file_adapter import FileAdapter, MultiFileAdapter
from .infrastructure.adapters.cloud_storage_adapter import (
    CloudStorageAdapter, S3Adapter, AzureBlobAdapter, GCSAdapter,
    get_cloud_storage_adapter, CloudDataProfiler
)

from .infrastructure.adapters.streaming_adapter import (
    StreamingAdapter, KafkaAdapter, KinesisAdapter, EventHubAdapter,
    get_streaming_adapter, StreamingProfiler
)

__version__ = "0.1.0"

__all__ = [
    # Core profiling engine and configuration
    'ProfilingEngine',
    'ProfilingConfig',
    
    # Analysis services
    'SchemaAnalysisService',
    'StatisticalProfilingService',
    'PatternDiscoveryService',
    'QualityAssessmentService',
    'PerformanceOptimizer',
    
    # Domain entities and value objects
    'DataProfile',
    'SchemaProfile',
    'ColumnProfile',
    'QualityAssessment',
    'ProfilingStatus',
    'DataType',
    'CardinalityLevel',
    'PatternType',
    'QualityIssueType',
    'ProfileId',
    'DatasetId',
    'ProfilingMetadata',
    'ValueDistribution',
    'StatisticalSummary',
    'Pattern',
    'QualityIssue',
    
    # Database adapters
    'DatabaseAdapter',
    'PostgreSQLAdapter',
    'MySQLAdapter',
    'SQLiteAdapter',
    'get_database_adapter',
    'DatabaseProfiler',
    
    # File adapters
    'FileAdapter',
    'MultiFileAdapter',
    
    # Cloud storage adapters
    'CloudStorageAdapter',
    'S3Adapter',
    'AzureBlobAdapter',
    'GCSAdapter',
    'get_cloud_storage_adapter',
    'CloudDataProfiler',
    
    # Streaming adapters
    'StreamingAdapter',
    'KafkaAdapter',
    'KinesisAdapter',
    'EventHubAdapter',
    'get_streaming_adapter',
    'StreamingProfiler'
]