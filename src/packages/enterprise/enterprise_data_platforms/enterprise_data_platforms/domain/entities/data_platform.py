"""
Data Platform domain entities for enterprise integrations.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class DataPlatformType(str, Enum):
    """Types of data platforms."""
    SNOWFLAKE = "snowflake"
    BIGQUERY = "bigquery"
    DATABRICKS = "databricks"
    REDSHIFT = "redshift"
    SYNAPSE = "synapse"
    ATHENA = "athena"
    CUSTOM = "custom"


class ConnectionStatus(str, Enum):
    """Data platform connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"
    TESTING = "testing"


class DataFormat(str, Enum):
    """Supported data formats."""
    JSON = "json"
    PARQUET = "parquet"
    CSV = "csv"
    AVRO = "avro"
    ORC = "orc"
    DELTA = "delta"
    ICEBERG = "iceberg"


class CompressionType(str, Enum):
    """Compression types for data."""
    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"
    ZSTD = "zstd"
    BROTLI = "brotli"


class DataPlatformConnection(BaseModel):
    """
    Data platform connection configuration.
    
    Represents connection details and credentials for
    various data platforms like Snowflake, BigQuery, and Databricks.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Connection identifier")
    
    # Connection identification
    name: str = Field(..., description="Connection name")
    platform_type: DataPlatformType = Field(..., description="Data platform type")
    tenant_id: UUID = Field(..., description="Owning tenant")
    
    # Connection configuration
    host: Optional[str] = Field(None, description="Platform host/endpoint")
    port: Optional[int] = Field(None, description="Connection port")
    database: Optional[str] = Field(None, description="Database name")
    schema: Optional[str] = Field(None, description="Default schema")
    warehouse: Optional[str] = Field(None, description="Compute warehouse (Snowflake)")
    
    # Authentication
    username: Optional[str] = Field(None, description="Username")
    password: Optional[str] = Field(None, description="Password (encrypted)")
    private_key: Optional[str] = Field(None, description="Private key for key-pair auth")
    private_key_passphrase: Optional[str] = Field(None, description="Private key passphrase")
    oauth_token: Optional[str] = Field(None, description="OAuth token")
    service_account_key: Optional[str] = Field(None, description="Service account key (GCP)")
    
    # Connection options
    connection_timeout: int = Field(default=30, description="Connection timeout in seconds")
    read_timeout: int = Field(default=300, description="Read timeout in seconds")
    retry_attempts: int = Field(default=3, description="Connection retry attempts")
    pool_size: int = Field(default=5, description="Connection pool size")
    
    # SSL/TLS configuration
    ssl_enabled: bool = Field(default=True, description="Enable SSL/TLS")
    ssl_verify: bool = Field(default=True, description="Verify SSL certificates")
    ssl_cert_path: Optional[str] = Field(None, description="SSL certificate path")
    
    # Additional properties
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional connection properties")
    
    # Status and monitoring
    status: ConnectionStatus = Field(default=ConnectionStatus.DISCONNECTED)
    last_connected: Optional[datetime] = Field(None, description="Last successful connection")
    last_error: Optional[str] = Field(None, description="Last connection error")
    
    # Metadata
    tags: Dict[str, str] = Field(default_factory=dict, description="Connection tags")
    description: Optional[str] = Field(None, description="Connection description")
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self.status == ConnectionStatus.CONNECTED
    
    def get_connection_string(self) -> str:
        """Get connection string for the platform."""
        if self.platform_type == DataPlatformType.SNOWFLAKE:
            return f"snowflake://{self.username}@{self.host}/{self.database}/{self.schema}?warehouse={self.warehouse}"
        elif self.platform_type == DataPlatformType.BIGQUERY:
            return f"bigquery://{self.database}"
        elif self.platform_type == DataPlatformType.DATABRICKS:
            return f"databricks://token:{self.oauth_token}@{self.host}/{self.database}"
        else:
            return f"{self.platform_type.value}://{self.host}:{self.port}/{self.database}"
    
    def update_status(self, status: ConnectionStatus, error: Optional[str] = None) -> None:
        """Update connection status."""
        self.status = status
        if status == ConnectionStatus.CONNECTED:
            self.last_connected = datetime.utcnow()
            self.last_error = None
        elif error:
            self.last_error = error
        self.updated_at = datetime.utcnow()
    
    def add_tag(self, key: str, value: str) -> None:
        """Add tag to connection."""
        self.tags[key] = value
        self.updated_at = datetime.utcnow()


class DataSource(BaseModel):
    """
    Data source configuration for streaming or batch processing.
    
    Represents various data sources including tables, views,
    streams, and files from different platforms.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Data source identifier")
    
    # Source identification
    name: str = Field(..., description="Data source name")
    connection_id: UUID = Field(..., description="Associated connection ID")
    tenant_id: UUID = Field(..., description="Owning tenant")
    
    # Source configuration
    source_type: str = Field(..., description="Source type (table, view, stream, file)")
    source_path: str = Field(..., description="Source path or identifier")
    
    # Schema and format
    schema_definition: Optional[Dict[str, Any]] = Field(None, description="Data schema definition")
    data_format: DataFormat = Field(default=DataFormat.JSON, description="Data format")
    compression: CompressionType = Field(default=CompressionType.NONE, description="Compression type")
    
    # Partitioning
    partition_columns: List[str] = Field(default_factory=list, description="Partition columns")
    partition_strategy: Optional[str] = Field(None, description="Partitioning strategy")
    
    # Data quality
    quality_checks: List[Dict[str, Any]] = Field(default_factory=list, description="Data quality checks")
    validation_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Validation rules")
    
    # Refresh configuration
    refresh_enabled: bool = Field(default=False, description="Enable data refresh")
    refresh_schedule: Optional[str] = Field(None, description="Refresh schedule (cron format)")
    refresh_interval_minutes: Optional[int] = Field(None, description="Refresh interval in minutes")
    incremental_column: Optional[str] = Field(None, description="Column for incremental refresh")
    
    # Performance settings
    batch_size: int = Field(default=1000, description="Batch size for processing")
    parallel_workers: int = Field(default=1, description="Number of parallel workers")
    cache_enabled: bool = Field(default=False, description="Enable result caching")
    cache_ttl_minutes: int = Field(default=60, description="Cache TTL in minutes")
    
    # Monitoring
    last_refresh: Optional[datetime] = Field(None, description="Last refresh timestamp")
    record_count: Optional[int] = Field(None, description="Number of records")
    data_size_bytes: Optional[int] = Field(None, description="Data size in bytes")
    
    # Metadata
    tags: Dict[str, str] = Field(default_factory=dict)
    description: Optional[str] = Field(None, description="Data source description")
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def is_streaming(self) -> bool:
        """Check if data source is streaming."""
        return self.source_type in ["stream", "topic", "queue"]
    
    def needs_refresh(self) -> bool:
        """Check if data source needs refresh."""
        if not self.refresh_enabled or not self.refresh_interval_minutes:
            return False
        
        if not self.last_refresh:
            return True
        
        next_refresh = self.last_refresh + timedelta(minutes=self.refresh_interval_minutes)
        return datetime.utcnow() >= next_refresh
    
    def update_stats(self, record_count: int, data_size_bytes: int) -> None:
        """Update data source statistics."""
        self.record_count = record_count
        self.data_size_bytes = data_size_bytes
        self.last_refresh = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def add_quality_check(self, check_type: str, config: Dict[str, Any]) -> None:
        """Add data quality check."""
        check = {
            "id": str(uuid4()),
            "type": check_type,
            "config": config,
            "created_at": datetime.utcnow().isoformat()
        }
        self.quality_checks.append(check)
        self.updated_at = datetime.utcnow()


class DataPipeline(BaseModel):
    """
    Data pipeline configuration for ETL/ELT processes.
    
    Represents data transformation and movement pipelines
    between different platforms and systems.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Pipeline identifier")
    
    # Pipeline identification
    name: str = Field(..., description="Pipeline name")
    tenant_id: UUID = Field(..., description="Owning tenant")
    
    # Pipeline configuration
    source_connection_id: UUID = Field(..., description="Source connection ID")
    target_connection_id: UUID = Field(..., description="Target connection ID")
    
    # Data sources and targets
    sources: List[UUID] = Field(default_factory=list, description="Source data source IDs")
    targets: List[Dict[str, Any]] = Field(default_factory=list, description="Target configurations")
    
    # Transformation
    transformations: List[Dict[str, Any]] = Field(default_factory=list, description="Data transformations")
    sql_query: Optional[str] = Field(None, description="SQL query for transformation")
    
    # Scheduling
    schedule_enabled: bool = Field(default=False, description="Enable scheduling")
    schedule_cron: Optional[str] = Field(None, description="Cron schedule expression")
    schedule_timezone: str = Field(default="UTC", description="Schedule timezone")
    
    # Execution settings
    execution_mode: str = Field(default="batch", description="Execution mode (batch/streaming)")
    batch_size: int = Field(default=10000, description="Batch size for processing")
    max_parallelism: int = Field(default=4, description="Maximum parallel workers")
    timeout_minutes: int = Field(default=60, description="Pipeline timeout in minutes")
    
    # Error handling
    retry_enabled: bool = Field(default=True, description="Enable retry on failure")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_backoff_seconds: int = Field(default=60, description="Retry backoff in seconds")
    
    # Monitoring and alerting
    monitoring_enabled: bool = Field(default=True, description="Enable monitoring")
    alert_on_failure: bool = Field(default=True, description="Alert on pipeline failure")
    alert_recipients: List[str] = Field(default_factory=list, description="Alert recipients")
    
    # Status
    status: str = Field(default="inactive", description="Pipeline status")
    last_run: Optional[datetime] = Field(None, description="Last execution time")
    next_run: Optional[datetime] = Field(None, description="Next scheduled run")
    last_success: Optional[datetime] = Field(None, description="Last successful run")
    last_error: Optional[str] = Field(None, description="Last error message")
    
    # Statistics
    total_runs: int = Field(default=0, description="Total number of runs")
    successful_runs: int = Field(default=0, description="Successful runs count")
    failed_runs: int = Field(default=0, description="Failed runs count")
    avg_duration_seconds: Optional[float] = Field(None, description="Average run duration")
    
    # Metadata
    tags: Dict[str, str] = Field(default_factory=dict)
    description: Optional[str] = Field(None, description="Pipeline description")
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def is_active(self) -> bool:
        """Check if pipeline is active."""
        return self.status == "active"
    
    def is_scheduled(self) -> bool:
        """Check if pipeline is scheduled."""
        return self.schedule_enabled and self.schedule_cron is not None
    
    def get_success_rate(self) -> float:
        """Get pipeline success rate."""
        if self.total_runs == 0:
            return 0.0
        return (self.successful_runs / self.total_runs) * 100
    
    def record_run_start(self) -> None:
        """Record pipeline run start."""
        self.status = "running"
        self.last_run = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def record_run_success(self, duration_seconds: float) -> None:
        """Record successful pipeline run."""
        self.status = "active"
        self.last_success = datetime.utcnow()
        self.last_error = None
        self.total_runs += 1
        self.successful_runs += 1
        
        # Update average duration
        if self.avg_duration_seconds is None:
            self.avg_duration_seconds = duration_seconds
        else:
            self.avg_duration_seconds = (
                (self.avg_duration_seconds * (self.total_runs - 1) + duration_seconds) / self.total_runs
            )
        
        self.updated_at = datetime.utcnow()
    
    def record_run_failure(self, error_message: str) -> None:
        """Record failed pipeline run."""
        self.status = "failed"
        self.last_error = error_message
        self.total_runs += 1
        self.failed_runs += 1
        self.updated_at = datetime.utcnow()
    
    def add_transformation(self, transformation_type: str, config: Dict[str, Any]) -> None:
        """Add data transformation step."""
        transformation = {
            "id": str(uuid4()),
            "type": transformation_type,
            "config": config,
            "order": len(self.transformations),
            "created_at": datetime.utcnow().isoformat()
        }
        self.transformations.append(transformation)
        self.updated_at = datetime.utcnow()


class DataQualityResult(BaseModel):
    """
    Data quality check result.
    
    Represents the outcome of data quality validations
    and checks performed on datasets.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Result identifier")
    
    # Association
    data_source_id: UUID = Field(..., description="Data source ID")
    pipeline_id: Optional[UUID] = Field(None, description="Pipeline ID if applicable")
    tenant_id: UUID = Field(..., description="Owning tenant")
    
    # Check details
    check_name: str = Field(..., description="Quality check name")
    check_type: str = Field(..., description="Type of quality check")
    check_config: Dict[str, Any] = Field(..., description="Check configuration")
    
    # Results
    passed: bool = Field(..., description="Whether check passed")
    score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Quality score (0-1)")
    
    # Metrics
    total_records: int = Field(..., description="Total records checked")
    passed_records: int = Field(..., description="Records that passed")
    failed_records: int = Field(..., description="Records that failed")
    
    # Details
    failure_reasons: List[str] = Field(default_factory=list, description="Reasons for failures")
    sample_failures: List[Dict[str, Any]] = Field(default_factory=list, description="Sample failure records")
    
    # Thresholds
    threshold_config: Dict[str, Any] = Field(default_factory=dict, description="Quality thresholds")
    
    # Execution details
    execution_time_seconds: float = Field(..., description="Execution time in seconds")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    
    # Metadata
    executed_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def get_pass_rate(self) -> float:
        """Get pass rate percentage."""
        if self.total_records == 0:
            return 0.0
        return (self.passed_records / self.total_records) * 100
    
    def get_failure_rate(self) -> float:
        """Get failure rate percentage."""
        return 100.0 - self.get_pass_rate()
    
    def is_within_threshold(self, threshold_field: str, threshold_value: float) -> bool:
        """Check if result is within specified threshold."""
        current_value = getattr(self, threshold_field, None)
        if current_value is None:
            return False
        
        if threshold_field in ["score", "pass_rate"]:
            return current_value >= threshold_value
        else:
            return current_value <= threshold_value