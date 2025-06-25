"""Configuration management for distributed processing systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field


class ClusterMode(str, Enum):
    """Cluster operation modes."""

    STANDALONE = "standalone"
    DISTRIBUTED = "distributed"
    HYBRID = "hybrid"


class CommunicationProtocol(str, Enum):
    """Communication protocols for distributed systems."""

    HTTP = "http"
    GRPC = "grpc"
    REDIS = "redis"
    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"


class PartitionStrategy(str, Enum):
    """Data partitioning strategies."""

    ROUND_ROBIN = "round_robin"
    HASH_BASED = "hash_based"
    SIZE_BASED = "size_based"
    FEATURE_BASED = "feature_based"
    ADAPTIVE = "adaptive"


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies."""

    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED = "weighted"
    ADAPTIVE = "adaptive"
    CAPABILITY_BASED = "capability_based"


class AggregationStrategy(str, Enum):
    """Result aggregation strategies."""

    SIMPLE_MERGE = "simple_merge"
    WEIGHTED_AVERAGE = "weighted_average"
    ENSEMBLE_VOTING = "ensemble_voting"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    HIERARCHICAL = "hierarchical"


class NetworkConfig(BaseModel):
    """Network configuration for distributed systems."""

    protocol: CommunicationProtocol = Field(
        default=CommunicationProtocol.HTTP, description="Communication protocol"
    )
    host: str = Field(default="localhost", description="Host address")
    port: int = Field(default=8080, ge=1, le=65535, description="Port number")

    # Connection settings
    connection_timeout: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Connection timeout in seconds"
    )
    read_timeout: float = Field(
        default=60.0, ge=1.0, le=600.0, description="Read timeout in seconds"
    )
    max_connections: int = Field(
        default=100, ge=1, le=10000, description="Maximum concurrent connections"
    )

    # Security settings
    enable_tls: bool = Field(default=False, description="Enable TLS/SSL")
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")
    cert_file: str | None = Field(default=None, description="SSL certificate file path")
    key_file: str | None = Field(default=None, description="SSL private key file path")
    ca_file: str | None = Field(default=None, description="CA certificate file path")

    # Authentication
    enable_auth: bool = Field(default=False, description="Enable authentication")
    auth_token: str | None = Field(default=None, description="Authentication token")
    api_key: str | None = Field(default=None, description="API key")


class WorkerConfig(BaseModel):
    """Configuration for worker nodes."""

    # Basic settings
    worker_id: str | None = Field(default=None, description="Unique worker identifier")
    max_concurrent_tasks: int = Field(
        default=4, ge=1, le=100, description="Maximum concurrent tasks"
    )
    task_timeout: float = Field(
        default=300.0, ge=10.0, le=3600.0, description="Task timeout in seconds"
    )

    # Resource limits
    max_memory_mb: int = Field(
        default=4096, ge=512, le=32768, description="Maximum memory usage in MB"
    )
    max_cpu_cores: int = Field(
        default=4, ge=1, le=64, description="Maximum CPU cores to use"
    )
    max_gpu_memory_mb: int = Field(
        default=0, ge=0, le=32768, description="Maximum GPU memory in MB"
    )

    # Capabilities
    supported_algorithms: set[str] = Field(
        default_factory=set, description="Supported algorithms"
    )
    hardware_acceleration: set[str] = Field(
        default_factory=set, description="Available hardware acceleration"
    )
    data_formats: set[str] = Field(
        default_factory=set, description="Supported data formats"
    )

    # Performance settings
    batch_size: int = Field(
        default=1000, ge=1, le=100000, description="Default batch size"
    )
    prefetch_batches: int = Field(
        default=2, ge=1, le=10, description="Number of batches to prefetch"
    )
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_size_mb: int = Field(
        default=512, ge=64, le=8192, description="Cache size in MB"
    )

    # Health monitoring
    heartbeat_interval: float = Field(
        default=30.0, ge=5.0, le=300.0, description="Heartbeat interval in seconds"
    )
    health_check_interval: float = Field(
        default=60.0, ge=10.0, le=600.0, description="Health check interval"
    )
    metrics_reporting_interval: float = Field(
        default=60.0, ge=10.0, le=600.0, description="Metrics reporting interval"
    )


class FaultToleranceConfig(BaseModel):
    """Fault tolerance configuration."""

    # Retry settings
    max_retries: int = Field(
        default=3, ge=0, le=10, description="Maximum retry attempts"
    )
    retry_delay: float = Field(
        default=1.0, ge=0.1, le=60.0, description="Initial retry delay in seconds"
    )
    retry_exponential_backoff: bool = Field(
        default=True, description="Use exponential backoff for retries"
    )
    max_retry_delay: float = Field(
        default=60.0, ge=1.0, le=600.0, description="Maximum retry delay"
    )

    # Circuit breaker
    enable_circuit_breaker: bool = Field(
        default=True, description="Enable circuit breaker pattern"
    )
    failure_threshold: int = Field(
        default=5, ge=1, le=100, description="Failure threshold for circuit breaker"
    )
    recovery_timeout: float = Field(
        default=60.0, ge=10.0, le=600.0, description="Circuit breaker recovery timeout"
    )

    # Failover
    enable_failover: bool = Field(default=True, description="Enable automatic failover")
    failover_timeout: float = Field(
        default=30.0, ge=5.0, le=300.0, description="Failover detection timeout"
    )
    backup_nodes: list[str] = Field(
        default_factory=list, description="Backup node addresses"
    )

    # Data redundancy
    replication_factor: int = Field(
        default=1, ge=1, le=5, description="Data replication factor"
    )
    enable_checkpointing: bool = Field(
        default=True, description="Enable task checkpointing"
    )
    checkpoint_interval: float = Field(
        default=300.0, ge=60.0, le=3600.0, description="Checkpoint interval"
    )


class ClusterConfig(BaseModel):
    """Cluster-wide configuration."""

    # Basic settings
    cluster_name: str = Field(default="pynomaly-cluster", description="Cluster name")
    cluster_mode: ClusterMode = Field(
        default=ClusterMode.DISTRIBUTED, description="Cluster operation mode"
    )

    # Node management
    min_nodes: int = Field(
        default=1, ge=1, le=100, description="Minimum required nodes"
    )
    max_nodes: int = Field(
        default=10, ge=1, le=1000, description="Maximum allowed nodes"
    )
    auto_scaling: bool = Field(default=False, description="Enable automatic scaling")
    scale_up_threshold: float = Field(
        default=0.8, ge=0.1, le=1.0, description="CPU threshold for scaling up"
    )
    scale_down_threshold: float = Field(
        default=0.3, ge=0.1, le=1.0, description="CPU threshold for scaling down"
    )

    # Coordination
    coordinator_nodes: list[str] = Field(
        default_factory=list, description="Coordinator node addresses"
    )
    election_timeout: float = Field(
        default=150.0, ge=50.0, le=500.0, description="Leader election timeout"
    )
    consensus_algorithm: str = Field(default="raft", description="Consensus algorithm")

    # Load balancing
    load_balancing_strategy: LoadBalancingStrategy = Field(
        default=LoadBalancingStrategy.LEAST_LOADED,
        description="Load balancing strategy",
    )
    rebalance_interval: float = Field(
        default=300.0, ge=60.0, le=3600.0, description="Load rebalancing interval"
    )

    # Monitoring
    enable_monitoring: bool = Field(
        default=True, description="Enable cluster monitoring"
    )
    metrics_retention_days: int = Field(
        default=7, ge=1, le=90, description="Metrics retention period"
    )
    log_level: str = Field(default="INFO", description="Logging level")


class DistributedConfig(BaseModel):
    """Main configuration for distributed processing."""

    # Core settings
    enabled: bool = Field(default=False, description="Enable distributed processing")
    mode: ClusterMode = Field(
        default=ClusterMode.STANDALONE, description="Operation mode"
    )

    # Sub-configurations
    network: NetworkConfig = Field(
        default_factory=NetworkConfig, description="Network configuration"
    )
    worker: WorkerConfig = Field(
        default_factory=WorkerConfig, description="Worker configuration"
    )
    cluster: ClusterConfig = Field(
        default_factory=ClusterConfig, description="Cluster configuration"
    )
    fault_tolerance: FaultToleranceConfig = Field(
        default_factory=FaultToleranceConfig,
        description="Fault tolerance configuration",
    )

    # Data processing
    partition_strategy: PartitionStrategy = Field(
        default=PartitionStrategy.SIZE_BASED, description="Data partitioning strategy"
    )
    aggregation_strategy: AggregationStrategy = Field(
        default=AggregationStrategy.WEIGHTED_AVERAGE,
        description="Result aggregation strategy",
    )

    # Performance settings
    chunk_size: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Default chunk size for processing",
    )
    parallel_chunks: int = Field(
        default=4, ge=1, le=100, description="Number of parallel chunks"
    )
    enable_compression: bool = Field(
        default=True, description="Enable data compression"
    )
    compression_algorithm: str = Field(
        default="gzip", description="Compression algorithm"
    )

    # Storage settings
    shared_storage_path: str | None = Field(
        default=None, description="Shared storage path"
    )
    temp_storage_path: str = Field(
        default="/tmp/pynomaly", description="Temporary storage path"
    )
    cleanup_temp_files: bool = Field(
        default=True, description="Cleanup temporary files"
    )

    # Security
    enable_encryption: bool = Field(default=False, description="Enable data encryption")
    encryption_algorithm: str = Field(
        default="AES-256", description="Encryption algorithm"
    )

    # Development settings
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    profile_performance: bool = Field(
        default=False, description="Enable performance profiling"
    )
    mock_workers: bool = Field(
        default=False, description="Use mock workers for testing"
    )


@dataclass
class RuntimeConfig:
    """Runtime configuration state."""

    # Current cluster state
    active_nodes: set[str] = field(default_factory=set)
    coordinator_node: str | None = None
    cluster_health: float = 1.0

    # Performance metrics
    total_tasks_processed: int = 0
    average_task_time: float = 0.0
    error_rate: float = 0.0

    # Resource usage
    total_cpu_usage: float = 0.0
    total_memory_usage: float = 0.0
    total_network_bandwidth: float = 0.0

    # Last update timestamp
    last_updated: float = 0.0


class ConfigurationManager:
    """Manages distributed processing configuration."""

    def __init__(self, config: DistributedConfig | None = None):
        """Initialize configuration manager.

        Args:
            config: Distributed configuration
        """
        self.config = config or DistributedConfig()
        self.runtime = RuntimeConfig()
        self._watchers = []

    def update_config(self, updates: dict[str, any]) -> None:
        """Update configuration with new values.

        Args:
            updates: Configuration updates
        """
        # Update the configuration
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            elif "." in key:
                # Handle nested updates
                parts = key.split(".")
                obj = self.config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)

        # Notify watchers
        self._notify_watchers()

    def add_config_watcher(self, callback) -> None:
        """Add configuration change watcher.

        Args:
            callback: Function to call on configuration changes
        """
        self._watchers.append(callback)

    def _notify_watchers(self) -> None:
        """Notify all configuration watchers."""
        for watcher in self._watchers:
            try:
                watcher(self.config)
            except Exception:
                # Log error but don't fail
                pass

    def validate_config(self) -> list[str]:
        """Validate configuration and return list of issues.

        Returns:
            List of validation issues
        """
        issues = []

        # Check network configuration
        if self.config.network.port <= 0 or self.config.network.port > 65535:
            issues.append("Invalid port number")

        # Check worker configuration
        if self.config.worker.max_concurrent_tasks <= 0:
            issues.append("Max concurrent tasks must be positive")

        # Check cluster configuration
        if self.config.cluster.min_nodes > self.config.cluster.max_nodes:
            issues.append("Min nodes cannot be greater than max nodes")

        # Check resource limits
        if self.config.worker.max_memory_mb < 512:
            issues.append("Minimum memory requirement is 512MB")

        return issues

    def get_effective_config(self) -> DistributedConfig:
        """Get effective configuration with environment overrides.

        Returns:
            Effective configuration
        """
        # Create a copy of the configuration
        effective_config = self.config.model_copy()

        # Apply environment variable overrides
        import os

        # Network overrides
        if os.getenv("PYNOMALY_CLUSTER_HOST"):
            effective_config.network.host = os.getenv("PYNOMALY_CLUSTER_HOST")
        if os.getenv("PYNOMALY_CLUSTER_PORT"):
            effective_config.network.port = int(os.getenv("PYNOMALY_CLUSTER_PORT"))

        # Worker overrides
        if os.getenv("PYNOMALY_MAX_WORKERS"):
            effective_config.worker.max_concurrent_tasks = int(
                os.getenv("PYNOMALY_MAX_WORKERS")
            )

        # Mode override
        if os.getenv("PYNOMALY_CLUSTER_MODE"):
            effective_config.mode = ClusterMode(os.getenv("PYNOMALY_CLUSTER_MODE"))

        return effective_config

    def export_config(self) -> dict[str, any]:
        """Export configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return {
            "config": self.config.model_dump(),
            "runtime": {
                "active_nodes": list(self.runtime.active_nodes),
                "coordinator_node": self.runtime.coordinator_node,
                "cluster_health": self.runtime.cluster_health,
                "total_tasks_processed": self.runtime.total_tasks_processed,
                "average_task_time": self.runtime.average_task_time,
                "error_rate": self.runtime.error_rate,
                "total_cpu_usage": self.runtime.total_cpu_usage,
                "total_memory_usage": self.runtime.total_memory_usage,
                "total_network_bandwidth": self.runtime.total_network_bandwidth,
                "last_updated": self.runtime.last_updated,
            },
        }

    def import_config(self, config_dict: dict[str, any]) -> None:
        """Import configuration from dictionary.

        Args:
            config_dict: Configuration dictionary
        """
        if "config" in config_dict:
            self.config = DistributedConfig(**config_dict["config"])

        if "runtime" in config_dict:
            runtime_data = config_dict["runtime"]
            self.runtime = RuntimeConfig(
                active_nodes=set(runtime_data.get("active_nodes", [])),
                coordinator_node=runtime_data.get("coordinator_node"),
                cluster_health=runtime_data.get("cluster_health", 1.0),
                total_tasks_processed=runtime_data.get("total_tasks_processed", 0),
                average_task_time=runtime_data.get("average_task_time", 0.0),
                error_rate=runtime_data.get("error_rate", 0.0),
                total_cpu_usage=runtime_data.get("total_cpu_usage", 0.0),
                total_memory_usage=runtime_data.get("total_memory_usage", 0.0),
                total_network_bandwidth=runtime_data.get(
                    "total_network_bandwidth", 0.0
                ),
                last_updated=runtime_data.get("last_updated", 0.0),
            )


# Global configuration manager instance
_config_manager: ConfigurationManager | None = None


def get_distributed_config_manager() -> ConfigurationManager:
    """Get global distributed configuration manager.

    Returns:
        Configuration manager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager


def init_distributed_config(
    config: DistributedConfig | None = None,
) -> ConfigurationManager:
    """Initialize global distributed configuration manager.

    Args:
        config: Distributed configuration

    Returns:
        Configuration manager instance
    """
    global _config_manager
    _config_manager = ConfigurationManager(config)
    return _config_manager
