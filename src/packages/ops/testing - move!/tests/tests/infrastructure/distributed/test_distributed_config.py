"""Test suite for distributed configuration management."""

import os
from unittest.mock import patch

import pytest

from monorepo.infrastructure.distributed.distributed_config import (
    AggregationStrategy,
    ClusterConfig,
    ClusterMode,
    CommunicationProtocol,
    ConfigurationManager,
    DistributedConfig,
    FaultToleranceConfig,
    LoadBalancingStrategy,
    NetworkConfig,
    PartitionStrategy,
    WorkerConfig,
)


class TestDistributedConfig:
    """Test cases for DistributedConfig."""

    def test_default_config_creation(self):
        """Test creating config with default values."""
        config = DistributedConfig()

        assert config.enabled is False
        assert config.mode == ClusterMode.STANDALONE
        assert config.partition_strategy == PartitionStrategy.SIZE_BASED
        assert config.aggregation_strategy == AggregationStrategy.WEIGHTED_AVERAGE
        assert config.chunk_size == 10000
        assert config.parallel_chunks == 4

    def test_network_config_validation(self):
        """Test network configuration validation."""
        # Valid configuration
        config = NetworkConfig(host="192.168.1.100", port=8080)
        assert config.host == "192.168.1.100"
        assert config.port == 8080

        # Invalid port
        with pytest.raises(ValueError):
            NetworkConfig(port=0)

        with pytest.raises(ValueError):
            NetworkConfig(port=70000)

    def test_worker_config_validation(self):
        """Test worker configuration validation."""
        config = WorkerConfig(
            max_concurrent_tasks=8, max_memory_mb=8192, max_cpu_cores=8
        )

        assert config.max_concurrent_tasks == 8
        assert config.max_memory_mb == 8192
        assert config.max_cpu_cores == 8

        # Test minimum values
        with pytest.raises(ValueError):
            WorkerConfig(max_concurrent_tasks=0)

        with pytest.raises(ValueError):
            WorkerConfig(max_memory_mb=100)  # Below minimum

    def test_cluster_config_validation(self):
        """Test cluster configuration validation."""
        config = ClusterConfig(min_nodes=2, max_nodes=10, auto_scaling=True)

        assert config.min_nodes == 2
        assert config.max_nodes == 10
        assert config.auto_scaling is True

        # Invalid node configuration
        with pytest.raises(ValueError):
            ClusterConfig(min_nodes=10, max_nodes=5)

    def test_fault_tolerance_config(self):
        """Test fault tolerance configuration."""
        config = FaultToleranceConfig(
            max_retries=5,
            enable_circuit_breaker=True,
            enable_failover=True,
            replication_factor=3,
        )

        assert config.max_retries == 5
        assert config.enable_circuit_breaker is True
        assert config.enable_failover is True
        assert config.replication_factor == 3

    def test_complete_distributed_config(self):
        """Test complete distributed configuration."""
        config = DistributedConfig(
            enabled=True,
            mode=ClusterMode.DISTRIBUTED,
            network=NetworkConfig(
                protocol=CommunicationProtocol.GRPC,
                host="cluster.example.com",
                port=9090,
            ),
            worker=WorkerConfig(max_concurrent_tasks=16, max_memory_mb=16384),
            cluster=ClusterConfig(
                cluster_name="test-cluster", min_nodes=3, max_nodes=20
            ),
        )

        assert config.enabled is True
        assert config.mode == ClusterMode.DISTRIBUTED
        assert config.network.protocol == CommunicationProtocol.GRPC
        assert config.network.host == "cluster.example.com"
        assert config.worker.max_concurrent_tasks == 16
        assert config.cluster.cluster_name == "test-cluster"


class TestConfigurationManager:
    """Test cases for ConfigurationManager."""

    def test_configuration_manager_creation(self):
        """Test creating configuration manager."""
        config = DistributedConfig(enabled=True)
        manager = ConfigurationManager(config)

        assert manager.config.enabled is True
        assert len(manager.runtime.active_nodes) == 0

    def test_config_update(self):
        """Test configuration updates."""
        manager = ConfigurationManager()

        # Update simple config
        manager.update_config({"enabled": True, "chunk_size": 20000})

        assert manager.config.enabled is True
        assert manager.config.chunk_size == 20000

        # Update nested config
        manager.update_config({"worker.max_concurrent_tasks": 12})

        assert manager.config.worker.max_concurrent_tasks == 12

    def test_config_watcher(self):
        """Test configuration change watchers."""
        manager = ConfigurationManager()

        # Track watcher calls
        watcher_calls = []

        def test_watcher(config):
            watcher_calls.append(config.enabled)

        manager.add_config_watcher(test_watcher)

        # Update config
        manager.update_config({"enabled": True})

        assert len(watcher_calls) == 1
        assert watcher_calls[0] is True

    def test_config_validation(self):
        """Test configuration validation."""
        manager = ConfigurationManager()

        # Valid configuration
        issues = manager.validate_config()
        assert len(issues) == 0

        # Invalid configuration
        manager.config.network.port = -1
        manager.config.worker.max_concurrent_tasks = 0
        manager.config.cluster.min_nodes = 10
        manager.config.cluster.max_nodes = 5
        manager.config.worker.max_memory_mb = 100

        issues = manager.validate_config()
        assert len(issues) > 0
        assert any("port" in issue.lower() for issue in issues)
        assert any("concurrent" in issue.lower() for issue in issues)
        assert any("min nodes" in issue.lower() for issue in issues)
        assert any("memory" in issue.lower() for issue in issues)

    def test_environment_overrides(self):
        """Test environment variable overrides."""
        base_config = DistributedConfig()
        manager = ConfigurationManager(base_config)

        with patch.dict(
            os.environ,
            {
                "PYNOMALY_CLUSTER_HOST": "192.168.1.50",
                "PYNOMALY_CLUSTER_PORT": "9090",
                "PYNOMALY_MAX_WORKERS": "8",
                "PYNOMALY_CLUSTER_MODE": "distributed",
            },
        ):
            effective_config = manager.get_effective_config()

            assert effective_config.network.host == "192.168.1.50"
            assert effective_config.network.port == 9090
            assert effective_config.worker.max_concurrent_tasks == 8
            assert effective_config.mode == ClusterMode.DISTRIBUTED

    def test_config_export_import(self):
        """Test configuration export and import."""
        original_config = DistributedConfig(
            enabled=True, mode=ClusterMode.HYBRID, chunk_size=15000
        )

        manager = ConfigurationManager(original_config)

        # Add some runtime state
        manager.runtime.active_nodes.add("worker-1")
        manager.runtime.active_nodes.add("worker-2")
        manager.runtime.coordinator_node = "coordinator-1"
        manager.runtime.cluster_health = 0.95

        # Export configuration
        exported = manager.export_config()

        assert "config" in exported
        assert "runtime" in exported
        assert exported["config"]["enabled"] is True
        assert exported["config"]["mode"] == "hybrid"
        assert len(exported["runtime"]["active_nodes"]) == 2
        assert exported["runtime"]["coordinator_node"] == "coordinator-1"
        assert exported["runtime"]["cluster_health"] == 0.95

        # Import into new manager
        new_manager = ConfigurationManager()
        new_manager.import_config(exported)

        assert new_manager.config.enabled is True
        assert new_manager.config.mode == ClusterMode.HYBRID
        assert new_manager.config.chunk_size == 15000
        assert len(new_manager.runtime.active_nodes) == 2
        assert "worker-1" in new_manager.runtime.active_nodes
        assert "worker-2" in new_manager.runtime.active_nodes
        assert new_manager.runtime.coordinator_node == "coordinator-1"


class TestConfigurationEnums:
    """Test configuration enum values."""

    def test_cluster_mode_enum(self):
        """Test ClusterMode enum values."""
        assert ClusterMode.STANDALONE == "standalone"
        assert ClusterMode.DISTRIBUTED == "distributed"
        assert ClusterMode.HYBRID == "hybrid"

    def test_communication_protocol_enum(self):
        """Test CommunicationProtocol enum values."""
        assert CommunicationProtocol.HTTP == "http"
        assert CommunicationProtocol.GRPC == "grpc"
        assert CommunicationProtocol.REDIS == "redis"
        assert CommunicationProtocol.RABBITMQ == "rabbitmq"
        assert CommunicationProtocol.KAFKA == "kafka"

    def test_partition_strategy_enum(self):
        """Test PartitionStrategy enum values."""
        assert PartitionStrategy.ROUND_ROBIN == "round_robin"
        assert PartitionStrategy.HASH_BASED == "hash_based"
        assert PartitionStrategy.SIZE_BASED == "size_based"
        assert PartitionStrategy.FEATURE_BASED == "feature_based"
        assert PartitionStrategy.ADAPTIVE == "adaptive"

    def test_load_balancing_strategy_enum(self):
        """Test LoadBalancingStrategy enum values."""
        assert LoadBalancingStrategy.ROUND_ROBIN == "round_robin"
        assert LoadBalancingStrategy.LEAST_LOADED == "least_loaded"
        assert LoadBalancingStrategy.WEIGHTED == "weighted"
        assert LoadBalancingStrategy.ADAPTIVE == "adaptive"
        assert LoadBalancingStrategy.CAPABILITY_BASED == "capability_based"

    def test_aggregation_strategy_enum(self):
        """Test AggregationStrategy enum values."""
        assert AggregationStrategy.SIMPLE_MERGE == "simple_merge"
        assert AggregationStrategy.WEIGHTED_AVERAGE == "weighted_average"
        assert AggregationStrategy.ENSEMBLE_VOTING == "ensemble_voting"
        assert AggregationStrategy.CONFIDENCE_WEIGHTED == "confidence_weighted"
        assert AggregationStrategy.HIERARCHICAL == "hierarchical"


class TestGlobalConfigurationFunctions:
    """Test global configuration functions."""

    def test_get_distributed_config_manager(self):
        """Test getting global configuration manager."""
        from monorepo.infrastructure.distributed.distributed_config import (
            get_distributed_config_manager,
        )

        manager1 = get_distributed_config_manager()
        manager2 = get_distributed_config_manager()

        # Should return the same instance
        assert manager1 is manager2

    def test_init_distributed_config(self):
        """Test initializing global configuration manager."""
        from monorepo.infrastructure.distributed.distributed_config import (
            init_distributed_config,
        )

        custom_config = DistributedConfig(enabled=True, chunk_size=25000)
        manager = init_distributed_config(custom_config)

        assert manager.config.enabled is True
        assert manager.config.chunk_size == 25000


class TestConfigurationIntegration:
    """Integration tests for configuration management."""

    def test_realistic_cluster_configuration(self):
        """Test realistic cluster configuration."""
        config = DistributedConfig(
            enabled=True,
            mode=ClusterMode.DISTRIBUTED,
            network=NetworkConfig(
                protocol=CommunicationProtocol.GRPC,
                host="pynomaly-cluster.internal",
                port=8080,
                enable_tls=True,
                connection_timeout=60.0,
                max_connections=200,
            ),
            worker=WorkerConfig(
                max_concurrent_tasks=8,
                max_memory_mb=8192,
                max_cpu_cores=8,
                supported_algorithms={
                    "isolation_forest",
                    "one_class_svm",
                    "local_outlier_factor",
                },
                hardware_acceleration={"cpu", "gpu"},
                batch_size=5000,
                enable_caching=True,
                cache_size_mb=1024,
            ),
            cluster=ClusterConfig(
                cluster_name="pynomaly-production",
                min_nodes=3,
                max_nodes=50,
                auto_scaling=True,
                load_balancing_strategy=LoadBalancingStrategy.LEAST_LOADED,
            ),
            fault_tolerance=FaultToleranceConfig(
                max_retries=3,
                enable_circuit_breaker=True,
                enable_failover=True,
                replication_factor=2,
                enable_checkpointing=True,
            ),
            partition_strategy=PartitionStrategy.ADAPTIVE,
            aggregation_strategy=AggregationStrategy.ENSEMBLE_VOTING,
            chunk_size=50000,
            parallel_chunks=16,
            enable_compression=True,
            enable_encryption=True,
        )

        # Validate the configuration
        manager = ConfigurationManager(config)
        issues = manager.validate_config()

        assert len(issues) == 0, f"Configuration validation failed: {issues}"
        assert config.enabled is True
        assert config.cluster.cluster_name == "pynomaly-production"
        assert config.worker.max_concurrent_tasks == 8
        assert "gpu" in config.worker.hardware_acceleration

    def test_development_vs_production_config(self):
        """Test different configurations for development vs production."""
        # Development configuration
        dev_config = DistributedConfig(
            enabled=False,  # Disabled for development
            mode=ClusterMode.STANDALONE,
            worker=WorkerConfig(
                max_concurrent_tasks=2, max_memory_mb=2048, enable_caching=False
            ),
            cluster=ClusterConfig(min_nodes=1, max_nodes=1, auto_scaling=False),
            debug_mode=True,
            mock_workers=True,
        )

        # Production configuration
        prod_config = DistributedConfig(
            enabled=True,
            mode=ClusterMode.DISTRIBUTED,
            worker=WorkerConfig(
                max_concurrent_tasks=16,
                max_memory_mb=32768,
                enable_caching=True,
                cache_size_mb=4096,
            ),
            cluster=ClusterConfig(min_nodes=5, max_nodes=100, auto_scaling=True),
            fault_tolerance=FaultToleranceConfig(
                max_retries=5,
                enable_circuit_breaker=True,
                enable_failover=True,
                replication_factor=3,
            ),
            enable_encryption=True,
            debug_mode=False,
            mock_workers=False,
        )

        # Validate both configurations
        dev_manager = ConfigurationManager(dev_config)
        prod_manager = ConfigurationManager(prod_config)

        dev_issues = dev_manager.validate_config()
        prod_issues = prod_manager.validate_config()

        assert len(dev_issues) == 0
        assert len(prod_issues) == 0

        # Verify key differences
        assert dev_config.enabled is False
        assert prod_config.enabled is True

        assert dev_config.mode == ClusterMode.STANDALONE
        assert prod_config.mode == ClusterMode.DISTRIBUTED

        assert dev_config.worker.max_concurrent_tasks == 2
        assert prod_config.worker.max_concurrent_tasks == 16

        assert dev_config.debug_mode is True
        assert prod_config.debug_mode is False
