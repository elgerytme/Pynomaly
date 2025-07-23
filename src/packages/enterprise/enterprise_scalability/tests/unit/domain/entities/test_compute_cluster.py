"""
Unit tests for ComputeCluster domain entities.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4, UUID

from enterprise_scalability.domain.entities.compute_cluster import (
    ComputeNode, ComputeCluster,
    ClusterType, ClusterStatus, NodeStatus, ResourceType, ScalingPolicy
)


class TestComputeNode:
    """Test cases for ComputeNode entity."""
    
    def test_compute_node_creation_basic(self):
        """Test basic compute node creation."""
        cluster_id = uuid4()
        
        node = ComputeNode(
            name="worker-001",
            cluster_id=cluster_id,
            node_type="standard-large",
            hostname="worker-001.cluster.local",
            ip_address="192.168.1.100",
            cpu_cores=8,
            memory_gb=32.0
        )
        
        assert isinstance(node.id, UUID)
        assert node.name == "worker-001"
        assert node.cluster_id == cluster_id
        assert node.node_type == "standard-large"
        assert node.hostname == "worker-001.cluster.local"
        assert node.ip_address == "192.168.1.100"
        assert node.cpu_cores == 8
        assert node.memory_gb == 32.0
        assert node.port == 8787
        assert node.gpu_count == 0
        assert node.gpu_type is None
        assert node.storage_gb == 100.0
        assert node.status == NodeStatus.PENDING
        assert node.health_score == 100.0
        assert node.last_heartbeat is None
        assert node.tasks_running == 0
        assert node.tasks_completed == 0
        assert node.tasks_failed == 0
        assert node.uptime_seconds == 0.0
        assert node.labels == {}
        assert node.taints == []
        assert node.drain_timeout_seconds == 300
        
    def test_compute_node_creation_comprehensive(self):
        """Test comprehensive compute node creation."""
        cluster_id = uuid4()
        labels = {"zone": "us-west-1a", "instance-type": "gpu"}
        taints = [{"key": "gpu", "value": "true", "effect": "NoSchedule"}]
        
        node = ComputeNode(
            name="gpu-worker-001",
            cluster_id=cluster_id,
            node_type="gpu-large",
            hostname="gpu-worker-001.cluster.local",
            ip_address="192.168.1.101",
            port=9999,
            cpu_cores=16,
            memory_gb=64.0,
            gpu_count=2,
            gpu_type="NVIDIA Tesla V100",
            storage_gb=500.0,
            cpu_usage_percent=25.5,
            memory_usage_gb=16.0,
            gpu_usage_percent=10.0,
            storage_usage_gb=50.0,
            network_io_mbps=100.0,
            status=NodeStatus.RUNNING,
            health_score=95.0,
            tasks_running=3,
            tasks_completed=100,
            tasks_failed=2,
            uptime_seconds=3600.0,
            labels=labels,
            taints=taints,
            drain_timeout_seconds=600
        )
        
        assert node.port == 9999
        assert node.gpu_count == 2
        assert node.gpu_type == "NVIDIA Tesla V100"
        assert node.storage_gb == 500.0
        assert node.cpu_usage_percent == 25.5
        assert node.memory_usage_gb == 16.0
        assert node.gpu_usage_percent == 10.0
        assert node.storage_usage_gb == 50.0
        assert node.network_io_mbps == 100.0
        assert node.status == NodeStatus.RUNNING
        assert node.health_score == 95.0
        assert node.tasks_running == 3
        assert node.tasks_completed == 100
        assert node.tasks_failed == 2
        assert node.uptime_seconds == 3600.0
        assert node.labels == labels
        assert node.taints == taints
        assert node.drain_timeout_seconds == 600
        
    def test_memory_usage_validation(self):
        """Test memory usage validation."""
        cluster_id = uuid4()
        
        with pytest.raises(ValueError, match="Memory usage cannot exceed capacity"):
            ComputeNode(
                name="worker-001",
                cluster_id=cluster_id,
                node_type="standard",
                hostname="worker-001.local",
                ip_address="192.168.1.100",
                cpu_cores=4,
                memory_gb=8.0,
                memory_usage_gb=10.0  # Exceeds capacity
            )
    
    def test_is_healthy_running_with_recent_heartbeat(self):
        """Test healthy node with running status and recent heartbeat."""
        node = ComputeNode(
            name="worker-001",
            cluster_id=uuid4(),
            node_type="standard",
            hostname="worker-001.local",
            ip_address="192.168.1.100",
            cpu_cores=4,
            memory_gb=8.0,
            status=NodeStatus.RUNNING,
            health_score=80.0,
            last_heartbeat=datetime.utcnow() - timedelta(minutes=2)
        )
        
        assert node.is_healthy() is True
        
    def test_is_healthy_running_with_old_heartbeat(self):
        """Test unhealthy node with old heartbeat."""
        node = ComputeNode(
            name="worker-001",
            cluster_id=uuid4(),
            node_type="standard",
            hostname="worker-001.local",
            ip_address="192.168.1.100",
            cpu_cores=4,
            memory_gb=8.0,
            status=NodeStatus.RUNNING,
            health_score=80.0,
            last_heartbeat=datetime.utcnow() - timedelta(minutes=10)
        )
        
        assert node.is_healthy() is False
        
    def test_is_healthy_low_health_score(self):
        """Test unhealthy node with low health score."""
        node = ComputeNode(
            name="worker-001",
            cluster_id=uuid4(),
            node_type="standard",
            hostname="worker-001.local",
            ip_address="192.168.1.100",
            cpu_cores=4,
            memory_gb=8.0,
            status=NodeStatus.RUNNING,
            health_score=50.0
        )
        
        assert node.is_healthy() is False
        
    def test_is_healthy_non_running_status(self):
        """Test unhealthy node with non-running status."""
        node = ComputeNode(
            name="worker-001",
            cluster_id=uuid4(),
            node_type="standard",
            hostname="worker-001.local",
            ip_address="192.168.1.100",
            cpu_cores=4,
            memory_gb=8.0,
            status=NodeStatus.STOPPED,
            health_score=90.0
        )
        
        assert node.is_healthy() is False
        
    def test_is_healthy_draining_status(self):
        """Test healthy node with draining status."""
        node = ComputeNode(
            name="worker-001",
            cluster_id=uuid4(),
            node_type="standard",
            hostname="worker-001.local",
            ip_address="192.168.1.100",
            cpu_cores=4,
            memory_gb=8.0,
            status=NodeStatus.DRAINING,
            health_score=80.0,
            last_heartbeat=datetime.utcnow() - timedelta(minutes=1)
        )
        
        assert node.is_healthy() is True
        
    def test_get_resource_utilization(self):
        """Test resource utilization calculation."""
        node = ComputeNode(
            name="worker-001",
            cluster_id=uuid4(),
            node_type="standard",
            hostname="worker-001.local",
            ip_address="192.168.1.100",
            cpu_cores=8,
            memory_gb=32.0,
            storage_gb=100.0,
            cpu_usage_percent=50.0,
            memory_usage_gb=16.0,
            gpu_usage_percent=25.0,
            storage_usage_gb=40.0
        )
        
        utilization = node.get_resource_utilization()
        
        assert utilization["cpu"] == 50.0
        assert utilization["memory"] == 50.0  # 16/32 * 100
        assert utilization["gpu"] == 25.0
        assert utilization["storage"] == 40.0  # 40/100 * 100
        
    def test_get_resource_utilization_zero_capacity(self):
        """Test resource utilization with zero capacity."""
        node = ComputeNode(
            name="worker-001",
            cluster_id=uuid4(),
            node_type="standard",
            hostname="worker-001.local",
            ip_address="192.168.1.100",
            cpu_cores=8,
            memory_gb=0.0,  # Zero memory capacity
            storage_gb=0.0,  # Zero storage capacity
            cpu_usage_percent=50.0,
            memory_usage_gb=0.0,
            storage_usage_gb=0.0
        )
        
        utilization = node.get_resource_utilization()
        
        assert utilization["cpu"] == 50.0
        assert utilization["memory"] == 0.0
        assert utilization["storage"] == 0.0
        
    def test_has_capacity_for_task_sufficient(self):
        """Test task capacity check with sufficient resources."""
        node = ComputeNode(
            name="worker-001",
            cluster_id=uuid4(),
            node_type="standard",
            hostname="worker-001.local",
            ip_address="192.168.1.100",
            cpu_cores=8,
            memory_gb=32.0,
            cpu_usage_percent=50.0,  # 4 cores used, 4 available
            memory_usage_gb=16.0     # 16 GB used, 16 available
        )
        
        # Request 2 cores and 8 GB - should have capacity
        assert node.has_capacity_for_task(2.0, 8.0) is True
        
    def test_has_capacity_for_task_insufficient_cpu(self):
        """Test task capacity check with insufficient CPU."""
        node = ComputeNode(
            name="worker-001",
            cluster_id=uuid4(),
            node_type="standard",
            hostname="worker-001.local",
            ip_address="192.168.1.100",
            cpu_cores=8,
            memory_gb=32.0,
            cpu_usage_percent=75.0,  # 6 cores used, 2 available
            memory_usage_gb=16.0     # 16 GB used, 16 available
        )
        
        # Request 4 cores and 8 GB - insufficient CPU
        assert node.has_capacity_for_task(4.0, 8.0) is False
        
    def test_has_capacity_for_task_insufficient_memory(self):
        """Test task capacity check with insufficient memory."""
        node = ComputeNode(
            name="worker-001",
            cluster_id=uuid4(),
            node_type="standard",
            hostname="worker-001.local",
            ip_address="192.168.1.100",
            cpu_cores=8,
            memory_gb=32.0,
            cpu_usage_percent=50.0,  # 4 cores used, 4 available
            memory_usage_gb=28.0     # 28 GB used, 4 available
        )
        
        # Request 2 cores and 8 GB - insufficient memory
        assert node.has_capacity_for_task(2.0, 8.0) is False
        
    def test_update_heartbeat(self):
        """Test heartbeat update."""
        node = ComputeNode(
            name="worker-001",
            cluster_id=uuid4(),
            node_type="standard",
            hostname="worker-001.local",
            ip_address="192.168.1.100",
            cpu_cores=4,
            memory_gb=8.0
        )
        
        original_updated_at = node.updated_at
        
        node.update_heartbeat()
        
        assert node.last_heartbeat is not None
        assert node.updated_at > original_updated_at
        
    def test_update_metrics(self):
        """Test metrics update."""
        node = ComputeNode(
            name="worker-001",
            cluster_id=uuid4(),
            node_type="standard",
            hostname="worker-001.local",
            ip_address="192.168.1.100",
            cpu_cores=8,
            memory_gb=32.0,
            storage_gb=100.0
        )
        
        original_updated_at = node.updated_at
        
        node.update_metrics(
            cpu_usage=75.0,
            memory_usage=20.0,
            gpu_usage=50.0,
            storage_usage=60.0
        )
        
        assert node.cpu_usage_percent == 75.0
        assert node.memory_usage_gb == 20.0
        assert node.gpu_usage_percent == 50.0
        assert node.storage_usage_gb == 60.0
        assert node.updated_at > original_updated_at
        
    def test_update_metrics_boundary_values(self):
        """Test metrics update with boundary values."""
        node = ComputeNode(
            name="worker-001",
            cluster_id=uuid4(),
            node_type="standard", 
            hostname="worker-001.local",
            ip_address="192.168.1.100",
            cpu_cores=8,
            memory_gb=32.0,
            storage_gb=100.0
        )
        
        # Test values exceeding limits are clamped
        node.update_metrics(
            cpu_usage=150.0,    # Should be clamped to 100.0
            memory_usage=50.0,  # Should be clamped to 32.0 (capacity)
            gpu_usage=-10.0,    # Should be clamped to 0.0
            storage_usage=200.0 # Should be clamped to 100.0 (capacity)
        )
        
        assert node.cpu_usage_percent == 100.0
        assert node.memory_usage_gb == 32.0
        assert node.gpu_usage_percent == 0.0
        assert node.storage_usage_gb == 100.0
        
    def test_drain(self):
        """Test node draining."""
        node = ComputeNode(
            name="worker-001",
            cluster_id=uuid4(),
            node_type="standard",
            hostname="worker-001.local",
            ip_address="192.168.1.100",
            cpu_cores=4,
            memory_gb=8.0,
            status=NodeStatus.RUNNING
        )
        
        original_updated_at = node.updated_at
        
        node.drain(timeout_seconds=600)
        
        assert node.status == NodeStatus.DRAINING
        assert node.drain_timeout_seconds == 600
        assert node.updated_at > original_updated_at


class TestComputeCluster:
    """Test cases for ComputeCluster entity."""
    
    def test_compute_cluster_creation_basic(self):
        """Test basic compute cluster creation."""
        tenant_id = uuid4()
        created_by = uuid4()
        
        cluster = ComputeCluster(
            name="data-processing-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=tenant_id,
            created_by=created_by,
            scheduler_address="tcp://scheduler:8786",
            services_covered=["data_processing", "ml_training"]
        )
        
        assert isinstance(cluster.id, UUID)
        assert cluster.name == "data-processing-cluster"
        assert cluster.description == ""
        assert cluster.cluster_type == ClusterType.DASK
        assert cluster.version == "2024.1.0"
        assert cluster.tenant_id == tenant_id
        assert cluster.created_by == created_by
        assert cluster.shared_users == []
        assert cluster.scheduler_address == "tcp://scheduler:8786"
        assert cluster.dashboard_port is None
        assert cluster.configuration == {}
        assert cluster.max_nodes == 100
        assert cluster.min_nodes == 1
        assert cluster.max_cpu_cores == 1000
        assert cluster.max_memory_gb == 1000.0
        assert cluster.status == ClusterStatus.PENDING
        assert cluster.current_nodes == 0
        assert cluster.active_nodes == 0
        assert cluster.total_cpu_cores == 0
        assert cluster.total_memory_gb == 0.0
        assert cluster.used_cpu_cores == 0.0
        assert cluster.used_memory_gb == 0.0
        assert cluster.scaling_policy == ScalingPolicy.MANUAL
        assert cluster.auto_scale_enabled is False
        assert cluster.scale_up_threshold == 80.0
        assert cluster.scale_down_threshold == 20.0
        assert cluster.scale_up_cooldown_minutes == 5
        assert cluster.scale_down_cooldown_minutes == 10
        assert cluster.tasks_submitted == 0
        assert cluster.tasks_completed == 0
        assert cluster.tasks_failed == 0
        assert cluster.avg_task_duration_seconds == 0.0
        assert cluster.health_score == 100.0
        assert cluster.last_scaling_action is None
        assert cluster.error_message is None
        
    def test_compute_cluster_creation_comprehensive(self):
        """Test comprehensive compute cluster creation."""
        tenant_id = uuid4()
        created_by = uuid4()
        shared_users = [uuid4(), uuid4()]
        configuration = {"worker_memory": "4GB", "scheduler_timeout": "30s"}
        
        cluster = ComputeCluster(
            name="ml-training-cluster",
            description="Cluster for machine learning model training",
            cluster_type=ClusterType.RAY,
            version="2.8.0",
            tenant_id=tenant_id,
            created_by=created_by,
            shared_users=shared_users,
            scheduler_address="ray://head-node:10001",
            dashboard_port=8265,
            configuration=configuration,
            max_nodes=50,
            min_nodes=2,
            max_cpu_cores=500,
            max_memory_gb=2000.0,
            status=ClusterStatus.RUNNING,
            current_nodes=5,
            active_nodes=4,
            total_cpu_cores=40,
            total_memory_gb=160.0,
            used_cpu_cores=20.0,
            used_memory_gb=80.0,
            scaling_policy=ScalingPolicy.AUTO_CPU,
            auto_scale_enabled=True,
            scale_up_threshold=75.0,
            scale_down_threshold=25.0,
            scale_up_cooldown_minutes=3,
            scale_down_cooldown_minutes=15,
            tasks_submitted=1000,
            tasks_completed=950,
            tasks_failed=50,
            avg_task_duration_seconds=120.5,
            health_score=85.0,
            services_covered=["ml_training", "hyperparameter_tuning"]
        )
        
        assert cluster.description == "Cluster for machine learning model training"
        assert cluster.cluster_type == ClusterType.RAY
        assert cluster.shared_users == shared_users
        assert cluster.dashboard_port == 8265
        assert cluster.configuration == configuration
        assert cluster.max_nodes == 50
        assert cluster.min_nodes == 2
        assert cluster.max_cpu_cores == 500
        assert cluster.max_memory_gb == 2000.0
        assert cluster.status == ClusterStatus.RUNNING
        assert cluster.current_nodes == 5
        assert cluster.active_nodes == 4
        assert cluster.total_cpu_cores == 40
        assert cluster.total_memory_gb == 160.0
        assert cluster.used_cpu_cores == 20.0
        assert cluster.used_memory_gb == 80.0
        assert cluster.scaling_policy == ScalingPolicy.AUTO_CPU
        assert cluster.auto_scale_enabled is True
        assert cluster.scale_up_threshold == 75.0
        assert cluster.scale_down_threshold == 25.0
        assert cluster.scale_up_cooldown_minutes == 3
        assert cluster.scale_down_cooldown_minutes == 15
        assert cluster.tasks_submitted == 1000
        assert cluster.tasks_completed == 950
        assert cluster.tasks_failed == 50
        assert cluster.avg_task_duration_seconds == 120.5
        assert cluster.health_score == 85.0
        
    def test_min_nodes_validation(self):
        """Test min_nodes validation against max_nodes."""
        tenant_id = uuid4()
        created_by = uuid4()
        
        with pytest.raises(ValueError, match="min_nodes cannot be greater than max_nodes"):
            ComputeCluster(
                name="test-cluster",
                cluster_type=ClusterType.DASK,
                version="2024.1.0",
                tenant_id=tenant_id,
                created_by=created_by,
                scheduler_address="tcp://scheduler:8786",
                services_covered=["test"],
                max_nodes=5,
                min_nodes=10  # Greater than max_nodes
            )
            
    def test_scale_thresholds_validation(self):
        """Test scale threshold validation."""
        tenant_id = uuid4()
        created_by = uuid4()
        
        with pytest.raises(ValueError, match="scale_down_threshold must be less than scale_up_threshold"):
            ComputeCluster(
                name="test-cluster",
                cluster_type=ClusterType.DASK,
                version="2024.1.0",
                tenant_id=tenant_id,
                created_by=created_by,
                scheduler_address="tcp://scheduler:8786",
                services_covered=["test"],
                scale_up_threshold=70.0,
                scale_down_threshold=80.0  # Greater than scale_up_threshold
            )
            
    def test_is_running_true(self):
        """Test is_running returns True for running cluster with active nodes."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            status=ClusterStatus.RUNNING,
            active_nodes=3
        )
        
        assert cluster.is_running() is True
        
    def test_is_running_false_wrong_status(self):
        """Test is_running returns False for non-running status."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            status=ClusterStatus.STOPPED,
            active_nodes=3
        )
        
        assert cluster.is_running() is False
        
    def test_is_running_false_no_active_nodes(self):
        """Test is_running returns False with no active nodes."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            status=ClusterStatus.RUNNING,
            active_nodes=0
        )
        
        assert cluster.is_running() is False
        
    def test_is_healthy_true(self):
        """Test is_healthy returns True for healthy running cluster."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            status=ClusterStatus.RUNNING,
            active_nodes=3,
            min_nodes=2,
            health_score=80.0
        )
        
        assert cluster.is_healthy() is True
        
    def test_is_healthy_false_not_running(self):
        """Test is_healthy returns False for non-running cluster."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            status=ClusterStatus.STOPPED,
            active_nodes=3,
            min_nodes=2,
            health_score=80.0
        )
        
        assert cluster.is_healthy() is False
        
    def test_is_healthy_false_low_health_score(self):
        """Test is_healthy returns False for low health score."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            status=ClusterStatus.RUNNING,
            active_nodes=3,
            min_nodes=2,
            health_score=50.0
        )
        
        assert cluster.is_healthy() is False
        
    def test_is_healthy_false_insufficient_nodes(self):
        """Test is_healthy returns False for insufficient active nodes."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            status=ClusterStatus.RUNNING,
            active_nodes=1,
            min_nodes=3,
            health_score=80.0
        )
        
        assert cluster.is_healthy() is False
        
    def test_get_resource_utilization(self):
        """Test resource utilization calculation."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            total_cpu_cores=100,
            used_cpu_cores=50.0,
            total_memory_gb=400.0,
            used_memory_gb=200.0,
            current_nodes=5,
            max_nodes=10
        )
        
        utilization = cluster.get_resource_utilization()
        
        assert utilization["cpu"] == 50.0      # 50/100 * 100
        assert utilization["memory"] == 50.0   # 200/400 * 100
        assert utilization["nodes"] == 50.0    # 5/10 * 100
        
    def test_get_resource_utilization_zero_totals(self):
        """Test resource utilization with zero totals."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            total_cpu_cores=0,
            used_cpu_cores=0.0,
            total_memory_gb=0.0,
            used_memory_gb=0.0,
            current_nodes=0,
            max_nodes=10
        )
        
        utilization = cluster.get_resource_utilization()
        
        assert utilization["cpu"] == 0.0
        assert utilization["memory"] == 0.0
        assert utilization["nodes"] == 0.0
        
    def test_should_scale_up_true(self):
        """Test should_scale_up returns True when conditions met."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            auto_scale_enabled=True,
            current_nodes=5,
            max_nodes=10,
            total_cpu_cores=40,
            used_cpu_cores=35.0,  # 87.5% utilization
            scale_up_threshold=80.0
        )
        
        assert cluster.should_scale_up() is True
        
    def test_should_scale_up_false_disabled(self):
        """Test should_scale_up returns False when auto-scaling disabled."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            auto_scale_enabled=False,
            current_nodes=5,
            max_nodes=10,
            total_cpu_cores=40,
            used_cpu_cores=35.0,
            scale_up_threshold=80.0
        )
        
        assert cluster.should_scale_up() is False
        
    def test_should_scale_up_false_at_max_nodes(self):
        """Test should_scale_up returns False when at max nodes."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            auto_scale_enabled=True,
            current_nodes=10,
            max_nodes=10,
            total_cpu_cores=80,
            used_cpu_cores=70.0,
            scale_up_threshold=80.0
        )
        
        assert cluster.should_scale_up() is False
        
    def test_should_scale_up_false_low_utilization(self):
        """Test should_scale_up returns False with low utilization."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            auto_scale_enabled=True,
            current_nodes=5,
            max_nodes=10,
            total_cpu_cores=40,
            used_cpu_cores=20.0,  # 50% utilization
            scale_up_threshold=80.0
        )
        
        assert cluster.should_scale_up() is False
        
    def test_should_scale_down_true(self):
        """Test should_scale_down returns True when conditions met."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            auto_scale_enabled=True,
            current_nodes=5,
            min_nodes=2,
            total_cpu_cores=40,
            used_cpu_cores=4.0,   # 10% utilization
            total_memory_gb=160.0,
            used_memory_gb=16.0,  # 10% utilization
            scale_down_threshold=20.0
        )
        
        assert cluster.should_scale_down() is True
        
    def test_should_scale_down_false_disabled(self):
        """Test should_scale_down returns False when auto-scaling disabled."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            auto_scale_enabled=False,
            current_nodes=5,
            min_nodes=2,
            total_cpu_cores=40,
            used_cpu_cores=4.0,
            total_memory_gb=160.0,
            used_memory_gb=16.0,
            scale_down_threshold=20.0
        )
        
        assert cluster.should_scale_down() is False
        
    def test_should_scale_down_false_at_min_nodes(self):
        """Test should_scale_down returns False when at min nodes."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            auto_scale_enabled=True,
            current_nodes=2,
            min_nodes=2,
            total_cpu_cores=16,
            used_cpu_cores=2.0,
            total_memory_gb=64.0,
            used_memory_gb=8.0,
            scale_down_threshold=20.0
        )
        
        assert cluster.should_scale_down() is False
        
    def test_should_scale_down_false_high_utilization(self):
        """Test should_scale_down returns False with high utilization."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            auto_scale_enabled=True,
            current_nodes=5,
            min_nodes=2,
            total_cpu_cores=40,
            used_cpu_cores=20.0,  # 50% utilization
            total_memory_gb=160.0,
            used_memory_gb=80.0,  # 50% utilization
            scale_down_threshold=20.0
        )
        
        assert cluster.should_scale_down() is False
        
    def test_can_scale_no_previous_action(self):
        """Test can_scale returns True with no previous scaling action."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            last_scaling_action=None
        )
        
        assert cluster.can_scale("up") is True
        assert cluster.can_scale("down") is True
        
    def test_can_scale_cooldown_not_expired(self):
        """Test can_scale returns False during cooldown period."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            last_scaling_action=datetime.utcnow() - timedelta(minutes=2),
            scale_up_cooldown_minutes=5,
            scale_down_cooldown_minutes=10
        )
        
        assert cluster.can_scale("up") is False
        assert cluster.can_scale("down") is False
        
    def test_can_scale_cooldown_expired(self):
        """Test can_scale returns True after cooldown period."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            last_scaling_action=datetime.utcnow() - timedelta(minutes=15),
            scale_up_cooldown_minutes=5,
            scale_down_cooldown_minutes=10
        )
        
        assert cluster.can_scale("up") is True
        assert cluster.can_scale("down") is True
        
    def test_update_resource_totals(self):
        """Test updating resource totals from node list."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"]
        )
        
        nodes = [
            ComputeNode(
                name="node-1",
                cluster_id=cluster.id,
                node_type="standard",
                hostname="node-1.local",
                ip_address="192.168.1.101",
                cpu_cores=8,
                memory_gb=32.0,
                status=NodeStatus.RUNNING,
                cpu_usage_percent=50.0,
                memory_usage_gb=16.0
            ),
            ComputeNode(
                name="node-2",
                cluster_id=cluster.id,
                node_type="standard",
                hostname="node-2.local",
                ip_address="192.168.1.102",
                cpu_cores=8,
                memory_gb=32.0,
                status=NodeStatus.RUNNING,
                cpu_usage_percent=25.0,
                memory_usage_gb=8.0
            ),
            ComputeNode(
                name="node-3",
                cluster_id=cluster.id,
                node_type="standard",
                hostname="node-3.local",
                ip_address="192.168.1.103",
                cpu_cores=8,
                memory_gb=32.0,
                status=NodeStatus.PENDING,
                cpu_usage_percent=0.0,
                memory_usage_gb=0.0
            )
        ]
        
        original_updated_at = cluster.updated_at
        
        cluster.update_resource_totals(nodes)
        
        assert cluster.current_nodes == 3
        assert cluster.active_nodes == 2  # Only RUNNING nodes
        assert cluster.total_cpu_cores == 24  # 8 + 8 + 8
        assert cluster.total_memory_gb == 96.0  # 32 + 32 + 32
        assert cluster.used_cpu_cores == 6.0  # (8*0.5) + (8*0.25) + (8*0.0)
        assert cluster.used_memory_gb == 24.0  # 16 + 8 + 0
        assert cluster.updated_at > original_updated_at
        
    def test_record_scaling_action(self):
        """Test recording scaling action."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"]
        )
        
        original_updated_at = cluster.updated_at
        
        cluster.record_scaling_action("up")
        
        assert cluster.last_scaling_action is not None
        assert cluster.updated_at > original_updated_at
        
    def test_start(self):
        """Test starting cluster."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            status=ClusterStatus.PENDING,
            error_message="Previous error"
        )
        
        original_updated_at = cluster.updated_at
        
        cluster.start()
        
        assert cluster.status == ClusterStatus.RUNNING
        assert cluster.started_at is not None
        assert cluster.updated_at > original_updated_at
        assert cluster.error_message is None
        
    def test_stop(self):
        """Test stopping cluster."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            status=ClusterStatus.RUNNING
        )
        
        original_updated_at = cluster.updated_at
        
        cluster.stop()
        
        assert cluster.status == ClusterStatus.STOPPED
        assert cluster.stopped_at is not None
        assert cluster.updated_at > original_updated_at
        
    def test_set_error(self):
        """Test setting cluster error state."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            status=ClusterStatus.RUNNING
        )
        
        original_updated_at = cluster.updated_at
        error_msg = "Scheduler connection failed"
        
        cluster.set_error(error_msg)
        
        assert cluster.status == ClusterStatus.ERROR
        assert cluster.error_message == error_msg
        assert cluster.updated_at > original_updated_at
        
    def test_get_cluster_summary(self):
        """Test getting cluster summary."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            status=ClusterStatus.RUNNING,
            current_nodes=5,
            active_nodes=4,
            max_nodes=10,
            min_nodes=2,
            total_cpu_cores=40,
            total_memory_gb=160.0,
            used_cpu_cores=20.0,
            used_memory_gb=80.0,
            health_score=85.0,
            auto_scale_enabled=True,
            tasks_completed=950,
            tasks_failed=50,
            started_at=datetime.utcnow() - timedelta(hours=2)
        )
        
        summary = cluster.get_cluster_summary()
        
        assert summary["id"] == str(cluster.id)
        assert summary["name"] == "test-cluster"
        assert summary["type"] == ClusterType.DASK
        assert summary["status"] == ClusterStatus.RUNNING
        assert summary["nodes"]["current"] == 5
        assert summary["nodes"]["active"] == 4
        assert summary["nodes"]["max"] == 10
        assert summary["nodes"]["min"] == 2
        assert summary["resources"]["cpu_cores"] == 40
        assert summary["resources"]["memory_gb"] == 160.0
        assert summary["resources"]["utilization"]["cpu"] == 50.0
        assert summary["health_score"] == 85.0
        assert summary["auto_scaling"] is True
        assert summary["tasks"]["completed"] == 950
        assert summary["tasks"]["failed"] == 50
        assert summary["tasks"]["success_rate"] == 95.0  # 950/(950+50) * 100
        assert summary["uptime_hours"] == pytest.approx(2.0, rel=0.1)
        
    def test_get_cluster_summary_no_tasks(self):
        """Test cluster summary with no completed tasks."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            tasks_completed=0,
            tasks_failed=0
        )
        
        summary = cluster.get_cluster_summary()
        
        assert summary["tasks"]["success_rate"] == 0.0
        
    def test_get_cluster_summary_no_start_time(self):
        """Test cluster summary with no start time."""
        cluster = ComputeCluster(
            name="test-cluster",
            cluster_type=ClusterType.DASK,
            version="2024.1.0",
            tenant_id=uuid4(),
            created_by=uuid4(),
            scheduler_address="tcp://scheduler:8786",
            services_covered=["test"],
            started_at=None
        )
        
        summary = cluster.get_cluster_summary()
        
        assert summary["uptime_hours"] == 0.0