"""Comprehensive tests for distributed processing infrastructure."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import uuid
from typing import Dict, List, Any

from pynomaly.infrastructure.distributed import (
    DistributedProcessingManager,
    DistributedWorker,
    DetectionCoordinator,
    LoadBalancer,
    TaskQueue,
    Task,
    TaskStatus,
    TaskPriority,
    WorkerPool,
    Worker,
    WorkerStatus,
    CoordinationService,
    Workflow,
    WorkflowStep,
    WorkflowStatus
)
from pynomaly.domain.exceptions import ProcessingError


class TestDistributedProcessingManager:
    """Test DistributedProcessingManager functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create a distributed processing manager instance."""
        return DistributedProcessingManager(
            max_workers=5,
            heartbeat_interval=10,
            task_timeout=300
        )
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, manager):
        """Test manager initialization."""
        assert manager.max_workers == 5
        assert manager.heartbeat_interval == 10
        assert manager.task_timeout == 300
        assert not manager._running
        assert len(manager.workers) == 0
    
    @pytest.mark.asyncio
    async def test_manager_start_stop(self, manager):
        """Test manager start and stop lifecycle."""
        # Start manager
        await manager.start()
        assert manager._running
        
        # Stop manager
        await manager.stop()
        assert not manager._running
    
    @pytest.mark.asyncio
    async def test_worker_registration(self, manager):
        """Test worker registration and deregistration."""
        worker_id = "test_worker_1"
        capabilities = ["sklearn", "pyod"]
        
        # Register worker
        success = await manager.register_worker(
            worker_id=worker_id,
            host="localhost",
            port=8000,
            capabilities=capabilities
        )
        assert success
        assert worker_id in manager.workers
        
        worker_info = manager.workers[worker_id]
        assert worker_info['host'] == "localhost"
        assert worker_info['port'] == 8000
        assert worker_info['capabilities'] == capabilities
        
        # Deregister worker
        success = await manager.deregister_worker(worker_id)
        assert success
        assert worker_id not in manager.workers
    
    @pytest.mark.asyncio
    async def test_worker_heartbeat(self, manager):
        """Test worker heartbeat handling."""
        worker_id = "test_worker_1"
        
        # Register worker first
        await manager.register_worker(
            worker_id=worker_id,
            host="localhost",
            port=8000
        )
        
        # Send heartbeat
        metrics = {"cpu_usage": 50.0, "memory_usage": 60.0}
        success = await manager.update_worker_heartbeat(
            worker_id=worker_id,
            status="idle",
            metrics=metrics
        )
        assert success
        
        worker_info = manager.workers[worker_id]
        assert worker_info['status'] == "idle"
        assert worker_info['metrics'] == metrics
        assert worker_info['last_heartbeat'] is not None
    
    @pytest.mark.asyncio
    async def test_task_distribution(self, manager):
        """Test task distribution to workers."""
        # Register multiple workers
        for i in range(3):
            await manager.register_worker(
                worker_id=f"worker_{i}",
                host="localhost",
                port=8000 + i
            )
        
        # Submit tasks
        task_ids = []
        for i in range(5):
            task_id = await manager.submit_task(
                task_type="anomaly_detection",
                payload={"data": f"test_data_{i}"},
                priority=TaskPriority.NORMAL.value
            )
            task_ids.append(task_id)
        
        assert len(task_ids) == 5
        for task_id in task_ids:
            assert task_id in manager.tasks
    
    @pytest.mark.asyncio
    async def test_system_metrics(self, manager):
        """Test system metrics collection."""
        # Start manager
        await manager.start()
        
        # Get initial metrics
        metrics = await manager.get_system_metrics()
        
        assert 'workers' in metrics
        assert 'tasks' in metrics
        assert 'performance' in metrics
        assert 'uptime' in metrics
        
        # Verify worker metrics
        assert metrics['workers']['total'] == 0
        assert metrics['workers']['active'] == 0
        assert metrics['workers']['utilization'] == 0.0
        
        await manager.stop()


class TestDistributedWorker:
    """Test DistributedWorker functionality."""
    
    @pytest.fixture
    def worker(self):
        """Create a distributed worker instance."""
        return DistributedWorker(
            worker_id="test_worker",
            host="localhost",
            port=8000,
            capabilities=["sklearn", "pyod"],
            max_concurrent_tasks=3
        )
    
    @pytest.mark.asyncio
    async def test_worker_initialization(self, worker):
        """Test worker initialization."""
        assert worker.worker_id == "test_worker"
        assert worker.host == "localhost"
        assert worker.port == 8000
        assert worker.capabilities == ["sklearn", "pyod"]
        assert worker.max_concurrent_tasks == 3
        assert not worker._running
    
    @pytest.mark.asyncio
    async def test_worker_start_stop(self, worker):
        """Test worker start and stop lifecycle."""
        # Start worker
        await worker.start()
        assert worker._running
        
        # Stop worker
        await worker.stop()
        assert not worker._running
    
    @pytest.mark.asyncio
    async def test_task_execution(self, worker):
        """Test task execution on worker."""
        task = Task(
            id="test_task",
            type="anomaly_detection",
            payload={"data": "test_data"},
            priority=TaskPriority.NORMAL.value
        )
        
        # Mock task execution
        with patch.object(worker, '_execute_task', return_value={"result": "success"}):
            result = await worker.execute_task(task)
            assert result == {"result": "success"}
    
    @pytest.mark.asyncio
    async def test_worker_status_reporting(self, worker):
        """Test worker status reporting."""
        status = await worker.get_status()
        
        assert 'worker_id' in status
        assert 'host' in status
        assert 'port' in status
        assert 'capabilities' in status
        assert 'current_load' in status
        assert 'max_concurrent_tasks' in status
        assert 'status' in status
        assert 'metrics' in status
        
        assert status['worker_id'] == "test_worker"
        assert status['host'] == "localhost"
        assert status['port'] == 8000
    
    @pytest.mark.asyncio
    async def test_capability_checking(self, worker):
        """Test worker capability checking."""
        # Test matching capabilities
        assert worker.has_capability("sklearn")
        assert worker.has_capability("pyod")
        assert worker.has_capabilities(["sklearn"])
        assert worker.has_capabilities(["sklearn", "pyod"])
        
        # Test non-matching capabilities
        assert not worker.has_capability("tensorflow")
        assert not worker.has_capabilities(["tensorflow"])
        assert not worker.has_capabilities(["sklearn", "tensorflow"])


class TestTaskQueue:
    """Test TaskQueue functionality."""
    
    @pytest.fixture
    def task_queue(self):
        """Create a task queue instance."""
        return TaskQueue(max_size=100, cleanup_interval=60)
    
    @pytest.mark.asyncio
    async def test_task_submission(self, task_queue):
        """Test task submission to queue."""
        await task_queue.start()
        
        task_id = await task_queue.submit_task(
            task_type="anomaly_detection",
            payload={"data": "test_data"},
            priority=TaskPriority.HIGH.value
        )
        
        assert task_id is not None
        task = await task_queue.get_task(task_id)
        assert task is not None
        assert task.type == "anomaly_detection"
        assert task.priority == TaskPriority.HIGH.value
        assert task.status == TaskStatus.QUEUED
        
        await task_queue.stop()
    
    @pytest.mark.asyncio
    async def test_task_priority_ordering(self, task_queue):
        """Test task priority ordering in queue."""
        await task_queue.start()
        
        # Submit tasks with different priorities
        low_task_id = await task_queue.submit_task(
            task_type="test", payload={}, priority=TaskPriority.LOW.value
        )
        high_task_id = await task_queue.submit_task(
            task_type="test", payload={}, priority=TaskPriority.HIGH.value
        )
        normal_task_id = await task_queue.submit_task(
            task_type="test", payload={}, priority=TaskPriority.NORMAL.value
        )
        
        # Get tasks - should be in priority order
        first_task = await task_queue.get_next_task()
        second_task = await task_queue.get_next_task()
        third_task = await task_queue.get_next_task()
        
        assert first_task.id == high_task_id
        assert second_task.id == normal_task_id
        assert third_task.id == low_task_id
        
        await task_queue.stop()
    
    @pytest.mark.asyncio
    async def test_task_dependencies(self, task_queue):
        """Test task dependency handling."""
        await task_queue.start()
        
        # Submit parent task
        parent_task_id = await task_queue.submit_task(
            task_type="parent", payload={}
        )
        
        # Submit dependent task
        child_task_id = await task_queue.submit_task(
            task_type="child", payload={}, dependencies=[parent_task_id]
        )
        
        # Child task should be pending until parent completes
        child_task = await task_queue.get_task(child_task_id)
        assert child_task.status == TaskStatus.PENDING
        
        # Complete parent task
        await task_queue.complete_task(parent_task_id, {"result": "success"})
        
        # Wait for dependency check
        await asyncio.sleep(0.1)
        
        # Child task should now be queued
        child_task = await task_queue.get_task(child_task_id)
        assert child_task.status == TaskStatus.QUEUED
        
        await task_queue.stop()
    
    @pytest.mark.asyncio
    async def test_task_retry_mechanism(self, task_queue):
        """Test task retry mechanism."""
        await task_queue.start()
        
        task_id = await task_queue.submit_task(
            task_type="test", payload={}, max_retries=2
        )
        
        # Get and assign task
        task = await task_queue.get_next_task()
        await task_queue.assign_task(task_id, "worker_1")
        
        # Fail task (should retry)
        await task_queue.fail_task(task_id, "Test error")
        
        # Task should be queued for retry
        task = await task_queue.get_task(task_id)
        assert task.status == TaskStatus.QUEUED
        assert task.retry_count == 1
        
        await task_queue.stop()
    
    @pytest.mark.asyncio
    async def test_queue_statistics(self, task_queue):
        """Test queue statistics."""
        await task_queue.start()
        
        # Submit various tasks
        for i in range(5):
            await task_queue.submit_task(
                task_type=f"type_{i % 3}", payload={}
            )
        
        stats = await task_queue.get_queue_stats()
        
        assert 'pending' in stats
        assert 'queued' in stats or 'assigned' in stats
        assert 'running' in stats
        assert 'completed' in stats
        assert 'failed' in stats
        assert 'total_active' in stats
        assert 'task_types' in stats
        
        await task_queue.stop()


class TestWorkflowCoordination:
    """Test workflow coordination functionality."""
    
    @pytest.fixture
    def coordination_service(self):
        """Create a coordination service instance."""
        task_queue = TaskQueue()
        worker_pool = WorkerPool(task_queue)
        processing_manager = DistributedProcessingManager()
        detection_coordinator = DetectionCoordinator(processing_manager)
        
        return CoordinationService(
            task_queue=task_queue,
            worker_pool=worker_pool,
            processing_manager=processing_manager,
            detection_coordinator=detection_coordinator
        )
    
    @pytest.mark.asyncio
    async def test_workflow_creation(self, coordination_service):
        """Test workflow creation."""
        workflow_steps = [
            {
                "id": "step1",
                "type": "data_loading",
                "name": "Load Data",
                "parameters": {"dataset_id": "test_dataset"}
            },
            {
                "id": "step2",
                "type": "anomaly_detection",
                "name": "Detect Anomalies",
                "dependencies": ["step1"],
                "parameters": {"algorithm": "isolation_forest"}
            }
        ]
        
        workflow_id = await coordination_service.create_workflow(
            name="Test Workflow",
            description="Test workflow for anomaly detection",
            steps=workflow_steps
        )
        
        assert workflow_id is not None
        
        # Get workflow status
        status = await coordination_service.get_workflow_status(workflow_id)
        assert status is not None
        assert status['name'] == "Test Workflow"
        assert status['status'] == WorkflowStatus.PENDING.value
        assert len(status['steps']) == 2
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, coordination_service):
        """Test workflow execution."""
        workflow_steps = [
            {
                "id": "step1",
                "type": "test_task",
                "name": "Test Step",
                "parameters": {"test": "value"}
            }
        ]
        
        workflow_id = await coordination_service.create_workflow(
            name="Test Workflow",
            description="Simple test workflow",
            steps=workflow_steps
        )
        
        # Start workflow execution
        success = await coordination_service.start_workflow(workflow_id)
        assert success
        
        # Check workflow status
        status = await coordination_service.get_workflow_status(workflow_id)
        assert status['status'] == WorkflowStatus.RUNNING.value
    
    @pytest.mark.asyncio
    async def test_workflow_templates(self, coordination_service):
        """Test workflow template functionality."""
        # Create workflow from template
        parameters = {
            "dataset_id": "test_dataset",
            "detector_id": "test_detector",
            "algorithm": "isolation_forest",
            "output_format": "json"
        }
        
        workflow_id = await coordination_service.create_workflow_from_template(
            template_name="basic_detection",
            parameters=parameters
        )
        
        assert workflow_id is not None
        
        # Verify workflow was created correctly
        status = await coordination_service.get_workflow_status(workflow_id)
        assert status is not None
        assert "Basic Anomaly Detection" in status['name']
        assert len(status['steps']) == 3  # load_data, detect_anomalies, save_results
    
    @pytest.mark.asyncio
    async def test_workflow_cancellation(self, coordination_service):
        """Test workflow cancellation."""
        workflow_steps = [
            {
                "id": "step1",
                "type": "long_running_task",
                "name": "Long Running Task",
                "parameters": {}
            }
        ]
        
        workflow_id = await coordination_service.create_workflow(
            name="Cancellable Workflow",
            description="Test workflow cancellation",
            steps=workflow_steps
        )
        
        # Start workflow
        await coordination_service.start_workflow(workflow_id)
        
        # Cancel workflow
        success = await coordination_service.cancel_workflow(workflow_id)
        assert success
        
        # Check workflow status
        status = await coordination_service.get_workflow_status(workflow_id)
        assert status['status'] == WorkflowStatus.CANCELLED.value


class TestLoadBalancer:
    """Test LoadBalancer functionality."""
    
    @pytest.fixture
    def load_balancer(self):
        """Create a load balancer instance."""
        return LoadBalancer(strategy="round_robin")
    
    def test_load_balancer_initialization(self, load_balancer):
        """Test load balancer initialization."""
        assert load_balancer.strategy == "round_robin"
        assert len(load_balancer.workers) == 0
    
    @pytest.mark.asyncio
    async def test_worker_registration(self, load_balancer):
        """Test worker registration with load balancer."""
        worker_info = {
            "id": "worker_1",
            "host": "localhost",
            "port": 8000,
            "capacity": 10,
            "current_load": 0,
            "capabilities": ["sklearn"]
        }
        
        await load_balancer.register_worker(worker_info)
        assert "worker_1" in load_balancer.workers
        assert load_balancer.workers["worker_1"] == worker_info
    
    @pytest.mark.asyncio
    async def test_round_robin_selection(self, load_balancer):
        """Test round-robin worker selection."""
        # Register multiple workers
        for i in range(3):
            worker_info = {
                "id": f"worker_{i}",
                "host": "localhost",
                "port": 8000 + i,
                "capacity": 10,
                "current_load": 0,
                "capabilities": ["sklearn"]
            }
            await load_balancer.register_worker(worker_info)
        
        # Select workers in round-robin fashion
        selected_workers = []
        for _ in range(6):
            worker = await load_balancer.select_worker()
            selected_workers.append(worker["id"])
        
        # Should cycle through workers
        expected = ["worker_0", "worker_1", "worker_2", "worker_0", "worker_1", "worker_2"]
        assert selected_workers == expected
    
    @pytest.mark.asyncio
    async def test_least_connections_selection(self, load_balancer):
        """Test least connections worker selection."""
        load_balancer.strategy = "least_connections"
        
        # Register workers with different loads
        worker_loads = [5, 2, 8]
        for i, load in enumerate(worker_loads):
            worker_info = {
                "id": f"worker_{i}",
                "host": "localhost",
                "port": 8000 + i,
                "capacity": 10,
                "current_load": load,
                "capabilities": ["sklearn"]
            }
            await load_balancer.register_worker(worker_info)
        
        # Should select worker with least connections (worker_1 with load 2)
        worker = await load_balancer.select_worker()
        assert worker["id"] == "worker_1"
        assert worker["current_load"] == 2
    
    @pytest.mark.asyncio
    async def test_capability_filtering(self, load_balancer):
        """Test worker selection with capability filtering."""
        # Register workers with different capabilities
        workers = [
            {"id": "worker_1", "capabilities": ["sklearn"]},
            {"id": "worker_2", "capabilities": ["pytorch"]},
            {"id": "worker_3", "capabilities": ["sklearn", "pytorch"]}
        ]
        
        for worker in workers:
            worker_info = {
                "host": "localhost",
                "port": 8000,
                "capacity": 10,
                "current_load": 0,
                **worker
            }
            await load_balancer.register_worker(worker_info)
        
        # Select worker with specific capabilities
        worker = await load_balancer.select_worker(required_capabilities=["pytorch"])
        assert worker["id"] in ["worker_2", "worker_3"]
        assert "pytorch" in worker["capabilities"]


class TestIntegrationScenarios:
    """Integration tests for distributed processing scenarios."""
    
    @pytest.fixture
    def distributed_system(self):
        """Create a complete distributed system setup."""
        task_queue = TaskQueue()
        worker_pool = WorkerPool(task_queue)
        processing_manager = DistributedProcessingManager()
        detection_coordinator = DetectionCoordinator(processing_manager)
        coordination_service = CoordinationService(
            task_queue, worker_pool, processing_manager, detection_coordinator
        )
        
        return {
            'task_queue': task_queue,
            'worker_pool': worker_pool,
            'processing_manager': processing_manager,
            'detection_coordinator': detection_coordinator,
            'coordination_service': coordination_service
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, distributed_system):
        """Test complete end-to-end workflow execution."""
        coordination_service = distributed_system['coordination_service']
        
        # Create and execute a workflow
        workflow_steps = [
            {
                "id": "load_data",
                "type": "data_loading",
                "name": "Load Dataset",
                "parameters": {"dataset_id": "test_dataset"}
            },
            {
                "id": "detect_anomalies",
                "type": "anomaly_detection",
                "name": "Detect Anomalies",
                "dependencies": ["load_data"],
                "parameters": {"algorithm": "isolation_forest"}
            }
        ]
        
        workflow_id = await coordination_service.create_workflow(
            name="End-to-End Test",
            description="Complete workflow test",
            steps=workflow_steps
        )
        
        # Start workflow
        success = await coordination_service.start_workflow(workflow_id)
        assert success
        
        # Verify workflow is running
        status = await coordination_service.get_workflow_status(workflow_id)
        assert status['status'] == WorkflowStatus.RUNNING.value
    
    @pytest.mark.asyncio
    async def test_system_metrics_collection(self, distributed_system):
        """Test comprehensive system metrics collection."""
        coordination_service = distributed_system['coordination_service']
        
        # Start coordination service
        await coordination_service.start()
        
        # Get system metrics
        metrics = await coordination_service.get_system_metrics()
        
        # Verify metrics structure
        assert 'workflows' in metrics
        assert 'task_queue' in metrics
        assert 'worker_pool' in metrics
        assert 'processing_manager' in metrics
        assert 'detection_coordinator' in metrics
        
        # Verify workflow metrics
        workflow_metrics = metrics['workflows']
        assert 'active' in workflow_metrics
        assert 'completed' in workflow_metrics
        assert 'total' in workflow_metrics
        
        await coordination_service.stop()
    
    @pytest.mark.asyncio
    async def test_fault_tolerance(self, distributed_system):
        """Test system fault tolerance and recovery."""
        task_queue = distributed_system['task_queue']
        worker_pool = distributed_system['worker_pool']
        
        await task_queue.start()
        await worker_pool.start()
        
        # Add workers
        for i in range(3):
            await worker_pool.add_worker(
                worker_id=f"worker_{i}",
                host="localhost",
                port=8000 + i,
                capacity=5
            )
        
        # Submit tasks
        task_ids = []
        for i in range(10):
            task_id = await task_queue.submit_task(
                task_type="test_task",
                payload={"data": f"test_{i}"}
            )
            task_ids.append(task_id)
        
        # Simulate worker failure
        await worker_pool.remove_worker("worker_0")
        
        # System should continue operating
        stats = await worker_pool.get_worker_stats()
        assert stats['total_workers'] == 2
        
        await task_queue.stop()
        await worker_pool.stop()