"""
Pytest configuration for Enterprise Scalability package testing.
Provides fixtures for distributed computing, streaming, and scalability testing.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import time
from uuid import uuid4, UUID
from datetime import datetime, timedelta
from unittest.mock import Mock
import asyncio


@pytest.fixture
def sample_cluster_config() -> Dict[str, Any]:
    """Sample compute cluster configuration."""
    return {
        'name': 'anomaly-detection-cluster',
        'cluster_type': 'dask',
        'min_nodes': 2,
        'max_nodes': 10,
        'node_config': {
            'cpu_cores': 4,
            'memory_gb': 16,
            'instance_type': 'compute-optimized',
            'storage_gb': 100
        },
        'auto_scale_config': {
            'enabled': True,
            'scale_up_threshold': 0.8,
            'scale_down_threshold': 0.3,
            'cooldown_minutes': 5
        },
        'networking': {
            'vpc_id': 'vpc-12345',
            'subnet_ids': ['subnet-123', 'subnet-456'],
            'security_groups': ['sg-compute']
        }
    }


@pytest.fixture
def sample_stream_processor_config() -> Dict[str, Any]:
    """Sample stream processor configuration."""
    return {
        'name': 'real-time-anomaly-stream',
        'sources': [
            {
                'name': 'kafka_transactions',
                'type': 'kafka',
                'connection_string': 'kafka.analytics.local:9092',
                'topics': ['customer-transactions', 'system-events'],
                'consumer_group': 'anomaly-detection',
                'format': 'json'
            },
            {
                'name': 'kinesis_logs',
                'type': 'kinesis',
                'connection_string': 'https://kinesis.us-east-1.amazonaws.com',
                'stream_name': 'application-logs',
                'format': 'json'
            }
        ],
        'sinks': [
            {
                'name': 'results_database',
                'type': 'postgresql',
                'connection_string': 'postgresql://analytics:password@db.local:5432/results',
                'destination': 'anomaly_results',
                'format': 'json'
            },
            {
                'name': 'alerts_queue',
                'type': 'sqs',
                'connection_string': 'https://sqs.us-east-1.amazonaws.com/123456789/alerts',
                'destination': 'high-severity-alerts',
                'format': 'json'
            }
        ],
        'processing_logic': """
        def process_record(record):
            # Extract features
            amount = record.get('transaction_amount', 0)
            velocity = record.get('transaction_velocity', 0)
            
            # Simple anomaly detection
            anomaly_score = calculate_anomaly_score(amount, velocity)
            
            return {
                'customer_id': record.get('customer_id'),
                'anomaly_score': anomaly_score,
                'is_anomaly': anomaly_score > 0.8,
                'processed_at': datetime.utcnow().isoformat()
            }
        """,
        'parallelism': 4,
        'checkpointing': {
            'enabled': True,
            'interval_seconds': 30,
            'storage_path': 's3://checkpoints/anomaly-stream'
        }
    }


@pytest.fixture
def distributed_task_configs() -> List[Dict[str, Any]]:
    """Sample distributed task configurations."""
    return [
        {
            'function_name': 'batch_anomaly_detection',
            'module_name': 'anomaly_detection.batch_processing',
            'args': ['customer_data.csv'],
            'kwargs': {'contamination': 0.05, 'n_estimators': 100},
            'type': 'anomaly_detection',
            'priority': 'high',
            'resources': {
                'cpu_cores': 4,
                'memory_gb': 8,
                'estimated_duration_minutes': 30
            }
        },
        {
            'function_name': 'feature_engineering',
            'module_name': 'data_processing.feature_engineering',
            'args': ['raw_data.parquet'],
            'kwargs': {'window_size': '24h', 'feature_set': 'comprehensive'},
            'type': 'data_processing',
            'priority': 'normal',
            'resources': {
                'cpu_cores': 2,
                'memory_gb': 4,
                'estimated_duration_minutes': 15
            }
        },
        {
            'function_name': 'model_training',
            'module_name': 'ml_pipeline.training',
            'args': ['training_dataset.csv'],
            'kwargs': {'algorithm': 'isolation_forest', 'cv_folds': 5},
            'type': 'model_training',
            'priority': 'high',
            'resources': {
                'cpu_cores': 8,
                'memory_gb': 16,
                'estimated_duration_minutes': 60,
                'gpu_required': False
            }
        }
    ]


@pytest.fixture
def scalability_test_scenarios() -> List[Dict[str, Any]]:
    """Scalability test scenarios."""
    return [
        {
            'scenario_name': 'horizontal_scaling_stress_test',
            'description': 'Test horizontal scaling under increasing load',
            'initial_nodes': 2,
            'max_nodes': 20,
            'load_pattern': 'linear_increase',
            'duration_minutes': 30,
            'target_throughput': 10000,  # requests per minute
            'expected_scale_triggers': 5
        },
        {
            'scenario_name': 'burst_traffic_handling',
            'description': 'Test handling sudden traffic spikes',
            'initial_nodes': 3,
            'max_nodes': 15,
            'load_pattern': 'sudden_spike',
            'duration_minutes': 20,
            'spike_multiplier': 10,
            'recovery_time_target_seconds': 300
        },
        {
            'scenario_name': 'sustained_high_load',
            'description': 'Test sustained high-load performance',
            'initial_nodes': 8,
            'max_nodes': 12,
            'load_pattern': 'sustained_high',
            'duration_minutes': 60,
            'target_throughput': 50000,
            'stability_threshold': 0.95
        },
        {
            'scenario_name': 'graceful_degradation',
            'description': 'Test graceful degradation under node failures',
            'initial_nodes': 10,
            'max_nodes': 10,
            'failure_simulation': True,
            'nodes_to_fail': 3,
            'expected_performance_retention': 0.7
        }
    ]


@pytest.fixture
def performance_benchmarks() -> Dict[str, Any]:
    """Performance benchmarks for scalability testing."""
    return {
        'cluster_provisioning': {
            'node_startup_time_seconds': 120,
            'cluster_ready_time_seconds': 300,
            'auto_scale_response_time_seconds': 180
        },
        'stream_processing': {
            'throughput_messages_per_second': 10000,
            'latency_p95_milliseconds': 100,
            'processing_accuracy': 0.999,
            'backpressure_recovery_seconds': 30
        },
        'distributed_tasks': {
            'task_scheduling_time_milliseconds': 50,
            'task_completion_rate': 0.98,
            'resource_utilization_target': 0.85,
            'fault_tolerance_rate': 0.95
        },
        'overall_system': {
            'availability_target': 0.9999,
            'response_time_p99_milliseconds': 500,
            'concurrent_users_supported': 10000,
            'data_processing_throughput_gb_per_hour': 1000
        }
    }


@pytest.fixture
def mock_compute_cluster():
    """Mock compute cluster for testing."""
    class MockComputeCluster:
        def __init__(self, cluster_id: UUID = None):
            self.id = cluster_id or uuid4()
            self.name = 'test-cluster'
            self.cluster_type = 'dask'
            self.status = 'running'
            self.current_nodes = 2
            self.min_nodes = 1
            self.max_nodes = 10
            self.total_cpu_cores = 8
            self.total_memory_gb = 32.0
            self.cpu_utilization = 0.3
            self.memory_utilization = 0.4
            self.auto_scale_enabled = True
            self.nodes = []
            
        def scale(self, target_nodes: int) -> Dict[str, Any]:
            if target_nodes < self.min_nodes or target_nodes > self.max_nodes:
                return {'success': False, 'error': 'Target nodes out of range'}
                
            old_nodes = self.current_nodes
            self.current_nodes = target_nodes
            
            # Update resources proportionally
            self.total_cpu_cores = target_nodes * 4
            self.total_memory_gb = target_nodes * 16.0
            
            return {
                'success': True,
                'old_nodes': old_nodes,
                'new_nodes': target_nodes,
                'scaling_time': np.random.uniform(30, 180)  # 30s to 3min
            }
            
        def get_metrics(self) -> Dict[str, Any]:
            return {
                'cluster_id': str(self.id),
                'status': self.status,
                'current_nodes': self.current_nodes,
                'cpu_utilization': self.cpu_utilization,
                'memory_utilization': self.memory_utilization,
                'total_cpu_cores': self.total_cpu_cores,
                'total_memory_gb': self.total_memory_gb,
                'tasks_running': np.random.randint(0, 20),
                'tasks_pending': np.random.randint(0, 10)
            }
            
        def simulate_load(self, load_factor: float):
            """Simulate load on the cluster."""
            self.cpu_utilization = min(0.95, load_factor * 0.8)
            self.memory_utilization = min(0.95, load_factor * 0.7)
            
        def should_scale_up(self) -> bool:
            return self.cpu_utilization > 0.8 or self.memory_utilization > 0.8
            
        def should_scale_down(self) -> bool:
            return self.cpu_utilization < 0.3 and self.memory_utilization < 0.3 and self.current_nodes > self.min_nodes
    
    return MockComputeCluster()


@pytest.fixture
def mock_stream_processor():
    """Mock stream processor for testing."""
    class MockStreamProcessor:
        def __init__(self, processor_id: UUID = None):
            self.id = processor_id or uuid4()
            self.name = 'test-stream-processor'
            self.status = 'stopped'
            self.current_parallelism = 1
            self.max_parallelism = 10
            self.throughput_per_second = 0
            self.latency_ms = 0
            self.backlog_size = 0
            self.error_rate = 0.0
            self.auto_scaling_enabled = True
            
        def start(self) -> Dict[str, Any]:
            self.status = 'running'
            self.throughput_per_second = 1000
            self.latency_ms = 50
            return {'success': True, 'startup_time': np.random.uniform(10, 60)}
            
        def stop(self) -> Dict[str, Any]:
            self.status = 'stopped'
            self.throughput_per_second = 0
            return {'success': True}
            
        def scale(self, target_parallelism: int) -> Dict[str, Any]:
            if target_parallelism < 1 or target_parallelism > self.max_parallelism:
                return {'success': False, 'error': 'Target parallelism out of range'}
                
            old_parallelism = self.current_parallelism
            self.current_parallelism = target_parallelism
            
            # Update performance metrics proportionally
            self.throughput_per_second = target_parallelism * 1000
            
            return {
                'success': True,
                'old_parallelism': old_parallelism,
                'new_parallelism': target_parallelism,
                'scaling_time': np.random.uniform(10, 30)
            }
            
        def get_metrics(self) -> Dict[str, Any]:
            return {
                'processor_id': str(self.id),
                'status': self.status,
                'current_parallelism': self.current_parallelism,
                'throughput_per_second': self.throughput_per_second,
                'latency_ms': self.latency_ms,
                'backlog_size': self.backlog_size,
                'error_rate': self.error_rate,
                'cpu_utilization': np.random.uniform(0.2, 0.8),
                'memory_utilization': np.random.uniform(0.3, 0.7)
            }
            
        def simulate_load(self, load_factor: float):
            """Simulate processing load."""
            self.backlog_size = int(load_factor * 10000)
            self.latency_ms = 50 + (load_factor * 100)  # Latency increases with load
            self.error_rate = min(0.05, load_factor * 0.02)  # Error rate increases with load
            
        def should_scale_up(self) -> bool:
            return self.backlog_size > 5000 or self.latency_ms > 100
            
        def should_scale_down(self) -> bool:
            return self.backlog_size < 1000 and self.latency_ms < 75 and self.current_parallelism > 1
    
    return MockStreamProcessor()


@pytest.fixture
def mock_task_scheduler():
    """Mock task scheduler for testing."""
    class MockTaskScheduler:
        def __init__(self):
            self.tasks = {}
            self.task_queue = []
            self.running_tasks = {}
            
        def submit_task(self, task_config: Dict[str, Any]) -> UUID:
            task_id = uuid4()
            task = {
                'id': task_id,
                'status': 'pending',
                'submitted_at': datetime.utcnow(),
                'started_at': None,
                'completed_at': None,
                'result': None,
                'error': None,
                **task_config
            }
            
            self.tasks[task_id] = task
            self.task_queue.append(task_id)
            
            return task_id
            
        def execute_tasks(self, max_concurrent: int = 5) -> Dict[str, Any]:
            """Simulate task execution."""
            executed_tasks = 0
            
            while self.task_queue and len(self.running_tasks) < max_concurrent:
                task_id = self.task_queue.pop(0)
                task = self.tasks[task_id]
                
                # Start task
                task['status'] = 'running'
                task['started_at'] = datetime.utcnow()
                self.running_tasks[task_id] = task
                executed_tasks += 1
                
            # Complete some running tasks
            completed_task_ids = []
            for task_id, task in self.running_tasks.items():
                if np.random.random() > 0.7:  # 30% chance to complete per execution cycle
                    task['status'] = 'completed' if np.random.random() > 0.1 else 'failed'
                    task['completed_at'] = datetime.utcnow()
                    
                    if task['status'] == 'completed':
                        task['result'] = {'processed_rows': np.random.randint(1000, 10000)}
                    else:
                        task['error'] = 'Simulated task failure'
                        
                    completed_task_ids.append(task_id)
                    
            for task_id in completed_task_ids:
                del self.running_tasks[task_id]
                
            return {
                'tasks_executed': executed_tasks,
                'tasks_completed': len(completed_task_ids),
                'tasks_pending': len(self.task_queue),
                'tasks_running': len(self.running_tasks)
            }
            
        def get_task_status(self, task_id: UUID) -> Dict[str, Any]:
            return self.tasks.get(task_id, {'error': 'Task not found'})
            
        def get_system_metrics(self) -> Dict[str, Any]:
            total_tasks = len(self.tasks)
            completed_tasks = sum(1 for t in self.tasks.values() if t['status'] == 'completed')
            failed_tasks = sum(1 for t in self.tasks.values() if t['status'] == 'failed')
            
            return {
                'total_tasks': total_tasks,
                'completed_tasks': completed_tasks,
                'failed_tasks': failed_tasks,
                'success_rate': completed_tasks / total_tasks if total_tasks > 0 else 0,
                'tasks_pending': len(self.task_queue),
                'tasks_running': len(self.running_tasks)
            }
    
    return MockTaskScheduler()


@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            
        def start(self):
            self.start_time = time.perf_counter()
            
        def stop(self):
            self.end_time = time.perf_counter()
            
        @property
        def elapsed(self) -> float:
            if self.start_time is None or self.end_time is None:
                return 0.0
            return self.end_time - self.start_time
    
    return Timer()


@pytest.fixture
def large_workload_simulation():
    """Generate large workload simulation data."""
    def generate_workload(duration_minutes: int, base_load: int, pattern: str = 'linear'):
        timestamps = pd.date_range(
            start=datetime.utcnow(),
            end=datetime.utcnow() + timedelta(minutes=duration_minutes),
            freq='1min'
        )
        
        if pattern == 'linear':
            loads = np.linspace(base_load, base_load * 5, len(timestamps))
        elif pattern == 'spike':
            loads = np.full(len(timestamps), base_load)
            spike_start = len(timestamps) // 4
            spike_end = spike_start + len(timestamps) // 10
            loads[spike_start:spike_end] *= 10
        elif pattern == 'sustained_high':
            loads = np.full(len(timestamps), base_load * 8)
        else:
            loads = np.random.poisson(base_load, len(timestamps))
            
        return pd.DataFrame({
            'timestamp': timestamps,
            'load_level': loads.astype(int),
            'cpu_demand': loads * 0.1,
            'memory_demand': loads * 0.05
        })
    
    return generate_workload


def pytest_configure(config):
    """Configure pytest markers for scalability testing."""
    markers = [
        "scalability: Enterprise scalability tests",
        "distributed_computing: Distributed computing tests",
        "stream_processing: Stream processing tests",
        "auto_scaling: Auto-scaling tests",
        "load_testing: Load and stress testing",
        "performance: Scalability performance tests",
        "integration: Scalability integration tests"
    ]
    
    for marker in markers:
        config.addinivalue_line("markers", marker)
