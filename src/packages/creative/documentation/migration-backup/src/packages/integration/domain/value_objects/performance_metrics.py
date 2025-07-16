"""
Performance metrics value objects for integration monitoring.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
from decimal import Decimal

from core.domain.abstractions.base_value_object import BaseValueObject


@dataclass(frozen=True)
class SystemMetrics(BaseValueObject):
    """System-level performance metrics."""
    
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    network_io_bytes_per_second: float
    disk_io_bytes_per_second: float
    active_connections: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate metrics after initialization."""
        if not (0 <= self.cpu_usage_percent <= 100):
            raise ValueError("CPU usage must be between 0 and 100")
        if not (0 <= self.memory_usage_percent <= 100):
            raise ValueError("Memory usage must be between 0 and 100")
        if not (0 <= self.disk_usage_percent <= 100):
            raise ValueError("Disk usage must be between 0 and 100")
        if self.network_io_bytes_per_second < 0:
            raise ValueError("Network I/O must be non-negative")
        if self.disk_io_bytes_per_second < 0:
            raise ValueError("Disk I/O must be non-negative")
        if self.active_connections < 0:
            raise ValueError("Active connections must be non-negative")


@dataclass(frozen=True)
class ApplicationMetrics(BaseValueObject):
    """Application-level performance metrics."""
    
    request_count: int
    error_count: int
    response_time_ms: float
    throughput_requests_per_second: float
    active_users: int
    memory_usage_mb: float
    cache_hit_rate: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate metrics after initialization."""
        if self.request_count < 0:
            raise ValueError("Request count must be non-negative")
        if self.error_count < 0:
            raise ValueError("Error count must be non-negative")
        if self.response_time_ms < 0:
            raise ValueError("Response time must be non-negative")
        if self.throughput_requests_per_second < 0:
            raise ValueError("Throughput must be non-negative")
        if self.active_users < 0:
            raise ValueError("Active users must be non-negative")
        if self.memory_usage_mb < 0:
            raise ValueError("Memory usage must be non-negative")
        if not (0 <= self.cache_hit_rate <= 1):
            raise ValueError("Cache hit rate must be between 0 and 1")
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate as a percentage."""
        if self.request_count == 0:
            return 0.0
        return (self.error_count / self.request_count) * 100
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        return 100.0 - self.error_rate


@dataclass(frozen=True)
class PackageMetrics(BaseValueObject):
    """Package-specific performance metrics."""
    
    package_name: str
    operation_count: int
    average_execution_time_ms: float
    max_execution_time_ms: float
    min_execution_time_ms: float
    success_count: int
    failure_count: int
    memory_usage_mb: float
    cpu_usage_percent: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate metrics after initialization."""
        if not self.package_name:
            raise ValueError("Package name cannot be empty")
        if self.operation_count < 0:
            raise ValueError("Operation count must be non-negative")
        if self.average_execution_time_ms < 0:
            raise ValueError("Average execution time must be non-negative")
        if self.max_execution_time_ms < 0:
            raise ValueError("Max execution time must be non-negative")
        if self.min_execution_time_ms < 0:
            raise ValueError("Min execution time must be non-negative")
        if self.success_count < 0:
            raise ValueError("Success count must be non-negative")
        if self.failure_count < 0:
            raise ValueError("Failure count must be non-negative")
        if self.memory_usage_mb < 0:
            raise ValueError("Memory usage must be non-negative")
        if not (0 <= self.cpu_usage_percent <= 100):
            raise ValueError("CPU usage must be between 0 and 100")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        total_operations = self.success_count + self.failure_count
        if total_operations == 0:
            return 0.0
        return (self.success_count / total_operations) * 100
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as a percentage."""
        return 100.0 - self.success_rate


@dataclass(frozen=True)
class WorkflowMetrics(BaseValueObject):
    """Workflow execution performance metrics."""
    
    workflow_id: str
    workflow_name: str
    total_steps: int
    completed_steps: int
    failed_steps: int
    execution_time_seconds: float
    throughput_steps_per_second: float
    memory_peak_mb: float
    data_processed_mb: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate metrics after initialization."""
        if not self.workflow_id:
            raise ValueError("Workflow ID cannot be empty")
        if not self.workflow_name:
            raise ValueError("Workflow name cannot be empty")
        if self.total_steps < 0:
            raise ValueError("Total steps must be non-negative")
        if self.completed_steps < 0:
            raise ValueError("Completed steps must be non-negative")
        if self.failed_steps < 0:
            raise ValueError("Failed steps must be non-negative")
        if self.execution_time_seconds < 0:
            raise ValueError("Execution time must be non-negative")
        if self.throughput_steps_per_second < 0:
            raise ValueError("Throughput must be non-negative")
        if self.memory_peak_mb < 0:
            raise ValueError("Memory peak must be non-negative")
        if self.data_processed_mb < 0:
            raise ValueError("Data processed must be non-negative")
    
    @property
    def completion_rate(self) -> float:
        """Calculate completion rate as a percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps / self.total_steps) * 100
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as a percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.failed_steps / self.total_steps) * 100
    
    @property
    def is_completed(self) -> bool:
        """Check if workflow is completed."""
        return self.completed_steps == self.total_steps
    
    @property
    def has_failures(self) -> bool:
        """Check if workflow has failures."""
        return self.failed_steps > 0


@dataclass(frozen=True)
class PerformanceMetrics(BaseValueObject):
    """Comprehensive performance metrics for integration monitoring."""
    
    system: SystemMetrics
    application: ApplicationMetrics
    packages: Dict[str, PackageMetrics] = field(default_factory=dict)
    workflows: Dict[str, WorkflowMetrics] = field(default_factory=dict)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def get_package_metrics(self, package_name: str) -> Optional[PackageMetrics]:
        """Get metrics for a specific package."""
        return self.packages.get(package_name)
    
    def get_workflow_metrics(self, workflow_id: str) -> Optional[WorkflowMetrics]:
        """Get metrics for a specific workflow."""
        return self.workflows.get(workflow_id)
    
    def get_overall_health_score(self) -> float:
        """Calculate overall health score (0-100)."""
        scores = []
        
        # System health (40% weight)
        system_score = (
            (100 - self.system.cpu_usage_percent) * 0.3 +
            (100 - self.system.memory_usage_percent) * 0.3 +
            (100 - self.system.disk_usage_percent) * 0.4
        )
        scores.append(system_score * 0.4)
        
        # Application health (30% weight)
        app_score = (
            self.application.success_rate * 0.5 +
            min(100, (1000 / max(1, self.application.response_time_ms)) * 100) * 0.3 +
            (self.application.cache_hit_rate * 100) * 0.2
        )
        scores.append(app_score * 0.3)
        
        # Package health (20% weight)
        if self.packages:
            package_scores = [pkg.success_rate for pkg in self.packages.values()]
            avg_package_score = sum(package_scores) / len(package_scores)
            scores.append(avg_package_score * 0.2)
        
        # Workflow health (10% weight)
        if self.workflows:
            workflow_scores = [wf.completion_rate for wf in self.workflows.values()]
            avg_workflow_score = sum(workflow_scores) / len(workflow_scores)
            scores.append(avg_workflow_score * 0.1)
        
        return sum(scores)
    
    def get_critical_alerts(self) -> List[str]:
        """Get list of critical performance alerts."""
        alerts = []
        
        # System alerts
        if self.system.cpu_usage_percent > 90:
            alerts.append(f"Critical: CPU usage at {self.system.cpu_usage_percent}%")
        if self.system.memory_usage_percent > 90:
            alerts.append(f"Critical: Memory usage at {self.system.memory_usage_percent}%")
        if self.system.disk_usage_percent > 95:
            alerts.append(f"Critical: Disk usage at {self.system.disk_usage_percent}%")
        
        # Application alerts
        if self.application.error_rate > 10:
            alerts.append(f"Critical: Error rate at {self.application.error_rate}%")
        if self.application.response_time_ms > 10000:
            alerts.append(f"Critical: Response time at {self.application.response_time_ms}ms")
        
        # Package alerts
        for name, metrics in self.packages.items():
            if metrics.failure_rate > 20:
                alerts.append(f"Critical: Package {name} failure rate at {metrics.failure_rate}%")
        
        return alerts