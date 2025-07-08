"""System health schemas for monitoring system resources and performance.

This module provides Pydantic schemas for system health monitoring including
resource utilization, performance metrics, and system status indicators.

Schemas:
    SystemHealthFrame: Main system health frame
    SystemResourceMetrics: Resource utilization metrics
    SystemPerformanceMetrics: Performance metrics
    SystemStatusMetrics: System status indicators
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime

from pydantic import BaseModel, Field, validator

from .base import RealTimeMetricFrame


class SystemStatus(str, Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"


class ServiceStatus(str, Enum):
    """Service status levels."""
    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    STOPPING = "stopping"
    ERROR = "error"


class SystemResourceMetrics(BaseModel):
    """System resource utilization metrics."""
    
    # CPU metrics
    cpu_usage_percent: float = Field(ge=0.0, le=100.0, description="CPU usage percentage")
    cpu_load_average: float = Field(ge=0.0, description="CPU load average")
    cpu_cores: int = Field(gt=0, description="Number of CPU cores")
    cpu_frequency: Optional[float] = Field(None, ge=0.0, description="CPU frequency in GHz")
    
    # Memory metrics
    memory_usage_percent: float = Field(ge=0.0, le=100.0, description="Memory usage percentage")
    memory_used_mb: float = Field(ge=0.0, description="Memory used in MB")
    memory_total_mb: float = Field(gt=0.0, description="Total memory in MB")
    memory_available_mb: float = Field(ge=0.0, description="Available memory in MB")
    
    # Disk metrics
    disk_usage_percent: float = Field(ge=0.0, le=100.0, description="Disk usage percentage")
    disk_used_gb: float = Field(ge=0.0, description="Disk used in GB")
    disk_total_gb: float = Field(gt=0.0, description="Total disk space in GB")
    disk_io_read_rate: float = Field(ge=0.0, description="Disk read rate in MB/s")
    disk_io_write_rate: float = Field(ge=0.0, description="Disk write rate in MB/s")
    
    # Network metrics
    network_bytes_sent_rate: float = Field(ge=0.0, description="Network bytes sent rate in MB/s")
    network_bytes_recv_rate: float = Field(ge=0.0, description="Network bytes received rate in MB/s")
    network_packets_sent_rate: float = Field(ge=0.0, description="Network packets sent rate per second")
    network_packets_recv_rate: float = Field(ge=0.0, description="Network packets received rate per second")
    
    # Process metrics
    process_count: int = Field(ge=0, description="Number of running processes")
    thread_count: int = Field(ge=0, description="Number of active threads")
    
    @validator('memory_used_mb')
    def validate_memory_used(cls, v: float, values: Dict[str, Any]) -> float:
        """Validate memory used doesn't exceed total."""
        if 'memory_total_mb' in values and v > values['memory_total_mb']:
            raise ValueError("Memory used cannot exceed total memory")
        return v
    
    @validator('disk_used_gb')
    def validate_disk_used(cls, v: float, values: Dict[str, Any]) -> float:
        """Validate disk used doesn't exceed total."""
        if 'disk_total_gb' in values and v > values['disk_total_gb']:
            raise ValueError("Disk used cannot exceed total disk space")
        return v


class SystemPerformanceMetrics(BaseModel):
    """System performance metrics."""
    
    # Response times
    avg_response_time_ms: float = Field(ge=0.0, description="Average response time in milliseconds")
    p95_response_time_ms: float = Field(ge=0.0, description="95th percentile response time in milliseconds")
    p99_response_time_ms: float = Field(ge=0.0, description="99th percentile response time in milliseconds")
    
    # Throughput metrics
    requests_per_second: float = Field(ge=0.0, description="Requests per second")
    transactions_per_second: float = Field(ge=0.0, description="Transactions per second")
    
    # Error metrics
    error_rate: float = Field(ge=0.0, le=1.0, description="Error rate (0-1)")
    timeout_rate: float = Field(ge=0.0, le=1.0, description="Timeout rate (0-1)")
    retry_rate: float = Field(ge=0.0, le=1.0, description="Retry rate (0-1)")
    
    # Database metrics
    database_connections: int = Field(ge=0, description="Number of database connections")
    database_query_time_ms: float = Field(ge=0.0, description="Average database query time in milliseconds")
    database_deadlocks: int = Field(ge=0, description="Number of database deadlocks")
    
    # Cache metrics
    cache_hit_rate: float = Field(ge=0.0, le=1.0, description="Cache hit rate (0-1)")
    cache_miss_rate: float = Field(ge=0.0, le=1.0, description="Cache miss rate (0-1)")
    
    # Queue metrics
    queue_depth: int = Field(ge=0, description="Queue depth")
    queue_processing_time_ms: float = Field(ge=0.0, description="Queue processing time in milliseconds")
    
    @validator('p95_response_time_ms')
    def validate_p95_response_time(cls, v: float, values: Dict[str, Any]) -> float:
        """Validate P95 response time is >= average."""
        if 'avg_response_time_ms' in values and v < values['avg_response_time_ms']:
            raise ValueError("P95 response time cannot be less than average response time")
        return v
    
    @validator('p99_response_time_ms')
    def validate_p99_response_time(cls, v: float, values: Dict[str, Any]) -> float:
        """Validate P99 response time is >= P95."""
        if 'p95_response_time_ms' in values and v < values['p95_response_time_ms']:
            raise ValueError("P99 response time cannot be less than P95 response time")
        return v


class SystemStatusMetrics(BaseModel):
    """System status indicators."""
    
    # Overall system status
    system_status: SystemStatus = Field(description="Overall system status")
    uptime_seconds: float = Field(ge=0.0, description="System uptime in seconds")
    
    # Service health
    services_total: int = Field(ge=0, description="Total number of services")
    services_healthy: int = Field(ge=0, description="Number of healthy services")
    services_degraded: int = Field(ge=0, description="Number of degraded services")
    services_failed: int = Field(ge=0, description="Number of failed services")
    
    # Alert information
    active_alerts: int = Field(ge=0, description="Number of active alerts")
    critical_alerts: int = Field(ge=0, description="Number of critical alerts")
    warning_alerts: int = Field(ge=0, description="Number of warning alerts")
    
    # Maintenance and deployment
    maintenance_mode: bool = Field(default=False, description="Whether system is in maintenance mode")
    deployment_in_progress: bool = Field(default=False, description="Whether deployment is in progress")
    
    # Security status
    security_scan_passed: bool = Field(default=True, description="Whether security scan passed")
    vulnerability_count: int = Field(ge=0, description="Number of known vulnerabilities")
    
    # Backup status
    last_backup_timestamp: Optional[datetime] = Field(None, description="Last backup timestamp")
    backup_status: Optional[str] = Field(None, description="Backup status")
    
    @validator('services_healthy')
    def validate_services_healthy(cls, v: int, values: Dict[str, Any]) -> int:
        """Validate healthy services don't exceed total."""
        if 'services_total' in values and v > values['services_total']:
            raise ValueError("Healthy services cannot exceed total services")
        return v
    
    @validator('critical_alerts')
    def validate_critical_alerts(cls, v: int, values: Dict[str, Any]) -> int:
        """Validate critical alerts don't exceed active alerts."""
        if 'active_alerts' in values and v > values['active_alerts']:
            raise ValueError("Critical alerts cannot exceed active alerts")
        return v


class SystemHealthFrame(RealTimeMetricFrame):
    """Main system health frame containing all system health metrics."""
    
    # Core health metrics
    resource_metrics: SystemResourceMetrics
    performance_metrics: SystemPerformanceMetrics
    status_metrics: SystemStatusMetrics
    
    # System identification
    hostname: str = Field(description="System hostname")
    environment: str = Field(description="Environment (production, staging, etc.)")
    region: Optional[str] = Field(None, description="Deployment region")
    
    # Health scores
    overall_health_score: float = Field(ge=0.0, le=1.0, description="Overall health score (0-1)")
    availability_score: float = Field(ge=0.0, le=1.0, description="Availability score (0-1)")
    reliability_score: float = Field(ge=0.0, le=1.0, description="Reliability score (0-1)")
    
    # Capacity planning
    capacity_utilization: float = Field(ge=0.0, le=1.0, description="Capacity utilization (0-1)")
    estimated_capacity_exhaustion: Optional[datetime] = Field(None, description="Estimated capacity exhaustion time")
    
    # Trend indicators
    cpu_trend: Optional[str] = Field(None, description="CPU usage trend (increasing, decreasing, stable)")
    memory_trend: Optional[str] = Field(None, description="Memory usage trend")
    disk_trend: Optional[str] = Field(None, description="Disk usage trend")
    
    # Additional context
    configuration_version: Optional[str] = Field(None, description="Configuration version")
    deployment_version: Optional[str] = Field(None, description="Deployment version")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"
    
    def get_resource_health_score(self) -> float:
        """Calculate resource health score based on utilization."""
        cpu_score = max(0.0, 1.0 - (self.resource_metrics.cpu_usage_percent / 100.0))
        memory_score = max(0.0, 1.0 - (self.resource_metrics.memory_usage_percent / 100.0))
        disk_score = max(0.0, 1.0 - (self.resource_metrics.disk_usage_percent / 100.0))
        
        return (cpu_score + memory_score + disk_score) / 3.0
    
    def get_performance_health_score(self) -> float:
        """Calculate performance health score."""
        # Lower response times are better
        response_score = max(0.0, 1.0 - min(1.0, self.performance_metrics.avg_response_time_ms / 1000.0))
        
        # Lower error rates are better
        error_score = max(0.0, 1.0 - self.performance_metrics.error_rate)
        
        # Higher cache hit rates are better
        cache_score = self.performance_metrics.cache_hit_rate
        
        return (response_score + error_score + cache_score) / 3.0
    
    def get_status_health_score(self) -> float:
        """Calculate status health score."""
        if self.status_metrics.services_total == 0:
            return 1.0
        
        service_health_ratio = self.status_metrics.services_healthy / self.status_metrics.services_total
        
        # Penalize for critical alerts
        alert_penalty = min(1.0, self.status_metrics.critical_alerts * 0.1)
        
        # Penalize for maintenance mode
        maintenance_penalty = 0.2 if self.status_metrics.maintenance_mode else 0.0
        
        return max(0.0, service_health_ratio - alert_penalty - maintenance_penalty)
    
    def is_healthy(self) -> bool:
        """Check if the system is healthy."""
        return all([
            self.overall_health_score >= 0.8,
            self.status_metrics.system_status in [SystemStatus.HEALTHY, SystemStatus.WARNING],
            self.resource_metrics.cpu_usage_percent <= 80.0,
            self.resource_metrics.memory_usage_percent <= 80.0,
            self.resource_metrics.disk_usage_percent <= 80.0,
            self.performance_metrics.error_rate <= 0.05,
            self.status_metrics.critical_alerts == 0,
        ])
    
    def needs_attention(self) -> bool:
        """Check if the system needs attention."""
        return any([
            self.overall_health_score < 0.8,
            self.status_metrics.system_status in [SystemStatus.CRITICAL, SystemStatus.DEGRADED],
            self.resource_metrics.cpu_usage_percent > 80.0,
            self.resource_metrics.memory_usage_percent > 80.0,
            self.resource_metrics.disk_usage_percent > 80.0,
            self.performance_metrics.error_rate > 0.05,
            self.status_metrics.critical_alerts > 0,
        ])
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of system health."""
        return {
            "overall_health_score": self.overall_health_score,
            "resource_health_score": self.get_resource_health_score(),
            "performance_health_score": self.get_performance_health_score(),
            "status_health_score": self.get_status_health_score(),
            "system_status": self.status_metrics.system_status.value,
            "is_healthy": self.is_healthy(),
            "needs_attention": self.needs_attention(),
            "uptime_hours": self.status_metrics.uptime_seconds / 3600.0,
            "active_alerts": self.status_metrics.active_alerts,
            "critical_alerts": self.status_metrics.critical_alerts,
        }
