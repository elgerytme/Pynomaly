"""
Pipeline Monitoring Infrastructure

Provides comprehensive monitoring and observability for ML pipelines including
execution tracking, performance metrics, and alerting capabilities.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

import psutil
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry
import structlog

from mlops.domain.entities.pipeline import Pipeline, PipelineRun, PipelineStatus, StageStatus


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PipelineAlert:
    """Pipeline monitoring alert."""
    id: str
    pipeline_id: str
    run_id: Optional[str] = None
    alert_type: str = ""
    severity: AlertSeverity = AlertSeverity.MEDIUM
    title: str = ""
    description: str = ""
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    is_resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringConfig:
    """Configuration for pipeline monitoring."""
    # Metrics collection
    enable_metrics: bool = True
    metrics_interval_seconds: int = 30
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    resource_monitoring_interval: int = 60
    
    # Alerting
    enable_alerting: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["log", "webhook"])
    webhook_url: Optional[str] = None
    
    # Thresholds
    stage_timeout_minutes: int = 120
    pipeline_timeout_minutes: int = 480
    memory_threshold_mb: int = 4096
    cpu_threshold_percent: float = 90.0
    
    # Retention
    metrics_retention_days: int = 30
    logs_retention_days: int = 7


class PipelineMonitor:
    """Comprehensive pipeline monitoring system."""
    
    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        self.logger = structlog.get_logger(__name__)
        
        # Monitoring state
        self.active_runs: Dict[str, PipelineRun] = {}
        self.pipeline_metrics: Dict[str, Dict] = {}
        self.alerts: List[PipelineAlert] = []
        self.alert_handlers: List[Callable] = []
        
        # Prometheus metrics
        self.registry = CollectorRegistry()
        self._init_metrics()
        
        # Background tasks
        self.monitoring_tasks: List[asyncio.Task] = []
        self.is_monitoring = False
    
    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        self.pipeline_runs_total = Counter(
            'ml_pipeline_runs_total',
            'Total number of pipeline runs',
            ['pipeline_id', 'pipeline_name', 'status'],
            registry=self.registry
        )
        
        self.pipeline_duration = Histogram(
            'ml_pipeline_duration_seconds',
            'Pipeline execution duration',
            ['pipeline_id', 'pipeline_name'],
            registry=self.registry
        )
        
        self.stage_duration = Histogram(
            'ml_pipeline_stage_duration_seconds',
            'Stage execution duration',
            ['pipeline_id', 'stage_name', 'stage_type'],
            registry=self.registry
        )
        
        self.active_pipelines = Gauge(
            'ml_pipeline_active_count',
            'Number of currently active pipelines',
            registry=self.registry
        )
        
        self.pipeline_success_rate = Gauge(
            'ml_pipeline_success_rate',
            'Pipeline success rate over time',
            ['pipeline_id', 'pipeline_name'],
            registry=self.registry
        )
        
        self.resource_usage = Gauge(
            'ml_pipeline_resource_usage',
            'Resource usage during pipeline execution',
            ['pipeline_id', 'run_id', 'resource_type'],
            registry=self.registry
        )
        
        self.stage_failures = Counter(
            'ml_pipeline_stage_failures_total',
            'Total number of stage failures',
            ['pipeline_id', 'stage_name', 'stage_type', 'error_type'],
            registry=self.registry
        )
    
    async def register_pipeline(self, pipeline: Pipeline) -> None:
        """Register a pipeline for monitoring."""
        pipeline_id = pipeline.id
        
        self.pipeline_metrics[pipeline_id] = {
            'name': pipeline.name,
            'type': pipeline.pipeline_type,
            'registered_at': datetime.utcnow(),
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'avg_duration': 0.0,
            'last_run_at': None,
            'stages': {stage.name: {'type': stage.stage_type} for stage in pipeline.stages}
        }
        
        self.logger.info(
            "Pipeline registered for monitoring",
            pipeline_id=pipeline_id,
            pipeline_name=pipeline.name
        )
    
    async def start_run_monitoring(self, run: PipelineRun) -> None:
        """Start monitoring a pipeline run."""
        self.active_runs[run.id] = run
        
        # Update metrics
        self.active_pipelines.set(len(self.active_runs))
        
        # Start resource monitoring for this run
        if self.config.enable_performance_monitoring:
            task = asyncio.create_task(self._monitor_run_resources(run))
            self.monitoring_tasks.append(task)
        
        self.logger.info(
            "Started run monitoring",
            run_id=run.id,
            pipeline_id=run.pipeline_id
        )
    
    async def stop_run_monitoring(self, run: PipelineRun) -> None:
        """Stop monitoring a pipeline run."""
        if run.id in self.active_runs:
            # Record final metrics
            await self._record_run_completion(run)
            
            # Remove from active runs
            del self.active_runs[run.id]
            self.active_pipelines.set(len(self.active_runs))
            
            self.logger.info(
                "Stopped run monitoring",
                run_id=run.id,
                pipeline_id=run.pipeline_id,
                final_status=run.status.value if hasattr(run.status, 'value') else str(run.status)
            )
    
    async def _record_run_completion(self, run: PipelineRun) -> None:
        """Record metrics when a run completes."""
        pipeline_id = run.pipeline_id
        
        # Update pipeline metrics
        if pipeline_id in self.pipeline_metrics:
            metrics = self.pipeline_metrics[pipeline_id]
            metrics['total_runs'] += 1
            metrics['last_run_at'] = run.started_at
            
            if run.status == PipelineStatus.COMPLETED:
                metrics['successful_runs'] += 1
            elif run.status == PipelineStatus.FAILED:
                metrics['failed_runs'] += 1
            
            # Update average duration
            if run.execution_time_seconds:
                total_duration = metrics['avg_duration'] * (metrics['total_runs'] - 1)
                metrics['avg_duration'] = (total_duration + run.execution_time_seconds) / metrics['total_runs']
        
        # Record Prometheus metrics
        pipeline_name = self.pipeline_metrics.get(pipeline_id, {}).get('name', 'unknown')
        
        self.pipeline_runs_total.labels(
            pipeline_id=pipeline_id,
            pipeline_name=pipeline_name,
            status=run.status.value if hasattr(run.status, 'value') else str(run.status)
        ).inc()
        
        if run.execution_time_seconds:
            self.pipeline_duration.labels(
                pipeline_id=pipeline_id,
                pipeline_name=pipeline_name
            ).observe(run.execution_time_seconds)
        
        # Record stage metrics
        for stage_name, stage_run in run.stage_runs.items():
            if stage_run.execution_time_seconds:
                stage_type = self.pipeline_metrics.get(pipeline_id, {}).get('stages', {}).get(stage_name, {}).get('type', 'unknown')
                
                self.stage_duration.labels(
                    pipeline_id=pipeline_id,
                    stage_name=stage_name,
                    stage_type=stage_type
                ).observe(stage_run.execution_time_seconds)
            
            # Record failures
            if stage_run.status == StageStatus.FAILED:
                error_type = self._classify_error(stage_run.error_message)
                stage_type = self.pipeline_metrics.get(pipeline_id, {}).get('stages', {}).get(stage_name, {}).get('type', 'unknown')
                
                self.stage_failures.labels(
                    pipeline_id=pipeline_id,
                    stage_name=stage_name,
                    stage_type=stage_type,
                    error_type=error_type
                ).inc()
        
        # Update success rate
        if pipeline_id in self.pipeline_metrics:
            metrics = self.pipeline_metrics[pipeline_id]
            if metrics['total_runs'] > 0:
                success_rate = metrics['successful_runs'] / metrics['total_runs']
                self.pipeline_success_rate.labels(
                    pipeline_id=pipeline_id,
                    pipeline_name=pipeline_name
                ).set(success_rate)
    
    async def _monitor_run_resources(self, run: PipelineRun) -> None:
        """Monitor resource usage during pipeline run."""
        while run.id in self.active_runs and not run.is_completed():
            try:
                # Get current resource usage
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Record metrics
                self.resource_usage.labels(
                    pipeline_id=run.pipeline_id,
                    run_id=run.id,
                    resource_type='cpu_percent'
                ).set(cpu_percent)
                
                self.resource_usage.labels(
                    pipeline_id=run.pipeline_id,
                    run_id=run.id,
                    resource_type='memory_mb'
                ).set(memory.used / 1024 / 1024)
                
                self.resource_usage.labels(
                    pipeline_id=run.pipeline_id,
                    run_id=run.id,
                    resource_type='disk_usage_percent'
                ).set(disk.percent)
                
                # Check thresholds and create alerts
                await self._check_resource_thresholds(run, cpu_percent, memory, disk)
                
                # Wait before next check
                await asyncio.sleep(self.config.resource_monitoring_interval)
                
            except Exception as e:
                self.logger.error(
                    "Error monitoring run resources",
                    run_id=run.id,
                    error=str(e)
                )
                break
    
    async def _check_resource_thresholds(self, run: PipelineRun, cpu_percent: float, memory, disk) -> None:
        """Check resource usage against thresholds and create alerts."""
        alerts = []
        
        # CPU threshold
        if cpu_percent > self.config.cpu_threshold_percent:
            alerts.append(PipelineAlert(
                id=f"cpu_{run.id}_{datetime.utcnow().timestamp()}",
                pipeline_id=run.pipeline_id,
                run_id=run.id,
                alert_type="high_cpu_usage",
                severity=AlertSeverity.HIGH,
                title="High CPU Usage",
                description=f"CPU usage ({cpu_percent:.1f}%) exceeds threshold ({self.config.cpu_threshold_percent}%)",
                metadata={"cpu_percent": cpu_percent, "threshold": self.config.cpu_threshold_percent}
            ))
        
        # Memory threshold
        memory_mb = memory.used / 1024 / 1024
        if memory_mb > self.config.memory_threshold_mb:
            alerts.append(PipelineAlert(
                id=f"memory_{run.id}_{datetime.utcnow().timestamp()}",
                pipeline_id=run.pipeline_id,
                run_id=run.id,
                alert_type="high_memory_usage",
                severity=AlertSeverity.HIGH,
                title="High Memory Usage",
                description=f"Memory usage ({memory_mb:.1f}MB) exceeds threshold ({self.config.memory_threshold_mb}MB)",
                metadata={"memory_mb": memory_mb, "threshold": self.config.memory_threshold_mb}
            ))
        
        # Disk threshold (warning at 80%, critical at 90%)
        if disk.percent > 90:
            alerts.append(PipelineAlert(
                id=f"disk_{run.id}_{datetime.utcnow().timestamp()}",
                pipeline_id=run.pipeline_id,
                run_id=run.id,
                alert_type="high_disk_usage",
                severity=AlertSeverity.CRITICAL,
                title="Critical Disk Usage",
                description=f"Disk usage ({disk.percent:.1f}%) is critically high",
                metadata={"disk_percent": disk.percent}
            ))
        elif disk.percent > 80:
            alerts.append(PipelineAlert(
                id=f"disk_{run.id}_{datetime.utcnow().timestamp()}",
                pipeline_id=run.pipeline_id,
                run_id=run.id,
                alert_type="high_disk_usage",
                severity=AlertSeverity.MEDIUM,
                title="High Disk Usage",
                description=f"Disk usage ({disk.percent:.1f}%) is high",
                metadata={"disk_percent": disk.percent}
            ))
        
        # Process alerts
        for alert in alerts:
            await self._process_alert(alert)
    
    async def _process_alert(self, alert: PipelineAlert) -> None:
        """Process and send an alert."""
        self.alerts.append(alert)
        
        # Log alert
        self.logger.warning(
            "Pipeline alert triggered",
            alert_id=alert.id,
            pipeline_id=alert.pipeline_id,
            run_id=alert.run_id,
            alert_type=alert.alert_type,
            severity=alert.severity.value,
            title=alert.title,
            description=alert.description
        )
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                self.logger.error(
                    "Error calling alert handler",
                    handler=str(handler),
                    error=str(e)
                )
    
    def add_alert_handler(self, handler: Callable[[PipelineAlert], None]) -> None:
        """Add an alert handler."""
        self.alert_handlers.append(handler)
    
    async def setup_model_monitoring(self, config: Dict[str, Any]) -> None:
        """Setup monitoring for deployed models."""
        deployment_id = config.get('deployment_id')
        model_version_id = config.get('model_version_id')
        
        self.logger.info(
            "Setting up model monitoring",
            deployment_id=deployment_id,
            model_version_id=model_version_id,
            config=config
        )
        
        # This would integrate with the model serving monitoring
        # For now, just log the setup
    
    async def get_run_status(self, run_id: str) -> Dict[str, Any]:
        """Get detailed status of a pipeline run."""
        if run_id in self.active_runs:
            run = self.active_runs[run_id]
            return {
                "run_id": run_id,
                "pipeline_id": run.pipeline_id,
                "status": run.status.value if hasattr(run.status, 'value') else str(run.status),
                "started_at": run.started_at.isoformat(),
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "execution_time_seconds": run.execution_time_seconds,
                "stage_count": len(run.stage_runs),
                "completed_stages": len([s for s in run.stage_runs.values() if s.status == StageStatus.COMPLETED]),
                "failed_stages": len([s for s in run.stage_runs.values() if s.status == StageStatus.FAILED]),
                "is_active": True
            }
        
        # Run not found in active runs - would check historical data
        return {
            "run_id": run_id,
            "status": "not_found",
            "is_active": False
        }
    
    async def get_pipeline_metrics(self, pipeline_id: str) -> Dict[str, Any]:
        """Get comprehensive metrics for a pipeline."""
        if pipeline_id not in self.pipeline_metrics:
            return {"error": "Pipeline not found"}
        
        metrics = self.pipeline_metrics[pipeline_id]
        
        # Calculate additional metrics
        success_rate = 0.0
        if metrics['total_runs'] > 0:
            success_rate = metrics['successful_runs'] / metrics['total_runs']
        
        failure_rate = 0.0
        if metrics['total_runs'] > 0:
            failure_rate = metrics['failed_runs'] / metrics['total_runs']
        
        return {
            "pipeline_id": pipeline_id,
            "name": metrics['name'],
            "type": metrics['type'],
            "registered_at": metrics['registered_at'].isoformat(),
            "total_runs": metrics['total_runs'],
            "successful_runs": metrics['successful_runs'],
            "failed_runs": metrics['failed_runs'],
            "success_rate": success_rate,
            "failure_rate": failure_rate,
            "avg_duration_seconds": metrics['avg_duration'],
            "last_run_at": metrics['last_run_at'].isoformat() if metrics['last_run_at'] else None,
            "stage_count": len(metrics['stages']),
            "stages": metrics['stages']
        }
    
    async def get_active_alerts(self, pipeline_id: str = None) -> List[Dict[str, Any]]:
        """Get active alerts for pipelines."""
        active_alerts = [alert for alert in self.alerts if not alert.is_resolved]
        
        if pipeline_id:
            active_alerts = [alert for alert in active_alerts if alert.pipeline_id == pipeline_id]
        
        return [
            {
                "id": alert.id,
                "pipeline_id": alert.pipeline_id,
                "run_id": alert.run_id,
                "alert_type": alert.alert_type,
                "severity": alert.severity.value,
                "title": alert.title,
                "description": alert.description,
                "triggered_at": alert.triggered_at.isoformat(),
                "metadata": alert.metadata
            }
            for alert in active_alerts
        ]
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.is_resolved:
                alert.is_resolved = True
                alert.resolved_at = datetime.utcnow()
                
                self.logger.info(
                    "Alert resolved",
                    alert_id=alert_id,
                    resolved_at=alert.resolved_at.isoformat()
                )
                
                return True
        
        return False
    
    def _classify_error(self, error_message: str) -> str:
        """Classify error type based on error message."""
        if not error_message:
            return "unknown"
        
        error_lower = error_message.lower()
        
        if "timeout" in error_lower:
            return "timeout"
        elif "memory" in error_lower or "out of memory" in error_lower:
            return "memory"
        elif "permission" in error_lower or "access denied" in error_lower:
            return "permission"
        elif "connection" in error_lower or "network" in error_lower:
            return "network"
        elif "file not found" in error_lower or "no such file" in error_lower:
            return "file_not_found"
        elif "validation" in error_lower or "invalid" in error_lower:
            return "validation"
        else:
            return "runtime"
    
    async def start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        
        # Start periodic cleanup task
        cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self.monitoring_tasks.append(cleanup_task)
        
        self.logger.info("Pipeline monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""
        self.is_monitoring = False
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        self.monitoring_tasks.clear()
        
        self.logger.info("Pipeline monitoring stopped")
    
    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of old metrics and alerts."""
        while self.is_monitoring:
            try:
                now = datetime.utcnow()
                
                # Clean up old alerts
                retention_threshold = now - timedelta(days=self.config.logs_retention_days)
                self.alerts = [
                    alert for alert in self.alerts
                    if alert.triggered_at > retention_threshold or not alert.is_resolved
                ]
                
                self.logger.debug(
                    "Completed periodic cleanup",
                    active_alerts=len([a for a in self.alerts if not a.is_resolved]),
                    total_alerts=len(self.alerts)
                )
                
                # Wait before next cleanup
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                self.logger.error(
                    "Error during periodic cleanup",
                    error=str(e)
                )
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    def get_metrics_registry(self) -> CollectorRegistry:
        """Get Prometheus metrics registry."""
        return self.registry
    
    async def export_metrics_summary(self) -> Dict[str, Any]:
        """Export a summary of all monitoring metrics."""
        return {
            "active_runs": len(self.active_runs),
            "total_pipelines": len(self.pipeline_metrics),
            "active_alerts": len([a for a in self.alerts if not a.is_resolved]),
            "total_alerts": len(self.alerts),
            "monitoring_enabled": self.is_monitoring,
            "config": {
                "metrics_enabled": self.config.enable_metrics,
                "performance_monitoring": self.config.enable_performance_monitoring,
                "alerting_enabled": self.config.enable_alerting,
                "metrics_interval": self.config.metrics_interval_seconds,
                "resource_monitoring_interval": self.config.resource_monitoring_interval
            },
            "pipeline_summaries": [
                {
                    "pipeline_id": pid,
                    "name": metrics["name"],
                    "total_runs": metrics["total_runs"],
                    "success_rate": metrics["successful_runs"] / metrics["total_runs"] if metrics["total_runs"] > 0 else 0,
                    "avg_duration": metrics["avg_duration"]
                }
                for pid, metrics in self.pipeline_metrics.items()
            ]
        }