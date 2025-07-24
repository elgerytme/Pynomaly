"""
Pipeline Health Service

Provides application-level services for monitoring pipeline health,
managing metrics, alerts, and generating health reports.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from uuid import UUID

from ...domain.entities.pipeline_health import (
    PipelineHealth,
    PipelineMetric,
    PipelineAlert,
    PipelineStatus,
    MetricType,
    AlertSeverity,
    MetricThreshold
)


from ...domain.repositories.pipeline_health_repository import PipelineHealthRepository
from ...infrastructure.errors.exceptions import PipelineError


class PipelineHealthService:
    """Service for managing pipeline health monitoring."""
    
    def __init__(self, repository: PipelineHealthRepository):
        self._repository = repository
        self._alert_handlers: Dict[AlertSeverity, List[Callable]] = {
            AlertSeverity.INFO: [],
            AlertSeverity.WARNING: [],
            AlertSeverity.CRITICAL: [],
            AlertSeverity.EMERGENCY: []
        }
        
        # Default thresholds for common metrics
        self._default_thresholds = {
            MetricType.THROUGHPUT: MetricThreshold(
                warning_threshold=100,
                critical_threshold=50,
                comparison_operator="<"
            ),
            MetricType.LATENCY: MetricThreshold(
                warning_threshold=1000,  # ms
                critical_threshold=5000,
                comparison_operator=">"
            ),
            MetricType.ERROR_RATE: MetricThreshold(
                warning_threshold=0.05,  # 5%
                critical_threshold=0.1,  # 10%
                comparison_operator=">"
            ),
            MetricType.AVAILABILITY: MetricThreshold(
                warning_threshold=99.0,  # 99%
                critical_threshold=95.0,
                comparison_operator="<"
            )
        }
    
    async def register_pipeline(self, pipeline_id: UUID, pipeline_name: str) -> PipelineHealth:
        """Register a new pipeline for health monitoring."""
        health = PipelineHealth(
            pipeline_id=pipeline_id,
            pipeline_name=pipeline_name,
            status=PipelineStatus.UNKNOWN
        )
        
        return await self._repository.save_pipeline_health(health)
    
    async def get_pipeline_health(self, pipeline_id: UUID) -> Optional[PipelineHealth]:
        """Get health status for a pipeline."""
        return await self._repository.get_pipeline_health(pipeline_id)
    
    async def get_all_pipelines(self) -> List[PipelineHealth]:
        """Get health status for all pipelines."""
        return await self._repository.get_all_pipeline_health()
    
    async def get_pipelines_by_status(self, status: PipelineStatus) -> List[PipelineHealth]:
        """Get all pipelines with a specific status."""
        all_pipelines = await self._repository.get_all_pipeline_health()
        return [
            pipeline for pipeline in all_pipelines
            if pipeline.status == status
        ]
    
    async def record_metric(
        self,
        pipeline_id: UUID,
        metric_type: MetricType,
        name: str,
        value: float,
        unit: str,
        labels: Dict[str, str] = None,
        source: str = None,
        threshold: MetricThreshold = None
    ) -> None:
        """Record a metric for a pipeline."""
        pipeline = await self._repository.get_pipeline_health(pipeline_id)
        if not pipeline:
            raise PipelineError(f"Pipeline {pipeline_id} not registered")
        
        # Use default threshold if not provided
        if threshold is None:
            threshold = self._default_thresholds.get(metric_type)
        
        metric = PipelineMetric(
            pipeline_id=pipeline_id,
            metric_type=metric_type,
            name=name,
            value=value,
            unit=unit,
            labels=labels or {},
            source=source,
            threshold=threshold
        )
        
        # Add to pipeline and history
        pipeline.add_metric(metric)
        await self._repository.add_metric(pipeline_id, metric)
        
        # Check for alert conditions
        await self._check_metric_alerts(pipeline, metric)
    
    async def create_alert(
        self,
        pipeline_id: UUID,
        severity: AlertSeverity,
        title: str,
        description: str,
        metric_id: UUID = None,
        triggered_by: str = None,
        current_value: float = None,
        threshold_value: float = None
    ) -> PipelineAlert:
        """Create an alert for a pipeline."""
        pipeline = await self._repository.get_pipeline_health(pipeline_id)
        if not pipeline:
            raise PipelineError(f"Pipeline {pipeline_id} not registered")
        
        alert = PipelineAlert(
            pipeline_id=pipeline_id,
            metric_id=metric_id,
            severity=severity,
            title=title,
            description=description,
            triggered_by=triggered_by,
            current_value=current_value,
            threshold_value=threshold_value
        )
        
        pipeline.add_alert(alert)
        await self._repository.add_alert(pipeline_id, alert)
        
        # Trigger alert handlers
        self._trigger_alert_handlers(alert)
        
        return alert
    
    async def resolve_alert(self, pipeline_id: UUID, alert_id: UUID) -> bool:
        """Resolve an alert."""
        return await self._repository.resolve_alert(pipeline_id, alert_id)
    
    async def acknowledge_alert(self, pipeline_id: UUID, alert_id: UUID, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        pipeline = await self._repository.get_pipeline_health(pipeline_id)
        if not pipeline:
            return False
        
        for alert in pipeline.active_alerts:
            if alert.id == alert_id:
                alert.acknowledge(acknowledged_by)
                await self._repository.save_pipeline_health(pipeline) # Save the updated pipeline health
                return True
        
        return False
    
    async def update_pipeline_execution(
        self,
        pipeline_id: UUID,
        success: bool,
        execution_duration: timedelta = None
    ) -> None:
        """Update pipeline execution statistics."""
        pipeline = await self._repository.get_pipeline_health(pipeline_id)
        if not pipeline:
            raise PipelineError(f"Pipeline {pipeline_id} not found")
        
        pipeline.last_execution = datetime.utcnow()
        
        if execution_duration:
            pipeline.execution_duration = execution_duration
        
        if success:
            pipeline.successful_executions += 1
        else:
            pipeline.failed_executions += 1
            
            # Create alert for failed execution
            await self.create_alert(
                pipeline_id=pipeline_id,
                severity=AlertSeverity.WARNING,
                title="Pipeline Execution Failed",
                description=f"Pipeline {pipeline.pipeline_name} execution failed"
            )
        
        # Update error rate
        total_executions = pipeline.successful_executions + pipeline.failed_executions
        if total_executions > 0:
            pipeline.error_rate = pipeline.failed_executions / total_executions
        
        await self._repository.save_pipeline_health(pipeline)
    
    async def get_metric_history(
        self,
        pipeline_id: UUID,
        metric_type: MetricType = None,
        hours: int = 24
    ) -> List[PipelineMetric]:
        """Get metric history for a pipeline."""
        return await self._repository.get_metric_history(pipeline_id, metric_type.value if metric_type else None, hours)
    
    async def get_health_dashboard(self) -> Dict[str, Any]:
        """Get health dashboard data for all pipelines."""
        all_pipelines = await self.get_all_pipelines()
        
        # Overall statistics
        total_pipelines = len(all_pipelines)
        healthy_pipelines = len(await self.get_pipelines_by_status(PipelineStatus.HEALTHY))
        warning_pipelines = len(await self.get_pipelines_by_status(PipelineStatus.WARNING))
        critical_pipelines = len(await self.get_pipelines_by_status(PipelineStatus.CRITICAL))
        failed_pipelines = len(await self.get_pipelines_by_status(PipelineStatus.FAILED))
        
        # Alert statistics
        total_alerts = sum(len(p.active_alerts) for p in all_pipelines)
        critical_alerts = sum(len(p.get_critical_alerts()) for p in all_pipelines)
        
        # Health score distribution
        health_scores = [p.get_health_score() for p in all_pipelines]
        avg_health_score = sum(health_scores) / len(health_scores) if health_scores else 0
        
        # Top problematic pipelines
        problematic_pipelines = sorted(
            all_pipelines,
            key=lambda p: (
                len(p.get_critical_alerts()),
                -p.get_health_score(),
                len(p.active_alerts)
            ),
            reverse=True
        )[:5]
        
        return {
            "overview": {
                "total_pipelines": total_pipelines,
                "healthy_pipelines": healthy_pipelines,
                "warning_pipelines": warning_pipelines,
                "critical_pipelines": critical_pipelines,
                "failed_pipelines": failed_pipelines,
                "total_alerts": total_alerts,
                "critical_alerts": critical_alerts,
                "avg_health_score": avg_health_score
            },
            "status_distribution": {
                "healthy": healthy_pipelines,
                "warning": warning_pipelines,
                "critical": critical_pipelines,
                "failed": failed_pipelines
            },
            "top_issues": [p.to_summary_dict() for p in problematic_pipelines],
            "recent_alerts": await self._get_recent_alerts(hours=24),
            "performance_trends": await self._get_performance_trends(hours=24)
        }
    
    async def get_pipeline_report(self, pipeline_id: UUID, hours: int = 24) -> Dict[str, Any]:
        """Generate a comprehensive report for a pipeline."""
        pipeline = await self._repository.get_pipeline_health(pipeline_id)
        if not pipeline:
            raise PipelineError(f"Pipeline {pipeline_id} not found")
        
        # Get metric history
        metric_history = await self.get_metric_history(pipeline_id, hours=hours)
        
        # Calculate trends
        trends = self._calculate_metric_trends(metric_history)
        
        # SLA compliance
        sla_compliance = self._calculate_sla_compliance(pipeline, hours)
        
        # Recommendations
        recommendations = self._generate_recommendations(pipeline, metric_history)
        
        return {
            "pipeline": pipeline.to_summary_dict(),
            "health_score": pipeline.get_health_score(),
            "uptime": pipeline.calculate_uptime(hours),
            "metric_trends": trends,
            "sla_compliance": sla_compliance,
            "active_alerts": [alert.dict() for alert in pipeline.active_alerts],
            "recommendations": recommendations,
            "metric_summary": self._get_metric_summary(metric_history)
        }
    
    def register_alert_handler(self, severity: AlertSeverity, handler: Callable[[PipelineAlert], None]) -> None:
        """Register a handler for alerts of a specific severity."""
        self._alert_handlers[severity].append(handler)
    
    def set_default_threshold(self, metric_type: MetricType, threshold: MetricThreshold) -> None:
        """Set default threshold for a metric type."""
        self._default_thresholds[metric_type] = threshold
    
    def _check_metric_alerts(self, pipeline: PipelineHealth, metric: PipelineMetric) -> None:
        """Check if a metric triggers any alerts."""
        if not metric.threshold:
            return
        
        status = metric.get_status()
        
        if status in [PipelineStatus.WARNING, PipelineStatus.CRITICAL]:
            severity = AlertSeverity.CRITICAL if status == PipelineStatus.CRITICAL else AlertSeverity.WARNING
            
            # Check if similar alert already exists
            existing_alert = None
            for alert in pipeline.active_alerts:
                if (alert.metric_id == metric.id or 
                    (alert.triggered_by == f"metric_{metric.metric_type}" and 
                     alert.severity == severity)):
                    existing_alert = alert
                    break
            
            if not existing_alert:
                threshold_value = (metric.threshold.critical_threshold 
                                if status == PipelineStatus.CRITICAL 
                                else metric.threshold.warning_threshold)
                
                self.create_alert(
                    pipeline_id=pipeline.pipeline_id,
                    severity=severity,
                    title=f"{metric.name} {status.value}",
                    description=f"{metric.name} value {metric.value} {metric.unit} exceeds threshold",
                    metric_id=metric.id,
                    triggered_by=f"metric_{metric.metric_type}",
                    current_value=metric.value,
                    threshold_value=threshold_value
                )
    
    def _trigger_alert_handlers(self, alert: PipelineAlert) -> None:
        """Trigger registered alert handlers."""
        handlers = self._alert_handlers.get(alert.severity, [])
        for handler in handlers:
            try:
                handler(alert)
            except Exception as e:
                # Log error but don't fail the alert creation
                print(f"Error in alert handler: {e}")
    
    def _get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts across all pipelines."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_alerts = []
        for pipeline in self._pipelines.values():
            for alert in pipeline.active_alerts:
                if alert.created_at >= cutoff_time:
                    alert_dict = alert.dict()
                    alert_dict['pipeline_name'] = pipeline.pipeline_name
                    recent_alerts.append(alert_dict)
        
        return sorted(recent_alerts, key=lambda a: a['created_at'], reverse=True)
    
    def _get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends across all pipelines."""
        trends = {}
        
        for metric_type in MetricType:
            values = []
            timestamps = []
            
            for pipeline_id in self._pipelines:
                metrics = self.get_metric_history(pipeline_id, metric_type, hours)
                for metric in metrics:
                    values.append(metric.value)
                    timestamps.append(metric.timestamp)
            
            if values:
                trends[metric_type.value] = {
                    "avg_value": sum(values) / len(values),
                    "min_value": min(values),
                    "max_value": max(values),
                    "data_points": len(values)
                }
        
        return trends
    
    def _calculate_metric_trends(self, metrics: List[PipelineMetric]) -> Dict[str, Dict[str, float]]:
        """Calculate trends for metrics."""
        trends = {}
        
        # Group metrics by type
        by_type = {}
        for metric in metrics:
            if metric.metric_type not in by_type:
                by_type[metric.metric_type] = []
            by_type[metric.metric_type].append(metric)
        
        # Calculate trends for each type
        for metric_type, type_metrics in by_type.items():
            if len(type_metrics) < 2:
                continue
            
            # Sort by timestamp
            sorted_metrics = sorted(type_metrics, key=lambda m: m.timestamp)
            
            # Calculate simple trend (first vs last)
            first_value = sorted_metrics[0].value
            last_value = sorted_metrics[-1].value
            
            trend_percentage = ((last_value - first_value) / first_value * 100) if first_value != 0 else 0
            
            trends[metric_type.value] = {
                "trend_percentage": trend_percentage,
                "first_value": first_value,
                "last_value": last_value,
                "avg_value": sum(m.value for m in sorted_metrics) / len(sorted_metrics)
            }
        
        return trends
    
    def _calculate_sla_compliance(self, pipeline: PipelineHealth, hours: int) -> Dict[str, Any]:
        """Calculate SLA compliance for a pipeline."""
        uptime = pipeline.calculate_uptime(hours)
        
        # Define SLA targets
        sla_targets = {
            "availability": 99.9,  # 99.9% uptime
            "error_rate": 0.01,    # Max 1% error rate
            "response_time": 1000   # Max 1 second response time
        }
        
        compliance = {
            "availability": {
                "target": sla_targets["availability"],
                "actual": uptime,
                "compliant": uptime >= sla_targets["availability"]
            },
            "error_rate": {
                "target": sla_targets["error_rate"],
                "actual": pipeline.error_rate,
                "compliant": pipeline.error_rate <= sla_targets["error_rate"]
            }
        }
        
        # Check response time compliance
        latency_metrics = [
            m for m in pipeline.current_metrics 
            if m.metric_type == MetricType.LATENCY
        ]
        if latency_metrics:
            avg_latency = sum(m.value for m in latency_metrics) / len(latency_metrics)
            compliance["response_time"] = {
                "target": sla_targets["response_time"],
                "actual": avg_latency,
                "compliant": avg_latency <= sla_targets["response_time"]
            }
        
        return compliance
    
    def _generate_recommendations(self, pipeline: PipelineHealth, metrics: List[PipelineMetric]) -> List[str]:
        """Generate recommendations for pipeline improvement."""
        recommendations = []
        
        # High error rate
        if pipeline.error_rate > 0.05:
            recommendations.append("Error rate is high. Investigate recent failures and implement retry mechanisms.")
        
        # Low availability
        if pipeline.availability_percentage < 99.0:
            recommendations.append("Availability is below target. Consider implementing redundancy and failover mechanisms.")
        
        # Performance issues
        latency_metrics = [m for m in metrics if m.metric_type == MetricType.LATENCY]
        if latency_metrics:
            avg_latency = sum(m.value for m in latency_metrics) / len(latency_metrics)
            if avg_latency > 2000:  # 2 seconds
                recommendations.append("High latency detected. Consider optimizing data processing and caching.")
        
        # Many alerts
        if len(pipeline.active_alerts) > 3:
            recommendations.append("Multiple active alerts. Review alert thresholds and resolve underlying issues.")
        
        # Stale data
        if pipeline.data_freshness and pipeline.data_freshness > timedelta(hours=4):
            recommendations.append("Data freshness is low. Check data ingestion pipelines and schedules.")
        
        return recommendations
    
    def _get_metric_summary(self, metrics: List[PipelineMetric]) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics for metrics."""
        summary = {}
        
        # Group by metric type
        by_type = {}
        for metric in metrics:
            if metric.metric_type not in by_type:
                by_type[metric.metric_type] = []
            by_type[metric.metric_type].append(metric.value)
        
        # Calculate summary stats
        for metric_type, values in by_type.items():
            if values:
                summary[metric_type.value] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1] if values else None
                }
        
        return summary