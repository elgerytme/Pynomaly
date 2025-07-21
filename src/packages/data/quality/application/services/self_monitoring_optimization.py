"""
Self-monitoring and optimization service for autonomous quality system management.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
from collections import defaultdict, deque

from data_quality.application.services.autonomous_quality_monitoring_service import AutonomousQualityMonitoringService
from data_quality.application.services.automated_remediation_engine import AutomatedRemediationEngine
from data_quality.application.services.adaptive_quality_controls import AdaptiveQualityControls
from data_quality.application.services.pipeline_integration_framework import PipelineIntegrationFramework
from data_quality.application.services.intelligent_quality_orchestration import IntelligentQualityOrchestration
from software.interfaces.data_quality_interface import DataQualityInterface
from software.interfaces.data_quality_interface import QualityReport


logger = logging.getLogger(__name__)


class SystemHealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class OptimizationType(Enum):
    """Types of optimization."""
    PERFORMANCE = "performance"
    COST = "cost"
    ACCURACY = "accuracy"
    RESOURCE_UTILIZATION = "resource_utilization"
    RESPONSE_TIME = "response_time"


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_usage: float
    error_rate: float
    response_time: float
    throughput: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ComponentHealth:
    """Health status of system component."""
    component_name: str
    health_status: SystemHealthStatus
    performance_score: float
    last_check: datetime
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation."""
    recommendation_id: str
    optimization_type: OptimizationType
    component: str
    description: str
    expected_improvement: float
    implementation_effort: str  # low, medium, high
    priority: str  # low, medium, high, critical
    cost_benefit_ratio: float
    implementation_steps: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    implemented: bool = False


@dataclass
class CostMetrics:
    """Cost metrics for quality operations."""
    processing_cost: float
    storage_cost: float
    network_cost: float
    human_intervention_cost: float
    total_cost: float
    cost_per_record: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EffectivenessMetrics:
    """Effectiveness metrics for quality system."""
    quality_improvement_rate: float
    false_positive_rate: float
    false_negative_rate: float
    automation_rate: float
    user_satisfaction_score: float
    mean_time_to_detection: float
    mean_time_to_resolution: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SystemAlert:
    """System alert for monitoring."""
    alert_id: str
    alert_type: str
    severity: str
    component: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class SelfMonitoringOptimization:
    """Service for self-monitoring and optimization of quality system performance."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the self-monitoring and optimization service."""
        # Initialize service configuration
        self.config = config
        
        # Initialize quality services (monitoring only, not controlling)
        self.monitoring_service = None
        self.remediation_engine = None
        self.adaptive_controls = None
        self.pipeline_framework = None
        self.orchestration_service = None
        
        # Self-monitoring data
        self.system_metrics_history: deque = deque(maxlen=1000)
        self.component_health: Dict[str, ComponentHealth] = {}
        self.optimization_recommendations: List[OptimizationRecommendation] = []
        self.cost_metrics_history: deque = deque(maxlen=100)
        self.effectiveness_metrics_history: deque = deque(maxlen=100)
        self.system_alerts: List[SystemAlert] = []
        
        # Performance baselines
        self.performance_baselines: Dict[str, float] = {}
        self.baseline_update_interval = config.get("baseline_update_interval", 86400)  # 24 hours
        
        # Configuration
        self.monitoring_interval = config.get("monitoring_interval", 60)  # 1 minute
        self.optimization_interval = config.get("optimization_interval", 3600)  # 1 hour
        self.alert_retention_days = config.get("alert_retention_days", 30)
        self.auto_optimization_enabled = config.get("auto_optimization_enabled", True)
        
        # Initialize baselines
        self._initialize_performance_baselines()
        
        # Start monitoring tasks
        asyncio.create_task(self._system_monitoring_task())
        asyncio.create_task(self._optimization_task())
        asyncio.create_task(self._alert_management_task())
        asyncio.create_task(self._baseline_update_task())
    
    def _initialize_performance_baselines(self) -> None:
        """Initialize performance baselines."""
        self.performance_baselines = {
            "cpu_usage": 70.0,
            "memory_usage": 80.0,
            "disk_usage": 85.0,
            "network_usage": 60.0,
            "error_rate": 5.0,
            "response_time": 1000.0,  # ms
            "throughput": 100.0,  # requests/second
            "quality_score": 0.85,
            "automation_rate": 0.90,
            "user_satisfaction": 4.0  # out of 5
        }
        
        logger.info("Initialized performance baselines")
    
    async def _system_monitoring_task(self) -> None:
        """Main system monitoring task."""
        while True:
            try:
                await asyncio.sleep(self.monitoring_interval)
                
                # Collect system metrics
                metrics = await self._collect_system_metrics()
                self.system_metrics_history.append(metrics)
                
                # Check component health
                await self._check_component_health()
                
                # Update effectiveness metrics
                await self._update_effectiveness_metrics()
                
                # Update cost metrics
                await self._update_cost_metrics()
                
                # Check for alerts
                await self._check_system_alerts()
                
            except Exception as e:
                logger.error(f"System monitoring task error: {str(e)}")
    
    async def _optimization_task(self) -> None:
        """System optimization task."""
        while True:
            try:
                await asyncio.sleep(self.optimization_interval)
                
                # Generate optimization recommendations
                await self._generate_optimization_recommendations()
                
                # Apply automatic optimizations
                if self.auto_optimization_enabled:
                    await self._apply_automatic_optimizations()
                
                # Update performance baselines
                await self._update_performance_baselines()
                
            except Exception as e:
                logger.error(f"Optimization task error: {str(e)}")
    
    async def _alert_management_task(self) -> None:
        """Alert management task."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Clean up old alerts
                await self._cleanup_old_alerts()
                
                # Auto-resolve resolved alerts
                await self._auto_resolve_alerts()
                
            except Exception as e:
                logger.error(f"Alert management task error: {str(e)}")
    
    async def _baseline_update_task(self) -> None:
        """Baseline update task."""
        while True:
            try:
                await asyncio.sleep(self.baseline_update_interval)
                
                # Update performance baselines based on historical data
                await self._recalculate_baselines()
                
            except Exception as e:
                logger.error(f"Baseline update task error: {str(e)}")
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            import psutil
            
            # System resources
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Application metrics (simulated)
            error_rate = 2.5  # Would be calculated from actual error logs
            response_time = 150.0  # Would be calculated from response times
            throughput = 85.0  # Would be calculated from request rates
            
            return SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                network_usage=50.0,  # Simulated
                error_rate=error_rate,
                response_time=response_time,
                throughput=throughput
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {str(e)}")
            return SystemMetrics(
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_usage=0.0,
                error_rate=0.0,
                response_time=0.0,
                throughput=0.0
            )
    
    async def _check_component_health(self) -> None:
        """Check health of all quality system components."""
        components = [
            "monitoring_service",
            "remediation_engine",
            "adaptive_controls",
            "pipeline_framework",
            "orchestration_service"
        ]
        
        for component in components:
            health = await self._assess_component_health(component)
            self.component_health[component] = health
    
    async def _assess_component_health(self, component_name: str) -> ComponentHealth:
        """Assess health of a specific component."""
        # Get recent metrics for this component
        recent_metrics = list(self.system_metrics_history)[-10:] if self.system_metrics_history else []
        
        if not recent_metrics:
            return ComponentHealth(
                component_name=component_name,
                health_status=SystemHealthStatus.HEALTHY,
                performance_score=1.0,
                last_check=datetime.utcnow()
            )
        
        # Calculate component performance score
        performance_score = 1.0
        issues = []
        recommendations = []
        
        # Check CPU usage
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        if avg_cpu > self.performance_baselines["cpu_usage"]:
            performance_score -= 0.1
            issues.append(f"High CPU usage: {avg_cpu:.1f}%")
            recommendations.append("Consider optimizing CPU-intensive operations")
        
        # Check memory usage
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        if avg_memory > self.performance_baselines["memory_usage"]:
            performance_score -= 0.1
            issues.append(f"High memory usage: {avg_memory:.1f}%")
            recommendations.append("Consider optimizing memory usage")
        
        # Check error rate
        avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        if avg_error_rate > self.performance_baselines["error_rate"]:
            performance_score -= 0.2
            issues.append(f"High error rate: {avg_error_rate:.1f}%")
            recommendations.append("Investigate and fix error sources")
        
        # Check response time
        avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
        if avg_response_time > self.performance_baselines["response_time"]:
            performance_score -= 0.15
            issues.append(f"High response time: {avg_response_time:.1f}ms")
            recommendations.append("Optimize response time performance")
        
        # Determine health status
        if performance_score >= 0.9:
            health_status = SystemHealthStatus.HEALTHY
        elif performance_score >= 0.7:
            health_status = SystemHealthStatus.DEGRADED
        elif performance_score >= 0.5:
            health_status = SystemHealthStatus.UNHEALTHY
        else:
            health_status = SystemHealthStatus.CRITICAL
        
        return ComponentHealth(
            component_name=component_name,
            health_status=health_status,
            performance_score=max(0.0, performance_score),
            last_check=datetime.utcnow(),
            issues=issues,
            recommendations=recommendations,
            metrics={
                "cpu_usage": avg_cpu,
                "memory_usage": avg_memory,
                "error_rate": avg_error_rate,
                "response_time": avg_response_time
            }
        )
    
    async def _update_effectiveness_metrics(self) -> None:
        """Update effectiveness metrics."""
        # Simulate effectiveness metrics calculation
        # In practice, these would be calculated from actual system data
        
        effectiveness = EffectivenessMetrics(
            quality_improvement_rate=0.15,  # 15% improvement
            false_positive_rate=0.08,       # 8% false positives
            false_negative_rate=0.05,       # 5% false negatives
            automation_rate=0.92,           # 92% automation
            user_satisfaction_score=4.2,    # out of 5
            mean_time_to_detection=45.0,    # seconds
            mean_time_to_resolution=180.0   # seconds
        )
        
        self.effectiveness_metrics_history.append(effectiveness)
    
    async def _update_cost_metrics(self) -> None:
        """Update cost metrics."""
        # Simulate cost metrics calculation
        # In practice, these would be calculated from actual usage data
        
        cost_metrics = CostMetrics(
            processing_cost=250.0,          # $250/hour
            storage_cost=50.0,              # $50/hour
            network_cost=30.0,              # $30/hour
            human_intervention_cost=80.0,   # $80/hour
            total_cost=410.0,               # $410/hour
            cost_per_record=0.0041          # $0.0041 per record
        )
        
        self.cost_metrics_history.append(cost_metrics)
    
    async def _check_system_alerts(self) -> None:
        """Check for system alerts."""
        # Check each component for alert conditions
        for component_name, health in self.component_health.items():
            if health.health_status == SystemHealthStatus.CRITICAL:
                alert = SystemAlert(
                    alert_id=f"critical_{component_name}_{datetime.utcnow().timestamp()}",
                    alert_type="health_critical",
                    severity="critical",
                    component=component_name,
                    message=f"Component {component_name} is in critical health state"
                )
                self.system_alerts.append(alert)
                logger.critical(f"CRITICAL ALERT: {alert.message}")
            
            elif health.health_status == SystemHealthStatus.UNHEALTHY:
                alert = SystemAlert(
                    alert_id=f"unhealthy_{component_name}_{datetime.utcnow().timestamp()}",
                    alert_type="health_unhealthy",
                    severity="high",
                    component=component_name,
                    message=f"Component {component_name} is unhealthy"
                )
                self.system_alerts.append(alert)
                logger.warning(f"HEALTH ALERT: {alert.message}")
        
        # Check system metrics for alerts
        if self.system_metrics_history:
            latest_metrics = self.system_metrics_history[-1]
            
            if latest_metrics.cpu_usage > 90:
                alert = SystemAlert(
                    alert_id=f"cpu_high_{datetime.utcnow().timestamp()}",
                    alert_type="resource_usage",
                    severity="high",
                    component="system",
                    message=f"High CPU usage: {latest_metrics.cpu_usage:.1f}%"
                )
                self.system_alerts.append(alert)
            
            if latest_metrics.memory_usage > 95:
                alert = SystemAlert(
                    alert_id=f"memory_high_{datetime.utcnow().timestamp()}",
                    alert_type="resource_usage",
                    severity="critical",
                    component="system",
                    message=f"Critical memory usage: {latest_metrics.memory_usage:.1f}%"
                )
                self.system_alerts.append(alert)
    
    async def _generate_optimization_recommendations(self) -> None:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Performance optimization recommendations
        if self.system_metrics_history:
            recent_metrics = list(self.system_metrics_history)[-10:]
            avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
            avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
            
            if avg_cpu > 80:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"cpu_optimization_{datetime.utcnow().timestamp()}",
                    optimization_type=OptimizationType.PERFORMANCE,
                    component="system",
                    description="Optimize CPU usage through algorithmic improvements",
                    expected_improvement=15.0,
                    implementation_effort="medium",
                    priority="high",
                    cost_benefit_ratio=2.5,
                    implementation_steps=[
                        "Profile CPU-intensive operations",
                        "Implement caching for repeated calculations",
                        "Optimize database queries",
                        "Consider parallel processing"
                    ],
                    risks=["Potential system instability during optimization"]
                ))
            
            if avg_memory > 85:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"memory_optimization_{datetime.utcnow().timestamp()}",
                    optimization_type=OptimizationType.RESOURCE_UTILIZATION,
                    component="system",
                    description="Optimize memory usage through better resource management",
                    expected_improvement=20.0,
                    implementation_effort="medium",
                    priority="high",
                    cost_benefit_ratio=3.0,
                    implementation_steps=[
                        "Implement memory pooling",
                        "Optimize data structures",
                        "Add garbage collection optimization",
                        "Review memory leaks"
                    ],
                    risks=["Potential memory fragmentation"]
                ))
            
            if avg_response_time > 500:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"response_time_optimization_{datetime.utcnow().timestamp()}",
                    optimization_type=OptimizationType.RESPONSE_TIME,
                    component="system",
                    description="Improve response time through optimization",
                    expected_improvement=30.0,
                    implementation_effort="high",
                    priority="medium",
                    cost_benefit_ratio=2.0,
                    implementation_steps=[
                        "Implement request caching",
                        "Optimize database indexes",
                        "Add connection pooling",
                        "Implement async processing"
                    ],
                    risks=["Increased complexity"]
                ))
        
        # Cost optimization recommendations
        if self.cost_metrics_history:
            latest_cost = self.cost_metrics_history[-1]
            
            if latest_cost.cost_per_record > 0.01:  # $0.01 per record
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"cost_optimization_{datetime.utcnow().timestamp()}",
                    optimization_type=OptimizationType.COST,
                    component="system",
                    description="Reduce cost per record through efficiency improvements",
                    expected_improvement=25.0,
                    implementation_effort="medium",
                    priority="medium",
                    cost_benefit_ratio=4.0,
                    implementation_steps=[
                        "Implement batch processing",
                        "Optimize resource allocation",
                        "Review storage costs",
                        "Implement auto-scaling"
                    ],
                    risks=["Potential service disruption"]
                ))
        
        # Add new recommendations
        for rec in recommendations:
            if not any(r.recommendation_id == rec.recommendation_id for r in self.optimization_recommendations):
                self.optimization_recommendations.append(rec)
        
        logger.info(f"Generated {len(recommendations)} new optimization recommendations")
    
    async def _apply_automatic_optimizations(self) -> None:
        """Apply automatic optimizations."""
        auto_optimizations = [
            rec for rec in self.optimization_recommendations
            if not rec.implemented 
            and rec.priority == "low"
            and rec.implementation_effort == "low"
            and rec.cost_benefit_ratio > 2.0
        ]
        
        for optimization in auto_optimizations:
            try:
                await self._implement_optimization(optimization)
                optimization.implemented = True
                logger.info(f"Automatically implemented optimization: {optimization.description}")
            except Exception as e:
                logger.error(f"Failed to implement optimization {optimization.recommendation_id}: {str(e)}")
    
    async def _implement_optimization(self, optimization: OptimizationRecommendation) -> None:
        """Implement a specific optimization."""
        # Simulate optimization implementation
        logger.info(f"Implementing optimization: {optimization.description}")
        
        # In practice, this would contain actual optimization logic
        # For now, we'll just simulate the implementation
        await asyncio.sleep(1)  # Simulate implementation time
        
        # Update performance baselines if successful
        if optimization.optimization_type == OptimizationType.PERFORMANCE:
            self.performance_baselines["cpu_usage"] *= 0.95  # 5% improvement
        elif optimization.optimization_type == OptimizationType.COST:
            # Cost optimization would update cost-related metrics
            pass
    
    async def _update_performance_baselines(self) -> None:
        """Update performance baselines based on recent performance."""
        if len(self.system_metrics_history) < 100:
            return
        
        # Calculate new baselines from recent performance
        recent_metrics = list(self.system_metrics_history)[-100:]
        
        # Use 90th percentile for upper bounds
        cpu_values = sorted([m.cpu_usage for m in recent_metrics])
        memory_values = sorted([m.memory_usage for m in recent_metrics])
        response_time_values = sorted([m.response_time for m in recent_metrics])
        
        # Update baselines (gradual adjustment)
        self.performance_baselines["cpu_usage"] = (
            self.performance_baselines["cpu_usage"] * 0.9 +
            cpu_values[int(len(cpu_values) * 0.9)] * 0.1
        )
        
        self.performance_baselines["memory_usage"] = (
            self.performance_baselines["memory_usage"] * 0.9 +
            memory_values[int(len(memory_values) * 0.9)] * 0.1
        )
        
        self.performance_baselines["response_time"] = (
            self.performance_baselines["response_time"] * 0.9 +
            response_time_values[int(len(response_time_values) * 0.9)] * 0.1
        )
    
    async def _cleanup_old_alerts(self) -> None:
        """Clean up old alerts."""
        cutoff_time = datetime.utcnow() - timedelta(days=self.alert_retention_days)
        
        old_alerts = [
            alert for alert in self.system_alerts
            if alert.timestamp < cutoff_time
        ]
        
        for alert in old_alerts:
            self.system_alerts.remove(alert)
        
        if old_alerts:
            logger.info(f"Cleaned up {len(old_alerts)} old alerts")
    
    async def _auto_resolve_alerts(self) -> None:
        """Auto-resolve alerts based on current system state."""
        for alert in self.system_alerts:
            if alert.resolved:
                continue
            
            # Check if alert condition is resolved
            should_resolve = False
            
            if alert.alert_type == "health_critical":
                component_health = self.component_health.get(alert.component)
                if component_health and component_health.health_status != SystemHealthStatus.CRITICAL:
                    should_resolve = True
            
            elif alert.alert_type == "resource_usage":
                if self.system_metrics_history:
                    latest_metrics = self.system_metrics_history[-1]
                    if "cpu_high" in alert.alert_id and latest_metrics.cpu_usage < 85:
                        should_resolve = True
                    elif "memory_high" in alert.alert_id and latest_metrics.memory_usage < 90:
                        should_resolve = True
            
            if should_resolve:
                alert.resolved = True
                alert.resolution_time = datetime.utcnow()
                logger.info(f"Auto-resolved alert: {alert.alert_id}")
    
    async def _recalculate_baselines(self) -> None:
        """Recalculate performance baselines based on historical data."""
        if not self.system_metrics_history:
            return
        
        # Use all available historical data
        all_metrics = list(self.system_metrics_history)
        
        # Calculate new baselines using statistical methods
        cpu_values = [m.cpu_usage for m in all_metrics]
        memory_values = [m.memory_usage for m in all_metrics]
        response_time_values = [m.response_time for m in all_metrics]
        
        # Use median + 1.5 * IQR as baseline
        def calculate_baseline(values):
            q1 = statistics.quantiles(values, n=4)[0]
            q3 = statistics.quantiles(values, n=4)[2]
            iqr = q3 - q1
            return statistics.median(values) + 1.5 * iqr
        
        self.performance_baselines["cpu_usage"] = calculate_baseline(cpu_values)
        self.performance_baselines["memory_usage"] = calculate_baseline(memory_values)
        self.performance_baselines["response_time"] = calculate_baseline(response_time_values)
        
        logger.info("Recalculated performance baselines")
    
    # Error handling would be managed by interface implementation
    async def register_quality_services(self, services: Dict[str, Any]) -> None:
        """Register quality services for monitoring."""
        self.monitoring_service = services.get("monitoring_service")
        self.remediation_engine = services.get("remediation_engine")
        self.adaptive_controls = services.get("adaptive_controls")
        self.pipeline_framework = services.get("pipeline_framework")
        self.orchestration_service = services.get("orchestration_service")
        
        logger.info("Registered quality services for monitoring")
    
    # Error handling would be managed by interface implementation
    async def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        overall_health = SystemHealthStatus.HEALTHY
        
        # Determine overall health from components
        for health in self.component_health.values():
            if health.health_status == SystemHealthStatus.CRITICAL:
                overall_health = SystemHealthStatus.CRITICAL
                break
            elif health.health_status == SystemHealthStatus.UNHEALTHY:
                overall_health = SystemHealthStatus.UNHEALTHY
            elif health.health_status == SystemHealthStatus.DEGRADED and overall_health == SystemHealthStatus.HEALTHY:
                overall_health = SystemHealthStatus.DEGRADED
        
        return {
            "overall_health": overall_health.value,
            "components": {
                name: {
                    "health_status": health.health_status.value,
                    "performance_score": health.performance_score,
                    "last_check": health.last_check,
                    "issues": health.issues,
                    "recommendations": health.recommendations
                }
                for name, health in self.component_health.items()
            },
            "system_metrics": {
                "current": self.system_metrics_history[-1].__dict__ if self.system_metrics_history else {},
                "averages": self._calculate_metric_averages()
            },
            "alerts": {
                "active": len([a for a in self.system_alerts if not a.resolved]),
                "critical": len([a for a in self.system_alerts if a.severity == "critical" and not a.resolved]),
                "recent": [
                    {
                        "alert_id": alert.alert_id,
                        "severity": alert.severity,
                        "component": alert.component,
                        "message": alert.message,
                        "timestamp": alert.timestamp
                    }
                    for alert in self.system_alerts[-10:]
                ]
            }
        }
    
    def _calculate_metric_averages(self) -> Dict[str, float]:
        """Calculate metric averages from recent history."""
        if not self.system_metrics_history:
            return {}
        
        recent_metrics = list(self.system_metrics_history)[-10:]
        
        return {
            "cpu_usage": sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            "memory_usage": sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
            "disk_usage": sum(m.disk_usage for m in recent_metrics) / len(recent_metrics),
            "error_rate": sum(m.error_rate for m in recent_metrics) / len(recent_metrics),
            "response_time": sum(m.response_time for m in recent_metrics) / len(recent_metrics),
            "throughput": sum(m.throughput for m in recent_metrics) / len(recent_metrics)
        }
    
    # Error handling would be managed by interface implementation
    async def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations."""
        return [
            {
                "recommendation_id": rec.recommendation_id,
                "optimization_type": rec.optimization_type.value,
                "component": rec.component,
                "description": rec.description,
                "expected_improvement": rec.expected_improvement,
                "implementation_effort": rec.implementation_effort,
                "priority": rec.priority,
                "cost_benefit_ratio": rec.cost_benefit_ratio,
                "implementation_steps": rec.implementation_steps,
                "risks": rec.risks,
                "created_at": rec.created_at,
                "implemented": rec.implemented
            }
            for rec in self.optimization_recommendations
        ]
    
    # Error handling would be managed by interface implementation
    async def implement_optimization_recommendation(self, recommendation_id: str) -> bool:
        """Implement a specific optimization recommendation."""
        recommendation = next(
            (rec for rec in self.optimization_recommendations if rec.recommendation_id == recommendation_id),
            None
        )
        
        if not recommendation:
            return False
        
        try:
            await self._implement_optimization(recommendation)
            recommendation.implemented = True
            logger.info(f"Implemented optimization recommendation: {recommendation_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to implement optimization {recommendation_id}: {str(e)}")
            return False
    
    # Error handling would be managed by interface implementation
    async def get_cost_effectiveness_analysis(self) -> Dict[str, Any]:
        """Get cost-effectiveness analysis."""
        if not self.cost_metrics_history or not self.effectiveness_metrics_history:
            return {}
        
        latest_cost = self.cost_metrics_history[-1]
        latest_effectiveness = self.effectiveness_metrics_history[-1]
        
        # Calculate trends
        cost_trend = "stable"
        effectiveness_trend = "stable"
        
        if len(self.cost_metrics_history) >= 2:
            cost_change = (latest_cost.total_cost - self.cost_metrics_history[-2].total_cost) / self.cost_metrics_history[-2].total_cost
            if cost_change > 0.05:
                cost_trend = "increasing"
            elif cost_change < -0.05:
                cost_trend = "decreasing"
        
        if len(self.effectiveness_metrics_history) >= 2:
            effectiveness_change = (latest_effectiveness.quality_improvement_rate - 
                                  self.effectiveness_metrics_history[-2].quality_improvement_rate)
            if effectiveness_change > 0.02:
                effectiveness_trend = "improving"
            elif effectiveness_change < -0.02:
                effectiveness_trend = "degrading"
        
        return {
            "cost_metrics": {
                "total_cost": latest_cost.total_cost,
                "cost_per_record": latest_cost.cost_per_record,
                "trend": cost_trend
            },
            "effectiveness_metrics": {
                "quality_improvement_rate": latest_effectiveness.quality_improvement_rate,
                "automation_rate": latest_effectiveness.automation_rate,
                "user_satisfaction": latest_effectiveness.user_satisfaction_score,
                "trend": effectiveness_trend
            },
            "cost_effectiveness_ratio": latest_effectiveness.quality_improvement_rate / latest_cost.cost_per_record,
            "recommendations": [
                "Implement batch processing to reduce cost per record",
                "Increase automation to improve effectiveness",
                "Optimize resource allocation during peak hours"
            ]
        }
    
    async def shutdown(self) -> None:
        """Shutdown the self-monitoring and optimization service."""
        logger.info("Shutting down self-monitoring and optimization service...")
        
        # Clear all data
        self.system_metrics_history.clear()
        self.component_health.clear()
        self.optimization_recommendations.clear()
        self.cost_metrics_history.clear()
        self.effectiveness_metrics_history.clear()
        self.system_alerts.clear()
        
        logger.info("Self-monitoring and optimization service shutdown complete")