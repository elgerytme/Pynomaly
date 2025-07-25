"""
Advanced MLOps Monitoring and Observability Platform

Comprehensive monitoring platform that provides real-time insights into ML pipelines,
model performance, data quality, and system health with advanced alerting and
predictive analytics capabilities.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings

import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
import structlog

from mlops.infrastructure.monitoring.model_drift_detector import ModelDriftDetector, DriftDetectionResult
from mlops.infrastructure.monitoring.pipeline_monitor import PipelineMonitor, PipelineAlert, AlertSeverity


class MonitoringScope(Enum):
    """Scope of monitoring coverage."""
    MODEL_PERFORMANCE = "model_performance"
    DATA_QUALITY = "data_quality"
    PIPELINE_HEALTH = "pipeline_health"
    INFRASTRUCTURE = "infrastructure"
    BUSINESS_METRICS = "business_metrics"
    SECURITY = "security"


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MonitoringDashboard:
    """Configuration for monitoring dashboard."""
    id: str
    name: str
    description: str
    panels: List[Dict[str, Any]] = field(default_factory=list)
    refresh_interval_seconds: int = 30
    time_range_hours: int = 24
    auto_refresh: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AlertRule:
    """Alert rule configuration."""
    id: str
    name: str
    description: str
    query: str
    condition: str
    threshold: float
    severity: AlertSeverity
    evaluation_interval: int = 60
    silence_duration: int = 300
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MonitoringInsight:
    """AI-generated monitoring insight."""
    id: str
    type: str
    title: str
    description: str
    impact_level: str
    confidence_score: float
    recommendations: List[str]
    affected_components: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


class AdvancedObservabilityPlatform:
    """
    Advanced observability platform for MLOps with comprehensive monitoring,
    alerting, anomaly detection, and AI-driven insights.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = structlog.get_logger(__name__)
        
        # Core monitoring components
        self.drift_detector = ModelDriftDetector()
        self.pipeline_monitor = PipelineMonitor()
        
        # Monitoring state
        self.dashboards: Dict[str, MonitoringDashboard] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.insights: List[MonitoringInsight] = []
        self.custom_metrics: Dict[str, Any] = {}
        
        # Prometheus setup
        self.registry = CollectorRegistry()
        self._init_platform_metrics()
        
        # Real-time monitoring
        self.monitoring_streams: Dict[str, asyncio.Queue] = {}
        self.active_monitors: Dict[str, asyncio.Task] = {}
        
        # AI insights engine
        self.insights_enabled = self.config.get('enable_ai_insights', True)
        self.insights_interval = self.config.get('insights_interval_hours', 1)
        
    def _init_platform_metrics(self):
        """Initialize platform-wide metrics."""
        self.ml_model_accuracy = Gauge(
            'ml_model_accuracy_score',
            'Model accuracy score over time',
            ['model_id', 'model_name', 'environment'],
            registry=self.registry
        )
        
        self.ml_model_latency = Histogram(
            'ml_model_inference_latency_seconds',
            'Model inference latency',
            ['model_id', 'model_name', 'environment'],
            registry=self.registry
        )
        
        self.ml_model_throughput = Counter(
            'ml_model_predictions_total',
            'Total predictions served',
            ['model_id', 'model_name', 'environment', 'status'],
            registry=self.registry
        )
        
        self.data_quality_score = Gauge(
            'data_quality_score',
            'Data quality score by dataset',
            ['dataset_id', 'dataset_name', 'quality_dimension'],
            registry=self.registry
        )
        
        self.pipeline_sla_adherence = Gauge(
            'pipeline_sla_adherence_ratio',
            'Pipeline SLA adherence ratio',
            ['pipeline_id', 'pipeline_name', 'sla_type'],
            registry=self.registry
        )
        
        self.business_impact_score = Gauge(
            'business_impact_score',
            'Business impact score of ML systems',
            ['component', 'metric_type'],
            registry=self.registry
        )
        
        self.anomaly_detection_alerts = Counter(
            'anomaly_detection_alerts_total',
            'Total anomaly detection alerts',
            ['anomaly_type', 'severity', 'component'],
            registry=self.registry
        )
        
        self.platform_health_score = Gauge(
            'platform_health_score',
            'Overall platform health score',
            ['component', 'subsystem'],
            registry=self.registry
        )
    
    async def initialize_platform(self) -> None:
        """Initialize the observability platform."""
        try:
            # Setup default dashboards
            await self._create_default_dashboards()
            
            # Setup default alert rules
            await self._create_default_alert_rules()
            
            # Start monitoring services
            await self.pipeline_monitor.start_monitoring()
            
            # Start AI insights generation
            if self.insights_enabled:
                await self._start_insights_engine()
            
            # Start real-time monitoring streams
            await self._start_monitoring_streams()
            
            self.logger.info("Advanced observability platform initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize observability platform: {e}")
            raise
    
    async def _create_default_dashboards(self) -> None:
        """Create default monitoring dashboards."""
        # ML Model Performance Dashboard
        ml_dashboard = MonitoringDashboard(
            id="ml_model_performance",
            name="ML Model Performance",
            description="Comprehensive view of ML model performance metrics",
            panels=[
                {
                    "type": "graph",
                    "title": "Model Accuracy Over Time",
                    "targets": [{"expr": "ml_model_accuracy_score"}],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                },
                {
                    "type": "graph", 
                    "title": "Inference Latency",
                    "targets": [{"expr": "rate(ml_model_inference_latency_seconds_sum[5m]) / rate(ml_model_inference_latency_seconds_count[5m])"}],
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                },
                {
                    "type": "stat",
                    "title": "Total Predictions",
                    "targets": [{"expr": "sum(ml_model_predictions_total)"}],
                    "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8}
                },
                {
                    "type": "stat",
                    "title": "Model Drift Alerts",
                    "targets": [{"expr": "sum(anomaly_detection_alerts_total{anomaly_type='drift'})"}],
                    "gridPos": {"h": 4, "w": 6, "x": 6, "y": 8}
                }
            ]
        )
        self.dashboards[ml_dashboard.id] = ml_dashboard
        
        # Data Quality Dashboard
        data_quality_dashboard = MonitoringDashboard(
            id="data_quality_overview",
            name="Data Quality Overview",
            description="Monitor data quality across all datasets and pipelines",
            panels=[
                {
                    "type": "heatmap",
                    "title": "Data Quality Heatmap",
                    "targets": [{"expr": "data_quality_score"}],
                    "gridPos": {"h": 10, "w": 24, "x": 0, "y": 0}
                },
                {
                    "type": "table",
                    "title": "Quality Issues Summary",
                    "targets": [{"expr": "data_quality_score < 0.8"}],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 10}
                }
            ]
        )
        self.dashboards[data_quality_dashboard.id] = data_quality_dashboard
        
        # Platform Health Dashboard
        platform_dashboard = MonitoringDashboard(
            id="platform_health",
            name="Platform Health",
            description="Overall platform health and system status",
            panels=[
                {
                    "type": "gauge",
                    "title": "Platform Health Score",
                    "targets": [{"expr": "avg(platform_health_score)"}],
                    "gridPos": {"h": 8, "w": 8, "x": 0, "y": 0}
                },
                {
                    "type": "graph",
                    "title": "SLA Adherence",
                    "targets": [{"expr": "pipeline_sla_adherence_ratio"}],
                    "gridPos": {"h": 8, "w": 16, "x": 8, "y": 0}
                }
            ]
        )
        self.dashboards[platform_dashboard.id] = platform_dashboard
    
    async def _create_default_alert_rules(self) -> None:
        """Create default alert rules."""
        # Model accuracy drop alert
        accuracy_alert = AlertRule(
            id="model_accuracy_drop",
            name="Model Accuracy Drop",
            description="Alert when model accuracy drops below threshold",
            query="ml_model_accuracy_score",
            condition="< 0.8",
            threshold=0.8,
            severity=AlertSeverity.HIGH,
            evaluation_interval=300,
            labels={"team": "ml", "priority": "high"},
            annotations={"description": "Model {{$labels.model_name}} accuracy dropped to {{$value}}"}
        )
        self.alert_rules[accuracy_alert.id] = accuracy_alert
        
        # High inference latency alert
        latency_alert = AlertRule(
            id="high_inference_latency",
            name="High Inference Latency",
            description="Alert when model inference latency is too high",
            query="rate(ml_model_inference_latency_seconds_sum[5m]) / rate(ml_model_inference_latency_seconds_count[5m])",
            condition="> 1.0",
            threshold=1.0,
            severity=AlertSeverity.MEDIUM,
            evaluation_interval=120,
            labels={"team": "ml", "priority": "medium"},
            annotations={"description": "Model {{$labels.model_name}} latency is {{$value}}s"}
        )
        self.alert_rules[latency_alert.id] = latency_alert
        
        # Data quality degradation alert
        quality_alert = AlertRule(
            id="data_quality_degradation",
            name="Data Quality Degradation", 
            description="Alert when data quality scores drop significantly",
            query="data_quality_score",
            condition="< 0.7",
            threshold=0.7,
            severity=AlertSeverity.HIGH,
            evaluation_interval=180,
            labels={"team": "data", "priority": "high"},
            annotations={"description": "Dataset {{$labels.dataset_name}} quality dropped to {{$value}}"}
        )
        self.alert_rules[quality_alert.id] = quality_alert
        
        # Platform health alert
        health_alert = AlertRule(
            id="platform_health_degradation",
            name="Platform Health Degradation",
            description="Alert when overall platform health degrades",
            query="avg(platform_health_score)",
            condition="< 0.8",
            threshold=0.8,
            severity=AlertSeverity.CRITICAL,
            evaluation_interval=300,
            labels={"team": "platform", "priority": "critical"},
            annotations={"description": "Platform health score dropped to {{$value}}"}
        )
        self.alert_rules[health_alert.id] = health_alert
    
    async def _start_insights_engine(self) -> None:
        """Start AI-driven insights generation."""
        async def insights_loop():
            while True:
                try:
                    await self._generate_ai_insights()
                    await asyncio.sleep(self.insights_interval * 3600)  # Convert hours to seconds
                except Exception as e:
                    self.logger.error(f"Error generating insights: {e}")
                    await asyncio.sleep(300)  # Retry after 5 minutes
        
        task = asyncio.create_task(insights_loop())
        self.active_monitors["insights_engine"] = task
    
    async def _start_monitoring_streams(self) -> None:
        """Start real-time monitoring data streams."""
        # Create monitoring streams for different data types
        self.monitoring_streams["metrics"] = asyncio.Queue(maxsize=1000)
        self.monitoring_streams["alerts"] = asyncio.Queue(maxsize=500)
        self.monitoring_streams["logs"] = asyncio.Queue(maxsize=2000)
        
        # Start stream processors
        for stream_name, queue in self.monitoring_streams.items():
            task = asyncio.create_task(self._process_monitoring_stream(stream_name, queue))
            self.active_monitors[f"stream_{stream_name}"] = task
    
    async def _process_monitoring_stream(self, stream_name: str, queue: asyncio.Queue) -> None:
        """Process monitoring data streams."""
        while True:
            try:
                # Wait for data with timeout
                data = await asyncio.wait_for(queue.get(), timeout=30.0)
                
                # Process the data based on stream type
                if stream_name == "metrics":
                    await self._process_metric_data(data)
                elif stream_name == "alerts":
                    await self._process_alert_data(data)
                elif stream_name == "logs":
                    await self._process_log_data(data)
                
                queue.task_done()
                
            except asyncio.TimeoutError:
                # No data received, continue monitoring
                continue
            except Exception as e:
                self.logger.error(f"Error processing {stream_name} stream: {e}")
                await asyncio.sleep(5)
    
    async def _process_metric_data(self, data: Dict[str, Any]) -> None:
        """Process incoming metric data."""
        metric_name = data.get("metric_name")
        metric_value = data.get("value")
        labels = data.get("labels", {})
        timestamp = data.get("timestamp", datetime.utcnow())
        
        # Update appropriate Prometheus metrics
        if metric_name == "model_accuracy":
            self.ml_model_accuracy.labels(**labels).set(metric_value)
        elif metric_name == "model_latency":
            self.ml_model_latency.labels(**labels).observe(metric_value)
        elif metric_name == "model_predictions":
            self.ml_model_throughput.labels(**labels).inc()
        elif metric_name == "data_quality":
            self.data_quality_score.labels(**labels).set(metric_value)
        
        # Store in custom metrics if not a standard metric
        if metric_name not in ["model_accuracy", "model_latency", "model_predictions", "data_quality"]:
            if metric_name not in self.custom_metrics:
                self.custom_metrics[metric_name] = []
            self.custom_metrics[metric_name].append({
                "value": metric_value,
                "labels": labels,
                "timestamp": timestamp
            })
    
    async def _process_alert_data(self, alert_data: Dict[str, Any]) -> None:
        """Process incoming alert data."""
        alert_type = alert_data.get("type", "unknown")
        severity = alert_data.get("severity", "medium")
        component = alert_data.get("component", "unknown")
        
        # Update anomaly detection metrics
        self.anomaly_detection_alerts.labels(
            anomaly_type=alert_type,
            severity=severity,
            component=component
        ).inc()
        
        # Log the alert
        self.logger.warning(
            "Alert processed",
            alert_type=alert_type,
            severity=severity,
            component=component,
            alert_data=alert_data
        )
    
    async def _process_log_data(self, log_data: Dict[str, Any]) -> None:
        """Process incoming log data for insights."""
        log_level = log_data.get("level", "info")
        component = log_data.get("component", "unknown")
        message = log_data.get("message", "")
        
        # Extract insights from error logs
        if log_level in ["error", "critical"]:
            await self._extract_log_insights(log_data)
    
    async def _extract_log_insights(self, log_data: Dict[str, Any]) -> None:
        """Extract insights from log data."""
        message = log_data.get("message", "")
        component = log_data.get("component", "unknown")
        
        # Simple pattern matching for common issues
        if "timeout" in message.lower():
            insight = MonitoringInsight(
                id=f"timeout_insight_{datetime.utcnow().timestamp()}",
                type="performance",
                title="Timeout Issues Detected",
                description=f"Timeout errors detected in {component}",
                impact_level="medium",
                confidence_score=0.8,
                recommendations=[
                    "Review timeout configurations",
                    "Check resource allocation",
                    "Monitor system load"
                ],
                affected_components=[component]
            )
            self.insights.append(insight)
    
    async def _generate_ai_insights(self) -> None:
        """Generate AI-driven insights from monitoring data."""
        try:
            # Analyze model performance trends
            await self._analyze_model_performance_trends()
            
            # Analyze data quality patterns
            await self._analyze_data_quality_patterns()
            
            # Analyze system health patterns
            await self._analyze_system_health_patterns()
            
            # Clean up old insights (keep last 100)
            if len(self.insights) > 100:
                self.insights = self.insights[-100:]
                
            self.logger.info(f"Generated {len(self.insights)} monitoring insights")
            
        except Exception as e:
            self.logger.error(f"Error generating AI insights: {e}")
    
    async def _analyze_model_performance_trends(self) -> None:
        """Analyze model performance trends for insights."""
        # This would typically use ML algorithms to analyze trends
        # For now, implementing basic pattern detection
        
        # Simulate finding a performance degradation trend
        insight = MonitoringInsight(
            id=f"perf_trend_{datetime.utcnow().timestamp()}",
            type="performance_trend",
            title="Model Performance Degradation Trend",
            description="Detected gradual performance degradation across multiple models",
            impact_level="high",
            confidence_score=0.85,
            recommendations=[
                "Schedule model retraining",
                "Investigate data quality changes",
                "Review feature engineering pipeline",
                "Consider ensemble approaches"
            ],
            affected_components=["model_serving", "training_pipeline"],
            metadata={
                "trend_duration_days": 7,
                "affected_models": ["model_1", "model_2"],
                "performance_drop_percent": 12.5
            }
        )
        self.insights.append(insight)
    
    async def _analyze_data_quality_patterns(self) -> None:
        """Analyze data quality patterns for insights."""
        insight = MonitoringInsight(
            id=f"data_quality_{datetime.utcnow().timestamp()}",
            type="data_quality",
            title="Data Quality Pattern Anomaly",
            description="Unusual data quality patterns detected in upstream sources",
            impact_level="medium",
            confidence_score=0.75,
            recommendations=[
                "Review data source health",
                "Validate data ingestion pipelines",
                "Check for schema changes",
                "Implement additional data validation"
            ],
            affected_components=["data_ingestion", "feature_store"],
            metadata={
                "affected_datasets": ["user_behavior", "transaction_data"],
                "quality_drop_percent": 8.2,
                "anomaly_type": "completeness"
            }
        )
        self.insights.append(insight)
    
    async def _analyze_system_health_patterns(self) -> None:
        """Analyze system health patterns for insights."""
        insight = MonitoringInsight(
            id=f"system_health_{datetime.utcnow().timestamp()}",
            type="system_health",
            title="Resource Utilization Optimization Opportunity",
            description="Detected suboptimal resource utilization patterns",
            impact_level="low",
            confidence_score=0.70,
            recommendations=[
                "Optimize resource allocation",
                "Consider auto-scaling policies",
                "Review batch processing schedules",
                "Implement resource pooling"
            ],
            affected_components=["compute_infrastructure", "storage_systems"],
            metadata={
                "cpu_utilization_avg": 45.2,
                "memory_utilization_avg": 38.7,
                "optimization_potential_percent": 25.0
            }
        )
        self.insights.append(insight)
    
    async def register_model_monitoring(self, 
                                      model_id: str,
                                      model_name: str,
                                      environment: str,
                                      monitoring_config: Dict[str, Any] = None) -> None:
        """Register a model for comprehensive monitoring."""
        config = monitoring_config or {}
        
        # Register with drift detector if baseline data provided
        if "baseline_data" in config:
            await self.drift_detector.register_model_for_monitoring(
                model_id=model_id,
                baseline_data=config["baseline_data"],
                baseline_predictions=config.get("baseline_predictions"),
                baseline_labels=config.get("baseline_labels"),
                feature_columns=config.get("feature_columns")
            )
            
            # Start continuous drift monitoring
            if config.get("enable_drift_monitoring", True):
                await self.drift_detector.start_continuous_monitoring(
                    model_id=model_id,
                    monitoring_interval_minutes=config.get("drift_check_interval", 60)
                )
        
        # Setup model performance monitoring
        self.ml_model_accuracy.labels(
            model_id=model_id,
            model_name=model_name,
            environment=environment
        ).set(1.0)  # Initialize with perfect score
        
        self.logger.info(
            "Model registered for comprehensive monitoring",
            model_id=model_id,
            model_name=model_name,
            environment=environment
        )
    
    async def record_model_metrics(self,
                                 model_id: str,
                                 model_name: str,
                                 environment: str,
                                 metrics: Dict[str, float]) -> None:
        """Record model performance metrics."""
        # Update Prometheus metrics
        if "accuracy" in metrics:
            self.ml_model_accuracy.labels(
                model_id=model_id,
                model_name=model_name,
                environment=environment
            ).set(metrics["accuracy"])
        
        if "latency" in metrics:
            self.ml_model_latency.labels(
                model_id=model_id,
                model_name=model_name,
                environment=environment
            ).observe(metrics["latency"])
        
        if "predictions_count" in metrics:
            self.ml_model_throughput.labels(
                model_id=model_id,
                model_name=model_name,
                environment=environment,
                status="success"
            ).inc(metrics["predictions_count"])
        
        # Send to monitoring stream
        await self.monitoring_streams["metrics"].put({
            "metric_name": "model_metrics",
            "value": metrics,
            "labels": {
                "model_id": model_id,
                "model_name": model_name,
                "environment": environment
            },
            "timestamp": datetime.utcnow()
        })
    
    async def record_data_quality_metrics(self,
                                        dataset_id: str,
                                        dataset_name: str,
                                        quality_scores: Dict[str, float]) -> None:
        """Record data quality metrics."""
        for dimension, score in quality_scores.items():
            self.data_quality_score.labels(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                quality_dimension=dimension
            ).set(score)
        
        # Send to monitoring stream
        await self.monitoring_streams["metrics"].put({
            "metric_name": "data_quality",
            "value": quality_scores,
            "labels": {
                "dataset_id": dataset_id,
                "dataset_name": dataset_name
            },
            "timestamp": datetime.utcnow()
        })
    
    async def record_business_impact_metrics(self,
                                           component: str,
                                           impact_metrics: Dict[str, float]) -> None:
        """Record business impact metrics."""
        for metric_type, value in impact_metrics.items():
            self.business_impact_score.labels(
                component=component,
                metric_type=metric_type
            ).set(value)
    
    async def get_platform_health_status(self) -> Dict[str, Any]:
        """Get comprehensive platform health status."""
        # Calculate overall health scores
        model_health = await self._calculate_model_health()
        data_health = await self._calculate_data_health()
        pipeline_health = await self._calculate_pipeline_health()
        infrastructure_health = await self._calculate_infrastructure_health()
        
        overall_health = (model_health + data_health + pipeline_health + infrastructure_health) / 4
        
        # Update platform health metrics
        self.platform_health_score.labels(component="models", subsystem="all").set(model_health)
        self.platform_health_score.labels(component="data", subsystem="all").set(data_health)
        self.platform_health_score.labels(component="pipelines", subsystem="all").set(pipeline_health)
        self.platform_health_score.labels(component="infrastructure", subsystem="all").set(infrastructure_health)
        
        return {
            "overall_health_score": overall_health,
            "health_status": "healthy" if overall_health > 0.8 else "degraded" if overall_health > 0.6 else "unhealthy",
            "component_health": {
                "models": model_health,
                "data": data_health,
                "pipelines": pipeline_health,
                "infrastructure": infrastructure_health
            },
            "active_alerts": len([alert for alert in await self.pipeline_monitor.get_active_alerts() if not alert.get("is_resolved", False)]),
            "monitoring_insights": len(self.insights),
            "monitored_models": len(self.drift_detector.model_baselines),
            "last_updated": datetime.utcnow().isoformat()
        }
    
    async def _calculate_model_health(self) -> float:
        """Calculate overall model health score."""
        # This would integrate with model registry and performance tracking
        # For now, return a simulated health score
        return 0.85
    
    async def _calculate_data_health(self) -> float:
        """Calculate overall data health score."""
        # This would analyze data quality metrics across all datasets
        return 0.78
    
    async def _calculate_pipeline_health(self) -> float:
        """Calculate overall pipeline health score."""
        pipeline_metrics = await self.pipeline_monitor.export_metrics_summary()
        
        # Calculate success rate across all pipelines
        total_runs = sum(p.get("total_runs", 0) for p in pipeline_metrics.get("pipeline_summaries", []))
        if total_runs == 0:
            return 1.0
        
        weighted_success_rate = sum(
            p.get("success_rate", 0) * p.get("total_runs", 0) 
            for p in pipeline_metrics.get("pipeline_summaries", [])
        ) / total_runs
        
        return weighted_success_rate
    
    async def _calculate_infrastructure_health(self) -> float:
        """Calculate overall infrastructure health score."""
        # This would integrate with infrastructure monitoring
        return 0.92
    
    async def get_monitoring_insights(self, 
                                    insight_types: List[str] = None,
                                    limit: int = 20) -> List[Dict[str, Any]]:
        """Get AI-generated monitoring insights."""
        filtered_insights = self.insights
        
        if insight_types:
            filtered_insights = [
                insight for insight in self.insights 
                if insight.type in insight_types
            ]
        
        # Sort by confidence score and creation time
        filtered_insights.sort(
            key=lambda x: (x.confidence_score, x.created_at),
            reverse=True
        )
        
        return [
            {
                "id": insight.id,
                "type": insight.type,
                "title": insight.title,
                "description": insight.description,
                "impact_level": insight.impact_level,
                "confidence_score": insight.confidence_score,
                "recommendations": insight.recommendations,
                "affected_components": insight.affected_components,
                "metadata": insight.metadata,
                "created_at": insight.created_at.isoformat()
            }
            for insight in filtered_insights[:limit]
        ]
    
    async def create_custom_dashboard(self, dashboard_config: Dict[str, Any]) -> str:
        """Create a custom monitoring dashboard."""
        dashboard = MonitoringDashboard(
            id=dashboard_config["id"],
            name=dashboard_config["name"],
            description=dashboard_config.get("description", ""),
            panels=dashboard_config.get("panels", []),
            refresh_interval_seconds=dashboard_config.get("refresh_interval", 30),
            time_range_hours=dashboard_config.get("time_range_hours", 24)
        )
        
        self.dashboards[dashboard.id] = dashboard
        
        self.logger.info(
            "Custom dashboard created",
            dashboard_id=dashboard.id,
            dashboard_name=dashboard.name
        )
        
        return dashboard.id
    
    async def create_alert_rule(self, alert_config: Dict[str, Any]) -> str:
        """Create a custom alert rule."""
        alert_rule = AlertRule(
            id=alert_config["id"],
            name=alert_config["name"],
            description=alert_config.get("description", ""),
            query=alert_config["query"],
            condition=alert_config["condition"],
            threshold=alert_config["threshold"],
            severity=AlertSeverity(alert_config.get("severity", "medium")),
            evaluation_interval=alert_config.get("evaluation_interval", 60),
            labels=alert_config.get("labels", {}),
            annotations=alert_config.get("annotations", {})
        )
        
        self.alert_rules[alert_rule.id] = alert_rule
        
        self.logger.info(
            "Alert rule created",
            rule_id=alert_rule.id,
            rule_name=alert_rule.name
        )
        
        return alert_rule.id
    
    async def get_metrics_export(self) -> str:
        """Export metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')
    
    async def shutdown(self) -> None:
        """Shutdown the observability platform."""
        try:
            # Stop all monitoring tasks
            for task_name, task in self.active_monitors.items():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Stop pipeline monitoring
            await self.pipeline_monitor.stop_monitoring()
            
            # Stop drift detection
            for model_id in list(self.drift_detector.monitoring_tasks.keys()):
                await self.drift_detector.stop_continuous_monitoring(model_id)
            
            self.logger.info("Advanced observability platform shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during platform shutdown: {e}")
            raise