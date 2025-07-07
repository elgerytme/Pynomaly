"""
Enterprise-grade observability and monitoring service using OpenTelemetry.

This service provides comprehensive monitoring capabilities:
- Distributed tracing with OpenTelemetry
- Metrics collection (Prometheus compatible)
- Application performance monitoring (APM)
- Health checks and readiness probes
- Business metrics and SLA monitoring
- Real-time dashboards and alerting
"""

from __future__ import annotations

import asyncio
import logging
import os
import psutil
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import numpy as np

# OpenTelemetry imports
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.b3 import B3MultiFormat
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.semconv.trace import SpanAttributes
    from opentelemetry.trace import Status, StatusCode
    
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics we collect."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram" 
    SUMMARY = "summary"


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    description: str
    check_function: callable
    critical: bool = False
    timeout_seconds: float = 5.0
    interval_seconds: float = 30.0
    last_check: Optional[datetime] = None
    last_status: HealthStatus = HealthStatus.HEALTHY
    last_error: Optional[str] = None


@dataclass
class MetricDefinition:
    """Metric definition for automatic collection."""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms


@dataclass
class Alert:
    """Alert definition."""
    id: str
    name: str
    condition: str
    threshold: float
    severity: str
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class ObservabilityConfig:
    """Configuration for observability service."""
    
    # Service identification
    service_name: str = "pynomaly"
    service_version: str = "0.1.0"
    environment: str = "development"
    
    # OpenTelemetry configuration
    enable_tracing: bool = True
    enable_metrics: bool = True
    trace_sample_rate: float = 1.0
    
    # Exporters
    jaeger_endpoint: Optional[str] = None
    otlp_endpoint: Optional[str] = None
    prometheus_port: int = 8090
    
    # Health checks
    enable_health_checks: bool = True
    health_check_port: int = 8091
    
    # Monitoring intervals
    metrics_collection_interval: float = 15.0
    health_check_interval: float = 30.0
    
    # Resource monitoring
    monitor_system_resources: bool = True
    monitor_application_metrics: bool = True
    monitor_business_metrics: bool = True
    
    # Alerting
    enable_alerting: bool = True
    alert_check_interval: float = 60.0


class SystemMetricsCollector:
    """Collects system-level metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Prometheus metrics (if available)
        if PROMETHEUS_AVAILABLE:
            self.registry = CollectorRegistry()
            self.cpu_usage = Gauge(
                'system_cpu_usage_percent', 
                'System CPU usage percentage',
                registry=self.registry
            )
            self.memory_usage = Gauge(
                'system_memory_usage_bytes',
                'System memory usage in bytes',
                ['type'],
                registry=self.registry
            )
            self.disk_usage = Gauge(
                'system_disk_usage_bytes',
                'System disk usage in bytes', 
                ['device', 'type'],
                registry=self.registry
            )
            self.network_io = Counter(
                'system_network_io_bytes_total',
                'System network I/O in bytes',
                ['direction'],
                registry=self.registry
            )
        
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        metrics = {}
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            
            metrics['cpu'] = {
                'usage_percent': cpu_percent,
                'count': cpu_count,
                'load_avg_1m': load_avg[0],
                'load_avg_5m': load_avg[1],
                'load_avg_15m': load_avg[2]
            }
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            metrics['memory'] = {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent': memory.percent,
                'swap_total': swap.total,
                'swap_used': swap.used,
                'swap_percent': swap.percent
            }
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            metrics['disk'] = {
                'total': disk_usage.total,
                'used': disk_usage.used,
                'free': disk_usage.free,
                'percent': (disk_usage.used / disk_usage.total) * 100,
                'read_bytes': disk_io.read_bytes if disk_io else 0,
                'write_bytes': disk_io.write_bytes if disk_io else 0
            }
            
            # Network metrics
            network_io = psutil.net_io_counters()
            
            metrics['network'] = {
                'bytes_sent': network_io.bytes_sent,
                'bytes_recv': network_io.bytes_recv,
                'packets_sent': network_io.packets_sent,
                'packets_recv': network_io.packets_recv
            }
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                self.cpu_usage.set(cpu_percent)
                self.memory_usage.labels(type='used').set(memory.used)
                self.memory_usage.labels(type='available').set(memory.available)
                self.disk_usage.labels(device='root', type='used').set(disk_usage.used)
                self.disk_usage.labels(device='root', type='free').set(disk_usage.free)
                
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            
        return metrics


class ApplicationMetricsCollector:
    """Collects application-specific metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
        
        # Application metrics
        self.request_counts = {}
        self.response_times = []
        self.error_counts = {}
        self.active_sessions = 0
        
        # ML model metrics
        self.model_predictions = 0
        self.model_training_jobs = 0
        self.anomalies_detected = 0
        self.datasets_processed = 0
        
        if PROMETHEUS_AVAILABLE:
            self.registry = CollectorRegistry()
            
            # Application metrics
            self.app_requests_total = Counter(
                'app_requests_total',
                'Total application requests',
                ['method', 'endpoint', 'status'],
                registry=self.registry
            )
            
            self.app_request_duration = Histogram(
                'app_request_duration_seconds',
                'Application request duration',
                ['method', 'endpoint'],
                buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
                registry=self.registry
            )
            
            self.app_active_sessions = Gauge(
                'app_active_sessions',
                'Number of active sessions',
                registry=self.registry
            )
            
            # ML model metrics
            self.ml_predictions_total = Counter(
                'ml_predictions_total',
                'Total ML model predictions',
                ['model_type', 'status'],
                registry=self.registry
            )
            
            self.ml_anomalies_detected = Counter(
                'ml_anomalies_detected_total',
                'Total anomalies detected',
                ['detector_type', 'severity'],
                registry=self.registry
            )
            
            self.ml_model_accuracy = Gauge(
                'ml_model_accuracy',
                'ML model accuracy score',
                ['model_name'],
                registry=self.registry
            )
            
            self.ml_dataset_size = Gauge(
                'ml_dataset_size_bytes',
                'Size of processed datasets',
                ['dataset_type'],
                registry=self.registry
            )
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        key = f"{method}:{endpoint}"
        self.request_counts[key] = self.request_counts.get(key, 0) + 1
        self.response_times.append(duration)
        
        if status_code >= 400:
            self.error_counts[key] = self.error_counts.get(key, 0) + 1
            
        if PROMETHEUS_AVAILABLE:
            self.app_requests_total.labels(
                method=method, 
                endpoint=endpoint, 
                status=str(status_code)
            ).inc()
            self.app_request_duration.labels(
                method=method, 
                endpoint=endpoint
            ).observe(duration)
    
    def record_ml_prediction(self, model_type: str, success: bool, anomaly_detected: bool = False):
        """Record ML prediction metrics."""
        self.model_predictions += 1
        
        if anomaly_detected:
            self.anomalies_detected += 1
            
        if PROMETHEUS_AVAILABLE:
            status = 'success' if success else 'failure'
            self.ml_predictions_total.labels(
                model_type=model_type,
                status=status
            ).inc()
            
            if anomaly_detected:
                self.ml_anomalies_detected.labels(
                    detector_type=model_type,
                    severity='unknown'
                ).inc()
    
    def update_model_accuracy(self, model_name: str, accuracy: float):
        """Update model accuracy metric."""
        if PROMETHEUS_AVAILABLE:
            self.ml_model_accuracy.labels(model_name=model_name).set(accuracy)
    
    def record_dataset_processed(self, dataset_type: str, size_bytes: int):
        """Record dataset processing metrics."""
        self.datasets_processed += 1
        
        if PROMETHEUS_AVAILABLE:
            self.ml_dataset_size.labels(dataset_type=dataset_type).set(size_bytes)
    
    def set_active_sessions(self, count: int):
        """Update active sessions count."""
        self.active_sessions = count
        
        if PROMETHEUS_AVAILABLE:
            self.app_active_sessions.set(count)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current application metrics."""
        uptime = time.time() - self.start_time
        
        # Calculate response time statistics
        response_stats = {}
        if self.response_times:
            response_stats = {
                'min': min(self.response_times),
                'max': max(self.response_times),
                'mean': np.mean(self.response_times),
                'p50': np.percentile(self.response_times, 50),
                'p95': np.percentile(self.response_times, 95),
                'p99': np.percentile(self.response_times, 99)
            }
        
        return {
            'uptime_seconds': uptime,
            'request_counts': self.request_counts,
            'error_counts': self.error_counts,
            'response_time_stats': response_stats,
            'active_sessions': self.active_sessions,
            'ml_metrics': {
                'total_predictions': self.model_predictions,
                'anomalies_detected': self.anomalies_detected,
                'datasets_processed': self.datasets_processed,
                'training_jobs': self.model_training_jobs
            }
        }


class HealthCheckService:
    """Service for health checks and readiness probes."""
    
    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.health_checks: Dict[str, HealthCheck] = {}
        self.overall_status = HealthStatus.HEALTHY
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        
        # Database connectivity check
        self.register_health_check(
            name="database",
            description="Database connectivity",
            check_function=self._check_database,
            critical=True
        )
        
        # Memory usage check
        self.register_health_check(
            name="memory",
            description="System memory usage",
            check_function=self._check_memory_usage,
            critical=False
        )
        
        # Disk space check
        self.register_health_check(
            name="disk_space",
            description="Available disk space",
            check_function=self._check_disk_space,
            critical=True
        )
        
        # External service dependencies
        self.register_health_check(
            name="external_services",
            description="External service dependencies",
            check_function=self._check_external_services,
            critical=False
        )
    
    def register_health_check(
        self,
        name: str,
        description: str,
        check_function: callable,
        critical: bool = False,
        timeout_seconds: float = 5.0,
        interval_seconds: float = 30.0
    ):
        """Register a new health check."""
        health_check = HealthCheck(
            name=name,
            description=description,
            check_function=check_function,
            critical=critical,
            timeout_seconds=timeout_seconds,
            interval_seconds=interval_seconds
        )
        
        self.health_checks[name] = health_check
        self.logger.info(f"Registered health check: {name}")
    
    async def run_health_check(self, name: str) -> HealthStatus:
        """Run a specific health check."""
        if name not in self.health_checks:
            return HealthStatus.UNHEALTHY
        
        health_check = self.health_checks[name]
        
        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                asyncio.create_task(health_check.check_function()),
                timeout=health_check.timeout_seconds
            )
            
            if result:
                health_check.last_status = HealthStatus.HEALTHY
                health_check.last_error = None
            else:
                health_check.last_status = HealthStatus.UNHEALTHY
                health_check.last_error = "Check returned False"
                
        except asyncio.TimeoutError:
            health_check.last_status = HealthStatus.UNHEALTHY
            health_check.last_error = f"Timeout after {health_check.timeout_seconds}s"
            
        except Exception as e:
            health_check.last_status = HealthStatus.UNHEALTHY
            health_check.last_error = str(e)
            
        health_check.last_check = datetime.utcnow()
        return health_check.last_status
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        critical_failures = 0
        total_failures = 0
        
        # Run all checks concurrently
        tasks = []
        for name in self.health_checks:
            task = asyncio.create_task(self.run_health_check(name))
            tasks.append((name, task))
        
        for name, task in tasks:
            status = await task
            health_check = self.health_checks[name]
            
            results[name] = {
                'status': status.value,
                'description': health_check.description,
                'critical': health_check.critical,
                'last_check': health_check.last_check.isoformat() if health_check.last_check else None,
                'error': health_check.last_error
            }
            
            if status != HealthStatus.HEALTHY:
                total_failures += 1
                if health_check.critical:
                    critical_failures += 1
        
        # Determine overall status
        if critical_failures > 0:
            self.overall_status = HealthStatus.CRITICAL
        elif total_failures > len(self.health_checks) // 2:
            self.overall_status = HealthStatus.UNHEALTHY
        elif total_failures > 0:
            self.overall_status = HealthStatus.DEGRADED
        else:
            self.overall_status = HealthStatus.HEALTHY
        
        return {
            'overall_status': self.overall_status.value,
            'checks': results,
            'summary': {
                'total_checks': len(self.health_checks),
                'failed_checks': total_failures,
                'critical_failures': critical_failures
            },
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _check_database(self) -> bool:
        """Check database connectivity."""
        try:
            # This would normally check actual database connection
            # For now, simulate a database check
            await asyncio.sleep(0.1)  # Simulate DB query
            return True
        except Exception:
            return False
    
    async def _check_memory_usage(self) -> bool:
        """Check system memory usage."""
        try:
            memory = psutil.virtual_memory()
            # Fail if memory usage > 90%
            return memory.percent < 90.0
        except Exception:
            return False
    
    async def _check_disk_space(self) -> bool:
        """Check available disk space."""
        try:
            disk = psutil.disk_usage('/')
            # Fail if disk usage > 85%
            usage_percent = (disk.used / disk.total) * 100
            return usage_percent < 85.0
        except Exception:
            return False
    
    async def _check_external_services(self) -> bool:
        """Check external service dependencies."""
        try:
            # This would check external APIs, services, etc.
            # For now, always return healthy
            return True
        except Exception:
            return False


class ObservabilityService:
    """Main observability service orchestrating all monitoring capabilities."""
    
    def __init__(self, config: Optional[ObservabilityConfig] = None):
        """Initialize observability service."""
        self.config = config or ObservabilityConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.system_metrics = SystemMetricsCollector()
        self.app_metrics = ApplicationMetricsCollector()
        self.health_service = HealthCheckService(self.config)
        
        # Metrics storage
        self.metrics_buffer: List[Dict[str, Any]] = []
        self.alerts: Dict[str, Alert] = {}
        
        # Background tasks
        self._running = False
        self._metrics_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._alerts_task: Optional[asyncio.Task] = None
        
        # Initialize OpenTelemetry
        if OPENTELEMETRY_AVAILABLE:
            self._setup_opentelemetry()
        else:
            self.logger.warning("OpenTelemetry not available - some features disabled")
        
        self.logger.info("Observability service initialized")
    
    def _setup_opentelemetry(self):
        """Set up OpenTelemetry tracing and metrics."""
        if not OPENTELEMETRY_AVAILABLE:
            return
        
        # Create resource
        resource = Resource.create({
            SERVICE_NAME: self.config.service_name,
            SERVICE_VERSION: self.config.service_version,
            "environment": self.config.environment
        })
        
        # Set up tracing
        if self.config.enable_tracing:
            tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(tracer_provider)
            
            # Add exporters
            exporters = []
            
            if self.config.jaeger_endpoint:
                jaeger_exporter = JaegerExporter(
                    agent_host_name="localhost",
                    agent_port=14268,
                )
                exporters.append(jaeger_exporter)
            
            if self.config.otlp_endpoint:
                otlp_exporter = OTLPSpanExporter(endpoint=self.config.otlp_endpoint)
                exporters.append(otlp_exporter)
            
            for exporter in exporters:
                span_processor = BatchSpanProcessor(exporter)
                tracer_provider.add_span_processor(span_processor)
            
            # Set propagator
            set_global_textmap(B3MultiFormat())
            
            self.tracer = trace.get_tracer(__name__)
        
        # Set up metrics
        if self.config.enable_metrics:
            readers = []
            
            # Prometheus reader
            if PROMETHEUS_AVAILABLE:
                prometheus_reader = PrometheusMetricReader(port=self.config.prometheus_port)
                readers.append(prometheus_reader)
            
            # OTLP reader
            if self.config.otlp_endpoint:
                otlp_reader = PeriodicExportingMetricReader(
                    OTLPMetricExporter(endpoint=self.config.otlp_endpoint),
                    export_interval_millis=int(self.config.metrics_collection_interval * 1000)
                )
                readers.append(otlp_reader)
            
            meter_provider = MeterProvider(resource=resource, metric_readers=readers)
            metrics.set_meter_provider(meter_provider)
            
            self.meter = metrics.get_meter(__name__)
    
    @contextmanager
    def trace_operation(self, operation_name: str, **attributes):
        """Context manager for tracing operations."""
        if not OPENTELEMETRY_AVAILABLE or not hasattr(self, 'tracer'):
            yield
            return
        
        with self.tracer.start_as_current_span(operation_name) as span:
            # Add attributes
            for key, value in attributes.items():
                span.set_attribute(key, value)
            
            start_time = time.time()
            
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
                
            finally:
                duration = time.time() - start_time
                span.set_attribute("duration_ms", duration * 1000)
    
    def record_business_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a business metric."""
        metric_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'name': name,
            'value': value,
            'labels': labels or {},
            'type': 'business'
        }
        
        self.metrics_buffer.append(metric_data)
        self.logger.debug(f"Recorded business metric: {name}={value}")
    
    def add_alert(self, name: str, condition: str, threshold: float, severity: str = "warning"):
        """Add a new alert rule."""
        alert = Alert(
            id=str(uuid4()),
            name=name,
            condition=condition,
            threshold=threshold,
            severity=severity
        )
        
        self.alerts[alert.id] = alert
        self.logger.info(f"Added alert: {name}")
        return alert.id
    
    async def start(self):
        """Start the observability service."""
        if self._running:
            return
        
        self._running = True
        self.logger.info("Starting observability service...")
        
        # Start background tasks
        if self.config.enable_metrics:
            self._metrics_task = asyncio.create_task(self._metrics_collection_loop())
        
        if self.config.enable_health_checks:
            self._health_task = asyncio.create_task(self._health_check_loop())
        
        if self.config.enable_alerting:
            self._alerts_task = asyncio.create_task(self._alerts_loop())
        
        self.logger.info("Observability service started")
    
    async def stop(self):
        """Stop the observability service."""
        if not self._running:
            return
        
        self._running = False
        self.logger.info("Stopping observability service...")
        
        # Cancel background tasks
        for task in [self._metrics_task, self._health_task, self._alerts_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.logger.info("Observability service stopped")
    
    async def _metrics_collection_loop(self):
        """Background loop for metrics collection."""
        while self._running:
            try:
                # Collect system metrics
                if self.config.monitor_system_resources:
                    system_metrics = self.system_metrics.collect_metrics()
                    
                    metric_data = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'type': 'system',
                        'data': system_metrics
                    }
                    self.metrics_buffer.append(metric_data)
                
                # Collect application metrics
                if self.config.monitor_application_metrics:
                    app_metrics = self.app_metrics.get_metrics()
                    
                    metric_data = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'type': 'application',
                        'data': app_metrics
                    }
                    self.metrics_buffer.append(metric_data)
                
                # Limit buffer size
                if len(self.metrics_buffer) > 1000:
                    self.metrics_buffer = self.metrics_buffer[-500:]
                
                await asyncio.sleep(self.config.metrics_collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(5)
    
    async def _health_check_loop(self):
        """Background loop for health checks."""
        while self._running:
            try:
                await self.health_service.run_all_checks()
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)
    
    async def _alerts_loop(self):
        """Background loop for alert evaluation."""
        while self._running:
            try:
                await self._evaluate_alerts()
                await asyncio.sleep(self.config.alert_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in alerts loop: {e}")
                await asyncio.sleep(5)
    
    async def _evaluate_alerts(self):
        """Evaluate all alert conditions."""
        current_time = datetime.utcnow()
        
        for alert in self.alerts.values():
            if not alert.enabled:
                continue
            
            try:
                # Get current metrics for evaluation
                # This is a simplified implementation
                should_trigger = await self._evaluate_alert_condition(alert)
                
                if should_trigger:
                    if not alert.last_triggered or (
                        current_time - alert.last_triggered
                    ).total_seconds() > 300:  # 5 minute cooldown
                        
                        alert.last_triggered = current_time
                        alert.trigger_count += 1
                        
                        self.logger.warning(
                            f"Alert triggered: {alert.name} (condition: {alert.condition})"
                        )
                        
                        # Send alert notification
                        await self._send_alert_notification(alert)
                        
            except Exception as e:
                self.logger.error(f"Error evaluating alert {alert.name}: {e}")
    
    async def _evaluate_alert_condition(self, alert: Alert) -> bool:
        """Evaluate if an alert condition is met."""
        # This is a simplified implementation
        # In practice, this would parse the condition and evaluate against current metrics
        
        if alert.condition == "cpu_usage > threshold":
            system_metrics = self.system_metrics.collect_metrics()
            cpu_usage = system_metrics.get('cpu', {}).get('usage_percent', 0)
            return cpu_usage > alert.threshold
        
        elif alert.condition == "memory_usage > threshold":
            system_metrics = self.system_metrics.collect_metrics()
            memory_percent = system_metrics.get('memory', {}).get('percent', 0)
            return memory_percent > alert.threshold
        
        elif alert.condition == "error_rate > threshold":
            app_metrics = self.app_metrics.get_metrics()
            total_requests = sum(app_metrics.get('request_counts', {}).values())
            total_errors = sum(app_metrics.get('error_counts', {}).values())
            error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
            return error_rate > alert.threshold
        
        return False
    
    async def _send_alert_notification(self, alert: Alert):
        """Send alert notification."""
        # This would integrate with notification systems like:
        # - Slack
        # - PagerDuty
        # - Email
        # - SMS
        
        self.logger.critical(
            f"ALERT: {alert.name} | Severity: {alert.severity} | "
            f"Condition: {alert.condition} | Threshold: {alert.threshold}"
        )
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus formatted metrics."""
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus not available"
        
        # Combine metrics from all collectors
        output = []
        
        # System metrics
        if hasattr(self.system_metrics, 'registry'):
            output.append(generate_latest(self.system_metrics.registry).decode())
        
        # Application metrics
        if hasattr(self.app_metrics, 'registry'):
            output.append(generate_latest(self.app_metrics.registry).decode())
        
        return '\n'.join(output)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            'service_info': {
                'name': self.config.service_name,
                'version': self.config.service_version,
                'environment': self.config.environment
            },
            'system_metrics': self.system_metrics.collect_metrics(),
            'application_metrics': self.app_metrics.get_metrics(),
            'health_status': self.health_service.overall_status.value,
            'active_alerts': len([a for a in self.alerts.values() if a.enabled]),
            'metrics_buffer_size': len(self.metrics_buffer),
            'opentelemetry_enabled': OPENTELEMETRY_AVAILABLE,
            'prometheus_enabled': PROMETHEUS_AVAILABLE,
            'timestamp': datetime.utcnow().isoformat()
        }


# Singleton instance
_observability_service: Optional[ObservabilityService] = None


def get_observability_service(config: Optional[ObservabilityConfig] = None) -> ObservabilityService:
    """Get singleton observability service."""
    global _observability_service
    if _observability_service is None:
        _observability_service = ObservabilityService(config)
    return _observability_service


def setup_instrumentation():
    """Set up automatic instrumentation for common libraries."""
    if not OPENTELEMETRY_AVAILABLE:
        return
    
    # Instrument common libraries
    RequestsInstrumentor().instrument()
    
    # FastAPI instrumentation will be done when the app is created
    # SQLAlchemy instrumentation will be done when the engine is created
    
    logging.getLogger(__name__).info("OpenTelemetry instrumentation set up")