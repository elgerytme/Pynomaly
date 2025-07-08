"""Enhanced Prometheus metrics for batch processing and orchestration."""

import logging
import time
from typing import Any, Dict, Optional
from contextlib import contextmanager

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Info,
        CollectorRegistry,
        generate_latest,
        push_to_gateway,
        CONTENT_TYPE_LATEST,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class PrometheusMetricsCollector:
    """Enhanced Prometheus metrics collector for batch and orchestration monitoring."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None, namespace: str = "pynomaly"):
        """Initialize metrics collector."""
        self.registry = registry
        self.namespace = namespace
        
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, metrics will be disabled")
            return
            
        # Job duration metrics
        self.job_duration = Histogram(
            f"{namespace}_job_duration_seconds",
            "Duration of anomaly detection jobs",
            labelnames=["job_type", "algorithm", "engine", "status"],
            registry=registry,
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0]
        )
        
        # Anomalies found metrics
        self.anomalies_found = Counter(
            f"{namespace}_anomalies_found_total",
            "Total number of anomalies found",
            labelnames=["job_type", "algorithm", "severity", "data_source"],
            registry=registry
        )
        
        # Retry metrics
        self.retry_count = Counter(
            f"{namespace}_job_retries_total",
            "Total number of job retries",
            labelnames=["job_type", "failure_reason", "retry_attempt"],
            registry=registry
        )
        
        # Memory usage metrics
        self.memory_usage = Gauge(
            f"{namespace}_memory_usage_mb",
            "Current memory usage in MB",
            labelnames=["job_type", "component", "instance"],
            registry=registry
        )
        
        # Peak memory usage
        self.peak_memory_usage = Gauge(
            f"{namespace}_peak_memory_usage_mb",
            "Peak memory usage in MB",
            labelnames=["job_type", "component"],
            registry=registry
        )
        
        # Job processing metrics
        self.jobs_total = Counter(
            f"{namespace}_jobs_total",
            "Total number of jobs processed",
            labelnames=["job_type", "status", "algorithm"],
            registry=registry
        )
        
        # Active jobs gauge
        self.active_jobs = Gauge(
            f"{namespace}_active_jobs",
            "Number of currently active jobs",
            labelnames=["job_type", "status"],
            registry=registry
        )
        
        # Data throughput metrics
        self.data_throughput = Histogram(
            f"{namespace}_data_throughput_records_per_second",
            "Data processing throughput",
            labelnames=["job_type", "algorithm", "engine"],
            registry=registry
        )
        
        # Chunk processing metrics
        self.chunk_processing_time = Histogram(
            f"{namespace}_chunk_processing_seconds",
            "Time to process individual chunks",
            labelnames=["job_type", "algorithm", "chunk_size_category"],
            registry=registry
        )
        
        # Error rate metrics
        self.error_rate = Counter(
            f"{namespace}_errors_total",
            "Total number of errors",
            labelnames=["job_type", "error_type", "component"],
            registry=registry
        )
        
        # SLA metrics
        self.sla_violations = Counter(
            f"{namespace}_sla_violations_total",
            "Total number of SLA violations",
            labelnames=["job_type", "sla_type", "severity"],
            registry=registry
        )
        
        # Resource utilization
        self.cpu_usage = Gauge(
            f"{namespace}_cpu_usage_percent",
            "CPU usage percentage",
            labelnames=["job_type", "component", "instance"],
            registry=registry
        )
        
        # Queue metrics
        self.queue_size = Gauge(
            f"{namespace}_queue_size",
            "Number of items in processing queue",
            labelnames=["queue_type", "priority"],
            registry=registry
        )
        
        # Session metrics for orchestrator
        self.session_count = Gauge(
            f"{namespace}_sessions_total",
            "Total number of processing sessions",
            labelnames=["session_type", "status"],
            registry=registry
        )
        
        # Processing latency
        self.processing_latency = Histogram(
            f"{namespace}_processing_latency_seconds",
            "End-to-end processing latency",
            labelnames=["job_type", "algorithm", "data_size_category"],
            registry=registry
        )
        
        # Data quality metrics
        self.data_quality_score = Gauge(
            f"{namespace}_data_quality_score",
            "Data quality score (0-1)",
            labelnames=["job_type", "data_source", "quality_metric"],
            registry=registry
        )
        
        # Model performance metrics
        self.model_accuracy = Gauge(
            f"{namespace}_model_accuracy",
            "Model accuracy score",
            labelnames=["algorithm", "dataset_type"],
            registry=registry
        )
        
        # Alert metrics
        self.alert_count = Counter(
            f"{namespace}_alerts_total",
            "Total number of alerts generated",
            labelnames=["alert_type", "severity", "source"],
            registry=registry
        )
        
        # Webhook/notification metrics
        self.notification_count = Counter(
            f"{namespace}_notifications_total",
            "Total number of notifications sent",
            labelnames=["notification_type", "destination", "status"],
            registry=registry
        )
        
        # System info
        self.system_info = Info(
            f"{namespace}_system_info",
            "System information",
            registry=registry
        )
        
        logger.info(f"Prometheus metrics collector initialized with namespace: {namespace}")
    
    def is_available(self) -> bool:
        """Check if Prometheus metrics are available."""
        return PROMETHEUS_AVAILABLE
    
    def record_job_duration(self, duration: float, job_type: str, algorithm: str, 
                          engine: str, status: str) -> None:
        """Record job duration."""
        if not self.is_available():
            return
        self.job_duration.labels(
            job_type=job_type,
            algorithm=algorithm,
            engine=engine,
            status=status
        ).observe(duration)
    
    def increment_anomalies_found(self, count: int, job_type: str, algorithm: str, 
                                severity: str, data_source: str) -> None:
        """Increment anomalies found counter."""
        if not self.is_available():
            return
        self.anomalies_found.labels(
            job_type=job_type,
            algorithm=algorithm,
            severity=severity,
            data_source=data_source
        ).inc(count)
    
    def increment_retry_count(self, job_type: str, failure_reason: str, retry_attempt: int) -> None:
        """Increment retry counter."""
        if not self.is_available():
            return
        self.retry_count.labels(
            job_type=job_type,
            failure_reason=failure_reason,
            retry_attempt=str(retry_attempt)
        ).inc()
    
    def set_memory_usage(self, memory_mb: float, job_type: str, component: str, 
                        instance: str) -> None:
        """Set current memory usage."""
        if not self.is_available():
            return
        self.memory_usage.labels(
            job_type=job_type,
            component=component,
            instance=instance
        ).set(memory_mb)
    
    def set_peak_memory_usage(self, memory_mb: float, job_type: str, component: str) -> None:
        """Set peak memory usage."""
        if not self.is_available():
            return
        self.peak_memory_usage.labels(
            job_type=job_type,
            component=component
        ).set(memory_mb)
    
    def increment_jobs_total(self, job_type: str, status: str, algorithm: str) -> None:
        """Increment total jobs counter."""
        if not self.is_available():
            return
        self.jobs_total.labels(
            job_type=job_type,
            status=status,
            algorithm=algorithm
        ).inc()
    
    def set_active_jobs(self, count: int, job_type: str, status: str) -> None:
        """Set active jobs gauge."""
        if not self.is_available():
            return
        self.active_jobs.labels(
            job_type=job_type,
            status=status
        ).set(count)
    
    def record_data_throughput(self, throughput: float, job_type: str, algorithm: str, 
                             engine: str) -> None:
        """Record data throughput."""
        if not self.is_available():
            return
        self.data_throughput.labels(
            job_type=job_type,
            algorithm=algorithm,
            engine=engine
        ).observe(throughput)
    
    def record_chunk_processing_time(self, duration: float, job_type: str, algorithm: str, 
                                   chunk_size_category: str) -> None:
        """Record chunk processing time."""
        if not self.is_available():
            return
        self.chunk_processing_time.labels(
            job_type=job_type,
            algorithm=algorithm,
            chunk_size_category=chunk_size_category
        ).observe(duration)
    
    def increment_error_count(self, job_type: str, error_type: str, component: str) -> None:
        """Increment error counter."""
        if not self.is_available():
            return
        self.error_rate.labels(
            job_type=job_type,
            error_type=error_type,
            component=component
        ).inc()
    
    def increment_sla_violations(self, job_type: str, sla_type: str, severity: str) -> None:
        """Increment SLA violations counter."""
        if not self.is_available():
            return
        self.sla_violations.labels(
            job_type=job_type,
            sla_type=sla_type,
            severity=severity
        ).inc()
    
    def set_cpu_usage(self, usage_percent: float, job_type: str, component: str, 
                     instance: str) -> None:
        """Set CPU usage."""
        if not self.is_available():
            return
        self.cpu_usage.labels(
            job_type=job_type,
            component=component,
            instance=instance
        ).set(usage_percent)
    
    def set_queue_size(self, size: int, queue_type: str, priority: str) -> None:
        """Set queue size."""
        if not self.is_available():
            return
        self.queue_size.labels(
            queue_type=queue_type,
            priority=priority
        ).set(size)
    
    def set_session_count(self, count: int, session_type: str, status: str) -> None:
        """Set session count."""
        if not self.is_available():
            return
        self.session_count.labels(
            session_type=session_type,
            status=status
        ).set(count)
    
    def record_processing_latency(self, latency: float, job_type: str, algorithm: str, 
                                data_size_category: str) -> None:
        """Record processing latency."""
        if not self.is_available():
            return
        self.processing_latency.labels(
            job_type=job_type,
            algorithm=algorithm,
            data_size_category=data_size_category
        ).observe(latency)
    
    def set_data_quality_score(self, score: float, job_type: str, data_source: str, 
                              quality_metric: str) -> None:
        """Set data quality score."""
        if not self.is_available():
            return
        self.data_quality_score.labels(
            job_type=job_type,
            data_source=data_source,
            quality_metric=quality_metric
        ).set(score)
    
    def set_model_accuracy(self, accuracy: float, algorithm: str, dataset_type: str) -> None:
        """Set model accuracy."""
        if not self.is_available():
            return
        self.model_accuracy.labels(
            algorithm=algorithm,
            dataset_type=dataset_type
        ).set(accuracy)
    
    def increment_alert_count(self, alert_type: str, severity: str, source: str) -> None:
        """Increment alert counter."""
        if not self.is_available():
            return
        self.alert_count.labels(
            alert_type=alert_type,
            severity=severity,
            source=source
        ).inc()
    
    def increment_notification_count(self, notification_type: str, destination: str, 
                                   status: str) -> None:
        """Increment notification counter."""
        if not self.is_available():
            return
        self.notification_count.labels(
            notification_type=notification_type,
            destination=destination,
            status=status
        ).inc()
    
    def set_system_info(self, info: Dict[str, str]) -> None:
        """Set system information."""
        if not self.is_available():
            return
        self.system_info.info(info)
    
    @contextmanager
    def time_operation(self, job_type: str, algorithm: str, engine: str, status: str = "success"):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        except Exception as e:
            status = "failed"
            raise
        finally:
            duration = time.time() - start_time
            self.record_job_duration(duration, job_type, algorithm, engine, status)
    
    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format."""
        if not self.is_available():
            return ""
        
        return generate_latest(self.registry)
    
    def push_to_gateway(self, gateway_url: str, job_name: str, 
                       grouping_key: Optional[Dict[str, str]] = None) -> None:
        """Push metrics to Prometheus Pushgateway."""
        if not self.is_available():
            return
        
        try:
            push_to_gateway(
                gateway_url, 
                job=job_name, 
                registry=self.registry,
                grouping_key=grouping_key or {}
            )
        except Exception as e:
            logger.error(f"Failed to push metrics to gateway: {e}")


# Global metrics collector instance
_metrics_collector: Optional[PrometheusMetricsCollector] = None


def get_metrics_collector(namespace: str = "pynomaly") -> PrometheusMetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = PrometheusMetricsCollector(namespace=namespace)
    return _metrics_collector


def categorize_chunk_size(size: int) -> str:
    """Categorize chunk size for metrics."""
    if size < 1000:
        return "small"
    elif size < 10000:
        return "medium"
    elif size < 100000:
        return "large"
    else:
        return "xlarge"


def categorize_data_size(size_mb: float) -> str:
    """Categorize data size for metrics."""
    if size_mb < 10:
        return "small"
    elif size_mb < 100:
        return "medium"
    elif size_mb < 1000:
        return "large"
    else:
        return "xlarge"


def categorize_severity(anomaly_count: int, total_samples: int) -> str:
    """Categorize anomaly severity based on contamination rate."""
    if total_samples == 0:
        return "unknown"
    
    contamination_rate = anomaly_count / total_samples
    
    if contamination_rate < 0.01:
        return "low"
    elif contamination_rate < 0.05:
        return "medium"
    elif contamination_rate < 0.1:
        return "high"
    else:
        return "critical"
