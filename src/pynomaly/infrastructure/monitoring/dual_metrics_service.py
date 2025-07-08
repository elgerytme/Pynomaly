"""Service for dual-writing metrics to both Prometheus and ExternalMonitoringService."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, Union

from pynomaly.infrastructure.monitoring.external_monitoring_service import (
    AlertSeverity,
    ExternalMonitoringService,
    MetricType,
    send_system_health_alert,
)
from pynomaly.infrastructure.monitoring.prometheus_metrics import (
    PrometheusMetricsService,
    get_metrics_service,
)

logger = logging.getLogger(__name__)


class DualMetricsService:
    """Service that forwards metrics to both Prometheus and ExternalMonitoringService."""
    
    def __init__(
        self,
        prometheus_service: Optional[PrometheusMetricsService] = None,
        external_service: Optional[ExternalMonitoringService] = None,
    ):
        """Initialize the dual metrics service.
        
        Args:
            prometheus_service: Prometheus metrics service instance
            external_service: External monitoring service instance
        """
        self.prometheus_service = prometheus_service or get_metrics_service()
        self.external_service = external_service
        
    async def record_http_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        response_size: Optional[int] = None,
    ) -> None:
        """Record HTTP request metrics to both services.
        
        Args:
            method: HTTP method
            endpoint: Request endpoint
            status_code: Response status code
            duration: Request duration in seconds
            response_size: Response size in bytes
        """
        # Record to Prometheus
        if self.prometheus_service:
            self.prometheus_service.record_http_request(method, endpoint, status_code, duration)
            if response_size:
                self.prometheus_service.record_api_response(endpoint, response_size)
        
        # Record to external monitoring
        if self.external_service:
            try:
                # Send request count metric
                await self.external_service.send_metric(
                    name="http_requests_total",
                    value=1,
                    metric_type=MetricType.COUNTER,
                    tags={
                        "method": method,
                        "endpoint": endpoint,
                        "status": str(status_code),
                    },
                )
                
                # Send duration metric
                await self.external_service.send_metric(
                    name="http_request_duration_seconds",
                    value=duration,
                    metric_type=MetricType.HISTOGRAM,
                    tags={
                        "method": method,
                        "endpoint": endpoint,
                    },
                )
                
                # Send response size metric if available
                if response_size:
                    await self.external_service.send_metric(
                        name="http_response_size_bytes",
                        value=response_size,
                        metric_type=MetricType.HISTOGRAM,
                        tags={"endpoint": endpoint},
                    )
            except Exception as e:
                logger.error(f"Failed to send HTTP metrics to external service: {e}")
    
    async def record_detection_metrics(
        self,
        algorithm: str,
        dataset_type: str,
        anomaly_count: int,
        total_samples: int,
        detection_time: float,
        accuracy_score: Optional[float] = None,
    ) -> None:
        """Record anomaly detection metrics to both services.
        
        Args:
            algorithm: Detection algorithm used
            dataset_type: Type of dataset processed
            anomaly_count: Number of anomalies detected
            total_samples: Total number of samples processed
            detection_time: Detection duration in seconds
            accuracy_score: Detection accuracy score (0-1)
        """
        # Record to Prometheus
        if self.prometheus_service:
            self.prometheus_service.record_detection_metrics(
                algorithm, dataset_type, anomaly_count, total_samples, detection_time, accuracy_score
            )
        
        # Record to external monitoring
        if self.external_service:
            try:
                tags = {"algorithm": algorithm, "dataset_type": dataset_type}
                
                # Detection count
                await self.external_service.send_metric(
                    name="anomaly_detections_total",
                    value=anomaly_count,
                    metric_type=MetricType.COUNTER,
                    tags=tags,
                )
                
                # Total samples
                await self.external_service.send_metric(
                    name="anomaly_samples_total",
                    value=total_samples,
                    metric_type=MetricType.GAUGE,
                    tags=tags,
                )
                
                # Detection time
                await self.external_service.send_metric(
                    name="anomaly_detection_time_seconds",
                    value=detection_time,
                    metric_type=MetricType.HISTOGRAM,
                    tags=tags,
                )
                
                # Accuracy score
                if accuracy_score is not None:
                    await self.external_service.send_metric(
                        name="anomaly_detection_accuracy",
                        value=accuracy_score,
                        metric_type=MetricType.GAUGE,
                        tags=tags,
                    )
                
                # Anomaly rate
                if total_samples > 0:
                    anomaly_rate = (anomaly_count / total_samples) * 100
                    await self.external_service.send_metric(
                        name="anomaly_rate_percent",
                        value=anomaly_rate,
                        metric_type=MetricType.GAUGE,
                        tags=tags,
                    )
                    
            except Exception as e:
                logger.error(f"Failed to send detection metrics to external service: {e}")
    
    async def record_training_metrics(
        self,
        algorithm: str,
        dataset_size: int,
        training_time: float,
        model_score: Optional[float] = None,
        trials_count: Optional[int] = None,
    ) -> None:
        """Record model training metrics to both services.
        
        Args:
            algorithm: Training algorithm used
            dataset_size: Size of training dataset
            training_time: Training duration in seconds
            model_score: Model performance score
            trials_count: Number of optimization trials
        """
        # Record to Prometheus
        if self.prometheus_service:
            self.prometheus_service.record_training_metrics(
                algorithm, dataset_size, training_time, model_score
            )
        
        # Record to external monitoring
        if self.external_service:
            try:
                tags = {"algorithm": algorithm}
                
                # Training requests
                await self.external_service.send_metric(
                    name="training_requests_total",
                    value=1,
                    metric_type=MetricType.COUNTER,
                    tags=tags,
                )
                
                # Training duration
                await self.external_service.send_metric(
                    name="training_duration_seconds",
                    value=training_time,
                    metric_type=MetricType.HISTOGRAM,
                    tags=tags,
                )
                
                # Model score
                if model_score is not None:
                    await self.external_service.send_metric(
                        name="model_score",
                        value=model_score,
                        metric_type=MetricType.GAUGE,
                        tags=tags,
                    )
                
                # Trials count
                if trials_count is not None:
                    await self.external_service.send_metric(
                        name="training_trials_count",
                        value=trials_count,
                        metric_type=MetricType.GAUGE,
                        tags=tags,
                    )
                    
            except Exception as e:
                logger.error(f"Failed to send training metrics to external service: {e}")
    
    async def record_error(
        self,
        error_type: str,
        component: str,
        severity: str = "error",
        exception: Optional[Exception] = None,
        send_alert: bool = True,
    ) -> None:
        """Record error metrics and optionally send alerts.
        
        Args:
            error_type: Type of error
            component: Component where error occurred
            severity: Error severity level
            exception: Exception object if available
            send_alert: Whether to send alert to external monitoring
        """
        # Record to Prometheus
        if self.prometheus_service:
            self.prometheus_service.record_error(error_type, component, severity)
        
        # Record to external monitoring and send alert
        if self.external_service:
            try:
                # Record error metric
                await self.external_service.send_metric(
                    name="errors_total",
                    value=1,
                    metric_type=MetricType.COUNTER,
                    tags={
                        "error_type": error_type,
                        "component": component,
                        "severity": severity,
                    },
                )
                
                # Send alert for high severity errors
                if send_alert and severity in ["error", "critical"]:
                    alert_severity = AlertSeverity.CRITICAL if severity == "critical" else AlertSeverity.HIGH
                    
                    error_message = f"Error in {component}: {error_type}"
                    if exception:
                        error_message += f" - {str(exception)}"
                    
                    await send_system_health_alert(
                        self.external_service,
                        component=component,
                        issue=error_message,
                        severity=alert_severity,
                    )
                    
            except Exception as e:
                logger.error(f"Failed to send error metrics to external service: {e}")
    
    async def record_system_metrics(
        self,
        active_models: int,
        active_streams: int,
        memory_usage: Dict[str, int],
        cpu_usage: Dict[str, float],
    ) -> None:
        """Record system metrics to both services.
        
        Args:
            active_models: Number of active models
            active_streams: Number of active streams
            memory_usage: Memory usage by component
            cpu_usage: CPU usage by component
        """
        # Record to Prometheus
        if self.prometheus_service:
            self.prometheus_service.update_system_metrics(
                active_models, active_streams, memory_usage, cpu_usage
            )
        
        # Record to external monitoring
        if self.external_service:
            try:
                # Active models
                await self.external_service.send_metric(
                    name="active_models",
                    value=active_models,
                    metric_type=MetricType.GAUGE,
                )
                
                # Active streams
                await self.external_service.send_metric(
                    name="active_streams",
                    value=active_streams,
                    metric_type=MetricType.GAUGE,
                )
                
                # Memory usage by component
                for component, usage in memory_usage.items():
                    await self.external_service.send_metric(
                        name="memory_usage_bytes",
                        value=usage,
                        metric_type=MetricType.GAUGE,
                        tags={"component": component},
                    )
                
                # CPU usage by component
                for component, usage in cpu_usage.items():
                    await self.external_service.send_metric(
                        name="cpu_usage_ratio",
                        value=usage,
                        metric_type=MetricType.GAUGE,
                        tags={"component": component},
                    )
                    
            except Exception as e:
                logger.error(f"Failed to send system metrics to external service: {e}")
    
    async def record_cache_metrics(
        self,
        cache_type: str,
        operation: str,
        hit: bool,
        cache_size: int,
        hit_ratio: Optional[float] = None,
    ) -> None:
        """Record cache metrics to both services.
        
        Args:
            cache_type: Type of cache
            operation: Cache operation (get, set, delete)
            hit: Whether operation was a hit
            cache_size: Current cache size
            hit_ratio: Cache hit ratio (0-1)
        """
        # Record to Prometheus
        if self.prometheus_service:
            self.prometheus_service.record_cache_metrics(cache_type, operation, hit, cache_size)
        
        # Record to external monitoring
        if self.external_service:
            try:
                status = "hit" if hit else "miss"
                tags = {"cache_type": cache_type}
                
                # Cache operations
                await self.external_service.send_metric(
                    name="cache_operations_total",
                    value=1,
                    metric_type=MetricType.COUNTER,
                    tags={**tags, "operation": operation, "status": status},
                )
                
                # Cache size
                await self.external_service.send_metric(
                    name="cache_size_items",
                    value=cache_size,
                    metric_type=MetricType.GAUGE,
                    tags=tags,
                )
                
                # Hit ratio
                if hit_ratio is not None:
                    await self.external_service.send_metric(
                        name="cache_hit_ratio",
                        value=hit_ratio,
                        metric_type=MetricType.GAUGE,
                        tags=tags,
                    )
                    
            except Exception as e:
                logger.error(f"Failed to send cache metrics to external service: {e}")


# Global dual metrics service instance
_dual_metrics_service: Optional[DualMetricsService] = None


def initialize_dual_metrics_service(
    prometheus_service: Optional[PrometheusMetricsService] = None,
    external_service: Optional[ExternalMonitoringService] = None,
) -> DualMetricsService:
    """Initialize the global dual metrics service.
    
    Args:
        prometheus_service: Prometheus metrics service instance
        external_service: External monitoring service instance
        
    Returns:
        DualMetricsService instance
    """
    global _dual_metrics_service
    _dual_metrics_service = DualMetricsService(prometheus_service, external_service)
    return _dual_metrics_service


def get_dual_metrics_service() -> Optional[DualMetricsService]:
    """Get the global dual metrics service instance.
    
    Returns:
        DualMetricsService instance or None if not initialized
    """
    return _dual_metrics_service


def set_external_monitoring_service(external_service: ExternalMonitoringService) -> None:
    """Set the external monitoring service in the global dual metrics service.
    
    Args:
        external_service: ExternalMonitoringService instance
    """
    global _dual_metrics_service
    if _dual_metrics_service:
        _dual_metrics_service.external_service = external_service
    else:
        _dual_metrics_service = DualMetricsService(external_service=external_service)


# Convenience functions for common use cases


async def record_http_request_dual(
    method: str,
    endpoint: str,
    status_code: int,
    duration: float,
    response_size: Optional[int] = None,
) -> None:
    """Record HTTP request metrics to both Prometheus and external monitoring.
    
    Args:
        method: HTTP method
        endpoint: Request endpoint
        status_code: Response status code
        duration: Request duration in seconds
        response_size: Response size in bytes
    """
    if _dual_metrics_service:
        await _dual_metrics_service.record_http_request(
            method, endpoint, status_code, duration, response_size
        )


async def record_detection_metrics_dual(
    algorithm: str,
    dataset_type: str,
    anomaly_count: int,
    total_samples: int,
    detection_time: float,
    accuracy_score: Optional[float] = None,
) -> None:
    """Record anomaly detection metrics to both services.
    
    Args:
        algorithm: Detection algorithm used
        dataset_type: Type of dataset processed
        anomaly_count: Number of anomalies detected
        total_samples: Total number of samples processed
        detection_time: Detection duration in seconds
        accuracy_score: Detection accuracy score (0-1)
    """
    if _dual_metrics_service:
        await _dual_metrics_service.record_detection_metrics(
            algorithm, dataset_type, anomaly_count, total_samples, detection_time, accuracy_score
        )


async def record_training_metrics_dual(
    algorithm: str,
    dataset_size: int,
    training_time: float,
    model_score: Optional[float] = None,
    trials_count: Optional[int] = None,
) -> None:
    """Record model training metrics to both services.
    
    Args:
        algorithm: Training algorithm used
        dataset_size: Size of training dataset
        training_time: Training duration in seconds
        model_score: Model performance score
        trials_count: Number of optimization trials
    """
    if _dual_metrics_service:
        await _dual_metrics_service.record_training_metrics(
            algorithm, dataset_size, training_time, model_score, trials_count
        )


async def record_error_dual(
    error_type: str,
    component: str,
    severity: str = "error",
    exception: Optional[Exception] = None,
    send_alert: bool = True,
) -> None:
    """Record error metrics and optionally send alerts to both services.
    
    Args:
        error_type: Type of error
        component: Component where error occurred
        severity: Error severity level
        exception: Exception object if available
        send_alert: Whether to send alert to external monitoring
    """
    if _dual_metrics_service:
        await _dual_metrics_service.record_error(
            error_type, component, severity, exception, send_alert
        )


async def record_system_metrics_dual(
    active_models: int,
    active_streams: int,
    memory_usage: Dict[str, int],
    cpu_usage: Dict[str, float],
) -> None:
    """Record system metrics to both services.
    
    Args:
        active_models: Number of active models
        active_streams: Number of active streams
        memory_usage: Memory usage by component
        cpu_usage: CPU usage by component
    """
    if _dual_metrics_service:
        await _dual_metrics_service.record_system_metrics(
            active_models, active_streams, memory_usage, cpu_usage
        )


async def record_cache_metrics_dual(
    cache_type: str,
    operation: str,
    hit: bool,
    cache_size: int,
    hit_ratio: Optional[float] = None,
) -> None:
    """Record cache metrics to both services.
    
    Args:
        cache_type: Type of cache
        operation: Cache operation (get, set, delete)
        hit: Whether operation was a hit
        cache_size: Current cache size
        hit_ratio: Cache hit ratio (0-1)
    """
    if _dual_metrics_service:
        await _dual_metrics_service.record_cache_metrics(
            cache_type, operation, hit, cache_size, hit_ratio
        )
