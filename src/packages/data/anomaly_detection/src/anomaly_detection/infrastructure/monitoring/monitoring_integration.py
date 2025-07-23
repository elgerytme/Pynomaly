"""Integration utilities for connecting monitoring system with detection services."""

from __future__ import annotations

import time
import functools
from typing import Any, Callable, Optional, Dict
from contextlib import contextmanager
import threading
import asyncio

from ..logging import get_logger
from .model_performance_monitor import get_model_performance_monitor, ModelPerformanceMetrics
from .alerting_system import get_alerting_system
from ...domain.entities.detection_result import DetectionResult

logger = get_logger(__name__)


class MonitoringIntegration:
    """Integration utilities for connecting monitoring with detection services."""
    
    def __init__(self):
        self.monitor = get_model_performance_monitor()
        self.alerting = get_alerting_system()
        self._is_enabled = True
        
        logger.info("Monitoring integration initialized")
    
    def enable_monitoring(self) -> None:
        """Enable monitoring integration."""
        self._is_enabled = True
        logger.info("Monitoring integration enabled")
    
    def disable_monitoring(self) -> None:
        """Disable monitoring integration."""
        self._is_enabled = False
        logger.info("Monitoring integration disabled")
    
    def monitor_detection(
        self,
        model_id: Optional[str] = None,
        algorithm: Optional[str] = None,
        record_performance: bool = True,
        record_memory: bool = False
    ):
        """Decorator for monitoring detection operations."""
        
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self._is_enabled:
                    return func(*args, **kwargs)
                
                # Extract monitoring parameters
                actual_model_id = model_id or kwargs.get('model_id', 'unknown')
                actual_algorithm = algorithm or kwargs.get('algorithm', 'unknown')
                
                start_time = time.time()
                memory_before = None
                
                if record_memory:
                    memory_before = self._get_memory_usage()
                
                try:
                    # Execute the function
                    result = func(*args, **kwargs)
                    
                    # Record metrics if result is DetectionResult
                    if isinstance(result, DetectionResult):
                        prediction_time_ms = (time.time() - start_time) * 1000
                        
                        # Record performance metrics
                        if record_performance:
                            memory_usage = None
                            if record_memory and memory_before is not None:
                                memory_after = self._get_memory_usage()
                                memory_usage = memory_after - memory_before
                            
                            self.monitor.record_prediction_metrics(
                                model_id=actual_model_id,
                                algorithm=actual_algorithm,
                                prediction_result=result,
                                prediction_time_ms=prediction_time_ms,
                                memory_usage_mb=memory_usage
                            )
                    
                    return result
                
                except Exception as e:
                    # Record error metrics
                    error_time_ms = (time.time() - start_time) * 1000
                    
                    metrics = ModelPerformanceMetrics(
                        model_id=actual_model_id,
                        algorithm=actual_algorithm,
                        timestamp=time.time(),
                        prediction_time_ms=error_time_ms,
                        custom_metrics={'error': 1.0, 'error_type': type(e).__name__}
                    )
                    
                    with self.monitor._lock:
                        self.monitor._metrics_buffer.append(metrics)
                    
                    raise
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self._is_enabled:
                    return await func(*args, **kwargs)
                
                # Extract monitoring parameters
                actual_model_id = model_id or kwargs.get('model_id', 'unknown')
                actual_algorithm = algorithm or kwargs.get('algorithm', 'unknown')
                
                start_time = time.time()
                memory_before = None
                
                if record_memory:
                    memory_before = self._get_memory_usage()
                
                try:
                    # Execute the async function
                    result = await func(*args, **kwargs)
                    
                    # Record metrics if result is DetectionResult
                    if isinstance(result, DetectionResult):
                        prediction_time_ms = (time.time() - start_time) * 1000
                        
                        # Record performance metrics
                        if record_performance:
                            memory_usage = None
                            if record_memory and memory_before is not None:
                                memory_after = self._get_memory_usage()
                                memory_usage = memory_after - memory_before
                            
                            self.monitor.record_prediction_metrics(
                                model_id=actual_model_id,
                                algorithm=actual_algorithm,
                                prediction_result=result,
                                prediction_time_ms=prediction_time_ms,
                                memory_usage_mb=memory_usage
                            )
                    
                    return result
                
                except Exception as e:
                    # Record error metrics
                    error_time_ms = (time.time() - start_time) * 1000
                    
                    metrics = ModelPerformanceMetrics(
                        model_id=actual_model_id,
                        algorithm=actual_algorithm,
                        timestamp=time.time(),
                        prediction_time_ms=error_time_ms,
                        custom_metrics={'error': 1.0, 'error_type': type(e).__name__}
                    )
                    
                    with self.monitor._lock:
                        self.monitor._metrics_buffer.append(metrics)
                    
                    raise
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def monitor_training(
        self,
        model_id: Optional[str] = None,
        algorithm: Optional[str] = None
    ):
        """Decorator for monitoring training operations."""
        
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self._is_enabled:
                    return func(*args, **kwargs)
                
                # Extract monitoring parameters
                actual_model_id = model_id or kwargs.get('model_id', 'unknown')
                actual_algorithm = algorithm or kwargs.get('algorithm', 'unknown')
                
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    
                    training_time_ms = (time.time() - start_time) * 1000
                    
                    # Extract dataset information if available
                    dataset_size = kwargs.get('dataset_size', 0)
                    feature_count = kwargs.get('feature_count', 0)
                    
                    # Try to infer from data parameter
                    data = kwargs.get('data')
                    if data is not None:
                        if hasattr(data, 'shape'):
                            dataset_size = data.shape[0]
                            feature_count = data.shape[1] if len(data.shape) > 1 else 1
                    
                    self.monitor.record_training_metrics(
                        model_id=actual_model_id,
                        algorithm=actual_algorithm,
                        training_time_ms=training_time_ms,
                        dataset_size=dataset_size,
                        feature_count=feature_count
                    )
                    
                    return result
                
                except Exception as e:
                    error_time_ms = (time.time() - start_time) * 1000
                    
                    self.monitor.record_training_metrics(
                        model_id=actual_model_id,
                        algorithm=actual_algorithm,
                        training_time_ms=error_time_ms,
                        dataset_size=0,
                        feature_count=0,
                        custom_metrics={'error': 1.0, 'error_type': type(e).__name__}
                    )
                    
                    raise
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self._is_enabled:
                    return await func(*args, **kwargs)
                
                # Extract monitoring parameters
                actual_model_id = model_id or kwargs.get('model_id', 'unknown')
                actual_algorithm = algorithm or kwargs.get('algorithm', 'unknown')
                
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    
                    training_time_ms = (time.time() - start_time) * 1000
                    
                    # Extract dataset information if available
                    dataset_size = kwargs.get('dataset_size', 0)
                    feature_count = kwargs.get('feature_count', 0)
                    
                    # Try to infer from data parameter
                    data = kwargs.get('data')
                    if data is not None:
                        if hasattr(data, 'shape'):
                            dataset_size = data.shape[0]
                            feature_count = data.shape[1] if len(data.shape) > 1 else 1
                    
                    self.monitor.record_training_metrics(
                        model_id=actual_model_id,
                        algorithm=actual_algorithm,
                        training_time_ms=training_time_ms,
                        dataset_size=dataset_size,
                        feature_count=feature_count
                    )
                    
                    return result
                
                except Exception as e:
                    error_time_ms = (time.time() - start_time) * 1000
                    
                    self.monitor.record_training_metrics(
                        model_id=actual_model_id,
                        algorithm=actual_algorithm,
                        training_time_ms=error_time_ms,
                        dataset_size=0,
                        feature_count=0,
                        custom_metrics={'error': 1.0, 'error_type': type(e).__name__}
                    )
                    
                    raise
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    @contextmanager
    def monitoring_context(
        self,
        model_id: str,
        algorithm: str,
        operation: str = "prediction"
    ):
        """Context manager for monitoring operations."""
        
        if not self._is_enabled:
            yield
            return
        
        start_time = time.time()
        
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            
            if operation == "prediction":
                # Create minimal metrics for context monitoring
                metrics = ModelPerformanceMetrics(
                    model_id=model_id,
                    algorithm=algorithm,
                    timestamp=time.time(),
                    prediction_time_ms=duration_ms
                )
                
                with self.monitor._lock:
                    self.monitor._metrics_buffer.append(metrics)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
        except Exception:
            return 0.0
    
    def record_custom_metric(
        self,
        model_id: str,
        metric_name: str,
        metric_value: float,
        algorithm: str = "custom"
    ) -> None:
        """Record a custom metric."""
        
        if not self._is_enabled:
            return
        
        metrics = ModelPerformanceMetrics(
            model_id=model_id,
            algorithm=algorithm,
            timestamp=time.time(),
            custom_metrics={metric_name: metric_value}
        )
        
        with self.monitor._lock:
            self.monitor._metrics_buffer.append(metrics)
        
        logger.debug("Custom metric recorded",
                    model_id=model_id,
                    metric_name=metric_name,
                    metric_value=metric_value)
    
    def setup_default_alert_callbacks(self) -> None:
        """Setup default alert callbacks for common actions."""
        
        def log_alert_callback(alert):
            """Log alert to application logs."""
            logger.warning("Performance alert triggered",
                          alert_id=alert.alert_id,
                          model_id=alert.model_id,
                          metric_name=alert.metric_name,
                          severity=alert.severity,
                          current_value=alert.current_value,
                          threshold_value=alert.threshold_value)
        
        def console_alert_callback(alert):
            """Print alert to console."""
            print(f"🚨 ALERT [{alert.severity.upper()}]: {alert.message}")
        
        # Add callbacks
        self.alerting.add_alert_callback(log_alert_callback)
        self.alerting.add_alert_callback(console_alert_callback)
        
        logger.info("Default alert callbacks configured")
    
    def configure_email_alerts(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        to_emails: List[str]
    ) -> None:
        """Configure email alerting."""
        
        from .alerting_system import EmailConfig
        
        email_config = EmailConfig(
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            username=username,
            password=password,
            from_email=from_email,
            to_emails=to_emails
        )
        
        self.alerting.add_email_channel("default_email", email_config)
        
        logger.info("Email alerting configured", to_emails=to_emails)
    
    def configure_slack_alerts(
        self,
        webhook_url: str,
        channel: Optional[str] = None
    ) -> None:
        """Configure Slack alerting."""
        
        from .alerting_system import SlackConfig
        
        slack_config = SlackConfig(
            webhook_url=webhook_url,
            channel=channel
        )
        
        self.alerting.add_slack_channel("default_slack", slack_config)
        
        logger.info("Slack alerting configured", channel=channel)
    
    def create_performance_summary_report(
        self,
        model_ids: Optional[List[str]] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Create a comprehensive performance summary report."""
        
        if model_ids is None:
            model_ids = list(self.monitor._performance_history.keys())
        
        report = {
            "report_timestamp": time.time(),
            "period_hours": hours,
            "models_analyzed": len(model_ids),
            "summary": {},
            "alerts": {
                "active_count": len(self.alerting.get_active_alerts()),
                "recent_stats": self.alerting.get_alert_statistics(hours)
            },
            "recommendations": []
        }
        
        # Generate summary for each model
        total_predictions = 0
        total_anomalies = 0
        avg_precision_scores = []
        avg_prediction_times = []
        
        for model_id in model_ids:
            model_summary = self.monitor.get_model_performance_summary(model_id, hours)
            report["summary"][model_id] = model_summary
            
            # Aggregate metrics
            total_predictions += model_summary.get("total_predictions", 0)
            total_anomalies += model_summary.get("total_anomalies_detected", 0)
            
            if model_summary.get("avg_precision") is not None:
                avg_precision_scores.append(model_summary["avg_precision"])
            
            if model_summary.get("avg_prediction_time_ms") is not None:
                avg_prediction_times.append(model_summary["avg_prediction_time_ms"])
        
        # Calculate overall metrics
        report["overall_metrics"] = {
            "total_predictions": total_predictions,
            "total_anomalies_detected": total_anomalies,
            "overall_anomaly_rate": total_anomalies / total_predictions if total_predictions > 0 else 0,
            "avg_precision": sum(avg_precision_scores) / len(avg_precision_scores) if avg_precision_scores else None,
            "avg_prediction_time_ms": sum(avg_prediction_times) / len(avg_prediction_times) if avg_prediction_times else None
        }
        
        # Generate recommendations
        report["recommendations"] = self._generate_performance_recommendations(report)
        
        return report
    
    def _generate_performance_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations."""
        
        recommendations = []
        overall_metrics = report["overall_metrics"]
        
        # Check precision
        avg_precision = overall_metrics.get("avg_precision")
        if avg_precision is not None and avg_precision < 0.7:
            recommendations.append(
                f"Average precision ({avg_precision:.3f}) is below recommended threshold (0.7). "
                "Consider tuning model parameters or using ensemble methods."
            )
        
        # Check prediction time
        avg_pred_time = overall_metrics.get("avg_prediction_time_ms")
        if avg_pred_time is not None and avg_pred_time > 2000:
            recommendations.append(
                f"Average prediction time ({avg_pred_time:.1f}ms) is high. "
                "Consider optimizing algorithms or using model compression techniques."
            )
        
        # Check alert frequency
        alert_count = report["alerts"]["active_count"]
        if alert_count > 5:
            recommendations.append(
                f"High number of active alerts ({alert_count}). "
                "Review alert thresholds and address performance issues."
            )
        
        # Check anomaly rate
        anomaly_rate = overall_metrics["overall_anomaly_rate"]
        if anomaly_rate > 0.2:
            recommendations.append(
                f"High anomaly rate ({anomaly_rate:.3f}) may indicate data quality issues "
                "or overly sensitive detection parameters."
            )
        elif anomaly_rate < 0.01:
            recommendations.append(
                f"Very low anomaly rate ({anomaly_rate:.3f}) may indicate "
                "under-sensitive detection parameters or lack of anomalies in data."
            )
        
        return recommendations
    
    async def initialize_monitoring_system(self) -> None:
        """Initialize the complete monitoring system."""
        
        # Start monitoring components
        await self.monitor.start_monitoring()
        
        # Setup default callbacks
        self.setup_default_alert_callbacks()
        
        logger.info("Monitoring system fully initialized")
    
    async def shutdown_monitoring_system(self) -> None:
        """Shutdown the monitoring system."""
        
        await self.monitor.stop_monitoring()
        
        logger.info("Monitoring system shutdown complete")


# Global instance
_monitoring_integration: Optional[MonitoringIntegration] = None


def get_monitoring_integration() -> MonitoringIntegration:
    """Get the global monitoring integration instance."""
    
    global _monitoring_integration
    
    if _monitoring_integration is None:
        _monitoring_integration = MonitoringIntegration()
    
    return _monitoring_integration


def initialize_monitoring_integration() -> MonitoringIntegration:
    """Initialize the global monitoring integration."""
    
    global _monitoring_integration
    _monitoring_integration = MonitoringIntegration()
    return _monitoring_integration


# Convenience decorators using global instance

def monitor_detection(
    model_id: Optional[str] = None,
    algorithm: Optional[str] = None,
    record_performance: bool = True,
    record_memory: bool = False
):
    """Convenience decorator for monitoring detection operations."""
    return get_monitoring_integration().monitor_detection(
        model_id=model_id,
        algorithm=algorithm,
        record_performance=record_performance,
        record_memory=record_memory
    )


def monitor_training(
    model_id: Optional[str] = None,
    algorithm: Optional[str] = None
):
    """Convenience decorator for monitoring training operations."""
    return get_monitoring_integration().monitor_training(
        model_id=model_id,
        algorithm=algorithm
    )


def monitoring_context(
    model_id: str,
    algorithm: str,
    operation: str = "prediction"
):
    """Convenience context manager for monitoring operations."""
    return get_monitoring_integration().monitoring_context(
        model_id=model_id,
        algorithm=algorithm,
        operation=operation
    )