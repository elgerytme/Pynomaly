"""Integration module for performance monitoring system."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from .performance_alerts import performance_monitor, AlertSeverity, MetricType
from .alert_handlers import create_alert_handlers

logger = logging.getLogger(__name__)


class PerformanceMonitoringIntegration:
    """Integration class for performance monitoring system."""
    
    def __init__(self, config_path: str = "config/monitoring/performance_alerts.json"):
        """Initialize performance monitoring integration."""
        self.config_path = config_path
        self.handlers_configured = False
        self.monitoring_started = False
    
    def configure_alert_handlers(self):
        """Configure alert handlers from configuration."""
        if self.handlers_configured:
            return
        
        try:
            import json
            config_file = Path(self.config_path)
            
            if not config_file.exists():
                logger.warning(f"Configuration file not found: {self.config_path}")
                return
            
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Create and add handlers
            alert_handlers = create_alert_handlers(config.get('alert_handlers', {}))
            
            # Clear existing handlers and add new ones
            performance_monitor.alert_handlers.clear()
            for handler in alert_handlers:
                performance_monitor.add_alert_handler(handler)
            
            self.handlers_configured = True
            logger.info(f"Configured {len(alert_handlers)} alert handlers")
            
        except Exception as e:
            logger.error(f"Failed to configure alert handlers: {e}")
    
    def start_monitoring(self):
        """Start performance monitoring system."""
        if self.monitoring_started:
            return
        
        try:
            # Configure handlers first
            self.configure_alert_handlers()
            
            # Start monitoring
            performance_monitor.start_monitoring()
            self.monitoring_started = True
            
            logger.info("Performance monitoring system started")
            
        except Exception as e:
            logger.error(f"Failed to start performance monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop performance monitoring system."""
        if not self.monitoring_started:
            return
        
        try:
            performance_monitor.stop_monitoring()
            self.monitoring_started = False
            
            logger.info("Performance monitoring system stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop performance monitoring: {e}")
    
    def get_system_status(self):
        """Get comprehensive system status."""
        return {
            "monitoring_active": self.monitoring_started,
            "handlers_configured": self.handlers_configured,
            "performance_monitor_status": {
                "active": performance_monitor.monitoring_active,
                "metrics_buffer_size": len(performance_monitor.metrics_buffer),
                "active_alerts": len(performance_monitor.active_alerts),
                "total_thresholds": len(performance_monitor.thresholds),
                "alert_handlers": len(performance_monitor.alert_handlers)
            },
            "statistics": performance_monitor.get_performance_stats()
        }
    
    def create_test_alert(self, severity: AlertSeverity = AlertSeverity.MEDIUM):
        """Create a test alert for testing purposes."""
        from .performance_alerts import PerformanceAlert
        
        test_alert = PerformanceAlert(
            alert_id=f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            severity=severity,
            metric_type=MetricType.RESPONSE_TIME,
            message=f"Test alert - {severity.value} severity",
            current_value=2500.0,
            threshold_value=1000.0,
            triggered_at=datetime.now(),
            tags={"test": "true"},
            metadata={"test_purpose": "integration_test"}
        )
        
        # Process the test alert
        performance_monitor.process_alert(test_alert)
        
        return test_alert
    
    def export_metrics_to_prometheus(self, output_file: str = "metrics.prom"):
        """Export metrics in Prometheus format."""
        try:
            metrics_content = []
            
            # Get current metrics
            stats = performance_monitor.get_performance_stats()
            
            # Total alerts metric
            metrics_content.append(f"# HELP pynomaly_alerts_total Total number of alerts generated")
            metrics_content.append(f"# TYPE pynomaly_alerts_total counter")
            metrics_content.append(f"pynomaly_alerts_total {stats['total_alerts']}")
            metrics_content.append("")
            
            # Alerts by severity
            metrics_content.append(f"# HELP pynomaly_alerts_by_severity_total Number of alerts by severity")
            metrics_content.append(f"# TYPE pynomaly_alerts_by_severity_total counter")
            for severity, count in stats['alerts_by_severity'].items():
                metrics_content.append(f"pynomaly_alerts_by_severity_total{{severity=\"{severity}\"}} {count}")
            metrics_content.append("")
            
            # Alerts by type
            metrics_content.append(f"# HELP pynomaly_alerts_by_type_total Number of alerts by metric type")
            metrics_content.append(f"# TYPE pynomaly_alerts_by_type_total counter")
            for metric_type, count in stats['alerts_by_type'].items():
                metrics_content.append(f"pynomaly_alerts_by_type_total{{metric_type=\"{metric_type}\"}} {count}")
            metrics_content.append("")
            
            # Active alerts
            metrics_content.append(f"# HELP pynomaly_active_alerts_total Number of currently active alerts")
            metrics_content.append(f"# TYPE pynomaly_active_alerts_total gauge")
            metrics_content.append(f"pynomaly_active_alerts_total {stats['active_alerts_count']}")
            metrics_content.append("")
            
            # Monitoring status
            metrics_content.append(f"# HELP pynomaly_monitoring_active Whether monitoring is active")
            metrics_content.append(f"# TYPE pynomaly_monitoring_active gauge")
            metrics_content.append(f"pynomaly_monitoring_active {1 if stats['monitoring_active'] else 0}")
            metrics_content.append("")
            
            # Metrics buffer size
            metrics_content.append(f"# HELP pynomaly_metrics_buffer_size Number of metrics in buffer")
            metrics_content.append(f"# TYPE pynomaly_metrics_buffer_size gauge")
            metrics_content.append(f"pynomaly_metrics_buffer_size {stats['metrics_buffer_size']}")
            metrics_content.append("")
            
            # Write to file
            with open(output_file, 'w') as f:
                f.write('\n'.join(metrics_content))
            
            logger.info(f"Metrics exported to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def get_health_check(self):
        """Get health check information."""
        try:
            stats = performance_monitor.get_performance_stats()
            
            # Determine health status
            health_status = "healthy"
            
            # Check if there are critical alerts
            if stats['alerts_by_severity'].get('critical', 0) > 0:
                health_status = "critical"
            elif stats['alerts_by_severity'].get('high', 0) > 0:
                health_status = "degraded"
            elif stats['alerts_by_severity'].get('medium', 0) > 0:
                health_status = "warning"
            
            return {
                "status": health_status,
                "timestamp": datetime.now().isoformat(),
                "monitoring_active": self.monitoring_started,
                "metrics": {
                    "total_alerts": stats['total_alerts'],
                    "active_alerts": stats['active_alerts_count'],
                    "buffer_size": stats['metrics_buffer_size'],
                    "thresholds": stats['thresholds_count']
                },
                "checks": {
                    "monitoring_system": "pass" if self.monitoring_started else "fail",
                    "alert_handlers": "pass" if self.handlers_configured else "fail",
                    "performance_monitor": "pass" if stats['monitoring_active'] else "fail"
                }
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }


# Global integration instance
monitoring_integration = PerformanceMonitoringIntegration()


def initialize_performance_monitoring():
    """Initialize performance monitoring system."""
    monitoring_integration.start_monitoring()


def shutdown_performance_monitoring():
    """Shutdown performance monitoring system."""
    monitoring_integration.stop_monitoring()


def get_monitoring_status():
    """Get current monitoring system status."""
    return monitoring_integration.get_system_status()


def create_test_alert(severity: AlertSeverity = AlertSeverity.MEDIUM):
    """Create a test alert for testing purposes."""
    return monitoring_integration.create_test_alert(severity)


def export_prometheus_metrics(output_file: str = "metrics.prom"):
    """Export metrics in Prometheus format."""
    monitoring_integration.export_metrics_to_prometheus(output_file)


def get_health_check():
    """Get health check information."""
    return monitoring_integration.get_health_check()