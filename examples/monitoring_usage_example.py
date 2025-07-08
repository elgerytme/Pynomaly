"""Example usage of the monitoring system."""

import asyncio
import os
from datetime import datetime

from pynomaly.infrastructure.config.settings import get_settings
from pynomaly.infrastructure.monitoring.service_initialization import (
    initialize_monitoring_service,
    shutdown_monitoring_service,
)
from pynomaly.infrastructure.monitoring.external_monitoring_service import (
    MetricType,
    AlertSeverity,
)


async def demo_monitoring_system():
    """Demonstrate the monitoring system functionality."""
    print("=== Pynomaly Monitoring System Demo ===\\n")
    
    # Configure providers via environment variables
    print("1. Configuring monitoring providers...")
    
    # Enable Grafana (example)
    os.environ["PYNOMALY_GRAFANA_ENABLED"] = "true"
    os.environ["PYNOMALY_GRAFANA_ENDPOINT"] = "http://localhost:3000"
    os.environ["PYNOMALY_GRAFANA_API_KEY"] = "your_grafana_api_key"
    
    # Enable custom webhook (example)
    os.environ["PYNOMALY_WEBHOOK_ENABLED"] = "true"
    os.environ["PYNOMALY_WEBHOOK_ENDPOINT"] = "http://localhost:8080/webhook"
    os.environ["PYNOMALY_WEBHOOK_API_KEY"] = "your_webhook_api_key"
    
    # Configure buffer settings
    os.environ["PYNOMALY_MONITORING_BUFFER_SIZE"] = "50"
    os.environ["PYNOMALY_MONITORING_FLUSH_INTERVAL"] = "30"
    
    print("✓ Environment variables configured")
    
    # Initialize monitoring service
    print("\\n2. Initializing monitoring service...")
    settings = get_settings()
    monitoring_service = await initialize_monitoring_service(settings)
    
    print(f"✓ Service initialized with {len(monitoring_service.providers)} providers")
    print(f"✓ Buffer size: {monitoring_service.buffer_size}")
    print(f"✓ Flush interval: {monitoring_service.flush_interval} seconds")
    
    # Test provider connections
    print("\\n3. Testing provider connections...")
    connection_results = await monitoring_service.test_all_providers()
    
    for provider_name, connected in connection_results.items():
        status = "✓" if connected else "✗"
        print(f"{status} {provider_name}: {'Connected' if connected else 'Failed'}")
    
    # Send metrics
    print("\\n4. Sending metrics...")
    
    # Counter metric
    await monitoring_service.send_metric(
        name="anomaly_detections_total",
        value=42,
        metric_type=MetricType.COUNTER,
        tags={"detector": "isolation_forest", "dataset": "sample_data"},
        buffered=True
    )
    print("✓ Sent counter metric: anomaly_detections_total")
    
    # Gauge metric
    await monitoring_service.send_metric(
        name="anomaly_rate_percent",
        value=3.5,
        metric_type=MetricType.GAUGE,
        tags={"detector": "isolation_forest", "dataset": "sample_data"},
        buffered=True
    )
    print("✓ Sent gauge metric: anomaly_rate_percent")
    
    # Timer metric
    await monitoring_service.send_metric(
        name="detection_time_seconds",
        value=1.234,
        metric_type=MetricType.TIMER,
        tags={"detector": "isolation_forest", "dataset": "sample_data"},
        buffered=False  # Send immediately
    )
    print("✓ Sent timer metric: detection_time_seconds")
    
    # Send alerts
    print("\\n5. Sending alerts...")
    
    # High severity alert
    await monitoring_service.send_alert(
        title="High Anomaly Rate Detected",
        message="Anomaly rate exceeded 5% threshold in production dataset",
        severity=AlertSeverity.HIGH,
        source="anomaly_detector",
        tags={"environment": "production", "dataset": "user_activity"},
        buffered=False
    )
    print("✓ Sent high severity alert")
    
    # Low severity alert
    await monitoring_service.send_alert(
        title="Model Retrain Recommended",
        message="Model performance has degraded below 90% accuracy",
        severity=AlertSeverity.LOW,
        source="model_monitor",
        tags={"environment": "production", "model": "isolation_forest_v2"},
        buffered=True
    )
    print("✓ Sent low severity alert")
    
    # Show buffer status
    print("\\n6. Buffer status:")
    status = monitoring_service.get_provider_status()
    for provider_name, provider_status in status.items():
        print(f"  - {provider_name}:")
        print(f"    Provider: {provider_status['provider']}")
        print(f"    Enabled: {provider_status['enabled']}")
        print(f"    Endpoint: {provider_status['endpoint']}")
        print(f"    Buffered metrics: {provider_status['buffered_metrics']}")
        print(f"    Buffered alerts: {provider_status['buffered_alerts']}")
    
    # Flush buffers
    print("\\n7. Flushing buffers...")
    await monitoring_service.flush_buffers()
    print("✓ Buffers flushed")
    
    # Shutdown service
    print("\\n8. Shutting down monitoring service...")
    await shutdown_monitoring_service(monitoring_service)
    print("✓ Service shutdown complete")
    
    print("\\n=== Demo Complete ===")


async def demo_convenience_functions():
    """Demonstrate convenience functions for common monitoring patterns."""
    print("\\n=== Convenience Functions Demo ===\\n")
    
    # Initialize service
    settings = get_settings()
    monitoring_service = await initialize_monitoring_service(settings)
    
    # Import convenience functions
    from pynomaly.infrastructure.monitoring.external_monitoring_service import (
        send_anomaly_detection_metrics,
        send_training_job_metrics,
        send_system_health_alert,
    )
    
    # Send anomaly detection metrics
    print("1. Sending anomaly detection metrics...")
    await send_anomaly_detection_metrics(
        service=monitoring_service,
        detector_name="isolation_forest",
        dataset_name="production_data",
        anomaly_count=15,
        total_samples=1000,
        detection_time=2.5,
        accuracy_score=0.95
    )
    print("✓ Anomaly detection metrics sent")
    
    # Send training job metrics
    print("\\n2. Sending training job metrics...")
    await send_training_job_metrics(
        service=monitoring_service,
        job_id="training_job_123456",
        algorithm="isolation_forest",
        trial_count=50,
        best_score=0.92,
        execution_time=300.0
    )
    print("✓ Training job metrics sent")
    
    # Send system health alert
    print("\\n3. Sending system health alert...")
    await send_system_health_alert(
        service=monitoring_service,
        component="database",
        issue="High connection pool utilization (>90%)",
        severity=AlertSeverity.MEDIUM
    )
    print("✓ System health alert sent")
    
    # Shutdown service
    await shutdown_monitoring_service(monitoring_service)
    print("\\n✓ Convenience functions demo complete")


if __name__ == "__main__":
    print("Starting monitoring system demo...")
    asyncio.run(demo_monitoring_system())
    asyncio.run(demo_convenience_functions())
