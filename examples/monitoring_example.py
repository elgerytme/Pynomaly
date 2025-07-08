"""
Example usage of the enhanced monitoring system with Prometheus metrics and alerting.
"""

import asyncio
import logging

from src.pynomaly.domain.services.advanced_detection_service import DetectionAlgorithm
from src.pynomaly.domain.services.processing_orchestrator import ProcessingOrchestrator

# Import the enhanced monitoring components
from src.pynomaly.infrastructure.batch.batch_processor import (
    BatchConfig,
    BatchEngine,
    BatchProcessor,
)
from src.pynomaly.infrastructure.monitoring.alerting_system import (
    AlertRule,
    AlertSeverity,
    AlertType,
    NotificationConfig,
    NotificationType,
    get_alerting_system,
    setup_default_alerts,
)
from src.pynomaly.infrastructure.monitoring.prometheus_metrics_enhanced import (
    get_metrics_collector,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_monitoring_system():
    """Setup the complete monitoring system with metrics and alerts."""
    logger.info("Setting up monitoring system...")

    # Get metrics collector
    metrics_collector = get_metrics_collector()

    # Setup alerting system
    alerting_system = get_alerting_system()

    # Setup default alerts
    setup_default_alerts()

    # Add custom alert rules
    custom_rules = [
        AlertRule(
            name="batch_job_timeout",
            description="Batch job taking too long to complete",
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=AlertSeverity.HIGH,
            condition="avg_job_duration > 3600",  # 1 hour
            threshold=3600,
            duration_seconds=300,
            cooldown_seconds=1800,
        ),
        AlertRule(
            name="high_memory_usage_critical",
            description="Critical memory usage detected",
            alert_type=AlertType.MEMORY_USAGE,
            severity=AlertSeverity.CRITICAL,
            condition="memory_usage > 0.95",
            threshold=0.95,
            duration_seconds=60,
            cooldown_seconds=300,
        ),
        AlertRule(
            name="anomaly_detection_spike",
            description="Unusually high number of anomalies detected",
            alert_type=AlertType.ANOMALY_SPIKE,
            severity=AlertSeverity.MEDIUM,
            condition="anomaly_rate > 0.2",
            threshold=0.2,
            duration_seconds=600,
            cooldown_seconds=1800,
        ),
    ]

    for rule in custom_rules:
        alerting_system.add_rule(rule)

    # Setup notification configurations
    notification_configs = [
        # Log notifications for all alerts
        NotificationConfig(
            type=NotificationType.LOG,
            enabled=True,
            config={"level": "WARNING"},
            min_severity=AlertSeverity.LOW,
        ),
        # Email notifications for high severity alerts
        NotificationConfig(
            type=NotificationType.EMAIL,
            enabled=True,
            config={
                "from": "alerts@example.com",
                "to": "admin@example.com",
                "smtp_server": "smtp.example.com",
                "smtp_port": 587,
                "use_tls": True,
                "username": "alerts@example.com",
                "password": "your_password",
            },
            min_severity=AlertSeverity.HIGH,
        ),
        # Webhook notifications for critical alerts
        NotificationConfig(
            type=NotificationType.WEBHOOK,
            enabled=True,
            config={
                "url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
                "headers": {"Content-Type": "application/json"},
                "timeout": 30,
            },
            min_severity=AlertSeverity.CRITICAL,
        ),
        # Slack notifications for job failures
        NotificationConfig(
            type=NotificationType.SLACK,
            enabled=True,
            config={
                "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
                "timeout": 30,
            },
            min_severity=AlertSeverity.MEDIUM,
            alert_types=[AlertType.JOB_FAILURE, AlertType.SLA_BREACH],
        ),
    ]

    for config in notification_configs:
        alerting_system.add_notification_config(config)

    # Add log patterns for monitoring
    alerting_system.add_log_pattern(
        "memory_leak_pattern",
        r"memory.*leak|OutOfMemoryError|heap.*exceeded",
        AlertSeverity.HIGH,
        threshold=3,
        window_seconds=300,
    )

    alerting_system.add_log_pattern(
        "database_error_pattern",
        r"database.*error|connection.*failed|timeout.*database",
        AlertSeverity.MEDIUM,
        threshold=5,
        window_seconds=300,
    )

    # Add SLA configurations
    alerting_system.add_sla_config(
        "batch_processing_sla",
        "avg_job_duration",
        threshold=1800,  # 30 minutes
        duration_seconds=300,
    )

    alerting_system.add_sla_config(
        "streaming_latency_sla",
        "avg_processing_latency",
        threshold=5.0,  # 5 seconds
        duration_seconds=300,
    )

    # Start alert monitoring
    await alerting_system.start_monitoring()

    logger.info("Monitoring system setup completed")
    return metrics_collector, alerting_system


async def demonstrate_batch_processing_with_metrics():
    """Demonstrate batch processing with metrics collection."""
    logger.info("Starting batch processing demonstration...")

    # Create batch configuration
    batch_config = BatchConfig(
        engine=BatchEngine.MULTIPROCESSING,
        max_workers=4,
        chunk_size=1000,
        detection_algorithm=DetectionAlgorithm.ISOLATION_FOREST,
        max_retries=3,
        retry_delay_seconds=5.0,
    )

    # Create batch processor
    batch_processor = BatchProcessor(batch_config)

    # Submit a batch job
    try:
        job_id = await batch_processor.submit_job(
            name="demo_batch_job",
            description="Demonstration of batch processing with metrics",
            input_path="examples/data/sample_data.csv",
            output_path="outputs/batch_results.json",
            config=batch_config,
        )

        logger.info(f"Submitted batch job: {job_id}")

        # Monitor job progress
        while True:
            job_status = await batch_processor.get_job_status(job_id)
            if job_status is None:
                break

            logger.info(
                f"Job {job_id} status: {job_status['status']} "
                f"({job_status['progress_percentage']:.1f}%)"
            )

            if job_status["status"] in ["completed", "failed", "cancelled"]:
                break

            await asyncio.sleep(10)

        logger.info("Batch processing demonstration completed")

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")

    finally:
        await batch_processor.shutdown()


async def demonstrate_orchestrator_with_metrics():
    """Demonstrate orchestrator with metrics collection."""
    logger.info("Starting orchestrator demonstration...")

    # Create orchestrator
    orchestrator = ProcessingOrchestrator()

    # Start monitoring
    await orchestrator.start_monitoring()

    try:
        # Start a streaming session
        stream_session_id = await orchestrator.start_streaming_session(
            name="demo_stream",
            source_config={
                "bootstrap_servers": "localhost:9092",
                "topic": "anomaly_data",
                "group_id": "demo_group",
            },
            detection_algorithm=DetectionAlgorithm.ISOLATION_FOREST,
        )

        logger.info(f"Started streaming session: {stream_session_id}")

        # Start a batch session
        batch_session_id = await orchestrator.start_batch_session(
            name="demo_batch",
            input_path="examples/data/sample_data.csv",
            output_path="outputs/orchestrator_batch_results.json",
            engine=BatchEngine.MULTIPROCESSING,
            detection_algorithm=DetectionAlgorithm.ISOLATION_FOREST,
        )

        logger.info(f"Started batch session: {batch_session_id}")

        # Monitor sessions
        await asyncio.sleep(30)

        # Get session statuses
        stream_status = await orchestrator.get_session_status(stream_session_id)
        batch_status = await orchestrator.get_session_status(batch_session_id)

        logger.info(f"Stream session status: {stream_status['status']}")
        logger.info(f"Batch session status: {batch_status['status']}")

        # Get system metrics
        system_metrics = await orchestrator.get_system_metrics()
        logger.info(f"System metrics: {system_metrics}")

        # Stop sessions
        await orchestrator.stop_session(stream_session_id)
        await orchestrator.stop_session(batch_session_id)

        logger.info("Orchestrator demonstration completed")

    except Exception as e:
        logger.error(f"Orchestrator demonstration failed: {e}")

    finally:
        await orchestrator.shutdown()


async def demonstrate_alerting_system():
    """Demonstrate the alerting system."""
    logger.info("Starting alerting system demonstration...")

    alerting_system = get_alerting_system()

    # Simulate some log entries that would trigger alerts
    log_entries = [
        "ERROR: Database connection failed",
        "CRITICAL: OutOfMemoryError in batch processor",
        "WARNING: High memory usage detected",
        "ERROR: Job execution failed",
        "FATAL: System resources exhausted",
    ]

    for entry in log_entries:
        alerting_system.process_log_entry(entry)
        await asyncio.sleep(1)  # Small delay between entries

    # Wait for alerts to be processed
    await asyncio.sleep(5)

    # Get alert statistics
    stats = alerting_system.get_alert_statistics()
    logger.info(f"Alert statistics: {stats}")

    # Get active alerts
    active_alerts = alerting_system.get_active_alerts()
    logger.info(f"Active alerts: {len(active_alerts)}")

    for alert in active_alerts:
        logger.info(
            f"Alert: {alert.rule.name} - {alert.rule.severity.value} - {alert.message}"
        )

    # Get alert history
    alert_history = alerting_system.get_alert_history(hours=1)
    logger.info(f"Alert history (last hour): {len(alert_history)} alerts")

    logger.info("Alerting system demonstration completed")


async def demonstrate_metrics_collection():
    """Demonstrate metrics collection and export."""
    logger.info("Starting metrics collection demonstration...")

    metrics_collector = get_metrics_collector()

    # Record some sample metrics
    metrics_collector.record_job_duration(
        120.5, "batch", "isolation_forest", "multiprocessing", "completed"
    )
    metrics_collector.increment_anomalies_found(
        15, "batch", "isolation_forest", "medium", "sample_data.csv"
    )
    metrics_collector.increment_retry_count("batch", "timeout", 2)
    metrics_collector.set_memory_usage(1024.5, "batch", "processor", "instance_1")
    metrics_collector.set_peak_memory_usage(2048.0, "batch", "processor")
    metrics_collector.increment_jobs_total("batch", "completed", "isolation_forest")
    metrics_collector.set_active_jobs(3, "batch", "running")

    # Record streaming metrics
    metrics_collector.record_processing_latency(
        2.5, "streaming", "isolation_forest", "medium"
    )
    metrics_collector.set_session_count(5, "streaming", "running")
    metrics_collector.increment_alert_count(
        "high_error_rate", "medium", "alerting_system"
    )
    metrics_collector.increment_notification_count(
        "email", "admin@example.com", "success"
    )

    # Export metrics (this would typically be scraped by Prometheus)
    if metrics_collector.is_available():
        metrics_output = metrics_collector.get_metrics()
        logger.info(f"Metrics exported: {len(metrics_output)} bytes")

        # Save to file for demonstration
        with open("outputs/metrics_export.txt", "w") as f:
            f.write(metrics_output)

        logger.info("Metrics saved to outputs/metrics_export.txt")

    logger.info("Metrics collection demonstration completed")


async def main():
    """Main demonstration function."""
    logger.info("Starting comprehensive monitoring system demonstration...")

    # Setup monitoring system
    metrics_collector, alerting_system = await setup_monitoring_system()

    try:
        # Demonstrate different components
        await demonstrate_metrics_collection()
        await demonstrate_alerting_system()
        await demonstrate_batch_processing_with_metrics()
        await demonstrate_orchestrator_with_metrics()

        logger.info("All demonstrations completed successfully!")

        # Final metrics and alert summary
        logger.info("\n=== FINAL SUMMARY ===")

        # Alert statistics
        alert_stats = alerting_system.get_alert_statistics()
        logger.info(f"Alert Statistics: {alert_stats}")

        # Active alerts
        active_alerts = alerting_system.get_active_alerts()
        logger.info(f"Active Alerts: {len(active_alerts)}")

        # Export final metrics
        if metrics_collector.is_available():
            final_metrics = metrics_collector.get_metrics()
            with open("outputs/final_metrics_export.txt", "w") as f:
                f.write(final_metrics)
            logger.info("Final metrics exported to outputs/final_metrics_export.txt")

    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise

    finally:
        # Cleanup
        await alerting_system.stop_monitoring()
        logger.info("Monitoring system demonstration completed")


if __name__ == "__main__":
    # Create outputs directory if it doesn't exist
    import os

    os.makedirs("outputs", exist_ok=True)

    # Run the demonstration
    asyncio.run(main())
