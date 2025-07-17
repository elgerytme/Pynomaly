"""Streaming pipeline manager for orchestrating real-time anomaly detection."""

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from pynomaly_detection.infrastructure.streaming.real_time_anomaly_pipeline import (
    AlertSeverity,
    KafkaDataSource,
    RealTimeAnomalyPipeline,
    StreamingAlert,
    StreamingAnomalyDetector,
    StreamingConfig,
    WebSocketDataSource,
)

logger = logging.getLogger(__name__)


class PipelineTemplate:
    """Template for creating streaming pipelines."""

    def __init__(
        self,
        name: str,
        description: str,
        data_source_type: str,
        data_source_config: dict[str, Any],
        detector_config: dict[str, Any],
        streaming_config: dict[str, Any],
    ):
        """Initialize pipeline template.

        Args:
            name: Template name
            description: Template description
            data_source_type: Type of data source (kafka, websocket)
            data_source_config: Data source configuration
            detector_config: Detector configuration
            streaming_config: Streaming configuration
        """
        self.name = name
        self.description = description
        self.data_source_type = data_source_type
        self.data_source_config = data_source_config
        self.detector_config = detector_config
        self.streaming_config = streaming_config


class StreamingPipelineManager:
    """Manager for orchestrating multiple streaming pipelines."""

    def __init__(self):
        """Initialize streaming pipeline manager."""
        self.pipelines: dict[str, RealTimeAnomalyPipeline] = {}
        self.templates: dict[str, PipelineTemplate] = {}
        self.alerts: list[StreamingAlert] = []
        self.max_alerts = 10000  # Maximum alerts to keep in memory

        # Register default templates
        self._register_default_templates()

    def _register_default_templates(self) -> None:
        """Register default pipeline templates."""
        # Kafka fraud detection template
        fraud_template = PipelineTemplate(
            name="fraud_detection_kafka",
            description="Real-time fraud detection from Kafka stream",
            data_source_type="kafka",
            data_source_config={
                "bootstrap_servers": "localhost:9092",
                "topic": "transactions",
                "group_id": "fraud_detection",
                "auto_offset_reset": "latest",
            },
            detector_config={
                "detector_algorithm": "isolation_forest",
                "contamination": 0.01,  # 1% fraud rate
                "window_size": 5000,
                "retraining_interval": 10000,
            },
            streaming_config={
                "batch_size": 50,
                "window_size": 1000,
                "window_slide": 100,
                "max_buffer_size": 5000,
                "processing_timeout": 10.0,
                "alert_thresholds": {
                    "anomaly_rate": 0.02,  # Alert if >2% anomalies
                    "processing_latency": 500.0,  # Alert if >500ms latency
                    "error_rate": 0.01,  # Alert if >1% errors
                },
            },
        )
        self.templates["fraud_detection_kafka"] = fraud_template

        # WebSocket IoT sensor template
        iot_template = PipelineTemplate(
            name="iot_sensors_websocket",
            description="Real-time IoT sensor anomaly detection",
            data_source_type="websocket",
            data_source_config={
                "websocket_url": "wss://iot-gateway.example.com/sensors",
                "headers": {"Authorization": "Bearer <token>"},
            },
            detector_config={
                "detector_algorithm": "one_class_svm",
                "contamination": 0.05,  # 5% anomaly rate for sensors
                "window_size": 2000,
                "retraining_interval": 5000,
            },
            streaming_config={
                "batch_size": 20,
                "window_size": 500,
                "window_slide": 50,
                "max_buffer_size": 2000,
                "processing_timeout": 5.0,
                "alert_thresholds": {
                    "anomaly_rate": 0.1,
                    "processing_latency": 200.0,
                    "error_rate": 0.02,
                },
            },
        )
        self.templates["iot_sensors_websocket"] = iot_template

        # High-frequency trading template
        trading_template = PipelineTemplate(
            name="hft_anomaly_detection",
            description="High-frequency trading anomaly detection",
            data_source_type="kafka",
            data_source_config={
                "bootstrap_servers": "localhost:9092",
                "topic": "market_data",
                "group_id": "trading_anomaly",
                "auto_offset_reset": "latest",
            },
            detector_config={
                "detector_algorithm": "local_outlier_factor",
                "contamination": 0.001,  # Very low anomaly rate
                "window_size": 10000,
                "retraining_interval": 50000,
            },
            streaming_config={
                "batch_size": 100,
                "window_size": 2000,
                "window_slide": 200,
                "max_buffer_size": 10000,
                "processing_timeout": 1.0,  # Very low latency
                "alert_thresholds": {
                    "anomaly_rate": 0.005,
                    "processing_latency": 100.0,  # <100ms required
                    "error_rate": 0.001,
                },
            },
        )
        self.templates["hft_anomaly_detection"] = trading_template

    async def create_pipeline_from_template(
        self,
        template_name: str,
        pipeline_id: str | None = None,
        override_config: dict[str, Any] | None = None,
    ) -> str:
        """Create a pipeline from a template.

        Args:
            template_name: Name of the template to use
            pipeline_id: Optional custom pipeline ID
            override_config: Optional configuration overrides

        Returns:
            Created pipeline ID

        Raises:
            ValueError: If template doesn't exist
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")

        template = self.templates[template_name]
        pipeline_id = pipeline_id or str(uuid4())

        # Apply overrides
        data_source_config = template.data_source_config.copy()
        detector_config = template.detector_config.copy()
        streaming_config = template.streaming_config.copy()

        if override_config:
            data_source_config.update(override_config.get("data_source", {}))
            detector_config.update(override_config.get("detector", {}))
            streaming_config.update(override_config.get("streaming", {}))

        # Create pipeline
        return await self.create_pipeline(
            pipeline_id=pipeline_id,
            data_source_type=template.data_source_type,
            data_source_config=data_source_config,
            detector_config=detector_config,
            streaming_config=streaming_config,
        )

    async def create_pipeline(
        self,
        pipeline_id: str,
        data_source_type: str,
        data_source_config: dict[str, Any],
        detector_config: dict[str, Any],
        streaming_config: dict[str, Any],
    ) -> str:
        """Create a new streaming pipeline.

        Args:
            pipeline_id: Unique pipeline identifier
            data_source_type: Type of data source
            data_source_config: Data source configuration
            detector_config: Detector configuration
            streaming_config: Streaming configuration

        Returns:
            Created pipeline ID
        """
        if pipeline_id in self.pipelines:
            raise ValueError(f"Pipeline '{pipeline_id}' already exists")

        # Create data source
        if data_source_type == "kafka":
            data_source = KafkaDataSource(**data_source_config)
        elif data_source_type == "websocket":
            data_source = WebSocketDataSource(**data_source_config)
        else:
            raise ValueError(f"Unsupported data source type: {data_source_type}")

        # Create detector
        detector = StreamingAnomalyDetector(**detector_config)

        # Create streaming config
        streaming_config["pipeline_id"] = pipeline_id
        config = StreamingConfig(**streaming_config)

        # Create pipeline
        pipeline = RealTimeAnomalyPipeline(
            config=config,
            data_source=data_source,
            detector=detector,
            alert_handler=self._handle_alert,
        )

        self.pipelines[pipeline_id] = pipeline

        logger.info(f"Created streaming pipeline: {pipeline_id}")
        return pipeline_id

    async def start_pipeline(self, pipeline_id: str) -> None:
        """Start a streaming pipeline.

        Args:
            pipeline_id: Pipeline ID to start

        Raises:
            ValueError: If pipeline doesn't exist
        """
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline '{pipeline_id}' not found")

        pipeline = self.pipelines[pipeline_id]
        await pipeline.start()

        logger.info(f"Started streaming pipeline: {pipeline_id}")

    async def stop_pipeline(self, pipeline_id: str) -> None:
        """Stop a streaming pipeline.

        Args:
            pipeline_id: Pipeline ID to stop

        Raises:
            ValueError: If pipeline doesn't exist
        """
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline '{pipeline_id}' not found")

        pipeline = self.pipelines[pipeline_id]
        await pipeline.stop()

        logger.info(f"Stopped streaming pipeline: {pipeline_id}")

    async def delete_pipeline(self, pipeline_id: str) -> None:
        """Delete a streaming pipeline.

        Args:
            pipeline_id: Pipeline ID to delete

        Raises:
            ValueError: If pipeline doesn't exist
        """
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline '{pipeline_id}' not found")

        # Stop pipeline if running
        pipeline = self.pipelines[pipeline_id]
        if pipeline.is_running:
            await pipeline.stop()

        # Remove from registry
        del self.pipelines[pipeline_id]

        logger.info(f"Deleted streaming pipeline: {pipeline_id}")

    async def start_all_pipelines(self) -> None:
        """Start all registered pipelines."""
        for pipeline_id in list(self.pipelines.keys()):
            try:
                await self.start_pipeline(pipeline_id)
            except Exception as e:
                logger.error(f"Failed to start pipeline {pipeline_id}: {e}")

    async def stop_all_pipelines(self) -> None:
        """Stop all running pipelines."""
        for pipeline_id in list(self.pipelines.keys()):
            try:
                await self.stop_pipeline(pipeline_id)
            except Exception as e:
                logger.error(f"Failed to stop pipeline {pipeline_id}: {e}")

    def get_pipeline_status(self, pipeline_id: str) -> dict[str, Any]:
        """Get status of a specific pipeline.

        Args:
            pipeline_id: Pipeline ID

        Returns:
            Pipeline status

        Raises:
            ValueError: If pipeline doesn't exist
        """
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline '{pipeline_id}' not found")

        pipeline = self.pipelines[pipeline_id]
        return pipeline.get_status()

    def get_all_pipeline_status(self) -> dict[str, Any]:
        """Get status of all pipelines.

        Returns:
            Status of all pipelines
        """
        status = {}
        for pipeline_id, pipeline in self.pipelines.items():
            try:
                status[pipeline_id] = pipeline.get_status()
            except Exception as e:
                status[pipeline_id] = {"error": str(e)}

        return status

    def get_pipeline_metrics(self, pipeline_id: str) -> dict[str, Any]:
        """Get metrics for a specific pipeline.

        Args:
            pipeline_id: Pipeline ID

        Returns:
            Pipeline metrics

        Raises:
            ValueError: If pipeline doesn't exist
        """
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline '{pipeline_id}' not found")

        pipeline = self.pipelines[pipeline_id]
        return pipeline.get_metrics().dict()

    def get_aggregated_metrics(self) -> dict[str, Any]:
        """Get aggregated metrics across all pipelines.

        Returns:
            Aggregated metrics
        """
        total_processed = 0
        total_anomalies = 0
        total_errors = 0
        running_pipelines = 0
        average_latency = 0.0

        for pipeline in self.pipelines.values():
            if pipeline.is_running:
                running_pipelines += 1
                metrics = pipeline.get_metrics()
                total_processed += metrics.processed_records
                total_anomalies += metrics.anomalies_detected
                total_errors += metrics.error_count
                average_latency += metrics.average_latency

        if running_pipelines > 0:
            average_latency /= running_pipelines

        return {
            "total_pipelines": len(self.pipelines),
            "running_pipelines": running_pipelines,
            "total_processed_records": total_processed,
            "total_anomalies_detected": total_anomalies,
            "total_errors": total_errors,
            "overall_anomaly_rate": (
                total_anomalies / total_processed if total_processed > 0 else 0.0
            ),
            "overall_error_rate": (
                total_errors / total_processed if total_processed > 0 else 0.0
            ),
            "average_processing_latency": average_latency,
            "total_alerts": len(self.alerts),
        }

    def get_recent_alerts(
        self,
        limit: int = 100,
        severity: AlertSeverity | None = None,
        pipeline_id: str | None = None,
    ) -> list[StreamingAlert]:
        """Get recent alerts with optional filtering.

        Args:
            limit: Maximum number of alerts to return
            severity: Optional severity filter
            pipeline_id: Optional pipeline ID filter

        Returns:
            List of recent alerts
        """
        # Filter alerts
        filtered_alerts = self.alerts

        if severity:
            filtered_alerts = [
                alert for alert in filtered_alerts if alert.severity == severity
            ]

        if pipeline_id:
            filtered_alerts = [
                alert
                for alert in filtered_alerts
                if alert.metadata.get("pipeline_id") == pipeline_id
            ]

        # Sort by timestamp (most recent first) and limit
        filtered_alerts.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_alerts[:limit]

    def get_alert_statistics(self) -> dict[str, Any]:
        """Get alert statistics.

        Returns:
            Alert statistics
        """
        if not self.alerts:
            return {
                "total_alerts": 0,
                "alerts_by_severity": {},
                "alerts_by_type": {},
                "alerts_by_pipeline": {},
                "recent_alert_rate": 0.0,
            }

        # Count by severity
        severity_counts = {}
        for alert in self.alerts:
            severity_counts[alert.severity.value] = (
                severity_counts.get(alert.severity.value, 0) + 1
            )

        # Count by type
        type_counts = {}
        for alert in self.alerts:
            type_counts[alert.alert_type] = type_counts.get(alert.alert_type, 0) + 1

        # Count by pipeline
        pipeline_counts = {}
        for alert in self.alerts:
            pipeline_id = alert.metadata.get("pipeline_id", "unknown")
            pipeline_counts[pipeline_id] = pipeline_counts.get(pipeline_id, 0) + 1

        # Calculate recent alert rate (last hour)
        recent_threshold = datetime.utcnow().timestamp() - 3600  # 1 hour ago
        recent_alerts = [
            alert
            for alert in self.alerts
            if alert.timestamp.timestamp() > recent_threshold
        ]

        return {
            "total_alerts": len(self.alerts),
            "alerts_by_severity": severity_counts,
            "alerts_by_type": type_counts,
            "alerts_by_pipeline": pipeline_counts,
            "recent_alert_rate": len(recent_alerts) / 60.0,  # alerts per minute
        }

    def register_template(self, template: PipelineTemplate) -> None:
        """Register a new pipeline template.

        Args:
            template: Pipeline template to register
        """
        self.templates[template.name] = template
        logger.info(f"Registered pipeline template: {template.name}")

    def get_templates(self) -> dict[str, PipelineTemplate]:
        """Get all registered templates.

        Returns:
            Dictionary of templates
        """
        return self.templates.copy()

    def _handle_alert(self, alert: StreamingAlert) -> None:
        """Handle alerts from pipelines.

        Args:
            alert: Alert to handle
        """
        # Add to alerts list
        self.alerts.append(alert)

        # Maintain max alerts limit
        if len(self.alerts) > self.max_alerts:
            # Remove oldest alerts
            self.alerts = self.alerts[-self.max_alerts :]

        # Log alert
        logger.info(
            f"Alert: {alert.alert_type} - {alert.severity.value} - {alert.message}"
        )

        # In a real implementation, you might:
        # - Send to external alerting system (PagerDuty, Slack, etc.)
        # - Store in database
        # - Trigger automated responses
        # - Update monitoring dashboards

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on all pipelines.

        Returns:
            Health check results
        """
        health_status = {
            "overall_status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "pipeline_health": {},
            "issues": [],
        }

        unhealthy_count = 0

        for pipeline_id, pipeline in self.pipelines.items():
            try:
                status = pipeline.get_status()
                metrics = pipeline.get_metrics()

                # Check for issues
                issues = []

                if not pipeline.is_running:
                    issues.append("Pipeline not running")

                if metrics.error_count > 0:
                    error_rate = metrics.error_count / max(metrics.processed_records, 1)
                    if error_rate > 0.05:  # >5% error rate
                        issues.append(f"High error rate: {error_rate:.2%}")

                if metrics.average_latency > 1000:  # >1 second latency
                    issues.append(f"High latency: {metrics.average_latency:.1f}ms")

                pipeline_status = "healthy" if not issues else "unhealthy"
                if issues:
                    unhealthy_count += 1

                health_status["pipeline_health"][pipeline_id] = {
                    "status": pipeline_status,
                    "is_running": pipeline.is_running,
                    "processed_records": metrics.processed_records,
                    "error_count": metrics.error_count,
                    "average_latency": metrics.average_latency,
                    "issues": issues,
                }

            except Exception as e:
                health_status["pipeline_health"][pipeline_id] = {
                    "status": "error",
                    "error": str(e),
                }
                unhealthy_count += 1

        # Determine overall status
        if unhealthy_count == 0:
            health_status["overall_status"] = "healthy"
        elif unhealthy_count < len(self.pipelines):
            health_status["overall_status"] = "degraded"
        else:
            health_status["overall_status"] = "unhealthy"

        return health_status
