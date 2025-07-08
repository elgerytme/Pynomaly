"""Real-time streaming service for anomaly detection."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator
from uuid import UUID

from pynomaly.domain.entities.anomaly_event import (
    AnomalyEvent,
    AnomalyEventData,
    EventSeverity,
    EventType,
)
from pynomaly.domain.entities.streaming_session import (
    SessionStatus,
    SessionSummary,
    StreamingAlert,
    StreamingConfiguration,
    StreamingDataSink,
    StreamingDataSource,
    StreamingMetrics,
    StreamingSession,
)
from pynomaly.shared.protocols.repository_protocol import (
    ModelRepositoryProtocol,
)
from pynomaly.infrastructure.monitoring.prometheus_metrics import get_metrics_service

logger = logging.getLogger(__name__)


class StreamingService:
    """Service for managing real-time anomaly detection streaming."""

    def __init__(
        self,
        model_repository: ModelRepositoryProtocol,
        streaming_repository: Any,  # StreamingRepositoryProtocol when implemented
        event_repository: Any,  # EventRepositoryProtocol when implemented
        detector_service: Any,  # DetectorService when implemented
        notification_service: Any,  # NotificationService when implemented
    ):
        """Initialize the streaming service.

        Args:
            model_repository: Model repository
            streaming_repository: Streaming session repository
            event_repository: Event repository
            detector_service: Detector service for predictions
            notification_service: Notification service
        """
        self.model_repository = model_repository
        self.streaming_repository = streaming_repository
        self.event_repository = event_repository
        self.detector_service = detector_service
        self.notification_service = notification_service
        self._active_sessions: dict[UUID, StreamingSession] = {}
        self._session_tasks: dict[UUID, asyncio.Task] = {}

    async def create_streaming_session(
        self,
        name: str,
        detector_id: UUID,
        data_source: StreamingDataSource,
        configuration: StreamingConfiguration,
        created_by: str,
        description: str | None = None,
        data_sink: StreamingDataSink | None = None,
        model_version: str | None = None,
        max_duration: timedelta | None = None,
        tags: list[str] | None = None,
    ) -> StreamingSession:
        """Create a new streaming session.

        Args:
            name: Session name
            detector_id: Detector to use for anomaly detection
            data_source: Input data source configuration
            configuration: Streaming configuration
            created_by: User creating the session
            description: Session description
            data_sink: Output data sink configuration
            model_version: Specific model version to use
            max_duration: Maximum session duration
            tags: Session tags

        Returns:
            Created streaming session
        """
        # Validate detector exists
        detector = await self.model_repository.get_by_id(detector_id)
        if not detector:
            raise ValueError(f"Detector {detector_id} does not exist")

        # Create session
        session = StreamingSession(
            name=name,
            description=description,
            detector_id=detector_id,
            model_version=model_version,
            data_source=data_source,
            data_sink=data_sink,
            configuration=configuration,
            max_duration=max_duration,
            created_by=created_by,
            tags=tags or [],
        )

        # Store session
        stored_session = await self.streaming_repository.create_session(session)

        return stored_session

    async def start_streaming_session(self, session_id: UUID) -> StreamingSession:
        """Start a streaming session.

        Args:
            session_id: Session to start

        Returns:
            Updated streaming session
        """
        session = await self.streaming_repository.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Start the session
        session.start_session()

        # Update in repository
        updated_session = await self.streaming_repository.update_session(session)

        # Start the streaming task
        task = asyncio.create_task(self._run_streaming_session(session))
        self._session_tasks[session_id] = task
        self._active_sessions[session_id] = updated_session

        return updated_session

    async def stop_streaming_session(
        self, session_id: UUID, error_message: str | None = None
    ) -> StreamingSession:
        """Stop a streaming session.

        Args:
            session_id: Session to stop
            error_message: Optional error message

        Returns:
            Updated streaming session
        """
        session = await self.streaming_repository.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Stop the session
        session.stop_session(error_message)

        # Cancel the streaming task
        if session_id in self._session_tasks:
            task = self._session_tasks.pop(session_id)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Remove from active sessions
        self._active_sessions.pop(session_id, None)

        # Complete the stop
        session.complete_stop()

        # Update in repository
        updated_session = await self.streaming_repository.update_session(session)

        return updated_session

    async def pause_streaming_session(self, session_id: UUID) -> StreamingSession:
        """Pause a streaming session.

        Args:
            session_id: Session to pause

        Returns:
            Updated streaming session
        """
        session = await self.streaming_repository.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        session.pause_session()

        # Update active session
        if session_id in self._active_sessions:
            self._active_sessions[session_id] = session

        updated_session = await self.streaming_repository.update_session(session)

        return updated_session

    async def resume_streaming_session(self, session_id: UUID) -> StreamingSession:
        """Resume a paused streaming session.

        Args:
            session_id: Session to resume

        Returns:
            Updated streaming session
        """
        session = await self.streaming_repository.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        session.resume_session()

        # Update active session
        if session_id in self._active_sessions:
            self._active_sessions[session_id] = session

        updated_session = await self.streaming_repository.update_session(session)

        return updated_session

    async def get_session_metrics(self, session_id: UUID) -> StreamingMetrics:
        """Get current metrics for a streaming session.

        Args:
            session_id: Session identifier

        Returns:
            Current streaming metrics
        """
        session = await self.streaming_repository.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        return session.current_metrics

    async def get_session_summary(self, session_id: UUID) -> SessionSummary:
        """Get summary for a streaming session.

        Args:
            session_id: Session identifier

        Returns:
            Session summary
        """
        session = await self.streaming_repository.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        uptime = session.get_uptime()
        uptime_seconds = uptime.total_seconds() if uptime else 0.0

        throughput_summary = session.get_throughput_summary()

        return SessionSummary(
            session_id=session.id,
            name=session.name,
            status=session.status,
            detector_id=session.detector_id,
            created_at=session.created_at,
            started_at=session.started_at,
            stopped_at=session.stopped_at,
            uptime_seconds=uptime_seconds,
            messages_processed=session.current_metrics.messages_processed,
            anomalies_detected=session.current_metrics.anomalies_detected,
            current_throughput=session.current_metrics.messages_per_second,
            avg_throughput=throughput_summary["avg_throughput"],
            error_rate=session.current_metrics.error_rate,
            anomaly_rate=session.current_metrics.anomaly_rate,
            created_by=session.created_by,
        )

    async def list_streaming_sessions(
        self,
        status: SessionStatus | None = None,
        detector_id: UUID | None = None,
        created_by: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SessionSummary]:
        """List streaming sessions with filters.

        Args:
            status: Filter by status
            detector_id: Filter by detector
            created_by: Filter by creator
            limit: Maximum results
            offset: Result offset

        Returns:
            List of session summaries
        """
        sessions = await self.streaming_repository.list_sessions(
            status=status,
            detector_id=detector_id,
            created_by=created_by,
            limit=limit,
            offset=offset,
        )

        summaries = []
        for session in sessions:
            summary = await self.get_session_summary(session.id)
            summaries.append(summary)

        return summaries

    async def create_streaming_alert(
        self,
        session_id: UUID,
        name: str,
        metric_name: str,
        threshold_value: float,
        comparison_operator: str,
        created_by: str,
        description: str | None = None,
        severity: str = "medium",
        duration_threshold: timedelta = timedelta(minutes=1),
        notification_channels: list[str] | None = None,
    ) -> StreamingAlert:
        """Create a streaming alert.

        Args:
            session_id: Session to monitor
            name: Alert name
            metric_name: Metric to monitor
            threshold_value: Alert threshold
            comparison_operator: Comparison operator
            created_by: Alert creator
            description: Alert description
            severity: Alert severity
            duration_threshold: Duration threshold
            notification_channels: Notification channels

        Returns:
            Created streaming alert
        """
        # Validate session exists
        session = await self.streaming_repository.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        alert = StreamingAlert(
            session_id=session_id,
            name=name,
            description=description,
            metric_name=metric_name,
            threshold_value=threshold_value,
            comparison_operator=comparison_operator,
            duration_threshold=duration_threshold,
            severity=severity,
            notification_channels=notification_channels or [],
            created_by=created_by,
        )

        stored_alert = await self.streaming_repository.create_alert(alert)

        return stored_alert

    async def process_streaming_data(
        self, session_id: UUID, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Process a single data point through streaming pipeline.

        Args:
            session_id: Session identifier
            data: Input data to process

        Returns:
            Processing result including anomaly detection
        """
        session = self._active_sessions.get(session_id)
        if not session or not session.is_active():
            raise ValueError(f"Session {session_id} is not active")

        start_time = time.time()
        try:
            # Preprocess data
            processed_data = await self._preprocess_data(data, session.configuration)

            # Run anomaly detection
            detection_result = await self.detector_service.predict(
                session.detector_id, processed_data
            )

            # Create result
            result = {
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "input_data": data,
                "processed_data": processed_data,
                "anomaly_score": detection_result.get("anomaly_score", 0.0),
                "is_anomaly": detection_result.get("is_anomaly", False),
                "confidence": detection_result.get("confidence", 0.0),
                "feature_contributions": detection_result.get(
                    "feature_contributions", {}
                ),
                "explanation": detection_result.get("explanation"),
            }

            # Record streaming metrics
            metrics_service = get_metrics_service()
            if metrics_service:
                processing_time = time.time() - start_time
                metrics_service.record_streaming_metrics(
                    stream_id=str(session_id),
                    samples_processed=1,
                    throughput=1.0 / processing_time if processing_time > 0 else 0.0,
                    buffer_utilization=0.5,  # Could be calculated from actual buffer
                )

            # Generate event if anomaly detected
            if result["is_anomaly"]:
                await self._generate_anomaly_event(session, data, detection_result)

            # Update session metrics
            await self._update_session_metrics(session, result)

            # Check alerts
            await self._check_session_alerts(session)

            # Send to output sink if configured
            if session.data_sink:
                await self._send_to_sink(session.data_sink, result)

            return result

        except Exception as e:
            logger.error(f"Error processing data for session {session_id}: {e}")
            await self._handle_processing_error(session, str(e))
            raise

    async def _run_streaming_session(self, session: StreamingSession) -> None:
        """Run the streaming session processing loop."""
        try:
            # Activate session
            session.activate_session()
            await self.streaming_repository.update_session(session)

            # Initialize data source
            data_stream = await self._initialize_data_source(session.data_source)

            logger.info(f"Started streaming session {session.id}")

            # Process data stream
            async for data_batch in data_stream:
                # Check if session should continue
                if not session.is_active():
                    break

                if session.is_expired():
                    session.stop_session("Session expired")
                    break

                # Process batch
                for data_point in data_batch:
                    if session.status == SessionStatus.PAUSED:
                        await asyncio.sleep(1)
                        continue

                    try:
                        await self.process_streaming_data(session.id, data_point)
                    except Exception as e:
                        logger.error(f"Error processing data point: {e}")
                        # Continue processing other points

                # Update session activity
                session.update_activity()

                # Checkpoint if enabled
                if session.configuration.enable_checkpointing:
                    await self._checkpoint_session(session)

        except Exception as e:
            logger.error(f"Streaming session {session.id} failed: {e}")
            session.stop_session(f"Session failed: {str(e)}")

        finally:
            # Clean up
            await self.streaming_repository.update_session(session)
            self._active_sessions.pop(session.id, None)
            logger.info(f"Stopped streaming session {session.id}")

    async def _initialize_data_source(
        self, data_source: StreamingDataSource
    ) -> AsyncGenerator[list[dict[str, Any]], None]:
        """Initialize and create data stream from source."""
        # This would implement actual data source connectors
        # For now, return a mock stream

        if data_source.source_type == "kafka":
            # Kafka connector implementation
            async for batch in self._create_kafka_stream(data_source):
                yield batch
        elif data_source.source_type == "kinesis":
            # Kinesis connector implementation
            async for batch in self._create_kinesis_stream(data_source):
                yield batch
        elif data_source.source_type == "websocket":
            # WebSocket connector implementation
            async for batch in self._create_websocket_stream(data_source):
                yield batch
        else:
            # Mock stream for testing
            async for batch in self._create_mock_stream():
                yield batch

    async def _create_mock_stream(self) -> AsyncGenerator[list[dict[str, Any]], None]:
        """Create a mock data stream for testing."""
        import random

        while True:
            # Generate mock batch
            batch = []
            for _ in range(random.randint(1, 10)):
                data_point = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "feature1": random.uniform(0, 100),
                    "feature2": random.uniform(-50, 50),
                    "feature3": random.choice(["A", "B", "C"]),
                    "value": random.uniform(0, 1000),
                }
                batch.append(data_point)

            yield batch
            await asyncio.sleep(1)  # 1 second delay between batches

    async def _create_kafka_stream(
        self, data_source: StreamingDataSource
    ) -> AsyncGenerator[list[dict[str, Any]], None]:
        """Create Kafka data stream."""
        # This would implement actual Kafka consumer
        # Placeholder implementation
        while True:
            yield [{"mock": "kafka_data"}]
            await asyncio.sleep(1)

    async def _create_kinesis_stream(
        self, data_source: StreamingDataSource
    ) -> AsyncGenerator[list[dict[str, Any]], None]:
        """Create Kinesis data stream."""
        # This would implement actual Kinesis consumer
        # Placeholder implementation
        while True:
            yield [{"mock": "kinesis_data"}]
            await asyncio.sleep(1)

    async def _create_websocket_stream(
        self, data_source: StreamingDataSource
    ) -> AsyncGenerator[list[dict[str, Any]], None]:
        """Create WebSocket data stream."""
        # This would implement actual WebSocket client
        # Placeholder implementation
        while True:
            yield [{"mock": "websocket_data"}]
            await asyncio.sleep(1)

    async def _preprocess_data(
        self, data: dict[str, Any], config: StreamingConfiguration
    ) -> dict[str, Any]:
        """Preprocess input data."""
        # Basic preprocessing
        processed = data.copy()

        # Schema validation if enabled
        if config.schema_validation:
            await self._validate_schema(processed)

        # Data cleaning and transformation
        processed = await self._clean_data(processed)

        return processed

    async def _validate_schema(self, data: dict[str, Any]) -> None:
        """Validate data schema."""
        # This would implement actual schema validation
        pass

    async def _clean_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Clean and transform data."""
        # This would implement data cleaning logic
        return data

    async def _generate_anomaly_event(
        self,
        session: StreamingSession,
        data: dict[str, Any],
        detection_result: dict[str, Any],
    ) -> None:
        """Generate anomaly event."""
        anomaly_data = AnomalyEventData(
            anomaly_score=detection_result["anomaly_score"],
            confidence=detection_result["confidence"],
            feature_contributions=detection_result.get("feature_contributions", {}),
            explanation=detection_result.get("explanation"),
            model_version=session.model_version,
        )

        event = AnomalyEvent(
            event_type=EventType.ANOMALY_DETECTED,
            severity=self._determine_severity(detection_result["anomaly_score"]),
            source_session_id=session.id,
            detector_id=session.detector_id,
            title=f"Anomaly detected in {session.name}",
            description=f"Anomaly score: {detection_result['anomaly_score']:.3f}",
            raw_data=data,
            anomaly_data=anomaly_data,
            event_time=datetime.utcnow(),
        )

        await self.event_repository.create_event(event)

    def _determine_severity(self, anomaly_score: float) -> EventSeverity:
        """Determine event severity based on anomaly score."""
        if anomaly_score >= 0.9:
            return EventSeverity.CRITICAL
        elif anomaly_score >= 0.7:
            return EventSeverity.HIGH
        elif anomaly_score >= 0.5:
            return EventSeverity.MEDIUM
        else:
            return EventSeverity.LOW

    async def _update_session_metrics(
        self, session: StreamingSession, result: dict[str, Any]
    ) -> None:
        """Update session metrics."""
        metrics = session.current_metrics

        # Update counters
        metrics.messages_processed += 1
        if result["is_anomaly"]:
            metrics.anomalies_detected += 1

        # Update rates
        uptime = session.get_uptime()
        if uptime and uptime.total_seconds() > 0:
            metrics.messages_per_second = (
                metrics.messages_processed / uptime.total_seconds()
            )
            metrics.anomaly_rate = (
                metrics.anomalies_detected / metrics.messages_processed
            )

        # Update in session
        session.update_metrics(metrics)

        # Persist to repository
        await self.streaming_repository.update_session(session)

    async def _check_session_alerts(self, session: StreamingSession) -> None:
        """Check and trigger session alerts."""
        alerts = await self.streaming_repository.get_session_alerts(session.id)

        for alert in alerts:
            if not alert.enabled:
                continue

            # Get current metric value
            current_value = getattr(session.current_metrics, alert.metric_name, 0.0)

            # Check condition
            if alert.evaluate_condition(current_value):
                if not alert.is_triggered:
                    alert.trigger_alert()
                    await self.streaming_repository.update_alert(alert)
                    await self._send_alert_notification(alert, current_value)
            else:
                if alert.is_triggered:
                    alert.resolve_alert()
                    await self.streaming_repository.update_alert(alert)

    async def _send_alert_notification(
        self, alert: StreamingAlert, current_value: float
    ) -> None:
        """Send alert notification."""
        # This would implement actual notification sending
        logger.warning(
            f"Alert triggered: {alert.name} - {alert.metric_name}={current_value} "
            f"{alert.comparison_operator} {alert.threshold_value}"
        )

    async def _send_to_sink(
        self, data_sink: StreamingDataSink, result: dict[str, Any]
    ) -> None:
        """Send result to output sink."""
        # This would implement actual sink connectors
        if data_sink.sink_type == "kafka":
            await self._send_to_kafka(data_sink, result)
        elif data_sink.sink_type == "database":
            await self._send_to_database(data_sink, result)
        # Add other sink implementations

    async def _send_to_kafka(
        self, sink: StreamingDataSink, result: dict[str, Any]
    ) -> None:
        """Send result to Kafka."""
        # Kafka producer implementation
        pass

    async def _send_to_database(
        self, sink: StreamingDataSink, result: dict[str, Any]
    ) -> None:
        """Send result to database."""
        # Database insert implementation
        pass

    async def _checkpoint_session(self, session: StreamingSession) -> None:
        """Create checkpoint for session."""
        # This would implement checkpointing logic
        session.current_metrics.last_checkpoint = datetime.utcnow()

    async def _handle_processing_error(
        self, session: StreamingSession, error: str
    ) -> None:
        """Handle processing error."""
        metrics = session.current_metrics
        metrics.failed_messages += 1

        uptime = session.get_uptime()
        if uptime and uptime.total_seconds() > 0:
            metrics.error_rate = metrics.failed_messages / metrics.messages_processed

        session.update_metrics(metrics)
        await self.streaming_repository.update_session(session)
