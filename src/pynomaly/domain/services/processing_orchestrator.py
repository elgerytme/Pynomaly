"""Processing orchestrator for streaming and batch anomaly detection."""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ...infrastructure.batch.batch_processor import (
    BatchConfig,
    BatchEngine,
    BatchProcessor,
    DataFormat,
    create_batch_processor,
)
from ...infrastructure.monitoring.distributed_tracing import trace_operation
from ...infrastructure.streaming.stream_processor import (
    StreamConfig,
    StreamFormat,
    StreamProcessor,
    StreamSource,
    create_stream_processor,
)
from ...shared.config import Config
from .advanced_detection_service import DetectionAlgorithm

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing modes available."""

    STREAMING = "streaming"
    BATCH = "batch"
    HYBRID = "hybrid"


class ProcessingStatus(Enum):
    """Processing status."""

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ProcessingSession:
    """Processing session information."""

    session_id: str
    mode: ProcessingMode
    status: ProcessingStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None

    # Configuration
    config: Optional[Union[StreamConfig, BatchConfig]] = None

    # Metrics
    total_records: int = 0
    total_anomalies: int = 0
    error_count: int = 0

    # References
    processor_instance: Optional[Union[StreamProcessor, BatchProcessor]] = None
    job_ids: List[str] = None


class ProcessingOrchestrator:
    """Orchestrator for streaming and batch processing modes."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize processing orchestrator."""
        self.config = config or Config()

        # Active sessions
        self.sessions: Dict[str, ProcessingSession] = {}

        # Processors (lazy initialized)
        self._stream_processors: Dict[str, StreamProcessor] = {}
        self._batch_processors: Dict[str, BatchProcessor] = {}

        # Global settings
        self.max_concurrent_sessions = self.config.get(
            "processing.max_concurrent_sessions", 10
        )
        self.enable_auto_scaling = self.config.get(
            "processing.enable_auto_scaling", True
        )

        # Monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_active = False

        logger.info("Processing orchestrator initialized")

    @trace_operation("start_streaming")
    async def start_streaming_session(
        self,
        name: str,
        source_config: Dict[str, Any],
        source_type: StreamSource = StreamSource.KAFKA,
        detection_algorithm: DetectionAlgorithm = DetectionAlgorithm.ISOLATION_FOREST,
        **kwargs,
    ) -> str:
        """Start a streaming processing session."""

        if len(self.sessions) >= self.max_concurrent_sessions:
            raise RuntimeError("Maximum concurrent sessions reached")

        session_id = str(uuid.uuid4())

        try:
            # Create stream configuration
            stream_config = StreamConfig(
                source_type=source_type,
                source_config=source_config,
                detection_algorithm=detection_algorithm,
                **kwargs,
            )

            # Create session
            session = ProcessingSession(
                session_id=session_id,
                mode=ProcessingMode.STREAMING,
                status=ProcessingStatus.STARTING,
                created_at=datetime.now(),
                config=stream_config,
                job_ids=[],
            )

            self.sessions[session_id] = session

            # Create and start stream processor
            stream_processor = create_stream_processor(stream_config)
            session.processor_instance = stream_processor
            self._stream_processors[session_id] = stream_processor

            # Start processing
            await stream_processor.start()

            session.status = ProcessingStatus.RUNNING
            session.started_at = datetime.now()

            logger.info(f"Started streaming session {session_id} ({name})")
            return session_id

        except Exception as e:
            if session_id in self.sessions:
                self.sessions[session_id].status = ProcessingStatus.ERROR
            logger.error(f"Failed to start streaming session: {e}")
            raise

    @trace_operation("start_batch")
    async def start_batch_session(
        self,
        name: str,
        input_path: str,
        output_path: str,
        engine: BatchEngine = BatchEngine.MULTIPROCESSING,
        detection_algorithm: DetectionAlgorithm = DetectionAlgorithm.ISOLATION_FOREST,
        **kwargs,
    ) -> str:
        """Start a batch processing session."""

        session_id = str(uuid.uuid4())

        try:
            # Create batch configuration
            batch_config = BatchConfig(
                engine=engine, detection_algorithm=detection_algorithm, **kwargs
            )

            # Create session
            session = ProcessingSession(
                session_id=session_id,
                mode=ProcessingMode.BATCH,
                status=ProcessingStatus.STARTING,
                created_at=datetime.now(),
                config=batch_config,
                job_ids=[],
            )

            self.sessions[session_id] = session

            # Create batch processor
            batch_processor = create_batch_processor(batch_config)
            session.processor_instance = batch_processor
            self._batch_processors[session_id] = batch_processor

            # Submit batch job
            job_id = await batch_processor.submit_job(
                name=name,
                description=f"Batch job for session {session_id}",
                input_path=input_path,
                output_path=output_path,
                config=batch_config,
            )

            session.job_ids.append(job_id)
            session.status = ProcessingStatus.RUNNING
            session.started_at = datetime.now()

            logger.info(
                f"Started batch session {session_id} ({name}) with job {job_id}"
            )
            return session_id

        except Exception as e:
            if session_id in self.sessions:
                self.sessions[session_id].status = ProcessingStatus.ERROR
            logger.error(f"Failed to start batch session: {e}")
            raise

    async def start_hybrid_session(
        self,
        name: str,
        stream_config: Dict[str, Any],
        batch_config: Dict[str, Any],
        detection_algorithm: DetectionAlgorithm = DetectionAlgorithm.ISOLATION_FOREST,
    ) -> str:
        """Start a hybrid processing session (both streaming and batch)."""

        session_id = str(uuid.uuid4())

        try:
            # Create session
            session = ProcessingSession(
                session_id=session_id,
                mode=ProcessingMode.HYBRID,
                status=ProcessingStatus.STARTING,
                created_at=datetime.now(),
                job_ids=[],
            )

            self.sessions[session_id] = session

            # Start streaming component
            stream_session_id = await self.start_streaming_session(
                name=f"{name}_stream",
                detection_algorithm=detection_algorithm,
                **stream_config,
            )

            # Start batch component
            batch_session_id = await self.start_batch_session(
                name=f"{name}_batch",
                detection_algorithm=detection_algorithm,
                **batch_config,
            )

            # Link sub-sessions
            session.job_ids = [stream_session_id, batch_session_id]
            session.status = ProcessingStatus.RUNNING
            session.started_at = datetime.now()

            logger.info(f"Started hybrid session {session_id} ({name})")
            return session_id

        except Exception as e:
            if session_id in self.sessions:
                self.sessions[session_id].status = ProcessingStatus.ERROR
            logger.error(f"Failed to start hybrid session: {e}")
            raise

    async def stop_session(self, session_id: str) -> bool:
        """Stop a processing session."""
        session = self.sessions.get(session_id)
        if not session:
            return False

        try:
            session.status = ProcessingStatus.STOPPING

            if session.mode == ProcessingMode.STREAMING:
                # Stop stream processor
                if session_id in self._stream_processors:
                    await self._stream_processors[session_id].stop()
                    del self._stream_processors[session_id]

            elif session.mode == ProcessingMode.BATCH:
                # Cancel batch jobs
                if session_id in self._batch_processors:
                    batch_processor = self._batch_processors[session_id]
                    for job_id in session.job_ids:
                        await batch_processor.cancel_job(job_id)
                    await batch_processor.shutdown()
                    del self._batch_processors[session_id]

            elif session.mode == ProcessingMode.HYBRID:
                # Stop linked sessions
                for linked_session_id in session.job_ids:
                    await self.stop_session(linked_session_id)

            session.status = ProcessingStatus.STOPPED
            session.stopped_at = datetime.now()

            logger.info(f"Stopped session {session_id}")
            return True

        except Exception as e:
            session.status = ProcessingStatus.ERROR
            logger.error(f"Failed to stop session {session_id}: {e}")
            return False

    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a processing session."""
        session = self.sessions.get(session_id)
        if not session:
            return None

        status_data = {
            "session_id": session.session_id,
            "mode": session.mode.value,
            "status": session.status.value,
            "created_at": session.created_at.isoformat(),
            "started_at": (
                session.started_at.isoformat() if session.started_at else None
            ),
            "stopped_at": (
                session.stopped_at.isoformat() if session.stopped_at else None
            ),
            "total_records": session.total_records,
            "total_anomalies": session.total_anomalies,
            "error_count": session.error_count,
        }

        # Add mode-specific details
        if (
            session.mode == ProcessingMode.STREAMING
            and session_id in self._stream_processors
        ):
            processor_status = self._stream_processors[session_id].get_status()
            status_data["streaming_details"] = processor_status

        elif (
            session.mode == ProcessingMode.BATCH
            and session_id in self._batch_processors
        ):
            batch_processor = self._batch_processors[session_id]
            job_statuses = []
            for job_id in session.job_ids:
                job_status = await batch_processor.get_job_status(job_id)
                if job_status:
                    job_statuses.append(job_status)
            status_data["batch_details"] = {"jobs": job_statuses}

        elif session.mode == ProcessingMode.HYBRID:
            # Get status of linked sessions
            linked_statuses = []
            for linked_session_id in session.job_ids:
                linked_status = await self.get_session_status(linked_session_id)
                if linked_status:
                    linked_statuses.append(linked_status)
            status_data["hybrid_details"] = {"linked_sessions": linked_statuses}

        return status_data

    async def list_sessions(
        self, mode: Optional[ProcessingMode] = None
    ) -> List[Dict[str, Any]]:
        """List all processing sessions."""
        sessions = list(self.sessions.values())

        if mode:
            sessions = [s for s in sessions if s.mode == mode]

        result = []
        for session in sessions:
            session_data = await self.get_session_status(session.session_id)
            if session_data:
                result.append(session_data)

        return sorted(result, key=lambda x: x["created_at"], reverse=True)

    async def add_stream_data(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Add data to a streaming session (for testing)."""
        if session_id not in self._stream_processors:
            return False

        try:
            await self._stream_processors[session_id].add_test_record(data)
            return True
        except Exception as e:
            logger.error(f"Failed to add stream data: {e}")
            return False

    async def get_processing_recommendations(
        self,
        data_size_mb: float,
        expected_throughput: float,
        latency_requirements: str = "normal",
    ) -> Dict[str, Any]:
        """Get recommendations for processing mode and configuration."""

        recommendations = {
            "recommended_mode": None,
            "configuration": {},
            "reasoning": [],
            "alternatives": [],
        }

        # Size-based recommendations
        if data_size_mb < 100:  # Small datasets
            recommendations["recommended_mode"] = ProcessingMode.STREAMING.value
            recommendations["configuration"] = {
                "batch_size": 1000,
                "engine": "memory",
                "max_workers": 2,
            }
            recommendations["reasoning"].append(
                "Small dataset suitable for streaming processing"
            )

        elif data_size_mb > 10000:  # Large datasets (>10GB)
            recommendations["recommended_mode"] = ProcessingMode.BATCH.value
            recommendations["configuration"] = {
                "engine": BatchEngine.DASK.value,
                "chunk_size": 50000,
                "max_workers": 8,
            }
            recommendations["reasoning"].append(
                "Large dataset requires batch processing for efficiency"
            )

        else:  # Medium datasets
            if latency_requirements == "low":
                recommendations["recommended_mode"] = ProcessingMode.STREAMING.value
                recommendations["configuration"] = {
                    "batch_size": 5000,
                    "batch_timeout_seconds": 10.0,
                    "max_workers": 4,
                }
                recommendations["reasoning"].append(
                    "Low latency requirements favor streaming"
                )
            else:
                recommendations["recommended_mode"] = ProcessingMode.BATCH.value
                recommendations["configuration"] = {
                    "engine": BatchEngine.MULTIPROCESSING.value,
                    "chunk_size": 10000,
                    "max_workers": 4,
                }
                recommendations["reasoning"].append(
                    "Medium dataset with normal latency requirements"
                )

        # Throughput-based adjustments
        if expected_throughput > 10000:  # High throughput
            if recommendations["recommended_mode"] == ProcessingMode.STREAMING.value:
                recommendations["configuration"]["max_workers"] = 8
                recommendations["configuration"]["enable_backpressure"] = True
                recommendations["reasoning"].append(
                    "High throughput requires more workers and backpressure"
                )
            else:
                recommendations["configuration"]["engine"] = BatchEngine.DASK.value
                recommendations["configuration"]["max_workers"] = 12
                recommendations["reasoning"].append(
                    "High throughput batch processing with Dask"
                )

        # Add alternatives
        if recommendations["recommended_mode"] == ProcessingMode.STREAMING.value:
            recommendations["alternatives"].append(
                {
                    "mode": ProcessingMode.BATCH.value,
                    "pros": [
                        "Better for large datasets",
                        "More fault tolerant",
                        "Better resource utilization",
                    ],
                    "cons": ["Higher latency", "More complex setup"],
                }
            )
        else:
            recommendations["alternatives"].append(
                {
                    "mode": ProcessingMode.STREAMING.value,
                    "pros": [
                        "Lower latency",
                        "Real-time processing",
                        "Simpler for small data",
                    ],
                    "cons": ["Limited scalability", "Memory constraints"],
                }
            )

        # Hybrid option for complex scenarios
        if data_size_mb > 1000 and expected_throughput > 5000:
            recommendations["alternatives"].append(
                {
                    "mode": ProcessingMode.HYBRID.value,
                    "pros": [
                        "Combines benefits of both modes",
                        "Flexible processing",
                        "Handles mixed workloads",
                    ],
                    "cons": ["More complex setup", "Higher resource requirements"],
                }
            )

        return recommendations

    async def start_monitoring(self) -> None:
        """Start background monitoring of sessions."""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started processing orchestrator monitoring")

    async def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        if not self._monitoring_active:
            return

        self._monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped processing orchestrator monitoring")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                await self._update_session_metrics()
                await self._check_session_health()
                await self._cleanup_finished_sessions()

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _update_session_metrics(self) -> None:
        """Update metrics for all active sessions."""
        for session_id, session in self.sessions.items():
            if session.status != ProcessingStatus.RUNNING:
                continue

            try:
                if (
                    session.mode == ProcessingMode.STREAMING
                    and session_id in self._stream_processors
                ):
                    processor = self._stream_processors[session_id]
                    status = processor.get_status()
                    session.total_records = status.get("processed_records", 0)
                    session.error_count = status.get("error_count", 0)

                elif (
                    session.mode == ProcessingMode.BATCH
                    and session_id in self._batch_processors
                ):
                    batch_processor = self._batch_processors[session_id]
                    for job_id in session.job_ids:
                        job_status = await batch_processor.get_job_status(job_id)
                        if job_status:
                            session.total_records = job_status.get("total_samples", 0)
                            session.total_anomalies = job_status.get(
                                "total_anomalies", 0
                            )

            except Exception as e:
                logger.warning(f"Error updating metrics for session {session_id}: {e}")

    async def _check_session_health(self) -> None:
        """Check health of all sessions."""
        for session_id, session in self.sessions.items():
            if session.status != ProcessingStatus.RUNNING:
                continue

            try:
                # Check if session has been running too long without progress
                if session.started_at:
                    runtime = (datetime.now() - session.started_at).total_seconds()

                    # Simple health check - could be more sophisticated
                    if (
                        runtime > 3600 and session.total_records == 0
                    ):  # 1 hour with no records
                        logger.warning(
                            f"Session {session_id} appears unhealthy - no records processed in 1 hour"
                        )
                        session.status = ProcessingStatus.ERROR

            except Exception as e:
                logger.warning(f"Error checking health for session {session_id}: {e}")

    async def _cleanup_finished_sessions(self) -> None:
        """Clean up finished sessions."""
        finished_sessions = []

        for session_id, session in self.sessions.items():
            if session.status in [ProcessingStatus.STOPPED, ProcessingStatus.ERROR]:
                if session.stopped_at:
                    # Clean up sessions older than 24 hours
                    age = (datetime.now() - session.stopped_at).total_seconds()
                    if age > 86400:  # 24 hours
                        finished_sessions.append(session_id)

        for session_id in finished_sessions:
            del self.sessions[session_id]
            logger.info(f"Cleaned up finished session {session_id}")

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get orchestrator system metrics."""
        total_sessions = len(self.sessions)
        running_sessions = len(
            [s for s in self.sessions.values() if s.status == ProcessingStatus.RUNNING]
        )

        mode_counts = {}
        for session in self.sessions.values():
            mode = session.mode.value
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        return {
            "total_sessions": total_sessions,
            "running_sessions": running_sessions,
            "max_concurrent_sessions": self.max_concurrent_sessions,
            "sessions_by_mode": mode_counts,
            "active_stream_processors": len(self._stream_processors),
            "active_batch_processors": len(self._batch_processors),
            "monitoring_active": self._monitoring_active,
            "auto_scaling_enabled": self.enable_auto_scaling,
        }

    async def shutdown(self) -> None:
        """Shutdown the orchestrator and all active sessions."""
        logger.info("Shutting down processing orchestrator...")

        # Stop monitoring
        await self.stop_monitoring()

        # Stop all active sessions
        for session_id in list(self.sessions.keys()):
            await self.stop_session(session_id)

        logger.info("Processing orchestrator shutdown complete")


# Global orchestrator instance
_orchestrator: Optional[ProcessingOrchestrator] = None


def get_processing_orchestrator(
    config: Optional[Config] = None,
) -> ProcessingOrchestrator:
    """Get the global processing orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ProcessingOrchestrator(config)
    return _orchestrator
