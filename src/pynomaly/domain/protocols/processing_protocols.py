"""Domain protocols for processing services."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable


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


class DetectionAlgorithm(Enum):
    """Detection algorithms for processing."""
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    AUTOENCODER = "autoencoder"


class StreamSource(Enum):
    """Stream source types."""
    KAFKA = "kafka"
    MQTT = "mqtt"
    WEBSOCKET = "websocket"
    FILE = "file"
    MEMORY = "memory"


class BatchEngine(Enum):
    """Batch processing engines."""
    MULTIPROCESSING = "multiprocessing"
    DASK = "dask"
    RAY = "ray"
    SPARK = "spark"


@dataclass
class ProcessingConfig:
    """Base processing configuration."""
    detection_algorithm: DetectionAlgorithm
    batch_size: int = 1000
    max_workers: int = 4
    timeout_seconds: float = 300.0
    extra_params: dict[str, Any] | None = None


@dataclass
class StreamConfig:
    """Stream processing configuration."""
    source_type: StreamSource
    source_config: dict[str, Any]
    detection_algorithm: DetectionAlgorithm
    batch_size: int = 1000
    max_workers: int = 4
    timeout_seconds: float = 300.0
    extra_params: dict[str, Any] | None = None
    batch_timeout_seconds: float = 10.0
    enable_backpressure: bool = False


@dataclass
class BatchConfig:
    """Batch processing configuration."""
    engine: BatchEngine
    detection_algorithm: DetectionAlgorithm
    batch_size: int = 1000
    max_workers: int = 4
    timeout_seconds: float = 300.0
    extra_params: dict[str, Any] | None = None
    chunk_size: int = 10000
    retry_attempts: int = 3
    checkpoint_interval: int = 1000


@dataclass
class ProcessingResult:
    """Processing operation result."""
    session_id: str
    total_records: int
    total_anomalies: int
    processing_time_seconds: float
    error_count: int = 0
    metadata: dict[str, Any] | None = None


@dataclass
class ProcessingSession:
    """Processing session information."""
    session_id: str
    mode: ProcessingMode
    status: ProcessingStatus
    created_at: datetime
    started_at: datetime | None = None
    stopped_at: datetime | None = None
    config: ProcessingConfig | None = None
    total_records: int = 0
    total_anomalies: int = 0
    error_count: int = 0
    job_ids: list[str] = None

    def __post_init__(self):
        if self.job_ids is None:
            self.job_ids = []


@runtime_checkable
class TracingProtocol(Protocol):
    """Protocol for distributed tracing operations."""

    def trace_operation(self, operation_name: str):
        """Create a trace decorator for an operation."""
        ...


@runtime_checkable
class StreamProcessorProtocol(Protocol):
    """Protocol for stream processors."""

    async def start(self) -> None:
        """Start the stream processor."""
        ...

    async def stop(self) -> None:
        """Stop the stream processor."""
        ...

    def get_status(self) -> dict[str, Any]:
        """Get current processor status."""
        ...

    async def add_test_record(self, data: dict[str, Any]) -> None:
        """Add test record for testing purposes."""
        ...


@runtime_checkable
class BatchProcessorProtocol(Protocol):
    """Protocol for batch processors."""

    async def submit_job(
        self,
        name: str,
        description: str,
        input_path: str,
        output_path: str,
        config: BatchConfig
    ) -> str:
        """Submit a batch job."""
        ...

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a batch job."""
        ...

    async def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get job status."""
        ...

    async def shutdown(self) -> None:
        """Shutdown the batch processor."""
        ...


@runtime_checkable
class ProcessorFactoryProtocol(Protocol):
    """Protocol for processor factories."""

    def create_stream_processor(self, config: StreamConfig) -> StreamProcessorProtocol:
        """Create a stream processor."""
        ...

    def create_batch_processor(self, config: BatchConfig) -> BatchProcessorProtocol:
        """Create a batch processor."""
        ...


@runtime_checkable
class ProcessingOrchestratorProtocol(Protocol):
    """Protocol for processing orchestrators."""

    async def start_streaming_session(
        self,
        name: str,
        source_config: dict[str, Any],
        source_type: StreamSource = StreamSource.KAFKA,
        detection_algorithm: DetectionAlgorithm = DetectionAlgorithm.ISOLATION_FOREST,
        **kwargs: Any
    ) -> str:
        """Start a streaming processing session."""
        ...

    async def start_batch_session(
        self,
        name: str,
        input_path: str,
        output_path: str,
        engine: BatchEngine = BatchEngine.MULTIPROCESSING,
        detection_algorithm: DetectionAlgorithm = DetectionAlgorithm.ISOLATION_FOREST,
        **kwargs: Any
    ) -> str:
        """Start a batch processing session."""
        ...

    async def start_hybrid_session(
        self,
        name: str,
        stream_config: dict[str, Any],
        batch_config: dict[str, Any],
        detection_algorithm: DetectionAlgorithm = DetectionAlgorithm.ISOLATION_FOREST,
    ) -> str:
        """Start a hybrid processing session."""
        ...

    async def stop_session(self, session_id: str) -> bool:
        """Stop a processing session."""
        ...

    async def get_session_status(self, session_id: str) -> dict[str, Any] | None:
        """Get status of a processing session."""
        ...

    async def list_sessions(
        self, mode: ProcessingMode | None = None
    ) -> list[dict[str, Any]]:
        """List all processing sessions."""
        ...

    async def get_processing_recommendations(
        self,
        data_size_mb: float,
        expected_throughput: float,
        latency_requirements: str = "normal",
    ) -> dict[str, Any]:
        """Get recommendations for processing mode and configuration."""
        ...

    async def get_system_metrics(self) -> dict[str, Any]:
        """Get orchestrator system metrics."""
        ...

    async def shutdown(self) -> None:
        """Shutdown the orchestrator."""
        ...


@runtime_checkable
class ConfigProtocol(Protocol):
    """Protocol for configuration services."""

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        ...


class ProcessingOrchestratorService(ABC):
    """Abstract base class for processing orchestrator services."""

    def __init__(
        self,
        processor_factory: ProcessorFactoryProtocol,
        tracing: TracingProtocol | None = None,
        config: ConfigProtocol | None = None
    ):
        self.processor_factory = processor_factory
        self.tracing = tracing
        self.config = config
        self.sessions: dict[str, ProcessingSession] = {}
        self.max_concurrent_sessions = 10
        self.enable_auto_scaling = True

    @abstractmethod
    async def start_streaming_session(
        self,
        name: str,
        source_config: dict[str, Any],
        source_type: StreamSource = StreamSource.KAFKA,
        detection_algorithm: DetectionAlgorithm = DetectionAlgorithm.ISOLATION_FOREST,
        **kwargs: Any
    ) -> str:
        """Start a streaming processing session."""
        ...

    @abstractmethod
    async def start_batch_session(
        self,
        name: str,
        input_path: str,
        output_path: str,
        engine: BatchEngine = BatchEngine.MULTIPROCESSING,
        detection_algorithm: DetectionAlgorithm = DetectionAlgorithm.ISOLATION_FOREST,
        **kwargs: Any
    ) -> str:
        """Start a batch processing session."""
        ...

    @abstractmethod
    async def stop_session(self, session_id: str) -> bool:
        """Stop a processing session."""
        ...

    @abstractmethod
    async def get_session_status(self, session_id: str) -> dict[str, Any] | None:
        """Get status of a processing session."""
        ...
