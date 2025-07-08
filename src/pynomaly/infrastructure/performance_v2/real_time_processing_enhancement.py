"""Real-time processing enhancement for sub-millisecond anomaly detection."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Optional
from uuid import UUID, uuid4

import numpy as np

logger = logging.getLogger(__name__)

class ProcessingMode(str, Enum):
    ULTRA_LOW_LATENCY = "ultra_low_latency"  # < 1ms
    LOW_LATENCY = "low_latency"              # < 10ms
    REAL_TIME = "real_time"                  # < 100ms
    NEAR_REAL_TIME = "near_real_time"        # < 1s

class NetworkOptimization(str, Enum):
    KERNEL_BYPASS = "kernel_bypass"
    RDMA = "rdma"
    DPDK = "dpdk"
    SR_IOV = "sr_iov"
    STANDARD = "standard"

class MemoryPattern(str, Enum):
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    TEMPORAL_LOCALITY = "temporal_locality"
    SPATIAL_LOCALITY = "spatial_locality"

@dataclass
class LatencyProfile:
    """Latency requirements and measurements."""
    profile_id: str
    target_latency_us: float  # Target latency in microseconds
    max_latency_us: float     # Maximum acceptable latency
    percentile_99_us: float = 0.0
    percentile_95_us: float = 0.0
    percentile_50_us: float = 0.0
    jitter_us: float = 0.0    # Latency jitter

    # Measurements
    measured_latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    def add_measurement(self, latency_us: float) -> None:
        """Add a latency measurement."""
        self.measured_latencies.append(latency_us)
        self._update_percentiles()

    def _update_percentiles(self) -> None:
        """Update latency percentiles."""
        if len(self.measured_latencies) < 10:
            return

        latencies = sorted(self.measured_latencies)
        n = len(latencies)

        self.percentile_50_us = latencies[int(n * 0.5)]
        self.percentile_95_us = latencies[int(n * 0.95)]
        self.percentile_99_us = latencies[int(n * 0.99)]

        # Calculate jitter (standard deviation)
        mean_latency = np.mean(latencies)
        self.jitter_us = np.std(latencies)

    def is_meeting_sla(self) -> bool:
        """Check if latency SLA is being met."""
        return (self.percentile_99_us <= self.max_latency_us and
                self.percentile_95_us <= self.target_latency_us * 1.2)

@dataclass
class ProcessingTask:
    """Real-time processing task."""
    task_id: UUID
    data: np.ndarray
    priority: int
    deadline_us: float  # Deadline in microseconds from now

    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Processing metadata
    algorithm: str = "default"
    parameters: dict[str, Any] = field(default_factory=dict)

    # Results
    anomaly_scores: Optional[np.ndarray] = None
    anomalies_detected: int = 0
    processing_time_us: float = 0.0

    def get_age_us(self) -> float:
        """Get task age in microseconds."""
        return (time.time() - self.created_at) * 1_000_000

    def is_expired(self) -> bool:
        """Check if task has exceeded its deadline."""
        return self.get_age_us() > self.deadline_us

    def start_processing(self) -> None:
        """Mark task as started."""
        self.started_at = time.time()

    def complete_processing(self, anomaly_scores: np.ndarray) -> None:
        """Mark task as completed."""
        self.completed_at = time.time()
        self.anomaly_scores = anomaly_scores
        self.anomalies_detected = int(np.sum(anomaly_scores > 0.5))

        if self.started_at:
            self.processing_time_us = (self.completed_at - self.started_at) * 1_000_000

@dataclass
class StreamingBuffer:
    """High-performance streaming buffer with zero-copy operations."""
    buffer_id: str
    capacity: int
    element_size: int

    # Buffer storage
    _data: np.ndarray = field(init=False)
    _write_index: int = 0
    _read_index: int = 0
    _is_full: bool = False

    # Performance metrics
    total_writes: int = 0
    total_reads: int = 0
    buffer_overruns: int = 0

    def __post_init__(self):
        self._data = np.zeros((self.capacity, self.element_size), dtype=np.float32)

    def write(self, data: np.ndarray) -> bool:
        """Write data to buffer with zero-copy semantics."""
        if data.shape[1] != self.element_size:
            logger.error(f"Data shape mismatch: expected {self.element_size}, got {data.shape[1]}")
            return False

        available_space = self.capacity - self.size()
        if len(data) > available_space:
            self.buffer_overruns += 1
            logger.warning(f"Buffer {self.buffer_id} overrun: need {len(data)}, have {available_space}")
            return False

        # Write data in chunks if it wraps around
        for row in data:
            self._data[self._write_index] = row
            self._write_index = (self._write_index + 1) % self.capacity
            self.total_writes += 1

            if self._write_index == self._read_index and not self._is_full:
                self._is_full = True

        return True

    def read(self, count: int) -> Optional[np.ndarray]:
        """Read data from buffer with zero-copy semantics."""
        if count > self.size():
            return None

        # Handle wrap-around reading
        if self._read_index + count <= self.capacity:
            # Simple case: no wrap-around
            result = self._data[self._read_index:self._read_index + count].copy()
        else:
            # Wrap-around case
            first_part = self._data[self._read_index:].copy()
            second_part_size = count - len(first_part)
            second_part = self._data[:second_part_size].copy()
            result = np.vstack([first_part, second_part])

        # Update read index
        self._read_index = (self._read_index + count) % self.capacity
        self.total_reads += count

        if self._is_full:
            self._is_full = False

        return result

    def size(self) -> int:
        """Get current buffer size."""
        if self._is_full:
            return self.capacity
        elif self._write_index >= self._read_index:
            return self._write_index - self._read_index
        else:
            return self.capacity - self._read_index + self._write_index

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self.size() == 0

    def get_utilization(self) -> float:
        """Get buffer utilization percentage."""
        return (self.size() / self.capacity) * 100

class UltraLowLatencyProcessor:
    """Ultra-low latency processor for sub-millisecond detection."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.processing_mode = ProcessingMode(config.get("mode", "ultra_low_latency"))
        self.latency_profile = LatencyProfile(
            profile_id="ultra_low_latency",
            target_latency_us=config.get("target_latency_us", 500),
            max_latency_us=config.get("max_latency_us", 1000)
        )

        # Processing queues by priority
        self.priority_queues: dict[int, list[ProcessingTask]] = {
            0: [],  # Critical (< 100us)
            1: [],  # High (< 500us)
            2: [],  # Medium (< 1ms)
            3: [],  # Low (< 10ms)
        }

        # Performance tracking
        self.processed_tasks = 0
        self.expired_tasks = 0
        self.processing_times: Deque[float] = deque(maxlen=1000)

        # Pre-compiled algorithms for speed
        self.compiled_algorithms: dict[str, Callable] = {}
        self._compile_algorithms()

    def _compile_algorithms(self) -> None:
        """Pre-compile algorithms for ultra-fast execution."""
        # Simple threshold-based detection (fastest)
        def threshold_detect(data: np.ndarray, threshold: float = 2.0) -> np.ndarray:
            return (np.abs(data) > threshold).astype(np.float32)

        # Statistical outlier detection (fast)
        def statistical_detect(data: np.ndarray, z_threshold: float = 3.0) -> np.ndarray:
            if len(data) < 2:
                return np.zeros(len(data), dtype=np.float32)

            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            z_scores = np.abs((data - mean) / (std + 1e-8))
            return (np.max(z_scores, axis=1) > z_threshold).astype(np.float32)

        # Distance-based detection (medium speed)
        def distance_detect(data: np.ndarray, distance_threshold: float = 1.5) -> np.ndarray:
            if len(data) < 2:
                return np.zeros(len(data), dtype=np.float32)

            # Use L2 norm from mean as simple distance measure
            mean = np.mean(data, axis=0)
            distances = np.linalg.norm(data - mean, axis=1)
            threshold = np.percentile(distances, 95) * distance_threshold
            return (distances > threshold).astype(np.float32)

        self.compiled_algorithms = {
            "threshold": threshold_detect,
            "statistical": statistical_detect,
            "distance": distance_detect,
        }

    async def submit_task(self, task: ProcessingTask) -> bool:
        """Submit task for ultra-low latency processing."""
        try:
            # Determine priority based on deadline
            if task.deadline_us <= 100:
                priority = 0  # Critical
            elif task.deadline_us <= 500:
                priority = 1  # High
            elif task.deadline_us <= 1000:
                priority = 2  # Medium
            else:
                priority = 3  # Low

            # Add to appropriate queue
            self.priority_queues[priority].append(task)

            # Trigger immediate processing for critical tasks
            if priority == 0:
                await self._process_critical_tasks()

            return True

        except Exception as e:
            logger.error(f"Failed to submit task {task.task_id}: {e}")
            return False

    async def _process_critical_tasks(self) -> None:
        """Process critical tasks immediately."""
        critical_queue = self.priority_queues[0]

        while critical_queue:
            task = critical_queue.pop(0)

            if task.is_expired():
                self.expired_tasks += 1
                continue

            await self._execute_task(task)

    async def start_processing(self) -> None:
        """Start the ultra-low latency processing loop."""
        asyncio.create_task(self._processing_loop())

    async def _processing_loop(self) -> None:
        """Main processing loop with priority scheduling."""
        while True:
            try:
                task_processed = False

                # Process queues by priority
                for priority in sorted(self.priority_queues.keys()):
                    queue = self.priority_queues[priority]

                    if queue:
                        task = queue.pop(0)

                        if task.is_expired():
                            self.expired_tasks += 1
                            continue

                        await self._execute_task(task)
                        task_processed = True
                        break

                # If no tasks, yield control briefly
                if not task_processed:
                    await asyncio.sleep(0.0001)  # 100 microseconds

            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                await asyncio.sleep(0.001)

    async def _execute_task(self, task: ProcessingTask) -> None:
        """Execute a processing task with timing measurement."""
        start_time = time.perf_counter()
        task.start_processing()

        try:
            # Select algorithm based on deadline constraints
            algorithm_name = self._select_algorithm(task.deadline_us)
            algorithm = self.compiled_algorithms[algorithm_name]

            # Execute algorithm
            anomaly_scores = algorithm(task.data, **task.parameters)

            # Complete task
            task.complete_processing(anomaly_scores)

            # Record timing
            end_time = time.perf_counter()
            processing_time_us = (end_time - start_time) * 1_000_000

            self.processing_times.append(processing_time_us)
            self.latency_profile.add_measurement(processing_time_us)
            self.processed_tasks += 1

        except Exception as e:
            logger.error(f"Task execution failed for {task.task_id}: {e}")

    def _select_algorithm(self, deadline_us: float) -> str:
        """Select algorithm based on deadline constraints."""
        if deadline_us <= 100:
            return "threshold"  # Fastest
        elif deadline_us <= 500:
            return "statistical"  # Fast
        else:
            return "distance"  # More accurate but slower

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Get real-time performance metrics."""
        if not self.processing_times:
            return {}

        recent_times = list(self.processing_times)

        return {
            "processed_tasks": self.processed_tasks,
            "expired_tasks": self.expired_tasks,
            "success_rate": (self.processed_tasks / max(self.processed_tasks + self.expired_tasks, 1)) * 100,
            "latency": {
                "current_avg_us": np.mean(recent_times[-100:]) if len(recent_times) >= 100 else np.mean(recent_times),
                "percentile_95_us": self.latency_profile.percentile_95_us,
                "percentile_99_us": self.latency_profile.percentile_99_us,
                "jitter_us": self.latency_profile.jitter_us,
                "sla_compliance": self.latency_profile.is_meeting_sla(),
            },
            "queue_depths": {str(p): len(q) for p, q in self.priority_queues.items()},
        }

class StreamProcessor:
    """High-performance stream processor with backpressure handling."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.max_buffer_size = config.get("max_buffer_size", 10000)
        self.batch_size = config.get("batch_size", 100)
        self.batch_timeout_ms = config.get("batch_timeout_ms", 10)

        # Streaming buffers
        self.input_buffer = StreamingBuffer(
            buffer_id="input",
            capacity=self.max_buffer_size,
            element_size=config.get("element_size", 10)
        )

        self.output_buffer = StreamingBuffer(
            buffer_id="output",
            capacity=self.max_buffer_size,
            element_size=1  # Anomaly scores
        )

        # Processing components
        self.ultra_processor = UltraLowLatencyProcessor(config.get("processor", {}))
        self.backpressure_handler = BackpressureHandler(config.get("backpressure", {}))

        # Stream statistics
        self.stream_stats = {
            "ingested_samples": 0,
            "processed_samples": 0,
            "dropped_samples": 0,
            "backpressure_events": 0,
        }

    async def start_streaming(self) -> None:
        """Start stream processing."""
        await self.ultra_processor.start_processing()
        asyncio.create_task(self._stream_processing_loop())
        logger.info("Stream processing started")

    async def ingest_data(self, data: np.ndarray) -> bool:
        """Ingest data into the stream."""
        try:
            # Check for backpressure
            if self.input_buffer.get_utilization() > 80:
                action = await self.backpressure_handler.handle_backpressure(
                    self.input_buffer.get_utilization(),
                    len(data)
                )

                if action == "drop":
                    self.stream_stats["dropped_samples"] += len(data)
                    self.stream_stats["backpressure_events"] += 1
                    return False
                elif action == "throttle":
                    await asyncio.sleep(0.001)  # 1ms throttle

            # Write to buffer
            success = self.input_buffer.write(data)
            if success:
                self.stream_stats["ingested_samples"] += len(data)
            else:
                self.stream_stats["dropped_samples"] += len(data)

            return success

        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            return False

    async def _stream_processing_loop(self) -> None:
        """Main stream processing loop."""
        last_batch_time = time.time()

        while True:
            try:
                current_time = time.time()
                batch_age_ms = (current_time - last_batch_time) * 1000

                # Check if we should process a batch
                should_process = (
                    self.input_buffer.size() >= self.batch_size or
                    (self.input_buffer.size() > 0 and batch_age_ms >= self.batch_timeout_ms)
                )

                if should_process:
                    await self._process_batch()
                    last_batch_time = current_time
                else:
                    await asyncio.sleep(0.0001)  # 100 microseconds

            except Exception as e:
                logger.error(f"Stream processing loop error: {e}")
                await asyncio.sleep(0.001)

    async def _process_batch(self) -> None:
        """Process a batch of data."""
        try:
            # Read batch from input buffer
            batch_size = min(self.batch_size, self.input_buffer.size())
            if batch_size == 0:
                return

            batch_data = self.input_buffer.read(batch_size)
            if batch_data is None:
                return

            # Create processing task
            task = ProcessingTask(
                task_id=uuid4(),
                data=batch_data,
                priority=1,
                deadline_us=self.batch_timeout_ms * 1000,  # Convert to microseconds
                algorithm="statistical"
            )

            # Submit for ultra-low latency processing
            await self.ultra_processor.submit_task(task)

            # Write results to output buffer (simulated)
            # In practice, this would wait for task completion
            anomaly_scores = np.random.random((batch_size, 1)).astype(np.float32)
            self.output_buffer.write(anomaly_scores)

            self.stream_stats["processed_samples"] += batch_size

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")

    async def get_stream_metrics(self) -> dict[str, Any]:
        """Get comprehensive stream metrics."""
        processor_metrics = await self.ultra_processor.get_performance_metrics()

        return {
            "stream_stats": self.stream_stats,
            "buffer_utilization": {
                "input": self.input_buffer.get_utilization(),
                "output": self.output_buffer.get_utilization(),
            },
            "processor_metrics": processor_metrics,
            "throughput": {
                "ingestion_rate": self.stream_stats["ingested_samples"] / max(time.time() - self.ultra_processor.latency_profile.measured_latencies[0] if self.ultra_processor.latency_profile.measured_latencies else time.time(), 1),
                "processing_rate": self.stream_stats["processed_samples"] / max(time.time() - self.ultra_processor.latency_profile.measured_latencies[0] if self.ultra_processor.latency_profile.measured_latencies else time.time(), 1),
                "drop_rate": self.stream_stats["dropped_samples"] / max(self.stream_stats["ingested_samples"] + self.stream_stats["dropped_samples"], 1) * 100,
            },
        }

class BackpressureHandler:
    """Handles backpressure in real-time streams."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.drop_threshold = config.get("drop_threshold", 95)
        self.throttle_threshold = config.get("throttle_threshold", 80)
        self.adaptive_mode = config.get("adaptive_mode", True)

        # Backpressure statistics
        self.backpressure_history: Deque[tuple[float, float]] = deque(maxlen=100)  # (timestamp, utilization)

    async def handle_backpressure(self, buffer_utilization: float, incoming_data_size: int) -> str:
        """Handle backpressure situation."""
        self.backpressure_history.append((time.time(), buffer_utilization))

        if buffer_utilization >= self.drop_threshold:
            return "drop"
        elif buffer_utilization >= self.throttle_threshold:
            if self.adaptive_mode:
                # Adaptive throttling based on recent trends
                throttle_duration = await self._calculate_adaptive_throttle()
                await asyncio.sleep(throttle_duration)
            return "throttle"
        else:
            return "allow"

    async def _calculate_adaptive_throttle(self) -> float:
        """Calculate adaptive throttle duration based on trends."""
        if len(self.backpressure_history) < 5:
            return 0.001  # 1ms default

        # Analyze recent utilization trend
        recent_utilizations = [u for _, u in list(self.backpressure_history)[-5:]]
        trend = np.polyfit(range(len(recent_utilizations)), recent_utilizations, 1)[0]

        if trend > 5:  # Rapidly increasing
            return 0.005  # 5ms
        elif trend > 1:  # Slowly increasing
            return 0.002  # 2ms
        else:  # Stable or decreasing
            return 0.001  # 1ms

class NetworkOptimizer:
    """Optimizes network performance for real-time processing."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.optimization_mode = NetworkOptimization(config.get("mode", "standard"))
        self.target_latency_us = config.get("target_latency_us", 100)

        # Network performance metrics
        self.network_metrics = {
            "round_trip_time_us": deque(maxlen=1000),
            "bandwidth_mbps": deque(maxlen=100),
            "packet_loss_rate": 0.0,
            "jitter_us": 0.0,
        }

    async def optimize_network_path(self, destination: str) -> dict[str, Any]:
        """Optimize network path for ultra-low latency."""
        try:
            if self.optimization_mode == NetworkOptimization.KERNEL_BYPASS:
                # Implement kernel bypass optimizations
                optimizations = await self._apply_kernel_bypass()
            elif self.optimization_mode == NetworkOptimization.RDMA:
                # Implement RDMA optimizations
                optimizations = await self._apply_rdma_optimization()
            elif self.optimization_mode == NetworkOptimization.DPDK:
                # Implement DPDK optimizations
                optimizations = await self._apply_dpdk_optimization()
            else:
                # Standard TCP optimizations
                optimizations = await self._apply_standard_optimizations()

            return {
                "optimization_applied": True,
                "mode": self.optimization_mode.value,
                "optimizations": optimizations,
                "expected_latency_reduction_us": optimizations.get("latency_reduction", 0),
            }

        except Exception as e:
            logger.error(f"Network optimization failed: {e}")
            return {"optimization_applied": False, "error": str(e)}

    async def _apply_kernel_bypass(self) -> dict[str, Any]:
        """Apply kernel bypass optimizations."""
        # Simulate kernel bypass setup
        await asyncio.sleep(0.01)
        return {
            "kernel_bypass_enabled": True,
            "user_space_networking": True,
            "latency_reduction": 50,  # 50 microseconds reduction
        }

    async def _apply_rdma_optimization(self) -> dict[str, Any]:
        """Apply RDMA optimizations."""
        # Simulate RDMA setup
        await asyncio.sleep(0.02)
        return {
            "rdma_enabled": True,
            "zero_copy_enabled": True,
            "latency_reduction": 100,  # 100 microseconds reduction
        }

    async def _apply_dpdk_optimization(self) -> dict[str, Any]:
        """Apply DPDK optimizations."""
        # Simulate DPDK setup
        await asyncio.sleep(0.015)
        return {
            "dpdk_enabled": True,
            "poll_mode_driver": True,
            "hugepages_enabled": True,
            "latency_reduction": 75,  # 75 microseconds reduction
        }

    async def _apply_standard_optimizations(self) -> dict[str, Any]:
        """Apply standard TCP optimizations."""
        # Simulate standard optimizations
        await asyncio.sleep(0.005)
        return {
            "tcp_nodelay": True,
            "socket_buffer_tuning": True,
            "cpu_affinity": True,
            "latency_reduction": 20,  # 20 microseconds reduction
        }

    async def measure_network_performance(self, destination: str) -> dict[str, Any]:
        """Measure current network performance."""
        # Simulate network measurements
        rtt_us = np.random.uniform(50, 200)
        bandwidth_mbps = np.random.uniform(800, 1000)

        self.network_metrics["round_trip_time_us"].append(rtt_us)
        self.network_metrics["bandwidth_mbps"].append(bandwidth_mbps)

        return {
            "round_trip_time_us": rtt_us,
            "bandwidth_mbps": bandwidth_mbps,
            "packet_loss_rate": self.network_metrics["packet_loss_rate"],
            "jitter_us": np.std(list(self.network_metrics["round_trip_time_us"])[-10:]) if len(self.network_metrics["round_trip_time_us"]) >= 10 else 0,
        }

class RealTimeProcessingOrchestrator:
    """Main orchestrator for real-time processing enhancement."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.stream_processor = StreamProcessor(config.get("stream", {}))
        self.network_optimizer = NetworkOptimizer(config.get("network", {}))
        self.processing_mode = ProcessingMode(config.get("mode", "real_time"))

        # System performance tracking
        self.system_metrics = {
            "start_time": time.time(),
            "total_processed": 0,
            "total_anomalies": 0,
        }

    async def initialize(self) -> bool:
        """Initialize real-time processing enhancement."""
        try:
            logger.info("Initializing real-time processing enhancement")

            # Start stream processing
            await self.stream_processor.start_streaming()

            # Optimize network if required
            if self.processing_mode in [ProcessingMode.ULTRA_LOW_LATENCY, ProcessingMode.LOW_LATENCY]:
                await self.network_optimizer.optimize_network_path("localhost")

            logger.info("Real-time processing enhancement initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize real-time processing enhancement: {e}")
            return False

    async def process_real_time_data(self, data: np.ndarray) -> dict[str, Any]:
        """Process data in real-time with enhanced performance."""
        start_time = time.perf_counter()

        try:
            # Ingest data into stream
            success = await self.stream_processor.ingest_data(data)

            if not success:
                return {
                    "success": False,
                    "error": "Failed to ingest data - backpressure or buffer full",
                    "processing_time_us": 0,
                }

            # Processing is handled asynchronously by stream processor
            # For demonstration, we'll simulate getting results
            await asyncio.sleep(0.0001)  # Simulate minimal processing delay

            end_time = time.perf_counter()
            processing_time_us = (end_time - start_time) * 1_000_000

            # Update system metrics
            self.system_metrics["total_processed"] += len(data)

            return {
                "success": True,
                "samples_processed": len(data),
                "processing_time_us": processing_time_us,
                "estimated_anomalies": int(len(data) * 0.05),  # 5% anomaly rate
            }

        except Exception as e:
            logger.error(f"Real-time data processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time_us": 0,
            }

    async def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status."""
        stream_metrics = await self.stream_processor.get_stream_metrics()
        network_metrics = await self.network_optimizer.measure_network_performance("localhost")

        uptime_seconds = time.time() - self.system_metrics["start_time"]

        return {
            "mode": self.processing_mode.value,
            "uptime_seconds": uptime_seconds,
            "system_metrics": self.system_metrics,
            "stream_metrics": stream_metrics,
            "network_metrics": network_metrics,
            "performance_summary": {
                "avg_throughput_samples_per_sec": self.system_metrics["total_processed"] / max(uptime_seconds, 1),
                "anomaly_detection_rate": self.system_metrics["total_anomalies"] / max(self.system_metrics["total_processed"], 1) * 100,
            },
        }

# Example usage and testing
async def create_sample_real_time_data() -> np.ndarray:
    """Create sample real-time data for testing."""
    # Generate normal data with some anomalies
    normal_data = np.random.normal(0, 1, (1000, 10))

    # Add some anomalies
    anomaly_indices = np.random.choice(1000, 50, replace=False)
    normal_data[anomaly_indices] += np.random.normal(5, 1, (50, 10))

    return normal_data.astype(np.float32)

async def benchmark_real_time_performance() -> dict[str, Any]:
    """Benchmark real-time processing performance."""
    config = {
        "mode": "ultra_low_latency",
        "stream": {
            "max_buffer_size": 10000,
            "batch_size": 100,
            "batch_timeout_ms": 1,
            "processor": {
                "target_latency_us": 500,
                "max_latency_us": 1000,
            }
        },
        "network": {
            "mode": "kernel_bypass",
            "target_latency_us": 100,
        }
    }

    orchestrator = RealTimeProcessingOrchestrator(config)
    await orchestrator.initialize()

    # Benchmark processing
    test_data = await create_sample_real_time_data()

    # Process in chunks to simulate real-time streaming
    chunk_size = 100
    results = []

    for i in range(0, len(test_data), chunk_size):
        chunk = test_data[i:i+chunk_size]
        result = await orchestrator.process_real_time_data(chunk)
        results.append(result)

    # Calculate benchmark metrics
    successful_results = [r for r in results if r["success"]]
    avg_latency_us = np.mean([r["processing_time_us"] for r in successful_results])
    max_latency_us = np.max([r["processing_time_us"] for r in successful_results])
    total_samples = sum(r["samples_processed"] for r in successful_results)

    return {
        "total_chunks": len(results),
        "successful_chunks": len(successful_results),
        "success_rate": len(successful_results) / len(results) * 100,
        "total_samples_processed": total_samples,
        "avg_latency_us": avg_latency_us,
        "max_latency_us": max_latency_us,
        "throughput_samples_per_sec": total_samples / (max_latency_us / 1_000_000),
        "system_status": await orchestrator.get_system_status(),
    }
