"""Distributed anomaly detection implementation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

# Import domain entities
from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore

from .data_partitioner import DataPartition, DataPartitioner, PartitionStrategy
from .distributed_config import get_distributed_config_manager
from .task_distributor import DistributedTask, TaskDistributor, TaskPriority, TaskType
from .worker_manager import WorkerManager

logger = logging.getLogger(__name__)


@dataclass
class DetectionChunk:
    """Represents a chunk of data for distributed detection."""

    chunk_id: str
    partition: DataPartition
    detector_config: dict[str, Any]
    algorithm_name: str
    parameters: dict[str, Any] = field(default_factory=dict)

    # Processing metadata
    assigned_worker: str | None = None
    processing_started_at: datetime | None = None
    estimated_duration: float = 0.0


@dataclass
class ChunkResult:
    """Result of processing a detection chunk."""

    chunk_id: str
    worker_id: str

    # Detection results
    anomalies: list[Anomaly]
    scores: list[AnomalyScore]

    # Metadata
    samples_processed: int
    processing_time_seconds: float
    memory_used_mb: float

    # Quality metrics
    detection_confidence: float = 0.0
    model_performance: dict[str, float] = field(default_factory=dict)

    # Error information (if any)
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


class DistributedDetectionResult(BaseModel):
    """Comprehensive result of distributed anomaly detection."""

    # Basic information
    detection_id: str = Field(..., description="Unique detection identifier")
    algorithm_name: str = Field(..., description="Algorithm used for detection")

    # Results
    anomalies: list[Anomaly] = Field(
        default_factory=list, description="Detected anomalies"
    )
    scores: list[AnomalyScore] = Field(
        default_factory=list, description="Anomaly scores"
    )

    # Processing metadata
    total_samples: int = Field(..., description="Total samples processed")
    chunks_processed: int = Field(..., description="Number of chunks processed")
    workers_used: int = Field(..., description="Number of workers used")

    # Timing information
    started_at: datetime = Field(..., description="Detection start time")
    completed_at: datetime = Field(..., description="Detection completion time")
    total_processing_time: float = Field(
        ..., description="Total processing time in seconds"
    )
    parallel_efficiency: float = Field(
        ..., description="Parallel processing efficiency"
    )

    # Resource usage
    total_memory_used_mb: float = Field(
        default=0.0, description="Total memory used in MB"
    )
    peak_memory_usage_mb: float = Field(
        default=0.0, description="Peak memory usage in MB"
    )

    # Quality metrics
    overall_confidence: float = Field(
        default=0.0, description="Overall detection confidence"
    )
    chunk_consistency: float = Field(
        default=0.0, description="Consistency between chunks"
    )

    # Performance metrics
    throughput_samples_per_second: float = Field(
        default=0.0, description="Processing throughput"
    )
    speedup_factor: float = Field(
        default=1.0, description="Speedup vs sequential processing"
    )

    # Error handling
    errors: list[str] = Field(default_factory=list, description="Processing errors")
    warnings: list[str] = Field(default_factory=list, description="Processing warnings")

    # Chunk-level details
    chunk_results: list[ChunkResult] = Field(
        default_factory=list, description="Individual chunk results"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def success_rate(self) -> float:
        """Calculate success rate based on chunk results."""
        if not self.chunk_results:
            return 0.0

        successful_chunks = sum(
            1 for result in self.chunk_results if result.error is None
        )
        return successful_chunks / len(self.chunk_results)

    @property
    def anomaly_rate(self) -> float:
        """Calculate overall anomaly rate."""
        if self.total_samples == 0:
            return 0.0

        return len(self.anomalies) / self.total_samples


class DistributedDetector:
    """Distributed anomaly detection system."""

    def __init__(
        self,
        task_distributor: TaskDistributor | None = None,
        worker_manager: WorkerManager | None = None,
        data_partitioner: DataPartitioner | None = None,
    ):
        """Initialize distributed detector.

        Args:
            task_distributor: Task distribution system
            worker_manager: Worker management system
            data_partitioner: Data partitioning system
        """
        self.config = get_distributed_config_manager().get_effective_config()

        # Core components
        self.task_distributor = task_distributor or TaskDistributor(self.config)
        self.worker_manager = worker_manager or WorkerManager(self.config.worker)
        self.data_partitioner = data_partitioner or DataPartitioner()

        # Detection state
        self.active_detections: dict[str, DistributedDetectionResult] = {}

        # Performance tracking
        self.detection_history: list[DistributedDetectionResult] = []
        self.performance_metrics: dict[str, float] = {}

    async def start(self) -> None:
        """Start the distributed detector."""
        await self.task_distributor.start()
        await self.worker_manager.start()
        logger.info("Distributed detector started")

    async def stop(self) -> None:
        """Stop the distributed detector."""
        await self.task_distributor.stop()
        await self.worker_manager.stop()
        logger.info("Distributed detector stopped")

    async def detect_anomalies(
        self,
        data: pd.DataFrame | np.ndarray,
        detector: Detector,
        num_partitions: int | None = None,
        strategy: PartitionStrategy | None = None,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> DistributedDetectionResult:
        """Perform distributed anomaly detection.

        Args:
            data: Input data for anomaly detection
            detector: Detector configuration
            num_partitions: Number of data partitions
            strategy: Partitioning strategy
            priority: Task priority

        Returns:
            Distributed detection result
        """
        detection_id = f"detection_{int(time.time() * 1000)}"
        started_at = datetime.now(UTC)

        logger.info(
            f"Starting distributed detection {detection_id} with detector {detector.name}"
        )

        try:
            # Create detection result object
            result = DistributedDetectionResult(
                detection_id=detection_id,
                algorithm_name=detector.algorithm,
                started_at=started_at,
                completed_at=started_at,  # Will be updated
                total_processing_time=0.0,
                parallel_efficiency=0.0,
                total_samples=len(data),
            )

            self.active_detections[detection_id] = result

            # Step 1: Partition the data
            partitions = await self._partition_data(data, num_partitions, strategy)
            result.chunks_processed = len(partitions)

            # Step 2: Create detection chunks
            chunks = await self._create_detection_chunks(partitions, detector)

            # Step 3: Distribute and execute detection tasks
            chunk_results = await self._execute_distributed_detection(chunks, priority)

            # Step 4: Aggregate results
            await self._aggregate_detection_results(result, chunk_results)

            # Step 5: Calculate final metrics
            await self._calculate_final_metrics(result)

            # Update completion time
            result.completed_at = datetime.now(UTC)
            result.total_processing_time = (
                result.completed_at - result.started_at
            ).total_seconds()

            # Store in history
            self.detection_history.append(result)

            # Remove from active
            del self.active_detections[detection_id]

            logger.info(
                f"Completed distributed detection {detection_id}: "
                f"{len(result.anomalies)} anomalies found in {result.total_processing_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Error in distributed detection {detection_id}: {e}")

            # Update result with error
            if detection_id in self.active_detections:
                result = self.active_detections[detection_id]
                result.errors.append(str(e))
                result.completed_at = datetime.now(UTC)
                result.total_processing_time = (
                    result.completed_at - result.started_at
                ).total_seconds()

                del self.active_detections[detection_id]
                return result

            raise

    async def _partition_data(
        self,
        data: pd.DataFrame | np.ndarray,
        num_partitions: int | None,
        strategy: PartitionStrategy | None,
    ) -> list[DataPartition]:
        """Partition data for distributed processing."""
        logger.debug("Partitioning data for distributed processing")

        # Use data partitioner to create partitions
        partitions = self.data_partitioner.partition_data(
            data=data, num_partitions=num_partitions, strategy=strategy
        )

        logger.info(f"Created {len(partitions)} data partitions")
        return partitions

    async def _create_detection_chunks(
        self, partitions: list[DataPartition], detector: Detector
    ) -> list[DetectionChunk]:
        """Create detection chunks from data partitions."""
        chunks = []

        for partition in partitions:
            chunk = DetectionChunk(
                chunk_id=f"chunk_{partition.partition_id}",
                partition=partition,
                detector_config=detector.get_params(),
                algorithm_name=detector.algorithm,
                parameters=detector.hyperparameters or {},
                estimated_duration=partition.metadata.estimated_processing_time,
            )
            chunks.append(chunk)

        logger.debug(f"Created {len(chunks)} detection chunks")
        return chunks

    async def _execute_distributed_detection(
        self, chunks: list[DetectionChunk], priority: TaskPriority
    ) -> list[ChunkResult]:
        """Execute detection tasks across distributed workers."""
        logger.debug(f"Executing {len(chunks)} detection tasks")

        # Create distributed tasks
        tasks = []
        for chunk in chunks:
            task = DistributedTask(
                task_type=TaskType.ANOMALY_DETECTION,
                priority=priority,
                function_name="anomaly_detection",
                arguments={
                    "data": chunk.partition.data,
                    "algorithm": chunk.algorithm_name,
                    "parameters": chunk.parameters,
                },
                kwargs={"detector_config": chunk.detector_config},
                context={
                    "chunk_id": chunk.chunk_id,
                    "partition_id": chunk.partition.partition_id,
                },
            )
            tasks.append((chunk, task))

        # Submit tasks to distributor
        task_futures = []
        for chunk, task in tasks:
            task_id = await self.task_distributor.submit_task(task)
            task_futures.append((chunk, task_id))

        # Wait for results
        chunk_results = []
        for chunk, task_id in task_futures:
            try:
                # Wait for task completion
                task_result = await self.task_distributor.get_task_result(
                    task_id,
                    timeout=chunk.estimated_duration * 3,  # 3x estimated time
                )

                if task_result and task_result.is_successful:
                    # Convert task result to chunk result
                    chunk_result = self._convert_task_result_to_chunk_result(
                        chunk, task_result
                    )
                    chunk_results.append(chunk_result)
                else:
                    # Create error result
                    error_msg = task_result.error if task_result else "Task timed out"
                    chunk_result = ChunkResult(
                        chunk_id=chunk.chunk_id,
                        worker_id=task_result.worker_id if task_result else "unknown",
                        anomalies=[],
                        scores=[],
                        samples_processed=0,
                        processing_time_seconds=0.0,
                        memory_used_mb=0.0,
                        error=error_msg,
                    )
                    chunk_results.append(chunk_result)

            except Exception as e:
                logger.error(f"Error processing chunk {chunk.chunk_id}: {e}")

                # Create error result
                chunk_result = ChunkResult(
                    chunk_id=chunk.chunk_id,
                    worker_id="unknown",
                    anomalies=[],
                    scores=[],
                    samples_processed=0,
                    processing_time_seconds=0.0,
                    memory_used_mb=0.0,
                    error=str(e),
                )
                chunk_results.append(chunk_result)

        logger.info(f"Completed {len(chunk_results)} detection chunks")
        return chunk_results

    def _convert_task_result_to_chunk_result(
        self, chunk: DetectionChunk, task_result
    ) -> ChunkResult:
        """Convert task result to chunk result."""
        result_data = task_result.result

        # Extract anomalies and scores from result
        anomalies = []
        scores = []

        if isinstance(result_data, dict):
            # Parse simulated result structure
            anomaly_indices = result_data.get("anomalies_detected", [])
            anomaly_scores = result_data.get("anomaly_scores", [])

            # Create Anomaly objects
            for i, score in enumerate(anomaly_scores):
                if i < len(anomaly_indices):
                    anomaly = Anomaly(
                        index=(
                            anomaly_indices[i]
                            if isinstance(anomaly_indices, list)
                            else i
                        ),
                        score=float(score),
                        features={},  # Would be populated with actual feature values
                    )
                    anomalies.append(anomaly)

                # Create AnomalyScore objects
                anomaly_score = AnomalyScore(
                    value=float(score),
                    confidence=min(1.0, score + 0.1),  # Simple confidence calculation
                    algorithm=chunk.algorithm_name,
                )
                scores.append(anomaly_score)

        return ChunkResult(
            chunk_id=chunk.chunk_id,
            worker_id=task_result.worker_id,
            anomalies=anomalies,
            scores=scores,
            samples_processed=chunk.partition.metadata.total_samples,
            processing_time_seconds=task_result.execution_time_seconds,
            memory_used_mb=task_result.memory_used_mb,
            detection_confidence=0.8,  # Would be calculated based on algorithm performance
        )

    async def _aggregate_detection_results(
        self, result: DistributedDetectionResult, chunk_results: list[ChunkResult]
    ) -> None:
        """Aggregate results from all chunks."""
        logger.debug("Aggregating detection results")

        # Store chunk results
        result.chunk_results = chunk_results

        # Aggregate anomalies and scores
        all_anomalies = []
        all_scores = []
        total_memory = 0.0
        successful_chunks = []

        for chunk_result in chunk_results:
            if chunk_result.error is None:
                all_anomalies.extend(chunk_result.anomalies)
                all_scores.extend(chunk_result.scores)
                total_memory += chunk_result.memory_used_mb
                successful_chunks.append(chunk_result)
            else:
                result.errors.append(
                    f"Chunk {chunk_result.chunk_id}: {chunk_result.error}"
                )

        # Update result
        result.anomalies = all_anomalies
        result.scores = all_scores
        result.total_memory_used_mb = total_memory
        result.peak_memory_usage_mb = max(
            [cr.memory_used_mb for cr in successful_chunks], default=0.0
        )
        result.workers_used = len({cr.worker_id for cr in successful_chunks})

        # Calculate confidence metrics
        if successful_chunks:
            confidences = [cr.detection_confidence for cr in successful_chunks]
            result.overall_confidence = sum(confidences) / len(confidences)

            # Simple consistency measure (would be more sophisticated in practice)
            anomaly_rates = [
                len(cr.anomalies) / max(cr.samples_processed, 1)
                for cr in successful_chunks
            ]
            if len(anomaly_rates) > 1:
                avg_rate = sum(anomaly_rates) / len(anomaly_rates)
                variance = sum((rate - avg_rate) ** 2 for rate in anomaly_rates) / len(
                    anomaly_rates
                )
                result.chunk_consistency = max(0.0, 1.0 - variance)
            else:
                result.chunk_consistency = 1.0

    async def _calculate_final_metrics(
        self, result: DistributedDetectionResult
    ) -> None:
        """Calculate final performance metrics."""
        # Throughput calculation
        if result.total_processing_time > 0:
            result.throughput_samples_per_second = (
                result.total_samples / result.total_processing_time
            )

        # Parallel efficiency calculation
        total_chunk_time = sum(
            cr.processing_time_seconds for cr in result.chunk_results
        )
        if total_chunk_time > 0:
            result.parallel_efficiency = min(
                1.0, total_chunk_time / result.total_processing_time
            )

        # Speedup factor (compared to estimated sequential processing)
        estimated_sequential_time = sum(
            chunk.partition.metadata.estimated_processing_time
            for chunk in list(result.chunk_results)
        )
        if estimated_sequential_time > 0:
            result.speedup_factor = (
                estimated_sequential_time / result.total_processing_time
            )

    def get_detection_status(self, detection_id: str) -> dict[str, Any] | None:
        """Get status of an active detection.

        Args:
            detection_id: Detection identifier

        Returns:
            Detection status or None if not found
        """
        if detection_id not in self.active_detections:
            return None

        result = self.active_detections[detection_id]
        current_time = datetime.now(UTC)

        return {
            "detection_id": detection_id,
            "algorithm": result.algorithm_name,
            "started_at": result.started_at.isoformat(),
            "elapsed_time": (current_time - result.started_at).total_seconds(),
            "total_samples": result.total_samples,
            "chunks_total": result.chunks_processed,
            "chunks_completed": len(result.chunk_results),
            "workers_used": result.workers_used,
            "errors": len(result.errors),
            "progress_percentage": (
                len(result.chunk_results) / max(result.chunks_processed, 1)
            )
            * 100,
        }

    def get_performance_statistics(self) -> dict[str, Any]:
        """Get performance statistics for distributed detection.

        Returns:
            Performance statistics
        """
        if not self.detection_history:
            return {
                "total_detections": 0,
                "average_processing_time": 0.0,
                "average_throughput": 0.0,
                "average_speedup": 1.0,
                "average_efficiency": 0.0,
                "success_rate": 0.0,
            }

        successful_detections = [
            d for d in self.detection_history if d.success_rate > 0.5
        ]

        if not successful_detections:
            return {
                "total_detections": len(self.detection_history),
                "successful_detections": 0,
                "success_rate": 0.0,
            }

        return {
            "total_detections": len(self.detection_history),
            "successful_detections": len(successful_detections),
            "success_rate": len(successful_detections) / len(self.detection_history),
            "average_processing_time": sum(
                d.total_processing_time for d in successful_detections
            )
            / len(successful_detections),
            "average_throughput": sum(
                d.throughput_samples_per_second for d in successful_detections
            )
            / len(successful_detections),
            "average_speedup": sum(d.speedup_factor for d in successful_detections)
            / len(successful_detections),
            "average_efficiency": sum(
                d.parallel_efficiency for d in successful_detections
            )
            / len(successful_detections),
            "average_workers_used": sum(d.workers_used for d in successful_detections)
            / len(successful_detections),
            "total_samples_processed": sum(
                d.total_samples for d in successful_detections
            ),
            "total_anomalies_found": sum(
                len(d.anomalies) for d in successful_detections
            ),
        }
