"""Large-scale batch processing for anomaly detection."""

import asyncio
import json
import logging
import pickle
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from ..monitoring.prometheus_metrics_enhanced import get_metrics_collector, categorize_chunk_size, categorize_severity

try:
    import dask
    import dask.dataframe as dd
    from dask.distributed import Client, as_completed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    MULTIPROCESSING_AVAILABLE = True
except ImportError:
    MULTIPROCESSING_AVAILABLE = False

from ...domain.entities.dataset import Dataset
from ...domain.entities.detection_result import DetectionResult
from ...domain.services.advanced_detection_service import (
    DetectionAlgorithm,
    get_detection_service,
)
from ..monitoring.distributed_tracing import trace_operation

logger = logging.getLogger(__name__)


class BatchEngine(Enum):
    """Available batch processing engines."""
    SEQUENTIAL = "sequential"
    MULTIPROCESSING = "multiprocessing"
    THREADING = "threading"
    DASK = "dask"
    RAY = "ray"


class BatchStatus(Enum):
    """Batch job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class DataFormat(Enum):
    """Supported data formats for batch processing."""
    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"
    PICKLE = "pickle"
    HDF5 = "hdf5"
    FEATHER = "feather"


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    # Processing engine
    engine: BatchEngine = BatchEngine.MULTIPROCESSING
    max_workers: int = 4

    # Data handling
    chunk_size: int = 10000
    max_memory_mb: int = 2048
    enable_caching: bool = True
    temp_dir: str | None = None

    # Detection configuration
    detection_algorithm: DetectionAlgorithm = DetectionAlgorithm.ISOLATION_FOREST
    detection_config: dict[str, Any] | None = None

    # Performance optimization
    enable_parallel_io: bool = True
    enable_compression: bool = True
    optimize_memory: bool = True

    # Error handling and reliability
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    checkpoint_frequency: int = 100  # Save progress every N chunks

    # Output configuration
    output_format: DataFormat = DataFormat.PARQUET
    save_intermediate_results: bool = False

    # Resource limits
    max_execution_time_seconds: int | None = None
    memory_limit_mb: int | None = None
    cpu_limit_percent: float | None = None


@dataclass
class BatchChunk:
    """Individual chunk for batch processing."""

    chunk_id: str
    chunk_index: int
    data: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    source_info: dict[str, Any] | None = None

    def size_mb(self) -> float:
        """Calculate chunk size in MB."""
        return self.data.memory_usage(deep=True).sum() / (1024 * 1024)

    def to_dataset(self, name_prefix: str = "batch_chunk") -> Dataset:
        """Convert chunk to Dataset."""
        return Dataset(
            id=self.chunk_id,
            name=f"{name_prefix}_{self.chunk_index}",
            description=f"Batch processing chunk {self.chunk_index}",
            data=self.data,
            metadata={
                "chunk_index": self.chunk_index,
                "chunk_size": len(self.data),
                "chunk_size_mb": self.size_mb(),
                **self.metadata
            }
        )


@dataclass
class BatchJob:
    """Batch processing job."""

    job_id: str
    name: str
    description: str
    config: BatchConfig
    input_path: str
    output_path: str

    # Job state
    status: BatchStatus = BatchStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Progress tracking
    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0

    # Results
    total_samples: int = 0
    total_anomalies: int = 0
    execution_time: float = 0.0

    # Error tracking
    errors: list[dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0

    # Resource usage
    peak_memory_mb: float = 0.0
    cpu_time_seconds: float = 0.0

    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_chunks == 0:
            return 0.0
        return (self.processed_chunks / self.total_chunks) * 100.0

    def add_error(self, error: Exception, chunk_id: str | None = None) -> None:
        """Add error to job."""
        self.errors.append({
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "chunk_id": chunk_id
        })


class BatchProcessor:
    """Large-scale batch processing for anomaly detection."""

    def __init__(self, config: BatchConfig):
    """Initialize batch processor."""
    self.config = config
    self.detection_service = get_detection_service()
    self.metrics_collector = get_metrics_collector()

    # Job management
    self.jobs: dict[str, BatchJob] = {}
    self.running_jobs: dict[str, asyncio.Task] = {}

    # Processing engine
    self.engine_client = None
    self.executor = None

    # Resource monitoring
    self.memory_monitor_task: asyncio.Task | None = None
    self.monitoring_active = False

    # Temporary directory for intermediate files
    if config.temp_dir:
        self.temp_dir = Path(config.temp_dir)
    else:
        self.temp_dir = Path(tempfile.gettempdir()) / "pynomaly_batch"

    self.temp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Batch processor initialized with engine: {config.engine.value}")

    async def initialize_engine(self) -> None:
        """Initialize the processing engine."""
        if self.config.engine == BatchEngine.DASK:
            if not DASK_AVAILABLE:
                raise ImportError("Dask is required for Dask engine")

            # Initialize Dask client
            self.engine_client = Client(
                n_workers=self.config.max_workers,
                threads_per_worker=2,
                memory_limit=f"{self.config.max_memory_mb // self.config.max_workers}MB"
            )
            logger.info(f"Dask client initialized: {self.engine_client}")

        elif self.config.engine == BatchEngine.RAY:
            if not RAY_AVAILABLE:
                raise ImportError("Ray is required for Ray engine")

            # Initialize Ray
            ray.init(
                num_cpus=self.config.max_workers,
                object_store_memory=self.config.max_memory_mb * 1024 * 1024
            )
            logger.info("Ray initialized")

        elif self.config.engine == BatchEngine.MULTIPROCESSING:
            if not MULTIPROCESSING_AVAILABLE:
                raise ImportError("Multiprocessing is required for multiprocessing engine")

            self.executor = ProcessPoolExecutor(max_workers=self.config.max_workers)
            logger.info(f"ProcessPoolExecutor initialized with {self.config.max_workers} workers")

        elif self.config.engine == BatchEngine.THREADING:
            self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
            logger.info(f"ThreadPoolExecutor initialized with {self.config.max_workers} workers")

    async def shutdown_engine(self) -> None:
        """Shutdown the processing engine."""
        if self.engine_client:
            if self.config.engine == BatchEngine.DASK:
                await self.engine_client.close()
            elif self.config.engine == BatchEngine.RAY:
                ray.shutdown()

        if self.executor:
            self.executor.shutdown(wait=True)

    @trace_operation("batch_processing")
    async def submit_job(
        self,
        name: str,
        description: str,
        input_path: str,
        output_path: str,
        config: BatchConfig | None = None
    ) -> str:
        """Submit a batch processing job."""

        job_config = config or self.config
        job_id = str(uuid.uuid4())

        # Create job
        job = BatchJob(
            job_id=job_id,
            name=name,
            description=description,
            config=job_config,
            input_path=input_path,
            output_path=output_path
        )

        self.jobs[job_id] = job

        # Start job processing
        task = asyncio.create_task(self._process_job(job))
        self.running_jobs[job_id] = task

        logger.info(f"Submitted batch job {job_id}: {name}")
        return job_id

    async def _process_job(self, job: BatchJob) -> None:
        """Process a batch job."""
        try:
            with self.metrics_collector.time_operation("batch", job.config.detection_algorithm.value, job.config.engine.value):
                job.status = BatchStatus.RUNNING
                job.started_at = datetime.now()
                start_time = time.time()

                # Initialize engine if needed
                if not self.engine_client and not self.executor:
                    await self.initialize_engine()

                # Load and chunk data
                chunks = await self._load_and_chunk_data(job)
                job.total_chunks = len(chunks)

                # Process chunks
                results = []
                if job.config.engine == BatchEngine.SEQUENTIAL:
                    results = await self._process_sequential(job, chunks)
                elif job.config.engine in [BatchEngine.MULTIPROCESSING, BatchEngine.THREADING]:
                    results = await self._process_executor(job, chunks)
                elif job.config.engine == BatchEngine.DASK:
                    results = await self._process_dask(job, chunks)
                elif job.config.engine == BatchEngine.RAY:
                    results = await self._process_ray(job, chunks)

                # Combine results
                combined_result = await self._combine_results(job, results)

                # Save final result
                await self._save_result(job, combined_result)

                # Update job status
                job.status = BatchStatus.COMPLETED
                job.completed_at = datetime.now()
                job.execution_time = time.time() - start_time

                logger.info(f"Batch job {job.job_id} completed successfully")

                # Record overall job duration
                self.metrics_collector.record_job_duration(
                    job.execution_time,
                    "batch",
                    job.config.detection_algorithm.value,
                    job.config.engine.value,
                    "completed"
                )

                # Increment total jobs processed
                self.metrics_collector.increment_jobs_total(
                    "batch",
                    "completed",
                    job.config.detection_algorithm.value
                )

                # Increment anomalies found
                severity = categorize_severity(combined_result.anomaly_count, combined_result.total_samples)
                self.metrics_collector.increment_anomalies_found(
                    combined_result.anomaly_count,
                    "batch",
                    job.config.detection_algorithm.value,
                    severity,
                    job.input_path
                )

        except Exception as e:
            job.status = BatchStatus.FAILED
            job.add_error(e)
            logger.error(f"Batch job {job.job_id} failed: {e}")

            # Record error metrics
            self.metrics_collector.increment_error_count(
                "batch",
                type(e).__name__,
                "batch_processor"
            )
            
            # Record failed job
            self.metrics_collector.increment_jobs_total(
                "batch",
                "failed",
                job.config.detection_algorithm.value
            )

            # Retry if configured
            if job.retry_count < job.config.max_retries:
                job.retry_count += 1
                job.status = BatchStatus.RETRYING
                
                # Record retry metrics
                self.metrics_collector.increment_retry_count(
                    "batch",
                    type(e).__name__,
                    job.retry_count
                )
                
                await asyncio.sleep(job.config.retry_delay_seconds)
                await self._process_job(job)  # Recursive retry

        finally:
            # Cleanup
            if job.job_id in self.running_jobs:
                del self.running_jobs[job.job_id]

    async def _load_and_chunk_data(self, job: BatchJob) -> list[BatchChunk]:
        """Load data and split into chunks."""
        input_path = Path(job.input_path)
        chunks = []

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Determine file format
        file_extension = input_path.suffix.lower()

        try:
            if file_extension == '.csv':
                # Read CSV in chunks
                chunk_iter = pd.read_csv(
                    input_path,
                    chunksize=job.config.chunk_size
                )

                for i, chunk_df in enumerate(chunk_iter):
                    chunk = BatchChunk(
                        chunk_id=f"{job.job_id}_chunk_{i}",
                        chunk_index=i,
                        data=chunk_df,
                        source_info={"file": str(input_path), "format": "csv"}
                    )
                    chunks.append(chunk)

            elif file_extension == '.parquet':
                # Read Parquet
                df = pd.read_parquet(input_path)
                chunks = self._split_dataframe_to_chunks(job, df)

            elif file_extension == '.json':
                # Read JSON
                df = pd.read_json(input_path)
                chunks = self._split_dataframe_to_chunks(job, df)

            elif file_extension == '.pkl' or file_extension == '.pickle':
                # Read Pickle
                with open(input_path, 'rb') as f:
                    df = pickle.load(f)
                if isinstance(df, pd.DataFrame):
                    chunks = self._split_dataframe_to_chunks(job, df)
                else:
                    raise ValueError("Pickle file does not contain a DataFrame")

            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            logger.info(f"Loaded data into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _split_dataframe_to_chunks(self, job: BatchJob, df: pd.DataFrame) -> list[BatchChunk]:
        """Split DataFrame into chunks."""
        chunks = []
        chunk_size = job.config.chunk_size

        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i + chunk_size].copy()
            chunk = BatchChunk(
                chunk_id=f"{job.job_id}_chunk_{i // chunk_size}",
                chunk_index=i // chunk_size,
                data=chunk_df,
                source_info={"total_rows": len(df), "chunk_start": i}
            )
            chunks.append(chunk)

        return chunks

    async def _process_sequential(self, job: BatchJob, chunks: list[BatchChunk]) -> list[DetectionResult]:
        """Process chunks sequentially."""
        results = []

        for chunk in chunks:
            try:
                result = await self._process_chunk(chunk, job.config)
                results.append(result)
                job.processed_chunks += 1

                # Update job statistics
                job.total_samples += result.total_samples
                job.total_anomalies += result.anomaly_count

                # Checkpoint if configured
                if job.processed_chunks % job.config.checkpoint_frequency == 0:
                    await self._save_checkpoint(job, results)

            except Exception as e:
                job.failed_chunks += 1
                job.add_error(e, chunk.chunk_id)
                logger.error(f"Error processing chunk {chunk.chunk_id}: {e}")

        return results

    async def _process_executor(self, job: BatchJob, chunks: list[BatchChunk]) -> list[DetectionResult]:
        """Process chunks using ThreadPoolExecutor or ProcessPoolExecutor."""
        results = []
        loop = asyncio.get_event_loop()

        # Submit all chunks
        futures = []
        for chunk in chunks:
            future = loop.run_in_executor(
                self.executor,
                self._process_chunk_sync,
                chunk,
                job.config
            )
            futures.append((chunk, future))

        # Collect results as they complete
        for chunk, future in futures:
            try:
                result = await future
                results.append(result)
                job.processed_chunks += 1

                # Update job statistics
                job.total_samples += result.total_samples
                job.total_anomalies += result.anomaly_count

            except Exception as e:
                job.failed_chunks += 1
                job.add_error(e, chunk.chunk_id)
                logger.error(f"Error processing chunk {chunk.chunk_id}: {e}")

        return results

    async def _process_dask(self, job: BatchJob, chunks: list[BatchChunk]) -> list[DetectionResult]:
        """Process chunks using Dask."""
        if not DASK_AVAILABLE or not self.engine_client:
            raise RuntimeError("Dask is not available")

        results = []

        # Submit chunks to Dask
        futures = []
        for chunk in chunks:
            future = self.engine_client.submit(
                self._process_chunk_sync,
                chunk,
                job.config
            )
            futures.append((chunk, future))

        # Collect results
        for chunk, future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                job.processed_chunks += 1

                # Update job statistics
                job.total_samples += result.total_samples
                job.total_anomalies += result.anomaly_count

            except Exception as e:
                job.failed_chunks += 1
                job.add_error(e, chunk.chunk_id)
                logger.error(f"Error processing chunk {chunk.chunk_id}: {e}")

        return results

    async def _process_ray(self, job: BatchJob, chunks: list[BatchChunk]) -> list[DetectionResult]:
        """Process chunks using Ray."""
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray is not available")

        # Define Ray remote function
        @ray.remote
        def process_chunk_ray(chunk: BatchChunk, config: BatchConfig) -> DetectionResult:
            return self._process_chunk_sync(chunk, config)

        results = []

        # Submit chunks to Ray
        futures = []
        for chunk in chunks:
            future = process_chunk_ray.remote(chunk, job.config)
            futures.append((chunk, future))

        # Collect results
        for chunk, future in futures:
            try:
                result = ray.get(future)
                results.append(result)
                job.processed_chunks += 1

                # Update job statistics
                job.total_samples += result.total_samples
                job.total_anomalies += result.anomaly_count

            except Exception as e:
                job.failed_chunks += 1
                job.add_error(e, chunk.chunk_id)
                logger.error(f"Error processing chunk {chunk.chunk_id}: {e}")

        return results

    def _process_chunk_sync(self, chunk: BatchChunk, config: BatchConfig) -> DetectionResult:
        """Synchronous chunk processing (for use with executors)."""
        # Create event loop if not in async context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self._process_chunk(chunk, config))

    async def _process_chunk(self, chunk: BatchChunk, config: BatchConfig) -> DetectionResult:
        """Process a single chunk for anomaly detection."""
        start_time = time.time()
        chunk_size_category = categorize_chunk_size(len(chunk.data))
        
        try:
            # Convert chunk to dataset
            dataset = chunk.to_dataset()

            # Get detection service (may need to recreate in subprocess)
            detection_service = get_detection_service()

            # Run anomaly detection
            result = await detection_service.detect_anomalies(
                dataset=dataset,
                algorithm=config.detection_algorithm,
                config=config.detection_config
            )

            # Add batch processing metadata
            result.metadata.update({
                "batch_processing": True,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "engine": config.engine.value
            })

            # Record chunk processing time
            duration = time.time() - start_time
            self.metrics_collector.record_chunk_processing_time(
                duration,
                "batch",
                config.detection_algorithm.value,
                chunk_size_category
            )

            return result

        except Exception as e:
            # Record chunk processing error
            self.metrics_collector.increment_error_count(
                "batch",
                type(e).__name__,
                "chunk_processor"
            )
            logger.error(f"Error processing chunk {chunk.chunk_id}: {e}")
            raise

    async def _combine_results(self, job: BatchJob, results: list[DetectionResult]) -> DetectionResult:
        """Combine results from multiple chunks."""
        if not results:
            raise ValueError("No results to combine")

        # Combine anomalies from all chunks
        all_anomalies = []
        total_samples = 0
        total_execution_time = 0.0

        for result in results:
            # Adjust anomaly indices for chunk offset
            chunk_index = result.metadata.get("chunk_index", 0)
            chunk_start = chunk_index * job.config.chunk_size

            for anomaly in result.anomalies:
                # Adjust index to global position
                anomaly.index += chunk_start
                all_anomalies.append(anomaly)

            total_samples += result.total_samples
            total_execution_time += result.execution_time

        # Create combined result
        combined_result = DetectionResult(
            dataset_id=f"batch_job_{job.job_id}",
            algorithm=f"batch_{job.config.detection_algorithm.value}",
            anomalies=all_anomalies,
            total_samples=total_samples,
            anomaly_count=len(all_anomalies),
            contamination_rate=results[0].contamination_rate,  # Use first chunk's rate
            execution_time=total_execution_time,
            metadata={
                "batch_job_id": job.job_id,
                "total_chunks": len(results),
                "processing_engine": job.config.engine.value,
                "combined_at": datetime.now().isoformat()
            }
        )

        return combined_result

    async def _save_result(self, job: BatchJob, result: DetectionResult) -> None:
        """Save the final result."""
        output_path = Path(job.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare result data
        result_data = {
            "job_id": job.job_id,
            "dataset_id": result.dataset_id,
            "algorithm": result.algorithm,
            "total_samples": result.total_samples,
            "anomaly_count": result.anomaly_count,
            "contamination_rate": result.contamination_rate.value,
            "execution_time": result.execution_time,
            "anomalies": [
                {
                    "index": anomaly.index,
                    "score": anomaly.score.value,
                    "confidence_lower": anomaly.confidence.lower,
                    "confidence_upper": anomaly.confidence.upper,
                    "explanation": anomaly.explanation,
                    "features": anomaly.features
                }
                for anomaly in result.anomalies
            ],
            "metadata": result.metadata
        }

        # Save based on output format
        if job.config.output_format == DataFormat.JSON:
            with open(output_path, 'w') as f:
                json.dump(result_data, f, indent=2, default=str)

        elif job.config.output_format == DataFormat.PICKLE:
            with open(output_path, 'wb') as f:
                pickle.dump(result_data, f)

        elif job.config.output_format == DataFormat.PARQUET:
            # Convert to DataFrame and save
            anomalies_df = pd.DataFrame([
                {
                    "index": anomaly.index,
                    "score": anomaly.score.value,
                    "confidence_lower": anomaly.confidence.lower,
                    "confidence_upper": anomaly.confidence.upper,
                    "explanation": anomaly.explanation
                }
                for anomaly in result.anomalies
            ])
            anomalies_df.to_parquet(output_path, index=False)

        else:
            # Default to JSON
            with open(output_path, 'w') as f:
                json.dump(result_data, f, indent=2, default=str)

        logger.info(f"Saved batch result to {output_path}")

    async def _save_checkpoint(self, job: BatchJob, partial_results: list[DetectionResult]) -> None:
        """Save intermediate checkpoint."""
        if job.config.save_intermediate_results:
            checkpoint_path = self.temp_dir / f"checkpoint_{job.job_id}_{job.processed_chunks}.pkl"

            checkpoint_data = {
                "job_id": job.job_id,
                "processed_chunks": job.processed_chunks,
                "timestamp": datetime.now().isoformat(),
                "partial_results": partial_results
            }

            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)

            logger.debug(f"Saved checkpoint for job {job.job_id}")

    async def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get status of a batch job."""
        job = self.jobs.get(job_id)
        if not job:
            return None

        return {
            "job_id": job.job_id,
            "name": job.name,
            "description": job.description,
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "progress_percentage": job.progress_percentage(),
            "total_chunks": job.total_chunks,
            "processed_chunks": job.processed_chunks,
            "failed_chunks": job.failed_chunks,
            "total_samples": job.total_samples,
            "total_anomalies": job.total_anomalies,
            "execution_time": job.execution_time,
            "retry_count": job.retry_count,
            "error_count": len(job.errors),
            "config": {
                "engine": job.config.engine.value,
                "chunk_size": job.config.chunk_size,
                "max_workers": job.config.max_workers,
                "detection_algorithm": job.config.detection_algorithm.value
            }
        }

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running batch job."""
        job = self.jobs.get(job_id)
        if not job:
            return False

        if job.status == BatchStatus.RUNNING:
            job.status = BatchStatus.CANCELLED

            # Cancel the task if running
            if job_id in self.running_jobs:
                task = self.running_jobs[job_id]
                task.cancel()
                del self.running_jobs[job_id]

            logger.info(f"Cancelled batch job {job_id}")
            return True

        return False

    async def list_jobs(self, status: BatchStatus | None = None) -> list[dict[str, Any]]:
        """List all batch jobs."""
        jobs = list(self.jobs.values())

        if status:
            jobs = [job for job in jobs if job.status == status]

        return [
            {
                "job_id": job.job_id,
                "name": job.name,
                "status": job.status.value,
                "created_at": job.created_at.isoformat(),
                "progress_percentage": job.progress_percentage(),
                "total_samples": job.total_samples,
                "total_anomalies": job.total_anomalies
            }
            for job in sorted(jobs, key=lambda j: j.created_at, reverse=True)
        ]

    async def cleanup_completed_jobs(self, older_than_days: int = 7) -> int:
        """Clean up completed jobs older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        cleaned_count = 0

        for job_id, job in list(self.jobs.items()):
            if (job.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED] and
                job.completed_at and job.completed_at < cutoff_date):

                del self.jobs[job_id]
                cleaned_count += 1

        logger.info(f"Cleaned up {cleaned_count} old batch jobs")
        return cleaned_count

    async def get_system_metrics(self) -> dict[str, Any]:
        """Get batch processor system metrics."""
        running_jobs = [job for job in self.jobs.values() if job.status == BatchStatus.RUNNING]

        return {
            "total_jobs": len(self.jobs),
            "running_jobs": len(running_jobs),
            "completed_jobs": len([job for job in self.jobs.values() if job.status == BatchStatus.COMPLETED]),
            "failed_jobs": len([job for job in self.jobs.values() if job.status == BatchStatus.FAILED]),
            "engine": self.config.engine.value,
            "max_workers": self.config.max_workers,
            "temp_dir": str(self.temp_dir),
            "engine_active": self.engine_client is not None or self.executor is not None
        }

    async def shutdown(self) -> None:
        """Shutdown the batch processor."""
        logger.info("Shutting down batch processor...")

        # Cancel running jobs
        for job_id in list(self.running_jobs.keys()):
            await self.cancel_job(job_id)

        # Shutdown engine
        await self.shutdown_engine()

        # Cleanup temp directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

        logger.info("Batch processor shutdown complete")


# Factory function
def create_batch_processor(config: BatchConfig) -> BatchProcessor:
    """Create a batch processor with the given configuration."""
    return BatchProcessor(config)
