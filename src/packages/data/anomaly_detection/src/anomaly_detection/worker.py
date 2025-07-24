"""Comprehensive background worker for anomaly detection tasks."""

import asyncio
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import structlog

from anomaly_detection.domain.services.detection_service import DetectionService
from anomaly_detection.domain.services.ensemble_service import EnsembleService
from anomaly_detection.domain.entities.dataset import Dataset, DatasetType
try:
    from ai.mlops.infrastructure.repositories.model_repository import ModelRepository
except ImportError:
    # Fallback to local copy
    from anomaly_detection.infrastructure.repositories.model_repository import ModelRepository
from anomaly_detection.infrastructure.monitoring import (
    get_metrics_collector, get_performance_monitor, get_monitoring_dashboard
)
from anomaly_detection.infrastructure.logging.error_handler import (
    AnomalyDetectionError, ErrorCategory, ErrorHandler
)

logger = structlog.get_logger()


class JobType(Enum):
    """Types of background jobs."""
    DETECTION = "detection"
    ENSEMBLE = "ensemble"
    BATCH_TRAINING = "batch_training"
    STREAM_MONITORING = "stream_monitoring"
    MODEL_VALIDATION = "model_validation"
    DATA_PREPROCESSING = "data_preprocessing"
    EXPLANATION_GENERATION = "explanation_generation"
    SCHEDULED_ANALYSIS = "scheduled_analysis"


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobPriority(Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class WorkerJob:
    """Background job representation."""
    job_id: str
    job_type: JobType
    priority: JobPriority
    status: JobStatus
    payload: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    result: Optional[Dict[str, Any]] = None
    progress: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary."""
        data = asdict(self)
        # Handle enum serialization
        data['job_type'] = self.job_type.value
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        # Handle datetime serialization
        data['created_at'] = self.created_at.isoformat()
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data


class JobQueue:
    """Simple in-memory job queue with priority support."""
    
    def __init__(self):
        self._jobs: List[WorkerJob] = []
        self._job_history: Dict[str, WorkerJob] = {}
        self._lock = asyncio.Lock()
    
    async def enqueue(self, job: WorkerJob) -> None:
        """Add job to queue with priority ordering."""
        async with self._lock:
            self._jobs.append(job)
            # Sort by priority (highest first) then by creation time
            self._jobs.sort(key=lambda j: (-j.priority.value, j.created_at))
            self._job_history[job.job_id] = job
    
    async def dequeue(self) -> Optional[WorkerJob]:
        """Get next job from queue."""
        async with self._lock:
            if self._jobs:
                job = self._jobs.pop(0)
                job.status = JobStatus.RUNNING
                job.started_at = datetime.utcnow()
                return job
            return None
    
    async def get_job(self, job_id: str) -> Optional[WorkerJob]:
        """Get job by ID."""
        return self._job_history.get(job_id)
    
    async def update_job(self, job: WorkerJob) -> None:
        """Update job in history."""
        async with self._lock:
            self._job_history[job.job_id] = job
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status information."""
        async with self._lock:
            pending_jobs = len(self._jobs)
            total_jobs = len(self._job_history)
            
            status_counts = {}
            for status in JobStatus:
                status_counts[status.value] = sum(
                    1 for job in self._job_history.values() 
                    if job.status == status
                )
            
            return {
                "pending_jobs": pending_jobs,
                "total_jobs": total_jobs,
                "status_counts": status_counts,
                "queue_length": pending_jobs
            }


class AnomalyDetectionWorker:
    """Comprehensive background worker for anomaly detection tasks."""
    
    def __init__(self, 
                 models_dir: Optional[str] = None,
                 max_concurrent_jobs: int = 3,
                 enable_monitoring: bool = True):
        """Initialize the worker."""
        self.logger = logger.bind(component="anomaly_worker")
        self.detection_service = DetectionService()
        self.ensemble_service = EnsembleService()
        self.model_repository = ModelRepository(models_dir or "./models")
        
        self.job_queue = JobQueue()
        self.max_concurrent_jobs = max_concurrent_jobs
        self.running_jobs: Dict[str, asyncio.Task] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        
        # Monitoring integration
        self.enable_monitoring = enable_monitoring
        if enable_monitoring:
            self.metrics_collector = get_metrics_collector()
            self.performance_monitor = get_performance_monitor()
            self.dashboard = get_monitoring_dashboard()
        
        self._shutdown_event = asyncio.Event()
        self.is_running = False
    
    async def submit_job(self, 
                        job_type: JobType,
                        payload: Dict[str, Any],
                        priority: JobPriority = JobPriority.NORMAL,
                        max_retries: int = 3,
                        timeout_seconds: int = 300) -> str:
        """Submit a new job to the worker."""
        job_id = str(uuid.uuid4())
        
        job = WorkerJob(
            job_id=job_id,
            job_type=job_type,
            priority=priority,
            status=JobStatus.PENDING,
            payload=payload,
            created_at=datetime.utcnow(),
            max_retries=max_retries,
            timeout_seconds=timeout_seconds
        )
        
        await self.job_queue.enqueue(job)
        
        self.logger.info(
            "Job submitted",
            job_id=job_id,
            job_type=job_type.value,
            priority=priority.value
        )
        
        if self.enable_monitoring:
            self.metrics_collector.increment_counter(
                "jobs_submitted", 
                tags={"job_type": job_type.value, "priority": priority.value}
            )
        
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and result."""
        job = await self.job_queue.get_job(job_id)
        if job:
            return job.to_dict()
        return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        job = await self.job_queue.get_job(job_id)
        if not job:
            return False
        
        if job.status == JobStatus.PENDING:
            job.status = JobStatus.CANCELLED
            await self.job_queue.update_job(job)
            return True
        elif job.status == JobStatus.RUNNING and job_id in self.running_jobs:
            self.running_jobs[job_id].cancel()
            job.status = JobStatus.CANCELLED
            await self.job_queue.update_job(job)
            return True
        
        return False
    
    async def _execute_job(self, job: WorkerJob) -> None:
        """Execute a single job with monitoring and error handling."""
        start_time = time.time()
        
        try:
            if self.enable_monitoring:
                operation_id = self.metrics_collector.start_operation(
                    f"worker_job_{job.job_type.value}"
                )
            
            # Route job to appropriate handler
            if job.job_type == JobType.DETECTION:
                result = await self._process_detection_job(job)
            elif job.job_type == JobType.ENSEMBLE:
                result = await self._process_ensemble_job(job)
            elif job.job_type == JobType.BATCH_TRAINING:
                result = await self._process_batch_training_job(job)
            elif job.job_type == JobType.STREAM_MONITORING:
                result = await self._process_stream_monitoring_job(job)
            elif job.job_type == JobType.MODEL_VALIDATION:
                result = await self._process_model_validation_job(job)
            elif job.job_type == JobType.DATA_PREPROCESSING:
                result = await self._process_data_preprocessing_job(job)
            elif job.job_type == JobType.EXPLANATION_GENERATION:
                result = await self._process_explanation_job(job)
            elif job.job_type == JobType.SCHEDULED_ANALYSIS:
                result = await self._process_scheduled_analysis_job(job)
            else:
                raise AnomalyDetectionError(
                    f"Unknown job type: {job.job_type}",
                    category=ErrorCategory.VALIDATION
                )
            
            # Job completed successfully
            job.status = JobStatus.COMPLETED
            job.result = result
            job.completed_at = datetime.utcnow()
            job.progress = 100.0
            
            duration_ms = (time.time() - start_time) * 1000
            
            if self.enable_monitoring:
                self.metrics_collector.end_operation(operation_id, success=True)
                self.metrics_collector.record_timing(
                    f"job_duration_{job.job_type.value}",
                    duration_ms,
                    tags={"status": "success"}
                )
            
            self.logger.info(
                "Job completed successfully",
                job_id=job.job_id,
                job_type=job.job_type.value,
                duration_ms=duration_ms
            )
            
        except asyncio.CancelledError:
            job.status = JobStatus.CANCELLED
            self.logger.info("Job cancelled", job_id=job.job_id)
            
        except Exception as e:
            error_msg = handle_error(e, self.logger, {"job_id": job.job_id})
            job.error_message = error_msg
            job.retry_count += 1
            
            if job.retry_count < job.max_retries:
                job.status = JobStatus.RETRYING
                # Re-queue job with exponential backoff
                await asyncio.sleep(2 ** job.retry_count)
                await self.job_queue.enqueue(job)
                
                self.logger.warning(
                    "Job failed, retrying",
                    job_id=job.job_id,
                    retry_count=job.retry_count,
                    max_retries=job.max_retries,
                    error=error_msg
                )
            else:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.utcnow()
                
                self.logger.error(
                    "Job failed permanently",
                    job_id=job.job_id,
                    retry_count=job.retry_count,
                    error=error_msg
                )
            
            if self.enable_monitoring:
                self.metrics_collector.end_operation(operation_id, success=False)
                self.metrics_collector.increment_counter(
                    "job_failures",
                    tags={"job_type": job.job_type.value}
                )
        
        finally:
            await self.job_queue.update_job(job)
            if job.job_id in self.running_jobs:
                del self.running_jobs[job.job_id]
    
    async def _process_detection_job(self, job: WorkerJob) -> Dict[str, Any]:
        """Process anomaly detection job."""
        payload = job.payload
        
        # Load data
        data_source = payload.get("data_source")
        if isinstance(data_source, str):
            # File path
            data_path = Path(data_source)
            if data_path.suffix.lower() == '.csv':
                df = pd.read_csv(data_path)
                data = df.select_dtypes(include=[np.number]).values
            else:
                raise AnomalyDetectionError(
                    f"Unsupported file format: {data_path.suffix}",
                    category=ErrorCategory.VALIDATION
                )
        elif isinstance(data_source, list):
            # Direct data
            data = np.array(data_source, dtype=np.float64)
        else:
            raise AnomalyDetectionError(
                "Invalid data source format",
                category=ErrorCategory.VALIDATION
            )
        
        # Update progress
        job.progress = 30.0
        await self.job_queue.update_job(job)
        
        # Run detection
        algorithm = payload.get("algorithm", "isolation_forest")
        contamination = payload.get("contamination", 0.1)
        parameters = payload.get("parameters", {})
        
        result = self.detection_service.detect_anomalies(
            data=data,
            algorithm=algorithm,
            contamination=contamination,
            **parameters
        )
        
        job.progress = 80.0
        await self.job_queue.update_job(job)
        
        # Store results if requested
        output_path = payload.get("output_path")
        if output_path:
            result_dict = result.to_dict()
            with open(output_path, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
        
        job.progress = 100.0
        await self.job_queue.update_job(job)
        
        return {
            "job_id": job.job_id,
            "algorithm": algorithm,
            "total_samples": result.total_samples,
            "anomalies_detected": result.anomaly_count,
            "anomaly_rate": result.anomaly_rate,
            "confidence_scores": result.confidence_scores.tolist() if result.confidence_scores is not None else None,
            "output_path": output_path
        }
    
    async def _process_ensemble_job(self, job: WorkerJob) -> Dict[str, Any]:
        """Process ensemble detection job."""
        payload = job.payload
        
        # Load data (same as detection job)
        data_source = payload.get("data_source")
        if isinstance(data_source, str):
            data_path = Path(data_source)
            if data_path.suffix.lower() == '.csv':
                df = pd.read_csv(data_path)
                data = df.select_dtypes(include=[np.number]).values
            else:
                raise AnomalyDetectionError(
                    f"Unsupported file format: {data_path.suffix}",
                    category=ErrorCategory.VALIDATION
                )
        elif isinstance(data_source, list):
            data = np.array(data_source, dtype=np.float64)
        else:
            raise AnomalyDetectionError(
                "Invalid data source format",
                category=ErrorCategory.VALIDATION
            )
        
        job.progress = 20.0
        await self.job_queue.update_job(job)
        
        # Run ensemble detection
        algorithms = payload.get("algorithms", ["isolation_forest", "lof"])
        method = payload.get("method", "majority")
        contamination = payload.get("contamination", 0.1)
        
        result = self.ensemble_service.detect_anomalies_ensemble(
            data=data,
            algorithms=algorithms,
            method=method,
            contamination=contamination
        )
        
        job.progress = 90.0
        await self.job_queue.update_job(job)
        
        # Store results if requested
        output_path = payload.get("output_path")
        if output_path:
            result_dict = result.to_dict()
            with open(output_path, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
        
        return {
            "job_id": job.job_id,
            "algorithms": algorithms,
            "method": method,
            "total_samples": result.total_samples,
            "anomalies_detected": result.anomaly_count,
            "anomaly_rate": result.anomaly_rate,
            "output_path": output_path
        }
    
    async def _process_batch_training_job(self, job: WorkerJob) -> Dict[str, Any]:
        """Process batch model training job."""
        payload = job.payload
        
        # Load training data
        data_source = payload.get("data_source")
        data_path = Path(data_source)
        df = pd.read_csv(data_path)
        data = df.select_dtypes(include=[np.number]).values
        
        job.progress = 25.0
        await self.job_queue.update_job(job)
        
        # Train model
        algorithm = payload.get("algorithm", "isolation_forest")
        contamination = payload.get("contamination", 0.1)
        model_name = payload.get("model_name", f"Batch Model {job.job_id[:8]}")
        
        self.detection_service.fit(
            data=data,
            algorithm=algorithm,
            contamination=contamination
        )
        
        job.progress = 75.0
        await self.job_queue.update_job(job)
        
        # Save model
        try:
            from data.processing.domain.entities.model import Model, ModelMetadata, ModelStatus
        except ImportError:
            from anomaly_detection.domain.entities.model import Model, ModelMetadata, ModelStatus
        
        metadata = ModelMetadata(
            model_id=f"batch-{job.job_id}",
            name=model_name,
            algorithm=algorithm,
            status=ModelStatus.TRAINED,
            training_samples=len(data),
            training_features=data.shape[1],
            contamination_rate=contamination,
            description=f"Batch trained model from job {job.job_id}"
        )
        
        trained_model = self.detection_service._fitted_models.get(algorithm)
        model = Model(metadata=metadata, model_object=trained_model)
        
        model_id = self.model_repository.save(model)
        
        return {
            "job_id": job.job_id,
            "model_id": model_id,
            "algorithm": algorithm,
            "training_samples": len(data),
            "training_features": data.shape[1],
            "contamination": contamination
        }
    
    async def _process_stream_monitoring_job(self, job: WorkerJob) -> Dict[str, Any]:
        """Process streaming monitoring job."""
        payload = job.payload
        
        # This would integrate with streaming platforms like Kafka, Pulsar, etc.
        # For demonstration, we'll simulate stream processing
        
        stream_config = payload.get("stream_config", {})
        window_size = stream_config.get("window_size", 100)
        monitoring_duration = payload.get("duration_seconds", 60)
        
        anomalies_detected = 0
        windows_processed = 0
        
        start_time = time.time()
        while time.time() - start_time < monitoring_duration:
            # Simulate receiving streaming data
            await asyncio.sleep(1)  # 1-second windows
            
            # Generate synthetic streaming data
            stream_data = np.random.randn(window_size, 3)
            
            # Add some anomalies
            if np.random.random() < 0.1:  # 10% chance of anomaly in window
                stream_data[-1] = [10, 10, 10]  # Clear anomaly
                anomalies_detected += 1
            
            windows_processed += 1
            
            # Update progress
            progress = min(((time.time() - start_time) / monitoring_duration) * 100, 100)
            job.progress = progress
            await self.job_queue.update_job(job)
        
        return {
            "job_id": job.job_id,
            "windows_processed": windows_processed,
            "anomalies_detected": anomalies_detected,
            "monitoring_duration": monitoring_duration,
            "anomaly_rate": anomalies_detected / windows_processed if windows_processed > 0 else 0
        }
    
    async def _process_model_validation_job(self, job: WorkerJob) -> Dict[str, Any]:
        """Process model validation job."""
        payload = job.payload
        
        model_id = payload.get("model_id")
        validation_data_source = payload.get("validation_data_source")
        
        # Load model
        model = self.model_repository.load(model_id)
        
        # Load validation data
        data_path = Path(validation_data_source)
        df = pd.read_csv(data_path)
        data = df.select_dtypes(include=[np.number]).values
        
        job.progress = 40.0
        await self.job_queue.update_job(job)
        
        # Run validation
        predictions = model.predict(data)
        
        job.progress = 80.0
        await self.job_queue.update_job(job)
        
        # Calculate metrics
        anomaly_count = np.sum(predictions == -1)
        anomaly_rate = anomaly_count / len(predictions)
        
        return {
            "job_id": job.job_id,
            "model_id": model_id,
            "validation_samples": len(data),
            "anomalies_detected": int(anomaly_count),
            "anomaly_rate": float(anomaly_rate),
            "model_performance": "good" if 0.01 <= anomaly_rate <= 0.2 else "needs_review"
        }
    
    async def _process_data_preprocessing_job(self, job: WorkerJob) -> Dict[str, Any]:
        """Process data preprocessing job."""
        payload = job.payload
        
        input_path = Path(payload.get("input_path"))
        output_path = Path(payload.get("output_path"))
        preprocessing_steps = payload.get("steps", ["normalize", "remove_outliers"])
        
        # Load data
        df = pd.read_csv(input_path)
        
        job.progress = 20.0
        await self.job_queue.update_job(job)
        
        # Apply preprocessing steps
        for i, step in enumerate(preprocessing_steps):
            if step == "normalize":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
            elif step == "remove_outliers":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            progress = 20 + (i + 1) / len(preprocessing_steps) * 70
            job.progress = progress
            await self.job_queue.update_job(job)
        
        # Save processed data
        df.to_csv(output_path, index=False)
        
        return {
            "job_id": job.job_id,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "original_samples": payload.get("original_samples", 0),
            "processed_samples": len(df),
            "preprocessing_steps": preprocessing_steps
        }
    
    async def _process_explanation_job(self, job: WorkerJob) -> Dict[str, Any]:
        """Process explanation generation job."""
        payload = job.payload
        
        # This would integrate with explanation libraries like SHAP, LIME, etc.
        # For demonstration, we'll simulate explanation generation
        
        anomaly_indices = payload.get("anomaly_indices", [])
        explanation_method = payload.get("method", "feature_importance")
        
        explanations = []
        for i, idx in enumerate(anomaly_indices):
            # Simulate explanation generation
            await asyncio.sleep(0.1)
            
            explanation = {
                "sample_index": idx,
                "method": explanation_method,
                "feature_importance": np.random.random(5).tolist(),
                "confidence": np.random.uniform(0.7, 0.95)
            }
            explanations.append(explanation)
            
            progress = (i + 1) / len(anomaly_indices) * 100
            job.progress = progress
            await self.job_queue.update_job(job)
        
        return {
            "job_id": job.job_id,
            "explanation_method": explanation_method,
            "explanations_generated": len(explanations),
            "explanations": explanations
        }
    
    async def _process_scheduled_analysis_job(self, job: WorkerJob) -> Dict[str, Any]:
        """Process scheduled analysis job."""
        payload = job.payload
        
        analysis_type = payload.get("analysis_type", "daily_summary")
        data_sources = payload.get("data_sources", [])
        
        results = []
        for i, source in enumerate(data_sources):
            # Simulate analysis
            await asyncio.sleep(0.5)
            
            # Generate mock analysis result
            result = {
                "source": source,
                "analysis_type": analysis_type,
                "anomalies_found": np.random.randint(0, 10),
                "data_quality_score": np.random.uniform(0.8, 1.0),
                "recommendations": ["Monitor feature X", "Review threshold settings"]
            }
            results.append(result)
            
            progress = (i + 1) / len(data_sources) * 100
            job.progress = progress
            await self.job_queue.update_job(job)
        
        return {
            "job_id": job.job_id,
            "analysis_type": analysis_type,
            "sources_analyzed": len(data_sources),
            "total_anomalies": sum(r["anomalies_found"] for r in results),
            "average_quality_score": np.mean([r["data_quality_score"] for r in results]),
            "analysis_results": results
        }
    
    async def start(self) -> None:
        """Start the worker."""
        self.is_running = True
        self.logger.info("Starting anomaly detection worker")
        
        while self.is_running and not self._shutdown_event.is_set():
            try:
                # Check if we can process more jobs
                if len(self.running_jobs) < self.max_concurrent_jobs:
                    job = await self.job_queue.dequeue()
                    
                    if job:
                        # Start job processing
                        task = asyncio.create_task(self._execute_job(job))
                        self.running_jobs[job.job_id] = task
                        
                        self.logger.info(
                            "Started processing job",
                            job_id=job.job_id,
                            job_type=job.job_type.value
                        )
                
                # Clean up completed tasks
                completed_jobs = [
                    job_id for job_id, task in self.running_jobs.items()
                    if task.done()
                ]
                for job_id in completed_jobs:
                    del self.running_jobs[job_id]
                
                # Wait a bit before checking for new jobs
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error("Worker error", error=str(e))
                await asyncio.sleep(5)  # Back off on errors
    
    async def stop(self) -> None:
        """Stop the worker gracefully."""
        self.logger.info("Stopping anomaly detection worker")
        self.is_running = False
        self._shutdown_event.set()
        
        # Cancel running jobs
        for job_id, task in self.running_jobs.items():
            if not task.done():
                task.cancel()
                self.logger.info("Cancelled job", job_id=job_id)
        
        # Wait for jobs to complete/cancel
        if self.running_jobs:
            await asyncio.gather(*self.running_jobs.values(), return_exceptions=True)
        
        self.executor.shutdown(wait=True)
        self.logger.info("Anomaly detection worker stopped")
    
    async def get_worker_status(self) -> Dict[str, Any]:
        """Get comprehensive worker status."""
        queue_status = await self.job_queue.get_queue_status()
        
        return {
            "is_running": self.is_running,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "currently_running_jobs": len(self.running_jobs),
            "running_job_ids": list(self.running_jobs.keys()),
            "queue_status": queue_status,
            "monitoring_enabled": self.enable_monitoring
        }


async def run_worker_demo() -> None:
    """Demo function to show comprehensive worker capabilities."""
    worker = AnomalyDetectionWorker(max_concurrent_jobs=2)
    
    # Start worker in background
    worker_task = asyncio.create_task(worker.start())
    
    try:
        # Submit various types of jobs
        
        # 1. Detection job with synthetic data
        detection_job_id = await worker.submit_job(
            JobType.DETECTION,
            {
                "data_source": np.random.randn(1000, 5).tolist(),
                "algorithm": "isolation_forest",
                "contamination": 0.1,
                "parameters": {"random_state": 42}
            },
            priority=JobPriority.HIGH
        )
        print(f"Submitted detection job: {detection_job_id}")
        
        # 2. Ensemble job
        ensemble_job_id = await worker.submit_job(
            JobType.ENSEMBLE,
            {
                "data_source": np.random.randn(500, 3).tolist(),
                "algorithms": ["isolation_forest", "lof"],
                "method": "majority",
                "contamination": 0.15
            },
            priority=JobPriority.NORMAL
        )
        print(f"Submitted ensemble job: {ensemble_job_id}")
        
        # 3. Explanation job
        explanation_job_id = await worker.submit_job(
            JobType.EXPLANATION_GENERATION,
            {
                "anomaly_indices": [10, 25, 47, 89],
                "method": "shap"
            },
            priority=JobPriority.LOW
        )
        print(f"Submitted explanation job: {explanation_job_id}")
        
        # Monitor job progress
        jobs_to_monitor = [detection_job_id, ensemble_job_id, explanation_job_id]
        
        while True:
            all_completed = True
            
            for job_id in jobs_to_monitor:
                status = await worker.get_job_status(job_id)
                if status:
                    job_status = JobStatus(status['status'])
                    progress = status.get('progress', 0)
                    print(f"Job {job_id[:8]}: {job_status.value} ({progress:.1f}%)")
                    
                    if job_status not in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                        all_completed = False
            
            if all_completed:
                break
            
            await asyncio.sleep(2)
        
        # Print final results
        print("\n=== Final Results ===")
        for job_id in jobs_to_monitor:
            status = await worker.get_job_status(job_id)
            if status and status.get('result'):
                print(f"\nJob {job_id[:8]} result:")
                print(json.dumps(status['result'], indent=2, default=str))
        
        # Show worker status
        worker_status = await worker.get_worker_status()
        print(f"\n=== Worker Status ===")
        print(json.dumps(worker_status, indent=2, default=str))
        
    finally:
        await worker.stop()
        worker_task.cancel()


def main() -> None:
    """Run the worker."""
    logger.info("Starting Anomaly Detection Worker Demo")
    
    try:
        asyncio.run(run_worker_demo())
    except KeyboardInterrupt:
        logger.info("Worker demo interrupted by user")
    except Exception as e:
        logger.error("Worker demo failed", error=str(e))
    finally:
        logger.info("Anomaly Detection Worker Demo completed")


# Export aliases for backward compatibility
Job = WorkerJob


if __name__ == "__main__":
    main()