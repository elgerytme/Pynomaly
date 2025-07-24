"""Batch processing service for large-scale anomaly detection operations."""

import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import structlog

from ..entities.dataset import Dataset, DatasetType, DatasetMetadata
from ..entities.detection_result import DetectionResult
from .detection_service import DetectionService
from ...infrastructure.repositories.model_repository import ModelRepository

logger = structlog.get_logger()


class BatchJobStatus(Enum):
    """Batch job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchJobType(Enum):
    """Batch job type enumeration."""
    DETECTION = "detection"
    TRAINING = "training"
    EVALUATION = "evaluation"
    DATA_PROCESSING = "data_processing"


@dataclass
class BatchJobConfig:
    """Batch job configuration."""
    chunk_size: int = 1000
    max_workers: int = 4
    timeout_seconds: int = 3600
    parallel_processing: bool = True
    save_intermediate: bool = True
    output_format: str = "json"  # json, csv, parquet


@dataclass
class BatchJobProgress:
    """Batch job progress tracking."""
    total_items: int
    processed_items: int
    successful_items: int
    failed_items: int
    current_chunk: int
    total_chunks: int
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    
    @property
    def completion_percentage(self) -> float:
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100


@dataclass
class BatchJob:
    """Batch job information."""
    job_id: str
    job_type: BatchJobType
    status: BatchJobStatus
    config: BatchJobConfig
    progress: BatchJobProgress
    input_paths: List[str]
    output_path: Optional[str]
    model_id: Optional[str]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    results_summary: Optional[Dict[str, Any]] = None


class BatchProcessingService:
    """Service for managing batch processing operations."""
    
    def __init__(self):
        self.jobs: Dict[str, BatchJob] = {}
        self.detection_service = DetectionService()
        self.model_repository = ModelRepository()
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.process_executor = ProcessPoolExecutor(max_workers=4)
        
    async def submit_detection_job(
        self,
        input_paths: List[str],
        output_path: str,
        model_id: str,
        config: BatchJobConfig = None
    ) -> str:
        """Submit a batch detection job."""
        job_id = str(uuid.uuid4())
        config = config or BatchJobConfig()
        
        # Validate inputs
        for path in input_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Input file not found: {path}")
        
        # Verify model exists
        model = self.model_repository.load(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Calculate total items (estimate)
        total_items = await self._estimate_total_items(input_paths)
        
        # Create job
        job = BatchJob(
            job_id=job_id,
            job_type=BatchJobType.DETECTION,
            status=BatchJobStatus.PENDING,
            config=config,
            progress=BatchJobProgress(
                total_items=total_items,
                processed_items=0,
                successful_items=0,
                failed_items=0,
                current_chunk=0,
                total_chunks=(total_items + config.chunk_size - 1) // config.chunk_size,
                start_time=datetime.now()
            ),
            input_paths=input_paths,
            output_path=output_path,
            model_id=model_id,
            created_at=datetime.now()
        )
        
        self.jobs[job_id] = job
        
        # Start processing in background
        asyncio.create_task(self._process_detection_job(job))
        
        logger.info("Batch detection job submitted", 
                   job_id=job_id,
                   input_files=len(input_paths),
                   model_id=model_id,
                   estimated_items=total_items)
        
        return job_id
    
    async def submit_training_job(
        self,
        input_paths: List[str],
        output_path: str,
        algorithm: str,
        training_params: Dict[str, Any],
        config: BatchJobConfig = None
    ) -> str:
        """Submit a batch training job."""
        job_id = str(uuid.uuid4())
        config = config or BatchJobConfig()
        
        # Validate inputs
        for path in input_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Input file not found: {path}")
        
        total_items = await self._estimate_total_items(input_paths)
        
        job = BatchJob(
            job_id=job_id,
            job_type=BatchJobType.TRAINING,
            status=BatchJobStatus.PENDING,
            config=config,
            progress=BatchJobProgress(
                total_items=total_items,
                processed_items=0,
                successful_items=0,
                failed_items=0,
                current_chunk=0,
                total_chunks=1,  # Training is typically single job
                start_time=datetime.now()
            ),
            input_paths=input_paths,
            output_path=output_path,
            model_id=None,
            created_at=datetime.now()
        )
        
        self.jobs[job_id] = job
        
        # Start processing in background
        asyncio.create_task(self._process_training_job(job, algorithm, training_params))
        
        logger.info("Batch training job submitted", 
                   job_id=job_id,
                   algorithm=algorithm,
                   input_files=len(input_paths))
        
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[BatchJob]:
        """Get status of a batch job."""
        return self.jobs.get(job_id)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a batch job."""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status in [BatchJobStatus.COMPLETED, BatchJobStatus.FAILED, BatchJobStatus.CANCELLED]:
            return False
        
        job.status = BatchJobStatus.CANCELLED
        job.completed_at = datetime.now()
        
        logger.info("Batch job cancelled", job_id=job_id)
        
        return True
    
    async def list_jobs(
        self,
        job_type: Optional[BatchJobType] = None,
        status: Optional[BatchJobStatus] = None,
        limit: int = 50
    ) -> List[BatchJob]:
        """List batch jobs with optional filtering."""
        jobs = list(self.jobs.values())
        
        if job_type:
            jobs = [job for job in jobs if job.job_type == job_type]
        
        if status:
            jobs = [job for job in jobs if job.status == status]
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        return jobs[:limit]
    
    async def get_job_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get results of a completed batch job."""
        job = self.jobs.get(job_id)
        if not job or job.status != BatchJobStatus.COMPLETED:
            return None
        
        if job.output_path and Path(job.output_path).exists():
            try:
                with open(job.output_path, 'r') as f:
                    if job.config.output_format == "json":
                        return json.load(f)
                    else:
                        return {"output_file": job.output_path}
            except Exception as e:
                logger.error("Failed to read job results", job_id=job_id, error=str(e))
        
        return job.results_summary
    
    async def _process_detection_job(self, job: BatchJob):
        """Process a batch detection job."""
        try:
            job.status = BatchJobStatus.RUNNING
            job.started_at = datetime.now()
            
            model = self.model_repository.load(job.model_id)
            if not model:
                raise ValueError(f"Model {job.model_id} not found")
            
            all_results = []
            
            for input_path in job.input_paths:
                try:
                    # Process file in chunks
                    file_results = await self._process_file_in_chunks(
                        input_path, model, job
                    )
                    all_results.extend(file_results)
                    
                except Exception as e:
                    logger.error("Failed to process file", 
                               job_id=job.job_id, 
                               file=input_path,
                               error=str(e))
                    job.progress.failed_items += 1
            
            # Save results
            if job.output_path:
                await self._save_batch_results(job.output_path, all_results, job.config.output_format)
            
            # Update job status
            job.status = BatchJobStatus.COMPLETED
            job.completed_at = datetime.now()
            job.results_summary = {
                "total_results": len(all_results),
                "total_anomalies": sum(1 for r in all_results if r.get("is_anomaly", False)),
                "processing_time": (job.completed_at - job.started_at).total_seconds(),
                "files_processed": len(job.input_paths)
            }
            
            logger.info("Batch detection job completed", 
                       job_id=job.job_id,
                       results_count=len(all_results))
            
        except Exception as e:
            job.status = BatchJobStatus.FAILED
            job.completed_at = datetime.now()
            job.error_message = str(e)
            
            logger.error("Batch detection job failed", 
                        job_id=job.job_id, 
                        error=str(e))
    
    async def _process_training_job(self, job: BatchJob, algorithm: str, training_params: Dict[str, Any]):
        """Process a batch training job."""
        try:
            job.status = BatchJobStatus.RUNNING
            job.started_at = datetime.now()
            
            # Load all training data
            all_data = []
            for input_path in job.input_paths:
                file_data = await self._load_file_data(input_path)
                all_data.extend(file_data)
                job.progress.processed_items += len(file_data)
            
            # Create dataset
            dataset = Dataset.from_dict_list(all_data)
            dataset.dataset_type = DatasetType.TRAINING
            
            # Train model
            trained_model = self.detection_service.train(dataset, algorithm, training_params)
            
            # Save model
            model_id = self.model_repository.save(trained_model)
            
            # Update job
            job.status = BatchJobStatus.COMPLETED
            job.completed_at = datetime.now()
            job.model_id = model_id
            job.results_summary = {
                "model_id": model_id,
                "algorithm": algorithm,
                "training_samples": len(all_data),
                "training_time": (job.completed_at - job.started_at).total_seconds()
            }
            
            logger.info("Batch training job completed", 
                       job_id=job.job_id,
                       model_id=model_id,
                       samples=len(all_data))
            
        except Exception as e:
            job.status = BatchJobStatus.FAILED
            job.completed_at = datetime.now()
            job.error_message = str(e)
            
            logger.error("Batch training job failed", 
                        job_id=job.job_id, 
                        error=str(e))
    
    async def _process_file_in_chunks(self, file_path: str, model, job: BatchJob) -> List[Dict[str, Any]]:
        """Process a file in chunks for detection."""
        import pandas as pd
        
        results = []
        chunk_size = job.config.chunk_size
        
        try:
            # Read file
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Process in chunks
            total_rows = len(df)
            for i in range(0, total_rows, chunk_size):
                if job.status == BatchJobStatus.CANCELLED:
                    break
                
                chunk_df = df.iloc[i:i + chunk_size]
                
                try:
                    # Create dataset from chunk
                    chunk_data = chunk_df.to_dict('records')
                    dataset = Dataset.from_dict_list(chunk_data)
                    
                    # Perform detection
                    detection_result = self.detection_service.predict(dataset, model)
                    
                    # Convert results
                    for j, (prediction, score) in enumerate(zip(detection_result.predictions, detection_result.anomaly_scores or [])):
                        results.append({
                            "file": file_path,
                            "row_index": i + j,
                            "prediction": int(prediction),
                            "is_anomaly": prediction == -1,
                            "anomaly_score": float(score) if score is not None else None,
                            "data": chunk_data[j]
                        })
                    
                    # Update progress
                    job.progress.current_chunk += 1
                    job.progress.processed_items += len(chunk_df)
                    job.progress.successful_items += len(chunk_df)
                    
                except Exception as e:
                    logger.error("Failed to process chunk", 
                               job_id=job.job_id,
                               file=file_path,
                               chunk=i,
                               error=str(e))
                    job.progress.failed_items += len(chunk_df)
        
        except Exception as e:
            logger.error("Failed to process file", 
                        job_id=job.job_id,
                        file=file_path,
                        error=str(e))
            raise
        
        return results
    
    async def _load_file_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from a file."""
        import pandas as pd
        
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            return df.to_dict('records')
            
        except Exception as e:
            logger.error("Failed to load file", file=file_path, error=str(e))
            raise
    
    async def _estimate_total_items(self, input_paths: List[str]) -> int:
        """Estimate total number of items to process."""
        total = 0
        
        for path in input_paths:
            try:
                # Quick estimation by counting lines (for CSV) or loading metadata
                if path.endswith('.csv'):
                    with open(path, 'r') as f:
                        total += sum(1 for _ in f) - 1  # Subtract header
                else:
                    # For other formats, use a rough estimate
                    file_size = Path(path).stat().st_size
                    total += max(file_size // 1000, 100)  # Rough estimate
            except Exception:
                # Fallback estimate
                total += 1000
        
        return total
    
    async def _save_batch_results(self, output_path: str, results: List[Dict[str, Any]], format: str):
        """Save batch results to file."""
        import pandas as pd
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == "json":
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            
            elif format == "csv":
                df = pd.DataFrame(results)
                df.to_csv(output_path, index=False)
            
            elif format == "parquet":
                df = pd.DataFrame(results)
                df.to_parquet(output_path, index=False)
            
            else:
                raise ValueError(f"Unsupported output format: {format}")
                
        except Exception as e:
            logger.error("Failed to save batch results", 
                        output_path=str(output_path),
                        format=format,
                        error=str(e))
            raise
    
    async def cleanup_old_jobs(self, days_old: int = 7):
        """Clean up old completed jobs."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        jobs_to_remove = []
        
        for job_id, job in self.jobs.items():
            if (job.status in [BatchJobStatus.COMPLETED, BatchJobStatus.FAILED, BatchJobStatus.CANCELLED] and
                job.completed_at and job.completed_at < cutoff_date):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.jobs[job_id]
            logger.info("Cleaned up old job", job_id=job_id)
        
        return len(jobs_to_remove)
    
    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        total_jobs = len(self.jobs)
        completed_jobs = sum(1 for job in self.jobs.values() if job.status == BatchJobStatus.COMPLETED)
        failed_jobs = sum(1 for job in self.jobs.values() if job.status == BatchJobStatus.FAILED)
        running_jobs = sum(1 for job in self.jobs.values() if job.status == BatchJobStatus.RUNNING)
        
        total_items_processed = sum(job.progress.processed_items for job in self.jobs.values())
        
        return {
            "total_jobs": total_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "running_jobs": running_jobs,
            "success_rate": (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0,
            "total_items_processed": total_items_processed,
            "average_processing_time": sum(
                (job.completed_at - job.started_at).total_seconds() 
                for job in self.jobs.values() 
                if job.completed_at and job.started_at
            ) / max(completed_jobs, 1)
        }
    
    async def shutdown(self):
        """Shutdown the batch processing service."""
        # Cancel all running jobs
        for job in self.jobs.values():
            if job.status == BatchJobStatus.RUNNING:
                job.status = BatchJobStatus.CANCELLED
                job.completed_at = datetime.now()
        
        # Shutdown executors
        self.executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        
        logger.info("Batch processing service shutdown complete")