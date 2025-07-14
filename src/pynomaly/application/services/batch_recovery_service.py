"""Batch Processing Recovery Service.

This module provides comprehensive error handling, recovery mechanisms,
and fault tolerance for batch processing operations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

from pydantic import BaseModel

from ...infrastructure.config.settings import Settings
from ...infrastructure.resilience.retry import retry_async, RetryPolicy
from .batch_processing_service import BatchJob, BatchStatus, BatchCheckpoint
from .batch_monitoring_service import BatchMonitoringService, AlertLevel

logger = logging.getLogger(__name__)


class FailureType(str, Enum):
    """Types of batch processing failures."""
    PROCESSING_ERROR = "processing_error"
    TIMEOUT = "timeout"
    MEMORY_ERROR = "memory_error"
    NETWORK_ERROR = "network_error"
    DATA_CORRUPTION = "data_corruption"
    SYSTEM_ERROR = "system_error"
    USER_CANCELLED = "user_cancelled"
    UNKNOWN = "unknown"


class RecoveryStrategy(str, Enum):
    """Recovery strategies for different failure types."""
    RETRY = "retry"
    RESTART_FROM_CHECKPOINT = "restart_from_checkpoint"
    RESTART_FROM_BEGINNING = "restart_from_beginning"
    SKIP_FAILED_BATCH = "skip_failed_batch"
    MANUAL_INTERVENTION = "manual_intervention"
    ABORT = "abort"


class FailureRecord(BaseModel):
    """Record of a batch processing failure."""
    
    job_id: str
    batch_index: Optional[int] = None
    failure_type: FailureType
    error_message: str
    timestamp: datetime
    stack_trace: Optional[str] = None
    system_state: Dict[str, Any] = {}
    recovery_attempted: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_successful: bool = False
    metadata: Dict[str, Any] = {}


class RecoveryConfig(BaseModel):
    """Configuration for recovery strategies."""
    
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 30.0
    exponential_backoff: bool = True
    max_retry_delay_seconds: float = 300.0
    
    # Checkpoint settings
    enable_checkpointing: bool = True
    checkpoint_interval_batches: int = 10
    checkpoint_storage_path: Optional[str] = None
    
    # Memory management
    enable_memory_monitoring: bool = True
    memory_threshold_percent: float = 85.0
    garbage_collect_on_failure: bool = True
    
    # System state preservation
    preserve_partial_results: bool = True
    backup_job_state: bool = True
    
    # Recovery strategies per failure type
    strategy_mapping: Dict[FailureType, RecoveryStrategy] = {
        FailureType.PROCESSING_ERROR: RecoveryStrategy.RETRY,
        FailureType.TIMEOUT: RecoveryStrategy.RESTART_FROM_CHECKPOINT,
        FailureType.MEMORY_ERROR: RecoveryStrategy.RESTART_FROM_CHECKPOINT,
        FailureType.NETWORK_ERROR: RecoveryStrategy.RETRY,
        FailureType.DATA_CORRUPTION: RecoveryStrategy.SKIP_FAILED_BATCH,
        FailureType.SYSTEM_ERROR: RecoveryStrategy.RESTART_FROM_CHECKPOINT,
        FailureType.USER_CANCELLED: RecoveryStrategy.ABORT,
        FailureType.UNKNOWN: RecoveryStrategy.MANUAL_INTERVENTION
    }


class BatchRecoveryService:
    """Service for handling batch processing failures and recovery."""
    
    def __init__(self,
                 monitoring_service: Optional[BatchMonitoringService] = None,
                 settings: Optional[Settings] = None):
        """Initialize the recovery service.
        
        Args:
            monitoring_service: Monitoring service for alerts
            settings: Application settings
        """
        self.settings = settings or Settings()
        self.monitoring_service = monitoring_service
        self.logger = logging.getLogger(__name__)
        
        # Recovery configuration
        self.config = RecoveryConfig()
        
        # Failure tracking
        self.failure_records: List[FailureRecord] = []
        self.job_retry_counts: Dict[str, int] = {}
        
        # Recovery handlers
        self.recovery_handlers: Dict[RecoveryStrategy, Callable] = {
            RecoveryStrategy.RETRY: self._handle_retry_recovery,
            RecoveryStrategy.RESTART_FROM_CHECKPOINT: self._handle_checkpoint_recovery,
            RecoveryStrategy.RESTART_FROM_BEGINNING: self._handle_restart_recovery,
            RecoveryStrategy.SKIP_FAILED_BATCH: self._handle_skip_batch_recovery,
            RecoveryStrategy.MANUAL_INTERVENTION: self._handle_manual_intervention,
            RecoveryStrategy.ABORT: self._handle_abort_recovery
        }
        
        # State storage
        self.checkpoint_storage_path = Path(
            self.config.checkpoint_storage_path or "./batch_checkpoints"
        )
        self.checkpoint_storage_path.mkdir(parents=True, exist_ok=True)
    
    def configure_recovery(self, config: RecoveryConfig) -> None:
        """Configure recovery settings.
        
        Args:
            config: Recovery configuration
        """
        self.config = config
        self.logger.info("Updated batch recovery configuration")
    
    async def handle_failure(self,
                           job: BatchJob,
                           error: Exception,
                           batch_index: Optional[int] = None,
                           context: Optional[Dict[str, Any]] = None) -> bool:
        """Handle a batch processing failure.
        
        Args:
            job: The failed batch job
            error: The exception that occurred
            batch_index: Index of the failed batch (if applicable)
            context: Additional context about the failure
            
        Returns:
            True if recovery was attempted, False otherwise
        """
        self.logger.error(f"Handling failure for job {job.id}: {error}")
        
        # Classify the failure
        failure_type = self._classify_failure(error, context)
        
        # Create failure record
        failure_record = FailureRecord(
            job_id=job.id,
            batch_index=batch_index,
            failure_type=failure_type,
            error_message=str(error),
            timestamp=datetime.now(timezone.utc),
            stack_trace=self._get_stack_trace(error),
            system_state=await self._capture_system_state(),
            metadata=context or {}
        )
        
        self.failure_records.append(failure_record)
        
        # Send alert
        if self.monitoring_service:
            await self.monitoring_service._create_alert(
                level=AlertLevel.ERROR,
                title=f"Batch Job Failure",
                message=f"Job {job.id} failed: {failure_type.value}",
                job_id=job.id,
                metadata={
                    "failure_type": failure_type.value,
                    "batch_index": batch_index,
                    "error_message": str(error)
                }
            )
        
        # Attempt recovery
        recovery_attempted = await self._attempt_recovery(job, failure_record)
        failure_record.recovery_attempted = recovery_attempted
        
        return recovery_attempted
    
    def _classify_failure(self, 
                         error: Exception, 
                         context: Optional[Dict[str, Any]] = None) -> FailureType:
        """Classify the type of failure based on the error and context."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Memory errors
        if ("memory" in error_str or "out of memory" in error_str or 
            error_type in ["memoryerror", "outofmemoryerror"]):
            return FailureType.MEMORY_ERROR
        
        # Timeout errors
        if ("timeout" in error_str or "timed out" in error_str or
            error_type in ["timeouterror", "asynciotimeouterror"]):
            return FailureType.TIMEOUT
        
        # Network errors
        if ("network" in error_str or "connection" in error_str or
            "socket" in error_str or error_type in ["connectionerror", "networkerror"]):
            return FailureType.NETWORK_ERROR
        
        # Data corruption
        if ("corrupt" in error_str or "invalid data" in error_str or
            "parse error" in error_str or "decode" in error_str):
            return FailureType.DATA_CORRUPTION
        
        # System errors
        if (error_type in ["oserror", "ioerror", "systemexit"] or
            "system" in error_str):
            return FailureType.SYSTEM_ERROR
        
        # User cancellation
        if (error_type in ["keyboardinterrupt", "cancellederror"] or
            "cancel" in error_str):
            return FailureType.USER_CANCELLED
        
        # Processing errors (most common)
        if (error_type in ["valueerror", "typeerror", "attributeerror",
                          "indexerror", "keyerror"] or 
            context and context.get("processing_stage")):
            return FailureType.PROCESSING_ERROR
        
        return FailureType.UNKNOWN
    
    def _get_stack_trace(self, error: Exception) -> Optional[str]:
        """Extract stack trace from exception."""
        import traceback
        try:
            return traceback.format_exc()
        except Exception:
            return None
    
    async def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for debugging."""
        import psutil
        try:
            memory = psutil.virtual_memory()
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "process_count": len(psutil.pids()),
                "disk_usage": psutil.disk_usage('/').percent
            }
        except Exception:
            return {"error": "Failed to capture system state"}
    
    async def _attempt_recovery(self, 
                              job: BatchJob, 
                              failure_record: FailureRecord) -> bool:
        """Attempt to recover from a failure.
        
        Args:
            job: The failed job
            failure_record: Record of the failure
            
        Returns:
            True if recovery was attempted
        """
        # Check retry limits
        retry_count = self.job_retry_counts.get(job.id, 0)
        if retry_count >= self.config.max_retry_attempts:
            self.logger.warning(f"Job {job.id} exceeded maximum retry attempts")
            return False
        
        # Determine recovery strategy
        strategy = self.config.strategy_mapping.get(
            failure_record.failure_type, 
            RecoveryStrategy.MANUAL_INTERVENTION
        )
        
        failure_record.recovery_strategy = strategy
        
        # Execute recovery
        try:
            handler = self.recovery_handlers.get(strategy)
            if handler:
                success = await handler(job, failure_record)
                failure_record.recovery_successful = success
                
                if success:
                    self.logger.info(f"Recovery successful for job {job.id} using {strategy.value}")
                else:
                    self.logger.warning(f"Recovery failed for job {job.id} using {strategy.value}")
                
                return True
            else:
                self.logger.error(f"No handler for recovery strategy: {strategy}")
                return False
                
        except Exception as e:
            self.logger.error(f"Recovery attempt failed for job {job.id}: {e}")
            failure_record.recovery_successful = False
            return False
    
    async def _handle_retry_recovery(self, 
                                   job: BatchJob, 
                                   failure_record: FailureRecord) -> bool:
        """Handle retry recovery strategy."""
        retry_count = self.job_retry_counts.get(job.id, 0)
        self.job_retry_counts[job.id] = retry_count + 1
        
        # Calculate delay with exponential backoff
        delay = self.config.retry_delay_seconds
        if self.config.exponential_backoff:
            delay = min(
                delay * (2 ** retry_count),
                self.config.max_retry_delay_seconds
            )
        
        self.logger.info(f"Retrying job {job.id} in {delay} seconds (attempt {retry_count + 1})")
        
        # Perform garbage collection if enabled
        if self.config.garbage_collect_on_failure:
            import gc
            gc.collect()
        
        # Wait before retry
        await asyncio.sleep(delay)
        
        # Update job status for retry
        job.status = BatchStatus.RETRYING
        job.retry_count = retry_count + 1
        job.last_error = failure_record.error_message
        
        return True
    
    async def _handle_checkpoint_recovery(self, 
                                        job: BatchJob, 
                                        failure_record: FailureRecord) -> bool:
        """Handle recovery from checkpoint."""
        # Find the most recent valid checkpoint
        checkpoint = self._find_latest_checkpoint(job.id)
        if not checkpoint:
            self.logger.warning(f"No checkpoint found for job {job.id}, attempting full restart")
            return await self._handle_restart_recovery(job, failure_record)
        
        self.logger.info(f"Recovering job {job.id} from checkpoint at batch {checkpoint.last_processed_batch}")
        
        try:
            # Restore job state from checkpoint
            await self._restore_from_checkpoint(job, checkpoint)
            
            # Update job status
            job.status = BatchStatus.RUNNING
            job.metrics = checkpoint.metrics_snapshot
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore from checkpoint: {e}")
            return False
    
    async def _handle_restart_recovery(self, 
                                     job: BatchJob, 
                                     failure_record: FailureRecord) -> bool:
        """Handle full restart recovery."""
        self.logger.info(f"Restarting job {job.id} from beginning")
        
        # Reset job state
        job.status = BatchStatus.PENDING
        job.metrics.processed_batches = 0
        job.metrics.processed_items = 0
        job.metrics.failed_batches = 0
        job.metrics.start_time = None
        job.metrics.end_time = None
        job.checkpoints.clear()
        
        # Clear any partial results if configured
        if self.config.preserve_partial_results:
            await self._backup_partial_results(job)
        
        return True
    
    async def _handle_skip_batch_recovery(self, 
                                        job: BatchJob, 
                                        failure_record: FailureRecord) -> bool:
        """Handle recovery by skipping the failed batch."""
        if failure_record.batch_index is None:
            self.logger.error("Cannot skip batch - no batch index provided")
            return False
        
        self.logger.info(f"Skipping failed batch {failure_record.batch_index} for job {job.id}")
        
        # Mark batch as skipped
        job.metrics.skipped_batches += 1
        job.metrics.processed_batches += 1  # Count as processed to continue
        
        # Continue with next batch
        job.status = BatchStatus.RUNNING
        
        return True
    
    async def _handle_manual_intervention(self, 
                                        job: BatchJob, 
                                        failure_record: FailureRecord) -> bool:
        """Handle manual intervention requirement."""
        self.logger.warning(f"Job {job.id} requires manual intervention")
        
        # Pause the job
        job.status = BatchStatus.PAUSED
        
        # Create critical alert
        if self.monitoring_service:
            await self.monitoring_service._create_alert(
                level=AlertLevel.CRITICAL,
                title="Manual Intervention Required",
                message=f"Job {job.id} requires manual intervention due to {failure_record.failure_type.value}",
                job_id=job.id,
                metadata={
                    "failure_record": failure_record.dict(),
                    "requires_action": True
                }
            )
        
        return True
    
    async def _handle_abort_recovery(self, 
                                   job: BatchJob, 
                                   failure_record: FailureRecord) -> bool:
        """Handle job abort."""
        self.logger.info(f"Aborting job {job.id}")
        
        job.status = BatchStatus.CANCELLED
        job.completed_at = datetime.now(timezone.utc)
        job.last_error = failure_record.error_message
        
        return True
    
    async def save_checkpoint(self, job: BatchJob, additional_data: Optional[Dict[str, Any]] = None) -> bool:
        """Save a checkpoint for a job.
        
        Args:
            job: Job to checkpoint
            additional_data: Additional data to save with checkpoint
            
        Returns:
            True if checkpoint was saved successfully
        """
        if not self.config.enable_checkpointing:
            return False
        
        try:
            checkpoint_data = {
                "job_id": job.id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "job_state": {
                    "status": job.status.value,
                    "metrics": job.metrics.dict(),
                    "config": job.config.dict(),
                    "processor_kwargs": job.processor_kwargs
                },
                "additional_data": additional_data or {}
            }
            
            # Save to file
            checkpoint_file = self.checkpoint_storage_path / f"{job.id}_checkpoint.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            
            self.logger.debug(f"Saved checkpoint for job {job.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint for job {job.id}: {e}")
            return False
    
    def _find_latest_checkpoint(self, job_id: str) -> Optional[BatchCheckpoint]:
        """Find the latest checkpoint for a job."""
        try:
            checkpoint_file = self.checkpoint_storage_path / f"{job_id}_checkpoint.json"
            if not checkpoint_file.exists():
                return None
            
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            # Convert back to BatchCheckpoint object
            job_state = data["job_state"]
            metrics = job_state["metrics"]
            
            from .batch_processing_service import BatchMetrics
            metrics_obj = BatchMetrics(**metrics)
            
            checkpoint = BatchCheckpoint(
                job_id=job_id,
                last_processed_batch=metrics_obj.processed_batches,
                processed_items=metrics_obj.processed_items,
                checkpoint_time=datetime.fromisoformat(data["timestamp"]),
                metrics_snapshot=metrics_obj,
                checkpoint_data=data.get("additional_data", {})
            )
            
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint for job {job_id}: {e}")
            return None
    
    async def _restore_from_checkpoint(self, job: BatchJob, checkpoint: BatchCheckpoint) -> None:
        """Restore job state from checkpoint."""
        # Restore metrics
        job.metrics = checkpoint.metrics_snapshot
        
        # Update timestamps
        job.metrics.last_checkpoint = checkpoint.checkpoint_time
        
        self.logger.info(f"Restored job {job.id} from checkpoint at {checkpoint.checkpoint_time}")
    
    async def _backup_partial_results(self, job: BatchJob) -> None:
        """Backup any partial results before restart."""
        try:
            backup_data = {
                "job_id": job.id,
                "backup_time": datetime.now(timezone.utc).isoformat(),
                "partial_metrics": job.metrics.dict(),
                "processed_batches": job.metrics.processed_batches,
                "processed_items": job.metrics.processed_items
            }
            
            backup_file = self.checkpoint_storage_path / f"{job.id}_backup.json"
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            self.logger.info(f"Backed up partial results for job {job.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to backup partial results for job {job.id}: {e}")
    
    def get_failure_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get failure statistics for the specified time period.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Failure statistics
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_failures = [f for f in self.failure_records if f.timestamp >= cutoff]
        
        if not recent_failures:
            return {
                "total_failures": 0,
                "period_hours": hours,
                "failure_types": {},
                "recovery_success_rate": 0.0
            }
        
        # Count by failure type
        failure_types = {}
        recovery_attempts = 0
        recovery_successes = 0
        
        for failure in recent_failures:
            failure_type = failure.failure_type.value
            failure_types[failure_type] = failure_types.get(failure_type, 0) + 1
            
            if failure.recovery_attempted:
                recovery_attempts += 1
                if failure.recovery_successful:
                    recovery_successes += 1
        
        recovery_rate = (recovery_successes / recovery_attempts * 100) if recovery_attempts > 0 else 0
        
        return {
            "total_failures": len(recent_failures),
            "period_hours": hours,
            "failure_types": failure_types,
            "recovery_attempts": recovery_attempts,
            "recovery_successes": recovery_successes,
            "recovery_success_rate": recovery_rate,
            "most_common_failure": max(failure_types.items(), key=lambda x: x[1])[0] if failure_types else None
        }
    
    def get_job_failure_history(self, job_id: str) -> List[FailureRecord]:
        """Get failure history for a specific job.
        
        Args:
            job_id: Job ID
            
        Returns:
            List of failure records for the job
        """
        return [f for f in self.failure_records if f.job_id == job_id]
    
    async def cleanup_old_records(self, days: int = 7) -> int:
        """Clean up old failure records and checkpoints.
        
        Args:
            days: Remove records older than this many days
            
        Returns:
            Number of records cleaned up
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Clean up failure records
        old_count = len(self.failure_records)
        self.failure_records = [f for f in self.failure_records if f.timestamp >= cutoff]
        cleaned_records = old_count - len(self.failure_records)
        
        # Clean up checkpoint files
        cleaned_files = 0
        try:
            for checkpoint_file in self.checkpoint_storage_path.glob("*.json"):
                if checkpoint_file.stat().st_mtime < cutoff.timestamp():
                    checkpoint_file.unlink()
                    cleaned_files += 1
        except Exception as e:
            self.logger.error(f"Failed to clean up checkpoint files: {e}")
        
        total_cleaned = cleaned_records + cleaned_files
        if total_cleaned > 0:
            self.logger.info(f"Cleaned up {cleaned_records} failure records and {cleaned_files} checkpoint files")
        
        return total_cleaned