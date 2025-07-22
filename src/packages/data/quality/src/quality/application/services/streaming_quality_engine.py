"""Streaming Quality Assessment Engine.

Real-time quality assessment engine for continuous data quality monitoring
with sliding window analysis, distributed processing, and state management.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, AsyncGenerator, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
from queue import Queue, Empty
import statistics
from abc import ABC, abstractmethod

from ...domain.entities.quality_monitoring import (
    StreamingQualityAssessment, QualityWindow, StreamingMetrics,
    QualityMonitoringJob, MonitoringJobStatus, WindowType, StreamId,
    MonitoringJobId, QualityThreshold, QualityAlert, AlertSeverity,
    AlertStatus, ThresholdId, AlertId
)
from ...domain.entities.quality_profile import DatasetId
from .quality_assessment_service import QualityAssessmentService, QualityDimension

logger = logging.getLogger(__name__)


class StreamProcessor(ABC):
    """Abstract base class for stream processors."""
    
    @abstractmethod
    async def process_batch(self, batch: pd.DataFrame) -> StreamingQualityAssessment:
        """Process a batch of data."""
        pass
    
    @abstractmethod
    async def process_window(self, window_data: pd.DataFrame) -> StreamingQualityAssessment:
        """Process a complete window of data."""
        pass


@dataclass
class WindowState:
    """State for a quality window."""
    window_id: str
    window_start: datetime
    window_end: datetime
    buffer: deque = field(default_factory=deque)
    record_count: int = 0
    last_assessment: Optional[StreamingQualityAssessment] = None
    is_complete: bool = False
    
    def add_batch(self, batch: pd.DataFrame) -> None:
        """Add a batch to the window buffer."""
        self.buffer.append(batch)
        self.record_count += len(batch)
    
    def get_combined_data(self) -> pd.DataFrame:
        """Get all data in the window combined."""
        if not self.buffer:
            return pd.DataFrame()
        
        return pd.concat(list(self.buffer), ignore_index=True)
    
    def is_ready_for_assessment(self, current_time: datetime) -> bool:
        """Check if window is ready for assessment."""
        return current_time >= self.window_end or self.is_complete
    
    def cleanup_old_data(self, retention_seconds: int) -> None:
        """Clean up old data from buffer."""
        cutoff_time = datetime.now() - timedelta(seconds=retention_seconds)
        
        # Remove old batches (simplified approach)
        while self.buffer and self.window_start < cutoff_time:
            self.buffer.popleft()


@dataclass
class StreamingEngineConfig:
    """Configuration for streaming quality engine."""
    
    # Processing configuration
    max_batch_size: int = 1000
    batch_timeout_seconds: float = 5.0
    max_concurrent_windows: int = 10
    processing_timeout_seconds: int = 30
    
    # Quality assessment
    enable_statistical_analysis: bool = True
    enable_anomaly_detection: bool = True
    enable_pattern_detection: bool = True
    
    # Performance
    max_memory_mb: int = 2048
    checkpoint_interval_seconds: int = 60
    state_retention_hours: int = 24
    
    # Threading
    thread_pool_size: int = 4
    enable_async_processing: bool = True
    
    # Metrics
    enable_detailed_metrics: bool = True
    metrics_buffer_size: int = 1000


class StreamingQualityEngine:
    """Real-time streaming quality assessment engine."""
    
    def __init__(self, config: StreamingEngineConfig = None):
        """Initialize streaming quality engine."""
        self.config = config or StreamingEngineConfig()
        self.quality_service = QualityAssessmentService()
        
        # State management
        self.active_windows: Dict[str, WindowState] = {}
        self.active_jobs: Dict[MonitoringJobId, QualityMonitoringJob] = {}
        self.stream_processors: Dict[StreamId, StreamProcessor] = {}
        
        # Processing infrastructure
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.thread_pool_size)
        self.processing_queue = Queue()
        self.metrics_buffer = deque(maxlen=self.config.metrics_buffer_size)
        
        # Control flags
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Performance tracking
        self.start_time = datetime.now()
        self.total_records_processed = 0
        self.total_assessments_completed = 0
        self.error_count = 0
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        logger.info("Streaming quality engine initialized")
    
    async def start_monitoring_job(self, job: QualityMonitoringJob) -> QualityMonitoringJob:
        """Start a real-time monitoring job."""
        try:
            # Validate job configuration
            self._validate_job_config(job)
            
            # Initialize job state
            started_job = job.start()
            self.active_jobs[job.job_id] = started_job
            
            # Create stream processor
            processor = DefaultStreamProcessor(
                job.stream_id,
                job.config,
                self.quality_service
            )
            self.stream_processors[job.stream_id] = processor
            
            # Start background processing task
            task = asyncio.create_task(self._process_job_stream(started_job))
            self.background_tasks.append(task)
            
            logger.info(f"Started monitoring job {job.job_id} for stream {job.stream_id}")
            return started_job
            
        except Exception as e:
            logger.error(f"Failed to start monitoring job {job.job_id}: {str(e)}")
            error_job = job.record_error(str(e))
            self.active_jobs[job.job_id] = error_job
            raise
    
    async def stop_monitoring_job(self, job_id: MonitoringJobId) -> Optional[QualityMonitoringJob]:
        """Stop a monitoring job."""
        if job_id not in self.active_jobs:
            logger.warning(f"Job {job_id} not found in active jobs")
            return None
        
        job = self.active_jobs[job_id]
        stopped_job = job.stop()
        self.active_jobs[job_id] = stopped_job
        
        # Cleanup stream processor
        if job.stream_id in self.stream_processors:
            del self.stream_processors[job.stream_id]
        
        # Cancel background task
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        logger.info(f"Stopped monitoring job {job_id}")
        return stopped_job
    
    async def process_data_batch(self, 
                                stream_id: StreamId, 
                                batch: pd.DataFrame) -> Optional[StreamingQualityAssessment]:
        """Process a batch of streaming data."""
        try:
            start_time = time.time()
            
            # Find active job for this stream
            job = self._find_job_by_stream(stream_id)
            if not job or not job.is_running():
                logger.warning(f"No active job found for stream {stream_id}")
                return None
            
            # Get stream processor
            processor = self.stream_processors.get(stream_id)
            if not processor:
                logger.error(f"No processor found for stream {stream_id}")
                return None
            
            # Process batch
            assessment = await processor.process_batch(batch)
            
            # Update job with assessment
            updated_job = job.add_assessment(assessment)
            self.active_jobs[job.job_id] = updated_job
            
            # Update metrics
            processing_time = time.time() - start_time
            await self._update_streaming_metrics(stream_id, batch, processing_time)
            
            # Check thresholds and generate alerts
            await self._check_thresholds(updated_job, assessment)
            
            self.total_records_processed += len(batch)
            self.total_assessments_completed += 1
            
            logger.debug(f"Processed batch of {len(batch)} records for stream {stream_id}")
            return assessment
            
        except Exception as e:
            logger.error(f"Error processing batch for stream {stream_id}: {str(e)}")
            self.error_count += 1
            
            # Update job with error
            if job:
                error_job = job.record_error(str(e))
                self.active_jobs[job.job_id] = error_job
            
            raise
    
    async def process_data_stream(self, 
                                 stream_id: StreamId, 
                                 data_stream: AsyncGenerator[pd.DataFrame, None]) -> None:
        """Process a continuous data stream."""
        try:
            async for batch in data_stream:
                if self.shutdown_event.is_set():
                    break
                
                await self.process_data_batch(stream_id, batch)
                
        except Exception as e:
            logger.error(f"Error processing data stream {stream_id}: {str(e)}")
            raise
    
    async def get_streaming_metrics(self, stream_id: StreamId) -> Optional[StreamingMetrics]:
        """Get current streaming metrics for a stream."""
        job = self._find_job_by_stream(stream_id)
        if job:
            return job.streaming_metrics
        return None
    
    async def get_active_jobs(self) -> List[QualityMonitoringJob]:
        """Get all active monitoring jobs."""
        return [job for job in self.active_jobs.values() if job.is_running()]
    
    async def get_job_status(self, job_id: MonitoringJobId) -> Optional[QualityMonitoringJob]:
        """Get status of a specific monitoring job."""
        return self.active_jobs.get(job_id)
    
    async def get_recent_assessments(self, 
                                   stream_id: StreamId, 
                                   limit: int = 10) -> List[StreamingQualityAssessment]:
        """Get recent quality assessments for a stream."""
        job = self._find_job_by_stream(stream_id)
        if job:
            return job.recent_assessments[-limit:]
        return []
    
    async def get_active_alerts(self, stream_id: StreamId) -> List[QualityAlert]:
        """Get active alerts for a stream."""
        job = self._find_job_by_stream(stream_id)
        if job:
            return [alert for alert in job.active_alerts if alert.is_active()]
        return []
    
    async def acknowledge_alert(self, 
                               alert_id: AlertId, 
                               user: str, 
                               notes: str = "") -> Optional[QualityAlert]:
        """Acknowledge an alert."""
        for job in self.active_jobs.values():
            for alert in job.active_alerts:
                if alert.alert_id == alert_id:
                    acknowledged_alert = alert.acknowledge(user, notes)
                    updated_job = job.update_alert(acknowledged_alert)
                    self.active_jobs[job.job_id] = updated_job
                    return acknowledged_alert
        return None
    
    async def resolve_alert(self, 
                           alert_id: AlertId, 
                           user: str, 
                           action: str, 
                           notes: str = "") -> Optional[QualityAlert]:
        """Resolve an alert."""
        for job in self.active_jobs.values():
            for alert in job.active_alerts:
                if alert.alert_id == alert_id:
                    resolved_alert = alert.resolve(user, action, notes)
                    updated_job = job.update_alert(resolved_alert)
                    self.active_jobs[job.job_id] = updated_job
                    return resolved_alert
        return None
    
    async def pause_monitoring_job(self, job_id: MonitoringJobId) -> Optional[QualityMonitoringJob]:
        """Pause a monitoring job."""
        if job_id not in self.active_jobs:
            return None
        
        job = self.active_jobs[job_id]
        # Create paused job (simplified - would need proper state management)
        paused_job = QualityMonitoringJob(
            job_id=job.job_id,
            stream_id=job.stream_id,
            job_name=job.job_name,
            description=job.description,
            config=job.config,
            status=MonitoringJobStatus.PAUSED,
            created_at=job.created_at,
            started_at=job.started_at,
            stopped_at=job.stopped_at,
            current_window_id=job.current_window_id,
            windows_processed=job.windows_processed,
            total_records_processed=job.total_records_processed,
            streaming_metrics=job.streaming_metrics,
            recent_assessments=job.recent_assessments,
            active_alerts=job.active_alerts,
            error_count=job.error_count,
            last_error=job.last_error,
            last_error_at=job.last_error_at
        )
        
        self.active_jobs[job_id] = paused_job
        return paused_job
    
    async def resume_monitoring_job(self, job_id: MonitoringJobId) -> Optional[QualityMonitoringJob]:
        """Resume a paused monitoring job."""
        if job_id not in self.active_jobs:
            return None
        
        job = self.active_jobs[job_id]
        if job.status != MonitoringJobStatus.PAUSED:
            return job
        
        # Resume job
        resumed_job = QualityMonitoringJob(
            job_id=job.job_id,
            stream_id=job.stream_id,
            job_name=job.job_name,
            description=job.description,
            config=job.config,
            status=MonitoringJobStatus.RUNNING,
            created_at=job.created_at,
            started_at=job.started_at,
            stopped_at=job.stopped_at,
            current_window_id=job.current_window_id,
            windows_processed=job.windows_processed,
            total_records_processed=job.total_records_processed,
            streaming_metrics=job.streaming_metrics,
            recent_assessments=job.recent_assessments,
            active_alerts=job.active_alerts,
            error_count=job.error_count,
            last_error=job.last_error,
            last_error_at=job.last_error_at
        )
        
        self.active_jobs[job_id] = resumed_job
        return resumed_job
    
    async def shutdown(self) -> None:
        """Shutdown the streaming engine."""
        logger.info("Shutting down streaming quality engine")
        
        self.shutdown_event.set()
        
        # Cancel all background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Stop all jobs
        for job_id in list(self.active_jobs.keys()):
            await self.stop_monitoring_job(job_id)
        
        # Cleanup thread pool
        self.thread_pool.shutdown(wait=True)
        
        logger.info("Streaming quality engine shut down completed")
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """Get engine performance statistics."""
        uptime = datetime.now() - self.start_time
        
        return {
            'uptime_seconds': uptime.total_seconds(),
            'total_records_processed': self.total_records_processed,
            'total_assessments_completed': self.total_assessments_completed,
            'error_count': self.error_count,
            'active_jobs': len(self.active_jobs),
            'active_windows': len(self.active_windows),
            'processing_rate': self.total_records_processed / uptime.total_seconds() if uptime.total_seconds() > 0 else 0,
            'success_rate': 1 - (self.error_count / max(1, self.total_assessments_completed)),
            'memory_usage_mb': self._get_memory_usage_mb(),
            'thread_pool_active': self.thread_pool._threads,
            'metrics_buffer_size': len(self.metrics_buffer)
        }
    
    # Private methods
    
    def _validate_job_config(self, job: QualityMonitoringJob) -> None:
        """Validate job configuration."""
        if not job.job_name:
            raise ValueError("Job name is required")
        
        if not job.config.quality_thresholds:
            logger.warning(f"Job {job.job_id} has no quality thresholds configured")
        
        if job.config.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if job.config.max_concurrent_windows <= 0:
            raise ValueError("Max concurrent windows must be positive")
    
    def _find_job_by_stream(self, stream_id: StreamId) -> Optional[QualityMonitoringJob]:
        """Find active job by stream ID."""
        for job in self.active_jobs.values():
            if job.stream_id == stream_id:
                return job
        return None
    
    async def _process_job_stream(self, job: QualityMonitoringJob) -> None:
        """Background task to process a job stream."""
        try:
            while not self.shutdown_event.is_set() and job.is_running():
                # Simulate processing (in real implementation, this would listen to actual stream)
                await asyncio.sleep(1)
                
                # Perform periodic maintenance
                await self._perform_maintenance(job)
                
        except Exception as e:
            logger.error(f"Error in background job processing {job.job_id}: {str(e)}")
            error_job = job.record_error(str(e))
            self.active_jobs[job.job_id] = error_job
    
    async def _update_streaming_metrics(self, 
                                       stream_id: StreamId, 
                                       batch: pd.DataFrame, 
                                       processing_time: float) -> None:
        """Update streaming metrics for a stream."""
        job = self._find_job_by_stream(stream_id)
        if not job:
            return
        
        # Calculate metrics
        throughput = len(batch) / processing_time if processing_time > 0 else 0
        latency_ms = processing_time * 1000
        
        # Update metrics
        new_metrics = StreamingMetrics(
            throughput_records_per_second=throughput,
            latency_ms=latency_ms,
            error_rate=self.error_count / max(1, self.total_assessments_completed),
            memory_usage_mb=self._get_memory_usage_mb(),
            cpu_usage_percent=self._get_cpu_usage_percent(),
            active_windows=len(self.active_windows),
            processed_records=job.total_records_processed,
            failed_records=self.error_count,
            backlog_size=self.processing_queue.qsize()
        )
        
        updated_job = job.update_metrics(new_metrics)
        self.active_jobs[job.job_id] = updated_job
    
    async def _check_thresholds(self, 
                               job: QualityMonitoringJob, 
                               assessment: StreamingQualityAssessment) -> None:
        """Check quality thresholds and generate alerts."""
        if not job.config.enable_alerting:
            return
        
        for threshold in job.config.quality_thresholds:
            # Get metric value from assessment
            metric_value = self._get_metric_value(assessment, threshold.metric_name)
            if metric_value is None:
                continue
            
            # Evaluate threshold
            if threshold.evaluate(metric_value):
                # Check for alert cooldown
                if self._is_alert_in_cooldown(job, threshold):
                    continue
                
                # Create alert
                alert = QualityAlert(
                    alert_id=AlertId(),
                    threshold_id=threshold.threshold_id,
                    monitoring_job_id=job.job_id,
                    stream_id=job.stream_id,
                    severity=threshold.alert_severity,
                    status=AlertStatus.ACTIVE,
                    title=f"Quality threshold violation: {threshold.name}",
                    description=threshold.format_alert_message(metric_value),
                    triggered_at=datetime.now(),
                    metric_name=threshold.metric_name,
                    actual_value=metric_value,
                    threshold_value=threshold.value,
                    affected_records=assessment.records_processed,
                    context_data={
                        'assessment_id': assessment.assessment_id,
                        'window_id': assessment.window_id,
                        'overall_score': assessment.overall_score
                    }
                )
                
                # Add alert to job
                updated_job = job.add_alert(alert)
                self.active_jobs[job.job_id] = updated_job
                
                logger.warning(f"Alert generated for stream {job.stream_id}: {alert.title}")
    
    def _get_metric_value(self, assessment: StreamingQualityAssessment, metric_name: str) -> Optional[float]:
        """Get metric value from assessment."""
        metric_mapping = {
            'overall_score': assessment.overall_score,
            'completeness_score': assessment.completeness_score,
            'accuracy_score': assessment.accuracy_score,
            'consistency_score': assessment.consistency_score,
            'validity_score': assessment.validity_score,
            'uniqueness_score': assessment.uniqueness_score,
            'timeliness_score': assessment.timeliness_score,
            'records_processed': assessment.records_processed,
            'processing_latency_ms': assessment.processing_latency_ms,
            'anomalies_count': len(assessment.anomalies_detected)
        }
        
        return metric_mapping.get(metric_name, assessment.quality_metrics.get(metric_name))
    
    def _is_alert_in_cooldown(self, job: QualityMonitoringJob, threshold: QualityThreshold) -> bool:
        """Check if alert is in cooldown period."""
        cooldown_duration = timedelta(seconds=job.config.alert_cooldown_seconds)
        cutoff_time = datetime.now() - cooldown_duration
        
        # Check recent alerts for this threshold
        for alert in job.active_alerts:
            if (alert.threshold_id == threshold.threshold_id and 
                alert.triggered_at > cutoff_time):
                return True
        
        return False
    
    async def _perform_maintenance(self, job: QualityMonitoringJob) -> None:
        """Perform periodic maintenance tasks."""
        # Cleanup old windows
        self._cleanup_old_windows()
        
        # Cleanup resolved alerts
        self._cleanup_resolved_alerts(job)
        
        # Update metrics buffer
        self._update_metrics_buffer()
    
    def _cleanup_old_windows(self) -> None:
        """Clean up old window states."""
        retention_seconds = self.config.state_retention_hours * 3600
        cutoff_time = datetime.now() - timedelta(seconds=retention_seconds)
        
        windows_to_remove = []
        for window_id, window_state in self.active_windows.items():
            if window_state.window_end < cutoff_time:
                windows_to_remove.append(window_id)
        
        for window_id in windows_to_remove:
            del self.active_windows[window_id]
    
    def _cleanup_resolved_alerts(self, job: QualityMonitoringJob) -> None:
        """Clean up resolved alerts older than retention period."""
        retention_hours = 24
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        
        active_alerts = []
        for alert in job.active_alerts:
            if alert.is_resolved() and alert.resolved_at and alert.resolved_at < cutoff_time:
                continue  # Skip old resolved alerts
            active_alerts.append(alert)
        
        if len(active_alerts) != len(job.active_alerts):
            # Update job with cleaned alerts
            cleaned_job = QualityMonitoringJob(
                job_id=job.job_id,
                stream_id=job.stream_id,
                job_name=job.job_name,
                description=job.description,
                config=job.config,
                status=job.status,
                created_at=job.created_at,
                started_at=job.started_at,
                stopped_at=job.stopped_at,
                current_window_id=job.current_window_id,
                windows_processed=job.windows_processed,
                total_records_processed=job.total_records_processed,
                streaming_metrics=job.streaming_metrics,
                recent_assessments=job.recent_assessments,
                active_alerts=active_alerts,
                error_count=job.error_count,
                last_error=job.last_error,
                last_error_at=job.last_error_at
            )
            
            self.active_jobs[job.job_id] = cleaned_job
    
    def _update_metrics_buffer(self) -> None:
        """Update metrics buffer with current statistics."""
        stats = self.get_engine_statistics()
        self.metrics_buffer.append({
            'timestamp': datetime.now(),
            'stats': stats
        })
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _get_cpu_usage_percent(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return 0.0


class DefaultStreamProcessor(StreamProcessor):
    """Default implementation of stream processor."""
    
    def __init__(self, 
                 stream_id: StreamId, 
                 config: 'MonitoringJobConfig',
                 quality_service: QualityAssessmentService):
        """Initialize default stream processor."""
        self.stream_id = stream_id
        self.config = config
        self.quality_service = quality_service
        
        # Window management
        self.current_window: Optional[WindowState] = None
        self.window_buffer = deque()
        
        logger.info(f"Initialized stream processor for stream {stream_id}")
    
    async def process_batch(self, batch: pd.DataFrame) -> StreamingQualityAssessment:
        """Process a batch of data."""
        start_time = datetime.now()
        
        try:
            # Ensure we have a current window
            if not self.current_window:
                self.current_window = self._create_new_window()
            
            # Add batch to current window
            self.current_window.add_batch(batch)
            
            # Check if window is ready for assessment
            if self.current_window.is_ready_for_assessment(datetime.now()):
                assessment = await self.process_window(self.current_window.get_combined_data())
                self.current_window.last_assessment = assessment
                
                # Create new window if needed
                if self.config.quality_window.window_type == WindowType.TUMBLING:
                    self.current_window = self._create_new_window()
                
                return assessment
            else:
                # Return partial assessment for real-time monitoring
                return await self._create_partial_assessment(batch, start_time)
                
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise
    
    async def process_window(self, window_data: pd.DataFrame) -> StreamingQualityAssessment:
        """Process a complete window of data."""
        start_time = datetime.now()
        
        try:
            # Run quality assessment
            assessment_result = await self._run_quality_assessment(window_data)
            
            # Detect anomalies
            anomalies = await self._detect_anomalies(window_data) if self.config.enable_anomaly_detection else []
            
            # Calculate processing latency
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create streaming assessment
            assessment = StreamingQualityAssessment(
                assessment_id=f"assessment_{int(datetime.now().timestamp())}",
                stream_id=self.stream_id,
                window_id=self.current_window.window_id if self.current_window else "unknown",
                window_start=start_time,
                window_end=datetime.now(),
                records_processed=len(window_data),
                overall_score=assessment_result.get('overall_score', 0.0),
                completeness_score=assessment_result.get('completeness_score', 0.0),
                accuracy_score=assessment_result.get('accuracy_score', 0.0),
                consistency_score=assessment_result.get('consistency_score', 0.0),
                validity_score=assessment_result.get('validity_score', 0.0),
                uniqueness_score=assessment_result.get('uniqueness_score', 0.0),
                timeliness_score=assessment_result.get('timeliness_score', 0.0),
                quality_metrics=assessment_result.get('metrics', {}),
                processing_latency_ms=processing_time,
                anomalies_detected=anomalies
            )
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error processing window: {str(e)}")
            raise
    
    async def _run_quality_assessment(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run quality assessment on data."""
        try:
            # Use the quality assessment service
            # This is a simplified version - in practice, you'd need to adapt the service for streaming
            
            # Calculate basic quality metrics
            total_cells = data.size
            missing_cells = data.isnull().sum().sum()
            completeness_score = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0.0
            
            # Simplified scoring for other dimensions
            accuracy_score = 0.9  # Would be calculated based on validation rules
            consistency_score = 0.85  # Would be calculated based on format consistency
            validity_score = 0.88  # Would be calculated based on business rules
            uniqueness_score = 0.95  # Would be calculated based on duplicate detection
            timeliness_score = 0.92  # Would be calculated based on data freshness
            
            # Overall score (weighted average)
            overall_score = (
                completeness_score * 0.2 +
                accuracy_score * 0.25 +
                consistency_score * 0.2 +
                validity_score * 0.2 +
                uniqueness_score * 0.1 +
                timeliness_score * 0.05
            )
            
            return {
                'overall_score': overall_score,
                'completeness_score': completeness_score,
                'accuracy_score': accuracy_score,
                'consistency_score': consistency_score,
                'validity_score': validity_score,
                'uniqueness_score': uniqueness_score,
                'timeliness_score': timeliness_score,
                'metrics': {
                    'total_records': len(data),
                    'total_columns': len(data.columns),
                    'missing_cells': missing_cells,
                    'completeness_rate': completeness_score
                }
            }
            
        except Exception as e:
            logger.error(f"Error in quality assessment: {str(e)}")
            return {
                'overall_score': 0.0,
                'completeness_score': 0.0,
                'accuracy_score': 0.0,
                'consistency_score': 0.0,
                'validity_score': 0.0,
                'uniqueness_score': 0.0,
                'timeliness_score': 0.0,
                'metrics': {}
            }
    
    async def _detect_anomalies(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect anomalies in the data."""
        anomalies = []
        
        try:
            # Simple anomaly detection based on statistical outliers
            for column in data.select_dtypes(include=[np.number]).columns:
                if len(data[column].dropna()) < 10:  # Need minimum samples
                    continue
                
                # Calculate z-scores
                mean_val = data[column].mean()
                std_val = data[column].std()
                
                if std_val == 0:  # Avoid division by zero
                    continue
                
                z_scores = np.abs((data[column] - mean_val) / std_val)
                outlier_mask = z_scores > 3  # 3-sigma rule
                
                if outlier_mask.any():
                    outlier_count = outlier_mask.sum()
                    anomalies.append({
                        'type': 'statistical_outlier',
                        'column': column,
                        'outlier_count': outlier_count,
                        'outlier_percentage': outlier_count / len(data) * 100,
                        'threshold': 3.0,
                        'severity': 'high' if outlier_count > len(data) * 0.1 else 'medium'
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return []
    
    async def _create_partial_assessment(self, batch: pd.DataFrame, start_time: datetime) -> StreamingQualityAssessment:
        """Create a partial assessment for real-time monitoring."""
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Quick quality check
        total_cells = batch.size
        missing_cells = batch.isnull().sum().sum()
        completeness_score = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0.0
        
        return StreamingQualityAssessment(
            assessment_id=f"partial_{int(datetime.now().timestamp())}",
            stream_id=self.stream_id,
            window_id=self.current_window.window_id if self.current_window else "unknown",
            window_start=start_time,
            window_end=datetime.now(),
            records_processed=len(batch),
            overall_score=completeness_score,  # Simplified for partial assessment
            completeness_score=completeness_score,
            accuracy_score=0.0,  # Not calculated for partial assessment
            consistency_score=0.0,
            validity_score=0.0,
            uniqueness_score=0.0,
            timeliness_score=0.0,
            quality_metrics={
                'total_records': len(batch),
                'missing_cells': missing_cells,
                'completeness_rate': completeness_score
            },
            processing_latency_ms=processing_time,
            anomalies_detected=[]
        )
    
    def _create_new_window(self) -> WindowState:
        """Create a new window state."""
        window_config = self.config.quality_window
        now = datetime.now()
        
        window_start = now
        window_end = now + timedelta(seconds=window_config.duration_seconds)
        
        return WindowState(
            window_id=f"window_{int(now.timestamp())}",
            window_start=window_start,
            window_end=window_end,
            buffer=deque(),
            record_count=0,
            is_complete=False
        )