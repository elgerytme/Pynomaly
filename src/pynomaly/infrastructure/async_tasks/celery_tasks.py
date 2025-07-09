"""
Asynchronous task queue system using Celery and Redis for heavy detection jobs.
Provides scalable background processing for computationally intensive anomaly detection tasks.
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from celery import Celery, Task
from celery.result import AsyncResult
from celery.signals import task_prerun, task_postrun, task_failure
from redis import Redis
from kombu import Queue

logger = logging.getLogger(__name__)

# Celery configuration
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Create Celery app
celery_app = Celery(
    'pynomaly_tasks',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=['pynomaly.infrastructure.async_tasks.celery_tasks']
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
    task_compression='gzip',
    result_compression='gzip',
    task_routes={
        'pynomaly.infrastructure.async_tasks.celery_tasks.heavy_detection_task': {'queue': 'heavy_detection'},
        'pynomaly.infrastructure.async_tasks.celery_tasks.ensemble_detection_task': {'queue': 'ensemble'},
        'pynomaly.infrastructure.async_tasks.celery_tasks.data_preprocessing_task': {'queue': 'preprocessing'},
        'pynomaly.infrastructure.async_tasks.celery_tasks.model_training_task': {'queue': 'training'},
    },
    task_default_queue='default',
    task_queues=(
        Queue('default', routing_key='default'),
        Queue('heavy_detection', routing_key='heavy_detection'),
        Queue('ensemble', routing_key='ensemble'),
        Queue('preprocessing', routing_key='preprocessing'),
        Queue('training', routing_key='training'),
    ),
    worker_send_task_events=True,
    task_send_sent_event=True,
    worker_hijack_root_logger=False,
    worker_log_color=False,
    task_time_limit=3600,  # 1 hour
    task_soft_time_limit=3300,  # 55 minutes
    task_always_eager=False,
    task_eager_propagates=True,
    task_ignore_result=False,
    task_track_started=True,
    task_reject_on_worker_lost=True,
    result_expires=3600,  # 1 hour
    beat_schedule={
        'cleanup-expired-results': {
            'task': 'pynomaly.infrastructure.async_tasks.celery_tasks.cleanup_expired_results',
            'schedule': timedelta(hours=1),
        },
        'monitor-system-health': {
            'task': 'pynomaly.infrastructure.async_tasks.celery_tasks.monitor_system_health',
            'schedule': timedelta(minutes=5),
        },
    },
)


@dataclass
class TaskResult:
    """Result from an asynchronous task."""
    task_id: str
    status: str
    result: Any
    error: Optional[str] = None
    traceback: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseTask(Task):
    """Base task class with common functionality."""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        logger.error(f"Task {task_id} failed: {exc}")
        logger.error(f"Traceback: {einfo}")
        
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success."""
        logger.info(f"Task {task_id} completed successfully")
        
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry."""
        logger.warning(f"Task {task_id} retrying due to: {exc}")


@celery_app.task(bind=True, base=BaseTask)
def heavy_detection_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Heavy anomaly detection task using numpy/pandas optimized operations.
    
    Args:
        data: Dictionary containing dataset and detection parameters
        
    Returns:
        Dictionary with detection results
    """
    try:
        # Update task state to PROGRESS
        self.update_state(
            state='PROGRESS',
            meta={'stage': 'initializing', 'progress': 0}
        )
        
        # Extract parameters
        dataset = np.array(data['dataset'])
        algorithm = data.get('algorithm', 'isolation_forest')
        contamination = data.get('contamination', 0.1)
        
        logger.info(f"Starting heavy detection task with {len(dataset)} samples")
        
        # Simulate heavy computation with progress updates
        stages = ['loading', 'preprocessing', 'training', 'detection', 'postprocessing']
        
        for i, stage in enumerate(stages):
            self.update_state(
                state='PROGRESS',
                meta={'stage': stage, 'progress': (i + 1) / len(stages) * 100}
            )
            
            # Simulate computation time
            time.sleep(2)
            
            if stage == 'preprocessing':
                # Optimize numpy operations
                dataset = _optimize_numpy_preprocessing(dataset)
            elif stage == 'training':
                # Simulate model training
                model = _train_detection_model(dataset, algorithm, contamination)
            elif stage == 'detection':
                # Perform detection
                anomaly_scores = _detect_anomalies(dataset, model)
                
        # Prepare results
        results = {
            'anomaly_scores': anomaly_scores.tolist(),
            'anomalies_detected': int(np.sum(anomaly_scores > 0.5)),
            'total_samples': len(dataset),
            'algorithm': algorithm,
            'contamination': contamination,
            'processing_time': time.time() - self.request.started_at if hasattr(self.request, 'started_at') else None
        }
        
        logger.info(f"Detection completed: {results['anomalies_detected']} anomalies found")
        return results
        
    except Exception as exc:
        logger.error(f"Heavy detection task failed: {exc}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(exc), 'traceback': traceback.format_exc()}
        )
        raise


@celery_app.task(bind=True, base=BaseTask)
def ensemble_detection_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensemble anomaly detection task using multiple algorithms.
    
    Args:
        data: Dictionary containing dataset and ensemble parameters
        
    Returns:
        Dictionary with ensemble detection results
    """
    try:
        dataset = np.array(data['dataset'])
        algorithms = data.get('algorithms', ['isolation_forest', 'one_class_svm', 'local_outlier_factor'])
        
        self.update_state(
            state='PROGRESS',
            meta={'stage': 'initializing', 'progress': 0}
        )
        
        results = {}
        ensemble_scores = np.zeros(len(dataset))
        
        for i, algorithm in enumerate(algorithms):
            self.update_state(
                state='PROGRESS',
                meta={
                    'stage': f'running_{algorithm}',
                    'progress': (i + 1) / len(algorithms) * 100
                }
            )
            
            # Train and run each algorithm
            model = _train_detection_model(dataset, algorithm, data.get('contamination', 0.1))
            scores = _detect_anomalies(dataset, model)
            
            results[algorithm] = {
                'scores': scores.tolist(),
                'anomalies': int(np.sum(scores > 0.5))
            }
            
            # Add to ensemble score
            ensemble_scores += scores
            
        # Calculate final ensemble score
        ensemble_scores /= len(algorithms)
        
        final_results = {
            'individual_results': results,
            'ensemble_scores': ensemble_scores.tolist(),
            'ensemble_anomalies': int(np.sum(ensemble_scores > 0.5)),
            'total_samples': len(dataset),
            'algorithms_used': algorithms
        }
        
        return final_results
        
    except Exception as exc:
        logger.error(f"Ensemble detection task failed: {exc}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(exc), 'traceback': traceback.format_exc()}
        )
        raise


@celery_app.task(bind=True, base=BaseTask)
def data_preprocessing_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Data preprocessing task optimized for pandas operations.
    
    Args:
        data: Dictionary containing raw data and preprocessing parameters
        
    Returns:
        Dictionary with preprocessed data
    """
    try:
        # Convert to pandas DataFrame
        df = pd.DataFrame(data['raw_data'])
        
        self.update_state(
            state='PROGRESS',
            meta={'stage': 'preprocessing', 'progress': 0}
        )
        
        # Optimized pandas operations
        stages = ['cleaning', 'normalization', 'feature_engineering', 'validation']
        
        for i, stage in enumerate(stages):
            self.update_state(
                state='PROGRESS',
                meta={'stage': stage, 'progress': (i + 1) / len(stages) * 100}
            )
            
            if stage == 'cleaning':
                # Remove duplicates and handle missing values
                df = df.drop_duplicates()
                df = df.fillna(df.mean())
                
            elif stage == 'normalization':
                # Normalize numerical columns
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std()
                
            elif stage == 'feature_engineering':
                # Create additional features
                if len(numeric_columns) > 1:
                    df['feature_sum'] = df[numeric_columns].sum(axis=1)
                    df['feature_mean'] = df[numeric_columns].mean(axis=1)
                    
            elif stage == 'validation':
                # Validate data quality
                assert not df.isnull().any().any(), "Data contains null values after preprocessing"
                assert len(df) > 0, "Dataset is empty after preprocessing"
                
        result = {
            'preprocessed_data': df.to_dict('records'),
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'data_types': df.dtypes.to_dict(),
            'preprocessing_summary': {
                'duplicates_removed': len(pd.DataFrame(data['raw_data'])) - len(df),
                'features_created': len(df.columns) - len(pd.DataFrame(data['raw_data']).columns),
                'missing_values_filled': True
            }
        }
        
        return result
        
    except Exception as exc:
        logger.error(f"Data preprocessing task failed: {exc}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(exc), 'traceback': traceback.format_exc()}
        )
        raise


@celery_app.task(bind=True, base=BaseTask)
def model_training_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Model training task for anomaly detection models.
    
    Args:
        data: Dictionary containing training data and model parameters
        
    Returns:
        Dictionary with trained model information
    """
    try:
        dataset = np.array(data['dataset'])
        algorithm = data.get('algorithm', 'isolation_forest')
        hyperparameters = data.get('hyperparameters', {})
        
        self.update_state(
            state='PROGRESS',
            meta={'stage': 'training', 'progress': 50}
        )
        
        # Train model
        model = _train_detection_model(dataset, algorithm, hyperparameters.get('contamination', 0.1))
        
        # Evaluate model
        validation_scores = _detect_anomalies(dataset, model)
        
        result = {
            'model_type': algorithm,
            'training_samples': len(dataset),
            'hyperparameters': hyperparameters,
            'validation_anomalies': int(np.sum(validation_scores > 0.5)),
            'model_id': f"{algorithm}_{int(time.time())}",
            'training_completed': True
        }
        
        return result
        
    except Exception as exc:
        logger.error(f"Model training task failed: {exc}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(exc), 'traceback': traceback.format_exc()}
        )
        raise


@celery_app.task
def cleanup_expired_results():
    """Cleanup expired task results from Redis."""
    try:
        redis_client = Redis.from_url(CELERY_RESULT_BACKEND)
        
        # Get all keys with celery-task-meta prefix
        keys = redis_client.keys('celery-task-meta-*')
        
        expired_count = 0
        for key in keys:
            # Check if key is older than 1 hour
            ttl = redis_client.ttl(key)
            if ttl == -1:  # No TTL set
                redis_client.expire(key, 3600)  # Set 1 hour expiry
            elif ttl < 0:  # Key expired
                redis_client.delete(key)
                expired_count += 1
                
        logger.info(f"Cleaned up {expired_count} expired task results")
        return {'cleaned_up': expired_count}
        
    except Exception as exc:
        logger.error(f"Cleanup task failed: {exc}")
        raise


@celery_app.task
def monitor_system_health():
    """Monitor system health and resource usage."""
    try:
        import psutil
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        # Get Celery worker status
        inspect = celery_app.control.inspect()
        active_tasks = inspect.active() or {}
        
        health_data = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'disk_percent': disk_percent
            },
            'celery': {
                'active_tasks': sum(len(tasks) for tasks in active_tasks.values()),
                'workers': list(active_tasks.keys())
            }
        }
        
        # Log alerts for high resource usage
        if cpu_percent > 80:
            logger.warning(f"High CPU usage detected: {cpu_percent}%")
        if memory_percent > 80:
            logger.warning(f"High memory usage detected: {memory_percent}%")
            
        return health_data
        
    except Exception as exc:
        logger.error(f"System health monitoring failed: {exc}")
        raise


# Helper functions for optimized numpy/pandas operations
def _optimize_numpy_preprocessing(data: np.ndarray) -> np.ndarray:
    """Optimize numpy preprocessing operations."""
    # Use vectorized operations instead of loops
    # Normalize data using broadcasting
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    # Avoid division by zero
    std[std == 0] = 1
    
    # Vectorized normalization
    normalized = (data - mean) / std
    
    return normalized


def _train_detection_model(data: np.ndarray, algorithm: str, contamination: float) -> Dict[str, Any]:
    """Train detection model (simplified implementation)."""
    # This is a simplified mock implementation
    # In practice, you would use actual ML libraries like scikit-learn
    
    model = {
        'algorithm': algorithm,
        'contamination': contamination,
        'training_samples': len(data),
        'features': data.shape[1] if len(data.shape) > 1 else 1,
        'threshold': np.percentile(np.random.random(len(data)), (1 - contamination) * 100)
    }
    
    return model


def _detect_anomalies(data: np.ndarray, model: Dict[str, Any]) -> np.ndarray:
    """Detect anomalies using the trained model."""
    # Simplified anomaly detection
    # Generate random scores for demonstration
    scores = np.random.random(len(data))
    
    # Apply threshold
    threshold = model.get('threshold', 0.5)
    anomaly_scores = (scores > threshold).astype(float)
    
    return anomaly_scores


class TaskManager:
    """Manager for asynchronous tasks."""
    
    def __init__(self):
        self.redis_client = Redis.from_url(CELERY_RESULT_BACKEND)
        
    def submit_heavy_detection(self, dataset: np.ndarray, **kwargs) -> str:
        """Submit heavy detection task."""
        task_data = {
            'dataset': dataset.tolist(),
            **kwargs
        }
        
        result = heavy_detection_task.delay(task_data)
        return result.id
        
    def submit_ensemble_detection(self, dataset: np.ndarray, algorithms: List[str], **kwargs) -> str:
        """Submit ensemble detection task."""
        task_data = {
            'dataset': dataset.tolist(),
            'algorithms': algorithms,
            **kwargs
        }
        
        result = ensemble_detection_task.delay(task_data)
        return result.id
        
    def submit_data_preprocessing(self, raw_data: List[Dict], **kwargs) -> str:
        """Submit data preprocessing task."""
        task_data = {
            'raw_data': raw_data,
            **kwargs
        }
        
        result = data_preprocessing_task.delay(task_data)
        return result.id
        
    def submit_model_training(self, dataset: np.ndarray, algorithm: str, **kwargs) -> str:
        """Submit model training task."""
        task_data = {
            'dataset': dataset.tolist(),
            'algorithm': algorithm,
            **kwargs
        }
        
        result = model_training_task.delay(task_data)
        return result.id
        
    def get_task_status(self, task_id: str) -> TaskResult:
        """Get task status and results."""
        result = AsyncResult(task_id, app=celery_app)
        
        task_result = TaskResult(
            task_id=task_id,
            status=result.status,
            result=result.result if result.ready() else None,
            metadata=result.info if result.info else {}
        )
        
        if result.failed():
            task_result.error = str(result.info.get('error', 'Unknown error'))
            task_result.traceback = result.info.get('traceback')
            
        return task_result
        
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        try:
            celery_app.control.revoke(task_id, terminate=True)
            return True
        except Exception as exc:
            logger.error(f"Failed to cancel task {task_id}: {exc}")
            return False
            
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get list of active tasks."""
        inspect = celery_app.control.inspect()
        active_tasks = inspect.active() or {}
        
        all_tasks = []
        for worker, tasks in active_tasks.items():
            for task in tasks:
                all_tasks.append({
                    'worker': worker,
                    'task_id': task['id'],
                    'name': task['name'],
                    'args': task['args'],
                    'kwargs': task['kwargs']
                })
                
        return all_tasks
        
    def get_task_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get task execution history."""
        # This would typically query a database or persistent storage
        # For now, return empty list
        return []


# Task signals for monitoring
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Handle task pre-run event."""
    logger.info(f"Task {task_id} starting: {task.name}")


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, 
                        retval=None, state=None, **kwds):
    """Handle task post-run event."""
    logger.info(f"Task {task_id} finished: {task.name} (state: {state})")


@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwds):
    """Handle task failure event."""
    logger.error(f"Task {task_id} failed: {exception}")
    logger.error(f"Traceback: {traceback}")


if __name__ == "__main__":
    # Example usage
    task_manager = TaskManager()
    
    # Submit a heavy detection task
    sample_data = np.random.rand(1000, 10)
    task_id = task_manager.submit_heavy_detection(
        dataset=sample_data,
        algorithm='isolation_forest',
        contamination=0.1
    )
    
    print(f"Submitted task: {task_id}")
    
    # Check task status
    status = task_manager.get_task_status(task_id)
    print(f"Task status: {status}")
