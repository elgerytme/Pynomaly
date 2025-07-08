#!/usr/bin/env python3
"""Training worker entry point with Prometheus metrics."""

import argparse
import asyncio
import logging
import os
import sys
from typing import Dict, Any

from pynomaly.infrastructure.monitoring.prometheus_metrics import initialize_metrics
from pynomaly.infrastructure.distributed.worker_manager import WorkerTaskExecutor
from pynomaly.infrastructure.config import create_container


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingWorker:
    """Worker process for training tasks."""
    
    def __init__(self, worker_id: str, metrics_port: int = 9090):
        """Initialize training worker.
        
        Args:
            worker_id: Unique worker identifier
            metrics_port: Port for Prometheus metrics server
        """
        self.worker_id = worker_id
        self.metrics_port = metrics_port
        self.executor = None
        self.running = False
        
    async def start(self):
        """Start the training worker."""
        logger.info(f"Starting training worker {self.worker_id}")
        
        # Initialize metrics service
        metrics_service = initialize_metrics(
            enable_default_metrics=True,
            namespace="pynomaly",
            port=self.metrics_port
        )
        
        # Set application info
        metrics_service.set_application_info(
            version="1.0.0",  # Could be loaded from config
            environment=os.getenv("PYNOMALY_ENVIRONMENT", "development"),
            build_time="unknown",
            git_commit="unknown"
        )
        
        # Create task executor
        self.executor = WorkerTaskExecutor(
            worker_id=self.worker_id,
            max_workers=int(os.getenv("WORKER_MAX_TASKS", "4")),
            metrics_port=self.metrics_port
        )
        
        # Set running flag
        self.running = True
        
        logger.info(f"Training worker {self.worker_id} started successfully")
        
        # Worker main loop
        try:
            while self.running:
                # Process any pending tasks
                await self._process_tasks()
                
                # Update system metrics
                await self._update_system_metrics(metrics_service)
                
                # Sleep briefly to prevent CPU spinning
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            logger.error(f"Worker error: {e}")
        finally:
            await self.shutdown()
    
    async def _process_tasks(self):
        """Process pending tasks."""
        # This would integrate with the task queue system
        # For now, just a placeholder
        pass
    
    async def _update_system_metrics(self, metrics_service):
        """Update system metrics."""
        try:
            import psutil
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            # Update metrics
            metrics_service.update_system_metrics(
                active_models=1,  # Would be actual count
                active_streams=0,
                memory_usage={"worker": memory.used},
                cpu_usage={"worker": cpu_percent / 100.0}
            )
            
        except ImportError:
            # psutil not available, skip metrics
            pass
        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")
    
    async def shutdown(self):
        """Shutdown the worker."""
        logger.info(f"Shutting down training worker {self.worker_id}")
        self.running = False
        
        if self.executor:
            # Shutdown executor
            if hasattr(self.executor, 'shutdown'):
                await self.executor.shutdown()


def record_training_metrics(algorithm: str, dataset_size: int, duration: float, 
                          success: bool, model_size_bytes: int = 0):
    """Record training metrics to Prometheus."""
    from pynomaly.infrastructure.monitoring.prometheus_metrics import get_metrics_service
    
    metrics_service = get_metrics_service()
    if metrics_service:
        metrics_service.record_training(
            algorithm=algorithm,
            dataset_size=dataset_size,
            duration=duration,
            model_size_bytes=model_size_bytes,
            success=success
        )


def record_detection_metrics(algorithm: str, dataset_type: str, dataset_size: int,
                           duration: float, anomalies_found: int, success: bool,
                           accuracy: float = None):
    """Record detection metrics to Prometheus."""
    from pynomaly.infrastructure.monitoring.prometheus_metrics import get_metrics_service
    
    metrics_service = get_metrics_service()
    if metrics_service:
        metrics_service.record_detection(
            algorithm=algorithm,
            dataset_type=dataset_type,
            dataset_size=dataset_size,
            duration=duration,
            anomalies_found=anomalies_found,
            success=success,
            accuracy=accuracy
        )


async def main():
    """Main entry point for training worker."""
    parser = argparse.ArgumentParser(description="Training Worker")
    parser.add_argument("--worker-id", default=f"training-worker-{os.getpid()}", 
                       help="Worker ID")
    parser.add_argument("--metrics-port", type=int, default=9090,
                       help="Prometheus metrics port")
    
    args = parser.parse_args()
    
    # Create and start worker
    worker = TrainingWorker(
        worker_id=args.worker_id,
        metrics_port=args.metrics_port
    )
    
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
