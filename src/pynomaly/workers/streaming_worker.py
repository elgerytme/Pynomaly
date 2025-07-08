#!/usr/bin/env python3
"""Streaming service worker entry point with Prometheus metrics."""

import argparse
import asyncio
import logging
import os
import sys
from typing import Dict, Any

from pynomaly.infrastructure.monitoring.prometheus_metrics import initialize_metrics
from pynomaly.application.services.streaming_service import StreamingService
from pynomaly.infrastructure.config import create_container


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StreamingWorker:
    """Worker process for streaming services."""
    
    def __init__(self, worker_id: str, metrics_port: int = 9091):
        """Initialize streaming worker.
        
        Args:
            worker_id: Unique worker identifier
            metrics_port: Port for Prometheus metrics server
        """
        self.worker_id = worker_id
        self.metrics_port = metrics_port
        self.streaming_service = None
        self.running = False
        
    async def start(self):
        """Start the streaming worker."""
        logger.info(f"Starting streaming worker {self.worker_id}")
        
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
        
        # Create container and dependencies
        container = create_container()
        
        # Create streaming service (would need proper dependencies)
        # For now, this is a placeholder
        self.streaming_service = StreamingService(
            model_repository=None,  # Would be injected
            streaming_repository=None,  # Would be injected
            event_repository=None,  # Would be injected
            detector_service=None,  # Would be injected
            notification_service=None,  # Would be injected
        )
        
        # Set running flag
        self.running = True
        
        logger.info(f"Streaming worker {self.worker_id} started successfully")
        
        # Worker main loop
        try:
            while self.running:
                # Process streaming tasks
                await self._process_streaming_tasks()
                
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
    
    async def _process_streaming_tasks(self):
        """Process streaming tasks."""
        # This would integrate with the streaming service
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
                active_models=0,  # Would be actual count
                active_streams=1,  # Would be actual count
                memory_usage={"streaming_worker": memory.used},
                cpu_usage={"streaming_worker": cpu_percent / 100.0}
            )
            
        except ImportError:
            # psutil not available, skip metrics
            pass
        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")
    
    async def shutdown(self):
        """Shutdown the worker."""
        logger.info(f"Shutting down streaming worker {self.worker_id}")
        self.running = False
        
        if self.streaming_service:
            # Shutdown streaming service
            if hasattr(self.streaming_service, 'shutdown'):
                await self.streaming_service.shutdown()


def record_streaming_metrics(stream_id: str, samples_processed: int, 
                           throughput: float, buffer_utilization: float,
                           backpressure_events: int = 0):
    """Record streaming metrics to Prometheus."""
    from pynomaly.infrastructure.monitoring.prometheus_metrics import get_metrics_service
    
    metrics_service = get_metrics_service()
    if metrics_service:
        metrics_service.record_streaming_metrics(
            stream_id=stream_id,
            samples_processed=samples_processed,
            throughput=throughput,
            buffer_utilization=buffer_utilization,
            backpressure_events=backpressure_events
        )


async def main():
    """Main entry point for streaming worker."""
    parser = argparse.ArgumentParser(description="Streaming Worker")
    parser.add_argument("--worker-id", default=f"streaming-worker-{os.getpid()}", 
                       help="Worker ID")
    parser.add_argument("--metrics-port", type=int, default=9091,
                       help="Prometheus metrics port")
    
    args = parser.parse_args()
    
    # Create and start worker
    worker = StreamingWorker(
        worker_id=args.worker_id,
        metrics_port=args.metrics_port
    )
    
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
