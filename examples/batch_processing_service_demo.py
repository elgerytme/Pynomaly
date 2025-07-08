#!/usr/bin/env python3
"""Demo script for BatchProcessingService usage."""

import asyncio
import logging
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

from pynomaly.application.services import (
    BatchProcessingService,
    JobSubmissionOptions,
    ProcessingHooks,
    create_batch_processing_service
)
from pynomaly.infrastructure.batch.batch_processor import BatchConfig, BatchEngine
from pynomaly.domain.services.advanced_detection_service import DetectionAlgorithm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data(size: int = 10000) -> pd.DataFrame:
    """Create sample data for demonstration."""
    np.random.seed(42)
    
    # Generate normal data
    normal_data = np.random.randn(size - 100, 3)
    
    # Generate anomalies (outliers)
    anomalies = np.random.randn(100, 3) * 5  # More spread out
    
    # Combine
    data = np.vstack([normal_data, anomalies])
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3'])
    
    return df


def create_processing_hooks() -> ProcessingHooks:
    """Create processing hooks for demonstration."""
    
    def pre_processing_hook(job_data):
        logger.info(f"Pre-processing hook called for job: {job_data['name']}")
        logger.info(f"Input file: {job_data['input_path']}")
        logger.info(f"Output file: {job_data['output_path']}")
        
        # You can modify job_data here
        job_data['processed_at'] = 'pre-processing'
        return job_data
    
    def post_processing_hook(job_status):
        logger.info(f"Post-processing hook called for job: {job_status['job_id']}")
        logger.info(f"Job status: {job_status['status']}")
        logger.info(f"Total samples: {job_status.get('total_samples', 0)}")
        logger.info(f"Total anomalies: {job_status.get('total_anomalies', 0)}")
        
        # You can modify job_status here
        job_status['post_processed'] = True
        return job_status
    
    def on_job_submitted(job_id, job_data):
        logger.info(f"✅ Job submitted: {job_id}")
        logger.info(f"   Name: {job_data['name']}")
    
    def on_job_started(job_id, status):
        logger.info(f"🚀 Job started: {job_id}")
        logger.info(f"   Engine: {status.get('engine_chosen', 'unknown')}")
        logger.info(f"   Chunk size: {status.get('chunk_size_chosen', 'unknown')}")
    
    def on_job_completed(job_id, status):
        logger.info(f"✅ Job completed: {job_id}")
        logger.info(f"   Execution time: {status.get('execution_time', 0):.2f}s")
        logger.info(f"   Total samples: {status.get('total_samples', 0)}")
        logger.info(f"   Total anomalies: {status.get('total_anomalies', 0)}")
    
    def on_job_failed(job_id, status):
        logger.error(f"❌ Job failed: {job_id}")
        logger.error(f"   Error count: {status.get('error_count', 0)}")
    
    def on_progress_update(job_id, progress):
        logger.info(f"📊 Job {job_id}: {progress:.1f}% complete")
    
    return ProcessingHooks(
        pre_processing=pre_processing_hook,
        post_processing=post_processing_hook,
        on_job_submitted=on_job_submitted,
        on_job_started=on_job_started,
        on_job_completed=on_job_completed,
        on_job_failed=on_job_failed,
        on_progress_update=on_progress_update
    )


async def demo_simple_usage():
    """Demonstrate simple usage of BatchProcessingService."""
    logger.info("🔄 Demo: Simple Usage")
    
    # Create service
    service = create_batch_processing_service()
    
    # Create sample data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_file = temp_path / "input_data.csv"
        output_file = temp_path / "output_results.json"
        
        # Create and save sample data
        df = create_sample_data(1000)
        df.to_csv(input_file, index=False)
        
        # Submit job
        job_id = await service.submit_simple_job(
            name="Simple Demo Job",
            description="A simple demonstration of batch processing",
            input_path=str(input_file),
            output_path=str(output_file),
            detection_algorithm=DetectionAlgorithm.ISOLATION_FOREST
        )
        
        logger.info(f"Submitted job: {job_id}")
        
        # Monitor job status
        while True:
            status = await service.get_job_status(job_id)
            if not status:
                break
            
            logger.info(f"Job {job_id}: {status['status']} ({status.get('progress_percentage', 0):.1f}%)")
            
            if status['status'] in ['completed', 'failed', 'cancelled']:
                break
            
            await asyncio.sleep(2)
        
        # Get final metrics
        metrics = await service.get_job_metrics(job_id)
        logger.info(f"Final metrics: {metrics}")
        
        # Clean up
        await service.shutdown()


async def demo_advanced_usage():
    """Demonstrate advanced usage with hooks and configuration."""
    logger.info("🔄 Demo: Advanced Usage with Hooks")
    
    # Create configuration
    config = BatchConfig(
        engine=BatchEngine.MULTIPROCESSING,
        chunk_size=500,
        max_workers=2,
        detection_algorithm=DetectionAlgorithm.ISOLATION_FOREST
    )
    
    # Create service
    service = create_batch_processing_service(config)
    
    # Set up hooks
    hooks = create_processing_hooks()
    service.set_hooks(hooks)
    
    # Create sample data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_file = temp_path / "large_input_data.csv"
        output_file = temp_path / "output_results.parquet"
        
        # Create and save larger sample data
        df = create_sample_data(5000)
        df.to_csv(input_file, index=False)
        
        # Submit job with options
        options = JobSubmissionOptions(
            name="Advanced Demo Job",
            description="Advanced demonstration with hooks and configuration",
            input_path=str(input_file),
            output_path=str(output_file),
            detection_algorithm=DetectionAlgorithm.ISOLATION_FOREST,
            chunk_size=1000,  # Override chunk size
            engine=BatchEngine.MULTIPROCESSING,  # Override engine
            config_overrides={'max_workers': 4}  # Override max workers
        )
        
        job_id = await service.submit_job(options)
        
        logger.info(f"Submitted advanced job: {job_id}")
        
        # Monitor job status with less frequent updates
        while True:
            status = await service.get_job_status(job_id)
            if not status:
                break
            
            if status['status'] in ['completed', 'failed', 'cancelled']:
                break
            
            await asyncio.sleep(1)
        
        # Get system metrics
        system_metrics = await service.get_system_metrics()
        logger.info(f"System metrics: {system_metrics}")
        
        # Clean up
        await service.shutdown()


async def demo_multiple_jobs():
    """Demonstrate handling multiple concurrent jobs."""
    logger.info("🔄 Demo: Multiple Concurrent Jobs")
    
    # Create service
    service = create_batch_processing_service()
    
    # Set up hooks
    hooks = create_processing_hooks()
    service.set_hooks(hooks)
    
    job_ids = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Submit multiple jobs
        for i in range(3):
            input_file = temp_path / f"input_data_{i}.csv"
            output_file = temp_path / f"output_results_{i}.json"
            
            # Create sample data
            df = create_sample_data(1000 + i * 500)
            df.to_csv(input_file, index=False)
            
            job_id = await service.submit_simple_job(
                name=f"Multi Job {i+1}",
                description=f"Job {i+1} in multi-job demo",
                input_path=str(input_file),
                output_path=str(output_file)
            )
            
            job_ids.append(job_id)
            logger.info(f"Submitted job {i+1}: {job_id}")
        
        # Monitor all jobs
        while True:
            jobs = await service.list_jobs()
            running_jobs = [j for j in jobs if j['status'] == 'running']
            
            if not running_jobs:
                break
            
            logger.info(f"Running jobs: {len(running_jobs)}")
            await asyncio.sleep(2)
        
        # Get final status for all jobs
        for job_id in job_ids:
            status = await service.get_job_status(job_id)
            logger.info(f"Job {job_id}: {status['status']} - {status.get('total_samples', 0)} samples processed")
        
        # Clean up
        await service.shutdown()


async def demo_engine_selection():
    """Demonstrate automatic engine selection based on data size."""
    logger.info("🔄 Demo: Engine Selection")
    
    service = create_batch_processing_service()
    
    # Test different data sizes
    test_sizes = [10, 500, 5000, 50000]  # MB
    
    for size_mb in test_sizes:
        engine = service.choose_engine(size_mb)
        chunk_size = service.choose_chunk_size(size_mb, engine)
        
        logger.info(f"Data size: {size_mb}MB -> Engine: {engine.value}, Chunk size: {chunk_size}")
    
    await service.shutdown()


async def main():
    """Run all demos."""
    logger.info("🚀 Starting BatchProcessingService Demo")
    
    try:
        await demo_simple_usage()
        await asyncio.sleep(1)
        
        await demo_advanced_usage()
        await asyncio.sleep(1)
        
        await demo_multiple_jobs()
        await asyncio.sleep(1)
        
        await demo_engine_selection()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    
    logger.info("✅ All demos completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
