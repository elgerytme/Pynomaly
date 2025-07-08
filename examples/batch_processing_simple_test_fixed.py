#!/usr/bin/env python3
"""Simple test script for BatchProcessingService using sequential engine."""

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


def create_sample_data(size: int = 1000) -> pd.DataFrame:
    """Create sample data for demonstration."""
    np.random.seed(42)
    
    # Generate normal data
    normal_data = np.random.randn(size - 50, 3)
    
    # Generate anomalies (outliers)
    anomalies = np.random.randn(50, 3) * 3  # More spread out
    
    # Combine
    data = np.vstack([normal_data, anomalies])
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3'])
    
    return df


async def test_engine_selection():
    """Test engine selection logic."""
    logger.info("🔄 Testing Engine Selection Logic")
    
    service = create_batch_processing_service()
    
    # Test different data sizes
    test_cases = [
        (10, "small"),
        (150, "medium"),
        (1500, "large"),
        (15000, "very_large")
    ]
    
    for size_mb, description in test_cases:
        engine = service.choose_engine(size_mb)
        chunk_size = service.choose_chunk_size(size_mb, engine)
        
        logger.info(f"{description.capitalize()} dataset ({size_mb}MB): Engine={engine.value}, Chunk size={chunk_size}")
    
    await service.shutdown()
    
    return True


async def test_simple_job_submission():
    """Test simple job submission method."""
    logger.info("🔄 Testing Simple Job Submission")
    
    service = create_batch_processing_service(
        BatchConfig(engine=BatchEngine.SEQUENTIAL, chunk_size=50)
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_file = temp_path / "simple_test_data.csv"
        output_file = temp_path / "simple_results.json"
        
        # Create and save sample data
        df = create_sample_data(200)
        df.to_csv(input_file, index=False)
        
        # Submit simple job
        job_id = await service.submit_simple_job(
            name="Simple Job Test",
            description="Test simple job submission",
            input_path=str(input_file),
            output_path=str(output_file)
        )
        
        logger.info(f"Submitted simple job: {job_id}")
        
        # Wait for completion
        max_iterations = 20
        iteration = 0
        
        while iteration < max_iterations:
            status = await service.get_job_status(job_id)
            if not status:
                break
            
            if status['status'] in ['completed', 'failed', 'cancelled']:
                logger.info(f"Job completed with status: {status['status']}")
                break
            
            await asyncio.sleep(1)
            iteration += 1
        
        await service.shutdown()
        
        return status and status['status'] == 'completed' if status else False


async def test_sequential_processing():
    """Test sequential processing which should work reliably."""
    logger.info("🔄 Testing Sequential Processing")
    
    # Create configuration with sequential engine
    config = BatchConfig(
        engine=BatchEngine.SEQUENTIAL,
        chunk_size=100,
        detection_algorithm=DetectionAlgorithm.ISOLATION_FOREST
    )
    
    # Create service
    service = create_batch_processing_service(config)
    
    # Set up hooks
    hooks = ProcessingHooks(
        pre_processing=lambda data: {**data, 'preprocessed': True},
        post_processing=lambda status: {**status, 'postprocessed': True},
        on_job_submitted=lambda job_id, data: logger.info(f"✅ Job submitted: {job_id}"),
        on_job_started=lambda job_id, status: logger.info(f"🚀 Job started: {job_id}"),
        on_job_completed=lambda job_id, status: logger.info(f"✅ Job completed: {job_id} - {status.get('total_samples', 0)} samples, {status.get('total_anomalies', 0)} anomalies"),
        on_job_failed=lambda job_id, status: logger.error(f"❌ Job failed: {job_id}"),
        on_progress_update=lambda job_id, progress: logger.info(f"📊 Progress: {progress:.1f}%")
    )
    
    service.set_hooks(hooks)
    
    # Create sample data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_file = temp_path / "sequential_test_data.csv"
        output_file = temp_path / "sequential_results.json"
        
        # Create and save sample data
        df = create_sample_data(500)
        df.to_csv(input_file, index=False)
        
        logger.info(f"Created sample data with {len(df)} rows")
        
        # Submit job using JobSubmissionOptions
        options = JobSubmissionOptions(
            name="Sequential Processing Test",
            description="Test sequential processing with hooks",
            input_path=str(input_file),
            output_path=str(output_file),
            detection_algorithm=DetectionAlgorithm.ISOLATION_FOREST,
            engine=BatchEngine.SEQUENTIAL,
            chunk_size=100
        )
        
        job_id = await service.submit_job(options)
        logger.info(f"Submitted job: {job_id}")
        
        # Monitor job status
        max_iterations = 30
        iteration = 0
        
        while iteration < max_iterations:
            status = await service.get_job_status(job_id)
            if not status:
                logger.error("Job status not found")
                break
            
            logger.info(f"Job {job_id}: {status['status']} ({status.get('progress_percentage', 0):.1f}%)")
            
            if status['status'] in ['completed', 'failed', 'cancelled']:
                break
            
            await asyncio.sleep(1)
            iteration += 1
        
        # Get final metrics
        final_status = await service.get_job_status(job_id)
        if final_status:
            logger.info(f"Final status: {final_status['status']}")
            logger.info(f"Total samples: {final_status.get('total_samples', 0)}")
            logger.info(f"Total anomalies: {final_status.get('total_anomalies', 0)}")
            logger.info(f"Execution time: {final_status.get('execution_time', 0):.2f}s")
        
        # Get job metrics
        metrics = await service.get_job_metrics(job_id)
        logger.info(f"Job metrics: {metrics}")
        
        # Test job listing
        jobs = await service.list_jobs()
        logger.info(f"Total jobs: {len(jobs)}")
        
        # Test system metrics
        system_metrics = await service.get_system_metrics()
        logger.info(f"System metrics: {system_metrics}")
        
        # Clean up
        await service.shutdown()
        
        return final_status['status'] == 'completed' if final_status else False


async def main():
    """Run all tests."""
    logger.info("🚀 Starting BatchProcessingService Tests")
    
    test_results = []
    
    try:
        # Test 1: Engine Selection
        result1 = await test_engine_selection()
        test_results.append(("Engine Selection", result1))
        
        # Test 2: Simple Job Submission
        result2 = await test_simple_job_submission()
        test_results.append(("Simple Job Submission", result2))
        
        # Test 3: Sequential Processing (most comprehensive)
        result3 = await test_sequential_processing()
        test_results.append(("Sequential Processing", result3))
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        test_results.append(("Error", False))
    
    # Report results
    logger.info("\n📊 Test Results:")
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    logger.info(f"\n🎯 Summary: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("✅ All tests passed! BatchProcessingService is working correctly.")
    else:
        logger.warning("⚠️  Some tests failed. Check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
