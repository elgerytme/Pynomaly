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


async def test_engine_selection():
    """Test engine selection logic."""
    logger.info("🔄 Testing Engine Selection Logic")\n    \n    service = create_batch_processing_service()\n    \n    # Test different data sizes\n    test_cases = [\n        (10, "small"),\n        (150, "medium"),\n        (1500, "large"),\n        (15000, "very_large")\n    ]\n    \n    for size_mb, description in test_cases:\n        engine = service.choose_engine(size_mb)\n        chunk_size = service.choose_chunk_size(size_mb, engine)\n        \n        logger.info(f"{description.capitalize()} dataset ({size_mb}MB): Engine={engine.value}, Chunk size={chunk_size}")\n    \n    await service.shutdown()\n    \n    return True


async def test_simple_job_submission():
    """Test simple job submission method."""\n    logger.info("🔄 Testing Simple Job Submission")\n    \n    service = create_batch_processing_service(\n        BatchConfig(engine=BatchEngine.SEQUENTIAL, chunk_size=50)\n    )\n    \n    with tempfile.TemporaryDirectory() as temp_dir:\n        temp_path = Path(temp_dir)\n        input_file = temp_path / "simple_test_data.csv"\n        output_file = temp_path / "simple_results.json"\n        \n        # Create and save sample data\n        df = create_sample_data(200)\n        df.to_csv(input_file, index=False)\n        \n        # Submit simple job\n        job_id = await service.submit_simple_job(\n            name="Simple Job Test",\n            description="Test simple job submission",\n            input_path=str(input_file),\n            output_path=str(output_file)\n        )\n        \n        logger.info(f"Submitted simple job: {job_id}")\n        \n        # Wait for completion\n        max_iterations = 20\n        iteration = 0\n        \n        while iteration < max_iterations:\n            status = await service.get_job_status(job_id)\n            if not status:\n                break\n            \n            if status['status'] in ['completed', 'failed', 'cancelled']:\n                logger.info(f"Job completed with status: {status['status']}")\n                break\n            \n            await asyncio.sleep(1)\n            iteration += 1\n        \n        await service.shutdown()\n        \n        return status and status['status'] == 'completed' if status else False


async def main():\n    """Run all tests."""\n    logger.info("🚀 Starting BatchProcessingService Tests")\n    \n    test_results = []\n    \n    try:\n        # Test 1: Engine Selection\n        result1 = await test_engine_selection()\n        test_results.append(("Engine Selection", result1))\n        \n        # Test 2: Simple Job Submission\n        result2 = await test_simple_job_submission()\n        test_results.append(("Simple Job Submission", result2))\n        \n        # Test 3: Sequential Processing (most comprehensive)\n        result3 = await test_sequential_processing()\n        test_results.append(("Sequential Processing", result3))\n        \n    except Exception as e:\n        logger.error(f"Test failed with error: {e}")\n        test_results.append(("Error", False))\n    \n    # Report results\n    logger.info("\\n📊 Test Results:")\n    for test_name, result in test_results:\n        status = "✅ PASS" if result else "❌ FAIL"\n        logger.info(f"  {test_name}: {status}")\n    \n    passed = sum(1 for _, result in test_results if result)\n    total = len(test_results)\n    logger.info(f"\\n🎯 Summary: {passed}/{total} tests passed")\n    \n    if passed == total:\n        logger.info("✅ All tests passed! BatchProcessingService is working correctly.")\n    else:\n        logger.warning("⚠️  Some tests failed. Check the implementation.")\n    \n    return passed == total


if __name__ == "__main__":\n    success = asyncio.run(main())\n    exit(0 if success else 1)
