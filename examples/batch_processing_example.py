"""Example usage of the Batch Processing Orchestration system.

This example demonstrates how to use the comprehensive batch processing
system for handling large datasets efficiently with progress tracking,
error recovery, and monitoring.
"""

import asyncio
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

# Import batch processing components
from src.pynomaly.application.services import (
    BatchOrchestrator,
    BatchJobRequest,
    BatchPriority,
    BatchMonitoringService,
    BatchRecoveryService,
    ProgressEvent,
    BatchAlert
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_dataset(size: int = 10000) -> pd.DataFrame:
    """Create a sample dataset for processing."""
    import numpy as np
    
    logger.info(f"Creating sample dataset with {size} rows")
    
    np.random.seed(42)  # For reproducible results
    
    data = {
        'feature_1': np.random.normal(0, 1, size),
        'feature_2': np.random.exponential(2, size),
        'feature_3': np.random.uniform(-5, 5, size),
        'feature_4': np.random.gamma(2, 2, size),
        'category': np.random.choice(['A', 'B', 'C', 'D'], size),
        'timestamp': pd.date_range('2024-01-01', periods=size, freq='1H'),
        'text_field': [f'sample_text_{i}' for i in range(size)]
    }
    
    # Add some anomalies
    anomaly_indices = np.random.choice(size, size // 20, replace=False)  # 5% anomalies
    data['feature_1'][anomaly_indices] = np.random.normal(10, 2, len(anomaly_indices))
    
    # Add some missing values
    missing_indices = np.random.choice(size, size // 50, replace=False)  # 2% missing
    for idx in missing_indices:
        col = np.random.choice(['feature_2', 'feature_3', 'feature_4'])
        data[col][idx] = np.nan
    
    df = pd.DataFrame(data)
    logger.info(f"Created dataset with shape {df.shape}")
    return df


def setup_progress_callback():
    """Setup a progress callback function."""
    def progress_callback(event: ProgressEvent):
        """Handle progress events."""
        if event.event_type == "started":
            logger.info(f"Job {event.job_id} started: {event.message}")
        elif event.event_type == "progress":
            logger.info(f"Job {event.job_id} progress: {event.progress_percentage:.1f}% - {event.message}")
        elif event.event_type == "completed":
            logger.info(f"Job {event.job_id} completed: {event.message}")
        elif event.event_type == "failed":
            logger.error(f"Job {event.job_id} failed: {event.message}")
    
    return progress_callback


def setup_alert_callback():
    """Setup an alert callback function."""
    def alert_callback(alert: BatchAlert):
        """Handle alerts."""
        level_emoji = {
            "info": "ℹ️",
            "warning": "⚠️", 
            "error": "❌",
            "critical": "🚨"
        }
        
        emoji = level_emoji.get(alert.level.value, "❓")
        logger.warning(f"{emoji} ALERT [{alert.level.value.upper()}]: {alert.title} - {alert.message}")
    
    return alert_callback


async def example_basic_batch_processing():
    """Example of basic batch processing."""
    logger.info("=== Basic Batch Processing Example ===")
    
    # Create orchestrator
    orchestrator = BatchOrchestrator()
    
    # Create sample data
    data = create_sample_dataset(1000)
    
    # Submit a data quality job
    job_request = BatchJobRequest(
        name="Data Quality Assessment",
        description="Assess data quality of sample dataset",
        processor_type="data_quality",
        data_source=data,
        priority=BatchPriority.HIGH,
        config_overrides={
            "batch_size": 100,
            "enable_progress_tracking": True
        }
    )
    
    job_id = await orchestrator.submit_job(job_request)
    logger.info(f"Submitted job: {job_id}")
    
    # Wait for completion
    while True:
        status = orchestrator.get_job_status(job_id)
        if status["status"] in ["completed", "failed", "cancelled"]:
            break
        await asyncio.sleep(1)
    
    logger.info(f"Job completed with status: {status['status']}")
    return job_id


async def example_job_dependencies():
    """Example of jobs with dependencies."""
    logger.info("=== Job Dependencies Example ===")
    
    orchestrator = BatchOrchestrator()
    data = create_sample_dataset(500)
    
    # Job 1: Data Profiling (no dependencies)
    profiling_request = BatchJobRequest(
        name="Data Profiling",
        description="Profile the dataset structure and statistics",
        processor_type="data_profiling",
        data_source=data,
        priority=BatchPriority.HIGH,
        config_overrides={"batch_size": 50}
    )
    
    profiling_job_id = await orchestrator.submit_job(profiling_request)
    logger.info(f"Submitted profiling job: {profiling_job_id}")
    
    # Job 2: Data Quality (depends on profiling)
    quality_request = BatchJobRequest(
        name="Data Quality Check",
        description="Check data quality based on profiling results",
        processor_type="data_quality",
        data_source=data,
        depends_on=[profiling_job_id],
        priority=BatchPriority.MEDIUM,
        config_overrides={"batch_size": 100}
    )
    
    quality_job_id = await orchestrator.submit_job(quality_request)
    logger.info(f"Submitted quality job: {quality_job_id}")
    
    # Job 3: Anomaly Detection (depends on quality check)
    anomaly_request = BatchJobRequest(
        name="Anomaly Detection",
        description="Detect anomalies after quality assessment",
        processor_type="anomaly_detection",
        data_source=data,
        depends_on=[quality_job_id],
        priority=BatchPriority.MEDIUM,
        config_overrides={"batch_size": 75}
    )
    
    anomaly_job_id = await orchestrator.submit_job(anomaly_request)
    logger.info(f"Submitted anomaly detection job: {anomaly_job_id}")
    
    # Wait for all jobs to complete
    all_job_ids = [profiling_job_id, quality_job_id, anomaly_job_id]
    completed_jobs = []
    
    while len(completed_jobs) < len(all_job_ids):
        for job_id in all_job_ids:
            if job_id not in completed_jobs:
                status = orchestrator.get_job_status(job_id)
                if status["status"] in ["completed", "failed", "cancelled"]:
                    completed_jobs.append(job_id)
                    logger.info(f"Job {job_id} completed with status: {status['status']}")
        
        await asyncio.sleep(1)
    
    logger.info("All dependent jobs completed!")
    return all_job_ids


async def example_concurrent_processing():
    """Example of concurrent job processing."""
    logger.info("=== Concurrent Processing Example ===")
    
    orchestrator = BatchOrchestrator()
    orchestrator.max_concurrent_jobs = 3  # Allow 3 concurrent jobs
    
    # Create multiple datasets
    datasets = [create_sample_dataset(300) for _ in range(5)]
    
    # Submit multiple independent jobs
    job_ids = []
    for i, data in enumerate(datasets):
        request = BatchJobRequest(
            name=f"Concurrent Job {i+1}",
            description=f"Process dataset {i+1} concurrently",
            processor_type="data_quality",
            data_source=data,
            priority=BatchPriority.MEDIUM,
            config_overrides={"batch_size": 50}
        )
        
        job_id = await orchestrator.submit_job(request)
        job_ids.append(job_id)
        logger.info(f"Submitted concurrent job {i+1}: {job_id}")
    
    # Monitor progress
    completed = set()
    while len(completed) < len(job_ids):
        system_status = orchestrator.get_system_status()
        logger.info(f"System status: {system_status['running_jobs']} running, "
                   f"{system_status['scheduled_jobs']} scheduled, "
                   f"{system_status['completed_jobs']} completed")
        
        for job_id in job_ids:
            if job_id not in completed:
                status = orchestrator.get_job_status(job_id)
                if status["status"] in ["completed", "failed", "cancelled"]:
                    completed.add(job_id)
                    logger.info(f"Job {job_id} finished: {status['status']}")
        
        await asyncio.sleep(2)
    
    logger.info("All concurrent jobs completed!")
    return job_ids


async def example_monitoring_and_alerts():
    """Example of monitoring and alerting."""
    logger.info("=== Monitoring and Alerts Example ===")
    
    # Setup monitoring
    monitoring_service = BatchMonitoringService()
    recovery_service = BatchRecoveryService(monitoring_service)
    
    # Setup callbacks
    progress_callback = setup_progress_callback()
    alert_callback = setup_alert_callback()
    
    monitoring_service.add_progress_callback(progress_callback)
    monitoring_service.add_alert_callback(alert_callback)
    
    # Start monitoring
    await monitoring_service.start_monitoring()
    
    try:
        # Create orchestrator with monitoring
        orchestrator = BatchOrchestrator()
        
        # Create a larger dataset to trigger monitoring
        large_data = create_sample_dataset(2000)
        
        # Submit job with monitoring
        request = BatchJobRequest(
            name="Monitored Job",
            description="Job with comprehensive monitoring",
            processor_type="anomaly_detection",
            data_source=large_data,
            priority=BatchPriority.HIGH,
            config_overrides={
                "batch_size": 200,
                "enable_progress_tracking": True
            }
        )
        
        job_id = await orchestrator.submit_job(request)
        logger.info(f"Submitted monitored job: {job_id}")
        
        # Monitor the job
        while True:
            status = orchestrator.get_job_status(job_id)
            if status["status"] in ["completed", "failed", "cancelled"]:
                break
            
            # Get monitoring data
            dashboard_data = monitoring_service.get_monitoring_dashboard_data()
            logger.info(f"System: CPU {dashboard_data['system_summary'].get('cpu_percent', 0):.1f}%, "
                       f"Memory {dashboard_data['system_summary'].get('memory_percent', 0):.1f}%")
            
            await asyncio.sleep(3)
        
        logger.info(f"Monitored job completed: {status['status']}")
        
        # Get final statistics
        dashboard_data = monitoring_service.get_monitoring_dashboard_data()
        logger.info(f"Final dashboard data: {dashboard_data}")
        
    finally:
        await monitoring_service.stop_monitoring()
    
    return job_id


async def example_error_handling():
    """Example of error handling and recovery."""
    logger.info("=== Error Handling Example ===")
    
    orchestrator = BatchOrchestrator()
    
    # Register a processor that will fail
    async def failing_processor(batch_data, context):
        """Processor that fails on specific batches."""
        batch_index = context["batch_index"]
        if batch_index == 2:  # Fail on third batch
            raise ValueError(f"Simulated failure on batch {batch_index}")
        
        # Simulate processing
        await asyncio.sleep(0.1)
        return {
            "batch_index": batch_index,
            "processed_rows": len(batch_data),
            "status": "success"
        }
    
    orchestrator.batch_service.register_processor("failing_processor", failing_processor)
    
    # Submit job that will fail
    data = create_sample_dataset(500)
    request = BatchJobRequest(
        name="Failing Job",
        description="Job that will encounter errors",
        processor_type="failing_processor",
        data_source=data,
        config_overrides={
            "batch_size": 100,
            "retry_attempts": 2
        }
    )
    
    job_id = await orchestrator.submit_job(request)
    logger.info(f"Submitted failing job: {job_id}")
    
    # Wait for completion (it will fail)
    while True:
        status = orchestrator.get_job_status(job_id)
        if status["status"] in ["completed", "failed", "cancelled"]:
            break
        await asyncio.sleep(1)
    
    logger.info(f"Job status: {status['status']}")
    if status["status"] == "failed":
        logger.info(f"Error message: {status.get('error_message', 'Unknown error')}")
    
    return job_id


async def example_configuration_optimization():
    """Example of automatic configuration optimization."""
    logger.info("=== Configuration Optimization Example ===")
    
    from src.pynomaly.application.services import BatchConfigurationManager
    
    config_manager = BatchConfigurationManager()
    
    # Create datasets of different sizes and complexities
    datasets = {
        "small_simple": pd.DataFrame({'numbers': range(100)}),
        "medium_complex": create_sample_dataset(1000),
        "large_simple": pd.DataFrame({'values': range(10000)}),
        "large_complex": create_sample_dataset(5000)
    }
    
    for name, data in datasets.items():
        logger.info(f"\n--- Optimizing configuration for {name} ---")
        
        # Get optimization recommendation
        result = config_manager.calculate_optimal_batch_config(
            data=data,
            processor_name="anomaly_detection"
        )
        
        logger.info(f"Dataset: {name}")
        logger.info(f"  Rows: {len(data)}, Columns: {len(data.columns)}")
        logger.info(f"  Recommended batch size: {result.recommended_batch_size}")
        logger.info(f"  Recommended concurrency: {result.recommended_concurrency}")
        logger.info(f"  Estimated memory usage: {result.estimated_memory_usage_mb:.1f} MB")
        logger.info(f"  Estimated processing time: {result.estimated_processing_time_seconds:.1f} seconds")
        logger.info(f"  Confidence score: {result.confidence_score:.2f}")
        
        if result.warnings:
            logger.warning(f"  Warnings: {result.warnings}")
    
    # Get system recommendations
    recommendations = config_manager.get_system_recommendations()
    logger.info(f"\nSystem Recommendations:")
    logger.info(f"  CPU Usage: {recommendations['system_status']['cpu_usage']:.1f}%")
    logger.info(f"  Memory Usage: {recommendations['system_status']['memory_usage']:.1f}%")
    for rec in recommendations['recommendations']:
        logger.info(f"  {rec['type'].upper()}: {rec['message']}")


async def main():
    """Run all examples."""
    logger.info("Starting Batch Processing Examples")
    logger.info("=" * 50)
    
    try:
        # Run examples
        await example_basic_batch_processing()
        await asyncio.sleep(1)
        
        await example_job_dependencies()
        await asyncio.sleep(1)
        
        await example_concurrent_processing()
        await asyncio.sleep(1)
        
        await example_monitoring_and_alerts()
        await asyncio.sleep(1)
        
        await example_error_handling()
        await asyncio.sleep(1)
        
        await example_configuration_optimization()
        
        logger.info("=" * 50)
        logger.info("All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())