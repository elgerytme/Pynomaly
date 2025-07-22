"""Integration tests for the worker system with real services."""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any

from anomaly_detection.worker import (
    AnomalyDetectionWorker, JobType, JobPriority, JobStatus
)
from anomaly_detection.domain.services.detection_service import DetectionService
from anomaly_detection.infrastructure.repositories.model_repository import ModelRepository


class TestWorkerServiceIntegration:
    """Test worker integration with real detection services."""
    
    @pytest.fixture
    def worker(self, temp_dir):
        """Create worker with real services."""
        return AnomalyDetectionWorker(
            models_dir=str(temp_dir / "worker_models"),
            max_concurrent_jobs=2,
            enable_monitoring=True  # Test with monitoring enabled
        )
    
    @pytest.fixture
    def sample_dataset(self, temp_dir):
        """Create realistic dataset for testing."""
        np.random.seed(42)
        
        # Normal behavior data
        normal_data = np.random.multivariate_normal(
            mean=[50, 100, 25],  # Price, quantity, customer_age
            cov=[[25, 10, 5], [10, 100, 15], [5, 15, 50]],
            size=500
        )
        
        # Anomalous behavior data
        anomaly_data = np.array([
            [200, 50, 20],   # High price, low quantity
            [300, 25, 65],   # Very high price, elderly customer
            [25, 500, 25],   # Low price, very high quantity
            [150, 200, 18],  # High price and quantity, young customer
            [400, 10, 45]    # Extremely high price
        ])
        
        # Combine data
        all_data = np.vstack([normal_data, anomaly_data])
        labels = np.hstack([np.ones(500), -np.ones(5)])  # 1=normal, -1=anomaly
        
        # Create DataFrame
        df = pd.DataFrame(all_data, columns=['price', 'quantity', 'customer_age'])
        df['is_anomaly'] = (labels == -1).astype(int)
        
        # Save to CSV
        csv_path = temp_dir / "sales_data.csv"
        df.to_csv(csv_path, index=False)
        
        return {
            'csv_path': csv_path,
            'data': all_data,
            'labels': labels,
            'n_samples': len(df),
            'n_anomalies': 5,
            'contamination_rate': 5/505
        }
    
    async def test_detection_job_with_real_service(self, worker, sample_dataset):
        """Test detection job using real detection service."""
        job_id = await worker.submit_job(
            JobType.DETECTION,
            {
                "data_source": str(sample_dataset['csv_path']),
                "algorithm": "isolation_forest",
                "contamination": 0.02,  # Expected contamination rate
                "parameters": {"random_state": 42}
            },
            priority=JobPriority.HIGH
        )
        
        # Start worker
        worker_task = asyncio.create_task(worker.start())
        
        try:
            # Monitor job progress
            completed = False
            max_wait = 60
            wait_time = 0
            
            while wait_time < max_wait and not completed:
                status = await worker.get_job_status(job_id)
                if status:
                    job_status = status["status"]
                    progress = status.get("progress", 0)
                    
                    print(f"Job {job_id[:8]}: {job_status} ({progress:.1f}%)")
                    
                    if job_status in ["completed", "failed"]:
                        completed = True
                        break
                
                await asyncio.sleep(2)
                wait_time += 2
            
            # Verify completion
            assert completed, "Job did not complete within timeout"
            
            final_status = await worker.get_job_status(job_id)
            assert final_status["status"] == "completed"
            
            result = final_status["result"]
            assert result is not None
            assert result["total_samples"] == sample_dataset['n_samples']
            assert result["anomalies_detected"] >= 0
            assert 0 <= result["anomaly_rate"] <= 1
            
            # Should detect some anomalies in our synthetic dataset
            assert result["anomalies_detected"] > 0, "Should detect some anomalies in synthetic data"
            
        finally:
            await worker.stop()
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
    
    async def test_ensemble_job_with_real_service(self, worker, sample_dataset):
        """Test ensemble detection job with real ensemble service."""
        job_id = await worker.submit_job(
            JobType.ENSEMBLE,
            {
                "data_source": str(sample_dataset['csv_path']),
                "algorithms": ["isolation_forest", "lof"],
                "method": "majority",
                "contamination": 0.02
            },
            priority=JobPriority.NORMAL
        )
        
        # Start worker
        worker_task = asyncio.create_task(worker.start())
        
        try:
            # Wait for completion
            completed = False
            max_wait = 90  # Ensemble jobs take longer
            wait_time = 0
            
            while wait_time < max_wait and not completed:
                status = await worker.get_job_status(job_id)
                if status and status["status"] in ["completed", "failed"]:
                    completed = True
                    break
                
                await asyncio.sleep(3)
                wait_time += 3
            
            assert completed, "Ensemble job did not complete within timeout"
            
            final_status = await worker.get_job_status(job_id)
            assert final_status["status"] == "completed"
            
            result = final_status["result"]
            assert result["algorithms"] == ["isolation_forest", "lof"]
            assert result["method"] == "majority"
            assert result["total_samples"] == sample_dataset['n_samples']
            assert result["anomalies_detected"] >= 0
            
        finally:
            await worker.stop()
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
    
    async def test_batch_training_job_with_model_persistence(self, worker, sample_dataset, temp_dir):
        """Test batch training job with model persistence."""
        job_id = await worker.submit_job(
            JobType.BATCH_TRAINING,
            {
                "data_source": str(sample_dataset['csv_path']),
                "algorithm": "isolation_forest",
                "contamination": 0.02,
                "model_name": "Sales Anomaly Detection Model"
            },
            priority=JobPriority.HIGH
        )
        
        # Start worker
        worker_task = asyncio.create_task(worker.start())
        
        try:
            # Wait for training completion
            completed = False
            max_wait = 120  # Training can take longer
            wait_time = 0
            
            while wait_time < max_wait and not completed:
                status = await worker.get_job_status(job_id)
                if status and status["status"] in ["completed", "failed"]:
                    completed = True
                    break
                
                await asyncio.sleep(3)
                wait_time += 3
            
            assert completed, "Training job did not complete within timeout"
            
            final_status = await worker.get_job_status(job_id)
            assert final_status["status"] == "completed"
            
            result = final_status["result"]
            assert "model_id" in result
            assert result["algorithm"] == "isolation_forest"
            assert result["training_samples"] > 0
            assert result["training_features"] == 3  # price, quantity, customer_age
            
            # Verify model was actually saved
            model_id = result["model_id"]
            model_repo = ModelRepository(str(temp_dir / "worker_models"))
            
            # Should be able to load the saved model
            saved_model = model_repo.load(model_id)
            assert saved_model is not None
            assert saved_model.metadata.name == "Sales Anomaly Detection Model"
            assert saved_model.metadata.algorithm == "isolation_forest"
            
            # Test the saved model can make predictions
            test_data = sample_dataset['data'][:10]
            predictions = saved_model.predict(test_data)
            assert len(predictions) == 10
            assert all(pred in [-1, 1] for pred in predictions)
            
        finally:
            await worker.stop()
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
    
    async def test_model_validation_job(self, worker, sample_dataset, temp_dir):
        """Test model validation job with real model repository."""
        # First, train a model using the detection service directly
        detection_service = DetectionService()
        data = sample_dataset['data']
        
        detection_service.fit(
            data=data,
            algorithm="isolation_forest",
            contamination=0.02,
            random_state=42
        )
        
        # Save the model
        from anomaly_detection.domain.entities.model import Model, ModelMetadata, ModelStatus
        
        metadata = ModelMetadata(
            model_id="validation-test-model",
            name="Test Validation Model",
            algorithm="isolation_forest",
            status=ModelStatus.TRAINED,
            training_samples=len(data),
            training_features=data.shape[1],
            contamination_rate=0.02
        )
        
        trained_model = detection_service._fitted_models["isolation_forest"]
        model = Model(metadata=metadata, model_object=trained_model)
        
        model_repo = ModelRepository(str(temp_dir / "worker_models"))
        model_id = model_repo.save(model)
        
        # Now submit validation job
        validation_job_id = await worker.submit_job(
            JobType.MODEL_VALIDATION,
            {
                "model_id": model_id,
                "validation_data_source": str(sample_dataset['csv_path'])
            },
            priority=JobPriority.NORMAL
        )
        
        # Start worker
        worker_task = asyncio.create_task(worker.start())
        
        try:
            # Wait for validation completion
            completed = False
            max_wait = 60
            wait_time = 0
            
            while wait_time < max_wait and not completed:
                status = await worker.get_job_status(validation_job_id)
                if status and status["status"] in ["completed", "failed"]:
                    completed = True
                    break
                
                await asyncio.sleep(2)
                wait_time += 2
            
            assert completed, "Validation job did not complete within timeout"
            
            final_status = await worker.get_job_status(validation_job_id)
            assert final_status["status"] == "completed"
            
            result = final_status["result"]
            assert result["model_id"] == model_id
            assert result["validation_samples"] == sample_dataset['n_samples']
            assert result["anomalies_detected"] >= 0
            assert 0 <= result["anomaly_rate"] <= 1
            assert result["model_performance"] in ["good", "needs_review"]
            
        finally:
            await worker.stop()
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
    
    async def test_data_preprocessing_with_real_data(self, worker, sample_dataset, temp_dir):
        """Test data preprocessing job with real data."""
        output_path = temp_dir / "processed_sales_data.csv"
        
        job_id = await worker.submit_job(
            JobType.DATA_PREPROCESSING,
            {
                "input_path": str(sample_dataset['csv_path']),
                "output_path": str(output_path),
                "steps": ["normalize", "remove_outliers"],
                "original_samples": sample_dataset['n_samples']
            },
            priority=JobPriority.NORMAL
        )
        
        # Start worker
        worker_task = asyncio.create_task(worker.start())
        
        try:
            # Wait for preprocessing completion
            completed = False
            max_wait = 45
            wait_time = 0
            
            while wait_time < max_wait and not completed:
                status = await worker.get_job_status(job_id)
                if status and status["status"] in ["completed", "failed"]:
                    completed = True
                    break
                
                await asyncio.sleep(2)
                wait_time += 2
            
            assert completed, "Preprocessing job did not complete within timeout"
            
            final_status = await worker.get_job_status(job_id)
            assert final_status["status"] == "completed"
            
            result = final_status["result"]
            assert result["original_samples"] == sample_dataset['n_samples']
            assert result["processed_samples"] <= sample_dataset['n_samples']  # May remove outliers
            assert result["preprocessing_steps"] == ["normalize", "remove_outliers"]
            
            # Verify output file exists and has correct structure
            assert output_path.exists()
            
            processed_df = pd.read_csv(output_path)
            assert len(processed_df) > 0
            assert "price" in processed_df.columns
            assert "quantity" in processed_df.columns
            assert "customer_age" in processed_df.columns
            
            # Data should be normalized (mean close to 0, std close to 1)
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'is_anomaly':  # Skip label column
                    assert abs(processed_df[col].mean()) < 0.5, f"Column {col} not properly normalized"
                    assert abs(processed_df[col].std() - 1.0) < 0.5, f"Column {col} std not close to 1"
            
        finally:
            await worker.stop()
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
    
    async def test_concurrent_job_processing(self, worker, sample_dataset):
        """Test processing multiple jobs concurrently."""
        # Submit multiple different types of jobs
        job_ids = []
        
        # Detection job
        detection_job_id = await worker.submit_job(
            JobType.DETECTION,
            {
                "data_source": str(sample_dataset['csv_path']),
                "algorithm": "lof",
                "contamination": 0.02
            },
            priority=JobPriority.HIGH
        )
        job_ids.append(("detection", detection_job_id))\n        
        # Explanation job
        explanation_job_id = await worker.submit_job(
            JobType.EXPLANATION_GENERATION,
            {
                "anomaly_indices": [100, 200, 300, 400, 500],  # Likely anomalous indices
                "method": "feature_importance"
            },
            priority=JobPriority.NORMAL
        )
        job_ids.append(("explanation", explanation_job_id))
        
        # Scheduled analysis job
        analysis_job_id = await worker.submit_job(
            JobType.SCHEDULED_ANALYSIS,
            {
                "analysis_type": "data_quality_check",
                "data_sources": ["sales", "inventory", "customer"]
            },
            priority=JobPriority.LOW
        )
        job_ids.append(("analysis", analysis_job_id))
        
        # Start worker
        worker_task = asyncio.create_task(worker.start())
        
        try:
            # Monitor all jobs
            max_wait = 180  # 3 minutes for all jobs
            wait_time = 0
            
            while wait_time < max_wait:
                all_complete = True
                
                for job_type, job_id in job_ids:
                    status = await worker.get_job_status(job_id)
                    if status:
                        job_status = status["status"]
                        progress = status.get("progress", 0)
                        print(f"{job_type} {job_id[:8]}: {job_status} ({progress:.1f}%)")
                        
                        if job_status not in ["completed", "failed"]:
                            all_complete = False
                    else:
                        all_complete = False
                
                if all_complete:
                    break
                
                await asyncio.sleep(3)
                wait_time += 3
            
            # Verify all jobs completed successfully
            for job_type, job_id in job_ids:
                final_status = await worker.get_job_status(job_id)
                assert final_status is not None, f"{job_type} job status not found"
                assert final_status["status"] == "completed", f"{job_type} job failed: {final_status.get('error_message')}"
                assert final_status["result"] is not None, f"{job_type} job has no result"
                
                print(f"✓ {job_type} job completed successfully")
            
            # Check worker status
            worker_status = await worker.get_worker_status()
            assert worker_status["currently_running_jobs"] == 0
            
            queue_status = worker_status["queue_status"]
            assert queue_status["status_counts"]["completed"] >= 3
            
        finally:
            await worker.stop()
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
    
    async def test_job_retry_mechanism(self, worker, temp_dir):
        """Test job retry mechanism with intentional failures."""
        # Create a job that will fail initially (non-existent file)
        job_id = await worker.submit_job(
            JobType.DETECTION,
            {
                "data_source": str(temp_dir / "nonexistent_file.csv"),
                "algorithm": "isolation_forest",
                "contamination": 0.1
            },
            priority=JobPriority.NORMAL
        )
        
        # Start worker
        worker_task = asyncio.create_task(worker.start())
        
        try:
            # Wait for job to fail and retry
            max_wait = 60
            wait_time = 0
            final_status = None
            
            while wait_time < max_wait:
                status = await worker.get_job_status(job_id)
                if status and status["status"] in ["failed", "completed"]:
                    final_status = status
                    break
                
                await asyncio.sleep(2)
                wait_time += 2
            
            assert final_status is not None
            assert final_status["status"] == "failed"  # Should fail after retries
            
            # Should have attempted retries
            job = await worker.job_queue.get_job(job_id)
            assert job.retry_count > 0, "Job should have been retried"
            assert job.error_message is not None, "Should have error message"
            
        finally:
            await worker.stop()
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
    
    async def test_worker_monitoring_integration(self, worker, sample_dataset):
        """Test worker integration with monitoring system."""
        # Submit a job to generate monitoring data
        job_id = await worker.submit_job(
            JobType.DETECTION,
            {
                "data_source": str(sample_dataset['csv_path']),
                "algorithm": "isolation_forest",
                "contamination": 0.02
            },
            priority=JobPriority.NORMAL
        )
        
        # Start worker with monitoring enabled
        worker_task = asyncio.create_task(worker.start())
        
        try:
            # Wait for job completion
            completed = False
            max_wait = 60
            wait_time = 0
            
            while wait_time < max_wait and not completed:
                status = await worker.get_job_status(job_id)
                if status and status["status"] in ["completed", "failed"]:
                    completed = True
                    break
                
                await asyncio.sleep(2)
                wait_time += 2
            
            assert completed, "Job did not complete within timeout"
            
            # Check that monitoring data was collected
            if worker.enable_monitoring:
                metrics_summary = worker.metrics_collector.get_summary_stats()
                assert metrics_summary["total_metrics"] > 0, "Should have collected metrics"
                
                # Should have recorded job submission and completion metrics
                assert "jobs_submitted" in str(metrics_summary), "Should track job submissions"
                
                perf_summary = worker.performance_monitor.get_performance_summary()
                # Performance monitoring may or may not have data depending on timing
                assert "total_profiles" in perf_summary
            
        finally:
            await worker.stop()
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
    
    async def test_job_cancellation(self, worker):
        """Test job cancellation functionality."""
        # Submit a long-running job
        job_id = await worker.submit_job(
            JobType.STREAM_MONITORING,
            {
                "stream_config": {"window_size": 100},
                "duration_seconds": 30  # Long duration
            },
            priority=JobPriority.NORMAL
        )
        
        # Start worker
        worker_task = asyncio.create_task(worker.start())
        
        try:
            # Wait for job to start
            await asyncio.sleep(3)
            
            # Check job is running
            status = await worker.get_job_status(job_id)
            assert status["status"] == "running"
            
            # Cancel the job
            success = await worker.cancel_job(job_id)
            assert success is True
            
            # Wait a bit for cancellation to take effect
            await asyncio.sleep(2)
            
            # Check job was cancelled
            final_status = await worker.get_job_status(job_id)
            assert final_status["status"] == "cancelled"
            
        finally:
            await worker.stop()
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass