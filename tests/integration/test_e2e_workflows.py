"""End-to-end workflow integration tests."""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from uuid import uuid4

from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.entities.training_job import TrainingJob
from pynomaly.domain.entities.anomaly_event import AnomalyEvent, EventType, EventSeverity
from pynomaly.domain.services.advanced_classification_service import AdvancedClassificationService
from pynomaly.domain.services.detection_pipeline_integration import DetectionPipelineIntegration
from pynomaly.domain.services.threshold_severity_classifier import ThresholdSeverityClassifier
from pynomaly.domain.value_objects import ContaminationRate


class MockDetectionService:
    """Mock detection service for testing."""
    
    async def detect_anomalies(self, detector: Detector, data: np.ndarray) -> List[float]:
        """Mock anomaly detection."""
        # Simulate anomaly scores based on data variance
        scores = []
        for row in data:
            variance = np.var(row)
            # Higher variance = higher anomaly score
            score = min(variance / 10.0, 1.0)
            scores.append(score)
        return scores


class MockTrainingService:
    """Mock training service for testing."""
    
    async def train_detector(self, training_job: TrainingJob) -> Detector:
        """Mock detector training."""
        # Create a fitted detector
        detector = Detector(
            id=training_job.detector_id,
            name=f"trained_{training_job.detector_id}",
            algorithm_name="mock_algorithm",
            contamination_rate=ContaminationRate.from_value(0.1),
            parameters=training_job.parameters,
            is_fitted=True,
            trained_at=datetime.utcnow(),
        )
        return detector


class TestE2EWorkflows:
    """End-to-end workflow testing covering complete user journeys."""

    @pytest.fixture(autouse=True)
    async def setup_e2e_environment(self):
        """Set up complete testing environment."""
        await self.setup_test_environment()
        self.data_generator = TestDataGenerator()
        
        # Create test data files
        self.test_data_dir = Path(tempfile.mkdtemp())
        self.sample_csv = self.test_data_dir / "sample_data.csv"
        self.sample_json = self.test_data_dir / "sample_data.json"
        
        # Generate test datasets
        self.data_generator.create_sample_csv(self.sample_csv, rows=1000, anomalies=True)
        self.data_generator.create_sample_json(self.sample_json, records=500)
        
        yield
        
        # Cleanup
        import shutil
        shutil.rmtree(self.test_data_dir, ignore_errors=True)

    @pytest.mark.integration
    @pytest.mark.e2e
    async def test_complete_anomaly_detection_workflow(self):
        """Test complete anomaly detection workflow from data load to results."""
        
        # Step 1: Data Loading
        dataset_service = self.container.dataset_service()
        dataset = await dataset_service.load_from_file(
            file_path=str(self.sample_csv),
            name="e2e_test_dataset",
            description="End-to-end test dataset"
        )
        
        assert dataset is not None
        assert dataset.name == "e2e_test_dataset"
        assert dataset.n_samples == 1000
        
        # Step 2: Data Preprocessing
        preprocessing_service = self.container.preprocessing_service()
        preprocessed_dataset = await preprocessing_service.preprocess(
            dataset=dataset,
            normalize=True,
            remove_missing=True,
            feature_selection=True
        )
        
        assert preprocessed_dataset is not None
        assert preprocessed_dataset.is_preprocessed
        
        # Step 3: Detector Creation and Training
        detector_service = self.container.detector_service()
        detector = await detector_service.create_detector(
            name="e2e_isolation_forest",
            algorithm="IsolationForest",
            parameters={"contamination": 0.1, "n_estimators": 100}
        )
        
        assert detector is not None
        assert detector.algorithm == "IsolationForest"
        
        # Train the detector
        training_result = await detector_service.train(
            detector=detector,
            dataset=preprocessed_dataset
        )
        
        assert training_result.success
        assert training_result.training_time > 0
        assert detector.is_trained
        
        # Step 4: Anomaly Detection
        detection_service = self.container.detection_service()
        results = await detection_service.detect(
            detector=detector,
            dataset=preprocessed_dataset
        )
        
        assert results is not None
        assert results.n_samples == preprocessed_dataset.n_samples
        assert 0 <= results.n_anomalies <= results.n_samples
        assert 0.0 <= results.anomaly_rate <= 1.0
        
        # Step 5: Model Evaluation
        evaluation_service = self.container.evaluation_service()
        evaluation_results = await evaluation_service.evaluate(
            detector=detector,
            test_dataset=preprocessed_dataset,
            cross_validate=True,
            n_folds=3
        )
        
        assert evaluation_results is not None
        assert "precision" in evaluation_results.metrics
        assert "recall" in evaluation_results.metrics
        assert "f1_score" in evaluation_results.metrics
        
        # Step 6: Results Export
        export_service = self.container.export_service()
        export_path = self.test_data_dir / "results.json"
        
        await export_service.export_results(
            results=results,
            format="json",
            output_path=str(export_path)
        )
        
        assert export_path.exists()
        
        # Verify exported data
        with open(export_path) as f:
            exported_data = json.load(f)
        
        assert "anomalies" in exported_data
        assert "metadata" in exported_data
        assert exported_data["metadata"]["algorithm"] == "IsolationForest"

    @pytest.mark.integration
    @pytest.mark.e2e
    async def test_multi_algorithm_comparison_workflow(self):
        """Test workflow comparing multiple algorithms."""
        
        # Load dataset
        dataset_service = self.container.dataset_service()
        dataset = await dataset_service.load_from_file(
            file_path=str(self.sample_csv),
            name="comparison_dataset"
        )
        
        # Create multiple detectors
        detector_service = self.container.detector_service()
        algorithms = [
            ("IsolationForest", {"contamination": 0.1, "n_estimators": 50}),
            ("LOF", {"contamination": 0.1, "n_neighbors": 20}),
            ("OneClassSVM", {"nu": 0.1, "kernel": "rbf"})
        ]
        
        detectors = []
        for algo_name, params in algorithms:
            detector = await detector_service.create_detector(
                name=f"comparison_{algo_name.lower()}",
                algorithm=algo_name,
                parameters=params
            )
            
            # Train detector
            await detector_service.train(detector=detector, dataset=dataset)
            detectors.append(detector)
        
        assert len(detectors) == 3
        assert all(d.is_trained for d in detectors)
        
        # Run comparison
        comparison_service = self.container.comparison_service()
        comparison_results = await comparison_service.compare_detectors(
            detectors=detectors,
            test_dataset=dataset,
            metrics=["precision", "recall", "f1_score", "auc_roc"]
        )
        
        assert comparison_results is not None
        assert len(comparison_results.results) == 3
        
        for result in comparison_results.results:
            assert result.detector_name in ["comparison_isolationforest", "comparison_lof", "comparison_oneclasssvm"]
            assert "precision" in result.metrics
            assert "recall" in result.metrics
            assert "f1_score" in result.metrics
        
        # Verify best detector selection
        best_detector = comparison_results.best_detector
        assert best_detector is not None
        assert best_detector.name in [d.name for d in detectors]

    @pytest.mark.integration
    @pytest.mark.e2e
    async def test_streaming_data_workflow(self):
        """Test streaming data processing workflow."""
        
        # Set up streaming service
        streaming_service = self.container.streaming_service()
        
        # Create streaming detector
        detector_service = self.container.detector_service()
        detector = await detector_service.create_detector(
            name="streaming_detector",
            algorithm="IsolationForest",
            parameters={"contamination": 0.1}
        )
        
        # Train on initial batch
        dataset_service = self.container.dataset_service()
        training_dataset = await dataset_service.load_from_file(
            file_path=str(self.sample_csv),
            name="streaming_training"
        )
        
        await detector_service.train(detector=detector, dataset=training_dataset)
        
        # Simulate streaming data
        streaming_results = []
        batch_size = 50
        
        for i in range(0, 500, batch_size):
            # Generate streaming batch
            streaming_batch = self.data_generator.create_streaming_batch(
                size=batch_size,
                anomaly_rate=0.05
            )
            
            # Process batch
            batch_results = await streaming_service.process_batch(
                detector=detector,
                data_batch=streaming_batch
            )
            
            streaming_results.append(batch_results)
        
        assert len(streaming_results) == 10  # 500/50 batches
        
        # Verify streaming processing
        total_processed = sum(r.n_samples for r in streaming_results)
        total_anomalies = sum(r.n_anomalies for r in streaming_results)
        
        assert total_processed == 500
        assert total_anomalies >= 0
        
        # Test streaming aggregation
        aggregated_results = await streaming_service.aggregate_results(
            streaming_results
        )
        
        assert aggregated_results.total_samples == 500
        assert aggregated_results.total_anomalies == total_anomalies

    @pytest.mark.integration
    @pytest.mark.e2e
    async def test_automl_workflow(self):
        """Test AutoML workflow for automatic algorithm selection."""
        
        # Load dataset
        dataset_service = self.container.dataset_service()
        dataset = await dataset_service.load_from_file(
            file_path=str(self.sample_csv),
            name="automl_dataset"
        )
        
        # Run AutoML
        automl_service = self.container.automl_service()
        automl_results = await automl_service.run_automl(
            dataset=dataset,
            max_algorithms=3,
            max_time_minutes=5,
            cv_folds=3,
            optimization_metric="f1_score"
        )
        
        assert automl_results is not None
        assert automl_results.best_detector is not None
        assert len(automl_results.algorithm_results) <= 3
        assert automl_results.total_time_seconds <= 300  # 5 minutes
        
        # Verify best detector performance
        best_detector = automl_results.best_detector
        assert best_detector.is_trained
        
        best_performance = automl_results.best_performance
        assert "f1_score" in best_performance
        assert 0.0 <= best_performance["f1_score"] <= 1.0
        
        # Test AutoML recommendations
        recommendations = automl_results.recommendations
        assert len(recommendations) > 0
        
        for rec in recommendations:
            assert "algorithm" in rec
            assert "confidence" in rec
            assert "reasoning" in rec

    @pytest.mark.integration
    @pytest.mark.e2e
    async def test_api_workflow_integration(self):
        """Test complete API workflow integration."""
        
        # Test API client
        api_client = self.container.api_client()
        
        # Upload dataset via API
        upload_response = await api_client.upload_dataset(
            file_path=str(self.sample_csv),
            dataset_name="api_test_dataset"
        )
        
        assert upload_response.success
        dataset_id = upload_response.dataset_id
        
        # Create detector via API
        detector_response = await api_client.create_detector(
            name="api_test_detector",
            algorithm="IsolationForest",
            parameters={"contamination": 0.1}
        )
        
        assert detector_response.success
        detector_id = detector_response.detector_id
        
        # Train detector via API
        training_response = await api_client.train_detector(
            detector_id=detector_id,
            dataset_id=dataset_id
        )
        
        assert training_response.success
        assert training_response.training_time > 0
        
        # Run detection via API
        detection_response = await api_client.run_detection(
            detector_id=detector_id,
            dataset_id=dataset_id
        )
        
        assert detection_response.success
        results = detection_response.results
        
        assert results.n_samples > 0
        assert results.n_anomalies >= 0
        assert 0.0 <= results.anomaly_rate <= 1.0
        
        # Get results via API
        results_response = await api_client.get_results(
            result_id=detection_response.result_id
        )
        
        assert results_response.success
        assert results_response.results.n_samples == results.n_samples

    @pytest.mark.integration
    @pytest.mark.e2e
    async def test_cli_workflow_integration(self):
        """Test CLI workflow integration."""
        
        # Test CLI through programmatic interface
        cli_service = self.container.cli_service()
        
        # Load dataset via CLI
        load_result = await cli_service.execute_command([
            "dataset", "load",
            str(self.sample_csv),
            "--name", "cli_test_dataset"
        ])
        
        assert load_result.exit_code == 0
        assert "successfully loaded" in load_result.output.lower()
        
        # Create detector via CLI
        create_result = await cli_service.execute_command([
            "detector", "create",
            "cli_test_detector",
            "--algorithm", "IsolationForest",
            "--contamination", "0.1"
        ])
        
        assert create_result.exit_code == 0
        assert "detector created" in create_result.output.lower()
        
        # Train detector via CLI
        train_result = await cli_service.execute_command([
            "detect", "train",
            "cli_test_detector",
            "cli_test_dataset"
        ])
        
        assert train_result.exit_code == 0
        assert "training completed" in train_result.output.lower()
        
        # Run detection via CLI
        detect_result = await cli_service.execute_command([
            "detect", "run",
            "cli_test_detector",
            "cli_test_dataset",
            "--output", str(self.test_data_dir / "cli_results.csv")
        ])
        
        assert detect_result.exit_code == 0
        assert "detection completed" in detect_result.output.lower()
        
        # Verify output file
        output_file = self.test_data_dir / "cli_results.csv"
        assert output_file.exists()

    @pytest.mark.integration
    @pytest.mark.e2e
    async def test_web_ui_workflow_integration(self):
        """Test Web UI workflow integration."""
        
        # Test web interface through API calls
        web_client = self.container.web_client()
        
        # Login to web interface
        login_response = await web_client.login(
            username="test_user",
            password="test_password"
        )
        
        assert login_response.success
        session_token = login_response.session_token
        
        # Upload dataset through web interface
        upload_response = await web_client.upload_dataset(
            session_token=session_token,
            file_path=str(self.sample_csv),
            dataset_name="web_test_dataset"
        )
        
        assert upload_response.success
        
        # Create and configure detector
        detector_response = await web_client.create_detector(
            session_token=session_token,
            name="web_test_detector",
            algorithm="IsolationForest",
            parameters={"contamination": 0.1}
        )
        
        assert detector_response.success
        
        # Start training job
        job_response = await web_client.start_training_job(
            session_token=session_token,
            detector_id=detector_response.detector_id,
            dataset_id=upload_response.dataset_id
        )
        
        assert job_response.success
        job_id = job_response.job_id
        
        # Poll job status
        max_polls = 30
        for _ in range(max_polls):
            status_response = await web_client.get_job_status(
                session_token=session_token,
                job_id=job_id
            )
            
            if status_response.status == "completed":
                break
            elif status_response.status == "failed":
                pytest.fail(f"Training job failed: {status_response.error}")
            
            await asyncio.sleep(1)
        else:
            pytest.fail("Training job did not complete within timeout")
        
        assert status_response.status == "completed"

    @pytest.mark.integration
    @pytest.mark.e2e
    async def test_error_handling_workflow(self):
        """Test error handling throughout the workflow."""
        
        dataset_service = self.container.dataset_service()
        detector_service = self.container.detector_service()
        
        # Test invalid file handling
        with pytest.raises(FileNotFoundError):
            await dataset_service.load_from_file(
                file_path="/nonexistent/file.csv",
                name="invalid_dataset"
            )
        
        # Test invalid algorithm handling
        with pytest.raises(ValueError):
            await detector_service.create_detector(
                name="invalid_detector",
                algorithm="NonExistentAlgorithm",
                parameters={}
            )
        
        # Test training without dataset
        detector = await detector_service.create_detector(
            name="test_detector",
            algorithm="IsolationForest",
            parameters={"contamination": 0.1}
        )
        
        with pytest.raises(ValueError):
            await detector_service.train(
                detector=detector,
                dataset=None
            )
        
        # Test detection on untrained detector
        dataset = await dataset_service.load_from_file(
            file_path=str(self.sample_csv),
            name="test_dataset"
        )
        
        detection_service = self.container.detection_service()
        
        with pytest.raises(ValueError):
            await detection_service.detect(
                detector=detector,  # Untrained
                dataset=dataset
            )

    @pytest.mark.integration
    @pytest.mark.e2e
    async def test_concurrent_workflow_execution(self):
        """Test concurrent execution of multiple workflows."""
        
        # Run multiple workflows concurrently
        tasks = [
            self._run_simple_detection_workflow(f"concurrent_{i}")
            for i in range(3)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all workflows completed successfully
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent workflow {i} failed: {result}")
            
            assert result.success
            assert result.n_samples > 0

    async def _run_simple_detection_workflow(self, suffix: str) -> Any:
        """Helper method for concurrent testing."""
        
        # Load dataset
        dataset_service = self.container.dataset_service()
        dataset = await dataset_service.load_from_file(
            file_path=str(self.sample_csv),
            name=f"concurrent_dataset_{suffix}"
        )
        
        # Create and train detector
        detector_service = self.container.detector_service()
        detector = await detector_service.create_detector(
            name=f"concurrent_detector_{suffix}",
            algorithm="IsolationForest",
            parameters={"contamination": 0.1, "n_estimators": 10}  # Small for speed
        )
        
        await detector_service.train(detector=detector, dataset=dataset)
        
        # Run detection
        detection_service = self.container.detection_service()
        results = await detection_service.detect(
            detector=detector,
            dataset=dataset
        )
        
        return results

    @pytest.mark.integration
    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_large_dataset_workflow(self):
        """Test workflow with large datasets."""
        
        # Generate large dataset
        large_csv = self.test_data_dir / "large_dataset.csv"
        self.data_generator.create_sample_csv(
            large_csv, 
            rows=10000,  # Larger dataset
            features=50,
            anomalies=True
        )
        
        # Load large dataset
        dataset_service = self.container.dataset_service()
        dataset = await dataset_service.load_from_file(
            file_path=str(large_csv),
            name="large_test_dataset"
        )
        
        assert dataset.n_samples == 10000
        assert dataset.n_features == 50
        
        # Create optimized detector for large data
        detector_service = self.container.detector_service()
        detector = await detector_service.create_detector(
            name="large_data_detector",
            algorithm="IsolationForest",
            parameters={
                "contamination": 0.1,
                "n_estimators": 100,
                "max_samples": "auto",
                "n_jobs": -1  # Use all CPU cores
            }
        )
        
        # Train with performance monitoring
        import time
        start_time = time.time()
        
        training_result = await detector_service.train(
            detector=detector,
            dataset=dataset
        )
        
        training_time = time.time() - start_time
        
        assert training_result.success
        assert training_time < 60  # Should complete within 1 minute
        
        # Run detection with monitoring
        start_time = time.time()
        
        detection_service = self.container.detection_service()
        results = await detection_service.detect(
            detector=detector,
            dataset=dataset
        )
        
        detection_time = time.time() - start_time
        
        assert results.n_samples == 10000
        assert detection_time < 30  # Should complete within 30 seconds
        
        # Verify memory usage didn't exceed limits
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        assert memory_mb < 1000  # Less than 1GB memory usage