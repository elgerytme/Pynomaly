"""
End-to-End Integration Tests

Tests complete workflows from data ingestion to anomaly detection results.
"""

from datetime import datetime

import pytest

from tests.integration.framework.integration_test_base import CrossLayerIntegrationTest


class TestEndToEndWorkflow(CrossLayerIntegrationTest):
    """Test complete end-to-end workflows."""

    async def test_complete_anomaly_detection_workflow(self):
        """Test complete anomaly detection workflow from start to finish."""

        async with self.setup_test_environment() as env:
            # Test data preparation
            input_data = {
                "dataset_config": {
                    "name": "e2e_test_dataset",
                    "size": 1000,
                    "anomaly_rate": 0.1,
                    "features": 5,
                    "data_type": "tabular",
                },
                "detector_config": {
                    "name": "e2e_test_detector",
                    "algorithm": "IsolationForest",
                    "parameters": {"contamination": 0.1, "random_state": 42},
                },
            }

            # Execute complete workflow
            result = await self.execute_complete_workflow(
                "anomaly_detection", input_data
            )

            # Verify workflow completion
            assert result["status"] == "completed"
            assert result["workflow"] == "anomaly_detection"
            assert result["dataset_id"] is not None
            assert result["detector_id"] is not None
            assert result["result_id"] is not None

            # Verify anomaly detection results
            assert result["anomaly_count"] > 0
            assert result["execution_time"] > 0

            # Verify data persistence
            dataset_repo = env.container.dataset_repository()
            detector_repo = env.container.detector_repository()
            result_repo = env.container.result_repository()

            dataset = await dataset_repo.get_by_id(result["dataset_id"])
            detector = await detector_repo.get_by_id(result["detector_id"])
            detection_result = await result_repo.get_by_id(result["result_id"])

            assert dataset is not None
            assert detector is not None
            assert detection_result is not None

            # Verify service health
            self.assert_service_health("database")
            self.assert_service_health("cache")

    async def test_streaming_anomaly_detection_workflow(self):
        """Test streaming anomaly detection workflow."""

        async with self.setup_test_environment() as env:
            # Create streaming data source
            streaming_config = {
                "name": "streaming_test_source",
                "event_rate": 10,
                "duration_seconds": 30,
                "anomaly_rate": 0.2,
            }

            streaming_source = await env.test_data_manager.create_streaming_data_source(
                **streaming_config
            )

            # Create detector for streaming data
            detector = await env.test_data_manager.create_detector(
                name="streaming_detector",
                algorithm="LocalOutlierFactor",
                parameters={"contamination": 0.2, "n_neighbors": 5},
            )

            # Verify streaming source
            assert streaming_source["event_count"] == 300  # 10 events/sec * 30 sec
            assert streaming_source["anomaly_rate"] == 0.2

            # Test streaming processing would go here
            # (simplified for integration test)

            # Verify detector configuration
            assert detector.algorithm_name == "LocalOutlierFactor"
            assert detector.parameters["contamination"] == 0.2

    async def test_batch_processing_workflow(self):
        """Test batch processing workflow with multiple datasets."""

        async with self.setup_test_environment() as env:
            # Create multiple datasets
            dataset_configs = [
                {"name": "batch_dataset_1", "size": 500, "anomaly_rate": 0.05},
                {"name": "batch_dataset_2", "size": 1000, "anomaly_rate": 0.15},
                {"name": "batch_dataset_3", "size": 750, "anomaly_rate": 0.1},
            ]

            datasets = []
            for config in dataset_configs:
                dataset = await env.test_data_manager.create_dataset(**config)
                datasets.append(dataset)

            # Create detector
            detector = await env.test_data_manager.create_detector(
                name="batch_detector",
                algorithm="OneClassSVM",
                parameters={"nu": 0.1, "gamma": "scale"},
            )

            # Process each dataset
            results = []
            for dataset in datasets:
                input_data = {
                    "dataset_config": {"id": dataset.id},
                    "detector_config": {"id": detector.id},
                }

                result = await self.execute_complete_workflow(
                    "anomaly_detection", input_data
                )
                results.append(result)

            # Verify all results
            assert len(results) == 3
            for result in results:
                assert result["status"] == "completed"
                assert result["anomaly_count"] > 0

            # Verify batch consistency
            execution_times = [r["execution_time"] for r in results]
            assert all(t > 0 for t in execution_times)

    async def test_error_handling_workflow(self):
        """Test error handling in workflows."""

        async with self.setup_test_environment() as env:
            # Test with invalid detector configuration
            invalid_input = {
                "dataset_config": {
                    "name": "error_test_dataset",
                    "size": 100,
                    "anomaly_rate": 0.1,
                },
                "detector_config": {
                    "name": "invalid_detector",
                    "algorithm": "NonExistentAlgorithm",
                    "parameters": {},
                },
            }

            # Verify error handling
            with pytest.raises(ValueError, match="Unknown algorithm"):
                await self.execute_complete_workflow("anomaly_detection", invalid_input)

            # Test with missing dataset
            missing_dataset_input = {
                "dataset_config": {"id": "non_existent_id"},
                "detector_config": {
                    "name": "valid_detector",
                    "algorithm": "IsolationForest",
                    "parameters": {"contamination": 0.1},
                },
            }

            with pytest.raises(Exception):
                await self.execute_complete_workflow(
                    "anomaly_detection", missing_dataset_input
                )

    async def test_performance_workflow(self):
        """Test workflow performance with larger datasets."""

        async with self.setup_test_environment() as env:
            # Create larger dataset
            large_dataset = await env.test_data_manager.create_dataset(
                name="performance_test_dataset",
                size=10000,
                anomaly_rate=0.05,
                features=20,
                data_type="tabular",
            )

            # Create optimized detector
            detector = await env.test_data_manager.create_detector(
                name="performance_detector",
                algorithm="IsolationForest",
                parameters={"contamination": 0.05, "n_estimators": 50},
            )

            # Measure execution time
            start_time = datetime.now()

            input_data = {
                "dataset_config": {"id": large_dataset.id},
                "detector_config": {"id": detector.id},
            }

            result = await self.execute_complete_workflow(
                "anomaly_detection", input_data
            )

            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()

            # Verify performance
            assert result["status"] == "completed"
            assert total_time < 30.0  # Should complete within 30 seconds
            assert result["execution_time"] > 0

            # Verify result quality
            expected_anomalies = int(10000 * 0.05)  # 5% of 10,000
            actual_anomalies = result["anomaly_count"]

            # Allow some tolerance in anomaly detection
            assert abs(actual_anomalies - expected_anomalies) < expected_anomalies * 0.5

    async def _prepare_dataset(self, dataset_repo, input_data):
        """Prepare dataset for testing."""
        if "id" in input_data["dataset_config"]:
            return await dataset_repo.get_by_id(input_data["dataset_config"]["id"])
        else:
            return await self.environment.test_data_manager.create_dataset(
                **input_data["dataset_config"]
            )

    async def _prepare_detector(self, detector_repo, input_data):
        """Prepare detector for testing."""
        if "id" in input_data["detector_config"]:
            return await detector_repo.get_by_id(input_data["detector_config"]["id"])
        else:
            return await self.environment.test_data_manager.create_detector(
                **input_data["detector_config"]
            )

    async def _execute_anomaly_detection_workflow(self, container, input_data):
        """Execute complete anomaly detection workflow."""
        # Get repositories
        dataset_repo = container.dataset_repository()
        detector_repo = container.detector_repository()
        result_repo = container.result_repository()

        # Prepare dataset and detector
        dataset = await self._prepare_dataset(dataset_repo, input_data)
        detector = await self._prepare_detector(detector_repo, input_data)

        # Execute detection
        detection_service = container.detection_service()
        result = await detection_service.detect_anomalies(
            detector_id=detector.id, dataset_id=dataset.id
        )

        # Store result
        await result_repo.save(result)

        return {
            "workflow": "anomaly_detection",
            "status": "completed",
            "dataset_id": dataset.id,
            "detector_id": detector.id,
            "result_id": result.id,
            "anomaly_count": result.n_anomalies,
            "execution_time": result.execution_time_ms,
        }
