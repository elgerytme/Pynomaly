"""Comprehensive integration tests for the complete Pynomaly system.

This test suite validates that all major components work together correctly
in production-like scenarios.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.pynomaly.domain.entities import Dataset, Detector
from src.pynomaly.infrastructure.adapters import SklearnAdapter
from src.pynomaly.infrastructure.data_processing import (
    DataValidator,
    MemoryOptimizedDataLoader,
    StreamingDataProcessor,
)
from src.pynomaly.infrastructure.monitoring import (
    PerformanceMonitor,
    get_monitor,
    init_monitor,
)
from src.pynomaly.infrastructure.repositories import (
    InMemoryDatasetRepository,
    InMemoryDetectorRepository,
)


class TestComprehensiveIntegration:
    """Test complete system integration."""

    def setup_method(self):
        """Setup for each test."""
        # Initialize clean monitoring
        init_monitor()

        # Create test repositories
        self.detector_repo = InMemoryDetectorRepository()
        self.dataset_repo = InMemoryDatasetRepository()

        # Create test data
        self.test_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.normal(0, 1, 1000),
                "feature3": np.random.uniform(-1, 1, 1000),
            }
        )

        # Add some anomalies
        self.test_data.iloc[950:, :] = (
            self.test_data.iloc[950:, :] * 5
        )  # Make last 50 rows anomalous

    def test_end_to_end_detection_workflow(self):
        """Test complete anomaly detection workflow."""
        monitor = get_monitor()

        with monitor.monitor_operation("e2e_detection_test", "integration_test"):
            # 1. Data Validation
            dataset = Dataset(
                id="test_dataset", name="Test Dataset", data=self.test_data
            )

            validator = DataValidator()
            validation_result = validator.validate_dataset(dataset)

            assert validation_result.is_valid, "Dataset should be valid"

            # 2. Memory-Efficient Processing
            loader = MemoryOptimizedDataLoader(chunk_size=200)
            chunks = list(loader.load_dataset(dataset))

            assert len(chunks) == 5  # 1000 / 200
            assert all(chunk.metadata["memory_optimized"] for chunk in chunks)

            # 3. Algorithm Adapter
            adapter = SklearnAdapter("IsolationForest")

            # 4. Detector Creation and Training
            Detector(
                id="test_detector",
                name="Test Detector",
                algorithm_name="IsolationForest",
                parameters={"contamination": 0.05},
            )

            # Train detector
            adapter.fit(self.test_data)

            # 5. Detection
            scores = adapter.predict(self.test_data)
            predictions = adapter.predict_proba(self.test_data)

            assert len(scores) == 1000
            assert len(predictions) == 1000

            # 6. Validate anomaly detection quality
            # Expect anomalies in the last 50 rows (we made them anomalous)
            anomaly_indices = np.where(scores == 1)[0]

            # Should detect some anomalies in the artificially created anomalous region
            anomalies_in_region = len([idx for idx in anomaly_indices if idx >= 950])
            assert anomalies_in_region > 0, (
                "Should detect some anomalies in the anomalous region"
            )

    def test_streaming_processing_integration(self):
        """Test streaming processing with validation and monitoring."""
        monitor = get_monitor()

        with monitor.monitor_operation("streaming_test", "integration_test"):
            # Create streaming processor
            streaming_processor = StreamingDataProcessor()

            # Mock processor for testing
            class MockDataProcessor:
                def __init__(self):
                    self.processed_chunks = []

                def process_chunk(self, chunk):
                    # Validate chunk data
                    validator = DataValidator()
                    dataset = Dataset(id="chunk", name="Chunk", data=chunk.data)
                    validation_result = validator.validate_dataset(dataset)

                    # Store validation results
                    chunk.metadata["validation_result"] = validation_result.is_valid
                    self.processed_chunks.append(chunk)
                    return chunk

                def finalize(self):
                    return {
                        "total_chunks": len(self.processed_chunks),
                        "all_valid": all(
                            chunk.metadata.get("validation_result", False)
                            for chunk in self.processed_chunks
                        ),
                    }

            # Create dataset
            dataset = Dataset(id="stream_test", name="Stream Test", data=self.test_data)

            # Process with streaming
            processor = MockDataProcessor()
            result = streaming_processor.process_dataset(dataset, processor)

            assert result["total_chunks"] > 0
            assert result["all_valid"], "All chunks should be valid"

    def test_monitoring_integration(self):
        """Test monitoring integration across components."""
        monitor = get_monitor()

        # Test nested monitoring operations
        with monitor.monitor_operation(
            "outer_operation", "integration_test"
        ):
            # Simulate data loading with monitoring
            with monitor.monitor_operation(
                "data_loading", "data_processor"
            ):
                dataset = Dataset(id="test", name="Test", data=self.test_data)
                assert dataset.data is not None

            # Simulate validation with monitoring
            with monitor.monitor_operation(
                "validation", "data_validator"
            ):
                validator = DataValidator()
                result = validator.validate_dataset(dataset)
                assert result.is_valid

            # Simulate detection with monitoring
            with monitor.monitor_operation("detection", "detector"):
                adapter = SklearnAdapter("IsolationForest")
                adapter.fit(self.test_data)
                scores = adapter.predict(self.test_data)
                assert len(scores) == 1000

        # Verify monitoring captured all operations
        log_entries = monitor.log_entries
        operation_names = [entry.operation for entry in log_entries]

        assert "outer_operation" in operation_names
        assert "data_loading" in operation_names
        assert "validation" in operation_names
        assert "detection" in operation_names

    def test_error_handling_integration(self):
        """Test error handling across components."""
        monitor = get_monitor()

        with monitor.monitor_operation("error_handling_test", "integration_test"):
            # Test validation with invalid data
            invalid_data = pd.DataFrame()  # Empty dataset
            invalid_dataset = Dataset(id="invalid", name="Invalid", data=invalid_data)

            validator = DataValidator()
            result = validator.validate_dataset(invalid_dataset)

            assert not result.is_valid
            assert result.has_critical_issues

            # Test adapter with invalid algorithm
            try:
                SklearnAdapter("NonExistentAlgorithm")
                raise AssertionError("Should have raised an exception")
            except Exception as e:
                error_id = monitor.report_error(e, "adapter_test", "initialization")
                assert error_id

        # Verify error was captured
        error_reports = monitor.error_reports
        assert len(error_reports) > 0

    def test_performance_monitoring_integration(self):
        """Test performance monitoring integration."""
        # Test with mock to avoid psutil dependencies in CI
        with patch("src.pynomaly.infrastructure.monitoring.performance_monitor.psutil"):
            performance_monitor = PerformanceMonitor()

            # Start operation tracking
            operation_id = performance_monitor.start_operation(
                "test_operation", "IsolationForest", dataset_size=1000
            )

            # Simulate some work
            import time

            time.sleep(0.01)

            # End operation tracking
            metrics = performance_monitor.end_operation(
                operation_id, samples_processed=1000
            )

            assert metrics.execution_time > 0
            assert metrics.samples_processed == 1000
            assert metrics.operation_name == "test_operation"
            assert metrics.algorithm_name == "IsolationForest"

    def test_memory_efficiency_integration(self):
        """Test memory efficiency across large dataset processing."""
        # Create larger test dataset
        large_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 10000),
                "feature2": np.random.normal(0, 1, 10000),
                "feature3": np.random.uniform(-1, 1, 10000),
            }
        )

        dataset = Dataset(id="large_test", name="Large Test", data=large_data)

        # Test memory-optimized loading
        loader = MemoryOptimizedDataLoader(chunk_size=1000, memory_limit_mb=100)

        with loader.monitor_memory_usage("large_dataset_test"):
            chunks = list(loader.load_dataset(dataset))

            assert len(chunks) == 10  # 10000 / 1000

            # Verify memory optimization was applied
            for chunk in chunks:
                assert chunk.metadata["memory_optimized"]
                assert chunk.size_mb > 0

    def test_file_processing_integration(self):
        """Test file-based processing integration."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            temp_path = Path(f.name)

        try:
            monitor = get_monitor()

            with monitor.monitor_operation("file_processing_test", "integration_test"):
                # 1. File validation
                from src.pynomaly.infrastructure.data_processing import (
                    validate_file_format,
                )

                file_validation = validate_file_format(temp_path)
                assert file_validation.is_valid

                # 2. Memory-optimized loading
                loader = MemoryOptimizedDataLoader(chunk_size=200)
                chunks = list(loader.load_csv(temp_path))

                assert len(chunks) == 5  # 1000 / 200

                # 3. Process each chunk with validation
                validator = DataValidator()
                valid_chunks = 0

                for chunk in chunks:
                    chunk_dataset = Dataset(
                        id=f"chunk_{chunk.chunk_id}",
                        name=f"Chunk {chunk.chunk_id}",
                        data=chunk.data,
                    )

                    validation_result = validator.validate_dataset(chunk_dataset)
                    if validation_result.is_valid:
                        valid_chunks += 1

                assert valid_chunks == len(chunks), "All chunks should be valid"

        finally:
            temp_path.unlink()

    def test_algorithm_adapter_integration(self):
        """Test algorithm adapter integration across different algorithms."""
        algorithms_to_test = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]

        monitor = get_monitor()

        for algorithm_name in algorithms_to_test:
            with monitor.monitor_operation(
                f"test_{algorithm_name}", "algorithm_integration"
            ):
                try:
                    # Create adapter
                    adapter = SklearnAdapter(algorithm_name)

                    # Fit and predict
                    adapter.fit(self.test_data)
                    scores = adapter.predict(self.test_data)
                    probabilities = adapter.predict_proba(self.test_data)

                    # Validate results
                    assert len(scores) == len(self.test_data)
                    assert len(probabilities) == len(self.test_data)
                    assert set(scores) <= {-1, 1}, (
                        f"Scores should be -1 or 1 for {algorithm_name}"
                    )

                except Exception as e:
                    # Log error but continue with other algorithms
                    monitor.report_error(e, "algorithm_integration", algorithm_name)
                    pytest.fail(f"Algorithm {algorithm_name} failed: {e}")

        # Verify all algorithms were tested
        log_entries = monitor.log_entries
        tested_algorithms = [
            entry.operation
            for entry in log_entries
            if entry.operation.startswith("test_")
        ]

        assert (
            len(tested_algorithms) >= len(algorithms_to_test) * 2
        )  # Start and end logs

    def test_production_readiness_checklist(self):
        """Test production readiness checklist."""
        monitor = get_monitor()

        with monitor.monitor_operation(
            "production_readiness_check", "integration_test"
        ):
            checklist_results = {}

            # 1. Core Detection Pipeline
            try:
                dataset = Dataset(id="test", name="Test", data=self.test_data)
                adapter = SklearnAdapter("IsolationForest")
                adapter.fit(self.test_data)
                adapter.predict(self.test_data)
                checklist_results["core_detection"] = True
            except Exception as e:
                checklist_results["core_detection"] = False
                monitor.report_error(e, "production_check", "core_detection")

            # 2. Data Validation
            try:
                validator = DataValidator()
                validation_result = validator.validate_dataset(dataset)
                checklist_results["data_validation"] = validation_result.is_valid
            except Exception as e:
                checklist_results["data_validation"] = False
                monitor.report_error(e, "production_check", "data_validation")

            # 3. Memory-Efficient Processing
            try:
                loader = MemoryOptimizedDataLoader()
                chunks = list(loader.load_dataset(dataset))
                checklist_results["memory_processing"] = len(chunks) > 0
            except Exception as e:
                checklist_results["memory_processing"] = False
                monitor.report_error(e, "production_check", "memory_processing")

            # 4. Monitoring and Logging
            try:
                metrics_summary = monitor.get_metrics_summary()
                checklist_results["monitoring"] = "logs" in metrics_summary
            except Exception as e:
                checklist_results["monitoring"] = False
                monitor.report_error(e, "production_check", "monitoring")

            # 5. Error Handling
            try:
                # Test error reporting
                test_error = ValueError("Test error for production check")
                error_id = monitor.report_error(
                    test_error, "production_check", "error_handling"
                )
                checklist_results["error_handling"] = bool(error_id)
            except Exception:
                checklist_results["error_handling"] = False

            # Calculate overall readiness score
            total_checks = len(checklist_results)
            passed_checks = sum(checklist_results.values())
            readiness_score = (passed_checks / total_checks) * 100

            monitor.info(
                f"Production readiness check complete: {readiness_score:.1f}%",
                operation="production_readiness_check",
                component="integration_test",
                checklist_results=checklist_results,
                readiness_score=readiness_score,
                passed_checks=passed_checks,
                total_checks=total_checks,
            )

            # Should achieve high production readiness
            assert readiness_score >= 80.0, (
                f"Production readiness score too low: {readiness_score}%"
            )
            assert checklist_results["core_detection"], "Core detection must work"
            assert checklist_results["monitoring"], "Monitoring must be functional"


@pytest.mark.asyncio
async def test_async_integration():
    """Test async integration capabilities."""
    monitor = get_monitor()

    async with monitor.monitor_async_operation(
        "async_integration_test", "integration_test"
    ):
        # Simulate async operations
        await asyncio.sleep(0.01)

        # Test that async monitoring works
        assert True  # Test passes if no exceptions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
