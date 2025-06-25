"""Comprehensive SDK streaming detection tests.

This module contains comprehensive tests for streaming anomaly detection
features including real-time processing, incremental learning, adaptive
thresholds, and streaming data pipelines.
"""

import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pynomaly.domain.entities import Dataset
from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter


class TestSDKStreamingDetection:
    """Test SDK streaming anomaly detection capabilities."""

    @pytest.fixture
    def mock_streaming_sdk(self):
        """Create mock SDK with streaming capabilities."""

        class MockStreamingSDK:
            def __init__(self):
                self.models = {}
                self.streams = {}
                self.stream_results = {}
                self.adaptive_thresholds = {}
                self.data_buffers = {}
                self.streaming_stats = {}

            def create_streaming_detector(self, algorithm: str, **parameters) -> str:
                """Create a streaming anomaly detector."""
                try:
                    adapter = SklearnAdapter(
                        algorithm_name=algorithm, parameters=parameters
                    )

                    detector_id = str(uuid.uuid4())
                    self.models[detector_id] = {
                        "adapter": adapter,
                        "algorithm": algorithm,
                        "parameters": parameters,
                        "trained": False,
                        "streaming_mode": True,
                        "buffer_size": parameters.get("buffer_size", 1000),
                        "update_frequency": parameters.get("update_frequency", 100),
                    }

                    # Initialize streaming components
                    self.data_buffers[detector_id] = deque(
                        maxlen=self.models[detector_id]["buffer_size"]
                    )
                    self.streaming_stats[detector_id] = {
                        "samples_processed": 0,
                        "anomalies_detected": 0,
                        "last_update": None,
                        "processing_times": deque(maxlen=100),
                    }

                    return detector_id

                except Exception as e:
                    raise ValueError(f"Failed to create streaming detector: {str(e)}")

            def train_streaming_detector(
                self, detector_id: str, initial_data: pd.DataFrame
            ) -> dict[str, Any]:
                """Train streaming detector with initial data."""
                if detector_id not in self.models:
                    raise ValueError(f"Detector {detector_id} not found")

                model = self.models[detector_id]

                try:
                    # Train on initial data
                    dataset = Dataset(name="Initial Training", data=initial_data)
                    adapter = model["adapter"]
                    adapter.fit(dataset)

                    model["trained"] = True
                    model["trained_at"] = datetime.utcnow()

                    # Initialize adaptive threshold
                    scores = adapter.score(dataset)
                    score_values = [score.value for score in scores]

                    self.adaptive_thresholds[detector_id] = {
                        "current_threshold": np.percentile(score_values, 90),
                        "score_history": deque(score_values[-100:], maxlen=1000),
                        "adaptation_rate": 0.1,
                        "min_threshold": 0.1,
                        "max_threshold": 0.9,
                    }

                    return {
                        "detector_id": detector_id,
                        "status": "success",
                        "initial_samples": len(initial_data),
                        "initial_threshold": self.adaptive_thresholds[detector_id][
                            "current_threshold"
                        ],
                        "trained_at": model["trained_at"].isoformat(),
                    }

                except Exception as e:
                    return {
                        "detector_id": detector_id,
                        "status": "failed",
                        "error": str(e),
                    }

            def process_streaming_sample(
                self, detector_id: str, sample: dict[str, float]
            ) -> dict[str, Any]:
                """Process a single streaming sample."""
                if detector_id not in self.models:
                    raise ValueError(f"Detector {detector_id} not found")

                model = self.models[detector_id]
                if not model["trained"]:
                    raise ValueError(f"Detector {detector_id} is not trained")

                start_time = time.time()

                try:
                    # Convert sample to DataFrame
                    sample_df = pd.DataFrame([sample])
                    dataset = Dataset(name="Streaming Sample", data=sample_df)

                    # Get anomaly score
                    adapter = model["adapter"]
                    scores = adapter.score(dataset)
                    score_value = scores[0].value

                    # Adaptive threshold detection
                    threshold_info = self.adaptive_thresholds[detector_id]
                    current_threshold = threshold_info["current_threshold"]

                    is_anomaly = score_value > current_threshold

                    # Update adaptive threshold
                    self._update_adaptive_threshold(detector_id, score_value)

                    # Update streaming statistics
                    stats = self.streaming_stats[detector_id]
                    stats["samples_processed"] += 1
                    if is_anomaly:
                        stats["anomalies_detected"] += 1
                    stats["last_update"] = datetime.utcnow()

                    processing_time = time.time() - start_time
                    stats["processing_times"].append(processing_time)

                    # Store in buffer for potential incremental learning
                    self.data_buffers[detector_id].append(sample)

                    result = {
                        "detector_id": detector_id,
                        "sample": sample,
                        "anomaly_score": score_value,
                        "threshold": current_threshold,
                        "is_anomaly": is_anomaly,
                        "confidence": abs(score_value - current_threshold),
                        "processing_time": processing_time,
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                    # Check if incremental update is needed
                    if stats["samples_processed"] % model["update_frequency"] == 0:
                        self._incremental_update(detector_id)
                        result["model_updated"] = True
                    else:
                        result["model_updated"] = False

                    return result

                except Exception as e:
                    return {
                        "detector_id": detector_id,
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat(),
                    }

            def process_streaming_batch(
                self, detector_id: str, batch: list[dict[str, float]]
            ) -> list[dict[str, Any]]:
                """Process a batch of streaming samples."""
                results = []

                for sample in batch:
                    result = self.process_streaming_sample(detector_id, sample)
                    results.append(result)

                return results

            def _update_adaptive_threshold(self, detector_id: str, new_score: float):
                """Update adaptive threshold based on new score."""
                threshold_info = self.adaptive_thresholds[detector_id]

                # Add score to history
                threshold_info["score_history"].append(new_score)

                # Recalculate threshold using recent scores
                recent_scores = list(threshold_info["score_history"])
                if len(recent_scores) >= 10:  # Need minimum samples
                    new_threshold = np.percentile(recent_scores, 90)

                    # Apply adaptive update
                    adaptation_rate = threshold_info["adaptation_rate"]
                    current_threshold = threshold_info["current_threshold"]

                    updated_threshold = (
                        1 - adaptation_rate
                    ) * current_threshold + adaptation_rate * new_threshold

                    # Apply bounds
                    updated_threshold = max(
                        threshold_info["min_threshold"],
                        min(threshold_info["max_threshold"], updated_threshold),
                    )

                    threshold_info["current_threshold"] = updated_threshold

            def _incremental_update(self, detector_id: str):
                """Perform incremental model update."""
                model = self.models[detector_id]
                buffer_data = list(self.data_buffers[detector_id])

                if len(buffer_data) < 50:  # Need minimum samples for update
                    return

                try:
                    # Create dataset from buffer
                    buffer_df = pd.DataFrame(buffer_data)
                    dataset = Dataset(name="Incremental Update", data=buffer_df)

                    # For this mock, we simulate incremental learning
                    # In a real implementation, this might involve partial_fit or retraining
                    adapter = model["adapter"]

                    # Simple simulation: retrain on recent data
                    # (In practice, this would be more sophisticated)
                    adapter.fit(dataset)

                    model["last_incremental_update"] = datetime.utcnow()

                except Exception:
                    # Incremental update failed, continue with existing model
                    pass

            def get_streaming_stats(self, detector_id: str) -> dict[str, Any]:
                """Get streaming detection statistics."""
                if detector_id not in self.models:
                    raise ValueError(f"Detector {detector_id} not found")

                stats = self.streaming_stats[detector_id]
                threshold_info = self.adaptive_thresholds.get(detector_id, {})

                processing_times = list(stats["processing_times"])

                return {
                    "detector_id": detector_id,
                    "samples_processed": stats["samples_processed"],
                    "anomalies_detected": stats["anomalies_detected"],
                    "anomaly_rate": stats["anomalies_detected"]
                    / max(1, stats["samples_processed"]),
                    "current_threshold": threshold_info.get("current_threshold"),
                    "buffer_size": len(self.data_buffers.get(detector_id, [])),
                    "avg_processing_time": np.mean(processing_times)
                    if processing_times
                    else 0,
                    "max_processing_time": np.max(processing_times)
                    if processing_times
                    else 0,
                    "samples_per_second": 1.0 / np.mean(processing_times)
                    if processing_times
                    else 0,
                    "last_update": stats["last_update"].isoformat()
                    if stats["last_update"]
                    else None,
                }

            def create_streaming_pipeline(
                self, detector_id: str, data_source: Iterator[dict[str, float]]
            ) -> str:
                """Create a streaming pipeline."""
                if detector_id not in self.models:
                    raise ValueError(f"Detector {detector_id} not found")

                pipeline_id = str(uuid.uuid4())

                self.streams[pipeline_id] = {
                    "detector_id": detector_id,
                    "data_source": data_source,
                    "active": False,
                    "results": [],
                    "created_at": datetime.utcnow(),
                }

                return pipeline_id

            def start_streaming_pipeline(self, pipeline_id: str) -> bool:
                """Start streaming pipeline."""
                if pipeline_id not in self.streams:
                    return False

                pipeline = self.streams[pipeline_id]
                pipeline["active"] = True
                pipeline["started_at"] = datetime.utcnow()

                # Start processing in background (simulated)
                def process_stream():
                    detector_id = pipeline["detector_id"]
                    data_source = pipeline["data_source"]

                    try:
                        for sample in data_source:
                            if not pipeline["active"]:
                                break

                            result = self.process_streaming_sample(detector_id, sample)
                            pipeline["results"].append(result)

                            # Simulate real-time processing delay
                            time.sleep(0.01)

                    except Exception:
                        pipeline["active"] = False

                # In a real implementation, this would be a proper background task
                threading.Thread(target=process_stream, daemon=True).start()

                return True

            def stop_streaming_pipeline(self, pipeline_id: str) -> bool:
                """Stop streaming pipeline."""
                if pipeline_id not in self.streams:
                    return False

                pipeline = self.streams[pipeline_id]
                pipeline["active"] = False
                pipeline["stopped_at"] = datetime.utcnow()

                return True

            def get_streaming_results(
                self, pipeline_id: str, limit: int = None
            ) -> list[dict[str, Any]]:
                """Get streaming pipeline results."""
                if pipeline_id not in self.streams:
                    return []

                results = self.streams[pipeline_id]["results"]

                if limit:
                    return results[-limit:]

                return results

        return MockStreamingSDK()

    def test_streaming_detector_creation_and_training(self, mock_streaming_sdk):
        """Test streaming detector creation and training."""
        try:
            sdk = mock_streaming_sdk

            # Create streaming detector
            detector_id = sdk.create_streaming_detector(
                algorithm="IsolationForest",
                contamination=0.1,
                n_estimators=50,
                random_state=42,
                buffer_size=500,
                update_frequency=50,
            )

            assert detector_id is not None

            # Create initial training data
            initial_data = pd.DataFrame(
                {
                    "sensor1": np.random.normal(0, 1, 1000),
                    "sensor2": np.random.normal(0, 1, 1000),
                    "sensor3": np.random.normal(0, 1, 1000),
                }
            )

            # Train streaming detector
            training_result = sdk.train_streaming_detector(detector_id, initial_data)

            assert training_result["status"] == "success"
            assert training_result["detector_id"] == detector_id
            assert training_result["initial_samples"] == 1000
            assert "initial_threshold" in training_result
            assert 0.0 <= training_result["initial_threshold"] <= 1.0

            # Verify detector statistics
            stats = sdk.get_streaming_stats(detector_id)
            assert stats["detector_id"] == detector_id
            assert stats["samples_processed"] == 0  # No streaming samples processed yet
            assert stats["current_threshold"] == training_result["initial_threshold"]

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_single_sample_streaming_detection(self, mock_streaming_sdk):
        """Test processing single streaming samples."""
        try:
            sdk = mock_streaming_sdk

            # Create and train detector
            detector_id = sdk.create_streaming_detector(
                algorithm="IsolationForest",
                contamination=0.1,
                n_estimators=30,
                random_state=42,
            )

            # Training data
            training_data = pd.DataFrame(
                {
                    "feature1": np.random.normal(0, 1, 500),
                    "feature2": np.random.normal(0, 1, 500),
                }
            )

            training_result = sdk.train_streaming_detector(detector_id, training_data)
            assert training_result["status"] == "success"

            # Process streaming samples
            test_samples = [
                {"feature1": 0.5, "feature2": -0.3},  # Normal sample
                {"feature1": 5.0, "feature2": 4.8},  # Anomaly sample
                {"feature1": -0.2, "feature2": 0.7},  # Normal sample
                {"feature1": -6.0, "feature2": -5.5},  # Anomaly sample
            ]

            results = []
            for sample in test_samples:
                result = sdk.process_streaming_sample(detector_id, sample)
                results.append(result)

            # Verify all samples were processed
            assert len(results) == 4

            for i, result in enumerate(results):
                assert result["detector_id"] == detector_id
                assert result["sample"] == test_samples[i]
                assert "anomaly_score" in result
                assert "threshold" in result
                assert "is_anomaly" in result
                assert "processing_time" in result
                assert "timestamp" in result

                # Verify score is valid
                assert 0.0 <= result["anomaly_score"] <= 1.0
                assert 0.0 <= result["threshold"] <= 1.0
                assert isinstance(result["is_anomaly"], bool)
                assert result["processing_time"] > 0

            # Verify anomaly detection patterns
            # Samples with extreme values should generally have higher scores
            normal_scores = [results[0]["anomaly_score"], results[2]["anomaly_score"]]
            anomaly_scores = [results[1]["anomaly_score"], results[3]["anomaly_score"]]

            # This is not guaranteed but should generally be true
            np.mean(normal_scores)
            np.mean(anomaly_scores)

            # At least one anomaly should have higher score than normal samples
            assert max(anomaly_scores) > min(normal_scores)

            # Verify statistics were updated
            stats = sdk.get_streaming_stats(detector_id)
            assert stats["samples_processed"] == 4
            assert stats["anomalies_detected"] >= 0
            assert stats["anomaly_rate"] <= 1.0
            assert stats["avg_processing_time"] > 0

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_batch_streaming_detection(self, mock_streaming_sdk):
        """Test processing batches of streaming samples."""
        try:
            sdk = mock_streaming_sdk

            # Create and train detector
            detector_id = sdk.create_streaming_detector(
                algorithm="IsolationForest",
                contamination=0.1,
                n_estimators=25,
                random_state=42,
            )

            # Training data
            training_data = pd.DataFrame(
                {"x": np.random.normal(0, 1, 300), "y": np.random.normal(0, 1, 300)}
            )

            training_result = sdk.train_streaming_detector(detector_id, training_data)
            assert training_result["status"] == "success"

            # Create batch of streaming samples
            batch_size = 20
            batch_samples = []

            for i in range(batch_size):
                if i < 15:
                    # Normal samples
                    sample = {"x": np.random.normal(0, 1), "y": np.random.normal(0, 1)}
                else:
                    # Anomaly samples
                    sample = {
                        "x": np.random.normal(3, 0.5),
                        "y": np.random.normal(3, 0.5),
                    }
                batch_samples.append(sample)

            # Process batch
            batch_results = sdk.process_streaming_batch(detector_id, batch_samples)

            # Verify batch processing
            assert len(batch_results) == batch_size

            for i, result in enumerate(batch_results):
                assert result["detector_id"] == detector_id
                assert result["sample"] == batch_samples[i]
                assert 0.0 <= result["anomaly_score"] <= 1.0
                assert isinstance(result["is_anomaly"], bool)

            # Verify statistics
            stats = sdk.get_streaming_stats(detector_id)
            assert stats["samples_processed"] == batch_size
            assert stats["anomalies_detected"] <= batch_size
            assert 0.0 <= stats["anomaly_rate"] <= 1.0

            # Processing should be reasonably fast
            assert stats["avg_processing_time"] < 1.0  # Less than 1 second per sample
            assert stats["samples_per_second"] > 1.0  # At least 1 sample per second

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_adaptive_threshold_adjustment(self, mock_streaming_sdk):
        """Test adaptive threshold adjustment during streaming."""
        try:
            sdk = mock_streaming_sdk

            # Create detector with adaptive thresholding
            detector_id = sdk.create_streaming_detector(
                algorithm="IsolationForest",
                contamination=0.1,
                n_estimators=30,
                random_state=42,
            )

            # Training data (normal distribution)
            training_data = pd.DataFrame({"value": np.random.normal(0, 1, 500)})

            training_result = sdk.train_streaming_detector(detector_id, training_data)
            training_result["initial_threshold"]

            # Process samples from shifted distribution (concept drift)
            drift_samples = []
            for i in range(100):
                # Gradually shift the distribution
                shift = i * 0.02  # Gradual shift
                sample = {"value": np.random.normal(shift, 1)}
                drift_samples.append(sample)

            # Process samples and track threshold changes
            thresholds = []
            for sample in drift_samples:
                result = sdk.process_streaming_sample(detector_id, sample)
                thresholds.append(result["threshold"])

            # Verify adaptive threshold behavior
            assert len(thresholds) == 100

            # Threshold should adapt to the changing distribution
            early_threshold = np.mean(thresholds[:20])
            late_threshold = np.mean(thresholds[-20:])

            # With concept drift, threshold should change
            threshold_change = abs(late_threshold - early_threshold)
            assert threshold_change > 0  # Some adaptation should occur

            # Threshold should remain within reasonable bounds
            for threshold in thresholds:
                assert 0.0 <= threshold <= 1.0

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_incremental_model_updates(self, mock_streaming_sdk):
        """Test incremental model updates during streaming."""
        try:
            sdk = mock_streaming_sdk

            # Create detector with frequent updates
            detector_id = sdk.create_streaming_detector(
                algorithm="IsolationForest",
                contamination=0.1,
                n_estimators=20,
                random_state=42,
                update_frequency=25,  # Update every 25 samples
            )

            # Initial training
            training_data = pd.DataFrame({"sensor": np.random.normal(0, 1, 200)})

            training_result = sdk.train_streaming_detector(detector_id, training_data)
            assert training_result["status"] == "success"

            # Process streaming samples to trigger updates
            samples_with_updates = []

            for i in range(75):  # Should trigger 3 updates (at 25, 50, 75)
                sample = {"sensor": np.random.normal(0, 1)}
                result = sdk.process_streaming_sample(detector_id, sample)

                if result.get("model_updated", False):
                    samples_with_updates.append(i + 1)

            # Verify incremental updates occurred
            expected_updates = [25, 50, 75]  # At these sample counts
            assert len(samples_with_updates) == 3

            for expected, actual in zip(expected_updates, samples_with_updates, strict=False):
                assert actual == expected

            # Verify final statistics
            stats = sdk.get_streaming_stats(detector_id)
            assert stats["samples_processed"] == 75

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_streaming_pipeline_management(self, mock_streaming_sdk):
        """Test streaming pipeline creation and management."""
        try:
            sdk = mock_streaming_sdk

            # Create and train detector
            detector_id = sdk.create_streaming_detector(
                algorithm="IsolationForest",
                contamination=0.1,
                n_estimators=20,
                random_state=42,
            )

            training_data = pd.DataFrame(
                {
                    "temperature": np.random.normal(20, 5, 300),
                    "pressure": np.random.normal(1013, 20, 300),
                }
            )

            training_result = sdk.train_streaming_detector(detector_id, training_data)
            assert training_result["status"] == "success"

            # Create data source generator
            def data_source():
                for _i in range(50):
                    yield {
                        "temperature": np.random.normal(20, 5),
                        "pressure": np.random.normal(1013, 20),
                    }
                    time.sleep(0.001)  # Simulate real-time data

            # Create streaming pipeline
            pipeline_id = sdk.create_streaming_pipeline(detector_id, data_source())
            assert pipeline_id is not None

            # Start pipeline
            start_success = sdk.start_streaming_pipeline(pipeline_id)
            assert start_success

            # Let pipeline process some data
            time.sleep(0.5)  # Allow processing time

            # Get intermediate results
            intermediate_results = sdk.get_streaming_results(pipeline_id, limit=10)
            assert len(intermediate_results) > 0
            assert len(intermediate_results) <= 10

            # Verify result structure
            for result in intermediate_results:
                assert "detector_id" in result
                assert "anomaly_score" in result
                assert "is_anomaly" in result
                assert "timestamp" in result

            # Stop pipeline
            stop_success = sdk.stop_streaming_pipeline(pipeline_id)
            assert stop_success

            # Get final results
            final_results = sdk.get_streaming_results(pipeline_id)
            assert len(final_results) > len(intermediate_results)

            # Verify all results have valid structure
            for result in final_results:
                assert result["detector_id"] == detector_id
                assert 0.0 <= result["anomaly_score"] <= 1.0
                assert isinstance(result["is_anomaly"], bool)

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_streaming_performance_monitoring(self, mock_streaming_sdk):
        """Test streaming performance monitoring and metrics."""
        try:
            sdk = mock_streaming_sdk

            # Create high-performance detector
            detector_id = sdk.create_streaming_detector(
                algorithm="IsolationForest",
                contamination=0.1,
                n_estimators=15,  # Smaller for faster processing
                random_state=42,
            )

            # Quick training
            training_data = pd.DataFrame(
                {
                    "metric1": np.random.normal(0, 1, 200),
                    "metric2": np.random.normal(0, 1, 200),
                }
            )

            training_result = sdk.train_streaming_detector(detector_id, training_data)
            assert training_result["status"] == "success"

            # Process samples and monitor performance
            start_time = time.time()
            num_samples = 100

            for _i in range(num_samples):
                sample = {
                    "metric1": np.random.normal(0, 1),
                    "metric2": np.random.normal(0, 1),
                }
                result = sdk.process_streaming_sample(detector_id, sample)

                # Verify each sample was processed successfully
                assert "anomaly_score" in result
                assert result["processing_time"] > 0

            total_time = time.time() - start_time

            # Get performance statistics
            stats = sdk.get_streaming_stats(detector_id)

            # Verify performance metrics
            assert stats["samples_processed"] == num_samples
            assert stats["avg_processing_time"] > 0
            assert stats["max_processing_time"] >= stats["avg_processing_time"]
            assert stats["samples_per_second"] > 0

            # Performance should be reasonable for streaming
            assert stats["avg_processing_time"] < 0.1  # Less than 100ms per sample
            assert stats["samples_per_second"] > 10  # At least 10 samples per second

            # Overall throughput check
            overall_throughput = num_samples / total_time
            assert overall_throughput > 5  # At least 5 samples per second overall

        except ImportError:
            pytest.skip("scikit-learn not available")
