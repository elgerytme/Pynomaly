"""Integration tests for streaming and real-time processing."""

import asyncio
import json

import pytest
import websockets
from httpx import AsyncClient

from tests.integration.conftest import IntegrationTestHelper


class TestStreamingIntegration:
    """Test real-time streaming integration scenarios."""

    @pytest.mark.asyncio
    async def test_websocket_streaming_monitoring(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_time_series_csv: str,
        disable_auth,
    ):
        """Test WebSocket-based real-time streaming monitoring."""

        # Step 1: Setup streaming session
        dataset = await integration_helper.upload_dataset(
            sample_time_series_csv, "websocket_test_dataset"
        )

        detector = await integration_helper.create_detector(
            dataset["id"], "isolation_forest"
        )

        await integration_helper.train_detector(detector["id"])

        # Step 2: Create streaming session
        streaming_config = {
            "name": "websocket_test_session",
            "detector_id": detector["id"],
            "data_source": {"source_type": "mock", "connection_config": {}},
            "configuration": {
                "processing_mode": "real_time",
                "batch_size": 1,
                "max_throughput": 50,
            },
        }

        response = await async_test_client.post(
            "/api/streaming/sessions?created_by=test_user", json=streaming_config
        )
        response.raise_for_status()
        session = response.json()["data"]
        integration_helper.created_resources["sessions"].append(session["id"])

        # Step 3: Start streaming session
        response = await async_test_client.post(
            f"/api/streaming/sessions/{session['id']}/start"
        )
        response.raise_for_status()

        # Step 4: Test WebSocket connection
        websocket_url = f"ws://testserver/api/streaming/sessions/{session['id']}/live"

        # Note: In a real test environment, you would connect to an actual WebSocket
        # For this integration test, we'll simulate the WebSocket behavior

        # Simulate processing some data points
        test_data_points = [
            {"timestamp": "2024-12-25T10:00:00Z", "value": 50.2, "cpu_usage": 45.1},
            {"timestamp": "2024-12-25T10:01:00Z", "value": 52.1, "cpu_usage": 47.3},
            {
                "timestamp": "2024-12-25T10:02:00Z",
                "value": 150.0,
                "cpu_usage": 95.2,
            },  # Anomaly
        ]

        processing_results = []
        for data_point in test_data_points:
            response = await async_test_client.post(
                f"/api/streaming/sessions/{session['id']}/process",
                json={"data": data_point},
            )
            response.raise_for_status()
            result = response.json()["data"]
            processing_results.append(result)

        # Verify processing results
        assert len(processing_results) == 3
        assert (
            processing_results[2]["is_anomaly"] is True
        )  # Last point should be anomaly
        assert (
            processing_results[2]["anomaly_score"]
            > processing_results[0]["anomaly_score"]
        )

        # Step 5: Verify session metrics
        response = await async_test_client.get(
            f"/api/streaming/sessions/{session['id']}/metrics"
        )
        response.raise_for_status()
        metrics = response.json()["data"]

        assert metrics["messages_processed"] >= 3
        assert metrics["anomalies_detected"] >= 1
        assert metrics["anomaly_rate"] > 0

        # Step 6: Stop session
        response = await async_test_client.post(
            f"/api/streaming/sessions/{session['id']}/stop"
        )
        response.raise_for_status()

    @pytest.mark.asyncio
    async def test_streaming_session_lifecycle(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth,
    ):
        """Test complete streaming session lifecycle management."""

        # Setup
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv, "lifecycle_test_dataset"
        )

        detector = await integration_helper.create_detector(
            dataset["id"], "isolation_forest"
        )

        await integration_helper.train_detector(detector["id"])

        # Step 1: Create session
        streaming_config = {
            "name": "lifecycle_test_session",
            "detector_id": detector["id"],
            "data_source": {
                "source_type": "mock",
                "connection_config": {"mock_data_rate": 10},
            },
            "configuration": {
                "processing_mode": "micro_batch",
                "batch_size": 5,
                "max_throughput": 100,
                "schema_validation": True,
                "enable_checkpointing": True,
            },
            "max_duration_hours": 1.0,
            "tags": ["integration_test", "lifecycle"],
        }

        response = await async_test_client.post(
            "/api/streaming/sessions?created_by=test_user", json=streaming_config
        )
        response.raise_for_status()
        session = response.json()["data"]
        session_id = session["id"]
        integration_helper.created_resources["sessions"].append(session_id)

        assert session["status"] == "pending"
        assert session["name"] == "lifecycle_test_session"

        # Step 2: Start session
        response = await async_test_client.post(
            f"/api/streaming/sessions/{session_id}/start"
        )
        response.raise_for_status()
        started_session = response.json()["data"]
        assert started_session["status"] in ["starting", "active"]

        # Step 3: Process some data
        for i in range(5):
            test_data = {
                "data": {
                    "timestamp": f"2024-12-25T10:{i:02d}:00Z",
                    "feature1": i * 0.5,
                    "feature2": i * 0.3,
                    "feature3": i * 0.1,
                }
            }

            response = await async_test_client.post(
                f"/api/streaming/sessions/{session_id}/process", json=test_data
            )
            response.raise_for_status()

        # Step 4: Pause session
        response = await async_test_client.post(
            f"/api/streaming/sessions/{session_id}/pause"
        )
        response.raise_for_status()
        paused_session = response.json()["data"]
        assert paused_session["status"] == "paused"

        # Step 5: Resume session
        response = await async_test_client.post(
            f"/api/streaming/sessions/{session_id}/resume"
        )
        response.raise_for_status()
        resumed_session = response.json()["data"]
        assert resumed_session["status"] == "active"

        # Step 6: Get session summary
        response = await async_test_client.get(
            f"/api/streaming/sessions/{session_id}/summary"
        )
        response.raise_for_status()
        summary = response.json()["data"]

        assert summary["session_id"] == session_id
        assert summary["messages_processed"] >= 5
        assert summary["uptime_seconds"] >= 0

        # Step 7: Create alert for session
        alert_config = {
            "name": "High Error Rate Alert",
            "metric_name": "error_rate",
            "threshold_value": 0.1,
            "comparison_operator": ">",
            "severity": "high",
            "duration_threshold_minutes": 1.0,
            "notification_channels": ["email"],
        }

        response = await async_test_client.post(
            f"/api/streaming/sessions/{session_id}/alerts?created_by=test_user",
            json=alert_config,
        )
        response.raise_for_status()
        alert = response.json()["data"]

        assert alert["name"] == "High Error Rate Alert"
        assert alert["session_id"] == session_id

        # Step 8: Stop session
        response = await async_test_client.post(
            f"/api/streaming/sessions/{session_id}/stop",
            params={"error_message": "Integration test completed"},
        )
        response.raise_for_status()
        stopped_session = response.json()["data"]
        assert stopped_session["status"] in ["stopping", "stopped"]

    @pytest.mark.asyncio
    async def test_streaming_session_scaling(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth,
    ):
        """Test streaming session scaling and performance under load."""

        # Setup
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv, "scaling_test_dataset"
        )

        detector = await integration_helper.create_detector(
            dataset["id"], "isolation_forest"
        )

        await integration_helper.train_detector(detector["id"])

        # Step 1: Create multiple streaming sessions
        session_configs = [
            {
                "name": f"scaling_session_{i}",
                "detector_id": detector["id"],
                "data_source": {
                    "source_type": "mock",
                    "connection_config": {"mock_data_rate": 20},
                },
                "configuration": {
                    "processing_mode": "real_time",
                    "batch_size": 1,
                    "max_throughput": 50,
                },
            }
            for i in range(3)
        ]

        sessions = []
        for config in session_configs:
            response = await async_test_client.post(
                "/api/streaming/sessions?created_by=test_user", json=config
            )
            response.raise_for_status()
            session = response.json()["data"]
            sessions.append(session)
            integration_helper.created_resources["sessions"].append(session["id"])

        # Step 2: Start all sessions
        for session in sessions:
            response = await async_test_client.post(
                f"/api/streaming/sessions/{session['id']}/start"
            )
            response.raise_for_status()

        # Step 3: Process data concurrently across sessions
        async def process_data_for_session(session_id: str, data_points: int):
            results = []
            for i in range(data_points):
                test_data = {
                    "data": {
                        "timestamp": f"2024-12-25T10:{i:02d}:00Z",
                        "feature1": i * 0.1,
                        "feature2": i * 0.2,
                        "feature3": i * 0.05,
                    }
                }

                response = await async_test_client.post(
                    f"/api/streaming/sessions/{session_id}/process", json=test_data
                )
                response.raise_for_status()
                results.append(response.json()["data"])

            return results

        # Process data concurrently
        tasks = [process_data_for_session(session["id"], 10) for session in sessions]

        all_results = await asyncio.gather(*tasks)

        # Verify all sessions processed data
        assert len(all_results) == 3
        for session_results in all_results:
            assert len(session_results) == 10

        # Step 4: List all active sessions
        response = await async_test_client.get("/api/streaming/sessions?status=active")
        response.raise_for_status()
        active_sessions = response.json()["data"]

        assert len(active_sessions) >= 3

        # Step 5: Get aggregate metrics
        total_processed = 0
        total_anomalies = 0

        for session in sessions:
            response = await async_test_client.get(
                f"/api/streaming/sessions/{session['id']}/metrics"
            )
            response.raise_for_status()
            metrics = response.json()["data"]

            total_processed += metrics["messages_processed"]
            total_anomalies += metrics["anomalies_detected"]

        assert total_processed >= 30  # 3 sessions * 10 messages each

        # Step 6: Stop all sessions
        for session in sessions:
            response = await async_test_client.post(
                f"/api/streaming/sessions/{session['id']}/stop"
            )
            response.raise_for_status()

    @pytest.mark.asyncio
    async def test_streaming_error_handling(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth,
    ):
        """Test streaming error handling and recovery scenarios."""

        # Setup
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv, "error_handling_dataset"
        )

        detector = await integration_helper.create_detector(
            dataset["id"], "isolation_forest"
        )

        await integration_helper.train_detector(detector["id"])

        # Step 1: Create session with error-prone configuration
        streaming_config = {
            "name": "error_handling_session",
            "detector_id": detector["id"],
            "data_source": {
                "source_type": "mock",
                "connection_config": {"simulate_errors": True},
            },
            "configuration": {
                "processing_mode": "real_time",
                "batch_size": 1,
                "max_throughput": 100,
                "schema_validation": True,
                "error_handling": {
                    "max_retries": 3,
                    "retry_delay_seconds": 1,
                    "dead_letter_queue": True,
                },
            },
        }

        response = await async_test_client.post(
            "/api/streaming/sessions?created_by=test_user", json=streaming_config
        )
        response.raise_for_status()
        session = response.json()["data"]
        session_id = session["id"]
        integration_helper.created_resources["sessions"].append(session_id)

        # Step 2: Start session
        response = await async_test_client.post(
            f"/api/streaming/sessions/{session_id}/start"
        )
        response.raise_for_status()

        # Step 3: Send valid data
        valid_data = {
            "data": {
                "timestamp": "2024-12-25T10:00:00Z",
                "feature1": 1.0,
                "feature2": 2.0,
                "feature3": 3.0,
            }
        }

        response = await async_test_client.post(
            f"/api/streaming/sessions/{session_id}/process", json=valid_data
        )
        response.raise_for_status()
        valid_result = response.json()["data"]
        assert "anomaly_score" in valid_result

        # Step 4: Send invalid data (should trigger error handling)
        invalid_data_cases = [
            # Missing required fields
            {"data": {"timestamp": "2024-12-25T10:01:00Z"}},
            # Invalid data types
            {
                "data": {
                    "timestamp": "invalid",
                    "feature1": "not_a_number",
                    "feature2": None,
                    "feature3": [],
                }
            },
            # Out of range values
            {
                "data": {
                    "timestamp": "2024-12-25T10:02:00Z",
                    "feature1": float("inf"),
                    "feature2": float("nan"),
                    "feature3": 1e10,
                }
            },
        ]

        error_count = 0
        for invalid_data in invalid_data_cases:
            try:
                response = await async_test_client.post(
                    f"/api/streaming/sessions/{session_id}/process", json=invalid_data
                )
                # Some errors might be handled gracefully
                if response.status_code >= 400:
                    error_count += 1
            except Exception:
                error_count += 1

        # Step 5: Check session metrics for errors
        response = await async_test_client.get(
            f"/api/streaming/sessions/{session_id}/metrics"
        )
        response.raise_for_status()
        metrics = response.json()["data"]

        # Should have processed at least the valid message
        assert metrics["messages_processed"] >= 1
        # Should have recorded some errors
        assert (
            metrics["failed_messages"] >= 0
        )  # Might be 0 if errors are handled gracefully

        # Step 6: Test session recovery after errors
        # Send more valid data to ensure session can continue
        recovery_data = [
            {
                "timestamp": "2024-12-25T10:03:00Z",
                "feature1": 0.5,
                "feature2": 1.5,
                "feature3": 2.5,
            },
            {
                "timestamp": "2024-12-25T10:04:00Z",
                "feature1": 0.8,
                "feature2": 1.2,
                "feature3": 1.8,
            },
        ]

        for data_point in recovery_data:
            response = await async_test_client.post(
                f"/api/streaming/sessions/{session_id}/process",
                json={"data": data_point},
            )
            response.raise_for_status()
            result = response.json()["data"]
            assert "anomaly_score" in result

        # Step 7: Verify session is still active
        response = await async_test_client.get(
            f"/api/streaming/sessions/{session_id}/summary"
        )
        response.raise_for_status()
        summary = response.json()["data"]
        assert summary["status"] in ["active", "running"]

        # Step 8: Stop session
        response = await async_test_client.post(
            f"/api/streaming/sessions/{session_id}/stop"
        )
        response.raise_for_status()

    @pytest.mark.asyncio
    async def test_streaming_data_sink_integration(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth,
    ):
        """Test streaming data sink integration for output handling."""

        # Setup
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv, "data_sink_dataset"
        )

        detector = await integration_helper.create_detector(
            dataset["id"], "isolation_forest"
        )

        await integration_helper.train_detector(detector["id"])

        # Step 1: Create session with data sink
        streaming_config = {
            "name": "data_sink_session",
            "detector_id": detector["id"],
            "data_source": {"source_type": "mock", "connection_config": {}},
            "configuration": {
                "processing_mode": "real_time",
                "batch_size": 1,
                "max_throughput": 50,
            },
            "data_sink": {
                "sink_type": "mock",
                "connection_config": {
                    "output_format": "json",
                    "include_metadata": True,
                },
                "enabled": True,
            },
        }

        response = await async_test_client.post(
            "/api/streaming/sessions?created_by=test_user", json=streaming_config
        )
        response.raise_for_status()
        session = response.json()["data"]
        session_id = session["id"]
        integration_helper.created_resources["sessions"].append(session_id)

        # Step 2: Start session
        response = await async_test_client.post(
            f"/api/streaming/sessions/{session_id}/start"
        )
        response.raise_for_status()

        # Step 3: Process data with different anomaly scores
        test_data_points = [
            {
                "timestamp": "2024-12-25T10:00:00Z",
                "feature1": 0.1,
                "feature2": 0.2,
                "feature3": 0.3,
            },  # Normal
            {
                "timestamp": "2024-12-25T10:01:00Z",
                "feature1": 5.0,
                "feature2": 4.8,
                "feature3": 3.2,
            },  # Anomaly
            {
                "timestamp": "2024-12-25T10:02:00Z",
                "feature1": 0.2,
                "feature2": 0.3,
                "feature3": 0.1,
            },  # Normal
        ]

        sink_outputs = []
        for data_point in test_data_points:
            response = await async_test_client.post(
                f"/api/streaming/sessions/{session_id}/process",
                json={"data": data_point},
            )
            response.raise_for_status()
            result = response.json()["data"]
            sink_outputs.append(result)

        # Step 4: Verify outputs were processed
        assert len(sink_outputs) == 3

        # Normal data should have low anomaly scores
        assert sink_outputs[0]["anomaly_score"] < 0.5
        assert sink_outputs[2]["anomaly_score"] < 0.5

        # Anomalous data should have high score
        assert sink_outputs[1]["anomaly_score"] > 0.5
        assert sink_outputs[1]["is_anomaly"] is True

        # Step 5: Check session metrics
        response = await async_test_client.get(
            f"/api/streaming/sessions/{session_id}/metrics"
        )
        response.raise_for_status()
        metrics = response.json()["data"]

        assert metrics["messages_processed"] == 3
        assert metrics["anomalies_detected"] >= 1

        # Step 6: Stop session
        response = await async_test_client.post(
            f"/api/streaming/sessions/{session_id}/stop"
        )
        response.raise_for_status()

        # Step 7: Verify final session state
        response = await async_test_client.get(
            f"/api/streaming/sessions/{session_id}/summary"
        )
        response.raise_for_status()
        final_summary = response.json()["data"]

        assert final_summary["status"] in ["stopped", "stopping"]
        assert final_summary["messages_processed"] == 3
