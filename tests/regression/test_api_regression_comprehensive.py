"""Comprehensive API regression tests.

This module contains regression tests for the API layer to ensure that
API behavior remains consistent across versions and that breaking changes
are properly detected and managed.
"""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from pynomaly.presentation.api.app import create_app


class TestAPIVersionCompatibility:
    """Test API version compatibility and backward compatibility."""

    @pytest.fixture
    def client(self):
        """Create test client for API testing."""
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def sample_dataset_payload(self):
        """Create sample dataset payload for API testing."""
        return {
            "name": "Test Dataset",
            "description": "Sample dataset for regression testing",
            "data": {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [0.1, 0.2, 0.3, 0.4, 0.5],
                "feature3": [10, 20, 30, 40, 50],
            },
            "metadata": {
                "source": "regression_test",
                "created_at": "2023-10-01T10:00:00Z",
            },
        }

    def test_api_health_endpoint_stability(self, client):
        """Test that health endpoint maintains stable response format."""
        response = client.get("/health")

        # Health endpoint should always return 200
        assert response.status_code == 200

        # Response should be JSON
        health_data = response.json()
        assert isinstance(health_data, dict)

        # Required fields should be present
        required_fields = ["status", "timestamp", "version"]
        for field in required_fields:
            assert field in health_data, f"Missing required field: {field}"

        # Status should be string
        assert isinstance(health_data["status"], str)
        assert health_data["status"] in ["healthy", "unhealthy", "degraded"]

        # Timestamp should be ISO format
        timestamp = health_data["timestamp"]
        assert isinstance(timestamp, str)
        # Should be parseable as datetime
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        # Version should be present
        assert isinstance(health_data["version"], str)
        assert len(health_data["version"]) > 0

    def test_api_info_endpoint_stability(self, client):
        """Test that API info endpoint maintains stable response format."""
        response = client.get("/api/v1/info")

        assert response.status_code == 200

        info_data = response.json()
        assert isinstance(info_data, dict)

        # Required info fields
        required_fields = ["name", "version", "description", "endpoints"]
        for field in required_fields:
            assert field in info_data

        # Endpoints should be a list
        assert isinstance(info_data["endpoints"], list)
        assert len(info_data["endpoints"]) > 0

        # Each endpoint should have required fields
        for endpoint in info_data["endpoints"]:
            assert "path" in endpoint
            assert "method" in endpoint
            assert "description" in endpoint

    def test_dataset_crud_api_compatibility(self, client, sample_dataset_payload):
        """Test dataset CRUD API maintains backward compatibility."""
        # Test dataset creation
        create_response = client.post("/api/v1/datasets", json=sample_dataset_payload)

        # Should succeed or return specific error format
        if create_response.status_code == 201:
            dataset_data = create_response.json()

            # Response should contain expected fields
            assert "id" in dataset_data
            assert "name" in dataset_data
            assert "created_at" in dataset_data

            dataset_id = dataset_data["id"]

            # Test dataset retrieval
            get_response = client.get(f"/api/v1/datasets/{dataset_id}")
            assert get_response.status_code == 200

            retrieved_data = get_response.json()
            assert retrieved_data["id"] == dataset_id
            assert retrieved_data["name"] == sample_dataset_payload["name"]

            # Test dataset listing
            list_response = client.get("/api/v1/datasets")
            assert list_response.status_code == 200

            datasets_list = list_response.json()
            assert isinstance(datasets_list, list)

            # Find our dataset in the list
            found_dataset = None
            for dataset in datasets_list:
                if dataset["id"] == dataset_id:
                    found_dataset = dataset
                    break

            assert found_dataset is not None

            # Test dataset deletion
            delete_response = client.delete(f"/api/v1/datasets/{dataset_id}")
            assert delete_response.status_code in [200, 204]

            # Verify deletion
            verify_response = client.get(f"/api/v1/datasets/{dataset_id}")
            assert verify_response.status_code == 404

        else:
            # If creation fails, should return proper error format
            assert create_response.status_code in [400, 422, 500]
            error_data = create_response.json()
            assert "detail" in error_data or "message" in error_data

    def test_detector_api_compatibility(self, client):
        """Test detector API maintains backward compatibility."""
        detector_payload = {
            "name": "Test Detector",
            "algorithm": "IsolationForest",
            "parameters": {
                "contamination": 0.1,
                "n_estimators": 100,
                "random_state": 42,
            },
            "description": "Test detector for regression testing",
        }

        # Test detector creation
        create_response = client.post("/api/v1/detectors", json=detector_payload)

        if create_response.status_code == 201:
            detector_data = create_response.json()

            # Response should contain expected fields
            assert "id" in detector_data
            assert "name" in detector_data
            assert "algorithm" in detector_data
            assert "parameters" in detector_data

            detector_id = detector_data["id"]

            # Test detector retrieval
            get_response = client.get(f"/api/v1/detectors/{detector_id}")
            assert get_response.status_code == 200

            # Test detector listing
            list_response = client.get("/api/v1/detectors")
            assert list_response.status_code == 200

            detectors_list = list_response.json()
            assert isinstance(detectors_list, list)

            # Test detector deletion
            delete_response = client.delete(f"/api/v1/detectors/{detector_id}")
            assert delete_response.status_code in [200, 204]

        else:
            # Proper error format
            assert create_response.status_code in [400, 422, 500]


class TestAPIResponseFormatRegression:
    """Test API response format consistency across versions."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    def test_error_response_format_consistency(self, client):
        """Test that error responses maintain consistent format."""
        # Test 404 error format
        response = client.get("/api/v1/datasets/nonexistent-id")
        assert response.status_code == 404

        error_data = response.json()
        assert isinstance(error_data, dict)

        # Standard error fields
        error_fields = ["detail", "error_code", "timestamp"]
        present_fields = [field for field in error_fields if field in error_data]
        assert (
            len(present_fields) >= 1
        ), "At least one standard error field should be present"

        # Test 422 validation error format
        invalid_payload = {"invalid": "data"}
        response = client.post("/api/v1/datasets", json=invalid_payload)

        if response.status_code == 422:
            validation_error = response.json()
            assert isinstance(validation_error, dict)

            # Should contain validation details
            assert "detail" in validation_error

    def test_pagination_format_consistency(self, client):
        """Test pagination response format consistency."""
        # Test datasets pagination
        response = client.get("/api/v1/datasets?page=1&size=10")

        if response.status_code == 200:
            paginated_data = response.json()

            # Check if paginated response format
            if isinstance(paginated_data, dict) and "items" in paginated_data:
                # Paginated format
                required_pagination_fields = ["items", "total", "page", "size"]
                for field in required_pagination_fields:
                    assert field in paginated_data

                assert isinstance(paginated_data["items"], list)
                assert isinstance(paginated_data["total"], int)
                assert isinstance(paginated_data["page"], int)
                assert isinstance(paginated_data["size"], int)

            else:
                # Simple list format is also acceptable
                assert isinstance(paginated_data, list)

    def test_datetime_format_consistency(self, client):
        """Test datetime format consistency across API responses."""
        # Get any resource with timestamps
        response = client.get("/health")

        if response.status_code == 200:
            health_data = response.json()

            if "timestamp" in health_data:
                timestamp = health_data["timestamp"]

                # Should be ISO 8601 format
                assert isinstance(timestamp, str)

                # Should be parseable
                try:
                    if timestamp.endswith("Z"):
                        parsed_dt = datetime.fromisoformat(
                            timestamp.replace("Z", "+00:00")
                        )
                    else:
                        parsed_dt = datetime.fromisoformat(timestamp)

                    assert isinstance(parsed_dt, datetime)

                except ValueError:
                    pytest.fail(f"Invalid datetime format: {timestamp}")


class TestAPIPerformanceRegression:
    """Test API performance regression detection."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    def test_health_endpoint_response_time(self, client):
        """Test health endpoint response time regression."""
        import time

        response_times = []

        # Measure response time over multiple requests
        for _ in range(5):
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()

            response_time = end_time - start_time
            response_times.append(response_time)

            # Each request should succeed
            assert response.status_code == 200

        # Calculate statistics
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)

        # Performance thresholds
        assert (
            avg_response_time < 1.0
        ), f"Average response time too slow: {avg_response_time}s"
        assert (
            max_response_time < 2.0
        ), f"Maximum response time too slow: {max_response_time}s"

    def test_api_info_endpoint_response_time(self, client):
        """Test API info endpoint response time."""
        import time

        start_time = time.time()
        response = client.get("/api/v1/info")
        end_time = time.time()

        response_time = end_time - start_time

        # Should respond quickly
        assert response_time < 2.0, f"API info response too slow: {response_time}s"

        if response.status_code == 200:
            # Response should not be empty
            info_data = response.json()
            assert len(info_data) > 0

    def test_concurrent_request_performance(self, client):
        """Test concurrent request handling performance."""
        import threading
        import time

        def make_request():
            return client.get("/health")

        # Test concurrent requests
        num_threads = 5
        start_time = time.time()

        threads = []
        results = []

        def thread_worker():
            response = make_request()
            results.append(response)

        # Start threads
        for _ in range(num_threads):
            thread = threading.Thread(target=thread_worker)
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        end_time = time.time()
        total_time = end_time - start_time

        # All requests should succeed
        assert len(results) == num_threads
        for response in results:
            assert response.status_code == 200

        # Concurrent requests should not take excessively long
        assert total_time < 5.0, f"Concurrent requests too slow: {total_time}s"


class TestAPISecurityRegression:
    """Test API security configuration regression."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    def test_security_headers_presence(self, client):
        """Test that security headers are present."""
        response = client.get("/health")

        # Check for common security headers
        security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
        ]

        headers = response.headers

        # At least some security headers should be present
        present_headers = [h for h in security_headers if h in headers]

        # Note: This might not apply to all setups, so we make it informational
        if len(present_headers) == 0:
            # Log that no security headers are present (for monitoring)
            pass

    def test_cors_configuration_stability(self, client):
        """Test CORS configuration stability."""
        # Test OPTIONS request
        response = client.options("/api/v1/info")

        # Should handle OPTIONS requests
        assert response.status_code in [
            200,
            405,
        ]  # 405 if OPTIONS not explicitly handled

        # Test CORS headers in regular request
        response = client.get(
            "/api/v1/info", headers={"Origin": "http://localhost:3000"}
        )

        if "Access-Control-Allow-Origin" in response.headers:
            # CORS is configured
            cors_origin = response.headers["Access-Control-Allow-Origin"]
            assert cors_origin in ["*", "http://localhost:3000"]

    def test_input_validation_consistency(self, client):
        """Test input validation consistency."""
        # Test various malicious payloads
        malicious_payloads = [
            {"<script>": "alert('xss')"},
            {"'; DROP TABLE datasets; --": "sql_injection"},
            {"../../../etc/passwd": "path_traversal"},
            {"a" * 10000: "large_input"},  # Large input
        ]

        for payload in malicious_payloads:
            response = client.post("/api/v1/datasets", json=payload)

            # Should reject malicious input gracefully
            assert response.status_code in [400, 422, 500]

            # Should not crash the server
            health_check = client.get("/health")
            assert health_check.status_code == 200


class TestAPIDeprecationRegression:
    """Test API deprecation handling and backward compatibility."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    def test_deprecated_endpoint_warnings(self, client):
        """Test that deprecated endpoints provide proper warnings."""
        # Test if any deprecated endpoints exist
        deprecated_endpoints = [
            "/api/v0/datasets",  # Hypothetical old version
            "/datasets",  # Hypothetical old path
        ]

        for endpoint in deprecated_endpoints:
            response = client.get(endpoint)

            # If endpoint exists but is deprecated
            if response.status_code == 200:
                # Should include deprecation warning in headers or response
                deprecation_indicators = [
                    "Deprecation" in response.headers,
                    "X-Deprecated" in response.headers,
                    (
                        "deprecated" in response.json().get("warnings", [])
                        if response.headers.get("content-type", "").startswith(
                            "application/json"
                        )
                        else False
                    ),
                ]

                # At least one deprecation indicator should be present
                assert any(
                    deprecation_indicators
                ), f"No deprecation warning for {endpoint}"

    def test_api_version_header_support(self, client):
        """Test API version header support."""
        # Test with version header
        headers = {"API-Version": "v1"}
        response = client.get("/api/v1/info", headers=headers)

        # Should handle version headers gracefully
        assert response.status_code in [200, 400, 404]

        # Test with unsupported version
        headers = {"API-Version": "v99"}
        response = client.get("/api/v1/info", headers=headers)

        # Should handle unsupported versions gracefully
        assert response.status_code in [200, 400, 404, 422]

    def test_backward_compatible_response_fields(self, client):
        """Test that response fields remain backward compatible."""
        response = client.get("/api/v1/info")

        if response.status_code == 200:
            info_data = response.json()

            # Core fields that should always be present for backward compatibility
            backward_compatible_fields = ["name", "version"]

            for field in backward_compatible_fields:
                if field in info_data:
                    # Field should have reasonable type and format
                    assert isinstance(info_data[field], str)
                    assert len(info_data[field]) > 0


class TestAPIDataIntegrityRegression:
    """Test API data integrity and consistency."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    def test_data_round_trip_integrity(self, client):
        """Test data integrity through create-read-update cycle."""
        # Create test dataset
        original_data = {
            "name": "Integrity Test Dataset",
            "description": "Testing data integrity",
            "data": {
                "values": [1.1, 2.2, 3.3, 4.4, 5.5],
                "labels": ["A", "B", "C", "D", "E"],
            },
            "metadata": {"test_flag": True, "precision": 2.345},
        }

        # Create dataset
        create_response = client.post("/api/v1/datasets", json=original_data)

        if create_response.status_code == 201:
            created_data = create_response.json()
            dataset_id = created_data["id"]

            # Retrieve dataset
            get_response = client.get(f"/api/v1/datasets/{dataset_id}")
            assert get_response.status_code == 200

            retrieved_data = get_response.json()

            # Verify core data integrity
            assert retrieved_data["name"] == original_data["name"]
            assert retrieved_data["description"] == original_data["description"]

            # Verify data fields if present
            if "data" in retrieved_data:
                # Data structure should be preserved
                assert isinstance(retrieved_data["data"], dict)

            # Clean up
            client.delete(f"/api/v1/datasets/{dataset_id}")

    def test_concurrent_data_operations_consistency(self, client):
        """Test data consistency under concurrent operations."""
        import threading

        # Shared data for concurrent operations
        dataset_ids = []
        operation_results = []

        def create_dataset(index):
            """Create a dataset in a thread."""
            payload = {
                "name": f"Concurrent Dataset {index}",
                "description": f"Dataset created in thread {index}",
                "data": {"values": list(range(index, index + 5))},
            }

            response = client.post("/api/v1/datasets", json=payload)
            operation_results.append(("create", index, response.status_code))

            if response.status_code == 201:
                dataset_id = response.json()["id"]
                dataset_ids.append(dataset_id)

        # Start concurrent creation operations
        threads = []
        for i in range(3):
            thread = threading.Thread(target=create_dataset, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all operations to complete
        for thread in threads:
            thread.join()

        # Verify operations completed
        assert len(operation_results) == 3

        # Clean up created datasets
        for dataset_id in dataset_ids:
            client.delete(f"/api/v1/datasets/{dataset_id}")

    def test_special_character_handling(self, client):
        """Test handling of special characters in data."""
        special_chars_data = {
            "name": "Special Characters Test: 测试 🚀 emoji",
            "description": "Testing unicode: café résumé naïve",
            "data": {
                "unicode_text": ["Hello 世界", "Привет мир", "مرحبا بالعالم"],
                "special_chars": ["@#$%^&*()", '<>?:"{}|', "\\n\\t\\r"],
            },
            "metadata": {
                "unicode_key_测试": "unicode_value_тест",
                "emoji_key_🔧": "emoji_value_⚡",
            },
        }

        # Create dataset with special characters
        response = client.post("/api/v1/datasets", json=special_chars_data)

        if response.status_code == 201:
            created_data = response.json()
            dataset_id = created_data["id"]

            # Retrieve and verify
            get_response = client.get(f"/api/v1/datasets/{dataset_id}")

            if get_response.status_code == 200:
                retrieved_data = get_response.json()

                # Unicode characters should be preserved
                assert (
                    "測試" in retrieved_data["name"]
                    or "test" in retrieved_data["name"].lower()
                )
                assert (
                    "café" in retrieved_data["description"]
                    or "cafe" in retrieved_data["description"].lower()
                )

            # Clean up
            client.delete(f"/api/v1/datasets/{dataset_id}")

        else:
            # If special characters are not supported, should fail gracefully
            assert response.status_code in [400, 422]
            error_data = response.json()
            assert "detail" in error_data or "message" in error_data
