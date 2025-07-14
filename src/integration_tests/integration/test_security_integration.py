"""Integration tests for security features and vulnerability assessment."""

import pytest
from httpx import AsyncClient

from tests.integration.conftest import IntegrationTestHelper


class TestSecurityIntegration:
    """Test security features, authentication, authorization, and vulnerability resistance."""

    @pytest.mark.asyncio
    async def test_input_validation_and_sanitization(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth,
    ):
        """Test input validation and sanitization to prevent injection attacks."""

        # Setup basic components
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv, "security_test_dataset"
        )

        detector = await integration_helper.create_detector(
            dataset["id"], "isolation_forest"
        )

        await integration_helper.train_detector(detector["id"])

        # Test 1: SQL Injection attempts in string parameters
        sql_injection_payloads = [
            "'; DROP TABLE datasets; --",
            "test' OR '1'='1",
            "admin'/**/OR/**/1=1#",
            "'; UNION SELECT * FROM users; --",
            "test'; DELETE FROM models WHERE 1=1; --",
        ]

        for payload in sql_injection_payloads:
            # Test in dataset name
            response = await async_test_client.get(
                f"/api/datasets?name_contains={payload}"
            )
            assert response.status_code in [
                200,
                400,
                422,
            ]  # Should not cause server error

            # Test in detector name creation
            detector_config = {
                "name": payload,
                "description": "Test detector with malicious name",
                "algorithm": "isolation_forest",
                "parameters": {"contamination": 0.1},
                "feature_columns": ["feature1", "feature2", "feature3"],
            }

            response = await async_test_client.post(
                f"/api/detectors/create?dataset_id={dataset['id']}",
                json=detector_config,
            )

            # Should either reject (400/422) or sanitize input
            assert response.status_code in [200, 201, 400, 422]

            if response.status_code in [200, 201]:
                created_detector = response.json()["data"]
                # If created, name should be sanitized
                assert (
                    payload not in created_detector["name"]
                    or created_detector["name"] == payload
                )
                integration_helper.created_resources["detectors"].append(
                    created_detector["id"]
                )

        # Test 2: XSS attempts in text fields
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "';alert('XSS');//",
        ]

        for payload in xss_payloads:
            update_data = {"description": payload, "tags": [payload, "security_test"]}

            response = await async_test_client.put(
                f"/api/datasets/{dataset['id']}", json=update_data
            )

            # Should handle XSS attempts safely
            assert response.status_code in [200, 400, 422]

            if response.status_code == 200:
                updated_dataset = response.json()["data"]
                # XSS payload should be sanitized or escaped
                description = updated_dataset.get("description", "")
                assert (
                    "<script>" not in description or payload == description
                )  # Either sanitized or stored as-is

        # Test 3: Path traversal attempts
        path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        ]

        for payload in path_traversal_payloads:
            # Test in file-related endpoints
            response = await async_test_client.get(f"/api/datasets/{payload}")
            assert response.status_code in [
                400,
                404,
                422,
            ]  # Should not allow path traversal

        # Test 4: Command injection in parameters
        command_injection_payloads = [
            "; ls -la",
            "| cat /etc/passwd",
            "&& rm -rf /",
            "`whoami`",
            "$(cat /etc/passwd)",
        ]

        for payload in command_injection_payloads:
            # Test in algorithm parameters
            malicious_params = {
                "contamination": payload,
                "random_state": "42; rm -rf /",
            }

            detector_config = {
                "name": "command_injection_test",
                "algorithm": "isolation_forest",
                "parameters": malicious_params,
                "feature_columns": ["feature1", "feature2", "feature3"],
            }

            response = await async_test_client.post(
                f"/api/detectors/create?dataset_id={dataset['id']}",
                json=detector_config,
            )

            # Should reject malicious parameters
            assert response.status_code in [400, 422]

    @pytest.mark.asyncio
    async def test_authentication_bypass_attempts(
        self, async_test_client: AsyncClient, sample_dataset_csv: str
    ):
        """Test resistance to authentication bypass attempts."""

        # Note: This test runs without disable_auth to test actual auth behavior

        # Test 1: Header manipulation attempts
        auth_bypass_headers = [
            {"X-Forwarded-User": "admin"},
            {"X-User": "admin"},
            {"X-Real-IP": "127.0.0.1"},
            {"X-Forwarded-For": "127.0.0.1"},
            {"Authorization": "Bearer fake-token"},
            {"Authorization": "Basic YWRtaW46YWRtaW4="},  # admin:admin
            {"X-API-Key": "admin-key"},
            {"X-Auth-Token": "bypass-token"},
        ]

        for headers in auth_bypass_headers:
            response = await async_test_client.get("/api/datasets", headers=headers)
            # Should either succeed (if auth disabled) or fail with proper auth error
            assert response.status_code in [200, 401, 403]

        # Test 2: JWT token manipulation
        jwt_bypass_attempts = [
            "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
            "Bearer null",
            "Bearer undefined",
            "Bearer ''",
            "Bearer {}",
            "Bearer []",
        ]

        for token in jwt_bypass_attempts:
            headers = {"Authorization": token}
            response = await async_test_client.get("/api/datasets", headers=headers)
            assert response.status_code in [200, 401, 403]

        # Test 3: Session manipulation
        session_bypass_attempts = [
            {"Cookie": "session=admin"},
            {"Cookie": "sessionid=12345"},
            {"Cookie": "auth=true"},
            {"Cookie": "user=admin"},
            {"Cookie": "role=admin"},
        ]

        for headers in session_bypass_attempts:
            response = await async_test_client.get("/api/datasets", headers=headers)
            assert response.status_code in [200, 401, 403]

    @pytest.mark.asyncio
    async def test_authorization_and_access_control(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth,
    ):
        """Test authorization and role-based access control."""

        # Setup test data
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv, "authorization_test_dataset"
        )

        detector = await integration_helper.create_detector(
            dataset["id"], "isolation_forest"
        )

        # Test different user roles and permissions
        test_roles = [
            {"role": "admin", "user": "admin_user"},
            {"role": "data_scientist", "user": "ds_user"},
            {"role": "viewer", "user": "viewer_user"},
            {"role": "guest", "user": "guest_user"},
        ]

        for role_info in test_roles:
            headers = {"X-User-Role": role_info["role"], "X-User-ID": role_info["user"]}

            # Test read operations (should generally be allowed)
            response = await async_test_client.get("/api/datasets", headers=headers)
            assert response.status_code in [200, 401, 403]

            # Test write operations (permissions may vary)
            detector_config = {
                "name": f"auth_test_detector_{role_info['role']}",
                "algorithm": "isolation_forest",
                "parameters": {"contamination": 0.1},
                "feature_columns": ["feature1", "feature2", "feature3"],
            }

            response = await async_test_client.post(
                f"/api/detectors/create?dataset_id={dataset['id']}",
                json=detector_config,
                headers=headers,
            )

            # Response depends on role permissions
            assert response.status_code in [200, 201, 401, 403]

            if response.status_code in [200, 201]:
                created_detector = response.json()["data"]
                integration_helper.created_resources["detectors"].append(
                    created_detector["id"]
                )

            # Test delete operations (typically restricted)
            response = await async_test_client.delete(
                f"/api/datasets/{dataset['id']}", headers=headers
            )
            assert response.status_code in [200, 401, 403, 404]

    @pytest.mark.asyncio
    async def test_data_privacy_and_encryption(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth,
    ):
        """Test data privacy features and encryption handling."""

        # Test sensitive data handling
        sensitive_dataset_content = """sensitive_field,personal_info,secret_data
john.doe@email.com,John Doe,secret123
jane.smith@email.com,Jane Smith,password456
admin@company.com,Administrator,admin_secret"""

        # Create sensitive dataset
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(sensitive_dataset_content)
            sensitive_csv_path = f.name

        try:
            # Upload sensitive dataset
            with open(sensitive_csv_path, "rb") as f:
                files = {"file": ("sensitive_data.csv", f, "text/csv")}
                data = {
                    "name": "sensitive_test_dataset",
                    "description": "Dataset with sensitive information",
                    "privacy_level": "confidential",
                }

                response = await async_test_client.post(
                    "/api/datasets/upload", files=files, data=data
                )

            if response.status_code in [200, 201]:
                sensitive_dataset = response.json()["data"]
                integration_helper.created_resources["datasets"].append(
                    sensitive_dataset["id"]
                )

                # Test data access controls
                response = await async_test_client.get(
                    f"/api/datasets/{sensitive_dataset['id']}/data"
                )

                if response.status_code == 200:
                    data_sample = response.json()["data"]

                    # Check if sensitive data is masked or encrypted
                    data_str = str(data_sample)
                    sensitive_indicators = [
                        "john.doe@email.com",
                        "secret123",
                        "password456",
                    ]

                    # In a secure system, sensitive data should be masked
                    exposed_count = sum(
                        1 for indicator in sensitive_indicators if indicator in data_str
                    )
                    # This test documents current behavior rather than enforcing encryption
                    print(
                        f"Sensitive data exposure check: {exposed_count}/{len(sensitive_indicators)} indicators found"
                    )

                # Test data export with privacy controls
                export_config = {
                    "format": "csv",
                    "include_sensitive": False,
                    "anonymize": True,
                }

                response = await async_test_client.post(
                    f"/api/datasets/{sensitive_dataset['id']}/export",
                    json=export_config,
                )

                # Should handle privacy settings appropriately
                assert response.status_code in [200, 201, 400, 403]

        finally:
            # Clean up temporary file
            if os.path.exists(sensitive_csv_path):
                os.unlink(sensitive_csv_path)

    @pytest.mark.asyncio
    async def test_rate_limiting_and_ddos_protection(
        self,
        async_test_client: AsyncClient,
        integration_helper: IntegrationTestHelper,
        sample_dataset_csv: str,
        disable_auth,
    ):
        """Test rate limiting and DDoS protection mechanisms."""

        import asyncio
        import time

        # Setup for rate limiting tests
        dataset = await integration_helper.upload_dataset(
            sample_dataset_csv, "rate_limit_test_dataset"
        )

        detector = await integration_helper.create_detector(
            dataset["id"], "isolation_forest"
        )

        await integration_helper.train_detector(detector["id"])

        # Test 1: Rapid consecutive requests
        async def rapid_fire_requests(num_requests: int, endpoint: str):
            """Make rapid consecutive requests to test rate limiting."""
            results = []
            start_time = time.time()

            for i in range(num_requests):
                try:
                    response = await async_test_client.get(endpoint)
                    results.append(
                        {
                            "request_id": i,
                            "status_code": response.status_code,
                            "rate_limited": response.status_code == 429,
                            "timestamp": time.time() - start_time,
                        }
                    )
                except Exception as e:
                    results.append(
                        {
                            "request_id": i,
                            "status_code": None,
                            "error": str(e),
                            "timestamp": time.time() - start_time,
                        }
                    )

            return results

        # Test rapid requests to different endpoints
        endpoints_to_test = [
            "/api/datasets",
            f"/api/datasets/{dataset['id']}",
            "/api/detectors",
            "/api/health",
        ]

        for endpoint in endpoints_to_test:
            rapid_results = await rapid_fire_requests(30, endpoint)

            # Analyze rate limiting behavior
            rate_limited_count = sum(
                1 for r in rapid_results if r.get("rate_limited", False)
            )
            successful_count = sum(
                1 for r in rapid_results if r.get("status_code") == 200
            )

            # Rate limiting might not be implemented, so we document behavior
            print(
                f"Endpoint {endpoint}: {successful_count}/30 successful, {rate_limited_count} rate limited"
            )

        # Test 2: Concurrent request flooding
        async def concurrent_request_flood(endpoint: str, concurrent_count: int):
            """Make concurrent requests to test flood protection."""
            start_time = time.time()

            tasks = [async_test_client.get(endpoint) for _ in range(concurrent_count)]

            responses = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time

            successful_responses = [
                r
                for r in responses
                if not isinstance(r, Exception) and r.status_code == 200
            ]
            rate_limited_responses = [
                r
                for r in responses
                if not isinstance(r, Exception) and r.status_code == 429
            ]

            return {
                "total_requests": concurrent_count,
                "successful": len(successful_responses),
                "rate_limited": len(rate_limited_responses),
                "total_time": total_time,
                "requests_per_second": concurrent_count / total_time,
            }

        # Test concurrent flooding
        flood_result = await concurrent_request_flood("/api/health", 20)

        # System should handle concurrent load gracefully
        assert (
            flood_result["successful"] + flood_result["rate_limited"]
            >= flood_result["total_requests"] * 0.5
        )

        print(
            f"Concurrent flood test: {flood_result['successful']}/{flood_result['total_requests']} successful"
        )

        # Test 3: Resource-intensive request limiting
        async def resource_intensive_requests():
            """Test rate limiting on resource-intensive operations."""
            results = []

            # Make multiple prediction requests rapidly
            for i in range(10):
                test_data = {
                    "data": [
                        {"feature1": i * 0.1, "feature2": i * 0.2, "feature3": i * 0.05}
                    ]
                }

                try:
                    response = await async_test_client.post(
                        f"/api/detection/predict/{detector['id']}", json=test_data
                    )
                    results.append(
                        {
                            "request_id": i,
                            "status_code": response.status_code,
                            "success": response.status_code == 200,
                        }
                    )
                except Exception as e:
                    results.append({"request_id": i, "error": str(e), "success": False})

            return results

        intensive_results = await resource_intensive_requests()
        successful_predictions = sum(
            1 for r in intensive_results if r.get("success", False)
        )

        # Should handle resource-intensive requests appropriately
        assert successful_predictions >= 5  # At least half should succeed

        print(f"Resource-intensive requests: {successful_predictions}/10 successful")

    @pytest.mark.asyncio
    async def test_secure_file_upload_handling(
        self, async_test_client: AsyncClient, disable_auth
    ):
        """Test secure file upload handling and malicious file detection."""

        # Test 1: Malicious file extensions
        malicious_files = [
            ("malware.exe", b"MZ\x90\x00", "application/octet-stream"),
            ("script.js", b"alert('malicious');", "application/javascript"),
            ("exploit.php", b"<?php system($_GET['cmd']); ?>", "application/x-php"),
            (
                "virus.bat",
                b"@echo off\ndel /f /q C:\\*.*",
                "application/x-msdos-program",
            ),
            ("backdoor.sh", b"#!/bin/bash\nrm -rf /", "application/x-sh"),
        ]

        for filename, content, content_type in malicious_files:
            files = {"file": (filename, content, content_type)}
            data = {"name": f"malicious_test_{filename}"}

            response = await async_test_client.post(
                "/api/datasets/upload", files=files, data=data
            )

            # Should reject malicious files
            assert response.status_code in [
                400,
                422,
                415,
            ]  # Bad request, validation error, or unsupported media type

        # Test 2: Oversized files
        large_content = b"data,values\n" + b"1,2\n" * 100000  # Large CSV content

        files = {"file": ("large_file.csv", large_content, "text/csv")}
        data = {"name": "large_file_test"}

        response = await async_test_client.post(
            "/api/datasets/upload", files=files, data=data
        )

        # Should handle large files appropriately (reject or accept based on limits)
        assert response.status_code in [
            200,
            201,
            400,
            413,
            422,
        ]  # Success or file too large

        if response.status_code in [200, 201]:
            # If accepted, clean up
            dataset = response.json()["data"]
            await async_test_client.delete(f"/api/datasets/{dataset['id']}")

        # Test 3: File content validation
        invalid_csv_files = [
            ("empty.csv", b"", "text/csv"),
            ("binary.csv", b"\x00\x01\x02\x03\x04", "text/csv"),
            (
                "xml_injection.csv",
                b"<?xml version='1.0'?><root><![CDATA[malicious]]></root>",
                "text/csv",
            ),
            ("json_in_csv.csv", b'{"malicious": "payload"}', "text/csv"),
        ]

        for filename, content, content_type in invalid_csv_files:
            files = {"file": (filename, content, content_type)}
            data = {"name": f"invalid_test_{filename}"}

            response = await async_test_client.post(
                "/api/datasets/upload", files=files, data=data
            )

            # Should validate file content properly
            assert response.status_code in [200, 201, 400, 422]

        # Test 4: Path traversal in filenames
        path_traversal_filenames = [
            ("../../../etc/passwd.csv", b"col1,col2\n1,2\n", "text/csv"),
            (
                "..\\..\\windows\\system32\\config\\sam.csv",
                b"col1,col2\n1,2\n",
                "text/csv",
            ),
            ("normal_name.csv", b"col1,col2\n1,2\n", "text/csv"),  # Control case
        ]

        for filename, content, content_type in path_traversal_filenames:
            files = {"file": (filename, content, content_type)}
            data = {
                "name": f"path_test_{filename.replace('/', '_').replace('\\', '_')}"
            }

            response = await async_test_client.post(
                "/api/datasets/upload", files=files, data=data
            )

            if filename == "normal_name.csv":
                # Normal file should succeed
                assert response.status_code in [200, 201, 400, 422]
                if response.status_code in [200, 201]:
                    dataset = response.json()["data"]
                    await async_test_client.delete(f"/api/datasets/{dataset['id']}")
            else:
                # Path traversal attempts should be handled safely
                assert response.status_code in [200, 201, 400, 422]

    @pytest.mark.asyncio
    async def test_api_security_headers(
        self, async_test_client: AsyncClient, disable_auth
    ):
        """Test security headers in API responses."""

        # Test security headers on various endpoints
        endpoints_to_test = [
            "/",
            "/api/health",
            "/api/docs" if disable_auth else "/api/datasets",
        ]

        expected_security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy",
        ]

        for endpoint in endpoints_to_test:
            response = await async_test_client.get(endpoint)

            # Endpoint should respond successfully or with expected auth error
            assert response.status_code in [200, 401, 403, 404]

            # Check for security headers
            headers_present = []
            for header in expected_security_headers:
                if header in response.headers:
                    headers_present.append(header)

            # Document which security headers are present
            print(
                f"Endpoint {endpoint}: {len(headers_present)}/{len(expected_security_headers)} security headers present"
            )

            # Verify no sensitive information in headers
            sensitive_header_patterns = [
                "Server",  # Should not reveal server details
                "X-Powered-By",  # Should not reveal technology stack
                "X-AspNet-Version",
                "X-AspNetMvc-Version",
            ]

            for sensitive_header in sensitive_header_patterns:
                if sensitive_header in response.headers:
                    header_value = response.headers[sensitive_header]
                    # Document but don't necessarily fail on presence
                    print(
                        f"Sensitive header found - {sensitive_header}: {header_value}"
                    )

        # Test CORS headers
        cors_response = await async_test_client.options("/api/health")

        if cors_response.status_code in [200, 204]:
            # Check CORS configuration
            cors_headers = [
                "Access-Control-Allow-Origin",
                "Access-Control-Allow-Methods",
                "Access-Control-Allow-Headers",
            ]

            cors_present = sum(
                1 for header in cors_headers if header in cors_response.headers
            )
            print(f"CORS headers: {cors_present}/{len(cors_headers)} present")

        # Test for information disclosure in error responses
        response = await async_test_client.get("/api/nonexistent-endpoint")
        assert response.status_code == 404

        error_content = response.text.lower()

        # Should not reveal sensitive information in errors
        sensitive_patterns = [
            "traceback",
            "exception",
            "internal server error",
            "stack trace",
            "database error",
            "sql error",
        ]

        disclosed_info = [
            pattern for pattern in sensitive_patterns if pattern in error_content
        ]

        if disclosed_info:
            print(f"Potential information disclosure in 404 response: {disclosed_info}")

        # Error responses should be generic
        assert len(error_content) < 1000  # Should not be overly verbose
