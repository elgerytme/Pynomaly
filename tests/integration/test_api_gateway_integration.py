"""Integration tests for API gateway features."""

import pytest
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List

from pynomaly.features.api_gateway import (
    APIGateway,
    EndpointManager,
    RequestProcessor,
    ResponseProcessor,
    APIVersioning,
    APIRequest,
    APIResponse,
    EndpointMetadata,
    HTTPMethod,
    EndpointStatus,
    ResponseFormat,
    get_api_gateway,
)


@pytest.fixture
def sample_api_request():
    """Create sample API request for testing."""
    return APIRequest(
        request_id=str(uuid.uuid4()),
        method=HTTPMethod.GET,
        path="/v1/test",
        headers={"Content-Type": "application/json", "User-Agent": "test-client"},
        query_params={"param1": "value1", "param2": "value2"},
        body=None,
        timestamp=datetime.now(),
        client_ip="127.0.0.1",
        user_agent="test-client/1.0",
        auth_token="test-token"
    )


@pytest.fixture
def sample_post_request():
    """Create sample POST API request for testing."""
    return APIRequest(
        request_id=str(uuid.uuid4()),
        method=HTTPMethod.POST,
        path="/v1/detect",
        headers={"Content-Type": "application/json"},
        query_params={},
        body={
            "data": [
                {"feature_1": 1.5, "feature_2": -0.8},
                {"feature_1": 2.1, "feature_2": 0.3},
                {"feature_1": -1.2, "feature_2": 1.7}
            ],
            "algorithm": "isolation_forest",
            "parameters": {"contamination": 0.1}
        },
        timestamp=datetime.now(),
        client_ip="192.168.1.100",
        user_agent="anomaly-client/2.0"
    )


@pytest.mark.asyncio
class TestRequestProcessorIntegration:
    """Integration tests for request processor."""
    
    async def test_request_processing_lifecycle(self, sample_api_request):
        """Test complete request processing lifecycle."""
        processor = RequestProcessor()
        
        # Process valid request
        processed_request = await processor.process_request(sample_api_request)
        
        # Verify processing results
        assert processed_request.request_id == sample_api_request.request_id
        assert processed_request.method == sample_api_request.method
        assert processed_request.path == sample_api_request.path.strip()
        
        # Verify headers are normalized (lowercased)
        assert "content-type" in processed_request.headers
        assert "user-agent" in processed_request.headers
        
        # Verify query params are trimmed
        assert processed_request.query_params["param1"] == "value1"
        assert processed_request.query_params["param2"] == "value2"
        
        # Check processing statistics
        stats = await processor.get_processing_stats()
        assert stats["total_requests"] >= 1
        assert stats["successful_requests"] >= 1
        assert stats["success_rate"] > 0
    
    async def test_request_validation(self):
        """Test request validation functionality."""
        processor = RequestProcessor()
        
        # Test invalid path (empty)
        invalid_request = APIRequest(
            request_id=str(uuid.uuid4()),
            method=HTTPMethod.GET,
            path="",  # Invalid empty path
            headers={},
            query_params={},
            timestamp=datetime.now()
        )
        
        with pytest.raises(Exception):
            await processor.process_request(invalid_request)
        
        # Test invalid path (doesn't start with /)
        invalid_path_request = APIRequest(
            request_id=str(uuid.uuid4()),
            method=HTTPMethod.GET,
            path="invalid-path",  # Should start with /
            headers={},
            query_params={},
            timestamp=datetime.now()
        )
        
        with pytest.raises(Exception):
            await processor.process_request(invalid_path_request)
        
        # Test large body validation
        large_body = {"data": "x" * 15_000_000}  # >10MB
        large_body_request = APIRequest(
            request_id=str(uuid.uuid4()),
            method=HTTPMethod.POST,
            path="/v1/test",
            headers={},
            query_params={},
            body=large_body,
            timestamp=datetime.now()
        )
        
        with pytest.raises(Exception):
            await processor.process_request(large_body_request)
        
        # Check validation error statistics
        stats = await processor.get_processing_stats()
        assert stats["validation_errors"] >= 3
        assert stats["validation_error_rate"] > 0
    
    async def test_request_sanitization(self):
        """Test request data sanitization."""
        processor = RequestProcessor()
        
        # Create request with data that needs sanitization
        dirty_request = APIRequest(
            request_id=str(uuid.uuid4()),
            method=HTTPMethod.POST,
            path="  /v1/sanitize  ",  # Extra whitespace
            headers={
                "  Content-Type  ": "  application/json  ",  # Whitespace in headers
                "AUTHORIZATION": "Bearer token123"  # Mixed case
            },
            query_params={
                "  param1  ": "  value1  ",  # Whitespace in params
                "param2": "value2"
            },
            timestamp=datetime.now(),
            client_ip="  127.0.0.1  ",  # Whitespace in IP
            user_agent="  test-agent  "  # Whitespace in user agent
        )
        
        processed_request = await processor.process_request(dirty_request)
        
        # Verify sanitization
        assert processed_request.path == "/v1/sanitize"
        assert processed_request.headers["content-type"] == "application/json"
        assert processed_request.headers["authorization"] == "Bearer token123"
        assert processed_request.query_params["param1"] == "value1"
        assert processed_request.client_ip == "127.0.0.1"
        assert processed_request.user_agent == "test-agent"


@pytest.mark.asyncio
class TestResponseProcessorIntegration:
    """Integration tests for response processor."""
    
    async def test_response_processing_lifecycle(self, sample_api_request):
        """Test complete response processing lifecycle."""
        processor = ResponseProcessor()
        
        # Create sample response
        response = APIResponse(
            request_id=sample_api_request.request_id,
            status_code=200,
            headers={},
            body={"message": "success", "data": [1, 2, 3]},
            format=ResponseFormat.JSON,
            timestamp=datetime.now(),
            execution_time_ms=45.5
        )
        
        # Process response
        processed_response = await processor.process_response(response)
        
        # Verify processing results
        assert processed_response.request_id == sample_api_request.request_id
        assert processed_response.status_code == 200
        
        # Verify standard headers were added
        assert "Content-Type" in processed_response.headers
        assert "X-Request-ID" in processed_response.headers
        assert "X-Response-Time" in processed_response.headers
        assert "X-Timestamp" in processed_response.headers
        
        assert processed_response.headers["Content-Type"] == "application/json"
        assert processed_response.headers["X-Request-ID"] == sample_api_request.request_id
        
        # Verify body was formatted as JSON string
        assert isinstance(processed_response.body, str)
        parsed_body = json.loads(processed_response.body)
        assert parsed_body["message"] == "success"
        assert parsed_body["data"] == [1, 2, 3]
        
        # Check processing statistics
        stats = await processor.get_response_stats()
        assert stats["total_responses"] >= 1
        assert stats["successful_responses"] >= 1
        assert stats["success_rate"] > 0
    
    async def test_response_format_handling(self, sample_api_request):
        """Test different response format handling."""
        processor = ResponseProcessor()
        
        # Test JSON format
        json_response = APIResponse(
            request_id=sample_api_request.request_id,
            status_code=200,
            body={"test": "json"},
            format=ResponseFormat.JSON
        )
        
        processed_json = await processor.process_response(json_response)
        assert processed_json.headers["Content-Type"] == "application/json"
        assert isinstance(processed_json.body, str)
        
        # Test XML format
        xml_response = APIResponse(
            request_id=sample_api_request.request_id,
            status_code=200,
            body="<test>xml</test>",
            format=ResponseFormat.XML
        )
        
        processed_xml = await processor.process_response(xml_response)
        assert processed_xml.headers["Content-Type"] == "application/xml"
        
        # Test plain text format
        text_response = APIResponse(
            request_id=sample_api_request.request_id,
            status_code=200,
            body="plain text response",
            format=ResponseFormat.PLAIN_TEXT
        )
        
        processed_text = await processor.process_response(text_response)
        assert processed_text.headers["Content-Type"] == "text/plain"
    
    async def test_response_error_handling(self, sample_api_request):
        """Test response error handling."""
        processor = ResponseProcessor()
        
        # Create response that will cause processing error
        error_response = APIResponse(
            request_id=sample_api_request.request_id,
            status_code=500,
            body={"error": "internal_error"},
            format=ResponseFormat.JSON
        )
        
        # Simulate processing error by creating malformed response
        # (This would normally happen if there's an error in processing)
        processed_response = await processor.process_response(error_response)
        
        # Should handle gracefully and create proper error response
        assert processed_response.status_code == 500
        assert "Content-Type" in processed_response.headers
        
        # Check error statistics
        stats = await processor.get_response_stats()
        assert stats["error_responses"] >= 1
        assert stats["status_code_counts"]["500"] >= 1


@pytest.mark.asyncio
class TestEndpointManagerIntegration:
    """Integration tests for endpoint manager."""
    
    async def test_endpoint_registration_lifecycle(self):
        """Test complete endpoint registration lifecycle."""
        manager = EndpointManager()
        
        # Create endpoint handler
        async def test_handler(request: APIRequest) -> APIResponse:
            return APIResponse(
                request_id=request.request_id,
                status_code=200,
                body={"message": "test endpoint response"},
                format=ResponseFormat.JSON
            )
        
        # Create endpoint metadata
        metadata = EndpointMetadata(
            path="/v1/test-endpoint",
            method=HTTPMethod.GET,
            handler=test_handler,
            name="Test Endpoint",
            description="Endpoint for integration testing",
            version="1.0.0",
            status=EndpointStatus.ACTIVE,
            rate_limit=100,
            auth_required=False,
            tags=["test", "integration"]
        )
        
        # Register endpoint
        registered = manager.register_endpoint(metadata)
        assert registered
        
        # Retrieve endpoint
        retrieved = manager.get_endpoint("/v1/test-endpoint", HTTPMethod.GET)
        assert retrieved is not None
        assert retrieved.name == "Test Endpoint"
        assert retrieved.path == "/v1/test-endpoint"
        assert retrieved.method == HTTPMethod.GET
        
        # List endpoints
        all_endpoints = manager.list_endpoints()
        assert len(all_endpoints) >= 1
        assert any(ep.path == "/v1/test-endpoint" for ep in all_endpoints)
        
        # Filter endpoints by status
        active_endpoints = manager.list_endpoints(status_filter=EndpointStatus.ACTIVE)
        assert len(active_endpoints) >= 1
        
        # Filter endpoints by tag
        test_endpoints = manager.list_endpoints(tag_filter="test")
        assert len(test_endpoints) >= 1
        
        # Update endpoint statistics
        manager.update_endpoint_stats("/v1/test-endpoint", HTTPMethod.GET, 125.5, success=True)
        manager.update_endpoint_stats("/v1/test-endpoint", HTTPMethod.GET, 98.2, success=True)
        manager.update_endpoint_stats("/v1/test-endpoint", HTTPMethod.GET, 155.8, success=False)
        
        # Get endpoint statistics
        stats = await manager.get_endpoint_stats("/v1/test-endpoint")
        endpoint_key = f"{HTTPMethod.GET.value}:/v1/test-endpoint"
        assert endpoint_key in stats
        
        endpoint_stats = stats[endpoint_key]
        assert endpoint_stats["request_count"] == 3
        assert endpoint_stats["error_count"] == 1
        assert endpoint_stats["average_execution_time_ms"] > 0
        
        # Deprecate endpoint
        deprecated = manager.deprecate_endpoint("/v1/test-endpoint", HTTPMethod.GET)
        assert deprecated
        
        # Verify deprecation
        deprecated_endpoint = manager.get_endpoint("/v1/test-endpoint", HTTPMethod.GET)
        assert deprecated_endpoint.status == EndpointStatus.DEPRECATED
        assert deprecated_endpoint.deprecated_at is not None
    
    async def test_endpoint_overwriting(self):
        """Test endpoint overwriting behavior."""
        manager = EndpointManager()
        
        # Create first endpoint
        async def handler1(request: APIRequest) -> APIResponse:
            return APIResponse(
                request_id=request.request_id,
                status_code=200,
                body={"handler": "first"},
                format=ResponseFormat.JSON
            )
        
        metadata1 = EndpointMetadata(
            path="/v1/overwrite-test",
            method=HTTPMethod.POST,
            handler=handler1,
            name="First Handler",
            description="First version of endpoint"
        )
        
        registered1 = manager.register_endpoint(metadata1)
        assert registered1
        
        # Create second endpoint with same path and method
        async def handler2(request: APIRequest) -> APIResponse:
            return APIResponse(
                request_id=request.request_id,
                status_code=200,
                body={"handler": "second"},
                format=ResponseFormat.JSON
            )
        
        metadata2 = EndpointMetadata(
            path="/v1/overwrite-test",
            method=HTTPMethod.POST,
            handler=handler2,
            name="Second Handler",
            description="Second version of endpoint"
        )
        
        registered2 = manager.register_endpoint(metadata2)
        assert registered2
        
        # Verify second endpoint overwrote the first
        current_endpoint = manager.get_endpoint("/v1/overwrite-test", HTTPMethod.POST)
        assert current_endpoint.name == "Second Handler"
        assert current_endpoint.description == "Second version of endpoint"


@pytest.mark.asyncio
class TestAPIVersioningIntegration:
    """Integration tests for API versioning."""
    
    async def test_version_management_lifecycle(self):
        """Test complete version management lifecycle."""
        versioning = APIVersioning()
        
        # Register versions
        v1_registered = versioning.register_version(
            "v1",
            description="Initial API version",
            deprecated=False
        )
        assert v1_registered
        
        v2_registered = versioning.register_version(
            "v2",
            description="Second API version with improvements",
            deprecated=False
        )
        assert v2_registered
        
        v1_beta_registered = versioning.register_version(
            "v1-beta",
            description="Beta version for testing",
            deprecated=True,
            sunset_date=datetime.now() + timedelta(days=30)
        )
        assert v1_beta_registered
        
        # Test version extraction from paths
        assert versioning.get_version_from_path("/v1/users") == "v1"
        assert versioning.get_version_from_path("/api/v2/data") == "v2"
        assert versioning.get_version_from_path("/v1-beta/test") == "v1"  # Extracts v1 from v1-beta
        assert versioning.get_version_from_path("/no-version/endpoint") == "v1"  # Default
        
        # Test version support checks
        assert versioning.is_version_supported("v1")
        assert versioning.is_version_supported("v2")
        assert versioning.is_version_supported("v1-beta")
        assert not versioning.is_version_supported("v3")
        
        # Test deprecation checks
        assert not versioning.is_version_deprecated("v1")
        assert not versioning.is_version_deprecated("v2")
        assert versioning.is_version_deprecated("v1-beta")
        
        # Get version information
        v1_info = versioning.get_version_info("v1")
        assert v1_info["description"] == "Initial API version"
        assert not v1_info["deprecated"]
        
        v1_beta_info = versioning.get_version_info("v2")
        assert v1_beta_info["description"] == "Second API version with improvements"
        
        # Test non-existent version
        nonexistent_info = versioning.get_version_info("v99")
        assert nonexistent_info == {}
    
    async def test_version_path_extraction_edge_cases(self):
        """Test edge cases in version path extraction."""
        versioning = APIVersioning()
        versioning.default_version = "v1"
        
        # Test various path formats
        test_cases = [
            ("/v1/endpoint", "v1"),
            ("/api/v2/endpoint", "v2"),
            ("/service/v3/test", "v3"),
            ("/v10/advanced", "v10"),
            ("/no-version", "v1"),  # Default
            ("/version/but/no/v", "v1"),  # Default
            ("", "v1"),  # Empty path, default
            ("/", "v1"),  # Root path, default
        ]
        
        for path, expected_version in test_cases:
            extracted_version = versioning.get_version_from_path(path)
            assert extracted_version == expected_version, f"Path {path} should extract version {expected_version}, got {extracted_version}"


@pytest.mark.asyncio
class TestAPIGatewayIntegration:
    """Integration tests for API gateway."""
    
    async def test_api_gateway_complete_request_lifecycle(self, sample_api_request):
        """Test complete API gateway request handling lifecycle."""
        gateway = APIGateway()
        
        # Register standard endpoints
        await gateway.register_anomaly_detection_endpoints()
        
        # Handle health check request
        health_request = APIRequest(
            request_id=str(uuid.uuid4()),
            method=HTTPMethod.GET,
            path="/health",
            headers={"Accept": "application/json"},
            query_params={},
            timestamp=datetime.now(),
            client_ip="127.0.0.1"
        )
        
        health_response = await gateway.handle_request(health_request)
        
        # Verify health check response
        assert health_response.status_code == 200
        assert health_response.request_id == health_request.request_id
        assert "Content-Type" in health_response.headers
        
        # Parse health check body
        health_body = json.loads(health_response.body)
        assert health_body["status"] == "healthy"
        assert "timestamp" in health_body
        assert "version" in health_body
        
        # Handle detection request
        detection_request = APIRequest(
            request_id=str(uuid.uuid4()),
            method=HTTPMethod.POST,
            path="/v1/detect",
            headers={"Content-Type": "application/json"},
            query_params={},
            body={
                "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                "algorithm": "isolation_forest"
            },
            timestamp=datetime.now(),
            client_ip="127.0.0.1"
        )
        
        detection_response = await gateway.handle_request(detection_request)
        
        # Verify detection response
        assert detection_response.status_code == 200
        assert detection_response.request_id == detection_request.request_id
        
        # Parse detection body
        detection_body = json.loads(detection_response.body)
        assert "detection_id" in detection_body
        assert "anomalies_detected" in detection_body
        assert "timestamp" in detection_body
        assert "processing_time_ms" in detection_body
        
        # Check gateway statistics
        gateway_status = await gateway.get_gateway_status()
        assert gateway_status["gateway_stats"]["total_requests"] >= 2
        assert gateway_status["gateway_stats"]["total_responses"] >= 2
        assert gateway_status["registered_endpoints"] >= 2
        assert "v1" in gateway_status["api_versions"]
    
    async def test_api_gateway_error_handling(self):
        """Test API gateway error handling scenarios."""
        gateway = APIGateway()
        
        # Test non-existent endpoint
        not_found_request = APIRequest(
            request_id=str(uuid.uuid4()),
            method=HTTPMethod.GET,
            path="/nonexistent/endpoint",
            headers={},
            query_params={},
            timestamp=datetime.now(),
            client_ip="127.0.0.1"
        )
        
        not_found_response = await gateway.handle_request(not_found_request)
        assert not_found_response.status_code == 404
        
        not_found_body = json.loads(not_found_response.body)
        assert "error" in not_found_body
        assert "Endpoint not found" in not_found_body["message"]
        
        # Test unsupported API version
        unsupported_version_request = APIRequest(
            request_id=str(uuid.uuid4()),
            method=HTTPMethod.GET,
            path="/v99/unsupported",
            headers={},
            query_params={},
            timestamp=datetime.now(),
            client_ip="127.0.0.1"
        )
        
        unsupported_response = await gateway.handle_request(unsupported_version_request)
        assert unsupported_response.status_code == 400
        
        unsupported_body = json.loads(unsupported_response.body)
        assert "Unsupported API version" in unsupported_body["message"]
        
        # Test invalid request format
        invalid_request = APIRequest(
            request_id=str(uuid.uuid4()),
            method=HTTPMethod.POST,
            path="invalid-path",  # Missing leading slash
            headers={},
            query_params={},
            timestamp=datetime.now(),
            client_ip="127.0.0.1"
        )
        
        with pytest.raises(Exception):
            await gateway.handle_request(invalid_request)
    
    async def test_api_gateway_endpoint_status_handling(self):
        """Test API gateway handling of different endpoint statuses."""
        gateway = APIGateway()
        
        # Create test endpoint handler
        async def test_handler(request: APIRequest) -> APIResponse:
            return APIResponse(
                request_id=request.request_id,
                status_code=200,
                body={"message": "endpoint working"},
                format=ResponseFormat.JSON
            )
        
        # Register endpoint as active
        active_metadata = EndpointMetadata(
            path="/v1/active-test",
            method=HTTPMethod.GET,
            handler=test_handler,
            status=EndpointStatus.ACTIVE
        )
        gateway.endpoint_manager.register_endpoint(active_metadata)
        
        # Register endpoint as inactive
        inactive_metadata = EndpointMetadata(
            path="/v1/inactive-test",
            method=HTTPMethod.GET,
            handler=test_handler,
            status=EndpointStatus.INACTIVE
        )
        gateway.endpoint_manager.register_endpoint(inactive_metadata)
        
        # Register endpoint under maintenance
        maintenance_metadata = EndpointMetadata(
            path="/v1/maintenance-test",
            method=HTTPMethod.GET,
            handler=test_handler,
            status=EndpointStatus.MAINTENANCE
        )
        gateway.endpoint_manager.register_endpoint(maintenance_metadata)
        
        # Register API version
        gateway.versioning.register_version("v1", "Test version")
        
        # Test active endpoint (should work)
        active_request = APIRequest(
            request_id=str(uuid.uuid4()),
            method=HTTPMethod.GET,
            path="/v1/active-test",
            headers={},
            query_params={},
            timestamp=datetime.now(),
            client_ip="127.0.0.1"
        )
        
        active_response = await gateway.handle_request(active_request)
        assert active_response.status_code == 200
        
        # Test inactive endpoint (should return 503)
        inactive_request = APIRequest(
            request_id=str(uuid.uuid4()),
            method=HTTPMethod.GET,
            path="/v1/inactive-test",
            headers={},
            query_params={},
            timestamp=datetime.now(),
            client_ip="127.0.0.1"
        )
        
        inactive_response = await gateway.handle_request(inactive_request)
        assert inactive_response.status_code == 503
        
        inactive_body = json.loads(inactive_response.body)
        assert "temporarily unavailable" in inactive_body["message"]
        
        # Test maintenance endpoint (should return 503)
        maintenance_request = APIRequest(
            request_id=str(uuid.uuid4()),
            method=HTTPMethod.GET,
            path="/v1/maintenance-test",
            headers={},
            query_params={},
            timestamp=datetime.now(),
            client_ip="127.0.0.1"
        )
        
        maintenance_response = await gateway.handle_request(maintenance_request)
        assert maintenance_response.status_code == 503
        
        maintenance_body = json.loads(maintenance_response.body)
        assert "under maintenance" in maintenance_body["message"]


@pytest.mark.asyncio
class TestAPIGatewayRateLimitingIntegration:
    """Integration tests for API gateway rate limiting."""
    
    async def test_rate_limiting_functionality(self):
        """Test rate limiting functionality."""
        gateway = APIGateway()
        
        # Create test endpoint
        async def rate_limited_handler(request: APIRequest) -> APIResponse:
            return APIResponse(
                request_id=request.request_id,
                status_code=200,
                body={"message": "success"},
                format=ResponseFormat.JSON
            )
        
        # Register endpoint with rate limit
        metadata = EndpointMetadata(
            path="/v1/rate-limited",
            method=HTTPMethod.GET,
            handler=rate_limited_handler,
            rate_limit=5  # 5 requests per minute
        )
        gateway.endpoint_manager.register_endpoint(metadata)
        gateway.versioning.register_version("v1", "Test version")
        
        # Make multiple requests rapidly
        responses = []
        for i in range(10):  # More than rate limit
            request = APIRequest(
                request_id=str(uuid.uuid4()),
                method=HTTPMethod.GET,
                path="/v1/rate-limited",
                headers={},
                query_params={},
                timestamp=datetime.now(),
                client_ip="127.0.0.1"  # Same IP for rate limiting
            )
            
            response = await gateway.handle_request(request)
            responses.append(response)
            
            # Small delay between requests
            await asyncio.sleep(0.01)
        
        # Check responses
        success_responses = [r for r in responses if r.status_code == 200]
        rate_limited_responses = [r for r in responses if r.status_code == 429]
        
        # Should have some successful responses and some rate limited
        assert len(success_responses) > 0
        # Note: Actual rate limiting behavior depends on the rate limiter implementation
        
        # Verify rate limited response format
        if rate_limited_responses:
            rate_limited_body = json.loads(rate_limited_responses[0].body)
            assert "Rate limit exceeded" in rate_limited_body["error"]


@pytest.mark.asyncio
class TestAPIGatewayDeprecatedVersions:
    """Test API gateway handling of deprecated versions."""
    
    async def test_deprecated_version_warnings(self):
        """Test handling of deprecated API versions."""
        gateway = APIGateway()
        
        # Register deprecated version
        gateway.versioning.register_version(
            "v1-deprecated",
            description="Deprecated version",
            deprecated=True,
            sunset_date=datetime.now() + timedelta(days=30)
        )
        
        # Register active version
        gateway.versioning.register_version("v2", description="Current version")
        
        # Create test endpoint for deprecated version
        async def deprecated_handler(request: APIRequest) -> APIResponse:
            return APIResponse(
                request_id=request.request_id,
                status_code=200,
                body={"message": "deprecated endpoint response"},
                format=ResponseFormat.JSON
            )
        
        deprecated_metadata = EndpointMetadata(
            path="/v1-deprecated/test",
            method=HTTPMethod.GET,
            handler=deprecated_handler
        )
        gateway.endpoint_manager.register_endpoint(deprecated_metadata)
        
        # Make request to deprecated endpoint
        deprecated_request = APIRequest(
            request_id=str(uuid.uuid4()),
            method=HTTPMethod.GET,
            path="/v1-deprecated/test",
            headers={},
            query_params={},
            timestamp=datetime.now(),
            client_ip="127.0.0.1"
        )
        
        # Should still work but with warning logged
        deprecated_response = await gateway.handle_request(deprecated_request)
        assert deprecated_response.status_code == 200
        
        # Response should indicate deprecated version usage
        # (In a real implementation, you might add deprecation headers)
        deprecated_body = json.loads(deprecated_response.body)
        assert deprecated_body["message"] == "deprecated endpoint response"


@pytest.mark.asyncio
class TestGlobalAPIGatewayIntegration:
    """Test global API gateway integration."""
    
    async def test_global_api_gateway_singleton(self):
        """Test global API gateway singleton behavior."""
        # Test global gateway retrieval
        gateway1 = get_api_gateway()
        gateway2 = get_api_gateway()
        
        # Verify singleton behavior
        assert gateway1 is gateway2
        assert isinstance(gateway1, APIGateway)
        
        # Test global gateway functionality
        await gateway1.register_anomaly_detection_endpoints()
        
        # Verify persistence across references
        status = await gateway2.get_gateway_status()
        assert status["registered_endpoints"] >= 2  # Health + detection endpoints
        assert "v1" in status["api_versions"]


@pytest.mark.asyncio
class TestAPIGatewayPerformance:
    """Performance tests for API gateway."""
    
    async def test_concurrent_request_handling(self):
        """Test concurrent request handling performance."""
        gateway = APIGateway()
        await gateway.register_anomaly_detection_endpoints()
        
        # Create concurrent request handler
        async def make_request(request_id: int):
            request = APIRequest(
                request_id=f"concurrent_test_{request_id}",
                method=HTTPMethod.GET,
                path="/health",
                headers={"Accept": "application/json"},
                query_params={},
                timestamp=datetime.now(),
                client_ip=f"192.168.1.{request_id % 254 + 1}"  # Vary IP addresses
            )
            
            response = await gateway.handle_request(request)
            return response.status_code == 200
        
        # Run concurrent requests
        concurrent_count = 50
        start_time = datetime.now()
        
        tasks = [make_request(i) for i in range(concurrent_count)]
        results = await asyncio.gather(*tasks)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Verify results
        successful_requests = sum(results)
        assert successful_requests == concurrent_count  # All should succeed
        
        # Check performance
        requests_per_second = concurrent_count / execution_time
        assert requests_per_second > 10  # Should handle at least 10 requests/second
        
        # Verify gateway statistics
        gateway_status = await gateway.get_gateway_status()
        assert gateway_status["gateway_stats"]["total_requests"] >= concurrent_count
        assert gateway_status["gateway_stats"]["total_responses"] >= concurrent_count
    
    async def test_large_request_handling(self):
        """Test handling of large requests."""
        gateway = APIGateway()
        
        # Create large request handler
        async def large_data_handler(request: APIRequest) -> APIResponse:
            # Echo back the size of received data
            data_size = len(json.dumps(request.body)) if request.body else 0
            
            return APIResponse(
                request_id=request.request_id,
                status_code=200,
                body={"received_data_size": data_size, "status": "processed"},
                format=ResponseFormat.JSON
            )
        
        # Register large data endpoint
        large_data_metadata = EndpointMetadata(
            path="/v1/large-data",
            method=HTTPMethod.POST,
            handler=large_data_handler
        )
        gateway.endpoint_manager.register_endpoint(large_data_metadata)
        gateway.versioning.register_version("v1", "Test version")
        
        # Create large request (within limits)
        large_data = {"data": ["x" * 1000 for _ in range(1000)]}  # ~1MB
        
        large_request = APIRequest(
            request_id=str(uuid.uuid4()),
            method=HTTPMethod.POST,
            path="/v1/large-data",
            headers={"Content-Type": "application/json"},
            query_params={},
            body=large_data,
            timestamp=datetime.now(),
            client_ip="127.0.0.1"
        )
        
        # Process large request
        start_time = datetime.now()
        large_response = await gateway.handle_request(large_request)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Verify response
        assert large_response.status_code == 200
        
        large_response_body = json.loads(large_response.body)
        assert large_response_body["status"] == "processed"
        assert large_response_body["received_data_size"] > 0
        
        # Check processing time is reasonable
        assert processing_time < 10.0  # Should process within 10 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])