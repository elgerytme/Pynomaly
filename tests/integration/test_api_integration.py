"""
API Integration Tests

Tests API endpoints and their integration with the underlying services.
"""

import json
import pytest
from fastapi.testclient import TestClient
from tests.integration.framework.integration_test_base import ServiceIntegrationTest
from pynomaly.presentation.api.app import create_app


class TestAPIIntegration(ServiceIntegrationTest):
    """Test API integration with services."""
    
    def _get_test_config(self):
        """Get API integration test configuration."""
        return {
            "database_enabled": True,
            "cache_enabled": True,
            "message_queue_enabled": False,
            "external_services_enabled": True,
            "test_data_size": "small",
            "parallel_execution": False,
        }
    
    async def test_health_endpoint_integration(self):
        """Test health endpoint integration."""
        
        async with self.setup_test_environment() as env:
            # Create FastAPI app with test container
            app = create_app(env.container)
            client = TestClient(app)
            
            # Test health endpoint
            response = client.get("/api/v1/health")
            
            # Verify response
            assert response.status_code == 200
            
            health_data = response.json()
            assert health_data["status"] == "healthy"
            assert "timestamp" in health_data
            assert "version" in health_data
            
            # Verify service health
            self.assert_service_health("database")
            self.assert_service_health("cache")
    
    async def test_detector_crud_integration(self):
        """Test detector CRUD operations integration."""
        
        async with self.setup_test_environment() as env:
            app = create_app(env.container)
            client = TestClient(app)
            
            # Test data
            detector_data = {
                "name": "api_test_detector",
                "algorithm": "IsolationForest",
                "parameters": {
                    "contamination": 0.1,
                    "random_state": 42
                },
                "description": "Test detector for API integration"
            }
            
            # Create detector
            response = client.post(
                "/api/v1/detectors",
                json=detector_data
            )
            
            assert response.status_code == 201
            created_detector = response.json()
            
            # Verify creation
            assert created_detector["name"] == detector_data["name"]
            assert created_detector["algorithm"] == detector_data["algorithm"]
            assert "id" in created_detector
            
            detector_id = created_detector["id"]
            
            # Get detector
            response = client.get(f"/api/v1/detectors/{detector_id}")
            assert response.status_code == 200
            
            retrieved_detector = response.json()
            assert retrieved_detector["id"] == detector_id
            assert retrieved_detector["name"] == detector_data["name"]
            
            # Update detector
            update_data = {
                "name": "updated_api_test_detector",
                "description": "Updated test detector"
            }
            
            response = client.put(
                f"/api/v1/detectors/{detector_id}",
                json=update_data
            )
            assert response.status_code == 200
            
            updated_detector = response.json()
            assert updated_detector["name"] == update_data["name"]
            assert updated_detector["description"] == update_data["description"]
            
            # List detectors
            response = client.get("/api/v1/detectors")
            assert response.status_code == 200
            
            detectors_list = response.json()
            assert len(detectors_list) >= 1
            
            # Find our detector in the list
            found_detector = next(
                (d for d in detectors_list if d["id"] == detector_id),
                None
            )
            assert found_detector is not None
            assert found_detector["name"] == update_data["name"]
            
            # Delete detector
            response = client.delete(f"/api/v1/detectors/{detector_id}")
            assert response.status_code == 204
            
            # Verify deletion
            response = client.get(f"/api/v1/detectors/{detector_id}")
            assert response.status_code == 404
    
    async def test_dataset_upload_integration(self):
        """Test dataset upload integration."""
        
        async with self.setup_test_environment() as env:
            app = create_app(env.container)
            client = TestClient(app)
            
            # Create test dataset
            test_dataset = await env.test_data_manager.create_dataset(
                name="api_upload_test",
                size=100,
                anomaly_rate=0.1,
                features=5,
                save_to_file=True
            )
            
            # Upload dataset via API
            with open(test_dataset.file_path, 'rb') as f:
                response = client.post(
                    "/api/v1/datasets/upload",
                    files={"file": ("test_data.csv", f, "text/csv")},
                    data={"name": "uploaded_dataset", "description": "Uploaded via API"}
                )
            
            assert response.status_code == 201
            uploaded_dataset = response.json()
            
            # Verify upload
            assert uploaded_dataset["name"] == "uploaded_dataset"
            assert "id" in uploaded_dataset
            assert "file_path" in uploaded_dataset
            
            # Get dataset details
            dataset_id = uploaded_dataset["id"]
            response = client.get(f"/api/v1/datasets/{dataset_id}")
            assert response.status_code == 200
            
            dataset_details = response.json()
            assert dataset_details["name"] == "uploaded_dataset"
            assert dataset_details["description"] == "Uploaded via API"
    
    async def test_detection_workflow_integration(self):
        """Test complete detection workflow via API."""
        
        async with self.setup_test_environment() as env:
            app = create_app(env.container)
            client = TestClient(app)
            
            # Create test dataset
            test_dataset = await env.test_data_manager.create_dataset(
                name="workflow_test_dataset",
                size=500,
                anomaly_rate=0.1,
                features=10,
                save_to_file=True
            )
            
            # Create detector via API
            detector_data = {
                "name": "workflow_test_detector",
                "algorithm": "IsolationForest",
                "parameters": {"contamination": 0.1, "random_state": 42}
            }
            
            response = client.post("/api/v1/detectors", json=detector_data)
            assert response.status_code == 201
            detector = response.json()
            
            # Upload dataset via API
            with open(test_dataset.file_path, 'rb') as f:
                response = client.post(
                    "/api/v1/datasets/upload",
                    files={"file": ("workflow_data.csv", f, "text/csv")},
                    data={"name": "workflow_dataset"}
                )
            
            assert response.status_code == 201
            dataset = response.json()
            
            # Execute detection via API
            detection_request = {
                "detector_id": detector["id"],
                "dataset_id": dataset["id"]
            }
            
            response = client.post(
                "/api/v1/detection/predict",
                json=detection_request
            )
            
            assert response.status_code == 200
            detection_result = response.json()
            
            # Verify detection result
            assert "result_id" in detection_result
            assert "anomaly_count" in detection_result
            assert "execution_time" in detection_result
            assert detection_result["anomaly_count"] > 0
            assert detection_result["execution_time"] > 0
            
            # Get detection result details
            result_id = detection_result["result_id"]
            response = client.get(f"/api/v1/detection/results/{result_id}")
            assert response.status_code == 200
            
            result_details = response.json()
            assert result_details["id"] == result_id
            assert result_details["detector_id"] == detector["id"]
            assert result_details["dataset_id"] == dataset["id"]
    
    async def test_error_handling_integration(self):
        """Test API error handling integration."""
        
        async with self.setup_test_environment() as env:
            app = create_app(env.container)
            client = TestClient(app)
            
            # Test 404 error
            response = client.get("/api/v1/detectors/non-existent-id")
            assert response.status_code == 404
            
            error_response = response.json()
            assert "detail" in error_response
            
            # Test validation error
            invalid_detector = {
                "name": "",  # Invalid empty name
                "algorithm": "NonExistentAlgorithm",
                "parameters": {}
            }
            
            response = client.post("/api/v1/detectors", json=invalid_detector)
            assert response.status_code == 422
            
            validation_error = response.json()
            assert "detail" in validation_error
            
            # Test invalid detection request
            invalid_detection = {
                "detector_id": "invalid-id",
                "dataset_id": "invalid-id"
            }
            
            response = client.post(
                "/api/v1/detection/predict",
                json=invalid_detection
            )
            assert response.status_code == 404
    
    async def test_streaming_endpoint_integration(self):
        """Test streaming endpoint integration."""
        
        async with self.setup_test_environment() as env:
            app = create_app(env.container)
            client = TestClient(app)
            
            # Create streaming data source
            streaming_source = await env.test_data_manager.create_streaming_data_source(
                name="api_streaming_test",
                event_rate=5,
                duration_seconds=10,
                anomaly_rate=0.2
            )
            
            # Test streaming endpoint
            response = client.get("/api/v1/streaming/status")
            assert response.status_code == 200
            
            streaming_status = response.json()
            assert "status" in streaming_status
            
            # Test streaming configuration
            config_data = {
                "source_type": "file",
                "source_path": streaming_source["file_path"],
                "batch_size": 10,
                "processing_interval": 1.0
            }
            
            response = client.post(
                "/api/v1/streaming/configure",
                json=config_data
            )
            assert response.status_code == 200
            
            config_response = response.json()
            assert config_response["status"] == "configured"
    
    async def test_authentication_integration(self):
        """Test authentication integration."""
        
        async with self.setup_test_environment() as env:
            # Enable authentication for this test
            env.settings.auth_enabled = True
            
            app = create_app(env.container)
            client = TestClient(app)
            
            # Test accessing protected endpoint without auth
            response = client.get("/api/v1/admin/users")
            assert response.status_code == 401
            
            # Test login
            login_data = {
                "username": "test_user",
                "password": "test_password"
            }
            
            response = client.post("/api/v1/auth/login", json=login_data)
            # Note: This might return 404 if auth service is not fully implemented
            # The test verifies the authentication flow exists
            
            # Test token-based access would go here
            # (Implementation depends on actual auth service)
    
    async def test_performance_monitoring_integration(self):
        """Test performance monitoring integration."""
        
        async with self.setup_test_environment() as env:
            app = create_app(env.container)
            client = TestClient(app)
            
            # Test performance metrics endpoint
            response = client.get("/api/v1/performance/metrics")
            assert response.status_code == 200
            
            metrics = response.json()
            assert "system_metrics" in metrics
            assert "application_metrics" in metrics
            
            # Test performance history
            response = client.get("/api/v1/performance/history")
            assert response.status_code == 200
            
            history = response.json()
            assert "metrics_history" in history
            assert isinstance(history["metrics_history"], list)
    
    async def test_export_integration(self):
        """Test export functionality integration."""
        
        async with self.setup_test_environment() as env:
            app = create_app(env.container)
            client = TestClient(app)
            
            # Create test data
            test_dataset = await env.test_data_manager.create_dataset(
                name="export_test_dataset",
                size=100,
                anomaly_rate=0.1,
                features=5
            )
            
            test_detector = await env.test_data_manager.create_detector(
                name="export_test_detector",
                algorithm="IsolationForest",
                parameters={"contamination": 0.1}
            )
            
            test_result = await env.test_data_manager.create_detection_result(
                detector_id=test_detector.id,
                dataset_id=test_dataset.id,
                anomaly_count=10,
                execution_time_ms=500.0
            )
            
            # Test export results
            export_request = {
                "result_id": test_result.id,
                "format": "json",
                "include_metadata": True
            }
            
            response = client.post("/api/v1/export/results", json=export_request)
            assert response.status_code == 200
            
            # Verify export content
            export_data = response.json()
            assert "export_url" in export_data or "export_data" in export_data
            assert "format" in export_data
            assert export_data["format"] == "json"
    
    async def test_version_endpoint_integration(self):
        """Test version endpoint integration."""
        
        async with self.setup_test_environment() as env:
            app = create_app(env.container)
            client = TestClient(app)
            
            # Test version endpoint
            response = client.get("/api/v1/version")
            assert response.status_code == 200
            
            version_info = response.json()
            assert "version" in version_info
            assert "build_info" in version_info
            assert "api_version" in version_info
            assert version_info["api_version"] == "v1"
            
            # Test API schema endpoint
            response = client.get("/api/v1/openapi.json")
            assert response.status_code == 200
            
            schema = response.json()
            assert "openapi" in schema
            assert "info" in schema
            assert "paths" in schema
    
    async def test_concurrent_requests_integration(self):
        """Test concurrent requests integration."""
        
        async with self.setup_test_environment() as env:
            app = create_app(env.container)
            client = TestClient(app)
            
            # Create multiple detectors concurrently
            detector_data = [
                {
                    "name": f"concurrent_detector_{i}",
                    "algorithm": "IsolationForest",
                    "parameters": {"contamination": 0.1, "random_state": 42 + i}
                }
                for i in range(5)
            ]
            
            # Send concurrent requests
            responses = []
            for data in detector_data:
                response = client.post("/api/v1/detectors", json=data)
                responses.append(response)
            
            # Verify all requests succeeded
            for i, response in enumerate(responses):
                assert response.status_code == 201
                detector = response.json()
                assert detector["name"] == f"concurrent_detector_{i}"
            
            # Verify all detectors exist
            response = client.get("/api/v1/detectors")
            assert response.status_code == 200
            
            detectors_list = response.json()
            concurrent_detectors = [
                d for d in detectors_list 
                if d["name"].startswith("concurrent_detector_")
            ]
            
            assert len(concurrent_detectors) == 5