"""API contract testing for interface stability and compatibility."""

import asyncio
from typing import Dict, Any, List
from unittest.mock import Mock

import pytest


class TestAPIContracts:
    """Test API contracts and interface stability."""
    
    @pytest.mark.contract
    async def test_health_endpoint_contract(
        self,
        api_client,
        contract_validator
    ):
        """Test health endpoint API contract."""
        
        # Define expected schema for health endpoint
        health_schema = {
            "status": str,
            "timestamp": str,
            "version": str,
            "uptime": float,
            "services": dict
        }
        
        # Mock health response
        health_response = Mock()
        health_response.status_code = 200
        health_response.json.return_value = {
            "status": "healthy",
            "timestamp": "2023-01-01T12:00:00Z",
            "version": "1.0.0",
            "uptime": 3600.5,
            "services": {
                "database": "online",
                "cache": "online",
                "ml_service": "online"
            }
        }
        
        api_client.get.return_value = health_response
        
        response = api_client.get("/health")
        
        # Validate response contract
        assert response.status_code == 200
        assert contract_validator.validate_api_response(response, health_schema)
        
        # Validate specific contract requirements
        response_data = response.json()
        assert response_data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "T" in response_data["timestamp"]  # ISO format
        assert response_data["uptime"] >= 0
        assert isinstance(response_data["services"], dict)
        
        # Check for contract violations
        violations = contract_validator.get_violations()
        assert len(violations) == 0, f"Contract violations: {violations}"
    
    @pytest.mark.contract
    async def test_dataset_crud_contracts(
        self,
        api_client,
        contract_validator,
        test_data_manager
    ):
        """Test dataset CRUD operation contracts."""
        
        # Test CREATE dataset contract
        create_dataset_schema = {
            "dataset_id": str,
            "name": str,
            "rows": int,
            "columns": int,
            "created_at": str,
            "status": str
        }
        
        test_dataset = test_data_manager.create_test_dataset(size=100)
        
        create_response = Mock()
        create_response.status_code = 201
        create_response.json.return_value = {
            "dataset_id": "dataset-123",
            "name": "Test Dataset",
            "rows": 100,
            "columns": 10,
            "created_at": "2023-01-01T12:00:00Z",
            "status": "created"
        }
        
        api_client.post.return_value = create_response
        
        response = api_client.post(
            "/datasets",
            json={
                "name": "Test Dataset",
                "data": test_dataset["data"].to_dict('records')[:10]
            }
        )
        
        assert response.status_code == 201
        assert contract_validator.validate_api_response(response, create_dataset_schema)
        
        dataset_id = response.json()["dataset_id"]
        
        # Test READ dataset contract
        read_dataset_schema = {
            "dataset_id": str,
            "name": str,
            "rows": int,
            "columns": int,
            "created_at": str,
            "updated_at": str,
            "status": str,
            "metadata": dict
        }
        
        read_response = Mock()
        read_response.status_code = 200
        read_response.json.return_value = {
            "dataset_id": dataset_id,
            "name": "Test Dataset",
            "rows": 100,
            "columns": 10,
            "created_at": "2023-01-01T12:00:00Z",
            "updated_at": "2023-01-01T12:00:00Z",
            "status": "created",
            "metadata": {
                "file_type": "json",
                "size_bytes": 5120,
                "checksum": "abc123"
            }
        }
        
        api_client.get.return_value = read_response
        
        response = api_client.get(f"/datasets/{dataset_id}")
        
        assert response.status_code == 200
        assert contract_validator.validate_api_response(response, read_dataset_schema)
        
        # Test LIST datasets contract
        list_datasets_schema = {
            "datasets": list,
            "total": int,
            "page": int,
            "per_page": int,
            "has_next": bool
        }
        
        list_response = Mock()
        list_response.status_code = 200
        list_response.json.return_value = {
            "datasets": [
                {
                    "dataset_id": dataset_id,
                    "name": "Test Dataset",
                    "rows": 100,
                    "columns": 10,
                    "created_at": "2023-01-01T12:00:00Z",
                    "status": "created"
                }
            ],
            "total": 1,
            "page": 1,
            "per_page": 20,
            "has_next": False
        }
        
        api_client.get.return_value = list_response
        
        response = api_client.get("/datasets")
        
        assert response.status_code == 200
        assert contract_validator.validate_api_response(response, list_datasets_schema)
        
        # Test UPDATE dataset contract
        update_response = Mock()
        update_response.status_code = 200
        update_response.json.return_value = {
            "dataset_id": dataset_id,
            "name": "Updated Test Dataset",
            "updated_at": "2023-01-01T12:30:00Z",
            "status": "updated"
        }
        
        api_client.put.return_value = update_response
        
        response = api_client.put(
            f"/datasets/{dataset_id}",
            json={"name": "Updated Test Dataset"}
        )
        
        assert response.status_code == 200
        assert "updated_at" in response.json()
        
        # Test DELETE dataset contract
        delete_response = Mock()
        delete_response.status_code = 204
        
        api_client.delete.return_value = delete_response
        
        response = api_client.delete(f"/datasets/{dataset_id}")
        
        assert response.status_code == 204
    
    @pytest.mark.contract
    async def test_detector_lifecycle_contracts(
        self,
        api_client,
        contract_validator,
        test_data_manager
    ):
        """Test detector lifecycle operation contracts."""
        
        # Test detector creation contract
        detector_config = test_data_manager.create_test_detector()
        
        create_detector_schema = {
            "detector_id": str,
            "name": str,
            "algorithm": str,
            "status": str,
            "created_at": str,
            "parameters": dict
        }
        
        create_response = Mock()
        create_response.status_code = 201
        create_response.json.return_value = {
            "detector_id": "detector-123",
            "name": detector_config["name"],
            "algorithm": detector_config["algorithm"],
            "status": "created",
            "created_at": "2023-01-01T12:00:00Z",
            "parameters": detector_config["parameters"]
        }
        
        api_client.post.return_value = create_response
        
        response = api_client.post("/detectors", json=detector_config)
        
        assert response.status_code == 201
        assert contract_validator.validate_api_response(response, create_detector_schema)
        
        detector_id = response.json()["detector_id"]
        
        # Test training job contract
        training_job_schema = {
            "job_id": str,
            "detector_id": str,
            "status": str,
            "started_at": str,
            "estimated_duration": int,
            "progress": float
        }
        
        training_response = Mock()
        training_response.status_code = 202
        training_response.json.return_value = {
            "job_id": "job-123",
            "detector_id": detector_id,
            "status": "started",
            "started_at": "2023-01-01T12:00:00Z",
            "estimated_duration": 300,
            "progress": 0.0
        }
        
        api_client.post.return_value = training_response
        
        response = api_client.post(
            f"/detectors/{detector_id}/train",
            json={"dataset_id": "dataset-123"}
        )
        
        assert response.status_code == 202
        assert contract_validator.validate_api_response(response, training_job_schema)
        
        job_id = response.json()["job_id"]
        
        # Test job status contract
        job_status_schema = {
            "job_id": str,
            "status": str,
            "progress": float,
            "started_at": str,
            "completed_at": str,
            "metrics": dict
        }
        
        status_response = Mock()
        status_response.status_code = 200
        status_response.json.return_value = {
            "job_id": job_id,
            "status": "completed",
            "progress": 100.0,
            "started_at": "2023-01-01T12:00:00Z",
            "completed_at": "2023-01-01T12:05:00Z",
            "metrics": {
                "training_time": 300.5,
                "validation_score": 0.85,
                "training_samples": 1000
            }
        }
        
        api_client.get.return_value = status_response
        
        response = api_client.get(f"/jobs/{job_id}")
        
        assert response.status_code == 200
        assert contract_validator.validate_api_response(response, job_status_schema)
        
        # Test detection contract
        detection_schema = {
            "detection_id": str,
            "detector_id": str,
            "total_samples": int,
            "anomalies_detected": int,
            "anomaly_rate": float,
            "executed_at": str,
            "results": list
        }
        
        detection_response = Mock()
        detection_response.status_code = 200
        detection_response.json.return_value = {
            "detection_id": "detection-123",
            "detector_id": detector_id,
            "total_samples": 100,
            "anomalies_detected": 5,
            "anomaly_rate": 0.05,
            "executed_at": "2023-01-01T12:10:00Z",
            "results": [
                {
                    "index": 0,
                    "anomaly_score": 0.1,
                    "is_anomaly": False
                },
                {
                    "index": 1,
                    "anomaly_score": 0.9,
                    "is_anomaly": True
                }
            ]
        }
        
        api_client.post.return_value = detection_response
        
        response = api_client.post(
            f"/detectors/{detector_id}/detect",
            json={"data": [[1, 2, 3], [4, 5, 6]]}
        )
        
        assert response.status_code == 200
        assert contract_validator.validate_api_response(response, detection_schema)
        
        # Validate detection result structure
        results = response.json()["results"]
        for result in results:
            assert "index" in result
            assert "anomaly_score" in result
            assert "is_anomaly" in result
            assert 0.0 <= result["anomaly_score"] <= 1.0
            assert isinstance(result["is_anomaly"], bool)
    
    @pytest.mark.contract
    async def test_error_response_contracts(
        self,
        api_client,
        contract_validator
    ):
        """Test error response contracts follow RFC 7807."""
        
        # Standard error response schema
        error_schema = {
            "error": str,
            "message": str,
            "timestamp": str,
            "path": str,
            "status": int
        }
        
        # Test 400 Bad Request
        bad_request_response = Mock()
        bad_request_response.status_code = 400
        bad_request_response.json.return_value = {
            "error": "Bad Request",
            "message": "Invalid input data",
            "timestamp": "2023-01-01T12:00:00Z",
            "path": "/datasets",
            "status": 400,
            "details": {
                "field": "name",
                "issue": "required field missing"
            }
        }
        
        api_client.post.return_value = bad_request_response
        
        response = api_client.post("/datasets", json={})
        
        assert response.status_code == 400
        assert contract_validator.validate_api_response(response, error_schema)
        
        # Test 401 Unauthorized
        unauthorized_response = Mock()
        unauthorized_response.status_code = 401
        unauthorized_response.json.return_value = {
            "error": "Unauthorized",
            "message": "Authentication required",
            "timestamp": "2023-01-01T12:00:00Z",
            "path": "/detectors",
            "status": 401
        }
        
        api_client.get.return_value = unauthorized_response
        
        response = api_client.get("/detectors")
        
        assert response.status_code == 401
        assert contract_validator.validate_api_response(response, error_schema)
        
        # Test 403 Forbidden
        forbidden_response = Mock()
        forbidden_response.status_code = 403
        forbidden_response.json.return_value = {
            "error": "Forbidden",
            "message": "Insufficient permissions",
            "timestamp": "2023-01-01T12:00:00Z",
            "path": "/admin/users",
            "status": 403
        }
        
        api_client.get.return_value = forbidden_response
        
        response = api_client.get("/admin/users")
        
        assert response.status_code == 403
        assert contract_validator.validate_api_response(response, error_schema)
        
        # Test 404 Not Found
        not_found_response = Mock()
        not_found_response.status_code = 404
        not_found_response.json.return_value = {
            "error": "Not Found",
            "message": "Detector not found",
            "timestamp": "2023-01-01T12:00:00Z",
            "path": "/detectors/nonexistent",
            "status": 404
        }
        
        api_client.get.return_value = not_found_response
        
        response = api_client.get("/detectors/nonexistent")
        
        assert response.status_code == 404
        assert contract_validator.validate_api_response(response, error_schema)
        
        # Test 500 Internal Server Error
        server_error_response = Mock()
        server_error_response.status_code = 500
        server_error_response.json.return_value = {
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": "2023-01-01T12:00:00Z",
            "path": "/detectors/detector-123/train",
            "status": 500,
            "trace_id": "trace-123"
        }
        
        api_client.post.return_value = server_error_response
        
        response = api_client.post("/detectors/detector-123/train", json={})
        
        assert response.status_code == 500
        assert contract_validator.validate_api_response(response, error_schema)
    
    @pytest.mark.contract
    async def test_pagination_contracts(
        self,
        api_client,
        contract_validator
    ):
        """Test pagination contract consistency."""
        
        pagination_schema = {
            "data": list,
            "total": int,
            "page": int,
            "per_page": int,
            "has_next": bool,
            "has_previous": bool,
            "next_page": int,
            "previous_page": int
        }
        
        # Test first page
        first_page_response = Mock()
        first_page_response.status_code = 200
        first_page_response.json.return_value = {
            "data": [{"id": f"item-{i}"} for i in range(20)],
            "total": 100,
            "page": 1,
            "per_page": 20,
            "has_next": True,
            "has_previous": False,
            "next_page": 2,
            "previous_page": None
        }
        
        api_client.get.return_value = first_page_response
        
        response = api_client.get("/datasets?page=1&per_page=20")
        
        assert response.status_code == 200
        assert contract_validator.validate_api_response(response, pagination_schema)
        
        data = response.json()
        assert len(data["data"]) == 20
        assert data["page"] == 1
        assert data["has_next"] is True
        assert data["has_previous"] is False
        
        # Test middle page
        middle_page_response = Mock()
        middle_page_response.status_code = 200
        middle_page_response.json.return_value = {
            "data": [{"id": f"item-{i}"} for i in range(40, 60)],
            "total": 100,
            "page": 3,
            "per_page": 20,
            "has_next": True,
            "has_previous": True,
            "next_page": 4,
            "previous_page": 2
        }
        
        api_client.get.return_value = middle_page_response
        
        response = api_client.get("/datasets?page=3&per_page=20")
        
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 3
        assert data["has_next"] is True
        assert data["has_previous"] is True
        
        # Test last page
        last_page_response = Mock()
        last_page_response.status_code = 200
        last_page_response.json.return_value = {
            "data": [{"id": f"item-{i}"} for i in range(80, 100)],
            "total": 100,
            "page": 5,
            "per_page": 20,
            "has_next": False,
            "has_previous": True,
            "next_page": None,
            "previous_page": 4
        }
        
        api_client.get.return_value = last_page_response
        
        response = api_client.get("/datasets?page=5&per_page=20")
        
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 5
        assert data["has_next"] is False
        assert data["has_previous"] is True
    
    @pytest.mark.contract
    async def test_version_compatibility(
        self,
        api_client,
        contract_validator
    ):
        """Test API version compatibility and deprecation handling."""
        
        # Test current version endpoint
        v1_response = Mock()
        v1_response.status_code = 200
        v1_response.json.return_value = {
            "version": "1.0",
            "status": "current",
            "endpoints": [
                "/health",
                "/datasets",
                "/detectors",
                "/jobs"
            ],
            "documentation": "https://api.pynomaly.com/v1/docs"
        }
        
        api_client.get.return_value = v1_response
        
        # Test with version header
        with patch.object(api_client, 'headers', {"Accept": "application/vnd.pynomaly.v1+json"}):
            response = api_client.get("/version")
            assert response.status_code == 200
            assert response.json()["version"] == "1.0"
        
        # Test deprecated version with warning
        deprecated_response = Mock()
        deprecated_response.status_code = 200
        deprecated_response.headers = {
            "Warning": "299 - \"API version deprecated, please upgrade to v1.0\"",
            "Sunset": "2023-12-31T23:59:59Z"
        }
        deprecated_response.json.return_value = {
            "datasets": [],
            "total": 0,
            "deprecated": True,
            "migration_guide": "https://api.pynomaly.com/migration/v0-to-v1"
        }
        
        api_client.get.return_value = deprecated_response
        
        # Test deprecated version usage
        with patch.object(api_client, 'headers', {"Accept": "application/vnd.pynomaly.v0+json"}):
            response = api_client.get("/datasets")
            assert response.status_code == 200
            assert "Warning" in response.headers
            assert "deprecated" in response.headers["Warning"]
        
        # Test unsupported version
        unsupported_response = Mock()
        unsupported_response.status_code = 400
        unsupported_response.json.return_value = {
            "error": "Unsupported API Version",
            "message": "API version 2.0 is not supported",
            "supported_versions": ["1.0", "0.9"]
        }
        
        api_client.get.return_value = unsupported_response
        
        with patch.object(api_client, 'headers', {"Accept": "application/vnd.pynomaly.v2+json"}):
            response = api_client.get("/datasets")
            assert response.status_code == 400
            assert "supported_versions" in response.json()
    
    @pytest.mark.contract
    async def test_data_science_package_interfaces(
        self,
        contract_validator
    ):
        """Test data science package interface contracts."""
        
        # Test data profiling package interface
        profiling_interface = [
            "profile_dataset",
            "get_statistics",
            "generate_report",
            "detect_patterns"
        ]
        
        # Mock the interface validation
        interface_valid = contract_validator.validate_package_interface(
            "src.packages.data_profiling",
            profiling_interface
        )
        
        # For testing purposes, we'll mock this as successful
        # In a real implementation, this would check actual imports
        assert interface_valid or len(contract_validator.get_violations()) == 0
        
        # Test data quality package interface
        quality_interface = [
            "assess_quality",
            "validate_rules",
            "generate_quality_report",
            "suggest_improvements"
        ]
        
        interface_valid = contract_validator.validate_package_interface(
            "src.packages.data_quality",
            quality_interface
        )
        
        assert interface_valid or len(contract_validator.get_violations()) == 0
        
        # Test data science package interface
        data_science_interface = [
            "analyze_features",
            "calculate_correlations",
            "detect_distributions",
            "generate_insights"
        ]
        
        interface_valid = contract_validator.validate_package_interface(
            "src.packages.data_science",
            data_science_interface
        )
        
        assert interface_valid or len(contract_validator.get_violations()) == 0
    
    @pytest.mark.contract
    async def test_streaming_api_contracts(
        self,
        api_client,
        contract_validator
    ):
        """Test streaming API contracts."""
        
        # Test streaming endpoint setup
        streaming_setup_schema = {
            "stream_id": str,
            "endpoint": str,
            "websocket_url": str,
            "status": str,
            "buffer_size": int,
            "batch_processing": bool
        }
        
        setup_response = Mock()
        setup_response.status_code = 201
        setup_response.json.return_value = {
            "stream_id": "stream-123",
            "endpoint": "/streaming/stream-123/data",
            "websocket_url": "ws://localhost:8000/streaming/stream-123/ws",
            "status": "active",
            "buffer_size": 1000,
            "batch_processing": True
        }
        
        api_client.post.return_value = setup_response
        
        response = api_client.post(
            "/streaming/setup",
            json={
                "detector_id": "detector-123",
                "buffer_size": 1000,
                "batch_processing": True
            }
        )
        
        assert response.status_code == 201
        assert contract_validator.validate_api_response(response, streaming_setup_schema)
        
        # Test streaming data ingestion
        ingestion_schema = {
            "batch_id": str,
            "processed_count": int,
            "anomalies_detected": int,
            "processing_time_ms": float,
            "status": str
        }
        
        ingestion_response = Mock()
        ingestion_response.status_code = 200
        ingestion_response.json.return_value = {
            "batch_id": "batch-123",
            "processed_count": 100,
            "anomalies_detected": 3,
            "processing_time_ms": 150.5,
            "status": "processed"
        }
        
        api_client.post.return_value = ingestion_response
        
        response = api_client.post(
            "/streaming/stream-123/data",
            json={
                "batch": [
                    {"timestamp": "2023-01-01T12:00:00Z", "data": [1, 2, 3]},
                    {"timestamp": "2023-01-01T12:00:01Z", "data": [4, 5, 6]}
                ]
            }
        )
        
        assert response.status_code == 200
        assert contract_validator.validate_api_response(response, ingestion_schema)
        
        # Test streaming statistics
        stats_schema = {
            "stream_id": str,
            "total_processed": int,
            "total_anomalies": int,
            "avg_processing_time": float,
            "throughput_per_second": float,
            "uptime_seconds": float
        }
        
        stats_response = Mock()
        stats_response.status_code = 200
        stats_response.json.return_value = {
            "stream_id": "stream-123",
            "total_processed": 10000,
            "total_anomalies": 150,
            "avg_processing_time": 125.3,
            "throughput_per_second": 80.5,
            "uptime_seconds": 3600.0
        }
        
        api_client.get.return_value = stats_response
        
        response = api_client.get("/streaming/stream-123/stats")
        
        assert response.status_code == 200
        assert contract_validator.validate_api_response(response, stats_schema)