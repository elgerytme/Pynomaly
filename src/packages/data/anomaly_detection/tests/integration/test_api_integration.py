"""Integration tests for FastAPI endpoints."""

import pytest
import json
from typing import Dict, Any
from fastapi.testclient import TestClient

from anomaly_detection.server import create_app


class TestAPIIntegration:
    """Integration tests for FastAPI endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    def test_health_endpoint(self, client: TestClient):
        """Test basic health endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "service" in data
        assert "version" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"
        assert data["service"] == "anomaly-detection-api"
    
    def test_detailed_health_check(self, client: TestClient):
        """Test detailed health check endpoint."""
        response = client.get("/api/v1/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "overall_status" in data
        assert "checks" in data
        assert "service_info" in data
        
        # Should have various health checks
        checks = data["checks"]
        expected_checks = ["algorithms", "model_repository", "memory", "disk"]
        
        for check_name in expected_checks:
            if check_name in checks:
                check_data = checks[check_name]
                assert "status" in check_data
                assert "message" in check_data
                assert "timestamp" in check_data
    
    def test_readiness_check(self, client: TestClient):
        """Test Kubernetes readiness probe."""
        response = client.get("/api/v1/health/readiness")
        
        # Should return 200 if ready or 503 if not ready
        assert response.status_code in [200, 503]
        
        data = response.json()
        assert "ready" in data
        assert "checks" in data
        assert "timestamp" in data
        
        if response.status_code == 200:
            assert data["ready"] is True
    
    def test_liveness_check(self, client: TestClient):
        """Test Kubernetes liveness probe."""
        response = client.get("/api/v1/health/liveness")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "alive" in data
        assert "timestamp" in data
        assert data["alive"] is True
    
    def test_anomaly_detection_endpoint(self, client: TestClient):
        """Test main anomaly detection endpoint."""
        # Prepare test data
        test_data = {
            "data": [
                [0.0, 0.0],
                [1.0, 1.0],
                [0.5, 0.5],
                [10.0, 10.0],  # Likely anomaly
                [0.2, 0.3]
            ],
            "algorithm": "isolation_forest",
            "contamination": 0.2,
            "parameters": {}
        }
        
        response = client.post("/api/v1/detect", json=test_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        required_fields = [
            "success", "anomalies", "algorithm", "total_samples", 
            "anomalies_detected", "anomaly_rate", "timestamp", 
            "processing_time_ms"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        assert data["success"] is True
        assert data["algorithm"] == "isolation_forest"
        assert data["total_samples"] == 5
        assert isinstance(data["anomalies"], list)
        assert isinstance(data["anomalies_detected"], int)
        assert 0 <= data["anomaly_rate"] <= 1
        assert data["processing_time_ms"] > 0
    
    def test_anomaly_detection_different_algorithms(self, client: TestClient):
        """Test detection with different algorithms."""
        test_data_base = {
            "data": [
                [0.0, 0.0], [1.0, 1.0], [0.5, 0.5], 
                [10.0, 10.0], [0.2, 0.3], [-0.1, 0.1]
            ],
            "contamination": 0.15
        }
        
        algorithms = ["isolation_forest", "local_outlier_factor", "lof"]
        
        for algorithm in algorithms:
            test_data = {**test_data_base, "algorithm": algorithm}
            
            response = client.post("/api/v1/detect", json=test_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["algorithm"] == algorithm
            assert data["total_samples"] == 6
    
    def test_ensemble_detection_endpoint(self, client: TestClient):
        """Test ensemble detection endpoint."""
        test_data = {
            "data": [
                [0.0, 0.0], [1.0, 1.0], [0.5, 0.5],
                [10.0, 10.0], [0.2, 0.3], [-0.1, 0.1],
                [15.0, -5.0]  # Another potential anomaly
            ],
            "algorithms": ["isolation_forest", "local_outlier_factor"],
            "method": "majority",
            "contamination": 0.2
        }
        
        response = client.post("/api/v1/ensemble", json=test_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["algorithm"] == "ensemble_majority"
        assert data["total_samples"] == 7
        assert isinstance(data["anomalies"], list)
    
    def test_ensemble_different_methods(self, client: TestClient):
        """Test ensemble with different combination methods."""
        test_data_base = {
            "data": [
                [0.0, 0.0], [1.0, 1.0], [0.5, 0.5],
                [10.0, 10.0], [0.2, 0.3]
            ],
            "algorithms": ["isolation_forest", "local_outlier_factor"],
            "contamination": 0.2
        }
        
        methods = ["majority", "average"]  # Test available methods
        
        for method in methods:
            test_data = {**test_data_base, "method": method}
            
            response = client.post("/api/v1/ensemble", json=test_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["algorithm"] == f"ensemble_{method}"
    
    def test_algorithms_list_endpoint(self, client: TestClient):
        """Test algorithm listing endpoint."""
        response = client.get("/api/v1/algorithms")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "single_algorithms" in data
        assert "ensemble_methods" in data
        assert "supported_formats" in data
        
        # Check that expected algorithms are listed
        single_algorithms = data["single_algorithms"]
        assert "isolation_forest" in single_algorithms
        assert "local_outlier_factor" in single_algorithms
        
        ensemble_methods = data["ensemble_methods"]
        assert "majority" in ensemble_methods
        assert "average" in ensemble_methods
    
    def test_models_list_endpoint(self, client: TestClient):
        """Test models listing endpoint."""
        response = client.get("/api/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "models" in data
        assert "total_count" in data
        
        # Initially should be empty or have test models
        assert isinstance(data["models"], list)
        assert isinstance(data["total_count"], int)
        assert data["total_count"] >= 0
    
    def test_metrics_endpoint(self, client: TestClient):
        """Test metrics endpoint."""
        response = client.get("/api/v1/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "metrics_summary" in data
        assert "performance_summary" in data
        assert "timestamp" in data
        
        # Check metrics structure
        metrics_summary = data["metrics_summary"]
        assert "collection_time" in metrics_summary
        assert "total_metrics" in metrics_summary
        
        performance_summary = data["performance_summary"]
        assert "total_profiles" in performance_summary
    
    def test_dashboard_summary_endpoint(self, client: TestClient):
        """Test dashboard summary endpoint."""
        response = client.get("/api/v1/dashboard/summary")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "summary" in data
        summary = data["summary"]
        
        expected_fields = [
            "overall_health_status", "total_operations", 
            "success_rate", "generated_at"
        ]
        
        for field in expected_fields:
            assert field in summary
    
    def test_input_validation_errors(self, client: TestClient):
        """Test API input validation."""
        # Empty data
        response = client.post("/api/v1/detect", json={
            "data": [],
            "algorithm": "isolation_forest"
        })
        assert response.status_code == 400
        
        # Invalid contamination rate
        response = client.post("/api/v1/detect", json={
            "data": [[1, 2], [3, 4]],
            "algorithm": "isolation_forest",
            "contamination": 1.5
        })
        assert response.status_code == 400
        
        # Invalid algorithm
        response = client.post("/api/v1/detect", json={
            "data": [[1, 2], [3, 4]],
            "algorithm": "nonexistent_algorithm"
        })
        assert response.status_code in [400, 500]  # Could be validation or processing error
        
        # Malformed JSON
        response = client.post("/api/v1/detect", 
                             data="invalid json",
                             headers={"Content-Type": "application/json"})
        assert response.status_code == 422  # FastAPI validation error
    
    def test_ensemble_validation_errors(self, client: TestClient):
        """Test ensemble endpoint validation."""
        # Single algorithm (should require at least 2)
        response = client.post("/api/v1/ensemble", json={
            "data": [[1, 2], [3, 4]],
            "algorithms": ["isolation_forest"]
        })
        assert response.status_code == 400
        
        # Empty algorithms list
        response = client.post("/api/v1/ensemble", json={
            "data": [[1, 2], [3, 4]],
            "algorithms": []
        })
        assert response.status_code == 400
    
    def test_error_response_format(self, client: TestClient):
        """Test that error responses have consistent format."""
        # Trigger a validation error
        response = client.post("/api/v1/detect", json={
            "data": [],
            "algorithm": "isolation_forest"
        })
        
        assert response.status_code == 400
        
        # Error response should be JSON
        assert response.headers.get("content-type", "").startswith("application/json")
        
        # Should have detail field (FastAPI standard)
        data = response.json()
        assert "detail" in data
    
    def test_performance_monitoring_integration(self, client: TestClient):
        """Test that API calls are properly monitored."""
        # Make a successful request
        response = client.post("/api/v1/detect", json={
            "data": [[0, 0], [1, 1], [10, 10]],
            "algorithm": "isolation_forest",
            "contamination": 0.3
        })
        
        assert response.status_code == 200
        
        # Check that metrics were recorded
        metrics_response = client.get("/api/v1/metrics")
        assert metrics_response.status_code == 200
        
        metrics_data = metrics_response.json()
        
        # Should have some performance data
        performance_summary = metrics_data.get("performance_summary", {})
        assert performance_summary.get("total_operations", 0) > 0
    
    def test_cors_headers(self, client: TestClient):
        """Test CORS headers are properly set."""
        response = client.options("/api/v1/detect")
        
        # Should have CORS headers
        headers = response.headers
        assert "access-control-allow-origin" in headers
        assert "access-control-allow-methods" in headers
        assert "access-control-allow-headers" in headers
    
    def test_large_payload_handling(self, client: TestClient):
        """Test handling of larger data payloads."""
        # Create larger dataset
        import numpy as np
        np.random.seed(42)
        
        large_data = np.random.randn(500, 5).tolist()
        
        response = client.post("/api/v1/detect", json={
            "data": large_data,
            "algorithm": "isolation_forest",
            "contamination": 0.1
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["total_samples"] == 500
        assert data["processing_time_ms"] > 0
    
    def test_concurrent_requests(self, client: TestClient):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        errors = []
        
        def make_request():
            try:
                response = client.post("/api/v1/detect", json={
                    "data": [[0, 0], [1, 1], [5, 5]],
                    "algorithm": "isolation_forest",
                    "contamination": 0.2
                })
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        assert all(status == 200 for status in results)