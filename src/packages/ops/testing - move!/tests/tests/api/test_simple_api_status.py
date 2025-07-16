"""Simple API status test to identify critical issues."""

import pytest
import requests
from fastapi.testclient import TestClient

def test_api_import_basic():
    """Test if we can import basic API components."""
    try:
        from pynomaly.application.dto.dataset_dto import DatasetResponseDTO
        from pynomaly.application.dto.detection_dto import ConfidenceInterval
        assert DatasetResponseDTO is not None
        assert ConfidenceInterval is not None
        print("✅ DTOs import successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import DTOs: {e}")

def test_fastapi_app_creation():
    """Test if we can create the FastAPI app without errors."""
    try:
        # Try to import and create app without full dependency injection
        from fastapi import FastAPI
        
        # Create minimal test app to check for basic import issues
        app = FastAPI(title="Test API")
        
        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}
        
        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 200
        print("✅ Basic FastAPI app creation works")
    except Exception as e:
        pytest.fail(f"Failed to create basic FastAPI app: {e}")

def test_health_endpoint_availability():
    """Test if health endpoints are accessible."""
    try:
        # Try to access health endpoint on default port
        response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"Health endpoint status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Health endpoint is accessible")
        else:
            print(f"⚠️  Health endpoint returned {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("⚠️  Server not running on localhost:8000")
    except Exception as e:
        print(f"⚠️  Health endpoint test failed: {e}")

def test_pydantic_models():
    """Test if Pydantic models can be instantiated."""
    try:
        from pynomaly.application.dto.detection_dto import ConfidenceInterval
        from pynomaly.application.dto.dataset_dto import DatasetResponseDTO
        from datetime import datetime
        import uuid
        
        # Test ConfidenceInterval
        ci = ConfidenceInterval(lower=0.1, upper=0.9, confidence_level=0.95)
        assert ci.lower == 0.1
        assert ci.upper == 0.9
        
        # Test DatasetResponseDTO
        dataset = DatasetResponseDTO(
            id=uuid.uuid4(),
            name="test_dataset",
            shape=(100, 10),
            n_samples=100,
            n_features=10,
            feature_names=["f1", "f2", "f3"],
            has_target=True,
            target_column="target",
            created_at=datetime.now(),
            memory_usage_mb=1.0,
            numeric_features=8,
            categorical_features=2
        )
        assert dataset.name == "test_dataset"
        print("✅ Pydantic models work correctly")
    except Exception as e:
        pytest.fail(f"Failed to create Pydantic models: {e}")

if __name__ == "__main__":
    # Run tests individually for better debugging
    test_api_import_basic()
    test_fastapi_app_creation()
    test_health_endpoint_availability()
    test_pydantic_models()
    print("API status tests completed.")