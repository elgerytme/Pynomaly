#!/usr/bin/env python3
"""Unit tests for OpenAPI generation."""

import pytest
from fastapi.openapi.utils import get_openapi

from pynomaly.presentation.api.app import create_app
from pynomaly.infrastructure.config import Container


class TestOpenAPIGeneration:
    """Test suite for OpenAPI schema generation."""

    def test_app_creation(self):
        """Test that the FastAPI app can be created successfully."""
        container = Container()
        app = create_app(container)
        
        assert app is not None
        assert app.title == "Pynomaly"
        assert app.docs_url == "/api/v1/docs"
        assert app.redoc_url == "/api/v1/redoc"
        assert app.openapi_url == "/api/v1/openapi.json" or app.openapi_url == "/api/openapi.json"

    def test_openapi_generation_no_exceptions(self):
        """Test that OpenAPI generation does not raise exceptions."""
        container = Container()
        app = create_app(container)
        
        # This should not raise any exceptions
        schema = app.openapi()
        
        assert schema is not None
        assert isinstance(schema, dict)

    def test_openapi_schema_valid_keys(self):
        """Test that OpenAPI schema contains expected keys."""
        container = Container()
        app = create_app(container)
        
        schema = app.openapi()
        
        # Check for required OpenAPI 3.0 keys
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        
        # Check info section
        assert "title" in schema["info"]
        assert "version" in schema["info"]
        assert schema["info"]["title"] == "Pynomaly"

    def test_openapi_paths_not_empty(self):
        """Test that OpenAPI schema contains API paths."""
        container = Container()
        app = create_app(container)
        
        schema = app.openapi()
        
        assert "paths" in schema
        paths = schema["paths"]
        assert len(paths) > 0
        
        # Check for some expected endpoints
        expected_paths = [
            "/api/v1/health",
            "/api/v1/version",
            "/api/v1/auth/login",
            "/api/v1/detectors",
            "/api/v1/datasets"
        ]
        
        for expected_path in expected_paths:
            assert expected_path in paths, f"Expected path {expected_path} not found in schema"

    def test_openapi_components_schemas(self):
        """Test that OpenAPI schema contains component schemas."""
        container = Container()
        app = create_app(container)
        
        schema = app.openapi()
        
        if "components" in schema and "schemas" in schema["components"]:
            schemas = schema["components"]["schemas"]
            assert len(schemas) > 0
            
            # Check for some expected DTOs
            expected_schemas = [
                "DetectorDTO",
                "DatasetDTO", 
                "DetectionRequestDTO",
                "TrainingRequestDTO"
            ]
            
            for expected_schema in expected_schemas:
                if expected_schema in schemas:
                    # Schema should have required properties
                    schema_def = schemas[expected_schema]
                    assert "type" in schema_def
                    assert "properties" in schema_def

    def test_direct_get_openapi_call(self):
        """Test calling get_openapi directly to isolate issues."""
        container = Container()
        app = create_app(container)
        
        # Call get_openapi directly with app parameters
        schema = get_openapi(
            title=app.title,
            version=app.version,
            openapi_version=app.openapi_version,
            description=app.description,
            routes=app.routes
        )
        
        assert schema is not None
        assert isinstance(schema, dict)
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema

    def test_no_circular_references_in_schemas(self):
        """Test that there are no circular references in schema definitions."""
        container = Container()
        app = create_app(container)
        
        schema = app.openapi()
        
        if "components" in schema and "schemas" in schema["components"]:
            schemas = schema["components"]["schemas"]
            
            # Check that no schema contains ForwardRef errors
            for schema_name, schema_def in schemas.items():
                assert "ForwardRef" not in str(schema_def), f"ForwardRef found in {schema_name}"
                assert "PydanticUndefined" not in str(schema_def), f"PydanticUndefined found in {schema_name}"


if __name__ == "__main__":
    # Run the tests
    test_instance = TestOpenAPIGeneration()
    
    try:
        test_instance.test_app_creation()
        print("✅ App creation test passed")
        
        test_instance.test_openapi_generation_no_exceptions()
        print("✅ OpenAPI generation test passed")
        
        test_instance.test_openapi_schema_valid_keys()
        print("✅ OpenAPI schema validation test passed")
        
        test_instance.test_openapi_paths_not_empty()
        print("✅ OpenAPI paths test passed")
        
        test_instance.test_openapi_components_schemas()
        print("✅ OpenAPI components test passed")
        
        test_instance.test_direct_get_openapi_call()
        print("✅ Direct get_openapi test passed")
        
        test_instance.test_no_circular_references_in_schemas()
        print("✅ Circular references test passed")
        
        print("\n🎉 All OpenAPI tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
