#!/usr/bin/env python3
"""Unit tests for OpenAPI utils."""

import pytest
from unittest.mock import Mock, patch
from fastapi import FastAPI
from fastapi.routing import APIRoute
from pydantic import BaseModel

try:
    from pynomaly.presentation.api.docs.openapi_utils import (
        get_all_pydantic_models,
        rebuild_model_references,
        generate_model_schemas,
        create_operation_objects,
        custom_openapi_generator,
        apply_custom_openapi_to_app,
        clear_openapi_cache,
    )
except ImportError as e:
    print(f"Could not import openapi_utils: {e}")
    print("Testing standalone functionality...")
    
    # Create minimal versions for testing
    def get_all_pydantic_models():
        return {}
    
    def rebuild_model_references(models):
        pass
    
    def generate_model_schemas(models):
        return {}
    
    def create_operation_objects(routes):
        return []
    
    def custom_openapi_generator(app, config=None):
        return lambda: {}
    
    def apply_custom_openapi_to_app(app, config=None):
        pass
    
    def clear_openapi_cache():
        pass


class TestModel(BaseModel):
    """Test Pydantic model."""
    name: str
    value: int


class TestNestedModel(BaseModel):
    """Test nested Pydantic model."""
    test_model: TestModel
    description: str


class TestOpenAPIUtils:
    """Test suite for OpenAPI utils."""

    def setup_method(self):
        """Setup test fixtures."""
        clear_openapi_cache()

    def test_get_all_pydantic_models(self):
        """Test getting all Pydantic models."""
        models = get_all_pydantic_models()
        
        assert isinstance(models, dict)
        assert len(models) > 0
        
        # Check that our test models are included
        model_names = list(models.keys())
        assert "TestModel" in model_names
        assert "TestNestedModel" in model_names
        
        # Check that models are BaseModel subclasses
        for name, model in models.items():
            assert issubclass(model, BaseModel)

    def test_rebuild_model_references(self):
        """Test rebuilding model references."""
        models = {"TestModel": TestModel}
        
        # Should not raise any exceptions
        rebuild_model_references(models)
        
        # Test with invalid model (should handle gracefully)
        invalid_model = Mock()
        invalid_model.model_rebuild.side_effect = Exception("Test error")
        models_with_invalid = {"TestModel": TestModel, "InvalidModel": invalid_model}
        
        # Should not raise exception, just log warning
        rebuild_model_references(models_with_invalid)

    def test_generate_model_schemas(self):
        """Test generating model schemas."""
        models = {"TestModel": TestModel, "TestNestedModel": TestNestedModel}
        
        schemas = generate_model_schemas(models)
        
        assert isinstance(schemas, dict)
        assert len(schemas) == 2
        assert "TestModel" in schemas
        assert "TestNestedModel" in schemas
        
        # Check schema structure
        test_model_schema = schemas["TestModel"]
        assert "type" in test_model_schema
        assert "properties" in test_model_schema
        assert "title" in test_model_schema
        
        # Check that properties are correct
        properties = test_model_schema["properties"]
        assert "name" in properties
        assert "value" in properties

    def test_generate_model_schemas_with_error(self):
        """Test generating model schemas with error handling."""
        # Create a mock model that raises an exception
        invalid_model = Mock()
        invalid_model.model_json_schema.side_effect = Exception("Test error")
        invalid_model.__name__ = "InvalidModel"
        
        models = {"InvalidModel": invalid_model}
        
        schemas = generate_model_schemas(models)
        
        assert isinstance(schemas, dict)
        assert "InvalidModel" in schemas
        
        # Should create fallback schema
        fallback_schema = schemas["InvalidModel"]
        assert fallback_schema["type"] == "object"
        assert fallback_schema["title"] == "InvalidModel"
        assert "fallback" in fallback_schema["description"]

    def test_create_operation_objects(self):
        """Test creating operation objects."""
        # Create mock routes
        route1 = Mock(spec=APIRoute)
        route1.path = "/test"
        route1.methods = ["GET"]
        route1.name = "test_endpoint"
        route1.tags = ["test"]
        route1.summary = "Test endpoint"
        route1.description = "Test description"
        route1.response_model = TestModel
        
        route2 = Mock(spec=APIRoute)
        route2.path = "/test2"
        route2.methods = ["POST"]
        route2.name = "test_endpoint2"
        route2.tags = []
        route2.summary = None
        route2.description = None
        route2.response_model = None
        
        routes = [route1, route2]
        
        operations = create_operation_objects(routes)
        
        assert isinstance(operations, list)
        assert len(operations) == 2
        
        # Check first operation
        op1 = operations[0]
        assert op1["path"] == "/test"
        assert op1["methods"] == ["GET"]
        assert op1["name"] == "test_endpoint"
        assert op1["tags"] == ["test"]
        assert op1["summary"] == "Test endpoint"
        assert op1["description"] == "Test description"
        assert op1["response_model"] == "TestModel"
        
        # Check second operation
        op2 = operations[1]
        assert op2["path"] == "/test2"
        assert op2["methods"] == ["POST"]
        assert op2["name"] == "test_endpoint2"
        assert op2["tags"] == []

    def test_custom_openapi_generator(self):
        """Test custom OpenAPI generator."""
        app = FastAPI(title="Test App", version="1.0.0")
        
        # Add a test route
        @app.get("/test")
        def test_endpoint():
            return {"message": "test"}
        
        generator = custom_openapi_generator(app)
        
        # Test that generator is callable
        assert callable(generator)
        
        # Test generating schema
        schema = generator()
        
        assert isinstance(schema, dict)
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        assert "components" in schema
        
        # Check info section
        assert schema["info"]["title"] == "Test App"
        assert schema["info"]["version"] == "1.0.0"
        
        # Check that schemas are included
        assert "schemas" in schema["components"]
        
        # Check custom metadata
        assert "x-operations" in schema
        assert "x-model-count" in schema
        assert "x-generated-at" in schema

    def test_custom_openapi_generator_caching(self):
        """Test that OpenAPI generator uses caching."""
        app = FastAPI(title="Test App", version="1.0.0")
        
        generator = custom_openapi_generator(app)
        
        # Generate schema twice
        schema1 = generator()
        schema2 = generator()
        
        # Should be the same object (cached)
        assert schema1 is schema2

    def test_apply_custom_openapi_to_app(self):
        """Test applying custom OpenAPI generator to app."""
        app = FastAPI(title="Test App", version="1.0.0")
        original_openapi = app.openapi
        
        apply_custom_openapi_to_app(app)
        
        # Check that openapi method was replaced
        assert app.openapi is not original_openapi
        assert callable(app.openapi)
        
        # Test that it generates valid schema
        schema = app.openapi()
        assert isinstance(schema, dict)
        assert "openapi" in schema

    def test_clear_openapi_cache(self):
        """Test clearing OpenAPI cache."""
        app = FastAPI(title="Test App", version="1.0.0")
        
        generator = custom_openapi_generator(app)
        
        # Generate schema to populate cache
        schema1 = generator()
        
        # Clear cache
        clear_openapi_cache()
        
        # Generate schema again
        schema2 = generator()
        
        # Should be different objects since cache was cleared
        assert schema1 is not schema2
        # Basic structure should be similar
        assert schema1["info"]["title"] == schema2["info"]["title"]
        assert "paths" in schema1 and "paths" in schema2


def test_standalone_schema_generation():
    """Test standalone schema generation without FastAPI app."""
    # Test that we can generate schemas for models independently
    models = get_all_pydantic_models()
    
    # Rebuild references
    rebuild_model_references(models)
    
    # Generate schemas
    schemas = generate_model_schemas(models)
    
    assert isinstance(schemas, dict)
    assert len(schemas) > 0
    
    # Check that test models are present
    assert "TestModel" in schemas
    assert "TestNestedModel" in schemas
    
    # Verify schema structure
    test_schema = schemas["TestModel"]
    assert test_schema["type"] == "object"
    assert "properties" in test_schema
    assert "name" in test_schema["properties"]
    assert "value" in test_schema["properties"]


if __name__ == "__main__":
    # Run the tests
    test_instance = TestOpenAPIUtils()
    
    try:
        test_instance.setup_method()
        test_instance.test_get_all_pydantic_models()
        print("✅ get_all_pydantic_models test passed")
        
        test_instance.test_rebuild_model_references()
        print("✅ rebuild_model_references test passed")
        
        test_instance.test_generate_model_schemas()
        print("✅ generate_model_schemas test passed")
        
        test_instance.test_generate_model_schemas_with_error()
        print("✅ generate_model_schemas_with_error test passed")
        
        test_instance.test_create_operation_objects()
        print("✅ create_operation_objects test passed")
        
        test_instance.test_custom_openapi_generator()
        print("✅ custom_openapi_generator test passed")
        
        test_instance.test_custom_openapi_generator_caching()
        print("✅ custom_openapi_generator_caching test passed")
        
        test_instance.test_apply_custom_openapi_to_app()
        print("✅ apply_custom_openapi_to_app test passed")
        
        test_instance.test_clear_openapi_cache()
        print("✅ clear_openapi_cache test passed")
        
        test_standalone_schema_generation()
        print("✅ standalone_schema_generation test passed")
        
        print("\n🎉 All OpenAPI utils tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
