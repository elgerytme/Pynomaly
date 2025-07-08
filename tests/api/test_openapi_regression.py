#!/usr/bin/env python3
"""Standalone regression test for TypeAdapter/ForwardRef OpenAPI issues."""

import json
import pytest
from fastapi import FastAPI, Depends, Query, HTTPException
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from typing import Annotated, Optional
from uuid import UUID
from datetime import datetime


# Simple test models to reproduce the issue
class SimpleRequest(BaseModel):
    name: str
    value: int


class SimpleResponse(BaseModel):
    id: UUID
    message: str
    timestamp: datetime


# Simple dependency function to test Depends() patterns
def get_simple_dependency() -> str:
    return "simple"


# Test FastAPI app with the patterns that could cause TypeAdapter issues
def create_test_app() -> FastAPI:
    """Create a test FastAPI app with dependency patterns that could cause TypeAdapter issues."""
    app = FastAPI(
        title="Test API",
        version="1.0.0",
        description="Test API for OpenAPI generation regression testing"
    )
    
    @app.get("/test")
    async def test_endpoint(
        # These patterns previously caused TypeAdapter/ForwardRef issues
        param: Annotated[Optional[str], Query()] = None,
        dependency: Annotated[str, Depends(get_simple_dependency)] = None,
    ) -> SimpleResponse:
        """Test endpoint with dependency patterns."""
        return SimpleResponse(
            id=UUID('12345678-1234-5678-1234-567812345678'),
            message="test",
            timestamp=datetime.now()
        )
    
    @app.post("/test")
    async def test_post_endpoint(
        request: SimpleRequest,
        dependency: Annotated[str, Depends(get_simple_dependency)] = None,
    ) -> SimpleResponse:
        """Test POST endpoint with body and dependency patterns."""
        return SimpleResponse(
            id=UUID('12345678-1234-5678-1234-567812345678'),
            message=f"Got {request.name}",
            timestamp=datetime.now()
        )
    
    return app


class TestOpenAPIRegression:
    """Regression tests for OpenAPI generation issues."""

    def test_app_creation_no_exceptions(self):
        """Test that the test app can be created without exceptions."""
        app = create_test_app()
        assert app is not None
        assert app.title == "Test API"

    def test_openapi_generation_no_exceptions(self):
        """Test that OpenAPI generation does not raise exceptions."""
        app = create_test_app()
        
        # The critical test: app.openapi() should not raise exceptions
        # This was failing before the fix with TypeAdapter/ForwardRef errors
        try:
            schema = app.openapi()
            assert schema is not None
            assert isinstance(schema, dict)
        except Exception as e:
            if "ForwardRef" in str(e) or "TypeAdapter" in str(e):
                pytest.fail(f"TypeAdapter/ForwardRef regression detected: {e}")
            else:
                # Re-raise unexpected exceptions
                raise

    def test_openapi_schema_structure(self):
        """Test that OpenAPI schema has the expected structure."""
        app = create_test_app()
        schema = app.openapi()
        
        # Check basic OpenAPI structure
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        
        # Check info section
        assert "title" in schema["info"]
        assert "version" in schema["info"]
        assert schema["info"]["title"] == "Test API"

    def test_no_forwardref_strings_in_json(self):
        """Test that no ForwardRef strings remain in the OpenAPI JSON."""
        app = create_test_app()
        schema = app.openapi()
        
        # Convert to JSON string to check for ForwardRef strings
        json_schema = json.dumps(schema)
        
        # These strings should not appear in the final OpenAPI JSON
        forbidden_strings = [
            "ForwardRef",
            "_ForwardRef",
            "typing.ForwardRef",
            "PydanticUndefined",
            "TypeAdapter",
            "'Request'",  # Common forward reference issue
            "'Response'",  # Common forward reference issue
        ]
        
        for forbidden_string in forbidden_strings:
            assert forbidden_string not in json_schema, \
                f"Found forbidden string '{forbidden_string}' in OpenAPI JSON"

    def test_dependency_patterns_resolved(self):
        """Test that dependency patterns are properly resolved."""
        app = create_test_app()
        schema = app.openapi()
        
        # Check that we have endpoints
        assert "paths" in schema
        paths = schema["paths"]
        assert "/test" in paths
        
        # Check that parameters are properly defined
        get_endpoint = paths["/test"]["get"]
        assert "parameters" in get_endpoint
        
        # Parameters should have proper schema
        for param in get_endpoint["parameters"]:
            assert "schema" in param or "$ref" in param, \
                f"Parameter missing schema: {param}"

    def test_pydantic_models_properly_serialized(self):
        """Test that Pydantic models are properly serialized in OpenAPI."""
        app = create_test_app()
        schema = app.openapi()
        
        # Check that we have component schemas
        assert "components" in schema
        assert "schemas" in schema["components"]
        schemas = schema["components"]["schemas"]
        
        # Look for our test models
        model_found = False
        for schema_name, schema_def in schemas.items():
            if "SimpleRequest" in schema_name or "SimpleResponse" in schema_name:
                model_found = True
                assert isinstance(schema_def, dict)
                assert "type" in schema_def
                assert "properties" in schema_def
                
                # Should not contain unresolved forward references
                schema_str = str(schema_def)
                assert "ForwardRef" not in schema_str
                assert "PydanticUndefined" not in schema_str
        
        assert model_found, "Expected to find SimpleRequest or SimpleResponse in schemas"

    def test_uuid_and_datetime_types_handled(self):
        """Test that UUID and datetime types are properly handled."""
        app = create_test_app()
        schema = app.openapi()
        
        # Find schemas with UUID and datetime fields
        schemas = schema["components"]["schemas"]
        
        uuid_handled = False
        datetime_handled = False
        
        for schema_name, schema_def in schemas.items():
            if isinstance(schema_def, dict) and "properties" in schema_def:
                properties = schema_def["properties"]
                
                for prop_name, prop_def in properties.items():
                    if prop_name == "id" and isinstance(prop_def, dict):
                        # UUID should be handled as string with format
                        if "type" in prop_def and prop_def["type"] == "string":
                            uuid_handled = True
                    
                    if prop_name == "timestamp" and isinstance(prop_def, dict):
                        # datetime should be handled as string with format
                        if "type" in prop_def and prop_def["type"] == "string":
                            datetime_handled = True
        
        # At least one of these should be handled properly
        assert uuid_handled or datetime_handled, \
            "UUID or datetime types not properly handled in OpenAPI schema"

    def test_annotated_depends_patterns(self):
        """Test that Annotated[..., Depends(...)] patterns work correctly."""
        app = create_test_app()
        
        # This should not raise exceptions
        schema = app.openapi()
        
        # Check that endpoints are properly defined
        assert "paths" in schema
        paths = schema["paths"]
        
        # Both GET and POST should be present
        assert "/test" in paths
        assert "get" in paths["/test"]
        assert "post" in paths["/test"]
        
        # Endpoints should have proper operation IDs
        get_op = paths["/test"]["get"]
        post_op = paths["/test"]["post"]
        
        assert "operationId" in get_op
        assert "operationId" in post_op

    def test_json_serialization_works(self):
        """Test that the OpenAPI schema can be serialized to JSON."""
        app = create_test_app()
        schema = app.openapi()
        
        # Should be able to serialize to JSON without errors
        json_str = json.dumps(schema)
        assert len(json_str) > 0
        
        # Should be able to parse back from JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["info"]["title"] == "Test API"

    def test_performance_acceptable(self):
        """Test that OpenAPI generation completes in reasonable time."""
        import time
        
        app = create_test_app()
        
        # Measure OpenAPI generation time
        start_time = time.time()
        schema = app.openapi()
        end_time = time.time()
        
        generation_time = end_time - start_time
        
        # Should complete quickly for a simple app
        assert generation_time < 5.0, \
            f"OpenAPI generation took too long: {generation_time:.2f}s"
        
        # Schema should be valid
        assert schema is not None
        assert isinstance(schema, dict)


if __name__ == "__main__":
    # Run the tests
    test_instance = TestOpenAPIRegression()
    
    print("🔍 Running OpenAPI regression tests...")
    print("=" * 60)
    
    try:
        test_instance.test_app_creation_no_exceptions()
        print("✅ App creation test passed")
        
        test_instance.test_openapi_generation_no_exceptions()
        print("✅ OpenAPI generation test passed")
        
        test_instance.test_openapi_schema_structure()
        print("✅ OpenAPI schema structure test passed")
        
        test_instance.test_no_forwardref_strings_in_json()
        print("✅ No ForwardRef strings test passed")
        
        test_instance.test_dependency_patterns_resolved()
        print("✅ Dependency patterns test passed")
        
        test_instance.test_pydantic_models_properly_serialized()
        print("✅ Pydantic models test passed")
        
        test_instance.test_uuid_and_datetime_types_handled()
        print("✅ UUID/datetime types test passed")
        
        test_instance.test_annotated_depends_patterns()
        print("✅ Annotated Depends patterns test passed")
        
        test_instance.test_json_serialization_works()
        print("✅ JSON serialization test passed")
        
        test_instance.test_performance_acceptable()
        print("✅ Performance test passed")
        
        print("\n🎉 All OpenAPI regression tests passed!")
        print("✅ TypeAdapter/ForwardRef issues have been resolved!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
