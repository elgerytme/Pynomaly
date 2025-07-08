#!/usr/bin/env python3
"""Test script to validate OpenAPI schema generation fixes."""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def test_basic_openapi():
    """Test basic OpenAPI schema generation."""
    app = FastAPI(title="Test App", version="1.0.0")
    
    @app.get("/test")
    def test_endpoint():
        return {"message": "test"}
    
    # Test basic OpenAPI generation
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description="Test API",
        routes=app.routes,
    )
    
    print(f"✅ Basic OpenAPI schema generated successfully")
    print(f"   Title: {schema['info']['title']}")
    print(f"   Version: {schema['info']['version']}")
    print(f"   Paths: {len(schema['paths'])}")
    
    return schema


def test_pynomaly_openapi_config():
    """Test Pynomaly OpenAPI configuration."""
    try:
        # Create a minimal FastAPI app
        app = FastAPI(title="Pynomaly", version="1.0.0")
        
        # Add a simple endpoint
        @app.get("/api/health")
        def health():
            return {"status": "healthy"}
        
        # Try to create OpenAPI config
        from pynomaly.presentation.api.docs.openapi_config import OpenAPIConfig
        from pynomaly.infrastructure.config.settings import Settings
        
        # Create minimal settings
        settings = Settings()
        config = OpenAPIConfig(settings)
        
        # Test schema generation
        schema = config.get_openapi_schema(app)
        
        print(f"✅ Pynomaly OpenAPI configuration works")
        print(f"   Title: {schema['info']['title']}")
        print(f"   Version: {schema['info']['version']}")
        print(f"   Paths: {len(schema.get('paths', {}))}")
        
        return schema
        
    except Exception as e:
        print(f"❌ Pynomaly OpenAPI configuration failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_openapi_endpoint():
    """Test OpenAPI endpoint generation."""
    try:
        # Create a minimal FastAPI app
        app = FastAPI(title="Pynomaly", version="1.0.0")
        
        @app.get("/api/health")
        def health():
            return {"status": "healthy"}
        
        # Configure OpenAPI with fallback handling
        from pynomaly.presentation.api.docs.openapi_config import configure_openapi_docs
        from pynomaly.infrastructure.config.settings import Settings
        
        settings = Settings()
        configure_openapi_docs(app, settings)
        
        # Test that openapi() method works
        schema = app.openapi()
        
        print(f"✅ OpenAPI endpoint generation works")
        print(f"   Title: {schema['info']['title']}")
        print(f"   OpenAPI version: {schema.get('openapi', 'unknown')}")
        print(f"   Paths: {len(schema.get('paths', {}))}")
        
        return schema
        
    except Exception as e:
        print(f"❌ OpenAPI endpoint generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("🔍 Testing OpenAPI schema generation...\n")
    
    print("1. Basic OpenAPI test:")
    test_basic_openapi()
    print()
    
    print("2. Pynomaly OpenAPI configuration test:")
    test_pynomaly_openapi_config()
    print()
    
    print("3. OpenAPI endpoint test:")
    test_openapi_endpoint()
    print()
    
    print("✅ All tests completed!")
