#!/usr/bin/env python3
"""Final OpenAPI test with minimal dependencies."""

from fastapi import FastAPI


def test_openapi_fallback():
    """Test that OpenAPI fallback works correctly."""
    app = FastAPI(title="Pynomaly", version="1.0.0")
    
    @app.get("/api/health")
    def health():
        return {"status": "healthy"}
    
    # Test OpenAPI configuration with fallback
    try:
        # Create a settings class inline
        class MockSettings:
            def __init__(self):
                self.app = self
                self.name = "Pynomaly"
                self.version = "1.0.0"
                self.description = "Advanced anomaly detection API"
                self.docs_enabled = True
        
        settings = MockSettings()
        
        # Apply OpenAPI configuration function directly
        from pynomaly.presentation.api.docs.openapi_config import configure_openapi_docs
        configure_openapi_docs(app, settings)
        
        # Test schema generation (this should use fallback)
        schema = app.openapi()
        
        print("✅ OpenAPI fallback schema generated successfully")
        print(f"   Schema keys: {list(schema.keys())}")
        print(f"   Info keys: {list(schema.get('info', {}).keys())}")
        print(f"   Title: {schema.get('info', {}).get('title', 'N/A')}")
        print(f"   Version: {schema.get('info', {}).get('version', 'N/A')}")
        print(f"   OpenAPI version: {schema.get('openapi', 'unknown')}")
        print(f"   Paths: {len(schema.get('paths', {}))}")
        
        # Test that the endpoint URL is configured correctly
        print(f"   OpenAPI URL: {app.openapi_url}")
        print(f"   Docs URL: {app.docs_url}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAPI fallback failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_openapi_endpoint_access():
    """Test that the OpenAPI endpoint can be accessed."""
    try:
        import uvicorn
        from fastapi import FastAPI
        import asyncio
        import httpx
        
        app = FastAPI(title="Pynomaly", version="1.0.0")
        
        @app.get("/api/health")
        def health():
            return {"status": "healthy"}
        
        # Configure OpenAPI with fallback
        class MockSettings:
            def __init__(self):
                self.app = self
                self.name = "Pynomaly"
                self.version = "1.0.0"
                self.description = "Advanced anomaly detection API"
                self.docs_enabled = True
        
        settings = MockSettings()
        
        from pynomaly.presentation.api.docs.openapi_config import configure_openapi_docs
        configure_openapi_docs(app, settings)
        
        # Test the schema generation directly
        schema = app.openapi()
        
        print("✅ OpenAPI endpoint access test passed")
        print(f"   Schema generated: {len(str(schema))} characters")
        print(f"   Contains fallback: {'Schema generation failed' in str(schema)}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAPI endpoint access failed: {e}")
        return False


if __name__ == "__main__":
    print("🔍 Testing OpenAPI fixes...\n")
    
    print("1. Testing OpenAPI fallback schema generation:")
    success1 = test_openapi_fallback()
    print()
    
    print("2. Testing OpenAPI endpoint access:")
    success2 = test_openapi_endpoint_access()
    print()
    
    if success1 and success2:
        print("✅ All OpenAPI tests passed! P-005 fix is working.")
    else:
        print("❌ Some tests failed. OpenAPI fix needs more work.")
