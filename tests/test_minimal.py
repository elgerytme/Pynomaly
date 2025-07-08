#!/usr/bin/env python3
"""Minimal test to isolate the Query+Request ForwardRef issue."""

import sys
sys.path.append('src')

from fastapi import FastAPI, Query, Depends
from pynomaly.infrastructure.config import Container


def simple_container() -> Container:
    """Simple container function that doesn't use Request."""
    return Container()


# Test 1: Simple endpoint with Query only
def test_query_only():
    """Test endpoint with just Query parameters."""
    app = FastAPI()
    
    @app.get("/test")
    async def test_endpoint(
        param: bool = Query(default=True, description="Test parameter")
    ):
        return {"param": param}
    
    try:
        schema = app.openapi()
        print("✅ Query-only test passed")
        return True
    except Exception as e:
        print(f"❌ Query-only test failed: {e}")
        return False


# Test 2: Query with container dependency
def test_query_with_container():
    """Test endpoint with Query and container dependency."""
    app = FastAPI()
    
    @app.get("/test")
    async def test_endpoint(
        container: Container = Depends(simple_container),
        param: bool = Query(default=True, description="Test parameter")
    ):
        return {"param": param}
    
    try:
        schema = app.openapi()
        print("✅ Query with simple container test passed")
        return True
    except Exception as e:
        print(f"❌ Query with simple container test failed: {e}")
        return False


# Test 3: Query with problematic get_container
def test_query_with_get_container():
    """Test endpoint with Query and problematic get_container."""
    from pynomaly.presentation.api.deps import get_container
    
    app = FastAPI()
    
    @app.get("/test")
    async def test_endpoint(
        container: Container = Depends(get_container),
        param: bool = Query(default=True, description="Test parameter")
    ):
        return {"param": param}
    
    try:
        schema = app.openapi()
        print("✅ Query with get_container test passed")
        return True
    except Exception as e:
        print(f"❌ Query with get_container test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Running minimal tests to isolate the issue...\n")
    
    test_query_only()
    test_query_with_container()
    test_query_with_get_container()
