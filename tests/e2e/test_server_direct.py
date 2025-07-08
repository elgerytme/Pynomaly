#!/usr/bin/env python3
import sys
import socket
from contextlib import closing

sys.path.insert(0, "src")


def find_free_port():
    """Find a free port to use for testing."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def test_server_direct_import():
    """Test that the server can be imported and app created successfully."""
    try:
        import uvicorn
        from pynomaly.presentation.api import create_app
        
        print("✅ Imports successful")
        
        # Create app
        app = create_app()
        print("✅ App created successfully")
        
        # Test that app is callable (basic ASGI app test)
        assert callable(app), "App should be callable"
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_server_startup_config():
    """Test server startup configuration without actually starting the server."""
    try:
        import uvicorn
        from pynomaly.presentation.api import create_app
        
        app = create_app()
        
        # Find a free port for testing
        port = find_free_port()
        
        # Test that we can create a server config without starting
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info")
        server = uvicorn.Server(config)
        
        assert server is not None, "Server should be creatable"
        assert config.host == "127.0.0.1", "Host should be set correctly"
        assert config.port == port, "Port should be set correctly"
        
        print(f"✅ Server configuration test passed (would use port {port})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise
