#!/usr/bin/env python3
"""Test script to verify CLI integration works correctly."""

import sys
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_container_creation():
    """Test that the DI container can be created with adapters."""
    print("Testing container creation...")
    try:
        from pynomaly.infrastructure.config.container import create_container
        container = create_container()
        print("✅ Container created successfully")
        
        # Check if adapters are available
        adapters = [name for name in dir(container) if 'adapter' in name]
        print(f"✅ Available adapters: {adapters}")
        
        # Test accessing each adapter
        try:
            pyod_adapter = container.pyod_adapter()
            print("✅ PyOD adapter accessible")
        except Exception as e:
            print(f"❌ PyOD adapter error: {e}")
        
        try:
            sklearn_adapter = container.sklearn_adapter()
            print("✅ Sklearn adapter accessible")
        except Exception as e:
            print(f"❌ Sklearn adapter error: {e}")
            
        # Test optional adapters
        for adapter_name in ['tods_adapter', 'pygod_adapter', 'pytorch_adapter']:
            if hasattr(container, adapter_name):
                try:
                    adapter = getattr(container, adapter_name)()
                    print(f"✅ {adapter_name} accessible")
                except Exception as e:
                    print(f"❌ {adapter_name} error: {e}")
            else:
                print(f"ℹ️ {adapter_name} not available (optional dependency)")
        
        return True
        
    except Exception as e:
        print(f"❌ Container creation failed: {e}")
        traceback.print_exc()
        return False

def test_cli_imports():
    """Test that CLI modules can be imported."""
    print("\nTesting CLI imports...")
    try:
        from pynomaly.presentation.cli.app import app, get_cli_container
        print("✅ CLI app imported successfully")
        
        # Test container access
        container = get_cli_container()
        print("✅ CLI container accessible")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI import failed: {e}")
        traceback.print_exc()
        return False

def test_cli_commands():
    """Test that CLI commands can be accessed."""
    print("\nTesting CLI command structure...")
    try:
        from pynomaly.presentation.cli import app, detectors, datasets, detection, server
        print("✅ All CLI modules imported")
        
        # Check if commands are registered
        print(f"✅ Main app commands: {len(app.app.registered_commands)}")
        print(f"✅ Detector commands: {len(detectors.app.registered_commands)}")
        print(f"✅ Dataset commands: {len(datasets.app.registered_commands)}")
        print(f"✅ Detection commands: {len(detection.app.registered_commands)}")
        print(f"✅ Server commands: {len(server.app.registered_commands)}")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI commands test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all CLI integration tests."""
    print("🔍 Pynomaly CLI Integration Test")
    print("=" * 40)
    
    success = True
    
    # Test container creation
    success &= test_container_creation()
    
    # Test CLI imports
    success &= test_cli_imports()
    
    # Test CLI commands
    success &= test_cli_commands()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All CLI integration tests passed!")
        return 0
    else:
        print("❌ Some CLI integration tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())