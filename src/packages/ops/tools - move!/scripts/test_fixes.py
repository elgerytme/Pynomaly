#!/usr/bin/env python3
"""
Test script to verify that the minor non-blocking issues have been fixed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_cli_async_utils():
    """Test that CLI async utilities work correctly."""
    print("Testing CLI async utilities...")
    
    try:
        from monorepo.presentation.cli.async_utils import cli_runner, run_async_safely
        
        # Test basic async operation
        import asyncio
        
        async def test_coroutine():
            await asyncio.sleep(0.001)  # Very short sleep
            return "success"
        
        result = run_async_safely(test_coroutine())
        assert result == "success"
        print("✓ CLI async utilities working correctly")
        
    except Exception as e:
        print(f"✗ CLI async utilities failed: {e}")
        return False
    
    return True

def test_config_attributes():
    """Test that configuration attributes are accessible."""
    print("Testing configuration attributes...")
    
    try:
        from monorepo.infrastructure.config.settings import Settings
        
        settings = Settings()
        
        # Test all the compatibility properties
        test_attrs = [
            'app_name', 'version', 'debug', 'storage_path', 
            'api_host', 'api_port', 'max_dataset_size_mb', 
            'default_contamination_rate', 'gpu_enabled'
        ]
        
        for attr in test_attrs:
            value = getattr(settings, attr)
            print(f"  {attr}: {value}")
        
        print("✓ Configuration attributes accessible")
        
    except Exception as e:
        print(f"✗ Configuration attributes failed: {e}")
        return False
    
    return True

def test_security_monitoring():
    """Test that security monitoring initializes without event loop errors."""
    print("Testing security monitoring...")
    
    try:
        from monorepo.presentation.web.security_monitoring import SecurityMonitoringService
        
        # This should not raise event loop errors
        service = SecurityMonitoringService()
        
        # Check that background tasks are not started yet
        assert not service._tasks_started
        assert len(service._background_tasks) == 0
        
        print("✓ Security monitoring initializes without errors")
        
    except Exception as e:
        print(f"✗ Security monitoring failed: {e}")
        return False
    
    return True

def test_cli_import():
    """Test that CLI can be imported without errors."""
    print("Testing CLI import...")
    
    try:
        from monorepo.presentation.cli.app import app
        
        # Check that the app has commands
        assert hasattr(app, 'registered_commands')
        
        print("✓ CLI imports successfully")
        
    except Exception as e:
        print(f"✗ CLI import failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Running tests for fixed minor issues...\n")
    
    tests = [
        test_cli_async_utils,
        test_config_attributes,
        test_security_monitoring,
        test_cli_import
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All fixes are working correctly!")
        return 0
    else:
        print("❌ Some fixes still need work")
        return 1

if __name__ == "__main__":
    sys.exit(main())