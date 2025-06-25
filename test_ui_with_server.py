#!/usr/bin/env python3
"""
Comprehensive UI testing with operational server integration.
This script starts the Pynomaly server and runs UI tests against it.
"""

import asyncio
import multiprocessing
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def start_server_process():
    """Start the Pynomaly server in a separate process."""
    try:
        # Start server using our CLI wrapper
        process = subprocess.Popen(
            ["python3", "pynomaly_cli.py", "server-start"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(project_root)
        )
        return process
    except Exception as e:
        print(f"Failed to start server: {e}")
        return None


def wait_for_server(host="127.0.0.1", port=8000, timeout=30):
    """Wait for server to be ready."""
    import requests
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://{host}:{port}/api/health")
            if response.status_code == 200:
                print("✅ Server is ready")
                return True
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            print(f"Unexpected error checking server: {e}")
        
        time.sleep(1)
    
    print("❌ Server failed to start within timeout")
    return False


def test_api_endpoints():
    """Test core API endpoints."""
    import requests
    
    print("\n🔍 Testing API endpoints...")
    
    base_url = "http://127.0.0.1:8000"
    endpoints = [
        ("/", "API root"),
        ("/api/health", "Health check"),
        ("/api/docs", "API documentation"),
        ("/api/detectors", "Detectors list"),
        ("/api/datasets", "Datasets list"),
    ]
    
    results = []
    
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}")
            success = 200 <= response.status_code < 300
            results.append({
                "endpoint": endpoint,
                "description": description,
                "status": response.status_code,
                "success": success
            })
            status_icon = "✅" if success else "❌"
            print(f"  {status_icon} {description}: {response.status_code}")
        except Exception as e:
            results.append({
                "endpoint": endpoint,
                "description": description,
                "status": None,
                "success": False,
                "error": str(e)
            })
            print(f"  ❌ {description}: Error - {e}")
    
    success_count = sum(1 for r in results if r["success"])
    print(f"\n📊 API Results: {success_count}/{len(results)} endpoints working")
    
    return results


def test_basic_ui_availability():
    """Test basic UI availability without full browser automation."""
    import requests
    
    print("\n🖥️ Testing UI availability...")
    
    base_url = "http://127.0.0.1:8000"
    
    try:
        # Test if we can reach the main page
        response = requests.get(base_url)
        if 200 <= response.status_code < 300:
            print("  ✅ Main page accessible")
            return True
        else:
            print(f"  ❌ Main page returned {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Cannot reach main page: {e}")
        return False


def run_server_integration_tests():
    """Run comprehensive server integration tests."""
    print("🚀 Starting Pynomaly Server Integration Tests")
    print("=" * 50)
    
    # Start server
    print("1. Starting server...")
    server_process = start_server_process()
    
    if not server_process:
        print("❌ Failed to start server process")
        return False
    
    try:
        # Wait for server to be ready
        print("2. Waiting for server to be ready...")
        if not wait_for_server():
            print("❌ Server startup failed")
            return False
        
        # Run API tests
        api_results = test_api_endpoints()
        
        # Run basic UI tests  
        ui_available = test_basic_ui_availability()
        
        # Calculate success rate
        api_success_count = sum(1 for r in api_results if r["success"])
        api_total = len(api_results)
        ui_success_count = 1 if ui_available else 0
        ui_total = 1
        
        total_success = api_success_count + ui_success_count
        total_tests = api_total + ui_total
        success_rate = (total_success / total_tests) * 100
        
        print(f"\n📊 Server Integration Test Results:")
        print(f"  API Endpoints: {api_success_count}/{api_total} working")
        print(f"  UI Availability: {ui_success_count}/{ui_total} working")
        print(f"  Overall Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("🎉 Server integration tests PASSED (≥80% success rate)")
            return True
        else:
            print("⚠️ Server integration tests below target (≥80% success rate)")
            return False
            
    finally:
        # Cleanup: terminate server
        print("\n3. Shutting down server...")
        try:
            server_process.terminate()
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()
        print("✅ Server shutdown complete")


def test_cli_server_integration():
    """Test CLI server integration specifically."""
    print("\n🔧 Testing CLI server integration...")
    
    # Test server startup and shutdown
    try:
        # Quick server test - start and stop
        process = subprocess.Popen(
            ["timeout", "5s", "python3", "pynomaly_cli.py", "server-start"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(project_root)
        )
        
        stdout, stderr = process.communicate()
        
        # Check for successful startup indicators
        output = stdout.decode() + stderr.decode()
        
        startup_indicators = [
            "Starting Pynomaly API server",
            "Uvicorn running on",
            "Application startup complete"
        ]
        
        startup_success = any(indicator in output for indicator in startup_indicators)
        
        if startup_success:
            print("  ✅ CLI server startup working")
            return True
        else:
            print("  ❌ CLI server startup issues detected")
            print(f"  Output: {output[:200]}...")
            return False
            
    except Exception as e:
        print(f"  ❌ CLI server test failed: {e}")
        return False


def main():
    """Main test execution."""
    print("🧪 Pynomaly UI and Server Integration Testing")
    print("=" * 60)
    
    # Test CLI server integration first
    cli_server_ok = test_cli_server_integration()
    
    # Run full server integration tests
    server_integration_ok = run_server_integration_tests()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 FINAL TEST SUMMARY:")
    print(f"  CLI Server Integration: {'✅ PASS' if cli_server_ok else '❌ FAIL'}")
    print(f"  Server Integration Tests: {'✅ PASS' if server_integration_ok else '❌ FAIL'}")
    
    overall_success = cli_server_ok and server_integration_ok
    print(f"  Overall Status: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)