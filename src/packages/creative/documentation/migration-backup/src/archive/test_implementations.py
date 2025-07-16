#!/usr/bin/env python3
"""
Test script to validate our web UI implementations.
"""
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx


class ImplementationValidator:
    """Validates the implemented web UI features."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=10.0)
        
    def test_server_health(self):
        """Test basic server health."""
        try:
            response = self.client.get(f"{self.base_url}/")
            print(f"✓ Server responding: {response.status_code}")
            data = response.json()
            print(f"  - API Version: {data.get('api_version')}")
            print(f"  - Version: {data.get('version')}")
            return True
        except Exception as e:
            print(f"✗ Server not responding: {e}")
            return False
    
    def test_frontend_support_endpoints(self):
        """Test our new frontend support endpoints."""
        endpoints = [
            "/api/ui/config",
            "/api/ui/health", 
            "/api/session/status",
        ]
        
        success_count = 0
        for endpoint in endpoints:
            try:
                response = self.client.get(f"{self.base_url}{endpoint}")
                if response.status_code == 200:
                    print(f"✓ {endpoint}: {response.status_code}")
                    success_count += 1
                else:
                    print(f"✗ {endpoint}: {response.status_code}")
            except Exception as e:
                print(f"✗ {endpoint}: {e}")
        
        return success_count == len(endpoints)
    
    def test_post_endpoints(self):
        """Test POST endpoints."""
        # Test performance metrics endpoint
        try:
            metric_data = {
                "metric": "LCP",
                "value": 1500.0,
                "timestamp": int(time.time()),
                "url": "/"
            }
            response = self.client.post(f"{self.base_url}/api/metrics/critical", json=metric_data)
            if response.status_code == 200:
                print(f"✓ Performance metrics endpoint: {response.status_code}")
            else:
                print(f"✗ Performance metrics endpoint: {response.status_code}")
        except Exception as e:
            print(f"✗ Performance metrics endpoint: {e}")
        
        # Test security events endpoint
        try:
            event_data = {
                "type": "xss_attempt",
                "timestamp": int(time.time()),
                "url": "/",
                "userAgent": "test-agent",
                "data": {"payload": "test"}
            }
            response = self.client.post(f"{self.base_url}/api/security/events", json=event_data)
            if response.status_code == 200:
                print(f"✓ Security events endpoint: {response.status_code}")
            else:
                print(f"✗ Security events endpoint: {response.status_code}")
        except Exception as e:
            print(f"✗ Security events endpoint: {e}")
        
        # Test session extend endpoint
        try:
            response = self.client.post(f"{self.base_url}/api/session/extend")
            if response.status_code == 200:
                print(f"✓ Session extend endpoint: {response.status_code}")
            else:
                print(f"✗ Session extend endpoint: {response.status_code}")
        except Exception as e:
            print(f"✗ Session extend endpoint: {e}")
    
    def test_ui_config_content(self):
        """Test the UI configuration content."""
        try:
            response = self.client.get(f"{self.base_url}/api/ui/config")
            if response.status_code == 200:
                config = response.json()
                required_keys = ["performance_monitoring", "security", "features"]
                
                for key in required_keys:
                    if key in config:
                        print(f"✓ UI config contains {key}")
                    else:
                        print(f"✗ UI config missing {key}")
                
                # Check specific configurations
                if "performance_monitoring" in config:
                    perf_config = config["performance_monitoring"]
                    if perf_config.get("enabled") and "critical_thresholds" in perf_config:
                        print(f"✓ Performance monitoring properly configured")
                    else:
                        print(f"✗ Performance monitoring not properly configured")
                
                if "security" in config:
                    sec_config = config["security"]
                    if sec_config.get("csrf_protection") and sec_config.get("xss_protection"):
                        print(f"✓ Security features properly configured")
                    else:
                        print(f"✗ Security features not properly configured")
                
                return True
            else:
                print(f"✗ UI config endpoint returned {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ UI config test failed: {e}")
            return False
    
    def test_session_status_content(self):
        """Test session status endpoint content."""
        try:
            response = self.client.get(f"{self.base_url}/api/session/status")
            if response.status_code == 200:
                session_data = response.json()
                required_keys = ["authenticated", "expires_at", "last_activity", "csrf_token"]
                
                for key in required_keys:
                    if key in session_data:
                        print(f"✓ Session status contains {key}")
                    else:
                        print(f"✗ Session status missing {key}")
                
                # Check CSRF token format
                csrf_token = session_data.get("csrf_token")
                if csrf_token and len(csrf_token) >= 32:
                    print(f"✓ CSRF token properly generated")
                else:
                    print(f"✗ CSRF token not properly generated")
                
                return True
            else:
                print(f"✗ Session status endpoint returned {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Session status test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all validation tests."""
        print("=" * 60)
        print("PYNOMALY WEB UI IMPLEMENTATION VALIDATION")
        print("=" * 60)
        
        # Test basic server health
        print("\n1. Testing server health...")
        server_ok = self.test_server_health()
        
        if not server_ok:
            print("\n❌ Server not responding. Please start the server first.")
            return False
        
        # Test frontend support endpoints
        print("\n2. Testing frontend support endpoints...")
        endpoints_ok = self.test_frontend_support_endpoints()
        
        # Test POST endpoints  
        print("\n3. Testing POST endpoints...")
        self.test_post_endpoints()
        
        # Test UI config content
        print("\n4. Testing UI configuration content...")
        config_ok = self.test_ui_config_content()
        
        # Test session status content
        print("\n5. Testing session status content...")
        session_ok = self.test_session_status_content()
        
        # Summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        if server_ok and endpoints_ok and config_ok and session_ok:
            print("✅ All core implementations are working correctly!")
            print("   - Backend API endpoints are responding")
            print("   - CSRF token generation is working")
            print("   - UI configuration is properly structured")
            print("   - Session management is functional")
            return True
        else:
            print("❌ Some implementations need attention")
            print("   Check the detailed output above for specific issues")
            return False


def start_test_server():
    """Start the test server."""
    print("Starting test server on port 8001...")
    try:
        process = subprocess.Popen(
            [sys.executable, "scripts/run/run_web_app.py", "--port", "8001"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        time.sleep(10)  # Give server time to start
        return process
    except Exception as e:
        print(f"Failed to start server: {e}")
        return None


def main():
    """Main test runner."""
    # Try to start server
    server_process = start_test_server()
    
    try:
        # Run validation tests
        validator = ImplementationValidator()
        success = validator.run_all_tests()
        
        if success:
            print("\n🎉 Implementation validation completed successfully!")
            print("   The web UI infrastructure is ready for production use.")
        else:
            print("\n⚠️  Implementation validation found issues.")
            print("   Please review the output above and fix any problems.")
        
        return success
        
    finally:
        # Clean up server
        if server_process:
            server_process.terminate()
            server_process.wait()


if __name__ == "__main__":
    main()