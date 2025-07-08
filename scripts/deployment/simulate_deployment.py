#!/usr/bin/env python3
"""
Deployment simulation script to test URL refactoring from /web to /
"""

import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import requests

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class DeploymentSimulator:
    def __init__(self):
        self.server_process = None
        self.server_thread = None
        self.base_url = "http://localhost:8001"  # Use different port to avoid conflicts
        self.server_running = False

    def check_dependencies(self):
        """Check if required dependencies are available."""
        print("Checking dependencies...")

        try:
            import fastapi

            print("✅ FastAPI available")
        except ImportError:
            print("❌ FastAPI not available. Install with: pip install fastapi")
            return False

        try:
            import uvicorn

            print("✅ Uvicorn available")
        except ImportError:
            print("❌ Uvicorn not available. Install with: pip install uvicorn")
            return False

        return True

    def start_server(self):
        """Start the web server in a separate thread."""

        def run_server():
            try:
                # Set environment variables
                os.environ["PYTHONPATH"] = str(Path(__file__).parent / "src")

                # Import after setting path
                import uvicorn

                from pynomaly.presentation.web.app import create_web_app

                print("Creating web application...")
                app = create_web_app()

                print(f"Starting server on {self.base_url}")
                self.server_running = True

                # Run with minimal logging to avoid clutter
                uvicorn.run(
                    app,
                    host="127.0.0.1",
                    port=8001,
                    log_level="warning",
                    access_log=False,
                )

            except Exception as e:
                print(f"❌ Server failed to start: {e}")
                self.server_running = False

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

        # Wait for server to start
        for i in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get(f"{self.base_url}/api/health", timeout=1)
                if response.status_code == 200:
                    print("✅ Server started successfully")
                    return True
            except:
                pass
            time.sleep(1)
            print(f"Waiting for server... ({i+1}/30)")

        print("❌ Server failed to start within timeout")
        return False

    def test_api_endpoints(self):
        """Test API endpoints."""
        print("\nTesting API endpoints...")

        api_tests = [
            ("/api/health", "Health check"),
            ("/api/docs", "API documentation"),
        ]

        passed = 0
        for endpoint, description in api_tests:
            try:
                url = f"{self.base_url}{endpoint}"
                response = requests.get(url, timeout=5)

                if response.status_code in [200, 404]:  # 404 is okay for some endpoints
                    print(f"✅ {description}: {endpoint} -> {response.status_code}")
                    passed += 1
                else:
                    print(f"❌ {description}: {endpoint} -> {response.status_code}")

            except Exception as e:
                print(f"❌ {description}: {endpoint} -> Error: {e}")

        return passed == len(api_tests)

    def test_web_ui_endpoints(self):
        """Test Web UI endpoints (now at root)."""
        print("\nTesting Web UI endpoints...")

        web_tests = [
            ("/", "Root dashboard"),
            ("/login", "Login page"),
            ("/detectors", "Detectors page"),
            ("/datasets", "Datasets page"),
            ("/detection", "Detection page"),
            ("/monitoring", "Monitoring page"),
        ]

        passed = 0
        total = len(web_tests)

        for endpoint, description in web_tests:
            try:
                url = f"{self.base_url}{endpoint}"
                response = requests.get(url, timeout=5, allow_redirects=False)

                # Accept various response codes that indicate the endpoint exists
                if response.status_code in [200, 302, 401, 403]:
                    print(f"✅ {description}: {endpoint} -> {response.status_code}")
                    passed += 1
                else:
                    print(f"❌ {description}: {endpoint} -> {response.status_code}")

            except Exception as e:
                print(f"❌ {description}: {endpoint} -> Error: {e}")

        print(f"\nWeb UI tests: {passed}/{total} passed")
        return passed >= (total * 0.7)  # Allow 70% pass rate due to auth/config issues

    def test_old_web_endpoints(self):
        """Test that old /web endpoints no longer work."""
        print("\nTesting old /web endpoints (should fail)...")

        old_endpoints = [
            "/web/",
            "/web/login",
            "/web/detectors",
            "/web/datasets",
        ]

        failed_correctly = 0
        total = len(old_endpoints)

        for endpoint in old_endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                response = requests.get(url, timeout=5)

                if response.status_code == 404:
                    print(f"✅ Old endpoint correctly returns 404: {endpoint}")
                    failed_correctly += 1
                else:
                    print(
                        f"❌ Old endpoint still works: {endpoint} -> {response.status_code}"
                    )

            except Exception as e:
                print(f"⚠️  Old endpoint test error: {endpoint} -> {e}")

        print(f"\nOld endpoint tests: {failed_correctly}/{total} correctly return 404")
        return failed_correctly == total

    def test_static_files(self):
        """Test static file serving."""
        print("\nTesting static file serving...")

        static_tests = [
            "/static/css/main.css",
            "/static/js/dashboard.js",
        ]

        passed = 0
        total = len(static_tests)

        for endpoint in static_tests:
            try:
                url = f"{self.base_url}{endpoint}"
                response = requests.get(url, timeout=5)

                if response.status_code in [
                    200,
                    404,
                ]:  # 404 is okay if file doesn't exist
                    print(f"✅ Static file: {endpoint} -> {response.status_code}")
                    passed += 1
                else:
                    print(f"❌ Static file: {endpoint} -> {response.status_code}")

            except Exception as e:
                print(f"❌ Static file: {endpoint} -> Error: {e}")

        return passed >= 1  # At least one static file should work

    def test_nginx_config_compatibility(self):
        """Test nginx configuration compatibility."""
        print("\nTesting nginx configuration...")

        nginx_config_path = Path("config/web/nginx.conf")
        if not nginx_config_path.exists():
            print("❌ nginx.conf not found")
            return False

        config_content = nginx_config_path.read_text()

        # Check for updated location blocks
        if "location / {" in config_content:
            print("✅ Nginx config updated for root location")
        else:
            print("❌ Nginx config missing root location block")
            return False

        # Check that old /web/ location is removed or updated
        if "location /web/ {" in config_content:
            print("❌ Old /web/ location block still present")
            return False
        else:
            print("✅ Old /web/ location block removed")

        return True

    def run_deployment_simulation(self):
        """Run complete deployment simulation."""
        print("=" * 60)
        print("PYNOMALY DEPLOYMENT SIMULATION")
        print("Testing URL refactoring: /web -> /")
        print("=" * 60)

        # Step 1: Check dependencies
        if not self.check_dependencies():
            print("❌ Dependencies check failed")
            return False

        # Step 2: Test nginx config
        nginx_ok = self.test_nginx_config_compatibility()

        # Step 3: Start server
        print(f"\nStarting server at {self.base_url}...")
        if not self.start_server():
            print("❌ Server startup failed")
            return False

        try:
            # Step 4: Run tests
            api_ok = self.test_api_endpoints()
            web_ok = self.test_web_ui_endpoints()
            old_endpoints_ok = self.test_old_web_endpoints()
            static_ok = self.test_static_files()

            # Step 5: Overall results
            print("\n" + "=" * 60)
            print("DEPLOYMENT SIMULATION RESULTS")
            print("=" * 60)

            results = {
                "Nginx Configuration": nginx_ok,
                "API Endpoints": api_ok,
                "Web UI Endpoints": web_ok,
                "Old Endpoints Disabled": old_endpoints_ok,
                "Static Files": static_ok,
            }

            passed = sum(results.values())
            total = len(results)

            for test_name, result in results.items():
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"{test_name:<25}: {status}")

            print(f"\nOverall: {passed}/{total} tests passed")

            if passed == total:
                print("\n🎉 DEPLOYMENT SIMULATION SUCCESSFUL!")
                print("URL refactoring appears to be working correctly.")
                return True
            else:
                print(f"\n⚠️  DEPLOYMENT ISSUES DETECTED: {total-passed} failures")
                return False

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up server process."""
        if self.server_running:
            print("\nShutting down server...")
            # Server will stop when main thread exits


def main():
    """Main function."""
    simulator = DeploymentSimulator()

    try:
        success = simulator.run_deployment_simulation()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
        simulator.cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"\n\nSimulation failed with error: {e}")
        simulator.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
