#!/usr/bin/env python3
"""
Simple smoke test for API endpoints after auth migration
"""

import sys
import time
from multiprocessing import Process

import requests
import uvicorn


def run_server():
    """Run the FastAPI server in a separate process"""
    try:
        # Add src to path for imports
        sys.path.insert(0, 'src')
        from monorepo.presentation.api.app import create_app

        app = create_app()
        uvicorn.run(app, host="127.0.0.1", port=8123, log_level="warning")
    except Exception as e:
        print(f"❌ Server startup failed: {e}")

def test_endpoints():
    """Test key endpoints to ensure they're working"""
    base_url = "http://127.0.0.1:8123"

    # Wait for server to start
    time.sleep(3)

    test_results = []

    # Test 1: API root endpoint
    try:
        response = requests.get(f"{base_url}/api", timeout=5)
        if response.status_code == 200:
            test_results.append(("✅ API root endpoint", "PASS"))
        else:
            test_results.append((f"❌ API root endpoint ({response.status_code})", "FAIL"))
    except Exception as e:
        test_results.append((f"❌ API root endpoint (error: {e})", "FAIL"))

    # Test 2: Health endpoint
    try:
        response = requests.get(f"{base_url}/api/v1/health", timeout=5)
        if response.status_code == 200:
            test_results.append(("✅ Health endpoint", "PASS"))
        else:
            test_results.append((f"❌ Health endpoint ({response.status_code})", "FAIL"))
    except Exception as e:
        test_results.append((f"❌ Health endpoint (error: {e})", "FAIL"))

    # Test 3: OpenAPI docs
    try:
        response = requests.get(f"{base_url}/api/v1/openapi.json", timeout=5)
        if response.status_code == 200:
            openapi_data = response.json()
            paths_count = len(openapi_data.get('paths', {}))
            test_results.append((f"✅ OpenAPI docs ({paths_count} endpoints)", "PASS"))
        else:
            test_results.append((f"❌ OpenAPI docs ({response.status_code})", "FAIL"))
    except Exception as e:
        test_results.append((f"❌ OpenAPI docs (error: {e})", "FAIL"))

    # Test 4: Check some migrated endpoints exist in OpenAPI
    try:
        response = requests.get(f"{base_url}/api/v1/openapi.json", timeout=5)
        if response.status_code == 200:
            openapi_data = response.json()
            paths = openapi_data.get('paths', {})

            # Check for migrated endpoints
            migrated_endpoints = [
                '/api/v1/automl/profile',
                '/api/v1/autonomous/detect',
                '/api/v1/ensemble/detect',
                '/api/v1/explainability/explain/prediction',
                '/api/v1/streaming/sessions'
            ]

            found_endpoints = []
            for endpoint in migrated_endpoints:
                if endpoint in paths:
                    found_endpoints.append(endpoint)

            test_results.append((f"✅ Migrated endpoints ({len(found_endpoints)}/{len(migrated_endpoints)})", "PASS"))
        else:
            test_results.append(("❌ Cannot verify migrated endpoints", "FAIL"))
    except Exception as e:
        test_results.append((f"❌ Migrated endpoints check (error: {e})", "FAIL"))

    return test_results

def main():
    """Main test runner"""
    print("🚀 Starting API Smoke Test...")

    # Start server in background
    server_process = Process(target=run_server)
    server_process.start()

    try:
        # Run tests
        results = test_endpoints()

        # Print results
        print("\n📊 Test Results:")
        print("=" * 50)

        passed = failed = 0
        for test_name, status in results:
            print(f"{test_name}")
            if status == "PASS":
                passed += 1
            else:
                failed += 1

        print("=" * 50)
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")

        if failed == 0:
            print("\n🎉 All smoke tests PASSED! API migration successful.")
            return 0
        else:
            print(f"\n⚠️  {failed} tests failed. Check server logs for details.")
            return 1

    finally:
        # Cleanup
        server_process.terminate()
        server_process.join(timeout=5)
        if server_process.is_alive():
            server_process.kill()

if __name__ == "__main__":
    sys.exit(main())
