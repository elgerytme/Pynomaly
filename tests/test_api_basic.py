#!/usr/bin/env python3
"""Test basic API functionality."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import threading
import time

import requests
import uvicorn

from pynomaly.presentation.api.app import create_app


def test_basic_api():
    """Test basic API functionality."""
    print("Creating API app...")
    try:
        app = create_app()
        print("✓ API app created successfully")

        # Start server in background thread
        def run_server():
            uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        # Wait for server to start
        time.sleep(2)

        # Test endpoints
        base_url = "http://127.0.0.1:8001"

        print("Testing root endpoint...")
        try:
            response = requests.get(base_url, timeout=5)
            if response.status_code == 200:
                print("✓ Root endpoint working")
                try:
                    print(f"  Response: {response.json()}")
                except:
                    print(f"  Response (text): {response.text}")
            else:
                print(f"✗ Root endpoint failed: {response.status_code}")
        except Exception as e:
            print(f"✗ Root endpoint error: {e}")

        print("Testing health endpoint...")
        try:
            response = requests.get(f"{base_url}/api/health/", timeout=5)
            if response.status_code == 200:
                print("✓ Health endpoint working")
                data = response.json()
                print(f"  Status: {data.get('overall_status', 'unknown')}")
            else:
                print(f"✗ Health endpoint failed: {response.status_code}")
                print(f"  Response: {response.text}")
        except Exception as e:
            print(f"✗ Health endpoint error: {e}")

        return True

    except Exception as e:
        print(f"✗ API app creation failed: {e}")
        return False


if __name__ == "__main__":
    success = test_basic_api()
    print(f"\nAPI test {'✓ PASSED' if success else '✗ FAILED'}")
