#!/usr/bin/env python3
"""Test script to verify API versioning implementation."""

import os
import re
import sys


def test_api_versioning():
    """Test that API versioning is properly implemented."""

    print("🔍 Testing API Versioning Implementation")
    print("=" * 50)

    # Read the main app.py file
    app_file_path = "src/pynomaly/presentation/api/app.py"

    if not os.path.exists(app_file_path):
        print("❌ app.py file not found!")
        return False

    with open(app_file_path, "r") as f:
        content = f.read()

    # Test 1: Check for versioned FastAPI configuration
    print("\n1. Checking FastAPI configuration...")
    if '"/api/v1/docs"' in content:
        print("✅ Docs URL properly versioned: /api/v1/docs")
    else:
        print("❌ Docs URL not properly versioned")

    if '"/api/v1/redoc"' in content:
        print("✅ Redoc URL properly versioned: /api/v1/redoc")
    else:
        print("❌ Redoc URL not properly versioned")

    if '"/api/v1/openapi.json"' in content:
        print("✅ OpenAPI URL properly versioned: /api/v1/openapi.json")
    else:
        print("❌ OpenAPI URL not properly versioned")

    # Test 2: Check for versioned router includes
    print("\n2. Checking router includes...")
    versioned_routers = [
        'prefix="/api/v1"',
        'prefix="/api/v1/auth"',
        'prefix="/api/v1/admin"',
        'prefix="/api/v1/autonomous"',
        'prefix="/api/v1/detectors"',
        'prefix="/api/v1/datasets"',
        'prefix="/api/v1/detection"',
        'prefix="/api/v1/automl"',
        'prefix="/api/v1/ensemble"',
        'prefix="/api/v1/explainability"',
        'prefix="/api/v1/experiments"',
        'prefix="/api/v1/performance"',
        'prefix="/api/v1/streaming"',
        'prefix="/api/v1/events"',
    ]

    found_versioned = 0
    for router in versioned_routers:
        if router in content:
            found_versioned += 1
            print(f"✅ Found versioned router: {router}")
        else:
            print(f"❌ Missing versioned router: {router}")

    # Test 3: Check for non-versioned routers (should not exist)
    print("\n3. Checking for non-versioned routers...")
    old_patterns = ['prefix="/api"', 'prefix="/api/auth"', 'prefix="/api/admin"']

    found_old = 0
    for pattern in old_patterns:
        matches = re.findall(pattern, content)
        if matches:
            found_old += len(matches)
            print(f"⚠️  Found non-versioned pattern: {pattern} ({len(matches)} times)")

    # Test 4: Check documentation strings
    print("\n4. Checking documentation strings...")
    if "/api/v1/auth/login" in content:
        print("✅ Quick start documentation uses versioned URLs")
    else:
        print("❌ Quick start documentation not updated")

    # Test 5: Check version endpoint import
    print("\n5. Checking version endpoint...")
    if (
        "version," in content
        and "from pynomaly.presentation.api.endpoints import" in content
    ):
        print("✅ Version endpoint imported")
    else:
        print("❌ Version endpoint not imported")

    if 'version.router, prefix="/api/v1"' in content:
        print("✅ Version router included with v1 prefix")
    else:
        print("❌ Version router not properly included")

    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)

    total_checks = len(versioned_routers) + 6  # Additional checks
    passed_checks = found_versioned + (6 if found_old == 0 else 5)

    print(f"Versioned routers found: {found_versioned}/{len(versioned_routers)}")
    print(f"Non-versioned patterns found: {found_old} (should be 0)")

    if found_versioned >= len(versioned_routers) * 0.8 and found_old == 0:
        print("✅ API versioning implementation: PASSED")
        return True
    else:
        print("❌ API versioning implementation: NEEDS WORK")
        return False


if __name__ == "__main__":
    success = test_api_versioning()
    sys.exit(0 if success else 1)
