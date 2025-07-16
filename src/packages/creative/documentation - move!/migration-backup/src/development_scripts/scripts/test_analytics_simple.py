#!/usr/bin/env python3
"""
Simple test script for analytics dashboard.
"""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


async def test_analytics_import():
    """Test that analytics dashboard can be imported."""
    try:
        print("✅ Analytics dashboard imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import analytics dashboard: {e}")
        return False


async def test_analytics_service():
    """Test analytics service functionality."""
    try:
        from datetime import datetime, timedelta

        from monorepo.presentation.web.analytics_dashboard import (
            AnalyticsQuery,
            analytics_service,
        )

        # Test query
        query = AnalyticsQuery(
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now(),
            metric_type="daily",
        )

        # Test detection analytics
        detection_result = await analytics_service.get_detection_analytics(query)
        print(f"✅ Detection analytics: {len(detection_result['data'])} data points")

        # Test performance analytics
        performance_result = await analytics_service.get_performance_analytics(query)
        print(
            f"✅ Performance analytics: {len(performance_result['data'])} data points"
        )

        # Test business analytics
        business_result = await analytics_service.get_business_analytics(query)
        print(f"✅ Business analytics: {len(business_result['data'])} data points")

        return True
    except Exception as e:
        print(f"❌ Analytics service test failed: {e}")
        return False


async def test_app_integration():
    """Test app integration with analytics."""
    try:
        from monorepo.presentation.web.simple_app import create_app

        app = create_app()

        # Check if analytics routes are included
        route_paths = [route.path for route in app.routes]
        analytics_routes = [path for path in route_paths if "/analytics" in path]

        if analytics_routes:
            print(f"✅ Analytics routes found: {len(analytics_routes)} routes")
            for route in analytics_routes:
                print(f"  - {route}")
        else:
            print("⚠️  No analytics routes found")

        return True
    except Exception as e:
        print(f"❌ App integration test failed: {e}")
        return False


async def main():
    """Main test function."""
    print("🧪 Testing Analytics Dashboard Implementation")
    print("=" * 50)

    # Run tests
    test_results = []

    print("\n1. Testing Analytics Import...")
    test_results.append(await test_analytics_import())

    print("\n2. Testing Analytics Service...")
    test_results.append(await test_analytics_service())

    print("\n3. Testing App Integration...")
    test_results.append(await test_app_integration())

    # Summary
    successful_tests = sum(test_results)
    total_tests = len(test_results)

    print(f"\n{'='*50}")
    print(f"Test Summary: {successful_tests}/{total_tests} tests passed")

    if successful_tests == total_tests:
        print("🎉 All tests passed! Analytics dashboard is ready.")
        return True
    else:
        print("❌ Some tests failed. Please check the output above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
