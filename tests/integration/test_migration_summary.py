#!/usr/bin/env python3
"""
Summary test for authentication migration completion
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

def test_migration_files():
    """Test that migration files exist and have correct content"""
    test_results = []

    # Test 1: auth_deps.py exists and has required functions
    try:
        with open('src/pynomaly/presentation/api/auth_deps.py', 'r') as f:
            content = f.read()

        required_functions = ['get_current_user_simple', 'get_current_user_model', 'get_container_simple']
        for func in required_functions:
            if f"def {func}" in content:
                test_results.append((f"✅ {func} function exists", "PASS"))
            else:
                test_results.append((f"❌ {func} function missing", "FAIL"))

    except FileNotFoundError:
        test_results.append(("❌ auth_deps.py file missing", "FAIL"))

    return test_results

def test_endpoint_migrations():
    """Test that endpoint files have been migrated"""
    test_results = []

    migrated_files = [
        'src/pynomaly/presentation/api/endpoints/automl.py',
        'src/pynomaly/presentation/api/endpoints/autonomous.py',
        'src/pynomaly/presentation/api/endpoints/ensemble.py',
        'src/pynomaly/presentation/api/endpoints/explainability.py',
        'src/pynomaly/presentation/api/endpoints/streaming.py',
        'src/pynomaly/presentation/api/endpoints/export.py',
        'src/pynomaly/presentation/api/endpoints/model_lineage.py',
        'src/pynomaly/presentation/api/endpoints/events.py'
    ]

    for file_path in migrated_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Check if file uses simplified auth
            if 'get_container_simple' in content:
                test_results.append((f"✅ {os.path.basename(file_path)} migrated", "PASS"))
            elif 'get_container' in content and 'get_container_simple' not in content:
                test_results.append((f"❌ {os.path.basename(file_path)} not migrated", "FAIL"))
            else:
                test_results.append((f"⚠️ {os.path.basename(file_path)} no auth deps", "WARN"))

        except FileNotFoundError:
            test_results.append((f"❌ {os.path.basename(file_path)} missing", "FAIL"))

    return test_results

def test_app_router_inclusion():
    """Test that routers are included in main app"""
    test_results = []

    try:
        with open('src/pynomaly/presentation/api/app.py', 'r') as f:
            content = f.read()

        # Check for active router inclusions (not commented out)
        active_routers = [
            'automl.router',
            'autonomous.router',
            'ensemble.router',
            'explainability.router',
            'streaming.router',
            'export.router',
            'model_lineage.router',
            'events.router',
            'performance.router'
        ]

        for router in active_routers:
            # Look for uncommented include_router lines
            lines = content.split('\n')
            router_included = False
            for line in lines:
                if router in line and 'include_router' in line and not line.strip().startswith('#'):
                    router_included = True
                    break

            if router_included:
                test_results.append((f"✅ {router} included", "PASS"))
            else:
                test_results.append((f"❌ {router} not included", "FAIL"))

    except FileNotFoundError:
        test_results.append(("❌ app.py file missing", "FAIL"))

    return test_results

def count_endpoints_from_app():
    """Count endpoints by checking app.py routers"""
    try:
        with open('src/pynomaly/presentation/api/app.py', 'r') as f:
            content = f.read()

        # Count include_router lines that are not commented
        lines = content.split('\n')
        active_routers = 0

        for line in lines:
            if 'include_router' in line and not line.strip().startswith('#'):
                active_routers += 1

        return active_routers
    except:
        return 0

def main():
    """Main test runner"""
    print("🚀 Authentication Migration Summary Test")
    print("=" * 60)

    all_results = []

    # Test 1: Migration Files
    print("📁 Testing migration files...")
    file_results = test_migration_files()
    all_results.extend(file_results)

    # Test 2: Endpoint Migrations
    print("🔗 Testing endpoint migrations...")
    endpoint_results = test_endpoint_migrations()
    all_results.extend(endpoint_results)

    # Test 3: App Router Inclusion
    print("📱 Testing app router inclusion...")
    app_results = test_app_router_inclusion()
    all_results.extend(app_results)

    # Print results
    print("\n📊 Migration Summary Results:")
    print("=" * 60)

    passed = failed = warnings = 0
    for test_name, status in all_results:
        print(f"{test_name}")
        if status == "PASS":
            passed += 1
        elif status == "FAIL":
            failed += 1
        else:
            warnings += 1

    # Additional info
    router_count = count_endpoints_from_app()
    print(f"\n📈 Migration Statistics:")
    print(f"  • Active routers in app.py: {router_count}")
    print(f"  • Migration tests passed: {passed}")
    print(f"  • Migration tests failed: {failed}")
    print(f"  • Warnings: {warnings}")

    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Warnings: {warnings}")
    success_rate = passed / (passed + failed) * 100 if (passed + failed) > 0 else 0
    print(f"📈 Success Rate: {success_rate:.1f}%")

    if failed == 0:
        print("\n🎉 Authentication migration COMPLETED successfully!")
        print("✅ All core routers migrated to simplified auth dependencies")
        print("✅ All routers enabled in main application")
        return 0
    else:
        print(f"\n⚠️  {failed} migration tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
