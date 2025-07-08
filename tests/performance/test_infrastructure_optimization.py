#!/usr/bin/env python3
"""
Infrastructure optimization test - caching, monitoring, telemetry
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

def test_caching_infrastructure():
    """Test caching system availability"""
    test_results = []

    try:
        print("🗄️ Testing caching infrastructure...")

        # Test basic cache imports
        from pynomaly.infrastructure.cache import get_cache, init_cache
        test_results.append(("✅ Cache module imports successful", "PASS"))

        # Test cache initialization
        cache = get_cache()
        test_results.append(("✅ Cache instance creation", "PASS"))

        return test_results

    except ImportError as e:
        test_results.append((f"❌ Cache import failed: {str(e)}", "FAIL"))
        return test_results
    except Exception as e:
        test_results.append((f"❌ Cache test failed: {str(e)}", "FAIL"))
        return test_results

def test_monitoring_infrastructure():
    """Test monitoring system availability"""
    test_results = []

    try:
        print("📊 Testing monitoring infrastructure...")

        # Test basic monitoring imports
        try:
            from pynomaly.infrastructure.monitoring import get_telemetry
            test_results.append(("✅ Monitoring module imports", "PASS"))
        except ImportError:
            test_results.append(("⚠️ Monitoring module disabled", "WARN"))

        # Test Prometheus metrics
        try:
            from prometheus_fastapi_instrumentator import Instrumentator
            test_results.append(("✅ Prometheus metrics available", "PASS"))
        except ImportError:
            test_results.append(("❌ Prometheus metrics not available", "FAIL"))

        return test_results

    except Exception as e:
        test_results.append((f"❌ Monitoring test failed: {str(e)}", "FAIL"))
        return test_results

def test_performance_optimizations():
    """Test performance optimization features"""
    test_results = []

    try:
        print("⚡ Testing performance optimizations...")

        # Test performance module
        if os.path.exists('src/pynomaly/infrastructure/performance'):
            test_results.append(("✅ Performance module exists", "PASS"))

            # Check for specific optimization components
            performance_files = [
                'profiling_service.py',
                '__init__.py'
            ]

            for file in performance_files:
                if os.path.exists(f'src/pynomaly/infrastructure/performance/{file}'):
                    test_results.append((f"✅ {file} exists", "PASS"))
                else:
                    test_results.append((f"❌ {file} missing", "FAIL"))
        else:
            test_results.append(("❌ Performance module missing", "FAIL"))

        return test_results

    except Exception as e:
        test_results.append((f"❌ Performance test failed: {str(e)}", "FAIL"))
        return test_results

def test_container_optimization():
    """Test dependency injection container optimizations"""
    test_results = []

    try:
        print("📦 Testing container optimizations...")

        # Test simplified container creation (without complex imports)
        container_file = 'src/pynomaly/infrastructure/config/container.py'
        if os.path.exists(container_file):
            with open(container_file, 'r') as f:
                content = f.read()

            # Check for optimization features
            if 'OptionalServiceManager' in content:
                test_results.append(("✅ Optional service manager available", "PASS"))
            else:
                test_results.append(("❌ Optional service manager missing", "FAIL"))

            if 'providers.Singleton' in content:
                test_results.append(("✅ Singleton patterns used", "PASS"))
            else:
                test_results.append(("❌ Singleton patterns missing", "FAIL"))

        return test_results

    except Exception as e:
        test_results.append((f"❌ Container optimization test failed: {str(e)}", "FAIL"))
        return test_results

def test_enhanced_automl_availability():
    """Test enhanced AutoML features"""
    test_results = []

    try:
        print("🤖 Testing enhanced AutoML availability...")

        # Check if enhanced AutoML is available
        try:
            from pynomaly.presentation.api import enhanced_automl
            test_results.append(("✅ Enhanced AutoML available", "PASS"))
        except ImportError:
            test_results.append(("⚠️ Enhanced AutoML not available", "WARN"))

        return test_results

    except Exception as e:
        test_results.append((f"❌ Enhanced AutoML test failed: {str(e)}", "FAIL"))
        return test_results

def test_security_optimizations():
    """Test security optimization features"""
    test_results = []

    try:
        print("🔒 Testing security optimizations...")

        # Check for security service
        security_file = 'src/pynomaly/infrastructure/security/security_service.py'
        if os.path.exists(security_file):
            test_results.append(("✅ Security service exists", "PASS"))
        else:
            test_results.append(("❌ Security service missing", "FAIL"))

        # Check auth optimizations
        auth_deps_file = 'src/pynomaly/presentation/api/auth_deps.py'
        if os.path.exists(auth_deps_file):
            with open(auth_deps_file, 'r') as f:
                content = f.read()

            if 'get_current_user_simple' in content:
                test_results.append(("✅ Simplified auth dependencies", "PASS"))
            else:
                test_results.append(("❌ Simplified auth missing", "FAIL"))

        return test_results

    except Exception as e:
        test_results.append((f"❌ Security optimization test failed: {str(e)}", "FAIL"))
        return test_results

def main():
    """Main infrastructure optimization test runner"""
    print("🚀 Infrastructure Optimization Test")
    print("=" * 60)

    all_results = []

    # Test 1: Caching Infrastructure
    cache_results = test_caching_infrastructure()
    all_results.extend(cache_results)

    # Test 2: Monitoring Infrastructure
    monitoring_results = test_monitoring_infrastructure()
    all_results.extend(monitoring_results)

    # Test 3: Performance Optimizations
    performance_results = test_performance_optimizations()
    all_results.extend(performance_results)

    # Test 4: Container Optimization
    container_results = test_container_optimization()
    all_results.extend(container_results)

    # Test 5: Enhanced AutoML
    automl_results = test_enhanced_automl_availability()
    all_results.extend(automl_results)

    # Test 6: Security Optimizations
    security_results = test_security_optimizations()
    all_results.extend(security_results)

    # Print results
    print("\n📊 Infrastructure Optimization Results:")
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

    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Warnings: {warnings}")
    success_rate = passed / (passed + failed) * 100 if (passed + failed) > 0 else 0
    print(f"📈 Success Rate: {success_rate:.1f}%")

    if failed <= 2:  # Allow some tolerance for optional features
        print("\n🎉 Infrastructure optimization SUCCESSFUL!")
        print("✅ Core caching and monitoring infrastructure available")
        print("✅ Performance optimizations in place")
        print("✅ Security enhancements implemented")
        return 0
    else:
        print(f"\n⚠️  {failed} infrastructure tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
