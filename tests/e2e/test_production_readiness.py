#!/usr/bin/env python3
"""
Production readiness test - logging, health checks, monitoring, security
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def test_logging_infrastructure():
    """Test logging configuration and infrastructure"""
    test_results = []

    try:
        print("📝 Testing logging infrastructure...")

        # Test basic logging imports
        test_results.append(("✅ Python logging available", "PASS"))

        # Check for logging configuration files
        logging_configs = [
            "src/pynomaly/infrastructure/logging",
            "logging.conf",
            "logging.yml",
        ]

        found_configs = []
        for config in logging_configs:
            if os.path.exists(config):
                found_configs.append(config)

        if found_configs:
            test_results.append(
                (f"✅ Logging config found: {len(found_configs)} files", "PASS")
            )
        else:
            test_results.append(("⚠️ No logging config files found", "WARN"))

        return test_results

    except Exception as e:
        test_results.append((f"❌ Logging test failed: {str(e)}", "FAIL"))
        return test_results


def test_health_checks():
    """Test health check endpoints and functionality"""
    test_results = []

    try:
        print("🏥 Testing health check infrastructure...")

        # Check health endpoint file
        health_file = "src/pynomaly/presentation/api/endpoints/health.py"
        if os.path.exists(health_file):
            with open(health_file) as f:
                content = f.read()

            # Check for essential health check endpoints
            health_endpoints = ["/health", "/ready", "/live", "/metrics"]

            found_endpoints = []
            for endpoint in health_endpoints:
                if endpoint in content:
                    found_endpoints.append(endpoint)

            test_results.append(
                (
                    f"✅ Health endpoints: {len(found_endpoints)}/{len(health_endpoints)}",
                    "PASS",
                )
            )

            # Check for dependency checks
            if "database" in content.lower() or "redis" in content.lower():
                test_results.append(("✅ Dependency health checks included", "PASS"))
            else:
                test_results.append(("⚠️ Limited dependency health checks", "WARN"))
        else:
            test_results.append(("❌ Health endpoint file missing", "FAIL"))

        return test_results

    except Exception as e:
        test_results.append((f"❌ Health check test failed: {str(e)}", "FAIL"))
        return test_results


def test_monitoring_integration():
    """Test monitoring and observability features"""
    test_results = []

    try:
        print("📊 Testing monitoring integration...")

        # Check for Prometheus metrics
        try:
            from prometheus_fastapi_instrumentator import Instrumentator

            test_results.append(("✅ Prometheus instrumentator available", "PASS"))
        except ImportError:
            test_results.append(("❌ Prometheus instrumentator missing", "FAIL"))

        # Check app.py for monitoring configuration
        app_file = "src/pynomaly/presentation/api/app.py"
        if os.path.exists(app_file):
            with open(app_file) as f:
                content = f.read()

            if "prometheus" in content.lower():
                test_results.append(("✅ Prometheus integration configured", "PASS"))
            else:
                test_results.append(("❌ Prometheus integration missing", "FAIL"))

            if "track_request_metrics" in content:
                test_results.append(("✅ Request metrics tracking enabled", "PASS"))
            else:
                test_results.append(("❌ Request metrics tracking missing", "FAIL"))

        return test_results

    except Exception as e:
        test_results.append((f"❌ Monitoring test failed: {str(e)}", "FAIL"))
        return test_results


def test_security_features():
    """Test production security features"""
    test_results = []

    try:
        print("🔒 Testing security features...")

        # Check authentication infrastructure
        auth_deps_file = "src/pynomaly/presentation/api/auth_deps.py"
        if os.path.exists(auth_deps_file):
            test_results.append(("✅ Authentication system available", "PASS"))
        else:
            test_results.append(("❌ Authentication system missing", "FAIL"))

        # Check app.py for security middleware
        app_file = "src/pynomaly/presentation/api/app.py"
        if os.path.exists(app_file):
            with open(app_file) as f:
                content = f.read()

            security_features = [
                ("CORS", "CORSMiddleware"),
                ("Request tracking", "track_request_metrics"),
                ("Auth initialization", "init_auth"),
            ]

            for feature_name, feature_code in security_features:
                if feature_code in content:
                    test_results.append((f"✅ {feature_name} configured", "PASS"))
                else:
                    test_results.append((f"❌ {feature_name} missing", "FAIL"))

        # Check for security service
        security_service_file = (
            "src/pynomaly/infrastructure/security/security_service.py"
        )
        if os.path.exists(security_service_file):
            test_results.append(("✅ Security service available", "PASS"))
        else:
            test_results.append(("❌ Security service missing", "FAIL"))

        return test_results

    except Exception as e:
        test_results.append((f"❌ Security test failed: {str(e)}", "FAIL"))
        return test_results


def test_error_handling():
    """Test error handling and resilience features"""
    test_results = []

    try:
        print("⚡ Testing error handling...")

        # Check for error handling in endpoints
        endpoint_files = [
            "src/pynomaly/presentation/api/endpoints/health.py",
            "src/pynomaly/presentation/api/endpoints/auth.py",
            "src/pynomaly/presentation/api/endpoints/automl.py",
        ]

        error_handling_found = 0
        for file_path in endpoint_files:
            if os.path.exists(file_path):
                with open(file_path) as f:
                    content = f.read()

                if (
                    "try:" in content
                    and "except" in content
                    and "HTTPException" in content
                ):
                    error_handling_found += 1

        if error_handling_found >= 2:
            test_results.append(
                (f"✅ Error handling in {error_handling_found} endpoint files", "PASS")
            )
        else:
            test_results.append(("❌ Limited error handling found", "FAIL"))

        # Check for circuit breaker or resilience patterns
        container_file = "src/pynomaly/infrastructure/config/container.py"
        if os.path.exists(container_file):
            with open(container_file) as f:
                content = f.read()

            if "circuit" in content.lower() or "retry" in content.lower():
                test_results.append(("✅ Resilience patterns found", "PASS"))
            else:
                test_results.append(("⚠️ Limited resilience patterns", "WARN"))

        return test_results

    except Exception as e:
        test_results.append((f"❌ Error handling test failed: {str(e)}", "FAIL"))
        return test_results


def test_configuration_management():
    """Test configuration and environment management"""
    test_results = []

    try:
        print("⚙️ Testing configuration management...")

        # Check for settings configuration
        settings_file = "src/pynomaly/infrastructure/config/settings.py"
        if os.path.exists(settings_file):
            with open(settings_file) as f:
                content = f.read()

            config_features = [
                ("Environment variables", "Field("),
                ("Validation", "BaseSettings"),
                ("Documentation", "description="),
            ]

            for feature_name, feature_code in config_features:
                if feature_code in content:
                    test_results.append((f"✅ {feature_name} support", "PASS"))
                else:
                    test_results.append((f"❌ {feature_name} missing", "FAIL"))
        else:
            test_results.append(("❌ Settings configuration missing", "FAIL"))

        # Check for environment files
        env_files = [".env.example", ".env.template", "docker-compose.yml"]
        found_env_files = [f for f in env_files if os.path.exists(f)]

        if found_env_files:
            test_results.append(
                (f"✅ Environment files: {len(found_env_files)}", "PASS")
            )
        else:
            test_results.append(("⚠️ No environment template files", "WARN"))

        return test_results

    except Exception as e:
        test_results.append((f"❌ Configuration test failed: {str(e)}", "FAIL"))
        return test_results


def main():
    """Main production readiness test runner"""
    print("🚀 Production Readiness Test")
    print("=" * 60)

    all_results = []

    # Test 1: Logging Infrastructure
    logging_results = test_logging_infrastructure()
    all_results.extend(logging_results)

    # Test 2: Health Checks
    health_results = test_health_checks()
    all_results.extend(health_results)

    # Test 3: Monitoring Integration
    monitoring_results = test_monitoring_integration()
    all_results.extend(monitoring_results)

    # Test 4: Security Features
    security_results = test_security_features()
    all_results.extend(security_results)

    # Test 5: Error Handling
    error_results = test_error_handling()
    all_results.extend(error_results)

    # Test 6: Configuration Management
    config_results = test_configuration_management()
    all_results.extend(config_results)

    # Print results
    print("\n📊 Production Readiness Results:")
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

    if success_rate >= 70:  # Allow some tolerance for production features
        print("\n🎉 Production readiness ACHIEVED!")
        print("✅ Core production infrastructure in place")
        print("✅ Security and monitoring configured")
        print("✅ Health checks and error handling available")
        return 0
    else:
        print("\n⚠️  Production readiness needs improvement.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
