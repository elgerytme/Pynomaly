#!/usr/bin/env python3
"""Health check script for integration testing infrastructure."""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from pynomaly.infrastructure.config import create_container
    from pynomaly.presentation.api.app import create_app
    from httpx import AsyncClient
    import uvicorn
    import threading
    import socket
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("  poetry install")
    sys.exit(1)


def find_free_port():
    """Find a free port to use for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


async def check_api_health():
    """Check if the API can start and respond to basic requests."""
    print("🔍 Testing API startup and basic functionality...")
    
    try:
        # Create test app
        container = create_container()
        app = create_app(container)
        
        # Test with in-memory client (no server needed)
        from httpx import ASGITransport
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            
            # Test root endpoint
            response = await client.get("/")
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            print(f"✅ Root endpoint: {data['message']}")
            
            # Test health endpoint
            response = await client.get("/api/health")
            assert response.status_code == 200
            health = response.json()
            assert health["status"] in ["healthy", "ok", "ready"]
            print(f"✅ Health endpoint: {health['status']}")
            
            # Test OpenAPI docs
            response = await client.get("/api/openapi.json")
            if response.status_code == 200:
                openapi = response.json()
                assert "openapi" in openapi
                print(f"✅ OpenAPI docs: Available ({len(openapi.get('paths', {}))} endpoints)")
            else:
                print(f"⚠️  OpenAPI docs: Not available (status: {response.status_code})")
            
            # Test datasets endpoint (basic structure)
            response = await client.get("/api/datasets")
            # Should either work or return auth error
            assert response.status_code in [200, 401, 403]
            if response.status_code == 200:
                print("✅ Datasets endpoint: Accessible")
            else:
                print(f"ℹ️  Datasets endpoint: Protected (status: {response.status_code})")
            
        print("✅ API health check passed!")
        return True
        
    except Exception as e:
        print(f"❌ API health check failed: {e}")
        return False


def check_dependencies():
    """Check if all required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "fastapi",
        "uvicorn", 
        "httpx",
        "pytest",
        "pytest-asyncio",
        "pydantic",
        "pandas",
        "numpy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: Available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}: Missing")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install missing packages with: poetry install")
        return False
    
    print("✅ All dependencies available!")
    return True


def check_test_infrastructure():
    """Check if test infrastructure is properly set up."""
    print("🔍 Checking test infrastructure...")
    
    # Check test directories
    test_dirs = [
        project_root / "tests",
        project_root / "tests" / "integration",
        project_root / "test_reports"
    ]
    
    for test_dir in test_dirs:
        if test_dir.exists():
            print(f"✅ {test_dir.name}: Directory exists")
        else:
            test_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 {test_dir.name}: Created directory")
    
    # Check test files
    integration_test_files = [
        "conftest.py",
        "test_api_workflows.py",
        "test_database_integration.py",
        "test_streaming_integration.py",
        "test_performance_integration.py",
        "test_security_integration.py",
        "test_end_to_end_scenarios.py",
        "test_regression_suite.py"
    ]
    
    integration_dir = project_root / "tests" / "integration"
    missing_files = []
    
    for test_file in integration_test_files:
        file_path = integration_dir / test_file
        if file_path.exists():
            print(f"✅ {test_file}: Available")
        else:
            missing_files.append(test_file)
            print(f"❌ {test_file}: Missing")
    
    if missing_files:
        print(f"\n❌ Missing test files: {', '.join(missing_files)}")
        return False
    
    # Check pytest configuration
    pytest_ini = project_root / "pytest.ini"
    if pytest_ini.exists():
        print("✅ pytest.ini: Configuration available")
    else:
        print("⚠️  pytest.ini: Configuration missing")
    
    print("✅ Test infrastructure check passed!")
    return True


def check_environment():
    """Check environment configuration."""
    print("🔍 Checking environment configuration...")
    
    import os
    
    # Set test environment variables
    test_env_vars = {
        "PYNOMALY_ENVIRONMENT": "testing",
        "PYNOMALY_LOG_LEVEL": "INFO",
        "PYNOMALY_CACHE_ENABLED": "false",
        "PYNOMALY_AUTH_ENABLED": "false",
        "PYNOMALY_DOCS_ENABLED": "true"
    }
    
    for var, value in test_env_vars.items():
        os.environ[var] = value
        print(f"✅ {var}: Set to '{value}'")
    
    # Check data directories
    data_dir = project_root / "test_data"
    data_dir.mkdir(exist_ok=True)
    print(f"✅ Test data directory: {data_dir}")
    
    print("✅ Environment configuration completed!")
    return True


async def run_sample_integration_test():
    """Run a simple integration test to verify everything works."""
    print("🔍 Running sample integration test...")
    
    try:
        # Create test app
        container = create_container()
        app = create_app(container)
        
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            
            # Simple workflow test
            print("  📋 Testing basic API workflow...")
            
            # 1. Check health
            response = await client.get("/api/health")
            assert response.status_code == 200
            print("    ✅ Health check passed")
            
            # 2. List datasets (should be empty initially)
            response = await client.get("/api/datasets")
            assert response.status_code in [200, 401, 403]
            print("    ✅ Dataset listing works")
            
            # 3. Check if we can access detector endpoints
            response = await client.get("/api/detectors")
            assert response.status_code in [200, 401, 403]
            print("    ✅ Detector endpoints accessible")
            
            # 4. Check streaming endpoints
            response = await client.get("/api/streaming/sessions")
            assert response.status_code in [200, 401, 403]
            print("    ✅ Streaming endpoints accessible")
            
        print("✅ Sample integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Sample integration test failed: {e}")
        return False


def create_sample_test_data():
    """Create sample test data for integration tests."""
    print("🔍 Creating sample test data...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create test data directory
        test_data_dir = project_root / "test_data"
        test_data_dir.mkdir(exist_ok=True)
        
        # Generate sample dataset
        np.random.seed(42)
        n_samples = 100
        
        # Normal data
        normal_data = np.random.multivariate_normal(
            mean=[0, 0, 0],
            cov=[[1, 0.3, 0.1], [0.3, 1, 0.2], [0.1, 0.2, 1]],
            size=n_samples - 10
        )
        
        # Anomalous data
        anomaly_data = np.random.multivariate_normal(
            mean=[3, 3, 3],
            cov=[[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
            size=10
        )
        
        # Combine data
        data = np.vstack([normal_data, anomaly_data])
        labels = np.hstack([np.zeros(n_samples - 10), np.ones(10)])
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3'])
        df['label'] = labels
        df['timestamp'] = pd.date_range('2024-01-01', periods=n_samples, freq='1H')
        
        # Save to CSV
        csv_path = test_data_dir / "sample_test_dataset.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"✅ Sample dataset created: {csv_path}")
        print(f"    📊 {n_samples} samples, {len(df.columns)} features")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create sample test data: {e}")
        return False


async def main():
    """Main health check function."""
    print("🚀 Pynomaly Integration Testing Health Check")
    print("=" * 50)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Environment", check_environment),
        ("Test Infrastructure", check_test_infrastructure),
        ("Sample Test Data", create_sample_test_data),
        ("API Health", check_api_health),
        ("Sample Integration Test", run_sample_integration_test)
    ]
    
    passed_checks = 0
    total_checks = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n🔎 {check_name}")
        print("-" * 30)
        
        try:
            if asyncio.iscoroutinefunction(check_func):
                success = await check_func()
            else:
                success = check_func()
            
            if success:
                passed_checks += 1
                print(f"✅ {check_name}: PASSED")
            else:
                print(f"❌ {check_name}: FAILED")
                
        except Exception as e:
            print(f"❌ {check_name}: ERROR - {e}")
    
    # Final summary
    print(f"\n{'='*50}")
    print("🏁 HEALTH CHECK SUMMARY")
    print(f"{'='*50}")
    print(f"Passed: {passed_checks}/{total_checks}")
    print(f"Success Rate: {(passed_checks/total_checks*100):.1f}%")
    
    if passed_checks == total_checks:
        print("🎉 All checks passed! Integration testing infrastructure is ready.")
        print("\nNext steps:")
        print("  1. Run full integration tests: python scripts/run_integration_tests.py")
        print("  2. Run specific test suite: python scripts/run_integration_tests.py --suite test_api_workflows.py")
        print("  3. Run with coverage: python scripts/run_integration_tests.py --coverage")
        return True
    else:
        print("⚠️  Some checks failed. Please address the issues above before running integration tests.")
        return False


if __name__ == "__main__":
    # Run health check
    success = asyncio.run(main())
    sys.exit(0 if success else 1)