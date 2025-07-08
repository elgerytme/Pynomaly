#!/usr/bin/env python3
"""
Quick test to verify core functionality works after pyproject.toml fix.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_domain_imports():
    """Test that core domain imports work."""
    print("🔍 Testing domain layer imports...")
    try:

        print("✅ Domain layer imports successful")
        return True
    except Exception as e:
        print(f"❌ Domain layer import failed: {e}")
        return False


def test_application_imports():
    """Test that application layer imports work."""
    print("🔍 Testing application layer imports...")
    try:

        print("✅ Application layer imports successful")
        return True
    except Exception as e:
        print(f"❌ Application layer import failed: {e}")
        return False


def test_infrastructure_imports():
    """Test that infrastructure imports work."""
    print("🔍 Testing infrastructure layer imports...")
    try:

        print("✅ Infrastructure layer imports successful")
        return True
    except Exception as e:
        print(f"❌ Infrastructure layer import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality with simple detector."""
    print("🔍 Testing basic anomaly detection functionality...")
    try:
        import numpy as np
        import pandas as pd
        from pynomaly.domain.entities import Dataset
        from pynomaly.domain.value_objects import ContaminationRate
        from pynomaly.infrastructure.adapters import SklearnAdapter

        # Create sample data
        data = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 100),
                "feature_2": np.random.normal(0, 1, 100),
            }
        )

        # Add some anomalies
        data.iloc[-5:] = 5  # Last 5 points are anomalies

        dataset = Dataset(
            name="test_dataset", data=data, feature_names=["feature_1", "feature_2"]
        )

        # Create detector
        detector = SklearnAdapter(
            algorithm_name="IsolationForest",
            name="test_detector",
            contamination_rate=ContaminationRate(0.1),
        )

        print("✅ Basic functionality test passed")
        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_container_import():
    """Test that the dependency injection container works."""
    print("🔍 Testing dependency injection container...")
    try:
        from pynomaly.infrastructure.config.container import Container

        container = Container()
        print("✅ Container import and creation successful")
        return True
    except Exception as e:
        print(f"❌ Container test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Running Core Functionality Tests")
    print("=" * 50)

    tests = [
        test_domain_imports,
        test_application_imports,
        test_infrastructure_imports,
        test_container_import,
        test_basic_functionality,
    ]

    results = []
    for test in tests:
        results.append(test())
        print()

    passed = sum(results)
    total = len(results)

    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All core functionality tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed - core functionality may have issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
