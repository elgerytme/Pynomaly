#!/usr/bin/env python3
"""Test API dependencies and container."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pynomaly.infrastructure.config import create_container


def test_api_dependencies():
    """Test API dependencies are working."""
    print("Testing API dependencies...")

    try:
        # Create container
        container = create_container(testing=False)
        print("✓ Container created")

        # Test detector repository
        detector_repo = container.detector_repository()
        detector_count = detector_repo.count()
        print(f"✓ Detector repository works: {detector_count} detectors")

        # Test dataset repository
        dataset_repo = container.dataset_repository()
        dataset_count = dataset_repo.count()
        print(f"✓ Dataset repository works: {dataset_count} datasets")

        # Test result repository
        result_repo = container.result_repository()
        result_count = result_repo.count()
        print(f"✓ Result repository works: {result_count} results")

        # Test PyOD adapter
        try:
            pyod_adapter = container.pyod_adapter()
            algorithms = pyod_adapter.list_algorithms()
            print(f"✓ PyOD adapter works: {len(algorithms)} algorithms available")
        except Exception as e:
            print(f"⚠ PyOD adapter issue: {e}")

        # Test settings
        settings = container.config()
        print(f"✓ Settings loaded: API on {settings.api_host}:{settings.api_port}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_api_dependencies()
    print(f"\nDependency test {'✓ PASSED' if success else '✗ FAILED'}")
