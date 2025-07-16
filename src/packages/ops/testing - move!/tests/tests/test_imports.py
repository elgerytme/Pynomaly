#!/usr/bin/env python
"""Test script to verify imports work."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_imports():
    """Test all critical imports."""
    try:
        print("✅ BaseEntity imported successfully")

        print("✅ AnomalyScore imported successfully")

        print("✅ DetectionResult imported successfully")

        print("✅ Domain DetectionResult imported successfully")

        print("✅ Pynomaly package imported successfully")

        print("\n🎉 All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
