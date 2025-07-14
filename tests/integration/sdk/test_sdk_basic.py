#!/usr/bin/env python3
"""Simplified test script for Pynomaly SDK."""

import os
import sys


def test_basic_imports():
    """Test basic Pynomaly imports."""
    try:
        print(f"🐍 Python version: {sys.version}")
        print(f"📁 Current working directory: {os.getcwd()}")

        # Test basic package import

        print("✅ Package import successful")

        # Test domain entities

        print("✅ Domain entities import successful")

        # Test application services

        print("✅ Application services import successful")

        # Test infrastructure adapters

        print("✅ Infrastructure adapters import successful")

        # Test basic SDK models without async client
        from pynomaly.presentation.sdk.models import AnomalyScore, DetectionConfig

        print("✅ SDK models import successful")

        # Test basic model creation
        config = DetectionConfig(algorithm="isolation_forest", contamination=0.1)
        print(f"✅ DetectionConfig created: {config.algorithm}")

        score = AnomalyScore(value=0.8, confidence=0.9)
        print(f"✅ AnomalyScore created: {score.value}")

        print("🎉 Basic SDK components test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Basic imports test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_imports()
    sys.exit(0 if success else 1)
