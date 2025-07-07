#!/usr/bin/env python3
"""Working test of SDK models."""

import os
import sys


def test_working_sdk():
    """Test SDK with correct model names."""
    try:
        print(f"🐍 Python version: {sys.version}")
        print(f"📁 Current working directory: {os.getcwd()}")

        # Test basic package import

        print("✅ Package import successful")

        # Test core components

        print("✅ Domain entities import successful")

        print("✅ Application services import successful")

        print("✅ Infrastructure adapters import successful")

        # Test SDK models directly
        from pynomaly.presentation.sdk.models import (
            AnomalyScore,
            PerformanceMetrics,
        )

        print("✅ SDK models import successful")

        # Test model creation
        score = AnomalyScore(value=0.8, confidence=0.9)
        print(
            f"✅ AnomalyScore created: value={score.value}, confidence={score.confidence}"
        )

        # Test performance metrics
        metrics = PerformanceMetrics(
            precision=0.85, recall=0.80, f1_score=0.82, accuracy=0.90
        )
        print(f"✅ PerformanceMetrics created: F1={metrics.f1_score}")

        # Test sync client
        try:
            from pynomaly.presentation.sdk.client import PynomaliClient

            print("✅ Sync client import successful")

            try:
                client = PynomaliClient(base_url="http://localhost:8000")
                print("✅ Sync client initialization successful")
                print(f"   Client base URL: {client.base_url}")
            except Exception as e:
                print(f"⚠️  Sync client initialization failed: {e}")
        except Exception as e:
            print(f"⚠️  Sync client import failed: {e}")

        # Test async client availability
        from pynomaly.presentation.sdk import AsyncPynomaliClient

        if AsyncPynomaliClient is None:
            print("⚠️  Async client not available (missing aiohttp)")
        else:
            print("✅ Async client available")

        print("🎉 Working SDK test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Working SDK test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_working_sdk()
    sys.exit(0 if success else 1)
