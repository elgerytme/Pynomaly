#!/usr/bin/env python3
"""PowerShell environment test script."""

import os
import sys


def test_powershell_env():
    """Test in PowerShell environment."""
    print("🔷 POWERSHELL Environment Test")
    print(f"🐍 Python version: {sys.version}")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🛤️  Python path: {sys.path[:3]}")

    try:
        # Add source to path
        pynomaly_src = "C:\\Users\\andre\\Pynomaly\\src"
        if os.path.exists(pynomaly_src):
            sys.path.insert(0, pynomaly_src)
            print(f"✅ Added Windows path: {pynomaly_src}")
        else:
            # Fallback to WSL path
            sys.path.insert(0, "/mnt/c/Users/andre/Pynomaly/src")
            print("✅ Added WSL path")

        print("✅ POWERSHELL - Package import successful")

        from monorepo.presentation.sdk.models import AnomalyScore, PerformanceMetrics

        print("✅ POWERSHELL - SDK models import successful")

        score = AnomalyScore(value=0.85, confidence=0.92)
        print(f"✅ POWERSHELL - AnomalyScore created: {score.value}")

        metrics = PerformanceMetrics(
            precision=0.88, recall=0.82, f1_score=0.84, accuracy=0.87
        )
        print(f"✅ POWERSHELL - PerformanceMetrics created: F1={metrics.f1_score}")

        print("✅ POWERSHELL - Client import successful")

        print("🎉 POWERSHELL - SDK test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ POWERSHELL - Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_powershell_env()
    sys.exit(0 if success else 1)
