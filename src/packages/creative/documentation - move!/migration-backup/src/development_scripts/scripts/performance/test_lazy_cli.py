#!/usr/bin/env python3
"""
Test script to measure CLI performance improvements with lazy loading.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def measure_cli_startup(use_lazy: bool = True) -> float:
    """Measure CLI startup time with and without lazy loading."""
    env = os.environ.copy()
    env["PYNOMALY_USE_LAZY_CLI"] = "true" if use_lazy else "false"
    env["PYNOMALY_USE_FAST_CLI"] = "true" if use_lazy else "false"

    start_time = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, "-m", "monorepo.presentation.cli.app", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=PROJECT_ROOT,
            env=env,
        )
        end_time = time.perf_counter()

        if result.returncode == 0:
            return end_time - start_time
        else:
            print(f"CLI failed with return code {result.returncode}")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return float("inf")
    except subprocess.TimeoutExpired:
        return 30.0
    except Exception as e:
        print(f"Error measuring CLI startup: {e}")
        return float("inf")


def main():
    """Main testing function."""
    print("🚀 Testing CLI Performance Improvements")
    print("=" * 50)

    # Test baseline (traditional loading)
    print("\n1. Testing traditional loading (baseline)...")
    baseline_time = measure_cli_startup(use_lazy=False)
    print(f"   Traditional loading: {baseline_time:.3f}s")

    # Test lazy loading
    print("\n2. Testing lazy loading...")
    lazy_time = measure_cli_startup(use_lazy=True)
    print(f"   Lazy loading: {lazy_time:.3f}s")

    # Calculate improvement
    if baseline_time != float("inf") and lazy_time != float("inf"):
        improvement = baseline_time - lazy_time
        improvement_percent = (improvement / baseline_time) * 100

        print("\n📊 Performance Results:")
        print(f"   Baseline: {baseline_time:.3f}s")
        print(f"   Optimized: {lazy_time:.3f}s")
        print(f"   Improvement: {improvement:.3f}s ({improvement_percent:.1f}%)")

        if improvement > 0:
            print(f"   🎉 CLI is {improvement_percent:.1f}% faster!")
        else:
            print("   ⚠️  No improvement detected")
    else:
        print("   ❌ Could not measure performance improvements")

    # Test different commands
    print("\n3. Testing specific commands...")
    commands = ["version", "status", "quickstart"]

    for cmd in commands:
        env = os.environ.copy()
        env["PYNOMALY_USE_LAZY_CLI"] = "true"
        env["PYNOMALY_USE_FAST_CLI"] = "true"

        start_time = time.perf_counter()
        try:
            result = subprocess.run(
                [sys.executable, "-m", "monorepo.presentation.cli.app", cmd],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=PROJECT_ROOT,
                env=env,
            )
            end_time = time.perf_counter()

            if result.returncode == 0:
                print(f"   {cmd}: {(end_time - start_time):.3f}s")
            else:
                print(f"   {cmd}: failed (return code {result.returncode})")
        except Exception as e:
            print(f"   {cmd}: error ({e})")


if __name__ == "__main__":
    main()
