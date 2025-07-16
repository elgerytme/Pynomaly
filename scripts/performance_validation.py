#!/usr/bin/env python3
"""Performance validation script to ensure reorganization didn't impact performance."""

import importlib
import sys
import time
from pathlib import Path


def test_import_performance():
    """Test import performance for key packages."""
    results = {}

    # Test packages after reorganization
    test_packages = [
        "src.packages.anomaly_detection",
        "src.packages.core",
        "src.packages.interfaces",
        "src.packages.infrastructure",
        "src.packages.services",
    ]

    print("🚀 Testing import performance...")

    for package in test_packages:
        try:
            start_time = time.time()

            # Add src to path temporarily
            sys.path.insert(0, str(Path(__file__).parent.parent))

            # Import the package
            importlib.import_module(package)

            end_time = time.time()
            import_time = (end_time - start_time) * 1000  # Convert to ms
            results[package] = import_time

            status = "✅" if import_time < 100 else "⚠️" if import_time < 500 else "❌"
            print(f"{status} {package}: {import_time:.2f}ms")

        except ImportError as e:
            results[package] = "FAILED"
            print(f"❌ {package}: Import failed - {e}")
        except Exception as e:
            results[package] = "ERROR"
            print(f"💥 {package}: Error - {e}")

    return results


def test_memory_usage():
    """Basic memory usage validation."""
    try:
        import os

        # TODO: fix this import issue
        import psutil

        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024

        print(f"\n💾 Memory usage: {memory_mb:.1f} MB")

        if memory_mb < 100:
            print("✅ Memory usage is optimal")
        elif memory_mb < 200:
            print("⚠️ Memory usage is acceptable")
        else:
            print("❌ Memory usage is high")

        return memory_mb

    except ImportError:
        print("⚠️ psutil not available - skipping memory check")
        return None


def validate_architecture():
    """Validate that domain separation is maintained."""
    print("\n🏗️ Validating architecture...")

    # Check if domain checker runs successfully
    try:
        import subprocess

        result = subprocess.run(
            [sys.executable, "src/packages/simple_domain_check.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        if result.returncode == 0:
            print("✅ Domain boundaries validated")
            return True
        else:
            print("❌ Domain boundary violations detected")
            print(result.stdout)
            return False

    except Exception as e:
        print(f"💥 Architecture validation failed: {e}")
        return False


def main():
    """Main performance validation."""
    print("=" * 60)
    print("🔍 PYNOMALY PERFORMANCE VALIDATION")
    print("=" * 60)

    # Test imports
    import_results = test_import_performance()

    # Test memory
    memory_usage = test_memory_usage()

    # Test architecture
    architecture_valid = validate_architecture()

    # Summary
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)

    # Import performance summary
    successful_imports = sum(1 for v in import_results.values() if isinstance(v, float))
    total_imports = len(import_results)

    print(f"📦 Package Imports: {successful_imports}/{total_imports} successful")

    if successful_imports > 0:
        avg_import_time = (
            sum(v for v in import_results.values() if isinstance(v, float))
            / successful_imports
        )
        print(f"⏱️ Average Import Time: {avg_import_time:.2f}ms")

        if avg_import_time < 50:
            print("✅ Import performance is excellent")
        elif avg_import_time < 100:
            print("✅ Import performance is good")
        else:
            print("⚠️ Import performance could be improved")

    # Memory summary
    if memory_usage:
        memory_status = (
            "excellent"
            if memory_usage < 100
            else "good" if memory_usage < 200 else "high"
        )
        print(f"💾 Memory Usage: {memory_usage:.1f}MB ({memory_status})")

    # Architecture summary
    print(
        f"🏗️ Architecture: {'✅ Valid' if architecture_valid else '❌ Issues detected'}"
    )

    # Overall status
    print("\n" + "=" * 60)
    overall_good = (
        successful_imports >= total_imports * 0.8  # 80% imports successful
        and (memory_usage is None or memory_usage < 200)  # Memory under 200MB
        and architecture_valid  # Architecture valid
    )

    if overall_good:
        print("🎉 OVERALL STATUS: ✅ PERFORMANCE VALIDATED")
        print("The repository reorganization maintains good performance!")
    else:
        print("⚠️ OVERALL STATUS: ❌ PERFORMANCE ISSUES DETECTED")
        print("Some performance concerns need attention.")

    print("=" * 60)

    return 0 if overall_good else 1


if __name__ == "__main__":
    exit(main())
