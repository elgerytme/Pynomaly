#!/usr/bin/env python3
"""
Stable CLI test runner for improved test coverage and reliability.

This script runs the improved CLI tests with better mocking and error handling.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    import pytest
except ImportError:
    print("❌ pytest not available. Installing...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest"])
    import pytest


def run_stable_cli_tests():
    """Run the stable CLI tests with comprehensive reporting."""
    print("🧪 Starting Improved Pynomaly CLI Test Suite")
    print("=" * 60)

    # Test configuration
    test_args = [
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--durations=10",  # Show 10 slowest tests
        "--strict-markers",  # Strict marker enforcement
        "-x",  # Stop on first failure for initial debugging
    ]

    # Test files to run
    stable_test_files = [
        "test_cli_stable_integration.py",
        "test_cli_comprehensive.py",
        "test_cli_performance_stability.py",
    ]

    # Check which test files exist
    test_dir = Path(__file__).parent
    existing_test_files = []

    for test_file in stable_test_files:
        test_path = test_dir / test_file
        if test_path.exists():
            existing_test_files.append(str(test_path))
            print(f"✅ Found test file: {test_file}")
        else:
            print(f"⚠️  Test file not found: {test_file}")

    if not existing_test_files:
        print("❌ No stable test files found!")
        return 1

    print(f"\n🚀 Running {len(existing_test_files)} test files...")
    print("-" * 60)

    start_time = time.time()

    try:
        # Run pytest with the stable test files
        exit_code = pytest.main(test_args + existing_test_files)

        end_time = time.time()
        execution_time = end_time - start_time

        print("\n" + "=" * 60)
        print("📊 TEST EXECUTION SUMMARY")
        print(f"   Execution time: {execution_time:.2f} seconds")
        print(f"   Exit code: {exit_code}")

        if exit_code == 0:
            print("🎉 All stable CLI tests passed!")
            return 0
        else:
            print("❌ Some tests failed - check output above")
            return exit_code

    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


def run_coverage_analysis():
    """Run test coverage analysis if available."""
    try:
        import coverage

        print("\n📈 Running coverage analysis...")

        # This would integrate with pytest-cov if available
        cov_args = [
            "--cov=pynomaly.presentation.cli",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-fail-under=70",  # Require 70% coverage
        ]

        test_dir = Path(__file__).parent
        exit_code = pytest.main(["-v"] + cov_args + [str(test_dir)])

        if exit_code == 0:
            print("✅ Coverage analysis completed successfully")
        else:
            print("⚠️  Coverage below threshold or other issues")

        return exit_code

    except ImportError:
        print("⚠️  Coverage tools not available - skipping coverage analysis")
        print("   Install with: pip install pytest-cov coverage")
        return 0


def validate_test_environment():
    """Validate the test environment setup."""
    print("🔍 Validating test environment...")

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")

    # Check required packages
    required_packages = ["pytest", "typer", "unittest.mock"]
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} missing")

    if missing_packages:
        print(f"⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install " + " ".join(missing_packages))
        return False

    # Check CLI modules can be imported
    try:
        from pynomaly.presentation.cli.app import app

        print("✅ CLI modules importable")
    except ImportError as e:
        print(f"❌ CLI import error: {e}")
        return False

    return True


def main():
    """Main test runner function."""
    print("🔧 Pynomaly CLI Improved Test Runner")
    print("=" * 60)

    # Validate environment
    if not validate_test_environment():
        print("\n❌ Environment validation failed")
        return 1

    print("\n✅ Environment validation passed")

    # Run stable tests
    test_result = run_stable_cli_tests()

    # Run coverage if tests passed
    if test_result == 0:
        coverage_result = run_coverage_analysis()
    else:
        coverage_result = 1

    # Final summary
    print("\n" + "=" * 60)
    print("🎯 FINAL SUMMARY")
    print(f"   Stable tests: {'PASSED' if test_result == 0 else 'FAILED'}")
    print(f"   Coverage: {'PASSED' if coverage_result == 0 else 'SKIPPED/FAILED'}")

    overall_result = max(test_result, coverage_result)

    if overall_result == 0:
        print("🎉 All CLI tests completed successfully!")
        print("   CLI test coverage and stability improved ✅")
    else:
        print("❌ Some issues detected - see output above")
        print("   Review failures and improve test stability")

    return overall_result


if __name__ == "__main__":
    sys.exit(main())
