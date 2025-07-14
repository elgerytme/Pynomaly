#!/usr/bin/env python3
"""
Test All Python Versions Script
Runs tests across all configured Python versions with comprehensive reporting.
"""

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test result for a single Python version."""

    version: str
    success: bool
    duration: float
    test_count: int
    failures: int
    errors: int
    coverage: float
    output: str
    error_output: str


@dataclass
class MultiVersionTestSummary:
    """Summary of multi-version testing."""

    total_versions: int
    successful_versions: int
    failed_versions: int
    total_duration: float
    results: list[TestResult]
    compatibility_matrix: dict
    recommendations: list[str]


class MultiVersionTester:
    """Runs tests across multiple Python versions."""

    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path.cwd()
        self.environments_dir = self.base_dir / "environments"
        self.reports_dir = self.base_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)

        # Python versions to test
        self.python_versions = ["3.11.4", "3.11.9", "3.12.8", "3.13.1", "3.14.0a3"]

        # Test configurations
        self.test_configs = {
            "basic": {
                "command": ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
                "timeout": 300,
                "description": "Basic unit tests",
            },
            "coverage": {
                "command": [
                    "python",
                    "-m",
                    "pytest",
                    "tests/",
                    "--cov=pynomaly",
                    "--cov-report=xml",
                    "--cov-report=term",
                ],
                "timeout": 600,
                "description": "Tests with coverage",
            },
            "compatibility": {
                "command": ["python", "-c", self._get_compatibility_test_code()],
                "timeout": 60,
                "description": "Python version compatibility checks",
            },
            "performance": {
                "command": ["python", "-c", self._get_performance_test_code()],
                "timeout": 120,
                "description": "Basic performance benchmarks",
            },
        }

    def _get_compatibility_test_code(self) -> str:
        """Get Python code for compatibility testing."""
        return """
import sys
import os
sys.path.insert(0, 'src')

print(f'Testing Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')

# Test basic imports
try:
    import numpy as np
    print(f'✓ NumPy {np.__version__}')
except ImportError as e:
    print(f'✗ NumPy: {e}')
    sys.exit(1)

try:
    import pandas as pd
    print(f'✓ Pandas {pd.__version__}')
except ImportError as e:
    print(f'✗ Pandas: {e}')
    sys.exit(1)

try:
    import sklearn
    print(f'✓ Scikit-learn {sklearn.__version__}')
except ImportError as e:
    print(f'✗ Scikit-learn: {e}')
    sys.exit(1)

# Test async support
if sys.version_info >= (3, 11):
    import asyncio
    async def test_async():
        return 'async works'
    result = asyncio.run(test_async())
    print(f'✓ Async support: {result}')

# Test type annotations
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

@dataclass
class TestClass:
    value: int
    optional: Optional[str] = None

test_obj = TestClass(42, "test")
print(f'✓ Type annotations and dataclasses: {test_obj}')

# Test pathlib
from pathlib import Path
test_path = Path('.')
print(f'✓ Pathlib: {test_path.absolute()}')

print('✓ All compatibility tests passed')
"""

    def _get_performance_test_code(self) -> str:
        """Get Python code for performance testing."""
        return """
import sys
import time
import platform
sys.path.insert(0, 'src')

print(f'Performance testing Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')
print(f'Platform: {platform.platform()}')

# Test basic Python performance
start = time.perf_counter()
result = sum(i*i for i in range(100000))
python_time = time.perf_counter() - start
print(f'Python computation: {python_time:.4f}s')

# Test NumPy performance
try:
    import numpy as np
    start = time.perf_counter()
    data = np.random.rand(5000, 50)
    result = np.dot(data, data.T)
    numpy_time = time.perf_counter() - start
    print(f'NumPy matrix multiplication: {numpy_time:.4f}s')
except ImportError:
    print('NumPy not available for performance testing')

# Test import time
start = time.perf_counter()
try:
    import pandas as pd
    import_time = time.perf_counter() - start
    print(f'Pandas import time: {import_time:.4f}s')
except ImportError:
    print('Pandas not available for import testing')

print('✓ Performance benchmarks completed')
"""

    def find_python_environments(self) -> dict[str, Path]:
        """Find available Python environments."""
        environments = {}

        for version in self.python_versions:
            env_name = f".venv_{version.replace('.', '_')}"
            env_path = self.environments_dir / env_name

            if env_path.exists():
                # Check for Python executable
                python_exe = env_path / "bin" / "python"
                if platform.system() == "Windows":
                    python_exe = env_path / "Scripts" / "python.exe"

                if python_exe.exists():
                    environments[version] = python_exe
                    logger.debug(f"Found Python {version} at {python_exe}")

        return environments

    def run_test_for_version(
        self, version: str, python_exe: Path, test_config: str = "basic"
    ) -> TestResult:
        """Run tests for a specific Python version."""
        logger.info(f"Running {test_config} tests for Python {version}...")
        start_time = time.time()

        config = self.test_configs[test_config]
        command = [str(python_exe)] + config["command"][
            1:
        ]  # Replace 'python' with actual executable

        try:
            result = subprocess.run(
                command,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=config["timeout"],
                env={**os.environ, "PYTHONPATH": str(self.base_dir / "src")},
            )

            duration = time.time() - start_time

            # Parse test results
            test_count = 0
            failures = 0
            errors = 0
            coverage = 0.0

            # Simple parsing for pytest output
            for line in result.stdout.splitlines():
                if "failed" in line.lower() and "passed" in line.lower():
                    # Try to extract test counts
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed" and i > 0:
                            try:
                                test_count += int(parts[i - 1])
                            except ValueError:
                                pass
                        elif part == "failed" and i > 0:
                            try:
                                failures = int(parts[i - 1])
                            except ValueError:
                                pass
                        elif part == "error" and i > 0:
                            try:
                                errors = int(parts[i - 1])
                            except ValueError:
                                pass
                elif "coverage" in line.lower() and "%" in line:
                    # Try to extract coverage percentage
                    for part in line.split():
                        if "%" in part:
                            try:
                                coverage = float(part.replace("%", ""))
                                break
                            except ValueError:
                                pass

            success = result.returncode == 0

            return TestResult(
                version=version,
                success=success,
                duration=duration,
                test_count=test_count,
                failures=failures,
                errors=errors,
                coverage=coverage,
                output=result.stdout,
                error_output=result.stderr,
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            logger.error(f"Tests for Python {version} timed out")

            return TestResult(
                version=version,
                success=False,
                duration=duration,
                test_count=0,
                failures=0,
                errors=1,
                coverage=0.0,
                output="",
                error_output="Test execution timed out",
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Tests for Python {version} failed: {e}")

            return TestResult(
                version=version,
                success=False,
                duration=duration,
                test_count=0,
                failures=0,
                errors=1,
                coverage=0.0,
                output="",
                error_output=str(e),
            )

    def run_all_tests(
        self, test_config: str = "basic", parallel: bool = True, max_workers: int = 3
    ) -> MultiVersionTestSummary:
        """Run tests across all Python versions."""
        logger.info(f"Starting multi-version testing with config: {test_config}")
        start_time = time.time()

        # Find available environments
        environments = self.find_python_environments()

        if not environments:
            logger.error("No Python environments found!")
            return MultiVersionTestSummary(
                total_versions=0,
                successful_versions=0,
                failed_versions=0,
                total_duration=0.0,
                results=[],
                compatibility_matrix={},
                recommendations=["Run setup_multi_python.py to create environments"],
            )

        logger.info(
            f"Found {len(environments)} Python environments: {list(environments.keys())}"
        )

        results = []

        if parallel and len(environments) > 1:
            # Run tests in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self.run_test_for_version, version, python_exe, test_config
                    ): version
                    for version, python_exe in environments.items()
                }

                for future in as_completed(futures):
                    version = futures[future]
                    try:
                        result = future.result()
                        results.append(result)

                        status = "✓" if result.success else "✗"
                        logger.info(
                            f"{status} Python {version}: {result.duration:.1f}s"
                        )

                    except Exception as e:
                        logger.error(f"✗ Python {version} test failed: {e}")
                        results.append(
                            TestResult(
                                version=version,
                                success=False,
                                duration=0.0,
                                test_count=0,
                                failures=0,
                                errors=1,
                                coverage=0.0,
                                output="",
                                error_output=str(e),
                            )
                        )
        else:
            # Run tests sequentially
            for version, python_exe in environments.items():
                result = self.run_test_for_version(version, python_exe, test_config)
                results.append(result)

                status = "✓" if result.success else "✗"
                logger.info(f"{status} Python {version}: {result.duration:.1f}s")

        # Sort results by version
        results.sort(key=lambda r: r.version)

        # Calculate summary
        total_duration = time.time() - start_time
        successful_versions = sum(1 for r in results if r.success)
        failed_versions = len(results) - successful_versions

        # Generate compatibility matrix
        compatibility_matrix = self._generate_compatibility_matrix(results)

        # Generate recommendations
        recommendations = self._generate_recommendations(results)

        summary = MultiVersionTestSummary(
            total_versions=len(results),
            successful_versions=successful_versions,
            failed_versions=failed_versions,
            total_duration=total_duration,
            results=results,
            compatibility_matrix=compatibility_matrix,
            recommendations=recommendations,
        )

        logger.info(f"Multi-version testing completed in {total_duration:.1f}s")
        logger.info(f"Results: {successful_versions} passed, {failed_versions} failed")

        return summary

    def _generate_compatibility_matrix(self, results: list[TestResult]) -> dict:
        """Generate compatibility matrix from test results."""
        matrix = {}

        for result in results:
            version_key = f"python_{result.version.replace('.', '_')}"
            matrix[version_key] = {
                "status": "compatible" if result.success else "incompatible",
                "test_count": result.test_count,
                "coverage": result.coverage,
                "duration": result.duration,
                "issues": result.failures + result.errors,
            }

        return matrix

    def _generate_recommendations(self, results: list[TestResult]) -> list[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        failed_versions = [r for r in results if not r.success]
        if failed_versions:
            recommendations.append(
                f"Fix compatibility issues in Python versions: {', '.join(r.version for r in failed_versions)}"
            )

        low_coverage = [r for r in results if r.coverage > 0 and r.coverage < 80]
        if low_coverage:
            recommendations.append(
                f"Improve test coverage for Python versions: {', '.join(r.version for r in low_coverage)}"
            )

        high_error_count = [r for r in results if r.errors > 0]
        if high_error_count:
            recommendations.append(
                f"Investigate test errors in Python versions: {', '.join(r.version for r in high_error_count)}"
            )

        slow_tests = [r for r in results if r.duration > 300]  # 5 minutes
        if slow_tests:
            recommendations.append(
                f"Optimize test performance for Python versions: {', '.join(r.version for r in slow_tests)}"
            )

        if not recommendations:
            recommendations.append(
                "All Python versions are working well! Consider adding more comprehensive tests."
            )

        return recommendations

    def save_results(self, summary: MultiVersionTestSummary, output_file: Path):
        """Save test results to file."""
        with open(output_file, "w") as f:
            json.dump(asdict(summary), f, indent=2, default=str)

        logger.info(f"Test results saved to {output_file}")

    def print_summary(self, summary: MultiVersionTestSummary):
        """Print human-readable test summary."""
        print("\n=== Multi-Version Python Test Summary ===")
        print(f"Total versions tested: {summary.total_versions}")
        print(f"Successful: {summary.successful_versions}")
        print(f"Failed: {summary.failed_versions}")
        print(f"Total duration: {summary.total_duration:.1f}s")

        print("\n=== Individual Results ===")
        for result in summary.results:
            status = "✓ PASS" if result.success else "✗ FAIL"
            print(
                f"Python {result.version:<8} {status:<8} "
                f"{result.duration:>6.1f}s "
                f"Tests: {result.test_count:>3} "
                f"Failures: {result.failures:>2} "
                f"Errors: {result.errors:>2} "
                f"Coverage: {result.coverage:>5.1f}%"
            )

        print("\n=== Compatibility Matrix ===")
        for version, info in summary.compatibility_matrix.items():
            version_display = version.replace("python_", "").replace("_", ".")
            status = info["status"]
            status_symbol = "✓" if status == "compatible" else "✗"
            print(
                f"  {status_symbol} Python {version_display:<8} {status:<12} "
                f"({info['issues']} issues, {info['coverage']:.1f}% coverage)"
            )

        if summary.recommendations:
            print("\n=== Recommendations ===")
            for i, rec in enumerate(summary.recommendations, 1):
                print(f"{i}. {rec}")


def main():
    """Main entry point for multi-version testing."""
    parser = argparse.ArgumentParser(description="Multi-Version Python Testing")
    parser.add_argument(
        "--config",
        choices=["basic", "coverage", "compatibility", "performance"],
        default="basic",
        help="Test configuration to use",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run tests sequentially instead of parallel",
    )
    parser.add_argument(
        "--max-workers", type=int, default=3, help="Maximum parallel workers"
    )
    parser.add_argument("--output", type=Path, help="Output file for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize tester
    tester = MultiVersionTester()

    try:
        # Run tests
        summary = tester.run_all_tests(
            test_config=args.config,
            parallel=not args.sequential,
            max_workers=args.max_workers,
        )

        # Print summary
        tester.print_summary(summary)

        # Save results
        if args.output:
            tester.save_results(summary, args.output)
        else:
            # Default output file
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = (
                Path("reports") / f"multi_python_test_{args.config}_{timestamp}.json"
            )
            tester.save_results(summary, output_file)

        # Exit with appropriate code
        sys.exit(0 if summary.failed_versions == 0 else 1)

    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Multi-version testing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import os

    main()
