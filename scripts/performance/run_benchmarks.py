#!/usr/bin/env python3
"""
Performance Benchmarking CLI Utility for CI/CD Pipeline

This script runs performance benchmarks by:
1. Invoking pytest with performance markers
2. Running standalone BenchmarkingService if needed
3. Producing current_results.json for regression analysis

Usage:
    python run_benchmarks.py [--config CONFIG_FILE] [--output OUTPUT_FILE] [--verbose]
"""

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PerformanceBenchmarkRunner:
    """Runs performance benchmarks using pytest and standalone services."""

    def __init__(self, config_path: Path | None = None, verbose: bool = False):
        self.config_path = (
            config_path or Path(__file__).parent / "performance_config.yml"
        )
        self.verbose = verbose
        self.config = self._load_config()

        # Set up paths
        self.project_root = Path(__file__).parent.parent.parent
        self.results_dir = self.project_root / "benchmark_results"
        self.results_dir.mkdir(exist_ok=True)

        if self.verbose:
            logger.setLevel(logging.DEBUG)

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return self._get_default_config()

        if yaml is None:
            logger.warning("PyYAML not available, using default config")
            return self._get_default_config()

        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration if YAML file is not available."""
        return {
            "benchmark_config": {
                "iterations": 3,
                "warmup_iterations": 1,
                "timeout_seconds": 900.0,
                "dataset_sizes": [1000, 5000, 10000],
                "feature_dimensions": [10, 50, 100],
                "contamination_rates": [0.01, 0.05, 0.1],
            },
            "performance_thresholds": {
                "execution_time": {"max_execution_time_seconds": 300.0},
                "memory_usage": {"max_memory_usage_mb": 4096.0},
                "throughput": {"min_throughput_samples_per_second": 10.0},
                "accuracy": {"min_accuracy_score": 0.75},
            },
            "ci_settings": {"save_detailed_results": True, "report_formats": ["json"]},
        }

    def run_pytest_performance_tests(self) -> dict[str, Any]:
        """Run pytest with performance markers."""
        logger.info("Running pytest performance tests...")

        # Create temporary file for pytest JSON output
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            pytest_json_path = f.name

        try:
            # Build pytest command
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                "-m",
                "performance",
                "--tb=short",
                "--disable-warnings",
                f"--json-report={pytest_json_path}",
                str(self.project_root / "tests"),
            ]

            if self.verbose:
                cmd.append("-v")
            else:
                cmd.append("-q")

            # Run pytest
            logger.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=self.config["benchmark_config"]["timeout_seconds"],
            )

            # Parse pytest JSON report
            pytest_results = self._parse_pytest_json_report(pytest_json_path)

            # Log results
            if result.returncode == 0:
                logger.info("Pytest performance tests completed successfully")
                logger.info(f"Tests run: {pytest_results.get('test_count', 0)}")
                logger.info(f"Passed: {pytest_results.get('passed', 0)}")
                logger.info(f"Failed: {pytest_results.get('failed', 0)}")
            else:
                logger.warning(f"Pytest completed with return code {result.returncode}")
                logger.warning(f"STDERR: {result.stderr}")

            if self.verbose:
                logger.debug(f"STDOUT: {result.stdout}")

            return pytest_results

        except subprocess.TimeoutExpired:
            logger.error("Pytest performance tests timed out")
            return {"error": "timeout", "test_count": 0, "passed": 0, "failed": 0}
        except Exception as e:
            logger.error(f"Error running pytest: {e}")
            return {"error": str(e), "test_count": 0, "passed": 0, "failed": 0}
        finally:
            # Clean up temporary file
            try:
                Path(pytest_json_path).unlink()
            except:
                pass

    def _parse_pytest_json_report(self, json_path: str) -> dict[str, Any]:
        """Parse pytest JSON report file."""
        try:
            with open(json_path) as f:
                data = json.load(f)

            # Extract key metrics
            summary = data.get("summary", {})
            tests = data.get("tests", [])

            # Calculate performance metrics from test results
            performance_metrics = []
            for test in tests:
                if test.get("outcome") == "passed":
                    # Extract performance data from test metadata
                    metrics = self._extract_performance_metrics(test)
                    if metrics:
                        performance_metrics.append(metrics)

            return {
                "test_count": summary.get("total", 0),
                "passed": summary.get("passed", 0),
                "failed": summary.get("failed", 0),
                "skipped": summary.get("skipped", 0),
                "duration": summary.get("duration", 0),
                "performance_metrics": performance_metrics,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error parsing pytest JSON report: {e}")
            return {"error": str(e), "test_count": 0, "passed": 0, "failed": 0}

    def _extract_performance_metrics(
        self, test_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Extract performance metrics from test data."""
        try:
            # Extract basic timing information
            duration = test_data.get("duration", 0)

            # Try to extract custom performance metrics from test output
            call_info = test_data.get("call", {})
            longrepr = call_info.get("longrepr", "")

            # Look for performance metrics in captured output
            metrics = {
                "test_name": test_data.get("nodeid", ""),
                "algorithm": self._extract_algorithm_name(test_data.get("nodeid", "")),
                "execution_time_seconds": duration,
                "success": True,
            }

            # Try to extract additional metrics from test output
            if "throughput" in longrepr.lower():
                # Extract throughput if mentioned in output
                import re

                throughput_match = re.search(r"(\d+\.?\d*)\s*samples/second", longrepr)
                if throughput_match:
                    metrics["throughput"] = float(throughput_match.group(1))

            if "memory" in longrepr.lower():
                # Extract memory usage if mentioned
                memory_match = re.search(r"(\d+\.?\d*)\s*MB", longrepr)
                if memory_match:
                    metrics["memory_usage_mb"] = float(memory_match.group(1))

            return metrics

        except Exception as e:
            logger.debug(f"Error extracting performance metrics: {e}")
            return None

    def _extract_algorithm_name(self, test_name: str) -> str:
        """Extract algorithm name from test name."""
        # Simple heuristic to extract algorithm name
        algorithms = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]
        for algo in algorithms:
            if algo.lower() in test_name.lower():
                return algo
        return "Unknown"

    async def run_standalone_benchmarks(self) -> dict[str, Any]:
        """Run standalone benchmarking service."""
        logger.info("Running standalone benchmarking service...")

        try:
            # Import benchmarking service
            sys.path.insert(0, str(self.project_root / "src"))
            from pynomaly.application.services.performance_benchmarking_service import (
                BenchmarkConfig,
                PerformanceBenchmarkingService,
            )

            # Create service
            service = PerformanceBenchmarkingService(self.results_dir)

            # Create benchmark configuration
            config = BenchmarkConfig(
                benchmark_name="CI Performance Benchmark",
                description="Automated performance benchmark for CI/CD pipeline",
                dataset_sizes=self.config["benchmark_config"]["dataset_sizes"],
                feature_dimensions=self.config["benchmark_config"][
                    "feature_dimensions"
                ],
                contamination_rates=self.config["benchmark_config"][
                    "contamination_rates"
                ],
                iterations=self.config["benchmark_config"]["iterations"],
                warmup_iterations=self.config["benchmark_config"]["warmup_iterations"],
                timeout_seconds=self.config["benchmark_config"]["timeout_seconds"],
                enable_memory_profiling=self.config["benchmark_config"].get(
                    "enable_memory_profiling", True
                ),
                enable_cpu_profiling=self.config["benchmark_config"].get(
                    "enable_cpu_profiling", True
                ),
            )

            # Create benchmark suite
            suite_id = await service.create_benchmark_suite(
                suite_name="CI Performance Suite",
                description="Performance benchmarks for CI/CD pipeline",
                config=config,
            )

            # Run benchmarks
            algorithms = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]
            suite = await service.run_comprehensive_benchmark(
                suite_id=suite_id, algorithms=algorithms
            )

            # Convert to serializable format
            results = {
                "suite_id": str(suite.suite_id),
                "suite_name": suite.suite_name,
                "start_time": suite.start_time.isoformat(),
                "end_time": suite.end_time.isoformat() if suite.end_time else None,
                "total_duration_seconds": suite.total_duration_seconds,
                "individual_results": [
                    {
                        "algorithm_name": r.algorithm_name,
                        "dataset_size": r.dataset_size,
                        "feature_dimension": r.feature_dimension,
                        "contamination_rate": r.contamination_rate,
                        "execution_time_seconds": r.execution_time_seconds,
                        "peak_memory_mb": r.peak_memory_mb,
                        "training_throughput": r.training_throughput,
                        "accuracy_score": r.accuracy_score,
                        "precision_score": r.precision_score,
                        "recall_score": r.recall_score,
                        "f1_score": r.f1_score,
                        "success": r.success,
                        "error_message": r.error_message,
                        "timestamp": r.timestamp.isoformat(),
                    }
                    for r in suite.individual_results
                ],
                "summary_stats": suite.summary_stats,
                "comparative_analysis": suite.comparative_analysis,
                "recommendations": suite.recommendations,
                "performance_grade": suite.performance_grade,
            }

            logger.info("Standalone benchmarks completed successfully")
            logger.info(f"Total test runs: {len(suite.individual_results)}")
            logger.info(f"Duration: {suite.total_duration_seconds:.2f} seconds")

            return results

        except ImportError as e:
            logger.error(f"Could not import benchmarking service: {e}")
            return {"error": "import_error", "message": str(e)}
        except Exception as e:
            logger.error(f"Error running standalone benchmarks: {e}")
            return {"error": "runtime_error", "message": str(e)}

    def combine_results(
        self, pytest_results: dict[str, Any], standalone_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Combine pytest and standalone benchmark results."""
        combined = {
            "metadata": {
                "timestamp": time.time(),
                "config_file": str(self.config_path),
                "project_root": str(self.project_root),
                "runner_version": "1.0.0",
            },
            "pytest_results": pytest_results,
            "standalone_results": standalone_results,
            "performance_metrics": {},
        }

        # Extract and combine performance metrics
        all_metrics = []

        # Add pytest metrics
        if "performance_metrics" in pytest_results:
            all_metrics.extend(pytest_results["performance_metrics"])

        # Add standalone metrics
        if "individual_results" in standalone_results:
            all_metrics.extend(standalone_results["individual_results"])

        # Group by algorithm
        by_algorithm = {}
        for metric in all_metrics:
            algorithm = metric.get("algorithm_name", metric.get("algorithm", "Unknown"))
            if algorithm not in by_algorithm:
                by_algorithm[algorithm] = []
            by_algorithm[algorithm].append(metric)

        combined["performance_metrics"] = by_algorithm

        # Calculate summary statistics
        combined["summary"] = {
            "total_tests": len(all_metrics),
            "algorithms_tested": len(by_algorithm),
            "pytest_test_count": pytest_results.get("test_count", 0),
            "standalone_test_count": len(
                standalone_results.get("individual_results", [])
            ),
            "overall_success": (
                pytest_results.get("failed", 0) == 0
                and not pytest_results.get("error")
                and not standalone_results.get("error")
            ),
        }

        return combined

    def save_results(self, results: dict[str, Any], output_path: Path):
        """Save results to JSON file."""
        try:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

    async def run_all_benchmarks(self, output_path: Path):
        """Run all benchmarks and save results."""
        logger.info("Starting performance benchmark run...")

        # Run pytest performance tests
        pytest_results = self.run_pytest_performance_tests()

        # Run standalone benchmarks
        standalone_results = await self.run_standalone_benchmarks()

        # Combine results
        combined_results = self.combine_results(pytest_results, standalone_results)

        # Save results
        self.save_results(combined_results, output_path)

        # Log summary
        summary = combined_results["summary"]
        logger.info("=== Performance Benchmark Summary ===")
        logger.info(f"Total tests: {summary['total_tests']}")
        logger.info(f"Algorithms tested: {summary['algorithms_tested']}")
        logger.info(f"Pytest tests: {summary['pytest_test_count']}")
        logger.info(f"Standalone tests: {summary['standalone_test_count']}")
        logger.info(f"Overall success: {summary['overall_success']}")

        if not summary["overall_success"]:
            logger.warning("Some benchmarks failed - check detailed results")

        logger.info(f"Detailed results saved to: {output_path}")

        return combined_results


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run performance benchmarks for CI/CD pipeline"
    )
    parser.add_argument(
        "--config", type=Path, help="Path to performance configuration YAML file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="current_results.json",
        help="Output file for benchmark results (default: current_results.json)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Create runner
    runner = PerformanceBenchmarkRunner(config_path=args.config, verbose=args.verbose)

    try:
        # Run benchmarks
        results = await runner.run_all_benchmarks(args.output)

        # Exit with success if all benchmarks passed
        if results["summary"]["overall_success"]:
            logger.info("All benchmarks completed successfully")
            sys.exit(0)
        else:
            logger.error("Some benchmarks failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
