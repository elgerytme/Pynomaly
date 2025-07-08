#!/usr/bin/env python3
"""Buck2 CI/CD Performance Monitoring Script.

This script monitors Buck2 build performance in CI/CD environments,
tracking cache effectiveness, build times, and providing alerts for regressions.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class Buck2CIPerformanceMonitor:
    """Monitor Buck2 performance in CI/CD environments."""

    def __init__(self):
        self.root_path = Path.cwd()
        self.performance_history_file = self.root_path / "ci-performance-history.json"
        self.is_ci = os.getenv("CI") == "true"
        self.github_actions = os.getenv("GITHUB_ACTIONS") == "true"

    def get_buck2_command(self) -> str:
        """Find the correct Buck2 command path."""
        buck2_paths = ["buck2", "/mnt/c/Users/andre/buck2.exe", "/usr/local/bin/buck2"]

        for buck2_path in buck2_paths:
            try:
                result = subprocess.run(
                    [buck2_path, "--version"], capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    return buck2_path
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue

        raise RuntimeError("Buck2 not found in any expected location")

    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system performance metrics."""
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": "ci" if self.is_ci else "local",
            "platform": os.uname().sysname if hasattr(os, "uname") else "unknown",
            "python_version": sys.version.split()[0],
        }

        # Add CI-specific metadata
        if self.github_actions:
            metrics.update(
                {
                    "github_runner": os.getenv("RUNNER_OS", "unknown"),
                    "github_workflow": os.getenv("GITHUB_WORKFLOW", "unknown"),
                    "github_ref": os.getenv("GITHUB_REF", "unknown"),
                    "github_sha": (
                        os.getenv("GITHUB_SHA", "unknown")[:8]
                        if os.getenv("GITHUB_SHA")
                        else "unknown"
                    ),
                }
            )

        # System resources (if available)
        try:
            import psutil

            metrics.update(
                {
                    "cpu_count": psutil.cpu_count(),
                    "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                    "disk_free_gb": round(psutil.disk_usage(".").free / (1024**3), 2),
                }
            )
        except ImportError:
            metrics["resource_info"] = "psutil not available"

        return metrics

    def run_performance_benchmark(self, targets: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive Buck2 performance benchmark."""
        if targets is None:
            targets = ["//:validation"]

        print(f"🚀 Starting Buck2 CI/CD performance benchmark...")
        print(f"📋 Targets: {', '.join(targets)}")

        buck2_cmd = self.get_buck2_command()
        results = {
            "system_metrics": self.collect_system_metrics(),
            "buck2_version": None,
            "targets": targets,
            "benchmark_results": {},
            "cache_metrics": {},
            "performance_summary": {},
        }

        try:
            # Get Buck2 version
            version_result = subprocess.run(
                [buck2_cmd, "--version"], capture_output=True, text=True, timeout=10
            )
            results["buck2_version"] = (
                version_result.stdout.strip()
                if version_result.returncode == 0
                else "unknown"
            )

            print(f"🔧 Buck2 version: {results['buck2_version']}")

            # Benchmark each target
            for target in targets:
                print(f"\n📊 Benchmarking target: {target}")
                target_results = self._benchmark_target(buck2_cmd, target)
                results["benchmark_results"][target] = target_results

            # Collect cache metrics
            results["cache_metrics"] = self._collect_cache_metrics()

            # Generate performance summary
            results["performance_summary"] = self._generate_performance_summary(results)

            print(f"\n✅ Performance benchmark completed successfully")

        except Exception as e:
            results["error"] = str(e)
            print(f"❌ Performance benchmark failed: {e}")

        return results

    def _benchmark_target(self, buck2_cmd: str, target: str) -> Dict[str, Any]:
        """Benchmark a specific Buck2 target."""
        print(f"  🧹 Clean build test for {target}...")

        # Clean build
        subprocess.run([buck2_cmd, "clean"], capture_output=True, cwd=self.root_path)

        start_time = time.time()
        clean_result = subprocess.run(
            [buck2_cmd, "build", target],
            capture_output=True,
            text=True,
            cwd=self.root_path,
        )
        clean_duration = time.time() - start_time

        print(f"  ⚡ Cached build test for {target}...")

        # Cached build
        start_time = time.time()
        cached_result = subprocess.run(
            [buck2_cmd, "build", target],
            capture_output=True,
            text=True,
            cwd=self.root_path,
        )
        cached_duration = time.time() - start_time

        # Incremental build test (touch a file and rebuild)
        print(f"  🔄 Incremental build test for {target}...")
        self._create_incremental_change()

        start_time = time.time()
        incremental_result = subprocess.run(
            [buck2_cmd, "build", target],
            capture_output=True,
            text=True,
            cwd=self.root_path,
        )
        incremental_duration = time.time() - start_time

        # Calculate performance metrics
        cache_speedup = clean_duration / cached_duration if cached_duration > 0 else 1.0
        incremental_speedup = (
            clean_duration / incremental_duration if incremental_duration > 0 else 1.0
        )

        results = {
            "clean_build": {
                "duration": clean_duration,
                "success": clean_result.returncode == 0,
                "stdout_lines": len(clean_result.stdout.splitlines()),
                "stderr_lines": len(clean_result.stderr.splitlines()),
            },
            "cached_build": {
                "duration": cached_duration,
                "success": cached_result.returncode == 0,
                "stdout_lines": len(cached_result.stdout.splitlines()),
                "stderr_lines": len(cached_result.stderr.splitlines()),
            },
            "incremental_build": {
                "duration": incremental_duration,
                "success": incremental_result.returncode == 0,
                "stdout_lines": len(incremental_result.stdout.splitlines()),
                "stderr_lines": len(incremental_result.stderr.splitlines()),
            },
            "performance_metrics": {
                "cache_speedup": cache_speedup,
                "incremental_speedup": incremental_speedup,
                "cache_effectiveness_pct": (
                    ((clean_duration - cached_duration) / clean_duration * 100)
                    if clean_duration > 0
                    else 0
                ),
                "incremental_effectiveness_pct": (
                    ((clean_duration - incremental_duration) / clean_duration * 100)
                    if clean_duration > 0
                    else 0
                ),
            },
        }

        print(
            f"    📈 Clean: {clean_duration:.3f}s | Cached: {cached_duration:.3f}s ({cache_speedup:.1f}x)"
        )
        print(
            f"    📈 Incremental: {incremental_duration:.3f}s ({incremental_speedup:.1f}x)"
        )

        return results

    def _create_incremental_change(self):
        """Create a minimal change to test incremental builds."""
        try:
            # Touch the BUCK file to trigger incremental build
            buck_file = self.root_path / "BUCK"
            if buck_file.exists():
                buck_file.touch()
            else:
                # Create a temporary file that Buck2 might track
                temp_file = self.root_path / "temp_incremental_test.txt"
                temp_file.write_text(f"incremental test at {time.time()}")
                # Clean up immediately
                temp_file.unlink()
        except Exception:
            pass  # Incremental test is optional

    def _collect_cache_metrics(self) -> Dict[str, Any]:
        """Collect Buck2 cache metrics."""
        cache_dir = self.root_path / ".buck-cache"
        metrics = {
            "cache_directory_exists": cache_dir.exists(),
            "cache_size_mb": 0,
            "cache_file_count": 0,
            "cache_subdirs": 0,
        }

        if cache_dir.exists():
            try:
                # Calculate cache size and file count
                total_size = 0
                file_count = 0
                subdir_count = 0

                for root, dirs, files in os.walk(cache_dir):
                    subdir_count += len(dirs)
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            total_size += os.path.getsize(file_path)
                            file_count += 1
                        except OSError:
                            pass

                metrics.update(
                    {
                        "cache_size_mb": round(total_size / (1024 * 1024), 2),
                        "cache_file_count": file_count,
                        "cache_subdirs": subdir_count,
                    }
                )

            except Exception as e:
                metrics["cache_error"] = str(e)

        return metrics

    def _generate_performance_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a performance summary from benchmark results."""
        if not results["benchmark_results"]:
            return {"error": "No benchmark results available"}

        # Aggregate metrics across all targets
        total_clean_time = 0
        total_cached_time = 0
        total_incremental_time = 0
        successful_targets = 0
        failed_targets = 0

        cache_speedups = []
        incremental_speedups = []

        for target, target_results in results["benchmark_results"].items():
            if target_results["clean_build"]["success"]:
                successful_targets += 1
                total_clean_time += target_results["clean_build"]["duration"]
                total_cached_time += target_results["cached_build"]["duration"]
                total_incremental_time += target_results["incremental_build"][
                    "duration"
                ]

                metrics = target_results["performance_metrics"]
                cache_speedups.append(metrics["cache_speedup"])
                incremental_speedups.append(metrics["incremental_speedup"])
            else:
                failed_targets += 1

        if successful_targets == 0:
            return {"error": "No successful builds to analyze"}

        avg_cache_speedup = sum(cache_speedups) / len(cache_speedups)
        avg_incremental_speedup = sum(incremental_speedups) / len(incremental_speedups)

        summary = {
            "total_targets": len(results["benchmark_results"]),
            "successful_targets": successful_targets,
            "failed_targets": failed_targets,
            "total_benchmark_time": total_clean_time
            + total_cached_time
            + total_incremental_time,
            "average_performance": {
                "clean_build_time": total_clean_time / successful_targets,
                "cached_build_time": total_cached_time / successful_targets,
                "incremental_build_time": total_incremental_time / successful_targets,
                "cache_speedup": avg_cache_speedup,
                "incremental_speedup": avg_incremental_speedup,
            },
            "performance_grade": self._calculate_performance_grade(
                avg_cache_speedup, avg_incremental_speedup
            ),
            "recommendations": self._generate_recommendations(results),
        }

        return summary

    def _calculate_performance_grade(
        self, cache_speedup: float, incremental_speedup: float
    ) -> str:
        """Calculate a performance grade based on speedups."""
        if cache_speedup >= 10 and incremental_speedup >= 15:
            return "A+ (Excellent)"
        elif cache_speedup >= 5 and incremental_speedup >= 10:
            return "A (Very Good)"
        elif cache_speedup >= 3 and incremental_speedup >= 5:
            return "B (Good)"
        elif cache_speedup >= 2 and incremental_speedup >= 3:
            return "C (Average)"
        else:
            return "D (Needs Improvement)"

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []

        summary = results.get("performance_summary", {})
        cache_metrics = results.get("cache_metrics", {})
        avg_perf = summary.get("average_performance", {})

        # Cache recommendations
        if avg_perf.get("cache_speedup", 0) < 3:
            recommendations.append(
                "Consider enabling remote caching or optimizing cache configuration"
            )

        if cache_metrics.get("cache_size_mb", 0) > 1000:
            recommendations.append(
                "Cache size is large - consider implementing cache cleanup policies"
            )

        if avg_perf.get("incremental_speedup", 0) < 5:
            recommendations.append(
                "Incremental builds could be faster - review dependency graph"
            )

        # Environment recommendations
        if self.is_ci:
            recommendations.append(
                "Running in CI - ensure cache persistence between builds"
            )
            if not cache_metrics.get("cache_directory_exists", False):
                recommendations.append(
                    "Cache directory not found - verify cache setup in CI"
                )

        # Build performance recommendations
        if summary.get("failed_targets", 0) > 0:
            recommendations.append(
                "Some targets failed - review build configuration and dependencies"
            )

        if avg_perf.get("clean_build_time", 0) > 30:
            recommendations.append(
                "Clean builds are slow - consider parallelization and dependency optimization"
            )

        if not recommendations:
            recommendations.append(
                "Performance looks good! Monitor for regressions over time."
            )

        return recommendations

    def save_performance_history(self, results: Dict[str, Any]):
        """Save performance results to history file."""
        history = []

        # Load existing history
        if self.performance_history_file.exists():
            try:
                with open(self.performance_history_file, "r") as f:
                    history = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                history = []

        # Add current results
        history.append(results)

        # Keep only last 50 runs to avoid file growth
        history = history[-50:]

        # Save updated history
        with open(self.performance_history_file, "w") as f:
            json.dump(history, f, indent=2)

        print(f"📊 Performance history saved to {self.performance_history_file}")

    def generate_ci_summary(self, results: Dict[str, Any]) -> str:
        """Generate a CI-friendly summary of performance results."""
        summary = results.get("performance_summary", {})

        if "error" in summary:
            return f"❌ Performance monitoring failed: {summary['error']}"

        avg_perf = summary.get("average_performance", {})
        grade = summary.get("performance_grade", "Unknown")

        ci_summary = f"""
🚀 Buck2 Performance Report

**Overall Grade:** {grade}
**Targets:** {summary.get('successful_targets', 0)}/{summary.get('total_targets', 0)} successful

**Performance Metrics:**
- Cache Speedup: {avg_perf.get('cache_speedup', 0):.1f}x
- Incremental Speedup: {avg_perf.get('incremental_speedup', 0):.1f}x
- Clean Build Time: {avg_perf.get('clean_build_time', 0):.2f}s
- Cached Build Time: {avg_perf.get('cached_build_time', 0):.2f}s

**Cache Status:**
- Size: {results.get('cache_metrics', {}).get('cache_size_mb', 0):.1f} MB
- Files: {results.get('cache_metrics', {}).get('cache_file_count', 0):,}

**Top Recommendations:**
""".strip()

        recommendations = summary.get("recommendations", [])[:3]
        for i, rec in enumerate(recommendations, 1):
            ci_summary += f"\n{i}. {rec}"

        return ci_summary

    def set_github_output(self, results: Dict[str, Any]):
        """Set GitHub Actions outputs for use in workflow."""
        if not self.github_actions:
            return

        github_output = os.getenv("GITHUB_OUTPUT")
        if not github_output:
            return

        summary = results.get("performance_summary", {})
        avg_perf = summary.get("average_performance", {})

        outputs = {
            "cache-speedup": f"{avg_perf.get('cache_speedup', 0):.1f}",
            "incremental-speedup": f"{avg_perf.get('incremental_speedup', 0):.1f}",
            "performance-grade": summary.get("performance_grade", "Unknown"),
            "successful-targets": str(summary.get("successful_targets", 0)),
            "total-targets": str(summary.get("total_targets", 0)),
            "cache-size-mb": f"{results.get('cache_metrics', {}).get('cache_size_mb', 0):.1f}",
        }

        try:
            with open(github_output, "a") as f:
                for key, value in outputs.items():
                    f.write(f"{key}={value}\n")
            print("📤 GitHub Actions outputs set successfully")
        except Exception as e:
            print(f"⚠️ Failed to set GitHub Actions outputs: {e}")


def main():
    """Main entry point for Buck2 CI/CD performance monitoring."""
    parser = argparse.ArgumentParser(description="Monitor Buck2 CI/CD performance")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["//:validation"],
        help="Buck2 targets to benchmark",
    )
    parser.add_argument(
        "--save-history",
        action="store_true",
        help="Save results to performance history",
    )
    parser.add_argument("--output", help="Output file for detailed results (JSON)")
    parser.add_argument(
        "--ci-summary", action="store_true", help="Generate CI-friendly summary"
    )
    parser.add_argument(
        "--github-outputs", action="store_true", help="Set GitHub Actions outputs"
    )

    args = parser.parse_args()

    monitor = Buck2CIPerformanceMonitor()

    try:
        # Run performance benchmark
        results = monitor.run_performance_benchmark(args.targets)

        # Save detailed results
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"📄 Detailed results saved to {args.output}")

        # Save to history
        if args.save_history:
            monitor.save_performance_history(results)

        # Generate CI summary
        if args.ci_summary:
            print("\n" + "=" * 60)
            print(monitor.generate_ci_summary(results))
            print("=" * 60)

        # Set GitHub Actions outputs
        if args.github_outputs:
            monitor.set_github_output(results)

        # Exit with appropriate code
        summary = results.get("performance_summary", {})
        if "error" in results or "error" in summary:
            sys.exit(1)
        elif summary.get("failed_targets", 0) > 0:
            sys.exit(2)  # Some targets failed
        else:
            print("\n✅ Buck2 CI/CD performance monitoring completed successfully")
            sys.exit(0)

    except Exception as e:
        print(f"❌ Performance monitoring failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
