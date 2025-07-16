#!/usr/bin/env python3
"""
CLI Startup Performance Profiler

Measures and analyzes CLI startup performance to identify bottlenecks.
"""

import cProfile
import importlib.util
import pstats
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class CLIStartupProfiler:
    """Profile CLI startup performance."""

    def __init__(self):
        self.results: dict[str, float] = {}
        self.baseline_time: float = 0

    def measure_import_time(self, module_name: str) -> float:
        """Measure time to import a specific module."""
        start_time = time.perf_counter()
        try:
            importlib.import_module(module_name)
            end_time = time.perf_counter()
            return end_time - start_time
        except ImportError as e:
            print(f"Warning: Could not import {module_name}: {e}")
            return 0.0

    def profile_cli_imports(self) -> dict[str, float]:
        """Profile import times for CLI modules."""
        cli_modules = [
            "monorepo.presentation.cli.app",
            "monorepo.presentation.cli.autonomous",
            "monorepo.presentation.cli.automl",
            "monorepo.presentation.cli.datasets",
            "monorepo.presentation.cli.deep_learning",
            "monorepo.presentation.cli.detection",
            "monorepo.presentation.cli.detectors",
            "monorepo.presentation.cli.explainability",
            "monorepo.presentation.cli.preprocessing",
            "monorepo.presentation.cli.selection",
            "monorepo.presentation.cli.server",
            "monorepo.presentation.cli.tdd",
            "monorepo.presentation.cli.validation",
        ]

        import_times = {}
        for module in cli_modules:
            import_times[module] = self.measure_import_time(module)

        return import_times

    def profile_container_creation(self) -> float:
        """Profile container creation time."""
        start_time = time.perf_counter()
        try:
            from monorepo.infrastructure.config.container import create_container

            container = create_container()
            end_time = time.perf_counter()
            return end_time - start_time
        except Exception as e:
            print(f"Warning: Could not create container: {e}")
            return 0.0

    def profile_cli_container(self) -> float:
        """Profile CLI container creation time."""
        start_time = time.perf_counter()
        try:
            from monorepo.presentation.cli.container import get_cli_container

            container = get_cli_container()
            end_time = time.perf_counter()
            return end_time - start_time
        except Exception as e:
            print(f"Warning: Could not create CLI container: {e}")
            return 0.0

    def measure_cli_command_startup(self, command: str = "--help") -> float:
        """Measure time for CLI command to start and return."""
        start_time = time.perf_counter()
        try:
            # Run CLI command in subprocess to get real startup time
            result = subprocess.run(
                [sys.executable, "-m", "monorepo.presentation.cli.app", command],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=PROJECT_ROOT,
            )
            end_time = time.perf_counter()
            return end_time - start_time
        except subprocess.TimeoutExpired:
            print(f"Warning: CLI command '{command}' timed out")
            return 30.0
        except Exception as e:
            print(f"Warning: Could not run CLI command '{command}': {e}")
            return 0.0

    def profile_heavy_services(self) -> dict[str, float]:
        """Profile heavy service imports."""
        services = {
            "deep_learning_service": "monorepo.application.services.deep_learning_integration_service",
            "automl_service": "monorepo.application.services.automl_service",
            "explainability_service": "monorepo.application.services.explainable_ai_service",
            "training_service": "monorepo.application.services.training_service",
            "autonomous_service": "monorepo.application.services.autonomous_service",
        }

        service_times = {}
        for name, module in services.items():
            service_times[name] = self.measure_import_time(module)

        return service_times

    def run_detailed_profiling(self) -> None:
        """Run detailed profiling with cProfile."""
        print("🔍 Running detailed profiling...")

        # Profile CLI app import
        pr = cProfile.Profile()
        pr.enable()

        try:
            from monorepo.presentation.cli.container import get_cli_container

            container = get_cli_container()
        except Exception as e:
            print(f"Error during profiling: {e}")

        pr.disable()

        # Save profile stats
        stats_file = (
            PROJECT_ROOT / "reports" / "performance" / "cli_startup_profile.stats"
        )
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        pr.dump_stats(str(stats_file))

        # Generate readable report
        stats = pstats.Stats(pr)
        stats.sort_stats("cumulative")

        report_file = (
            PROJECT_ROOT / "reports" / "performance" / "cli_startup_profile.txt"
        )
        with open(report_file, "w") as f:
            stats.print_stats(50, file=f)

        print(f"✅ Detailed profile saved to {report_file}")

        # Show top bottlenecks
        print("\n📊 Top 10 Performance Bottlenecks:")
        stats.print_stats(10)

    def run_comprehensive_analysis(self) -> None:
        """Run comprehensive startup performance analysis."""
        print("🚀 Pynomaly CLI Startup Performance Analysis")
        print("=" * 50)

        # Measure baseline CLI startup
        print("\n1. Measuring baseline CLI startup time...")
        baseline_time = self.measure_cli_command_startup("--help")
        print(f"   Baseline CLI startup: {baseline_time:.3f}s")

        # Profile CLI imports
        print("\n2. Profiling CLI module imports...")
        import_times = self.profile_cli_imports()
        total_import_time = sum(import_times.values())
        print(f"   Total CLI import time: {total_import_time:.3f}s")

        # Show slowest imports
        sorted_imports = sorted(import_times.items(), key=lambda x: x[1], reverse=True)
        print("   Slowest imports:")
        for module, time_taken in sorted_imports[:5]:
            print(f"     {module}: {time_taken:.3f}s")

        # Profile container creation
        print("\n3. Profiling container creation...")
        container_time = self.profile_container_creation()
        print(f"   Container creation: {container_time:.3f}s")

        cli_container_time = self.profile_cli_container()
        print(f"   CLI container creation: {cli_container_time:.3f}s")

        # Profile heavy services
        print("\n4. Profiling heavy services...")
        service_times = self.profile_heavy_services()
        total_service_time = sum(service_times.values())
        print(f"   Total service import time: {total_service_time:.3f}s")

        # Show slowest services
        sorted_services = sorted(
            service_times.items(), key=lambda x: x[1], reverse=True
        )
        print("   Slowest services:")
        for service, time_taken in sorted_services:
            print(f"     {service}: {time_taken:.3f}s")

        # Summary
        print("\n📋 Performance Summary:")
        print(f"   Total CLI startup time: {baseline_time:.3f}s")
        print(
            f"   CLI imports: {total_import_time:.3f}s ({(total_import_time/baseline_time)*100:.1f}%)"
        )
        print(
            f"   Container creation: {container_time:.3f}s ({(container_time/baseline_time)*100:.1f}%)"
        )
        print(
            f"   Service imports: {total_service_time:.3f}s ({(total_service_time/baseline_time)*100:.1f}%)"
        )

        # Optimization recommendations
        print("\n🎯 Optimization Recommendations:")

        if container_time > 0.5:
            print("   🔴 HIGH: Container creation is slow - implement lazy loading")

        if total_import_time > 0.3:
            print("   🔴 HIGH: CLI imports are slow - defer heavy imports")

        if total_service_time > 0.4:
            print("   🟡 MEDIUM: Service imports are slow - use lazy initialization")

        if baseline_time > 2.0:
            print("   🔴 HIGH: Overall startup is very slow - needs optimization")
        elif baseline_time > 1.0:
            print("   🟡 MEDIUM: Startup time could be improved")
        else:
            print("   🟢 GOOD: Startup time is acceptable")

    def save_results(self) -> None:
        """Save profiling results to file."""
        results_file = (
            PROJECT_ROOT / "reports" / "performance" / "cli_startup_results.json"
        )
        results_file.parent.mkdir(parents=True, exist_ok=True)

        import json
        from datetime import datetime

        results = {
            "timestamp": datetime.now().isoformat(),
            "measurements": self.results,
            "baseline_time": self.baseline_time,
            "recommendations": [
                "Implement lazy loading for OptionalServiceManager",
                "Cache container instances",
                "Defer heavy imports in CLI modules",
                "Skip database operations for CLI",
            ],
        }

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"✅ Results saved to {results_file}")


def main():
    """Main profiling function."""
    profiler = CLIStartupProfiler()

    # Run comprehensive analysis
    profiler.run_comprehensive_analysis()

    # Run detailed profiling
    profiler.run_detailed_profiling()

    # Save results
    profiler.save_results()

    print("\n🎉 Performance profiling complete!")
    print("📁 Check reports/performance/ for detailed results")


if __name__ == "__main__":
    main()
