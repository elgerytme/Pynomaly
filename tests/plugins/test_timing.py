"""
Pytest plugin for test timing and performance monitoring.
"""

import json
import time
from pathlib import Path

import pytest


class TestTimingPlugin:
    """Plugin to track test execution times and identify slow tests."""

    def __init__(self):
        self.test_times = {}
        self.slow_tests = []
        self.start_time = None

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        """Track individual test execution time."""
        start = time.time()
        yield
        end = time.time()

        execution_time = end - start
        test_name = f"{item.module.__name__}::{item.name}"

        self.test_times[test_name] = execution_time

        # Flag slow tests (> 10 seconds)
        if execution_time > 10:
            self.slow_tests.append(
                {
                    "test": test_name,
                    "time": execution_time,
                    "file": str(item.fspath),
                    "line": item.location[1],
                }
            )

    def pytest_sessionfinish(self, session, exitstatus):
        """Generate timing report at end of session."""
        if not self.test_times:
            return

        # Find slowest tests
        sorted_tests = sorted(self.test_times.items(), key=lambda x: x[1], reverse=True)

        print("\n" + "=" * 50)
        print("🕒 TEST TIMING REPORT")
        print("=" * 50)

        print(f"📊 Total tests: {len(self.test_times)}")
        print(f"⏱️  Total time: {sum(self.test_times.values()):.2f}s")
        print(f"🐌 Slow tests (>10s): {len(self.slow_tests)}")

        if self.slow_tests:
            print("\n🐌 SLOW TESTS:")
            for test in self.slow_tests[:5]:  # Top 5 slowest
                print(f"   {test['test']}: {test['time']:.2f}s")

        print("\n🚀 FASTEST TESTS:")
        for test_name, time_taken in sorted_tests[-5:]:  # 5 fastest
            print(f"   {test_name}: {time_taken:.3f}s")

        print("\n💡 OPTIMIZATION SUGGESTIONS:")
        if len(self.slow_tests) > 0:
            print("   - Consider mocking external dependencies in slow tests")
            print("   - Use fixtures for expensive setup operations")
            print("   - Consider parallelizing slow test suites")
        else:
            print("   - Test performance looks good!")

        # Save detailed timing data
        timing_file = Path("test-timing-report.json")
        with open(timing_file, "w") as f:
            json.dump(
                {
                    "test_times": self.test_times,
                    "slow_tests": self.slow_tests,
                    "total_time": sum(self.test_times.values()),
                    "test_count": len(self.test_times),
                },
                f,
                indent=2,
            )

        print(f"📄 Detailed timing report saved to: {timing_file}")


def pytest_configure(config):
    """Register the timing plugin."""
    config.pluginmanager.register(TestTimingPlugin(), "test_timing")
