"""
Pytest plugin for memory usage monitoring during tests.
"""

import os

import psutil
import pytest


class MemoryMonitorPlugin:
    """Plugin to monitor memory usage during test execution."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.memory_usage = {}
        self.peak_memory = 0
        self.initial_memory = 0

    def pytest_sessionstart(self, session):
        """Record initial memory usage."""
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        print(f"\n🧠 Initial memory usage: {self.initial_memory:.1f} MB")

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        """Monitor memory during test execution."""
        memory_before = self.process.memory_info().rss / 1024 / 1024  # MB

        yield

        memory_after = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = memory_after - memory_before

        test_name = f"{item.module.__name__}::{item.name}"
        self.memory_usage[test_name] = {
            "before": memory_before,
            "after": memory_after,
            "delta": memory_delta,
        }

        # Track peak memory
        if memory_after > self.peak_memory:
            self.peak_memory = memory_after

    def pytest_sessionfinish(self, session, exitstatus):
        """Generate memory usage report."""
        if not self.memory_usage:
            return

        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        # Find memory-intensive tests
        memory_hogs = sorted(
            [(test, data["delta"]) for test, data in self.memory_usage.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        print("\n" + "=" * 50)
        print("🧠 MEMORY USAGE REPORT")
        print("=" * 50)

        print(f"📊 Initial memory: {self.initial_memory:.1f} MB")
        print(f"📈 Peak memory: {self.peak_memory:.1f} MB")
        print(f"📊 Final memory: {final_memory:.1f} MB")
        print(f"📊 Total growth: {final_memory - self.initial_memory:.1f} MB")

        print("\n🔥 MEMORY-INTENSIVE TESTS:")
        for test_name, memory_delta in memory_hogs[:5]:
            if memory_delta > 10:  # Only show tests that use >10MB
                print(f"   {test_name}: +{memory_delta:.1f} MB")

        if self.peak_memory > 1000:  # > 1GB
            print("\n⚠️  HIGH MEMORY USAGE DETECTED")
            print("   Consider optimizing memory-intensive tests")
        else:
            print("\n✅ Memory usage within acceptable limits")


def pytest_configure(config):
    """Register the memory monitoring plugin."""
    if psutil:
        config.pluginmanager.register(MemoryMonitorPlugin(), "memory_monitor")
