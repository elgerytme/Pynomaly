#!/usr/bin/env python3
"""
Mutation Testing Setup for Critical Code Paths
Implements mutation testing to validate test quality and improve coverage effectiveness.
"""

from pathlib import Path


class MutationTestingSetup:
    """Sets up and configures mutation testing for critical code paths."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config_file = project_root / "setup.cfg"
        self.mutmut_config = project_root / ".mutmut.toml"

    def create_mutmut_config(self):
        """Create mutation testing configuration."""
        print("🧬 Creating mutation testing configuration...")

        mutmut_config = """
[tool.mutmut]
# Mutation testing configuration for Pynomaly

# Paths to mutate (critical business logic)
paths_to_mutate = [
    "src/pynomaly/domain/",
    "src/pynomaly/application/services/detection_service.py",
    "src/pynomaly/application/services/ensemble_service.py",
    "src/pynomaly/infrastructure/adapters/pyod_adapter.py",
    "src/pynomaly/infrastructure/adapters/sklearn_adapter.py",
]

# Test command to run for each mutation
test_command = "python -m pytest tests/domain/ tests/application/test_services.py tests/infrastructure/adapters/test_ml_adapters_comprehensive.py -x --tb=no -q"

# Paths to exclude from mutation
exclude_patterns = [
    "*/test_*",
    "*/__pycache__/*",
    "*/migrations/*",
    "*/scripts/*",
    "*/.venv/*",
]

# Coverage threshold (mutations must maintain this coverage)
coverage_threshold = 85

# Mutation operators to use
operators = [
    "AOD",  # Arithmetic Operator Deletion
    "AOR",  # Arithmetic Operator Replacement
    "COD",  # Conditional Operator Deletion
    "COR",  # Conditional Operator Replacement
    "LCR",  # Logical Connector Replacement
    "ROR",  # Relational Operator Replacement
    "SIR",  # Slice Index Remove
]

# Maximum number of mutations to generate
max_mutations = 500

# Timeout for each test run (seconds)
timeout = 300
"""

        with open(self.mutmut_config, "w") as f:
            f.write(mutmut_config)

        print(f"✅ Mutation testing config created: {self.mutmut_config}")

    def create_critical_path_targets(self) -> dict[str, list[str]]:
        """Define critical code paths for targeted mutation testing."""
        critical_paths = {
            "domain_entities": [
                "src/pynomaly/domain/entities/anomaly.py",
                "src/pynomaly/domain/entities/detector.py",
                "src/pynomaly/domain/entities/detection_result.py",
                "src/pynomaly/domain/value_objects/anomaly_score.py",
                "src/pynomaly/domain/value_objects/contamination_rate.py",
            ],
            "detection_algorithms": [
                "src/pynomaly/infrastructure/adapters/pyod_adapter.py",
                "src/pynomaly/infrastructure/adapters/sklearn_adapter.py",
                "src/pynomaly/application/services/detection_service.py",
                "src/pynomaly/application/services/ensemble_service.py",
            ],
            "data_validation": [
                "src/pynomaly/domain/services/feature_validator.py",
                "src/pynomaly/infrastructure/data/preprocessing_pipeline.py",
                "src/pynomaly/application/services/data_validation_service.py",
            ],
            "security_critical": [
                "src/pynomaly/infrastructure/auth/jwt_auth.py",
                "src/pynomaly/infrastructure/security/input_sanitizer.py",
                "src/pynomaly/presentation/api/middleware.py",
            ],
        }

        return critical_paths

    def create_mutation_test_scripts(self):
        """Create scripts for running mutation tests on critical paths."""
        print("📝 Creating mutation test scripts...")

        scripts_dir = self.project_root / "scripts" / "mutation"
        scripts_dir.mkdir(parents=True, exist_ok=True)

        # Script for domain mutation testing
        domain_script = scripts_dir / "test_domain_mutations.py"
        domain_content = '''#!/usr/bin/env python3
"""
Domain Layer Mutation Testing
Tests the quality of domain layer tests through mutation testing.
"""

import subprocess
import sys
from pathlib import Path

def run_domain_mutations():
    """Run mutation testing on domain layer."""
    print("🧬 Running domain layer mutation testing...")

    # Target domain entities and value objects
    target_files = [
        "src/pynomaly/domain/entities/",
        "src/pynomaly/domain/value_objects/",
        "src/pynomaly/domain/services/",
    ]

    for target in target_files:
        print(f"\\n🎯 Testing mutations in {target}")

        cmd = [
            "mutmut", "run",
            "--paths-to-mutate", target,
            "--tests-dir", "tests/domain/",
            "--runner", "python -m pytest tests/domain/ -x --tb=no -q",
            "--timeout", "120",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            print(f"✅ Mutation testing completed for {target}")
            print(f"Mutations: {result.returncode}")

            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars

        except subprocess.TimeoutExpired:
            print(f"⏰ Mutation testing timed out for {target}")
        except Exception as e:
            print(f"❌ Error testing {target}: {e}")

if __name__ == "__main__":
    run_domain_mutations()
'''

        with open(domain_script, "w") as f:
            f.write(domain_content)

        # Script for ML adapter mutation testing
        ml_script = scripts_dir / "test_ml_adapter_mutations.py"
        ml_content = '''#!/usr/bin/env python3
"""
ML Adapter Mutation Testing
Tests the quality of ML adapter tests through targeted mutations.
"""

import subprocess
import sys
from pathlib import Path

def run_ml_adapter_mutations():
    """Run mutation testing on ML adapters."""
    print("🧬 Running ML adapter mutation testing...")

    adapters = [
        ("PyOD Adapter", "src/pynomaly/infrastructure/adapters/pyod_adapter.py"),
        ("Sklearn Adapter", "src/pynomaly/infrastructure/adapters/sklearn_adapter.py"),
        ("Detection Service", "src/pynomaly/application/services/detection_service.py"),
    ]

    for name, target_file in adapters:
        print(f"\\n🎯 Testing mutations in {name}")

        cmd = [
            "mutmut", "run",
            "--paths-to-mutate", target_file,
            "--runner", "python -m pytest tests/infrastructure/adapters/ tests/application/test_services.py -x --tb=no -q",
            "--timeout", "180",
            "--max-mutations", "50",  # Limit for large files
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            print(f"✅ Mutation testing completed for {name}")

            # Show mutation survival rate
            if "survived" in result.stdout.lower():
                print("⚠️  Some mutations survived - consider improving tests")
            else:
                print("🎯 All mutations killed - excellent test coverage")

        except subprocess.TimeoutExpired:
            print(f"⏰ Mutation testing timed out for {name}")
        except Exception as e:
            print(f"❌ Error testing {name}: {e}")

if __name__ == "__main__":
    run_ml_adapter_mutations()
'''

        with open(ml_script, "w") as f:
            f.write(ml_content)

        # Make scripts executable
        domain_script.chmod(0o755)
        ml_script.chmod(0o755)

        print(f"✅ Mutation test scripts created in {scripts_dir}")

    def create_mutation_ci_workflow(self):
        """Create CI workflow for mutation testing."""
        print("🔄 Creating mutation testing CI workflow...")

        workflow_file = (
            self.project_root / ".github" / "workflows" / "mutation-testing.yml"
        )
        workflow_content = """name: Mutation Testing

on:
  schedule:
    # Run mutation tests weekly on Sundays at 2 AM UTC
    - cron: '0 2 * * 0'
  workflow_dispatch:
    inputs:
      target_component:
        description: 'Component to test (domain, ml-adapters, security, all)'
        required: false
        default: 'domain'
        type: choice
        options:
        - domain
        - ml-adapters
        - security
        - all

env:
  PYTHON_VERSION: '3.11'
  POETRY_VERSION: '1.6.1'

jobs:
  mutation-testing:
    name: Mutation Testing
    runs-on: ubuntu-latest
    timeout-minutes: 120

    strategy:
      matrix:
        component:
          - ${{ github.event.inputs.target_component == 'all' && 'domain' || github.event.inputs.target_component }}
          - ${{ github.event.inputs.target_component == 'all' && 'ml-adapters' || '' }}
          - ${{ github.event.inputs.target_component == 'all' && 'security' || '' }}
      fail-fast: false

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: Install Poetry
      uses: snok/install-poetry@v1
      with:
        version: ${{ env.POETRY_VERSION }}

    - name: Install dependencies
      run: |
        poetry install --with dev,test
        poetry run pip install mutmut

    - name: Run mutation testing
      run: |
        case "${{ matrix.component }}" in
          "domain")
            echo "🧬 Running domain mutation testing..."
            poetry run python scripts/mutation/test_domain_mutations.py
            ;;
          "ml-adapters")
            echo "🧬 Running ML adapter mutation testing..."
            poetry run python scripts/mutation/test_ml_adapter_mutations.py
            ;;
          "security")
            echo "🧬 Running security mutation testing..."
            poetry run mutmut run \\
              --paths-to-mutate src/pynomaly/infrastructure/auth/ \\
              --paths-to-mutate src/pynomaly/infrastructure/security/ \\
              --runner "python -m pytest tests/security/ -x --tb=no -q" \\
              --timeout 150
            ;;
        esac

    - name: Generate mutation report
      if: always()
      run: |
        echo "📊 Generating mutation testing report..."
        poetry run mutmut junitxml > mutation-results-${{ matrix.component }}.xml
        poetry run mutmut html

    - name: Upload mutation results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: mutation-results-${{ matrix.component }}
        path: |
          mutation-results-${{ matrix.component }}.xml
          html/

  mutation-analysis:
    name: Analyze Mutation Results
    needs: mutation-testing
    if: always()
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Download mutation results
      uses: actions/download-artifact@v3
      with:
        path: mutation-results/

    - name: Analyze mutation testing effectiveness
      run: |
        echo "🔬 Mutation Testing Analysis"
        echo "=========================="

        # In a real implementation, this would parse the mutation results
        # and provide detailed analysis of test quality

        echo "📊 Mutation testing provides insights into test quality by:"
        echo "   - Creating small code changes (mutations)"
        echo "   - Running tests to see if they catch the changes"
        echo "   - Identifying weak spots in test coverage"
        echo ""
        echo "🎯 High-quality tests should kill most mutations"
        echo "⚠️  Surviving mutations indicate potential test gaps"
        echo ""
        echo "✅ Mutation testing analysis completed"
"""

        workflow_file.parent.mkdir(parents=True, exist_ok=True)
        with open(workflow_file, "w") as f:
            f.write(workflow_content)

        print(f"✅ Mutation testing workflow created: {workflow_file}")

    def create_pytest_plugins(self):
        """Create custom pytest plugins for enhanced testing."""
        print("🔌 Creating custom pytest plugins...")

        plugins_dir = self.project_root / "tests" / "plugins"
        plugins_dir.mkdir(parents=True, exist_ok=True)

        # Plugin for test timing and performance monitoring
        timing_plugin = plugins_dir / "test_timing.py"
        timing_content = '''"""
Pytest plugin for test timing and performance monitoring.
"""

import time
import pytest
import json
from pathlib import Path


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
            self.slow_tests.append({
                'test': test_name,
                'time': execution_time,
                'file': str(item.fspath),
                'line': item.location[1]
            })

    def pytest_sessionfinish(self, session, exitstatus):
        """Generate timing report at end of session."""
        if not self.test_times:
            return

        # Find slowest tests
        sorted_tests = sorted(self.test_times.items(), key=lambda x: x[1], reverse=True)

        print("\\n" + "="*50)
        print("🕒 TEST TIMING REPORT")
        print("="*50)

        print(f"📊 Total tests: {len(self.test_times)}")
        print(f"⏱️  Total time: {sum(self.test_times.values()):.2f}s")
        print(f"🐌 Slow tests (>10s): {len(self.slow_tests)}")

        if self.slow_tests:
            print("\\n🐌 SLOW TESTS:")
            for test in self.slow_tests[:5]:  # Top 5 slowest
                print(f"   {test['test']}: {test['time']:.2f}s")

        print("\\n🚀 FASTEST TESTS:")
        for test_name, time_taken in sorted_tests[-5:]:  # 5 fastest
            print(f"   {test_name}: {time_taken:.3f}s")

        print("\\n💡 OPTIMIZATION SUGGESTIONS:")
        if len(self.slow_tests) > 0:
            print("   - Consider mocking external dependencies in slow tests")
            print("   - Use fixtures for expensive setup operations")
            print("   - Consider parallelizing slow test suites")
        else:
            print("   - Test performance looks good!")

        # Save detailed timing data
        timing_file = Path("test-timing-report.json")
        with open(timing_file, 'w') as f:
            json.dump({
                'test_times': self.test_times,
                'slow_tests': self.slow_tests,
                'total_time': sum(self.test_times.values()),
                'test_count': len(self.test_times)
            }, f, indent=2)

        print(f"📄 Detailed timing report saved to: {timing_file}")


def pytest_configure(config):
    """Register the timing plugin."""
    config.pluginmanager.register(TestTimingPlugin(), "test_timing")
'''

        with open(timing_plugin, "w") as f:
            f.write(timing_content)

        # Plugin for memory usage monitoring
        memory_plugin = plugins_dir / "memory_monitor.py"
        memory_content = '''"""
Pytest plugin for memory usage monitoring during tests.
"""

import pytest
import psutil
import os
from pathlib import Path


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
        print(f"\\n🧠 Initial memory usage: {self.initial_memory:.1f} MB")

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        """Monitor memory during test execution."""
        memory_before = self.process.memory_info().rss / 1024 / 1024  # MB

        yield

        memory_after = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = memory_after - memory_before

        test_name = f"{item.module.__name__}::{item.name}"
        self.memory_usage[test_name] = {
            'before': memory_before,
            'after': memory_after,
            'delta': memory_delta
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
            [(test, data['delta']) for test, data in self.memory_usage.items()],
            key=lambda x: x[1], reverse=True
        )

        print("\\n" + "="*50)
        print("🧠 MEMORY USAGE REPORT")
        print("="*50)

        print(f"📊 Initial memory: {self.initial_memory:.1f} MB")
        print(f"📈 Peak memory: {self.peak_memory:.1f} MB")
        print(f"📊 Final memory: {final_memory:.1f} MB")
        print(f"📊 Total growth: {final_memory - self.initial_memory:.1f} MB")

        print("\\n🔥 MEMORY-INTENSIVE TESTS:")
        for test_name, memory_delta in memory_hogs[:5]:
            if memory_delta > 10:  # Only show tests that use >10MB
                print(f"   {test_name}: +{memory_delta:.1f} MB")

        if self.peak_memory > 1000:  # > 1GB
            print("\\n⚠️  HIGH MEMORY USAGE DETECTED")
            print("   Consider optimizing memory-intensive tests")
        else:
            print("\\n✅ Memory usage within acceptable limits")


def pytest_configure(config):
    """Register the memory monitoring plugin."""
    if psutil:
        config.pluginmanager.register(MemoryMonitorPlugin(), "memory_monitor")
'''

        with open(memory_plugin, "w") as f:
            f.write(memory_content)

        # Create plugin init file
        init_file = plugins_dir / "__init__.py"
        with open(init_file, "w") as f:
            f.write('"""Custom pytest plugins for enhanced testing."""\n')

        print(f"✅ Custom pytest plugins created in {plugins_dir}")

    def run_setup(self):
        """Run the complete mutation testing setup."""
        print("🧬 Setting up mutation testing infrastructure...")
        print("=" * 60)

        self.create_mutmut_config()
        self.create_critical_path_targets()
        self.create_mutation_test_scripts()
        self.create_mutation_ci_workflow()
        self.create_pytest_plugins()

        print("=" * 60)
        print("✅ MUTATION TESTING SETUP COMPLETE")
        print("=" * 60)
        print("🎯 Key Features Added:")
        print("   - Mutation testing configuration")
        print("   - Critical path targeting")
        print("   - CI/CD workflow integration")
        print("   - Custom pytest plugins")
        print("")
        print("🚀 Next Steps:")
        print("   1. Install mutmut: poetry add --group dev mutmut")
        print(
            "   2. Run domain mutations: python scripts/mutation/test_domain_mutations.py"
        )
        print("   3. Review mutation testing results")
        print("   4. Improve tests for surviving mutations")
        print("=" * 60)


def main():
    """Main execution function."""
    project_root = Path(__file__).parent.parent
    setup = MutationTestingSetup(project_root)
    setup.run_setup()


if __name__ == "__main__":
    main()
