#!/usr/bin/env python3
"""Comprehensive integration test runner for Pynomaly."""

import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml


class IntegrationTestRunner:
    """Comprehensive integration test runner."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the test runner."""
        self.config_path = config_path or Path(__file__).parent / "config" / "test_config.yaml"
        self.config = self._load_config()
        self.test_results = {}
        
    def _load_config(self) -> Dict:
        """Load test configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "performance": {"max_response_time": 2.0, "concurrent_users": 10},
            "security": {"enable_auth": True},
            "monitoring": {"enabled": True}
        }
    
    def run_test_suite(self, suite: str, args: Optional[List[str]] = None) -> bool:
        """Run a specific test suite."""
        print(f"\\n🧪 Running {suite} test suite...")
        
        base_args = [
            "python", "-m", "pytest",
            f"tests/integration_testing/{suite}/",
            "-v",
            "--tb=short",
            "--durations=10"
        ]
        
        # Add suite-specific arguments
        if suite == "performance":
            base_args.extend([
                "-m", "performance",
                "--timeout=300",
                f"--maxfail=3"
            ])
        elif suite == "security":
            base_args.extend([
                "-m", "security",
                "--strict-markers"
            ])
        elif suite == "end_to_end":
            base_args.extend([
                "-m", "end_to_end",
                "--timeout=600"
            ])
        elif suite == "contracts":
            base_args.extend([
                "-m", "contract",
                "--strict-config"
            ])
        
        if args:
            base_args.extend(args)
        
        start_time = time.time()
        try:
            result = subprocess.run(base_args, capture_output=True, text=True)
            execution_time = time.time() - start_time
            
            self.test_results[suite] = {
                "success": result.returncode == 0,
                "execution_time": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            if result.returncode == 0:
                print(f"✅ {suite} tests passed in {execution_time:.2f}s")
                return True
            else:
                print(f"❌ {suite} tests failed in {execution_time:.2f}s")
                print("STDOUT:", result.stdout[-500:])  # Last 500 chars
                print("STDERR:", result.stderr[-500:])
                return False
                
        except Exception as e:
            print(f"❌ Error running {suite} tests: {e}")
            self.test_results[suite] = {
                "success": False,
                "execution_time": time.time() - start_time,
                "error": str(e)
            }
            return False
    
    def run_all_tests(self, parallel: bool = False) -> bool:
        """Run all integration test suites."""
        print("🚀 Starting comprehensive integration testing...")
        
        test_suites = [
            "end_to_end",
            "performance", 
            "security",
            "multi_tenant",
            "disaster_recovery",
            "contracts"
        ]
        
        if parallel:
            return self._run_parallel_tests(test_suites)
        else:
            return self._run_sequential_tests(test_suites)
    
    def _run_sequential_tests(self, test_suites: List[str]) -> bool:
        """Run test suites sequentially."""
        all_passed = True
        
        for suite in test_suites:
            if not self.run_test_suite(suite):
                all_passed = False
                
        return all_passed
    
    def _run_parallel_tests(self, test_suites: List[str]) -> bool:
        """Run test suites in parallel (where possible)."""
        # Some tests need to run sequentially (e.g., disaster recovery)
        sequential_suites = ["disaster_recovery", "performance"]
        parallel_suites = [s for s in test_suites if s not in sequential_suites]
        
        all_passed = True
        
        # Run parallel suites
        if parallel_suites:
            print(f"Running {len(parallel_suites)} suites in parallel...")
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(self.run_test_suite, suite): suite 
                    for suite in parallel_suites
                }
                
                for future in concurrent.futures.as_completed(futures):
                    suite = futures[future]
                    try:
                        success = future.result()
                        if not success:
                            all_passed = False
                    except Exception as e:
                        print(f"Error in {suite}: {e}")
                        all_passed = False
        
        # Run sequential suites
        for suite in sequential_suites:
            if not self.run_test_suite(suite):
                all_passed = False
                
        return all_passed
    
    def run_smoke_tests(self) -> bool:
        """Run quick smoke tests."""
        print("💨 Running smoke tests...")
        
        smoke_args = [
            "-m", "not slow",
            "--maxfail=5",
            "-x"  # Stop on first failure
        ]
        
        return self.run_test_suite("end_to_end", smoke_args)
    
    def run_performance_benchmarks(self) -> bool:
        """Run performance benchmarks."""
        print("⚡ Running performance benchmarks...")
        
        perf_args = [
            "-m", "performance",
            "--benchmark-only",
            "--benchmark-sort=mean"
        ]
        
        return self.run_test_suite("performance", perf_args)
    
    def run_security_audit(self) -> bool:
        """Run security audit tests."""
        print("🔒 Running security audit...")
        
        security_args = [
            "-m", "security",
            "--strict-markers"
        ]
        
        return self.run_test_suite("security", security_args)
    
    def generate_report(self) -> None:
        """Generate test execution report."""
        print("\\n📊 Test Execution Report")
        print("=" * 50)
        
        total_time = sum(
            result.get("execution_time", 0) 
            for result in self.test_results.values()
        )
        
        passed_suites = sum(
            1 for result in self.test_results.values() 
            if result.get("success", False)
        )
        
        total_suites = len(self.test_results)
        
        print(f"Total Execution Time: {total_time:.2f}s")
        print(f"Test Suites: {passed_suites}/{total_suites} passed")
        print(f"Success Rate: {(passed_suites/total_suites)*100:.1f}%")
        
        print("\\nSuite Details:")
        for suite, result in self.test_results.items():
            status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
            time_str = f"{result.get('execution_time', 0):.2f}s"
            print(f"  {suite:20} {status:8} {time_str:>8}")
        
        # Generate detailed report file
        self._write_detailed_report()
    
    def _write_detailed_report(self) -> None:
        """Write detailed report to file."""
        report_path = Path("test_results") / "integration_test_report.txt"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("Integration Test Execution Report\\n")
            f.write("=" * 50 + "\\n\\n")
            
            for suite, result in self.test_results.items():
                f.write(f"Suite: {suite}\\n")
                f.write(f"Success: {result.get('success', False)}\\n")
                f.write(f"Execution Time: {result.get('execution_time', 0):.2f}s\\n")
                
                if 'stdout' in result:
                    f.write(f"STDOUT:\\n{result['stdout']}\\n")
                
                if 'stderr' in result:
                    f.write(f"STDERR:\\n{result['stderr']}\\n")
                
                f.write("\\n" + "-" * 40 + "\\n\\n")
        
        print(f"\\nDetailed report written to: {report_path}")
    
    def check_prerequisites(self) -> bool:
        """Check if prerequisites are met."""
        print("🔍 Checking prerequisites...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            return False
        
        # Check required packages
        required_packages = [
            "pytest", "pytest-asyncio", "pytest-timeout", 
            "pydantic", "fastapi", "pandas", "numpy"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing packages: {', '.join(missing_packages)}")
            return False
        
        # Check test environment
        if not os.environ.get("PYNOMALY_ENVIRONMENT"):
            os.environ["PYNOMALY_ENVIRONMENT"] = "integration_test"
        
        print("✅ Prerequisites check passed")
        return True
    
    def setup_test_environment(self) -> bool:
        """Set up test environment."""
        print("⚙️  Setting up test environment...")
        
        # Set environment variables
        test_env = {
            "PYNOMALY_ENVIRONMENT": "integration_test",
            "PYNOMALY_DEBUG": "true",
            "PYNOMALY_LOG_LEVEL": "DEBUG",
            "PYTHONPATH": str(Path.cwd() / "src"),
            "PYTHONDONTWRITEBYTECODE": "1"
        }
        
        for key, value in test_env.items():
            os.environ[key] = value
        
        # Create test directories
        test_dirs = [
            "test_results",
            "test_results/coverage",
            "test_results/reports"
        ]
        
        for dir_path in test_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        print("✅ Test environment ready")
        return True
    
    def cleanup_test_environment(self) -> None:
        """Clean up test environment."""
        print("🧹 Cleaning up test environment...")
        
        # Clean up temporary files if configured
        if self.config.get("environment", {}).get("cleanup_after_tests", True):
            import tempfile
            import shutil
            
            temp_dirs = [
                Path(tempfile.gettempdir()) / "pynomaly_test_*"
            ]
            
            for temp_pattern in temp_dirs:
                for temp_path in temp_pattern.parent.glob(temp_pattern.name):
                    if temp_path.is_dir():
                        shutil.rmtree(temp_path, ignore_errors=True)
        
        print("✅ Cleanup completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Pynomaly Integration Test Runner")
    parser.add_argument(
        "command",
        choices=["all", "smoke", "performance", "security", "suite"],
        help="Test command to run"
    )
    parser.add_argument(
        "--suite",
        choices=["end_to_end", "performance", "security", "multi_tenant", "disaster_recovery", "contracts"],
        help="Specific test suite to run (when command=suite)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel where possible"
    )
    parser.add_argument(
        "--config",
        help="Path to test configuration file"
    )
    parser.add_argument(
        "--no-prereq-check",
        action="store_true",
        help="Skip prerequisite checks"
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = IntegrationTestRunner(args.config)
    
    # Check prerequisites
    if not args.no_prereq_check and not runner.check_prerequisites():
        sys.exit(1)
    
    # Set up test environment
    if not runner.setup_test_environment():
        sys.exit(1)
    
    try:
        # Run tests based on command
        success = False
        
        if args.command == "all":
            success = runner.run_all_tests(parallel=args.parallel)
        elif args.command == "smoke":
            success = runner.run_smoke_tests()
        elif args.command == "performance":
            success = runner.run_performance_benchmarks()
        elif args.command == "security":
            success = runner.run_security_audit()
        elif args.command == "suite":
            if not args.suite:
                print("Error: --suite is required when command=suite")
                sys.exit(1)
            success = runner.run_test_suite(args.suite)
        
        # Generate report
        runner.generate_report()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    finally:
        # Clean up
        runner.cleanup_test_environment()


if __name__ == "__main__":
    main()