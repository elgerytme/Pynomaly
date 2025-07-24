"""Comprehensive test runner and validation script for the anomaly detection system."""

import pytest
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import subprocess
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tests.utils.test_factories import (
    TestScenarioBuilder, 
    TestDataValidator,
    create_simple_dataset,
    create_large_dataset
)


@dataclass
class TestResult:
    """Test execution result."""
    name: str
    passed: bool
    duration_seconds: float
    error_message: Optional[str] = None
    coverage_percentage: Optional[float] = None
    performance_metrics: Optional[Dict[str, Any]] = None


@dataclass
class TestSuiteResults:
    """Complete test suite results."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    total_duration_seconds: float
    coverage_percentage: float
    test_results: List[TestResult]
    system_validation_passed: bool
    performance_validation_passed: bool
    timestamp: str


class ComprehensiveTestRunner:
    """Comprehensive test runner with validation and reporting."""
    
    def __init__(self, project_root: Path):
        """Initialize test runner.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.test_dir = project_root / "tests"
        self.src_dir = project_root / "src"
        self.reports_dir = project_root / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Test configuration
        self.test_config = {
            "coverage_threshold": 85.0,
            "performance_threshold_ms": 5000,
            "memory_threshold_mb": 1000,
            "timeout_seconds": 300
        }
    
    def run_all_tests(self) -> TestSuiteResults:
        """Run all tests and return comprehensive results."""
        print("🚀 Starting comprehensive test suite validation...")
        start_time = time.time()
        
        results = []
        
        # 1. Run unit tests
        print("\n📋 Running unit tests...")
        unit_result = self._run_pytest_category("unit")
        results.append(unit_result)
        
        # 2. Run integration tests
        print("\n🔗 Running integration tests...")
        integration_result = self._run_pytest_category("integration")
        results.append(integration_result)
        
        # 3. Run end-to-end tests
        print("\n🎯 Running end-to-end tests...")
        e2e_result = self._run_pytest_category("e2e")
        results.append(e2e_result)
        
        # 4. Run system validation tests
        print("\n✅ Running system validation...")
        system_result = self._run_system_validation()
        results.append(system_result)
        
        # 5. Run performance tests
        print("\n⚡ Running performance validation...")
        performance_result = self._run_performance_validation()
        results.append(performance_result)
        
        # 6. Generate coverage report
        print("\n📊 Generating coverage report...")
        coverage_result = self._generate_coverage_report()
        
        # Calculate overall results
        total_duration = time.time() - start_time
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = len(results) - passed_tests
        
        suite_results = TestSuiteResults(
            total_tests=len(results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=0,
            total_duration_seconds=total_duration,
            coverage_percentage=coverage_result.coverage_percentage or 0.0,
            test_results=results,
            system_validation_passed=system_result.passed,
            performance_validation_passed=performance_result.passed,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Generate reports
        self._generate_test_report(suite_results)
        self._generate_junit_report(suite_results)
        
        return suite_results
    
    def _run_pytest_category(self, category: str) -> TestResult:
        """Run pytest for a specific category."""
        start_time = time.time()
        
        # Build pytest command
        test_path = self.test_dir / category
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_path),
            "--tb=short",
            "--verbose",
            f"--cov={self.src_dir}",
            "--cov-report=term-missing",
            "--cov-report=xml",
            "--cov-report=html",
            f"--timeout={self.test_config['timeout_seconds']}",
            "--junit-xml=" + str(self.reports_dir / f"junit_{category}.xml")
        ]
        
        try:
            # Run pytest
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=self.test_config["timeout_seconds"]
            )
            
            duration = time.time() - start_time
            
            # Parse results
            passed = result.returncode == 0
            error_message = result.stderr if not passed else None
            
            # Extract coverage from output if available
            coverage_percentage = self._extract_coverage_from_output(result.stdout)
            
            return TestResult(
                name=f"{category}_tests",
                passed=passed,
                duration_seconds=duration,
                error_message=error_message,
                coverage_percentage=coverage_percentage
            )
            
        except subprocess.TimeoutExpired:
            return TestResult(
                name=f"{category}_tests",
                passed=False,
                duration_seconds=self.test_config["timeout_seconds"],
                error_message=f"Tests timed out after {self.test_config['timeout_seconds']} seconds"
            )
        except Exception as e:
            return TestResult(
                name=f"{category}_tests",
                passed=False,
                duration_seconds=time.time() - start_time,
                error_message=f"Test execution failed: {str(e)}"
            )
    
    def _run_system_validation(self) -> TestResult:
        """Run comprehensive system validation."""
        start_time = time.time()
        
        try:
            # Import and run system validation
            from anomaly_detection.domain.services.detection_service import DetectionService
            from anomaly_detection.domain.services.ensemble_service import EnsembleService
            from anomaly_detection.infrastructure.validation.comprehensive_validators import ComprehensiveValidator
            
            validation_passed = True
            errors = []
            
            # Test 1: Basic detection service functionality
            try:
                detection_service = DetectionService()
                data, _ = create_simple_dataset(100, 5)
                
                result = detection_service.detect_anomalies(
                    data=data,
                    algorithm="iforest",
                    contamination=0.1
                )
                
                if not result.success:
                    validation_passed = False
                    errors.append("Basic detection service failed")
                    
            except Exception as e:
                validation_passed = False
                errors.append(f"Detection service error: {str(e)}")
            
            # Test 2: Ensemble service functionality
            try:
                ensemble_service = EnsembleService()
                data, _ = create_simple_dataset(200, 4)
                
                result = ensemble_service.detect_ensemble(
                    data=data,
                    algorithms=["iforest", "lof"],
                    method="majority",
                    contamination=0.1
                )
                
                if not result.success:
                    validation_passed = False
                    errors.append("Ensemble service failed")
                    
            except Exception as e:
                validation_passed = False
                errors.append(f"Ensemble service error: {str(e)}")
            
            # Test 3: Validation system
            try:
                validator = ComprehensiveValidator()
                
                validation_result = validator.validate_detection_request(
                    data=data,
                    algorithm="iforest",
                    contamination=0.1
                )
                
                if not validation_result.is_valid:
                    validation_passed = False
                    errors.append("Validation system failed")
                    
            except Exception as e:
                validation_passed = False
                errors.append(f"Validation system error: {str(e)}")
            
            duration = time.time() - start_time
            
            return TestResult(
                name="system_validation",
                passed=validation_passed,
                duration_seconds=duration,
                error_message="; ".join(errors) if errors else None
            )
            
        except Exception as e:
            return TestResult(
                name="system_validation",
                passed=False,
                duration_seconds=time.time() - start_time,
                error_message=f"System validation failed: {str(e)}"
            )
    
    def _run_performance_validation(self) -> TestResult:
        """Run performance validation tests."""
        start_time = time.time()
        
        try:
            from anomaly_detection.domain.services.detection_service import DetectionService
            
            performance_metrics = {}
            performance_passed = True
            errors = []
            
            # Test different dataset sizes
            test_sizes = [100, 1000, 5000]
            
            for size in test_sizes:
                data, _ = create_simple_dataset(size, 5)
                detection_service = DetectionService()
                
                # Measure detection time
                test_start = time.time()
                result = detection_service.detect_anomalies(
                    data=data,
                    algorithm="iforest",
                    contamination=0.1
                )
                test_duration = (time.time() - test_start) * 1000  # ms
                
                performance_metrics[f"detection_time_ms_{size}"] = test_duration
                
                # Check performance threshold
                if test_duration > self.test_config["performance_threshold_ms"]:
                    performance_passed = False
                    errors.append(f"Detection with {size} samples took {test_duration:.1f}ms (threshold: {self.test_config['performance_threshold_ms']}ms)")
                
                if not result.success:
                    performance_passed = False
                    errors.append(f"Detection failed for dataset size {size}")
            
            # Test memory usage (if psutil is available)
            try:
                import psutil
                import os
                
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024
                performance_metrics["memory_usage_mb"] = memory_mb
                
                if memory_mb > self.test_config["memory_threshold_mb"]:
                    performance_passed = False
                    errors.append(f"Memory usage {memory_mb:.1f}MB exceeds threshold {self.test_config['memory_threshold_mb']}MB")
                    
            except ImportError:
                performance_metrics["memory_usage_mb"] = None
            
            duration = time.time() - start_time
            
            return TestResult(
                name="performance_validation",
                passed=performance_passed,
                duration_seconds=duration,
                error_message="; ".join(errors) if errors else None,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            return TestResult(
                name="performance_validation",
                passed=False,
                duration_seconds=time.time() - start_time,
                error_message=f"Performance validation failed: {str(e)}"
            )
    
    def _generate_coverage_report(self) -> TestResult:
        """Generate comprehensive coverage report."""
        start_time = time.time()
        
        try:
            # Run coverage report generation
            cmd = [
                sys.executable, "-m", "coverage", "report",
                "--show-missing",
                "--format=json"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0 and result.stdout:
                try:
                    coverage_data = json.loads(result.stdout)
                    coverage_percentage = coverage_data.get("totals", {}).get("percent_covered", 0.0)
                except json.JSONDecodeError:
                    coverage_percentage = 0.0
            else:
                coverage_percentage = 0.0
            
            # Check coverage threshold
            passed = coverage_percentage >= self.test_config["coverage_threshold"]
            error_message = None if passed else f"Coverage {coverage_percentage:.1f}% below threshold {self.test_config['coverage_threshold']}%"
            
            duration = time.time() - start_time
            
            return TestResult(
                name="coverage_report",
                passed=passed,
                duration_seconds=duration,
                error_message=error_message,
                coverage_percentage=coverage_percentage
            )
            
        except Exception as e:
            return TestResult(
                name="coverage_report",
                passed=False,
                duration_seconds=time.time() - start_time,
                error_message=f"Coverage report generation failed: {str(e)}"
            )
    
    def _extract_coverage_from_output(self, output: str) -> Optional[float]:
        """Extract coverage percentage from pytest output."""
        lines = output.split('\n')
        for line in lines:
            if 'TOTAL' in line and '%' in line:
                try:
                    # Extract percentage from line like "TOTAL    1234    567    54%"
                    parts = line.split()
                    for part in parts:
                        if part.endswith('%'):
                            return float(part[:-1])
                except (ValueError, IndexError):
                    continue
        return None
    
    def _generate_test_report(self, results: TestSuiteResults) -> None:
        """Generate comprehensive test report."""
        report_path = self.reports_dir / "test_report.json"
        
        with open(report_path, 'w') as f:
            json.dump(asdict(results), f, indent=2, default=str)
        
        # Generate human-readable report
        html_report_path = self.reports_dir / "test_report.html"
        self._generate_html_report(results, html_report_path)
        
        print(f"\n📄 Test reports generated:")
        print(f"  JSON: {report_path}")
        print(f"  HTML: {html_report_path}")
    
    def _generate_html_report(self, results: TestSuiteResults, output_path: Path) -> None:
        """Generate HTML test report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Anomaly Detection Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
                .metric {{ background: white; padding: 15px; border: 1px solid #ddd; border-radius: 5px; text-align: center; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .test-result {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ddd; }}
                .test-passed {{ border-left-color: green; }}
                .test-failed {{ border-left-color: red; }}
                .error {{ background: #ffe6e6; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Anomaly Detection System - Test Report</h1>
                <p>Generated: {results.timestamp}</p>
                <p>Total Duration: {results.total_duration_seconds:.2f} seconds</p>
            </div>
            
            <div class="summary">
                <div class="metric">
                    <h3>Total Tests</h3>
                    <div>{results.total_tests}</div>
                </div>
                <div class="metric">
                    <h3 class="passed">Passed</h3>
                    <div>{results.passed_tests}</div>
                </div>
                <div class="metric">
                    <h3 class="failed">Failed</h3>
                    <div>{results.failed_tests}</div>
                </div>
                <div class="metric">
                    <h3>Coverage</h3>
                    <div>{results.coverage_percentage:.1f}%</div>
                </div>
            </div>
            
            <h2>Test Results</h2>
        """
        
        for test_result in results.test_results:
            status_class = "test-passed" if test_result.passed else "test-failed"
            status_text = "✅ PASSED" if test_result.passed else "❌ FAILED"
            
            html_content += f"""
            <div class="test-result {status_class}">
                <h3>{test_result.name} - {status_text}</h3>
                <p>Duration: {test_result.duration_seconds:.2f} seconds</p>
                {f'<p>Coverage: {test_result.coverage_percentage:.1f}%</p>' if test_result.coverage_percentage else ''}
                {f'<div class="error"><strong>Error:</strong> {test_result.error_message}</div>' if test_result.error_message else ''}
                {f'<p><strong>Performance Metrics:</strong> {test_result.performance_metrics}</p>' if test_result.performance_metrics else ''}
            </div>
            """
        
        html_content += """
            </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _generate_junit_report(self, results: TestSuiteResults) -> None:
        """Generate JUnit XML report for CI/CD integration."""
        junit_path = self.reports_dir / "junit_summary.xml"
        
        junit_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="anomaly_detection" tests="{results.total_tests}" failures="{results.failed_tests}" time="{results.total_duration_seconds:.3f}">
    <testsuite name="comprehensive_validation" tests="{results.total_tests}" failures="{results.failed_tests}" time="{results.total_duration_seconds:.3f}">
"""
        
        for test_result in results.test_results:
            junit_content += f'        <testcase name="{test_result.name}" time="{test_result.duration_seconds:.3f}"'
            
            if test_result.passed:
                junit_content += ' />\n'
            else:
                junit_content += '>\n'
                junit_content += f'            <failure message="{test_result.error_message or "Unknown error"}" />\n'
                junit_content += '        </testcase>\n'
        
        junit_content += """    </testsuite>
</testsuites>"""
        
        with open(junit_path, 'w') as f:
            f.write(junit_content)
    
    def print_summary(self, results: TestSuiteResults) -> None:
        """Print test summary to console."""
        print("\n" + "="*60)
        print("🎯 ANOMALY DETECTION TEST SUITE SUMMARY")
        print("="*60)
        
        # Overall status
        if results.failed_tests == 0:
            print("✅ ALL TESTS PASSED")
        else:
            print(f"❌ {results.failed_tests} TESTS FAILED")
        
        print(f"\n📊 Results:")
        print(f"  Total Tests: {results.total_tests}")
        print(f"  Passed: {results.passed_tests}")
        print(f"  Failed: {results.failed_tests}")
        print(f"  Duration: {results.total_duration_seconds:.2f}s")
        print(f"  Coverage: {results.coverage_percentage:.1f}%")
        
        print(f"\n🔍 Validation Status:")
        print(f"  System Validation: {'✅ PASSED' if results.system_validation_passed else '❌ FAILED'}")
        print(f"  Performance Validation: {'✅ PASSED' if results.performance_validation_passed else '❌ FAILED'}")
        
        # Failed tests details
        if results.failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for test_result in results.test_results:
                if not test_result.passed:
                    print(f"  - {test_result.name}: {test_result.error_message}")
        
        print("\n" + "="*60)


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description="Comprehensive anomaly detection test runner")
    parser.add_argument("--coverage-threshold", type=float, default=85.0, help="Coverage threshold percentage")
    parser.add_argument("--performance-threshold", type=int, default=5000, help="Performance threshold in milliseconds")
    parser.add_argument("--timeout", type=int, default=300, help="Test timeout in seconds")
    parser.add_argument("--reports-dir", type=str, help="Custom reports directory")
    
    args = parser.parse_args()
    
    # Determine project root
    project_root = Path(__file__).parent.parent
    
    # Initialize test runner
    runner = ComprehensiveTestRunner(project_root)
    
    # Update configuration
    runner.test_config.update({
        "coverage_threshold": args.coverage_threshold,
        "performance_threshold_ms": args.performance_threshold,
        "timeout_seconds": args.timeout
    })
    
    if args.reports_dir:
        runner.reports_dir = Path(args.reports_dir)
        runner.reports_dir.mkdir(exist_ok=True)
    
    # Run tests
    try:
        results = runner.run_all_tests()
        runner.print_summary(results)
        
        # Exit with appropriate code
        sys.exit(0 if results.failed_tests == 0 else 1)
        
    except KeyboardInterrupt:
        print("\n⏹️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test runner failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()