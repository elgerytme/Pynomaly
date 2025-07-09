#!/usr/bin/env python3
"""
Integration Test Runner

Comprehensive test runner for all integration tests with reporting and monitoring.
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pytest
from tests.integration.framework.integration_test_base import IntegrationTestRunner
from tests.integration.test_end_to_end_workflow import TestEndToEndWorkflow
from tests.integration.test_api_integration import TestAPIIntegration
from tests.integration.test_service_integration import TestServiceIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('integration_tests.log')
    ]
)
logger = logging.getLogger(__name__)


class IntegrationTestSuite:
    """Comprehensive integration test suite runner."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize test suite runner."""
        self.config = config or self._get_default_config()
        self.results: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
        self.test_classes = [
            TestEndToEndWorkflow,
            TestAPIIntegration,
            TestServiceIntegration
        ]
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default test configuration."""
        return {
            "test_environment": "integration",
            "parallel_execution": False,
            "verbose": True,
            "generate_reports": True,
            "cleanup_after_tests": True,
            "test_timeout": 300,  # 5 minutes per test
            "max_retries": 2,
            "fail_fast": False,
            "coverage_enabled": True,
            "performance_monitoring": True,
            "output_format": "json",
            "results_directory": "test-results/integration",
            "log_level": "INFO"
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        logger.info("Starting comprehensive integration test suite...")
        self.start_time = time.time()
        
        # Create results directory
        results_dir = Path(self.config["results_directory"])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize test results
        self.results = {
            "start_time": datetime.now().isoformat(),
            "config": self.config,
            "test_results": {},
            "summary": {},
            "errors": [],
            "performance_metrics": {}
        }
        
        try:
            # Run test suites
            await self._run_test_suites()
            
            # Generate summary
            self._generate_summary()
            
            # Generate reports
            if self.config["generate_reports"]:
                await self._generate_reports()
            
            # Performance analysis
            if self.config["performance_monitoring"]:
                await self._analyze_performance()
            
            logger.info("Integration test suite completed successfully")
            
        except Exception as e:
            logger.error(f"Integration test suite failed: {str(e)}")
            self.results["errors"].append({
                "type": "suite_failure",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
            raise
        
        finally:
            # Cleanup
            if self.config["cleanup_after_tests"]:
                await self._cleanup()
            
            # Save results
            await self._save_results()
        
        return self.results
    
    async def _run_test_suites(self) -> None:
        """Run individual test suites."""
        logger.info(f"Running {len(self.test_classes)} test suites...")
        
        for test_class in self.test_classes:
            suite_name = test_class.__name__
            logger.info(f"Running test suite: {suite_name}")
            
            try:
                # Run test suite
                suite_results = await self._run_test_suite(test_class)
                self.results["test_results"][suite_name] = suite_results
                
                # Log results
                logger.info(f"Test suite {suite_name} completed: "
                           f"{suite_results['passed']}/{suite_results['total']} passed")
                
                # Fail fast if enabled
                if self.config["fail_fast"] and suite_results["failed"] > 0:
                    logger.error(f"Failing fast due to failures in {suite_name}")
                    break
                    
            except Exception as e:
                logger.error(f"Test suite {suite_name} failed with error: {str(e)}")
                self.results["errors"].append({
                    "type": "suite_error",
                    "suite": suite_name,
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                
                if self.config["fail_fast"]:
                    raise
    
    async def _run_test_suite(self, test_class) -> Dict[str, Any]:
        """Run a single test suite."""
        suite_start_time = time.time()
        
        # Create test runner
        runner = IntegrationTestRunner(self.config)
        
        # Run tests
        results = await runner.run_test_suite([test_class])
        
        # Add timing information
        results["suite_execution_time"] = time.time() - suite_start_time
        results["timestamp"] = datetime.now().isoformat()
        
        return results
    
    def _generate_summary(self) -> None:
        """Generate test summary."""
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_execution_time = 0
        
        for suite_name, suite_results in self.results["test_results"].items():
            total_tests += suite_results["total_tests"]
            total_passed += suite_results["passed"]
            total_failed += suite_results["failed"]
            total_execution_time += suite_results["execution_time"]
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        self.results["summary"] = {
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "success_rate": success_rate,
            "total_execution_time": total_execution_time,
            "average_test_time": total_execution_time / total_tests if total_tests > 0 else 0,
            "end_time": datetime.now().isoformat(),
            "overall_status": "PASSED" if total_failed == 0 else "FAILED"
        }
    
    async def _generate_reports(self) -> None:
        """Generate test reports."""
        logger.info("Generating integration test reports...")
        
        results_dir = Path(self.config["results_directory"])
        
        # JSON report
        json_report_path = results_dir / "integration_test_report.json"
        with open(json_report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # HTML report
        html_report_path = results_dir / "integration_test_report.html"
        await self._generate_html_report(html_report_path)
        
        # JUnit XML report
        junit_report_path = results_dir / "integration_test_junit.xml"
        await self._generate_junit_report(junit_report_path)
        
        logger.info(f"Reports generated in: {results_dir}")
    
    async def _generate_html_report(self, output_path: Path) -> None:
        """Generate HTML test report."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Integration Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .suite {{ margin: 20px 0; border: 1px solid #ddd; padding: 10px; }}
                .suite-header {{ font-weight: bold; font-size: 1.2em; }}
                .metrics {{ display: flex; justify-content: space-between; }}
                .metric {{ text-align: center; }}
                .error {{ background-color: #ffebee; padding: 10px; border-radius: 3px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <h1>Integration Test Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <div class="metrics">
                    <div class="metric">
                        <div>Total Tests</div>
                        <div>{total_tests}</div>
                    </div>
                    <div class="metric">
                        <div>Passed</div>
                        <div class="passed">{passed}</div>
                    </div>
                    <div class="metric">
                        <div>Failed</div>
                        <div class="failed">{failed}</div>
                    </div>
                    <div class="metric">
                        <div>Success Rate</div>
                        <div>{success_rate:.1f}%</div>
                    </div>
                    <div class="metric">
                        <div>Execution Time</div>
                        <div>{execution_time:.2f}s</div>
                    </div>
                </div>
                <p><strong>Status:</strong> 
                   <span class="{status_class}">{status}</span>
                </p>
            </div>
            
            <h2>Test Suites</h2>
            {suites_html}
            
            {errors_html}
        </body>
        </html>
        """
        
        # Generate suites HTML
        suites_html = ""
        for suite_name, suite_results in self.results["test_results"].items():
            suite_html = f"""
            <div class="suite">
                <div class="suite-header">{suite_name}</div>
                <p>Tests: {suite_results['total_tests']} | 
                   Passed: <span class="passed">{suite_results['passed']}</span> | 
                   Failed: <span class="failed">{suite_results['failed']}</span> | 
                   Time: {suite_results['execution_time']:.2f}s</p>
            </div>
            """
            suites_html += suite_html
        
        # Generate errors HTML
        errors_html = ""
        if self.results["errors"]:
            errors_html = "<h2>Errors</h2>"
            for error in self.results["errors"]:
                errors_html += f"""
                <div class="error">
                    <strong>{error['type']}</strong>: {error['message']}
                    <br><small>{error['timestamp']}</small>
                </div>
                """
        
        # Format template
        html_content = html_template.format(
            total_tests=self.results["summary"]["total_tests"],
            passed=self.results["summary"]["passed"],
            failed=self.results["summary"]["failed"],
            success_rate=self.results["summary"]["success_rate"],
            execution_time=self.results["summary"]["total_execution_time"],
            status=self.results["summary"]["overall_status"],
            status_class="passed" if self.results["summary"]["overall_status"] == "PASSED" else "failed",
            suites_html=suites_html,
            errors_html=errors_html
        )
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    async def _generate_junit_report(self, output_path: Path) -> None:
        """Generate JUnit XML test report."""
        junit_template = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="Integration Tests" tests="{total_tests}" failures="{failed}" time="{execution_time}">
{suites_xml}
</testsuites>"""
        
        suites_xml = ""
        for suite_name, suite_results in self.results["test_results"].items():
            suite_xml = f"""    <testsuite name="{suite_name}" tests="{suite_results['total_tests']}" failures="{suite_results['failed']}" time="{suite_results['execution_time']}">
        <testcase name="{suite_name}" classname="integration" time="{suite_results['execution_time']}"/>
    </testsuite>
"""
            suites_xml += suite_xml
        
        junit_content = junit_template.format(
            total_tests=self.results["summary"]["total_tests"],
            failed=self.results["summary"]["failed"],
            execution_time=self.results["summary"]["total_execution_time"],
            suites_xml=suites_xml
        )
        
        with open(output_path, 'w') as f:
            f.write(junit_content)
    
    async def _analyze_performance(self) -> None:
        """Analyze performance metrics."""
        logger.info("Analyzing performance metrics...")
        
        performance_data = {
            "total_execution_time": self.results["summary"]["total_execution_time"],
            "average_test_time": self.results["summary"]["average_test_time"],
            "suite_performance": {},
            "performance_trends": {},
            "bottlenecks": []
        }
        
        # Analyze suite performance
        for suite_name, suite_results in self.results["test_results"].items():
            performance_data["suite_performance"][suite_name] = {
                "execution_time": suite_results["execution_time"],
                "tests_per_second": suite_results["total_tests"] / suite_results["execution_time"] if suite_results["execution_time"] > 0 else 0,
                "average_test_time": suite_results["execution_time"] / suite_results["total_tests"] if suite_results["total_tests"] > 0 else 0
            }
        
        # Identify bottlenecks
        slowest_suite = max(
            self.results["test_results"].items(),
            key=lambda x: x[1]["execution_time"]
        )
        
        if slowest_suite[1]["execution_time"] > 60:  # More than 1 minute
            performance_data["bottlenecks"].append({
                "type": "slow_suite",
                "suite": slowest_suite[0],
                "execution_time": slowest_suite[1]["execution_time"],
                "recommendation": "Consider optimizing test setup/teardown or splitting into smaller tests"
            })
        
        self.results["performance_metrics"] = performance_data
    
    async def _cleanup(self) -> None:
        """Cleanup test resources."""
        logger.info("Cleaning up test resources...")
        
        # Cleanup test data directories
        test_data_dirs = [
            Path("tests/integration/data"),
            Path("./test_storage"),
            Path("./test.db")
        ]
        
        for data_dir in test_data_dirs:
            if data_dir.exists():
                if data_dir.is_file():
                    data_dir.unlink()
                else:
                    import shutil
                    shutil.rmtree(data_dir, ignore_errors=True)
                logger.debug(f"Cleaned up: {data_dir}")
    
    async def _save_results(self) -> None:
        """Save test results."""
        results_dir = Path(self.config["results_directory"])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_file = results_dir / f"integration_test_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save latest results
        latest_file = results_dir / "latest_integration_results.json"
        with open(latest_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {results_file}")


async def main():
    """Main entry point for integration test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run integration tests")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel execution")
    parser.add_argument("--coverage", action="store_true", help="Enable coverage reporting")
    parser.add_argument("--output-dir", type=str, default="test-results/integration", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override with command line arguments
    if args.verbose:
        config["verbose"] = True
    if args.fail_fast:
        config["fail_fast"] = True
    if args.parallel:
        config["parallel_execution"] = True
    if args.coverage:
        config["coverage_enabled"] = True
    if args.output_dir:
        config["results_directory"] = args.output_dir
    
    # Run test suite
    test_suite = IntegrationTestSuite(config)
    
    try:
        results = await test_suite.run_all_tests()
        
        # Print summary
        summary = results["summary"]
        print("\n" + "="*50)
        print("INTEGRATION TEST SUMMARY")
        print("="*50)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Execution Time: {summary['total_execution_time']:.2f}s")
        print(f"Status: {summary['overall_status']}")
        print("="*50)
        
        # Exit with appropriate code
        sys.exit(0 if summary["overall_status"] == "PASSED" else 1)
        
    except Exception as e:
        logger.error(f"Integration test suite failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())