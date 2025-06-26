#!/usr/bin/env python3
"""Script to run integration tests with various configurations and reporting."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any


class IntegrationTestRunner:
    """Runner for integration tests with configuration and reporting."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_dir = project_root / "tests" / "integration"
        self.reports_dir = project_root / "test_reports"
        self.reports_dir.mkdir(exist_ok=True)
        
    def setup_environment(self, config: Dict[str, Any]) -> None:
        """Setup test environment variables."""
        env_vars = {
            "PYNOMALY_ENVIRONMENT": "testing",
            "PYNOMALY_LOG_LEVEL": config.get("log_level", "INFO"),
            "PYNOMALY_CACHE_ENABLED": "false",
            "PYNOMALY_AUTH_ENABLED": str(config.get("auth_enabled", False)).lower(),
            "PYNOMALY_DOCS_ENABLED": "true",
            "PYNOMALY_CORS_ENABLED": "true",
            "PYNOMALY_MONITORING_METRICS_ENABLED": "false",
            "PYNOMALY_MONITORING_TRACING_ENABLED": "false",
            "PYNOMALY_MONITORING_PROMETHEUS_ENABLED": "false",
        }
        
        # Set database URL for testing
        if config.get("database_url"):
            env_vars["PYNOMALY_DATABASE_URL"] = config["database_url"]
        
        # Apply environment variables
        for key, value in env_vars.items():
            os.environ[key] = value
            print(f"Set {key}={value}")
    
    def run_test_suite(self, test_pattern: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific test suite and return results."""
        print(f"\n{'='*60}")
        print(f"Running test suite: {test_pattern}")
        print(f"{'='*60}")
        
        # Setup environment
        self.setup_environment(config)
        
        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir / test_pattern),
            "-v",
            "--tb=short",
            f"--maxfail={config.get('max_failures', 5)}",
            "--color=yes"
        ]
        
        # Add coverage if requested
        if config.get("coverage", False):
            coverage_dir = self.reports_dir / "coverage"
            coverage_dir.mkdir(exist_ok=True)
            cmd.extend([
                "--cov=src/pynomaly",
                f"--cov-report=html:{coverage_dir}",
                f"--cov-report=xml:{coverage_dir}/coverage.xml",
                "--cov-report=term-missing"
            ])
        
        # Add junit XML report
        if config.get("junit_report", True):
            junit_file = self.reports_dir / f"junit_{test_pattern.replace('*', 'all').replace('/', '_')}.xml"
            cmd.extend([f"--junit-xml={junit_file}"])
        
        # Add markers if specified
        if config.get("markers"):
            for marker in config["markers"]:
                cmd.extend(["-m", marker])
        
        # Add additional pytest args
        if config.get("pytest_args"):
            cmd.extend(config["pytest_args"])
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run tests
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=config.get("timeout", 1800)  # 30 minutes default
            )
            end_time = time.time()
            
            return {
                "pattern": test_pattern,
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "duration": end_time - start_time,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {
                "pattern": test_pattern,
                "success": False,
                "returncode": -1,
                "duration": config.get("timeout", 1800),
                "stdout": "",
                "stderr": "Test suite timed out"
            }
        except Exception as e:
            return {
                "pattern": test_pattern,
                "success": False,
                "returncode": -2,
                "duration": time.time() - start_time,
                "stdout": "",
                "stderr": f"Error running tests: {str(e)}"
            }
    
    def run_all_suites(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run all integration test suites."""
        
        # Define test suites in order of execution
        test_suites = [
            "test_api_workflows.py",
            "test_database_integration.py", 
            "test_streaming_integration.py",
            "test_performance_integration.py",
            "test_security_integration.py",
            "test_end_to_end_scenarios.py",
            "test_regression_suite.py"
        ]
        
        results = []
        
        for test_suite in test_suites:
            if config.get("fail_fast", False) and results and not results[-1]["success"]:
                print(f"Skipping {test_suite} due to previous failure (fail_fast=True)")
                continue
                
            result = self.run_test_suite(test_suite, config)
            results.append(result)
            
            # Print summary for this suite
            status = "PASSED" if result["success"] else "FAILED"
            print(f"\n{test_suite}: {status} ({result['duration']:.1f}s)")
            
            if not result["success"]:
                print("STDERR:", result["stderr"][:500])  # First 500 chars
        
        return results
    
    def generate_summary_report(self, results: List[Dict[str, Any]]) -> None:
        """Generate summary report of test results."""
        
        # Calculate summary statistics
        total_suites = len(results)
        passed_suites = sum(1 for r in results if r["success"])
        failed_suites = total_suites - passed_suites
        total_duration = sum(r["duration"] for r in results)
        
        # Generate HTML report
        html_report = self.reports_dir / "integration_test_summary.html"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Pynomaly Integration Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        .suite {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
        .suite.passed {{ border-left: 5px solid green; }}
        .suite.failed {{ border-left: 5px solid red; }}
        pre {{ background: #f8f8f8; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>Pynomaly Integration Test Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Suites:</strong> {total_suites}</p>
        <p><strong>Passed:</strong> <span class="passed">{passed_suites}</span></p>
        <p><strong>Failed:</strong> <span class="failed">{failed_suites}</span></p>
        <p><strong>Success Rate:</strong> {(passed_suites/total_suites*100):.1f}%</p>
        <p><strong>Total Duration:</strong> {total_duration:.1f} seconds</p>
        <p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <h2>Test Suite Results</h2>
"""
        
        for result in results:
            status_class = "passed" if result["success"] else "failed"
            status_text = "PASSED" if result["success"] else "FAILED"
            
            html_content += f"""
    <div class="suite {status_class}">
        <h3>{result['pattern']} - <span class="{status_class}">{status_text}</span></h3>
        <p><strong>Duration:</strong> {result['duration']:.1f} seconds</p>
        <p><strong>Return Code:</strong> {result['returncode']}</p>
"""
            
            if result["stderr"]:
                html_content += f"""
        <h4>Error Output:</h4>
        <pre>{result['stderr'][:1000]}</pre>
"""
            
            if result["stdout"] and not result["success"]:
                html_content += f"""
        <h4>Standard Output:</h4>
        <pre>{result['stdout'][:1000]}</pre>
"""
            
            html_content += "    </div>\n"
        
        html_content += """
</body>
</html>
"""
        
        with open(html_report, 'w') as f:
            f.write(html_content)
        
        print(f"\nHTML report generated: {html_report}")
        
        # Generate text summary
        text_report = self.reports_dir / "integration_test_summary.txt"
        
        with open(text_report, 'w') as f:
            f.write("PYNOMALY INTEGRATION TEST SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Suites: {total_suites}\n")
            f.write(f"Passed: {passed_suites}\n")
            f.write(f"Failed: {failed_suites}\n")
            f.write(f"Success Rate: {(passed_suites/total_suites*100):.1f}%\n")
            f.write(f"Total Duration: {total_duration:.1f} seconds\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DETAILED RESULTS\n")
            f.write("-" * 30 + "\n\n")
            
            for result in results:
                status = "PASSED" if result["success"] else "FAILED"
                f.write(f"{result['pattern']}: {status} ({result['duration']:.1f}s)\n")
                if not result["success"]:
                    f.write(f"  Error: {result['stderr'][:200]}\n")
                f.write("\n")
        
        print(f"Text report generated: {text_report}")
        
        # Print console summary
        print(f"\n{'='*60}")
        print("INTEGRATION TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Suites: {total_suites}")
        print(f"Passed: {passed_suites}")
        print(f"Failed: {failed_suites}")
        print(f"Success Rate: {(passed_suites/total_suites*100):.1f}%")
        print(f"Total Duration: {total_duration:.1f} seconds")
        
        if failed_suites > 0:
            print(f"\nFAILED SUITES:")
            for result in results:
                if not result["success"]:
                    print(f"  - {result['pattern']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Pynomaly integration tests")
    
    parser.add_argument(
        "--suite", 
        help="Specific test suite to run (e.g., test_api_workflows.py)",
        default=None
    )
    parser.add_argument(
        "--coverage", 
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--auth", 
        action="store_true",
        help="Enable authentication for tests"
    )
    parser.add_argument(
        "--fail-fast", 
        action="store_true",
        help="Stop on first test suite failure"
    )
    parser.add_argument(
        "--max-failures", 
        type=int,
        default=5,
        help="Maximum failures per test suite"
    )
    parser.add_argument(
        "--timeout", 
        type=int,
        default=1800,
        help="Timeout per test suite in seconds"
    )
    parser.add_argument(
        "--markers", 
        nargs="+",
        help="Pytest markers to filter tests"
    )
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level for tests"
    )
    parser.add_argument(
        "--database-url", 
        help="Database URL for testing"
    )
    
    args = parser.parse_args()
    
    # Find project root
    project_root = Path(__file__).parent.parent
    
    # Create test runner
    runner = IntegrationTestRunner(project_root)
    
    # Build configuration
    config = {
        "coverage": args.coverage,
        "auth_enabled": args.auth,
        "fail_fast": args.fail_fast,
        "max_failures": args.max_failures,
        "timeout": args.timeout,
        "markers": args.markers,
        "log_level": args.log_level,
        "database_url": args.database_url,
        "junit_report": True
    }
    
    # Run tests
    if args.suite:
        # Run specific suite
        result = runner.run_test_suite(args.suite, config)
        results = [result]
    else:
        # Run all suites
        results = runner.run_all_suites(config)
    
    # Generate reports
    runner.generate_summary_report(results)
    
    # Exit with appropriate code
    failed_count = sum(1 for r in results if not r["success"])
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()