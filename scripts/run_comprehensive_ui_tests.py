#!/usr/bin/env python3
"""Comprehensive UI test runner with visual regression, accessibility, and performance testing."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import test modules
try:
    from tests.ui.conftest import TEST_CONFIG
    from tests.ui.test_accessibility_enhanced import TestAccessibilityEnhanced
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed: poetry install")
    sys.exit(1)


class ComprehensiveUITestRunner:
    """Comprehensive UI test runner with enhanced reporting and analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.project_root = project_root
        self.reports_dir = project_root / "test_reports" / "ui_comprehensive"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Test categories and their priorities
        self.test_categories = {
            "visual_regression": {"priority": "high", "timeout": 600},
            "accessibility": {"priority": "high", "timeout": 300},
            "performance": {"priority": "medium", "timeout": 300},
            "cross_browser": {"priority": "medium", "timeout": 900},
            "responsive": {"priority": "medium", "timeout": 450},
            "bdd_workflows": {"priority": "high", "timeout": 1200},
            "security": {"priority": "low", "timeout": 300}
        }
        
        self.results = {
            "start_time": time.time(),
            "test_results": {},
            "summary": {},
            "issues": [],
            "recommendations": []
        }

    def setup_environment(self):
        """Setup test environment variables and configuration."""
        # Set comprehensive testing environment variables
        test_env = {
            "PYNOMALY_ENVIRONMENT": "testing",
            "PYNOMALY_LOG_LEVEL": "WARNING",
            "PYNOMALY_AUTH_ENABLED": "false",
            "PYNOMALY_DOCS_ENABLED": "true",
            "PYNOMALY_CORS_ENABLED": "true",
            
            # UI Testing specific
            "HEADLESS": str(self.config.get("headless", True)).lower(),
            "SLOW_MO": str(self.config.get("slow_mo", 0)),
            "TAKE_SCREENSHOTS": str(self.config.get("screenshots", True)).lower(),
            "RECORD_VIDEOS": str(self.config.get("videos", False)).lower(),
            "RECORD_TRACES": str(self.config.get("traces", False)).lower(),
            "VISUAL_TESTING": str(self.config.get("visual_testing", True)).lower(),
            "ACCESSIBILITY_TESTING": str(self.config.get("accessibility_testing", True)).lower(),
            "PERFORMANCE_TESTING": str(self.config.get("performance_testing", True)).lower(),
        }
        
        # Apply environment variables
        for key, value in test_env.items():
            os.environ[key] = value
            if self.config.get("verbose", False):
                print(f"Set {key}={value}")

    def run_visual_regression_tests(self) -> Dict[str, Any]:
        """Run comprehensive visual regression testing."""
        print("\n🎨 Running Visual Regression Tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.project_root / "tests" / "ui" / "test_visual_regression.py"),
            "-v", "--tb=short",
            f"--maxfail={self.config.get('max_failures', 5)}",
            "--color=yes"
        ]
        
        if self.config.get("markers"):
            for marker in self.config["markers"]:
                cmd.extend(["-m", marker])
        
        # Add junit XML report
        junit_file = self.reports_dir / "visual_regression_junit.xml"
        cmd.extend([f"--junit-xml={junit_file}"])
        
        return self._run_test_command(cmd, "visual_regression")

    def run_accessibility_tests(self) -> Dict[str, Any]:
        """Run comprehensive accessibility testing."""
        print("\n♿ Running Accessibility Tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.project_root / "tests" / "ui" / "test_accessibility_enhanced.py"),
            "-v", "--tb=short",
            f"--maxfail={self.config.get('max_failures', 3)}",
            "--color=yes"
        ]
        
        # Add junit XML report
        junit_file = self.reports_dir / "accessibility_junit.xml"
        cmd.extend([f"--junit-xml={junit_file}"])
        
        return self._run_test_command(cmd, "accessibility")

    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance testing."""
        print("\n⚡ Running Performance Tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.project_root / "tests" / "ui" / "test_performance_monitoring.py"),
            "-v", "--tb=short",
            f"--maxfail={self.config.get('max_failures', 3)}",
            "--color=yes"
        ]
        
        # Add junit XML report
        junit_file = self.reports_dir / "performance_junit.xml"
        cmd.extend([f"--junit-xml={junit_file}"])
        
        return self._run_test_command(cmd, "performance")

    def run_responsive_tests(self) -> Dict[str, Any]:
        """Run responsive design testing."""
        print("\n📱 Running Responsive Design Tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.project_root / "tests" / "ui" / "test_responsive_design.py"),
            "-v", "--tb=short",
            f"--maxfail={self.config.get('max_failures', 3)}",
            "--color=yes"
        ]
        
        # Add junit XML report
        junit_file = self.reports_dir / "responsive_junit.xml"
        cmd.extend([f"--junit-xml={junit_file}"])
        
        return self._run_test_command(cmd, "responsive")

    def run_bdd_workflow_tests(self) -> Dict[str, Any]:
        """Run BDD workflow testing."""
        print("\n🎭 Running BDD Workflow Tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.project_root / "tests" / "ui" / "bdd" / "test_user_workflows.py"),
            "-v", "--tb=short",
            f"--maxfail={self.config.get('max_failures', 2)}",
            "--color=yes"
        ]
        
        # Add junit XML report
        junit_file = self.reports_dir / "bdd_workflows_junit.xml"
        cmd.extend([f"--junit-xml={junit_file}"])
        
        return self._run_test_command(cmd, "bdd_workflows")

    def run_cross_browser_tests(self) -> Dict[str, Any]:
        """Run cross-browser compatibility testing."""
        print("\n🌐 Running Cross-Browser Tests...")
        
        # Run tests across different browsers
        browsers = self.config.get("browsers", ["chromium", "firefox"])
        results = []
        
        for browser in browsers:
            print(f"  Testing with {browser}...")
            
            cmd = [
                sys.executable, "-m", "pytest",
                str(self.project_root / "tests" / "ui" / "test_visual_regression.py"),
                "-v", "--tb=short",
                f"--maxfail={self.config.get('max_failures', 2)}",
                "--color=yes",
                "-k", "cross_browser"
            ]
            
            # Set browser environment
            os.environ["BROWSER"] = browser
            
            # Add junit XML report
            junit_file = self.reports_dir / f"cross_browser_{browser}_junit.xml"
            cmd.extend([f"--junit-xml={junit_file}"])
            
            browser_result = self._run_test_command(cmd, f"cross_browser_{browser}")
            results.append(browser_result)
        
        # Aggregate results
        total_duration = sum(r["duration"] for r in results)
        success = all(r["success"] for r in results)
        
        return {
            "category": "cross_browser",
            "success": success,
            "duration": total_duration,
            "browser_results": results,
            "browsers_tested": browsers
        }

    def run_playwright_tests(self) -> Dict[str, Any]:
        """Run Playwright-based tests if Playwright is configured."""
        print("\n🎭 Running Playwright Tests...")
        
        playwright_config = self.project_root / "playwright.config.ts"
        if not playwright_config.exists():
            return {
                "category": "playwright",
                "success": False,
                "duration": 0,
                "error": "Playwright configuration not found"
            }
        
        try:
            # Install Playwright browsers if needed
            subprocess.run(["npx", "playwright", "install"], 
                         cwd=self.project_root, check=False, capture_output=True)
            
            # Run Playwright tests
            cmd = ["npx", "playwright", "test", "--reporter=html,json"]
            
            if self.config.get("headless", True):
                cmd.append("--headed=false")
            else:
                cmd.append("--headed=true")
            
            return self._run_test_command(cmd, "playwright", cwd=self.project_root)
            
        except Exception as e:
            return {
                "category": "playwright",
                "success": False,
                "duration": 0,
                "error": str(e)
            }

    def _run_test_command(self, cmd: List[str], category: str, cwd: Path = None) -> Dict[str, Any]:
        """Run a test command and capture results."""
        start_time = time.time()
        cwd = cwd or self.project_root
        
        timeout = self.test_categories.get(category, {}).get("timeout", 300)
        
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            duration = time.time() - start_time
            
            return {
                "category": category,
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd)
            }
            
        except subprocess.TimeoutExpired:
            return {
                "category": category,
                "success": False,
                "returncode": -1,
                "duration": timeout,
                "stdout": "",
                "stderr": f"Test category {category} timed out after {timeout} seconds",
                "command": " ".join(cmd)
            }
        except Exception as e:
            return {
                "category": category,
                "success": False,
                "returncode": -2,
                "duration": time.time() - start_time,
                "stdout": "",
                "stderr": f"Error running {category} tests: {str(e)}",
                "command": " ".join(cmd)
            }

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive UI tests."""
        print("🚀 Starting Comprehensive UI Test Suite")
        print("=" * 60)
        
        # Setup environment
        self.setup_environment()
        
        # Define test execution order (high priority first)
        test_execution_plan = [
            ("accessibility", self.run_accessibility_tests),
            ("visual_regression", self.run_visual_regression_tests),
            ("bdd_workflows", self.run_bdd_workflow_tests),
            ("performance", self.run_performance_tests),
            ("responsive", self.run_responsive_tests),
            ("cross_browser", self.run_cross_browser_tests),
        ]
        
        # Add Playwright tests if available
        if (self.project_root / "playwright.config.ts").exists():
            test_execution_plan.append(("playwright", self.run_playwright_tests))
        
        # Execute tests
        for category, test_func in test_execution_plan:
            if self.config.get("categories") and category not in self.config["categories"]:
                print(f"⏭️  Skipping {category} (not in selected categories)")
                continue
            
            if self.config.get("fail_fast") and self.results["test_results"]:
                # Check if any previous test failed
                previous_failures = [r for r in self.results["test_results"].values() if not r.get("success", False)]
                if previous_failures:
                    print(f"⏭️  Skipping {category} due to previous failure (fail_fast=True)")
                    continue
            
            try:
                result = test_func()
                self.results["test_results"][category] = result
                
                # Print immediate feedback
                status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
                duration = result.get("duration", 0)
                print(f"{status} {category}: {duration:.1f}s")
                
                if not result.get("success", False) and result.get("stderr"):
                    print(f"   Error: {result['stderr'][:200]}...")
                
            except Exception as e:
                self.results["test_results"][category] = {
                    "category": category,
                    "success": False,
                    "error": str(e),
                    "duration": 0
                }
                print(f"❌ FAILED {category}: Exception - {str(e)}")
        
        # Generate final summary
        self._generate_summary()
        
        return self.results

    def _generate_summary(self):
        """Generate comprehensive test summary."""
        total_tests = len(self.results["test_results"])
        passed_tests = sum(1 for r in self.results["test_results"].values() if r.get("success", False))
        failed_tests = total_tests - passed_tests
        total_duration = sum(r.get("duration", 0) for r in self.results["test_results"].values())
        
        self.results["summary"] = {
            "total_categories": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "total_duration": total_duration,
            "end_time": time.time()
        }
        
        # Identify issues and generate recommendations
        self._analyze_results()

    def _analyze_results(self):
        """Analyze test results and generate recommendations."""
        issues = []
        recommendations = []
        
        for category, result in self.results["test_results"].items():
            if not result.get("success", False):
                priority = self.test_categories.get(category, {}).get("priority", "medium")
                issues.append({
                    "category": category,
                    "priority": priority,
                    "error": result.get("stderr", result.get("error", "Unknown error"))
                })
        
        # Generate recommendations based on failures
        high_priority_failures = [i for i in issues if i["priority"] == "high"]
        
        if high_priority_failures:
            recommendations.append("Address high-priority failures immediately: " + 
                                 ", ".join([i["category"] for i in high_priority_failures]))
        
        if len(issues) > len(self.results["test_results"]) * 0.5:
            recommendations.append("More than 50% of test categories failed - review test environment and infrastructure")
        
        if any("timeout" in str(i.get("error", "")).lower() for i in issues):
            recommendations.append("Some tests timed out - consider increasing timeout values or optimizing test performance")
        
        if any("accessibility" in i["category"] for i in issues):
            recommendations.append("Accessibility issues found - prioritize WCAG compliance fixes")
        
        if any("visual" in i["category"] for i in issues):
            recommendations.append("Visual regression detected - review UI changes and update baselines if needed")
        
        self.results["issues"] = issues
        self.results["recommendations"] = recommendations

    def generate_reports(self):
        """Generate comprehensive test reports."""
        # JSON report
        json_report = self.reports_dir / "comprehensive_ui_test_report.json"
        with open(json_report, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # HTML report
        html_report = self.reports_dir / "comprehensive_ui_test_report.html"
        self._generate_html_report(html_report)
        
        # Text summary
        text_report = self.reports_dir / "comprehensive_ui_test_summary.txt"
        self._generate_text_report(text_report)
        
        print(f"\n📊 Reports generated:")
        print(f"   📄 JSON: {json_report}")
        print(f"   🌐 HTML: {html_report}")
        print(f"   📝 Text: {text_report}")

    def _generate_html_report(self, filepath: Path):
        """Generate HTML test report."""
        summary = self.results["summary"]
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Pynomaly Comprehensive UI Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .passed {{ color: green; font-weight: bold; }}
        .failed {{ color: red; font-weight: bold; }}
        .category {{ margin: 15px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
        .category.passed {{ border-left: 5px solid green; }}
        .category.failed {{ border-left: 5px solid red; }}
        .recommendations {{ background: #fff3cd; padding: 15px; border-radius: 5px; border: 1px solid #ffeaa7; }}
        .issues {{ background: #f8d7da; padding: 15px; border-radius: 5px; border: 1px solid #f5c6cb; }}
        pre {{ background: #f8f8f8; padding: 10px; overflow-x: auto; max-height: 200px; }}
        .priority-high {{ color: #dc3545; }}
        .priority-medium {{ color: #fd7e14; }}
        .priority-low {{ color: #6c757d; }}
    </style>
</head>
<body>
    <h1>Pynomaly Comprehensive UI Test Report</h1>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <p><strong>Total Test Categories:</strong> {summary['total_categories']}</p>
        <p><strong>Passed:</strong> <span class="passed">{summary['passed']}</span></p>
        <p><strong>Failed:</strong> <span class="failed">{summary['failed']}</span></p>
        <p><strong>Success Rate:</strong> {summary['success_rate']:.1f}%</p>
        <p><strong>Total Duration:</strong> {summary['total_duration']:.1f} seconds</p>
        <p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""
        
        # Add recommendations if any
        if self.results["recommendations"]:
            html_content += """
    <div class="recommendations">
        <h3>🎯 Recommendations</h3>
        <ul>
"""
            for rec in self.results["recommendations"]:
                html_content += f"            <li>{rec}</li>\n"
            html_content += "        </ul>\n    </div>\n"
        
        # Add issues if any
        if self.results["issues"]:
            html_content += """
    <div class="issues">
        <h3>⚠️ Issues Found</h3>
        <ul>
"""
            for issue in self.results["issues"]:
                priority_class = f"priority-{issue['priority']}"
                html_content += f"""            <li class="{priority_class}">
                <strong>{issue['category']}</strong> ({issue['priority']} priority): {issue['error'][:100]}...
            </li>\n"""
            html_content += "        </ul>\n    </div>\n"
        
        # Add detailed results
        html_content += "\n    <h2>Detailed Test Results</h2>\n"
        
        for category, result in self.results["test_results"].items():
            status_class = "passed" if result.get("success", False) else "failed"
            status_text = "PASSED" if result.get("success", False) else "FAILED"
            duration = result.get("duration", 0)
            
            html_content += f"""
    <div class="category {status_class}">
        <h3>{category.replace('_', ' ').title()} - <span class="{status_class}">{status_text}</span></h3>
        <p><strong>Duration:</strong> {duration:.1f} seconds</p>
        <p><strong>Return Code:</strong> {result.get('returncode', 'N/A')}</p>
"""
            
            if result.get("stderr"):
                html_content += f"""
        <h4>Error Output:</h4>
        <pre>{result['stderr'][:1000]}</pre>
"""
            
            if result.get("command"):
                html_content += f"""
        <h4>Command:</h4>
        <pre>{result['command']}</pre>
"""
            
            html_content += "    </div>\n"
        
        html_content += """
</body>
</html>
"""
        
        with open(filepath, "w") as f:
            f.write(html_content)

    def _generate_text_report(self, filepath: Path):
        """Generate text summary report."""
        summary = self.results["summary"]
        
        with open(filepath, "w") as f:
            f.write("PYNOMALY COMPREHENSIVE UI TEST SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Categories: {summary['total_categories']}\n")
            f.write(f"Passed: {summary['passed']}\n")
            f.write(f"Failed: {summary['failed']}\n")
            f.write(f"Success Rate: {summary['success_rate']:.1f}%\n")
            f.write(f"Total Duration: {summary['total_duration']:.1f} seconds\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if self.results["recommendations"]:
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 20 + "\n")
                for rec in self.results["recommendations"]:
                    f.write(f"• {rec}\n")
                f.write("\n")
            
            f.write("DETAILED RESULTS\n")
            f.write("-" * 20 + "\n\n")
            
            for category, result in self.results["test_results"].items():
                status = "PASSED" if result.get("success", False) else "FAILED"
                duration = result.get("duration", 0)
                f.write(f"{category}: {status} ({duration:.1f}s)\n")
                if not result.get("success", False) and result.get("stderr"):
                    f.write(f"  Error: {result['stderr'][:200]}...\n")
                f.write("\n")

    def print_final_summary(self):
        """Print final test summary to console."""
        summary = self.results["summary"]
        
        print(f"\n{'='*60}")
        print("🏁 COMPREHENSIVE UI TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Categories: {summary['total_categories']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Duration: {summary['total_duration']:.1f} seconds")
        
        if self.results["issues"]:
            print(f"\n❌ FAILED CATEGORIES:")
            for issue in self.results["issues"]:
                priority_indicator = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                indicator = priority_indicator.get(issue["priority"], "⚪")
                print(f"  {indicator} {issue['category']} ({issue['priority']} priority)")
        
        if self.results["recommendations"]:
            print(f"\n🎯 RECOMMENDATIONS:")
            for rec in self.results["recommendations"]:
                print(f"  • {rec}")


def main():
    """Main entry point for comprehensive UI testing."""
    parser = argparse.ArgumentParser(description="Run comprehensive UI tests for Pynomaly")
    
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["accessibility", "visual_regression", "performance", "responsive", 
                "cross_browser", "bdd_workflows", "playwright"],
        help="Specific test categories to run"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run tests in headless mode"
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run tests in headed mode (opposite of headless)"
    )
    parser.add_argument(
        "--screenshots",
        action="store_true",
        default=True,
        help="Take screenshots during tests"
    )
    parser.add_argument(
        "--videos",
        action="store_true",
        help="Record videos during tests"
    )
    parser.add_argument(
        "--traces",
        action="store_true",
        help="Record traces during tests"
    )
    parser.add_argument(
        "--visual-testing",
        action="store_true",
        default=True,
        help="Enable visual regression testing"
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first category failure"
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=5,
        help="Maximum failures per test category"
    )
    parser.add_argument(
        "--slow-mo",
        type=int,
        default=0,
        help="Slow motion delay in milliseconds"
    )
    parser.add_argument(
        "--browsers",
        nargs="+",
        default=["chromium", "firefox"],
        help="Browsers to test with for cross-browser testing"
    )
    parser.add_argument(
        "--markers",
        nargs="+",
        help="Pytest markers to filter tests"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Configure test runner
    config = {
        "headless": args.headless and not args.headed,
        "screenshots": args.screenshots,
        "videos": args.videos,
        "traces": args.traces,
        "visual_testing": args.visual_testing,
        "accessibility_testing": True,
        "performance_testing": True,
        "fail_fast": args.fail_fast,
        "max_failures": args.max_failures,
        "slow_mo": args.slow_mo,
        "browsers": args.browsers,
        "markers": args.markers,
        "verbose": args.verbose,
        "categories": args.categories
    }
    
    # Create and run comprehensive test suite
    runner = ComprehensiveUITestRunner(config)
    results = runner.run_all_tests()
    
    # Generate reports
    runner.generate_reports()
    
    # Print final summary
    runner.print_final_summary()
    
    # Exit with appropriate code
    failed_count = results["summary"]["failed"]
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()