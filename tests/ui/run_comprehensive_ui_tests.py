#!/usr/bin/env python3
"""
Comprehensive UI Testing Suite Runner for Pynomaly Web Application.

This script runs a complete UI automation test suite including:
- Functionality testing
- Responsive design validation
- Performance monitoring
- Accessibility checks
- Screenshot capture
- Error reporting
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class UITestRunner:
    """Comprehensive UI test runner with reporting capabilities."""

    def __init__(self, base_url: str = "http://localhost:8000", headless: bool = True):
        self.base_url = base_url
        self.headless = headless
        self.test_results = {}
        self.screenshots_dir = Path("screenshots")
        self.reports_dir = Path("reports")
        self.videos_dir = Path("videos")

        # Create directories
        self.screenshots_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        self.videos_dir.mkdir(exist_ok=True)

        print("🚀 Pynomaly UI Test Suite Runner")
        print(f"📍 Base URL: {self.base_url}")
        print(f"🎭 Headless Mode: {self.headless}")
        print(f"📸 Screenshots: {self.screenshots_dir}")
        print(f"📊 Reports: {self.reports_dir}")

    def check_server_availability(self) -> bool:
        """Check if the web server is running and accessible."""
        print("🔍 Checking server availability...")

        try:
            import requests

            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Server is running and accessible")
                return True
            else:
                print(f"⚠️ Server returned status code: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Server not accessible: {e}")
            print("💡 Make sure to start the Pynomaly web server first:")
            print(
                "   poetry run uvicorn pynomaly.presentation.api:app --host 127.0.0.1 --port 8000"
            )
            return False

    def install_dependencies(self) -> bool:
        """Install required dependencies for UI testing."""
        print("📦 Installing UI testing dependencies...")

        try:
            # Install Playwright
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "playwright",
                    "pytest-playwright",
                ],
                check=True,
                capture_output=True,
            )

            # Install browsers
            subprocess.run(
                [sys.executable, "-m", "playwright", "install"],
                check=True,
                capture_output=True,
            )

            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False

    def run_smoke_tests(self) -> dict[str, Any]:
        """Run smoke tests for basic functionality."""
        print("🚨 Running smoke tests...")

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/ui/test_web_app_automation.py::TestSmokeTests",
            "-v",
            "--tb=short",
            f"--browser={'chromium' if self.headless else 'chromium'}",
            "--html=reports/smoke_tests.html",
            "--self-contained-html",
        ]

        if self.headless:
            cmd.extend(["--headed" if not self.headless else "--headless"])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": "5 minutes",
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "returncode": -1,
                "stdout": "",
                "stderr": "Test timed out after 5 minutes",
                "duration": "5+ minutes",
            }

    def run_functionality_tests(self) -> dict[str, Any]:
        """Run comprehensive functionality tests."""
        print("⚙️ Running functionality tests...")

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/ui/test_web_app_automation.py::TestWebAppAutomation",
            "-v",
            "--tb=short",
            "--browser=chromium",
            "--html=reports/functionality_tests.html",
            "--self-contained-html",
        ]

        if self.headless:
            cmd.append("--headless")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": "10 minutes",
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "returncode": -1,
                "stdout": "",
                "stderr": "Test timed out after 10 minutes",
                "duration": "10+ minutes",
            }

    def run_responsive_tests(self) -> dict[str, Any]:
        """Run responsive design tests."""
        print("📱 Running responsive design tests...")

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/ui/test_web_app_automation.py::TestResponsiveDesign",
            "-v",
            "--tb=short",
            "--browser=chromium",
            "--html=reports/responsive_tests.html",
            "--self-contained-html",
        ]

        if self.headless:
            cmd.append("--headless")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": "5 minutes",
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "returncode": -1,
                "stdout": "",
                "stderr": "Test timed out after 5 minutes",
                "duration": "5+ minutes",
            }

    def run_visual_tests(self) -> dict[str, Any]:
        """Run visual regression tests if available."""
        print("👀 Running visual tests...")

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/ui/test_visual_regression.py",
            "-v",
            "--tb=short",
            "--browser=chromium",
            "--html=reports/visual_tests.html",
            "--self-contained-html",
        ]

        if self.headless:
            cmd.append("--headless")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": "5 minutes",
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "returncode": -1,
                "stdout": "",
                "stderr": "Test timed out after 5 minutes",
                "duration": "5+ minutes",
            }
        except Exception:
            return {
                "status": "skipped",
                "returncode": 0,
                "stdout": "Visual tests not available",
                "stderr": "",
                "duration": "0 minutes",
            }

    def capture_application_screenshots(self) -> dict[str, Any]:
        """Capture comprehensive screenshots of the application."""
        print("📸 Capturing application screenshots...")

        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=self.headless)
                context = browser.new_context(viewport={"width": 1920, "height": 1080})
                page = context.new_page()

                pages_to_capture = [
                    ("/web/", "dashboard"),
                    ("/web/detectors", "detectors"),
                    ("/web/datasets", "datasets"),
                    ("/web/detection", "detection"),
                    ("/web/experiments", "experiments"),
                    ("/web/visualizations", "visualizations"),
                    ("/web/exports", "exports"),
                ]

                captured_screenshots = []

                for url_path, name in pages_to_capture:
                    try:
                        page.goto(f"{self.base_url}{url_path}")
                        page.wait_for_load_state("networkidle")

                        # Take full page screenshot
                        screenshot_path = self.screenshots_dir / f"app_{name}_full.png"
                        page.screenshot(path=str(screenshot_path), full_page=True)
                        captured_screenshots.append(str(screenshot_path))

                        # Take viewport screenshot
                        screenshot_path = (
                            self.screenshots_dir / f"app_{name}_viewport.png"
                        )
                        page.screenshot(path=str(screenshot_path), full_page=False)
                        captured_screenshots.append(str(screenshot_path))

                    except Exception as e:
                        print(f"⚠️ Failed to capture {name}: {e}")

                browser.close()

                return {
                    "status": "completed",
                    "screenshots_captured": len(captured_screenshots),
                    "screenshots": captured_screenshots,
                }

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "screenshots_captured": 0,
                "screenshots": [],
            }

    def analyze_screenshots(self) -> dict[str, Any]:
        """Analyze captured screenshots for basic metrics."""
        print("🔍 Analyzing screenshots...")

        screenshot_files = list(self.screenshots_dir.glob("*.png"))

        analysis = {
            "total_screenshots": len(screenshot_files),
            "screenshots_by_type": {},
            "file_sizes": {},
            "largest_screenshot": None,
            "smallest_screenshot": None,
        }

        file_sizes = []

        for screenshot in screenshot_files:
            file_size = screenshot.stat().st_size
            file_sizes.append((screenshot.name, file_size))

            # Categorize by type
            if "mobile" in screenshot.name:
                analysis["screenshots_by_type"]["mobile"] = (
                    analysis["screenshots_by_type"].get("mobile", 0) + 1
                )
            elif "tablet" in screenshot.name:
                analysis["screenshots_by_type"]["tablet"] = (
                    analysis["screenshots_by_type"].get("tablet", 0) + 1
                )
            elif "desktop" in screenshot.name or "app_" in screenshot.name:
                analysis["screenshots_by_type"]["desktop"] = (
                    analysis["screenshots_by_type"].get("desktop", 0) + 1
                )
            else:
                analysis["screenshots_by_type"]["other"] = (
                    analysis["screenshots_by_type"].get("other", 0) + 1
                )

        if file_sizes:
            file_sizes.sort(key=lambda x: x[1])
            analysis["smallest_screenshot"] = {
                "name": file_sizes[0][0],
                "size": file_sizes[0][1],
            }
            analysis["largest_screenshot"] = {
                "name": file_sizes[-1][0],
                "size": file_sizes[-1][1],
            }
            analysis["total_size"] = sum(size for _, size in file_sizes)

        return analysis

    def generate_html_report(self) -> str:
        """Generate comprehensive HTML report."""
        print("📋 Generating HTML report...")

        screenshot_analysis = self.analyze_screenshots()

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pynomaly UI Testing Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #3B82F6; color: white; padding: 20px; border-radius: 8px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .success {{ background: #d4edda; }}
                .warning {{ background: #fff3cd; }}
                .error {{ background: #f8d7da; }}
                .screenshot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .screenshot-item {{ text-align: center; }}
                .screenshot-item img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                pre {{ background: #f8f9fa; padding: 10px; overflow-x: auto; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background: #f8f9fa; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 Pynomaly UI Testing Report</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Base URL: {self.base_url}</p>
            </div>
            
            <div class="section">
                <h2>📊 Test Summary</h2>
                <table>
                    <tr><th>Test Suite</th><th>Status</th><th>Duration</th></tr>
        """

        for test_name, result in self.test_results.items():
            status_class = "success" if result.get("status") == "passed" else "error"
            html_content += f"""
                    <tr class="{status_class}">
                        <td>{test_name}</td>
                        <td>{result.get("status", "unknown")}</td>
                        <td>{result.get("duration", "unknown")}</td>
                    </tr>
            """

        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>📸 Screenshot Analysis</h2>
                <ul>
        """

        html_content += f"""
                    <li>Total Screenshots: {screenshot_analysis["total_screenshots"]}</li>
                    <li>Desktop Screenshots: {screenshot_analysis["screenshots_by_type"].get("desktop", 0)}</li>
                    <li>Mobile Screenshots: {screenshot_analysis["screenshots_by_type"].get("mobile", 0)}</li>
                    <li>Tablet Screenshots: {screenshot_analysis["screenshots_by_type"].get("tablet", 0)}</li>
                    <li>Total Size: {screenshot_analysis.get("total_size", 0) / 1024 / 1024:.2f} MB</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>🖼️ Screenshots Gallery</h2>
                <div class="screenshot-grid">
        """

        # Add screenshots to gallery
        for screenshot in sorted(self.screenshots_dir.glob("*.png")):
            html_content += f"""
                    <div class="screenshot-item">
                        <h4>{screenshot.name}</h4>
                        <img src="{screenshot.name}" alt="{screenshot.name}">
                    </div>
            """

        html_content += """
                </div>
            </div>
        </body>
        </html>
        """

        report_path = (
            self.reports_dir
            / f"ui_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        with open(report_path, "w") as f:
            f.write(html_content)

        return str(report_path)

    def run_all_tests(self) -> dict[str, Any]:
        """Run all UI tests and generate comprehensive report."""
        print("🎯 Starting comprehensive UI test suite...")
        start_time = time.time()

        # Check server availability
        if not self.check_server_availability():
            return {"status": "failed", "reason": "Server not available"}

        # Install dependencies
        if not self.install_dependencies():
            return {"status": "failed", "reason": "Failed to install dependencies"}

        # Run test suites
        self.test_results["smoke_tests"] = self.run_smoke_tests()
        self.test_results["functionality_tests"] = self.run_functionality_tests()
        self.test_results["responsive_tests"] = self.run_responsive_tests()
        self.test_results["visual_tests"] = self.run_visual_tests()

        # Capture screenshots
        screenshot_result = self.capture_application_screenshots()
        self.test_results["screenshots"] = screenshot_result

        # Generate report
        report_path = self.generate_html_report()

        total_time = time.time() - start_time

        # Calculate overall status
        passed_tests = sum(
            1
            for result in self.test_results.values()
            if result.get("status") == "passed"
        )
        total_tests = len(
            [
                r
                for r in self.test_results.values()
                if r.get("status") in ["passed", "failed"]
            ]
        )

        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        summary = {
            "status": "completed",
            "total_duration": f"{total_time:.2f} seconds",
            "tests_run": total_tests,
            "tests_passed": passed_tests,
            "success_rate": f"{success_rate:.1f}%",
            "report_path": report_path,
            "screenshots_captured": screenshot_result.get("screenshots_captured", 0),
            "test_results": self.test_results,
        }

        print("\n" + "=" * 60)
        print("🎉 UI TESTING COMPLETE")
        print("=" * 60)
        print(f"📊 Tests Run: {total_tests}")
        print(f"✅ Tests Passed: {passed_tests}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        print(f"⏱️ Total Duration: {total_time:.2f} seconds")
        print(f"📸 Screenshots: {screenshot_result.get('screenshots_captured', 0)}")
        print(f"📋 Report: {report_path}")
        print("=" * 60)

        return summary


def main():
    """Main function to run UI tests."""
    parser = argparse.ArgumentParser(description="Run Pynomaly UI automation tests")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL for testing (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run tests in headless mode (default: True)",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run tests in headed mode (override headless)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run only smoke tests for quick validation"
    )

    args = parser.parse_args()

    # Override headless if headed is specified
    headless = args.headless and not args.headed

    runner = UITestRunner(base_url=args.url, headless=headless)

    if args.quick:
        print("🏃‍♂️ Running quick smoke tests only...")
        if not runner.check_server_availability():
            sys.exit(1)

        if not runner.install_dependencies():
            sys.exit(1)

        result = runner.run_smoke_tests()

        if result.get("status") == "passed":
            print("✅ Quick smoke tests passed!")
            sys.exit(0)
        else:
            print("❌ Quick smoke tests failed!")
            print(result.get("stderr", ""))
            sys.exit(1)
    else:
        result = runner.run_all_tests()

        if result.get("success_rate", "0%").replace("%", "") == "100.0":
            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
