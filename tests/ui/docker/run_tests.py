#!/usr/bin/env python3
"""
Test runner script for Docker-based UI testing with Chrome and Firefox.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")

    return result


def build_docker_image():
    """Build the Docker image for UI testing."""
    print("Building Docker image...")

    # Create requirements.txt for Docker
    requirements_content = """
playwright==1.40.0
pytest==7.4.3
pytest-playwright==0.4.3
pytest-xdist==3.5.0
pytest-html==4.1.1
pytest-json-report==1.5.0
pytest-timeout==2.2.0
pytest-flaky==3.7.0
pandas==2.1.4
numpy==1.24.4
"""

    with open("requirements.txt", "w") as f:
        f.write(requirements_content)

    result = run_command(["docker", "build", "-t", "pynomaly-ui-tests", "."])

    if result.returncode != 0:
        print("Failed to build Docker image")
        return False

    return True


def run_tests_in_docker(browser="chrome", parallel=True, verbose=True):
    """Run tests in Docker container."""
    print(f"Running tests with {browser} browser...")

    # Create test reports directory
    reports_dir = Path(f"test_reports/{browser}")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Docker run command
    cmd = [
        "docker",
        "run",
        "--rm",
        "-e",
        f"BROWSER={browser}",
        "-e",
        "HEADLESS=true",
        "-e",
        "TAKE_SCREENSHOTS=true",
        "-e",
        "RECORD_VIDEOS=true",
        "-e",
        "PYNOMALY_BASE_URL=http://host.docker.internal:8000",
        "-v",
        f"{reports_dir.absolute()}:/app/test_reports",
        "--network",
        "host",
        "pynomaly-ui-tests",
    ]

    # Add pytest arguments
    pytest_args = [
        "python",
        "-m",
        "pytest",
        "tests/ui/",
        "-v" if verbose else "",
        f"--html=test_reports/report_{browser}.html",
        "--self-contained-html",
        "--json-report",
        f"--json-report-file=test_reports/report_{browser}.json",
        "--timeout=60",
    ]

    if parallel:
        pytest_args.extend(["-n", "2"])

    # Filter empty strings
    pytest_args = [arg for arg in pytest_args if arg]

    cmd.extend(pytest_args)

    result = run_command(cmd)

    return result.returncode == 0


def run_docker_compose_tests():
    """Run tests using Docker Compose."""
    print("Running tests with Docker Compose...")

    # Build and run with docker-compose
    result = run_command(
        ["docker-compose", "up", "--build", "--abort-on-container-exit"]
    )

    if result.returncode != 0:
        print("Docker Compose tests failed")
        return False

    return True


def generate_combined_report(browsers=["chrome", "firefox"]):
    """Generate combined test report from multiple browsers."""
    print("Generating combined test report...")

    combined_results = {
        "browsers": {},
        "summary": {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "duration": 0,
        },
    }

    for browser in browsers:
        report_file = Path(f"test_reports/{browser}/report_{browser}.json")

        if report_file.exists():
            with open(report_file) as f:
                browser_results = json.load(f)
                combined_results["browsers"][browser] = browser_results

                # Update summary
                if "summary" in browser_results:
                    summary = browser_results["summary"]
                    combined_results["summary"]["total_tests"] += summary.get(
                        "total", 0
                    )
                    combined_results["summary"]["passed"] += summary.get("passed", 0)
                    combined_results["summary"]["failed"] += summary.get("failed", 0)
                    combined_results["summary"]["skipped"] += summary.get("skipped", 0)
                    combined_results["summary"]["duration"] += summary.get(
                        "duration", 0
                    )

    # Save combined report
    combined_report_file = Path("test_reports/combined_report.json")
    with open(combined_report_file, "w") as f:
        json.dump(combined_results, f, indent=2)

    print(f"Combined report saved to {combined_report_file}")

    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Total tests: {combined_results['summary']['total_tests']}")
    print(f"Passed: {combined_results['summary']['passed']}")
    print(f"Failed: {combined_results['summary']['failed']}")
    print(f"Skipped: {combined_results['summary']['skipped']}")
    print(f"Duration: {combined_results['summary']['duration']:.2f}s")

    for browser, results in combined_results["browsers"].items():
        if "summary" in results:
            summary = results["summary"]
            print(f"\n{browser.upper()} Browser:")
            print(f"  Passed: {summary.get('passed', 0)}")
            print(f"  Failed: {summary.get('failed', 0)}")
            print(f"  Skipped: {summary.get('skipped', 0)}")

    return combined_results["summary"]["failed"] == 0


def main():
    """Main function to run UI tests."""
    parser = argparse.ArgumentParser(description="Run UI tests with Docker")
    parser.add_argument(
        "--browser",
        choices=["chrome", "firefox", "both"],
        default="both",
        help="Browser to test with",
    )
    parser.add_argument(
        "--build", action="store_true", help="Build Docker image before running tests"
    )
    parser.add_argument(
        "--parallel", action="store_true", default=True, help="Run tests in parallel"
    )
    parser.add_argument(
        "--compose", action="store_true", help="Use Docker Compose to run tests"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Verbose output"
    )

    args = parser.parse_args()

    # Change to Docker directory
    docker_dir = Path(__file__).parent
    os.chdir(docker_dir)

    success = True

    if args.build:
        if not build_docker_image():
            return 1

    if args.compose:
        success = run_docker_compose_tests()
    else:
        if args.browser == "both":
            browsers = ["chrome", "firefox"]
        else:
            browsers = [args.browser]

        for browser in browsers:
            if not run_tests_in_docker(browser, args.parallel, args.verbose):
                success = False

        # Generate combined report
        if len(browsers) > 1:
            success = generate_combined_report(browsers) and success

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
