#!/usr/bin/env python3
"""
Comprehensive BDD Test Runner for Pynomaly
Executes behavior-driven development tests with detailed reporting
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class BDDTestRunner:
    """Enhanced BDD test runner with comprehensive reporting"""

    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.project_root = project_root
        self.results_dir = self.test_dir / "bdd_results"
        self.reports_dir = Path("test_reports") / "bdd"

        # Ensure directories exist
        self.results_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "features": {},
            "scenarios": {},
            "steps": {},
            "errors": [],
        }

    def discover_feature_files(self) -> List[Path]:
        """Discover all BDD feature files"""
        features_dir = self.test_dir / "bdd" / "features"
        if not features_dir.exists():
            print(f"Features directory not found: {features_dir}")
            return []

        feature_files = list(features_dir.glob("*.feature"))
        print(f"Discovered {len(feature_files)} feature files:")
        for feature_file in feature_files:
            print(f"  - {feature_file.name}")

        return feature_files

    def analyze_feature_files(self, feature_files: List[Path]) -> Dict[str, Any]:
        """Analyze feature files to extract scenarios and steps"""
        analysis = {
            "total_features": len(feature_files),
            "total_scenarios": 0,
            "total_steps": 0,
            "features": {},
        }

        for feature_file in feature_files:
            try:
                with open(feature_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Parse feature content
                lines = content.split("\n")
                feature_info = {
                    "name": "",
                    "description": "",
                    "scenarios": [],
                    "background_steps": [],
                    "tags": [],
                }

                current_scenario = None
                current_steps = []
                in_background = False

                for line in lines:
                    line = line.strip()

                    if line.startswith("Feature:"):
                        feature_info["name"] = line.replace("Feature:", "").strip()
                    elif line.startswith("Scenario:"):
                        if current_scenario and current_steps:
                            current_scenario["steps"] = current_steps
                            feature_info["scenarios"].append(current_scenario)

                        current_scenario = {
                            "name": line.replace("Scenario:", "").strip(),
                            "steps": [],
                        }
                        current_steps = []
                        in_background = False
                        analysis["total_scenarios"] += 1
                    elif line.startswith("Background:"):
                        in_background = True
                        current_steps = []
                    elif line.startswith(("Given ", "When ", "Then ", "And ", "But ")):
                        step = line
                        current_steps.append(step)
                        analysis["total_steps"] += 1

                        if in_background:
                            feature_info["background_steps"].append(step)
                    elif line.startswith("@"):
                        feature_info["tags"].extend(line.split())

                # Add last scenario
                if current_scenario and current_steps:
                    current_scenario["steps"] = current_steps
                    feature_info["scenarios"].append(current_scenario)

                analysis["features"][feature_file.name] = feature_info

            except Exception as e:
                print(f"Error analyzing {feature_file}: {e}")
                self.test_results["errors"].append(f"Feature analysis error: {e}")

        return analysis

    def run_pytest_bdd(self, test_categories: List[str] = None) -> Dict[str, Any]:
        """Run pytest-bdd tests with comprehensive reporting"""

        # Prepare pytest command
        pytest_cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(self.test_dir / "bdd"),
            "-v",
            "--tb=short",
            f"--junitxml={self.reports_dir}/bdd_junit.xml",
            f"--html={self.reports_dir}/bdd_report.html",
            "--self-contained-html",
            "-p",
            "no:warnings",
        ]

        # Add category filters if specified
        if test_categories:
            for category in test_categories:
                pytest_cmd.extend(["-k", category])

        print(f"Running BDD tests with command: {' '.join(pytest_cmd)}")

        try:
            # Run pytest
            result = subprocess.run(
                pytest_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            execution_result = {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
            }

            print(f"BDD tests completed with return code: {result.returncode}")
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)

            return execution_result

        except subprocess.TimeoutExpired:
            error_msg = "BDD tests timed out after 5 minutes"
            print(error_msg)
            self.test_results["errors"].append(error_msg)
            return {
                "return_code": -1,
                "stdout": "",
                "stderr": error_msg,
                "success": False,
            }
        except Exception as e:
            error_msg = f"Error running BDD tests: {e}"
            print(error_msg)
            self.test_results["errors"].append(error_msg)
            return {
                "return_code": -1,
                "stdout": "",
                "stderr": error_msg,
                "success": False,
            }

    def parse_junit_results(self) -> Dict[str, Any]:
        """Parse JUnit XML results for detailed analysis"""
        junit_file = self.reports_dir / "bdd_junit.xml"

        if not junit_file.exists():
            return {"error": "JUnit XML file not found"}

        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(junit_file)
            root = tree.getroot()

            results = {
                "total_tests": int(root.get("tests", 0)),
                "failures": int(root.get("failures", 0)),
                "errors": int(root.get("errors", 0)),
                "skipped": int(root.get("skipped", 0)),
                "time": float(root.get("time", 0)),
                "test_cases": [],
            }

            for testcase in root.findall(".//testcase"):
                case_info = {
                    "name": testcase.get("name"),
                    "classname": testcase.get("classname"),
                    "time": float(testcase.get("time", 0)),
                    "status": "passed",
                }

                if testcase.find("failure") is not None:
                    case_info["status"] = "failed"
                    case_info["failure"] = testcase.find("failure").text
                elif testcase.find("error") is not None:
                    case_info["status"] = "error"
                    case_info["error"] = testcase.find("error").text
                elif testcase.find("skipped") is not None:
                    case_info["status"] = "skipped"

                results["test_cases"].append(case_info)

            return results

        except Exception as e:
            return {"error": f"Error parsing JUnit XML: {e}"}

    def generate_bdd_report(
        self,
        feature_analysis: Dict[str, Any],
        execution_result: Dict[str, Any],
        junit_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate comprehensive BDD test report"""

        report = {
            "timestamp": datetime.now().isoformat(),
            "execution_summary": {
                "success": execution_result["success"],
                "return_code": execution_result["return_code"],
                "total_features": feature_analysis["total_features"],
                "total_scenarios": feature_analysis["total_scenarios"],
                "total_steps": feature_analysis["total_steps"],
            },
            "test_results": junit_results,
            "feature_analysis": feature_analysis,
            "execution_details": execution_result,
            "recommendations": [],
        }

        # Generate recommendations based on results
        if not execution_result["success"]:
            report["recommendations"].append(
                "Review failed tests and fix implementation issues"
            )

        if junit_results.get("failures", 0) > 0:
            report["recommendations"].append(
                f"Address {junit_results['failures']} test failures"
            )

        if junit_results.get("errors", 0) > 0:
            report["recommendations"].append(
                f"Fix {junit_results['errors']} test errors"
            )

        if feature_analysis["total_scenarios"] == 0:
            report["recommendations"].append(
                "Add more BDD scenarios to improve test coverage"
            )

        # Calculate success rate
        total_tests = junit_results.get("total_tests", 0)
        if total_tests > 0:
            passed_tests = (
                total_tests
                - junit_results.get("failures", 0)
                - junit_results.get("errors", 0)
            )
            success_rate = (passed_tests / total_tests) * 100
            report["execution_summary"]["success_rate"] = round(success_rate, 2)

        return report

    def save_results(self, report: Dict[str, Any]):
        """Save test results to files"""

        # Save JSON report
        json_file = self.reports_dir / "bdd_comprehensive_report.json"
        with open(json_file, "w") as f:
            json.dump(report, f, indent=2)

        # Save text summary
        summary_file = self.reports_dir / "bdd_summary.txt"
        with open(summary_file, "w") as f:
            f.write("BDD Test Execution Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {report['timestamp']}\n")
            f.write(f"Success: {report['execution_summary']['success']}\n")
            f.write(
                f"Total Features: {report['execution_summary']['total_features']}\n"
            )
            f.write(
                f"Total Scenarios: {report['execution_summary']['total_scenarios']}\n"
            )
            f.write(f"Total Steps: {report['execution_summary']['total_steps']}\n")

            if "success_rate" in report["execution_summary"]:
                f.write(
                    f"Success Rate: {report['execution_summary']['success_rate']}%\n"
                )

            f.write("\nTest Results:\n")
            test_results = report.get("test_results", {})
            f.write(f"  Total Tests: {test_results.get('total_tests', 0)}\n")
            f.write(
                f"  Passed: {test_results.get('total_tests', 0) - test_results.get('failures', 0) - test_results.get('errors', 0)}\n"
            )
            f.write(f"  Failed: {test_results.get('failures', 0)}\n")
            f.write(f"  Errors: {test_results.get('errors', 0)}\n")
            f.write(f"  Skipped: {test_results.get('skipped', 0)}\n")

            if report.get("recommendations"):
                f.write("\nRecommendations:\n")
                for i, rec in enumerate(report["recommendations"], 1):
                    f.write(f"  {i}. {rec}\n")

        print(f"BDD test results saved to:")
        print(f"  - JSON: {json_file}")
        print(f"  - Summary: {summary_file}")
        print(f"  - HTML: {self.reports_dir}/bdd_report.html")
        print(f"  - JUnit XML: {self.reports_dir}/bdd_junit.xml")

    def run_comprehensive_bdd_tests(
        self, test_categories: List[str] = None
    ) -> Dict[str, Any]:
        """Run comprehensive BDD test suite"""

        print("Starting Comprehensive BDD Test Execution")
        print("=" * 50)

        # Discover and analyze feature files
        feature_files = self.discover_feature_files()
        if not feature_files:
            print("No feature files found. Exiting.")
            return {"error": "No feature files found"}

        feature_analysis = self.analyze_feature_files(feature_files)
        print(
            f"Analysis complete: {feature_analysis['total_features']} features, "
            f"{feature_analysis['total_scenarios']} scenarios, "
            f"{feature_analysis['total_steps']} steps"
        )

        # Run pytest-bdd tests
        execution_result = self.run_pytest_bdd(test_categories)

        # Parse results
        junit_results = self.parse_junit_results()

        # Generate comprehensive report
        report = self.generate_bdd_report(
            feature_analysis, execution_result, junit_results
        )

        # Save results
        self.save_results(report)

        return report


def main():
    """Main entry point for BDD test runner"""

    import argparse

    parser = argparse.ArgumentParser(description="Run Pynomaly BDD Tests")
    parser.add_argument(
        "--categories",
        nargs="*",
        help="Test categories to run (e.g. workflow security performance)",
    )
    parser.add_argument(
        "--output-dir",
        default="test_reports/bdd",
        help="Output directory for test reports",
    )

    args = parser.parse_args()

    # Initialize test runner
    runner = BDDTestRunner()
    if args.output_dir:
        runner.reports_dir = Path(args.output_dir)
        runner.reports_dir.mkdir(parents=True, exist_ok=True)

    # Run tests
    try:
        report = runner.run_comprehensive_bdd_tests(args.categories)

        # Print summary
        print("\n" + "=" * 50)
        print("BDD Test Execution Complete")
        print("=" * 50)

        if "error" in report:
            print(f"Error: {report['error']}")
            return 1

        success = report["execution_summary"]["success"]
        print(f"Overall Success: {success}")

        if "success_rate" in report["execution_summary"]:
            print(f"Success Rate: {report['execution_summary']['success_rate']}%")

        test_results = report.get("test_results", {})
        total_tests = test_results.get("total_tests", 0)
        failures = test_results.get("failures", 0)
        errors = test_results.get("errors", 0)

        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_tests - failures - errors}")
        print(f"Failed: {failures}")
        print(f"Errors: {errors}")

        if report.get("recommendations"):
            print("\nRecommendations:")
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"  {i}. {rec}")

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\nBDD test execution interrupted by user")
        return 1
    except Exception as e:
        print(f"Error during BDD test execution: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
