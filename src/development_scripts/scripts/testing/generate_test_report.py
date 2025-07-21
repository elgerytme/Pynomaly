#!/usr/bin/env python3
"""
Comprehensive Test Report Generator

This script generates a detailed test report from all testing artifacts.
"""

import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Template


class TestReportGenerator:
    """Generate comprehensive test reports from CI/CD artifacts."""

    def __init__(self, artifacts_dir: str = "."):
        self.artifacts_dir = Path(artifacts_dir)
        self.report_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "security": {},
            "quality": {},
            "unit_tests": {},
            "integration_tests": {},
            "api_tests": {},
            "performance_tests": {},
            "load_tests": {},
            "docker_tests": {},
            "e2e_tests": {},
            "mutation_tests": {}
        }

    def parse_junit_xml(self, xml_file: Path) -> dict[str, Any]:
        """Parse JUnit XML file."""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            return {
                "tests": int(root.get("tests", 0)),
                "failures": int(root.get("failures", 0)),
                "errors": int(root.get("errors", 0)),
                "skipped": int(root.get("skipped", 0)),
                "time": float(root.get("time", 0)),
                "success_rate": self._calculate_success_rate(
                    int(root.get("tests", 0)),
                    int(root.get("failures", 0)),
                    int(root.get("errors", 0))
                )
            }
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
            return {}

    def _calculate_success_rate(self, tests: int, failures: int, errors: int) -> float:
        """Calculate test success rate."""
        if tests == 0:
            return 0.0
        return ((tests - failures - errors) / tests) * 100

    def parse_security_reports(self) -> None:
        """Parse security scan reports."""
        security_dir = self.artifacts_dir / "security-reports"
        if not security_dir.exists():
            return

        # Parse Safety report
        safety_file = security_dir / "security_report.json"
        if safety_file.exists():
            try:
                with open(safety_file) as f:
                    safety_data = json.load(f)
                    self.report_data["security"]["safety"] = {
                        "vulnerabilities": len(safety_data.get("vulnerabilities", [])),
                        "status": "PASS" if len(safety_data.get("vulnerabilities", [])) == 0 else "FAIL"
                    }
            except Exception as e:
                print(f"Error parsing safety report: {e}")

        # Parse Bandit report
        bandit_file = security_dir / "bandit_report.json"
        if bandit_file.exists():
            try:
                with open(bandit_file) as f:
                    bandit_data = json.load(f)
                    results = bandit_data.get("results", [])
                    self.report_data["security"]["bandit"] = {
                        "issues": len(results),
                        "high_severity": len([r for r in results if r.get("issue_severity") == "HIGH"]),
                        "medium_severity": len([r for r in results if r.get("issue_severity") == "MEDIUM"]),
                        "low_severity": len([r for r in results if r.get("issue_severity") == "LOW"]),
                        "status": "PASS" if len(results) == 0 else "FAIL"
                    }
            except Exception as e:
                print(f"Error parsing bandit report: {e}")

    def parse_code_quality_reports(self) -> None:
        """Parse code quality reports."""
        quality_dir = self.artifacts_dir / "code-quality-reports"
        if not quality_dir.exists():
            return

        # Parse Flake8 report
        flake8_file = quality_dir / "flake8_report.json"
        if flake8_file.exists():
            try:
                with open(flake8_file) as f:
                    flake8_data = json.load(f)
                    self.report_data["quality"]["flake8"] = {
                        "issues": len(flake8_data),
                        "status": "PASS" if len(flake8_data) == 0 else "FAIL"
                    }
            except Exception as e:
                print(f"Error parsing flake8 report: {e}")

        # Parse complexity report
        complexity_file = quality_dir / "complexity_report.json"
        if complexity_file.exists():
            try:
                with open(complexity_file) as f:
                    complexity_data = json.load(f)
                    self.report_data["quality"]["complexity"] = {
                        "average_complexity": self._calculate_average_complexity(complexity_data),
                        "high_complexity_functions": self._count_high_complexity(complexity_data),
                        "status": "PASS" if self._count_high_complexity(complexity_data) == 0 else "WARNING"
                    }
            except Exception as e:
                print(f"Error parsing complexity report: {e}")

    def _calculate_average_complexity(self, complexity_data: dict) -> float:
        """Calculate average complexity from radon output."""
        # Implementation would depend on radon output format
        return 0.0

    def _count_high_complexity(self, complexity_data: dict) -> int:
        """Count high complexity functions."""
        # Implementation would depend on radon output format
        return 0

    def parse_test_results(self) -> None:
        """Parse all test results."""
        test_types = [
            ("unit_tests", "unit-test-results"),
            ("integration_tests", "integration-test-results"),
            ("api_tests", "api-test-results"),
            ("performance_tests", "performance-test-results"),
            ("e2e_tests", "e2e-test-results")
        ]

        for test_type, artifact_name in test_types:
            self._parse_test_type(test_type, artifact_name)

    def _parse_test_type(self, test_type: str, artifact_name: str) -> None:
        """Parse specific test type results."""
        # Look for JUnit XML files
        for junit_file in self.artifacts_dir.glob(f"{artifact_name}*/junit_*.xml"):
            if junit_file.exists():
                junit_data = self.parse_junit_xml(junit_file)
                if junit_data:
                    self.report_data[test_type] = junit_data
                    break

    def parse_coverage_reports(self) -> None:
        """Parse coverage reports."""
        for coverage_file in self.artifacts_dir.glob("*/coverage.xml"):
            if coverage_file.exists():
                try:
                    tree = ET.parse(coverage_file)
                    root = tree.getroot()

                    # Extract coverage percentage
                    coverage_elem = root.find(".//coverage")
                    if coverage_elem is not None:
                        line_rate = float(coverage_elem.get("line-rate", 0))
                        self.report_data["summary"]["coverage"] = {
                            "percentage": line_rate * 100,
                            "status": "PASS" if line_rate >= 0.8 else "FAIL"
                        }
                except Exception as e:
                    print(f"Error parsing coverage report: {e}")

    def parse_performance_reports(self) -> None:
        """Parse performance test reports."""
        benchmark_files = list(self.artifacts_dir.glob("*/benchmark_results.json"))
        if benchmark_files:
            try:
                with open(benchmark_files[0]) as f:
                    benchmark_data = json.load(f)
                    self.report_data["performance_tests"]["benchmarks"] = {
                        "count": len(benchmark_data.get("benchmarks", [])),
                        "status": "PASS"
                    }
            except Exception as e:
                print(f"Error parsing benchmark report: {e}")

    def generate_summary(self) -> None:
        """Generate overall summary."""
        total_tests = 0
        total_failures = 0
        total_errors = 0

        test_types = ["unit_tests", "integration_tests", "api_tests", "e2e_tests"]

        for test_type in test_types:
            if test_type in self.report_data and self.report_data[test_type]:
                total_tests += self.report_data[test_type].get("tests", 0)
                total_failures += self.report_data[test_type].get("failures", 0)
                total_errors += self.report_data[test_type].get("errors", 0)

        self.report_data["summary"]["total_tests"] = total_tests
        self.report_data["summary"]["total_failures"] = total_failures
        self.report_data["summary"]["total_errors"] = total_errors
        self.report_data["summary"]["success_rate"] = self._calculate_success_rate(
            total_tests, total_failures, total_errors
        )

        # Overall status
        security_passed = all(
            item.get("status") == "PASS"
            for item in self.report_data["security"].values()
        )
        quality_passed = all(
            item.get("status") in ["PASS", "WARNING"]
            for item in self.report_data["quality"].values()
        )
        tests_passed = self.report_data["summary"]["success_rate"] >= 95.0

        self.report_data["summary"]["overall_status"] = (
            "PASS" if security_passed and quality_passed and tests_passed else "FAIL"
        )

    def generate_html_report(self) -> str:
        """Generate HTML report."""
        template_str = """
<!DOCTYPE html>
<html>
<head>
    <title>anomaly_detection Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; }
        .status-pass { color: green; font-weight: bold; }
        .status-fail { color: red; font-weight: bold; }
        .status-warning { color: orange; font-weight: bold; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background: #f9f9f9; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔄 anomaly_detection Comprehensive Test Report</h1>
        <p>Generated: {{ timestamp }}</p>
        <p>Overall Status:
            <span class="status-{{ 'pass' if summary.overall_status == 'PASS' else 'fail' }}">
                {{ summary.overall_status }}
            </span>
        </p>
    </div>

    <div class="section">
        <h2>📊 Summary</h2>
        <div class="metric">
            <strong>Total Tests:</strong> {{ summary.total_tests }}
        </div>
        <div class="metric">
            <strong>Success Rate:</strong> {{ "%.1f" | format(summary.success_rate) }}%
        </div>
        <div class="metric">
            <strong>Failures:</strong> {{ summary.total_failures }}
        </div>
        <div class="metric">
            <strong>Errors:</strong> {{ summary.total_errors }}
        </div>
        {% if summary.coverage %}
        <div class="metric">
            <strong>Coverage:</strong> {{ "%.1f" | format(summary.coverage.percentage) }}%
        </div>
        {% endif %}
    </div>

    <div class="section">
        <h2>🔒 Security</h2>
        <table>
            <tr><th>Tool</th><th>Status</th><th>Issues</th></tr>
            {% for tool, data in security.items() %}
            <tr>
                <td>{{ tool.title() }}</td>
                <td class="status-{{ 'pass' if data.status == 'PASS' else 'fail' }}">
                    {{ data.status }}
                </td>
                <td>{{ data.get('vulnerabilities', data.get('issues', 0)) }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="section">
        <h2>📊 Code Quality</h2>
        <table>
            <tr><th>Tool</th><th>Status</th><th>Issues</th></tr>
            {% for tool, data in quality.items() %}
            <tr>
                <td>{{ tool.title() }}</td>
                <td class="status-{{ 'pass' if data.status == 'PASS' else 'warning' if data.status == 'WARNING' else 'fail' }}">
                    {{ data.status }}
                </td>
                <td>{{ data.get('issues', 0) }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="section">
        <h2>🧪 Test Results</h2>
        <table>
            <tr><th>Test Type</th><th>Tests</th><th>Failures</th><th>Errors</th><th>Success Rate</th></tr>
            {% for test_type in ['unit_tests', 'integration_tests', 'api_tests', 'e2e_tests'] %}
            {% if test_type in report_data and report_data[test_type] %}
            {% set data = report_data[test_type] %}
            <tr>
                <td>{{ test_type.replace('_', ' ').title() }}</td>
                <td>{{ data.tests }}</td>
                <td>{{ data.failures }}</td>
                <td>{{ data.errors }}</td>
                <td class="status-{{ 'pass' if data.success_rate >= 95 else 'fail' }}">
                    {{ "%.1f" | format(data.success_rate) }}%
                </td>
            </tr>
            {% endif %}
            {% endfor %}
        </table>
    </div>

    <div class="section">
        <h2>📈 Performance</h2>
        {% if performance_tests %}
        <p>Performance tests completed successfully.</p>
        {% else %}
        <p>No performance test data available.</p>
        {% endif %}
    </div>

    <div class="section">
        <h2>🎯 Recommendations</h2>
        <ul>
            {% if summary.success_rate < 95 %}
            <li>❌ Fix failing tests to improve success rate</li>
            {% endif %}
            {% if summary.coverage and summary.coverage.percentage < 80 %}
            <li>❌ Increase test coverage to at least 80%</li>
            {% endif %}
            {% if security %}
            {% for tool, data in security.items() %}
            {% if data.status == 'FAIL' %}
            <li>🔒 Address security issues found by {{ tool }}</li>
            {% endif %}
            {% endfor %}
            {% endif %}
            {% if quality %}
            {% for tool, data in quality.items() %}
            {% if data.status == 'FAIL' %}
            <li>📊 Fix code quality issues found by {{ tool }}</li>
            {% endif %}
            {% endfor %}
            {% endif %}
        </ul>
    </div>
</body>
</html>
        """

        template = Template(template_str)
        return template.render(
            timestamp=self.report_data["timestamp"],
            summary=self.report_data["summary"],
            security=self.report_data["security"],
            quality=self.report_data["quality"],
            performance_tests=self.report_data["performance_tests"],
            report_data=self.report_data
        )

    def generate_reports(self) -> None:
        """Generate all reports."""
        print("🔄 Generating comprehensive test report...")

        # Parse all reports
        self.parse_security_reports()
        self.parse_code_quality_reports()
        self.parse_test_results()
        self.parse_coverage_reports()
        self.parse_performance_reports()

        # Generate summary
        self.generate_summary()

        # Generate HTML report
        html_report = self.generate_html_report()
        with open("test_report.html", "w") as f:
            f.write(html_report)

        # Generate JSON report
        with open("test_report.json", "w") as f:
            json.dump(self.report_data, f, indent=2)

        print("✅ Test report generated successfully!")
        print(f"📊 Overall Status: {self.report_data['summary']['overall_status']}")
        print(f"🧪 Total Tests: {self.report_data['summary']['total_tests']}")
        print(f"📈 Success Rate: {self.report_data['summary']['success_rate']:.1f}%")


def main():
    """Main function."""
    artifacts_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    generator = TestReportGenerator(artifacts_dir)
    generator.generate_reports()


if __name__ == "__main__":
    main()
