#!/usr/bin/env python3
"""
Comprehensive Test Report Generator
Generates HTML reports from all test artifacts including coverage, performance, and security results.
"""

import argparse
import html
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path


class TestReportGenerator:
    """Generates comprehensive test reports from CI/CD artifacts."""

    def __init__(self, input_dir: Path, output_file: Path):
        self.input_dir = input_dir
        self.output_file = output_file
        self.test_results = {}
        self.coverage_data = {}
        self.performance_data = {}
        self.security_data = {}

    def collect_test_artifacts(self):
        """Collect all test artifacts from the input directory."""
        print("🔍 Collecting test artifacts...")

        # Collect JUnit XML files
        for junit_file in self.input_dir.rglob("junit-*.xml"):
            self._parse_junit_xml(junit_file)

        # Collect coverage XML files
        for coverage_file in self.input_dir.rglob("coverage-*.xml"):
            self._parse_coverage_xml(coverage_file)

        # Collect performance JSON files
        for perf_file in self.input_dir.rglob("benchmark-results.json"):
            self._parse_performance_json(perf_file)

        # Collect security reports
        for sec_file in self.input_dir.rglob("security-report.json"):
            self._parse_security_json(sec_file)

        print(f"✅ Collected {len(self.test_results)} test suites")

    def _parse_junit_xml(self, file_path: Path):
        """Parse JUnit XML test results."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            suite_name = root.get("name", file_path.stem)

            self.test_results[suite_name] = {
                "tests": int(root.get("tests", 0)),
                "failures": int(root.get("failures", 0)),
                "errors": int(root.get("errors", 0)),
                "skipped": int(root.get("skipped", 0)),
                "time": float(root.get("time", 0)),
                "testcases": [],
            }

            for testcase in root.findall(".//testcase"):
                case_data = {
                    "name": testcase.get("name"),
                    "classname": testcase.get("classname"),
                    "time": float(testcase.get("time", 0)),
                    "status": "passed",
                }

                if testcase.find("failure") is not None:
                    case_data["status"] = "failed"
                    case_data["failure"] = testcase.find("failure").text
                elif testcase.find("error") is not None:
                    case_data["status"] = "error"
                    case_data["error"] = testcase.find("error").text
                elif testcase.find("skipped") is not None:
                    case_data["status"] = "skipped"

                self.test_results[suite_name]["testcases"].append(case_data)

        except Exception as e:
            print(f"⚠️ Error parsing {file_path}: {e}")

    def _parse_coverage_xml(self, file_path: Path):
        """Parse coverage XML reports."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            coverage_type = file_path.stem.replace("coverage-", "")

            # Extract overall coverage metrics
            coverage_elem = root.find(".//coverage")
            if coverage_elem is not None:
                self.coverage_data[coverage_type] = {
                    "line_rate": float(coverage_elem.get("line-rate", 0)) * 100,
                    "branch_rate": float(coverage_elem.get("branch-rate", 0)) * 100,
                    "lines_covered": int(coverage_elem.get("lines-covered", 0)),
                    "lines_valid": int(coverage_elem.get("lines-valid", 0)),
                    "branches_covered": int(coverage_elem.get("branches-covered", 0)),
                    "branches_valid": int(coverage_elem.get("branches-valid", 0)),
                    "packages": [],
                }

                # Extract package-level details
                for package in root.findall(".//package"):
                    package_data = {
                        "name": package.get("name"),
                        "line_rate": float(package.get("line-rate", 0)) * 100,
                        "branch_rate": float(package.get("branch-rate", 0)) * 100,
                        "classes": [],
                    }

                    for class_elem in package.findall(".//class"):
                        class_data = {
                            "name": class_elem.get("name"),
                            "filename": class_elem.get("filename"),
                            "line_rate": float(class_elem.get("line-rate", 0)) * 100,
                            "branch_rate": float(class_elem.get("branch-rate", 0))
                            * 100,
                        }
                        package_data["classes"].append(class_data)

                    self.coverage_data[coverage_type]["packages"].append(package_data)

        except Exception as e:
            print(f"⚠️ Error parsing coverage {file_path}: {e}")

    def _parse_performance_json(self, file_path: Path):
        """Parse performance benchmark results."""
        try:
            with open(file_path) as f:
                data = json.load(f)

            self.performance_data = {
                "benchmarks": data.get("benchmarks", []),
                "machine_info": data.get("machine_info", {}),
                "commit_info": data.get("commit_info", {}),
            }

        except Exception as e:
            print(f"⚠️ Error parsing performance data {file_path}: {e}")

    def _parse_security_json(self, file_path: Path):
        """Parse security scan results."""
        try:
            with open(file_path) as f:
                data = json.load(f)

            self.security_data = {
                "results": data.get("results", []),
                "metrics": data.get("metrics", {}),
                "generated_at": data.get("generated_at", ""),
            }

        except Exception as e:
            print(f"⚠️ Error parsing security data {file_path}: {e}")

    def generate_html_report(
        self, include_coverage=True, include_performance=True, include_security=True
    ):
        """Generate comprehensive HTML test report."""
        print("📊 Generating HTML report...")

        html_content = self._generate_html_structure()
        html_content += self._generate_summary_section()
        html_content += self._generate_test_results_section()

        if include_coverage and self.coverage_data:
            html_content += self._generate_coverage_section()

        if include_performance and self.performance_data:
            html_content += self._generate_performance_section()

        if include_security and self.security_data:
            html_content += self._generate_security_section()

        html_content += self._generate_html_footer()

        # Write to file
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"✅ Report generated: {self.output_file}")

    def _generate_html_structure(self) -> str:
        """Generate HTML structure and CSS."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pynomaly Comprehensive Test Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #2196F3;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #1976D2;
            margin: 0;
            font-size: 2.5em;
        }}
        .header .subtitle {{
            color: #666;
            font-size: 1.1em;
            margin-top: 10px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .metric-card.success {{
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        }}
        .metric-card.warning {{
            background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
        }}
        .metric-card.error {{
            background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 0;
        }}
        .metric-label {{
            font-size: 1em;
            opacity: 0.9;
            margin-top: 5px;
        }}
        .section {{
            margin: 40px 0;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }}
        .section-header {{
            background: #f8f9fa;
            padding: 15px 25px;
            border-bottom: 1px solid #e0e0e0;
            font-size: 1.3em;
            font-weight: 600;
            color: #333;
        }}
        .section-content {{
            padding: 25px;
        }}
        .test-suite {{
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 6px;
        }}
        .test-suite-header {{
            background: #f1f3f4;
            padding: 12px 20px;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .status-badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .status-passed {{ background: #d4edda; color: #155724; }}
        .status-failed {{ background: #f8d7da; color: #721c24; }}
        .status-error {{ background: #fff3cd; color: #856404; }}
        .status-skipped {{ background: #e2e3e5; color: #383d41; }}
        .progress-bar {{
            background: #e0e0e0;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin: 15px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #45a049);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .table-responsive {{
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #333;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
        }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Pynomaly Test Report</h1>
            <div class="subtitle">Comprehensive Testing Results</div>
            <div class="timestamp">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}</div>
        </div>
"""

    def _generate_summary_section(self) -> str:
        """Generate executive summary section."""
        total_tests = sum(suite["tests"] for suite in self.test_results.values())
        total_failures = sum(suite["failures"] for suite in self.test_results.values())
        total_errors = sum(suite["errors"] for suite in self.test_results.values())
        total_passed = total_tests - total_failures - total_errors

        pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

        # Calculate overall coverage
        avg_line_coverage = 0
        avg_branch_coverage = 0
        if self.coverage_data:
            line_rates = [data["line_rate"] for data in self.coverage_data.values()]
            branch_rates = [data["branch_rate"] for data in self.coverage_data.values()]
            avg_line_coverage = sum(line_rates) / len(line_rates) if line_rates else 0
            avg_branch_coverage = (
                sum(branch_rates) / len(branch_rates) if branch_rates else 0
            )

        return f"""
        <div class="summary-grid">
            <div class="metric-card {"success" if pass_rate >= 95 else "warning" if pass_rate >= 80 else "error"}">
                <div class="metric-value">{pass_rate:.1f}%</div>
                <div class="metric-label">Test Pass Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{total_tests:,}</div>
                <div class="metric-label">Total Tests</div>
            </div>
            <div class="metric-card {"success" if avg_line_coverage >= 80 else "warning" if avg_line_coverage >= 60 else "error"}">
                <div class="metric-value">{avg_line_coverage:.1f}%</div>
                <div class="metric-label">Line Coverage</div>
            </div>
            <div class="metric-card {"success" if avg_branch_coverage >= 65 else "warning" if avg_branch_coverage >= 45 else "error"}">
                <div class="metric-value">{avg_branch_coverage:.1f}%</div>
                <div class="metric-label">Branch Coverage</div>
            </div>
        </div>
"""

    def _generate_test_results_section(self) -> str:
        """Generate detailed test results section."""
        content = """
        <div class="section">
            <div class="section-header">📋 Test Results by Suite</div>
            <div class="section-content">
"""

        for suite_name, results in self.test_results.items():
            pass_rate = (
                (
                    (results["tests"] - results["failures"] - results["errors"])
                    / results["tests"]
                    * 100
                )
                if results["tests"] > 0
                else 0
            )

            status_class = (
                "success"
                if results["failures"] == 0 and results["errors"] == 0
                else "error"
            )

            content += f"""
                <div class="test-suite">
                    <div class="test-suite-header">
                        <span>{suite_name}</span>
                        <span class="status-badge status-{"passed" if status_class == "success" else "failed"}">
                            {results["tests"]} tests, {results["failures"]} failures, {results["errors"]} errors
                        </span>
                    </div>
                    <div style="padding: 15px;">
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {pass_rate}%">{pass_rate:.1f}%</div>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 10px;">
                            <div>✅ Passed: {results["tests"] - results["failures"] - results["errors"]}</div>
                            <div>❌ Failed: {results["failures"]}</div>
                            <div>⚠️ Errors: {results["errors"]}</div>
                            <div>⏱️ Time: {results["time"]:.2f}s</div>
                        </div>
                    </div>
                </div>
"""

        content += """
            </div>
        </div>
"""
        return content

    def _generate_coverage_section(self) -> str:
        """Generate coverage analysis section."""
        content = """
        <div class="section">
            <div class="section-header">📊 Code Coverage Analysis</div>
            <div class="section-content">
                <div class="table-responsive">
                    <table>
                        <thead>
                            <tr>
                                <th>Component</th>
                                <th>Line Coverage</th>
                                <th>Branch Coverage</th>
                                <th>Lines Covered</th>
                                <th>Branches Covered</th>
                            </tr>
                        </thead>
                        <tbody>
"""

        for component, data in self.coverage_data.items():
            line_coverage = data["line_rate"]
            branch_coverage = data["branch_rate"]

            content += f"""
                            <tr>
                                <td><strong>{component.title()}</strong></td>
                                <td>
                                    <div class="progress-bar" style="width: 100px; height: 15px;">
                                        <div class="progress-fill" style="width: {line_coverage}%; font-size: 0.75em;">
                                            {line_coverage:.1f}%
                                        </div>
                                    </div>
                                </td>
                                <td>
                                    <div class="progress-bar" style="width: 100px; height: 15px;">
                                        <div class="progress-fill" style="width: {branch_coverage}%; font-size: 0.75em;">
                                            {branch_coverage:.1f}%
                                        </div>
                                    </div>
                                </td>
                                <td>{data["lines_covered"]} / {data["lines_valid"]}</td>
                                <td>{data["branches_covered"]} / {data["branches_valid"]}</td>
                            </tr>
"""

        content += """
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
"""
        return content

    def _generate_performance_section(self) -> str:
        """Generate performance benchmark section."""
        if not self.performance_data.get("benchmarks"):
            return ""

        content = """
        <div class="section">
            <div class="section-header">⚡ Performance Benchmarks</div>
            <div class="section-content">
                <div class="table-responsive">
                    <table>
                        <thead>
                            <tr>
                                <th>Benchmark</th>
                                <th>Mean Time</th>
                                <th>Min Time</th>
                                <th>Max Time</th>
                                <th>Std Dev</th>
                                <th>Operations/sec</th>
                            </tr>
                        </thead>
                        <tbody>
"""

        for benchmark in self.performance_data["benchmarks"]:
            stats = benchmark.get("stats", {})
            mean_time = stats.get("mean", 0)
            min_time = stats.get("min", 0)
            max_time = stats.get("max", 0)
            stddev = stats.get("stddev", 0)
            ops_per_sec = 1 / mean_time if mean_time > 0 else 0

            content += f"""
                            <tr>
                                <td><strong>{html.escape(benchmark.get("name", "Unknown"))}</strong></td>
                                <td>{mean_time:.4f}s</td>
                                <td>{min_time:.4f}s</td>
                                <td>{max_time:.4f}s</td>
                                <td>{stddev:.4f}s</td>
                                <td>{ops_per_sec:.2f}</td>
                            </tr>
"""

        content += """
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
"""
        return content

    def _generate_security_section(self) -> str:
        """Generate security scan results section."""
        if not self.security_data.get("results"):
            return ""

        # Count security issues by severity
        severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for result in self.security_data["results"]:
            severity = result.get("issue_severity", "LOW").upper()
            if severity in severity_counts:
                severity_counts[severity] += 1

        content = f"""
        <div class="section">
            <div class="section-header">🔒 Security Scan Results</div>
            <div class="section-content">
                <div class="summary-grid">
                    <div class="metric-card {"error" if severity_counts["HIGH"] > 0 else "success"}">
                        <div class="metric-value">{severity_counts["HIGH"]}</div>
                        <div class="metric-label">High Severity</div>
                    </div>
                    <div class="metric-card {"warning" if severity_counts["MEDIUM"] > 0 else "success"}">
                        <div class="metric-value">{severity_counts["MEDIUM"]}</div>
                        <div class="metric-label">Medium Severity</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{severity_counts["LOW"]}</div>
                        <div class="metric-label">Low Severity</div>
                    </div>
                </div>
"""

        if any(severity_counts.values()):
            content += """
                <div class="table-responsive">
                    <table>
                        <thead>
                            <tr>
                                <th>Severity</th>
                                <th>Issue</th>
                                <th>File</th>
                                <th>Line</th>
                                <th>Description</th>
                            </tr>
                        </thead>
                        <tbody>
"""

            for result in self.security_data["results"]:
                severity = result.get("issue_severity", "LOW").upper()
                issue_text = result.get("issue_text", "Unknown issue")
                filename = result.get("filename", "Unknown file")
                line_number = result.get("line_number", "N/A")
                description = result.get("issue_description", "")

                severity_class = (
                    "error"
                    if severity == "HIGH"
                    else "warning"
                    if severity == "MEDIUM"
                    else ""
                )

                content += f"""
                            <tr>
                                <td><span class="status-badge status-{severity_class}">{severity}</span></td>
                                <td>{html.escape(issue_text)}</td>
                                <td>{html.escape(filename)}</td>
                                <td>{line_number}</td>
                                <td>{html.escape(description)}</td>
                            </tr>
"""

            content += """
                        </tbody>
                    </table>
                </div>
"""

        content += """
            </div>
        </div>
"""
        return content

    def _generate_html_footer(self) -> str:
        """Generate HTML footer."""
        return """
        <div class="footer">
            <p>Generated by Pynomaly Comprehensive Test Report Generator</p>
            <p>🎯 Enterprise-Grade Anomaly Detection Platform</p>
        </div>
    </div>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive test report")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing test artifacts",
    )
    parser.add_argument(
        "--output-file", type=Path, required=True, help="Output HTML report file"
    )
    parser.add_argument(
        "--include-coverage",
        action="store_true",
        default=True,
        help="Include coverage analysis",
    )
    parser.add_argument(
        "--include-performance",
        action="store_true",
        default=True,
        help="Include performance benchmarks",
    )
    parser.add_argument(
        "--include-security",
        action="store_true",
        default=True,
        help="Include security scan results",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    # Generate report
    generator = TestReportGenerator(args.input_dir, args.output_file)
    generator.collect_test_artifacts()
    generator.generate_html_report(
        include_coverage=args.include_coverage,
        include_performance=args.include_performance,
        include_security=args.include_security,
    )


if __name__ == "__main__":
    main()
