#!/usr/bin/env python3
"""
Automated Test Coverage Analysis Script
Generates comprehensive test coverage reports and identifies gaps automatically.
"""

import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import click


class TestCoverageAnalyzer:
    """Automated test coverage analysis and reporting."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src" / "anomaly_detection"
        self.tests_dir = self.project_root / "tests"
        self.reports_dir = self.project_root / "reports"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Ensure reports directory exists
        self.reports_dir.mkdir(exist_ok=True)

    def analyze_file_structure(self) -> dict[str, Any]:
        """Analyze the file structure and count source/test files."""
        click.echo("🔍 Analyzing file structure...")

        structure = {
            "timestamp": self.timestamp,
            "source_files": {},
            "test_files": {},
            "coverage_ratios": {},
            "areas": {},
            "layers": {},
            "test_types": {},
        }

        # Analyze source files by area and layer
        structure["source_files"] = self._analyze_source_files()
        structure["test_files"] = self._analyze_test_files()
        structure["coverage_ratios"] = self._calculate_coverage_ratios(
            structure["source_files"], structure["test_files"]
        )
        structure["areas"] = self._analyze_by_area()
        structure["layers"] = self._analyze_by_layer()
        structure["test_types"] = self._analyze_test_types()

        return structure

    def _analyze_source_files(self) -> dict[str, Any]:
        """Analyze source files by category."""
        source_analysis = {
            "total": 0,
            "by_area": defaultdict(int),
            "by_layer": defaultdict(int),
            "files": [],
        }

        for py_file in self.src_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            rel_path = py_file.relative_to(self.src_dir)
            source_analysis["total"] += 1
            source_analysis["files"].append(str(rel_path))

            # Categorize by layer
            if "domain" in str(rel_path):
                source_analysis["by_layer"]["domain"] += 1
            elif "application" in str(rel_path):
                source_analysis["by_layer"]["application"] += 1
            elif "infrastructure" in str(rel_path):
                source_analysis["by_layer"]["infrastructure"] += 1
            elif "presentation" in str(rel_path):
                source_analysis["by_layer"]["presentation"] += 1

            # Categorize by area
            if "cli" in str(rel_path):
                source_analysis["by_area"]["cli"] += 1
            elif "api" in str(rel_path):
                source_analysis["by_area"]["web_api"] += 1
            elif "web" in str(rel_path):
                source_analysis["by_area"]["web_ui"] += 1
            elif any(x in str(rel_path) for x in ["domain", "application"]):
                source_analysis["by_area"]["core"] += 1
            elif "sdk" in str(rel_path) or "shared" in str(rel_path):
                source_analysis["by_area"]["sdk"] += 1

        return dict(source_analysis)

    def _analyze_test_files(self) -> dict[str, Any]:
        """Analyze test files by category."""
        test_analysis = {
            "total": 0,
            "by_type": defaultdict(int),
            "by_layer": defaultdict(int),
            "by_area": defaultdict(int),
            "files": [],
        }

        for py_file in self.tests_dir.rglob("*.py"):
            if "__pycache__" in str(py_file) or "conftest.py" in str(py_file):
                continue

            rel_path = py_file.relative_to(self.tests_dir)
            test_analysis["total"] += 1
            test_analysis["files"].append(str(rel_path))

            # Categorize by test type
            if "unit" in str(rel_path):
                test_analysis["by_type"]["unit"] += 1
            elif "integration" in str(rel_path):
                test_analysis["by_type"]["integration"] += 1
            elif "e2e" in str(rel_path):
                test_analysis["by_type"]["e2e"] += 1
            elif "performance" in str(rel_path):
                test_analysis["by_type"]["performance"] += 1
            elif "ui" in str(rel_path):
                test_analysis["by_type"]["ui"] += 1
            elif "bdd" in str(rel_path):
                test_analysis["by_type"]["bdd"] += 1
            elif "regression" in str(rel_path):
                test_analysis["by_type"]["regression"] += 1

            # Categorize by layer (for unit tests)
            if "domain" in str(rel_path):
                test_analysis["by_layer"]["domain"] += 1
            elif "application" in str(rel_path):
                test_analysis["by_layer"]["application"] += 1
            elif "infrastructure" in str(rel_path):
                test_analysis["by_layer"]["infrastructure"] += 1
            elif "presentation" in str(rel_path):
                test_analysis["by_layer"]["presentation"] += 1

            # Categorize by area
            if "cli" in str(rel_path):
                test_analysis["by_area"]["cli"] += 1
            elif "api" in str(rel_path):
                test_analysis["by_area"]["web_api"] += 1
            elif "ui" in str(rel_path):
                test_analysis["by_area"]["web_ui"] += 1
            elif "sdk" in str(rel_path):
                test_analysis["by_area"]["sdk"] += 1
            elif any(x in str(rel_path) for x in ["domain", "application", "unit"]):
                test_analysis["by_area"]["core"] += 1

        return dict(test_analysis)

    def _calculate_coverage_ratios(
        self, source_files: dict, test_files: dict
    ) -> dict[str, float]:
        """Calculate coverage ratios."""
        ratios = {}

        # Overall ratio
        ratios["overall"] = (
            (test_files["total"] / source_files["total"]) * 100
            if source_files["total"] > 0
            else 0
        )

        # By layer
        for layer in ["domain", "application", "infrastructure", "presentation"]:
            source_count = source_files["by_layer"].get(layer, 0)
            test_count = test_files["by_layer"].get(layer, 0)
            ratios[f"layer_{layer}"] = (
                (test_count / source_count) * 100 if source_count > 0 else 0
            )

        # By area
        for area in ["core", "sdk", "cli", "web_api", "web_ui"]:
            source_count = source_files["by_area"].get(area, 0)
            test_count = test_files["by_area"].get(area, 0)
            ratios[f"area_{area}"] = (
                (test_count / source_count) * 100 if source_count > 0 else 0
            )

        return ratios

    def _analyze_by_area(self) -> dict[str, Any]:
        """Analyze coverage by functional area."""
        return {
            "core": {
                "description": "Domain and application logic",
                "priority": "high",
                "target_coverage": 80,
            },
            "sdk": {
                "description": "SDK and library interfaces",
                "priority": "medium",
                "target_coverage": 90,
            },
            "cli": {
                "description": "Command line interface",
                "priority": "critical",
                "target_coverage": 60,
            },
            "web_api": {
                "description": "Web API endpoints",
                "priority": "high",
                "target_coverage": 80,
            },
            "web_ui": {
                "description": "Web user interface",
                "priority": "medium",
                "target_coverage": 70,
            },
        }

    def _analyze_by_layer(self) -> dict[str, Any]:
        """Analyze coverage by architectural layer."""
        return {
            "domain": {
                "description": "Business logic and domain entities",
                "priority": "critical",
                "target_coverage": 90,
            },
            "application": {
                "description": "Application services and use cases",
                "priority": "high",
                "target_coverage": 80,
            },
            "infrastructure": {
                "description": "External integrations and persistence",
                "priority": "high",
                "target_coverage": 70,
            },
            "presentation": {
                "description": "APIs, CLI, and UI interfaces",
                "priority": "medium",
                "target_coverage": 60,
            },
        }

    def _analyze_test_types(self) -> dict[str, Any]:
        """Analyze test types and their characteristics."""
        return {
            "unit": {
                "description": "Isolated component testing",
                "quality_indicators": ["mocking", "isolation", "fast_execution"],
            },
            "integration": {
                "description": "Component interaction testing",
                "quality_indicators": ["cross_component", "realistic_scenarios"],
            },
            "e2e": {
                "description": "End-to-end workflow testing",
                "quality_indicators": ["complete_workflows", "real_data"],
            },
            "performance": {
                "description": "Performance and benchmarking",
                "quality_indicators": ["throughput", "memory_usage", "scalability"],
            },
            "ui": {
                "description": "User interface automation",
                "quality_indicators": ["cross_browser", "accessibility", "responsive"],
            },
            "bdd": {
                "description": "Behavior driven development",
                "quality_indicators": ["user_stories", "gherkin_scenarios"],
            },
            "regression": {
                "description": "Regression prevention",
                "quality_indicators": ["baseline_comparison", "automated_detection"],
            },
        }

    def run_coverage_tests(self) -> dict[str, Any]:
        """Run actual test coverage analysis."""
        click.echo("🧪 Running coverage tests...")

        coverage_results = {
            "timestamp": self.timestamp,
            "test_execution": {},
            "coverage_metrics": {},
            "quality_metrics": {},
        }

        try:
            # Run pytest with coverage
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                "--cov=anomaly_detection",
                "--cov-report=json",
                "--cov-report=html",
                "--cov-report=xml",
                "--tb=short",
                "-v",
                str(self.tests_dir),
            ]

            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
            )

            coverage_results["test_execution"] = {
                "return_code": result.returncode,
                "stdout_lines": len(result.stdout.splitlines()),
                "stderr_lines": len(result.stderr.splitlines()),
                "success": result.returncode == 0,
            }

            # Parse coverage JSON if available
            coverage_json_path = self.project_root / "coverage.json"
            if coverage_json_path.exists():
                with open(coverage_json_path) as f:
                    coverage_data = json.load(f)
                    coverage_results["coverage_metrics"] = {
                        "line_coverage": coverage_data["totals"]["percent_covered"],
                        "branch_coverage": coverage_data["totals"].get(
                            "percent_covered_display", "N/A"
                        ),
                        "total_lines": coverage_data["totals"]["num_statements"],
                        "covered_lines": coverage_data["totals"]["covered_lines"],
                    }

        except subprocess.TimeoutExpired:
            coverage_results["test_execution"] = {
                "return_code": -1,
                "error": "Test execution timed out after 5 minutes",
                "success": False,
            }
        except Exception as e:
            coverage_results["test_execution"] = {
                "return_code": -1,
                "error": str(e),
                "success": False,
            }

        return coverage_results

    def identify_gaps(
        self, structure: dict[str, Any], coverage_results: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Identify coverage gaps and create improvement recommendations."""
        click.echo("📊 Identifying coverage gaps...")

        gaps = []

        # Check coverage ratios against targets
        ratios = structure["coverage_ratios"]
        areas = structure["areas"]
        layers = structure["layers"]

        # Area gaps
        for area, config in areas.items():
            ratio_key = f"area_{area}"
            current_coverage = ratios.get(ratio_key, 0)
            target_coverage = config["target_coverage"]

            if current_coverage < target_coverage:
                gap = {
                    "type": "area",
                    "category": area,
                    "current_coverage": current_coverage,
                    "target_coverage": target_coverage,
                    "gap_percentage": target_coverage - current_coverage,
                    "priority": config["priority"],
                    "description": config["description"],
                    "recommendations": self._get_area_recommendations(
                        area, current_coverage, target_coverage
                    ),
                }
                gaps.append(gap)

        # Layer gaps
        for layer, config in layers.items():
            ratio_key = f"layer_{layer}"
            current_coverage = ratios.get(ratio_key, 0)
            target_coverage = config["target_coverage"]

            if current_coverage < target_coverage:
                gap = {
                    "type": "layer",
                    "category": layer,
                    "current_coverage": current_coverage,
                    "target_coverage": target_coverage,
                    "gap_percentage": target_coverage - current_coverage,
                    "priority": config["priority"],
                    "description": config["description"],
                    "recommendations": self._get_layer_recommendations(
                        layer, current_coverage, target_coverage
                    ),
                }
                gaps.append(gap)

        # Test type gaps
        test_types = structure["test_files"]["by_type"]
        if test_types.get("unit", 0) == 0:
            gaps.append(
                {
                    "type": "test_type",
                    "category": "system",
                    "current_coverage": 0,
                    "target_coverage": 100,
                    "gap_percentage": 100,
                    "priority": "critical",
                    "description": "Missing dedicated system test category",
                    "recommendations": [
                        "Create tests/system/ directory",
                        "Add end-to-end system tests",
                        "Implement deployment validation tests",
                    ],
                }
            )

        return sorted(
            gaps,
            key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}[
                x["priority"]
            ],
        )

    def _get_area_recommendations(
        self, area: str, current: float, target: float
    ) -> list[str]:
        """Get specific recommendations for area improvements."""
        recommendations = {
            "cli": [
                "Create comprehensive command-specific tests",
                "Add CLI workflow integration tests",
                "Implement argument validation testing",
                "Add help system validation",
            ],
            "core": [
                "Expand domain service testing",
                "Add more use case integration tests",
                "Implement property-based testing",
                "Add business rule validation tests",
            ],
            "web_api": [
                "Add more endpoint-specific tests",
                "Implement comprehensive error handling tests",
                "Add authentication/authorization tests",
                "Create API contract tests",
            ],
            "web_ui": [
                "Expand component testing",
                "Add visual regression tests",
                "Implement accessibility testing",
                "Add responsive design tests",
            ],
            "sdk": [
                "Add more integration scenarios",
                "Implement client testing",
                "Add protocol compliance tests",
                "Create SDK documentation tests",
            ],
        }
        return recommendations.get(
            area, ["Add comprehensive testing", "Improve test coverage"]
        )

    def _get_layer_recommendations(
        self, layer: str, current: float, target: float
    ) -> list[str]:
        """Get specific recommendations for layer improvements."""
        recommendations = {
            "domain": [
                "Add comprehensive entity testing",
                "Implement value object validation tests",
                "Add domain service testing",
                "Create business rule tests",
            ],
            "application": [
                "Add use case integration tests",
                "Implement service orchestration tests",
                "Add workflow validation tests",
                "Create cross-service tests",
            ],
            "infrastructure": [
                "Add repository integration tests",
                "Implement external service tests",
                "Add caching layer tests",
                "Create database tests",
            ],
            "presentation": [
                "Add controller testing",
                "Implement request/response validation",
                "Add authentication tests",
                "Create interface tests",
            ],
        }
        return recommendations.get(
            layer, ["Add comprehensive testing", "Improve test coverage"]
        )

    def generate_reports(
        self,
        structure: dict[str, Any],
        coverage_results: dict[str, Any],
        gaps: list[dict[str, Any]],
    ) -> None:
        """Generate comprehensive reports."""
        click.echo("📝 Generating reports...")

        # Generate summary report
        summary_file = self.reports_dir / f"test_coverage_summary_{self.timestamp}.json"
        summary = {
            "metadata": {
                "timestamp": self.timestamp,
                "project_root": str(self.project_root),
                "generated_by": "automated_test_coverage_analysis.py",
            },
            "structure": structure,
            "coverage_results": coverage_results,
            "gaps": gaps,
            "recommendations": self._generate_prioritized_recommendations(gaps),
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        click.echo(f"✅ Summary report saved to: {summary_file}")

        # Generate markdown report
        self._generate_markdown_report(summary)

    def _generate_prioritized_recommendations(
        self, gaps: list[dict[str, Any]]
    ) -> dict[str, list[str]]:
        """Generate prioritized recommendations."""
        recommendations = {"critical": [], "high": [], "medium": [], "low": []}

        for gap in gaps:
            priority = gap["priority"]
            rec = f"{gap['type'].title()} {gap['category']}: {gap['description']} (Current: {gap['current_coverage']:.1f}%, Target: {gap['target_coverage']}%)"
            recommendations[priority].append(rec)

        return recommendations

    def _generate_markdown_report(self, summary: dict[str, Any]) -> None:
        """Generate markdown report."""
        report_file = (
            self.reports_dir / f"automated_test_coverage_report_{self.timestamp}.md"
        )

        with open(report_file, "w") as f:
            f.write("# Automated Test Coverage Report\n\n")
            f.write(f"**Generated**: {summary['metadata']['timestamp']}\n")
            f.write(f"**Project**: {summary['metadata']['project_root']}\n\n")

            # Overall statistics
            structure = summary["structure"]
            f.write("## Overview\n\n")
            f.write(f"- **Total Source Files**: {structure['source_files']['total']}\n")
            f.write(f"- **Total Test Files**: {structure['test_files']['total']}\n")
            f.write(
                f"- **Overall Coverage Ratio**: {structure['coverage_ratios']['overall']:.1f}%\n\n"
            )

            # Coverage by area
            f.write("## Coverage by Area\n\n")
            f.write("| Area | Coverage | Target | Status |\n")
            f.write("|------|----------|--------|--------|\n")
            for area in ["core", "sdk", "cli", "web_api", "web_ui"]:
                coverage = structure["coverage_ratios"].get(f"area_{area}", 0)
                target = structure["areas"][area]["target_coverage"]
                status = (
                    "✅"
                    if coverage >= target
                    else "⚠️"
                    if coverage >= target * 0.7
                    else "❌"
                )
                f.write(
                    f"| {area.title()} | {coverage:.1f}% | {target}% | {status} |\n"
                )
            f.write("\n")

            # Coverage by layer
            f.write("## Coverage by Layer\n\n")
            f.write("| Layer | Coverage | Target | Status |\n")
            f.write("|-------|----------|--------|--------|\n")
            for layer in ["domain", "application", "infrastructure", "presentation"]:
                coverage = structure["coverage_ratios"].get(f"layer_{layer}", 0)
                target = structure["layers"][layer]["target_coverage"]
                status = (
                    "✅"
                    if coverage >= target
                    else "⚠️"
                    if coverage >= target * 0.7
                    else "❌"
                )
                f.write(
                    f"| {layer.title()} | {coverage:.1f}% | {target}% | {status} |\n"
                )
            f.write("\n")

            # Critical gaps
            gaps = summary["gaps"]
            critical_gaps = [g for g in gaps if g["priority"] == "critical"]
            if critical_gaps:
                f.write("## Critical Gaps\n\n")
                for gap in critical_gaps:
                    f.write(f"### {gap['category'].title()} ({gap['type'].title()})\n")
                    f.write(f"- **Current Coverage**: {gap['current_coverage']:.1f}%\n")
                    f.write(f"- **Target Coverage**: {gap['target_coverage']}%\n")
                    f.write(f"- **Gap**: {gap['gap_percentage']:.1f}%\n")
                    f.write(f"- **Description**: {gap['description']}\n")
                    f.write("- **Recommendations**:\n")
                    for rec in gap["recommendations"]:
                        f.write(f"  - {rec}\n")
                    f.write("\n")

            # Recommendations
            recommendations = summary["recommendations"]
            f.write("## Prioritized Recommendations\n\n")
            for priority in ["critical", "high", "medium", "low"]:
                if recommendations[priority]:
                    f.write(f"### {priority.title()} Priority\n\n")
                    for rec in recommendations[priority]:
                        f.write(f"- {rec}\n")
                    f.write("\n")

        click.echo(f"✅ Markdown report saved to: {report_file}")

    def run_full_analysis(self) -> dict[str, Any]:
        """Run complete automated analysis."""
        click.echo("🚀 Starting automated test coverage analysis...")

        # Analyze file structure
        structure = self.analyze_file_structure()

        # Run coverage tests
        coverage_results = self.run_coverage_tests()

        # Identify gaps
        gaps = self.identify_gaps(structure, coverage_results)

        # Generate reports
        self.generate_reports(structure, coverage_results, gaps)

        # Return summary for further processing
        return {
            "structure": structure,
            "coverage_results": coverage_results,
            "gaps": gaps,
            "critical_gaps_count": len(
                [g for g in gaps if g["priority"] == "critical"]
            ),
            "total_gaps_count": len(gaps),
        }


@click.command()
@click.option("--project-root", default=".", help="Project root directory")
@click.option(
    "--output-format",
    default="both",
    type=click.Choice(["json", "markdown", "both"]),
    help="Output format",
)
@click.option("--run-tests", is_flag=True, help="Run actual test coverage (slower)")
def main(project_root: str, output_format: str, run_tests: bool):
    """Automated test coverage analysis and reporting."""
    analyzer = TestCoverageAnalyzer(project_root)

    if run_tests:
        results = analyzer.run_full_analysis()
    else:
        # Quick analysis without running tests
        structure = analyzer.analyze_file_structure()
        gaps = analyzer.identify_gaps(structure, {})
        analyzer.generate_reports(structure, {}, gaps)
        results = {"structure": structure, "gaps": gaps}

    # Print summary
    click.echo("\n📋 Analysis Summary:")
    click.echo(
        f"├── Total Source Files: {results['structure']['source_files']['total']}"
    )
    click.echo(f"├── Total Test Files: {results['structure']['test_files']['total']}")
    click.echo(
        f"├── Overall Coverage: {results['structure']['coverage_ratios']['overall']:.1f}%"
    )
    if "critical_gaps_count" in results:
        click.echo(f"├── Critical Gaps: {results['critical_gaps_count']}")
        click.echo(f"└── Total Gaps: {results['total_gaps_count']}")

    click.echo(f"\n✅ Reports generated in: {analyzer.reports_dir}")


if __name__ == "__main__":
    main()
