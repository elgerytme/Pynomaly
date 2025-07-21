#!/usr/bin/env python3
"""
Enhanced Mutation Testing Integration for Coverage Quality Validation

This module provides comprehensive mutation testing capabilities integrated
with the coverage monitoring system to ensure test quality beyond just coverage percentage.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import click
import structlog

logger = structlog.get_logger(__name__)


class MutationTestRunner:
    """Enhanced mutation testing with integration to coverage monitoring."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.reports_dir = self.project_root / "reports" / "mutation"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def run_mutmut_analysis(self, target_paths: list[str] = None) -> dict:
        """Run mutmut mutation testing analysis.

        Args:
            target_paths: Specific paths to target for mutation testing

        Returns:
            Mutation testing results
        """
        target_paths = target_paths or ["src/anomaly_detection"]

        logger.info("Starting mutation testing analysis", targets=target_paths)

        try:
            # Run mutmut
            cmd = [
                "mutmut",
                "run",
                "--paths-to-mutate",
                ",".join(target_paths),
                "--runner",
                "python -m pytest tests/",
                "--tests-dir",
                "tests",
                "--use-coverage",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=1800,  # 30 minute timeout
            )

            # Parse mutmut results
            mutation_data = self._parse_mutmut_results(result)

            # Generate mutation report
            self._generate_mutation_report(mutation_data)

            return {
                "success": result.returncode == 0,
                "mutation_data": mutation_data,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        except subprocess.TimeoutExpired:
            logger.error("Mutation testing timed out")
            return {"success": False, "error": "Timeout"}
        except Exception as e:
            logger.error("Mutation testing failed", error=str(e))
            return {"success": False, "error": str(e)}

    def _parse_mutmut_results(self, result: subprocess.CompletedProcess) -> dict:
        """Parse mutmut results from command output."""

        # Try to get mutmut results
        try:
            status_result = subprocess.run(
                ["mutmut", "results"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            # Parse the output for mutation statistics
            output_lines = status_result.stdout.split("\n")

            mutation_data = {
                "timestamp": datetime.now().isoformat(),
                "total_mutants": 0,
                "killed_mutants": 0,
                "survived_mutants": 0,
                "suspicious_mutants": 0,
                "timeout_mutants": 0,
                "mutation_score": 0.0,
                "test_quality": "unknown",
            }

            # Basic parsing of mutmut output
            for line in output_lines:
                if "killed" in line.lower():
                    # Extract numbers from status line
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.isdigit():
                            mutation_data["killed_mutants"] = int(part)
                            break
                elif "survived" in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.isdigit():
                            mutation_data["survived_mutants"] = int(part)
                            break

            # Calculate mutation score
            total = mutation_data["killed_mutants"] + mutation_data["survived_mutants"]
            if total > 0:
                mutation_data["total_mutants"] = total
                mutation_data["mutation_score"] = (
                    mutation_data["killed_mutants"] / total
                ) * 100

                # Determine test quality
                if mutation_data["mutation_score"] >= 90:
                    mutation_data["test_quality"] = "excellent"
                elif mutation_data["mutation_score"] >= 80:
                    mutation_data["test_quality"] = "good"
                elif mutation_data["mutation_score"] >= 70:
                    mutation_data["test_quality"] = "acceptable"
                else:
                    mutation_data["test_quality"] = "needs_improvement"

            return mutation_data

        except Exception as e:
            logger.error("Failed to parse mutation results", error=str(e))
            return {
                "timestamp": datetime.now().isoformat(),
                "total_mutants": 0,
                "killed_mutants": 0,
                "survived_mutants": 0,
                "mutation_score": 0.0,
                "test_quality": "unknown",
                "error": str(e),
            }

    def _generate_mutation_report(self, mutation_data: dict):
        """Generate comprehensive mutation testing report."""

        report = {
            "mutation_testing_summary": mutation_data,
            "recommendations": self._generate_recommendations(mutation_data),
            "quality_assessment": self._assess_test_quality(mutation_data),
        }

        # Save JSON report
        report_file = (
            self.reports_dir
            / f"mutation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        # Generate HTML report
        html_report = self._generate_html_report(report)
        html_file = (
            self.reports_dir
            / f"mutation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        with open(html_file, "w") as f:
            f.write(html_report)

        # Create latest symlinks
        latest_json = self.reports_dir / "latest_mutation_report.json"
        latest_html = self.reports_dir / "latest_mutation_report.html"

        if latest_json.exists():
            latest_json.unlink()
        if latest_html.exists():
            latest_html.unlink()

        latest_json.symlink_to(report_file.name)
        latest_html.symlink_to(html_file.name)

        logger.info(
            "Mutation testing reports generated",
            json_report=str(report_file),
            html_report=str(html_file),
        )

    def _generate_recommendations(self, mutation_data: dict) -> list[str]:
        """Generate recommendations based on mutation testing results."""
        recommendations = []

        mutation_score = mutation_data.get("mutation_score", 0)
        survived_mutants = mutation_data.get("survived_mutants", 0)

        if mutation_score < 70:
            recommendations.append(
                "❌ Mutation score below 70% indicates weak tests. Consider adding more assertion-based tests."
            )
        elif mutation_score < 80:
            recommendations.append(
                "⚠️ Mutation score below 80%. Review test cases for edge cases and boundary conditions."
            )
        elif mutation_score < 90:
            recommendations.append(
                "✅ Good mutation score. Consider targeting specific modules with low scores for improvement."
            )
        else:
            recommendations.append(
                "🎉 Excellent mutation score! Your tests are catching code changes effectively."
            )

        if survived_mutants > 0:
            recommendations.append(
                f"🔍 {survived_mutants} mutants survived. Review these to identify test gaps."
            )

        recommendations.extend(
            [
                "📋 Run mutation testing regularly to maintain test quality",
                "🎯 Focus on critical business logic paths for mutation testing",
                "⚡ Consider using property-based testing for better mutation coverage",
            ]
        )

        return recommendations

    def _assess_test_quality(self, mutation_data: dict) -> dict:
        """Assess overall test quality based on mutation results."""
        mutation_score = mutation_data.get("mutation_score", 0)

        quality_metrics = {
            "mutation_effectiveness": mutation_score,
            "test_strength": mutation_data.get("test_quality", "unknown"),
            "areas_for_improvement": [],
        }

        if mutation_score < 90:
            quality_metrics["areas_for_improvement"].append(
                "Increase mutation score to 90%+"
            )

        if mutation_data.get("survived_mutants", 0) > 5:
            quality_metrics["areas_for_improvement"].append(
                "Reduce survived mutants count"
            )

        return quality_metrics

    def _generate_html_report(self, report: dict) -> str:
        """Generate HTML mutation testing report."""
        mutation_data = report["mutation_testing_summary"]
        recommendations = report["recommendations"]
        quality = report["quality_assessment"]

        mutation_score = mutation_data.get("mutation_score", 0)

        # Determine score color
        if mutation_score >= 90:
            score_color = "#28a745"
        elif mutation_score >= 80:
            score_color = "#ffc107"
        elif mutation_score >= 70:
            score_color = "#fd7e14"
        else:
            score_color = "#dc3545"

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>anomaly_detection Mutation Testing Report</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; background: #f6f8fa; }}
                .container {{ max-width: 1000px; margin: 0 auto; padding: 20px; }}
                .header {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }}
                .metric h3 {{ margin: 0 0 10px 0; color: #24292e; font-size: 14px; font-weight: 600; text-transform: uppercase; }}
                .metric .value {{ font-size: 28px; font-weight: 700; margin: 10px 0; }}
                .recommendations {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 20px 0; }}
                .recommendation {{ padding: 10px 0; border-bottom: 1px solid #eaecef; }}
                .recommendation:last-child {{ border-bottom: none; }}
                .score-circle {{ width: 120px; height: 120px; border-radius: 50%; margin: 0 auto; display: flex; align-items: center; justify-content: center; color: white; font-size: 24px; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🧬 Mutation Testing Report</h1>
                    <p><strong>Generated:</strong> {mutation_data.get('timestamp', 'Unknown')}</p>
                    <p><strong>Purpose:</strong> Validate test quality beyond coverage percentage</p>
                </div>

                <div class="metrics">
                    <div class="metric">
                        <h3>Mutation Score</h3>
                        <div class="score-circle" style="background-color: {score_color};">
                            {mutation_score:.1f}%
                        </div>
                        <small>Target: 90%+</small>
                    </div>
                    <div class="metric">
                        <h3>Total Mutants</h3>
                        <div class="value" style="color: #17a2b8;">{mutation_data.get('total_mutants', 0)}</div>
                        <small>Code mutations tested</small>
                    </div>
                    <div class="metric">
                        <h3>Killed Mutants</h3>
                        <div class="value" style="color: #28a745;">{mutation_data.get('killed_mutants', 0)}</div>
                        <small>Tests caught changes</small>
                    </div>
                    <div class="metric">
                        <h3>Survived Mutants</h3>
                        <div class="value" style="color: #dc3545;">{mutation_data.get('survived_mutants', 0)}</div>
                        <small>Tests missed changes</small>
                    </div>
                    <div class="metric">
                        <h3>Test Quality</h3>
                        <div class="value" style="color: {score_color};">{mutation_data.get('test_quality', 'Unknown').title()}</div>
                        <small>Overall assessment</small>
                    </div>
                </div>

                <div class="recommendations">
                    <h2>📋 Recommendations</h2>
        """

        for rec in recommendations:
            html += f"<div class='recommendation'>{rec}</div>"

        html += f"""
                </div>

                <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <h2>🎯 Quality Assessment</h2>
                    <p><strong>Mutation Effectiveness:</strong> {quality['mutation_effectiveness']:.1f}%</p>
                    <p><strong>Test Strength:</strong> {quality['test_strength'].title()}</p>

                    {f"<h3>Areas for Improvement:</h3><ul>{''.join(f'<li>{area}</li>' for area in quality['areas_for_improvement'])}</ul>" if quality['areas_for_improvement'] else "<p><strong>Status:</strong> ✅ All quality metrics met!</p>"}
                </div>

                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 20px; font-size: 14px; color: #6c757d;">
                    <p><strong>Note:</strong> Mutation testing validates that your tests can detect code changes.
                    A high mutation score indicates that your tests are effective at catching bugs,
                    not just achieving coverage.</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html


@click.group()
def cli():
    """Enhanced Mutation Testing CLI."""
    pass


@cli.command()
@click.option(
    "--target-paths", multiple=True, help="Paths to target for mutation testing"
)
@click.option("--output-format", type=click.Choice(["json", "text"]), default="text")
def run(target_paths, output_format):
    """Run comprehensive mutation testing analysis."""
    runner = MutationTestRunner()
    result = runner.run_mutmut_analysis(list(target_paths) if target_paths else None)

    if output_format == "json":
        click.echo(json.dumps(result, indent=2))
    else:
        if result["success"]:
            mutation_data = result["mutation_data"]
            click.echo("🧬 Mutation Testing Analysis Complete!")
            click.echo(
                f"📊 Mutation Score: {mutation_data.get('mutation_score', 0):.1f}%"
            )
            click.echo(f"🎯 Total Mutants: {mutation_data.get('total_mutants', 0)}")
            click.echo(f"✅ Killed: {mutation_data.get('killed_mutants', 0)}")
            click.echo(f"❌ Survived: {mutation_data.get('survived_mutants', 0)}")
            click.echo(
                f"🏆 Test Quality: {mutation_data.get('test_quality', 'Unknown').title()}"
            )
        else:
            click.echo(
                f"❌ Mutation testing failed: {result.get('error', 'Unknown error')}"
            )


@cli.command()
def status():
    """Show current mutation testing status."""
    runner = MutationTestRunner()

    # Check for latest report
    latest_report = runner.reports_dir / "latest_mutation_report.json"
    if latest_report.exists():
        with open(latest_report) as f:
            report = json.load(f)

        mutation_data = report["mutation_testing_summary"]
        click.echo("🧬 Latest Mutation Testing Status")
        click.echo("-" * 40)
        click.echo(f"Timestamp: {mutation_data.get('timestamp', 'Unknown')}")
        click.echo(f"Mutation Score: {mutation_data.get('mutation_score', 0):.1f}%")
        click.echo(
            f"Test Quality: {mutation_data.get('test_quality', 'Unknown').title()}"
        )
        click.echo(f"Total Mutants: {mutation_data.get('total_mutants', 0)}")
        click.echo(f"Killed: {mutation_data.get('killed_mutants', 0)}")
        click.echo(f"Survived: {mutation_data.get('survived_mutants', 0)}")
    else:
        click.echo("❌ No mutation testing data available")
        click.echo("Run 'mutation_testing_enhanced.py run' to generate initial data")


if __name__ == "__main__":
    cli()
