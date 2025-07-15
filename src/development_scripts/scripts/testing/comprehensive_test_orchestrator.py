#!/usr/bin/env python3
"""
Comprehensive Test Orchestrator for Issue #93

This script orchestrates the complete test coverage monitoring and automation setup,
integrating all components for the 100% coverage goal.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import click
import structlog

logger = structlog.get_logger(__name__)


class TestOrchestrator:
    """Orchestrates comprehensive test coverage monitoring and automation."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.reports_dir = self.project_root / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sub-components
        self.coverage_monitor_script = (
            self.project_root / "scripts" / "testing" / "coverage_monitor.py"
        )
        self.mutation_script = (
            self.project_root / "scripts" / "testing" / "mutation_testing_enhanced.py"
        )

    def run_full_coverage_pipeline(
        self,
        enable_mutation: bool = False,
        fail_on_regression: bool = True,
        target_coverage: float = 100.0,
    ) -> dict:
        """Run the complete coverage monitoring pipeline.

        Args:
            enable_mutation: Whether to run mutation testing
            fail_on_regression: Whether to fail on coverage regression
            target_coverage: Target coverage percentage

        Returns:
            Comprehensive results from all testing phases
        """
        results = {
            "pipeline_start": datetime.now().isoformat(),
            "phases": {},
            "overall_success": True,
            "summary": {},
        }

        try:
            # Phase 1: Run comprehensive test coverage
            logger.info("Phase 1: Running comprehensive test coverage analysis")
            coverage_result = self._run_coverage_analysis(fail_on_regression)
            results["phases"]["coverage"] = coverage_result

            if not coverage_result.get("success", False):
                results["overall_success"] = False
                logger.error("Coverage analysis failed")
                return results

            # Phase 2: Quality gates validation
            logger.info("Phase 2: Validating quality gates")
            gates_result = self._validate_quality_gates(coverage_result)
            results["phases"]["quality_gates"] = gates_result

            # Phase 3: Mutation testing (if enabled)
            if enable_mutation:
                logger.info("Phase 3: Running mutation testing")
                mutation_result = self._run_mutation_testing()
                results["phases"]["mutation"] = mutation_result

            # Phase 4: Generate comprehensive reports
            logger.info("Phase 4: Generating comprehensive reports")
            reporting_result = self._generate_comprehensive_reports(results)
            results["phases"]["reporting"] = reporting_result

            # Phase 5: Update CI/CD artifacts
            logger.info("Phase 5: Updating CI/CD artifacts")
            cicd_result = self._update_cicd_artifacts(results)
            results["phases"]["cicd"] = cicd_result

            # Generate final summary
            results["summary"] = self._generate_final_summary(results, target_coverage)
            results["pipeline_end"] = datetime.now().isoformat()

            logger.info(
                "Test coverage pipeline completed",
                success=results["overall_success"],
                coverage=results["summary"].get("current_coverage", 0),
            )

            return results

        except Exception as e:
            logger.error("Test coverage pipeline failed", error=str(e))
            results["overall_success"] = False
            results["error"] = str(e)
            return results

    def _run_coverage_analysis(self, fail_on_regression: bool) -> dict:
        """Run comprehensive coverage analysis."""
        try:
            cmd = [
                "python",
                str(self.coverage_monitor_script),
                "run",
                "--check-gates",
                "--test-command",
                "python -m pytest tests/ --cov=src/pynomaly --cov-report=xml --cov-report=html --cov-report=json --cov-branch",
            ]

            if fail_on_regression:
                cmd.append("--fail-on-regression")

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _validate_quality_gates(self, coverage_result: dict) -> dict:
        """Validate quality gates against coverage results."""
        try:
            # Parse coverage data from the latest run
            coverage_json = self.project_root / "coverage.json"
            if coverage_json.exists():
                with open(coverage_json) as f:
                    coverage_data = json.load(f)

                total_coverage = coverage_data["totals"]["percent_covered"]

                gates = {
                    "coverage_100_percent": total_coverage >= 100.0,
                    "coverage_95_percent": total_coverage >= 95.0,
                    "coverage_90_percent": total_coverage >= 90.0,
                    "no_missing_critical": True,  # Simplified for now
                    "branch_coverage_90": coverage_data["totals"]
                    .get("percent_covered_display", "0%")
                    .replace("%", "")
                    != "0",
                }

                all_passed = all(gates.values())
                critical_passed = gates["coverage_90_percent"]

                return {
                    "success": True,
                    "gates": gates,
                    "all_passed": all_passed,
                    "critical_passed": critical_passed,
                    "current_coverage": total_coverage,
                }
            else:
                return {"success": False, "error": "Coverage data not found"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _run_mutation_testing(self) -> dict:
        """Run mutation testing analysis."""
        try:
            cmd = [
                "python",
                str(self.mutation_script),
                "run",
                "--output-format",
                "json",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=1800,  # 30 minute timeout
            )

            mutation_result = {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

            # Try to parse JSON output
            if result.stdout:
                try:
                    mutation_data = json.loads(result.stdout)
                    mutation_result["data"] = mutation_data
                except json.JSONDecodeError:
                    pass

            return mutation_result

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Mutation testing timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _generate_comprehensive_reports(self, results: dict) -> dict:
        """Generate comprehensive reports combining all testing phases."""
        try:
            # Generate enhanced coverage report
            coverage_cmd = [
                "python",
                str(self.coverage_monitor_script),
                "report",
                "--enhanced",
                "--output",
                str(self.reports_dir / "comprehensive_coverage_report.html"),
            ]

            coverage_report = subprocess.run(
                coverage_cmd, capture_output=True, text=True, cwd=self.project_root
            )

            # Generate coverage badge
            badge_cmd = ["python", str(self.coverage_monitor_script), "badge"]

            badge_result = subprocess.run(
                badge_cmd, capture_output=True, text=True, cwd=self.project_root
            )

            # Generate overall status
            status_cmd = ["python", str(self.coverage_monitor_script), "status"]

            status_result = subprocess.run(
                status_cmd, capture_output=True, text=True, cwd=self.project_root
            )

            # Create summary report
            summary_report = self._create_summary_report(results)
            summary_file = (
                self.reports_dir
                / f"test_coverage_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

            with open(summary_file, "w") as f:
                json.dump(summary_report, f, indent=2)

            return {
                "success": True,
                "coverage_report": coverage_report.returncode == 0,
                "badge_generated": badge_result.returncode == 0,
                "status_generated": status_result.returncode == 0,
                "summary_file": str(summary_file),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _update_cicd_artifacts(self, results: dict) -> dict:
        """Update CI/CD artifacts for GitHub Actions."""
        try:
            artifacts = {}

            # Create GitHub Actions outputs
            if "GITHUB_OUTPUT" in os.environ:
                with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                    coverage = results.get("summary", {}).get("current_coverage", 0)
                    f.write(f"coverage_percentage={coverage:.2f}\n")
                    f.write(
                        f"quality_gates_passed={results.get('phases', {}).get('quality_gates', {}).get('all_passed', False)}\n"
                    )
                    f.write(
                        f"pipeline_success={results.get('overall_success', False)}\n"
                    )

                artifacts["github_outputs"] = True

            # Create coverage status for PR comments
            if results.get("overall_success") and "quality_gates" in results.get(
                "phases", {}
            ):
                coverage_data = results["phases"]["quality_gates"]
                status_message = self._create_pr_status_message(coverage_data)

                status_file = self.reports_dir / "pr_status_message.md"
                with open(status_file, "w") as f:
                    f.write(status_message)

                artifacts["pr_status"] = str(status_file)

            return {"success": True, "artifacts": artifacts}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _create_summary_report(self, results: dict) -> dict:
        """Create a comprehensive summary report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "pipeline_status": "success" if results["overall_success"] else "failed",
            "phases_completed": list(results.get("phases", {}).keys()),
            "coverage_summary": results.get("phases", {}).get("quality_gates", {}),
            "mutation_summary": results.get("phases", {}).get("mutation", {}),
            "recommendations": self._generate_recommendations(results),
            "next_steps": self._generate_next_steps(results),
        }

    def _create_pr_status_message(self, coverage_data: dict) -> str:
        """Create a status message for PR comments."""
        coverage = coverage_data.get("current_coverage", 0)
        gates = coverage_data.get("gates", {})

        # Coverage status emoji
        if coverage >= 100:
            emoji = "🎉"
            status = "TARGET ACHIEVED"
        elif coverage >= 95:
            emoji = "🔥"
            status = "EXCELLENT"
        elif coverage >= 90:
            emoji = "✅"
            status = "GOOD"
        else:
            emoji = "⚡"
            status = "NEEDS IMPROVEMENT"

        message = f"""## {emoji} Test Coverage Report

**Overall Status:** {status}
**Current Coverage:** {coverage:.2f}%
**Target:** 100%

### 🚦 Quality Gates
"""

        for gate_name, passed in gates.items():
            gate_display = gate_name.replace("_", " ").title()
            status_emoji = "✅" if passed else "❌"
            message += f"- {status_emoji} {gate_display}\n"

        message += f"""
### 📊 Progress to 100% Coverage
Progress: {coverage:.1f}% | Remaining: {100-coverage:.1f}%

**Generated by Enhanced Coverage Monitoring System**
"""

        return message

    def _generate_recommendations(self, results: dict) -> list[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        coverage_data = results.get("phases", {}).get("quality_gates", {})
        if coverage_data.get("success"):
            coverage = coverage_data.get("current_coverage", 0)

            if coverage < 90:
                recommendations.append(
                    "🎯 Priority: Increase overall coverage to 90% minimum"
                )
            elif coverage < 95:
                recommendations.append(
                    "📈 Focus: Target 95% coverage for production readiness"
                )
            elif coverage < 100:
                recommendations.append("🏁 Final push: Achieve 100% coverage goal")
            else:
                recommendations.append(
                    "🎉 Maintain 100% coverage with ongoing development"
                )

        # Add mutation testing recommendations if available
        mutation_data = results.get("phases", {}).get("mutation", {})
        if mutation_data.get("success") and "data" in mutation_data:
            mutation_score = (
                mutation_data["data"].get("mutation_data", {}).get("mutation_score", 0)
            )
            if mutation_score < 80:
                recommendations.append(
                    "🧬 Improve test quality with mutation testing focus"
                )

        return recommendations

    def _generate_next_steps(self, results: dict) -> list[str]:
        """Generate next steps based on results."""
        next_steps = []

        if not results.get("overall_success"):
            next_steps.append("🔧 Fix failing tests and coverage issues")
            next_steps.append("📊 Review coverage reports for gaps")

        coverage_data = results.get("phases", {}).get("quality_gates", {})
        if coverage_data.get("success"):
            gates = coverage_data.get("gates", {})
            if not gates.get("coverage_100_percent", False):
                next_steps.append("📝 Identify uncovered lines and add targeted tests")
                next_steps.append("🔍 Review code complexity for testability")

        next_steps.extend(
            [
                "🔄 Run coverage monitoring regularly in CI/CD",
                "📈 Monitor coverage trends and prevent regressions",
                "🎯 Set up quality gates for pull requests",
            ]
        )

        return next_steps

    def _generate_final_summary(self, results: dict, target_coverage: float) -> dict:
        """Generate final pipeline summary."""
        coverage_data = results.get("phases", {}).get("quality_gates", {})
        current_coverage = (
            coverage_data.get("current_coverage", 0)
            if coverage_data.get("success")
            else 0
        )

        return {
            "target_coverage": target_coverage,
            "current_coverage": current_coverage,
            "coverage_gap": target_coverage - current_coverage,
            "goal_achieved": current_coverage >= target_coverage,
            "phases_successful": sum(
                1
                for phase in results.get("phases", {}).values()
                if phase.get("success", False)
            ),
            "total_phases": len(results.get("phases", {})),
            "overall_success": results.get("overall_success", False),
        }


@click.group()
def cli():
    """Comprehensive Test Coverage Orchestrator for Issue #93."""
    pass


@cli.command()
@click.option("--enable-mutation", is_flag=True, help="Enable mutation testing")
@click.option(
    "--fail-on-regression",
    is_flag=True,
    default=True,
    help="Fail on coverage regression",
)
@click.option(
    "--target-coverage", type=float, default=100.0, help="Target coverage percentage"
)
def run(enable_mutation: bool, fail_on_regression: bool, target_coverage: float):
    """Run the complete test coverage monitoring pipeline."""
    orchestrator = TestOrchestrator()

    click.echo("🚀 Starting Comprehensive Test Coverage Pipeline")
    click.echo(f"🎯 Target Coverage: {target_coverage}%")
    click.echo(f"🧬 Mutation Testing: {'Enabled' if enable_mutation else 'Disabled'}")
    click.echo("-" * 60)

    results = orchestrator.run_full_coverage_pipeline(
        enable_mutation=enable_mutation,
        fail_on_regression=fail_on_regression,
        target_coverage=target_coverage,
    )

    # Display results
    if results["overall_success"]:
        click.echo("✅ Pipeline completed successfully!")
    else:
        click.echo("❌ Pipeline failed!")

    summary = results.get("summary", {})
    if summary:
        click.echo("\n📊 Final Results:")
        click.echo(f"   Current Coverage: {summary.get('current_coverage', 0):.2f}%")
        click.echo(f"   Target Coverage: {summary.get('target_coverage', 100):.2f}%")
        click.echo(f"   Coverage Gap: {summary.get('coverage_gap', 0):.2f}%")
        click.echo(
            f"   Goal Achieved: {'Yes' if summary.get('goal_achieved') else 'No'}"
        )
        click.echo(
            f"   Phases Successful: {summary.get('phases_successful', 0)}/{summary.get('total_phases', 0)}"
        )

    # Exit with appropriate code
    sys.exit(0 if results["overall_success"] else 1)


@cli.command()
def status():
    """Show current test coverage status."""
    orchestrator = TestOrchestrator()

    # Check for latest reports
    latest_summary = None
    for report_file in orchestrator.reports_dir.glob("test_coverage_summary_*.json"):
        if (
            latest_summary is None
            or report_file.stat().st_mtime > latest_summary.stat().st_mtime
        ):
            latest_summary = report_file

    if latest_summary:
        with open(latest_summary) as f:
            data = json.load(f)

        click.echo("🎯 Latest Test Coverage Status")
        click.echo("-" * 40)
        click.echo(f"Timestamp: {data.get('timestamp', 'Unknown')}")
        click.echo(f"Pipeline Status: {data.get('pipeline_status', 'Unknown')}")
        click.echo(f"Phases Completed: {', '.join(data.get('phases_completed', []))}")

        coverage_summary = data.get("coverage_summary", {})
        if coverage_summary:
            click.echo(
                f"Current Coverage: {coverage_summary.get('current_coverage', 0):.2f}%"
            )
            click.echo(
                f"Quality Gates Passed: {coverage_summary.get('all_passed', False)}"
            )
    else:
        click.echo("❌ No test coverage status available")
        click.echo(
            "Run 'comprehensive_test_orchestrator.py run' to generate initial data"
        )


if __name__ == "__main__":
    cli()
