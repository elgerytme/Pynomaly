#!/usr/bin/env python3
"""Mutation testing script for test quality assessment."""

import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import click


class MutationTester:
    """Run mutation testing to assess test quality."""

    def __init__(self, project_root: str = "."):
        """Initialize mutation tester.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)
        self.results = {}

    def run_mutation_testing(
        self,
        target_paths: List[str],
        test_command: str = "python -m pytest",
        timeout_factor: float = 2.0,
        max_mutations: Optional[int] = None,
    ) -> Dict[str, any]:
        """Run mutation testing on specified paths.

        Args:
            target_paths: List of paths to mutate
            test_command: Command to run tests
            timeout_factor: Timeout multiplier for test execution
            max_mutations: Maximum number of mutations to test

        Returns:
            Mutation testing results
        """
        self.logger.info("Starting mutation testing...")

        # Validate paths
        valid_paths = []
        for path in target_paths:
            path_obj = self.project_root / path
            if path_obj.exists():
                valid_paths.append(str(path_obj))
            else:
                self.logger.warning(f"Path does not exist: {path}")

        if not valid_paths:
            raise ValueError("No valid paths provided for mutation testing")

        # Run baseline test to ensure tests pass
        self.logger.info("Running baseline tests...")
        baseline_result = self._run_baseline_tests(test_command)
        if not baseline_result["success"]:
            raise RuntimeError(f"Baseline tests failed: {baseline_result['output']}")

        results = {
            "start_time": datetime.utcnow().isoformat(),
            "target_paths": valid_paths,
            "test_command": test_command,
            "baseline_duration": baseline_result["duration"],
            "mutations": [],
            "summary": {},
        }

        # Run mutations for each path
        for path in valid_paths:
            self.logger.info(f"Running mutations for: {path}")
            path_results = self._run_mutations_for_path(
                path, test_command, timeout_factor, max_mutations
            )
            results["mutations"].extend(path_results)

        # Calculate summary statistics
        results["summary"] = self._calculate_summary(results["mutations"])
        results["end_time"] = datetime.utcnow().isoformat()

        self.results = results
        return results

    def _run_baseline_tests(self, test_command: str) -> Dict[str, any]:
        """Run baseline tests to ensure they pass.

        Args:
            test_command: Command to run tests

        Returns:
            Baseline test results
        """
        start_time = time.time()

        try:
            result = subprocess.run(
                test_command.split(),
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=self.project_root,
            )

            duration = time.time() - start_time

            return {
                "success": result.returncode == 0,
                "duration": duration,
                "output": result.stdout + result.stderr,
                "returncode": result.returncode,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "duration": time.time() - start_time,
                "output": "Test execution timed out",
                "returncode": -1,
            }
        except Exception as e:
            return {
                "success": False,
                "duration": time.time() - start_time,
                "output": str(e),
                "returncode": -1,
            }

    def _run_mutations_for_path(
        self,
        path: str,
        test_command: str,
        timeout_factor: float,
        max_mutations: Optional[int],
    ) -> List[Dict[str, any]]:
        """Run mutations for a specific path.

        Args:
            path: Path to mutate
            test_command: Command to run tests
            timeout_factor: Timeout multiplier
            max_mutations: Maximum mutations to test

        Returns:
            List of mutation results
        """
        mutations = []

        try:
            # Use mutmut to run mutations
            mutmut_command = [
                "python",
                "-m",
                "mutmut",
                "run",
                "--paths-to-mutate",
                path,
                "--runner",
                test_command,
                "--timeout-factor",
                str(timeout_factor),
                "--no-progress",
            ]

            if max_mutations:
                mutmut_command.extend(["--max-mutations", str(max_mutations)])

            self.logger.info(f"Running: {' '.join(mutmut_command)}")

            result = subprocess.run(
                mutmut_command,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                cwd=self.project_root,
            )

            # Parse mutmut output
            mutations = self._parse_mutmut_output(result.stdout, result.stderr, path)

        except subprocess.TimeoutExpired:
            self.logger.error(f"Mutation testing timed out for path: {path}")
            mutations.append(
                {
                    "path": path,
                    "status": "timeout",
                    "error": "Mutation testing timed out",
                }
            )
        except Exception as e:
            self.logger.error(f"Mutation testing failed for path {path}: {e}")
            mutations.append({"path": path, "status": "error", "error": str(e)})

        return mutations

    def _parse_mutmut_output(
        self, stdout: str, stderr: str, path: str
    ) -> List[Dict[str, any]]:
        """Parse mutmut output to extract mutation results.

        Args:
            stdout: Standard output from mutmut
            stderr: Standard error from mutmut
            path: Path that was mutated

        Returns:
            List of parsed mutation results
        """
        mutations = []

        # Try to get detailed results from mutmut
        try:
            # Run mutmut results to get detailed information
            result_command = ["python", "-m", "mutmut", "results"]
            result_proc = subprocess.run(
                result_command,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self.project_root,
            )

            if result_proc.returncode == 0:
                # Parse the results
                for line in result_proc.stdout.split("\n"):
                    if line.strip() and not line.startswith("#"):
                        parts = line.split()
                        if len(parts) >= 2:
                            mutation_id = parts[0]
                            status = parts[1] if len(parts) > 1 else "unknown"

                            mutations.append(
                                {
                                    "path": path,
                                    "mutation_id": mutation_id,
                                    "status": status,
                                    "line": parts[2] if len(parts) > 2 else "unknown",
                                    "description": (
                                        " ".join(parts[3:]) if len(parts) > 3 else ""
                                    ),
                                }
                            )

        except Exception as e:
            self.logger.warning(f"Could not parse detailed mutmut results: {e}")

        # Fallback: parse basic info from stdout
        if not mutations:
            lines = stdout.split("\n") + stderr.split("\n")
            for line in lines:
                if "KILLED" in line or "SURVIVED" in line or "SKIPPED" in line:
                    mutations.append(
                        {
                            "path": path,
                            "status": "parsed_from_output",
                            "description": line.strip(),
                        }
                    )

        return mutations

    def _calculate_summary(self, mutations: List[Dict[str, any]]) -> Dict[str, any]:
        """Calculate summary statistics from mutation results.

        Args:
            mutations: List of mutation results

        Returns:
            Summary statistics
        """
        if not mutations:
            return {
                "total_mutations": 0,
                "killed": 0,
                "survived": 0,
                "skipped": 0,
                "timeout": 0,
                "error": 0,
                "mutation_score": 0.0,
            }

        status_counts = {}
        for mutation in mutations:
            status = mutation.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        total = len(mutations)
        killed = status_counts.get("killed", 0) + status_counts.get("KILLED", 0)
        survived = status_counts.get("survived", 0) + status_counts.get("SURVIVED", 0)
        skipped = status_counts.get("skipped", 0) + status_counts.get("SKIPPED", 0)
        timeout = status_counts.get("timeout", 0) + status_counts.get("TIMEOUT", 0)
        error = status_counts.get("error", 0) + status_counts.get("ERROR", 0)

        # Mutation score = killed / (killed + survived)
        tested_mutations = killed + survived
        mutation_score = (
            (killed / tested_mutations * 100) if tested_mutations > 0 else 0.0
        )

        return {
            "total_mutations": total,
            "killed": killed,
            "survived": survived,
            "skipped": skipped,
            "timeout": timeout,
            "error": error,
            "mutation_score": mutation_score,
            "status_distribution": status_counts,
        }

    def generate_report(self, output_file: str = "mutation_report.html") -> str:
        """Generate HTML report from mutation testing results.

        Args:
            output_file: Output file path

        Returns:
            Path to generated report
        """
        if not self.results:
            raise ValueError("No mutation testing results available")

        summary = self.results["summary"]

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mutation Testing Report - Pynomaly</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
                .metric {{ background: white; padding: 15px; border: 1px solid #ddd; border-radius: 5px; flex: 1; }}
                .metric h3 {{ margin: 0 0 10px 0; color: #333; }}
                .metric .value {{ font-size: 24px; font-weight: bold; }}
                .good {{ color: #28a745; }}
                .warning {{ color: #ffc107; }}
                .danger {{ color: #dc3545; }}
                .score {{ font-size: 48px; text-align: center; margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #f8f9fa; }}
                .status-killed {{ color: #28a745; font-weight: bold; }}
                .status-survived {{ color: #dc3545; font-weight: bold; }}
                .status-skipped {{ color: #6c757d; }}
                pre {{ background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Mutation Testing Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Target paths: {', '.join(self.results['target_paths'])}</p>
            </div>

            <div class="score {'good' if summary['mutation_score'] >= 80 else 'warning' if summary['mutation_score'] >= 60 else 'danger'}">
                Mutation Score: {summary['mutation_score']:.1f}%
            </div>

            <div class="summary">
                <div class="metric">
                    <h3>Total Mutations</h3>
                    <div class="value">{summary['total_mutations']}</div>
                </div>
                <div class="metric">
                    <h3>Killed</h3>
                    <div class="value good">{summary['killed']}</div>
                </div>
                <div class="metric">
                    <h3>Survived</h3>
                    <div class="value danger">{summary['survived']}</div>
                </div>
                <div class="metric">
                    <h3>Skipped</h3>
                    <div class="value">{summary['skipped']}</div>
                </div>
            </div>

            <h2>Mutation Details</h2>
            <table>
                <thead>
                    <tr>
                        <th>Path</th>
                        <th>Mutation ID</th>
                        <th>Status</th>
                        <th>Line</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
        """

        for mutation in self.results["mutations"]:
            status = mutation.get("status", "unknown")
            status_class = f"status-{status.lower()}"

            html_content += f"""
                    <tr>
                        <td>{mutation.get('path', 'N/A')}</td>
                        <td>{mutation.get('mutation_id', 'N/A')}</td>
                        <td class="{status_class}">{status.upper()}</td>
                        <td>{mutation.get('line', 'N/A')}</td>
                        <td>{mutation.get('description', 'N/A')}</td>
                    </tr>
            """

        html_content += """
                </tbody>
            </table>

            <h2>Interpretation</h2>
            <div>
                <h3>Mutation Score Interpretation:</h3>
                <ul>
                    <li><strong>80-100%:</strong> Excellent test quality</li>
                    <li><strong>60-79%:</strong> Good test quality, some improvements possible</li>
                    <li><strong>40-59%:</strong> Fair test quality, significant improvements needed</li>
                    <li><strong>0-39%:</strong> Poor test quality, major improvements required</li>
                </ul>

                <h3>Status Meanings:</h3>
                <ul>
                    <li><strong class="status-killed">KILLED:</strong> Mutation was detected by tests (good)</li>
                    <li><strong class="status-survived">SURVIVED:</strong> Mutation was not detected by tests (bad)</li>
                    <li><strong class="status-skipped">SKIPPED:</strong> Mutation was skipped (syntax error, etc.)</li>
                </ul>
            </div>

            <h2>Recommendations</h2>
            <div>
        """

        # Generate recommendations based on mutation score
        if summary["mutation_score"] >= 80:
            html_content += "<p>✅ Excellent test quality! Your tests are very effective at detecting defects.</p>"
        elif summary["mutation_score"] >= 60:
            html_content += "<p>⚠️ Good test quality, but there's room for improvement. Consider adding more edge case tests.</p>"
        else:
            html_content += (
                "<p>❌ Test quality needs significant improvement. Focus on:</p>"
            )
            html_content += "<ul>"
            html_content += "<li>Adding tests for edge cases and error conditions</li>"
            html_content += "<li>Improving assertion quality</li>"
            html_content += "<li>Testing boundary conditions</li>"
            html_content += "<li>Adding negative test cases</li>"
            html_content += "</ul>"

        if summary["survived"] > 0:
            html_content += f"<p>Focus on the {summary['survived']} survived mutations to improve test coverage.</p>"

        html_content += """
            </div>
        </body>
        </html>
        """

        output_path = self.project_root / output_file
        with open(output_path, "w") as f:
            f.write(html_content)

        return str(output_path)

    def save_results(self, output_file: str = "mutation_results.json") -> str:
        """Save mutation testing results to JSON file.

        Args:
            output_file: Output file path

        Returns:
            Path to saved results file
        """
        if not self.results:
            raise ValueError("No mutation testing results available")

        output_path = self.project_root / output_file
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)

        return str(output_path)


@click.group()
def cli():
    """Mutation testing CLI for test quality assessment."""
    logging.basicConfig(level=logging.INFO)


@cli.command()
@click.option("--paths", "-p", multiple=True, required=True, help="Paths to mutate")
@click.option("--test-command", "-c", default="python -m pytest", help="Test command")
@click.option("--timeout-factor", "-t", default=2.0, help="Timeout multiplier")
@click.option("--max-mutations", "-m", type=int, help="Maximum mutations to test")
@click.option("--output", "-o", default="mutation_results.json", help="Output file")
@click.option("--report", "-r", default="mutation_report.html", help="HTML report file")
def run(paths, test_command, timeout_factor, max_mutations, output, report):
    """Run mutation testing."""
    tester = MutationTester()

    try:
        click.echo("🧬 Starting mutation testing...")
        results = tester.run_mutation_testing(
            target_paths=list(paths),
            test_command=test_command,
            timeout_factor=timeout_factor,
            max_mutations=max_mutations,
        )

        # Save results
        results_path = tester.save_results(output)
        click.echo(f"📄 Results saved to: {results_path}")

        # Generate report
        report_path = tester.generate_report(report)
        click.echo(f"📊 Report generated: {report_path}")

        # Print summary
        summary = results["summary"]
        click.echo(f"\n🎯 Mutation Score: {summary['mutation_score']:.1f}%")
        click.echo(f"   Total mutations: {summary['total_mutations']}")
        click.echo(f"   Killed: {summary['killed']}")
        click.echo(f"   Survived: {summary['survived']}")
        click.echo(f"   Skipped: {summary['skipped']}")

        # Exit with non-zero code if mutation score is too low
        if summary["mutation_score"] < 60:
            click.echo("⚠️  Mutation score below 60%, consider improving test quality")
            sys.exit(1)
        else:
            click.echo("✅ Good mutation score!")

    except Exception as e:
        click.echo(f"❌ Mutation testing failed: {e}")
        sys.exit(1)


@cli.command()
@click.option("--domain-only", is_flag=True, help="Test only domain layer")
@click.option(
    "--fast", is_flag=True, help="Run fast mutation testing (limited mutations)"
)
def quick(domain_only, fast):
    """Run quick mutation testing on critical paths."""
    tester = MutationTester()

    if domain_only:
        paths = ["src/pynomaly/domain/"]
        click.echo("🎯 Quick mutation testing: Domain layer only")
    else:
        paths = ["src/pynomaly/domain/", "src/pynomaly/application/"]
        click.echo("🎯 Quick mutation testing: Domain + Application layers")

    max_mutations = 50 if fast else None
    test_command = "python -m pytest tests/domain/ tests/application/ -x"

    try:
        results = tester.run_mutation_testing(
            target_paths=paths,
            test_command=test_command,
            timeout_factor=1.5,
            max_mutations=max_mutations,
        )

        # Generate quick report
        report_path = tester.generate_report("quick_mutation_report.html")

        summary = results["summary"]
        click.echo(f"\n🎯 Quick Mutation Score: {summary['mutation_score']:.1f}%")
        click.echo(f"📊 Report: {report_path}")

    except Exception as e:
        click.echo(f"❌ Quick mutation testing failed: {e}")
        sys.exit(1)


@cli.command()
@click.argument("results_file")
def report(results_file):
    """Generate HTML report from existing results."""
    tester = MutationTester()

    try:
        with open(results_file) as f:
            tester.results = json.load(f)

        report_path = tester.generate_report()
        click.echo(f"📊 Report generated: {report_path}")

    except Exception as e:
        click.echo(f"❌ Report generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
