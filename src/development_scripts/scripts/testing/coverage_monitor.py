#!/usr/bin/env python3
"""Coverage monitoring and reporting script."""

import json
import logging
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click


class CoverageMonitor:
    """Monitor and track test coverage over time."""

    def __init__(self, db_path: str = "coverage_history.db"):
        """Initialize coverage monitor.

        Args:
            db_path: Path to SQLite database for storing coverage history
        """
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the coverage tracking database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS coverage_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    branch TEXT,
                    commit_hash TEXT,
                    total_coverage REAL NOT NULL,
                    lines_covered INTEGER NOT NULL,
                    lines_total INTEGER NOT NULL,
                    files_covered INTEGER NOT NULL,
                    files_total INTEGER NOT NULL,
                    test_command TEXT,
                    metadata TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS file_coverage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    file_path TEXT NOT NULL,
                    coverage_percent REAL NOT NULL,
                    lines_covered INTEGER NOT NULL,
                    lines_total INTEGER NOT NULL,
                    missing_lines TEXT,
                    FOREIGN KEY (run_id) REFERENCES coverage_runs (id)
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_coverage_runs_timestamp
                ON coverage_runs (timestamp)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_file_coverage_run_id
                ON file_coverage (run_id)
            """
            )

    def run_coverage(self, test_command: str = None) -> dict[str, Any]:
        """Run test coverage and collect results.

        Args:
            test_command: Custom test command to run

        Returns:
            Coverage results dictionary
        """
        if test_command is None:
            test_command = "python -m pytest tests/ --cov=src/anomaly_detection --cov-report=json --cov-report=term"

        self.logger.info(f"Running coverage with command: {test_command}")

        try:
            # Run the test command
            result = subprocess.run(
                test_command.split(),
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            if result.returncode != 0:
                self.logger.error(f"Test command failed: {result.stderr}")
                raise RuntimeError(f"Test execution failed: {result.stderr}")

            # Load coverage results
            coverage_file = Path("coverage.json")
            if not coverage_file.exists():
                raise FileNotFoundError("Coverage report not generated")

            with open(coverage_file) as f:
                coverage_data = json.load(f)

            return self._parse_coverage_data(coverage_data, test_command)

        except subprocess.TimeoutExpired:
            self.logger.error("Test command timed out")
            raise RuntimeError("Test execution timed out")
        except Exception as e:
            self.logger.error(f"Coverage run failed: {e}")
            raise

    def _parse_coverage_data(
        self, coverage_data: dict[str, Any], test_command: str
    ) -> dict[str, Any]:
        """Parse coverage data from JSON report.

        Args:
            coverage_data: Raw coverage data from JSON report
            test_command: Test command that was executed

        Returns:
            Parsed coverage results
        """
        totals = coverage_data.get("totals", {})
        files = coverage_data.get("files", {})

        # Get git information
        branch = self._get_git_branch()
        commit_hash = self._get_git_commit()

        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "branch": branch,
            "commit_hash": commit_hash,
            "total_coverage": totals.get("percent_covered", 0.0),
            "lines_covered": totals.get("covered_lines", 0),
            "lines_total": totals.get("num_statements", 0),
            "files_covered": len(
                [
                    f
                    for f in files.values()
                    if f.get("summary", {}).get("percent_covered", 0) > 0
                ]
            ),
            "files_total": len(files),
            "test_command": test_command,
            "file_coverage": {},
            "metadata": {
                "missing_lines": totals.get("missing_lines", 0),
                "excluded_lines": totals.get("excluded_lines", 0),
                "branches_covered": totals.get("covered_branches", 0),
                "branches_total": totals.get("num_branches", 0),
            },
        }

        # Parse file-level coverage
        for file_path, file_data in files.items():
            summary = file_data.get("summary", {})
            result["file_coverage"][file_path] = {
                "coverage_percent": summary.get("percent_covered", 0.0),
                "lines_covered": summary.get("covered_lines", 0),
                "lines_total": summary.get("num_statements", 0),
                "missing_lines": ",".join(map(str, file_data.get("missing_lines", []))),
            }

        return result

    def store_coverage(self, coverage_data: dict[str, Any]) -> int:
        """Store coverage data in database.

        Args:
            coverage_data: Coverage data to store

        Returns:
            ID of the stored coverage run
        """
        with sqlite3.connect(self.db_path) as conn:
            # Insert main coverage run
            cursor = conn.execute(
                """
                INSERT INTO coverage_runs (
                    timestamp, branch, commit_hash, total_coverage,
                    lines_covered, lines_total, files_covered, files_total,
                    test_command, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    coverage_data["timestamp"],
                    coverage_data["branch"],
                    coverage_data["commit_hash"],
                    coverage_data["total_coverage"],
                    coverage_data["lines_covered"],
                    coverage_data["lines_total"],
                    coverage_data["files_covered"],
                    coverage_data["files_total"],
                    coverage_data["test_command"],
                    json.dumps(coverage_data["metadata"]),
                ),
            )

            run_id = cursor.lastrowid

            # Insert file-level coverage
            for file_path, file_data in coverage_data["file_coverage"].items():
                conn.execute(
                    """
                    INSERT INTO file_coverage (
                        run_id, file_path, coverage_percent,
                        lines_covered, lines_total, missing_lines
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        run_id,
                        file_path,
                        file_data["coverage_percent"],
                        file_data["lines_covered"],
                        file_data["lines_total"],
                        file_data["missing_lines"],
                    ),
                )

            return run_id

    def get_coverage_trends(self, days: int = 30) -> list[dict[str, Any]]:
        """Get coverage trends over time.

        Args:
            days: Number of days to look back

        Returns:
            List of coverage data points
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"""
                SELECT
                    timestamp, branch, commit_hash, total_coverage,
                    lines_covered, lines_total, files_covered, files_total
                FROM coverage_runs
                WHERE timestamp >= datetime('now', '-{days} days')
                ORDER BY timestamp
            """
            )

            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]

    def check_coverage_regression(self, threshold: float = 2.0) -> dict[str, Any]:
        """Check for coverage regression.

        Args:
            threshold: Percentage threshold for regression detection

        Returns:
            Regression analysis results
        """
        trends = self.get_coverage_trends(days=7)  # Last week

        if len(trends) < 2:
            return {
                "has_regression": False,
                "message": "Insufficient data for regression analysis",
            }

        latest = trends[-1]
        previous = trends[-2]

        coverage_change = latest["total_coverage"] - previous["total_coverage"]

        has_regression = coverage_change < -threshold

        return {
            "has_regression": has_regression,
            "coverage_change": coverage_change,
            "latest_coverage": latest["total_coverage"],
            "previous_coverage": previous["total_coverage"],
            "threshold": threshold,
            "message": f"Coverage {'decreased' if coverage_change < 0 else 'increased'} by {abs(coverage_change):.2f}%",
        }

    def generate_coverage_report(self) -> str:
        """Generate a comprehensive coverage report.

        Returns:
            HTML coverage report
        """
        trends = self.get_coverage_trends()
        regression = self.check_coverage_regression()

        if not trends:
            return "<html><body><h1>No coverage data available</h1></body></html>"

        latest = trends[-1]

        # Calculate coverage by module
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT file_path, coverage_percent, lines_covered, lines_total
                FROM file_coverage
                WHERE run_id = (SELECT id FROM coverage_runs ORDER BY timestamp DESC LIMIT 1)
                ORDER BY coverage_percent ASC
            """
            )

            file_coverage = cursor.fetchall()

        # Build HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>anomaly_detection Coverage Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .metrics {{ display: flex; gap: 20px; margin: 20px 0; }}
                .metric {{ background: white; padding: 15px; border: 1px solid #ddd; border-radius: 5px; flex: 1; }}
                .metric h3 {{ margin: 0 0 10px 0; color: #333; }}
                .metric .value {{ font-size: 24px; font-weight: bold; }}
                .good {{ color: #28a745; }}
                .warning {{ color: #ffc107; }}
                .danger {{ color: #dc3545; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #f8f9fa; }}
                .progress {{ width: 100%; height: 20px; background: #f0f0f0; border-radius: 10px; }}
                .progress-bar {{ height: 100%; border-radius: 10px; transition: width 0.3s; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>anomaly_detection Test Coverage Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Branch: {latest['branch']} | Commit: {latest['commit_hash'][:8]}</p>
            </div>

            <div class="metrics">
                <div class="metric">
                    <h3>Total Coverage</h3>
                    <div class="value {'good' if latest['total_coverage'] >= 80 else 'warning' if latest['total_coverage'] >= 60 else 'danger'}">
                        {latest['total_coverage']:.1f}%
                    </div>
                </div>
                <div class="metric">
                    <h3>Lines Covered</h3>
                    <div class="value">{latest['lines_covered']:,} / {latest['lines_total']:,}</div>
                </div>
                <div class="metric">
                    <h3>Files Covered</h3>
                    <div class="value">{latest['files_covered']} / {latest['files_total']}</div>
                </div>
                <div class="metric">
                    <h3>Trend</h3>
                    <div class="value {'good' if not regression['has_regression'] else 'danger'}">
                        {regression['message']}
                    </div>
                </div>
            </div>

            <h2>File Coverage Details</h2>
            <table>
                <thead>
                    <tr>
                        <th>File</th>
                        <th>Coverage</th>
                        <th>Lines</th>
                        <th>Progress</th>
                    </tr>
                </thead>
                <tbody>
        """

        for file_path, coverage_percent, lines_covered, lines_total in file_coverage:
            color_class = (
                "good"
                if coverage_percent >= 80
                else "warning"
                if coverage_percent >= 60
                else "danger"
            )
            html += f"""
                    <tr>
                        <td>{file_path}</td>
                        <td class="{color_class}">{coverage_percent:.1f}%</td>
                        <td>{lines_covered} / {lines_total}</td>
                        <td>
                            <div class="progress">
                                <div class="progress-bar" style="width: {coverage_percent}%; background-color: {'#28a745' if coverage_percent >= 80 else '#ffc107' if coverage_percent >= 60 else '#dc3545'};"></div>
                            </div>
                        </td>
                    </tr>
            """

        html += """
                </tbody>
            </table>
        </body>
        </html>
        """

        return html

    def _get_git_branch(self) -> str:
        """Get current git branch."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return "unknown"

    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return "unknown"


@click.group()
def cli():
    """Coverage monitoring CLI."""
    logging.basicConfig(level=logging.INFO)


@cli.command()
@click.option("--test-command", "-c", help="Custom test command to run")
@click.option("--db-path", default="coverage_history.db", help="Database path")
def run(test_command: str, db_path: str):
    """Run coverage analysis and store results."""
    monitor = CoverageMonitor(db_path)

    try:
        coverage_data = monitor.run_coverage(test_command)
        run_id = monitor.store_coverage(coverage_data)

        click.echo("✅ Coverage analysis complete!")
        click.echo(f"📊 Total coverage: {coverage_data['total_coverage']:.1f}%")
        click.echo(f"📁 Run ID: {run_id}")

        # Check for regression
        regression = monitor.check_coverage_regression()
        if regression["has_regression"]:
            click.echo(f"⚠️  Coverage regression detected: {regression['message']}")
            sys.exit(1)
        else:
            click.echo(f"✅ {regression['message']}")

    except Exception as e:
        click.echo(f"❌ Coverage analysis failed: {e}")
        sys.exit(1)


@cli.command()
@click.option("--days", "-d", default=30, help="Days to look back")
@click.option("--db-path", default="coverage_history.db", help="Database path")
def trends(days: int, db_path: str):
    """Show coverage trends."""
    monitor = CoverageMonitor(db_path)
    trends_data = monitor.get_coverage_trends(days)

    if not trends_data:
        click.echo("No coverage data found")
        return

    click.echo(f"\n📈 Coverage trends (last {days} days):")
    click.echo("-" * 60)

    for trend in trends_data[-10:]:  # Show last 10 entries
        date = datetime.fromisoformat(trend["timestamp"]).strftime("%Y-%m-%d %H:%M")
        click.echo(f"{date} | {trend['total_coverage']:6.1f}% | {trend['branch']}")


@cli.command()
@click.option("--output", "-o", default="coverage_report.html", help="Output file")
@click.option("--db-path", default="coverage_history.db", help="Database path")
def report(output: str, db_path: str):
    """Generate HTML coverage report."""
    monitor = CoverageMonitor(db_path)
    html_report = monitor.generate_coverage_report()

    with open(output, "w") as f:
        f.write(html_report)

    click.echo(f"📄 Coverage report generated: {output}")


@cli.command()
@click.option("--threshold", "-t", default=2.0, help="Regression threshold (%)")
@click.option("--db-path", default="coverage_history.db", help="Database path")
def check(threshold: float, db_path: str):
    """Check for coverage regression."""
    monitor = CoverageMonitor(db_path)
    regression = monitor.check_coverage_regression(threshold)

    if regression["has_regression"]:
        click.echo("❌ Coverage regression detected!")
        click.echo(f"   {regression['message']}")
        click.echo(f"   Threshold: {threshold}%")
        sys.exit(1)
    else:
        click.echo("✅ No coverage regression detected")
        click.echo(f"   {regression['message']}")


if __name__ == "__main__":
    cli()
