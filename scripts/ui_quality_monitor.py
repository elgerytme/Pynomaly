#!/usr/bin/env python3
"""
UI Quality Monitoring Script
Tracks UI quality metrics over time and generates trend reports.
"""

import argparse
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any


class UIQualityMonitor:
    """Monitor and track UI quality metrics over time."""

    def __init__(self, db_path: str = "ui_quality_metrics.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for metrics storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ui_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                commit_hash TEXT,
                branch TEXT,
                overall_score REAL NOT NULL,
                layout_score REAL,
                ux_score REAL,
                visual_score REAL,
                accessibility_score REAL,
                responsive_score REAL,
                critical_issues INTEGER,
                warnings INTEGER,
                tests_passed INTEGER,
                tests_failed INTEGER,
                performance_score REAL,
                lighthouse_accessibility REAL,
                report_path TEXT
            )
        """)

        # Create issues table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ui_issues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_id INTEGER,
                category TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT NOT NULL,
                location TEXT,
                recommendation TEXT,
                wcag_criteria TEXT,
                FOREIGN KEY (metric_id) REFERENCES ui_metrics (id)
            )
        """)

        conn.commit()
        conn.close()

    def record_metrics(
        self, test_results: dict[str, Any], commit_hash: str = None, branch: str = None
    ):
        """Record UI test metrics to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Extract metrics from test results
        timestamp = datetime.now().isoformat()
        overall_score = test_results.get("overall_score", 0)

        # Category scores
        category_scores = test_results.get("critique", {}).get("category_scores", {})
        layout_score = category_scores.get("layout", 0)
        ux_score = category_scores.get("ux_flows", 0)
        visual_score = category_scores.get("visual", 0)
        accessibility_score = category_scores.get("accessibility", 0)
        responsive_score = category_scores.get("responsive", 0)

        # Issue counts
        critical_issues = test_results.get("critique", {}).get("critical_issues", 0)
        warnings = test_results.get("critique", {}).get("warnings", 0)

        # Test execution stats
        test_execution = test_results.get("test_execution", {})
        tests_passed = test_execution.get("passed", 0)
        tests_failed = test_execution.get("failed", 0)

        # Performance metrics
        performance = test_results.get("performance", {})
        lighthouse = performance.get("lighthouse_scores", {})
        performance_score = lighthouse.get("performance", 0)
        lighthouse_accessibility = lighthouse.get("accessibility", 0)

        # Insert main metrics
        cursor.execute(
            """
            INSERT INTO ui_metrics (
                timestamp, commit_hash, branch, overall_score,
                layout_score, ux_score, visual_score, accessibility_score, responsive_score,
                critical_issues, warnings, tests_passed, tests_failed,
                performance_score, lighthouse_accessibility
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                timestamp,
                commit_hash,
                branch,
                overall_score,
                layout_score,
                ux_score,
                visual_score,
                accessibility_score,
                responsive_score,
                critical_issues,
                warnings,
                tests_passed,
                tests_failed,
                performance_score,
                lighthouse_accessibility,
            ),
        )

        metric_id = cursor.lastrowid

        # Insert individual issues
        for category_name, category_data in test_results.items():
            if isinstance(category_data, dict) and "issues" in category_data:
                for issue in category_data.get("issues", []):
                    cursor.execute(
                        """
                        INSERT INTO ui_issues (
                            metric_id, category, severity, description, location, recommendation, wcag_criteria
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            metric_id,
                            category_name,
                            issue.get("severity", issue.get("type", "unknown")),
                            issue.get("description", ""),
                            issue.get("location", ""),
                            issue.get("recommendation", ""),
                            issue.get("wcag_criteria", ""),
                        ),
                    )

        conn.commit()
        conn.close()

        return metric_id

    def get_quality_trend(self, days: int = 30) -> list[dict[str, Any]]:
        """Get quality trend over specified number of days."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute(
            """
            SELECT timestamp, overall_score, accessibility_score, critical_issues, warnings
            FROM ui_metrics
            WHERE timestamp >= ?
            ORDER BY timestamp
        """,
            (cutoff_date,),
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "timestamp": row[0],
                    "overall_score": row[1],
                    "accessibility_score": row[2],
                    "critical_issues": row[3],
                    "warnings": row[4],
                }
            )

        conn.close()
        return results

    def get_latest_metrics(self) -> dict[str, Any] | None:
        """Get the latest recorded metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT *
            FROM ui_metrics
            ORDER BY timestamp DESC
            LIMIT 1
        """)

        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        columns = [desc[0] for desc in cursor.description]
        result = dict(zip(columns, row, strict=False))

        conn.close()
        return result

    def generate_trend_report(self, days: int = 30) -> str:
        """Generate HTML trend report."""
        trend_data = self.get_quality_trend(days)
        latest_metrics = self.get_latest_metrics()

        if not trend_data:
            return "No data available for trend analysis."

        # Calculate trend direction
        if len(trend_data) >= 2:
            recent_score = trend_data[-1]["overall_score"]
            older_score = trend_data[0]["overall_score"]
            trend_direction = (
                "📈"
                if recent_score > older_score
                else "📉"
                if recent_score < older_score
                else "➡️"
            )
            score_change = recent_score - older_score
        else:
            trend_direction = "➡️"
            score_change = 0

        # Generate report
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UI Quality Trend Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f8fafc;
        }}
        .card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .metric {{
            text-align: center;
            padding: 20px;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-label {{
            color: #6b7280;
            font-size: 0.9em;
        }}
        .trend-up {{ color: #10b981; }}
        .trend-down {{ color: #ef4444; }}
        .trend-stable {{ color: #6b7280; }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .chart-container {{
            position: relative;
            height: 400px;
            width: 100%;
        }}
    </style>
</head>
<body>
    <div class="card">
        <h1>📊 UI Quality Trend Report</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Period: Last {days} days</p>
    </div>
    
    <div class="grid">
        <div class="card metric">
            <div class="metric-value {("trend-up" if score_change > 0 else "trend-down" if score_change < 0 else "trend-stable")}">
                {latest_metrics["overall_score"] if latest_metrics else "N/A"}
            </div>
            <div class="metric-label">Overall Score {trend_direction}</div>
            {f"<small>Change: {score_change:+.1f} points</small>" if score_change != 0 else ""}
        </div>
        
        <div class="card metric">
            <div class="metric-value">
                {latest_metrics["accessibility_score"] if latest_metrics else "N/A"}
            </div>
            <div class="metric-label">Accessibility Score</div>
        </div>
        
        <div class="card metric">
            <div class="metric-value {("trend-down" if latest_metrics and latest_metrics["critical_issues"] == 0 else "trend-up")}">
                {latest_metrics["critical_issues"] if latest_metrics else "N/A"}
            </div>
            <div class="metric-label">Critical Issues</div>
        </div>
        
        <div class="card metric">
            <div class="metric-value">
                {latest_metrics["tests_passed"] if latest_metrics else "N/A"}/{(latest_metrics["tests_passed"] + latest_metrics["tests_failed"]) if latest_metrics else "N/A"}
            </div>
            <div class="metric-label">Tests Passed</div>
        </div>
    </div>
    
    <div class="card">
        <h2>📈 Quality Trend</h2>
        <div class="chart-container">
            <canvas id="trendChart"></canvas>
        </div>
    </div>
    
    <div class="card">
        <h2>🚨 Issue Tracking</h2>
        <div class="chart-container">
            <canvas id="issuesChart"></canvas>
        </div>
    </div>
    
    <script>
        // Trend chart
        const trendCtx = document.getElementById('trendChart').getContext('2d');
        new Chart(trendCtx, {{
            type: 'line',
            data: {{
                labels: {json.dumps([item["timestamp"][:10] for item in trend_data])},
                datasets: [{{
                    label: 'Overall Score',
                    data: {json.dumps([item["overall_score"] for item in trend_data])},
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4
                }}, {{
                    label: 'Accessibility Score',
                    data: {json.dumps([item["accessibility_score"] for item in trend_data])},
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100
                    }}
                }}
            }}
        }});
        
        // Issues chart
        const issuesCtx = document.getElementById('issuesChart').getContext('2d');
        new Chart(issuesCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps([item["timestamp"][:10] for item in trend_data])},
                datasets: [{{
                    label: 'Critical Issues',
                    data: {json.dumps([item["critical_issues"] for item in trend_data])},
                    backgroundColor: '#ef4444'
                }}, {{
                    label: 'Warnings',
                    data: {json.dumps([item["warnings"] for item in trend_data])},
                    backgroundColor: '#f59e0b'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

        return html_content

    def export_metrics(self, format: str = "json", days: int = 30) -> str:
        """Export metrics in specified format."""
        trend_data = self.get_quality_trend(days)

        if format.lower() == "json":
            return json.dumps(trend_data, indent=2)
        elif format.lower() == "csv":
            import csv
            import io

            output = io.StringIO()
            if trend_data:
                writer = csv.DictWriter(output, fieldnames=trend_data[0].keys())
                writer.writeheader()
                writer.writerows(trend_data)

            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")

    def check_quality_gates(
        self, thresholds: dict[str, float] = None
    ) -> dict[str, Any]:
        """Check if current metrics meet quality gates."""
        if thresholds is None:
            thresholds = {
                "overall_score": 85.0,
                "accessibility_score": 80.0,
                "critical_issues": 0,
                "warnings": 5,
            }

        latest = self.get_latest_metrics()
        if not latest:
            return {"passed": False, "reason": "No metrics available"}

        results = {"passed": True, "gates": {}, "failed_gates": []}

        for metric, threshold in thresholds.items():
            current_value = latest.get(metric, 0)

            if metric in ["critical_issues", "warnings"]:
                passed = current_value <= threshold
            else:
                passed = current_value >= threshold

            results["gates"][metric] = {
                "current": current_value,
                "threshold": threshold,
                "passed": passed,
            }

            if not passed:
                results["passed"] = False
                results["failed_gates"].append(metric)

        return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="UI Quality Monitoring Tool")
    parser.add_argument("--record", type=str, help="Record metrics from JSON file")
    parser.add_argument(
        "--trend-report", type=str, help="Generate trend report HTML file"
    )
    parser.add_argument("--export", choices=["json", "csv"], help="Export metrics data")
    parser.add_argument(
        "--days", type=int, default=30, help="Number of days for trend analysis"
    )
    parser.add_argument(
        "--quality-gates", action="store_true", help="Check quality gates"
    )
    parser.add_argument("--commit", type=str, help="Commit hash for recording")
    parser.add_argument("--branch", type=str, help="Branch name for recording")

    args = parser.parse_args()

    monitor = UIQualityMonitor()

    if args.record:
        # Record metrics from file
        with open(args.record) as f:
            test_results = json.load(f)

        metric_id = monitor.record_metrics(test_results, args.commit, args.branch)
        print(f"✅ Recorded metrics with ID: {metric_id}")

    elif args.trend_report:
        # Generate trend report
        report_html = monitor.generate_trend_report(args.days)

        with open(args.trend_report, "w") as f:
            f.write(report_html)

        print(f"📊 Trend report generated: {args.trend_report}")

    elif args.export:
        # Export metrics
        exported_data = monitor.export_metrics(args.export, args.days)
        print(exported_data)

    elif args.quality_gates:
        # Check quality gates
        gates = monitor.check_quality_gates()

        if gates["passed"]:
            print("✅ All quality gates passed!")
        else:
            print("❌ Quality gates failed:")
            for gate in gates["failed_gates"]:
                gate_info = gates["gates"][gate]
                print(
                    f"  {gate}: {gate_info['current']} (threshold: {gate_info['threshold']})"
                )

        exit(0 if gates["passed"] else 1)

    else:
        # Show latest metrics
        latest = monitor.get_latest_metrics()
        if latest:
            print("📊 Latest UI Quality Metrics:")
            print(f"  Overall Score: {latest['overall_score']}")
            print(f"  Accessibility: {latest['accessibility_score']}")
            print(f"  Critical Issues: {latest['critical_issues']}")
            print(f"  Warnings: {latest['warnings']}")
            print(f"  Timestamp: {latest['timestamp']}")
        else:
            print("No metrics recorded yet.")


if __name__ == "__main__":
    main()
