#!/usr/bin/env python3
"""Coverage dashboard generator for Pynomaly."""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List


class CoverageDashboard:
    """Generate HTML dashboard for coverage monitoring."""
    
    def __init__(self, root_dir: Path = None):
        """Initialize coverage dashboard."""
        self.root_dir = root_dir or Path.cwd()
        self.reports_dir = self.root_dir / "reports"
        self.dashboard_dir = self.reports_dir / "dashboard"
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize coverage history database
        self.db_path = self.dashboard_dir / "coverage_history.db"
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database for coverage history."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS coverage_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    overall_coverage REAL NOT NULL,
                    line_coverage REAL NOT NULL,
                    branch_coverage REAL NOT NULL,
                    total_lines INTEGER NOT NULL,
                    covered_lines INTEGER NOT NULL,
                    total_branches INTEGER NOT NULL,
                    covered_branches INTEGER NOT NULL,
                    commit_hash TEXT,
                    branch_name TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS module_coverage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    history_id INTEGER NOT NULL,
                    module_name TEXT NOT NULL,
                    coverage REAL NOT NULL,
                    lines INTEGER NOT NULL,
                    covered_lines INTEGER NOT NULL,
                    FOREIGN KEY (history_id) REFERENCES coverage_history (id)
                )
            """)
    
    def record_coverage_data(self, coverage_data: Dict) -> None:
        """Record coverage data to history database."""
        try:
            # Get git info
            import subprocess
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
            branch_name = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
            ).strip()
        except:
            commit_hash = "unknown"
            branch_name = "unknown"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert main coverage record
            cursor.execute("""
                INSERT INTO coverage_history (
                    timestamp, overall_coverage, line_coverage, branch_coverage,
                    total_lines, covered_lines, total_branches, covered_branches,
                    commit_hash, branch_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                coverage_data.get("coverage_percent", 0),
                coverage_data.get("line_coverage_percent", 0),
                coverage_data.get("branch_coverage_percent", 0),
                coverage_data.get("lines_total", 0),
                coverage_data.get("lines_covered", 0),
                coverage_data.get("branches_total", 0),
                coverage_data.get("branches_covered", 0),
                commit_hash,
                branch_name
            ))
            
            history_id = cursor.lastrowid
            
            # Insert module coverage data
            if "files" in coverage_data:
                for module_name, module_data in coverage_data["files"].items():
                    summary = module_data.get("summary", {})
                    cursor.execute("""
                        INSERT INTO module_coverage (
                            history_id, module_name, coverage, lines, covered_lines
                        ) VALUES (?, ?, ?, ?, ?)
                    """, (
                        history_id,
                        module_name,
                        summary.get("percent_covered", 0),
                        summary.get("num_statements", 0),
                        summary.get("covered_lines", 0)
                    ))
    
    def get_coverage_trend(self, days: int = 30) -> List[Dict]:
        """Get coverage trend data for the last N days."""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, overall_coverage, line_coverage, branch_coverage,
                       commit_hash, branch_name
                FROM coverage_history
                WHERE timestamp > ?
                ORDER BY timestamp
            """, (cutoff_date,))
            
            return [
                {
                    "timestamp": row[0],
                    "overall_coverage": row[1],
                    "line_coverage": row[2],
                    "branch_coverage": row[3],
                    "commit_hash": row[4][:8] if row[4] else "unknown",
                    "branch_name": row[5]
                }
                for row in cursor.fetchall()
            ]
    
    def get_module_coverage_summary(self) -> Dict:
        """Get latest module coverage summary."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT mc.module_name, mc.coverage, mc.lines, mc.covered_lines
                FROM module_coverage mc
                JOIN coverage_history ch ON mc.history_id = ch.id
                WHERE ch.id = (SELECT MAX(id) FROM coverage_history)
                ORDER BY mc.coverage ASC
            """)
            
            return {
                row[0]: {
                    "coverage": row[1],
                    "lines": row[2],
                    "covered_lines": row[3]
                }
                for row in cursor.fetchall()
            }
    
    def generate_html_dashboard(self) -> None:
        """Generate HTML dashboard."""
        print("📊 Generating coverage dashboard...")
        
        # Get data
        trend_data = self.get_coverage_trend()
        module_data = self.get_module_coverage_summary()
        
        # Load latest coverage data if available
        coverage_file = self.reports_dir / "coverage" / "coverage.json"
        latest_coverage = {}
        if coverage_file.exists():
            with open(coverage_file) as f:
                latest_coverage = json.load(f)
        
        html_content = self._generate_dashboard_html(trend_data, module_data, latest_coverage)
        
        dashboard_file = self.dashboard_dir / "index.html"
        with open(dashboard_file, "w") as f:
            f.write(html_content)
        
        print(f"✅ Dashboard generated: {dashboard_file}")
    
    def _generate_dashboard_html(self, trend_data: List[Dict], 
                                module_data: Dict, latest_coverage: Dict) -> str:
        """Generate the HTML content for the dashboard."""
        
        # Get latest stats
        latest_stats = trend_data[-1] if trend_data else {
            "overall_coverage": 0, "line_coverage": 0, "branch_coverage": 0
        }
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pynomaly - Coverage Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
        }}
        .stat-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            border-left: 4px solid;
        }}
        .stat-card.excellent {{ border-left-color: #28a745; }}
        .stat-card.good {{ border-left-color: #17a2b8; }}
        .stat-card.warning {{ border-left-color: #ffc107; }}
        .stat-card.danger {{ border-left-color: #dc3545; }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .stat-label {{
            color: #6c757d;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .chart-container {{
            padding: 30px;
            border-top: 1px solid #eee;
        }}
        .module-list {{
            padding: 30px;
            border-top: 1px solid #eee;
        }}
        .module-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }}
        .module-name {{
            font-family: monospace;
            font-size: 0.9em;
        }}
        .coverage-bar {{
            width: 200px;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 0 15px;
        }}
        .coverage-fill {{
            height: 100%;
            transition: width 0.3s ease;
        }}
        .coverage-text {{
            font-weight: bold;
            min-width: 50px;
            text-align: right;
        }}
        .timestamp {{
            color: #6c757d;
            font-size: 0.8em;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 Pynomaly Coverage Dashboard</h1>
            <p>Real-time test coverage monitoring and trends</p>
            <div class="timestamp">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card {self._get_coverage_class(latest_stats.get('overall_coverage', 0))}">
                <div class="stat-value">{latest_stats.get('overall_coverage', 0):.1f}%</div>
                <div class="stat-label">Overall Coverage</div>
            </div>
            <div class="stat-card {self._get_coverage_class(latest_stats.get('line_coverage', 0))}">
                <div class="stat-value">{latest_stats.get('line_coverage', 0):.1f}%</div>
                <div class="stat-label">Line Coverage</div>
            </div>
            <div class="stat-card {self._get_coverage_class(latest_stats.get('branch_coverage', 0))}">
                <div class="stat-value">{latest_stats.get('branch_coverage', 0):.1f}%</div>
                <div class="stat-label">Branch Coverage</div>
            </div>
            <div class="stat-card excellent">
                <div class="stat-value">{len(module_data)}</div>
                <div class="stat-label">Modules Tracked</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h2>📈 Coverage Trend (Last 30 Days)</h2>
            <canvas id="trendChart" width="800" height="300"></canvas>
        </div>
        
        <div class="module-list">
            <h2>📁 Module Coverage</h2>
            {self._generate_module_list_html(module_data)}
        </div>
    </div>
    
    <script>
        // Trend chart
        const ctx = document.getElementById('trendChart').getContext('2d');
        const trendData = {json.dumps(trend_data)};
        
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: trendData.map(d => new Date(d.timestamp).toLocaleDateString()),
                datasets: [{{
                    label: 'Overall Coverage',
                    data: trendData.map(d => d.overall_coverage),
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    fill: true
                }}, {{
                    label: 'Line Coverage',
                    data: trendData.map(d => d.line_coverage),
                    borderColor: '#17a2b8',
                    backgroundColor: 'rgba(23, 162, 184, 0.1)',
                    fill: false
                }}, {{
                    label: 'Branch Coverage',
                    data: trendData.map(d => d.branch_coverage),
                    borderColor: '#28a745',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    fill: false
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        ticks: {{
                            callback: function(value) {{
                                return value + '%';
                            }}
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        position: 'top'
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return context.dataset.label + ': ' + context.parsed.y.toFixed(1) + '%';
                            }}
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        return html
    
    def _get_coverage_class(self, coverage: float) -> str:
        """Get CSS class based on coverage percentage."""
        if coverage >= 90:
            return "excellent"
        elif coverage >= 80:
            return "good"
        elif coverage >= 70:
            return "warning"
        else:
            return "danger"
    
    def _get_coverage_color(self, coverage: float) -> str:
        """Get color based on coverage percentage."""
        if coverage >= 90:
            return "#28a745"
        elif coverage >= 80:
            return "#17a2b8"
        elif coverage >= 70:
            return "#ffc107"
        else:
            return "#dc3545"
    
    def _generate_module_list_html(self, module_data: Dict) -> str:
        """Generate HTML for module coverage list."""
        if not module_data:
            return "<p>No module data available</p>"
        
        html = ""
        for module_name, data in sorted(module_data.items(), key=lambda x: x[1]["coverage"]):
            coverage = data["coverage"]
            color = self._get_coverage_color(coverage)
            
            html += f"""
            <div class="module-item">
                <div class="module-name">{module_name}</div>
                <div class="coverage-bar">
                    <div class="coverage-fill" style="width: {coverage}%; background-color: {color};"></div>
                </div>
                <div class="coverage-text">{coverage:.1f}%</div>
            </div>
            """
        
        return html


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Coverage dashboard generator")
    parser.add_argument("--record", help="Record coverage data from JSON file")
    parser.add_argument("--generate", action="store_true", help="Generate HTML dashboard")
    
    args = parser.parse_args()
    
    dashboard = CoverageDashboard()
    
    if args.record:
        with open(args.record) as f:
            coverage_data = json.load(f)
        dashboard.record_coverage_data(coverage_data)
        print(f"✅ Coverage data recorded from {args.record}")
    
    if args.generate:
        dashboard.generate_html_dashboard()
    
    if not args.record and not args.generate:
        # Default: generate dashboard
        dashboard.generate_html_dashboard()


if __name__ == "__main__":
    main()