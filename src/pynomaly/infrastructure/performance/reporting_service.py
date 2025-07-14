"""
Performance Reporting Service.

Generates comprehensive performance reports with visualizations, trend analysis,
and actionable insights for CI/CD pipelines and performance monitoring.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import base64
import io

import numpy as np
import pandas as pd
from jinja2 import Template

# Optional dependencies for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class PerformanceReportGenerator:
    """Generates comprehensive performance reports with visualizations."""
    
    def __init__(self, output_dir: str = "performance_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if VISUALIZATION_AVAILABLE:
            # Set up plotting style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
    
    def generate_html_report(self, performance_data: Dict[str, Any], 
                           historical_data: List[Dict[str, Any]] = None) -> str:
        """Generate comprehensive HTML performance report."""
        
        # Prepare data for template
        template_data = {
            'report_title': 'Performance Regression Report',
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'run_id': performance_data.get('run_id', 'unknown'),
            'ci_status': performance_data.get('ci_status', 'UNKNOWN'),
            'performance_data': performance_data,
            'regression_summary': performance_data.get('regression_summary', {}),
            'baseline_status': performance_data.get('baseline_status', {}),
            'recommendations': performance_data.get('recommendations', []),
            'has_charts': VISUALIZATION_AVAILABLE
        }
        
        # Generate charts if visualization is available
        if VISUALIZATION_AVAILABLE and historical_data:
            template_data['charts'] = self._generate_charts(performance_data, historical_data)
        
        # Use template to generate HTML
        html_template = self._get_html_template()
        template = Template(html_template)
        html_content = template.render(**template_data)
        
        # Save HTML report
        report_file = self.output_dir / f"performance_report_{performance_data.get('run_id', 'latest')}.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {report_file}")
        return str(report_file)
    
    def _generate_charts(self, performance_data: Dict[str, Any], 
                        historical_data: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate performance charts and return as base64 encoded images."""
        charts = {}
        
        try:
            # Extract time series data
            metrics_data = self._extract_time_series_data(historical_data)
            
            if metrics_data:
                # Generate trend chart
                charts['trend_chart'] = self._create_trend_chart(metrics_data)
                
                # Generate regression severity chart
                charts['severity_chart'] = self._create_severity_chart(performance_data)
                
                # Generate baseline health chart
                charts['baseline_health_chart'] = self._create_baseline_health_chart(performance_data)
                
                # Generate performance distribution chart
                charts['distribution_chart'] = self._create_distribution_chart(metrics_data)
        
        except Exception as e:
            logger.error(f"Failed to generate charts: {e}")
        
        return charts
    
    def _extract_time_series_data(self, historical_data: List[Dict[str, Any]]) -> Dict[str, List[Tuple[datetime, float]]]:
        """Extract time series data from historical performance results."""
        metrics_data = {}
        
        for result in historical_data:
            timestamp = datetime.fromisoformat(result.get('timestamp', datetime.now().isoformat()))
            
            for test_name, test_data in result.get('test_results', {}).get('test_results', {}).items():
                if test_data.get('status') != 'success':
                    continue
                
                for metric in test_data.get('metrics', []):
                    metric_key = f"{test_name}_{metric['name']}"
                    
                    if metric_key not in metrics_data:
                        metrics_data[metric_key] = []
                    
                    metrics_data[metric_key].append((timestamp, metric['value']))
        
        # Sort by timestamp
        for key in metrics_data:
            metrics_data[key].sort(key=lambda x: x[0])
        
        return metrics_data
    
    def _create_trend_chart(self, metrics_data: Dict[str, List[Tuple[datetime, float]]]) -> str:
        """Create performance trend chart."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Trends Over Time', fontsize=16, fontweight='bold')
        
        # Select top 4 metrics by data points
        top_metrics = sorted(metrics_data.items(), key=lambda x: len(x[1]), reverse=True)[:4]
        
        for i, (metric_name, data) in enumerate(top_metrics):
            if not data:
                continue
                
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            timestamps, values = zip(*data)
            
            # Plot trend line
            ax.plot(timestamps, values, marker='o', linewidth=2, markersize=4)
            ax.set_title(metric_name.replace('_', ' ').title(), fontweight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(data) > 2:
                x_numeric = np.arange(len(timestamps))
                z = np.polyfit(x_numeric, values, 1)
                p = np.poly1d(z)
                ax.plot(timestamps, p(x_numeric), "--", alpha=0.8, color='red')
        
        # Hide empty subplots
        for i in range(len(top_metrics), 4):
            row, col = i // 2, i % 2
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_severity_chart(self, performance_data: Dict[str, Any]) -> str:
        """Create regression severity breakdown chart."""
        regression_summary = performance_data.get('regression_summary', {})
        severity_data = regression_summary.get('regressions_by_severity', {})
        
        if not any(severity_data.values()):
            # Create empty chart
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No Regressions Detected\n🎉', 
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=20, fontweight='bold', transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plt.title('Regression Severity Breakdown', fontsize=16, fontweight='bold')
            return self._fig_to_base64(fig)
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        
        labels = []
        sizes = []
        colors = []
        
        color_map = {
            'critical': '#ff4444',
            'high': '#ff8800',
            'medium': '#ffcc00',
            'low': '#88cc00'
        }
        
        for severity, count in severity_data.items():
            if count > 0:
                labels.append(f'{severity.title()} ({count})')
                sizes.append(count)
                colors.append(color_map.get(severity, '#cccccc'))
        
        if sizes:
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                            autopct='%1.1f%%', startangle=90)
            
            # Enhance text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        ax.set_title('Regression Severity Breakdown', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_baseline_health_chart(self, performance_data: Dict[str, Any]) -> str:
        """Create baseline health status chart."""
        baseline_status = performance_data.get('baseline_status', {})
        metrics_status = baseline_status.get('metrics', {})
        
        if not metrics_status:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No Baseline Data Available', 
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=16, transform=ax.transAxes)
            ax.axis('off')
            return self._fig_to_base64(fig)
        
        # Extract health scores
        metric_names = []
        health_scores = []
        
        for metric_name, status in metrics_status.items():
            metric_names.append(metric_name.replace('_', ' ').title()[:20])  # Truncate long names
            health_scores.append(status.get('health_score', 0.0))
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, max(6, len(metric_names) * 0.5)))
        
        # Color bars based on health score
        colors = ['#ff4444' if score < 0.6 else '#ffcc00' if score < 0.8 else '#88cc00' 
                 for score in health_scores]
        
        bars = ax.barh(metric_names, health_scores, color=colors)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, health_scores)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.2f}', va='center', fontweight='bold')
        
        ax.set_xlabel('Health Score')
        ax.set_title('Baseline Health Scores', fontsize=16, fontweight='bold')
        ax.set_xlim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add color legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#88cc00', label='Healthy (≥0.8)'),
            Patch(facecolor='#ffcc00', label='Warning (0.6-0.8)'),
            Patch(facecolor='#ff4444', label='Degraded (<0.6)')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_distribution_chart(self, metrics_data: Dict[str, List[Tuple[datetime, float]]]) -> str:
        """Create performance distribution chart."""
        if not metrics_data:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No Distribution Data Available', 
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=16, transform=ax.transAxes)
            ax.axis('off')
            return self._fig_to_base64(fig)
        
        # Select top 4 metrics with most data
        top_metrics = sorted(metrics_data.items(), key=lambda x: len(x[1]), reverse=True)[:4]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Value Distributions', fontsize=16, fontweight='bold')
        
        for i, (metric_name, data) in enumerate(top_metrics):
            if len(data) < 2:
                continue
                
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            values = [value for _, value in data]
            
            # Create histogram with KDE overlay
            ax.hist(values, bins=min(20, len(values)//2), alpha=0.7, density=True, edgecolor='black')
            
            if len(values) > 3:
                # Add KDE curve
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(values)
                x_range = np.linspace(min(values), max(values), 100)
                ax.plot(x_range, kde(x_range), linewidth=2, color='red')
            
            ax.set_title(metric_name.replace('_', ' ').title(), fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(top_metrics), 4):
            row, col = i // 2, i % 2
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 encoded string."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return img_base64
    
    def _get_html_template(self) -> str:
        """Get HTML template for performance report."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report_title }}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header .meta {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .status-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .status-passed { background-color: #d4edda; color: #155724; }
        .status-warning { background-color: #fff3cd; color: #856404; }
        .status-failed { background-color: #f8d7da; color: #721c24; }
        
        .section {
            background: white;
            margin-bottom: 30px;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .section h2 {
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }
        
        .metric-card h3 {
            color: #495057;
            margin-bottom: 8px;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }
        
        .chart-container {
            text-align: center;
            margin: 20px 0;
        }
        
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .recommendations {
            background: #e8f4fd;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #0066cc;
        }
        
        .recommendations ul {
            list-style: none;
            padding-left: 0;
        }
        
        .recommendations li {
            margin-bottom: 10px;
            padding: 8px;
            background: white;
            border-radius: 4px;
        }
        
        .severity-critical { border-left: 4px solid #dc3545; }
        .severity-high { border-left: 4px solid #fd7e14; }
        .severity-medium { border-left: 4px solid #ffc107; }
        .severity-low { border-left: 4px solid #28a745; }
        
        .regression-details {
            margin-top: 20px;
        }
        
        .regression-item {
            background: #fff;
            margin-bottom: 10px;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #ccc;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        th {
            background-color: #f8f9fa;
            font-weight: bold;
            color: #495057;
        }
        
        tr:hover {
            background-color: #f5f5f5;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>{{ report_title }}</h1>
            <div class="meta">
                <div>Generated: {{ generated_at }}</div>
                <div>Run ID: {{ run_id }}</div>
                <div class="status-badge status-{{ ci_status.lower() }}">
                    Status: {{ ci_status }}
                </div>
            </div>
        </div>
        
        <!-- Summary Metrics -->
        <div class="section">
            <h2>📊 Performance Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Total Tests</h3>
                    <div class="metric-value">{{ performance_data.test_results.total_tests }}</div>
                </div>
                <div class="metric-card">
                    <h3>Successful Tests</h3>
                    <div class="metric-value">{{ performance_data.test_results.successful_tests }}</div>
                </div>
                <div class="metric-card">
                    <h3>Total Regressions</h3>
                    <div class="metric-value">{{ regression_summary.total_regressions }}</div>
                </div>
                <div class="metric-card">
                    <h3>Critical Regressions</h3>
                    <div class="metric-value">{{ regression_summary.regressions_by_severity.critical }}</div>
                </div>
                <div class="metric-card">
                    <h3>Improvements</h3>
                    <div class="metric-value">{{ regression_summary.total_improvements }}</div>
                </div>
                <div class="metric-card">
                    <h3>Health Score</h3>
                    <div class="metric-value">{{ "%.2f"|format(baseline_status.average_health_score) }}</div>
                </div>
            </div>
        </div>
        
        <!-- Charts Section -->
        {% if has_charts and charts %}
        <div class="section">
            <h2>📈 Performance Visualizations</h2>
            
            {% if charts.trend_chart %}
            <h3>Performance Trends</h3>
            <div class="chart-container">
                <img src="data:image/png;base64,{{ charts.trend_chart }}" alt="Performance Trends">
            </div>
            {% endif %}
            
            {% if charts.severity_chart %}
            <h3>Regression Severity Breakdown</h3>
            <div class="chart-container">
                <img src="data:image/png;base64,{{ charts.severity_chart }}" alt="Regression Severity">
            </div>
            {% endif %}
            
            {% if charts.baseline_health_chart %}
            <h3>Baseline Health Status</h3>
            <div class="chart-container">
                <img src="data:image/png;base64,{{ charts.baseline_health_chart }}" alt="Baseline Health">
            </div>
            {% endif %}
        </div>
        {% endif %}
        
        <!-- Regression Analysis -->
        {% if regression_summary.total_regressions > 0 %}
        <div class="section">
            <h2>⚠️ Regression Analysis</h2>
            <div class="regression-details">
                {% for result in regression_summary.results %}
                {% if result.is_regression %}
                <div class="regression-item severity-{{ result.severity }}">
                    <h4>{{ result.metric_name }}</h4>
                    <p><strong>Current Value:</strong> {{ "%.2f"|format(result.current_value) }}</p>
                    <p><strong>Baseline Mean:</strong> {{ "%.2f"|format(result.baseline_mean) }}</p>
                    <p><strong>Deviation:</strong> {{ "%.2f"|format(result.deviation_std) }} standard deviations</p>
                    <p><strong>Confidence:</strong> {{ "%.1f"|format(result.confidence * 100) }}%</p>
                    <p><strong>Severity:</strong> {{ result.severity.title() }}</p>
                </div>
                {% endif %}
                {% endfor %}
            </div>
        </div>
        {% endif %}
        
        <!-- Recommendations -->
        <div class="section">
            <h2>💡 Recommendations</h2>
            <div class="recommendations">
                <ul>
                    {% for recommendation in recommendations %}
                    <li>{{ recommendation }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>
        
        <!-- Environment Info -->
        <div class="section">
            <h2>🖥️ Environment Information</h2>
            <table>
                <tr><th>Property</th><th>Value</th></tr>
                <tr><td>Python Version</td><td>{{ performance_data.environment.python_version }}</td></tr>
                <tr><td>CPU Cores</td><td>{{ performance_data.environment.cpu_count }}</td></tr>
                <tr><td>Memory (GB)</td><td>{{ performance_data.environment.memory_gb }}</td></tr>
                <tr><td>Platform</td><td>{{ performance_data.environment.platform }}</td></tr>
                <tr><td>Test Duration</td><td>{{ "%.1f"|format(performance_data.duration_seconds) }} seconds</td></tr>
            </table>
        </div>
        
        <div class="footer">
            <p>Performance Regression Report generated by Pynomaly CI/CD Pipeline</p>
            <p>For more information, visit the project repository</p>
        </div>
    </div>
</body>
</html>
"""
    
    def generate_json_report(self, performance_data: Dict[str, Any], 
                           output_file: str = None) -> str:
        """Generate detailed JSON report."""
        if output_file is None:
            output_file = self.output_dir / f"performance_report_{performance_data.get('run_id', 'latest')}.json"
        
        output_file = Path(output_file)
        
        with open(output_file, 'w') as f:
            json.dump(performance_data, f, indent=2, default=str)
        
        logger.info(f"JSON report generated: {output_file}")
        return str(output_file)
    
    def generate_csv_metrics(self, performance_data: Dict[str, Any], 
                           output_file: str = None) -> str:
        """Generate CSV file with performance metrics."""
        if output_file is None:
            output_file = self.output_dir / f"performance_metrics_{performance_data.get('run_id', 'latest')}.csv"
        
        output_file = Path(output_file)
        
        # Extract metrics data
        metrics_rows = []
        run_id = performance_data.get('run_id', 'unknown')
        timestamp = performance_data.get('timestamp', datetime.now().isoformat())
        
        for test_name, test_data in performance_data.get('test_results', {}).get('test_results', {}).items():
            if test_data.get('status') != 'success':
                continue
            
            for metric in test_data.get('metrics', []):
                metrics_rows.append({
                    'run_id': run_id,
                    'timestamp': timestamp,
                    'test_name': test_name,
                    'metric_name': metric['name'],
                    'value': metric['value'],
                    'unit': metric['unit']
                })
        
        # Create DataFrame and save
        df = pd.DataFrame(metrics_rows)
        df.to_csv(output_file, index=False)
        
        logger.info(f"CSV metrics generated: {output_file}")
        return str(output_file)


class PerformanceDashboard:
    """Simple dashboard for viewing performance trends."""
    
    def __init__(self, data_source: str):
        self.data_source = Path(data_source)
    
    def generate_dashboard_data(self, days: int = 30) -> Dict[str, Any]:
        """Generate data for performance dashboard."""
        # This would typically read from a database or time series store
        # For now, we'll create a simple structure
        
        dashboard_data = {
            'summary': {
                'total_runs': 0,
                'avg_health_score': 0.0,
                'recent_regressions': 0,
                'trend_direction': 'stable'
            },
            'metrics_trends': {},
            'recent_runs': [],
            'alerts': []
        }
        
        return dashboard_data


# Example usage
if __name__ == "__main__":
    # Example performance data
    example_data = {
        'run_id': 'test_20241201_143022',
        'timestamp': datetime.now().isoformat(),
        'duration_seconds': 45.2,
        'ci_status': 'WARNING',
        'test_results': {
            'total_tests': 4,
            'successful_tests': 4,
            'test_results': {
                'health_check': {
                    'status': 'success',
                    'metrics': [
                        {'name': 'response_time_mean', 'value': 125.5, 'unit': 'ms'},
                        {'name': 'error_rate', 'value': 0.0, 'unit': 'percent'}
                    ]
                }
            }
        },
        'regression_summary': {
            'total_regressions': 1,
            'total_improvements': 0,
            'regressions_by_severity': {'critical': 0, 'high': 1, 'medium': 0, 'low': 0},
            'has_critical_regressions': False
        },
        'baseline_status': {
            'average_health_score': 0.85,
            'metrics': {
                'response_time_mean': {'health_score': 0.9},
                'error_rate': {'health_score': 0.8}
            }
        },
        'recommendations': [
            '⚠️ 1 performance regression found. Consider investigating affected components.',
            '📊 Baseline health is good. Continue monitoring trends.'
        ],
        'environment': {
            'python_version': '3.12',
            'cpu_count': 4,
            'memory_gb': 16,
            'platform': 'Linux'
        }
    }
    
    # Generate reports
    generator = PerformanceReportGenerator()
    
    # Generate HTML report
    html_file = generator.generate_html_report(example_data)
    print(f"HTML report: {html_file}")
    
    # Generate JSON report
    json_file = generator.generate_json_report(example_data)
    print(f"JSON report: {json_file}")
    
    # Generate CSV metrics
    csv_file = generator.generate_csv_metrics(example_data)
    print(f"CSV metrics: {csv_file}")