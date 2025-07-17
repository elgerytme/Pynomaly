"""
HTML reporter for repository governance.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_reporter import BaseReporter


class HTMLReporter(BaseReporter):
    """HTML reporter for governance results."""
    
    def __init__(self, output_path: Optional[Path] = None, include_charts: bool = True):
        """Initialize the HTML reporter."""
        super().__init__(output_path)
        self.include_charts = include_charts
    
    @property
    def format_name(self) -> str:
        """Name of the report format."""
        return "HTML"
    
    @property
    def file_extension(self) -> str:
        """File extension for this report format."""
        return "html"
    
    def generate_report(self, check_results: Dict[str, Any], fix_results: Dict[str, Any] = None) -> str:
        """Generate an HTML report from check and fix results."""
        html_parts = []
        
        # HTML header
        html_parts.append(self._generate_html_header())
        
        # Body start
        html_parts.append('<body>')
        html_parts.append('<div class="container">')
        
        # Title
        html_parts.append('<h1>Repository Governance Report</h1>')
        html_parts.append(f'<p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')
        
        # Summary section
        html_parts.append(self._generate_summary_section(check_results, fix_results))
        
        # Score section
        html_parts.append(self._generate_score_section(check_results))
        
        # Charts if enabled
        if self.include_charts:
            html_parts.append(self._generate_charts_section(check_results))
        
        # Check results section
        html_parts.append(self._generate_check_results_section(check_results))
        
        # Fix results section
        if fix_results:
            html_parts.append(self._generate_fix_results_section(fix_results))
        
        # Recommendations section
        html_parts.append(self._generate_recommendations_section(check_results))
        
        # Body end
        html_parts.append('</div>')
        html_parts.append('</body>')
        html_parts.append('</html>')
        
        return '\\n'.join(html_parts)
    
    def save_report(self, report_content: str, filename: str = None) -> bool:
        """Save the report to a file."""
        try:
            if filename is None:
                filename = self.create_default_filename()
            
            if self.output_path is None:
                print(report_content)
                return True
            
            output_file = self.output_path / filename
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            output_file.write_text(report_content, encoding='utf-8')
            print(f"HTML report saved to: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save HTML report: {e}")
            return False
    
    def _generate_html_header(self) -> str:
        """Generate HTML header with CSS styles."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Repository Governance Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }
        
        h3 {
            color: #2c3e50;
            margin-top: 25px;
        }
        
        .timestamp {
            color: #7f8c8d;
            font-style: italic;
            margin-bottom: 30px;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .summary-card {
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        
        .summary-card h4 {
            margin: 0 0 10px 0;
            color: #2c3e50;
        }
        
        .summary-card .number {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }
        
        .score-container {
            display: flex;
            align-items: center;
            gap: 30px;
            margin: 20px 0;
        }
        
        .score-circle {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            font-weight: bold;
            color: white;
        }
        
        .score-details {
            flex: 1;
        }
        
        .severity-high { background-color: #e74c3c; }
        .severity-medium { background-color: #f39c12; }
        .severity-low { background-color: #f1c40f; color: #2c3e50; }
        .severity-info { background-color: #27ae60; }
        
        .violation-item {
            background-color: #f8f9fa;
            border-left: 4px solid #e74c3c;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }
        
        .violation-item.medium {
            border-left-color: #f39c12;
        }
        
        .violation-item.low {
            border-left-color: #f1c40f;
        }
        
        .violation-item.info {
            border-left-color: #27ae60;
        }
        
        .violation-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }
        
        .severity-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
            color: white;
        }
        
        .checker-section {
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
        }
        
        .checker-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 15px;
        }
        
        .checker-status {
            font-size: 24px;
        }
        
        .checker-score {
            background-color: #3498db;
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: bold;
        }
        
        .fix-result {
            background-color: #f8f9fa;
            border-left: 4px solid #27ae60;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }
        
        .fix-result.failed {
            border-left-color: #e74c3c;
        }
        
        .recommendations {
            background-color: #e8f5e8;
            border: 1px solid #27ae60;
            border-radius: 8px;
            padding: 20px;
        }
        
        .recommendations ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        
        .recommendations li {
            margin: 8px 0;
        }
        
        .chart-container {
            margin: 20px 0;
            text-align: center;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        
        .no-violations {
            color: #27ae60;
            font-weight: bold;
            text-align: center;
            padding: 20px;
            background-color: #e8f5e8;
            border-radius: 8px;
        }
    </style>
</head>'''
    
    def _generate_summary_section(self, check_results: Dict[str, Any], fix_results: Dict[str, Any] = None) -> str:
        """Generate the summary section."""
        stats = self.extract_summary_stats(check_results)
        
        html = '<h2>Summary</h2>'
        html += '<div class="summary-grid">'
        
        # Check summary cards
        html += f'''
        <div class="summary-card">
            <h4>Checkers Run</h4>
            <div class="number">{stats['checkers_run']}</div>
        </div>
        <div class="summary-card">
            <h4>Passed</h4>
            <div class="number" style="color: #27ae60;">{stats['checkers_passed']}</div>
        </div>
        <div class="summary-card">
            <h4>Failed</h4>
            <div class="number" style="color: #e74c3c;">{stats['checkers_failed']}</div>
        </div>
        <div class="summary-card">
            <h4>Total Violations</h4>
            <div class="number" style="color: #e74c3c;">{stats['total_violations']}</div>
        </div>
        '''
        
        # Severity breakdown
        html += f'''
        <div class="summary-card">
            <h4>High Severity</h4>
            <div class="number" style="color: #e74c3c;">{stats['high_severity']}</div>
        </div>
        <div class="summary-card">
            <h4>Medium Severity</h4>
            <div class="number" style="color: #f39c12;">{stats['medium_severity']}</div>
        </div>
        <div class="summary-card">
            <h4>Low Severity</h4>
            <div class="number" style="color: #f1c40f;">{stats['low_severity']}</div>
        </div>
        <div class="summary-card">
            <h4>Info</h4>
            <div class="number" style="color: #27ae60;">{stats['info_severity']}</div>
        </div>
        '''
        
        html += '</div>'
        
        # Fix summary if available
        if fix_results:
            fix_stats = self.extract_fix_summary(fix_results)
            html += '<h3>Fix Summary</h3>'
            html += '<div class="summary-grid">'
            html += f'''
            <div class="summary-card">
                <h4>Fixes Attempted</h4>
                <div class="number">{fix_stats['total_fixes_attempted']}</div>
            </div>
            <div class="summary-card">
                <h4>Successful</h4>
                <div class="number" style="color: #27ae60;">{fix_stats['successful_fixes']}</div>
            </div>
            <div class="summary-card">
                <h4>Failed</h4>
                <div class="number" style="color: #e74c3c;">{fix_stats['failed_fixes']}</div>
            </div>
            <div class="summary-card">
                <h4>Files Changed</h4>
                <div class="number">{fix_stats['files_changed']}</div>
            </div>
            '''
            html += '</div>'
        
        return html
    
    def _generate_score_section(self, check_results: Dict[str, Any]) -> str:
        """Generate the score section."""
        overall_score = self.calculate_overall_score(check_results)
        grade = self.get_score_grade(overall_score)
        
        # Determine color based on score
        if overall_score >= 80:
            color = '#27ae60'
        elif overall_score >= 60:
            color = '#f39c12'
        else:
            color = '#e74c3c'
        
        html = '<h2>Overall Score</h2>'
        html += '<div class="score-container">'
        html += f'<div class="score-circle" style="background-color: {color};">'
        html += f'{overall_score:.1f}<br><small>{grade}</small>'
        html += '</div>'
        html += '<div class="score-details">'
        html += '<h3>Score Breakdown by Checker</h3>'
        html += '<table>'
        html += '<tr><th>Checker</th><th>Score</th><th>Status</th></tr>'
        
        for checker_name, result in check_results.items():
            if isinstance(result, dict):
                score = result.get("score", 0)
                violations = result.get("violations", [])
                status = "✅ Pass" if not violations else "❌ Fail"
                html += f'<tr><td>{checker_name}</td><td>{score:.1f}</td><td>{status}</td></tr>'
        
        html += '</table>'
        html += '</div>'
        html += '</div>'
        
        return html
    
    def _generate_charts_section(self, check_results: Dict[str, Any]) -> str:
        """Generate charts section using Chart.js."""
        stats = self.extract_summary_stats(check_results)
        
        html = '<h2>Visual Overview</h2>'
        html += '<div class="chart-container">'
        
        # Simple text-based chart for now (could be enhanced with Chart.js)
        html += '<h3>Severity Distribution</h3>'
        html += '<div style="display: flex; justify-content: center; gap: 20px; margin: 20px 0;">'
        
        total_violations = stats['total_violations']
        if total_violations > 0:
            high_pct = (stats['high_severity'] / total_violations) * 100
            medium_pct = (stats['medium_severity'] / total_violations) * 100
            low_pct = (stats['low_severity'] / total_violations) * 100
            info_pct = (stats['info_severity'] / total_violations) * 100
            
            html += f'<div style="text-align: center;"><div style="width: 50px; height: {high_pct * 2}px; background-color: #e74c3c; margin: 0 auto;"></div><br>High<br>{stats["high_severity"]}</div>'
            html += f'<div style="text-align: center;"><div style="width: 50px; height: {medium_pct * 2}px; background-color: #f39c12; margin: 0 auto;"></div><br>Medium<br>{stats["medium_severity"]}</div>'
            html += f'<div style="text-align: center;"><div style="width: 50px; height: {low_pct * 2}px; background-color: #f1c40f; margin: 0 auto;"></div><br>Low<br>{stats["low_severity"]}</div>'
            html += f'<div style="text-align: center;"><div style="width: 50px; height: {info_pct * 2}px; background-color: #27ae60; margin: 0 auto;"></div><br>Info<br>{stats["info_severity"]}</div>'
        else:
            html += '<div class="no-violations">No violations found! 🎉</div>'
        
        html += '</div>'
        html += '</div>'
        
        return html
    
    def _generate_check_results_section(self, check_results: Dict[str, Any]) -> str:
        """Generate the check results section."""
        html = '<h2>Check Results</h2>'
        
        for checker_name, result in check_results.items():
            if not isinstance(result, dict):
                continue
            
            violations = result.get("violations", [])
            score = result.get("score", 0)
            
            html += '<div class="checker-section">'
            html += '<div class="checker-header">'
            html += f'<div class="checker-status">{"✅" if not violations else "❌"}</div>'
            html += f'<h3>{checker_name}</h3>'
            html += f'<div class="checker-score">Score: {score:.1f}</div>'
            html += '</div>'
            
            if violations:
                for violation in violations:
                    severity = violation.get("severity", "info")
                    message = violation.get("message", "No message")
                    total_count = violation.get("total_count", 0)
                    
                    html += f'<div class="violation-item {severity}">'
                    html += '<div class="violation-header">'
                    html += f'<span class="severity-badge severity-{severity}">{severity}</span>'
                    html += f'<strong>{message}</strong>'
                    if total_count > 0:
                        html += f'<span style="color: #7f8c8d;">(Count: {total_count})</span>'
                    html += '</div>'
                    
                    # Show violation details
                    violation_details = violation.get("violations", [])
                    if violation_details:
                        html += '<div style="margin-top: 10px;">'
                        html += '<strong>Affected Files:</strong>'
                        html += '<ul>'
                        for detail in violation_details[:5]:  # Show first 5
                            if isinstance(detail, dict):
                                file_path = detail.get("file", "")
                                line = detail.get("line", "")
                                if file_path:
                                    html += f'<li><code>{file_path}</code>{f":{line}" if line else ""}</li>'
                        if len(violation_details) > 5:
                            html += f'<li>... and {len(violation_details) - 5} more files</li>'
                        html += '</ul>'
                        html += '</div>'
                    
                    html += '</div>'
            else:
                html += '<div class="no-violations">No violations found</div>'
            
            html += '</div>'
        
        return html
    
    def _generate_fix_results_section(self, fix_results: Dict[str, Any]) -> str:
        """Generate the fix results section."""
        html = '<h2>Fix Results</h2>'
        
        for fixer_name, results in fix_results.items():
            if not isinstance(results, list):
                continue
            
            html += f'<h3>{fixer_name}</h3>'
            
            for result in results:
                if not isinstance(result, dict):
                    continue
                
                success = result.get("success", False)
                message = result.get("message", "No message")
                files_changed = result.get("files_changed", [])
                
                css_class = "fix-result" if success else "fix-result failed"
                status = "✅" if success else "❌"
                
                html += f'<div class="{css_class}">'
                html += f'<div><strong>{status} {message}</strong></div>'
                
                if files_changed:
                    html += f'<div style="margin-top: 10px; color: #7f8c8d;">Files changed: {len(files_changed)}</div>'
                
                html += '</div>'
        
        return html
    
    def _generate_recommendations_section(self, check_results: Dict[str, Any]) -> str:
        """Generate the recommendations section."""
        recommendations = self.get_recommendations(check_results)
        
        html = '<h2>Recommendations</h2>'
        html += '<div class="recommendations">'
        
        if recommendations:
            html += '<ul>'
            for recommendation in recommendations:
                html += f'<li>{recommendation}</li>'
            html += '</ul>'
        else:
            html += '<div class="no-violations">No recommendations - all checks passed! 🎉</div>'
        
        html += '</div>'
        
        return html