#!/usr/bin/env python3
"""
Report Generator
================
Generates comprehensive reports in multiple formats from validation results
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import asdict

from ..core.base_validator import ValidationReport, ValidationResult, RuleViolation


class ReportGenerator:
    """
    Generates validation reports in multiple formats.
    
    Supported formats:
    - HTML: Interactive web dashboard
    - Markdown: GitHub/GitLab compatible reports
    - JSON: Machine-readable structured data
    - SARIF: Static Analysis Results Interchange Format
    - JUnit: CI/CD compatible test results
    """
    
    def __init__(self, template_path: Optional[Union[str, Path]] = None):
        self.template_path = Path(template_path) if template_path else None
        self.templates_dir = Path(__file__).parent.parent / "templates"
    
    def generate_html_report(self, report: Union[ValidationReport, Dict[str, Any]]) -> str:
        """Generate interactive HTML dashboard report"""
        if isinstance(report, dict):
            data = report
        else:
            data = asdict(report)
        
        # HTML template with embedded CSS and JavaScript
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Best Practices Validation Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6; color: #333; background: #f5f7fa;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: #fff; border-radius: 8px; padding: 30px; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .score-card {{ text-align: center; margin-bottom: 30px; }}
        .score {{ font-size: 4rem; font-weight: bold; margin-bottom: 10px; }}
        .score.grade-a {{ color: #10b981; }}
        .score.grade-b {{ color: #f59e0b; }}
        .score.grade-c {{ color: #ef4444; }}
        .grade {{ font-size: 1.5rem; color: #6b7280; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 30px; }}
        .stat {{ background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat-value {{ font-size: 2rem; font-weight: bold; margin-bottom: 5px; }}
        .stat-label {{ color: #6b7280; font-size: 0.9rem; }}
        .categories {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .category {{ background: #fff; border-radius: 8px; padding: 25px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .category-header {{ display: flex; justify-content: between; align-items: center; margin-bottom: 15px; }}
        .category-title {{ font-size: 1.25rem; font-weight: bold; text-transform: capitalize; }}
        .category-score {{ font-size: 1.5rem; font-weight: bold; }}
        .violations {{ margin-top: 30px; }}
        .violation {{ background: #fff; border-left: 4px solid #ef4444; padding: 15px; margin-bottom: 10px; border-radius: 0 8px 8px 0; }}
        .violation.high {{ border-left-color: #f59e0b; }}
        .violation.medium {{ border-left-color: #3b82f6; }}
        .violation.low {{ border-left-color: #6b7280; }}
        .violation-header {{ font-weight: bold; margin-bottom: 5px; }}
        .violation-details {{ font-size: 0.9rem; color: #6b7280; }}
        .recommendation {{ background: #eff6ff; border: 1px solid #dbeafe; padding: 15px; border-radius: 8px; margin: 10px 0; }}
        .footer {{ text-align: center; margin-top: 50px; padding: 20px; color: #6b7280; }}
        .progress-bar {{ width: 100%; height: 8px; background: #e5e7eb; border-radius: 4px; overflow: hidden; margin: 10px 0; }}
        .progress-fill {{ height: 100%; transition: width 0.3s ease; }}
        .progress-fill.grade-a {{ background: #10b981; }}
        .progress-fill.grade-b {{ background: #f59e0b; }}
        .progress-fill.grade-c {{ background: #ef4444; }}
        .tab-container {{ margin: 30px 0; }}
        .tabs {{ display: flex; border-bottom: 2px solid #e5e7eb; }}
        .tab {{ padding: 12px 24px; cursor: pointer; border-bottom: 2px solid transparent; }}
        .tab.active {{ border-bottom-color: #3b82f6; color: #3b82f6; font-weight: bold; }}
        .tab-content {{ display: none; padding: 20px 0; }}
        .tab-content.active {{ display: block; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="score-card">
                <div class="score grade-{self._get_grade_class(data.get('compliance_score', {}).get('grade', 'F'))}">
                    {data.get('compliance_score', {}).get('overall_score', 0):.1f}%
                </div>
                <div class="grade">Grade: {data.get('compliance_score', {}).get('grade', 'F')}</div>
                <div class="progress-bar">
                    <div class="progress-fill grade-{self._get_grade_class(data.get('compliance_score', {}).get('grade', 'F'))}" 
                         style="width: {data.get('compliance_score', {}).get('overall_score', 0)}%"></div>
                </div>
            </div>
            
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{data.get('compliance_score', {}).get('total_violations', 0)}</div>
                    <div class="stat-label">Total Violations</div>
                </div>
                <div class="stat">
                    <div class="stat-value" style="color: #ef4444;">{data.get('compliance_score', {}).get('critical_violations', 0)}</div>
                    <div class="stat-label">Critical</div>
                </div>
                <div class="stat">
                    <div class="stat-value" style="color: #f59e0b;">{data.get('compliance_score', {}).get('high_violations', 0)}</div>
                    <div class="stat-label">High Priority</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{len(data.get('category_results', {}))}</div>
                    <div class="stat-label">Categories Validated</div>
                </div>
            </div>
        </div>

        <div class="tab-container">
            <div class="tabs">
                <div class="tab active" onclick="showTab('overview')">Overview</div>
                <div class="tab" onclick="showTab('categories')">Categories</div>
                <div class="tab" onclick="showTab('violations')">Violations</div>
                <div class="tab" onclick="showTab('recommendations')">Recommendations</div>
            </div>

            <div id="overview" class="tab-content active">
                <h2>Project Overview</h2>
                <p><strong>Project:</strong> {data.get('project_name', 'Unknown')}</p>
                <p><strong>Validation Date:</strong> {data.get('compliance_score', {}).get('timestamp', 'Unknown')}</p>
                <p><strong>Execution Time:</strong> {data.get('compliance_score', {}).get('execution_time', 0):.2f} seconds</p>
                
                {self._generate_recommendations_html(data.get('compliance_score', {}).get('recommendations', []))}
            </div>

            <div id="categories" class="tab-content">
                <h2>Category Breakdown</h2>
                <div class="categories">
                    {self._generate_categories_html(data.get('category_results', {}), data.get('compliance_score', {}).get('category_scores', {}))}
                </div>
            </div>

            <div id="violations" class="tab-content">
                <h2>All Violations</h2>
                <div class="violations">
                    {self._generate_violations_html(data.get('all_violations', []))}
                </div>
            </div>

            <div id="recommendations" class="tab-content">
                <h2>Recommendations</h2>
                {self._generate_recommendations_html(data.get('compliance_score', {}).get('recommendations', []))}
            </div>
        </div>

        <div class="footer">
            <p>Generated by Best Practices Framework v1.0.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>

    <script>
        function showTab(tabName) {{
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {{
                content.classList.remove('active');
            }});
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }}
    </script>
</body>
</html>"""
        
        return html_template
    
    def generate_markdown_report(self, report: Union[ValidationReport, Dict[str, Any]]) -> str:
        """Generate markdown report for GitHub/GitLab"""
        if isinstance(report, dict):
            data = report
        else:
            data = asdict(report)
        
        score = data.get('compliance_score', {}).get('overall_score', 0)
        grade = data.get('compliance_score', {}).get('grade', 'F')
        
        # Choose emoji based on grade
        grade_emoji = {'A+': '🏆', 'A': '🥇', 'A-': '🥈', 'B+': '🥉', 'B': '✅'}.get(grade, '❌')
        
        markdown = f"""# 🏗️ Best Practices Validation Report

## {grade_emoji} Overall Score: {score:.1f}% (Grade {grade})

**Project:** {data.get('project_name', 'Unknown')}  
**Validation Date:** {data.get('compliance_score', {}).get('timestamp', 'Unknown')}  
**Execution Time:** {data.get('compliance_score', {}).get('execution_time', 0):.2f} seconds

---

## 📊 Summary

| Metric | Value |
|--------|-------|
| Overall Score | {score:.1f}% |
| Grade | {grade} |
| Total Violations | {data.get('compliance_score', {}).get('total_violations', 0)} |
| Critical Violations | {data.get('compliance_score', {}).get('critical_violations', 0)} |
| High Priority Violations | {data.get('compliance_score', {}).get('high_violations', 0)} |
| Categories Validated | {len(data.get('category_results', {}))} |

---

## 🎯 Category Breakdown

{self._generate_categories_markdown(data.get('category_results', {}), data.get('compliance_score', {}).get('category_scores', {}))}

---

## 🚨 Top Violations

{self._generate_violations_markdown(data.get('all_violations', [])[:10])}

---

## 💡 Recommendations

{self._generate_recommendations_markdown(data.get('compliance_score', {}).get('recommendations', []))}

---

*Report generated by Best Practices Framework v1.0.0*
"""
        
        return markdown
    
    def generate_json_report(self, report: Union[ValidationReport, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate machine-readable JSON report"""
        if isinstance(report, dict):
            return report
        
        return asdict(report)
    
    def generate_sarif_report(self, report: Union[ValidationReport, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate SARIF (Static Analysis Results Interchange Format) report"""
        if isinstance(report, dict):
            violations = report.get('all_violations', [])
        else:
            violations = report.all_violations
        
        # Convert violations to SARIF format
        results = []
        rules = {}
        
        for violation in violations:
            rule_id = violation.get('rule_id') if isinstance(violation, dict) else violation.rule_id
            message = violation.get('message') if isinstance(violation, dict) else violation.message
            file_path = violation.get('file_path') if isinstance(violation, dict) else violation.file_path
            line_number = violation.get('line_number') if isinstance(violation, dict) else violation.line_number
            severity = violation.get('severity') if isinstance(violation, dict) else violation.severity
            
            # Map severity to SARIF levels
            sarif_level = {
                'critical': 'error',
                'high': 'error', 
                'medium': 'warning',
                'low': 'note',
                'info': 'note'
            }.get(severity, 'warning')
            
            # Add rule definition
            if rule_id not in rules:
                rules[rule_id] = {
                    "id": rule_id,
                    "shortDescription": {"text": message},
                    "fullDescription": {"text": message},
                    "defaultConfiguration": {"level": sarif_level}
                }
            
            # Create result
            result = {
                "ruleId": rule_id,
                "message": {"text": message},
                "level": sarif_level,
                "locations": []
            }
            
            if file_path and line_number:
                result["locations"].append({
                    "physicalLocation": {
                        "artifactLocation": {"uri": str(file_path)},
                        "region": {"startLine": line_number}
                    }
                })
            
            results.append(result)
        
        # Generate SARIF document
        sarif = {
            "version": "2.1.0",
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": "Best Practices Framework",
                        "version": "1.0.0",
                        "informationUri": "https://github.com/best-practices-framework/best-practices-framework",
                        "rules": list(rules.values())
                    }
                },
                "results": results
            }]
        }
        
        return sarif
    
    def generate_junit_report(self, report: Union[ValidationReport, Dict[str, Any]]) -> str:
        """Generate JUnit XML report for CI/CD integration"""
        if isinstance(report, dict):
            data = report
        else:
            data = asdict(report)
        
        violations = data.get('all_violations', [])
        category_results = data.get('category_results', {})
        
        # Count failures by severity
        failures = sum(1 for v in violations if v.get('severity') in ['critical', 'high'])
        errors = sum(1 for v in violations if v.get('severity') == 'critical')
        
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="BestPracticesValidation" tests="{len(category_results)}" failures="{failures}" errors="{errors}" time="{data.get('compliance_score', {}).get('execution_time', 0):.3f}">
"""
        
        for category, results in category_results.items():
            category_violations = [v for v in violations if v.get('category') == category]
            category_failures = sum(1 for v in category_violations if v.get('severity') in ['critical', 'high'])
            category_errors = sum(1 for v in category_violations if v.get('severity') == 'critical')
            
            xml += f'  <testsuite name="{category}" tests="{len(results)}" failures="{category_failures}" errors="{category_errors}" time="0">\n'
            
            for result in results:
                result_violations = [v for v in category_violations if any(r.get('validator_name', '') == result.get('validator_name', '') for r in [result])]
                
                if result_violations:
                    xml += f'    <testcase name="{result.get("validator_name", "unknown")}" classname="{category}">\n'
                    
                    for violation in result_violations:
                        if violation.get('severity') == 'critical':
                            xml += f'      <error message="{violation.get("message", "")}" type="{violation.get("rule_id", "")}">\n'
                            xml += f'        File: {violation.get("file_path", "N/A")}\n'
                            xml += f'        Line: {violation.get("line_number", "N/A")}\n'
                            xml += f'        Suggestion: {violation.get("suggestion", "N/A")}\n'
                            xml += '      </error>\n'
                        elif violation.get('severity') == 'high':
                            xml += f'      <failure message="{violation.get("message", "")}" type="{violation.get("rule_id", "")}">\n'
                            xml += f'        File: {violation.get("file_path", "N/A")}\n'
                            xml += f'        Line: {violation.get("line_number", "N/A")}\n'
                            xml += f'        Suggestion: {violation.get("suggestion", "N/A")}\n'
                            xml += '      </failure>\n'
                    
                    xml += '    </testcase>\n'
                else:
                    xml += f'    <testcase name="{result.get("validator_name", "unknown")}" classname="{category}"/>\n'
            
            xml += '  </testsuite>\n'
        
        xml += '</testsuites>\n'
        
        return xml
    
    def _get_grade_class(self, grade: str) -> str:
        """Get CSS class for grade"""
        if grade in ['A+', 'A', 'A-']:
            return 'a'
        elif grade in ['B+', 'B', 'B-']:
            return 'b' 
        else:
            return 'c'
    
    def _generate_categories_html(self, category_results: Dict, category_scores: Dict) -> str:
        """Generate HTML for category breakdown"""
        html = ""
        
        for category, results in category_results.items():
            score = category_scores.get(category, 0)
            violations_count = sum(len(r.get('violations', [])) for r in results)
            
            html += f"""
            <div class="category">
                <div class="category-header">
                    <div class="category-title">{category.replace('_', ' ')}</div>
                    <div class="category-score" style="color: {'#10b981' if score >= 80 else '#f59e0b' if score >= 60 else '#ef4444'}">{score:.1f}%</div>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill grade-{'a' if score >= 80 else 'b' if score >= 60 else 'c'}" style="width: {score}%"></div>
                </div>
                <p>{len(results)} validators, {violations_count} violations</p>
            </div>
            """
        
        return html
    
    def _generate_violations_html(self, violations: List) -> str:
        """Generate HTML for violations list"""
        if not violations:
            return "<p>🎉 No violations found!</p>"
        
        html = ""
        for violation in violations[:20]:  # Show top 20 violations
            severity = violation.get('severity', 'medium')
            html += f"""
            <div class="violation {severity}">
                <div class="violation-header">{violation.get('rule_id', 'Unknown')}: {violation.get('message', '')}</div>
                <div class="violation-details">
                    File: {violation.get('file_path', 'N/A')} | Line: {violation.get('line_number', 'N/A')} | Severity: {severity.upper()}
                </div>
                {f'<div class="violation-details">💡 {violation.get("suggestion", "")}</div>' if violation.get('suggestion') else ''}
            </div>
            """
        
        if len(violations) > 20:
            html += f"<p><em>... and {len(violations) - 20} more violations</em></p>"
        
        return html
    
    def _generate_recommendations_html(self, recommendations: List[str]) -> str:
        """Generate HTML for recommendations"""
        if not recommendations:
            return "<p>No specific recommendations at this time.</p>"
        
        html = ""
        for i, rec in enumerate(recommendations, 1):
            html += f'<div class="recommendation">{i}. {rec}</div>'
        
        return html
    
    def _generate_categories_markdown(self, category_results: Dict, category_scores: Dict) -> str:
        """Generate markdown for category breakdown"""
        markdown = "| Category | Score | Grade | Violations |\n|----------|-------|-------|------------|\n"
        
        for category, results in category_results.items():
            score = category_scores.get(category, 0)
            violations_count = sum(len(r.get('violations', [])) for r in results)
            grade = self._calculate_grade(score)
            
            markdown += f"| {category.replace('_', ' ').title()} | {score:.1f}% | {grade} | {violations_count} |\n"
        
        return markdown
    
    def _generate_violations_markdown(self, violations: List) -> str:
        """Generate markdown for violations"""
        if not violations:
            return "🎉 **No violations found!**"
        
        markdown = ""
        for violation in violations:
            severity_emoji = {'critical': '🚨', 'high': '⚠️', 'medium': '⚡', 'low': '💡', 'info': 'ℹ️'}.get(violation.get('severity', 'medium'), '⚡')
            
            markdown += f"""
### {severity_emoji} {violation.get('rule_id', 'Unknown')}

**Message:** {violation.get('message', '')}  
**File:** `{violation.get('file_path', 'N/A')}:{violation.get('line_number', 'N/A')}`  
**Severity:** {violation.get('severity', 'medium').upper()}  
"""
            if violation.get('suggestion'):
                markdown += f"**Suggestion:** {violation.get('suggestion')}\n"
            
            markdown += "\n---\n"
        
        return markdown
    
    def _generate_recommendations_markdown(self, recommendations: List[str]) -> str:
        """Generate markdown for recommendations"""
        if not recommendations:
            return "No specific recommendations at this time."
        
        markdown = ""
        for i, rec in enumerate(recommendations, 1):
            markdown += f"{i}. {rec}\n"
        
        return markdown
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from score"""
        if score >= 95:
            return 'A+'
        elif score >= 90:
            return 'A'
        elif score >= 85:
            return 'A-'
        elif score >= 80:
            return 'B+'
        elif score >= 75:
            return 'B'
        elif score >= 70:
            return 'B-'
        elif score >= 65:
            return 'C+'
        elif score >= 60:
            return 'C'
        elif score >= 55:
            return 'C-'
        elif score >= 50:
            return 'D'
        else:
            return 'F'