#!/usr/bin/env python3
"""Generate human-readable complexity reports from JSON analysis data.

This script converts JSON complexity analysis results into various formats
including Markdown, HTML, and text for easier consumption and sharing.
"""

import argparse
import json
import sys
from pathlib import Path


def generate_markdown_report(data: dict) -> str:
    """Generate Markdown format report."""
    lines = []

    # Header
    lines.append("# Complexity Analysis Report")
    lines.append("")
    lines.append(f"**Generated:** {data.get('timestamp', 'Unknown')}")
    lines.append(f"**Project:** {data.get('project_path', 'Unknown')}")
    lines.append("")

    # Executive Summary
    quality = data.get("quality_assessment", {})
    maintainability = data.get("detailed_analysis", {}).get("maintainability_score", {})

    lines.append("## 📊 Executive Summary")
    lines.append("")
    lines.append(f"- **Quality Status:** {quality.get('overall', 'Unknown').upper()}")
    lines.append(f"- **Maintainability Grade:** {maintainability.get('grade', 'N/A')}")
    lines.append(
        f"- **Maintainability Score:** {maintainability.get('overall_score', 0):.1f}/100"
    )
    lines.append("")

    # Key Metrics
    metrics = data.get("metrics", {})
    lines.append("## 📈 Key Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Files | {metrics.get('total_files', 0):,} |")
    lines.append(f"| Python Files | {metrics.get('python_files', 0):,} |")
    lines.append(f"| Total Lines | {metrics.get('total_lines', 0):,} |")
    lines.append(
        f"| Cyclomatic Complexity | {metrics.get('cyclomatic_complexity', 0):.1f} |"
    )
    lines.append(f"| Dependencies | {metrics.get('total_dependencies', 0)} |")
    lines.append("")

    # Trends (if available)
    trends = data.get("baseline_comparison")
    if trends:
        lines.append("## 📈 Changes from Baseline")
        lines.append("")

        critical_trends = []
        warning_trends = []
        info_trends = []

        for metric_key, trend in trends.items():
            if trend["severity"] == "critical":
                critical_trends.append(trend)
            elif trend["severity"] == "warning":
                warning_trends.append(trend)
            else:
                info_trends.append(trend)

        if critical_trends:
            lines.append("### 🚨 Critical Changes")
            for trend in critical_trends:
                icon = (
                    "📈"
                    if trend["trend"] == "increasing"
                    else "📉"
                    if trend["trend"] == "decreasing"
                    else "➡️"
                )
                lines.append(
                    f"- {icon} **{trend['name']}**: {trend['change_percent']:+.1f}% ({trend['current']} vs {trend['baseline']})"
                )
            lines.append("")

        if warning_trends:
            lines.append("### ⚠️ Warning Changes")
            for trend in warning_trends:
                icon = (
                    "📈"
                    if trend["trend"] == "increasing"
                    else "📉"
                    if trend["trend"] == "decreasing"
                    else "➡️"
                )
                lines.append(
                    f"- {icon} **{trend['name']}**: {trend['change_percent']:+.1f}% ({trend['current']} vs {trend['baseline']})"
                )
            lines.append("")

    # Quality Issues
    if quality.get("critical_issues") or quality.get("warnings"):
        lines.append("## 🚨 Quality Issues")
        lines.append("")

        if quality.get("critical_issues"):
            lines.append("### Critical Issues")
            for issue in quality["critical_issues"]:
                lines.append(f"- ❌ {issue}")
            lines.append("")

        if quality.get("warnings"):
            lines.append("### Warnings")
            for warning in quality["warnings"]:
                lines.append(f"- ⚠️ {warning}")
            lines.append("")

    # Complexity Hotspots
    hotspots = data.get("detailed_analysis", {}).get("hotspots", [])
    if hotspots:
        lines.append("## 🔥 Complexity Hotspots")
        lines.append("")
        lines.append("| File | Lines | Complexity | Recommendation |")
        lines.append("|------|-------|------------|----------------|")
        for hotspot in hotspots[:5]:  # Top 5
            lines.append(
                f"| `{hotspot['file']}` | {hotspot['lines']} | {hotspot['estimated_complexity']:.1f} | {hotspot['recommendation']} |"
            )
        lines.append("")

    # Technical Debt
    debt = data.get("detailed_analysis", {}).get("technical_debt", {})
    if debt:
        lines.append("## 💳 Technical Debt Assessment")
        lines.append("")
        lines.append(f"- **Debt Level:** {debt.get('level', 'Unknown').upper()}")
        lines.append(f"- **Debt Score:** {debt.get('score', 0):.1f}")
        lines.append(f"- **Estimated Days:** {debt.get('estimated_days', 0):.1f}")
        lines.append("")

        if debt.get("factors"):
            lines.append("### Contributing Factors")
            for factor in debt["factors"]:
                lines.append(f"- {factor}")
            lines.append("")

    # Recommendations
    recommendations = data.get("recommendations", [])
    if recommendations:
        lines.append("## 💡 Recommendations")
        lines.append("")
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    # CI Metadata
    ci_meta = data.get("ci_metadata", {})
    if ci_meta:
        lines.append("## 🔧 Build Information")
        lines.append("")
        lines.append(f"- **Git Ref:** {ci_meta.get('git_ref', 'Unknown')}")
        lines.append(f"- **Git Commit:** {ci_meta.get('git_commit', 'Unknown')}")
        lines.append(f"- **Run ID:** {ci_meta.get('run_id', 'Unknown')}")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by Pynomaly automated complexity monitoring*")

    return "\n".join(lines)


def generate_html_report(data: dict) -> str:
    """Generate HTML format report."""
    quality = data.get("quality_assessment", {})
    maintainability = data.get("detailed_analysis", {}).get("maintainability_score", {})
    metrics = data.get("metrics", {})

    # Determine status color
    status_color = {"good": "#28a745", "warning": "#ffc107", "critical": "#dc3545"}.get(
        quality.get("overall", "good"), "#6c757d"
    )

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Complexity Analysis Report</title>
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            margin: 20px; 
            line-height: 1.6; 
            color: #333;
        }}
        .header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            padding: 20px; 
            border-radius: 8px; 
            margin-bottom: 20px;
        }}
        .status {{ 
            display: inline-block; 
            padding: 4px 12px; 
            border-radius: 20px; 
            font-weight: bold; 
            background: {status_color}; 
            color: white;
        }}
        .metric-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 15px; 
            margin: 20px 0;
        }}
        .metric-card {{ 
            background: #f8f9fa; 
            padding: 15px; 
            border-radius: 8px; 
            border-left: 4px solid #007bff;
        }}
        .metric-value {{ 
            font-size: 1.5em; 
            font-weight: bold; 
            color: #007bff;
        }}
        .section {{ 
            margin: 20px 0; 
            padding: 15px; 
            border: 1px solid #dee2e6; 
            border-radius: 8px;
        }}
        .critical {{ border-left: 4px solid #dc3545; }}
        .warning {{ border-left: 4px solid #ffc107; }}
        .info {{ border-left: 4px solid #17a2b8; }}
        table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin: 10px 0;
        }}
        th, td {{ 
            text-align: left; 
            padding: 8px; 
            border-bottom: 1px solid #ddd;
        }}
        th {{ background-color: #f2f2f2; }}
        .code {{ 
            font-family: 'Courier New', monospace; 
            background: #f4f4f4; 
            padding: 2px 4px; 
            border-radius: 3px;
        }}
        .footer {{ 
            margin-top: 30px; 
            padding-top: 15px; 
            border-top: 1px solid #dee2e6; 
            color: #6c757d; 
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Complexity Analysis Report</h1>
        <p><strong>Generated:</strong> {data.get("timestamp", "Unknown")}</p>
        <p><strong>Project:</strong> {data.get("project_path", "Unknown")}</p>
        <p><strong>Status:</strong> <span class="status">{quality.get("overall", "Unknown").upper()}</span></p>
    </div>
    
    <div class="section">
        <h2>📈 Key Metrics</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{metrics.get("total_files", 0):,}</div>
                <div>Total Files</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get("python_files", 0):,}</div>
                <div>Python Files</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get("total_lines", 0):,}</div>
                <div>Total Lines</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get("cyclomatic_complexity", 0):.1f}</div>
                <div>Cyclomatic Complexity</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{maintainability.get("grade", "N/A")}</div>
                <div>Maintainability Grade</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{maintainability.get("overall_score", 0):.1f}</div>
                <div>Maintainability Score</div>
            </div>
        </div>
    </div>
    """

    # Quality Issues
    if quality.get("critical_issues") or quality.get("warnings"):
        html += '<div class="section critical">'
        html += "<h2>🚨 Quality Issues</h2>"

        if quality.get("critical_issues"):
            html += "<h3>Critical Issues</h3><ul>"
            for issue in quality["critical_issues"]:
                html += f"<li>❌ {issue}</li>"
            html += "</ul>"

        if quality.get("warnings"):
            html += "<h3>Warnings</h3><ul>"
            for warning in quality["warnings"]:
                html += f"<li>⚠️ {warning}</li>"
            html += "</ul>"

        html += "</div>"

    # Complexity Hotspots
    hotspots = data.get("detailed_analysis", {}).get("hotspots", [])
    if hotspots:
        html += '<div class="section warning">'
        html += "<h2>🔥 Complexity Hotspots</h2>"
        html += "<table><tr><th>File</th><th>Lines</th><th>Complexity</th><th>Recommendation</th></tr>"

        for hotspot in hotspots[:5]:
            html += f"""
            <tr>
                <td><span class="code">{hotspot["file"]}</span></td>
                <td>{hotspot["lines"]}</td>
                <td>{hotspot["estimated_complexity"]:.1f}</td>
                <td>{hotspot["recommendation"]}</td>
            </tr>
            """

        html += "</table></div>"

    # Recommendations
    recommendations = data.get("recommendations", [])
    if recommendations:
        html += '<div class="section info">'
        html += "<h2>💡 Recommendations</h2><ol>"

        for rec in recommendations:
            html += f"<li>{rec}</li>"

        html += "</ol></div>"

    html += """
    <div class="footer">
        <p><em>Generated by Pynomaly automated complexity monitoring</em></p>
    </div>
</body>
</html>
    """

    return html


def generate_text_report(data: dict) -> str:
    """Generate plain text format report."""
    lines = []

    # Header
    lines.append("=" * 60)
    lines.append("COMPLEXITY ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append(f"Generated: {data.get('timestamp', 'Unknown')}")
    lines.append(f"Project: {data.get('project_path', 'Unknown')}")
    lines.append("")

    # Summary
    quality = data.get("quality_assessment", {})
    maintainability = data.get("detailed_analysis", {}).get("maintainability_score", {})

    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 20)
    lines.append(f"Quality Status: {quality.get('overall', 'Unknown').upper()}")
    lines.append(f"Maintainability Grade: {maintainability.get('grade', 'N/A')}")
    lines.append(
        f"Maintainability Score: {maintainability.get('overall_score', 0):.1f}/100"
    )
    lines.append("")

    # Metrics
    metrics = data.get("metrics", {})
    lines.append("KEY METRICS")
    lines.append("-" * 15)
    lines.append(f"Total Files: {metrics.get('total_files', 0):,}")
    lines.append(f"Python Files: {metrics.get('python_files', 0):,}")
    lines.append(f"Total Lines: {metrics.get('total_lines', 0):,}")
    lines.append(
        f"Cyclomatic Complexity: {metrics.get('cyclomatic_complexity', 0):.1f}"
    )
    lines.append(f"Dependencies: {metrics.get('total_dependencies', 0)}")
    lines.append("")

    # Quality Issues
    if quality.get("critical_issues") or quality.get("warnings"):
        lines.append("QUALITY ISSUES")
        lines.append("-" * 15)

        if quality.get("critical_issues"):
            lines.append("Critical Issues:")
            for issue in quality["critical_issues"]:
                lines.append(f"  ❌ {issue}")
            lines.append("")

        if quality.get("warnings"):
            lines.append("Warnings:")
            for warning in quality["warnings"]:
                lines.append(f"  ⚠️ {warning}")
            lines.append("")

    # Recommendations
    recommendations = data.get("recommendations", [])
    if recommendations:
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 15)
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("Generated by Pynomaly automated complexity monitoring")

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate complexity reports")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input JSON file from complexity analysis",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["markdown", "html", "text"],
        default="markdown",
        help="Output format",
    )
    parser.add_argument(
        "--output", "-o", type=Path, help="Output file path (stdout if not specified)"
    )

    args = parser.parse_args()

    try:
        # Load analysis data
        with open(args.input) as f:
            data = json.load(f)

        # Generate report
        if args.format == "markdown":
            report = generate_markdown_report(data)
        elif args.format == "html":
            report = generate_html_report(data)
        elif args.format == "text":
            report = generate_text_report(data)
        else:
            raise ValueError(f"Unsupported format: {args.format}")

        # Output report
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                f.write(report)
            print(f"✅ Report generated: {args.output}")
        else:
            print(report)

    except Exception as e:
        print(f"❌ Failed to generate report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
