"""UI Test Report Generator with Critique and Recommendations."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class UITestReportGenerator:
    """Generate comprehensive UI test reports with critiques and recommendations."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_comprehensive_report(self, test_results: dict[str, Any]) -> str:
        """Generate a comprehensive HTML report with critiques and recommendations."""

        # Analyze results and generate critiques
        analysis = self._analyze_test_results(test_results)

        # Generate HTML report
        html_content = self._generate_html_report(test_results, analysis)

        # Save report
        report_path = self.output_dir / f"ui_test_report_{self.timestamp}.html"
        report_path.write_text(html_content, encoding="utf-8")

        # Generate JSON summary
        json_summary = self._generate_json_summary(test_results, analysis)
        json_path = self.output_dir / f"ui_test_summary_{self.timestamp}.json"
        json_path.write_text(json.dumps(json_summary, indent=2), encoding="utf-8")

        return str(report_path)

    def _analyze_test_results(self, results: dict[str, Any]) -> dict[str, Any]:
        """Analyze test results and provide critiques."""
        analysis = {
            "overall_score": 0,
            "critical_issues": [],
            "warnings": [],
            "recommendations": [],
            "strengths": [],
            "category_scores": {},
        }

        # Analyze each category
        if "layout_validation" in results:
            analysis["category_scores"]["layout"] = self._analyze_layout_results(
                results["layout_validation"], analysis
            )

        if "ux_flows" in results:
            analysis["category_scores"]["ux"] = self._analyze_ux_results(
                results["ux_flows"], analysis
            )

        if "visual_regression" in results:
            analysis["category_scores"]["visual"] = self._analyze_visual_results(
                results["visual_regression"], analysis
            )

        if "accessibility" in results:
            analysis["category_scores"]["accessibility"] = (
                self._analyze_accessibility_results(results["accessibility"], analysis)
            )

        if "responsive_design" in results:
            analysis["category_scores"]["responsive"] = (
                self._analyze_responsive_results(results["responsive_design"], analysis)
            )

        # Calculate overall score
        scores = list(analysis["category_scores"].values())
        analysis["overall_score"] = sum(scores) / len(scores) if scores else 0

        # Generate overall recommendations
        self._generate_overall_recommendations(analysis)

        return analysis

    def _analyze_layout_results(self, results: dict, analysis: dict) -> float:
        """Analyze layout validation results."""
        score = 85  # Start with good score

        # Check for critical layout issues
        if not results.get("navigation_consistent", True):
            analysis["critical_issues"].append(
                "Navigation inconsistency detected across pages"
            )
            score -= 20

        if not results.get("responsive_navigation", True):
            analysis["warnings"].append("Mobile navigation may not be fully functional")
            score -= 10

        # Check for semantic HTML issues
        if not results.get("semantic_html", True):
            analysis["warnings"].append("Missing or improper semantic HTML elements")
            score -= 5

        # Add recommendations
        if score < 80:
            analysis["recommendations"].append(
                "🔧 **Layout Issues**: Review navigation consistency and semantic HTML structure"
            )
        else:
            analysis["strengths"].append(
                "✅ Layout structure is well-organized and consistent"
            )

        return max(0, min(100, score))

    def _analyze_ux_results(self, results: dict, analysis: dict) -> float:
        """Analyze UX flow results."""
        score = 90  # Start optimistic

        # Check for UX flow failures
        failed_flows = results.get("failed_flows", [])
        if failed_flows:
            analysis["critical_issues"].append(
                f"Critical UX flows failed: {', '.join(failed_flows)}"
            )
            score -= len(failed_flows) * 15

        # Check for error handling
        if not results.get("error_recovery", True):
            analysis["warnings"].append("Error recovery mechanisms may be inadequate")
            score -= 10

        # Check for performance issues
        if results.get("slow_interactions", False):
            analysis["warnings"].append("Some interactions are slower than optimal")
            score -= 5

        # Add recommendations
        if score < 75:
            analysis["recommendations"].append(
                "🚀 **UX Improvements**: Focus on error handling and interaction performance"
            )
        else:
            analysis["strengths"].append(
                "✅ User experience flows are smooth and intuitive"
            )

        return max(0, min(100, score))

    def _analyze_visual_results(self, results: dict, analysis: dict) -> float:
        """Analyze visual regression results."""
        score = 95  # Visual should be stable

        # Check for visual regressions
        regressions = results.get("regressions", [])
        if regressions:
            analysis["critical_issues"].append(
                f"Visual regressions detected in: {', '.join(regressions)}"
            )
            score -= len(regressions) * 10

        # Check baseline coverage
        baseline_coverage = results.get("baseline_coverage", 0)
        if baseline_coverage < 80:
            analysis["warnings"].append("Visual test baseline coverage is below 80%")
            score -= 5

        # Add recommendations
        if score < 85:
            analysis["recommendations"].append(
                "👁️ **Visual Consistency**: Update visual baselines and review regression areas"
            )
        else:
            analysis["strengths"].append(
                "✅ Visual consistency is maintained across components"
            )

        return max(0, min(100, score))

    def _analyze_accessibility_results(self, results: dict, analysis: dict) -> float:
        """Analyze accessibility results."""
        score = 80  # Accessibility is challenging

        # Check for critical accessibility issues
        a11y_issues = results.get("issues", [])
        critical_a11y = [
            issue for issue in a11y_issues if issue.get("severity") == "critical"
        ]

        if critical_a11y:
            analysis["critical_issues"].append(
                f"Critical accessibility violations: {len(critical_a11y)} found"
            )
            score -= len(critical_a11y) * 15

        # Check for warnings
        warning_a11y = [
            issue for issue in a11y_issues if issue.get("severity") == "warning"
        ]
        if warning_a11y:
            analysis["warnings"].append(
                f"Accessibility warnings: {len(warning_a11y)} found"
            )
            score -= len(warning_a11y) * 2

        # Check keyboard navigation
        if not results.get("keyboard_navigation", True):
            analysis["critical_issues"].append(
                "Keyboard navigation is not fully functional"
            )
            score -= 20

        # Add recommendations
        if score < 70:
            analysis["recommendations"].append(
                "♿ **Accessibility**: Priority focus needed on ARIA labels, keyboard navigation, and color contrast"
            )
        elif score < 85:
            analysis["recommendations"].append(
                "♿ **Accessibility**: Good foundation, minor improvements needed"
            )
        else:
            analysis["strengths"].append(
                "✅ Accessibility standards are well implemented"
            )

        return max(0, min(100, score))

    def _analyze_responsive_results(self, results: dict, analysis: dict) -> float:
        """Analyze responsive design results."""
        score = 85

        # Check for responsive issues
        failed_viewports = results.get("failed_viewports", [])
        if failed_viewports:
            analysis["critical_issues"].append(
                f"Responsive design failures on: {', '.join(failed_viewports)}"
            )
            score -= len(failed_viewports) * 10

        # Check for touch target issues
        if results.get("touch_target_issues", False):
            analysis["warnings"].append(
                "Some touch targets may be too small for mobile devices"
            )
            score -= 8

        # Check for horizontal scroll
        if results.get("horizontal_scroll", False):
            analysis["warnings"].append(
                "Horizontal scrolling detected on some viewports"
            )
            score -= 10

        # Add recommendations
        if score < 75:
            analysis["recommendations"].append(
                "📱 **Responsive Design**: Review mobile layouts and touch target sizes"
            )
        else:
            analysis["strengths"].append(
                "✅ Responsive design works well across devices"
            )

        return max(0, min(100, score))

    def _generate_overall_recommendations(self, analysis: dict):
        """Generate overall recommendations based on analysis."""
        overall_score = analysis["overall_score"]

        if overall_score >= 90:
            analysis["recommendations"].insert(
                0,
                "🌟 **Excellent**: Your UI is in great shape! Focus on maintaining quality.",
            )
        elif overall_score >= 80:
            analysis["recommendations"].insert(
                0, "👍 **Good**: Solid UI foundation with room for refinement."
            )
        elif overall_score >= 70:
            analysis["recommendations"].insert(
                0,
                "⚠️ **Needs Attention**: Several areas require improvement for optimal UX.",
            )
        else:
            analysis["recommendations"].insert(
                0, "🚨 **Critical**: Significant UI/UX issues need immediate attention."
            )

        # Add specific action items
        if len(analysis["critical_issues"]) > 0:
            analysis["recommendations"].append(
                f"🔴 **Immediate Action**: Address {len(analysis['critical_issues'])} critical issues first"
            )

        if len(analysis["warnings"]) > 3:
            analysis["recommendations"].append(
                "🟡 **Next Steps**: Create a backlog to systematically address warnings"
            )

    def _generate_html_report(self, results: dict, analysis: dict) -> str:
        """Generate HTML report."""
        # Get score color
        score = analysis["overall_score"]
        if score >= 90:
            score_color = "#22c55e"  # Green
        elif score >= 80:
            score_color = "#84cc16"  # Lime
        elif score >= 70:
            score_color = "#eab308"  # Yellow
        else:
            score_color = "#ef4444"  # Red

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UI Test Report - {self.timestamp}</title>
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
        .header {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            text-align: center;
        }}
        .score-circle {{
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: conic-gradient({score_color} {score * 3.6}deg, #e5e7eb 0deg);
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px;
            position: relative;
        }}
        .score-circle::before {{
            content: '';
            width: 90px;
            height: 90px;
            background: white;
            border-radius: 50%;
            position: absolute;
        }}
        .score-text {{
            font-size: 24px;
            font-weight: bold;
            color: {score_color};
            z-index: 1;
        }}
        .card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .category-scores {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .category-card {{
            text-align: center;
            padding: 20px;
        }}
        .category-score {{
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .critical {{
            color: #ef4444;
            background: #fef2f2;
            border-left: 4px solid #ef4444;
            padding: 15px;
            margin: 10px 0;
        }}
        .warning {{
            color: #f59e0b;
            background: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 15px;
            margin: 10px 0;
        }}
        .strength {{
            color: #10b981;
            background: #f0fdf4;
            border-left: 4px solid #10b981;
            padding: 15px;
            margin: 10px 0;
        }}
        .recommendation {{
            background: #f0f9ff;
            border-left: 4px solid #3b82f6;
            padding: 15px;
            margin: 10px 0;
        }}
        .screenshot {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin: 10px 0;
        }}
        h1, h2, h3 {{
            color: #1f2937;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .timestamp {{
            color: #6b7280;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 UI Test Report</h1>
        <div class="score-circle">
            <div class="score-text">{score:.0f}</div>
        </div>
        <h2>Overall UI Quality Score</h2>
        <p class="timestamp">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <div class="category-scores">
        {self._render_category_scores(analysis["category_scores"])}
    </div>

    <div class="grid">
        <div class="card">
            <h2>🚨 Critical Issues</h2>
            {self._render_issues(analysis["critical_issues"], "critical")}
        </div>

        <div class="card">
            <h2>⚠️ Warnings</h2>
            {self._render_issues(analysis["warnings"], "warning")}
        </div>
    </div>

    <div class="card">
        <h2>💪 Strengths</h2>
        {self._render_issues(analysis["strengths"], "strength")}
    </div>

    <div class="card">
        <h2>🎯 Recommendations</h2>
        {self._render_issues(analysis["recommendations"], "recommendation")}
    </div>

    <div class="card">
        <h2>📊 Detailed Test Results</h2>
        {self._render_detailed_results(results)}
    </div>

    <div class="card">
        <h2>📸 Screenshots</h2>
        {self._render_screenshots()}
    </div>

</body>
</html>
"""
        return html

    def _render_category_scores(self, scores: dict[str, float]) -> str:
        """Render category scores."""
        html = ""
        category_names = {
            "layout": "Layout",
            "ux": "UX Flows",
            "visual": "Visual",
            "accessibility": "A11y",
            "responsive": "Responsive",
        }

        for category, score in scores.items():
            name = category_names.get(category, category.title())
            color = (
                "#22c55e" if score >= 80 else "#f59e0b" if score >= 60 else "#ef4444"
            )

            html += f"""
            <div class="card category-card">
                <div class="category-score" style="color: {color};">{score:.0f}</div>
                <div>{name}</div>
            </div>
            """

        return html

    def _render_issues(self, issues: list[str], issue_type: str) -> str:
        """Render issues list."""
        if not issues:
            return f'<div class="{issue_type}">None found! 🎉</div>'

        html = ""
        for issue in issues:
            html += f'<div class="{issue_type}">{html.escape(issue)}</div>'

        return html

    def _render_detailed_results(self, results: dict) -> str:
        """Render detailed test results."""
        html = "<pre style='background: #f8fafc; padding: 20px; border-radius: 8px; overflow-x: auto;'>"
        html += html.escape(json.dumps(results, indent=2))
        html += "</pre>"
        return html

    def _render_screenshots(self) -> str:
        """Render screenshots section."""
        screenshots_dir = Path("screenshots")
        if not screenshots_dir.exists():
            return "<p>No screenshots available.</p>"

        html = "<div class='grid'>"

        for screenshot in screenshots_dir.glob("*.png"):
            html += f"""
            <div>
                <h4>{screenshot.stem.replace("_", " ").title()}</h4>
                <img src="../screenshots/{screenshot.name}" class="screenshot" alt="{screenshot.stem}">
            </div>
            """

        html += "</div>"
        return html

    def _generate_json_summary(self, results: dict, analysis: dict) -> dict:
        """Generate JSON summary for programmatic use."""
        return {
            "timestamp": self.timestamp,
            "overall_score": analysis["overall_score"],
            "category_scores": analysis["category_scores"],
            "critical_issues_count": len(analysis["critical_issues"]),
            "warnings_count": len(analysis["warnings"]),
            "strengths_count": len(analysis["strengths"]),
            "recommendations_count": len(analysis["recommendations"]),
            "test_results": results,
            "analysis": analysis,
        }
