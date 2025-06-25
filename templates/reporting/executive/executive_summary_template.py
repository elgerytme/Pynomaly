#!/usr/bin/env python3
"""
Executive Summary Report Template for Pynomaly Anomaly Detection

This template generates business-focused executive summaries with high-level metrics,
business impact analysis, and strategic recommendations.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import jinja2
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class ExecutiveSummaryData:
    """Data structure for executive summary metrics."""

    # Basic metrics
    total_records: int
    anomaly_count: int
    detection_rate: float
    false_positive_rate: float
    processing_time_hours: float

    # Business metrics
    financial_impact_usd: float | None = None
    risk_score: float | None = None
    trend_direction: str | None = None  # "increasing", "decreasing", "stable"

    # Compliance metrics
    regulatory_violations: int | None = None
    compliance_score: float | None = None

    # System metrics
    system_uptime: float | None = None
    alert_volume: int | None = None

    # Metadata
    report_period: str | None = None
    data_sources: list[str] | None = None
    algorithms_used: list[str] | None = None


class ExecutiveReportGenerator:
    """Generate executive summary reports for anomaly detection results."""

    def __init__(self, template_dir: str | None = None):
        """Initialize the executive report generator.

        Args:
            template_dir: Directory containing Jinja2 templates
        """
        self.template_dir = (
            Path(template_dir) if template_dir else Path(__file__).parent
        )
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir)),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
        )

    def generate_summary(
        self,
        data: ExecutiveSummaryData,
        output_path: str,
        format_type: str = "html",
        include_charts: bool = True,
        template_name: str | None = None,
    ) -> str:
        """Generate executive summary report.

        Args:
            data: Executive summary data
            output_path: Output file path
            format_type: Output format ("html", "pdf", "markdown")
            include_charts: Whether to include visualization charts
            template_name: Custom template name (optional)

        Returns:
            Path to generated report
        """
        # Calculate derived metrics
        anomaly_rate = (
            data.anomaly_count / data.total_records if data.total_records > 0 else 0
        )

        # Determine risk level
        risk_level = self._calculate_risk_level(data)

        # Generate charts if requested
        chart_paths = []
        if include_charts:
            chart_paths = self._generate_charts(data, output_path)

        # Prepare template context
        context = {
            "data": data,
            "anomaly_rate": anomaly_rate,
            "risk_level": risk_level,
            "chart_paths": chart_paths,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "recommendations": self._generate_recommendations(data),
            "key_findings": self._generate_key_findings(data),
            "next_steps": self._generate_next_steps(data),
        }

        # Select template
        template_file = template_name or f"executive_summary.{format_type}.j2"

        try:
            template = self.jinja_env.get_template(template_file)
            rendered_content = template.render(**context)

            # Save rendered report
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(rendered_content)

            return str(output_file)

        except jinja2.TemplateNotFound:
            # Fallback to programmatic generation
            return self._generate_fallback_report(data, output_path, format_type)

    def _calculate_risk_level(self, data: ExecutiveSummaryData) -> str:
        """Calculate overall risk level based on metrics."""
        score = 0

        # Anomaly rate factor
        anomaly_rate = (
            data.anomaly_count / data.total_records if data.total_records > 0 else 0
        )
        if anomaly_rate > 0.1:
            score += 3
        elif anomaly_rate > 0.05:
            score += 2
        elif anomaly_rate > 0.02:
            score += 1

        # False positive rate factor
        if data.false_positive_rate > 0.2:
            score += 2
        elif data.false_positive_rate > 0.1:
            score += 1

        # Financial impact factor
        if data.financial_impact_usd and data.financial_impact_usd > 1000000:
            score += 3
        elif data.financial_impact_usd and data.financial_impact_usd > 100000:
            score += 2
        elif data.financial_impact_usd and data.financial_impact_usd > 10000:
            score += 1

        # Risk score factor
        if data.risk_score and data.risk_score > 0.8:
            score += 2
        elif data.risk_score and data.risk_score > 0.6:
            score += 1

        # Determine risk level
        if score >= 7:
            return "HIGH"
        elif score >= 4:
            return "MEDIUM"
        elif score >= 2:
            return "LOW"
        else:
            return "MINIMAL"

    def _generate_charts(
        self, data: ExecutiveSummaryData, output_path: str
    ) -> list[str]:
        """Generate visualization charts for the report."""
        charts = []
        output_dir = Path(output_path).parent / "charts"
        output_dir.mkdir(exist_ok=True)

        # Set style
        plt.style.use("default")
        sns.set_palette("husl")

        # Chart 1: Anomaly Detection Overview
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(
            "Anomaly Detection Executive Dashboard", fontsize=16, fontweight="bold"
        )

        # Anomaly vs Normal pie chart
        labels = ["Normal", "Anomalies"]
        sizes = [data.total_records - data.anomaly_count, data.anomaly_count]
        colors = ["#2ecc71", "#e74c3c"]
        ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax1.set_title("Detection Results")

        # Risk level gauge (approximation with bar chart)
        risk_levels = ["MINIMAL", "LOW", "MEDIUM", "HIGH"]
        risk_colors = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]
        current_risk = self._calculate_risk_level(data)
        risk_values = [1 if level == current_risk else 0.3 for level in risk_levels]
        ax2.bar(risk_levels, risk_values, color=risk_colors, alpha=0.8)
        ax2.set_title("Risk Level")
        ax2.set_ylabel("Risk Score")

        # Performance metrics
        metrics = ["Detection Rate", "False Positive Rate"]
        values = [data.detection_rate, data.false_positive_rate]
        colors_perf = ["#3498db", "#e74c3c"]
        bars = ax3.bar(metrics, values, color=colors_perf, alpha=0.8)
        ax3.set_title("Performance Metrics")
        ax3.set_ylabel("Rate")
        ax3.set_ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, values, strict=False):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{value:.1%}",
                ha="center",
                va="bottom",
            )

        # Financial impact (if available)
        if data.financial_impact_usd:
            impact_data = ["Potential Loss\nPrevented", "Investigation\nCosts"]
            impact_values = [data.financial_impact_usd, data.financial_impact_usd * 0.1]
            ax4.bar(impact_data, impact_values, color=["#27ae60", "#e67e22"], alpha=0.8)
            ax4.set_title("Financial Impact (USD)")
            ax4.set_ylabel("Amount")

            # Format y-axis as currency
            ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
        else:
            ax4.text(
                0.5,
                0.5,
                "Financial Impact\nData Not Available",
                ha="center",
                va="center",
                transform=ax4.transAxes,
                fontsize=12,
                style="italic",
            )
            ax4.set_title("Financial Impact")

        plt.tight_layout()
        chart_path = output_dir / "executive_dashboard.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()
        charts.append(str(chart_path))

        return charts

    def _generate_recommendations(self, data: ExecutiveSummaryData) -> list[str]:
        """Generate strategic recommendations based on data."""
        recommendations = []

        anomaly_rate = (
            data.anomaly_count / data.total_records if data.total_records > 0 else 0
        )

        # Anomaly rate recommendations
        if anomaly_rate > 0.1:
            recommendations.append(
                "🔴 **HIGH PRIORITY**: Anomaly rate exceeds 10%. "
                "Immediate investigation required to identify root causes."
            )
        elif anomaly_rate > 0.05:
            recommendations.append(
                "🟡 **MEDIUM PRIORITY**: Elevated anomaly rate detected. "
                "Consider adjusting detection thresholds or investigating patterns."
            )

        # False positive recommendations
        if data.false_positive_rate > 0.2:
            recommendations.append(
                "🔧 **PROCESS IMPROVEMENT**: High false positive rate (>20%). "
                "Recommend model retraining and threshold optimization."
            )
        elif data.false_positive_rate < 0.05:
            recommendations.append(
                "✅ **OPTIMIZATION**: Excellent false positive rate (<5%). "
                "Current configuration is performing well."
            )

        # Financial impact recommendations
        if data.financial_impact_usd and data.financial_impact_usd > 1000000:
            recommendations.append(
                "💰 **INVESTMENT JUSTIFICATION**: High financial impact detected. "
                "Consider increasing monitoring resources and automation."
            )

        # Compliance recommendations
        if data.regulatory_violations and data.regulatory_violations > 0:
            recommendations.append(
                "⚖️ **COMPLIANCE ALERT**: Regulatory violations detected. "
                "Immediate compliance team notification and remediation required."
            )

        # System performance recommendations
        if data.processing_time_hours > 24:
            recommendations.append(
                "⚡ **PERFORMANCE**: Processing time exceeds 24 hours. "
                "Consider infrastructure scaling or algorithm optimization."
            )

        # Default recommendations if none specific
        if not recommendations:
            recommendations.extend(
                [
                    "📊 **MONITORING**: Continue regular monitoring and review cycles.",
                    "🔄 **MAINTENANCE**: Schedule monthly model performance reviews.",
                    "📈 **OPTIMIZATION**: Consider A/B testing new detection algorithms.",
                ]
            )

        return recommendations

    def _generate_key_findings(self, data: ExecutiveSummaryData) -> list[str]:
        """Generate key findings summary."""
        findings = []

        anomaly_rate = (
            data.anomaly_count / data.total_records if data.total_records > 0 else 0
        )

        # Primary finding
        findings.append(
            f"🎯 **Primary Result**: Detected {data.anomaly_count:,} anomalies "
            f"from {data.total_records:,} records ({anomaly_rate:.1%} anomaly rate)"
        )

        # Performance finding
        findings.append(
            f"📊 **Performance**: {data.detection_rate:.1%} detection rate with "
            f"{data.false_positive_rate:.1%} false positive rate"
        )

        # Risk assessment
        risk_level = self._calculate_risk_level(data)
        risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "MINIMAL": "⚪"}
        findings.append(
            f"{risk_emoji.get(risk_level, '📊')} **Risk Assessment**: "
            f"Overall risk level classified as {risk_level}"
        )

        # Financial impact (if available)
        if data.financial_impact_usd:
            findings.append(
                f"💰 **Financial Impact**: Estimated ${data.financial_impact_usd:,.0f} "
                f"in potential losses prevented"
            )

        # Compliance status (if available)
        if data.compliance_score:
            findings.append(
                f"⚖️ **Compliance**: {data.compliance_score:.0%} compliance score achieved"
            )

        return findings

    def _generate_next_steps(self, data: ExecutiveSummaryData) -> list[str]:
        """Generate recommended next steps."""
        next_steps = []

        (
            data.anomaly_count / data.total_records if data.total_records > 0 else 0
        )
        risk_level = self._calculate_risk_level(data)

        # Immediate actions based on risk level
        if risk_level == "HIGH":
            next_steps.extend(
                [
                    "🚨 **IMMEDIATE**: Convene incident response team within 24 hours",
                    "🔍 **INVESTIGATE**: Conduct detailed analysis of high-risk anomalies",
                    "📞 **NOTIFY**: Alert senior management and compliance teams",
                ]
            )
        elif risk_level == "MEDIUM":
            next_steps.extend(
                [
                    "📅 **SCHEDULE**: Plan detailed review within 48 hours",
                    "🔧 **OPTIMIZE**: Review and adjust detection parameters",
                    "📊 **MONITOR**: Increase monitoring frequency temporarily",
                ]
            )
        else:
            next_steps.extend(
                [
                    "✅ **MAINTAIN**: Continue current monitoring procedures",
                    "📈 **OPTIMIZE**: Consider gradual performance improvements",
                    "🔄 **REVIEW**: Schedule next quarterly review",
                ]
            )

        # Standard operational steps
        next_steps.extend(
            [
                "📋 **DOCUMENT**: Record findings in compliance database",
                "🎯 **TRACK**: Monitor trend changes over next reporting period",
                "🤝 **COMMUNICATE**: Share results with relevant stakeholders",
            ]
        )

        return next_steps

    def _generate_fallback_report(
        self, data: ExecutiveSummaryData, output_path: str, format_type: str
    ) -> str:
        """Generate a basic report when templates are not available."""
        anomaly_rate = (
            data.anomaly_count / data.total_records if data.total_records > 0 else 0
        )
        risk_level = self._calculate_risk_level(data)

        report_content = f"""
# Anomaly Detection Executive Summary

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Report Period:** {data.report_period or "Current Period"}

## Executive Overview

### Key Metrics
- **Total Records Processed:** {data.total_records:,}
- **Anomalies Detected:** {data.anomaly_count:,} ({anomaly_rate:.1%})
- **Detection Rate:** {data.detection_rate:.1%}
- **False Positive Rate:** {data.false_positive_rate:.1%}
- **Risk Level:** {risk_level}

### Financial Impact
{f"- **Estimated Impact:** ${data.financial_impact_usd:,.0f}" if data.financial_impact_usd else "- Financial impact data not available"}

### System Performance
- **Processing Time:** {data.processing_time_hours:.1f} hours
{f"- **System Uptime:** {data.system_uptime:.1%}" if data.system_uptime else ""}

## Key Findings
"""

        for finding in self._generate_key_findings(data):
            report_content += f"\n- {finding}"

        report_content += "\n\n## Recommendations\n"
        for rec in self._generate_recommendations(data):
            report_content += f"\n- {rec}"

        report_content += "\n\n## Next Steps\n"
        for step in self._generate_next_steps(data):
            report_content += f"\n- {step}"

        # Save report
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_content)

        return str(output_file)


# Example usage and testing
if __name__ == "__main__":
    # Sample data for testing
    sample_data = ExecutiveSummaryData(
        total_records=100000,
        anomaly_count=2500,
        detection_rate=0.87,
        false_positive_rate=0.08,
        processing_time_hours=2.5,
        financial_impact_usd=750000,
        risk_score=0.65,
        trend_direction="increasing",
        regulatory_violations=0,
        compliance_score=0.95,
        system_uptime=0.998,
        alert_volume=145,
        report_period="Q2 2025",
        data_sources=["Transaction Database", "Customer Records", "External Feeds"],
        algorithms_used=["Isolation Forest", "Local Outlier Factor", "One-Class SVM"],
    )

    # Generate report
    generator = ExecutiveReportGenerator()
    report_path = generator.generate_summary(
        data=sample_data,
        output_path="executive_summary_sample.html",
        include_charts=True,
    )

    print(f"Executive summary generated: {report_path}")
