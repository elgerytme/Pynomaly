"""Business intelligence reporting service for executive and technical insights."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pynomaly.application.services.algorithm_benchmark import BenchmarkResult
from pynomaly.infrastructure.config.feature_flags import require_feature


@dataclass
class ExecutiveReport:
    """Executive summary report for business stakeholders."""

    # High-level metrics
    total_algorithms_tested: int = 0
    best_performing_algorithm: str = ""
    average_accuracy: float = 0.0
    performance_improvement: float = 0.0

    # Business impact
    estimated_time_savings: float = 0.0  # hours
    cost_reduction_percentage: float = 0.0
    automation_coverage: float = 0.0

    # Quality metrics
    false_positive_reduction: float = 0.0
    detection_reliability: float = 0.0
    risk_mitigation_score: float = 0.0

    # Strategic insights
    key_findings: list[str] = field(default_factory=list)
    business_recommendations: list[str] = field(default_factory=list)
    roi_projection: dict[str, float] = field(default_factory=dict)

    # Metadata
    report_period: str = ""
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "summary": {
                "total_algorithms_tested": self.total_algorithms_tested,
                "best_performing_algorithm": self.best_performing_algorithm,
                "average_accuracy": self.average_accuracy,
                "performance_improvement": self.performance_improvement,
            },
            "business_impact": {
                "estimated_time_savings_hours": self.estimated_time_savings,
                "cost_reduction_percentage": self.cost_reduction_percentage,
                "automation_coverage": self.automation_coverage,
            },
            "quality_metrics": {
                "false_positive_reduction": self.false_positive_reduction,
                "detection_reliability": self.detection_reliability,
                "risk_mitigation_score": self.risk_mitigation_score,
            },
            "insights": {
                "key_findings": self.key_findings,
                "business_recommendations": self.business_recommendations,
                "roi_projection": self.roi_projection,
            },
            "metadata": {
                "report_period": self.report_period,
                "generated_at": self.generated_at.isoformat(),
            },
        }


@dataclass
class TechnicalReport:
    """Detailed technical report for development and operations teams."""

    # Algorithm performance details
    algorithm_rankings: list[dict[str, Any]] = field(default_factory=list)
    performance_trends: dict[str, list[float]] = field(default_factory=dict)
    optimization_opportunities: list[str] = field(default_factory=list)

    # Infrastructure metrics
    resource_utilization: dict[str, float] = field(default_factory=dict)
    scaling_analysis: dict[str, Any] = field(default_factory=dict)
    bottleneck_identification: list[str] = field(default_factory=list)

    # Quality assurance
    test_coverage: float = 0.0
    code_quality_score: float = 0.0
    technical_debt_score: float = 0.0

    # Operational insights
    deployment_readiness: str = ""
    monitoring_recommendations: list[str] = field(default_factory=list)
    maintenance_schedule: dict[str, str] = field(default_factory=dict)

    # Metadata
    report_period: str = ""
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "algorithm_analysis": {
                "rankings": self.algorithm_rankings,
                "performance_trends": self.performance_trends,
                "optimization_opportunities": self.optimization_opportunities,
            },
            "infrastructure": {
                "resource_utilization": self.resource_utilization,
                "scaling_analysis": self.scaling_analysis,
                "bottleneck_identification": self.bottleneck_identification,
            },
            "quality_assurance": {
                "test_coverage": self.test_coverage,
                "code_quality_score": self.code_quality_score,
                "technical_debt_score": self.technical_debt_score,
            },
            "operations": {
                "deployment_readiness": self.deployment_readiness,
                "monitoring_recommendations": self.monitoring_recommendations,
                "maintenance_schedule": self.maintenance_schedule,
            },
            "metadata": {
                "report_period": self.report_period,
                "generated_at": self.generated_at.isoformat(),
            },
        }


class ReportingService:
    """Service for generating business intelligence reports."""

    @require_feature("business_intelligence")
    def __init__(self, output_directory: Path | None = None):
        """Initialize reporting service.

        Args:
            output_directory: Directory to save generated reports
        """
        self.output_directory = output_directory or Path.cwd() / "reports"
        self.output_directory.mkdir(exist_ok=True)

    def generate_executive_report(
        self,
        benchmark_results: list[BenchmarkResult],
        baseline_metrics: dict[str, float] | None = None,
        business_context: dict[str, Any] | None = None,
    ) -> ExecutiveReport:
        """Generate executive summary report.

        Args:
            benchmark_results: Algorithm benchmark results
            baseline_metrics: Previous performance baseline for comparison
            business_context: Additional business context and parameters

        Returns:
            Executive report with business insights
        """
        if not benchmark_results:
            return ExecutiveReport()

        report = ExecutiveReport()

        # Basic metrics
        report.total_algorithms_tested = len(benchmark_results)

        # Find best performing algorithm
        best_result = max(benchmark_results, key=lambda r: r.overall_score())
        report.best_performing_algorithm = best_result.algorithm_name

        # Calculate average accuracy
        accuracies = [r.accuracy for r in benchmark_results if r.accuracy > 0]
        report.average_accuracy = (
            sum(accuracies) / len(accuracies) if accuracies else 0.0
        )

        # Performance improvement calculation
        if baseline_metrics and "average_accuracy" in baseline_metrics:
            baseline_accuracy = baseline_metrics["average_accuracy"]
            improvement = (
                (report.average_accuracy - baseline_accuracy) / baseline_accuracy
            ) * 100
            report.performance_improvement = max(0, improvement)

        # Business impact calculations
        report.estimated_time_savings = self._calculate_time_savings(
            benchmark_results, business_context
        )
        report.cost_reduction_percentage = self._calculate_cost_reduction(
            benchmark_results, business_context
        )
        report.automation_coverage = self._calculate_automation_coverage(
            benchmark_results, business_context
        )

        # Quality metrics
        report.false_positive_reduction = self._calculate_fp_reduction(
            benchmark_results, baseline_metrics
        )
        report.detection_reliability = self._calculate_reliability(benchmark_results)
        report.risk_mitigation_score = self._calculate_risk_mitigation(
            benchmark_results
        )

        # Strategic insights
        report.key_findings = self._generate_key_findings(benchmark_results)
        report.business_recommendations = self._generate_business_recommendations(
            benchmark_results
        )
        report.roi_projection = self._calculate_roi_projection(
            benchmark_results, business_context
        )

        # Report metadata
        report.report_period = f"{datetime.now().strftime('%Y-%m-%d')}"

        return report

    def generate_technical_report(
        self,
        benchmark_results: list[BenchmarkResult],
        infrastructure_metrics: dict[str, Any] | None = None,
        quality_metrics: dict[str, float] | None = None,
    ) -> TechnicalReport:
        """Generate detailed technical report.

        Args:
            benchmark_results: Algorithm benchmark results
            infrastructure_metrics: Infrastructure performance data
            quality_metrics: Code quality and testing metrics

        Returns:
            Technical report with operational insights
        """
        if not benchmark_results:
            return TechnicalReport()

        report = TechnicalReport()

        # Algorithm performance details
        report.algorithm_rankings = self._create_algorithm_rankings(benchmark_results)
        report.performance_trends = self._analyze_performance_trends(benchmark_results)
        report.optimization_opportunities = self._identify_optimization_opportunities(
            benchmark_results
        )

        # Infrastructure metrics
        if infrastructure_metrics:
            report.resource_utilization = infrastructure_metrics.get(
                "resource_utilization", {}
            )
            report.scaling_analysis = infrastructure_metrics.get("scaling_analysis", {})
            report.bottleneck_identification = infrastructure_metrics.get(
                "bottlenecks", []
            )

        # Quality assurance
        if quality_metrics:
            report.test_coverage = quality_metrics.get("test_coverage", 0.0)
            report.code_quality_score = quality_metrics.get("code_quality", 0.0)
            report.technical_debt_score = quality_metrics.get("technical_debt", 0.0)

        # Operational insights
        report.deployment_readiness = self._assess_deployment_readiness(
            benchmark_results
        )
        report.monitoring_recommendations = self._generate_monitoring_recommendations(
            benchmark_results
        )
        report.maintenance_schedule = self._create_maintenance_schedule(
            benchmark_results
        )

        # Report metadata
        report.report_period = f"{datetime.now().strftime('%Y-%m-%d')}"

        return report

    def export_report(
        self,
        report: ExecutiveReport | TechnicalReport,
        format: str = "json",
        filename: str | None = None,
    ) -> Path:
        """Export report to file.

        Args:
            report: Report to export
            format: Export format ("json", "html", "pdf")
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Path to exported file
        """
        report_type = (
            "executive" if isinstance(report, ExecutiveReport) else "technical"
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if not filename:
            filename = f"{report_type}_report_{timestamp}.{format}"

        output_path = self.output_directory / filename

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(report.to_dict(), f, indent=2, default=str)

        elif format == "html":
            html_content = self._generate_html_report(report)
            with open(output_path, "w") as f:
                f.write(html_content)

        elif format == "pdf":
            # Would require additional dependencies (reportlab, weasyprint, etc.)
            # For now, generate HTML and note PDF requirement
            html_path = self.export_report(
                report, "html", filename.replace(".pdf", ".html")
            )
            return html_path

        return output_path

    def _calculate_time_savings(
        self,
        benchmark_results: list[BenchmarkResult],
        business_context: dict[str, Any] | None,
    ) -> float:
        """Calculate estimated time savings from automation."""
        if not business_context:
            return 0.0

        # Get processing volumes and manual effort estimates
        daily_samples = business_context.get("daily_samples", 1000)
        manual_processing_time = business_context.get(
            "manual_processing_minutes_per_sample", 0.5
        )

        # Calculate average processing time from benchmarks
        avg_processing_time = sum(r.total_time for r in benchmark_results) / len(
            benchmark_results
        )
        avg_samples_per_second = (
            1.0 / avg_processing_time if avg_processing_time > 0 else 0
        )

        # Time savings calculation
        manual_time_per_day = (daily_samples * manual_processing_time) / 60  # hours
        automated_time_per_day = (
            daily_samples / (avg_samples_per_second * 3600)
            if avg_samples_per_second > 0
            else 0
        )

        return max(0, manual_time_per_day - automated_time_per_day)

    def _calculate_cost_reduction(
        self,
        benchmark_results: list[BenchmarkResult],
        business_context: dict[str, Any] | None,
    ) -> float:
        """Calculate cost reduction percentage."""
        if not business_context:
            return 0.0

        # Use time savings to estimate cost reduction
        time_savings = self._calculate_time_savings(benchmark_results, business_context)
        hourly_rate = business_context.get("analyst_hourly_rate", 50.0)
        current_daily_cost = business_context.get(
            "current_daily_analysis_cost", time_savings * hourly_rate
        )

        if current_daily_cost > 0:
            cost_savings = time_savings * hourly_rate
            return (cost_savings / current_daily_cost) * 100

        return 0.0

    def _calculate_automation_coverage(
        self,
        benchmark_results: list[BenchmarkResult],
        business_context: dict[str, Any] | None,
    ) -> float:
        """Calculate percentage of processes that can be automated."""
        if not business_context:
            return 85.0  # Default estimate

        # Calculate based on algorithm reliability and business requirements
        avg_accuracy = sum(r.accuracy for r in benchmark_results) / len(
            benchmark_results
        )
        min_accuracy_threshold = business_context.get("min_automation_accuracy", 0.8)

        if avg_accuracy >= min_accuracy_threshold:
            return min(95.0, avg_accuracy * 100)
        else:
            # Partial automation for high-confidence cases
            return avg_accuracy * 70

    def _calculate_fp_reduction(
        self,
        benchmark_results: list[BenchmarkResult],
        baseline_metrics: dict[str, float] | None,
    ) -> float:
        """Calculate false positive reduction."""
        if not baseline_metrics or "false_positive_rate" not in baseline_metrics:
            return 0.0

        # Calculate current average false positive rate
        current_fp_rate = sum(
            1 - r.precision for r in benchmark_results if r.precision > 0
        ) / len(benchmark_results)

        baseline_fp_rate = baseline_metrics["false_positive_rate"]

        if baseline_fp_rate > 0:
            reduction = ((baseline_fp_rate - current_fp_rate) / baseline_fp_rate) * 100
            return max(0, reduction)

        return 0.0

    def _calculate_reliability(self, benchmark_results: list[BenchmarkResult]) -> float:
        """Calculate overall detection reliability score."""
        if not benchmark_results:
            return 0.0

        # Weighted combination of precision, recall, and consistency
        reliabilities = []
        for result in benchmark_results:
            precision_score = result.precision
            recall_score = result.recall
            f1_score = result.f1_score

            # Reliability is high when all metrics are balanced and high
            reliability = precision_score * 0.4 + recall_score * 0.4 + f1_score * 0.2
            reliabilities.append(reliability)

        return (sum(reliabilities) / len(reliabilities)) * 100

    def _calculate_risk_mitigation(
        self, benchmark_results: list[BenchmarkResult]
    ) -> float:
        """Calculate risk mitigation score."""
        if not benchmark_results:
            return 0.0

        # Risk mitigation is higher with better recall (fewer missed anomalies)
        avg_recall = sum(r.recall for r in benchmark_results) / len(benchmark_results)

        # Also consider consistency across algorithms
        recall_variance = sum(
            (r.recall - avg_recall) ** 2 for r in benchmark_results
        ) / len(benchmark_results)
        consistency_score = 1.0 / (1.0 + recall_variance)

        return (avg_recall * 0.7 + consistency_score * 0.3) * 100

    def _generate_key_findings(
        self, benchmark_results: list[BenchmarkResult]
    ) -> list[str]:
        """Generate key business findings."""
        findings = []

        if not benchmark_results:
            return ["No benchmark data available for analysis."]

        # Best performing algorithm
        best_result = max(benchmark_results, key=lambda r: r.overall_score())
        findings.append(
            f"{best_result.algorithm_name} achieved the highest overall performance "
            f"with {best_result.overall_score():.1f}% effectiveness"
        )

        # Speed insights
        fastest_result = min(benchmark_results, key=lambda r: r.total_time)
        findings.append(
            f"{fastest_result.algorithm_name} provides fastest processing at "
            f"{fastest_result.total_time:.3f} seconds per analysis"
        )

        # Accuracy insights
        most_accurate = max(benchmark_results, key=lambda r: r.accuracy)
        findings.append(
            f"{most_accurate.algorithm_name} delivers highest accuracy at "
            f"{most_accurate.accuracy:.1%} detection rate"
        )

        # Performance spread
        accuracies = [r.accuracy for r in benchmark_results]
        accuracy_range = max(accuracies) - min(accuracies)
        if accuracy_range < 0.1:
            findings.append(
                "Algorithm performance is consistently high across all tested methods"
            )
        else:
            findings.append(
                f"Algorithm performance varies significantly ({accuracy_range:.1%} range)"
            )

        return findings

    def _generate_business_recommendations(
        self, benchmark_results: list[BenchmarkResult]
    ) -> list[str]:
        """Generate business recommendations."""
        recommendations = []

        if not benchmark_results:
            return ["Conduct algorithm benchmarking before deployment."]

        # Performance-based recommendations
        best_result = max(benchmark_results, key=lambda r: r.overall_score())
        recommendations.append(
            f"Deploy {best_result.algorithm_name} for production use based on superior overall performance"
        )

        # Speed vs accuracy trade-offs
        fastest = min(benchmark_results, key=lambda r: r.total_time)
        most_accurate = max(benchmark_results, key=lambda r: r.accuracy)

        if fastest.algorithm_name != most_accurate.algorithm_name:
            recommendations.append(
                f"Consider {fastest.algorithm_name} for high-volume scenarios requiring speed, "
                f"{most_accurate.algorithm_name} for critical accuracy requirements"
            )

        # Resource optimization
        avg_time = sum(r.total_time for r in benchmark_results) / len(benchmark_results)
        fast_algorithms = [
            r for r in benchmark_results if r.total_time < avg_time * 0.8
        ]

        if fast_algorithms:
            recommendations.append(
                "Implement auto-scaling based on processing volume to optimize resource utilization"
            )

        # Quality assurance
        high_accuracy_algorithms = [r for r in benchmark_results if r.accuracy > 0.9]
        if len(high_accuracy_algorithms) > 1:
            recommendations.append(
                "Deploy ensemble of top-performing algorithms for maximum reliability"
            )

        return recommendations

    def _calculate_roi_projection(
        self,
        benchmark_results: list[BenchmarkResult],
        business_context: dict[str, Any] | None,
    ) -> dict[str, float]:
        """Calculate ROI projections."""
        if not business_context:
            return {}

        # Time and cost savings
        time_savings = self._calculate_time_savings(benchmark_results, business_context)
        self._calculate_cost_reduction(benchmark_results, business_context)

        # Implementation costs (estimated)
        implementation_cost = business_context.get("implementation_cost", 50000)
        annual_maintenance = business_context.get("annual_maintenance_cost", 10000)

        # Annual savings
        daily_savings = time_savings * business_context.get("analyst_hourly_rate", 50.0)
        annual_savings = daily_savings * 250  # Working days

        # ROI calculations
        first_year_roi = (
            (annual_savings - implementation_cost - annual_maintenance)
            / implementation_cost
        ) * 100
        three_year_roi = (
            (annual_savings * 3 - implementation_cost - annual_maintenance * 3)
            / implementation_cost
        ) * 100

        return {
            "first_year_roi_percent": max(0, first_year_roi),
            "three_year_roi_percent": max(0, three_year_roi),
            "annual_savings_usd": annual_savings,
            "payback_period_months": (
                max(1, implementation_cost / (annual_savings / 12))
                if annual_savings > 0
                else 0
            ),
        }

    def _create_algorithm_rankings(
        self, benchmark_results: list[BenchmarkResult]
    ) -> list[dict[str, Any]]:
        """Create detailed algorithm rankings."""
        rankings = []

        # Sort by overall score
        sorted_results = sorted(
            benchmark_results, key=lambda r: r.overall_score(), reverse=True
        )

        for i, result in enumerate(sorted_results):
            rankings.append(
                {
                    "rank": i + 1,
                    "algorithm": result.algorithm_name,
                    "overall_score": result.overall_score(),
                    "accuracy": result.accuracy,
                    "precision": result.precision,
                    "recall": result.recall,
                    "f1_score": result.f1_score,
                    "processing_time": result.total_time,
                    "efficiency_score": result.efficiency_score(),
                    "recommendation": (
                        "Primary choice"
                        if i == 0
                        else "Alternative option" if i < 3 else "Specialized use case"
                    ),
                }
            )

        return rankings

    def _analyze_performance_trends(
        self, benchmark_results: list[BenchmarkResult]
    ) -> dict[str, list[float]]:
        """Analyze performance trends across algorithms."""
        trends = {
            "accuracy_trend": [r.accuracy for r in benchmark_results],
            "speed_trend": [
                1.0 / r.total_time if r.total_time > 0 else 0 for r in benchmark_results
            ],
            "efficiency_trend": [r.efficiency_score() for r in benchmark_results],
            "memory_trend": [r.memory_usage for r in benchmark_results],
        }

        return trends

    def _identify_optimization_opportunities(
        self, benchmark_results: list[BenchmarkResult]
    ) -> list[str]:
        """Identify optimization opportunities."""
        opportunities = []

        # Performance analysis
        avg_time = sum(r.total_time for r in benchmark_results) / len(benchmark_results)
        slow_algorithms = [
            r for r in benchmark_results if r.total_time > avg_time * 1.5
        ]

        if slow_algorithms:
            opportunities.append(
                f"Optimize {len(slow_algorithms)} algorithms with above-average processing time"
            )

        # Memory analysis
        avg_memory = sum(r.memory_usage for r in benchmark_results) / len(
            benchmark_results
        )
        memory_heavy = [
            r for r in benchmark_results if r.memory_usage > avg_memory * 1.5
        ]

        if memory_heavy:
            opportunities.append(
                f"Reduce memory usage in {len(memory_heavy)} algorithms for better scalability"
            )

        # Accuracy analysis
        high_accuracy_threshold = 0.9
        low_accuracy = [
            r for r in benchmark_results if r.accuracy < high_accuracy_threshold
        ]

        if low_accuracy:
            opportunities.append(
                f"Improve accuracy in {len(low_accuracy)} algorithms through hyperparameter tuning"
            )

        return opportunities

    def _assess_deployment_readiness(
        self, benchmark_results: list[BenchmarkResult]
    ) -> str:
        """Assess deployment readiness."""
        if not benchmark_results:
            return "Not ready - no benchmark data"

        # Check minimum performance thresholds
        avg_accuracy = sum(r.accuracy for r in benchmark_results) / len(
            benchmark_results
        )
        avg_time = sum(r.total_time for r in benchmark_results) / len(benchmark_results)

        if avg_accuracy >= 0.85 and avg_time <= 5.0:  # 5 seconds max processing time
            return "Ready for production deployment"
        elif avg_accuracy >= 0.75:
            return "Ready for staging environment testing"
        else:
            return "Requires optimization before deployment"

    def _generate_monitoring_recommendations(
        self, benchmark_results: list[BenchmarkResult]
    ) -> list[str]:
        """Generate monitoring recommendations."""
        recommendations = [
            "Monitor algorithm accuracy on a daily basis",
            "Set up alerts for processing time exceeding baseline by 50%",
            "Track memory usage trends to prevent resource exhaustion",
            "Implement automated performance regression testing",
            "Monitor false positive rates in production data",
        ]

        # Add specific recommendations based on results
        avg_time = sum(r.total_time for r in benchmark_results) / len(benchmark_results)
        if avg_time > 1.0:
            recommendations.append(
                "Consider implementing caching for frequently processed data patterns"
            )

        return recommendations

    def _create_maintenance_schedule(
        self, benchmark_results: list[BenchmarkResult]
    ) -> dict[str, str]:
        """Create maintenance schedule recommendations."""
        return {
            "daily": "Monitor performance metrics and error rates",
            "weekly": "Review algorithm performance trends and adjust thresholds",
            "monthly": "Conduct comprehensive benchmark comparison",
            "quarterly": "Evaluate new algorithm implementations and optimizations",
            "annually": "Full performance audit and algorithm lifecycle review",
        }

    def _generate_html_report(self, report: ExecutiveReport | TechnicalReport) -> str:
        """Generate HTML version of the report."""
        report_data = report.to_dict()
        report_type = (
            "Executive" if isinstance(report, ExecutiveReport) else "Technical"
        )

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_type} Report - Pynomaly</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #2563eb; color: white; padding: 20px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #2563eb; }}
                .metric {{ margin: 10px 0; }}
                .recommendation {{ background-color: #f0f9ff; padding: 10px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report_type} Report</h1>
                <p>Generated: {report.generated_at.strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
        """

        # Add report-specific content
        if isinstance(report, ExecutiveReport):
            html += self._generate_executive_html_content(report_data)
        else:
            html += self._generate_technical_html_content(report_data)

        html += """
        </body>
        </html>
        """

        return html

    def _generate_executive_html_content(self, report_data: dict[str, Any]) -> str:
        """Generate HTML content for executive report."""
        summary = report_data["summary"]
        impact = report_data["business_impact"]
        insights = report_data["insights"]

        return f"""
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metric">Algorithms Tested: {summary["total_algorithms_tested"]}</div>
                <div class="metric">Best Performer: {summary["best_performing_algorithm"]}</div>
                <div class="metric">Average Accuracy: {summary["average_accuracy"]:.1%}</div>
            </div>

            <div class="section">
                <h2>Business Impact</h2>
                <div class="metric">Estimated Time Savings: {impact["estimated_time_savings_hours"]:.1f} hours/day</div>
                <div class="metric">Cost Reduction: {impact["cost_reduction_percentage"]:.1f}%</div>
                <div class="metric">Automation Coverage: {impact["automation_coverage"]:.1f}%</div>
            </div>

            <div class="section">
                <h2>Key Findings</h2>
                {"".join(f'<div class="recommendation">• {finding}</div>' for finding in insights["key_findings"])}
            </div>

            <div class="section">
                <h2>Recommendations</h2>
                {"".join(f'<div class="recommendation">• {rec}</div>' for rec in insights["business_recommendations"])}
            </div>
        """

    def _generate_technical_html_content(self, report_data: dict[str, Any]) -> str:
        """Generate HTML content for technical report."""
        algorithm = report_data["algorithm_analysis"]
        infrastructure = report_data["infrastructure"]
        quality = report_data["quality_assurance"]

        return f"""
            <div class="section">
                <h2>Algorithm Performance</h2>
                <div class="metric">Optimization Opportunities: {len(algorithm["optimization_opportunities"])}</div>
                {"".join(f'<div class="recommendation">• {opp}</div>' for opp in algorithm["optimization_opportunities"])}
            </div>

            <div class="section">
                <h2>Infrastructure Analysis</h2>
                <div class="metric">Bottlenecks Identified: {len(infrastructure["bottleneck_identification"])}</div>
                {"".join(f'<div class="recommendation">• {bottleneck}</div>' for bottleneck in infrastructure["bottleneck_identification"])}
            </div>

            <div class="section">
                <h2>Quality Metrics</h2>
                <div class="metric">Test Coverage: {quality["test_coverage"]:.1f}%</div>
                <div class="metric">Code Quality Score: {quality["code_quality_score"]:.1f}/100</div>
                <div class="metric">Technical Debt Score: {quality["technical_debt_score"]:.1f}/100</div>
            </div>
        """
