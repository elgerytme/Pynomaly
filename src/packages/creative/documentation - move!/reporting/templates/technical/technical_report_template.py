#!/usr/bin/env python3
"""
Technical Report Template for Pynomaly Anomaly Detection

This template generates detailed technical reports with algorithm performance metrics,
statistical analysis, and model validation results for technical stakeholders.
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

import jinja2


@dataclass
class AlgorithmMetrics:
    """Performance metrics for a single algorithm."""

    name: str
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    auc_pr: float
    accuracy: float | None = None
    specificity: float | None = None
    training_time_seconds: float | None = None
    prediction_time_seconds: float | None = None
    memory_usage_mb: float | None = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    confusion_matrix: list[list[int]] | None = None
    feature_importance: dict[str, float] | None = None


@dataclass
class TechnicalReportData:
    """Data structure for technical report."""

    # Dataset information
    dataset_name: str
    dataset_shape: tuple[int, int]
    dataset_features: list[str]
    target_distribution: dict[str, int]

    # Algorithm results
    algorithms: list[AlgorithmMetrics]

    # Statistical analysis
    statistical_significance: dict[str, Any] | None = None
    cross_validation_results: dict[str, list[float]] | None = None

    # Data quality metrics
    missing_values_pct: float | None = None
    duplicate_rows_pct: float | None = None
    correlation_analysis: dict[str, Any] | None = None

    # Experiment metadata
    experiment_id: str | None = None
    preprocessing_steps: list[str] | None = None
    validation_method: str | None = None
    random_seed: int | None = None

    # Performance analysis
    learning_curves: dict[str, Any] | None = None
    threshold_analysis: dict[str, Any] | None = None


class TechnicalReportGenerator:
    """Generate technical reports for anomaly detection experiments."""

    def __init__(self, template_dir: str | None = None):
        """Initialize the technical report generator.

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

    def generate_report(
        self,
        data: TechnicalReportData,
        output_path: str,
        format_type: str = "html",
        include_charts: bool = True,
        include_statistical_tests: bool = True,
        template_name: str | None = None,
    ) -> str:
        """Generate comprehensive technical report.

        Args:
            data: Technical report data
            output_path: Output file path
            format_type: Output format ("html", "pdf", "markdown")
            include_charts: Whether to include visualization charts
            include_statistical_tests: Whether to include statistical analysis
            template_name: Custom template name (optional)

        Returns:
            Path to generated report
        """
        # Generate statistical analysis
        if include_statistical_tests and len(data.algorithms) > 1:
            statistical_analysis = self._perform_statistical_analysis(data)
        else:
            statistical_analysis = None

        # Generate performance comparison
        performance_comparison = self._generate_performance_comparison(data)

        # Generate charts if requested
        chart_paths = []
        if include_charts:
            chart_paths = self._generate_technical_charts(data, output_path)

        # Best algorithm analysis
        best_algorithm = self._identify_best_algorithm(data)

        # Prepare template context
        context = {
            "data": data,
            "performance_comparison": performance_comparison,
            "statistical_analysis": statistical_analysis,
            "best_algorithm": best_algorithm,
            "chart_paths": chart_paths,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "technical_recommendations": self._generate_technical_recommendations(data),
            "methodology_notes": self._generate_methodology_notes(data),
            "validation_summary": self._generate_validation_summary(data),
        }

        # Select template
        template_file = template_name or f"technical_report.{format_type}.j2"

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
            return self._generate_fallback_report(
                data, output_path, format_type, context
            )

    def _perform_statistical_analysis(
        self, data: TechnicalReportData
    ) -> dict[str, Any]:
        """Perform statistical significance testing between algorithms."""
        if len(data.algorithms) < 2:
            return {}

        analysis = {}

        # Extract F1 scores for comparison
        {alg.name: alg.f1_score for alg in data.algorithms}

        # If cross-validation results are available, use them
        if data.cross_validation_results:
            cv_results = data.cross_validation_results

            # Friedman test for multiple algorithms
            if len(cv_results) > 2:
                algorithm_names = list(cv_results.keys())
                scores_matrix = [cv_results[name] for name in algorithm_names]

                try:
                    statistic, p_value = stats.friedmanchisquare(*scores_matrix)
                    analysis["friedman_test"] = {
                        "statistic": statistic,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                        "interpretation": self._interpret_friedman_test(p_value),
                    }
                except Exception as e:
                    analysis["friedman_test"] = {"error": str(e)}

            # Pairwise Wilcoxon signed-rank tests
            pairwise_tests = {}
            algorithm_names = list(cv_results.keys())

            for i, alg1 in enumerate(algorithm_names):
                for alg2 in algorithm_names[i + 1 :]:
                    try:
                        statistic, p_value = stats.wilcoxon(
                            cv_results[alg1], cv_results[alg2], alternative="two-sided"
                        )
                        pairwise_tests[f"{alg1}_vs_{alg2}"] = {
                            "statistic": statistic,
                            "p_value": p_value,
                            "significant": p_value < 0.05,
                            "better_algorithm": (
                                alg1
                                if np.mean(cv_results[alg1]) > np.mean(cv_results[alg2])
                                else alg2
                            ),
                        }
                    except Exception as e:
                        pairwise_tests[f"{alg1}_vs_{alg2}"] = {"error": str(e)}

            analysis["pairwise_tests"] = pairwise_tests

        # Effect size calculations
        if len(data.algorithms) == 2:
            alg1, alg2 = data.algorithms[:2]
            cohen_d = self._calculate_cohens_d(alg1.f1_score, alg2.f1_score)
            analysis["effect_size"] = {
                "cohens_d": cohen_d,
                "interpretation": self._interpret_cohens_d(cohen_d),
            }

        return analysis

    def _generate_performance_comparison(
        self, data: TechnicalReportData
    ) -> dict[str, Any]:
        """Generate comprehensive performance comparison."""
        metrics = ["precision", "recall", "f1_score", "auc_roc", "auc_pr"]

        comparison = {"summary_table": [], "rankings": {}, "performance_gaps": {}}

        # Create summary table
        for algorithm in data.algorithms:
            row = {
                "algorithm": algorithm.name,
                "precision": algorithm.precision,
                "recall": algorithm.recall,
                "f1_score": algorithm.f1_score,
                "auc_roc": algorithm.auc_roc,
                "auc_pr": algorithm.auc_pr,
                "training_time": algorithm.training_time_seconds,
                "prediction_time": algorithm.prediction_time_seconds,
                "memory_usage": algorithm.memory_usage_mb,
            }
            comparison["summary_table"].append(row)

        # Calculate rankings for each metric
        for metric in metrics:
            metric_values = [
                (alg.name, getattr(alg, metric)) for alg in data.algorithms
            ]
            metric_values.sort(key=lambda x: x[1], reverse=True)
            comparison["rankings"][metric] = [alg_name for alg_name, _ in metric_values]

        # Calculate performance gaps
        for metric in metrics:
            values = [getattr(alg, metric) for alg in data.algorithms]
            comparison["performance_gaps"][metric] = {
                "max": max(values),
                "min": min(values),
                "gap": max(values) - min(values),
                "std": np.std(values),
                "coefficient_of_variation": (
                    np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                ),
            }

        return comparison

    def _identify_best_algorithm(self, data: TechnicalReportData) -> dict[str, Any]:
        """Identify the best performing algorithm based on multiple criteria."""
        if not data.algorithms:
            return {}

        # Define weights for different metrics (customizable)
        weights = {
            "f1_score": 0.3,
            "auc_roc": 0.25,
            "precision": 0.2,
            "recall": 0.15,
            "auc_pr": 0.1,
        }

        algorithm_scores = {}

        for algorithm in data.algorithms:
            weighted_score = (
                weights["f1_score"] * algorithm.f1_score
                + weights["auc_roc"] * algorithm.auc_roc
                + weights["precision"] * algorithm.precision
                + weights["recall"] * algorithm.recall
                + weights["auc_pr"] * algorithm.auc_pr
            )
            algorithm_scores[algorithm.name] = weighted_score

        best_algorithm_name = max(algorithm_scores, key=algorithm_scores.get)
        best_algorithm = next(
            alg for alg in data.algorithms if alg.name == best_algorithm_name
        )

        return {
            "name": best_algorithm_name,
            "weighted_score": algorithm_scores[best_algorithm_name],
            "algorithm": best_algorithm,
            "scoring_method": "Weighted combination of F1, AUC-ROC, Precision, Recall, AUC-PR",
            "weights_used": weights,
            "all_scores": algorithm_scores,
        }

    def _generate_technical_charts(
        self, data: TechnicalReportData, output_path: str
    ) -> list[str]:
        """Generate technical visualization charts."""
        charts = []
        output_dir = Path(output_path).parent / "technical_charts"
        output_dir.mkdir(exist_ok=True)

        # Set style
        plt.style.use("default")
        sns.set_palette("husl")

        # Chart 1: Performance Comparison Radar Chart
        metrics = ["precision", "recall", "f1_score", "auc_roc", "auc_pr"]

        if len(data.algorithms) <= 5:  # Radar chart works best with few algorithms
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": "polar"})

            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle

            colors = plt.cm.Set1(np.linspace(0, 1, len(data.algorithms)))

            for i, algorithm in enumerate(data.algorithms):
                values = [getattr(algorithm, metric) for metric in metrics]
                values += values[:1]  # Complete the circle

                ax.plot(
                    angles,
                    values,
                    "o-",
                    linewidth=2,
                    label=algorithm.name,
                    color=colors[i],
                )
                ax.fill(angles, values, alpha=0.25, color=colors[i])

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([metric.replace("_", " ").title() for metric in metrics])
            ax.set_ylim(0, 1)
            ax.set_title(
                "Algorithm Performance Comparison", size=16, fontweight="bold", pad=20
            )
            ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))
            ax.grid(True)

            chart_path = output_dir / "performance_radar.png"
            plt.savefig(chart_path, dpi=300, bbox_inches="tight")
            plt.close()
            charts.append(str(chart_path))

        # Chart 2: Performance Metrics Bar Chart
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(
            "Detailed Performance Metrics Comparison", fontsize=16, fontweight="bold"
        )

        metrics_with_names = [
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1_score", "F1 Score"),
            ("auc_roc", "AUC-ROC"),
            ("auc_pr", "AUC-PR"),
            ("accuracy", "Accuracy"),
        ]

        algorithm_names = [alg.name for alg in data.algorithms]
        colors = plt.cm.Set2(np.linspace(0, 1, len(data.algorithms)))

        for idx, (metric, title) in enumerate(metrics_with_names):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]

            if metric == "accuracy" and any(
                getattr(alg, metric) is None for alg in data.algorithms
            ):
                # Skip accuracy if not available
                ax.text(
                    0.5,
                    0.5,
                    "Accuracy\nNot Available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                    style="italic",
                )
                ax.set_title(title)
                continue

            values = [getattr(alg, metric) or 0 for alg in data.algorithms]
            bars = ax.bar(algorithm_names, values, color=colors, alpha=0.8)

            # Add value labels on bars
            for bar, value in zip(bars, values, strict=False):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            ax.set_title(title, fontweight="bold")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1.1)
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        chart_path = output_dir / "performance_metrics.png"
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()
        charts.append(str(chart_path))

        # Chart 3: Confusion Matrices (if available)
        algorithms_with_cm = [
            alg for alg in data.algorithms if alg.confusion_matrix is not None
        ]
        if algorithms_with_cm:
            n_algorithms = len(algorithms_with_cm)
            cols = min(3, n_algorithms)
            rows = (n_algorithms + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            fig.suptitle("Confusion Matrices", fontsize=16, fontweight="bold")

            if n_algorithms == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes if isinstance(axes, list | np.ndarray) else [axes]
            else:
                axes = axes.flatten()

            for idx, algorithm in enumerate(algorithms_with_cm):
                ax = axes[idx]
                cm = np.array(algorithm.confusion_matrix)

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    ax=ax,
                    xticklabels=["Normal", "Anomaly"],
                    yticklabels=["Normal", "Anomaly"],
                )
                ax.set_title(f"{algorithm.name}", fontweight="bold")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")

            # Hide unused subplots
            for idx in range(n_algorithms, len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()
            chart_path = output_dir / "confusion_matrices.png"
            plt.savefig(chart_path, dpi=300, bbox_inches="tight")
            plt.close()
            charts.append(str(chart_path))

        # Chart 4: Training Time vs Performance Trade-off
        if any(alg.training_time_seconds is not None for alg in data.algorithms):
            fig, ax = plt.subplots(figsize=(10, 6))

            training_times = [alg.training_time_seconds or 0 for alg in data.algorithms]
            f1_scores = [alg.f1_score for alg in data.algorithms]

            ax.scatter(training_times, f1_scores, s=100, alpha=0.7, c=colors)

            # Add algorithm labels
            for i, alg in enumerate(data.algorithms):
                ax.annotate(
                    alg.name,
                    (training_times[i], f1_scores[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=10,
                )

            ax.set_xlabel("Training Time (seconds)", fontweight="bold")
            ax.set_ylabel("F1 Score", fontweight="bold")
            ax.set_title(
                "Training Time vs Performance Trade-off", fontsize=14, fontweight="bold"
            )
            ax.grid(True, alpha=0.3)

            chart_path = output_dir / "time_performance_tradeoff.png"
            plt.savefig(chart_path, dpi=300, bbox_inches="tight")
            plt.close()
            charts.append(str(chart_path))

        return charts

    def _generate_technical_recommendations(
        self, data: TechnicalReportData
    ) -> list[str]:
        """Generate technical recommendations based on analysis."""
        recommendations = []

        best_alg = self._identify_best_algorithm(data)

        if best_alg:
            recommendations.append(
                f"🥇 **Best Overall Performance**: {best_alg['name']} achieves the highest "
                f"weighted performance score ({best_alg['weighted_score']:.3f})"
            )

        # Performance analysis
        performance_comp = self._generate_performance_comparison(data)

        # Check for significant performance gaps
        high_variance_metrics = []
        for metric, gap_info in performance_comp["performance_gaps"].items():
            if gap_info["coefficient_of_variation"] > 0.1:  # 10% CV threshold
                high_variance_metrics.append(metric)

        if high_variance_metrics:
            recommendations.append(
                f"📊 **Performance Variance**: High variance detected in {', '.join(high_variance_metrics)}. "
                f"Consider algorithm selection based on specific metric priorities."
            )

        # Memory and time recommendations
        memory_intensive = [
            alg
            for alg in data.algorithms
            if alg.memory_usage_mb and alg.memory_usage_mb > 1000
        ]
        if memory_intensive:
            alg_names = [alg.name for alg in memory_intensive]
            recommendations.append(
                f"💾 **Memory Optimization**: {', '.join(alg_names)} require high memory (>1GB). "
                f"Consider memory-efficient alternatives for large-scale deployment."
            )

        slow_algorithms = [
            alg
            for alg in data.algorithms
            if alg.training_time_seconds and alg.training_time_seconds > 300
        ]
        if slow_algorithms:
            alg_names = [alg.name for alg in slow_algorithms]
            recommendations.append(
                f"⏱️ **Training Time**: {', '.join(alg_names)} require extended training time (>5min). "
                f"Consider parallel processing or algorithm optimization for production use."
            )

        # Feature importance recommendations
        algs_with_features = [alg for alg in data.algorithms if alg.feature_importance]
        if algs_with_features:
            recommendations.append(
                f"🔍 **Feature Analysis**: Feature importance analysis available for "
                f"{len(algs_with_features)} algorithms. Review for feature selection insights."
            )

        # Cross-validation recommendations
        if data.cross_validation_results:
            cv_std = {
                name: np.std(scores)
                for name, scores in data.cross_validation_results.items()
            }
            unstable_algorithms = [name for name, std in cv_std.items() if std > 0.05]

            if unstable_algorithms:
                recommendations.append(
                    f"⚠️ **Model Stability**: {', '.join(unstable_algorithms)} show high variance "
                    f"across cross-validation folds. Consider ensemble methods or hyperparameter tuning."
                )

        # Data quality recommendations
        if data.missing_values_pct and data.missing_values_pct > 5:
            recommendations.append(
                f"🧹 **Data Quality**: {data.missing_values_pct:.1f}% missing values detected. "
                f"Consider advanced imputation strategies for improved performance."
            )

        if data.duplicate_rows_pct and data.duplicate_rows_pct > 1:
            recommendations.append(
                f"🔄 **Data Cleaning**: {data.duplicate_rows_pct:.1f}% duplicate rows found. "
                f"Remove duplicates to prevent overfitting and improve generalization."
            )

        return recommendations

    def _generate_methodology_notes(self, data: TechnicalReportData) -> list[str]:
        """Generate methodology and experimental setup notes."""
        notes = []

        # Dataset information
        notes.append(
            f"📊 **Dataset**: {data.dataset_name} with {data.dataset_shape[0]:,} samples "
            f"and {data.dataset_shape[1]} features"
        )

        if data.target_distribution:
            total_samples = sum(data.target_distribution.values())
            normal_pct = (
                data.target_distribution.get("normal", 0) / total_samples
            ) * 100
            anomaly_pct = (
                data.target_distribution.get("anomaly", 0) / total_samples
            ) * 100
            notes.append(
                f"🎯 **Class Distribution**: {normal_pct:.1f}% normal, {anomaly_pct:.1f}% anomalies"
            )

        # Preprocessing steps
        if data.preprocessing_steps:
            notes.append(f"🔧 **Preprocessing**: {', '.join(data.preprocessing_steps)}")

        # Validation method
        if data.validation_method:
            notes.append(f"✅ **Validation**: {data.validation_method}")

        # Random seed
        if data.random_seed:
            notes.append(
                f"🎲 **Reproducibility**: Random seed {data.random_seed} used for reproducible results"
            )

        # Algorithm count and types
        algorithm_types = list({alg.name.split("(")[0] for alg in data.algorithms})
        notes.append(
            f"🤖 **Algorithms Tested**: {len(data.algorithms)} configurations across "
            f"{len(algorithm_types)} algorithm types"
        )

        return notes

    def _generate_validation_summary(self, data: TechnicalReportData) -> dict[str, Any]:
        """Generate validation methodology summary."""
        summary = {}

        if data.cross_validation_results:
            cv_results = data.cross_validation_results
            summary["cross_validation"] = {
                "method": "K-Fold Cross Validation",
                "folds": len(list(cv_results.values())[0]) if cv_results else 0,
                "algorithms_tested": len(cv_results),
                "mean_scores": {
                    name: np.mean(scores) for name, scores in cv_results.items()
                },
                "std_scores": {
                    name: np.std(scores) for name, scores in cv_results.items()
                },
            }

        if data.statistical_significance:
            summary["statistical_testing"] = {
                "tests_performed": list(data.statistical_significance.keys()),
                "significance_level": 0.05,
            }

        return summary

    def _calculate_cohens_d(
        self, mean1: float, mean2: float, std1: float = 0.1, std2: float = 0.1
    ) -> float:
        """Calculate Cohen's d effect size."""
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        return abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        if d < 0.2:
            return "Small effect"
        elif d < 0.5:
            return "Medium effect"
        elif d < 0.8:
            return "Large effect"
        else:
            return "Very large effect"

    def _interpret_friedman_test(self, p_value: float) -> str:
        """Interpret Friedman test results."""
        if p_value < 0.001:
            return "Highly significant differences between algorithms (p < 0.001)"
        elif p_value < 0.01:
            return "Significant differences between algorithms (p < 0.01)"
        elif p_value < 0.05:
            return "Statistically significant differences between algorithms (p < 0.05)"
        else:
            return "No statistically significant differences between algorithms"

    def _generate_fallback_report(
        self,
        data: TechnicalReportData,
        output_path: str,
        format_type: str,
        context: dict[str, Any],
    ) -> str:
        """Generate a basic technical report when templates are not available."""

        report_content = f"""
# Technical Anomaly Detection Report

**Generated:** {context["generated_at"]}
**Experiment ID:** {data.experiment_id or "N/A"}

## Dataset Information

- **Name:** {data.dataset_name}
- **Shape:** {data.dataset_shape[0]:,} samples × {data.dataset_shape[1]} features
- **Features:** {", ".join(data.dataset_features[:10])}{"..." if len(data.dataset_features) > 10 else ""}

## Algorithm Performance Comparison

| Algorithm | Precision | Recall | F1-Score | AUC-ROC | AUC-PR |
|-----------|-----------|--------|----------|---------|--------|
"""

        for alg in data.algorithms:
            report_content += f"| {alg.name} | {alg.precision:.3f} | {alg.recall:.3f} | {alg.f1_score:.3f} | {alg.auc_roc:.3f} | {alg.auc_pr:.3f} |\n"

        # Best algorithm
        best_alg = context["best_algorithm"]
        if best_alg:
            report_content += f"\n## Best Algorithm\n\n**{best_alg['name']}** (Weighted Score: {best_alg['weighted_score']:.3f})\n\n"

        # Technical recommendations
        report_content += "\n## Technical Recommendations\n\n"
        for rec in context["technical_recommendations"]:
            report_content += f"- {rec}\n"

        # Methodology notes
        report_content += "\n## Methodology\n\n"
        for note in context["methodology_notes"]:
            report_content += f"- {note}\n"

        # Statistical analysis
        if context["statistical_analysis"]:
            report_content += "\n## Statistical Analysis\n\n"
            stat_analysis = context["statistical_analysis"]

            if "friedman_test" in stat_analysis:
                friedman = stat_analysis["friedman_test"]
                if "error" not in friedman:
                    report_content += f"**Friedman Test:** χ² = {friedman['statistic']:.3f}, p = {friedman['p_value']:.3f}\n"
                    report_content += (
                        f"**Interpretation:** {friedman['interpretation']}\n\n"
                    )

        # Save report
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_content)

        return str(output_file)


# Example usage and testing
if __name__ == "__main__":
    # Sample algorithm metrics
    algorithms = [
        AlgorithmMetrics(
            name="IsolationForest",
            precision=0.85,
            recall=0.78,
            f1_score=0.81,
            auc_roc=0.89,
            auc_pr=0.82,
            accuracy=0.94,
            training_time_seconds=45.2,
            prediction_time_seconds=2.1,
            memory_usage_mb=150.5,
            hyperparameters={"contamination": 0.1, "n_estimators": 100},
            confusion_matrix=[[8500, 150], [220, 780]],
            feature_importance={
                "feature_1": 0.25,
                "feature_2": 0.18,
                "feature_3": 0.15,
            },
        ),
        AlgorithmMetrics(
            name="LocalOutlierFactor",
            precision=0.82,
            recall=0.85,
            f1_score=0.83,
            auc_roc=0.87,
            auc_pr=0.80,
            accuracy=0.93,
            training_time_seconds=120.7,
            prediction_time_seconds=5.3,
            memory_usage_mb=280.2,
            hyperparameters={"contamination": 0.1, "n_neighbors": 20},
            confusion_matrix=[[8450, 200], [150, 850]],
        ),
        AlgorithmMetrics(
            name="OneClassSVM",
            precision=0.79,
            recall=0.72,
            f1_score=0.75,
            auc_roc=0.84,
            auc_pr=0.77,
            accuracy=0.91,
            training_time_seconds=300.5,
            prediction_time_seconds=8.7,
            memory_usage_mb=95.8,
            hyperparameters={"nu": 0.1, "gamma": "scale"},
            confusion_matrix=[[8600, 50], [280, 720]],
        ),
    ]

    # Sample technical data
    sample_data = TechnicalReportData(
        dataset_name="Financial_Transactions_Dataset",
        dataset_shape=(10000, 15),
        dataset_features=[
            "amount",
            "hour",
            "day_of_week",
            "merchant_category",
            "customer_age",
        ],
        target_distribution={"normal": 9000, "anomaly": 1000},
        algorithms=algorithms,
        cross_validation_results={
            "IsolationForest": [0.81, 0.83, 0.79, 0.82, 0.80],
            "LocalOutlierFactor": [0.83, 0.85, 0.81, 0.84, 0.82],
            "OneClassSVM": [0.75, 0.77, 0.73, 0.76, 0.74],
        },
        missing_values_pct=2.3,
        duplicate_rows_pct=0.5,
        experiment_id="EXP_2025_001",
        preprocessing_steps=["StandardScaler", "PCA(n_components=10)"],
        validation_method="5-Fold Cross Validation",
        random_seed=42,
    )

    # Generate report
    generator = TechnicalReportGenerator()
    report_path = generator.generate_report(
        data=sample_data,
        output_path="technical_report_sample.html",
        include_charts=True,
        include_statistical_tests=True,
    )

    print(f"Technical report generated: {report_path}")
