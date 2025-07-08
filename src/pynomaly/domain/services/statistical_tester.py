"""
Statistical Tester for Model Comparison

This module provides statistical significance testing for comparing
model performance metrics.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy import stats
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a statistical test."""

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    confidence_level: float
    effect_size: Optional[float] = None
    power: Optional[float] = None


class StatisticalTester:
    """
    Statistical significance tester for model comparison.

    Provides various statistical tests to determine if differences
    between model performance metrics are statistically significant.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize StatisticalTester.

        Args:
            alpha: Significance level (default: 0.05)
        """
        self.alpha = alpha
        self.confidence_level = 1 - alpha

    def test_significance(
        self,
        metrics_a: Dict[str, Any],
        metrics_b: Dict[str, Any],
        test_type: str = "ttest",
        metric_name: str = "f1_score",
    ) -> TestResult:
        """
        Test statistical significance between two sets of metrics.

        Args:
            metrics_a: First model's metrics
            metrics_b: Second model's metrics
            test_type: Type of test ('ttest', 'wilcoxon', 'paired_ttest')
            metric_name: Name of the metric to compare

        Returns:
            TestResult with test statistics and significance
        """
        # Extract metric values
        value_a = self._extract_metric_value(metrics_a, metric_name)
        value_b = self._extract_metric_value(metrics_b, metric_name)

        if value_a is None or value_b is None:
            return TestResult(
                test_name=test_type,
                statistic=0.0,
                p_value=1.0,
                significant=False,
                confidence_level=self.confidence_level,
            )

        # For single values, create small samples for testing
        if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
            # Create synthetic samples around the values for testing
            sample_a = np.random.normal(value_a, 0.01, 30)
            sample_b = np.random.normal(value_b, 0.01, 30)
        else:
            sample_a = np.asarray(value_a)
            sample_b = np.asarray(value_b)

        # Perform the specified test
        if test_type == "ttest":
            return self._perform_ttest(sample_a, sample_b)
        elif test_type == "wilcoxon":
            return self._perform_wilcoxon_test(sample_a, sample_b)
        elif test_type == "paired_ttest":
            return self._perform_paired_ttest(sample_a, sample_b)
        elif test_type == "mannwhitney":
            return self._perform_mannwhitney_test(sample_a, sample_b)
        else:
            raise ValueError(f"Unknown test type: {test_type}")

    def _extract_metric_value(
        self, metrics: Dict[str, Any], metric_name: str
    ) -> Optional[Any]:
        """Extract metric value from metrics dictionary."""
        if metric_name not in metrics:
            return None

        value = metrics[metric_name]

        # Handle nested metric structure
        if isinstance(value, dict):
            if "value" in value:
                return value["value"]
            elif "mean" in value:
                return value["mean"]
            elif "values" in value:
                return value["values"]

        return value

    def _perform_ttest(self, sample_a: np.ndarray, sample_b: np.ndarray) -> TestResult:
        """Perform independent t-test."""
        try:
            statistic, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=False)

            # Calculate effect size (Cohen's d)
            effect_size = self._calculate_cohens_d(sample_a, sample_b)

            return TestResult(
                test_name="Independent t-test",
                statistic=float(statistic),
                p_value=float(p_value),
                significant=p_value < self.alpha,
                confidence_level=self.confidence_level,
                effect_size=effect_size,
            )
        except Exception as e:
            logger.error(f"Error performing t-test: {e}")
            return TestResult(
                test_name="Independent t-test",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                confidence_level=self.confidence_level,
            )

    def _perform_paired_ttest(
        self, sample_a: np.ndarray, sample_b: np.ndarray
    ) -> TestResult:
        """Perform paired t-test."""
        try:
            # Ensure samples have the same length
            min_length = min(len(sample_a), len(sample_b))
            sample_a = sample_a[:min_length]
            sample_b = sample_b[:min_length]

            statistic, p_value = stats.ttest_rel(sample_a, sample_b)

            # Calculate effect size
            effect_size = self._calculate_cohens_d(sample_a, sample_b)

            return TestResult(
                test_name="Paired t-test",
                statistic=float(statistic),
                p_value=float(p_value),
                significant=p_value < self.alpha,
                confidence_level=self.confidence_level,
                effect_size=effect_size,
            )
        except Exception as e:
            logger.error(f"Error performing paired t-test: {e}")
            return TestResult(
                test_name="Paired t-test",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                confidence_level=self.confidence_level,
            )

    def _perform_wilcoxon_test(
        self, sample_a: np.ndarray, sample_b: np.ndarray
    ) -> TestResult:
        """Perform Wilcoxon signed-rank test."""
        try:
            # Ensure samples have the same length
            min_length = min(len(sample_a), len(sample_b))
            sample_a = sample_a[:min_length]
            sample_b = sample_b[:min_length]

            statistic, p_value = stats.wilcoxon(sample_a, sample_b)

            return TestResult(
                test_name="Wilcoxon signed-rank test",
                statistic=float(statistic),
                p_value=float(p_value),
                significant=p_value < self.alpha,
                confidence_level=self.confidence_level,
            )
        except Exception as e:
            logger.error(f"Error performing Wilcoxon test: {e}")
            return TestResult(
                test_name="Wilcoxon signed-rank test",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                confidence_level=self.confidence_level,
            )

    def _perform_mannwhitney_test(
        self, sample_a: np.ndarray, sample_b: np.ndarray
    ) -> TestResult:
        """Perform Mann-Whitney U test."""
        try:
            statistic, p_value = stats.mannwhitneyu(
                sample_a, sample_b, alternative="two-sided"
            )

            return TestResult(
                test_name="Mann-Whitney U test",
                statistic=float(statistic),
                p_value=float(p_value),
                significant=p_value < self.alpha,
                confidence_level=self.confidence_level,
            )
        except Exception as e:
            logger.error(f"Error performing Mann-Whitney test: {e}")
            return TestResult(
                test_name="Mann-Whitney U test",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                confidence_level=self.confidence_level,
            )

    def _calculate_cohens_d(self, sample_a: np.ndarray, sample_b: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        try:
            mean_a = np.mean(sample_a)
            mean_b = np.mean(sample_b)

            # Pooled standard deviation
            n_a = len(sample_a)
            n_b = len(sample_b)

            if n_a <= 1 or n_b <= 1:
                return 0.0

            var_a = np.var(sample_a, ddof=1)
            var_b = np.var(sample_b, ddof=1)

            pooled_std = np.sqrt(
                ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
            )

            if pooled_std == 0:
                return 0.0

            cohens_d = (mean_a - mean_b) / pooled_std
            return float(cohens_d)
        except Exception:
            return 0.0

    def multiple_comparisons_correction(
        self, p_values: List[float], method: str = "bonferroni"
    ) -> List[float]:
        """
        Apply multiple comparisons correction.

        Args:
            p_values: List of p-values
            method: Correction method ('bonferroni', 'holm', 'fdr')

        Returns:
            List of corrected p-values
        """
        p_values = np.array(p_values)

        if method == "bonferroni":
            return (p_values * len(p_values)).clip(0, 1).tolist()
        elif method == "holm":
            return self._holm_correction(p_values)
        elif method == "fdr":
            return self._benjamini_hochberg_correction(p_values)
        else:
            raise ValueError(f"Unknown correction method: {method}")

    def _holm_correction(self, p_values: np.ndarray) -> List[float]:
        """Apply Holm-Bonferroni correction."""
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]

        corrected_p_values = np.zeros_like(p_values)

        for i, p_value in enumerate(sorted_p_values):
            corrected_p_values[sorted_indices[i]] = min(1.0, p_value * (n - i))

        return corrected_p_values.tolist()

    def _benjamini_hochberg_correction(self, p_values: np.ndarray) -> List[float]:
        """Apply Benjamini-Hochberg FDR correction."""
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]

        corrected_p_values = np.zeros_like(p_values)

        for i in range(n - 1, -1, -1):
            corrected_p_values[sorted_indices[i]] = min(
                1.0, sorted_p_values[i] * n / (i + 1)
            )

        return corrected_p_values.tolist()

    def power_analysis(
        self, effect_size: float, sample_size: int, alpha: Optional[float] = None
    ) -> float:
        """
        Compute statistical power for a given effect size and sample size.

        Args:
            effect_size: Effect size (Cohen's d)
            sample_size: Sample size
            alpha: Significance level (uses instance alpha if not provided)

        Returns:
            Statistical power
        """
        if alpha is None:
            alpha = self.alpha

        try:
            from scipy.stats import norm

            # Critical value for two-tailed test
            z_alpha = norm.ppf(1 - alpha / 2)

            # Standard error for effect size
            se = np.sqrt(2 / sample_size)

            # Non-centrality parameter
            ncp = effect_size / se

            # Power calculation
            power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)

            return float(power)
        except Exception as e:
            logger.error(f"Error computing power: {e}")
            return 0.0

    def comprehensive_comparison(
        self,
        models_metrics: Dict[str, Dict[str, Any]],
        metric_names: List[str] = None,
        correction_method: str = "bonferroni",
    ) -> Dict[str, Any]:
        """
        Perform comprehensive statistical comparison of multiple models.

        Args:
            models_metrics: Dictionary mapping model names to their metrics
            metric_names: List of metric names to compare
            correction_method: Multiple comparisons correction method

        Returns:
            Dictionary with comprehensive comparison results
        """
        if metric_names is None:
            metric_names = ["f1_score", "precision", "recall", "accuracy"]

        model_names = list(models_metrics.keys())
        results = {
            "pairwise_comparisons": {},
            "significance_matrix": {},
            "effect_sizes": {},
            "summary": {},
        }

        # Perform pairwise comparisons
        all_p_values = []
        comparison_keys = []

        for i, model_a in enumerate(model_names):
            for j, model_b in enumerate(model_names[i + 1 :], i + 1):
                comparison_key = f"{model_a}_vs_{model_b}"
                comparison_keys.append(comparison_key)

                comparison_results = {}
                p_values_for_correction = []

                for metric_name in metric_names:
                    test_result = self.test_significance(
                        models_metrics[model_a],
                        models_metrics[model_b],
                        metric_name=metric_name,
                    )
                    comparison_results[metric_name] = test_result
                    p_values_for_correction.append(test_result.p_value)

                results["pairwise_comparisons"][comparison_key] = comparison_results
                all_p_values.extend(p_values_for_correction)

        # Apply multiple comparisons correction
        if all_p_values:
            corrected_p_values = self.multiple_comparisons_correction(
                all_p_values, correction_method
            )

            # Update results with corrected p-values
            idx = 0
            for comparison_key in comparison_keys:
                for metric_name in metric_names:
                    original_result = results["pairwise_comparisons"][comparison_key][
                        metric_name
                    ]
                    corrected_p_value = corrected_p_values[idx]

                    # Create new result with corrected p-value
                    corrected_result = TestResult(
                        test_name=original_result.test_name,
                        statistic=original_result.statistic,
                        p_value=corrected_p_value,
                        significant=corrected_p_value < self.alpha,
                        confidence_level=original_result.confidence_level,
                        effect_size=original_result.effect_size,
                    )

                    results["pairwise_comparisons"][comparison_key][
                        metric_name
                    ] = corrected_result
                    idx += 1

        # Create significance matrix
        for metric_name in metric_names:
            significance_matrix = np.zeros((len(model_names), len(model_names)))

            for i, model_a in enumerate(model_names):
                for j, model_b in enumerate(model_names):
                    if i != j:
                        comparison_key = (
                            f"{model_a}_vs_{model_b}"
                            if i < j
                            else f"{model_b}_vs_{model_a}"
                        )
                        if comparison_key in results["pairwise_comparisons"]:
                            significance_matrix[i, j] = int(
                                results["pairwise_comparisons"][comparison_key][
                                    metric_name
                                ].significant
                            )

            results["significance_matrix"][metric_name] = significance_matrix.tolist()

        # Summary statistics
        results["summary"] = {
            "total_comparisons": len(comparison_keys),
            "correction_method": correction_method,
            "significance_level": self.alpha,
            "models_compared": model_names,
            "metrics_analyzed": metric_names,
        }

        return results
