"""
MetricsCalculator Domain Service

Provides comprehensive metrics computation for anomaly detection models
with support for both classification and anomaly score-based tasks.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)


class MetricsCalculator:
    """
    Static and async helper class for computing comprehensive anomaly detection metrics.

    Supports both classification-based and anomaly score-based evaluation tasks,
    providing mean, std, and confidence intervals where applicable.
    """

    @staticmethod
    def compute(
        y_true: np.ndarray | list[int | float],
        y_pred: np.ndarray | list[int | float],
        proba: np.ndarray | list[float] | None = None,
        task_type: str = "anomaly",
        confidence_level: float = 0.95,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Compute comprehensive metrics for anomaly detection models.

        Args:
            y_true: True labels (0 for normal, 1 for anomaly)
            y_pred: Predicted labels (0 for normal, 1 for anomaly)
            proba: Predicted probabilities/anomaly scores (optional)
            task_type: Type of task ("anomaly", "classification", "clustering")
            confidence_level: Confidence level for intervals (default: 0.95)
            **kwargs: Additional parameters for specific metrics

        Returns:
            Dictionary containing comprehensive metrics with statistics
        """
        # Convert inputs to numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if proba is not None:
            proba = np.asarray(proba)

        # Validate inputs
        MetricsCalculator._validate_inputs(y_true, y_pred, proba)

        # Compute metrics based on task type
        if task_type.lower() == "anomaly":
            return MetricsCalculator._compute_anomaly_metrics(
                y_true, y_pred, proba, confidence_level, **kwargs
            )
        elif task_type.lower() == "classification":
            return MetricsCalculator._compute_classification_metrics(
                y_true, y_pred, proba, confidence_level, **kwargs
            )
        elif task_type.lower() == "clustering":
            return MetricsCalculator._compute_clustering_metrics(
                y_true, y_pred, confidence_level, **kwargs
            )
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

    @staticmethod
    async def compute_async(
        y_true: np.ndarray | list[int | float],
        y_pred: np.ndarray | list[int | float],
        proba: np.ndarray | list[float] | None = None,
        task_type: str = "anomaly",
        confidence_level: float = 0.95,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Asynchronously compute comprehensive metrics for anomaly detection models.

        Same parameters as compute() but runs in a separate thread to avoid blocking.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            MetricsCalculator.compute,
            y_true,
            y_pred,
            proba,
            task_type,
            confidence_level,
            **kwargs,
        )

    @staticmethod
    def _validate_inputs(
        y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None = None
    ) -> None:
        """Validate input arrays for metrics computation."""
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true and y_pred must have same length: {len(y_true)} vs {len(y_pred)}"
            )

        if proba is not None and len(proba) != len(y_true):
            raise ValueError(
                f"proba must have same length as y_true: {len(proba)} vs {len(y_true)}"
            )

        if len(y_true) == 0:
            raise ValueError("Input arrays cannot be empty")

        # Check for valid label values
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)

        if not all(label in [0, 1] for label in unique_true):
            raise ValueError(
                "y_true must contain only 0 (normal) and 1 (anomaly) labels"
            )

        if not all(label in [0, 1] for label in unique_pred):
            raise ValueError(
                "y_pred must contain only 0 (normal) and 1 (anomaly) labels"
            )

    @staticmethod
    def _compute_anomaly_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        proba: np.ndarray | None,
        confidence_level: float,
        **kwargs,
    ) -> dict[str, Any]:
        """Compute metrics specific to anomaly detection tasks."""
        results = {}

        # Basic classification metrics
        basic_metrics = MetricsCalculator._compute_basic_metrics(y_true, y_pred)
        results.update(basic_metrics)

        # Anomaly-specific metrics
        if proba is not None:
            anomaly_metrics = MetricsCalculator._compute_anomaly_score_metrics(
                y_true, proba, confidence_level
            )
            results.update(anomaly_metrics)

        # Confusion matrix analysis
        confusion_metrics = MetricsCalculator._compute_confusion_matrix_metrics(
            y_true, y_pred
        )
        results.update(confusion_metrics)

        # Statistical analysis
        stats = MetricsCalculator._compute_statistical_analysis(
            basic_metrics, confidence_level
        )
        results.update(stats)

        return results

    @staticmethod
    def _compute_basic_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, Any]:
        """Compute basic classification metrics."""
        metrics = {}

        try:
            # Accuracy
            accuracy = accuracy_score(y_true, y_pred)
            metrics["accuracy"] = {"value": accuracy, "mean": accuracy, "std": 0.0}

            # Precision
            precision = precision_score(y_true, y_pred, zero_division=0)
            metrics["precision"] = {"value": precision, "mean": precision, "std": 0.0}

            # Recall
            recall = recall_score(y_true, y_pred, zero_division=0)
            metrics["recall"] = {"value": recall, "mean": recall, "std": 0.0}

            # F1 Score
            f1 = f1_score(y_true, y_pred, zero_division=0)
            metrics["f1_score"] = {"value": f1, "mean": f1, "std": 0.0}

            # Specificity (True Negative Rate)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics["specificity"] = {
                "value": specificity,
                "mean": specificity,
                "std": 0.0,
            }

        except Exception as e:
            metrics["basic_metrics_error"] = str(e)

        return metrics

    @staticmethod
    def _compute_anomaly_score_metrics(
        y_true: np.ndarray, proba: np.ndarray, confidence_level: float
    ) -> dict[str, Any]:
        """Compute metrics specific to anomaly scores."""
        metrics = {}

        try:
            # ROC AUC
            if len(np.unique(y_true)) > 1:
                roc_auc = roc_auc_score(y_true, proba)
                metrics["roc_auc"] = {"value": roc_auc, "mean": roc_auc, "std": 0.0}

            # Precision-Recall AUC
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, proba)
            pr_auc = auc(recall_curve, precision_curve)
            metrics["pr_auc"] = {"value": pr_auc, "mean": pr_auc, "std": 0.0}

            # Average Precision
            avg_precision = average_precision_score(y_true, proba)
            metrics["average_precision"] = {
                "value": avg_precision,
                "mean": avg_precision,
                "std": 0.0,
            }

            # Anomaly score statistics
            anomaly_scores = proba[y_true == 1]
            normal_scores = proba[y_true == 0]

            if len(anomaly_scores) > 0:
                metrics["anomaly_score_stats"] = {
                    "mean": float(np.mean(anomaly_scores)),
                    "std": float(np.std(anomaly_scores)),
                    "min": float(np.min(anomaly_scores)),
                    "max": float(np.max(anomaly_scores)),
                    "median": float(np.median(anomaly_scores)),
                }

            if len(normal_scores) > 0:
                metrics["normal_score_stats"] = {
                    "mean": float(np.mean(normal_scores)),
                    "std": float(np.std(normal_scores)),
                    "min": float(np.min(normal_scores)),
                    "max": float(np.max(normal_scores)),
                    "median": float(np.median(normal_scores)),
                }

            # Score separation metrics
            if len(anomaly_scores) > 0 and len(normal_scores) > 0:
                separation = np.mean(anomaly_scores) - np.mean(normal_scores)
                metrics["score_separation"] = {
                    "value": float(separation),
                    "mean": float(separation),
                    "std": 0.0,
                }

        except Exception as e:
            metrics["anomaly_score_error"] = str(e)

        return metrics

    @staticmethod
    def _compute_confusion_matrix_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, Any]:
        """Compute detailed confusion matrix metrics."""
        metrics = {}

        try:
            cm = confusion_matrix(y_true, y_pred)

            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()

                metrics["confusion_matrix"] = {
                    "true_negatives": int(tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "true_positives": int(tp),
                    "matrix": cm.tolist(),
                }

                # Derived metrics
                total = tn + fp + fn + tp
                metrics["confusion_matrix_derived"] = {
                    "true_positive_rate": float(tp / (tp + fn))
                    if (tp + fn) > 0
                    else 0.0,
                    "false_positive_rate": float(fp / (fp + tn))
                    if (fp + tn) > 0
                    else 0.0,
                    "true_negative_rate": float(tn / (tn + fp))
                    if (tn + fp) > 0
                    else 0.0,
                    "false_negative_rate": float(fn / (fn + tp))
                    if (fn + tp) > 0
                    else 0.0,
                    "positive_predictive_value": float(tp / (tp + fp))
                    if (tp + fp) > 0
                    else 0.0,
                    "negative_predictive_value": float(tn / (tn + fn))
                    if (tn + fn) > 0
                    else 0.0,
                    "prevalence": float((tp + fn) / total) if total > 0 else 0.0,
                    "detection_rate": float(tp / total) if total > 0 else 0.0,
                    "detection_prevalence": float((tp + fp) / total)
                    if total > 0
                    else 0.0,
                    "balanced_accuracy": float((tp / (tp + fn) + tn / (tn + fp)) / 2)
                    if (tp + fn) > 0 and (tn + fp) > 0
                    else 0.0,
                }

        except Exception as e:
            metrics["confusion_matrix_error"] = str(e)

        return metrics

    @staticmethod
    def _compute_statistical_analysis(
        metrics: dict[str, Any], confidence_level: float
    ) -> dict[str, Any]:
        """Compute statistical analysis and confidence intervals."""
        results = {}

        try:
            # Add confidence intervals to existing metrics
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and "value" in metric_data:
                    value = metric_data["value"]
                    # For single values, confidence interval is just the value
                    metric_data["confidence_interval"] = (value, value)

            # Overall performance summary
            performance_metrics = ["accuracy", "precision", "recall", "f1_score"]
            available_metrics = [
                metrics[m]["value"]
                for m in performance_metrics
                if m in metrics and "value" in metrics[m]
            ]

            if available_metrics:
                results["performance_summary"] = {
                    "mean": float(np.mean(available_metrics)),
                    "std": float(np.std(available_metrics)),
                    "min": float(np.min(available_metrics)),
                    "max": float(np.max(available_metrics)),
                    "confidence_interval": MetricsCalculator._compute_confidence_interval(
                        np.array(available_metrics), confidence_level
                    ),
                }

        except Exception as e:
            results["statistical_analysis_error"] = str(e)

        return results

    @staticmethod
    def _compute_confidence_interval(
        values: np.ndarray, confidence_level: float
    ) -> tuple[float, float]:
        """Compute confidence interval for a set of values."""
        if len(values) < 2:
            return (
                (float(values[0]), float(values[0])) if len(values) == 1 else (0.0, 0.0)
            )

        mean = np.mean(values)
        std_err = np.std(values, ddof=1) / np.sqrt(len(values))

        # Use t-distribution for small samples
        from scipy import stats

        alpha = 1 - confidence_level
        df = len(values) - 1
        t_critical = stats.t.ppf(1 - alpha / 2, df)

        margin_of_error = t_critical * std_err
        lower = mean - margin_of_error
        upper = mean + margin_of_error

        return (float(lower), float(upper))