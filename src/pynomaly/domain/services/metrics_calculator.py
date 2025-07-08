"""
MetricsCalculator Domain Service

Provides comprehensive metrics computation for anomaly detection models
with support for both classification and anomaly score-based tasks.
"""

from __future__ import annotations

import asyncio
import statistics
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    average_precision_score,
    confusion_matrix,
    classification_report,
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


class MetricsCalculator:
    """
    Static and async helper class for computing comprehensive anomaly detection metrics.

    Supports both classification-based and anomaly score-based evaluation tasks,
    providing mean, std, and confidence intervals where applicable.
    """

    @staticmethod
    def compute(
        y_true: Union[np.ndarray, List[Union[int, float]]],
        y_pred: Union[np.ndarray, List[Union[int, float]]],
        proba: Optional[Union[np.ndarray, List[float]]] = None,
        task_type: str = "anomaly",
        confidence_level: float = 0.95,
        **kwargs
    ) -> Dict[str, Any]:
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
        y_true: Union[np.ndarray, List[Union[int, float]]],
        y_pred: Union[np.ndarray, List[Union[int, float]]],
        proba: Optional[Union[np.ndarray, List[float]]] = None,
        task_type: str = "anomaly",
        confidence_level: float = 0.95,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Asynchronously compute comprehensive metrics for anomaly detection models.

        Same parameters as compute() but runs in a separate thread to avoid blocking.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            MetricsCalculator.compute,
            y_true, y_pred, proba, task_type, confidence_level, **kwargs
        )

    @staticmethod
    def _validate_inputs(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        proba: Optional[np.ndarray] = None
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
            raise ValueError("y_true must contain only 0 (normal) and 1 (anomaly) labels")

        if not all(label in [0, 1] for label in unique_pred):
            raise ValueError("y_pred must contain only 0 (normal) and 1 (anomaly) labels")

    @staticmethod
    def _compute_anomaly_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        proba: Optional[np.ndarray],
        confidence_level: float,
        **kwargs
    ) -> Dict[str, Any]:
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
        confusion_metrics = MetricsCalculator._compute_confusion_matrix_metrics(y_true, y_pred)
        results.update(confusion_metrics)

        # Statistical analysis
        stats = MetricsCalculator._compute_statistical_analysis(
            basic_metrics, confidence_level
        )
        results.update(stats)

        return results

    @staticmethod
    def _compute_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        proba: Optional[np.ndarray],
        confidence_level: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Compute metrics for general classification tasks."""
        results = {}

        # Basic classification metrics
        basic_metrics = MetricsCalculator._compute_basic_metrics(y_true, y_pred)
        results.update(basic_metrics)

        # Probability-based metrics
        if proba is not None:
            prob_metrics = MetricsCalculator._compute_probability_metrics(
                y_true, proba, confidence_level
            )
            results.update(prob_metrics)

        # Detailed classification report
        try:
            report = classification_report(y_true, y_pred, output_dict=True)
            results['classification_report'] = report
        except Exception as e:
            results['classification_report_error'] = str(e)

        # Statistical analysis
        stats = MetricsCalculator._compute_statistical_analysis(
            basic_metrics, confidence_level
        )
        results.update(stats)

        return results

    @staticmethod
    def _compute_clustering_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidence_level: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Compute metrics for clustering evaluation."""
        results = {}

        try:
            # Adjusted Rand Index
            ari = adjusted_rand_score(y_true, y_pred)
            results['adjusted_rand_index'] = {
                'value': ari,
                'mean': ari,
                'std': 0.0,
                'confidence_interval': (ari, ari)
            }

            # Normalized Mutual Information
            nmi = normalized_mutual_info_score(y_true, y_pred)
            results['normalized_mutual_info'] = {
                'value': nmi,
                'mean': nmi,
                'std': 0.0,
                'confidence_interval': (nmi, nmi)
            }

            # Silhouette score (requires feature data)
            if 'X' in kwargs:
                X = kwargs['X']
                if len(np.unique(y_pred)) > 1:
                    silhouette = silhouette_score(X, y_pred)
                    results['silhouette_score'] = {
                        'value': silhouette,
                        'mean': silhouette,
                        'std': 0.0,
                        'confidence_interval': (silhouette, silhouette)
                    }

        except Exception as e:
            results['clustering_error'] = str(e)

        return results

    @staticmethod
    def _compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Compute basic classification metrics."""
        metrics = {}

        try:
            # Accuracy
            accuracy = accuracy_score(y_true, y_pred)
            metrics['accuracy'] = {
                'value': accuracy,
                'mean': accuracy,
                'std': 0.0
            }

            # Precision
            precision = precision_score(y_true, y_pred, zero_division=0)
            metrics['precision'] = {
                'value': precision,
                'mean': precision,
                'std': 0.0
            }

            # Recall
            recall = recall_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = {
                'value': recall,
                'mean': recall,
                'std': 0.0
            }

            # F1 Score
            f1 = f1_score(y_true, y_pred, zero_division=0)
            metrics['f1_score'] = {
                'value': f1,
                'mean': f1,
                'std': 0.0
            }

            # Specificity (True Negative Rate)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['specificity'] = {
                'value': specificity,
                'mean': specificity,
                'std': 0.0
            }

        except Exception as e:
            metrics['basic_metrics_error'] = str(e)

        return metrics

    @staticmethod
    def _compute_anomaly_score_metrics(
        y_true: np.ndarray,
        proba: np.ndarray,
        confidence_level: float
    ) -> Dict[str, Any]:
        """Compute metrics specific to anomaly scores."""
        metrics = {}

        try:
            # ROC AUC
            if len(np.unique(y_true)) > 1:
                roc_auc = roc_auc_score(y_true, proba)
                metrics['roc_auc'] = {
                    'value': roc_auc,
                    'mean': roc_auc,
                    'std': 0.0
                }

            # Precision-Recall AUC
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, proba)
            pr_auc = auc(recall_curve, precision_curve)
            metrics['pr_auc'] = {
                'value': pr_auc,
                'mean': pr_auc,
                'std': 0.0
            }

            # Average Precision
            avg_precision = average_precision_score(y_true, proba)
            metrics['average_precision'] = {
                'value': avg_precision,
                'mean': avg_precision,
                'std': 0.0
            }

            # Anomaly score statistics
            anomaly_scores = proba[y_true == 1]
            normal_scores = proba[y_true == 0]

            if len(anomaly_scores) > 0:
                metrics['anomaly_score_stats'] = {
                    'mean': float(np.mean(anomaly_scores)),
                    'std': float(np.std(anomaly_scores)),
                    'min': float(np.min(anomaly_scores)),
                    'max': float(np.max(anomaly_scores)),
                    'median': float(np.median(anomaly_scores))
                }

            if len(normal_scores) > 0:
                metrics['normal_score_stats'] = {
                    'mean': float(np.mean(normal_scores)),
                    'std': float(np.std(normal_scores)),
                    'min': float(np.min(normal_scores)),
                    'max': float(np.max(normal_scores)),
                    'median': float(np.median(normal_scores))
                }

            # Score separation metrics
            if len(anomaly_scores) > 0 and len(normal_scores) > 0:
                separation = np.mean(anomaly_scores) - np.mean(normal_scores)
                metrics['score_separation'] = {
                    'value': float(separation),
                    'mean': float(separation),
                    'std': 0.0
                }

        except Exception as e:
            metrics['anomaly_score_error'] = str(e)

        return metrics

    @staticmethod
    def _compute_probability_metrics(
        y_true: np.ndarray,
        proba: np.ndarray,
        confidence_level: float
    ) -> Dict[str, Any]:
        """Compute probability-based metrics."""
        metrics = {}

        try:
            # Calibration metrics
            positive_proba = proba[y_true == 1]
            negative_proba = proba[y_true == 0]

            if len(positive_proba) > 0:
                metrics['positive_class_proba'] = {
                    'mean': float(np.mean(positive_proba)),
                    'std': float(np.std(positive_proba)),
                    'confidence_interval': MetricsCalculator._compute_confidence_interval(
                        positive_proba, confidence_level
                    )
                }

            if len(negative_proba) > 0:
                metrics['negative_class_proba'] = {
                    'mean': float(np.mean(negative_proba)),
                    'std': float(np.std(negative_proba)),
                    'confidence_interval': MetricsCalculator._compute_confidence_interval(
                        negative_proba, confidence_level
                    )
                }

            # Probability distribution metrics
            metrics['probability_stats'] = {
                'mean': float(np.mean(proba)),
                'std': float(np.std(proba)),
                'min': float(np.min(proba)),
                'max': float(np.max(proba)),
                'median': float(np.median(proba))
            }

        except Exception as e:
            metrics['probability_error'] = str(e)

        return metrics

    @staticmethod
    def _compute_confusion_matrix_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Compute detailed confusion matrix metrics."""
        metrics = {}

        try:
            cm = confusion_matrix(y_true, y_pred)

            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()

                metrics['confusion_matrix'] = {
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'true_positives': int(tp),
                    'matrix': cm.tolist()
                }

                # Derived metrics
                total = tn + fp + fn + tp
                metrics['confusion_matrix_derived'] = {
                    'true_positive_rate': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                    'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
                    'true_negative_rate': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                    'false_negative_rate': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
                    'positive_predictive_value': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                    'negative_predictive_value': float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
                    'prevalence': float((tp + fn) / total) if total > 0 else 0.0,
                    'detection_rate': float(tp / total) if total > 0 else 0.0,
                    'detection_prevalence': float((tp + fp) / total) if total > 0 else 0.0,
                    'balanced_accuracy': float((tp/(tp+fn) + tn/(tn+fp)) / 2) if (tp+fn) > 0 and (tn+fp) > 0 else 0.0
                }

        except Exception as e:
            metrics['confusion_matrix_error'] = str(e)

        return metrics

    @staticmethod
    def _compute_statistical_analysis(
        metrics: Dict[str, Any],
        confidence_level: float
    ) -> Dict[str, Any]:
        """Compute statistical analysis and confidence intervals."""
        results = {}

        try:
            # Add confidence intervals to existing metrics
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and 'value' in metric_data:
                    value = metric_data['value']
                    # For single values, confidence interval is just the value
                    metric_data['confidence_interval'] = (value, value)

            # Overall performance summary
            performance_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            available_metrics = [
                metrics[m]['value'] for m in performance_metrics
                if m in metrics and 'value' in metrics[m]
            ]

            if available_metrics:
                results['performance_summary'] = {
                    'mean': float(np.mean(available_metrics)),
                    'std': float(np.std(available_metrics)),
                    'min': float(np.min(available_metrics)),
                    'max': float(np.max(available_metrics)),
                    'confidence_interval': MetricsCalculator._compute_confidence_interval(
                        np.array(available_metrics), confidence_level
                    )
                }

        except Exception as e:
            results['statistical_analysis_error'] = str(e)

        return results

    @staticmethod
    def _compute_confidence_interval(
        values: np.ndarray,
        confidence_level: float
    ) -> Tuple[float, float]:
        """Compute confidence interval for a set of values."""
        if len(values) < 2:
            return (float(values[0]), float(values[0])) if len(values) == 1 else (0.0, 0.0)

        mean = np.mean(values)
        std_err = np.std(values, ddof=1) / np.sqrt(len(values))

        # Use t-distribution for small samples
        from scipy import stats
        alpha = 1 - confidence_level
        df = len(values) - 1
        t_critical = stats.t.ppf(1 - alpha/2, df)

        margin_of_error = t_critical * std_err
        lower = mean - margin_of_error
        upper = mean + margin_of_error

        return (float(lower), float(upper))

    @staticmethod
    def compute_cross_validation_metrics(
        cv_results: List[Dict[str, Any]],
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Compute aggregated metrics from cross-validation results.

        Args:
            cv_results: List of metric dictionaries from each CV fold
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary with aggregated metrics including mean, std, and confidence intervals
        """
        if not cv_results:
            raise ValueError("cv_results cannot be empty")

        aggregated = {}

        # Get all metric names from first fold
        metric_names = set()
        for fold_results in cv_results:
            for metric_name in fold_results.keys():
                if isinstance(fold_results[metric_name], dict) and 'value' in fold_results[metric_name]:
                    metric_names.add(metric_name)

        # Aggregate each metric across folds
        for metric_name in metric_names:
            values = []
            for fold_results in cv_results:
                if metric_name in fold_results and 'value' in fold_results[metric_name]:
                    values.append(fold_results[metric_name]['value'])

            if values:
                values_array = np.array(values)
                aggregated[metric_name] = {
                    'values': values,
                    'mean': float(np.mean(values_array)),
                    'std': float(np.std(values_array, ddof=1)),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array)),
                    'confidence_interval': MetricsCalculator._compute_confidence_interval(
                        values_array, confidence_level
                    )
                }

        return aggregated

    @staticmethod
    def compare_models(
        model_results: Dict[str, Dict[str, Any]],
        primary_metric: str = 'f1_score'
    ) -> Dict[str, Any]:
        """
        Compare multiple models based on their metrics.

        Args:
            model_results: Dictionary mapping model names to their metrics
            primary_metric: Primary metric for ranking models

        Returns:
            Dictionary with comparison results and rankings
        """
        if not model_results:
            raise ValueError("model_results cannot be empty")

        comparison = {
            'rankings': {},
            'comparisons': {},
            'summary': {}
        }

        # Extract primary metric values
        primary_values = {}
        for model_name, results in model_results.items():
            if primary_metric in results:
                if isinstance(results[primary_metric], dict) and 'value' in results[primary_metric]:
                    primary_values[model_name] = results[primary_metric]['value']
                else:
                    primary_values[model_name] = results[primary_metric]

        # Rank models by primary metric
        sorted_models = sorted(primary_values.items(), key=lambda x: x[1], reverse=True)
        comparison['rankings'][primary_metric] = [
            {'model': model, 'value': value, 'rank': rank + 1}
            for rank, (model, value) in enumerate(sorted_models)
        ]

        # Pairwise comparisons
        model_names = list(model_results.keys())
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                comparison_key = f"{model1}_vs_{model2}"

                # Compare all common metrics
                common_metrics = set(model_results[model1].keys()) & set(model_results[model2].keys())
                metric_comparisons = {}

                for metric in common_metrics:
                    val1 = model_results[model1][metric]
                    val2 = model_results[model2][metric]

                    if isinstance(val1, dict) and isinstance(val2, dict):
                        if 'value' in val1 and 'value' in val2:
                            diff = val1['value'] - val2['value']
                            metric_comparisons[metric] = {
                                'difference': diff,
                                'better_model': model1 if diff > 0 else model2,
                                'model1_value': val1['value'],
                                'model2_value': val2['value']
                            }

                comparison['comparisons'][comparison_key] = metric_comparisons

        # Summary statistics
        if primary_values:
            values = list(primary_values.values())
            comparison['summary'] = {
                'best_model': max(primary_values, key=primary_values.get),
                'worst_model': min(primary_values, key=primary_values.get),
                'mean_performance': float(np.mean(values)),
                'std_performance': float(np.std(values)),
                'performance_range': float(max(values) - min(values))
            }

        return comparison
