"""Performance baseline comparison service for model degradation detection."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from statistics import mean, stdev

from packages.data_science.domain.value_objects.model_performance_metrics import (
    ModelPerformanceMetrics,
    ModelTask,
)
from packages.data_science.domain.value_objects.performance_degradation_metrics import (
    PerformanceDegradationMetrics,
    DegradationMetricType,
    DegradationSeverity,
)


class PerformanceBaselineService:
    """Service for managing performance baselines and comparisons.
    
    This service provides functionality to establish performance baselines,
    compare current performance against baselines, and detect statistically
    significant degradations.
    """
    
    def __init__(self, confidence_level: float = 0.95, min_samples: int = 30):
        """Initialize the baseline service.
        
        Args:
            confidence_level: Statistical confidence level for significance testing
            min_samples: Minimum number of samples required for baseline establishment
        """
        self.confidence_level = confidence_level
        self.min_samples = min_samples
    
    def establish_baseline(
        self,
        historical_metrics: List[ModelPerformanceMetrics],
        baseline_method: str = "recent_average",
        lookback_days: int = 30,
    ) -> ModelPerformanceMetrics:
        """Establish a performance baseline from historical metrics.
        
        Args:
            historical_metrics: List of historical performance metrics
            baseline_method: Method for baseline calculation
            lookback_days: Number of days to look back for baseline calculation
        
        Returns:
            Baseline performance metrics
        
        Raises:
            ValueError: If insufficient data or invalid method
        """
        if not historical_metrics:
            raise ValueError("Historical metrics cannot be empty")
        
        if len(historical_metrics) < self.min_samples:
            raise ValueError(f"Need at least {self.min_samples} samples for baseline")
        
        # Filter metrics by lookback period if evaluation_date is available
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
        recent_metrics = [
            metric for metric in historical_metrics
            if hasattr(metric, 'evaluation_date') and 
            metric.evaluation_date and 
            metric.evaluation_date >= cutoff_date
        ]
        
        if not recent_metrics:
            recent_metrics = historical_metrics[-min(len(historical_metrics), lookback_days):]
        
        if baseline_method == "recent_average":
            return self._calculate_average_baseline(recent_metrics)
        elif baseline_method == "best_performance":
            return self._calculate_best_performance_baseline(recent_metrics)
        elif baseline_method == "median_performance":
            return self._calculate_median_baseline(recent_metrics)
        elif baseline_method == "weighted_average":
            return self._calculate_weighted_average_baseline(recent_metrics)
        else:
            raise ValueError(f"Unknown baseline method: {baseline_method}")
    
    def _calculate_average_baseline(self, metrics: List[ModelPerformanceMetrics]) -> ModelPerformanceMetrics:
        """Calculate baseline using average of historical metrics."""
        if not metrics:
            raise ValueError("No metrics provided for baseline calculation")
        
        # Get the first metric to determine task type and structure
        first_metric = metrics[0]
        task_type = first_metric.task_type
        sample_size = sum(m.sample_size for m in metrics) // len(metrics)
        
        # Calculate averages for each metric
        baseline_data = {
            "task_type": task_type,
            "sample_size": sample_size,
        }
        
        # Classification metrics
        if task_type in [ModelTask.BINARY_CLASSIFICATION, ModelTask.MULTICLASS_CLASSIFICATION]:
            baseline_data.update({
                "accuracy": self._safe_average([m.accuracy for m in metrics if m.accuracy is not None]),
                "precision": self._safe_average([m.precision for m in metrics if m.precision is not None]),
                "recall": self._safe_average([m.recall for m in metrics if m.recall is not None]),
                "f1_score": self._safe_average([m.f1_score for m in metrics if m.f1_score is not None]),
                "roc_auc": self._safe_average([m.roc_auc for m in metrics if m.roc_auc is not None]),
                "pr_auc": self._safe_average([m.pr_auc for m in metrics if m.pr_auc is not None]),
            })
        
        # Regression metrics
        if task_type in [ModelTask.REGRESSION, ModelTask.TIME_SERIES]:
            baseline_data.update({
                "mse": self._safe_average([m.mse for m in metrics if m.mse is not None]),
                "rmse": self._safe_average([m.rmse for m in metrics if m.rmse is not None]),
                "mae": self._safe_average([m.mae for m in metrics if m.mae is not None]),
                "r2_score": self._safe_average([m.r2_score for m in metrics if m.r2_score is not None]),
            })
        
        # Common metrics
        baseline_data.update({
            "prediction_time_seconds": self._safe_average([
                m.prediction_time_seconds for m in metrics 
                if m.prediction_time_seconds is not None
            ]),
            "prediction_confidence": self._safe_average([
                m.prediction_confidence for m in metrics 
                if m.prediction_confidence is not None
            ]),
            "prediction_stability": self._safe_average([
                m.prediction_stability for m in metrics 
                if m.prediction_stability is not None
            ]),
        })
        
        # Remove None values
        baseline_data = {k: v for k, v in baseline_data.items() if v is not None}
        
        return ModelPerformanceMetrics(**baseline_data)
    
    def _calculate_best_performance_baseline(self, metrics: List[ModelPerformanceMetrics]) -> ModelPerformanceMetrics:
        """Calculate baseline using best historical performance."""
        if not metrics:
            raise ValueError("No metrics provided for baseline calculation")
        
        # Find the metric with the best primary metric
        best_metric = max(metrics, key=lambda m: m.get_primary_metric() or 0)
        return best_metric
    
    def _calculate_median_baseline(self, metrics: List[ModelPerformanceMetrics]) -> ModelPerformanceMetrics:
        """Calculate baseline using median of historical metrics."""
        if not metrics:
            raise ValueError("No metrics provided for baseline calculation")
        
        first_metric = metrics[0]
        task_type = first_metric.task_type
        sample_size = sum(m.sample_size for m in metrics) // len(metrics)
        
        baseline_data = {
            "task_type": task_type,
            "sample_size": sample_size,
        }
        
        # Calculate medians for each metric
        if task_type in [ModelTask.BINARY_CLASSIFICATION, ModelTask.MULTICLASS_CLASSIFICATION]:
            baseline_data.update({
                "accuracy": self._safe_median([m.accuracy for m in metrics if m.accuracy is not None]),
                "precision": self._safe_median([m.precision for m in metrics if m.precision is not None]),
                "recall": self._safe_median([m.recall for m in metrics if m.recall is not None]),
                "f1_score": self._safe_median([m.f1_score for m in metrics if m.f1_score is not None]),
                "roc_auc": self._safe_median([m.roc_auc for m in metrics if m.roc_auc is not None]),
            })
        
        if task_type in [ModelTask.REGRESSION, ModelTask.TIME_SERIES]:
            baseline_data.update({
                "mse": self._safe_median([m.mse for m in metrics if m.mse is not None]),
                "rmse": self._safe_median([m.rmse for m in metrics if m.rmse is not None]),
                "mae": self._safe_median([m.mae for m in metrics if m.mae is not None]),
                "r2_score": self._safe_median([m.r2_score for m in metrics if m.r2_score is not None]),
            })
        
        # Remove None values
        baseline_data = {k: v for k, v in baseline_data.items() if v is not None}
        
        return ModelPerformanceMetrics(**baseline_data)
    
    def _calculate_weighted_average_baseline(self, metrics: List[ModelPerformanceMetrics]) -> ModelPerformanceMetrics:
        """Calculate baseline using weighted average (more recent metrics have higher weight)."""
        if not metrics:
            raise ValueError("No metrics provided for baseline calculation")
        
        # Create weights that increase linearly with recency
        weights = [i + 1 for i in range(len(metrics))]
        total_weight = sum(weights)
        
        first_metric = metrics[0]
        task_type = first_metric.task_type
        sample_size = sum(m.sample_size * w for m, w in zip(metrics, weights)) // total_weight
        
        baseline_data = {
            "task_type": task_type,
            "sample_size": sample_size,
        }
        
        # Calculate weighted averages
        if task_type in [ModelTask.BINARY_CLASSIFICATION, ModelTask.MULTICLASS_CLASSIFICATION]:
            baseline_data.update({
                "accuracy": self._weighted_average([m.accuracy for m in metrics], weights),
                "precision": self._weighted_average([m.precision for m in metrics], weights),
                "recall": self._weighted_average([m.recall for m in metrics], weights),
                "f1_score": self._weighted_average([m.f1_score for m in metrics], weights),
                "roc_auc": self._weighted_average([m.roc_auc for m in metrics], weights),
            })
        
        if task_type in [ModelTask.REGRESSION, ModelTask.TIME_SERIES]:
            baseline_data.update({
                "mse": self._weighted_average([m.mse for m in metrics], weights),
                "rmse": self._weighted_average([m.rmse for m in metrics], weights),
                "mae": self._weighted_average([m.mae for m in metrics], weights),
                "r2_score": self._weighted_average([m.r2_score for m in metrics], weights),
            })
        
        # Remove None values
        baseline_data = {k: v for k, v in baseline_data.items() if v is not None}
        
        return ModelPerformanceMetrics(**baseline_data)
    
    def compare_against_baseline(
        self,
        current_metrics: ModelPerformanceMetrics,
        baseline_metrics: ModelPerformanceMetrics,
        degradation_thresholds: Dict[DegradationMetricType, float],
    ) -> List[Dict[str, any]]:
        """Compare current metrics against baseline and detect degradations.
        
        Args:
            current_metrics: Current performance metrics
            baseline_metrics: Baseline performance metrics
            degradation_thresholds: Thresholds for each degradation metric type
        
        Returns:
            List of detected degradations with details
        """
        if current_metrics.task_type != baseline_metrics.task_type:
            raise ValueError("Current and baseline metrics must have same task type")
        
        degradations = []
        
        for metric_type, threshold_pct in degradation_thresholds.items():
            baseline_value = self._extract_metric_value(baseline_metrics, metric_type)
            current_value = self._extract_metric_value(current_metrics, metric_type)
            
            if baseline_value is None or current_value is None:
                continue
            
            # Calculate degradation percentage
            degradation_pct = self._calculate_degradation_percentage(
                current_value, baseline_value, metric_type
            )
            
            # Check if degradation exceeds threshold
            if degradation_pct > threshold_pct:
                severity = self._determine_severity(degradation_pct)
                
                degradations.append({
                    "metric_type": metric_type,
                    "baseline_value": baseline_value,
                    "current_value": current_value,
                    "degradation_percentage": degradation_pct,
                    "threshold_percentage": threshold_pct,
                    "severity": severity,
                    "is_significant": self._is_statistically_significant(
                        current_value, baseline_value, metric_type
                    ),
                })
        
        return degradations
    
    def calculate_baseline_stability(
        self,
        historical_metrics: List[ModelPerformanceMetrics],
        metric_type: DegradationMetricType,
    ) -> Dict[str, float]:
        """Calculate stability metrics for a baseline.
        
        Args:
            historical_metrics: Historical performance metrics
            metric_type: Type of metric to analyze
        
        Returns:
            Dictionary with stability metrics
        """
        values = [
            self._extract_metric_value(metric, metric_type)
            for metric in historical_metrics
        ]
        values = [v for v in values if v is not None]
        
        if len(values) < 2:
            return {"stability_score": 0.0, "coefficient_of_variation": 0.0}
        
        mean_val = mean(values)
        std_val = stdev(values)
        
        coefficient_of_variation = std_val / mean_val if mean_val != 0 else 0
        stability_score = max(0, 1 - coefficient_of_variation)
        
        return {
            "stability_score": stability_score,
            "coefficient_of_variation": coefficient_of_variation,
            "mean_value": mean_val,
            "std_value": std_val,
            "min_value": min(values),
            "max_value": max(values),
            "sample_count": len(values),
        }
    
    def suggest_degradation_thresholds(
        self,
        historical_metrics: List[ModelPerformanceMetrics],
        task_type: ModelTask,
    ) -> Dict[DegradationMetricType, float]:
        """Suggest degradation thresholds based on historical performance variability.
        
        Args:
            historical_metrics: Historical performance metrics
            task_type: Type of ML task
        
        Returns:
            Dictionary of suggested thresholds
        """
        thresholds = {}
        
        # Get relevant metrics for task type
        relevant_metrics = self._get_relevant_metrics_for_task(task_type)
        
        for metric_type in relevant_metrics:
            stability = self.calculate_baseline_stability(historical_metrics, metric_type)
            
            # Base threshold on coefficient of variation
            cv = stability["coefficient_of_variation"]
            
            if cv < 0.05:  # Very stable
                threshold = 5.0
            elif cv < 0.1:  # Moderately stable
                threshold = 10.0
            elif cv < 0.2:  # Somewhat unstable
                threshold = 15.0
            else:  # Highly variable
                threshold = 25.0
            
            thresholds[metric_type] = threshold
        
        return thresholds
    
    def _extract_metric_value(self, metrics: ModelPerformanceMetrics, metric_type: DegradationMetricType) -> Optional[float]:
        """Extract specific metric value from performance metrics."""
        metric_mapping = {
            DegradationMetricType.ACCURACY_DROP: metrics.accuracy,
            DegradationMetricType.PRECISION_DROP: metrics.precision,
            DegradationMetricType.RECALL_DROP: metrics.recall,
            DegradationMetricType.F1_SCORE_DROP: metrics.f1_score,
            DegradationMetricType.AUC_DROP: metrics.roc_auc,
            DegradationMetricType.MSE_INCREASE: metrics.mse,
            DegradationMetricType.RMSE_INCREASE: metrics.rmse,
            DegradationMetricType.MAE_INCREASE: metrics.mae,
            DegradationMetricType.R2_SCORE_DROP: metrics.r2_score,
            DegradationMetricType.PREDICTION_TIME_INCREASE: metrics.prediction_time_seconds,
            DegradationMetricType.CONFIDENCE_DROP: metrics.prediction_confidence,
            DegradationMetricType.STABILITY_DECREASE: metrics.prediction_stability,
        }
        
        return metric_mapping.get(metric_type)
    
    def _calculate_degradation_percentage(self, current: float, baseline: float, metric_type: DegradationMetricType) -> float:
        """Calculate degradation percentage."""
        if baseline == 0:
            return 0.0
        
        # For metrics where higher is better
        if metric_type in [
            DegradationMetricType.ACCURACY_DROP,
            DegradationMetricType.PRECISION_DROP,
            DegradationMetricType.RECALL_DROP,
            DegradationMetricType.F1_SCORE_DROP,
            DegradationMetricType.AUC_DROP,
            DegradationMetricType.R2_SCORE_DROP,
            DegradationMetricType.CONFIDENCE_DROP,
            DegradationMetricType.STABILITY_DECREASE,
        ]:
            if current < baseline:
                return ((baseline - current) / baseline) * 100
        else:
            # For metrics where lower is better
            if current > baseline:
                return ((current - baseline) / baseline) * 100
        
        return 0.0
    
    def _determine_severity(self, degradation_percentage: float) -> DegradationSeverity:
        """Determine severity based on degradation percentage."""
        if degradation_percentage >= 50:
            return DegradationSeverity.CRITICAL
        elif degradation_percentage >= 30:
            return DegradationSeverity.MAJOR
        elif degradation_percentage >= 15:
            return DegradationSeverity.MODERATE
        else:
            return DegradationSeverity.MINOR
    
    def _is_statistically_significant(self, current: float, baseline: float, metric_type: DegradationMetricType) -> bool:
        """Check if the difference is statistically significant."""
        # Simplified significance test - in production, this would use proper statistical tests
        difference = abs(current - baseline)
        relative_difference = difference / baseline if baseline != 0 else 0
        
        # Consider significant if relative difference is > 10%
        return relative_difference > 0.1
    
    def _get_relevant_metrics_for_task(self, task_type: ModelTask) -> List[DegradationMetricType]:
        """Get relevant degradation metrics for a task type."""
        if task_type in [ModelTask.BINARY_CLASSIFICATION, ModelTask.MULTICLASS_CLASSIFICATION]:
            return [
                DegradationMetricType.ACCURACY_DROP,
                DegradationMetricType.PRECISION_DROP,
                DegradationMetricType.RECALL_DROP,
                DegradationMetricType.F1_SCORE_DROP,
                DegradationMetricType.AUC_DROP,
                DegradationMetricType.CONFIDENCE_DROP,
                DegradationMetricType.PREDICTION_TIME_INCREASE,
            ]
        elif task_type in [ModelTask.REGRESSION, ModelTask.TIME_SERIES]:
            return [
                DegradationMetricType.MSE_INCREASE,
                DegradationMetricType.RMSE_INCREASE,
                DegradationMetricType.MAE_INCREASE,
                DegradationMetricType.R2_SCORE_DROP,
                DegradationMetricType.CONFIDENCE_DROP,
                DegradationMetricType.PREDICTION_TIME_INCREASE,
            ]
        else:
            return [
                DegradationMetricType.CONFIDENCE_DROP,
                DegradationMetricType.PREDICTION_TIME_INCREASE,
                DegradationMetricType.STABILITY_DECREASE,
            ]
    
    def _safe_average(self, values: List[Optional[float]]) -> Optional[float]:
        """Calculate average safely handling None values."""
        filtered_values = [v for v in values if v is not None]
        return mean(filtered_values) if filtered_values else None
    
    def _safe_median(self, values: List[Optional[float]]) -> Optional[float]:
        """Calculate median safely handling None values."""
        filtered_values = [v for v in values if v is not None]
        if not filtered_values:
            return None
        
        sorted_values = sorted(filtered_values)
        n = len(sorted_values)
        
        if n % 2 == 0:
            return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            return sorted_values[n//2]
    
    def _weighted_average(self, values: List[Optional[float]], weights: List[float]) -> Optional[float]:
        """Calculate weighted average safely handling None values."""
        filtered_pairs = [(v, w) for v, w in zip(values, weights) if v is not None]
        
        if not filtered_pairs:
            return None
        
        total_weighted = sum(v * w for v, w in filtered_pairs)
        total_weight = sum(w for v, w in filtered_pairs)
        
        return total_weighted / total_weight if total_weight > 0 else None