"""Domain service for model performance degradation detection.

This module implements the domain service for D-003: Model Performance Degradation Detection.
It provides business logic for detecting when model performance drops below acceptable
thresholds and determining appropriate actions.
"""

from __future__ import annotations

import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from pynomaly.domain.entities.performance_degradation import (
    DegradationSeverity,
    DegradationStatus,
    PerformanceBaseline,
    PerformanceDegradationEvent,
    PerformanceMetric,
    PerformanceMetricType,
    PerformanceMonitoringConfiguration,
    PerformanceThreshold,
)
from pynomaly.domain.exceptions import ValidationError

logger = logging.getLogger(__name__)


class PerformanceDegradationService:
    """Domain service for performance degradation detection and analysis."""
    
    def __init__(self):
        """Initialize the performance degradation service."""
        self.statistical_significance_threshold = 0.05  # p-value threshold
        self.min_baseline_samples = 30  # Minimum samples for reliable baseline
        self.trend_analysis_window = timedelta(days=14)  # Window for trend analysis
        
    def evaluate_performance_degradation(
        self,
        current_metrics: List[PerformanceMetric],
        baseline: PerformanceBaseline,
        configuration: PerformanceMonitoringConfiguration,
        historical_metrics: Optional[List[PerformanceMetric]] = None
    ) -> List[PerformanceDegradationEvent]:
        """Evaluate performance degradation against baseline and thresholds.
        
        Args:
            current_metrics: Recent performance metrics
            baseline: Performance baseline for comparison
            configuration: Monitoring configuration with thresholds
            historical_metrics: Optional historical metrics for trend analysis
            
        Returns:
            List of degradation events detected
            
        Raises:
            ValidationError: If inputs are invalid
        """
        self._validate_inputs(current_metrics, baseline, configuration)
        
        degradation_events = []
        
        # Group metrics by type
        metrics_by_type = self._group_metrics_by_type(current_metrics)
        
        # Check each metric type against thresholds
        for metric_type, metrics in metrics_by_type.items():
            if not metrics:
                continue
                
            # Get relevant thresholds for this metric type
            thresholds = [
                t for t in configuration.performance_thresholds 
                if t.metric_type == metric_type
            ]
            
            if not thresholds:
                continue
                
            # Get current value (most recent metric)
            current_metric = max(metrics, key=lambda m: m.timestamp)
            current_value = current_metric.value
            
            # Get baseline value
            baseline_value = baseline.get_baseline_value(metric_type)
            if baseline_value is None:
                logger.warning(f"No baseline value for {metric_type.value}")
                continue
            
            # Check each threshold
            for threshold in thresholds:
                if self._is_threshold_violated(current_value, threshold):
                    # Check if degradation is statistically significant
                    if self._is_statistically_significant(
                        current_value, 
                        baseline_value, 
                        baseline, 
                        metric_type
                    ):
                        # Create degradation event
                        event = self._create_degradation_event(
                            current_metric,
                            threshold,
                            baseline_value,
                            current_value,
                            configuration,
                            historical_metrics
                        )
                        degradation_events.append(event)
                        
                        logger.warning(
                            f"Performance degradation detected for {metric_type.value}: "
                            f"current={current_value:.4f}, baseline={baseline_value:.4f}, "
                            f"threshold={threshold.threshold_value:.4f}, "
                            f"severity={threshold.severity.value}"
                        )
        
        return degradation_events
    
    def calculate_baseline_statistics(
        self,
        metrics: List[PerformanceMetric],
        confidence_level: float = 0.95
    ) -> Dict[PerformanceMetricType, Dict[str, float]]:
        """Calculate baseline statistics from historical metrics.
        
        Args:
            metrics: Historical performance metrics
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with statistics for each metric type
        """
        if not metrics:
            return {}
        
        # Group metrics by type
        metrics_by_type = self._group_metrics_by_type(metrics)
        
        statistics_by_type = {}
        
        for metric_type, type_metrics in metrics_by_type.items():
            if len(type_metrics) < self.min_baseline_samples:
                logger.warning(
                    f"Insufficient samples for {metric_type.value}: "
                    f"got {len(type_metrics)}, need {self.min_baseline_samples}"
                )
                continue
            
            values = [m.value for m in type_metrics]
            
            # Calculate basic statistics
            mean_value = statistics.mean(values)
            std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
            median_value = statistics.median(values)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(
                values, confidence_level
            )
            
            # Calculate percentiles
            percentiles = {
                'p10': self._percentile(values, 10),
                'p25': self._percentile(values, 25),
                'p75': self._percentile(values, 75),
                'p90': self._percentile(values, 90),
                'p95': self._percentile(values, 95),
                'p99': self._percentile(values, 99),
            }
            
            statistics_by_type[metric_type] = {
                'mean': mean_value,
                'std_dev': std_dev,
                'median': median_value,
                'min': min(values),
                'max': max(values),
                'confidence_interval_lower': confidence_interval[0],
                'confidence_interval_upper': confidence_interval[1],
                'sample_count': len(values),
                **percentiles
            }
        
        return statistics_by_type
    
    def analyze_performance_trend(
        self,
        metrics: List[PerformanceMetric],
        metric_type: PerformanceMetricType,
        window: Optional[timedelta] = None
    ) -> Dict[str, float]:
        """Analyze performance trend for a specific metric type.
        
        Args:
            metrics: Historical performance metrics
            metric_type: Type of metric to analyze
            window: Time window for analysis (defaults to service window)
            
        Returns:
            Dictionary with trend analysis results
        """
        if window is None:
            window = self.trend_analysis_window
        
        # Filter metrics by type and time window
        cutoff_time = datetime.utcnow() - window
        relevant_metrics = [
            m for m in metrics 
            if m.metric_type == metric_type and m.timestamp >= cutoff_time
        ]
        
        if len(relevant_metrics) < 2:
            return {'trend': 'insufficient_data', 'slope': 0.0, 'r_squared': 0.0}
        
        # Sort by timestamp
        relevant_metrics.sort(key=lambda m: m.timestamp)
        
        # Calculate trend using linear regression
        x_values = [(m.timestamp - relevant_metrics[0].timestamp).total_seconds() 
                   for m in relevant_metrics]
        y_values = [m.value for m in relevant_metrics]
        
        slope, intercept, r_squared = self._linear_regression(x_values, y_values)
        
        # Determine trend direction
        if abs(slope) < 1e-10:
            trend = 'stable'
        elif slope > 0:
            trend = 'improving'
        else:
            trend = 'declining'
        
        # Calculate rate of change (per day)
        slope_per_day = slope * 86400  # Convert from per second to per day
        
        return {
            'trend': trend,
            'slope': slope,
            'slope_per_day': slope_per_day,
            'intercept': intercept,
            'r_squared': r_squared,
            'sample_count': len(relevant_metrics),
            'time_span_days': window.days,
            'recent_value': relevant_metrics[-1].value,
            'initial_value': relevant_metrics[0].value,
            'total_change': relevant_metrics[-1].value - relevant_metrics[0].value,
            'percentage_change': (
                (relevant_metrics[-1].value - relevant_metrics[0].value) / 
                relevant_metrics[0].value * 100
                if relevant_metrics[0].value != 0 else 0.0
            )
        }
    
    def determine_degradation_severity(
        self,
        current_value: float,
        baseline_value: float,
        metric_type: PerformanceMetricType,
        thresholds: List[PerformanceThreshold]
    ) -> DegradationSeverity:
        """Determine the severity of performance degradation.
        
        Args:
            current_value: Current metric value
            baseline_value: Baseline metric value
            metric_type: Type of metric
            thresholds: List of thresholds to check
            
        Returns:
            Severity level of degradation
        """
        # Sort thresholds by severity (most severe first)
        severity_order = [
            DegradationSeverity.CRITICAL,
            DegradationSeverity.MAJOR,
            DegradationSeverity.MODERATE,
            DegradationSeverity.MINOR,
            DegradationSeverity.NONE
        ]
        
        relevant_thresholds = [
            t for t in thresholds 
            if t.metric_type == metric_type
        ]
        
        # Check thresholds in order of severity
        for severity in severity_order:
            severity_thresholds = [
                t for t in relevant_thresholds 
                if t.severity == severity
            ]
            
            for threshold in severity_thresholds:
                if self._is_threshold_violated(current_value, threshold):
                    return severity
        
        return DegradationSeverity.NONE
    
    def create_monitoring_configuration(
        self,
        detector_id: str,
        metric_thresholds: Dict[PerformanceMetricType, List[Tuple[float, str, DegradationSeverity]]],
        monitoring_interval: timedelta = timedelta(hours=1),
        evaluation_window: timedelta = timedelta(days=7),
        baseline_window: timedelta = timedelta(days=30)
    ) -> PerformanceMonitoringConfiguration:
        """Create a monitoring configuration with recommended thresholds.
        
        Args:
            detector_id: ID of the detector to monitor
            metric_thresholds: Dictionary mapping metric types to threshold definitions
            monitoring_interval: How often to check performance
            evaluation_window: Window for performance evaluation
            baseline_window: Window for baseline calculation
            
        Returns:
            Performance monitoring configuration
        """
        thresholds = []
        
        for metric_type, threshold_definitions in metric_thresholds.items():
            for threshold_value, comparison_operator, severity in threshold_definitions:
                threshold = PerformanceThreshold(
                    metric_type=metric_type,
                    threshold_value=threshold_value,
                    comparison_operator=comparison_operator,
                    severity=severity,
                    description=f"{metric_type.value} {comparison_operator} {threshold_value}"
                )
                thresholds.append(threshold)
        
        return PerformanceMonitoringConfiguration(
            detector_id=detector_id,
            monitoring_interval=monitoring_interval,
            evaluation_window=evaluation_window,
            baseline_window=baseline_window,
            performance_thresholds=thresholds,
            enabled=True,
            alert_on_degradation=True,
            auto_trigger_retraining=True
        )
    
    def _validate_inputs(
        self,
        current_metrics: List[PerformanceMetric],
        baseline: PerformanceBaseline,
        configuration: PerformanceMonitoringConfiguration
    ) -> None:
        """Validate inputs for degradation evaluation."""
        if not current_metrics:
            raise ValidationError("Current metrics are required")
        
        if not baseline.is_valid:
            raise ValidationError("Baseline is not valid")
        
        if not configuration.enabled:
            raise ValidationError("Monitoring configuration is disabled")
        
        if not configuration.performance_thresholds:
            raise ValidationError("No performance thresholds configured")
        
        # Check if current metrics are recent enough
        oldest_allowed = datetime.utcnow() - configuration.evaluation_window
        recent_metrics = [m for m in current_metrics if m.timestamp >= oldest_allowed]
        
        if len(recent_metrics) < configuration.min_samples_required:
            raise ValidationError(
                f"Insufficient recent metrics: got {len(recent_metrics)}, "
                f"need {configuration.min_samples_required}"
            )
    
    def _group_metrics_by_type(
        self, 
        metrics: List[PerformanceMetric]
    ) -> Dict[PerformanceMetricType, List[PerformanceMetric]]:
        """Group metrics by type."""
        groups = {}
        for metric in metrics:
            if metric.metric_type not in groups:
                groups[metric.metric_type] = []
            groups[metric.metric_type].append(metric)
        return groups
    
    def _is_threshold_violated(
        self, 
        value: float, 
        threshold: PerformanceThreshold
    ) -> bool:
        """Check if a value violates a threshold."""
        return not threshold.evaluate(value)
    
    def _is_statistically_significant(
        self,
        current_value: float,
        baseline_value: float,
        baseline: PerformanceBaseline,
        metric_type: PerformanceMetricType
    ) -> bool:
        """Check if degradation is statistically significant."""
        # Get confidence interval from baseline
        confidence_interval = baseline.get_confidence_interval(metric_type)
        
        if confidence_interval is None:
            # No statistical test available, assume significant if outside baseline
            return abs(current_value - baseline_value) > abs(baseline_value * 0.05)
        
        # Check if current value is outside confidence interval
        lower, upper = confidence_interval
        return not (lower <= current_value <= upper)
    
    def _create_degradation_event(
        self,
        current_metric: PerformanceMetric,
        violated_threshold: PerformanceThreshold,
        baseline_value: float,
        current_value: float,
        configuration: PerformanceMonitoringConfiguration,
        historical_metrics: Optional[List[PerformanceMetric]] = None
    ) -> PerformanceDegradationEvent:
        """Create a degradation event."""
        # Calculate degradation percentage
        degradation_percentage = None
        if baseline_value != 0:
            degradation_percentage = (
                (current_value - baseline_value) / baseline_value * 100
            )
        
        # Analyze trend if historical metrics available
        trend_analysis = None
        if historical_metrics:
            trend_analysis = self.analyze_performance_trend(
                historical_metrics,
                current_metric.metric_type,
                configuration.evaluation_window
            )
        
        event = PerformanceDegradationEvent(
            detector_id=current_metric.detector_id,
            severity=violated_threshold.severity,
            status=DegradationStatus.DEGRADED,
            violated_threshold=violated_threshold,
            trigger_metric=current_metric,
            baseline_value=baseline_value,
            current_value=current_value,
            degradation_percentage=degradation_percentage,
            evaluation_window=configuration.evaluation_window,
            baseline_window=configuration.baseline_window,
            historical_metrics=historical_metrics or [],
            trend_analysis=trend_analysis
        )
        
        return event
    
    def _calculate_confidence_interval(
        self, 
        values: List[float], 
        confidence_level: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for a list of values."""
        if len(values) < 2:
            mean_val = values[0] if values else 0.0
            return (mean_val, mean_val)
        
        mean_val = statistics.mean(values)
        std_dev = statistics.stdev(values)
        
        # Use t-distribution for small samples
        if len(values) < 30:
            # Simplified t-distribution (would use scipy.stats.t in production)
            t_value = 2.0  # Approximate t-value for 95% confidence
        else:
            # Normal distribution
            t_value = 1.96  # Z-value for 95% confidence
        
        margin_of_error = t_value * (std_dev / (len(values) ** 0.5))
        
        return (mean_val - margin_of_error, mean_val + margin_of_error)
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            weight = index - lower_index
            
            if upper_index >= len(sorted_values):
                return sorted_values[lower_index]
            
            return (sorted_values[lower_index] * (1 - weight) + 
                   sorted_values[upper_index] * weight)
    
    def _linear_regression(
        self, 
        x_values: List[float], 
        y_values: List[float]
    ) -> Tuple[float, float, float]:
        """Perform simple linear regression."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0, 0.0, 0.0
        
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x_squared = sum(x * x for x in x_values)
        sum_y_squared = sum(y * y for y in y_values)
        
        # Calculate slope and intercept
        denominator = n * sum_x_squared - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0, statistics.mean(y_values), 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        # Calculate R-squared
        y_mean = sum_y / n
        ss_tot = sum((y - y_mean) ** 2 for y in y_values)
        ss_res = sum((y - (slope * x + intercept)) ** 2 
                    for x, y in zip(x_values, y_values))
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return slope, intercept, r_squared
