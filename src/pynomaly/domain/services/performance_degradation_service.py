"""Performance degradation detection service."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np
from scipy import stats

from packages.data_science.domain.value_objects.model_performance_metrics import (
    ModelPerformanceMetrics, 
    ModelTask
)
from pynomaly.domain.value_objects.performance_degradation_metrics import (
    DegradationAlert,
    DegradationReport,
    DegradationSeverity,
    DegradationType,
    MetricThreshold,
    PerformanceBaseline,
    PerformanceDegradation,
)
from pynomaly.shared.protocols.repository_protocol import ModelRepositoryProtocol

logger = logging.getLogger(__name__)


class PerformanceDegradationService:
    """Service for detecting and monitoring model performance degradation."""

    def __init__(
        self,
        model_repository: ModelRepositoryProtocol,
        performance_repository: Any,  # PerformanceRepositoryProtocol when implemented
        alert_service: Any,  # AlertService when implemented
    ):
        """Initialize the performance degradation service.
        
        Args:
            model_repository: Repository for model data
            performance_repository: Repository for performance metrics
            alert_service: Service for sending alerts
        """
        self.model_repository = model_repository
        self.performance_repository = performance_repository
        self.alert_service = alert_service
        self._default_thresholds = self._create_default_thresholds()

    def _create_default_thresholds(self) -> Dict[str, MetricThreshold]:
        """Create default threshold configurations for common metrics."""
        return {
            # Classification metrics (decrease is bad)
            "accuracy": MetricThreshold(
                metric_name="accuracy",
                warning_threshold=5.0,
                critical_threshold=10.0,
                threshold_type="percentage",
                direction="decrease",
                min_samples=10
            ),
            "precision": MetricThreshold(
                metric_name="precision",
                warning_threshold=5.0,
                critical_threshold=10.0,
                threshold_type="percentage",
                direction="decrease",
                min_samples=10
            ),
            "recall": MetricThreshold(
                metric_name="recall",
                warning_threshold=5.0,
                critical_threshold=10.0,
                threshold_type="percentage",
                direction="decrease",
                min_samples=10
            ),
            "f1_score": MetricThreshold(
                metric_name="f1_score",
                warning_threshold=5.0,
                critical_threshold=10.0,
                threshold_type="percentage",
                direction="decrease",
                min_samples=10
            ),
            "roc_auc": MetricThreshold(
                metric_name="roc_auc",
                warning_threshold=3.0,
                critical_threshold=7.0,
                threshold_type="percentage",
                direction="decrease",
                min_samples=10
            ),
            "pr_auc": MetricThreshold(
                metric_name="pr_auc",
                warning_threshold=5.0,
                critical_threshold=10.0,
                threshold_type="percentage",
                direction="decrease",
                min_samples=10
            ),
            # Regression metrics (increase is bad for error metrics)
            "mse": MetricThreshold(
                metric_name="mse",
                warning_threshold=15.0,
                critical_threshold=30.0,
                threshold_type="percentage",
                direction="increase",
                min_samples=10
            ),
            "rmse": MetricThreshold(
                metric_name="rmse",
                warning_threshold=15.0,
                critical_threshold=30.0,
                threshold_type="percentage",
                direction="increase",
                min_samples=10
            ),
            "mae": MetricThreshold(
                metric_name="mae",
                warning_threshold=15.0,
                critical_threshold=30.0,
                threshold_type="percentage",
                direction="increase",
                min_samples=10
            ),
            "mape": MetricThreshold(
                metric_name="mape",
                warning_threshold=20.0,
                critical_threshold=40.0,
                threshold_type="percentage",
                direction="increase",
                min_samples=10
            ),
            "r2_score": MetricThreshold(
                metric_name="r2_score",
                warning_threshold=5.0,
                critical_threshold=15.0,
                threshold_type="percentage",
                direction="decrease",
                min_samples=10
            ),
            # Efficiency metrics
            "training_time_seconds": MetricThreshold(
                metric_name="training_time_seconds",
                warning_threshold=25.0,
                critical_threshold=50.0,
                threshold_type="percentage",
                direction="increase",
                min_samples=5
            ),
            "prediction_time_seconds": MetricThreshold(
                metric_name="prediction_time_seconds",
                warning_threshold=30.0,
                critical_threshold=60.0,
                threshold_type="percentage",
                direction="increase",
                min_samples=5
            ),
            "memory_usage_mb": MetricThreshold(
                metric_name="memory_usage_mb",
                warning_threshold=25.0,
                critical_threshold=50.0,
                threshold_type="percentage",
                direction="increase",
                min_samples=5
            ),
            # Stability metrics
            "prediction_stability": MetricThreshold(
                metric_name="prediction_stability",
                warning_threshold=10.0,
                critical_threshold=20.0,
                threshold_type="percentage",
                direction="decrease",
                min_samples=10
            )
        }

    async def detect_degradation(
        self,
        model_id: UUID,
        current_metrics: ModelPerformanceMetrics,
        custom_thresholds: Optional[Dict[str, MetricThreshold]] = None,
        lookback_days: int = 30,
        detection_method: str = "baseline_comparison"
    ) -> List[PerformanceDegradation]:
        """Detect performance degradation for a model.
        
        Args:
            model_id: ID of the model to check
            current_metrics: Current performance metrics
            custom_thresholds: Custom threshold configurations
            lookback_days: Days to look back for baseline
            detection_method: Method for detection
            
        Returns:
            List of detected degradations
        """
        degradations = []
        
        # Get performance history
        performance_history = await self.performance_repository.get_model_performance_history(
            model_id=model_id,
            start_date=datetime.utcnow() - timedelta(days=lookback_days),
            end_date=datetime.utcnow()
        )
        
        if not performance_history:
            logger.warning(f"No performance history found for model {model_id}")
            return degradations
        
        # Use custom thresholds or defaults
        thresholds = custom_thresholds or self._default_thresholds
        
        # Get baselines for each metric
        baselines = await self._establish_baselines(performance_history)
        
        # Check each metric for degradation
        metrics_dict = current_metrics.dict()
        for metric_name, current_value in metrics_dict.items():
            if current_value is None or metric_name not in thresholds:
                continue
                
            baseline = baselines.get(metric_name)
            if not baseline:
                continue
                
            threshold = thresholds[metric_name]
            
            # Check for degradation
            is_degraded, severity = baseline.is_degraded(current_value, threshold)
            
            if is_degraded:
                degradation = self._create_degradation(
                    metric_name=metric_name,
                    current_value=current_value,
                    baseline=baseline,
                    threshold=threshold,
                    severity=severity,
                    detection_method=detection_method,
                    samples_used=baseline.sample_count
                )
                degradations.append(degradation)
        
        return degradations

    async def monitor_continuous_degradation(
        self,
        model_id: UUID,
        monitoring_window_hours: int = 24,
        min_samples: int = 10
    ) -> List[PerformanceDegradation]:
        """Monitor for continuous degradation over a time window.
        
        Args:
            model_id: ID of the model to monitor
            monitoring_window_hours: Hours to monitor for degradation
            min_samples: Minimum samples required for detection
            
        Returns:
            List of detected degradations
        """
        degradations = []
        
        # Get recent performance data
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=monitoring_window_hours)
        
        recent_metrics = await self.performance_repository.get_model_performance_history(
            model_id=model_id,
            start_date=start_time,
            end_date=end_time
        )
        
        if len(recent_metrics) < min_samples:
            logger.info(f"Insufficient samples ({len(recent_metrics)}) for continuous monitoring")
            return degradations
        
        # Get baseline from older data
        baseline_start = start_time - timedelta(days=30)
        baseline_end = start_time
        
        baseline_metrics = await self.performance_repository.get_model_performance_history(
            model_id=model_id,
            start_date=baseline_start,
            end_date=baseline_end
        )
        
        if not baseline_metrics:
            logger.warning(f"No baseline data found for model {model_id}")
            return degradations
        
        # Analyze trends
        degradations.extend(await self._analyze_performance_trends(
            recent_metrics=recent_metrics,
            baseline_metrics=baseline_metrics,
            model_id=model_id
        ))
        
        return degradations

    async def _establish_baselines(
        self, 
        performance_history: List[ModelPerformanceMetrics]
    ) -> Dict[str, PerformanceBaseline]:
        """Establish baselines from performance history.
        
        Args:
            performance_history: Historical performance metrics
            
        Returns:
            Dictionary of metric baselines
        """
        baselines = {}
        
        # Group metrics by name
        metric_groups = {}
        for metrics in performance_history:
            metrics_dict = metrics.dict()
            for metric_name, value in metrics_dict.items():
                if value is not None:
                    if metric_name not in metric_groups:
                        metric_groups[metric_name] = []
                    metric_groups[metric_name].append(value)
        
        # Create baselines
        for metric_name, values in metric_groups.items():
            if len(values) >= 5:  # Minimum samples for baseline
                baseline = self._create_baseline(metric_name, values)
                baselines[metric_name] = baseline
        
        return baselines

    def _create_baseline(self, metric_name: str, values: List[float]) -> PerformanceBaseline:
        """Create baseline from metric values.
        
        Args:
            metric_name: Name of the metric
            values: Historical values
            
        Returns:
            Performance baseline
        """
        np_values = np.array(values)
        
        # Remove outliers using IQR method
        q1 = np.percentile(np_values, 25)
        q3 = np.percentile(np_values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        clean_values = np_values[(np_values >= lower_bound) & (np_values <= upper_bound)]
        
        if len(clean_values) < 3:
            clean_values = np_values  # Use all values if too few after cleaning
        
        mean_val = float(np.mean(clean_values))
        std_val = float(np.std(clean_values))
        
        # Calculate confidence interval
        confidence_margin = 1.96 * (std_val / np.sqrt(len(clean_values)))
        
        # Assess trend
        x_vals = np.arange(len(clean_values))
        if len(clean_values) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, clean_values)
            trend_slope = float(slope)
            trend_r_squared = float(r_value ** 2)
        else:
            trend_slope = None
            trend_r_squared = None
        
        return PerformanceBaseline(
            metric_name=metric_name,
            baseline_value=mean_val,
            standard_deviation=std_val,
            sample_count=len(clean_values),
            confidence_interval_lower=mean_val - confidence_margin,
            confidence_interval_upper=mean_val + confidence_margin,
            min_value=float(np.min(clean_values)),
            max_value=float(np.max(clean_values)),
            percentile_25=float(np.percentile(clean_values, 25)),
            percentile_75=float(np.percentile(clean_values, 75)),
            median_value=float(np.median(clean_values)),
            trend_slope=trend_slope,
            trend_r_squared=trend_r_squared,
            is_stable=std_val <= (mean_val * 0.1) if mean_val != 0 else True
        )

    def _create_degradation(
        self,
        metric_name: str,
        current_value: float,
        baseline: PerformanceBaseline,
        threshold: MetricThreshold,
        severity: DegradationSeverity,
        detection_method: str,
        samples_used: int
    ) -> PerformanceDegradation:
        """Create a performance degradation instance.
        
        Args:
            metric_name: Name of the degraded metric
            current_value: Current metric value
            baseline: Baseline for comparison
            threshold: Threshold configuration
            severity: Severity of degradation
            detection_method: Method used for detection
            samples_used: Number of samples used
            
        Returns:
            Performance degradation instance
        """
        degradation_amount = current_value - baseline.baseline_value
        degradation_percentage = ((current_value - baseline.baseline_value) / baseline.baseline_value) * 100 if baseline.baseline_value != 0 else 0
        
        # Determine degradation type
        degradation_type = self._determine_degradation_type(metric_name, threshold.direction)
        
        # Calculate confidence level
        confidence_level = self._calculate_confidence(current_value, baseline, threshold)
        
        # Determine threshold violated
        threshold_violated = "critical" if severity == DegradationSeverity.CRITICAL else "warning"
        
        return PerformanceDegradation(
            degradation_type=degradation_type,
            severity=severity,
            metric_name=metric_name,
            current_value=current_value,
            baseline_value=baseline.baseline_value,
            degradation_amount=degradation_amount,
            degradation_percentage=degradation_percentage,
            threshold_violated=threshold_violated,
            confidence_level=confidence_level,
            detection_method=detection_method,
            samples_used=samples_used
        )

    def _determine_degradation_type(self, metric_name: str, direction: str) -> DegradationType:
        """Determine the type of degradation based on metric name."""
        type_mapping = {
            "accuracy": DegradationType.ACCURACY_DROP,
            "precision": DegradationType.PRECISION_DROP,
            "recall": DegradationType.RECALL_DROP,
            "f1_score": DegradationType.F1_DROP,
            "roc_auc": DegradationType.AUC_DROP,
            "pr_auc": DegradationType.AUC_DROP,
            "rmse": DegradationType.RMSE_INCREASE,
            "mae": DegradationType.MAE_INCREASE,
            "r2_score": DegradationType.R2_DROP,
            "prediction_time_seconds": DegradationType.LATENCY_INCREASE,
            "memory_usage_mb": DegradationType.MEMORY_INCREASE,
            "prediction_stability": DegradationType.STABILITY_DROP,
            "business_value_score": DegradationType.BUSINESS_VALUE_DROP
        }
        
        return type_mapping.get(metric_name, DegradationType.ACCURACY_DROP)

    def _calculate_confidence(
        self, 
        current_value: float, 
        baseline: PerformanceBaseline, 
        threshold: MetricThreshold
    ) -> float:
        """Calculate confidence level for degradation detection."""
        if baseline.standard_deviation == 0:
            return 1.0
        
        # Calculate z-score
        z_score = abs(current_value - baseline.baseline_value) / baseline.standard_deviation
        
        # Convert to confidence level
        confidence = min(z_score / 3.0, 1.0)  # Max confidence at 3 std deviations
        
        # Adjust based on sample size
        sample_adjustment = min(baseline.sample_count / 30.0, 1.0)
        
        return confidence * sample_adjustment

    async def _analyze_performance_trends(
        self,
        recent_metrics: List[ModelPerformanceMetrics],
        baseline_metrics: List[ModelPerformanceMetrics],
        model_id: UUID
    ) -> List[PerformanceDegradation]:
        """Analyze performance trends for degradation.
        
        Args:
            recent_metrics: Recent performance metrics
            baseline_metrics: Baseline performance metrics
            model_id: Model ID
            
        Returns:
            List of detected degradations
        """
        degradations = []
        
        # Extract time series for each metric
        recent_series = self._extract_metric_series(recent_metrics)
        baseline_series = self._extract_metric_series(baseline_metrics)
        
        for metric_name in recent_series:
            if metric_name not in baseline_series:
                continue
            
            recent_values = recent_series[metric_name]
            baseline_values = baseline_series[metric_name]
            
            if len(recent_values) < 5 or len(baseline_values) < 5:
                continue
            
            # Perform statistical test
            degradation = await self._perform_trend_analysis(
                metric_name=metric_name,
                recent_values=recent_values,
                baseline_values=baseline_values
            )
            
            if degradation:
                degradations.append(degradation)
        
        return degradations

    def _extract_metric_series(self, metrics_list: List[ModelPerformanceMetrics]) -> Dict[str, List[float]]:
        """Extract time series for each metric."""
        series = {}
        
        for metrics in metrics_list:
            metrics_dict = metrics.dict()
            for metric_name, value in metrics_dict.items():
                if value is not None:
                    if metric_name not in series:
                        series[metric_name] = []
                    series[metric_name].append(value)
        
        return series

    async def _perform_trend_analysis(
        self,
        metric_name: str,
        recent_values: List[float],
        baseline_values: List[float]
    ) -> Optional[PerformanceDegradation]:
        """Perform statistical analysis for trend degradation."""
        
        # Perform two-sample t-test
        try:
            t_stat, p_value = stats.ttest_ind(recent_values, baseline_values)
            
            # Check if degradation is significant
            if p_value > 0.05:  # Not statistically significant
                return None
            
            recent_mean = np.mean(recent_values)
            baseline_mean = np.mean(baseline_values)
            
            # Determine if this is actually degradation
            threshold = self._default_thresholds.get(metric_name)
            if not threshold:
                return None
            
            # Check direction
            is_degradation = False
            if threshold.direction == "decrease" and recent_mean < baseline_mean:
                is_degradation = True
            elif threshold.direction == "increase" and recent_mean > baseline_mean:
                is_degradation = True
            
            if not is_degradation:
                return None
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(recent_values) - 1) * np.var(recent_values) + 
                                (len(baseline_values) - 1) * np.var(baseline_values)) / 
                               (len(recent_values) + len(baseline_values) - 2))
            
            effect_size = abs(recent_mean - baseline_mean) / pooled_std
            
            # Determine severity based on effect size
            if effect_size >= 0.8:  # Large effect
                severity = DegradationSeverity.CRITICAL
            elif effect_size >= 0.5:  # Medium effect
                severity = DegradationSeverity.HIGH
            else:
                severity = DegradationSeverity.MEDIUM
            
            # Create baseline for comparison
            baseline = self._create_baseline(metric_name, baseline_values)
            
            degradation = self._create_degradation(
                metric_name=metric_name,
                current_value=recent_mean,
                baseline=baseline,
                threshold=threshold,
                severity=severity,
                detection_method="trend_analysis",
                samples_used=len(recent_values)
            )
            
            return degradation
            
        except Exception as e:
            logger.error(f"Error in trend analysis for {metric_name}: {e}")
            return None

    async def generate_degradation_report(
        self,
        model_id: UUID,
        degradations: List[PerformanceDegradation],
        time_period_start: datetime,
        time_period_end: datetime
    ) -> DegradationReport:
        """Generate comprehensive degradation report.
        
        Args:
            model_id: Model ID
            degradations: Detected degradations
            time_period_start: Start of analysis period
            time_period_end: End of analysis period
            
        Returns:
            Degradation report
        """
        report_id = str(uuid4())
        
        # Calculate overall health score
        health_score = self._calculate_health_score(degradations)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(degradations)
        
        # Create degradation summary
        degradation_summary = {
            "total_degradations": len(degradations),
            "severity_breakdown": {},
            "affected_metrics": list(set(d.metric_name for d in degradations)),
            "most_severe": max(degradations, key=lambda x: x.severity.value if degradations else 0)
        }
        
        # Count by severity
        for severity in DegradationSeverity:
            count = sum(1 for d in degradations if d.severity == severity)
            degradation_summary["severity_breakdown"][severity.value] = count
        
        return DegradationReport(
            report_id=report_id,
            model_id=str(model_id),
            time_period_start=time_period_start,
            time_period_end=time_period_end,
            degradations=degradations,
            overall_health_score=health_score,
            degradation_summary=degradation_summary,
            recommendations=recommendations
        )

    def _calculate_health_score(self, degradations: List[PerformanceDegradation]) -> float:
        """Calculate overall health score based on degradations."""
        if not degradations:
            return 1.0
        
        # Weight degradations by severity
        severity_weights = {
            DegradationSeverity.CRITICAL: 0.5,
            DegradationSeverity.HIGH: 0.3,
            DegradationSeverity.MEDIUM: 0.15,
            DegradationSeverity.LOW: 0.05,
            DegradationSeverity.NONE: 0.0
        }
        
        total_penalty = sum(severity_weights[d.severity] for d in degradations)
        health_score = max(0.0, 1.0 - total_penalty)
        
        return health_score

    def _generate_recommendations(self, degradations: List[PerformanceDegradation]) -> List[str]:
        """Generate recommendations based on degradations."""
        recommendations = []
        
        if not degradations:
            recommendations.append("Model performance is stable. Continue monitoring.")
            return recommendations
        
        # Critical degradations
        critical_degradations = [d for d in degradations if d.severity == DegradationSeverity.CRITICAL]
        if critical_degradations:
            recommendations.append(
                f"🚨 URGENT: {len(critical_degradations)} critical performance degradations detected. "
                "Consider immediate model retraining or rollback."
            )
        
        # High degradations
        high_degradations = [d for d in degradations if d.severity == DegradationSeverity.HIGH]
        if high_degradations:
            recommendations.append(
                f"⚠️ {len(high_degradations)} high-severity degradations detected. "
                "Plan for model update within 24-48 hours."
            )
        
        # Specific metric recommendations
        metric_recommendations = {
            "accuracy": "Review training data quality and feature engineering",
            "precision": "Check for class imbalance or false positive patterns",
            "recall": "Investigate potential data drift or missing features",
            "f1_score": "Balance precision and recall optimization",
            "roc_auc": "Review model calibration and decision thresholds",
            "rmse": "Check for outliers and feature scaling issues",
            "r2_score": "Evaluate feature selection and model complexity",
            "prediction_time_seconds": "Optimize model architecture or hardware",
            "memory_usage_mb": "Consider model compression techniques"
        }
        
        degraded_metrics = set(d.metric_name for d in degradations)
        for metric in degraded_metrics:
            if metric in metric_recommendations:
                recommendations.append(f"For {metric}: {metric_recommendations[metric]}")
        
        # General recommendations
        if len(degradations) > 5:
            recommendations.append("Multiple metrics affected - consider comprehensive model audit")
        
        recommendations.append("Monitor trends closely and establish alerts for future degradation")
        
        return recommendations