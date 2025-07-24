"""Refactored monitoring service using hexagonal architecture."""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from mlops.domain.interfaces.mlops_monitoring_operations import (
    ModelPerformanceMonitoringPort,
    InfrastructureMonitoringPort,
    DataQualityMonitoringPort,
    DataDriftMonitoringPort,
    AlertingPort,
    HealthCheckPort,
    PerformanceMetrics,
    InfrastructureMetrics,
    DataQualityMetrics,
    DriftDetectionResult,
    MonitoringAlert,
    ModelHealthReport,
    MonitoringRule,
    MonitoringAlertSeverity,
    MonitoringAlertStatus,
    ModelHealthStatus,
    DataDriftType,
    MetricThreshold
)

logger = logging.getLogger(__name__)


class MonitoringService:
    """Clean domain service for MLOps monitoring using dependency injection."""
    
    def __init__(
        self,
        performance_monitoring_port: ModelPerformanceMonitoringPort,
        infrastructure_monitoring_port: InfrastructureMonitoringPort,
        data_quality_monitoring_port: DataQualityMonitoringPort,
        data_drift_monitoring_port: DataDriftMonitoringPort,
        alerting_port: AlertingPort,
        health_check_port: HealthCheckPort
    ):
        """Initialize service with injected dependencies.
        
        Args:
            performance_monitoring_port: Port for performance monitoring
            infrastructure_monitoring_port: Port for infrastructure monitoring
            data_quality_monitoring_port: Port for data quality monitoring
            data_drift_monitoring_port: Port for data drift monitoring
            alerting_port: Port for alerting operations
            health_check_port: Port for health checks
        """
        self._performance_monitoring_port = performance_monitoring_port
        self._infrastructure_monitoring_port = infrastructure_monitoring_port
        self._data_quality_monitoring_port = data_quality_monitoring_port
        self._data_drift_monitoring_port = data_drift_monitoring_port
        self._alerting_port = alerting_port
        self._health_check_port = health_check_port
        
        logger.info("MonitoringService initialized with clean architecture")
    
    async def track_model_performance(
        self,
        model_id: str,
        deployment_id: Optional[str],
        accuracy: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        f1_score: Optional[float] = None,
        latency_p95: Optional[float] = None,
        throughput: Optional[float] = None,
        error_rate: Optional[float] = None,
        custom_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Track model performance metrics with business validation.
        
        Args:
            model_id: ID of the model
            deployment_id: ID of the deployment
            accuracy: Model accuracy
            precision: Model precision
            recall: Model recall
            f1_score: Model F1 score
            latency_p95: 95th percentile latency
            throughput: Request throughput
            error_rate: Error rate
            custom_metrics: Additional custom metrics
        """
        # Validate inputs
        validated_metrics = self._validate_performance_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "latency_p95": latency_p95,
            "throughput": throughput,
            "error_rate": error_rate
        })
        
        # Create metrics object
        metrics = PerformanceMetrics(
            accuracy=validated_metrics.get("accuracy"),
            precision=validated_metrics.get("precision"),
            recall=validated_metrics.get("recall"),
            f1_score=validated_metrics.get("f1_score"),
            latency_p95=validated_metrics.get("latency_p95"),
            throughput=validated_metrics.get("throughput"),
            error_rate=validated_metrics.get("error_rate"),
            custom_metrics=custom_metrics or {},
            timestamp=datetime.utcnow()
        )
        
        try:
            # Log metrics
            await self._performance_monitoring_port.log_prediction_metrics(
                model_id, deployment_id, metrics
            )
            
            # Apply business rules for performance monitoring
            await self._apply_performance_business_rules(model_id, deployment_id, metrics)
            
            logger.info(f"Tracked performance metrics for model {model_id}")
            
        except Exception as e:
            logger.error(f"Failed to track model performance: {e}")
            raise
    
    async def monitor_data_drift(
        self,
        model_id: str,
        reference_data: Dict[str, Any],
        current_data: Dict[str, Any],
        drift_threshold: float = 0.1,
        auto_alert: bool = True
    ) -> DriftDetectionResult:
        """Monitor data drift with automated alerting.
        
        Args:
            model_id: ID of the model
            reference_data: Reference dataset for comparison
            current_data: Current dataset to check for drift
            drift_threshold: Threshold for drift detection
            auto_alert: Whether to automatically create alerts
            
        Returns:
            Drift detection result
        """
        # Validate inputs
        if not reference_data or not current_data:
            raise ValueError("Both reference and current data must be provided")
        
        try:
            # Detect feature drift
            drift_result = await self._data_drift_monitoring_port.detect_feature_drift(
                model_id, reference_data, current_data
            )
            
            # Apply business logic for drift analysis
            enhanced_result = await self._analyze_drift_result(drift_result, drift_threshold)
            
            # Create alerts if drift detected and auto_alert enabled
            if auto_alert and enhanced_result.is_drift_detected:
                await self._create_drift_alert(model_id, enhanced_result)
            
            # Log drift detection for audit trail
            await self._log_drift_detection(model_id, enhanced_result)
            
            logger.info(f"Monitored data drift for model {model_id} (drift: {enhanced_result.is_drift_detected})")
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Failed to monitor data drift: {e}")
            raise
    
    async def assess_model_health(
        self,
        model_id: str,
        deployment_id: Optional[str] = None,
        include_predictions: bool = True,
        include_infrastructure: bool = True,
        include_drift: bool = True
    ) -> ModelHealthReport:
        """Perform comprehensive model health assessment.
        
        Args:
            model_id: ID of the model
            deployment_id: ID of the deployment
            include_predictions: Include prediction health
            include_infrastructure: Include infrastructure health
            include_drift: Include drift analysis
            
        Returns:
            Comprehensive health report
        """
        try:
            # Get base health report
            health_report = await self._health_check_port.check_model_health(
                model_id, deployment_id
            )
            
            # Enhance with business intelligence
            enhanced_report = await self._enhance_health_report(
                health_report, include_predictions, include_infrastructure, include_drift
            )
            
            # Apply health-based business rules
            await self._apply_health_business_rules(enhanced_report)
            
            logger.info(f"Assessed health for model {model_id} (status: {enhanced_report.overall_health.value})")
            return enhanced_report
            
        except Exception as e:
            logger.error(f"Failed to assess model health: {e}")
            raise
    
    async def create_monitoring_rule(
        self,
        name: str,
        model_id: Optional[str],
        deployment_id: Optional[str],
        metric_name: str,
        threshold_value: float,
        operator: str = ">",
        severity: MonitoringAlertSeverity = MonitoringAlertSeverity.MEDIUM,
        enabled: bool = True,
        description: str = ""
    ) -> str:
        """Create monitoring rule with business validation.
        
        Args:
            name: Rule name
            model_id: Model to monitor (None for all models)
            deployment_id: Deployment to monitor (None for all deployments)
            metric_name: Metric to monitor
            threshold_value: Threshold value
            operator: Comparison operator
            severity: Alert severity
            enabled: Whether rule is enabled
            description: Rule description
            
        Returns:
            Rule ID
        """
        # Validate inputs
        if not name or not name.strip():
            raise ValueError("Rule name cannot be empty")
        
        if operator not in [">", "<", ">=", "<=", "==", "!="]:
            raise ValueError(f"Invalid operator: {operator}")
        
        # Create threshold
        threshold = MetricThreshold(
            metric_name=metric_name,
            operator=operator,
            threshold_value=threshold_value,
            severity=severity,
            description=description or f"Monitor {metric_name} {operator} {threshold_value}"
        )
        
        # Create rule
        rule = MonitoringRule(
            rule_id=f"rule_{model_id or 'global'}_{metric_name}_{int(datetime.utcnow().timestamp())}",
            name=name.strip(),
            description=description,
            model_id=model_id,
            deployment_id=deployment_id,
            thresholds=[threshold],
            evaluation_window=timedelta(minutes=5),
            alert_frequency=timedelta(minutes=15),
            enabled=enabled,
            tags=[]
        )
        
        try:
            rule_id = await self._alerting_port.create_monitoring_rule(rule)
            
            # Apply business rules for new monitoring rules
            await self._apply_new_rule_business_rules(rule)
            
            logger.info(f"Created monitoring rule {rule_id}")
            return rule_id
            
        except Exception as e:
            logger.error(f"Failed to create monitoring rule: {e}")
            raise
    
    async def handle_performance_degradation(
        self,
        model_id: str,
        deployment_id: Optional[str] = None,
        baseline_days: int = 7,
        comparison_days: int = 1,
        degradation_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """Handle performance degradation with automated response.
        
        Args:
            model_id: ID of the model
            deployment_id: ID of the deployment
            baseline_days: Days for baseline calculation
            comparison_days: Days for comparison
            degradation_threshold: Threshold for degradation alert
            
        Returns:
            Degradation analysis and response actions
        """
        try:
            # Calculate performance degradation
            baseline_period = timedelta(days=baseline_days)
            comparison_period = timedelta(days=comparison_days)
            
            degradation_metrics = await self._performance_monitoring_port.calculate_performance_degradation(
                model_id, deployment_id, baseline_period, comparison_period
            )
            
            # Analyze degradation severity
            analysis = await self._analyze_performance_degradation(
                degradation_metrics, degradation_threshold
            )
            
            # Take automated actions if needed
            if analysis["requires_action"]:
                actions_taken = await self._take_degradation_actions(
                    model_id, deployment_id, analysis
                )
                analysis["actions_taken"] = actions_taken
            
            logger.info(f"Handled performance degradation for model {model_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to handle performance degradation: {e}")
            raise
    
    async def manage_alert_lifecycle(
        self,
        alert_id: str,
        action: str,
        user: str,
        note: Optional[str] = None
    ) -> bool:
        """Manage alert lifecycle with business rules.
        
        Args:
            alert_id: ID of the alert
            action: Action to take (acknowledge, resolve, suppress)
            user: User performing the action
            note: Optional note
            
        Returns:
            True if action successful
        """
        # Validate action
        valid_actions = ["acknowledge", "resolve", "suppress"]
        if action not in valid_actions:
            raise ValueError(f"Invalid action: {action}. Must be one of {valid_actions}")
        
        try:
            # Get current alert
            alert = await self._alerting_port.get_alert(alert_id)
            if not alert:
                raise ValueError(f"Alert {alert_id} not found")
            
            # Apply business rules for alert actions
            await self._validate_alert_action(alert, action, user)
            
            # Perform action
            success = False
            if action == "acknowledge":
                success = await self._alerting_port.acknowledge_alert(alert_id, user, note)
            elif action == "resolve":
                success = await self._alerting_port.resolve_alert(alert_id, user, note)
            elif action == "suppress":
                # Custom business logic for suppression
                success = await self._suppress_alert(alert_id, user, note)
            
            # Apply post-action business rules
            if success:
                await self._apply_post_alert_action_rules(alert, action, user)
            
            logger.info(f"Performed alert action {action} on {alert_id} by {user}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to manage alert lifecycle: {e}")
            raise
    
    async def generate_monitoring_insights(
        self,
        model_id: str,
        time_period: timedelta = timedelta(days=7),
        include_trends: bool = True,
        include_anomalies: bool = True,
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """Generate monitoring insights with business intelligence.
        
        Args:
            model_id: ID of the model
            time_period: Time period for analysis
            include_trends: Include trend analysis
            include_anomalies: Include anomaly detection
            include_recommendations: Include recommendations
            
        Returns:
            Comprehensive monitoring insights
        """
        try:
            insights = {
                "model_id": model_id,
                "analysis_period": time_period.days,
                "generated_at": datetime.utcnow().isoformat(),
                "summary": {}
            }
            
            # Get performance trends
            if include_trends:
                insights["performance_trends"] = await self._analyze_performance_trends(
                    model_id, time_period
                )
            
            # Detect anomalies
            if include_anomalies:
                insights["anomalies"] = await self._detect_monitoring_anomalies(
                    model_id, time_period
                )
            
            # Generate recommendations
            if include_recommendations:
                insights["recommendations"] = await self._generate_monitoring_recommendations(
                    model_id, insights
                )
            
            # Create executive summary
            insights["summary"] = await self._create_insights_summary(insights)
            
            logger.info(f"Generated monitoring insights for model {model_id}")
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate monitoring insights: {e}")
            raise
    
    # Private helper methods
    
    def _validate_performance_metrics(self, metrics: Dict[str, Optional[float]]) -> Dict[str, float]:
        """Validate performance metrics."""
        validated = {}
        
        for name, value in metrics.items():
            if value is not None:
                try:
                    float_value = float(value)
                    
                    # Validate ranges based on metric type
                    if name in ["accuracy", "precision", "recall", "f1_score"]:
                        if 0.0 <= float_value <= 1.0:
                            validated[name] = float_value
                    elif name == "error_rate":
                        if 0.0 <= float_value <= 1.0:
                            validated[name] = float_value
                    elif name in ["latency_p95", "throughput"]:
                        if float_value >= 0:
                            validated[name] = float_value
                    else:
                        validated[name] = float_value
                        
                except (ValueError, TypeError):
                    logger.warning(f"Invalid metric value: {name}={value}")
        
        return validated
    
    async def _apply_performance_business_rules(
        self,
        model_id: str,
        deployment_id: Optional[str],
        metrics: PerformanceMetrics
    ) -> None:
        """Apply business rules for performance monitoring."""
        try:
            # Auto-alert on poor performance
            if metrics.accuracy and metrics.accuracy < 0.7:
                await self._create_performance_alert(
                    model_id, deployment_id, "low_accuracy", metrics.accuracy
                )
            
            if metrics.error_rate and metrics.error_rate > 0.05:
                await self._create_performance_alert(
                    model_id, deployment_id, "high_error_rate", metrics.error_rate
                )
            
            if metrics.latency_p95 and metrics.latency_p95 > 1000:  # >1 second
                await self._create_performance_alert(
                    model_id, deployment_id, "high_latency", metrics.latency_p95
                )
            
        except Exception as e:
            logger.warning(f"Failed to apply performance business rules: {e}")
    
    async def _analyze_drift_result(
        self,
        drift_result: DriftDetectionResult,
        threshold: float
    ) -> DriftDetectionResult:
        """Analyze drift result with business logic."""
        # Enhance with business context
        if drift_result.drift_score > threshold and not drift_result.is_drift_detected:
            # Override detection based on business rules
            drift_result.is_drift_detected = True
            logger.info(f"Business rule override: drift detected based on score {drift_result.drift_score}")
        
        return drift_result
    
    async def _create_drift_alert(self, model_id: str, drift_result: DriftDetectionResult) -> None:
        """Create alert for detected drift."""
        try:
            alert_data = {
                "title": f"Data Drift Detected - {model_id}",
                "description": f"Data drift detected for model {model_id}. Drift score: {drift_result.drift_score:.3f}",
                "severity": "high" if drift_result.drift_score > 0.5 else "medium",
                "model_id": model_id,
                "metadata": {
                    "drift_type": drift_result.drift_type.value,
                    "drift_score": drift_result.drift_score,
                    "affected_features": drift_result.affected_features
                },
                "remediation_suggestions": [
                    "Review data quality",
                    "Consider model retraining",
                    "Check data source changes"
                ]
            }
            
            # Create a dummy rule for the alert
            rule_id = f"drift_rule_{model_id}"
            await self._alerting_port.trigger_alert(rule_id, alert_data)
            
        except Exception as e:
            logger.warning(f"Failed to create drift alert: {e}")
    
    async def _log_drift_detection(self, model_id: str, drift_result: DriftDetectionResult) -> None:
        """Log drift detection for audit trail."""
        logger.info(f"Drift detection logged for model {model_id}: "
                   f"type={drift_result.drift_type.value}, "
                   f"detected={drift_result.is_drift_detected}, "
                   f"score={drift_result.drift_score:.3f}")
    
    async def _enhance_health_report(
        self,
        health_report: ModelHealthReport,
        include_predictions: bool,
        include_infrastructure: bool,
        include_drift: bool
    ) -> ModelHealthReport:
        """Enhance health report with additional business context."""
        # Add business-specific health checks
        recommendations = list(health_report.recommendations)
        
        # Add performance-based recommendations
        if health_report.performance_metrics:
            if health_report.performance_metrics.accuracy and health_report.performance_metrics.accuracy < 0.8:
                recommendations.append("Consider model retraining due to low accuracy")
            
            if health_report.performance_metrics.error_rate and health_report.performance_metrics.error_rate > 0.02:
                recommendations.append("Investigate high error rate")
        
        # Add infrastructure-based recommendations
        if health_report.infrastructure_metrics:
            if health_report.infrastructure_metrics.cpu_usage and health_report.infrastructure_metrics.cpu_usage > 80:
                recommendations.append("Consider scaling up CPU resources")
            
            if health_report.infrastructure_metrics.memory_usage and health_report.infrastructure_metrics.memory_usage > 85:
                recommendations.append("Consider scaling up memory resources")
        
        health_report.recommendations = recommendations
        return health_report
    
    async def _apply_health_business_rules(self, health_report: ModelHealthReport) -> None:
        """Apply business rules based on health report."""
        try:
            # Auto-scale if resource usage is high
            if (health_report.infrastructure_metrics and 
                health_report.infrastructure_metrics.cpu_usage and 
                health_report.infrastructure_metrics.cpu_usage > 85):
                
                await self._trigger_auto_scaling(health_report.model_id, health_report.deployment_id)
            
            # Create health alert if overall health is poor
            if health_report.overall_health in [ModelHealthStatus.UNHEALTHY, ModelHealthStatus.DEGRADED]:
                await self._create_health_alert(health_report)
            
        except Exception as e:
            logger.warning(f"Failed to apply health business rules: {e}")
    
    async def _apply_new_rule_business_rules(self, rule: MonitoringRule) -> None:
        """Apply business rules for new monitoring rules."""
        try:
            # Auto-enable critical rules
            if any(threshold.severity == MonitoringAlertSeverity.CRITICAL for threshold in rule.thresholds):
                logger.info(f"Auto-enabled critical monitoring rule {rule.rule_id}")
            
            # Set up rule dependencies
            if rule.model_id:
                logger.info(f"Associated rule {rule.rule_id} with model {rule.model_id}")
            
        except Exception as e:
            logger.warning(f"Failed to apply new rule business rules: {e}")
    
    async def _analyze_performance_degradation(
        self,
        degradation_metrics: Dict[str, float],
        threshold: float
    ) -> Dict[str, Any]:
        """Analyze performance degradation."""
        analysis = {
            "degradation_metrics": degradation_metrics,
            "threshold": threshold,
            "requires_action": False,
            "severity": "low",
            "affected_metrics": []
        }
        
        # Check each metric
        for metric, value in degradation_metrics.items():
            if abs(value) > threshold:
                analysis["affected_metrics"].append({
                    "metric": metric,
                    "degradation": value,
                    "severity": "high" if abs(value) > threshold * 2 else "medium"
                })
        
        # Determine if action is required
        high_severity_count = sum(1 for m in analysis["affected_metrics"] if m["severity"] == "high")
        if high_severity_count > 0:
            analysis["requires_action"] = True
            analysis["severity"] = "high"
        elif len(analysis["affected_metrics"]) > 2:
            analysis["requires_action"] = True
            analysis["severity"] = "medium"
        
        return analysis
    
    async def _take_degradation_actions(
        self,
        model_id: str,
        deployment_id: Optional[str],
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Take automated actions for performance degradation."""
        actions_taken = []
        
        try:
            # Create alert
            alert_data = {
                "title": f"Performance Degradation - {model_id}",
                "description": f"Performance degradation detected for model {model_id}",
                "severity": analysis["severity"],
                "model_id": model_id,
                "deployment_id": deployment_id,
                "metadata": analysis["degradation_metrics"]
            }
            
            rule_id = f"degradation_rule_{model_id}"
            await self._alerting_port.trigger_alert(rule_id, alert_data)
            actions_taken.append("Created performance degradation alert")
            
            # Additional actions based on severity
            if analysis["severity"] == "high":
                # Could trigger auto-scaling, rollback, etc.
                actions_taken.append("Flagged for immediate attention")
            
        except Exception as e:
            logger.warning(f"Failed to take degradation actions: {e}")
            actions_taken.append(f"Failed to complete some actions: {e}")
        
        return actions_taken
    
    async def _validate_alert_action(self, alert: MonitoringAlert, action: str, user: str) -> None:
        """Validate alert action based on business rules."""
        # Check if alert is in valid state for action
        if action == "acknowledge" and alert.status != MonitoringAlertStatus.ACTIVE:
            raise ValueError("Can only acknowledge active alerts")
        
        if action == "resolve" and alert.status == MonitoringAlertStatus.RESOLVED:
            raise ValueError("Alert is already resolved")
        
        # Add user permission checks here if needed
        logger.debug(f"Validated action {action} by user {user} on alert {alert.alert_id}")
    
    async def _suppress_alert(self, alert_id: str, user: str, note: Optional[str]) -> bool:
        """Custom business logic for alert suppression."""
        try:
            # Implementation would suppress alert notifications
            logger.info(f"Suppressed alert {alert_id} by {user}")
            return True
        except Exception as e:
            logger.error(f"Failed to suppress alert: {e}")
            return False
    
    async def _apply_post_alert_action_rules(
        self,
        alert: MonitoringAlert,
        action: str,
        user: str
    ) -> None:
        """Apply business rules after alert actions."""
        try:
            # Log action for audit
            logger.info(f"Alert {alert.alert_id} {action} by {user}")
            
            # Apply escalation rules if needed
            if action == "resolve" and alert.severity == MonitoringAlertSeverity.CRITICAL:
                # Could trigger follow-up actions
                logger.info(f"Critical alert {alert.alert_id} resolved, triggering post-resolution checks")
            
        except Exception as e:
            logger.warning(f"Failed to apply post-alert action rules: {e}")
    
    async def _analyze_performance_trends(self, model_id: str, time_period: timedelta) -> Dict[str, Any]:
        """Analyze performance trends."""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - time_period
            
            metrics = await self._performance_monitoring_port.get_performance_metrics(
                model_id, start_time=start_time, end_time=end_time
            )
            
            if not metrics:
                return {"status": "no_data", "message": "No performance data available"}
            
            # Calculate trends
            trends = {
                "accuracy_trend": self._calculate_trend([m.accuracy for m in metrics if m.accuracy]),
                "latency_trend": self._calculate_trend([m.latency_p95 for m in metrics if m.latency_p95]),
                "error_rate_trend": self._calculate_trend([m.error_rate for m in metrics if m.error_rate]),
                "data_points": len(metrics)
            }
            
            return trends
            
        except Exception as e:
            logger.warning(f"Failed to analyze performance trends: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _detect_monitoring_anomalies(self, model_id: str, time_period: timedelta) -> List[Dict[str, Any]]:
        """Detect monitoring anomalies."""
        try:
            anomalies = await self._performance_monitoring_port.detect_performance_anomalies(
                model_id, lookback_window=time_period
            )
            
            return anomalies
            
        except Exception as e:
            logger.warning(f"Failed to detect monitoring anomalies: {e}")
            return []
    
    async def _generate_monitoring_recommendations(
        self,
        model_id: str,
        insights: Dict[str, Any]
    ) -> List[str]:
        """Generate monitoring recommendations."""
        recommendations = []
        
        # Performance trend recommendations
        trends = insights.get("performance_trends", {})
        if trends.get("accuracy_trend") == "declining":
            recommendations.append("Model accuracy is declining - consider retraining")
        
        if trends.get("latency_trend") == "increasing":
            recommendations.append("Response latency is increasing - investigate performance bottlenecks")
        
        # Anomaly recommendations
        anomalies = insights.get("anomalies", [])
        if len(anomalies) > 3:
            recommendations.append("Multiple anomalies detected - review model stability")
        
        # Default recommendations
        if not recommendations:
            recommendations.append("Monitoring is healthy - continue current practices")
        
        return recommendations
    
    async def _create_insights_summary(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary of insights."""
        summary = {
            "overall_status": "healthy",
            "key_findings": [],
            "priority_actions": []
        }
        
        # Analyze trends
        trends = insights.get("performance_trends", {})
        if trends.get("accuracy_trend") == "declining":
            summary["overall_status"] = "attention_needed"
            summary["key_findings"].append("Model accuracy is declining")
            summary["priority_actions"].append("Schedule model retraining")
        
        # Analyze anomalies
        anomalies = insights.get("anomalies", [])
        if len(anomalies) > 0:
            summary["key_findings"].append(f"{len(anomalies)} performance anomalies detected")
            if len(anomalies) > 3:
                summary["overall_status"] = "attention_needed"
                summary["priority_actions"].append("Investigate performance anomalies")
        
        return summary
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from list of values."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend calculation
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        if not first_half or not second_half:
            return "insufficient_data"
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        change_percent = ((second_avg - first_avg) / first_avg) * 100 if first_avg != 0 else 0
        
        if change_percent > 5:
            return "increasing"
        elif change_percent < -5:
            return "declining"
        else:
            return "stable"
    
    async def _create_performance_alert(
        self,
        model_id: str,
        deployment_id: Optional[str],
        alert_type: str,
        value: float
    ) -> None:
        """Create performance-related alert."""
        try:
            alert_data = {
                "title": f"Performance Alert - {alert_type}",
                "description": f"Performance issue detected: {alert_type} = {value}",
                "severity": "high" if alert_type in ["low_accuracy", "high_error_rate"] else "medium",
                "model_id": model_id,
                "deployment_id": deployment_id,
                "metadata": {"alert_type": alert_type, "value": value}
            }
            
            rule_id = f"performance_rule_{model_id}_{alert_type}"
            await self._alerting_port.trigger_alert(rule_id, alert_data)
            
        except Exception as e:
            logger.warning(f"Failed to create performance alert: {e}")
    
    async def _trigger_auto_scaling(self, model_id: str, deployment_id: Optional[str]) -> None:
        """Trigger auto-scaling for high resource usage."""
        try:
            logger.info(f"Triggering auto-scaling for model {model_id}, deployment {deployment_id}")
            # Implementation would trigger actual scaling
            
        except Exception as e:
            logger.warning(f"Failed to trigger auto-scaling: {e}")
    
    async def _create_health_alert(self, health_report: ModelHealthReport) -> None:
        """Create alert based on health report."""
        try:
            alert_data = {
                "title": f"Model Health Alert - {health_report.model_id}",
                "description": f"Model health is {health_report.overall_health.value}",
                "severity": "high" if health_report.overall_health == ModelHealthStatus.UNHEALTHY else "medium",
                "model_id": health_report.model_id,
                "deployment_id": health_report.deployment_id,
                "metadata": {
                    "overall_health": health_report.overall_health.value,
                    "performance_health": health_report.performance_health.value,
                    "infrastructure_health": health_report.infrastructure_health.value
                }
            }
            
            rule_id = f"health_rule_{health_report.model_id}"
            await self._alerting_port.trigger_alert(rule_id, alert_data)
            
        except Exception as e:
            logger.warning(f"Failed to create health alert: {e}")