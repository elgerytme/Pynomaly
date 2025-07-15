"""Model performance monitoring service that orchestrates degradation detection."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from packages.data_science.domain.value_objects.model_performance_metrics import (
    ModelPerformanceMetrics,
    ModelTask
)
from pynomaly.application.services.performance_alert_service import PerformanceAlertService
from pynomaly.domain.services.performance_degradation_service import PerformanceDegradationService
from pynomaly.domain.value_objects.performance_degradation_metrics import (
    DegradationReport,
    DegradationSeverity,
    MetricThreshold,
    PerformanceBaseline,
    PerformanceDegradation,
)
from pynomaly.infrastructure.repositories.performance_degradation_repository import (
    PerformanceDegradationRepository,
)
from pynomaly.shared.protocols.repository_protocol import ModelRepositoryProtocol

logger = logging.getLogger(__name__)


class ModelPerformanceMonitor:
    """High-level service for monitoring model performance and detecting degradation."""
    
    def __init__(
        self,
        model_repository: ModelRepositoryProtocol,
        performance_repository: PerformanceDegradationRepository,
        degradation_service: PerformanceDegradationService,
        alert_service: PerformanceAlertService,
        monitoring_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the performance monitor.
        
        Args:
            model_repository: Repository for model data
            performance_repository: Repository for performance data
            degradation_service: Service for degradation detection
            alert_service: Service for alerting
            monitoring_config: Monitoring configuration
        """
        self.model_repository = model_repository
        self.performance_repository = performance_repository
        self.degradation_service = degradation_service
        self.alert_service = alert_service
        self.config = monitoring_config or self._default_config()
        
        # Performance tracking
        self._monitoring_active = False
        self._monitored_models: Dict[UUID, Dict[str, Any]] = {}
        
    def _default_config(self) -> Dict[str, Any]:
        """Default monitoring configuration."""
        return {
            'monitoring_interval_minutes': 15,
            'baseline_update_frequency_days': 7,
            'history_retention_days': 90,
            'auto_alert': True,
            'auto_baseline_update': True,
            'degradation_detection': {
                'lookback_days': 30,
                'min_samples': 10,
                'continuous_monitoring_hours': 24
            },
            'thresholds': {
                'default_warning_percentage': 10.0,
                'default_critical_percentage': 20.0,
                'efficiency_warning_percentage': 25.0,
                'efficiency_critical_percentage': 50.0
            }
        }
    
    async def add_model_monitoring(
        self,
        model_id: UUID,
        monitoring_config: Optional[Dict[str, Any]] = None,
        custom_thresholds: Optional[Dict[str, MetricThreshold]] = None
    ) -> bool:
        """Add a model to performance monitoring.
        
        Args:
            model_id: Model to monitor
            monitoring_config: Model-specific monitoring config
            custom_thresholds: Custom threshold configurations
            
        Returns:
            True if added successfully
        """
        try:
            # Validate model exists
            model = await self.model_repository.get_by_id(model_id)
            if not model:
                logger.error(f"Model {model_id} not found")
                return False
            
            # Store monitoring configuration
            self._monitored_models[model_id] = {
                'config': monitoring_config or {},
                'added_at': datetime.utcnow(),
                'last_check': None,
                'alert_count': 0,
                'degradation_count': 0
            }
            
            # Store custom thresholds if provided
            if custom_thresholds:
                await self.performance_repository.store_threshold_config(model_id, custom_thresholds)
            
            logger.info(f"Added model {model_id} to performance monitoring")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add model {model_id} to monitoring: {e}")
            return False
    
    async def remove_model_monitoring(self, model_id: UUID) -> bool:
        """Remove a model from performance monitoring.
        
        Args:
            model_id: Model to remove
            
        Returns:
            True if removed successfully
        """
        try:
            if model_id in self._monitored_models:
                del self._monitored_models[model_id]
                logger.info(f"Removed model {model_id} from performance monitoring")
                return True
            else:
                logger.warning(f"Model {model_id} not in monitoring list")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove model {model_id} from monitoring: {e}")
            return False
    
    async def record_performance_metrics(
        self,
        model_id: UUID,
        metrics: ModelPerformanceMetrics,
        timestamp: Optional[datetime] = None,
        auto_check_degradation: bool = True
    ) -> Optional[DegradationReport]:
        """Record performance metrics and optionally check for degradation.
        
        Args:
            model_id: Model identifier
            metrics: Performance metrics to record
            timestamp: Timestamp for the metrics
            auto_check_degradation: Whether to automatically check for degradation
            
        Returns:
            Degradation report if degradation detected
        """
        try:
            # Store metrics
            await self.performance_repository.store_performance_metrics(
                model_id=model_id,
                metrics=metrics,
                timestamp=timestamp or datetime.utcnow()
            )
            
            # Update monitoring tracking
            if model_id in self._monitored_models:
                self._monitored_models[model_id]['last_check'] = datetime.utcnow()
            
            # Check for degradation if enabled
            if auto_check_degradation and model_id in self._monitored_models:
                return await self.check_model_degradation(model_id, metrics)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to record performance metrics for model {model_id}: {e}")
            return None
    
    async def check_model_degradation(
        self,
        model_id: UUID,
        current_metrics: Optional[ModelPerformanceMetrics] = None
    ) -> Optional[DegradationReport]:
        """Check a model for performance degradation.
        
        Args:
            model_id: Model to check
            current_metrics: Current metrics (if not provided, will get latest)
            
        Returns:
            Degradation report if degradation detected
        """
        try:
            # Get current metrics if not provided
            if not current_metrics:
                current_metrics = await self.performance_repository.get_latest_performance_metrics(model_id)
                if not current_metrics:
                    logger.warning(f"No performance metrics found for model {model_id}")
                    return None
            
            # Get custom thresholds
            custom_thresholds = await self.performance_repository.get_threshold_config(model_id)
            
            # Detect degradation
            degradation_config = self.config['degradation_detection']
            degradations = await self.degradation_service.detect_degradation(
                model_id=model_id,
                current_metrics=current_metrics,
                custom_thresholds=custom_thresholds,
                lookback_days=degradation_config['lookback_days']
            )
            
            # Also check for continuous degradation
            continuous_degradations = await self.degradation_service.monitor_continuous_degradation(
                model_id=model_id,
                monitoring_window_hours=degradation_config['continuous_monitoring_hours'],
                min_samples=degradation_config['min_samples']
            )
            
            # Combine degradations
            all_degradations = degradations + continuous_degradations
            
            # Generate report if degradations found
            if all_degradations:
                # Store degradations
                for degradation in all_degradations:
                    await self.performance_repository.store_degradation(model_id, degradation)
                
                # Generate report
                report = await self.degradation_service.generate_degradation_report(
                    model_id=model_id,
                    degradations=all_degradations,
                    time_period_start=datetime.utcnow() - timedelta(days=degradation_config['lookback_days']),
                    time_period_end=datetime.utcnow()
                )
                
                # Store report
                await self.performance_repository.store_report(report)
                
                # Update monitoring stats
                if model_id in self._monitored_models:
                    self._monitored_models[model_id]['degradation_count'] += len(all_degradations)
                
                # Send alerts if enabled
                if self.config['auto_alert']:
                    await self._handle_degradation_alerts(model_id, all_degradations)
                
                logger.info(f"Detected {len(all_degradations)} degradations for model {model_id}")
                return report
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to check degradation for model {model_id}: {e}")
            return None
    
    async def update_model_baselines(
        self,
        model_id: UUID,
        force_update: bool = False,
        baseline_window_days: int = 30
    ) -> Dict[str, bool]:
        """Update performance baselines for a model.
        
        Args:
            model_id: Model to update baselines for
            force_update: Force update even if recently updated
            baseline_window_days: Days of data to use for baseline
            
        Returns:
            Dictionary of metric name to update success
        """
        results = {}
        
        try:
            # Get recent performance history
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=baseline_window_days)
            
            performance_history = await self.performance_repository.get_model_performance_history(
                model_id=model_id,
                start_date=start_date,
                end_date=end_date
            )
            
            if not performance_history:
                logger.warning(f"No performance history found for model {model_id}")
                return results
            
            # Extract metrics by name
            metric_groups = {}
            for metrics in performance_history:
                metrics_dict = metrics.dict()
                for metric_name, value in metrics_dict.items():
                    if value is not None:
                        if metric_name not in metric_groups:
                            metric_groups[metric_name] = []
                        metric_groups[metric_name].append(value)
            
            # Update baselines
            for metric_name, values in metric_groups.items():
                try:
                    if len(values) >= 10:  # Minimum samples for baseline
                        # Check if update is needed
                        existing_baseline = await self.performance_repository.get_baseline(model_id, metric_name)
                        
                        should_update = force_update
                        if not should_update and existing_baseline:
                            days_since_update = (datetime.utcnow() - existing_baseline.last_updated).days
                            should_update = days_since_update >= self.config['baseline_update_frequency_days']
                        
                        if should_update or not existing_baseline:
                            # Create new baseline
                            new_baseline = self._create_baseline_from_values(metric_name, values)
                            
                            if existing_baseline:
                                await self.performance_repository.update_baseline(model_id, new_baseline)
                                logger.info(f"Updated baseline for {metric_name} on model {model_id}")
                            else:
                                await self.performance_repository.store_baseline(model_id, new_baseline)
                                logger.info(f"Created baseline for {metric_name} on model {model_id}")
                            
                            results[metric_name] = True
                        else:
                            results[metric_name] = False  # Not updated (too recent)
                    else:
                        results[metric_name] = False  # Insufficient data
                        
                except Exception as e:
                    logger.error(f"Failed to update baseline for {metric_name}: {e}")
                    results[metric_name] = False
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to update baselines for model {model_id}: {e}")
            return results
    
    async def get_model_health_status(self, model_id: UUID) -> Dict[str, Any]:
        """Get comprehensive health status for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Health status dictionary
        """
        try:
            # Get basic health summary
            health_summary = await self.performance_repository.get_model_health_summary(model_id)
            
            # Get recent degradations
            recent_degradations = await self.performance_repository.get_recent_degradations(
                model_id=model_id,
                hours=24
            )
            
            # Get active alerts
            active_alerts = await self.alert_service.get_active_alerts(model_id)
            
            # Get latest metrics
            latest_metrics = await self.performance_repository.get_latest_performance_metrics(model_id)
            
            # Get baselines
            baselines = await self.performance_repository.get_all_baselines(model_id)
            
            # Calculate trend scores
            trend_analysis = await self._analyze_health_trends(model_id)
            
            # Monitoring status
            monitoring_status = self._monitored_models.get(model_id, {})
            
            health_status = {
                **health_summary,
                'monitoring_active': model_id in self._monitored_models,
                'monitoring_since': monitoring_status.get('added_at'),
                'last_check': monitoring_status.get('last_check'),
                'recent_degradations': len(recent_degradations),
                'active_alerts': len(active_alerts),
                'has_latest_metrics': latest_metrics is not None,
                'baseline_coverage': len(baselines),
                'trend_analysis': trend_analysis,
                'status_summary': self._get_status_summary(health_summary, recent_degradations, active_alerts)
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Failed to get health status for model {model_id}: {e}")
            return {'error': str(e)}
    
    async def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get system-wide monitoring dashboard data.
        
        Returns:
            Dashboard data dictionary
        """
        try:
            # System-wide health
            system_health = await self.performance_repository.get_system_wide_health()
            
            # Monitored models summary
            monitored_models_summary = []
            for model_id, monitoring_info in self._monitored_models.items():
                health_status = await self.get_model_health_status(model_id)
                monitored_models_summary.append({
                    'model_id': str(model_id),
                    'health_score': health_status.get('health_score', 0),
                    'recent_degradations': health_status.get('recent_degradations', 0),
                    'active_alerts': health_status.get('active_alerts', 0),
                    'monitoring_since': monitoring_info.get('added_at'),
                    'last_check': monitoring_info.get('last_check'),
                    'status': health_status.get('status_summary', 'unknown')
                })
            
            # Recent system-wide alerts
            all_active_alerts = await self.alert_service.get_active_alerts()
            
            dashboard = {
                'system_health': system_health,
                'monitored_models_count': len(self._monitored_models),
                'monitored_models': monitored_models_summary,
                'total_active_alerts': len(all_active_alerts),
                'alerts_by_severity': self._count_alerts_by_severity(all_active_alerts),
                'monitoring_status': {
                    'active': self._monitoring_active,
                    'interval_minutes': self.config['monitoring_interval_minutes']
                },
                'last_updated': datetime.utcnow()
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Failed to generate monitoring dashboard: {e}")
            return {'error': str(e)}
    
    async def start_continuous_monitoring(self) -> None:
        """Start continuous monitoring of all registered models."""
        if self._monitoring_active:
            logger.warning("Continuous monitoring already active")
            return
        
        self._monitoring_active = True
        logger.info("Started continuous performance monitoring")
        
        # Run monitoring loop
        asyncio.create_task(self._monitoring_loop())
    
    async def stop_continuous_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self._monitoring_active = False
        logger.info("Stopped continuous performance monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                await self._run_monitoring_cycle()
                
                # Wait for next cycle
                interval_seconds = self.config['monitoring_interval_minutes'] * 60
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _run_monitoring_cycle(self) -> None:
        """Run a single monitoring cycle."""
        logger.debug("Running monitoring cycle")
        
        for model_id in list(self._monitored_models.keys()):
            try:
                # Check for degradation
                await self.check_model_degradation(model_id)
                
                # Update baselines if configured
                if self.config['auto_baseline_update']:
                    await self.update_model_baselines(model_id)
                
            except Exception as e:
                logger.error(f"Error monitoring model {model_id}: {e}")
        
        # Cleanup old data
        await self._cleanup_old_data()
    
    async def _handle_degradation_alerts(
        self,
        model_id: UUID,
        degradations: List[PerformanceDegradation]
    ) -> None:
        """Handle alerts for detected degradations."""
        try:
            alerts = await self.alert_service.process_degradations(
                model_id=model_id,
                degradations=degradations,
                send_alerts=True
            )
            
            # Update monitoring stats
            if model_id in self._monitored_models:
                self._monitored_models[model_id]['alert_count'] += len(alerts)
            
            logger.info(f"Sent {len(alerts)} alerts for model {model_id}")
            
        except Exception as e:
            logger.error(f"Failed to handle alerts for model {model_id}: {e}")
    
    def _create_baseline_from_values(self, metric_name: str, values: List[float]) -> PerformanceBaseline:
        """Create a baseline from metric values."""
        import numpy as np
        
        # Remove outliers using IQR method
        np_values = np.array(values)
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
            is_stable=std_val <= (mean_val * 0.1) if mean_val != 0 else True
        )
    
    async def _analyze_health_trends(self, model_id: UUID) -> Dict[str, Any]:
        """Analyze health trends for a model."""
        try:
            # Get degradation trends for key metrics
            key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'rmse', 'r2_score']
            trends = {}
            
            for metric in key_metrics:
                degradation_trend = await self.performance_repository.get_degradation_trends(
                    model_id=model_id,
                    metric_name=metric,
                    days=30
                )
                
                if degradation_trend:
                    trends[metric] = {
                        'degradation_count': len(degradation_trend),
                        'latest_severity': degradation_trend[-1].severity.value if degradation_trend else None,
                        'trend_direction': 'worsening' if len(degradation_trend) > 1 else 'stable'
                    }
            
            return {
                'metric_trends': trends,
                'overall_trend': 'improving' if not trends else 'degrading' if any(
                    t['trend_direction'] == 'worsening' for t in trends.values()
                ) else 'stable'
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze health trends for model {model_id}: {e}")
            return {}
    
    def _get_status_summary(
        self,
        health_summary: Dict[str, Any],
        recent_degradations: List[PerformanceDegradation],
        active_alerts: List[Any]
    ) -> str:
        """Get status summary string."""
        health_score = health_summary.get('health_score', 1.0)
        
        if any(d.severity == DegradationSeverity.CRITICAL for d in recent_degradations):
            return 'critical'
        elif any(d.severity == DegradationSeverity.HIGH for d in recent_degradations):
            return 'degraded'
        elif len(active_alerts) > 0:
            return 'warning'
        elif health_score >= 0.9:
            return 'healthy'
        elif health_score >= 0.7:
            return 'fair'
        else:
            return 'poor'
    
    def _count_alerts_by_severity(self, alerts: List[Any]) -> Dict[str, int]:
        """Count alerts by severity level."""
        counts = {severity.value: 0 for severity in DegradationSeverity}
        for alert in alerts:
            if hasattr(alert, 'alert_level'):
                counts[alert.alert_level.value] += 1
        return counts
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old monitoring data."""
        try:
            retention_days = self.config['history_retention_days']
            cleanup_summary = await self.performance_repository.cleanup_old_data(retention_days)
            
            if any(cleanup_summary.values()):
                logger.info(f"Cleaned up old data: {cleanup_summary}")
                
        except Exception as e:
            logger.error(f"Failed to clean up old data: {e}")