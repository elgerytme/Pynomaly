"""Integration layer for concept drift detection with monitoring infrastructure."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import threading
from collections import defaultdict

from ...domain.services.concept_drift_detection_service import (
    ConceptDriftDetectionService,
    DriftDetectionMethod,
    DriftSeverity,
    DriftAnalysisReport
)
from ...domain.entities.detection_result import DetectionResult
from ..logging import get_logger
from .model_performance_monitor import get_model_performance_monitor
from .alerting_system import get_alerting_system

logger = get_logger(__name__)


@dataclass
class DriftMonitoringConfig:
    """Configuration for drift monitoring integration."""
    enabled: bool = True
    check_interval_minutes: int = 60
    auto_analysis_enabled: bool = True
    alert_on_drift: bool = True
    methods: List[DriftDetectionMethod] = None
    thresholds: Dict[DriftDetectionMethod, float] = None
    min_samples_before_analysis: int = 100
    reference_window_hours: int = 168  # 1 week
    current_window_hours: int = 24     # 1 day
    
    def __post_init__(self):
        if self.methods is None:
            self.methods = [
                DriftDetectionMethod.STATISTICAL_DISTANCE,
                DriftDetectionMethod.POPULATION_STABILITY_INDEX,
                DriftDetectionMethod.DISTRIBUTION_SHIFT,
                DriftDetectionMethod.PERFORMANCE_DEGRADATION
            ]
        
        if self.thresholds is None:
            self.thresholds = {
                DriftDetectionMethod.STATISTICAL_DISTANCE: 0.05,
                DriftDetectionMethod.POPULATION_STABILITY_INDEX: 0.1,
                DriftDetectionMethod.JENSEN_SHANNON_DIVERGENCE: 0.05,
                DriftDetectionMethod.KOLMOGOROV_SMIRNOV: 0.05,
                DriftDetectionMethod.DISTRIBUTION_SHIFT: 0.1,
                DriftDetectionMethod.PERFORMANCE_DEGRADATION: 0.15,
                DriftDetectionMethod.PREDICTION_DRIFT: 0.2,
                DriftDetectionMethod.FEATURE_IMPORTANCE_DRIFT: 0.1
            }


class DriftMonitoringIntegration:
    """Integration layer for automated drift monitoring and alerting."""
    
    def __init__(self, config: Optional[DriftMonitoringConfig] = None):
        """Initialize drift monitoring integration.
        
        Args:
            config: Configuration for drift monitoring
        """
        self.config = config or DriftMonitoringConfig()
        
        # Initialize services
        self.drift_service = ConceptDriftDetectionService(
            window_size=1000,
            reference_window_size=2000,
            drift_threshold=0.05,
            min_samples=self.config.min_samples_before_analysis
        )
        
        self.performance_monitor = get_model_performance_monitor()
        self.alerting_system = get_alerting_system()
        
        # Tracking state
        self._monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        self._model_data_counts: Dict[str, int] = defaultdict(int)
        self._last_analysis_times: Dict[str, datetime] = {}
        self._drift_callbacks: List[Callable[[str, DriftAnalysisReport], None]] = []
        
        logger.info("Drift monitoring integration initialized",
                   enabled=self.config.enabled,
                   check_interval_minutes=self.config.check_interval_minutes,
                   methods=len(self.config.methods))
    
    async def start_monitoring(self) -> None:
        """Start automated drift monitoring."""
        if not self.config.enabled:
            logger.info("Drift monitoring disabled in configuration")
            return
        
        if self._monitoring_active:
            logger.warning("Drift monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Started automated drift monitoring",
                   interval_minutes=self.config.check_interval_minutes)
    
    async def stop_monitoring(self) -> None:
        """Stop automated drift monitoring."""
        self._monitoring_active = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
        
        logger.info("Stopped automated drift monitoring")
    
    def add_drift_callback(self, callback: Callable[[str, DriftAnalysisReport], None]) -> None:
        """Add callback function to be called when drift is detected.
        
        Args:
            callback: Function that takes (model_id, drift_report) as arguments
        """
        self._drift_callbacks.append(callback)
        logger.debug("Added drift detection callback")
    
    def remove_drift_callback(self, callback: Callable[[str, DriftAnalysisReport], None]) -> None:
        """Remove drift detection callback."""
        if callback in self._drift_callbacks:
            self._drift_callbacks.remove(callback)
            logger.debug("Removed drift detection callback")
    
    def record_prediction_data(
        self,
        model_id: str,
        input_data: Any,
        prediction_result: DetectionResult,
        performance_metrics: Optional[Dict[str, float]] = None,
        is_reference: bool = False
    ) -> None:
        """Record prediction data for drift monitoring.
        
        Args:
            model_id: Model identifier
            input_data: Input features used for prediction
            prediction_result: Prediction results
            performance_metrics: Optional performance metrics
            is_reference: Whether this data should be used as reference
        """
        if not self.config.enabled:
            return
        
        try:
            import numpy as np
            
            # Convert input data to numpy array if needed
            if hasattr(input_data, 'values'):  # pandas DataFrame
                data_array = input_data.values
            elif hasattr(input_data, 'numpy'):  # torch tensor
                data_array = input_data.numpy()
            else:
                data_array = np.array(input_data)
            
            # Ensure 2D array
            if len(data_array.shape) == 1:
                data_array = data_array.reshape(1, -1)
            
            predictions = prediction_result.predictions
            
            with self._lock:
                if is_reference:
                    self.drift_service.add_reference_data(
                        model_id=model_id,
                        data=data_array,
                        predictions=predictions,
                        performance_metrics=performance_metrics
                    )
                else:
                    self.drift_service.add_current_data(
                        model_id=model_id,
                        data=data_array,
                        predictions=predictions,
                        performance_metrics=performance_metrics
                    )
                
                # Update data count
                self._model_data_counts[model_id] += len(data_array)
            
            logger.debug("Recorded prediction data for drift monitoring",
                        model_id=model_id,
                        data_points=len(data_array),
                        is_reference=is_reference,
                        total_count=self._model_data_counts[model_id])
            
        except Exception as e:
            logger.error("Failed to record prediction data for drift monitoring",
                        model_id=model_id,
                        error=str(e))
    
    async def analyze_drift(
        self,
        model_id: str,
        force_analysis: bool = False,
        custom_methods: Optional[List[DriftDetectionMethod]] = None,
        custom_thresholds: Optional[Dict[DriftDetectionMethod, float]] = None
    ) -> Optional[DriftAnalysisReport]:
        """Analyze drift for a specific model.
        
        Args:
            model_id: Model identifier
            force_analysis: Force analysis even if insufficient data
            custom_methods: Override configured methods
            custom_thresholds: Override configured thresholds
            
        Returns:
            Drift analysis report or None if insufficient data
        """
        if not self.config.enabled:
            logger.warning("Drift monitoring is disabled")
            return None
        
        try:
            # Check if we have enough data for analysis
            if not force_analysis and self._model_data_counts[model_id] < self.config.min_samples_before_analysis:
                logger.debug("Insufficient data for drift analysis",
                           model_id=model_id,
                           current_count=self._model_data_counts[model_id],
                           required_count=self.config.min_samples_before_analysis)
                return None
            
            # Use custom parameters or fall back to config
            methods = custom_methods or self.config.methods
            thresholds = custom_thresholds or self.config.thresholds
            
            # Perform drift analysis
            logger.info("Starting drift analysis", model_id=model_id, methods=len(methods))
            
            report = self.drift_service.detect_drift(
                model_id=model_id,
                methods=methods,
                custom_thresholds=thresholds
            )
            
            # Update last analysis time
            self._last_analysis_times[model_id] = datetime.utcnow()
            
            # Process results
            await self._process_drift_report(model_id, report)
            
            logger.info("Drift analysis completed",
                       model_id=model_id,
                       drift_detected=report.overall_drift_detected,
                       severity=report.overall_severity.value,
                       consensus_score=report.consensus_score)
            
            return report
            
        except Exception as e:
            logger.error("Failed to analyze drift",
                        model_id=model_id,
                        error=str(e))
            return None
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop that periodically checks for drift."""
        logger.info("Started drift monitoring loop")
        
        try:
            while self._monitoring_active:
                await asyncio.sleep(self.config.check_interval_minutes * 60)
                
                if not self._monitoring_active:
                    break
                
                try:
                    await self._periodic_drift_check()
                except Exception as e:
                    logger.error("Error in periodic drift check", error=str(e))
                    
        except asyncio.CancelledError:
            logger.info("Drift monitoring loop cancelled")
            raise
        except Exception as e:
            logger.error("Drift monitoring loop failed", error=str(e))
        finally:
            self._monitoring_active = False
    
    async def _periodic_drift_check(self) -> None:
        """Perform periodic drift check for all monitored models."""
        if not self.config.auto_analysis_enabled:
            return
        
        # Get models that need analysis
        models_to_check = []
        current_time = datetime.utcnow()
        
        with self._lock:
            for model_id, count in self._model_data_counts.items():
                if count < self.config.min_samples_before_analysis:
                    continue
                
                last_analysis = self._last_analysis_times.get(model_id)
                if (last_analysis is None or 
                    current_time - last_analysis >= timedelta(minutes=self.config.check_interval_minutes)):
                    models_to_check.append(model_id)
        
        if not models_to_check:
            logger.debug("No models require drift analysis")
            return
        
        logger.info("Performing periodic drift check",
                   models_count=len(models_to_check))
        
        # Analyze drift for each model
        for model_id in models_to_check:
            try:
                await self.analyze_drift(model_id)
            except Exception as e:
                logger.error("Failed to analyze drift for model",
                           model_id=model_id,
                           error=str(e))
    
    async def _process_drift_report(self, model_id: str, report: DriftAnalysisReport) -> None:
        """Process drift analysis report (alerting, callbacks, etc.)."""
        
        # Send alerts if drift detected
        if self.config.alert_on_drift and report.overall_drift_detected:
            await self._send_drift_alert(model_id, report)
        
        # Call registered callbacks
        for callback in self._drift_callbacks:
            try:
                callback(model_id, report)
            except Exception as e:
                logger.error("Error in drift callback", 
                           model_id=model_id,
                           error=str(e))
        
        # Store report for later retrieval
        await self._store_drift_report(model_id, report)
    
    async def _send_drift_alert(self, model_id: str, report: DriftAnalysisReport) -> None:
        """Send drift detection alert."""
        try:
            severity_mapping = {
                DriftSeverity.LOW: "warning",
                DriftSeverity.MEDIUM: "warning", 
                DriftSeverity.HIGH: "critical",
                DriftSeverity.CRITICAL: "critical"
            }
            
            alert_severity = severity_mapping.get(report.overall_severity, "warning")
            
            # Create alert message
            detected_methods = [r.method.value for r in report.detection_results if r.drift_detected]
            
            message = (
                f"Concept drift detected for model '{model_id}'\n"
                f"Severity: {report.overall_severity.value.upper()}\n"
                f"Consensus Score: {report.consensus_score:.2f}\n"
                f"Detection Methods: {', '.join(detected_methods)}\n"
                f"Recommendations: {'; '.join(report.recommendations[:3])}"
            )
            
            # Send through alerting system
            await self.alerting_system.send_alert(
                title=f"Concept Drift Alert - {model_id}",
                message=message,
                severity=alert_severity,
                metadata={
                    "model_id": model_id,
                    "drift_severity": report.overall_severity.value,
                    "consensus_score": report.consensus_score,
                    "detection_methods": detected_methods,
                    "timestamp": report.timestamp.isoformat()
                }
            )
            
            logger.info("Sent drift detection alert",
                       model_id=model_id,
                       severity=report.overall_severity.value,
                       alert_severity=alert_severity)
            
        except Exception as e:
            logger.error("Failed to send drift alert",
                        model_id=model_id,
                        error=str(e))
    
    async def _store_drift_report(self, model_id: str, report: DriftAnalysisReport) -> None:
        """Store drift report for historical analysis."""
        try:
            # Convert report to dictionary for storage
            report_data = {
                "model_id": report.model_id,
                "timestamp": report.timestamp.isoformat(),
                "reference_period": {
                    "start": report.reference_period[0].isoformat(),
                    "end": report.reference_period[1].isoformat()
                },
                "current_period": {
                    "start": report.current_period[0].isoformat(),
                    "end": report.current_period[1].isoformat()
                },
                "overall_drift_detected": report.overall_drift_detected,
                "overall_severity": report.overall_severity.value,
                "consensus_score": report.consensus_score,
                "recommendations": report.recommendations,
                "detection_results": []
            }
            
            # Add detection results
            for result in report.detection_results:
                result_data = {
                    "method": result.method.value,
                    "drift_detected": result.drift_detected,
                    "drift_score": result.drift_score,
                    "severity": result.severity.value,
                    "p_value": result.p_value,
                    "affected_features": result.affected_features,
                    "threshold": result.threshold,
                    "confidence": result.confidence,
                    "timestamp": result.timestamp.isoformat(),
                    "metadata": result.metadata
                }
                report_data["detection_results"].append(result_data)
            
            # Store using performance monitor (could be extended to use dedicated storage)
            self.performance_monitor.record_drift_metrics(
                model_id=model_id,
                data_drift_score=report.consensus_score,
                concept_drift_score=report.consensus_score,
                drift_detected=report.overall_drift_detected,
                severity=report.overall_severity.value,
                report_data=report_data
            )
            
            logger.debug("Stored drift report",
                        model_id=model_id,
                        timestamp=report.timestamp)
            
        except Exception as e:
            logger.error("Failed to store drift report",
                        model_id=model_id,
                        error=str(e))
    
    def get_model_drift_status(self, model_id: str) -> Dict[str, Any]:
        """Get current drift monitoring status for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dictionary with drift monitoring status
        """
        with self._lock:
            data_count = self._model_data_counts.get(model_id, 0)
            last_analysis = self._last_analysis_times.get(model_id)
        
        return {
            "model_id": model_id,
            "monitoring_enabled": self.config.enabled,
            "data_points_collected": data_count,
            "sufficient_data": data_count >= self.config.min_samples_before_analysis,
            "last_analysis_time": last_analysis.isoformat() if last_analysis else None,
            "next_scheduled_analysis": (
                (last_analysis + timedelta(minutes=self.config.check_interval_minutes)).isoformat()
                if last_analysis else "pending"
            ),
            "configured_methods": [m.value for m in self.config.methods],
            "alert_on_drift": self.config.alert_on_drift
        }
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get overall drift monitoring summary.
        
        Returns:
            Dictionary with monitoring summary
        """
        with self._lock:
            total_models = len(self._model_data_counts)
            models_with_sufficient_data = sum(
                1 for count in self._model_data_counts.values()
                if count >= self.config.min_samples_before_analysis
            )
            total_data_points = sum(self._model_data_counts.values())
        
        return {
            "monitoring_active": self._monitoring_active,
            "config": asdict(self.config),
            "total_models_monitored": total_models,
            "models_with_sufficient_data": models_with_sufficient_data,
            "total_data_points_collected": total_data_points,
            "registered_callbacks": len(self._drift_callbacks),
            "last_check_times": {
                model_id: time.isoformat() if time else None
                for model_id, time in self._last_analysis_times.items()
            }
        }
    
    def update_config(self, **config_updates: Any) -> None:
        """Update monitoring configuration.
        
        Args:
            **config_updates: Configuration parameters to update
        """
        for key, value in config_updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info("Updated drift monitoring config",
                           parameter=key,
                           new_value=value)
            else:
                logger.warning("Unknown config parameter",
                              parameter=key)
    
    def reset_model_data(self, model_id: str) -> None:
        """Reset stored data for a model.
        
        Args:
            model_id: Model identifier
        """
        with self._lock:
            self.drift_service.clear_data(model_id)
            self._model_data_counts.pop(model_id, None)
            self._last_analysis_times.pop(model_id, None)
        
        logger.info("Reset drift monitoring data for model", model_id=model_id)
    
    def get_drift_history(self, model_id: str, hours: int = 168) -> List[Dict[str, Any]]:
        """Get drift detection history for a model.
        
        Args:
            model_id: Model identifier
            hours: Hours of history to retrieve
            
        Returns:
            List of drift analysis reports
        """
        # This would typically query stored drift reports
        # For now, return empty list as placeholder
        return self.drift_service.get_drift_history(model_id, hours)


# Global instance management
_drift_monitoring_integration: Optional[DriftMonitoringIntegration] = None


def get_drift_monitoring_integration() -> DriftMonitoringIntegration:
    """Get the global drift monitoring integration instance."""
    global _drift_monitoring_integration
    
    if _drift_monitoring_integration is None:
        _drift_monitoring_integration = DriftMonitoringIntegration()
    
    return _drift_monitoring_integration


def initialize_drift_monitoring(config: Optional[DriftMonitoringConfig] = None) -> DriftMonitoringIntegration:
    """Initialize drift monitoring integration with custom configuration."""
    global _drift_monitoring_integration
    
    _drift_monitoring_integration = DriftMonitoringIntegration(config)
    return _drift_monitoring_integration


async def start_drift_monitoring(config: Optional[DriftMonitoringConfig] = None) -> DriftMonitoringIntegration:
    """Start drift monitoring integration."""
    integration = initialize_drift_monitoring(config)
    await integration.start_monitoring()
    return integration


async def stop_drift_monitoring() -> None:
    """Stop drift monitoring integration."""
    global _drift_monitoring_integration
    
    if _drift_monitoring_integration:
        await _drift_monitoring_integration.stop_monitoring()