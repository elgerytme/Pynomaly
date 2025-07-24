"""Data pipeline monitoring with drift detection and automated retraining."""

from __future__ import annotations

import logging
import time
import asyncio
import threading
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import numpy.typing as npt

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .metrics import get_metrics_collector
from .tracing import get_tracer

logger = logging.getLogger(__name__)


class DriftStatus(Enum):
    """Data drift detection status."""
    NO_DRIFT = "no_drift"
    MINOR_DRIFT = "minor_drift"
    MAJOR_DRIFT = "major_drift"
    SEVERE_DRIFT = "severe_drift"


class ModelStatus(Enum):
    """Model performance status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class DriftDetectionConfig:
    """Configuration for drift detection."""
    statistical_threshold: float = 0.05  # p-value threshold
    psi_threshold: float = 0.1  # Population Stability Index threshold
    kl_divergence_threshold: float = 0.1
    cosine_similarity_threshold: float = 0.95
    window_size: int = 1000
    detection_interval: int = 3600  # seconds
    enable_feature_drift: bool = True
    enable_prediction_drift: bool = True
    enable_performance_drift: bool = True


@dataclass
class RetrainingConfig:
    """Configuration for automated retraining."""
    performance_threshold: float = 0.8  # Minimum acceptable performance
    drift_threshold: DriftStatus = DriftStatus.MAJOR_DRIFT
    min_data_points: int = 1000
    retrain_interval: int = 86400  # 24 hours in seconds
    max_retrain_attempts: int = 3
    validation_split: float = 0.2
    enable_auto_retrain: bool = True
    enable_champion_challenger: bool = True


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics."""
    timestamp: datetime
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    processing_time: Optional[float] = None
    data_quality_score: Optional[float] = None
    drift_score: Optional[float] = None
    drift_status: DriftStatus = DriftStatus.NO_DRIFT
    model_status: ModelStatus = ModelStatus.HEALTHY
    anomaly_rate: Optional[float] = None
    false_positive_rate: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataDriftDetector:
    """Statistical data drift detection."""
    
    def __init__(self, config: DriftDetectionConfig):
        """Initialize drift detector.
        
        Args:
            config: Drift detection configuration
        """
        self.config = config
        self.reference_data: Optional[npt.NDArray] = None
        self.reference_stats: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        
    def set_reference_data(
        self,
        data: npt.NDArray,
        feature_names: Optional[List[str]] = None
    ) -> None:
        """Set reference data for drift detection.
        
        Args:
            data: Reference dataset
            feature_names: Names of features
        """
        self.reference_data = data
        self.feature_names = feature_names or [f"feature_{i}" for i in range(data.shape[1])]
        
        # Calculate reference statistics
        self._calculate_reference_stats()
        
        logger.info(f"Reference data set with {data.shape[0]} samples, {data.shape[1]} features")
    
    def _calculate_reference_stats(self) -> None:
        """Calculate statistics for reference data."""
        if self.reference_data is None:
            return
        
        self.reference_stats = {
            "mean": np.mean(self.reference_data, axis=0),
            "std": np.std(self.reference_data, axis=0),
            "min": np.min(self.reference_data, axis=0),
            "max": np.max(self.reference_data, axis=0),
            "percentiles": {
                "25": np.percentile(self.reference_data, 25, axis=0),
                "50": np.percentile(self.reference_data, 50, axis=0),
                "75": np.percentile(self.reference_data, 75, axis=0)
            }
        }
    
    def detect_drift(
        self,
        current_data: npt.NDArray,
        method: str = "ks_test"
    ) -> Dict[str, Any]:
        """Detect drift in current data compared to reference.
        
        Args:
            current_data: Current dataset
            method: Drift detection method
            
        Returns:
            Drift detection results
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        if not SCIPY_AVAILABLE and method in ["ks_test", "mannwhitney"]:
            logger.warning("SciPy not available, falling back to statistical comparison")
            method = "statistical"
        
        results = {
            "method": method,
            "timestamp": datetime.now(),
            "overall_drift_status": DriftStatus.NO_DRIFT,
            "overall_drift_score": 0.0,
            "feature_results": {},
            "summary": {}
        }
        
        drift_scores = []
        
        for i, feature_name in enumerate(self.feature_names):
            if i >= current_data.shape[1]:
                break
            
            feature_result = self._detect_feature_drift(
                self.reference_data[:, i],
                current_data[:, i],
                feature_name,
                method
            )
            
            results["feature_results"][feature_name] = feature_result
            drift_scores.append(feature_result["drift_score"])
        
        # Calculate overall drift
        overall_drift_score = np.mean(drift_scores) if drift_scores else 0.0
        results["overall_drift_score"] = overall_drift_score
        
        # Determine overall drift status
        if overall_drift_score > 0.5:
            results["overall_drift_status"] = DriftStatus.SEVERE_DRIFT
        elif overall_drift_score > 0.3:
            results["overall_drift_status"] = DriftStatus.MAJOR_DRIFT
        elif overall_drift_score > 0.1:
            results["overall_drift_status"] = DriftStatus.MINOR_DRIFT
        
        # Summary statistics
        results["summary"] = {
            "total_features": len(self.feature_names),
            "features_with_drift": sum(1 for score in drift_scores if score > 0.1),
            "max_drift_score": max(drift_scores) if drift_scores else 0.0,
            "min_drift_score": min(drift_scores) if drift_scores else 0.0
        }
        
        return results
    
    def _detect_feature_drift(
        self,
        reference_feature: npt.NDArray,
        current_feature: npt.NDArray,
        feature_name: str,
        method: str
    ) -> Dict[str, Any]:
        """Detect drift for a single feature.
        
        Args:
            reference_feature: Reference feature data
            current_feature: Current feature data
            feature_name: Name of the feature
            method: Detection method
            
        Returns:
            Feature drift results
        """
        result = {
            "feature_name": feature_name,
            "method": method,
            "drift_score": 0.0,
            "drift_status": DriftStatus.NO_DRIFT,
            "p_value": None,
            "statistic": None,
            "metadata": {}
        }
        
        try:
            if method == "ks_test" and SCIPY_AVAILABLE:
                statistic, p_value = stats.ks_2samp(reference_feature, current_feature)
                result["statistic"] = float(statistic)
                result["p_value"] = float(p_value)
                result["drift_score"] = float(statistic)
                
            elif method == "mannwhitney" and SCIPY_AVAILABLE:
                statistic, p_value = stats.mannwhitneyu(
                    reference_feature, current_feature, alternative='two-sided'
                )
                result["statistic"] = float(statistic)
                result["p_value"] = float(p_value)
                # Normalize statistic to 0-1 range
                max_statistic = len(reference_feature) * len(current_feature)
                result["drift_score"] = float(statistic / max_statistic)
                
            elif method == "psi":
                # Population Stability Index
                drift_score = self._calculate_psi(reference_feature, current_feature)
                result["drift_score"] = drift_score
                result["statistic"] = drift_score
                
            elif method == "statistical":
                # Simple statistical comparison
                ref_mean, ref_std = np.mean(reference_feature), np.std(reference_feature)
                cur_mean, cur_std = np.mean(current_feature), np.std(current_feature)
                
                mean_diff = abs(cur_mean - ref_mean) / (ref_std + 1e-8)
                std_diff = abs(cur_std - ref_std) / (ref_std + 1e-8)
                
                drift_score = (mean_diff + std_diff) / 2
                result["drift_score"] = float(drift_score)
                result["metadata"] = {
                    "ref_mean": float(ref_mean),
                    "cur_mean": float(cur_mean),
                    "ref_std": float(ref_std),
                    "cur_std": float(cur_std),
                    "mean_diff": float(mean_diff),
                    "std_diff": float(std_diff)
                }
            
            # Determine drift status
            if result["drift_score"] > 0.3:
                result["drift_status"] = DriftStatus.SEVERE_DRIFT
            elif result["drift_score"] > 0.2:
                result["drift_status"] = DriftStatus.MAJOR_DRIFT
            elif result["drift_score"] > 0.1:
                result["drift_status"] = DriftStatus.MINOR_DRIFT
                
        except Exception as e:
            logger.error(f"Error detecting drift for feature {feature_name}: {e}")
            result["error"] = str(e)
        
        return result
    
    def _calculate_psi(
        self,
        reference: npt.NDArray,
        current: npt.NDArray,
        bins: int = 10
    ) -> float:
        """Calculate Population Stability Index.
        
        Args:
            reference: Reference feature values
            current: Current feature values
            bins: Number of bins for discretization
            
        Returns:
            PSI value
        """
        try:
            # Create bins based on reference data
            bin_edges = np.percentile(reference, np.linspace(0, 100, bins + 1))
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            
            # Calculate distributions
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            cur_counts, _ = np.histogram(current, bins=bin_edges)
            
            # Convert to proportions
            ref_props = ref_counts / len(reference)
            cur_props = cur_counts / len(current)
            
            # Add small constant to avoid log(0)
            ref_props = ref_props + 1e-8
            cur_props = cur_props + 1e-8
            
            # Calculate PSI
            psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
            
            return float(psi)
            
        except Exception as e:
            logger.error(f"Error calculating PSI: {e}")
            return 0.0


class ModelPerformanceMonitor:
    """Monitor model performance and detect degradation."""
    
    def __init__(self, config: RetrainingConfig):
        """Initialize performance monitor.
        
        Args:
            config: Retraining configuration
        """
        self.config = config
        self.performance_history: List[PipelineMetrics] = []
        self.baseline_performance: Optional[Dict[str, float]] = None
        
    def set_baseline_performance(self, metrics: Dict[str, float]) -> None:
        """Set baseline performance metrics.
        
        Args:
            metrics: Baseline performance metrics
        """
        self.baseline_performance = metrics
        logger.info(f"Baseline performance set: {metrics}")
    
    def record_performance(
        self,
        predictions: npt.NDArray,
        true_labels: Optional[npt.NDArray] = None,
        processing_time: Optional[float] = None,
        additional_metrics: Optional[Dict[str, Any]] = None
    ) -> PipelineMetrics:
        """Record model performance metrics.
        
        Args:
            predictions: Model predictions
            true_labels: True labels (if available)
            processing_time: Processing time in seconds
            additional_metrics: Additional metrics
            
        Returns:
            Pipeline metrics
        """
        metrics = PipelineMetrics(
            timestamp=datetime.now(),
            processing_time=processing_time,
            metadata=additional_metrics or {}
        )
        
        # Calculate performance metrics if labels available
        if true_labels is not None and SKLEARN_AVAILABLE:
            try:
                # Convert predictions to binary if needed
                binary_predictions = (predictions > 0.5).astype(int) if predictions.dtype == float else predictions
                binary_labels = (true_labels > 0.5).astype(int) if true_labels.dtype == float else true_labels
                
                metrics.accuracy = float(accuracy_score(binary_labels, binary_predictions))
                metrics.precision = float(precision_score(binary_labels, binary_predictions, average='weighted', zero_division=0))
                metrics.recall = float(recall_score(binary_labels, binary_predictions, average='weighted', zero_division=0))
                metrics.f1_score = float(f1_score(binary_labels, binary_predictions, average='weighted', zero_division=0))
                
                # Calculate false positive rate
                tn = np.sum((binary_labels == 0) & (binary_predictions == 0))
                fp = np.sum((binary_labels == 0) & (binary_predictions == 1))
                metrics.false_positive_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
                
            except Exception as e:
                logger.error(f"Error calculating performance metrics: {e}")
        
        # Calculate anomaly rate
        if predictions.dtype == int:
            metrics.anomaly_rate = float(np.mean(predictions))
        else:
            metrics.anomaly_rate = float(np.mean(predictions > 0.5))
        
        # Determine model status
        metrics.model_status = self._determine_model_status(metrics)
        
        # Add to history
        self.performance_history.append(metrics)
        
        # Keep only recent history (last 1000 records)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        return metrics
    
    def _determine_model_status(self, metrics: PipelineMetrics) -> ModelStatus:
        """Determine model status based on metrics.
        
        Args:
            metrics: Current metrics
            
        Returns:
            Model status
        """
        if not self.baseline_performance:
            return ModelStatus.HEALTHY
        
        # Check accuracy degradation
        if metrics.accuracy is not None:
            baseline_accuracy = self.baseline_performance.get("accuracy", 0.8)
            
            if metrics.accuracy < baseline_accuracy * 0.5:
                return ModelStatus.FAILED
            elif metrics.accuracy < baseline_accuracy * 0.7:
                return ModelStatus.CRITICAL
            elif metrics.accuracy < baseline_accuracy * 0.9:
                return ModelStatus.DEGRADED
        
        # Check false positive rate
        if metrics.false_positive_rate is not None:
            baseline_fpr = self.baseline_performance.get("false_positive_rate", 0.1)
            
            if metrics.false_positive_rate > baseline_fpr * 3:
                return ModelStatus.CRITICAL
            elif metrics.false_positive_rate > baseline_fpr * 2:
                return ModelStatus.DEGRADED
        
        return ModelStatus.HEALTHY
    
    def get_performance_trend(self, window_size: int = 100) -> Dict[str, Any]:
        """Get performance trend analysis.
        
        Args:
            window_size: Window size for trend analysis
            
        Returns:
            Trend analysis results
        """
        if len(self.performance_history) < 10:
            return {"status": "insufficient_data"}
        
        recent_metrics = self.performance_history[-window_size:]
        
        # Calculate trends
        trends = {}
        
        for metric_name in ["accuracy", "precision", "recall", "f1_score", "false_positive_rate"]:
            values = [getattr(m, metric_name) for m in recent_metrics if getattr(m, metric_name) is not None]
            
            if len(values) >= 5:
                # Simple linear trend
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                
                trends[metric_name] = {
                    "current": values[-1],
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "trend_slope": float(slope),
                    "trend_direction": "increasing" if slope > 0.001 else "decreasing" if slope < -0.001 else "stable"
                }
        
        return {
            "status": "ok",
            "window_size": len(recent_metrics),
            "trends": trends,
            "timestamp": datetime.now()
        }


class PipelineMonitor:
    """Comprehensive data pipeline monitoring system."""
    
    def __init__(
        self,
        drift_config: DriftDetectionConfig,
        retrain_config: RetrainingConfig,
        model_registry: Optional[Any] = None
    ):
        """Initialize pipeline monitor.
        
        Args:
            drift_config: Drift detection configuration
            retrain_config: Retraining configuration
            model_registry: Model registry for retraining
        """
        self.drift_config = drift_config
        self.retrain_config = retrain_config
        self.model_registry = model_registry
        
        # Components
        self.drift_detector = DataDriftDetector(drift_config)
        self.performance_monitor = ModelPerformanceMonitor(retrain_config)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Callbacks
        self.drift_callbacks: List[Callable] = []
        self.performance_callbacks: List[Callable] = []
        self.retrain_callbacks: List[Callable] = []
        
        # Metrics and tracing
        self.metrics_collector = get_metrics_collector()
        self.tracer = get_tracer()
        
        logger.info("Pipeline monitor initialized")
    
    def set_reference_data(
        self,
        data: npt.NDArray,
        feature_names: Optional[List[str]] = None
    ) -> None:
        """Set reference data for drift detection.
        
        Args:
            data: Reference dataset
            feature_names: Feature names
        """
        self.drift_detector.set_reference_data(data, feature_names)
    
    def set_baseline_performance(self, metrics: Dict[str, float]) -> None:
        """Set baseline performance metrics.
        
        Args:
            metrics: Baseline performance metrics
        """
        self.performance_monitor.set_baseline_performance(metrics)
    
    def monitor_prediction_batch(
        self,
        input_data: npt.NDArray,
        predictions: npt.NDArray,
        true_labels: Optional[npt.NDArray] = None,
        processing_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """Monitor a batch of predictions.
        
        Args:
            input_data: Input data
            predictions: Model predictions
            true_labels: True labels (if available)
            processing_time: Processing time
            
        Returns:
            Monitoring results
        """
        results = {
            "timestamp": datetime.now(),
            "batch_size": len(input_data),
            "drift_results": None,
            "performance_metrics": None,
            "recommendations": []
        }
        
        try:
            # Detect data drift
            if self.drift_config.enable_feature_drift and self.drift_detector.reference_data is not None:
                with self.tracer.trace_operation("drift_detection") if self.tracer else nullcontext():
                    drift_results = self.drift_detector.detect_drift(input_data)
                    results["drift_results"] = drift_results
                    
                    # Record drift metrics
                    if self.metrics_collector:
                        self.metrics_collector.record_data_quality(
                            dataset="current_batch",
                            dimension="drift",
                            score=1.0 - drift_results["overall_drift_score"]
                        )
                    
                    # Check for significant drift
                    if drift_results["overall_drift_status"] in [DriftStatus.MAJOR_DRIFT, DriftStatus.SEVERE_DRIFT]:
                        results["recommendations"].append("Consider model retraining due to data drift")
                        
                        # Trigger drift callbacks
                        for callback in self.drift_callbacks:
                            try:
                                callback(drift_results)
                            except Exception as e:
                                logger.error(f"Drift callback failed: {e}")
            
            # Monitor performance
            with self.tracer.trace_operation("performance_monitoring") if self.tracer else nullcontext():
                performance_metrics = self.performance_monitor.record_performance(
                    predictions, true_labels, processing_time
                )
                results["performance_metrics"] = performance_metrics
                
                # Record performance metrics
                if self.metrics_collector:
                    if performance_metrics.accuracy is not None:
                        self.metrics_collector.record_model_performance(
                            model_name="current_model",
                            metrics={
                                "accuracy": performance_metrics.accuracy,
                                "precision": performance_metrics.precision or 0.0,
                                "recall": performance_metrics.recall or 0.0,
                                "f1_score": performance_metrics.f1_score or 0.0
                            }
                        )
                
                # Check for performance degradation
                if performance_metrics.model_status in [ModelStatus.CRITICAL, ModelStatus.FAILED]:
                    results["recommendations"].append("Model performance is degraded, immediate attention required")
                elif performance_metrics.model_status == ModelStatus.DEGRADED:
                    results["recommendations"].append("Model performance is declining, consider retraining")
                
                # Trigger performance callbacks
                for callback in self.performance_callbacks:
                    try:
                        callback(performance_metrics)
                    except Exception as e:
                        logger.error(f"Performance callback failed: {e}")
            
            # Check retraining conditions
            if self._should_trigger_retraining(results):
                results["recommendations"].append("Automated retraining triggered")
                
                if self.retrain_config.enable_auto_retrain:
                    self._trigger_retraining(results)
            
        except Exception as e:
            logger.error(f"Error during pipeline monitoring: {e}")
            results["error"] = str(e)
        
        return results
    
    def _should_trigger_retraining(self, monitoring_results: Dict[str, Any]) -> bool:
        """Check if retraining should be triggered.
        
        Args:
            monitoring_results: Current monitoring results
            
        Returns:
            True if retraining should be triggered
        """
        # Check drift threshold
        drift_results = monitoring_results.get("drift_results")
        if drift_results and drift_results["overall_drift_status"] >= self.retrain_config.drift_threshold:
            return True
        
        # Check performance threshold
        performance_metrics = monitoring_results.get("performance_metrics")
        if performance_metrics:
            if performance_metrics.accuracy is not None:
                if performance_metrics.accuracy < self.retrain_config.performance_threshold:
                    return True
            
            if performance_metrics.model_status in [ModelStatus.CRITICAL, ModelStatus.FAILED]:
                return True
        
        return False
    
    def _trigger_retraining(self, monitoring_results: Dict[str, Any]) -> None:
        """Trigger automated retraining.
        
        Args:
            monitoring_results: Monitoring results that triggered retraining
        """
        logger.info("Triggering automated model retraining")
        
        # Trigger retrain callbacks
        for callback in self.retrain_callbacks:
            try:
                callback(monitoring_results)
            except Exception as e:
                logger.error(f"Retrain callback failed: {e}")
        
        # Record retraining event
        if self.metrics_collector:
            self.metrics_collector.record_task_execution(
                task_type="auto_retrain",
                duration=0.0,  # Will be updated when complete
                status="triggered"
            )
    
    def add_drift_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback for drift detection events.
        
        Args:
            callback: Callback function
        """
        self.drift_callbacks.append(callback)
    
    def add_performance_callback(self, callback: Callable[[PipelineMetrics], None]) -> None:
        """Add callback for performance monitoring events.
        
        Args:
            callback: Callback function
        """
        self.performance_callbacks.append(callback)
    
    def add_retrain_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback for retraining events.
        
        Args:
            callback: Callback function
        """
        self.retrain_callbacks.append(callback)
    
    def start_continuous_monitoring(self) -> None:
        """Start continuous monitoring in background thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.stop_event.clear()
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("Continuous pipeline monitoring started")
    
    def stop_continuous_monitoring(self) -> None:
        """Stop continuous monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self.stop_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Continuous pipeline monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self.stop_event.wait(self.drift_config.detection_interval):
            try:
                # Perform periodic checks
                self._periodic_health_check()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def _periodic_health_check(self) -> None:
        """Perform periodic health checks."""
        # Check performance trends
        trend_analysis = self.performance_monitor.get_performance_trend()
        
        if trend_analysis.get("status") == "ok":
            trends = trend_analysis.get("trends", {})
            
            for metric_name, trend_data in trends.items():
                if trend_data["trend_direction"] == "decreasing" and metric_name != "false_positive_rate":
                    logger.warning(f"Declining trend detected in {metric_name}: {trend_data['trend_slope']}")
                elif trend_data["trend_direction"] == "increasing" and metric_name == "false_positive_rate":
                    logger.warning(f"Increasing false positive rate: {trend_data['trend_slope']}")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary.
        
        Returns:
            Monitoring summary
        """
        summary = {
            "timestamp": datetime.now(),
            "monitoring_active": self.monitoring_active,
            "drift_detection": {
                "enabled": self.drift_config.enable_feature_drift,
                "reference_data_set": self.drift_detector.reference_data is not None,
                "last_detection": None
            },
            "performance_monitoring": {
                "enabled": True,
                "baseline_set": self.performance_monitor.baseline_performance is not None,
                "history_size": len(self.performance_monitor.performance_history)
            },
            "retraining": {
                "auto_enabled": self.retrain_config.enable_auto_retrain,
                "champion_challenger": self.retrain_config.enable_champion_challenger
            }
        }
        
        # Add recent performance metrics
        if self.performance_monitor.performance_history:
            latest_metrics = self.performance_monitor.performance_history[-1]
            summary["latest_performance"] = {
                "timestamp": latest_metrics.timestamp,
                "accuracy": latest_metrics.accuracy,
                "model_status": latest_metrics.model_status.value,
                "anomaly_rate": latest_metrics.anomaly_rate
            }
        
        # Add performance trends
        trend_analysis = self.performance_monitor.get_performance_trend()
        if trend_analysis.get("status") == "ok":
            summary["performance_trends"] = trend_analysis["trends"]
        
        return summary


# Utility context manager for null operations
class nullcontext:
    """Null context manager for when tracing is disabled."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass