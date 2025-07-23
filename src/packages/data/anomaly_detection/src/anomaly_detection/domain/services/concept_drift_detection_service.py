"""Advanced concept drift detection service with multiple detection methods."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import numpy.typing as npt
from collections import deque
import warnings

from ..entities.detection_result import DetectionResult
from ...infrastructure.logging import get_logger
from ...infrastructure.monitoring.model_performance_monitor import get_model_performance_monitor

logger = get_logger(__name__)


class DriftDetectionMethod(Enum):
    """Available drift detection methods."""
    STATISTICAL_DISTANCE = "statistical_distance"
    POPULATION_STABILITY_INDEX = "population_stability_index"
    JENSEN_SHANNON_DIVERGENCE = "jensen_shannon_divergence"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    DISTRIBUTION_SHIFT = "distribution_shift"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    PREDICTION_DRIFT = "prediction_drift"
    FEATURE_IMPORTANCE_DRIFT = "feature_importance_drift"


class DriftSeverity(Enum):
    """Drift severity levels."""
    NO_DRIFT = "no_drift"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis."""
    method: DriftDetectionMethod
    drift_detected: bool
    drift_score: float
    severity: DriftSeverity
    p_value: Optional[float]
    affected_features: List[str]
    threshold: float
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]
    
    @property
    def summary(self) -> str:
        """Get a human-readable summary of the drift detection result."""
        status = "DETECTED" if self.drift_detected else "NOT DETECTED"
        return (f"Drift {status} using {self.method.value} "
               f"(score: {self.drift_score:.4f}, severity: {self.severity.value})")


@dataclass
class DriftAnalysisReport:
    """Comprehensive drift analysis report across multiple methods."""
    model_id: str
    timestamp: datetime
    reference_period: Tuple[datetime, datetime]
    current_period: Tuple[datetime, datetime]
    detection_results: List[DriftDetectionResult]
    overall_drift_detected: bool
    overall_severity: DriftSeverity
    consensus_score: float
    recommendations: List[str]
    
    def get_results_by_method(self, method: DriftDetectionMethod) -> Optional[DriftDetectionResult]:
        """Get results for a specific detection method."""
        for result in self.detection_results:
            if result.method == method:
                return result
        return None
    
    def get_detected_drifts(self) -> List[DriftDetectionResult]:
        """Get only the results where drift was detected."""
        return [r for r in self.detection_results if r.drift_detected]


class ConceptDriftDetectionService:
    """Advanced concept drift detection service with multiple detection methods."""
    
    def __init__(
        self,
        window_size: int = 1000,
        reference_window_size: int = 2000,
        drift_threshold: float = 0.05,
        min_samples: int = 100
    ):
        """Initialize concept drift detection service.
        
        Args:
            window_size: Size of the current window for comparison
            reference_window_size: Size of the reference window
            drift_threshold: Default threshold for drift detection
            min_samples: Minimum samples required for drift detection
        """
        self.window_size = window_size
        self.reference_window_size = reference_window_size
        self.drift_threshold = drift_threshold
        self.min_samples = min_samples
        
        # Data storage for drift analysis
        self._reference_data: Dict[str, deque] = {}
        self._current_data: Dict[str, deque] = {}
        self._prediction_history: Dict[str, deque] = {}
        self._performance_history: Dict[str, deque] = {}
        
        # Monitoring integration
        self.monitor = get_model_performance_monitor()
        
        logger.info("Concept drift detection service initialized",
                   window_size=window_size,
                   reference_window_size=reference_window_size,
                   drift_threshold=drift_threshold)
    
    def add_reference_data(
        self,
        model_id: str,
        data: npt.NDArray[np.floating],
        predictions: Optional[npt.NDArray[np.integer]] = None,
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Add reference data for drift detection.
        
        Args:
            model_id: Model identifier
            data: Reference feature data
            predictions: Reference predictions (optional)
            performance_metrics: Reference performance metrics (optional)
        """
        if model_id not in self._reference_data:
            self._reference_data[model_id] = deque(maxlen=self.reference_window_size)
            self._prediction_history[model_id] = deque(maxlen=self.reference_window_size)
            self._performance_history[model_id] = deque(maxlen=self.reference_window_size)
        
        # Store data samples
        for sample in data:
            self._reference_data[model_id].append(sample)
        
        # Store predictions if provided
        if predictions is not None:
            for pred in predictions:
                self._prediction_history[model_id].append(pred)
        
        # Store performance metrics if provided
        if performance_metrics is not None:
            self._performance_history[model_id].append(performance_metrics)
        
        logger.debug("Added reference data for drift detection",
                    model_id=model_id,
                    data_samples=len(data),
                    total_reference_samples=len(self._reference_data[model_id]))
    
    def add_current_data(
        self,
        model_id: str,
        data: npt.NDArray[np.floating],
        predictions: Optional[npt.NDArray[np.integer]] = None,
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Add current data for drift detection.
        
        Args:
            model_id: Model identifier
            data: Current feature data
            predictions: Current predictions (optional)
            performance_metrics: Current performance metrics (optional)
        """
        if model_id not in self._current_data:
            self._current_data[model_id] = deque(maxlen=self.window_size)
        
        # Store data samples
        for sample in data:
            self._current_data[model_id].append(sample)
        
        logger.debug("Added current data for drift detection",
                    model_id=model_id,
                    data_samples=len(data),
                    total_current_samples=len(self._current_data[model_id]))
    
    def detect_drift(
        self,
        model_id: str,
        methods: Optional[List[DriftDetectionMethod]] = None,
        custom_thresholds: Optional[Dict[DriftDetectionMethod, float]] = None
    ) -> DriftAnalysisReport:
        """Detect concept drift using multiple methods.
        
        Args:
            model_id: Model identifier
            methods: List of detection methods to use (default: all available)
            custom_thresholds: Custom thresholds for specific methods
            
        Returns:
            Comprehensive drift analysis report
        """
        if methods is None:
            methods = list(DriftDetectionMethod)
        
        if custom_thresholds is None:
            custom_thresholds = {}
        
        # Check if we have sufficient data
        if not self._has_sufficient_data(model_id):
            logger.warning("Insufficient data for drift detection",
                          model_id=model_id,
                          reference_samples=len(self._reference_data.get(model_id, [])),
                          current_samples=len(self._current_data.get(model_id, [])))
            return self._create_empty_report(model_id)
        
        # Run detection methods
        detection_results = []
        
        for method in methods:
            try:
                threshold = custom_thresholds.get(method, self.drift_threshold)
                result = self._run_detection_method(model_id, method, threshold)
                detection_results.append(result)
                
                logger.debug("Drift detection method completed",
                           model_id=model_id,
                           method=method.value,
                           drift_detected=result.drift_detected,
                           drift_score=result.drift_score)
                
            except Exception as e:
                logger.error("Drift detection method failed",
                           model_id=model_id,
                           method=method.value,
                           error=str(e))
                continue
        
        # Generate comprehensive report
        report = self._generate_analysis_report(model_id, detection_results)
        
        # Record drift metrics for monitoring
        self._record_drift_metrics(model_id, report)
        
        logger.info("Drift detection analysis completed",
                   model_id=model_id,
                   methods_used=len(detection_results),
                   overall_drift_detected=report.overall_drift_detected,
                   overall_severity=report.overall_severity.value,
                   consensus_score=report.consensus_score)
        
        return report
    
    def _has_sufficient_data(self, model_id: str) -> bool:
        """Check if we have sufficient data for drift detection."""
        ref_data = self._reference_data.get(model_id, deque())
        curr_data = self._current_data.get(model_id, deque())
        
        return (len(ref_data) >= self.min_samples and 
                len(curr_data) >= self.min_samples)
    
    def _run_detection_method(
        self,
        model_id: str,
        method: DriftDetectionMethod,
        threshold: float
    ) -> DriftDetectionResult:
        """Run a specific drift detection method."""
        
        if method == DriftDetectionMethod.STATISTICAL_DISTANCE:
            return self._detect_statistical_distance_drift(model_id, threshold)
        elif method == DriftDetectionMethod.POPULATION_STABILITY_INDEX:
            return self._detect_psi_drift(model_id, threshold)
        elif method == DriftDetectionMethod.JENSEN_SHANNON_DIVERGENCE:
            return self._detect_js_divergence_drift(model_id, threshold)
        elif method == DriftDetectionMethod.KOLMOGOROV_SMIRNOV:
            return self._detect_ks_drift(model_id, threshold)
        elif method == DriftDetectionMethod.DISTRIBUTION_SHIFT:
            return self._detect_distribution_shift(model_id, threshold)
        elif method == DriftDetectionMethod.PERFORMANCE_DEGRADATION:
            return self._detect_performance_degradation(model_id, threshold)
        elif method == DriftDetectionMethod.PREDICTION_DRIFT:
            return self._detect_prediction_drift(model_id, threshold)
        elif method == DriftDetectionMethod.FEATURE_IMPORTANCE_DRIFT:
            return self._detect_feature_importance_drift(model_id, threshold)
        else:
            raise ValueError(f"Unknown drift detection method: {method}")
    
    def _detect_statistical_distance_drift(
        self,
        model_id: str,
        threshold: float
    ) -> DriftDetectionResult:
        """Detect drift using statistical distance measures."""
        ref_data = np.array(list(self._reference_data[model_id]))
        curr_data = np.array(list(self._current_data[model_id]))
        
        # Calculate statistical distances for each feature
        n_features = ref_data.shape[1]
        feature_distances = []
        affected_features = []
        
        for i in range(n_features):
            ref_feature = ref_data[:, i]
            curr_feature = curr_data[:, i]
            
            # Calculate Wasserstein distance (approximation)
            distance = self._wasserstein_distance(ref_feature, curr_feature)
            feature_distances.append(distance)
            
            if distance > threshold:
                affected_features.append(f"feature_{i}")
        
        # Overall drift score (mean of feature distances)
        drift_score = np.mean(feature_distances)
        drift_detected = drift_score > threshold
        severity = self._calculate_severity(drift_score, threshold)
        
        return DriftDetectionResult(
            method=DriftDetectionMethod.STATISTICAL_DISTANCE,
            drift_detected=drift_detected,
            drift_score=drift_score,
            severity=severity,
            p_value=None,
            affected_features=affected_features,
            threshold=threshold,
            confidence=min(0.95, 1.0 - (threshold / max(drift_score, threshold))),
            timestamp=datetime.utcnow(),
            metadata={
                "feature_distances": feature_distances,
                "n_features": n_features
            }
        )
    
    def _detect_psi_drift(
        self,
        model_id: str,
        threshold: float
    ) -> DriftDetectionResult:
        """Detect drift using Population Stability Index (PSI)."""
        ref_data = np.array(list(self._reference_data[model_id]))
        curr_data = np.array(list(self._current_data[model_id]))
        
        n_features = ref_data.shape[1]
        psi_scores = []
        affected_features = []
        
        for i in range(n_features):
            ref_feature = ref_data[:, i]
            curr_feature = curr_data[:, i]
            
            # Calculate PSI for this feature
            psi = self._calculate_psi(ref_feature, curr_feature)
            psi_scores.append(psi)
            
            # PSI interpretation: >0.1 = some shift, >0.25 = significant shift
            feature_threshold = threshold * 5  # Scale threshold for PSI
            if psi > feature_threshold:
                affected_features.append(f"feature_{i}")
        
        # Overall PSI score
        drift_score = np.mean(psi_scores)
        drift_detected = drift_score > threshold * 5  # Scale threshold for PSI
        severity = self._calculate_severity(drift_score / 5, threshold)  # Normalize for severity
        
        return DriftDetectionResult(
            method=DriftDetectionMethod.POPULATION_STABILITY_INDEX,
            drift_detected=drift_detected,
            drift_score=drift_score,
            severity=severity,
            p_value=None,
            affected_features=affected_features,
            threshold=threshold * 5,
            confidence=0.90,
            timestamp=datetime.utcnow(),
            metadata={
                "psi_scores": psi_scores,
                "interpretation": "PSI > 0.1: some shift, PSI > 0.25: significant shift"
            }
        )
    
    def _detect_js_divergence_drift(
        self,
        model_id: str,
        threshold: float
    ) -> DriftDetectionResult:
        """Detect drift using Jensen-Shannon divergence."""
        ref_data = np.array(list(self._reference_data[model_id]))
        curr_data = np.array(list(self._current_data[model_id]))
        
        n_features = ref_data.shape[1]
        js_scores = []
        affected_features = []
        
        for i in range(n_features):
            ref_feature = ref_data[:, i]
            curr_feature = curr_data[:, i]
            
            # Calculate Jensen-Shannon divergence
            js_div = self._jensen_shannon_divergence(ref_feature, curr_feature)
            js_scores.append(js_div)
            
            if js_div > threshold:
                affected_features.append(f"feature_{i}")
        
        drift_score = np.mean(js_scores)
        drift_detected = drift_score > threshold
        severity = self._calculate_severity(drift_score, threshold)
        
        return DriftDetectionResult(
            method=DriftDetectionMethod.JENSEN_SHANNON_DIVERGENCE,
            drift_detected=drift_detected,
            drift_score=drift_score,
            severity=severity,
            p_value=None,
            affected_features=affected_features,
            threshold=threshold,
            confidence=0.92,
            timestamp=datetime.utcnow(),
            metadata={
                "js_divergences": js_scores,
                "max_js_divergence": 1.0  # JS divergence is bounded [0, 1]
            }
        )
    
    def _detect_ks_drift(
        self,
        model_id: str,
        threshold: float
    ) -> DriftDetectionResult:
        """Detect drift using Kolmogorov-Smirnov test."""
        ref_data = np.array(list(self._reference_data[model_id]))
        curr_data = np.array(list(self._current_data[model_id]))
        
        n_features = ref_data.shape[1]
        ks_statistics = []
        p_values = []
        affected_features = []
        
        try:
            from scipy import stats
            
            for i in range(n_features):
                ref_feature = ref_data[:, i]
                curr_feature = curr_data[:, i]
                
                # Perform KS test
                ks_stat, p_val = stats.ks_2samp(ref_feature, curr_feature)
                ks_statistics.append(ks_stat)
                p_values.append(p_val)
                
                # Drift detected if p-value < threshold (typically 0.05)
                if p_val < threshold:
                    affected_features.append(f"feature_{i}")
            
            # Overall drift assessment
            min_p_value = min(p_values) if p_values else 1.0
            drift_detected = min_p_value < threshold
            drift_score = 1.0 - min_p_value  # Convert p-value to score
            severity = self._calculate_severity(drift_score, 1.0 - threshold)
            
            return DriftDetectionResult(
                method=DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
                drift_detected=drift_detected,
                drift_score=drift_score,
                severity=severity,
                p_value=min_p_value,
                affected_features=affected_features,
                threshold=threshold,
                confidence=0.95,
                timestamp=datetime.utcnow(),
                metadata={
                    "ks_statistics": ks_statistics,
                    "p_values": p_values,
                    "test_type": "two_sample_ks"
                }
            )
            
        except ImportError:
            logger.warning("SciPy not available for KS test, using approximation")
            return self._detect_statistical_distance_drift(model_id, threshold)
    
    def _detect_distribution_shift(
        self,
        model_id: str,
        threshold: float
    ) -> DriftDetectionResult:
        """Detect drift using distribution shift analysis."""
        ref_data = np.array(list(self._reference_data[model_id]))
        curr_data = np.array(list(self._current_data[model_id]))
        
        # Calculate distribution statistics
        ref_means = np.mean(ref_data, axis=0)
        curr_means = np.mean(curr_data, axis=0)
        ref_stds = np.std(ref_data, axis=0)
        curr_stds = np.std(curr_data, axis=0)
        
        # Calculate shifts in mean and variance
        mean_shifts = np.abs(curr_means - ref_means) / (ref_stds + 1e-8)
        std_shifts = np.abs(curr_stds - ref_stds) / (ref_stds + 1e-8)
        
        # Combined shift score
        shift_scores = (mean_shifts + std_shifts) / 2
        
        affected_features = []
        for i, score in enumerate(shift_scores):
            if score > threshold:
                affected_features.append(f"feature_{i}")
        
        drift_score = np.mean(shift_scores)
        drift_detected = drift_score > threshold
        severity = self._calculate_severity(drift_score, threshold)
        
        return DriftDetectionResult(
            method=DriftDetectionMethod.DISTRIBUTION_SHIFT,
            drift_detected=drift_detected,
            drift_score=drift_score,
            severity=severity,
            p_value=None,
            affected_features=affected_features,
            threshold=threshold,
            confidence=0.88,
            timestamp=datetime.utcnow(),
            metadata={
                "mean_shifts": mean_shifts.tolist(),
                "std_shifts": std_shifts.tolist(),
                "ref_means": ref_means.tolist(),
                "curr_means": curr_means.tolist()
            }
        )
    
    def _detect_performance_degradation(
        self,
        model_id: str,
        threshold: float
    ) -> DriftDetectionResult:
        """Detect drift based on model performance degradation."""
        if model_id not in self._performance_history or len(self._performance_history[model_id]) < 2:
            return DriftDetectionResult(
                method=DriftDetectionMethod.PERFORMANCE_DEGRADATION,
                drift_detected=False,
                drift_score=0.0,
                severity=DriftSeverity.NO_DRIFT,
                p_value=None,
                affected_features=[],
                threshold=threshold,
                confidence=0.5,
                timestamp=datetime.utcnow(),
                metadata={"error": "Insufficient performance history"}
            )
        
        # Get recent performance metrics
        perf_history = list(self._performance_history[model_id])
        recent_metrics = perf_history[-10:]  # Last 10 measurements
        baseline_metrics = perf_history[:len(perf_history)//2]  # First half as baseline
        
        # Calculate performance degradation
        performance_drops = []
        affected_metrics = []
        
        for metric_name in ["accuracy", "precision", "recall", "f1_score"]:
            if baseline_metrics and recent_metrics:
                baseline_values = [m.get(metric_name, 0) for m in baseline_metrics if metric_name in m]
                recent_values = [m.get(metric_name, 0) for m in recent_metrics if metric_name in m]
                
                if baseline_values and recent_values:
                    baseline_mean = np.mean(baseline_values)
                    recent_mean = np.mean(recent_values)
                    
                    if baseline_mean > 0:
                        drop = (baseline_mean - recent_mean) / baseline_mean
                        performance_drops.append(max(0, drop))  # Only positive drops
                        
                        if drop > threshold:
                            affected_metrics.append(metric_name)
        
        if not performance_drops:
            drift_score = 0.0
        else:
            drift_score = max(performance_drops)
        
        drift_detected = drift_score > threshold
        severity = self._calculate_severity(drift_score, threshold)
        
        return DriftDetectionResult(
            method=DriftDetectionMethod.PERFORMANCE_DEGRADATION,
            drift_detected=drift_detected,
            drift_score=drift_score,
            severity=severity,
            p_value=None,
            affected_features=affected_metrics,
            threshold=threshold,
            confidence=0.85,
            timestamp=datetime.utcnow(),
            metadata={
                "performance_drops": performance_drops,
                "baseline_period_size": len(baseline_metrics),
                "recent_period_size": len(recent_metrics)
            }
        )
    
    def _detect_prediction_drift(
        self,
        model_id: str,
        threshold: float
    ) -> DriftDetectionResult:
        """Detect drift in prediction patterns."""
        if model_id not in self._prediction_history or len(self._prediction_history[model_id]) < self.min_samples:
            return DriftDetectionResult(
                method=DriftDetectionMethod.PREDICTION_DRIFT,
                drift_detected=False,
                drift_score=0.0,
                severity=DriftSeverity.NO_DRIFT,
                p_value=None,
                affected_features=[],
                threshold=threshold,
                confidence=0.5,
                timestamp=datetime.utcnow(),
                metadata={"error": "Insufficient prediction history"}
            )
        
        # Get prediction history
        pred_history = list(self._prediction_history[model_id])
        
        # Split into reference and current periods
        split_point = len(pred_history) // 2
        ref_predictions = pred_history[:split_point]
        curr_predictions = pred_history[split_point:]
        
        # Calculate prediction statistics
        ref_anomaly_rate = np.mean([1 if p == -1 else 0 for p in ref_predictions])
        curr_anomaly_rate = np.mean([1 if p == -1 else 0 for p in curr_predictions])
        
        # Calculate drift in anomaly rate
        if ref_anomaly_rate > 0:
            drift_score = abs(curr_anomaly_rate - ref_anomaly_rate) / ref_anomaly_rate
        else:
            drift_score = curr_anomaly_rate
        
        drift_detected = drift_score > threshold
        severity = self._calculate_severity(drift_score, threshold)
        
        affected_features = ["anomaly_rate"] if drift_detected else []
        
        return DriftDetectionResult(
            method=DriftDetectionMethod.PREDICTION_DRIFT,
            drift_detected=drift_detected,
            drift_score=drift_score,
            severity=severity,
            p_value=None,
            affected_features=affected_features,
            threshold=threshold,
            confidence=0.80,
            timestamp=datetime.utcnow(),
            metadata={
                "ref_anomaly_rate": ref_anomaly_rate,
                "curr_anomaly_rate": curr_anomaly_rate,
                "ref_period_size": len(ref_predictions),
                "curr_period_size": len(curr_predictions)
            }
        )
    
    def _detect_feature_importance_drift(
        self,
        model_id: str,
        threshold: float
    ) -> DriftDetectionResult:
        """Detect drift in feature importance patterns."""
        # This is a placeholder for feature importance drift detection
        # In a real implementation, this would require access to model internals
        # or separate feature importance tracking
        
        return DriftDetectionResult(
            method=DriftDetectionMethod.FEATURE_IMPORTANCE_DRIFT,
            drift_detected=False,
            drift_score=0.0,
            severity=DriftSeverity.NO_DRIFT,
            p_value=None,
            affected_features=[],
            threshold=threshold,
            confidence=0.60,
            timestamp=datetime.utcnow(),
            metadata={
                "note": "Feature importance drift detection requires model internals access",
                "implementation": "placeholder"
            }
        )
    
    def _wasserstein_distance(self, x: npt.NDArray, y: npt.NDArray) -> float:
        """Calculate approximated Wasserstein distance between two distributions."""
        # Simple approximation using sorted samples
        x_sorted = np.sort(x)
        y_sorted = np.sort(y)
        
        # Interpolate to same length
        min_len = min(len(x_sorted), len(y_sorted))
        x_interp = np.interp(np.linspace(0, 1, min_len), 
                           np.linspace(0, 1, len(x_sorted)), x_sorted)
        y_interp = np.interp(np.linspace(0, 1, min_len), 
                           np.linspace(0, 1, len(y_sorted)), y_sorted)
        
        return np.mean(np.abs(x_interp - y_interp))
    
    def _calculate_psi(self, ref: npt.NDArray, curr: npt.NDArray, bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI)."""
        # Create bins based on reference data
        try:
            bin_edges = np.histogram_bin_edges(ref, bins=bins)
            
            # Calculate distributions
            ref_counts, _ = np.histogram(ref, bins=bin_edges)
            curr_counts, _ = np.histogram(curr, bins=bin_edges)
            
            # Convert to proportions
            ref_props = ref_counts / len(ref)
            curr_props = curr_counts / len(curr)
            
            # Avoid division by zero
            ref_props = np.where(ref_props == 0, 1e-6, ref_props)
            curr_props = np.where(curr_props == 0, 1e-6, curr_props)
            
            # Calculate PSI
            psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
            
            return abs(psi)
            
        except Exception as e:
            logger.warning(f"Error calculating PSI: {e}")
            return 0.0
    
    def _jensen_shannon_divergence(self, x: npt.NDArray, y: npt.NDArray, bins: int = 10) -> float:
        """Calculate Jensen-Shannon divergence between two distributions."""
        try:
            # Create histograms
            min_val = min(np.min(x), np.min(y))
            max_val = max(np.max(x), np.max(y))
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            
            hist_x, _ = np.histogram(x, bins=bin_edges, density=True)
            hist_y, _ = np.histogram(y, bins=bin_edges, density=True)
            
            # Normalize to probabilities
            p = hist_x / np.sum(hist_x) if np.sum(hist_x) > 0 else np.ones(len(hist_x)) / len(hist_x)
            q = hist_y / np.sum(hist_y) if np.sum(hist_y) > 0 else np.ones(len(hist_y)) / len(hist_y)
            
            # Avoid zeros
            p = np.where(p == 0, 1e-10, p)
            q = np.where(q == 0, 1e-10, q)
            
            # Calculate JS divergence
            m = (p + q) / 2
            js_div = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
            
            return min(1.0, max(0.0, js_div))  # Bound between 0 and 1
            
        except Exception as e:
            logger.warning(f"Error calculating JS divergence: {e}")
            return 0.0
    
    def _calculate_severity(self, score: float, threshold: float) -> DriftSeverity:
        """Calculate drift severity based on score and threshold."""
        if score <= threshold:
            return DriftSeverity.NO_DRIFT
        elif score <= threshold * 2:
            return DriftSeverity.LOW
        elif score <= threshold * 4:
            return DriftSeverity.MEDIUM
        elif score <= threshold * 8:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL
    
    def _generate_analysis_report(
        self,
        model_id: str,
        detection_results: List[DriftDetectionResult]
    ) -> DriftAnalysisReport:
        """Generate comprehensive drift analysis report."""
        # Calculate consensus
        drift_detections = [r.drift_detected for r in detection_results]
        drift_scores = [r.drift_score for r in detection_results]
        
        overall_drift_detected = sum(drift_detections) > len(drift_detections) / 2
        consensus_score = sum(drift_detections) / len(drift_detections) if drift_detections else 0.0
        
        # Determine overall severity
        severities = [r.severity for r in detection_results if r.drift_detected]
        if not severities:
            overall_severity = DriftSeverity.NO_DRIFT
        else:
            # Get the most severe
            severity_order = [DriftSeverity.NO_DRIFT, DriftSeverity.LOW, 
                            DriftSeverity.MEDIUM, DriftSeverity.HIGH, DriftSeverity.CRITICAL]
            max_severity_idx = max(severity_order.index(s) for s in severities)
            overall_severity = severity_order[max_severity_idx]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(detection_results, overall_severity)
        
        return DriftAnalysisReport(
            model_id=model_id,
            timestamp=datetime.utcnow(),
            reference_period=(datetime.utcnow() - timedelta(days=7), datetime.utcnow() - timedelta(days=3)),
            current_period=(datetime.utcnow() - timedelta(days=3), datetime.utcnow()),
            detection_results=detection_results,
            overall_drift_detected=overall_drift_detected,
            overall_severity=overall_severity,
            consensus_score=consensus_score,
            recommendations=recommendations
        )
    
    def _generate_recommendations(
        self,
        detection_results: List[DriftDetectionResult],
        overall_severity: DriftSeverity
    ) -> List[str]:
        """Generate actionable recommendations based on drift detection results."""
        recommendations = []
        
        if overall_severity == DriftSeverity.NO_DRIFT:
            recommendations.append("No significant drift detected. Continue monitoring.")
        else:
            if overall_severity in [DriftSeverity.MEDIUM, DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                recommendations.append("Significant drift detected. Consider model retraining.")
                recommendations.append("Investigate data quality and collection processes.")
            
            if overall_severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                recommendations.append("URGENT: Critical drift detected. Immediate action required.")
                recommendations.append("Consider temporarily switching to a backup model.")
            
            # Method-specific recommendations
            for result in detection_results:
                if result.drift_detected:
                    if result.method == DriftDetectionMethod.PERFORMANCE_DEGRADATION:
                        recommendations.append("Performance degradation detected. Review model accuracy.")
                    elif result.method == DriftDetectionMethod.PREDICTION_DRIFT:
                        recommendations.append("Prediction patterns have changed. Validate against expected behavior.")
                    elif result.method in [DriftDetectionMethod.STATISTICAL_DISTANCE, 
                                         DriftDetectionMethod.DISTRIBUTION_SHIFT]:
                        recommendations.append("Input data distribution has shifted. Check data sources.")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _create_empty_report(self, model_id: str) -> DriftAnalysisReport:
        """Create an empty drift analysis report for insufficient data cases."""
        return DriftAnalysisReport(
            model_id=model_id,
            timestamp=datetime.utcnow(),
            reference_period=(datetime.utcnow() - timedelta(days=7), datetime.utcnow()),
            current_period=(datetime.utcnow(), datetime.utcnow()),
            detection_results=[],
            overall_drift_detected=False,
            overall_severity=DriftSeverity.NO_DRIFT,
            consensus_score=0.0,
            recommendations=["Insufficient data for drift detection. Collect more samples."]
        )
    
    def _record_drift_metrics(self, model_id: str, report: DriftAnalysisReport) -> None:
        """Record drift metrics for monitoring."""
        try:
            # Calculate aggregate drift scores
            data_drift_score = 0.0
            concept_drift_score = 0.0
            
            for result in report.detection_results:
                if result.method in [DriftDetectionMethod.STATISTICAL_DISTANCE,
                                   DriftDetectionMethod.DISTRIBUTION_SHIFT,
                                   DriftDetectionMethod.JENSEN_SHANNON_DIVERGENCE]:
                    data_drift_score = max(data_drift_score, result.drift_score)
                elif result.method in [DriftDetectionMethod.PERFORMANCE_DEGRADATION,
                                     DriftDetectionMethod.PREDICTION_DRIFT]:
                    concept_drift_score = max(concept_drift_score, result.drift_score)
            
            # Record metrics
            self.monitor.record_drift_metrics(
                model_id=model_id,
                data_drift_score=data_drift_score,
                concept_drift_score=concept_drift_score,
                drift_detected=report.overall_drift_detected,
                severity=report.overall_severity.value,
                consensus_score=report.consensus_score,
                methods_used=[r.method.value for r in report.detection_results],
                affected_features=list(set().union(*[r.affected_features for r in report.detection_results]))
            )
            
        except Exception as e:
            logger.error("Failed to record drift metrics", error=str(e))
    
    def get_drift_history(self, model_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get drift detection history for a model."""
        # This would typically query stored drift reports
        # For now, return empty list as placeholder
        return []
    
    def clear_data(self, model_id: str) -> None:
        """Clear stored data for a model."""
        if model_id in self._reference_data:
            del self._reference_data[model_id]
        if model_id in self._current_data:
            del self._current_data[model_id]
        if model_id in self._prediction_history:
            del self._prediction_history[model_id]
        if model_id in self._performance_history:
            del self._performance_history[model_id]
        
        logger.info("Cleared drift detection data", model_id=model_id)