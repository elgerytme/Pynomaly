"""Model drift detection service."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd
from scipy import stats

from pynomaly.domain.entities.drift_report import (
    DriftConfiguration,
    DriftDetectionMethod,
    DriftMonitor,
    DriftReport,
    DriftSeverity,
    DriftType,
    FeatureDrift,
)
from pynomaly.shared.protocols.repository_protocol import (
    DatasetRepositoryProtocol,
    ModelRepositoryProtocol,
)


class ModelDriftDetectionService:
    """Service for detecting and monitoring model drift."""

    def __init__(
        self,
        model_repository: ModelRepositoryProtocol,
        dataset_repository: DatasetRepositoryProtocol,
        drift_repository: Any,  # DriftRepositoryProtocol when implemented
        model_serving_service: Any,  # ModelServingService when implemented
    ):
        """Initialize the drift detection service.
        
        Args:
            model_repository: Model repository
            dataset_repository: Dataset repository
            drift_repository: Drift repository
            model_serving_service: Model serving service
        """
        self.model_repository = model_repository
        self.dataset_repository = dataset_repository
        self.drift_repository = drift_repository
        self.model_serving_service = model_serving_service

    async def create_drift_monitor(
        self,
        model_id: UUID,
        name: str,
        configuration: DriftConfiguration,
        created_by: str,
        description: str | None = None,
        monitoring_frequency: str = "daily",
        alert_recipients: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> DriftMonitor:
        """Create a drift monitor for a model.
        
        Args:
            model_id: Model to monitor
            name: Monitor name
            configuration: Drift detection configuration
            created_by: User creating the monitor
            description: Monitor description
            monitoring_frequency: Monitoring frequency
            alert_recipients: Alert recipients
            tags: Monitor tags
            
        Returns:
            Created drift monitor
        """
        # Validate model exists
        model = await self.model_repository.get_by_id(model_id)
        if not model:
            raise ValueError(f"Model {model_id} does not exist")
        
        # Create monitor
        monitor = DriftMonitor(
            model_id=model_id,
            name=name,
            description=description,
            configuration=configuration,
            monitoring_frequency=monitoring_frequency,
            alert_recipients=alert_recipients or [],
            created_by=created_by,
            tags=tags or [],
        )
        
        # Store in repository
        stored_monitor = await self.drift_repository.create_monitor(monitor)
        
        return stored_monitor

    async def detect_drift(
        self,
        model_id: UUID,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        configuration: DriftConfiguration | None = None,
        created_by: str | None = None,
    ) -> DriftReport:
        """Detect drift between reference and current data.
        
        Args:
            model_id: Model identifier
            reference_data: Reference data
            current_data: Current data
            configuration: Drift detection configuration
            created_by: User initiating detection
            
        Returns:
            Drift detection report
        """
        if configuration is None:
            configuration = DriftConfiguration()
        
        detection_start_time = datetime.utcnow()
        
        # Validate data
        self._validate_data_compatibility(reference_data, current_data)
        
        # Filter features to monitor
        features_to_check = self._get_features_to_monitor(
            reference_data.columns.tolist(), configuration
        )
        
        # Detect drift for each feature
        feature_drift_results = {}
        for feature in features_to_check:
            feature_drift = await self._detect_feature_drift(
                feature, reference_data[feature], current_data[feature], configuration
            )
            feature_drift_results[feature] = feature_drift
        
        # Detect multivariate drift
        multivariate_drift_score, multivariate_drift_detected = await self._detect_multivariate_drift(
            reference_data[features_to_check], current_data[features_to_check], configuration
        )
        
        # Detect concept drift
        concept_drift_score, concept_drift_detected = await self._detect_concept_drift(
            model_id, reference_data, current_data, configuration
        )
        
        # Determine overall drift
        drifted_features = [
            name for name, drift in feature_drift_results.items() if drift.is_drifted
        ]
        
        overall_drift_detected = (
            len(drifted_features) > 0 or
            multivariate_drift_detected or
            concept_drift_detected
        )
        
        overall_drift_severity = self._calculate_overall_severity(
            feature_drift_results, multivariate_drift_detected, concept_drift_detected
        )
        
        # Determine drift types
        drift_types_detected = []
        if len(drifted_features) > 0:
            drift_types_detected.append(DriftType.DATA_DRIFT)
        if multivariate_drift_detected:
            drift_types_detected.append(DriftType.COVARIATE_SHIFT)
        if concept_drift_detected:
            drift_types_detected.append(DriftType.CONCEPT_DRIFT)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_drift_severity, drift_types_detected, drifted_features
        )
        
        detection_end_time = datetime.utcnow()
        
        # Create drift report
        drift_report = DriftReport(
            model_id=model_id,
            reference_sample_size=len(reference_data),
            current_sample_size=len(current_data),
            overall_drift_detected=overall_drift_detected,
            overall_drift_severity=overall_drift_severity,
            drift_types_detected=drift_types_detected,
            feature_drift=feature_drift_results,
            drifted_features=drifted_features,
            multivariate_drift_score=multivariate_drift_score,
            multivariate_drift_detected=multivariate_drift_detected,
            concept_drift_score=concept_drift_score,
            concept_drift_detected=concept_drift_detected,
            configuration=configuration,
            detection_start_time=detection_start_time,
            detection_end_time=detection_end_time,
            recommendations=recommendations,
            created_by=created_by,
        )
        
        # Store report
        stored_report = await self.drift_repository.create_report(drift_report)
        
        return stored_report

    async def run_scheduled_monitoring(self) -> list[DriftReport]:
        """Run scheduled drift monitoring for all active monitors.
        
        Returns:
            List of generated drift reports
        """
        # Get monitors that need checking
        monitors = await self.drift_repository.get_monitors_due_for_check()
        
        reports = []
        for monitor in monitors:
            try:
                report = await self._run_monitor_check(monitor)
                if report:
                    reports.append(report)
            except Exception as e:
                # Log error but continue with other monitors
                print(f"Error running monitor {monitor.id}: {e}")
        
        return reports

    async def get_drift_history(
        self,
        model_id: UUID,
        days: int = 30,
        severity_filter: DriftSeverity | None = None,
    ) -> list[DriftReport]:
        """Get drift history for a model.
        
        Args:
            model_id: Model identifier
            days: Number of days of history
            severity_filter: Filter by severity
            
        Returns:
            List of drift reports
        """
        since_date = datetime.utcnow() - timedelta(days=days)
        
        return await self.drift_repository.get_reports(
            model_id=model_id,
            since_date=since_date,
            severity_filter=severity_filter,
        )

    async def get_monitor_status(self, monitor_id: UUID) -> dict[str, Any]:
        """Get drift monitor status.
        
        Args:
            monitor_id: Monitor identifier
            
        Returns:
            Monitor status information
        """
        monitor = await self.drift_repository.get_monitor(monitor_id)
        if not monitor:
            raise ValueError(f"Monitor {monitor_id} not found")
        
        # Get recent reports
        recent_reports = await self.drift_repository.get_reports(
            model_id=monitor.model_id,
            limit=10,
        )
        
        return {
            "monitor": monitor,
            "recent_reports": recent_reports,
            "last_check": monitor.last_check_time,
            "next_check": monitor.next_check_time,
            "current_severity": monitor.current_drift_severity,
            "consecutive_detections": monitor.consecutive_drift_detections,
        }

    async def _detect_feature_drift(
        self,
        feature_name: str,
        reference_values: pd.Series,
        current_values: pd.Series,
        configuration: DriftConfiguration,
    ) -> FeatureDrift:
        """Detect drift for a single feature.
        
        Args:
            feature_name: Feature name
            reference_values: Reference feature values
            current_values: Current feature values
            configuration: Drift configuration
            
        Returns:
            Feature drift analysis
        """
        # Use primary detection method
        primary_method = configuration.enabled_methods[0]
        threshold = configuration.method_thresholds.get(
            primary_method.value, configuration.drift_threshold
        )
        
        # Calculate drift score and p-value based on method
        if primary_method == DriftDetectionMethod.KOLMOGOROV_SMIRNOV:
            drift_score, p_value = self._ks_test(reference_values, current_values)
        elif primary_method == DriftDetectionMethod.POPULATION_STABILITY_INDEX:
            drift_score = self._psi_test(reference_values, current_values)
            p_value = None
        elif primary_method == DriftDetectionMethod.JENSEN_SHANNON_DIVERGENCE:
            drift_score = self._js_divergence(reference_values, current_values)
            p_value = None
        elif primary_method == DriftDetectionMethod.WASSERSTEIN_DISTANCE:
            drift_score = self._wasserstein_distance(reference_values, current_values)
            p_value = None
        else:
            # Default to KS test
            drift_score, p_value = self._ks_test(reference_values, current_values)
        
        # Determine if drift detected
        is_drifted = drift_score > threshold
        
        # Calculate severity
        severity = self._calculate_feature_severity(drift_score, configuration)
        
        # Calculate statistics
        ref_mean = float(reference_values.mean()) if reference_values.dtype in ['int64', 'float64'] else None
        cur_mean = float(current_values.mean()) if current_values.dtype in ['int64', 'float64'] else None
        ref_std = float(reference_values.std()) if reference_values.dtype in ['int64', 'float64'] else None
        cur_std = float(current_values.std()) if current_values.dtype in ['int64', 'float64'] else None
        
        return FeatureDrift(
            feature_name=feature_name,
            drift_score=drift_score,
            p_value=p_value,
            threshold=threshold,
            is_drifted=is_drifted,
            severity=severity,
            method=primary_method,
            reference_mean=ref_mean,
            current_mean=cur_mean,
            reference_std=ref_std,
            current_std=cur_std,
        )

    async def _detect_multivariate_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        configuration: DriftConfiguration,
    ) -> tuple[float, bool]:
        """Detect multivariate drift.
        
        Args:
            reference_data: Reference data
            current_data: Current data
            configuration: Configuration
            
        Returns:
            Tuple of (drift_score, is_drifted)
        """
        if not configuration.enable_multivariate_detection:
            return 0.0, False
        
        # Simple multivariate approach: compare distributions using energy distance
        try:
            # Convert to numpy arrays
            ref_array = reference_data.select_dtypes(include=[np.number]).values
            cur_array = current_data.select_dtypes(include=[np.number]).values
            
            if ref_array.shape[1] == 0 or cur_array.shape[1] == 0:
                return 0.0, False
            
            # Calculate energy distance (simplified implementation)
            drift_score = self._energy_distance(ref_array, cur_array)
            
            threshold = configuration.drift_threshold
            is_drifted = drift_score > threshold
            
            return drift_score, is_drifted
            
        except Exception:
            return 0.0, False

    async def _detect_concept_drift(
        self,
        model_id: UUID,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        configuration: DriftConfiguration,
    ) -> tuple[float, bool]:
        """Detect concept drift by comparing model predictions.
        
        Args:
            model_id: Model identifier
            reference_data: Reference data
            current_data: Current data
            configuration: Configuration
            
        Returns:
            Tuple of (drift_score, is_drifted)
        """
        if not configuration.enable_concept_drift:
            return 0.0, False
        
        try:
            # Get model predictions on both datasets
            ref_predictions = await self._get_model_predictions(model_id, reference_data)
            cur_predictions = await self._get_model_predictions(model_id, current_data)
            
            # Compare prediction distributions
            drift_score, _ = self._ks_test(
                pd.Series(ref_predictions),
                pd.Series(cur_predictions)
            )
            
            threshold = configuration.drift_threshold
            is_drifted = drift_score > threshold
            
            return drift_score, is_drifted
            
        except Exception:
            return 0.0, False

    async def _run_monitor_check(self, monitor: DriftMonitor) -> DriftReport | None:
        """Run drift check for a monitor.
        
        Args:
            monitor: Drift monitor
            
        Returns:
            Drift report if drift detected, None otherwise
        """
        if not monitor.should_check_now():
            return None
        
        # Get reference and current data
        # This would typically be implemented based on your data storage strategy
        reference_data = await self._get_reference_data(monitor)
        current_data = await self._get_current_data(monitor)
        
        if reference_data is None or current_data is None:
            return None
        
        # Run drift detection
        report = await self.detect_drift(
            monitor.model_id,
            reference_data,
            current_data,
            monitor.configuration,
            "scheduled_monitor",
        )
        
        # Update monitor state
        monitor.record_drift_detection(report.overall_drift_severity, report.id)
        await self.drift_repository.update_monitor(monitor)
        
        # Send alert if needed
        if monitor.needs_alert(report.overall_drift_severity):
            await self._send_drift_alert(monitor, report)
        
        return report

    def _validate_data_compatibility(
        self, reference_data: pd.DataFrame, current_data: pd.DataFrame
    ) -> None:
        """Validate that reference and current data are compatible."""
        if reference_data.empty or current_data.empty:
            raise ValueError("Reference and current data cannot be empty")
        
        ref_columns = set(reference_data.columns)
        cur_columns = set(current_data.columns)
        
        if ref_columns != cur_columns:
            missing_in_current = ref_columns - cur_columns
            extra_in_current = cur_columns - ref_columns
            
            error_msg = "Data schema mismatch:"
            if missing_in_current:
                error_msg += f" Missing columns in current data: {missing_in_current}"
            if extra_in_current:
                error_msg += f" Extra columns in current data: {extra_in_current}"
            
            raise ValueError(error_msg)

    def _get_features_to_monitor(
        self, all_features: list[str], configuration: DriftConfiguration
    ) -> list[str]:
        """Get list of features to monitor based on configuration."""
        if configuration.features_to_monitor:
            features = configuration.features_to_monitor
        else:
            features = all_features
        
        # Remove excluded features
        features = [f for f in features if f not in configuration.exclude_features]
        
        return features

    def _calculate_overall_severity(
        self,
        feature_drift_results: dict[str, FeatureDrift],
        multivariate_drift: bool,
        concept_drift: bool,
    ) -> DriftSeverity:
        """Calculate overall drift severity."""
        if concept_drift:
            return DriftSeverity.CRITICAL
        
        if multivariate_drift:
            return DriftSeverity.HIGH
        
        severities = [drift.severity for drift in feature_drift_results.values()]
        
        if DriftSeverity.CRITICAL in severities:
            return DriftSeverity.CRITICAL
        elif DriftSeverity.HIGH in severities:
            return DriftSeverity.HIGH
        elif DriftSeverity.MEDIUM in severities:
            return DriftSeverity.MEDIUM
        elif DriftSeverity.LOW in severities:
            return DriftSeverity.LOW
        else:
            return DriftSeverity.NONE

    def _calculate_feature_severity(
        self, drift_score: float, configuration: DriftConfiguration
    ) -> DriftSeverity:
        """Calculate feature drift severity."""
        thresholds = configuration.severity_thresholds
        
        if drift_score >= thresholds["critical"]:
            return DriftSeverity.CRITICAL
        elif drift_score >= thresholds["high"]:
            return DriftSeverity.HIGH
        elif drift_score >= thresholds["medium"]:
            return DriftSeverity.MEDIUM
        elif drift_score >= thresholds["low"]:
            return DriftSeverity.LOW
        else:
            return DriftSeverity.NONE

    def _generate_recommendations(
        self,
        severity: DriftSeverity,
        drift_types: list[DriftType],
        drifted_features: list[str],
    ) -> list[str]:
        """Generate recommendations based on drift analysis."""
        recommendations = []
        
        if severity == DriftSeverity.CRITICAL:
            recommendations.append("URGENT: Stop model serving and investigate immediately")
            recommendations.append("Consider emergency model rollback")
        elif severity == DriftSeverity.HIGH:
            recommendations.append("Schedule immediate model retraining")
            recommendations.append("Increase monitoring frequency")
        elif severity == DriftSeverity.MEDIUM:
            recommendations.append("Plan model retraining within next release cycle")
            recommendations.append("Monitor performance metrics closely")
        
        if DriftType.CONCEPT_DRIFT in drift_types:
            recommendations.append("Investigate changes in target variable distribution")
            recommendations.append("Review business logic and data pipeline")
        
        if DriftType.DATA_DRIFT in drift_types and drifted_features:
            recommendations.append(f"Focus on features with drift: {', '.join(drifted_features[:5])}")
        
        return recommendations

    def _ks_test(self, ref_values: pd.Series, cur_values: pd.Series) -> tuple[float, float]:
        """Perform Kolmogorov-Smirnov test."""
        statistic, p_value = stats.ks_2samp(ref_values.dropna(), cur_values.dropna())
        return float(statistic), float(p_value)

    def _psi_test(self, ref_values: pd.Series, cur_values: pd.Series) -> float:
        """Calculate Population Stability Index."""
        # Bin the data
        _, bin_edges = np.histogram(ref_values.dropna(), bins=10)
        
        ref_counts, _ = np.histogram(ref_values.dropna(), bins=bin_edges)
        cur_counts, _ = np.histogram(cur_values.dropna(), bins=bin_edges)
        
        # Calculate proportions
        ref_props = ref_counts / ref_counts.sum()
        cur_props = cur_counts / cur_counts.sum()
        
        # Calculate PSI
        psi = 0.0
        for i in range(len(ref_props)):
            if ref_props[i] > 0 and cur_props[i] > 0:
                psi += (cur_props[i] - ref_props[i]) * np.log(cur_props[i] / ref_props[i])
        
        return float(psi)

    def _js_divergence(self, ref_values: pd.Series, cur_values: pd.Series) -> float:
        """Calculate Jensen-Shannon divergence."""
        # Bin the data
        _, bin_edges = np.histogram(
            pd.concat([ref_values.dropna(), cur_values.dropna()]), bins=10
        )
        
        ref_counts, _ = np.histogram(ref_values.dropna(), bins=bin_edges)
        cur_counts, _ = np.histogram(cur_values.dropna(), bins=bin_edges)
        
        # Calculate distributions
        ref_dist = ref_counts / ref_counts.sum()
        cur_dist = cur_counts / cur_counts.sum()
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        ref_dist = ref_dist + epsilon
        cur_dist = cur_dist + epsilon
        
        # Calculate JS divergence
        m = 0.5 * (ref_dist + cur_dist)
        js_div = 0.5 * stats.entropy(ref_dist, m) + 0.5 * stats.entropy(cur_dist, m)
        
        return float(js_div)

    def _wasserstein_distance(self, ref_values: pd.Series, cur_values: pd.Series) -> float:
        """Calculate Wasserstein distance."""
        distance = stats.wasserstein_distance(ref_values.dropna(), cur_values.dropna())
        return float(distance)

    def _energy_distance(self, ref_array: np.ndarray, cur_array: np.ndarray) -> float:
        """Calculate energy distance (simplified implementation)."""
        # This is a simplified implementation
        # In practice, you'd use a proper energy distance calculation
        ref_mean = np.mean(ref_array, axis=0)
        cur_mean = np.mean(cur_array, axis=0)
        
        distance = np.linalg.norm(ref_mean - cur_mean)
        return float(distance)

    async def _get_model_predictions(
        self, model_id: UUID, data: pd.DataFrame
    ) -> list[float]:
        """Get model predictions for data."""
        # This would use the model serving service
        # For now, return dummy predictions
        return [0.5] * len(data)

    async def _get_reference_data(self, monitor: DriftMonitor) -> pd.DataFrame | None:
        """Get reference data for a monitor."""
        # This would implement the logic to get reference data
        # based on the monitor configuration
        return None

    async def _get_current_data(self, monitor: DriftMonitor) -> pd.DataFrame | None:
        """Get current data for a monitor."""
        # This would implement the logic to get current data
        # based on the monitor configuration
        return None

    async def _send_drift_alert(self, monitor: DriftMonitor, report: DriftReport) -> None:
        """Send drift alert to recipients."""
        # This would implement alert sending logic
        pass