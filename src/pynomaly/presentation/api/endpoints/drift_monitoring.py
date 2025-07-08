"""API endpoints for data drift detection and model degradation monitoring."""

from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from pynomaly.application.services.drift_detection_service import DriftDetectionService
from pynomaly.application.use_cases.drift_monitoring_use_case import (
    DriftMonitoringUseCase,
)
from pynomaly.domain.entities.drift_detection import (
    DriftAlert,
    DriftDetectionMethod,
    DriftDetectionResult,
    DriftMonitoringStatus,
    DriftReport,
    DriftSeverity,
    ModelMonitoringConfig,
)
from pynomaly.presentation.api.docs.response_models import (
    HTTPResponses,
    SuccessResponse,
)

router = APIRouter(
    prefix="/drift",
    tags=["Data Drift & Model Degradation"],
    responses={
        401: HTTPResponses.unauthorized_401(),
        403: HTTPResponses.forbidden_403(),
        500: HTTPResponses.server_error_500(),
    },
)


# Request/Response Models


class DriftCheckRequest(BaseModel):
    """Request for performing drift check."""

    reference_data: list[list[float]] = Field(..., description="Reference dataset")
    current_data: list[list[float]] = Field(
        ..., description="Current dataset to compare"
    )
    feature_names: Optional[list[str]] = Field(None, description="Feature names")
    detection_methods: Optional[list[str]] = Field(
        default=["kolmogorov_smirnov", "jensen_shannon", "population_stability_index"],
        description="Detection methods to use",
    )


class PerformanceDriftRequest(BaseModel):
    """Request for performance drift check."""

    reference_metrics: dict[str, float] = Field(
        ..., description="Reference performance metrics"
    )
    current_metrics: dict[str, float] = Field(
        ..., description="Current performance metrics"
    )
    threshold: float = Field(
        default=0.05, description="Minimum change to consider drift"
    )


class MonitoringConfigRequest(BaseModel):
    """Request for configuring drift monitoring."""

    enabled: bool = Field(default=True, description="Enable monitoring")
    check_interval_hours: int = Field(default=24, description="Hours between checks")
    reference_window_days: int = Field(
        default=30, description="Reference data window in days"
    )
    comparison_window_days: int = Field(
        default=7, description="Comparison data window in days"
    )
    min_sample_size: int = Field(
        default=100, description="Minimum sample size for checks"
    )

    enabled_methods: list[str] = Field(
        default=["kolmogorov_smirnov", "jensen_shannon", "population_stability_index"],
        description="Drift detection methods to enable",
    )

    drift_thresholds: dict[str, float] = Field(
        default={
            "ks_statistic": 0.2,
            "js_divergence": 0.1,
            "psi_score": 0.25,
            "performance_drop": 0.05,
        },
        description="Drift detection thresholds",
    )

    alert_on_severity: list[str] = Field(
        default=["high", "critical"], description="Severity levels that trigger alerts"
    )
    notification_channels: list[str] = Field(
        default=[], description="Notification channels (email, slack, webhook)"
    )
    max_alerts_per_day: int = Field(default=10, description="Maximum alerts per day")

    features_to_monitor: list[str] = Field(
        default=[], description="Specific features to monitor"
    )
    exclude_features: list[str] = Field(
        default=[], description="Features to exclude from monitoring"
    )


class AlertActionRequest(BaseModel):
    """Request for alert actions."""

    action: str = Field(..., description="Action description")
    user: str = Field(..., description="User performing action")


# Dependency injection
async def get_drift_monitoring_use_case() -> DriftMonitoringUseCase:
    """Get drift monitoring use case."""
    # In real implementation, this would be properly injected
    drift_service = DriftDetectionService()
    return DriftMonitoringUseCase(drift_service)


@router.post(
    "/check/{detector_id}",
    response_model=SuccessResponse[DriftDetectionResult],
    summary="Perform Drift Check",
    description="""
    Perform immediate data drift check for a specific detector.

    **Drift Detection Methods:**
    - **Kolmogorov-Smirnov**: Statistical test for distribution differences
    - **Jensen-Shannon**: Divergence measure for probability distributions
    - **Population Stability Index**: Measure categorical feature drift

    **Detection Process:**
    1. Compare reference and current datasets
    2. Apply selected statistical tests
    3. Calculate feature-level drift scores
    4. Determine overall drift severity
    5. Generate actionable recommendations

    **Severity Levels:**
    - `low`: Minor drift, continue monitoring
    - `medium`: Moderate drift, investigate further
    - `high`: Significant drift, consider action
    - `critical`: Severe drift, immediate intervention required

    **Example Request:**
    ```json
    {
      "reference_data": [[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]],
      "current_data": [[1.5, 2.5], [1.6, 2.6], [1.4, 2.4]],
      "feature_names": ["feature_1", "feature_2"],
      "detection_methods": ["kolmogorov_smirnov", "jensen_shannon"]
    }
    ```
    """,
    responses={
        200: HTTPResponses.ok_200("Drift check completed successfully"),
        400: HTTPResponses.bad_request_400("Invalid drift check request"),
        404: HTTPResponses.not_found_404("Detector not found"),
    },
)
async def perform_drift_check(
    detector_id: str,
    request: DriftCheckRequest,
    use_case: DriftMonitoringUseCase = Depends(get_drift_monitoring_use_case),
) -> SuccessResponse[DriftDetectionResult]:
    """Perform immediate drift check for a detector."""
    try:
        # Convert request data to numpy arrays
        reference_data = np.array(request.reference_data)
        current_data = np.array(request.current_data)

        # Validate data shapes
        if reference_data.shape[1] != current_data.shape[1]:
            raise HTTPException(
                status_code=400,
                detail="Reference and current data must have same number of features",
            )

        # Perform drift check
        result = await use_case.perform_drift_check(
            detector_id=detector_id,
            reference_data=reference_data,
            current_data=current_data,
            feature_names=request.feature_names,
        )

        return SuccessResponse(
            data=result, message=f"Drift check completed for detector {detector_id}"
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift check failed: {str(e)}")


@router.post(
    "/performance/{detector_id}",
    response_model=SuccessResponse[DriftDetectionResult],
    summary="Check Performance Drift",
    description="""
    Check for performance drift by comparing performance metrics.

    **Performance Metrics Monitored:**
    - **Accuracy**: Overall model accuracy
    - **Precision**: Precision for each class
    - **Recall**: Recall for each class
    - **F1-Score**: F1 score for each class
    - **AUC**: Area under ROC curve

    **Drift Detection:**
    - Compares current metrics against reference baseline
    - Identifies statistically significant performance drops
    - Calculates relative and absolute metric changes
    - Determines if changes exceed configured thresholds

    **Common Causes of Performance Drift:**
    - Data distribution changes (covariate shift)
    - Concept drift (relationship changes)
    - Data quality degradation
    - Seasonal patterns in data
    - Model staleness over time

    **Example Request:**
    ```json
    {
      "reference_metrics": {
        "accuracy": 0.95,
        "precision": 0.93,
        "recall": 0.92,
        "f1": 0.925
      },
      "current_metrics": {
        "accuracy": 0.88,
        "precision": 0.85,
        "recall": 0.87,
        "f1": 0.86
      },
      "threshold": 0.05
    }
    ```
    """,
    responses={
        200: HTTPResponses.ok_200("Performance drift check completed"),
        400: HTTPResponses.bad_request_400("Invalid performance metrics"),
    },
)
async def check_performance_drift(
    detector_id: str,
    request: PerformanceDriftRequest,
    use_case: DriftMonitoringUseCase = Depends(get_drift_monitoring_use_case),
) -> SuccessResponse[DriftDetectionResult]:
    """Check for performance drift in a detector."""
    try:
        result = await use_case.check_performance_drift(
            detector_id=detector_id,
            reference_metrics=request.reference_metrics,
            current_metrics=request.current_metrics,
            threshold=request.threshold,
        )

        return SuccessResponse(
            data=result,
            message=f"Performance drift check completed for detector {detector_id}",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Performance drift check failed: {str(e)}"
        )


@router.post(
    "/monitoring/{detector_id}/configure",
    response_model=SuccessResponse[DriftMonitoringStatus],
    summary="Configure Drift Monitoring",
    description="""
    Configure automated drift monitoring for a detector.

    **Monitoring Features:**
    - **Scheduled Checks**: Automatic drift detection at configured intervals
    - **Multi-Method Detection**: Combine multiple statistical tests
    - **Threshold Management**: Configurable sensitivity for each method
    - **Alert Management**: Automated alerts based on severity levels
    - **Notification Integration**: Email, Slack, webhook notifications

    **Configuration Options:**
    - **Check Interval**: How often to perform drift checks (hours)
    - **Data Windows**: Size of reference and comparison datasets
    - **Detection Methods**: Which statistical tests to apply
    - **Alert Thresholds**: When to trigger notifications
    - **Feature Selection**: Monitor all or specific features

    **Monitoring Lifecycle:**
    1. **Setup**: Configure monitoring parameters
    2. **Active**: Continuously monitor for drift
    3. **Alert**: Notify when drift exceeds thresholds
    4. **Action**: Take corrective measures
    5. **Review**: Analyze drift patterns and adjust

    **Best Practices:**
    - Start with conservative thresholds and adjust based on experience
    - Monitor critical features more frequently
    - Set up multiple notification channels for redundancy
    - Regular review drift reports to identify patterns
    """,
    responses={
        201: HTTPResponses.created_201("Monitoring configured successfully"),
        400: HTTPResponses.bad_request_400("Invalid monitoring configuration"),
    },
)
async def configure_monitoring(
    detector_id: str,
    request: MonitoringConfigRequest,
    background_tasks: BackgroundTasks,
    use_case: DriftMonitoringUseCase = Depends(get_drift_monitoring_use_case),
) -> SuccessResponse[DriftMonitoringStatus]:
    """Configure drift monitoring for a detector."""
    try:
        # Convert request to domain model
        config = ModelMonitoringConfig(
            detector_id=detector_id,
            enabled=request.enabled,
            check_interval_hours=request.check_interval_hours,
            reference_window_days=request.reference_window_days,
            comparison_window_days=request.comparison_window_days,
            min_sample_size=request.min_sample_size,
            enabled_methods=[
                DriftDetectionMethod(method) for method in request.enabled_methods
            ],
            drift_thresholds=request.drift_thresholds,
            alert_on_severity=[
                DriftSeverity(severity) for severity in request.alert_on_severity
            ],
            notification_channels=request.notification_channels,
            max_alerts_per_day=request.max_alerts_per_day,
            features_to_monitor=request.features_to_monitor,
            exclude_features=request.exclude_features,
        )

        # Configure monitoring
        status = await use_case.configure_monitoring(detector_id, config)

        return SuccessResponse(
            data=status,
            message=f"Drift monitoring configured for detector {detector_id}",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to configure monitoring: {str(e)}"
        )


@router.get(
    "/monitoring/{detector_id}/status",
    response_model=SuccessResponse[DriftMonitoringStatus],
    summary="Get Monitoring Status",
    description="""
    Get current drift monitoring status for a detector.

    **Status Information:**
    - **Current State**: Active, paused, stopped, error
    - **Check History**: Number of checks performed, last check time
    - **Drift Summary**: Recent drift detections, health score
    - **Error Information**: Consecutive failures, last error details

    **Health Score Calculation:**
    - Starts at 1.0 (perfect health)
    - Decreases with drift detections (weighted by severity)
    - Gradually improves when no drift detected
    - Range: 0.0 (critical) to 1.0 (healthy)

    **Status Monitoring:**
    - Use this endpoint to monitor system health
    - Track drift detection patterns over time
    - Identify detectors needing attention
    - Plan maintenance and model updates
    """,
    responses={
        200: HTTPResponses.ok_200("Monitoring status retrieved"),
        404: HTTPResponses.not_found_404("Monitoring not configured"),
    },
)
async def get_monitoring_status(
    detector_id: str,
    use_case: DriftMonitoringUseCase = Depends(get_drift_monitoring_use_case),
) -> SuccessResponse[DriftMonitoringStatus]:
    """Get monitoring status for a detector."""
    try:
        status = await use_case.get_monitoring_status(detector_id)

        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"No monitoring configured for detector {detector_id}",
            )

        return SuccessResponse(
            data=status,
            message=f"Retrieved monitoring status for detector {detector_id}",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get monitoring status: {str(e)}"
        )


@router.post(
    "/monitoring/{detector_id}/pause",
    response_model=SuccessResponse[str],
    summary="Pause Monitoring",
    description="""
    Temporarily pause drift monitoring for a detector.

    **Pause Effects:**
    - Stops scheduled drift checks
    - Preserves monitoring configuration
    - Maintains historical data and status
    - Can be resumed without reconfiguration

    **Use Cases:**
    - Temporary maintenance windows
    - Known data quality issues
    - Model retraining in progress
    - Testing new configurations
    """,
)
async def pause_monitoring(
    detector_id: str,
    use_case: DriftMonitoringUseCase = Depends(get_drift_monitoring_use_case),
) -> SuccessResponse[str]:
    """Pause drift monitoring for a detector."""
    try:
        success = await use_case.pause_monitoring(detector_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"No active monitoring found for detector {detector_id}",
            )

        return SuccessResponse(
            data="paused", message=f"Monitoring paused for detector {detector_id}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to pause monitoring: {str(e)}"
        )


@router.post(
    "/monitoring/{detector_id}/resume",
    response_model=SuccessResponse[str],
    summary="Resume Monitoring",
    description="""
    Resume paused drift monitoring for a detector.

    **Resume Effects:**
    - Restarts scheduled drift checks
    - Uses existing monitoring configuration
    - Schedules next check based on interval
    - Continues from previous monitoring state
    """,
)
async def resume_monitoring(
    detector_id: str,
    use_case: DriftMonitoringUseCase = Depends(get_drift_monitoring_use_case),
) -> SuccessResponse[str]:
    """Resume drift monitoring for a detector."""
    try:
        success = await use_case.resume_monitoring(detector_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"No paused monitoring found for detector {detector_id}",
            )

        return SuccessResponse(
            data="resumed", message=f"Monitoring resumed for detector {detector_id}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to resume monitoring: {str(e)}"
        )


@router.delete(
    "/monitoring/{detector_id}",
    response_model=SuccessResponse[str],
    summary="Stop Monitoring",
    description="""
    Stop drift monitoring for a detector completely.

    **Stop Effects:**
    - Terminates all monitoring activities
    - Preserves historical data and alerts
    - Requires reconfiguration to restart
    - Stops all notifications
    """,
)
async def stop_monitoring(
    detector_id: str,
    use_case: DriftMonitoringUseCase = Depends(get_drift_monitoring_use_case),
) -> SuccessResponse[str]:
    """Stop drift monitoring for a detector."""
    try:
        success = await use_case.stop_monitoring(detector_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"No monitoring found for detector {detector_id}",
            )

        return SuccessResponse(
            data="stopped", message=f"Monitoring stopped for detector {detector_id}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to stop monitoring: {str(e)}"
        )


@router.get(
    "/monitoring/active",
    response_model=SuccessResponse[list[str]],
    summary="List Active Monitors",
    description="""
    List all detectors with active drift monitoring.

    **Response Includes:**
    - Detector IDs with active monitoring
    - Current monitoring status summary
    - System-wide monitoring health
    """,
)
async def list_active_monitors(
    use_case: DriftMonitoringUseCase = Depends(get_drift_monitoring_use_case),
) -> SuccessResponse[list[str]]:
    """List all actively monitored detectors."""
    try:
        active_monitors = await use_case.list_active_monitors()

        return SuccessResponse(
            data=active_monitors,
            message=f"Retrieved {len(active_monitors)} active monitors",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list active monitors: {str(e)}"
        )


@router.get(
    "/alerts",
    response_model=SuccessResponse[list[DriftAlert]],
    summary="Get Drift Alerts",
    description="""
    Retrieve drift alerts with optional filtering.

    **Filter Options:**
    - **Detector ID**: Get alerts for specific detector
    - **Severity**: Filter by alert severity level
    - **Active Only**: Show only unresolved alerts
    - **Time Range**: Alerts within specific period

    **Alert Lifecycle:**
    1. **Triggered**: Alert created when drift detected
    2. **Acknowledged**: Human acknowledges alert
    3. **Investigating**: Action being taken
    4. **Resolved**: Issue addressed and alert closed

    **Alert Information:**
    - Drift type and severity
    - Affected features and metrics
    - Recommended actions
    - Timeline and responsible users
    """,
)
async def get_drift_alerts(
    detector_id: Optional[str] = Query(None, description="Filter by detector ID"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    active_only: bool = Query(True, description="Only return active alerts"),
    limit: int = Query(100, description="Maximum number of alerts"),
    use_case: DriftMonitoringUseCase = Depends(get_drift_monitoring_use_case),
) -> SuccessResponse[list[DriftAlert]]:
    """Get drift alerts with optional filtering."""
    try:
        severity_filter = DriftSeverity(severity) if severity else None

        alerts = await use_case.get_drift_alerts(
            detector_id=detector_id,
            severity=severity_filter,
            active_only=active_only,
            limit=limit,
        )

        return SuccessResponse(
            data=alerts, message=f"Retrieved {len(alerts)} drift alerts"
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get drift alerts: {str(e)}"
        )


@router.post(
    "/alerts/{alert_id}/acknowledge",
    response_model=SuccessResponse[str],
    summary="Acknowledge Alert",
    description="""
    Acknowledge a drift alert to indicate it's being addressed.

    **Acknowledgment Process:**
    - Records who acknowledged the alert
    - Timestamps the acknowledgment
    - Maintains alert as active until resolved
    - Tracks accountability and response times
    """,
)
async def acknowledge_alert(
    alert_id: str,
    request: AlertActionRequest,
    use_case: DriftMonitoringUseCase = Depends(get_drift_monitoring_use_case),
) -> SuccessResponse[str]:
    """Acknowledge a drift alert."""
    try:
        success = await use_case.acknowledge_alert(alert_id, request.user)

        if not success:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

        return SuccessResponse(
            data="acknowledged",
            message=f"Alert {alert_id} acknowledged by {request.user}",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to acknowledge alert: {str(e)}"
        )


@router.post(
    "/alerts/{alert_id}/resolve",
    response_model=SuccessResponse[str],
    summary="Resolve Alert",
    description="""
    Resolve a drift alert after taking corrective action.

    **Resolution Process:**
    - Records resolution action taken
    - Timestamps the resolution
    - Marks alert as inactive/resolved
    - Documents the solution for future reference

    **Common Resolution Actions:**
    - Model retraining with recent data
    - Data pipeline fixes
    - Threshold adjustments
    - Feature engineering updates
    """,
)
async def resolve_alert(
    alert_id: str,
    request: AlertActionRequest,
    use_case: DriftMonitoringUseCase = Depends(get_drift_monitoring_use_case),
) -> SuccessResponse[str]:
    """Resolve a drift alert."""
    try:
        success = await use_case.resolve_alert(alert_id, request.user, request.action)

        if not success:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

        return SuccessResponse(
            data="resolved", message=f"Alert {alert_id} resolved by {request.user}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to resolve alert: {str(e)}"
        )


@router.get(
    "/reports/{detector_id}",
    response_model=SuccessResponse[DriftReport],
    summary="Generate Drift Report",
    description="""
    Generate comprehensive drift monitoring report for a detector.

    **Report Contents:**
    - **Executive Summary**: Overall drift status and trends
    - **Detection History**: Timeline of drift events
    - **Feature Analysis**: Most affected features and patterns
    - **Performance Impact**: Correlation with model performance
    - **Recommendations**: Actionable insights and next steps

    **Report Metrics:**
    - Drift detection rate over time
    - Feature-level drift patterns
    - Alert frequency and resolution times
    - System health trends
    - Comparative analysis across detectors

    **Use Cases:**
    - Monthly/quarterly model reviews
    - Regulatory compliance reporting
    - Performance analysis and optimization
    - Stakeholder communication
    """,
)
async def generate_drift_report(
    detector_id: str,
    period_days: int = Query(30, description="Report period in days"),
    use_case: DriftMonitoringUseCase = Depends(get_drift_monitoring_use_case),
) -> SuccessResponse[DriftReport]:
    """Generate drift monitoring report for a detector."""
    try:
        if period_days < 1 or period_days > 365:
            raise HTTPException(
                status_code=400, detail="Report period must be between 1 and 365 days"
            )

        report = await use_case.generate_drift_report(detector_id, period_days)

        return SuccessResponse(
            data=report, message=f"Drift report generated for detector {detector_id}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate drift report: {str(e)}"
        )


@router.get(
    "/system/health",
    response_model=SuccessResponse[dict[str, Any]],
    summary="Get System Health",
    description="""
    Get overall drift monitoring system health and status.

    **Health Indicators:**
    - **Active Monitors**: Number of detectors being monitored
    - **System Status**: Overall health (healthy, degraded, error)
    - **Recent Activity**: Drift detections, alerts, resolutions
    - **Performance**: Average health scores, error rates

    **Health Scoring:**
    - Healthy: All monitors functioning, low drift rates
    - Degraded: Some issues detected, monitoring continues
    - Error: Critical failures, immediate attention needed

    **Monitoring Dashboard:**
    Use this endpoint to build real-time monitoring dashboards
    showing the overall health of your drift detection system.
    """,
)
async def get_system_health(
    use_case: DriftMonitoringUseCase = Depends(get_drift_monitoring_use_case),
) -> SuccessResponse[dict[str, Any]]:
    """Get overall drift monitoring system health."""
    try:
        health = await use_case.get_system_health()

        return SuccessResponse(
            data=health, message="System health retrieved successfully"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get system health: {str(e)}"
        )
