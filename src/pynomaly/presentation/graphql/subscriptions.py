"""GraphQL subscriptions for real-time updates in the Pynomaly API."""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Dict, Any, Optional
from uuid import UUID

import strawberry
from strawberry.types import Info

from pynomaly.application.services.detection_service import DetectionService
from pynomaly.application.services.training_service import TrainingService
from pynomaly.application.services.monitoring_service import MonitoringService
from pynomaly.application.services.audit_service import AuditService
from pynomaly.domain.entities.user import User
from pynomaly.presentation.graphql.types import (
    DetectionAlert,
    JobProgress,
    SystemHealth,
    WebSocketEvent,
    AuditEvent,
    SecurityMetrics,
    PerformanceMetrics,
)


@strawberry.type
class Subscription:
    """GraphQL subscriptions for real-time updates."""

    @strawberry.subscription
    async def training_progress(
        self,
        info: Info,
        detector_id: strawberry.ID
    ) -> AsyncGenerator[JobProgress, None]:
        """Subscribe to training progress updates for a specific detector."""
        current_user: User = info.context.get("user")
        
        if not current_user or not current_user.has_permission("detector.train"):
            raise ValueError("Insufficient permissions to subscribe to training progress")
        
        training_service: TrainingService = info.context["container"].get(TrainingService)
        detector_uuid = UUID(str(detector_id))
        
        try:
            async for progress_update in training_service.subscribe_to_training_progress(detector_uuid):
                yield JobProgress(
                    job_id=str(progress_update.job_id),
                    detector_id=str(detector_uuid),
                    status=progress_update.status,
                    progress=progress_update.progress,
                    message=progress_update.message,
                    current_step=progress_update.current_step,
                    total_steps=progress_update.total_steps,
                    metrics=progress_update.metrics,
                    estimated_time_remaining=progress_update.estimated_time_remaining,
                    timestamp=progress_update.timestamp
                )
        except Exception as e:
            yield JobProgress(
                job_id="error",
                detector_id=str(detector_uuid),
                status="failed",
                progress=0,
                message=f"Training subscription error: {str(e)}",
                current_step=0,
                total_steps=0,
                metrics={},
                estimated_time_remaining=None,
                timestamp=None
            )

    @strawberry.subscription
    async def detection_results(
        self,
        info: Info,
        detector_id: Optional[strawberry.ID] = None
    ) -> AsyncGenerator[DetectionAlert, None]:
        """Subscribe to real-time detection results."""
        current_user: User = info.context.get("user")
        
        if not current_user or not current_user.has_permission("detection.read"):
            raise ValueError("Insufficient permissions to subscribe to detection results")
        
        detection_service: DetectionService = info.context["container"].get(DetectionService)
        detector_uuid = UUID(str(detector_id)) if detector_id else None
        
        try:
            async for detection_result in detection_service.subscribe_to_detection_results(
                detector_id=detector_uuid,
                user_id=current_user.id,
                tenant_id=current_user.tenant_id
            ):
                yield DetectionAlert(
                    id=str(detection_result.id),
                    detector_id=str(detection_result.detector_id),
                    severity=detection_result.severity,
                    score=detection_result.anomaly_score,
                    message=detection_result.message,
                    data_point=detection_result.data_point,
                    features=detection_result.features,
                    metadata=detection_result.metadata,
                    timestamp=detection_result.timestamp,
                    threshold=detection_result.threshold,
                    confidence=detection_result.confidence
                )
        except Exception as e:
            yield DetectionAlert(
                id="error",
                detector_id=str(detector_uuid) if detector_uuid else "unknown",
                severity="error",
                score=0.0,
                message=f"Detection subscription error: {str(e)}",
                data_point={},
                features=[],
                metadata={},
                timestamp=None,
                threshold=0.0,
                confidence=0.0
            )

    @strawberry.subscription
    async def system_health(
        self,
        info: Info,
        interval_seconds: int = 30
    ) -> AsyncGenerator[SystemHealth, None]:
        """Subscribe to system health updates."""
        current_user: User = info.context.get("user")
        
        if not current_user or not current_user.has_permission("system.monitor"):
            raise ValueError("Insufficient permissions to subscribe to system health")
        
        monitoring_service: MonitoringService = info.context["container"].get(MonitoringService)
        
        try:
            while True:
                health_data = await monitoring_service.get_system_health()
                
                yield SystemHealth(
                    status=health_data.status,
                    timestamp=health_data.timestamp,
                    uptime=health_data.uptime,
                    cpu_usage=health_data.cpu_usage,
                    memory_usage=health_data.memory_usage,
                    disk_usage=health_data.disk_usage,
                    active_connections=health_data.active_connections,
                    database_status=health_data.database_status,
                    cache_status=health_data.cache_status,
                    service_status=health_data.service_status,
                    error_rate=health_data.error_rate,
                    response_time=health_data.response_time
                )
                
                await asyncio.sleep(interval_seconds)
                
        except Exception as e:
            yield SystemHealth(
                status="error",
                timestamp=None,
                uptime=0,
                cpu_usage=0,
                memory_usage=0,
                disk_usage=0,
                active_connections=0,
                database_status="unknown",
                cache_status="unknown",
                service_status={},
                error_rate=0,
                response_time=0
            )

    @strawberry.subscription
    async def audit_events(
        self,
        info: Info,
        event_types: Optional[list[str]] = None,
        severity_min: Optional[str] = None
    ) -> AsyncGenerator[AuditEvent, None]:
        """Subscribe to security audit events."""
        current_user: User = info.context.get("user")
        
        if not current_user or not current_user.has_permission("audit.read"):
            raise ValueError("Insufficient permissions to subscribe to audit events")
        
        audit_service: AuditService = info.context["container"].get(AuditService)
        
        try:
            async for audit_event in audit_service.subscribe_to_audit_events(
                event_types=event_types,
                severity_min=severity_min,
                user_id=current_user.id,
                tenant_id=current_user.tenant_id
            ):
                yield AuditEvent(
                    id=str(audit_event.id),
                    event_type=audit_event.event_type,
                    user_id=str(audit_event.user_id) if audit_event.user_id else None,
                    tenant_id=str(audit_event.tenant_id) if audit_event.tenant_id else None,
                    timestamp=audit_event.timestamp,
                    severity=audit_event.severity,
                    description=audit_event.description,
                    details=audit_event.details,
                    ip_address=audit_event.ip_address,
                    user_agent=audit_event.user_agent,
                    session_id=audit_event.session_id,
                    correlation_id=audit_event.correlation_id,
                    risk_score=audit_event.risk_score
                )
        except Exception as e:
            yield AuditEvent(
                id="error",
                event_type="system_error",
                user_id=str(current_user.id),
                tenant_id=str(current_user.tenant_id),
                timestamp=None,
                severity="error",
                description=f"Audit subscription error: {str(e)}",
                details={},
                ip_address=None,
                user_agent=None,
                session_id=None,
                correlation_id=None,
                risk_score=100
            )

    @strawberry.subscription
    async def performance_metrics(
        self,
        info: Info,
        interval_seconds: int = 60
    ) -> AsyncGenerator[PerformanceMetrics, None]:
        """Subscribe to performance metrics updates."""
        current_user: User = info.context.get("user")
        
        if not current_user or not current_user.has_permission("metrics.read"):
            raise ValueError("Insufficient permissions to subscribe to performance metrics")
        
        monitoring_service: MonitoringService = info.context["container"].get(MonitoringService)
        
        try:
            while True:
                metrics = await monitoring_service.get_performance_metrics()
                
                yield PerformanceMetrics(
                    timestamp=metrics.timestamp,
                    request_count=metrics.request_count,
                    request_rate=metrics.request_rate,
                    average_response_time=metrics.average_response_time,
                    error_rate=metrics.error_rate,
                    throughput=metrics.throughput,
                    active_users=metrics.active_users,
                    active_detectors=metrics.active_detectors,
                    detection_rate=metrics.detection_rate,
                    training_jobs=metrics.training_jobs,
                    queue_size=metrics.queue_size,
                    cache_hit_rate=metrics.cache_hit_rate,
                    database_connections=metrics.database_connections,
                    memory_usage_mb=metrics.memory_usage_mb,
                    cpu_usage_percent=metrics.cpu_usage_percent
                )
                
                await asyncio.sleep(interval_seconds)
                
        except Exception as e:
            yield PerformanceMetrics(
                timestamp=None,
                request_count=0,
                request_rate=0,
                average_response_time=0,
                error_rate=0,
                throughput=0,
                active_users=0,
                active_detectors=0,
                detection_rate=0,
                training_jobs=0,
                queue_size=0,
                cache_hit_rate=0,
                database_connections=0,
                memory_usage_mb=0,
                cpu_usage_percent=0
            )

    @strawberry.subscription
    async def websocket_events(
        self,
        info: Info,
        channel: str = "general"
    ) -> AsyncGenerator[WebSocketEvent, None]:
        """Subscribe to general WebSocket events for a specific channel."""
        current_user: User = info.context.get("user")
        
        if not current_user:
            raise ValueError("Authentication required for WebSocket subscriptions")
        
        # This would integrate with your existing WebSocket infrastructure
        websocket_manager = info.context.get("websocket_manager")
        
        if not websocket_manager:
            raise ValueError("WebSocket manager not available")
        
        try:
            async for event in websocket_manager.subscribe_to_channel(
                channel=channel,
                user_id=current_user.id,
                tenant_id=current_user.tenant_id
            ):
                yield WebSocketEvent(
                    event_type=event.event_type,
                    data=event.data,
                    timestamp=event.timestamp,
                    channel=channel,
                    user_id=str(current_user.id),
                    correlation_id=event.correlation_id
                )
        except Exception as e:
            yield WebSocketEvent(
                event_type="error",
                data={"message": f"WebSocket subscription error: {str(e)}"},
                timestamp=None,
                channel=channel,
                user_id=str(current_user.id),
                correlation_id=None
            )

    @strawberry.subscription
    async def security_metrics(
        self,
        info: Info,
        interval_seconds: int = 120
    ) -> AsyncGenerator[SecurityMetrics, None]:
        """Subscribe to security metrics updates."""
        current_user: User = info.context.get("user")
        
        if not current_user or not current_user.has_permission("security.monitor"):
            raise ValueError("Insufficient permissions to subscribe to security metrics")
        
        monitoring_service: MonitoringService = info.context["container"].get(MonitoringService)
        
        try:
            while True:
                security_data = await monitoring_service.get_security_metrics()
                
                yield SecurityMetrics(
                    timestamp=security_data.timestamp,
                    failed_login_attempts=security_data.failed_login_attempts,
                    successful_logins=security_data.successful_logins,
                    active_sessions=security_data.active_sessions,
                    blocked_ips=security_data.blocked_ips,
                    security_alerts=security_data.security_alerts,
                    high_risk_events=security_data.high_risk_events,
                    authentication_rate=security_data.authentication_rate,
                    authorization_failures=security_data.authorization_failures,
                    data_access_violations=security_data.data_access_violations,
                    privilege_escalation_attempts=security_data.privilege_escalation_attempts,
                    suspicious_activities=security_data.suspicious_activities,
                    compliance_violations=security_data.compliance_violations,
                    average_risk_score=security_data.average_risk_score
                )
                
                await asyncio.sleep(interval_seconds)
                
        except Exception as e:
            yield SecurityMetrics(
                timestamp=None,
                failed_login_attempts=0,
                successful_logins=0,
                active_sessions=0,
                blocked_ips=0,
                security_alerts=0,
                high_risk_events=0,
                authentication_rate=0,
                authorization_failures=0,
                data_access_violations=0,
                privilege_escalation_attempts=0,
                suspicious_activities=0,
                compliance_violations=0,
                average_risk_score=0
            )

    @strawberry.subscription
    async def model_training_status(
        self,
        info: Info,
        user_id: Optional[strawberry.ID] = None
    ) -> AsyncGenerator[JobProgress, None]:
        """Subscribe to all training jobs status for a user or tenant."""
        current_user: User = info.context.get("user")
        
        if not current_user or not current_user.has_permission("detector.train"):
            raise ValueError("Insufficient permissions to subscribe to training status")
        
        training_service: TrainingService = info.context["container"].get(TrainingService)
        target_user_id = UUID(str(user_id)) if user_id else current_user.id
        
        # Verify user can monitor other users' training jobs
        if target_user_id != current_user.id and not current_user.has_permission("training.monitor_all"):
            raise ValueError("Insufficient permissions to monitor other users' training jobs")
        
        try:
            async for training_update in training_service.subscribe_to_all_training_jobs(
                user_id=target_user_id,
                tenant_id=current_user.tenant_id
            ):
                yield JobProgress(
                    job_id=str(training_update.job_id),
                    detector_id=str(training_update.detector_id),
                    status=training_update.status,
                    progress=training_update.progress,
                    message=training_update.message,
                    current_step=training_update.current_step,
                    total_steps=training_update.total_steps,
                    metrics=training_update.metrics,
                    estimated_time_remaining=training_update.estimated_time_remaining,
                    timestamp=training_update.timestamp
                )
        except Exception as e:
            yield JobProgress(
                job_id="error",
                detector_id="unknown",
                status="failed",
                progress=0,
                message=f"Training status subscription error: {str(e)}",
                current_step=0,
                total_steps=0,
                metrics={},
                estimated_time_remaining=None,
                timestamp=None
            )