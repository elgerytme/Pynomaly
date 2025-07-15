"""GraphQL queries for Pynomaly API."""

from __future__ import annotations

from typing import List, Optional
from uuid import UUID

import strawberry
from strawberry.types import Info

from .context import GraphQLContext
from .types import (
    AnomalyDetectionResult,
    AuditLog,
    AuditLogConnection,
    AuditLogFilter,
    AuditLogSort,
    Detector,
    DetectorConnection,
    DetectorFilter,
    DetectorSort,
    DetectionJob,
    DetectionResultConnection,
    DetectionResultFilter,
    DetectionResultSort,
    PerformanceMetrics,
    SecurityMetrics,
    SystemHealth,
    TrainingJob,
    User,
)


@strawberry.type
class Query:
    """Root query type for GraphQL."""

    @strawberry.field
    async def hello(self) -> str:
        """Simple hello query for testing."""
        return "Hello from Pynomaly GraphQL API!"

    @strawberry.field
    async def system_health(self, info: Info[GraphQLContext, None]) -> SystemHealth:
        """Get system health status."""
        container = info.context.container
        health_service = container.health_service()
        
        health_data = await health_service.get_health_status()
        
        return SystemHealth(
            status=health_data.get("status", "unknown"),
            timestamp=health_data.get("timestamp"),
            version=health_data.get("version", "unknown"),
            services=health_data.get("services", {}),
            metrics=health_data.get("metrics", {}),
        )

    @strawberry.field
    async def current_user(self, info: Info[GraphQLContext, None]) -> Optional[User]:
        """Get current authenticated user."""
        user_id = info.context.user_id
        if not user_id:
            return None
            
        container = info.context.container
        user_service = container.user_service()
        
        user_data = await user_service.get_user_by_id(user_id)
        if not user_data:
            return None
            
        return User(
            id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            roles=user_data.get("roles", []),
            permissions=user_data.get("permissions", []),
            is_active=user_data.get("is_active", False),
            is_verified=user_data.get("is_verified", False),
            created_at=user_data["created_at"],
            last_login=user_data.get("last_login"),
            failed_login_attempts=user_data.get("failed_login_attempts", 0),
            two_factor_enabled=user_data.get("two_factor_enabled", False),
        )

    @strawberry.field
    async def detector(
        self, 
        info: Info[GraphQLContext, None], 
        id: UUID
    ) -> Optional[Detector]:
        """Get detector by ID."""
        container = info.context.container
        detector_repo = container.detector_repository()
        
        detector_entity = detector_repo.find_by_id(id)
        if not detector_entity:
            return None
            
        return Detector(
            id=detector_entity.id,
            name=detector_entity.name,
            algorithm_name=detector_entity.algorithm_name,
            contamination_rate=detector_entity.contamination_rate.value,
            parameters=detector_entity.parameters,
            metadata=detector_entity.metadata,
            is_fitted=detector_entity.is_fitted,
            created_at=detector_entity.created_at,
            updated_at=detector_entity.updated_at,
        )

    @strawberry.field
    async def detectors(
        self,
        info: Info[GraphQLContext, None],
        filter: Optional[DetectorFilter] = None,
        sort: Optional[DetectorSort] = None,
        first: Optional[int] = None,
        after: Optional[str] = None,
    ) -> DetectorConnection:
        """Get detectors with pagination and filtering."""
        container = info.context.container
        detector_repo = container.detector_repository()
        
        # Build filter criteria
        filter_criteria = {}
        if filter:
            if filter.name:
                filter_criteria["name"] = filter.name
            if filter.algorithm_name:
                filter_criteria["algorithm_name"] = filter.algorithm_name
            if filter.is_fitted is not None:
                filter_criteria["is_fitted"] = filter.is_fitted
            if filter.created_after:
                filter_criteria["created_after"] = filter.created_after
            if filter.created_before:
                filter_criteria["created_before"] = filter.created_before

        # Get detectors
        detectors = detector_repo.find_all(
            filter=filter_criteria,
            sort=sort.field if sort else "created_at",
            order=sort.order.value if sort else "desc",
            limit=first or 50,
            offset=0,  # Implement cursor-based pagination
        )
        
        # Convert to GraphQL types
        detector_types = [
            Detector(
                id=d.id,
                name=d.name,
                algorithm_name=d.algorithm_name,
                contamination_rate=d.contamination_rate.value,
                parameters=d.parameters,
                metadata=d.metadata,
                is_fitted=d.is_fitted,
                created_at=d.created_at,
                updated_at=d.updated_at,
            )
            for d in detectors
        ]
        
        # Get total count
        total_count = detector_repo.count(filter=filter_criteria)
        
        return DetectorConnection(
            edges=detector_types,
            page_info={
                "has_next_page": len(detector_types) == (first or 50),
                "has_previous_page": False,  # Implement proper cursor logic
                "start_cursor": None,
                "end_cursor": None,
            },
            total_count=total_count,
        )

    @strawberry.field
    async def detection_result(
        self,
        info: Info[GraphQLContext, None],
        id: UUID,
    ) -> Optional[AnomalyDetectionResult]:
        """Get detection result by ID."""
        container = info.context.container
        detection_repo = container.detection_repository()
        
        result = detection_repo.find_by_id(id)
        if not result:
            return None
            
        return AnomalyDetectionResult(
            id=result.id,
            detector_id=result.detector_id,
            dataset_id=result.dataset_id,
            anomaly_indices=result.anomaly_indices,
            anomaly_scores=result.anomaly_scores,
            threshold=result.threshold,
            processing_time_ms=result.processing_time_ms,
            algorithm_metadata=result.algorithm_metadata,
            created_at=result.created_at,
        )

    @strawberry.field
    async def detection_results(
        self,
        info: Info[GraphQLContext, None],
        filter: Optional[DetectionResultFilter] = None,
        sort: Optional[DetectionResultSort] = None,
        first: Optional[int] = None,
        after: Optional[str] = None,
    ) -> DetectionResultConnection:
        """Get detection results with pagination and filtering."""
        container = info.context.container
        detection_repo = container.detection_repository()
        
        # Build filter criteria
        filter_criteria = {}
        if filter:
            if filter.detector_id:
                filter_criteria["detector_id"] = filter.detector_id
            if filter.dataset_id:
                filter_criteria["dataset_id"] = filter.dataset_id
            if filter.created_after:
                filter_criteria["created_after"] = filter.created_after
            if filter.created_before:
                filter_criteria["created_before"] = filter.created_before
            if filter.min_anomaly_count:
                filter_criteria["min_anomaly_count"] = filter.min_anomaly_count
            if filter.max_anomaly_count:
                filter_criteria["max_anomaly_count"] = filter.max_anomaly_count

        # Get results
        results = detection_repo.find_all(
            filter=filter_criteria,
            sort=sort.field if sort else "created_at",
            order=sort.order.value if sort else "desc",
            limit=first or 50,
            offset=0,
        )
        
        # Convert to GraphQL types
        result_types = [
            AnomalyDetectionResult(
                id=r.id,
                detector_id=r.detector_id,
                dataset_id=r.dataset_id,
                anomaly_indices=r.anomaly_indices,
                anomaly_scores=r.anomaly_scores,
                threshold=r.threshold,
                processing_time_ms=r.processing_time_ms,
                algorithm_metadata=r.algorithm_metadata,
                created_at=r.created_at,
            )
            for r in results
        ]
        
        # Get total count
        total_count = detection_repo.count(filter=filter_criteria)
        
        return DetectionResultConnection(
            edges=result_types,
            page_info={
                "has_next_page": len(result_types) == (first or 50),
                "has_previous_page": False,
                "start_cursor": None,
                "end_cursor": None,
            },
            total_count=total_count,
        )

    @strawberry.field
    async def detection_job(
        self,
        info: Info[GraphQLContext, None],
        id: UUID,
    ) -> Optional[DetectionJob]:
        """Get detection job by ID."""
        container = info.context.container
        job_service = container.job_service()
        
        job_data = await job_service.get_detection_job(id)
        if not job_data:
            return None
            
        return DetectionJob(
            id=job_data["id"],
            detector_id=job_data["detector_id"],
            dataset_id=job_data["dataset_id"],
            status=job_data["status"],
            progress=job_data.get("progress", 0.0),
            started_at=job_data["started_at"],
            completed_at=job_data.get("completed_at"),
            error_message=job_data.get("error_message"),
            result=job_data.get("result"),
        )

    @strawberry.field
    async def training_job(
        self,
        info: Info[GraphQLContext, None],
        id: UUID,
    ) -> Optional[TrainingJob]:
        """Get training job by ID."""
        container = info.context.container
        job_service = container.job_service()
        
        job_data = await job_service.get_training_job(id)
        if not job_data:
            return None
            
        return TrainingJob(
            id=job_data["id"],
            detector_id=job_data["detector_id"],
            dataset_id=job_data["dataset_id"],
            status=job_data["status"],
            progress=job_data.get("progress", 0.0),
            started_at=job_data["started_at"],
            completed_at=job_data.get("completed_at"),
            training_time_ms=job_data.get("training_time_ms"),
            validation_results=job_data.get("validation_results"),
            error_message=job_data.get("error_message"),
        )

    @strawberry.field
    async def audit_logs(
        self,
        info: Info[GraphQLContext, None],
        filter: Optional[AuditLogFilter] = None,
        sort: Optional[AuditLogSort] = None,
        first: Optional[int] = None,
        after: Optional[str] = None,
    ) -> AuditLogConnection:
        """Get audit logs with pagination and filtering."""
        container = info.context.container
        security_manager = container.security_manager()
        
        # Build filter criteria
        filter_criteria = {}
        if filter:
            if filter.user_id:
                filter_criteria["user_id"] = str(filter.user_id)
            if filter.action:
                filter_criteria["action"] = filter.action
            if filter.resource:
                filter_criteria["resource"] = filter.resource
            if filter.success is not None:
                filter_criteria["success"] = filter.success
            if filter.min_risk_score:
                filter_criteria["min_risk_score"] = filter.min_risk_score
            if filter.date_from:
                filter_criteria["date_from"] = filter.date_from
            if filter.date_to:
                filter_criteria["date_to"] = filter.date_to

        # Get audit logs
        logs = security_manager.get_audit_logs(
            filter=filter_criteria,
            sort=sort.field if sort else "timestamp",
            order=sort.order.value if sort else "desc",
            limit=first or 50,
            offset=0,
        )
        
        # Convert to GraphQL types
        log_types = [
            AuditLog(
                id=log.log_id,
                user_id=UUID(log.user_id) if log.user_id else None,
                action=log.action,
                resource=log.resource,
                timestamp=log.timestamp,
                ip_address=log.ip_address,
                user_agent=log.user_agent,
                success=log.success,
                details=log.details,
                risk_score=log.risk_score,
            )
            for log in logs
        ]
        
        # Get total count
        total_count = len(security_manager.audit_logs)
        
        return AuditLogConnection(
            edges=log_types,
            page_info={
                "has_next_page": len(log_types) == (first or 50),
                "has_previous_page": False,
                "start_cursor": None,
                "end_cursor": None,
            },
            total_count=total_count,
        )

    @strawberry.field
    async def security_metrics(
        self,
        info: Info[GraphQLContext, None],
    ) -> SecurityMetrics:
        """Get security metrics."""
        container = info.context.container
        security_manager = container.security_manager()
        
        metrics = security_manager.get_security_metrics()
        
        return SecurityMetrics(
            total_users=metrics["total_users"],
            active_sessions=metrics["active_sessions"],
            api_keys_active=metrics["api_keys_active"],
            failed_actions_24h=metrics["failed_actions_24h"],
            high_risk_events_24h=metrics["high_risk_events_24h"],
            blocked_ips=metrics["blocked_ips"],
            audit_logs_total=metrics["audit_logs_total"],
        )

    @strawberry.field
    async def performance_metrics(
        self,
        info: Info[GraphQLContext, None],
    ) -> PerformanceMetrics:
        """Get performance metrics."""
        container = info.context.container
        metrics_service = container.metrics_service()
        
        metrics = await metrics_service.get_performance_metrics()
        
        return PerformanceMetrics(
            avg_detection_time_ms=metrics.get("avg_detection_time_ms", 0.0),
            avg_training_time_ms=metrics.get("avg_training_time_ms", 0.0),
            total_detections=metrics.get("total_detections", 0),
            total_trainings=metrics.get("total_trainings", 0),
            success_rate=metrics.get("success_rate", 0.0),
            error_rate=metrics.get("error_rate", 0.0),
            throughput_per_hour=metrics.get("throughput_per_hour", 0.0),
        )

    @strawberry.field
    async def search_detectors(
        self,
        info: Info[GraphQLContext, None],
        query: str,
        limit: int = 10,
    ) -> List[Detector]:
        """Search detectors by name or algorithm."""
        container = info.context.container
        detector_repo = container.detector_repository()
        
        # Implement search logic
        detectors = detector_repo.search(query, limit=limit)
        
        return [
            Detector(
                id=d.id,
                name=d.name,
                algorithm_name=d.algorithm_name,
                contamination_rate=d.contamination_rate.value,
                parameters=d.parameters,
                metadata=d.metadata,
                is_fitted=d.is_fitted,
                created_at=d.created_at,
                updated_at=d.updated_at,
            )
            for d in detectors
        ]

    @strawberry.field
    async def detector_algorithms(
        self,
        info: Info[GraphQLContext, None],
    ) -> List[str]:
        """Get available detector algorithms."""
        container = info.context.container
        algorithm_service = container.algorithm_service()
        
        return algorithm_service.get_available_algorithms()

    @strawberry.field
    async def detector_metrics(
        self,
        info: Info[GraphQLContext, None],
        detector_id: UUID,
    ) -> Optional[dict]:
        """Get detector performance metrics."""
        container = info.context.container
        metrics_service = container.metrics_service()
        
        return await metrics_service.get_detector_metrics(detector_id)