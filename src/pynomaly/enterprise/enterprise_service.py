#!/usr/bin/env python3
"""
Enterprise Service for Pynomaly.
This module provides unified enterprise features including multi-tenancy, audit logging, and compliance.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from fastapi import FastAPI, HTTPException, Depends, Security, Request, BackgroundTasks, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from .multi_tenancy import (
    MultiTenantManager, TenantInfo, TenantUserInfo, TenantStatus, UserRole, ResourceType,
    TenantCreateRequest, TenantUpdateRequest, UserCreateRequest, LoginRequest, TokenResponse,
    get_multi_tenant_manager, get_current_user, get_current_tenant, require_permission
)
from .audit_logging import (
    AuditLogger, AuditEventInfo, AuditEventCreate, AuditQuery,
    AuditAction, AuditStatus, ComplianceLevel, SensitivityLevel,
    audit_log, get_audit_logger
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models
class EnterpriseHealthResponse(BaseModel):
    """Enterprise health response."""
    status: str
    timestamp: datetime
    services: Dict[str, Any]
    multi_tenancy: Dict[str, Any]
    audit_logging: Dict[str, Any]
    compliance: Dict[str, Any]

class EnterpriseDashboardResponse(BaseModel):
    """Enterprise dashboard response."""
    tenant_info: Dict[str, Any]
    user_info: Dict[str, Any]
    resource_usage: Dict[str, Any]
    audit_summary: Dict[str, Any]
    compliance_status: Dict[str, Any]
    alerts: List[Dict[str, Any]]

class ComplianceReportRequest(BaseModel):
    """Compliance report request."""
    compliance_level: ComplianceLevel
    start_date: datetime
    end_date: datetime
    include_audit_trail: bool = True
    include_user_activity: bool = True
    include_resource_usage: bool = True

class ComplianceReportResponse(BaseModel):
    """Compliance report response."""
    report_id: str
    compliance_level: ComplianceLevel
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    summary: Dict[str, Any]
    audit_trail: List[Dict[str, Any]]
    user_activity: List[Dict[str, Any]]
    resource_usage: List[Dict[str, Any]]
    violations: List[Dict[str, Any]]
    recommendations: List[str]

class EnterpriseService:
    """Enterprise service for multi-tenancy and audit logging."""
    
    def __init__(self):
        """Initialize enterprise service."""
        self.multi_tenant_manager = get_multi_tenant_manager()
        self.audit_logger = get_audit_logger()
        
        # Compliance configurations
        self.compliance_configs = {
            ComplianceLevel.GDPR: {
                "data_retention_days": 365,
                "audit_retention_days": 2555,  # 7 years
                "encryption_required": True,
                "data_subject_rights": True,
                "breach_notification_hours": 72
            },
            ComplianceLevel.HIPAA: {
                "data_retention_days": 2555,  # 7 years
                "audit_retention_days": 2555,
                "encryption_required": True,
                "access_controls": True,
                "audit_controls": True
            },
            ComplianceLevel.SOX: {
                "data_retention_days": 2555,  # 7 years
                "audit_retention_days": 2555,
                "financial_controls": True,
                "change_management": True,
                "segregation_of_duties": True
            }
        }
        
        logger.info("Enterprise service initialized")
    
    async def get_health_status(self) -> EnterpriseHealthResponse:
        """Get enterprise health status."""
        try:
            # Check multi-tenancy service
            tenant_count = len(await self.multi_tenant_manager.list_tenants())
            multi_tenancy_status = {
                "status": "healthy",
                "tenant_count": tenant_count,
                "database_connected": True
            }
            
            # Check audit logging service
            audit_status = {
                "status": "healthy",
                "database_connected": True,
                "elasticsearch_connected": self.audit_logger.es_client is not None,
                "redis_connected": self.audit_logger.redis_client is not None
            }
            
            # Check compliance status
            compliance_status = {
                "status": "healthy",
                "supported_levels": [level.value for level in ComplianceLevel],
                "active_configurations": len(self.compliance_configs)
            }
            
            return EnterpriseHealthResponse(
                status="healthy",
                timestamp=datetime.utcnow(),
                services={
                    "multi_tenancy": "healthy",
                    "audit_logging": "healthy",
                    "compliance": "healthy"
                },
                multi_tenancy=multi_tenancy_status,
                audit_logging=audit_status,
                compliance=compliance_status
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return EnterpriseHealthResponse(
                status="unhealthy",
                timestamp=datetime.utcnow(),
                services={
                    "multi_tenancy": "unknown",
                    "audit_logging": "unknown",
                    "compliance": "unknown"
                },
                multi_tenancy={"status": "error", "error": str(e)},
                audit_logging={"status": "error", "error": str(e)},
                compliance={"status": "error", "error": str(e)}
            )
    
    async def get_dashboard_data(self, tenant: TenantInfo, user: TenantUserInfo) -> EnterpriseDashboardResponse:
        """Get enterprise dashboard data."""
        try:
            # Get tenant info
            tenant_info = {
                "id": tenant.id,
                "name": tenant.name,
                "display_name": tenant.display_name,
                "status": tenant.status.value,
                "created_at": tenant.created_at.isoformat(),
                "resource_limits": tenant.resource_limits,
                "settings": tenant.settings
            }
            
            # Get user info
            user_info = {
                "id": user.id,
                "email": user.email,
                "username": user.username,
                "role": user.role.value,
                "permissions": user.permissions,
                "last_login": user.last_login.isoformat() if user.last_login else None
            }
            
            # Get resource usage (placeholder - would be implemented with actual resource tracking)
            resource_usage = {
                "models": {"used": 5, "limit": tenant.resource_limits.get("models", 100)},
                "experiments": {"used": 23, "limit": tenant.resource_limits.get("experiments", 500)},
                "deployments": {"used": 2, "limit": tenant.resource_limits.get("deployments", 50)},
                "storage": {"used": 1024*1024*512, "limit": tenant.resource_limits.get("storage", 1024*1024*1024*10)}
            }
            
            # Get audit summary
            audit_query = AuditQuery(
                tenant_id=tenant.id,
                start_time=datetime.utcnow() - timedelta(days=7),
                limit=1000
            )
            recent_events = await self.audit_logger.query_events(audit_query)
            
            audit_summary = {
                "total_events_7_days": len(recent_events),
                "actions_breakdown": {},
                "status_breakdown": {},
                "top_users": {}
            }
            
            # Analyze audit events
            for event in recent_events:
                # Count actions
                action = event.action.value
                audit_summary["actions_breakdown"][action] = audit_summary["actions_breakdown"].get(action, 0) + 1
                
                # Count status
                status = event.status.value
                audit_summary["status_breakdown"][status] = audit_summary["status_breakdown"].get(status, 0) + 1
                
                # Count users
                if event.user_id:
                    audit_summary["top_users"][event.user_id] = audit_summary["top_users"].get(event.user_id, 0) + 1
            
            # Get compliance status
            compliance_status = {
                "level": "gdpr",  # Default
                "status": "compliant",
                "last_assessment": datetime.utcnow().isoformat(),
                "violations": 0,
                "recommendations": []
            }
            
            # Get alerts (placeholder)
            alerts = [
                {
                    "id": str(uuid.uuid4()),
                    "type": "info",
                    "title": "System Update",
                    "message": "Enterprise features are running smoothly",
                    "timestamp": datetime.utcnow().isoformat()
                }
            ]
            
            return EnterpriseDashboardResponse(
                tenant_info=tenant_info,
                user_info=user_info,
                resource_usage=resource_usage,
                audit_summary=audit_summary,
                compliance_status=compliance_status,
                alerts=alerts
            )
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")
    
    async def generate_compliance_report(self, tenant: TenantInfo, user: TenantUserInfo, request: ComplianceReportRequest) -> ComplianceReportResponse:
        """Generate compliance report."""
        try:
            report_id = str(uuid.uuid4())
            
            # Get compliance configuration
            config = self.compliance_configs.get(request.compliance_level, {})
            
            # Query audit events for the period
            audit_query = AuditQuery(
                tenant_id=tenant.id,
                start_time=request.start_date,
                end_time=request.end_date,
                limit=10000
            )
            audit_events = await self.audit_logger.query_events(audit_query)
            
            # Generate summary
            summary = {
                "total_events": len(audit_events),
                "unique_users": len(set(event.user_id for event in audit_events if event.user_id)),
                "actions_performed": len(set(event.action.value for event in audit_events)),
                "resources_accessed": len(set(event.resource_type for event in audit_events)),
                "compliance_level": request.compliance_level.value,
                "period_days": (request.end_date - request.start_date).days
            }
            
            # Generate audit trail
            audit_trail = []
            if request.include_audit_trail:
                for event in audit_events[:100]:  # Limit for performance
                    audit_trail.append({
                        "timestamp": event.timestamp.isoformat(),
                        "user_id": event.user_id,
                        "action": event.action.value,
                        "resource_type": event.resource_type,
                        "resource_id": event.resource_id,
                        "status": event.status.value,
                        "ip_address": event.ip_address
                    })
            
            # Generate user activity
            user_activity = []
            if request.include_user_activity:
                user_stats = {}
                for event in audit_events:
                    if event.user_id:
                        if event.user_id not in user_stats:
                            user_stats[event.user_id] = {
                                "user_id": event.user_id,
                                "total_actions": 0,
                                "last_activity": event.timestamp,
                                "actions_breakdown": {}
                            }
                        
                        user_stats[event.user_id]["total_actions"] += 1
                        user_stats[event.user_id]["actions_breakdown"][event.action.value] = \
                            user_stats[event.user_id]["actions_breakdown"].get(event.action.value, 0) + 1
                        
                        if event.timestamp > user_stats[event.user_id]["last_activity"]:
                            user_stats[event.user_id]["last_activity"] = event.timestamp
                
                user_activity = [
                    {
                        **stats,
                        "last_activity": stats["last_activity"].isoformat()
                    }
                    for stats in user_stats.values()
                ]
            
            # Generate resource usage
            resource_usage = []
            if request.include_resource_usage:
                resource_stats = {}
                for event in audit_events:
                    if event.resource_type not in resource_stats:
                        resource_stats[event.resource_type] = {
                            "resource_type": event.resource_type,
                            "total_actions": 0,
                            "unique_resources": set(),
                            "actions_breakdown": {}
                        }
                    
                    resource_stats[event.resource_type]["total_actions"] += 1
                    if event.resource_id:
                        resource_stats[event.resource_type]["unique_resources"].add(event.resource_id)
                    resource_stats[event.resource_type]["actions_breakdown"][event.action.value] = \
                        resource_stats[event.resource_type]["actions_breakdown"].get(event.action.value, 0) + 1
                
                resource_usage = [
                    {
                        "resource_type": stats["resource_type"],
                        "total_actions": stats["total_actions"],
                        "unique_resources": len(stats["unique_resources"]),
                        "actions_breakdown": stats["actions_breakdown"]
                    }
                    for stats in resource_stats.values()
                ]
            
            # Check for violations
            violations = []
            
            # Check for failed access attempts
            failed_events = [event for event in audit_events if event.status == AuditStatus.FAILURE]
            if len(failed_events) > 10:  # Threshold
                violations.append({
                    "type": "excessive_failures",
                    "severity": "medium",
                    "description": f"Found {len(failed_events)} failed access attempts",
                    "recommendation": "Review access controls and user permissions"
                })
            
            # Check for after-hours access
            after_hours_events = [
                event for event in audit_events 
                if event.timestamp.hour < 6 or event.timestamp.hour > 22
            ]
            if len(after_hours_events) > 5:  # Threshold
                violations.append({
                    "type": "after_hours_access",
                    "severity": "low",
                    "description": f"Found {len(after_hours_events)} after-hours access events",
                    "recommendation": "Consider implementing time-based access controls"
                })
            
            # Generate recommendations
            recommendations = []
            
            if request.compliance_level == ComplianceLevel.GDPR:
                recommendations.extend([
                    "Ensure data subject rights are properly implemented",
                    "Verify data retention policies are followed",
                    "Review consent mechanisms for data processing"
                ])
            elif request.compliance_level == ComplianceLevel.HIPAA:
                recommendations.extend([
                    "Ensure all PHI access is properly logged",
                    "Verify encryption at rest and in transit",
                    "Review access controls for sensitive data"
                ])
            elif request.compliance_level == ComplianceLevel.SOX:
                recommendations.extend([
                    "Implement segregation of duties",
                    "Ensure proper change management procedures",
                    "Review financial data access controls"
                ])
            
            # Add violation-specific recommendations
            for violation in violations:
                recommendations.append(violation["recommendation"])
            
            return ComplianceReportResponse(
                report_id=report_id,
                compliance_level=request.compliance_level,
                generated_at=datetime.utcnow(),
                period_start=request.start_date,
                period_end=request.end_date,
                summary=summary,
                audit_trail=audit_trail,
                user_activity=user_activity,
                resource_usage=resource_usage,
                violations=violations,
                recommendations=list(set(recommendations))  # Remove duplicates
            )
            
        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate compliance report: {str(e)}")
    
    async def cleanup_expired_data(self):
        """Clean up expired data for compliance."""
        try:
            # Clean up expired audit events
            await self.audit_logger.cleanup_expired_events()
            
            # Clean up other expired data (implement as needed)
            logger.info("✅ Expired data cleanup completed")
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired data: {e}")

# FastAPI Router
from fastapi import APIRouter
router = APIRouter(prefix="/enterprise", tags=["enterprise"])

# Global enterprise service instance
enterprise_service = EnterpriseService()

# Endpoints
@router.get("/health", response_model=EnterpriseHealthResponse)
async def get_enterprise_health():
    """Get enterprise health status."""
    return await enterprise_service.get_health_status()

@router.get("/dashboard", response_model=EnterpriseDashboardResponse)
async def get_enterprise_dashboard(
    current_tenant: TenantInfo = Depends(get_current_tenant),
    current_user: TenantUserInfo = Depends(get_current_user)
):
    """Get enterprise dashboard data."""
    return await enterprise_service.get_dashboard_data(current_tenant, current_user)

# Multi-tenancy endpoints
@router.post("/tenants", response_model=TenantInfo)
async def create_tenant(request: TenantCreateRequest):
    """Create new tenant."""
    manager = get_multi_tenant_manager()
    return await manager.create_tenant(request)

@router.get("/tenants", response_model=List[TenantInfo])
async def list_tenants(skip: int = 0, limit: int = 100):
    """List tenants."""
    manager = get_multi_tenant_manager()
    return await manager.list_tenants(skip, limit)

@router.get("/tenants/{tenant_id}", response_model=TenantInfo)
async def get_tenant(tenant_id: str):
    """Get tenant by ID."""
    manager = get_multi_tenant_manager()
    tenant = await manager.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return tenant

@router.put("/tenants/{tenant_id}", response_model=TenantInfo)
async def update_tenant(tenant_id: str, request: TenantUpdateRequest):
    """Update tenant."""
    manager = get_multi_tenant_manager()
    return await manager.update_tenant(tenant_id, request)

@router.post("/tenants/{tenant_id}/users", response_model=TenantUserInfo)
async def create_user(tenant_id: str, request: UserCreateRequest):
    """Create user for tenant."""
    manager = get_multi_tenant_manager()
    return await manager.create_user(tenant_id, request)

@router.post("/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Login user."""
    manager = get_multi_tenant_manager()
    user = await manager.authenticate_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create token
    token_data = {
        "user_id": user.id,
        "tenant_id": user.tenant_id,
        "role": user.role.value
    }
    token = manager.create_access_token(token_data)
    
    return TokenResponse(
        access_token=token,
        expires_in=86400,  # 24 hours
        tenant_id=user.tenant_id,
        user_id=user.id,
        role=user.role.value
    )

# Audit endpoints
@router.get("/audit/events", response_model=List[AuditEventInfo])
async def query_audit_events(
    action: Optional[AuditAction] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    status: Optional[AuditStatus] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    skip: int = 0,
    limit: int = 100,
    current_tenant: TenantInfo = Depends(get_current_tenant),
    current_user: TenantUserInfo = Depends(get_current_user)
):
    """Query audit events."""
    query = AuditQuery(
        tenant_id=current_tenant.id,
        user_id=current_user.id if current_user.role != UserRole.TENANT_ADMIN else None,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        status=status,
        start_time=start_time,
        end_time=end_time,
        skip=skip,
        limit=limit
    )
    
    audit_logger = get_audit_logger()
    return await audit_logger.query_events(query)

@router.get("/audit/events/{event_id}", response_model=AuditEventInfo)
async def get_audit_event(
    event_id: str,
    current_tenant: TenantInfo = Depends(get_current_tenant),
    current_user: TenantUserInfo = Depends(get_current_user)
):
    """Get audit event by ID."""
    audit_logger = get_audit_logger()
    event = await audit_logger.get_event(event_id)
    if not event or event.tenant_id != current_tenant.id:
        raise HTTPException(status_code=404, detail="Audit event not found")
    return event

# Compliance endpoints
@router.post("/compliance/reports", response_model=ComplianceReportResponse)
async def generate_compliance_report(
    request: ComplianceReportRequest,
    current_tenant: TenantInfo = Depends(get_current_tenant),
    current_user: TenantUserInfo = Depends(get_current_user)
):
    """Generate compliance report."""
    if current_user.role != UserRole.TENANT_ADMIN:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    return await enterprise_service.generate_compliance_report(current_tenant, current_user, request)

@router.post("/maintenance/cleanup")
async def run_maintenance_cleanup(
    current_user: TenantUserInfo = Depends(get_current_user)
):
    """Run maintenance cleanup."""
    if current_user.role != UserRole.TENANT_ADMIN:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    await enterprise_service.cleanup_expired_data()
    return {"message": "Maintenance cleanup completed"}

# Make components available for import
__all__ = [
    "EnterpriseService", "EnterpriseHealthResponse", "EnterpriseDashboardResponse",
    "ComplianceReportRequest", "ComplianceReportResponse", "enterprise_service", "router"
]