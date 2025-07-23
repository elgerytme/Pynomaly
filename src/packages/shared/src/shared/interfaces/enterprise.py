"""Enterprise domain interfaces for cross-package communication."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from uuid import UUID

from ..domain.abstractions import ServiceInterface


class AuthServiceInterface(ServiceInterface):
    """Interface for authentication services."""
    
    @abstractmethod
    async def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with username/password."""
        pass
    
    @abstractmethod
    async def authenticate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with JWT token."""
        pass
    
    @abstractmethod
    async def create_user(self, user_data: Dict[str, Any]) -> UUID:
        """Create a new user."""
        pass
    
    @abstractmethod
    async def get_user(self, user_id: UUID) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        pass
    
    @abstractmethod
    async def update_user(self, user_id: UUID, user_data: Dict[str, Any]) -> bool:
        """Update user information."""
        pass
    
    @abstractmethod
    async def delete_user(self, user_id: UUID) -> bool:
        """Delete a user."""
        pass
    
    @abstractmethod
    async def assign_role(self, user_id: UUID, role: str) -> bool:
        """Assign role to user."""
        pass
    
    @abstractmethod
    async def revoke_role(self, user_id: UUID, role: str) -> bool:
        """Revoke role from user."""
        pass
    
    @abstractmethod
    async def check_permission(self, user_id: UUID, resource: str, action: str) -> bool:
        """Check if user has permission for action on resource."""
        pass


class MultiTenantInterface(ServiceInterface):
    """Interface for multi-tenant services."""
    
    @abstractmethod
    async def create_tenant(self, tenant_data: Dict[str, Any]) -> UUID:
        """Create a new tenant."""
        pass
    
    @abstractmethod
    async def get_tenant(self, tenant_id: UUID) -> Optional[Dict[str, Any]]:
        """Get tenant by ID."""
        pass
    
    @abstractmethod
    async def list_tenants(self, user_id: Optional[UUID] = None) -> List[Dict[str, Any]]:
        """List tenants, optionally filtered by user."""
        pass
    
    @abstractmethod
    async def update_tenant(self, tenant_id: UUID, tenant_data: Dict[str, Any]) -> bool:
        """Update tenant information."""
        pass
    
    @abstractmethod
    async def delete_tenant(self, tenant_id: UUID) -> bool:
        """Delete a tenant."""
        pass
    
    @abstractmethod
    async def add_user_to_tenant(self, tenant_id: UUID, user_id: UUID, role: Optional[str] = None) -> bool:
        """Add user to tenant with optional role."""
        pass
    
    @abstractmethod
    async def remove_user_from_tenant(self, tenant_id: UUID, user_id: UUID) -> bool:
        """Remove user from tenant."""
        pass
    
    @abstractmethod
    async def get_tenant_users(self, tenant_id: UUID) -> List[Dict[str, Any]]:
        """Get all users in a tenant."""
        pass
    
    @abstractmethod
    async def get_user_tenants(self, user_id: UUID) -> List[Dict[str, Any]]:
        """Get all tenants for a user."""
        pass
    
    @abstractmethod
    async def isolate_data(self, tenant_id: UUID, query: str) -> str:
        """Add tenant isolation to a database query."""
        pass


class OperationsInterface(ServiceInterface):
    """Interface for operations and monitoring services."""
    
    @abstractmethod
    async def collect_metrics(self, service_name: str) -> Dict[str, Any]:
        """Collect metrics for a service."""
        pass
    
    @abstractmethod
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        pass
    
    @abstractmethod
    async def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get status of a specific service."""
        pass
    
    @abstractmethod
    async def create_alert(self, alert_data: Dict[str, Any]) -> UUID:
        """Create a new alert."""
        pass
    
    @abstractmethod
    async def get_alerts(self, service_name: Optional[str] = None, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get alerts with optional filters."""
        pass
    
    @abstractmethod
    async def acknowledge_alert(self, alert_id: UUID, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        pass
    
    @abstractmethod
    async def resolve_alert(self, alert_id: UUID, resolved_by: str, resolution_notes: str) -> bool:
        """Resolve an alert."""
        pass
    
    @abstractmethod
    async def get_logs(self, service_name: str, time_range: Dict[str, Any], level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get logs for a service."""
        pass
    
    @abstractmethod
    async def trigger_backup(self, service_name: str) -> UUID:
        """Trigger backup for a service."""
        pass
    
    @abstractmethod
    async def get_backup_status(self, backup_id: UUID) -> Dict[str, Any]:
        """Get backup status."""
        pass


class AuditInterface(ServiceInterface):
    """Interface for audit and compliance services."""
    
    @abstractmethod
    async def log_action(self, user_id: UUID, action: str, resource: str, details: Optional[Dict[str, Any]] = None) -> UUID:
        """Log an audit action."""
        pass
    
    @abstractmethod
    async def get_audit_log(self, resource: Optional[str] = None, user_id: Optional[UUID] = None, time_range: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get audit log entries with optional filters."""
        pass
    
    @abstractmethod
    async def create_compliance_rule(self, rule_data: Dict[str, Any]) -> UUID:
        """Create a compliance rule."""
        pass
    
    @abstractmethod
    async def check_compliance(self, resource: str, resource_id: UUID) -> Dict[str, Any]:
        """Check compliance for a resource."""
        pass
    
    @abstractmethod
    async def generate_compliance_report(self, report_type: str, time_range: Dict[str, Any]) -> UUID:
        """Generate compliance report."""
        pass
    
    @abstractmethod
    async def get_compliance_report(self, report_id: UUID) -> Optional[Dict[str, Any]]:
        """Get compliance report."""
        pass


class GovernanceInterface(ServiceInterface):
    """Interface for data governance services."""
    
    @abstractmethod
    async def create_policy(self, policy_data: Dict[str, Any]) -> UUID:
        """Create a governance policy."""
        pass
    
    @abstractmethod
    async def get_policy(self, policy_id: UUID) -> Optional[Dict[str, Any]]:
        """Get governance policy."""
        pass
    
    @abstractmethod
    async def list_policies(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List governance policies."""
        pass
    
    @abstractmethod
    async def apply_policy(self, policy_id: UUID, resource: str, resource_id: UUID) -> Dict[str, Any]:
        """Apply policy to a resource."""
        pass
    
    @abstractmethod
    async def check_policy_compliance(self, resource: str, resource_id: UUID) -> Dict[str, Any]:
        """Check policy compliance for a resource."""
        pass
    
    @abstractmethod
    async def create_data_lineage(self, lineage_data: Dict[str, Any]) -> UUID:
        """Create data lineage record."""
        pass
    
    @abstractmethod
    async def get_data_lineage(self, dataset_id: UUID) -> Optional[Dict[str, Any]]:
        """Get data lineage for dataset."""
        pass
    
    @abstractmethod
    async def classify_data(self, dataset_id: UUID, classification_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Classify data according to governance rules."""
        pass


class ScalabilityInterface(ServiceInterface):
    """Interface for scalability and performance services."""
    
    @abstractmethod
    async def scale_service(self, service_name: str, target_replicas: int) -> bool:
        """Scale a service to target number of replicas."""
        pass
    
    @abstractmethod
    async def get_scaling_metrics(self, service_name: str) -> Dict[str, Any]:
        """Get scaling metrics for a service."""
        pass
    
    @abstractmethod
    async def create_auto_scaling_policy(self, service_name: str, policy_data: Dict[str, Any]) -> UUID:
        """Create auto-scaling policy."""
        pass
    
    @abstractmethod
    async def get_resource_usage(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """Get resource usage metrics."""
        pass
    
    @abstractmethod
    async def optimize_resources(self, service_name: str) -> Dict[str, Any]:
        """Optimize resource allocation for a service."""
        pass
    
    @abstractmethod
    async def get_performance_recommendations(self, service_name: str) -> List[Dict[str, Any]]:
        """Get performance optimization recommendations."""
        pass