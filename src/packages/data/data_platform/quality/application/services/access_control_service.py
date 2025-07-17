"""Access control services for role-based and attribute-based authorization."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from ...domain.entities.security_entity import (
    UserRole, AccessPermission, SecurityEvent, SecurityEventType
)


class RoleBasedAccessControlService:
    """Service for role-based access control (RBAC)."""
    
    def __init__(self):
        """Initialize RBAC service."""
        self.roles: Dict[str, UserRole] = {}
        self.user_roles: Dict[str, Set[str]] = {}
        self.permission_cache: Dict[str, Dict[str, bool]] = {}
        self.access_log: List[SecurityEvent] = []
        self._initialize_default_roles()
    
    def create_role(self, role_name: str, description: str, permissions: Set[str], 
                   created_by: str, inherited_roles: Optional[Set[str]] = None) -> UserRole:
        """
        Create a new role.
        
        Args:
            role_name: Role name
            description: Role description
            permissions: Set of permissions
            created_by: User creating the role
            inherited_roles: Roles to inherit from
            
        Returns:
            Created role
        """
        role = UserRole(
            name=role_name,
            description=description,
            permissions=permissions,
            inherited_roles=inherited_roles or set(),
            created_by=created_by
        )
        
        self.roles[role_name] = role
        
        # Log role creation
        self._log_security_event(
            event_type=SecurityEventType.AUTHORIZATION,
            action="role_created",
            result="success",
            details={
                'role_name': role_name,
                'permissions_count': len(permissions),
                'created_by': created_by
            }
        )
        
        return role
    
    def assign_role(self, user_id: str, role_name: str, assigned_by: str) -> bool:
        """
        Assign role to user.
        
        Args:
            user_id: User identifier
            role_name: Role name
            assigned_by: User assigning the role
            
        Returns:
            Whether assignment was successful
        """
        if role_name not in self.roles:
            return False
        
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        
        self.user_roles[user_id].add(role_name)
        
        # Clear permission cache for user
        if user_id in self.permission_cache:
            del self.permission_cache[user_id]
        
        # Log role assignment
        self._log_security_event(
            event_type=SecurityEventType.AUTHORIZATION,
            action="role_assigned",
            result="success",
            details={
                'user_id': user_id,
                'role_name': role_name,
                'assigned_by': assigned_by
            }
        )
        
        return True
    
    def revoke_role(self, user_id: str, role_name: str, revoked_by: str) -> bool:
        """
        Revoke role from user.
        
        Args:
            user_id: User identifier
            role_name: Role name
            revoked_by: User revoking the role
            
        Returns:
            Whether revocation was successful
        """
        if user_id not in self.user_roles or role_name not in self.user_roles[user_id]:
            return False
        
        self.user_roles[user_id].discard(role_name)
        
        # Clear permission cache for user
        if user_id in self.permission_cache:
            del self.permission_cache[user_id]
        
        # Log role revocation
        self._log_security_event(
            event_type=SecurityEventType.AUTHORIZATION,
            action="role_revoked",
            result="success",
            details={
                'user_id': user_id,
                'role_name': role_name,
                'revoked_by': revoked_by
            }
        )
        
        return True
    
    def check_permission(self, user_id: str, permission: str) -> bool:
        """
        Check if user has permission.
        
        Args:
            user_id: User identifier
            permission: Permission to check
            
        Returns:
            Whether user has permission
        """
        # Check cache first
        if user_id in self.permission_cache and permission in self.permission_cache[user_id]:
            return self.permission_cache[user_id][permission]
        
        # Get user permissions
        user_permissions = self.get_user_permissions(user_id)
        has_permission = permission in user_permissions
        
        # Cache result
        if user_id not in self.permission_cache:
            self.permission_cache[user_id] = {}
        self.permission_cache[user_id][permission] = has_permission
        
        # Log permission check
        self._log_security_event(
            event_type=SecurityEventType.AUTHORIZATION,
            action="permission_check",
            result="success" if has_permission else "denied",
            details={
                'user_id': user_id,
                'permission': permission,
                'granted': has_permission
            }
        )
        
        return has_permission
    
    def get_user_permissions(self, user_id: str) -> Set[str]:
        """
        Get all permissions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Set of permissions
        """
        if user_id not in self.user_roles:
            return set()
        
        permissions = set()
        
        # Get permissions from direct roles
        for role_name in self.user_roles[user_id]:
            if role_name in self.roles:
                role = self.roles[role_name]
                permissions.update(role.permissions)
                
                # Get permissions from inherited roles
                for inherited_role in role.inherited_roles:
                    if inherited_role in self.roles:
                        permissions.update(self.roles[inherited_role].permissions)
        
        return permissions
    
    def get_user_roles(self, user_id: str) -> Set[str]:
        """
        Get all roles for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Set of role names
        """
        return self.user_roles.get(user_id, set())
    
    def update_role_permissions(self, role_name: str, permissions: Set[str], 
                               updated_by: str) -> bool:
        """
        Update role permissions.
        
        Args:
            role_name: Role name
            permissions: New permissions
            updated_by: User updating the role
            
        Returns:
            Whether update was successful
        """
        if role_name not in self.roles:
            return False
        
        old_permissions = self.roles[role_name].permissions.copy()
        self.roles[role_name].permissions = permissions
        
        # Clear permission cache for all users with this role
        for user_id, user_roles in self.user_roles.items():
            if role_name in user_roles and user_id in self.permission_cache:
                del self.permission_cache[user_id]
        
        # Log role update
        self._log_security_event(
            event_type=SecurityEventType.AUTHORIZATION,
            action="role_updated",
            result="success",
            details={
                'role_name': role_name,
                'old_permissions': list(old_permissions),
                'new_permissions': list(permissions),
                'updated_by': updated_by
            }
        )
        
        return True
    
    def _initialize_default_roles(self) -> None:
        """Initialize default roles."""
        # Administrator role
        self.create_role(
            role_name="admin",
            description="System administrator with full access",
            permissions={
                "system.admin", "user.create", "user.read", "user.update", "user.delete",
                "role.create", "role.read", "role.update", "role.delete",
                "data.read", "data.write", "data.delete",
                "audit.read", "security.admin"
            },
            created_by="system"
        )
        
        # Data scientist role
        self.create_role(
            role_name="data_scientist",
            description="Data scientist with data analysis permissions",
            permissions={
                "data.read", "data.analyze", "model.create", "model.read", "model.update",
                "experiment.create", "experiment.read", "experiment.update"
            },
            created_by="system"
        )
        
        # Data steward role
        self.create_role(
            role_name="data_steward",
            description="Data steward with data governance permissions",
            permissions={
                "data.read", "data.write", "data.validate", "quality.read", "quality.write",
                "governance.read", "governance.write", "compliance.read"
            },
            created_by="system"
        )
        
        # Auditor role
        self.create_role(
            role_name="auditor",
            description="Auditor with read-only access for compliance",
            permissions={
                "audit.read", "compliance.read", "data.read", "security.read"
            },
            created_by="system"
        )
        
        # Viewer role
        self.create_role(
            role_name="viewer",
            description="Read-only access to data and reports",
            permissions={
                "data.read", "report.read", "dashboard.read"
            },
            created_by="system"
        )
    
    def _log_security_event(self, event_type: SecurityEventType, action: str, 
                           result: str, details: Dict[str, Any]) -> None:
        """Log security event."""
        event = SecurityEvent(
            event_type=event_type,
            action=action,
            result=result,
            details=details
        )
        self.access_log.append(event)


class AttributeBasedAccessControlService:
    """Service for attribute-based access control (ABAC)."""
    
    def __init__(self):
        """Initialize ABAC service."""
        self.policies: Dict[str, Dict[str, Any]] = {}
        self.attribute_providers: Dict[str, callable] = {}
        self.decision_cache: Dict[str, Dict[str, Any]] = {}
        self.access_log: List[SecurityEvent] = []
    
    def create_policy(self, policy_id: str, policy_config: Dict[str, Any]) -> None:
        """
        Create access policy.
        
        Args:
            policy_id: Policy identifier
            policy_config: Policy configuration
        """
        self.policies[policy_id] = {
            'config': policy_config,
            'created_at': datetime.now(),
            'active': True
        }
        
        # Log policy creation
        self._log_security_event(
            event_type=SecurityEventType.AUTHORIZATION,
            action="policy_created",
            result="success",
            details={
                'policy_id': policy_id,
                'policy_type': policy_config.get('type', 'unknown')
            }
        )
    
    def evaluate_access(self, subject: Dict[str, Any], resource: Dict[str, Any], 
                       action: str, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate access request.
        
        Args:
            subject: Subject attributes
            resource: Resource attributes
            action: Action being requested
            environment: Environment attributes
            
        Returns:
            Access decision
        """
        # Create context
        context = {
            'subject': subject,
            'resource': resource,
            'action': action,
            'environment': environment,
            'timestamp': datetime.now()
        }
        
        # Check cache
        cache_key = self._generate_cache_key(context)
        if cache_key in self.decision_cache:
            cached_decision = self.decision_cache[cache_key]
            if cached_decision['expires_at'] > datetime.now():
                return cached_decision['decision']
        
        # Evaluate policies
        decision = self._evaluate_policies(context)
        
        # Cache decision
        self.decision_cache[cache_key] = {
            'decision': decision,
            'expires_at': datetime.now() + timedelta(minutes=5)
        }
        
        # Log access decision
        self._log_security_event(
            event_type=SecurityEventType.AUTHORIZATION,
            action="access_evaluated",
            result="granted" if decision['permit'] else "denied",
            details={
                'subject_id': subject.get('id'),
                'resource_id': resource.get('id'),
                'action': action,
                'decision': decision['permit'],
                'reasons': decision.get('reasons', [])
            }
        )
        
        return decision
    
    def register_attribute_provider(self, attribute_type: str, provider: callable) -> None:
        """
        Register attribute provider.
        
        Args:
            attribute_type: Type of attributes provided
            provider: Provider function
        """
        self.attribute_providers[attribute_type] = provider
    
    def _evaluate_policies(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate access policies."""
        decision = {
            'permit': False,
            'reasons': [],
            'applicable_policies': [],
            'obligations': []
        }
        
        # Evaluate each policy
        for policy_id, policy_data in self.policies.items():
            if not policy_data['active']:
                continue
            
            policy_config = policy_data['config']
            
            # Check if policy applies
            if self._policy_applies(policy_config, context):
                decision['applicable_policies'].append(policy_id)
                
                # Evaluate policy condition
                if self._evaluate_condition(policy_config.get('condition', {}), context):
                    if policy_config.get('effect') == 'permit':
                        decision['permit'] = True
                        decision['reasons'].append(f'Policy {policy_id} permits access')
                    else:
                        decision['permit'] = False
                        decision['reasons'].append(f'Policy {policy_id} denies access')
                        break  # Deny takes precedence
                    
                    # Add obligations
                    if 'obligations' in policy_config:
                        decision['obligations'].extend(policy_config['obligations'])
        
        return decision
    
    def _policy_applies(self, policy_config: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if policy applies to the context."""
        # Check target
        target = policy_config.get('target', {})
        
        # Check subject attributes
        subject_match = self._match_attributes(
            target.get('subject', {}), 
            context['subject']
        )
        
        # Check resource attributes
        resource_match = self._match_attributes(
            target.get('resource', {}), 
            context['resource']
        )
        
        # Check action
        action_match = True
        if 'action' in target:
            action_match = context['action'] in target['action']
        
        return subject_match and resource_match and action_match
    
    def _match_attributes(self, required: Dict[str, Any], actual: Dict[str, Any]) -> bool:
        """Match required attributes against actual attributes."""
        for attr_name, attr_value in required.items():
            if attr_name not in actual:
                return False
            
            if isinstance(attr_value, list):
                if actual[attr_name] not in attr_value:
                    return False
            else:
                if actual[attr_name] != attr_value:
                    return False
        
        return True
    
    def _evaluate_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate policy condition."""
        if not condition:
            return True
        
        condition_type = condition.get('type', 'and')
        
        if condition_type == 'and':
            return all(self._evaluate_condition(sub_condition, context) 
                      for sub_condition in condition.get('conditions', []))
        
        elif condition_type == 'or':
            return any(self._evaluate_condition(sub_condition, context) 
                      for sub_condition in condition.get('conditions', []))
        
        elif condition_type == 'not':
            return not self._evaluate_condition(condition.get('condition', {}), context)
        
        elif condition_type == 'attribute':
            return self._evaluate_attribute_condition(condition, context)
        
        elif condition_type == 'time':
            return self._evaluate_time_condition(condition, context)
        
        elif condition_type == 'location':
            return self._evaluate_location_condition(condition, context)
        
        return False
    
    def _evaluate_attribute_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate attribute condition."""
        attribute_path = condition.get('attribute', '')
        operator = condition.get('operator', 'equals')
        value = condition.get('value')
        
        # Get attribute value from context
        actual_value = self._get_attribute_value(attribute_path, context)
        
        # Evaluate based on operator
        if operator == 'equals':
            return actual_value == value
        elif operator == 'not_equals':
            return actual_value != value
        elif operator == 'in':
            return actual_value in value
        elif operator == 'not_in':
            return actual_value not in value
        elif operator == 'greater_than':
            return actual_value > value
        elif operator == 'less_than':
            return actual_value < value
        elif operator == 'contains':
            return value in str(actual_value)
        
        return False
    
    def _evaluate_time_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate time-based condition."""
        current_time = context['timestamp']
        
        if 'time_range' in condition:
            time_range = condition['time_range']
            start_time = datetime.fromisoformat(time_range['start'])
            end_time = datetime.fromisoformat(time_range['end'])
            return start_time <= current_time <= end_time
        
        if 'day_of_week' in condition:
            allowed_days = condition['day_of_week']
            return current_time.weekday() in allowed_days
        
        if 'time_of_day' in condition:
            time_range = condition['time_of_day']
            start_hour = time_range['start_hour']
            end_hour = time_range['end_hour']
            return start_hour <= current_time.hour <= end_hour
        
        return True
    
    def _evaluate_location_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate location-based condition."""
        user_location = context['environment'].get('location', {})
        
        if 'allowed_countries' in condition:
            user_country = user_location.get('country')
            return user_country in condition['allowed_countries']
        
        if 'allowed_ip_ranges' in condition:
            user_ip = context['environment'].get('ip_address')
            # In a real implementation, this would check IP ranges
            return user_ip is not None
        
        return True
    
    def _get_attribute_value(self, attribute_path: str, context: Dict[str, Any]) -> Any:
        """Get attribute value from context."""
        parts = attribute_path.split('.')
        current = context
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current
    
    def _generate_cache_key(self, context: Dict[str, Any]) -> str:
        """Generate cache key for access decision."""
        key_parts = [
            context['subject'].get('id', ''),
            context['resource'].get('id', ''),
            context['action'],
            str(context['timestamp'].hour)  # Cache per hour
        ]
        return '|'.join(key_parts)
    
    def _log_security_event(self, event_type: SecurityEventType, action: str, 
                           result: str, details: Dict[str, Any]) -> None:
        """Log security event."""
        event = SecurityEvent(
            event_type=event_type,
            action=action,
            result=result,
            details=details
        )
        self.access_log.append(event)


class AccessControlOrchestrationService:
    """Service for orchestrating RBAC and ABAC."""
    
    def __init__(self, rbac_service: RoleBasedAccessControlService, 
                 abac_service: AttributeBasedAccessControlService):
        """
        Initialize access control orchestration service.
        
        Args:
            rbac_service: RBAC service
            abac_service: ABAC service
        """
        self.rbac_service = rbac_service
        self.abac_service = abac_service
    
    def check_access(self, user_id: str, resource: Dict[str, Any], action: str, 
                    context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check access using both RBAC and ABAC.
        
        Args:
            user_id: User identifier
            resource: Resource being accessed
            action: Action being performed
            context: Additional context
            
        Returns:
            Access decision
        """
        # Check RBAC first
        permission = f"{resource.get('type', 'unknown')}.{action}"
        rbac_permitted = self.rbac_service.check_permission(user_id, permission)
        
        # If RBAC denies, no need to check ABAC
        if not rbac_permitted:
            return {
                'permitted': False,
                'reason': 'RBAC denied access',
                'method': 'RBAC'
            }
        
        # Check ABAC for fine-grained control
        subject = {
            'id': user_id,
            'roles': list(self.rbac_service.get_user_roles(user_id)),
            'permissions': list(self.rbac_service.get_user_permissions(user_id))
        }
        
        abac_decision = self.abac_service.evaluate_access(
            subject=subject,
            resource=resource,
            action=action,
            environment=context
        )
        
        return {
            'permitted': rbac_permitted and abac_decision['permit'],
            'reason': 'RBAC and ABAC both permit' if abac_decision['permit'] else 'ABAC denied access',
            'method': 'RBAC + ABAC',
            'rbac_result': rbac_permitted,
            'abac_result': abac_decision,
            'obligations': abac_decision.get('obligations', [])
        }