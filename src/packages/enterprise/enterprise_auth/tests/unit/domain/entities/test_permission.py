"""
Unit tests for Permission and Role domain entities.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4, UUID

from enterprise_auth.domain.entities.permission import (
    Permission, Role, RolePermission, UserRole,
    PermissionScope, PermissionAction, ResourceType,
    SYSTEM_PERMISSIONS, SYSTEM_ROLES
)


class TestPermission:
    """Test cases for Permission entity."""
    
    def test_permission_creation_basic(self):
        """Test permission creation with basic fields."""
        permission = Permission(
            name="Read Users",
            code="USER.READ",
            description="Permission to read user information",
            scope=PermissionScope.TENANT,
            action=PermissionAction.READ,
            resource_type=ResourceType.USER,
            category="User Management"
        )
        
        assert isinstance(permission.id, UUID)
        assert permission.name == "Read Users"
        assert permission.code == "USER.READ"
        assert permission.description == "Permission to read user information"
        assert permission.scope == PermissionScope.TENANT
        assert permission.action == PermissionAction.READ
        assert permission.resource_type == ResourceType.USER
        assert permission.category == "User Management"
        assert permission.parent_id is None
        assert permission.conditions == {}
        assert permission.constraints == {}
        assert permission.is_active is True
        assert permission.is_system is False
        
    def test_permission_creation_with_all_fields(self):
        """Test permission creation with all fields."""
        parent_id = uuid4()
        created_by = uuid4()
        conditions = {"tenant_type": "enterprise"}
        constraints = {"resource_ids": ["123", "456"]}
        
        permission = Permission(
            name="Admin Users",
            code="user.admin",
            description="Full user administration",
            scope=PermissionScope.GLOBAL,
            action=PermissionAction.ADMIN,
            resource_type=ResourceType.USER,
            category="Administration",
            parent_id=parent_id,
            conditions=conditions,
            constraints=constraints,
            is_active=False,
            is_system=True,
            created_by=created_by
        )
        
        assert permission.name == "Admin Users"
        assert permission.code == "USER.ADMIN"  # Should be uppercase
        assert permission.scope == PermissionScope.GLOBAL
        assert permission.action == PermissionAction.ADMIN
        assert permission.resource_type == ResourceType.USER
        assert permission.parent_id == parent_id
        assert permission.conditions == conditions
        assert permission.constraints == constraints
        assert permission.is_active is False
        assert permission.is_system is True
        assert permission.created_by == created_by
        
    def test_code_validation_uppercase(self):
        """Test permission code is converted to uppercase."""
        permission = Permission(
            name="Test Permission",
            code="test.permission",
            description="Test permission",
            scope=PermissionScope.TENANT,
            action=PermissionAction.READ,
            resource_type=ResourceType.USER,
            category="Test"
        )
        assert permission.code == "TEST.PERMISSION"
        
    def test_code_validation_invalid_characters(self):
        """Test permission code validation with invalid characters."""
        with pytest.raises(ValueError, match="Permission code must contain only alphanumeric characters"):
            Permission(
                name="Test Permission",
                code="test@permission",
                description="Test permission",
                scope=PermissionScope.TENANT,
                action=PermissionAction.READ,
                resource_type=ResourceType.USER,
                category="Test"
            )
            
    def test_name_validation_strip(self):
        """Test permission name is stripped of whitespace."""
        permission = Permission(
            name="  Test Permission  ",
            code="TEST.PERMISSION",
            description="Test permission",
            scope=PermissionScope.TENANT,
            action=PermissionAction.READ,
            resource_type=ResourceType.USER,
            category="Test"
        )
        assert permission.name == "Test Permission"
        
    def test_matches_resource_same_type(self):
        """Test matches_resource with same resource type."""
        permission = Permission(
            name="Read Users",
            code="USER.READ",
            description="Read user permission",
            scope=PermissionScope.TENANT,
            action=PermissionAction.READ,
            resource_type=ResourceType.USER,
            category="User Management"
        )
        
        assert permission.matches_resource(ResourceType.USER) is True
        assert permission.matches_resource(ResourceType.ROLE) is False
        
    def test_matches_resource_with_constraints(self):
        """Test matches_resource with resource constraints."""
        resource_id = uuid4()
        permission = Permission(
            name="Read Specific User",
            code="USER.READ.SPECIFIC",
            description="Read specific user permission",
            scope=PermissionScope.RESOURCE,
            action=PermissionAction.READ,
            resource_type=ResourceType.USER,
            category="User Management",
            constraints={"resource_ids": [str(resource_id), "other-id"]}
        )
        
        assert permission.matches_resource(ResourceType.USER, resource_id) is True
        assert permission.matches_resource(ResourceType.USER, uuid4()) is False
        
    def test_can_perform_action_exact_match(self):
        """Test can_perform_action with exact action match."""
        permission = Permission(
            name="Read Users",
            code="USER.READ",
            description="Read user permission",
            scope=PermissionScope.TENANT,
            action=PermissionAction.READ,
            resource_type=ResourceType.USER,
            category="User Management"
        )
        
        assert permission.can_perform_action(PermissionAction.READ) is True
        assert permission.can_perform_action(PermissionAction.UPDATE) is False
        
    def test_can_perform_action_admin_includes_all(self):
        """Test can_perform_action where ADMIN includes all actions."""
        permission = Permission(
            name="Admin Users",
            code="USER.ADMIN",
            description="Admin user permission",
            scope=PermissionScope.TENANT,
            action=PermissionAction.ADMIN,
            resource_type=ResourceType.USER,
            category="User Management"
        )
        
        assert permission.can_perform_action(PermissionAction.READ) is True
        assert permission.can_perform_action(PermissionAction.UPDATE) is True
        assert permission.can_perform_action(PermissionAction.DELETE) is True
        assert permission.can_perform_action(PermissionAction.CREATE) is True
        
    def test_can_perform_action_manage_includes_all(self):
        """Test can_perform_action where MANAGE includes all actions."""
        permission = Permission(
            name="Manage Users",
            code="USER.MANAGE",
            description="Manage user permission",
            scope=PermissionScope.TENANT,
            action=PermissionAction.MANAGE,
            resource_type=ResourceType.USER,
            category="User Management"
        )
        
        assert permission.can_perform_action(PermissionAction.READ) is True
        assert permission.can_perform_action(PermissionAction.UPDATE) is True
        assert permission.can_perform_action(PermissionAction.DELETE) is True
        
    def test_is_valid_for_scope_active(self):
        """Test is_valid_for_scope for active permission."""
        permission = Permission(
            name="Test Permission",
            code="TEST.PERMISSION",
            description="Test permission",
            scope=PermissionScope.TENANT,
            action=PermissionAction.READ,
            resource_type=ResourceType.USER,
            category="Test",
            is_active=True
        )
        
        assert permission.is_valid_for_scope({}) is True
        
    def test_is_valid_for_scope_inactive(self):
        """Test is_valid_for_scope for inactive permission."""
        permission = Permission(
            name="Test Permission",
            code="TEST.PERMISSION",
            description="Test permission",
            scope=PermissionScope.TENANT,
            action=PermissionAction.READ,
            resource_type=ResourceType.USER,
            category="Test",
            is_active=False
        )
        
        assert permission.is_valid_for_scope({}) is False
        
    def test_is_valid_for_scope_with_conditions_match(self):
        """Test is_valid_for_scope with matching conditions."""
        permission = Permission(
            name="Test Permission",
            code="TEST.PERMISSION",
            description="Test permission",
            scope=PermissionScope.TENANT,
            action=PermissionAction.READ,
            resource_type=ResourceType.USER,
            category="Test",
            conditions={"tenant_type": "enterprise", "region": "us"}
        )
        
        context = {"tenant_type": "enterprise", "region": "us", "extra": "data"}
        assert permission.is_valid_for_scope(context) is True
        
    def test_is_valid_for_scope_with_conditions_no_match(self):
        """Test is_valid_for_scope with non-matching conditions."""
        permission = Permission(
            name="Test Permission",
            code="TEST.PERMISSION",
            description="Test permission",
            scope=PermissionScope.TENANT,
            action=PermissionAction.READ,
            resource_type=ResourceType.USER,
            category="Test",
            conditions={"tenant_type": "enterprise"}
        )
        
        context = {"tenant_type": "free"}
        assert permission.is_valid_for_scope(context) is False
        
    def test_is_valid_for_scope_with_list_conditions_match(self):
        """Test is_valid_for_scope with list conditions that match."""
        permission = Permission(
            name="Test Permission",
            code="TEST.PERMISSION",
            description="Test permission",
            scope=PermissionScope.TENANT,
            action=PermissionAction.READ,
            resource_type=ResourceType.USER,
            category="Test",
            conditions={"regions": ["us", "eu", "asia"]}
        )
        
        context = {"regions": "us"}
        assert permission.is_valid_for_scope(context) is True
        
    def test_is_valid_for_scope_with_list_conditions_no_match(self):
        """Test is_valid_for_scope with list conditions that don't match."""
        permission = Permission(
            name="Test Permission",
            code="TEST.PERMISSION",
            description="Test permission",
            scope=PermissionScope.TENANT,
            action=PermissionAction.READ,
            resource_type=ResourceType.USER,
            category="Test",
            conditions={"regions": ["us", "eu"]}
        )
        
        context = {"regions": "asia"}
        assert permission.is_valid_for_scope(context) is False
        
    def test_full_code_property(self):
        """Test full_code property."""
        permission = Permission(
            name="Read Users",
            code="USER.READ",
            description="Read user permission",
            scope=PermissionScope.TENANT,
            action=PermissionAction.READ,
            resource_type=ResourceType.USER,
            category="User Management"
        )
        
        assert permission.full_code == "tenant:user:read"
        
    def test_to_dict(self):
        """Test to_dict method."""
        permission = Permission(
            name="Test Permission",
            code="TEST.PERMISSION",
            description="Test permission",
            scope=PermissionScope.TENANT,
            action=PermissionAction.READ,
            resource_type=ResourceType.USER,
            category="Test"
        )
        
        data = permission.to_dict()
        assert isinstance(data, dict)
        assert data["name"] == "Test Permission"
        assert data["code"] == "TEST.PERMISSION"
        

class TestRole:
    """Test cases for Role entity."""
    
    def test_role_creation_basic(self):
        """Test role creation with basic fields."""
        role = Role(
            name="Analyst",
            code="ANALYST",
            description="Data analyst role"
        )
        
        assert isinstance(role.id, UUID)
        assert role.name == "Analyst"
        assert role.code == "ANALYST"
        assert role.description == "Data analyst role"
        assert role.tenant_id is None
        assert role.parent_id is None
        assert role.level == 0
        assert role.permission_ids == set()
        assert role.inherited_permissions == set()
        assert role.is_default is False
        assert role.is_system is False
        assert role.is_active is True
        assert role.max_users is None
        assert role.conditions == {}
        
    def test_role_creation_with_all_fields(self):
        """Test role creation with all fields."""
        tenant_id = uuid4()
        parent_id = uuid4()
        created_by = uuid4()
        permission_ids = {uuid4(), uuid4()}
        inherited_permissions = {uuid4()}
        conditions = {"department": "engineering"}
        
        role = Role(
            tenant_id=tenant_id,
            name="Senior Analyst",
            code="senior_analyst",
            description="Senior data analyst role",
            parent_id=parent_id,
            level=2,
            permission_ids=permission_ids,
            inherited_permissions=inherited_permissions,
            is_default=True,
            is_system=True,
            is_active=False,
            max_users=50,
            conditions=conditions,
            created_by=created_by
        )
        
        assert role.tenant_id == tenant_id
        assert role.name == "Senior Analyst"
        assert role.code == "SENIOR_ANALYST"  # Should be uppercase
        assert role.parent_id == parent_id
        assert role.level == 2
        assert role.permission_ids == permission_ids
        assert role.inherited_permissions == inherited_permissions
        assert role.is_default is True
        assert role.is_system is True
        assert role.is_active is False
        assert role.max_users == 50
        assert role.conditions == conditions
        assert role.created_by == created_by
        
    def test_code_validation_uppercase(self):
        """Test role code is converted to uppercase."""
        role = Role(
            name="Test Role",
            code="test-role_123",
            description="Test role"
        )
        assert role.code == "TEST-ROLE_123"
        
    def test_code_validation_invalid_characters(self):
        """Test role code validation with invalid characters."""
        with pytest.raises(ValueError, match="Role code must contain only alphanumeric characters"):
            Role(
                name="Test Role",
                code="test@role",
                description="Test role"
            )
            
    def test_permission_ids_list_conversion(self):
        """Test permission_ids list is converted to set."""
        permission_id1 = uuid4()
        permission_id2 = uuid4()
        
        role = Role(
            name="Test Role",
            code="TEST_ROLE",
            description="Test role",
            permission_ids=[permission_id1, permission_id2, permission_id1]  # Duplicate
        )
        
        assert role.permission_ids == {permission_id1, permission_id2}
        
    def test_inherited_permissions_list_conversion(self):
        """Test inherited_permissions list is converted to set."""
        permission_id1 = uuid4()
        permission_id2 = uuid4()
        
        role = Role(
            name="Test Role",
            code="TEST_ROLE",
            description="Test role",
            inherited_permissions=[permission_id1, permission_id2, permission_id1]  # Duplicate
        )
        
        assert role.inherited_permissions == {permission_id1, permission_id2}
        
    def test_add_permission(self):
        """Test add_permission method."""
        role = Role(
            name="Test Role",
            code="TEST_ROLE",
            description="Test role"
        )
        
        permission_id = uuid4()
        original_updated_at = role.updated_at
        
        role.add_permission(permission_id)
        
        assert permission_id in role.permission_ids
        assert role.updated_at > original_updated_at
        
    def test_remove_permission(self):
        """Test remove_permission method."""
        permission_id1 = uuid4()
        permission_id2 = uuid4()
        
        role = Role(
            name="Test Role",
            code="TEST_ROLE",
            description="Test role",
            permission_ids={permission_id1, permission_id2}
        )
        
        original_updated_at = role.updated_at
        role.remove_permission(permission_id1)
        
        assert permission_id1 not in role.permission_ids
        assert permission_id2 in role.permission_ids
        assert role.updated_at > original_updated_at
        
    def test_has_permission_direct(self):
        """Test has_permission method with direct permission."""
        permission_id = uuid4()
        
        role = Role(
            name="Test Role",
            code="TEST_ROLE",
            description="Test role",
            permission_ids={permission_id}
        )
        
        assert role.has_permission(permission_id) is True
        assert role.has_permission(uuid4()) is False
        
    def test_has_permission_inherited(self):
        """Test has_permission method with inherited permission."""
        permission_id = uuid4()
        
        role = Role(
            name="Test Role",
            code="TEST_ROLE",
            description="Test role",
            inherited_permissions={permission_id}
        )
        
        assert role.has_permission(permission_id) is True
        
    def test_get_all_permissions(self):
        """Test get_all_permissions method."""
        direct_permission = uuid4()
        inherited_permission = uuid4()
        
        role = Role(
            name="Test Role",
            code="TEST_ROLE",
            description="Test role",
            permission_ids={direct_permission},
            inherited_permissions={inherited_permission}
        )
        
        all_permissions = role.get_all_permissions()
        assert all_permissions == {direct_permission, inherited_permission}
        
    def test_inherit_from_parent(self):
        """Test inherit_from_parent method."""
        role = Role(
            name="Test Role",
            code="TEST_ROLE",
            description="Test role"
        )
        
        parent_permissions = {uuid4(), uuid4()}
        original_updated_at = role.updated_at
        
        role.inherit_from_parent(parent_permissions)
        
        assert role.inherited_permissions == parent_permissions
        assert role.updated_at > original_updated_at
        
    def test_is_valid_for_tenant_system_role(self):
        """Test is_valid_for_tenant for system role."""
        role = Role(
            name="System Role",
            code="SYSTEM_ROLE",
            description="System role",
            is_system=True
        )
        
        assert role.is_valid_for_tenant(uuid4()) is True
        
    def test_is_valid_for_tenant_global_role(self):
        """Test is_valid_for_tenant for global role."""
        role = Role(
            name="Global Role",
            code="GLOBAL_ROLE",
            description="Global role",
            tenant_id=None
        )
        
        assert role.is_valid_for_tenant(uuid4()) is True
        
    def test_is_valid_for_tenant_specific_role_match(self):
        """Test is_valid_for_tenant for tenant-specific role with match."""
        tenant_id = uuid4()
        role = Role(
            name="Tenant Role",
            code="TENANT_ROLE",
            description="Tenant role",
            tenant_id=tenant_id
        )
        
        assert role.is_valid_for_tenant(tenant_id) is True
        
    def test_is_valid_for_tenant_specific_role_no_match(self):
        """Test is_valid_for_tenant for tenant-specific role without match."""
        tenant_id = uuid4()
        other_tenant_id = uuid4()
        
        role = Role(
            name="Tenant Role",
            code="TENANT_ROLE",
            description="Tenant role",
            tenant_id=tenant_id
        )
        
        assert role.is_valid_for_tenant(other_tenant_id) is False
        
    def test_total_permissions_count_property(self):
        """Test total_permissions_count property."""
        direct_permission = uuid4()
        inherited_permission = uuid4()
        
        role = Role(
            name="Test Role",
            code="TEST_ROLE",
            description="Test role",
            permission_ids={direct_permission},
            inherited_permissions={inherited_permission}
        )
        
        assert role.total_permissions_count == 2
        
    def test_to_dict(self):
        """Test to_dict method."""
        role = Role(
            name="Test Role",
            code="TEST_ROLE",
            description="Test role"
        )
        
        data = role.to_dict()
        assert isinstance(data, dict)
        assert data["name"] == "Test Role"
        assert data["code"] == "TEST_ROLE"
        

class TestRolePermission:
    """Test cases for RolePermission entity."""
    
    def test_role_permission_creation(self):
        """Test role permission association creation."""
        role_id = uuid4()
        permission_id = uuid4()
        granted_by = uuid4()
        
        role_permission = RolePermission(
            role_id=role_id,
            permission_id=permission_id,
            granted_by=granted_by
        )
        
        assert isinstance(role_permission.id, UUID)
        assert role_permission.role_id == role_id
        assert role_permission.permission_id == permission_id
        assert role_permission.granted_by == granted_by
        assert role_permission.conditions == {}
        assert role_permission.expires_at is None
        assert role_permission.is_active is True
        assert role_permission.revoked_at is None
        assert role_permission.revoked_by is None
        
    def test_role_permission_with_expiration(self):
        """Test role permission association with expiration."""
        expires_at = datetime.utcnow() + timedelta(days=30)
        
        role_permission = RolePermission(
            role_id=uuid4(),
            permission_id=uuid4(),
            granted_by=uuid4(),
            expires_at=expires_at,
            conditions={"reason": "temporary access"}
        )
        
        assert role_permission.expires_at == expires_at
        assert role_permission.conditions == {"reason": "temporary access"}
        
    def test_is_valid_true(self):
        """Test is_valid returns True for valid association."""
        role_permission = RolePermission(
            role_id=uuid4(),
            permission_id=uuid4(),
            granted_by=uuid4(),
            expires_at=datetime.utcnow() + timedelta(days=30)
        )
        
        assert role_permission.is_valid() is True
        
    def test_is_valid_false_inactive(self):
        """Test is_valid returns False for inactive association."""
        role_permission = RolePermission(
            role_id=uuid4(),
            permission_id=uuid4(),
            granted_by=uuid4(),
            is_active=False
        )
        
        assert role_permission.is_valid() is False
        
    def test_is_valid_false_revoked(self):
        """Test is_valid returns False for revoked association."""
        role_permission = RolePermission(
            role_id=uuid4(),
            permission_id=uuid4(),
            granted_by=uuid4(),
            revoked_at=datetime.utcnow()
        )
        
        assert role_permission.is_valid() is False
        
    def test_is_valid_false_expired(self):
        """Test is_valid returns False for expired association."""
        role_permission = RolePermission(
            role_id=uuid4(),
            permission_id=uuid4(),
            granted_by=uuid4(),
            expires_at=datetime.utcnow() - timedelta(days=1)
        )
        
        assert role_permission.is_valid() is False
        
    def test_revoke(self):
        """Test revoke method."""
        role_permission = RolePermission(
            role_id=uuid4(),
            permission_id=uuid4(),
            granted_by=uuid4()
        )
        
        revoked_by = uuid4()
        reason = "Security policy change"
        
        role_permission.revoke(revoked_by, reason)
        
        assert role_permission.is_active is False
        assert role_permission.revoked_at is not None
        assert role_permission.revoked_by == revoked_by
        assert role_permission.conditions["revocation_reason"] == reason
        
    def test_extend_expiration(self):
        """Test extend_expiration method."""
        original_expiration = datetime.utcnow() + timedelta(days=30)
        new_expiration = datetime.utcnow() + timedelta(days=60)
        
        role_permission = RolePermission(
            role_id=uuid4(),
            permission_id=uuid4(),
            granted_by=uuid4(),
            expires_at=original_expiration
        )
        
        role_permission.extend_expiration(new_expiration)
        
        assert role_permission.expires_at == new_expiration
        

class TestUserRole:
    """Test cases for UserRole entity."""
    
    def test_user_role_creation(self):
        """Test user role assignment creation."""
        user_id = uuid4()
        role_id = uuid4()
        tenant_id = uuid4()
        assigned_by = uuid4()
        
        user_role = UserRole(
            user_id=user_id,
            role_id=role_id,
            tenant_id=tenant_id,
            assigned_by=assigned_by
        )
        
        assert isinstance(user_role.id, UUID)
        assert user_role.user_id == user_id
        assert user_role.role_id == role_id
        assert user_role.tenant_id == tenant_id
        assert user_role.assigned_by == assigned_by
        assert user_role.conditions == {}
        assert user_role.expires_at is None
        assert user_role.scope_limitations == {}
        assert user_role.is_active is True
        assert user_role.revoked_at is None
        assert user_role.revoked_by is None
        
    def test_user_role_with_limitations(self):
        """Test user role assignment with scope limitations."""
        expires_at = datetime.utcnow() + timedelta(days=30)
        conditions = {"department": "engineering"}
        scope_limitations = {"dataset_access": ["public"], "region": "us-east"}
        
        user_role = UserRole(
            user_id=uuid4(),
            role_id=uuid4(),
            tenant_id=uuid4(),
            assigned_by=uuid4(),
            expires_at=expires_at,
            conditions=conditions,
            scope_limitations=scope_limitations
        )
        
        assert user_role.expires_at == expires_at
        assert user_role.conditions == conditions
        assert user_role.scope_limitations == scope_limitations
        
    def test_is_valid_true(self):
        """Test is_valid returns True for valid assignment."""
        user_role = UserRole(
            user_id=uuid4(),
            role_id=uuid4(),
            tenant_id=uuid4(),
            assigned_by=uuid4(),
            expires_at=datetime.utcnow() + timedelta(days=30)
        )
        
        assert user_role.is_valid() is True
        
    def test_is_valid_false_inactive(self):
        """Test is_valid returns False for inactive assignment."""
        user_role = UserRole(
            user_id=uuid4(),
            role_id=uuid4(),
            tenant_id=uuid4(),
            assigned_by=uuid4(),
            is_active=False
        )
        
        assert user_role.is_valid() is False
        
    def test_is_valid_false_revoked(self):
        """Test is_valid returns False for revoked assignment."""
        user_role = UserRole(
            user_id=uuid4(),
            role_id=uuid4(),
            tenant_id=uuid4(),
            assigned_by=uuid4(),
            revoked_at=datetime.utcnow()
        )
        
        assert user_role.is_valid() is False
        
    def test_is_valid_false_expired(self):
        """Test is_valid returns False for expired assignment."""
        user_role = UserRole(
            user_id=uuid4(),
            role_id=uuid4(),
            tenant_id=uuid4(),
            assigned_by=uuid4(),
            expires_at=datetime.utcnow() - timedelta(days=1)
        )
        
        assert user_role.is_valid() is False
        
    def test_revoke(self):
        """Test revoke method."""
        user_role = UserRole(
            user_id=uuid4(),
            role_id=uuid4(),
            tenant_id=uuid4(),
            assigned_by=uuid4()
        )
        
        revoked_by = uuid4()
        reason = "Role change"
        
        user_role.revoke(revoked_by, reason)
        
        assert user_role.is_active is False
        assert user_role.revoked_at is not None
        assert user_role.revoked_by == revoked_by
        assert user_role.conditions["revocation_reason"] == reason
        
    def test_has_scope_limitation(self):
        """Test has_scope_limitation method."""
        user_role = UserRole(
            user_id=uuid4(),
            role_id=uuid4(),
            tenant_id=uuid4(),
            assigned_by=uuid4(),
            scope_limitations={"dataset_access": ["public"], "region": "us-east"}
        )
        
        assert user_role.has_scope_limitation("dataset_access") is True
        assert user_role.has_scope_limitation("time_access") is False
        
    def test_get_scope_limitation(self):
        """Test get_scope_limitation method."""
        limitations = {"dataset_access": ["public"], "region": "us-east"}
        user_role = UserRole(
            user_id=uuid4(),
            role_id=uuid4(),
            tenant_id=uuid4(),
            assigned_by=uuid4(),
            scope_limitations=limitations
        )
        
        assert user_role.get_scope_limitation("dataset_access") == ["public"]
        assert user_role.get_scope_limitation("region") == "us-east"
        assert user_role.get_scope_limitation("nonexistent") is None
        
    def test_add_scope_limitation(self):
        """Test add_scope_limitation method."""
        user_role = UserRole(
            user_id=uuid4(),
            role_id=uuid4(),
            tenant_id=uuid4(),
            assigned_by=uuid4()
        )
        
        user_role.add_scope_limitation("dataset_access", ["public", "internal"])
        
        assert user_role.scope_limitations["dataset_access"] == ["public", "internal"]
        
    def test_remove_scope_limitation(self):
        """Test remove_scope_limitation method."""
        user_role = UserRole(
            user_id=uuid4(),
            role_id=uuid4(),
            tenant_id=uuid4(),
            assigned_by=uuid4(),
            scope_limitations={"dataset_access": ["public"], "region": "us-east"}
        )
        
        user_role.remove_scope_limitation("dataset_access")
        
        assert "dataset_access" not in user_role.scope_limitations
        assert "region" in user_role.scope_limitations
        

class TestSystemPermissions:
    """Test cases for system-defined permissions."""
    
    def test_system_permissions_exist(self):
        """Test that system permissions are defined."""
        assert len(SYSTEM_PERMISSIONS) > 0
        assert "USER.CREATE" in SYSTEM_PERMISSIONS
        assert "USER.READ" in SYSTEM_PERMISSIONS
        assert "ROLE.CREATE" in SYSTEM_PERMISSIONS
        assert "TENANT.ADMIN" in SYSTEM_PERMISSIONS
        assert "SYSTEM.ADMIN" in SYSTEM_PERMISSIONS
        
    def test_system_permission_structure(self):
        """Test system permission structure."""
        user_create = SYSTEM_PERMISSIONS["USER.CREATE"]
        
        assert isinstance(user_create, Permission)
        assert user_create.name == "Create User"
        assert user_create.code == "USER.CREATE"
        assert user_create.scope == PermissionScope.TENANT
        assert user_create.action == PermissionAction.CREATE
        assert user_create.resource_type == ResourceType.USER
        assert user_create.is_system is True
        
    def test_system_admin_permission(self):
        """Test system admin permission."""
        system_admin = SYSTEM_PERMISSIONS["SYSTEM.ADMIN"]
        
        assert system_admin.scope == PermissionScope.GLOBAL
        assert system_admin.action == PermissionAction.ADMIN
        assert system_admin.is_system is True
        

class TestSystemRoles:
    """Test cases for system-defined roles."""
    
    def test_system_roles_exist(self):
        """Test that system roles are defined."""
        assert len(SYSTEM_ROLES) > 0
        assert "SUPER_ADMIN" in SYSTEM_ROLES
        assert "TENANT_ADMIN" in SYSTEM_ROLES
        assert "USER_ADMIN" in SYSTEM_ROLES
        assert "ANALYST" in SYSTEM_ROLES
        assert "VIEWER" in SYSTEM_ROLES
        
    def test_system_role_structure(self):
        """Test system role structure."""
        super_admin = SYSTEM_ROLES["SUPER_ADMIN"]
        
        assert isinstance(super_admin, Role)
        assert super_admin.name == "Super Administrator"
        assert super_admin.code == "SUPER_ADMIN"
        assert super_admin.level == 0
        assert super_admin.is_system is True
        assert len(super_admin.permission_ids) > 0
        
    def test_viewer_role_default(self):
        """Test viewer role is marked as default."""
        viewer = SYSTEM_ROLES["VIEWER"]
        
        assert viewer.is_default is True
        assert viewer.level == 4  # Lowest privilege level
        
    def test_role_hierarchy_levels(self):
        """Test role hierarchy levels are correct."""
        assert SYSTEM_ROLES["SUPER_ADMIN"].level == 0
        assert SYSTEM_ROLES["TENANT_ADMIN"].level == 1
        assert SYSTEM_ROLES["USER_ADMIN"].level == 2
        assert SYSTEM_ROLES["ANALYST"].level == 3
        assert SYSTEM_ROLES["VIEWER"].level == 4