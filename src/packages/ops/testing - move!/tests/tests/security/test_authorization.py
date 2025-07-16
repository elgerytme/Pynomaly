"""
Security Testing Suite - Authorization
Comprehensive security tests for authorization and access control.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from monorepo.domain.exceptions import AuthorizationError, SecurityError
from monorepo.infrastructure.auth.access_control import AccessControlManager
from monorepo.infrastructure.auth.rbac import Permission, Role, RoleBasedAccessControl


class TestRoleBasedAccessControl:
    """Test suite for Role-Based Access Control (RBAC)."""

    @pytest.fixture
    def rbac_system(self):
        """Create RBAC system with test roles and permissions."""
        rbac = RoleBasedAccessControl()

        # Define permissions
        read_datasets = Permission("read:datasets", "Read dataset information")
        write_datasets = Permission("write:datasets", "Create and modify datasets")
        delete_datasets = Permission("delete:datasets", "Delete datasets")
        admin_users = Permission("admin:users", "Manage users")
        admin_system = Permission("admin:system", "System administration")

        # Define roles
        viewer_role = Role("viewer", "Can view datasets")
        viewer_role.add_permission(read_datasets)

        user_role = Role("user", "Standard user")
        user_role.add_permission(read_datasets)
        user_role.add_permission(write_datasets)

        admin_role = Role("admin", "Administrator")
        admin_role.add_permission(read_datasets)
        admin_role.add_permission(write_datasets)
        admin_role.add_permission(delete_datasets)
        admin_role.add_permission(admin_users)
        admin_role.add_permission(admin_system)

        # Add roles to RBAC
        rbac.add_role(viewer_role)
        rbac.add_role(user_role)
        rbac.add_role(admin_role)

        return rbac

    @pytest.fixture
    def test_users(self):
        """Create test users with different roles."""
        return {
            "viewer": {
                "id": "viewer123",
                "email": "viewer@example.com",
                "role": "viewer",
                "permissions": ["read:datasets"],
            },
            "user": {
                "id": "user123",
                "email": "user@example.com",
                "role": "user",
                "permissions": ["read:datasets", "write:datasets"],
            },
            "admin": {
                "id": "admin123",
                "email": "admin@example.com",
                "role": "admin",
                "permissions": [
                    "read:datasets",
                    "write:datasets",
                    "delete:datasets",
                    "admin:users",
                    "admin:system",
                ],
            },
        }

    # Permission Tests

    def test_permission_creation(self):
        """Test permission object creation."""
        perm = Permission("read:data", "Read data permission")

        assert perm.name == "read:data"
        assert perm.description == "Read data permission"
        assert str(perm) == "read:data"

    def test_permission_equality(self):
        """Test permission equality comparison."""
        perm1 = Permission("read:data", "Description 1")
        perm2 = Permission("read:data", "Description 2")
        perm3 = Permission("write:data", "Description 3")

        assert perm1 == perm2  # Same name
        assert perm1 != perm3  # Different name

    # Role Tests

    def test_role_creation(self):
        """Test role object creation."""
        role = Role("user", "Standard user role")

        assert role.name == "user"
        assert role.description == "Standard user role"
        assert len(role.permissions) == 0

    def test_role_add_permission(self):
        """Test adding permissions to role."""
        role = Role("user", "User role")
        perm = Permission("read:data", "Read permission")

        role.add_permission(perm)

        assert perm in role.permissions
        assert role.has_permission("read:data")

    def test_role_remove_permission(self):
        """Test removing permissions from role."""
        role = Role("user", "User role")
        perm = Permission("read:data", "Read permission")

        role.add_permission(perm)
        assert role.has_permission("read:data")

        role.remove_permission(perm)
        assert not role.has_permission("read:data")

    def test_role_permission_inheritance(self):
        """Test role inheritance of permissions."""
        parent_role = Role("user", "User role")
        child_role = Role("premium_user", "Premium user role")

        read_perm = Permission("read:data", "Read permission")
        write_perm = Permission("write:data", "Write permission")

        parent_role.add_permission(read_perm)
        child_role.inherit_from(parent_role)
        child_role.add_permission(write_perm)

        assert child_role.has_permission("read:data")  # Inherited
        assert child_role.has_permission("write:data")  # Own permission

    # RBAC System Tests

    def test_rbac_check_permission_valid(self, rbac_system, test_users):
        """Test valid permission check."""
        user = test_users["user"]

        # User should have read and write permissions
        assert rbac_system.check_permission(user, "read:datasets")
        assert rbac_system.check_permission(user, "write:datasets")

    def test_rbac_check_permission_invalid(self, rbac_system, test_users):
        """Test invalid permission check."""
        user = test_users["viewer"]

        # Viewer should not have write or delete permissions
        assert not rbac_system.check_permission(user, "write:datasets")
        assert not rbac_system.check_permission(user, "delete:datasets")

    def test_rbac_admin_permissions(self, rbac_system, test_users):
        """Test admin has all permissions."""
        admin = test_users["admin"]

        all_permissions = [
            "read:datasets",
            "write:datasets",
            "delete:datasets",
            "admin:users",
            "admin:system",
        ]

        for permission in all_permissions:
            assert rbac_system.check_permission(admin, permission)

    def test_rbac_role_hierarchy(self, rbac_system):
        """Test role hierarchy and inheritance."""
        # Create role hierarchy: admin > manager > user > viewer
        rbac_system.set_role_hierarchy(
            {
                "admin": ["manager", "user", "viewer"],
                "manager": ["user", "viewer"],
                "user": ["viewer"],
            }
        )

        admin_user = {"role": "admin"}

        # Admin should inherit all lower role permissions
        assert rbac_system.check_permission(
            admin_user, "read:datasets"
        )  # viewer permission

    # Access Control Manager Tests

    def test_access_control_manager_initialization(self):
        """Test access control manager initialization."""
        acm = AccessControlManager()

        assert acm is not None
        assert hasattr(acm, "check_access")
        assert hasattr(acm, "grant_access")
        assert hasattr(acm, "revoke_access")

    def test_resource_based_access_control(self):
        """Test resource-based access control."""
        acm = AccessControlManager()

        # Define resource with access rules
        dataset_resource = {
            "id": "dataset123",
            "type": "dataset",
            "owner": "user123",
            "access_level": "private",
        }

        # Owner should have full access
        owner_user = {"id": "user123", "role": "user"}
        assert acm.check_resource_access(owner_user, dataset_resource, "read")
        assert acm.check_resource_access(owner_user, dataset_resource, "write")
        assert acm.check_resource_access(owner_user, dataset_resource, "delete")

        # Other user should not have access to private resource
        other_user = {"id": "user456", "role": "user"}
        assert not acm.check_resource_access(other_user, dataset_resource, "read")

    def test_shared_resource_access(self):
        """Test access to shared resources."""
        acm = AccessControlManager()

        # Shared dataset
        shared_dataset = {
            "id": "dataset456",
            "type": "dataset",
            "owner": "user123",
            "access_level": "shared",
            "shared_with": ["user456", "user789"],
            "permissions": {"user456": ["read"], "user789": ["read", "write"]},
        }

        # Shared user with read permission
        read_user = {"id": "user456", "role": "user"}
        assert acm.check_resource_access(read_user, shared_dataset, "read")
        assert not acm.check_resource_access(read_user, shared_dataset, "write")

        # Shared user with write permission
        write_user = {"id": "user789", "role": "user"}
        assert acm.check_resource_access(write_user, shared_dataset, "read")
        assert acm.check_resource_access(write_user, shared_dataset, "write")

    # Time-Based Access Control Tests

    def test_time_based_access_control(self):
        """Test time-based access restrictions."""
        acm = AccessControlManager()

        # User with time-restricted access (business hours only)
        time_restricted_user = {
            "id": "user123",
            "role": "user",
            "access_schedule": {
                "start_time": "09:00",
                "end_time": "17:00",
                "timezone": "UTC",
                "days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
            },
        }

        # Mock current time to business hours
        with patch("datetime.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 15, 14, 0)  # Monday 2 PM
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            assert acm.check_time_based_access(time_restricted_user)

        # Mock current time to after hours
        with patch("datetime.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 15, 20, 0)  # Monday 8 PM
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            assert not acm.check_time_based_access(time_restricted_user)

    def test_temporary_access_grants(self):
        """Test temporary access grants with expiration."""
        acm = AccessControlManager()

        user = {"id": "user123", "role": "user"}
        resource = {"id": "dataset123", "type": "dataset"}

        # Grant temporary access for 1 hour
        expiry = datetime.utcnow() + timedelta(hours=1)
        acm.grant_temporary_access(user["id"], resource["id"], "write", expiry)

        # Should have access now
        assert acm.check_temporary_access(user["id"], resource["id"], "write")

        # Mock expired time
        with patch("datetime.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value = datetime.utcnow() + timedelta(hours=2)

            assert not acm.check_temporary_access(user["id"], resource["id"], "write")

    # Attribute-Based Access Control Tests

    def test_attribute_based_access_control(self):
        """Test attribute-based access control (ABAC)."""
        acm = AccessControlManager()

        # User attributes
        user = {
            "id": "user123",
            "role": "analyst",
            "department": "security",
            "clearance_level": "confidential",
            "project_assignments": ["project_alpha", "project_beta"],
        }

        # Resource attributes
        resource = {
            "id": "dataset123",
            "classification": "confidential",
            "department": "security",
            "project": "project_alpha",
        }

        # Define access policy
        policy = {
            "rules": [
                {
                    "condition": "user.department == resource.department",
                    "effect": "allow",
                },
                {
                    "condition": "user.clearance_level >= resource.classification",
                    "effect": "allow",
                },
                {
                    "condition": "resource.project in user.project_assignments",
                    "effect": "allow",
                },
            ],
            "default": "deny",
        }

        # Should have access based on multiple matching conditions
        assert acm.evaluate_abac_policy(user, resource, "read", policy)

    def test_dynamic_permission_evaluation(self):
        """Test dynamic permission evaluation based on context."""
        acm = AccessControlManager()

        user = {"id": "user123", "role": "user"}
        context = {
            "ip_address": "192.168.1.100",
            "time": datetime.utcnow(),
            "location": "office",
            "device_trusted": True,
            "mfa_verified": True,
        }

        # High-security action requires additional context validation
        security_policy = {
            "action": "delete:sensitive_data",
            "requirements": {
                "location": "office",
                "device_trusted": True,
                "mfa_verified": True,
                "ip_whitelist": ["192.168.1.0/24"],
            },
        }

        assert acm.evaluate_context_policy(user, context, security_policy)

    # Multi-Tenancy Access Control Tests

    def test_multi_tenant_access_control(self):
        """Test access control in multi-tenant environment."""
        acm = AccessControlManager()

        # Users from different tenants
        tenant_a_user = {"id": "user123", "role": "user", "tenant_id": "tenant_a"}

        tenant_b_user = {"id": "user456", "role": "user", "tenant_id": "tenant_b"}

        # Resource belonging to tenant A
        tenant_a_resource = {
            "id": "dataset123",
            "type": "dataset",
            "tenant_id": "tenant_a",
        }

        # Tenant A user should have access to their tenant's resources
        assert acm.check_tenant_access(tenant_a_user, tenant_a_resource)

        # Tenant B user should NOT have access to tenant A's resources
        assert not acm.check_tenant_access(tenant_b_user, tenant_a_resource)

    # Security Policy Tests

    def test_security_policy_enforcement(self):
        """Test security policy enforcement."""
        acm = AccessControlManager()

        # Define security policy
        security_policy = {
            "password_policy": {
                "min_length": 12,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_numbers": True,
                "require_symbols": True,
            },
            "session_policy": {
                "max_idle_time": 1800,  # 30 minutes
                "max_session_time": 28800,  # 8 hours
                "require_mfa": True,
            },
            "access_policy": {
                "max_failed_attempts": 5,
                "lockout_duration": 900,  # 15 minutes
                "ip_restrictions": True,
            },
        }

        # Test password policy
        weak_password = "password123"
        strong_password = "MyStr0ng!P@ssw0rd"

        assert not acm.validate_password_policy(
            weak_password, security_policy["password_policy"]
        )
        assert acm.validate_password_policy(
            strong_password, security_policy["password_policy"]
        )

    # Error Handling Tests

    def test_authorization_error_handling(self, rbac_system, test_users):
        """Test proper error handling for authorization failures."""
        user = test_users["viewer"]

        # Should raise AuthorizationError for insufficient permissions
        with pytest.raises(AuthorizationError):
            rbac_system.require_permission(user, "delete:datasets")

    def test_invalid_role_handling(self, rbac_system):
        """Test handling of invalid roles."""
        invalid_user = {"id": "user123", "role": "nonexistent_role"}

        with pytest.raises(SecurityError):
            rbac_system.check_permission(invalid_user, "read:datasets")

    def test_malformed_permission_handling(self, rbac_system, test_users):
        """Test handling of malformed permission strings."""
        user = test_users["user"]

        invalid_permissions = [
            "",
            None,
            "invalid-format",
            "too:many:colons:here",
            123,  # Non-string
            {"invalid": "object"},
        ]

        for invalid_perm in invalid_permissions:
            with pytest.raises((ValueError, TypeError)):
                rbac_system.check_permission(user, invalid_perm)

    # Integration Tests

    def test_complete_authorization_workflow(self, rbac_system, test_users):
        """Test complete authorization workflow."""
        # 1. User attempts to access resource
        user = test_users["user"]

        # 2. Check basic permissions
        assert rbac_system.check_permission(user, "read:datasets")

        # 3. Check resource-specific access
        acm = AccessControlManager()
        resource = {"id": "dataset123", "owner": user["id"], "type": "dataset"}

        assert acm.check_resource_access(user, resource, "read")

        # 4. Log access attempt
        access_log = acm.log_access_attempt(user, resource, "read", True)

        assert access_log["user_id"] == user["id"]
        assert access_log["resource_id"] == resource["id"]
        assert access_log["action"] == "read"
        assert access_log["granted"] is True

    def test_privilege_escalation_prevention(self, rbac_system, test_users):
        """Test prevention of privilege escalation attacks."""
        user = test_users["user"]

        # User should not be able to escalate to admin permissions
        with pytest.raises(AuthorizationError):
            rbac_system.require_permission(user, "admin:system")

        # User should not be able to modify their own role
        with pytest.raises(SecurityError):
            rbac_system.assign_role(user["id"], "admin", user)

    def test_cross_tenant_access_prevention(self):
        """Test prevention of cross-tenant data access."""
        acm = AccessControlManager()

        user = {"id": "user123", "tenant_id": "tenant_a"}
        other_tenant_resource = {"id": "dataset456", "tenant_id": "tenant_b"}

        # Should prevent cross-tenant access
        with pytest.raises(AuthorizationError):
            acm.require_tenant_access(user, other_tenant_resource)
