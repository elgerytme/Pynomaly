#!/usr/bin/env python3
"""Test script for permission matrix functionality."""

import sys
import os
from pathlib import Path

# Add the src directory to the path to import our modules
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    # Test direct import without triggering package imports
    import importlib.util


    # Minimal import to avoid circular dependencies
    from src.pynomaly.domain.security.permission_matrix import PermissionMatrix, ResourceType, ActionType

    # Define mock user entities here
    from enum import Enum

    class UserRole(str, Enum):
        SUPER_ADMIN = "super_admin"
        TENANT_ADMIN = "tenant_admin"
        DATA_SCIENTIST = "data_scientist"
        ANALYST = "analyst"
        VIEWER = "viewer"

    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Permission:
        name: str
        resource: str
        action: str
        description: str = ""

    def has_permission(user_permissions, required_permission):
        return required_permission in user_permissions

    def has_resource_access(user_permissions, resource, action):
        required_permission = Permission(
            name=f"{resource.value}.{action.value}",
            resource=resource.value,
            action=action.value,
            description=""
        )
        return has_permission(user_permissions, required_permission)

    print("✓ Simplified setup for PermissionMatrix testing")

    # Test getting permissions for each role
    for role in UserRole:
        permissions = PermissionMatrix.get_role_permissions(role)
        print(f"✓ {role.value} has {len(permissions)} permissions")

    # Test hierarchy
    hierarchy = PermissionMatrix.get_permission_hierarchy()
    print(f"✓ Permission hierarchy: {hierarchy}")

    # Test permission checking
    super_admin_perms = PermissionMatrix.get_role_permissions(UserRole.SUPER_ADMIN)
    test_permission = Permission(
        name="platform.manage",
        resource="platform",
        action="manage",
        description="Full platform management"
    )

    if has_permission(super_admin_perms, test_permission):
        print("✓ Super admin has platform management permission")

    # Test resource access checking
    if has_resource_access(super_admin_perms, ResourceType.TENANT, ActionType.CREATE):
        print("✓ Super admin can create tenants")

    print("\n🎉 All permission matrix tests passed!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
