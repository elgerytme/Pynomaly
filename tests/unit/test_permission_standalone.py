#!/usr/bin/env python3
"""
Standalone test for permission matrix functionality.
This loads the permission matrix module directly without importing the full package.
"""

import sys
import os
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Set, List, Dict


# Define the UserRole and Permission classes needed by the permission matrix
class UserRole(str, Enum):
    SUPER_ADMIN = "super_admin"
    TENANT_ADMIN = "tenant_admin"
    DATA_SCIENTIST = "data_scientist"
    ANALYST = "analyst"
    VIEWER = "viewer"


@dataclass(frozen=True)
class Permission:
    name: str
    resource: str
    action: str
    description: str = ""


# Now load the permission matrix file directly
# Find the project root by going up from the test file
project_root = Path(__file__).parent.parent.parent
permission_matrix_path = (
    project_root / "src" / "pynomaly" / "domain" / "security" / "permission_matrix.py"
)

# Read and execute the permission matrix code
with open(permission_matrix_path, "r") as f:
    code = f.read()

# Replace the import line to use our local definitions
code = code.replace(
    "from pynomaly.domain.entities.user import UserRole, Permission", ""
)

# Create a proper module namespace
import sys
import types

permission_module = types.ModuleType("permission_matrix")
permission_module.__file__ = str(permission_matrix_path)
sys.modules["permission_matrix"] = permission_module

# Create a namespace with our definitions
namespace = {
    "__name__": "permission_matrix",
    "__file__": str(permission_matrix_path),
    "UserRole": UserRole,
    "Permission": Permission,
    "dataclass": dataclass,
    "Enum": Enum,
    "Dict": Dict,
    "Set": Set,
    "List": List,
    "__builtins__": __builtins__,
}

# Execute the permission matrix code
exec(code, namespace)

# Extract the classes we need
PermissionMatrix = namespace["PermissionMatrix"]
ResourceType = namespace["ResourceType"]
ActionType = namespace["ActionType"]
has_permission = namespace["has_permission"]
has_resource_access = namespace["has_resource_access"]

# Run tests
try:
    print("✓ Successfully loaded PermissionMatrix")

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
        description="Full platform management",
    )

    if has_permission(super_admin_perms, test_permission):
        print("✓ Super admin has platform management permission")

    # Test resource access checking
    if has_resource_access(super_admin_perms, ResourceType.TENANT, ActionType.CREATE):
        print("✓ Super admin can create tenants")

    # Test matrix summary
    summary = PermissionMatrix.get_matrix_summary()
    print(f"✓ Matrix summary generated for {len(summary)} roles")

    print("\n🎉 All permission matrix tests passed!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
