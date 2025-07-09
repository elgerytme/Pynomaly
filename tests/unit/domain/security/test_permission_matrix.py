import unittest
from src.pynomaly.domain.security.permission_matrix import PermissionMatrix
from src.pynomaly.domain.entities.user import UserRole

class TestPermissionMatrix(unittest.TestCase):

    def test_get_role_permissions_non_empty(self):
        # Check that role permissions are non-empty for each role
        roles = [UserRole.SUPER_ADMIN, UserRole.TENANT_ADMIN, UserRole.DATA_SCIENTIST,
                 UserRole.ANALYST, UserRole.VIEWER]
        for role in roles:
            with self.subTest(role=role):
                permissions = PermissionMatrix.get_role_permissions(role)
                self.assertTrue(len(permissions) > 0, f"Permissions should not be empty for role {role}")

    def test_can_role_grant_permission_super_admin(self):
        # Super Admin should be able to grant any permission
        role = UserRole.SUPER_ADMIN
        permissions = PermissionMatrix.get_role_permissions(role)
        for permission in permissions:
            with self.subTest(permission=permission):
                can_grant = PermissionMatrix.can_role_grant_permission(role, permission)
                self.assertTrue(can_grant, f"Super Admin should grant permission {permission}")

    def test_can_role_grant_permission_data_scientist(self):
        # Data Scientist should not be able to grant other roles' permissions
        role = UserRole.DATA_SCIENTIST
        permissions = PermissionMatrix.get_role_permissions(UserRole.SUPER_ADMIN)
        for permission in permissions:
            with self.subTest(permission=permission):
                can_grant = PermissionMatrix.can_role_grant_permission(role, permission)
                self.assertFalse(can_grant, f"Data Scientist should not grant permission {permission}")

if __name__ == '__main__':
    unittest.main()

