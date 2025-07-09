import unittest
from uuid import uuid4
from pynomaly.domain.entities.security import (
    SecurityPolicy,
    User,
    AuditEvent,
    AccessRequest,
    PermissionType,
    ActionType,
)

class SecurityEntityTests(unittest.TestCase):
    def test_audit_event_creation(self):
        event_id = uuid4()
        user_id = uuid4()
        event = AuditEvent(
            event_id=event_id,
            user_id=user_id,
            username="test_user",
            action=ActionType.CREATE_MODEL,
            resource_type="model",
            resource_id="test_resource",
            resource_name="Test Resource",
            ip_address="192.168.1.1",
            success=True
        )
        self.assertEqual(event.action, ActionType.CREATE_MODEL)
        self.assertEqual(event.resource_type, "model")
        self.assertEqual(event.username, "test_user")
        self.assertTrue(event.success)

    def test_security_policy_creation(self):
        policy = SecurityPolicy(
            policy_name="Test Policy",
            password_min_length=10,
            require_2fa=True
        )
        self.assertEqual(policy.policy_name, "Test Policy")
        self.assertEqual(policy.password_min_length, 10)
        self.assertTrue(policy.require_2fa)

    def test_access_request(self):
        request_id = uuid4()
        requester_id = uuid4()
        request = AccessRequest(
            request_id=request_id,
            requester_id=requester_id,
            requester_username="test_user",
            requested_permission=PermissionType.READ_MODELS,
            resource_type="model",
            justification="Need access for testing"
        )
        self.assertEqual(request.requester_username, "test_user")
        self.assertEqual(request.requested_permission, PermissionType.READ_MODELS)
        self.assertEqual(request.resource_type, "model")
        self.assertEqual(request.approval_status, "pending")

    def test_user_creation(self):
        user_id = uuid4()
        user = User(
            user_id=user_id,
            username="test_user",
            email="test@example.com",
            password_hash="hashed_password",
            salt="salt123",
            roles={"admin", "user"}
        )
        self.assertEqual(user.username, "test_user")
        self.assertIn("admin", user.roles)
        self.assertIn("user", user.roles)
        self.assertTrue(user.is_active)

    def test_user_permissions(self):
        user_id = uuid4()
        user = User(
            user_id=user_id,
            username="test_user",
            email="test@example.com",
            password_hash="hashed_password",
            salt="salt123",
            roles={"admin"}
        )
        # Admin should have all permissions
        self.assertTrue(user.has_permission(PermissionType.READ_MODELS))
        self.assertTrue(user.has_permission(PermissionType.WRITE_MODELS))
        self.assertTrue(user.has_permission(PermissionType.DELETE_MODELS))

if __name__ == '__main__':
    unittest.main()

