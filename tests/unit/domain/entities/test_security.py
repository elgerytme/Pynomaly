import unittest
from pynomaly.domain.entities import AuditEvent, SecurityPolicy, AccessRequest, User
from pynomaly.domain.entities.security import PermissionType, ActionType

class SecurityEntityTests(unittest.TestCase):
    def test_audit_event_creation(self):
        event = AuditEvent(action=ActionType.CREATE, resource="test_resource", actor="test_actor")
        self.assertEqual(event.action, ActionType.CREATE)
        self.assertEqual(event.resource, "test_resource")
        self.assertEqual(event.actor, "test_actor")

    def test_security_policy_creation(self):
        policy = SecurityPolicy(name="Test Policy", permissions=[PermissionType.READ, PermissionType.WRITE])
        self.assertEqual(policy.name, "Test Policy")
        self.assertIn(PermissionType.READ, policy.permissions)
        self.assertIn(PermissionType.WRITE, policy.permissions)

    def test_access_request(self):
        request = AccessRequest(user="test_user", resource="test_resource", requested_action=ActionType.UPDATE)
        self.assertEqual(request.user, "test_user")
        self.assertEqual(request.resource, "test_resource")
        self.assertEqual(request.requested_action, ActionType.UPDATE)

    def test_user_creation(self):
        user = User(username="test_user", roles=["admin", "user"])
        self.assertEqual(user.username, "test_user")
        self.assertIn("admin", user.roles)
        self.assertIn("user", user.roles)

if __name__ == '__main__':
    unittest.main()

