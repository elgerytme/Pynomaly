import unittest
from datetime import datetime, timedelta
from uuid import uuid4
from src.pynomaly.domain.entities.security import AccessRequest, PermissionType


class TestAccessRequest(unittest.TestCase):

    def setUp(self):
        self.requester_id = uuid4()
        self.requester_username = "test_user"
        self.permission = PermissionType.READ_MODELS
        self.resource_type = "dataset"
        self.justification = "Required for testing"

    def test_access_request_creation(self):
        access_request = AccessRequest(
            request_id=uuid4(),
            requester_id=self.requester_id,
            requester_username=self.requester_username,
            requested_permission=self.permission,
            resource_type=self.resource_type,
            resource_id=None,
            justification=self.justification,
        )

        self.assertEqual(access_request.requester_id, self.requester_id)
        self.assertEqual(access_request.requested_permission, self.permission)
        self.assertEqual(access_request.resource_type, self.resource_type)
        self.assertEqual(access_request.justification, self.justification)
        self.assertEqual(access_request.approval_status, "pending")

    def test_access_request_approval(self):
        access_request = AccessRequest(
            request_id=uuid4(),
            requester_id=self.requester_id,
            requester_username=self.requester_username,
            requested_permission=self.permission,
            resource_type=self.resource_type,
            resource_id=None,
            justification=self.justification,
            requested_start_time=datetime.utcnow(),
            requested_end_time=datetime.utcnow() + timedelta(days=1)
        )

        access_request.approval_status = "approved"
        access_request.granted_start_time = datetime.utcnow()
        access_request.granted_end_time = datetime.utcnow() + timedelta(hours=2)

        self.assertTrue(access_request.is_active())
        self.assertFalse(access_request.is_expired())

    def test_access_request_expiry(self):
        access_request = AccessRequest(
            request_id=uuid4(),
            requester_id=self.requester_id,
            requester_username=self.requester_username,
            requested_permission=self.permission,
            resource_type=self.resource_type,
            resource_id=None,
            justification=self.justification,
            requested_start_time=datetime.utcnow(),
            requested_end_time=datetime.utcnow() - timedelta(days=1)
        )

        access_request.approval_status = "approved"
        access_request.granted_start_time = datetime.utcnow() - timedelta(days=2)
        access_request.granted_end_time = datetime.utcnow() - timedelta(days=1)

        self.assertFalse(access_request.is_active())
        self.assertTrue(access_request.is_expired())


if __name__ == '__main__':
    unittest.main()

