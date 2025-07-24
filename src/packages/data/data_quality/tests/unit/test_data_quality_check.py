
import unittest
from datetime import datetime
from uuid import UUID, uuid4

from src.packages.data.data_quality.src.data_quality.domain.entities.data_quality_check import DataQualityCheck, CheckType, CheckStatus, CheckSeverity, CheckResult


class TestDataQualityCheck(unittest.TestCase):

    def test_data_quality_check_creation(self):
        rule_id = uuid4()
        check = DataQualityCheck(
            name="test_check",
            description="A test data quality check",
            check_type=CheckType.COMPLETENESS,
            rule_id=rule_id,
            dataset_name="test_dataset"
        )
        self.assertIsInstance(check.id, UUID)
        self.assertEqual(check.name, "test_check")
        self.assertEqual(check.check_type, CheckType.COMPLETENESS)
        self.assertEqual(check.rule_id, rule_id)
        self.assertEqual(check.status, CheckStatus.PENDING)

    def test_check_result_creation(self):
        result = CheckResult(
            check_id=uuid4(),
            dataset_name="test_dataset",
            passed=True,
            score=0.99,
            total_records=100,
            passed_records=99,
            failed_records=1
        )
        self.assertIsInstance(result.id, UUID)
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 0.99)

    def test_check_result_pass_fail_rate(self):
        result = CheckResult(
            check_id=uuid4(),
            dataset_name="test_dataset",
            passed=True,
            score=0.90,
            total_records=100,
            passed_records=90,
            failed_records=10
        )
        self.assertEqual(result.pass_rate, 90.0)
        self.assertEqual(result.fail_rate, 10.0)

    def test_check_activation_deactivation(self):
        rule_id = uuid4()
        check = DataQualityCheck(
            name="test_check",
            description="A test data quality check",
            check_type=CheckType.COMPLETENESS,
            rule_id=rule_id,
            dataset_name="test_dataset"
        )
        check.activate()
        self.assertTrue(check.is_active)
        self.assertEqual(check.status, CheckStatus.PENDING)

        check.deactivate()
        self.assertFalse(check.is_active)
        self.assertEqual(check.status, CheckStatus.CANCELLED)

    def test_check_execution_simulation(self):
        rule_id = uuid4()
        check = DataQualityCheck(
            name="test_check",
            description="A test data quality check",
            check_type=CheckType.COMPLETENESS,
            rule_id=rule_id,
            dataset_name="test_dataset"
        )
        check.activate()
        result = check.execute()

        self.assertEqual(check.status, CheckStatus.COMPLETED)
        self.assertIsNotNone(check.last_executed_at)
        self.assertEqual(check.execution_count, 1)
        self.assertIsNotNone(check.last_result)
        self.assertEqual(result.check_id, check.id)

    def test_check_execution_failure_tracking(self):
        rule_id = uuid4()
        check = DataQualityCheck(
            name="test_check",
            description="A test data quality check",
            check_type=CheckType.COMPLETENESS,
            rule_id=rule_id,
            dataset_name="test_dataset",
            threshold=1.0 # Make it fail easily
        )
        check.activate()
        result = check.execute()
        self.assertFalse(result.passed)
        self.assertEqual(check.consecutive_failures, 1)
        self.assertEqual(check.success_rate, 0.0)

        result = check.execute() # Run again to test success rate update
        self.assertEqual(check.consecutive_failures, 2)
        self.assertLess(check.success_rate, 0.0)

    def test_to_dict(self):
        rule_id = uuid4()
        check = DataQualityCheck(
            name="test_check",
            description="A test data quality check",
            check_type=CheckType.COMPLETENESS,
            rule_id=rule_id,
            dataset_name="test_dataset"
        )
        check_dict = check.to_dict()
        self.assertIsInstance(check_dict, dict)
        self.assertIn("id", check_dict)
        self.assertEqual(check_dict["name"], "test_check")
        self.assertEqual(check_dict["rule_id"], str(rule_id))
