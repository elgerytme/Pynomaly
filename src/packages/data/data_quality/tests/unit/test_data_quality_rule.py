
import unittest
from datetime import datetime
from uuid import UUID, uuid4

from src.data_quality.domain.entities.data_quality_rule import DataQualityRule, RuleType, RuleSeverity, RuleCondition, RuleOperator


class TestDataQualityRule(unittest.TestCase):

    def test_data_quality_rule_creation(self):
        rule = DataQualityRule(
            name="test_rule",
            description="A test data quality rule",
            rule_type=RuleType.NOT_NULL,
            severity=RuleSeverity.ERROR,
            dataset_name="test_dataset"
        )
        self.assertIsInstance(rule.id, UUID)
        self.assertEqual(rule.name, "test_rule")
        self.assertEqual(rule.rule_type, RuleType.NOT_NULL)
        self.assertEqual(rule.severity, RuleSeverity.ERROR)

    def test_rule_condition_creation(self):
        condition = RuleCondition(
            column_name="col1",
            operator=RuleOperator.EQUALS,
            value="expected_value"
        )
        self.assertIsInstance(condition.id, UUID)
        self.assertEqual(condition.column_name, "col1")
        self.assertEqual(condition.operator, RuleOperator.EQUALS)
        self.assertEqual(condition.value, "expected_value")

    def test_add_condition_to_rule(self):
        rule = DataQualityRule(
            name="test_rule",
            description="A test data quality rule",
            rule_type=RuleType.NOT_NULL,
            severity=RuleSeverity.ERROR,
            dataset_name="test_dataset"
        )
        condition = RuleCondition(
            column_name="col1",
            operator=RuleOperator.EQUALS,
            value="expected_value"
        )
        rule.add_condition(condition)
        self.assertEqual(len(rule.conditions), 1)

    def test_evaluate_record_single_condition(self):
        rule = DataQualityRule(
            name="test_rule",
            description="A test data quality rule",
            rule_type=RuleType.NOT_NULL,
            severity=RuleSeverity.ERROR,
            dataset_name="test_dataset"
        )
        condition = RuleCondition(
            column_name="col1",
            operator=RuleOperator.EQUALS,
            value="expected_value"
        )
        rule.add_condition(condition)

        self.assertTrue(rule.evaluate_record({"col1": "expected_value"}))
        self.assertFalse(rule.evaluate_record({"col1": "other_value"}))

    def test_evaluate_record_multiple_conditions_and(self):
        rule = DataQualityRule(
            name="test_rule",
            description="A test data quality rule",
            rule_type=RuleType.NOT_NULL,
            severity=RuleSeverity.ERROR,
            dataset_name="test_dataset",
            logical_operator="AND"
        )
        condition1 = RuleCondition(
            column_name="col1",
            operator=RuleOperator.EQUALS,
            value="value1"
        )
        condition2 = RuleCondition(
            column_name="col2",
            operator=RuleOperator.NOT_NULL
        )
        rule.add_condition(condition1)
        rule.add_condition(condition2)

        self.assertTrue(rule.evaluate_record({"col1": "value1", "col2": "value2"}))
        self.assertFalse(rule.evaluate_record({"col1": "value1", "col2": None}))
        self.assertFalse(rule.evaluate_record({"col1": "other", "col2": "value2"}))

    def test_evaluate_record_multiple_conditions_or(self):
        rule = DataQualityRule(
            name="test_rule",
            description="A test data quality rule",
            rule_type=RuleType.NOT_NULL,
            severity=RuleSeverity.ERROR,
            dataset_name="test_dataset",
            logical_operator="OR"
        )
        condition1 = RuleCondition(
            column_name="col1",
            operator=RuleOperator.EQUALS,
            value="value1"
        )
        condition2 = RuleCondition(
            column_name="col2",
            operator=RuleOperator.NOT_NULL
        )
        rule.add_condition(condition1)
        rule.add_condition(condition2)

        self.assertTrue(rule.evaluate_record({"col1": "value1", "col2": None}))
        self.assertTrue(rule.evaluate_record({"col1": "other", "col2": "value2"}))
        self.assertFalse(rule.evaluate_record({"col1": "other", "col2": None}))

    def test_record_evaluation(self):
        rule = DataQualityRule(
            name="test_rule",
            description="A test data quality rule",
            rule_type=RuleType.NOT_NULL,
            severity=RuleSeverity.ERROR,
            dataset_name="test_dataset"
        )
        rule.record_evaluation(True)
        self.assertEqual(rule.evaluation_count, 1)
        self.assertEqual(rule.violation_count, 1)
        self.assertIsNotNone(rule.last_evaluated_at)
        self.assertIsNotNone(rule.last_violation_at)

        rule.record_evaluation(False)
        self.assertEqual(rule.evaluation_count, 2)
        self.assertEqual(rule.violation_count, 1)

    def test_to_dict(self):
        rule = DataQualityRule(
            name="test_rule",
            description="A test data quality rule",
            rule_type=RuleType.NOT_NULL,
            severity=RuleSeverity.ERROR,
            dataset_name="test_dataset"
        )
        rule_dict = rule.to_dict()
        self.assertIsInstance(rule_dict, dict)
        self.assertIn("id", rule_dict)
        self.assertEqual(rule_dict["name"], "test_rule")
