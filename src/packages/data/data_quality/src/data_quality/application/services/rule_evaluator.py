
from typing import Any, Dict

from ...domain.entities.data_quality_rule import DataQualityRule, RuleCondition, RuleOperator


class RuleEvaluator:
    """Evaluates data quality rules against data records."""

    def evaluate_record(self, rule: DataQualityRule, record: Dict[str, Any]) -> bool:
        """Evaluates a single record against the rule's conditions.

        Args:
            rule: The DataQualityRule to evaluate.
            record: A dictionary representing the data record.

        Returns:
            True if the record passes the rule, False otherwise.
        """
        if not rule.conditions:
            return True  # No conditions means the rule always passes

        results = []
        for condition in rule.conditions:
            column_value = record.get(condition.column_name)
            results.append(self._evaluate_condition(condition, column_value))

        if rule.logical_operator == "AND":
            return all(results)
        else:  # OR
            return any(results)

    def _evaluate_condition(self, condition: RuleCondition, value: Any) -> bool:
        """Evaluates a single condition against a value."""
        if condition.operator == RuleOperator.IS_NULL:
            return value is None
        elif condition.operator == RuleOperator.IS_NOT_NULL:
            return value is not None
        elif condition.operator == RuleOperator.EQUALS:
            return value == condition.value
        elif condition.operator == RuleOperator.NOT_EQUALS:
            return value != condition.value
        # Add more operators as needed
        return False
