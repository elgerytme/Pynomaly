from typing import Any, Dict, List, Callable
import re

class ValidationResult:
    def __init__(self, rule_id: str, passed: bool, failed_records: int = 0, error_details: List[Dict[str, Any]] = None):
        self.rule_id = rule_id
        self.passed = passed
        self.failed_records = failed_records
        self.error_details = error_details or []

class ValidationRule:
    """Abstract base class for validation rules."""
    def __init__(self, rule_id: str, description: str = '', severity: str = 'medium'):
        self.rule_id = rule_id
        self.description = description
        self.severity = severity

    def validate(self, record: Dict[str, Any]) -> bool:
        """Validate a single record. Returns True if it passes, False otherwise."""
        raise NotImplementedError

class RangeRule(ValidationRule):
    """Validate that a numeric field falls within a specified range."""
    def __init__(self, rule_id: str, field: str, min_value: float = None, max_value: float = None, description: str = '', severity: str = 'medium'):
        super().__init__(rule_id, description, severity)
        self.field = field
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, record: Dict[str, Any]) -> bool:
        value = record.get(self.field)
        if value is None:
            return False
        try:
            v = float(value)
        except (TypeError, ValueError):
            return False
        if self.min_value is not None and v < self.min_value:
            return False
        if self.max_value is not None and v > self.max_value:
            return False
        return True

class FormatRule(ValidationRule):
    """Validate that a string field matches a regular expression."""
    def __init__(self, rule_id: str, field: str, pattern: str, description: str = '', severity: str = 'medium'):
        super().__init__(rule_id, description, severity)
        self.field = field
        self.pattern = re.compile(pattern)

    def validate(self, record: Dict[str, Any]) -> bool:
        value = record.get(self.field)
        if value is None:
            return False
        return bool(self.pattern.match(str(value)))

class CompletenessRule(ValidationRule):
    """Validate that a field is not null or empty."""
    def __init__(self, rule_id: str, field: str, description: str = '', severity: str = 'medium'):
        super().__init__(rule_id, description, severity)
        self.field = field

    def validate(self, record: Dict[str, Any]) -> bool:
        value = record.get(self.field)
        return value is not None and value != ''

class UniquenessRule(ValidationRule):
    """Validate that field values are unique across records."""
    def __init__(self, rule_id: str, field: str, description: str = '', severity: str = 'medium'):
        super().__init__(rule_id, description, severity)
        self.field = field
        self._seen = set()

    def validate(self, record: Dict[str, Any]) -> bool:
        value = record.get(self.field)
        if value in self._seen:
            return False
        self._seen.add(value)
        return True

class ConsistencyRule(ValidationRule):
    """Validate consistency between two fields via a comparator function."""
    def __init__(self, rule_id: str, field_a: str, field_b: str, comparator: Callable[[Any, Any], bool], description: str = '', severity: str = 'medium'):
        super().__init__(rule_id, description, severity)
        self.field_a = field_a
        self.field_b = field_b
        self.comparator = comparator

    def validate(self, record: Dict[str, Any]) -> bool:
        a = record.get(self.field_a)
        b = record.get(self.field_b)
        if a is None or b is None:
            return False
        try:
            return bool(self.comparator(a, b))
        except Exception:
            return False

class ValidationEngine:
    """Engine to apply multiple validation rules to a dataset."""
    def __init__(self, rules: List[ValidationRule]):
        self.rules = rules

    def run(self, records: List[Dict[str, Any]]) -> List[ValidationResult]:
        results: List[ValidationResult] = []
        for rule in self.rules:
            failed_details: List[Dict[str, Any]] = []
            if isinstance(rule, UniquenessRule):
                rule._seen.clear()
            for record in records:
                if not rule.validate(record):
                    failed_details.append(record)
            results.append(ValidationResult(
                rule_id=rule.rule_id,
                passed=(len(failed_details) == 0),
                failed_records=len(failed_details),
                error_details=failed_details
            ))
        return results