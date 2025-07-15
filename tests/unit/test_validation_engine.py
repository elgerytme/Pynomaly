"""
Unit tests for the data quality ValidationEngine and rule types.
"""
import operator

import pytest

from packages.data_quality.domain.services.validation_engine import (
    ValidationEngine,
    RangeRule,
    FormatRule,
    CompletenessRule,
    UniquenessRule,
    ConsistencyRule,
    ValidationResult,
)

@pytest.mark.unit
def test_range_rule():
    records = [{'x': 1}, {'x': 5}, {'x': 10}]
    rule = RangeRule('r1', field='x', min_value=2, max_value=8)
    engine = ValidationEngine([rule])
    results = engine.run(records)
    assert len(results) == 1
    res: ValidationResult = results[0]
    assert res.rule_id == 'r1'
    assert not res.passed
    assert res.failed_records == 2
    # Check that the correct records are marked failed
    assert records[0] in res.error_details
    assert records[2] in res.error_details

@pytest.mark.unit
def test_format_rule():
    records = [{'id': 'abc123'}, {'id': '123abc'}, {'id': 'abc001'}]
    rule = FormatRule('r2', field='id', pattern=r'^[a-z]+[0-9]+$')
    engine = ValidationEngine([rule])
    res: ValidationResult = engine.run(records)[0]
    assert not res.passed
    assert res.failed_records == 1
    assert {'id': '123abc'} in res.error_details

@pytest.mark.unit
def test_completeness_rule():
    records = [{'a': 'value'}, {'a': ''}, {}]
    rule = CompletenessRule('r3', field='a')
    engine = ValidationEngine([rule])
    res: ValidationResult = engine.run(records)[0]
    assert not res.passed
    assert res.failed_records == 2

@pytest.mark.unit
def test_uniqueness_rule():
    records = [{'u': 1}, {'u': 2}, {'u': 1}, {'u': None}, {'u': None}]
    rule = UniquenessRule('r4', field='u')
    engine = ValidationEngine([rule])
    res: ValidationResult = engine.run(records)[0]
    assert not res.passed
    # Expected duplicates: second occurrence of 1 and second occurrence of None
    assert res.failed_records == 2

@pytest.mark.unit
def test_consistency_rule():
    records = [{'a': 5, 'b': 3}, {'a': 2, 'b': 4}, {'a': None, 'b': 1}]
    rule = ConsistencyRule('r5', field_a='a', field_b='b', comparator=operator.gt)
    engine = ValidationEngine([rule])
    res: ValidationResult = engine.run(records)[0]
    assert not res.passed
    # Records failing: second (2 > 4 false), third (None fails)
    assert res.failed_records == 2