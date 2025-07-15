"""Tests for Quality Rule entity."""

import pytest
from datetime import datetime
from uuid import uuid4

from ..domain.entities.quality_rule import (
    QualityRule, RuleId, DatasetId, UserId, RuleType, RuleStatus,
    QualityCategory, Severity, ValidationLogic, RuleMetadata,
    ValidationResult, QualityThreshold
)


class TestQualityRule:
    """Test cases for Quality Rule entity."""
    
    def test_create_quality_rule(self):
        """Test creating a basic quality rule."""
        validation_logic = ValidationLogic(
            logic_type="completeness",
            expression="NOT NULL",
            error_message_template="Field is required"
        )
        
        metadata = RuleMetadata(
            description="Test completeness rule",
            business_justification="Ensure required fields are populated"
        )
        
        rule = QualityRule(
            rule_id=RuleId(),
            rule_name="test_completeness_rule",
            rule_type=RuleType.COMPLETENESS,
            category=QualityCategory.DATA_INTEGRITY,
            severity=Severity.HIGH,
            validation_logic=validation_logic,
            metadata=metadata,
            created_by=UserId()
        )
        
        assert rule.rule_name == "test_completeness_rule"
        assert rule.rule_type == RuleType.COMPLETENESS
        assert rule.status == RuleStatus.DRAFT
        assert rule.is_active() is False
    
    def test_activate_rule(self):
        """Test activating a quality rule."""
        validation_logic = ValidationLogic(
            logic_type="completeness",
            expression="NOT NULL",
            error_message_template="Field is required"
        )
        
        metadata = RuleMetadata(
            description="Test rule",
            business_justification="Test justification"
        )
        
        rule = QualityRule(
            rule_id=RuleId(),
            rule_name="test_rule",
            rule_type=RuleType.COMPLETENESS,
            category=QualityCategory.DATA_INTEGRITY,
            severity=Severity.MEDIUM,
            validation_logic=validation_logic,
            metadata=metadata,
            created_by=UserId()
        )
        
        approved_by = UserId()
        rule.activate(approved_by)
        
        assert rule.status == RuleStatus.ACTIVE
        assert rule.approved_by == approved_by
        assert rule.approved_at is not None
        assert rule.is_active() is True
    
    def test_deactivate_rule(self):
        """Test deactivating a quality rule."""
        validation_logic = ValidationLogic(
            logic_type="completeness",
            expression="NOT NULL",
            error_message_template="Field is required"
        )
        
        metadata = RuleMetadata(
            description="Test rule",
            business_justification="Test justification"
        )
        
        rule = QualityRule(
            rule_id=RuleId(),
            rule_name="test_rule",
            rule_type=RuleType.COMPLETENESS,
            category=QualityCategory.DATA_INTEGRITY,
            severity=Severity.MEDIUM,
            validation_logic=validation_logic,
            metadata=metadata,
            created_by=UserId()
        )
        
        # Activate first
        rule.activate(UserId())
        assert rule.is_active() is True
        
        # Then deactivate
        rule.deactivate()
        assert rule.status == RuleStatus.INACTIVE
        assert rule.is_active() is False
    
    def test_add_validation_result(self):
        """Test adding validation results to a rule."""
        validation_logic = ValidationLogic(
            logic_type="completeness",
            expression="NOT NULL",
            error_message_template="Field is required"
        )
        
        metadata = RuleMetadata(
            description="Test rule",
            business_justification="Test justification"
        )
        
        rule = QualityRule(
            rule_id=RuleId(),
            rule_name="test_rule",
            rule_type=RuleType.COMPLETENESS,
            category=QualityCategory.DATA_INTEGRITY,
            severity=Severity.MEDIUM,
            validation_logic=validation_logic,
            metadata=metadata,
            created_by=UserId()
        )
        
        result = ValidationResult(
            rule_id=rule.rule_id,
            dataset_id=DatasetId(),
            status="passed",
            total_records=100,
            records_passed=95,
            records_failed=5,
            pass_rate=0.95,
            execution_time_seconds=1.5
        )
        
        rule.add_validation_result(result)
        
        assert len(rule.recent_results) == 1
        assert rule.get_latest_result() == result
        assert rule.get_current_pass_rate() == 0.95
    
    def test_threshold_violation_detection(self):
        """Test threshold violation detection."""
        validation_logic = ValidationLogic(
            logic_type="completeness",
            expression="NOT NULL",
            error_message_template="Field is required"
        )
        
        metadata = RuleMetadata(
            description="Test rule",
            business_justification="Test justification"
        )
        
        # Create rule with strict thresholds
        thresholds = QualityThreshold(
            pass_rate_threshold=0.95,
            warning_threshold=0.90,
            critical_threshold=0.80
        )
        
        rule = QualityRule(
            rule_id=RuleId(),
            rule_name="test_rule",
            rule_type=RuleType.COMPLETENESS,
            category=QualityCategory.DATA_INTEGRITY,
            severity=Severity.MEDIUM,
            validation_logic=validation_logic,
            metadata=metadata,
            thresholds=thresholds,
            created_by=UserId()
        )
        
        # Add result that violates threshold
        failing_result = ValidationResult(
            rule_id=rule.rule_id,
            dataset_id=DatasetId(),
            status="failed",
            total_records=100,
            records_passed=85,
            records_failed=15,
            pass_rate=0.85,  # Below 0.95 threshold
            execution_time_seconds=1.5
        )
        
        rule.add_validation_result(failing_result)
        
        assert rule.is_threshold_violated() is True
        assert rule.get_violation_severity_level() == "minor"
        
        # Add critical failure
        critical_result = ValidationResult(
            rule_id=rule.rule_id,
            dataset_id=DatasetId(),
            status="failed",
            total_records=100,
            records_passed=70,
            records_failed=30,
            pass_rate=0.70,  # Below critical threshold
            execution_time_seconds=1.5
        )
        
        rule.add_validation_result(critical_result)
        
        assert rule.is_critical_violation() is True
        assert rule.get_violation_severity_level() == "critical"


class TestValidationLogic:
    """Test cases for ValidationLogic value object."""
    
    def test_create_validation_logic(self):
        """Test creating validation logic."""
        logic = ValidationLogic(
            logic_type="regex",
            expression="^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}$",
            parameters={"field": "email"},
            error_message_template="Invalid email format",
            success_criteria="Email must match standard format"
        )
        
        assert logic.logic_type == "regex"
        assert "email" in logic.expression
        assert logic.parameters["field"] == "email"


class TestQualityThreshold:
    """Test cases for QualityThreshold value object."""
    
    def test_create_quality_threshold(self):
        """Test creating quality thresholds."""
        threshold = QualityThreshold(
            pass_rate_threshold=0.95,
            warning_threshold=0.90,
            critical_threshold=0.80
        )
        
        assert threshold.pass_rate_threshold == 0.95
        assert threshold.warning_threshold == 0.90
        assert threshold.critical_threshold == 0.80
    
    def test_default_thresholds(self):
        """Test default threshold values."""
        threshold = QualityThreshold()
        
        assert threshold.pass_rate_threshold == 0.95
        assert threshold.warning_threshold == 0.90
        assert threshold.critical_threshold == 0.80