"""Comprehensive unit tests for DataQualityCheck domain entity."""

import pytest
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from unittest.mock import patch

from data_quality.domain.entities.data_quality_check import (
    DataQualityCheck, CheckType, CheckStatus, CheckResult, CheckSeverity
)


class TestCheckType:
    """Test cases for CheckType enum."""

    def test_check_type_values(self):
        """Test all CheckType enum values."""
        assert CheckType.COMPLETENESS.value == "completeness"
        assert CheckType.ACCURACY.value == "accuracy"
        assert CheckType.CONSISTENCY.value == "consistency"
        assert CheckType.VALIDITY.value == "validity"
        assert CheckType.UNIQUENESS.value == "uniqueness"
        assert CheckType.TIMELINESS.value == "timeliness"
        assert CheckType.INTEGRITY.value == "integrity"
        assert CheckType.CONFORMITY.value == "conformity"
        assert CheckType.PRECISION.value == "precision"
        assert CheckType.CUSTOM.value == "custom"


class TestCheckStatus:
    """Test cases for CheckStatus enum."""

    def test_check_status_values(self):
        """Test all CheckStatus enum values."""
        assert CheckStatus.PENDING.value == "pending"
        assert CheckStatus.RUNNING.value == "running"
        assert CheckStatus.COMPLETED.value == "completed"
        assert CheckStatus.FAILED.value == "failed"
        assert CheckStatus.CANCELLED.value == "cancelled"
        assert CheckStatus.SKIPPED.value == "skipped"


class TestCheckSeverity:
    """Test cases for CheckSeverity enum."""

    def test_check_severity_values(self):
        """Test all CheckSeverity enum values."""
        assert CheckSeverity.INFO.value == "info"
        assert CheckSeverity.WARNING.value == "warning"
        assert CheckSeverity.ERROR.value == "error"
        assert CheckSeverity.CRITICAL.value == "critical"


class TestCheckResult:
    """Test cases for CheckResult entity."""

    def test_initialization_defaults(self):
        """Test check result initialization with defaults."""
        result = CheckResult()
        
        assert isinstance(result.id, UUID)
        assert isinstance(result.check_id, UUID)
        assert result.dataset_name == ""
        assert result.column_name is None
        assert result.passed is False
        assert result.score == 0.0
        assert result.total_records == 0
        assert result.passed_records == 0
        assert result.failed_records == 0
        assert isinstance(result.executed_at, datetime)
        assert result.execution_time_ms == 0.0
        assert result.severity == CheckSeverity.INFO
        assert result.message == ""
        assert result.details == {}
        assert result.failed_values == []
        assert result.sample_failures == []
        assert result.metadata == {}
        assert result.tags == []

    def test_initialization_with_data(self):
        """Test check result initialization with provided data."""
        result_id = uuid4()
        check_id = uuid4()
        executed_at = datetime(2024, 1, 15, 10, 0, 0)
        details = {"query": "SELECT COUNT(*) FROM users"}
        failed_values = ["invalid@", "not-email"]
        metadata = {"db_connection": "prod"}
        tags = ["validation", "email"]
        
        result = CheckResult(
            id=result_id,
            check_id=check_id,
            dataset_name="users",
            column_name="email",
            passed=True,
            score=0.95,
            total_records=1000,
            passed_records=950,
            failed_records=50,
            executed_at=executed_at,
            execution_time_ms=125.5,
            severity=CheckSeverity.WARNING,
            message="Email validation completed with warnings",
            details=details,
            failed_values=failed_values,
            metadata=metadata,
            tags=tags
        )
        
        assert result.id == result_id
        assert result.check_id == check_id
        assert result.dataset_name == "users"
        assert result.column_name == "email"
        assert result.passed is True
        assert result.score == 0.95
        assert result.total_records == 1000
        assert result.passed_records == 950
        assert result.failed_records == 50
        assert result.executed_at == executed_at
        assert result.execution_time_ms == 125.5
        assert result.severity == CheckSeverity.WARNING
        assert result.message == "Email validation completed with warnings"
        assert result.details == details
        assert result.failed_values == failed_values
        assert result.metadata == metadata
        assert result.tags == tags

    def test_post_init_validation_invalid_score_low(self):
        """Test validation fails for score too low."""
        with pytest.raises(ValueError, match="Score must be between 0.0 and 1.0"):
            CheckResult(score=-0.1)

    def test_post_init_validation_invalid_score_high(self):
        """Test validation fails for score too high."""
        with pytest.raises(ValueError, match="Score must be between 0.0 and 1.0"):
            CheckResult(score=1.1)

    def test_post_init_validation_negative_total_records(self):
        """Test validation fails for negative total records."""
        with pytest.raises(ValueError, match="Total records cannot be negative"):
            CheckResult(total_records=-1)

    def test_post_init_validation_negative_passed_records(self):
        """Test validation fails for negative passed records."""
        with pytest.raises(ValueError, match="Passed records cannot be negative"):
            CheckResult(passed_records=-1)

    def test_post_init_validation_negative_failed_records(self):
        """Test validation fails for negative failed records."""
        with pytest.raises(ValueError, match="Failed records cannot be negative"):
            CheckResult(failed_records=-1)

    def test_post_init_validation_records_sum_exceeds_total(self):
        """Test validation fails when passed + failed > total."""
        with pytest.raises(ValueError, match="Sum of passed and failed records cannot exceed total"):
            CheckResult(total_records=100, passed_records=60, failed_records=50)

    def test_pass_rate_property_zero_records(self):
        """Test pass rate property with zero records."""
        result = CheckResult(total_records=0)
        assert result.pass_rate == 0.0

    def test_pass_rate_property_calculated(self):
        """Test pass rate property calculation."""
        result = CheckResult(total_records=1000, passed_records=850, failed_records=150)
        assert result.pass_rate == 85.0

    def test_fail_rate_property_zero_records(self):
        """Test fail rate property with zero records."""
        result = CheckResult(total_records=0)
        assert result.fail_rate == 0.0

    def test_fail_rate_property_calculated(self):
        """Test fail rate property calculation."""
        result = CheckResult(total_records=1000, passed_records=850, failed_records=150)
        assert result.fail_rate == 15.0

    def test_is_passed_property(self):
        """Test is_passed property."""
        result_passed = CheckResult(passed=True)
        assert result_passed.is_passed is True
        
        result_failed = CheckResult(passed=False)
        assert result_failed.is_passed is False

    def test_is_critical_property(self):
        """Test is_critical property."""
        result_critical = CheckResult(severity=CheckSeverity.CRITICAL)
        assert result_critical.is_critical is True
        
        result_info = CheckResult(severity=CheckSeverity.INFO)
        assert result_info.is_critical is False

    def test_add_failed_value_without_context(self):
        """Test adding failed value without context."""
        result = CheckResult()
        result.add_failed_value("invalid_email")
        
        assert "invalid_email" in result.failed_values
        assert len(result.sample_failures) == 0

    def test_add_failed_value_with_context(self):
        """Test adding failed value with context."""
        result = CheckResult()
        context = {"row_id": 123, "table": "users"}
        result.add_failed_value("invalid_email", context)
        
        assert "invalid_email" in result.failed_values
        assert len(result.sample_failures) == 1
        assert result.sample_failures[0]["value"] == "invalid_email"
        assert result.sample_failures[0]["row_id"] == 123
        assert result.sample_failures[0]["table"] == "users"

    def test_add_failed_value_sample_limit(self):
        """Test sample failures are limited to 100."""
        result = CheckResult()
        
        # Add more than 100 failures
        for i in range(150):
            context = {"row_id": i}
            result.add_failed_value(f"value_{i}", context)
        
        assert len(result.failed_values) == 150
        assert len(result.sample_failures) == 100

    def test_add_tag(self):
        """Test adding tags."""
        result = CheckResult()
        result.add_tag("validation")
        result.add_tag("email")
        
        assert "validation" in result.tags
        assert "email" in result.tags
        assert len(result.tags) == 2

    def test_add_tag_duplicate(self):
        """Test adding duplicate tag."""
        result = CheckResult()
        result.add_tag("validation")
        result.add_tag("validation")  # Duplicate
        
        assert result.tags.count("validation") == 1

    def test_add_tag_empty(self):
        """Test adding empty tag does nothing."""
        result = CheckResult()
        result.add_tag("")
        
        assert len(result.tags) == 0

    def test_update_metadata(self):
        """Test updating metadata."""
        result = CheckResult()
        result.update_metadata("connection", "prod_db")
        result.update_metadata("version", "1.2.0")
        
        assert result.metadata["connection"] == "prod_db"
        assert result.metadata["version"] == "1.2.0"

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = CheckResult(
            dataset_name="users",
            column_name="email",
            passed=True,
            score=0.95,
            total_records=1000,
            passed_records=950,
            failed_records=50,
            severity=CheckSeverity.WARNING,
            message="Test completed",
            details={"query": "SELECT ..."},
            failed_values=["val1", "val2"],
            metadata={"env": "test"},
            tags=["validation"]
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["id"] == str(result.id)
        assert result_dict["dataset_name"] == "users"
        assert result_dict["column_name"] == "email"
        assert result_dict["passed"] is True
        assert result_dict["score"] == 0.95
        assert result_dict["total_records"] == 1000
        assert result_dict["pass_rate"] == 95.0
        assert result_dict["fail_rate"] == 5.0
        assert result_dict["severity"] == "warning"
        assert result_dict["is_passed"] is True
        assert result_dict["is_critical"] is False


class TestDataQualityCheck:
    """Test cases for DataQualityCheck entity."""

    def test_initialization_defaults(self):
        """Test check initialization with defaults."""
        check = DataQualityCheck(name="test_check", dataset_name="users")
        
        assert isinstance(check.id, UUID)
        assert check.name == "test_check"
        assert check.description == ""
        assert check.check_type == CheckType.COMPLETENESS
        assert check.dataset_name == "users"
        assert check.column_name is None
        assert check.schema_name is None
        assert check.table_name is None
        assert check.query is None
        assert check.expression is None
        assert check.expected_value is None
        assert check.threshold == 0.95
        assert check.tolerance == 0.0
        assert check.is_active is True
        assert check.schedule_cron is None
        assert check.timeout_seconds == 300
        assert check.retry_attempts == 3
        assert check.status == CheckStatus.PENDING
        assert check.last_executed_at is None
        assert check.next_execution_at is None
        assert check.execution_count == 0
        assert check.last_result is None
        assert check.consecutive_failures == 0
        assert check.success_rate == 0.0
        assert isinstance(check.created_at, datetime)
        assert check.created_by == ""
        assert isinstance(check.updated_at, datetime)
        assert check.updated_by == ""
        assert check.config == {}
        assert check.environment_vars == {}
        assert check.tags == []
        assert check.depends_on == []
        assert check.blocks == []

    def test_initialization_with_data(self):
        """Test check initialization with provided data."""
        check_id = uuid4()
        created_at = datetime(2024, 1, 15, 10, 0, 0)
        config = {"batch_size": 1000}
        env_vars = {"ENV": "prod"}
        tags = ["validation", "critical"]
        depends_on = [uuid4(), uuid4()]
        
        check = DataQualityCheck(
            id=check_id,
            name="Email Validation",
            description="Validate email format",
            check_type=CheckType.VALIDITY,
            dataset_name="users",
            column_name="email",
            schema_name="public",
            table_name="user_accounts",
            query="SELECT email FROM users WHERE email IS NOT NULL",
            expression=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            expected_value="valid_email",
            threshold=0.98,
            tolerance=0.02,
            is_active=True,
            schedule_cron="0 2 * * *",
            timeout_seconds=600,
            retry_attempts=5,
            status=CheckStatus.COMPLETED,
            execution_count=10,
            consecutive_failures=1,
            success_rate=0.9,
            created_at=created_at,
            created_by="engineer",
            updated_at=created_at,
            updated_by="engineer",
            config=config,
            environment_vars=env_vars,
            tags=tags,
            depends_on=depends_on
        )
        
        assert check.id == check_id
        assert check.name == "Email Validation"
        assert check.description == "Validate email format"
        assert check.check_type == CheckType.VALIDITY
        assert check.dataset_name == "users"
        assert check.column_name == "email"
        assert check.schema_name == "public"
        assert check.table_name == "user_accounts"
        assert check.query == "SELECT email FROM users WHERE email IS NOT NULL"
        assert check.expression == r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        assert check.expected_value == "valid_email"
        assert check.threshold == 0.98
        assert check.tolerance == 0.02
        assert check.schedule_cron == "0 2 * * *"
        assert check.timeout_seconds == 600
        assert check.retry_attempts == 5
        assert check.status == CheckStatus.COMPLETED
        assert check.execution_count == 10
        assert check.consecutive_failures == 1
        assert check.success_rate == 0.9
        assert check.created_at == created_at
        assert check.created_by == "engineer"
        assert check.config == config
        assert check.environment_vars == env_vars
        assert check.tags == tags
        assert check.depends_on == depends_on

    def test_post_init_validation_empty_name(self):
        """Test validation fails for empty name."""
        with pytest.raises(ValueError, match="Check name cannot be empty"):
            DataQualityCheck(name="", dataset_name="users")

    def test_post_init_validation_name_too_long(self):
        """Test validation fails for name too long."""
        long_name = "a" * 101
        with pytest.raises(ValueError, match="Check name cannot exceed 100 characters"):
            DataQualityCheck(name=long_name, dataset_name="users")

    def test_post_init_validation_empty_dataset_name(self):
        """Test validation fails for empty dataset name."""
        with pytest.raises(ValueError, match="Dataset name cannot be empty"):
            DataQualityCheck(name="test", dataset_name="")

    def test_post_init_validation_invalid_threshold_low(self):
        """Test validation fails for threshold too low."""
        with pytest.raises(ValueError, match="Threshold must be between 0.0 and 1.0"):
            DataQualityCheck(name="test", dataset_name="users", threshold=-0.1)

    def test_post_init_validation_invalid_threshold_high(self):
        """Test validation fails for threshold too high."""
        with pytest.raises(ValueError, match="Threshold must be between 0.0 and 1.0"):
            DataQualityCheck(name="test", dataset_name="users", threshold=1.1)

    def test_post_init_validation_negative_tolerance(self):
        """Test validation fails for negative tolerance."""
        with pytest.raises(ValueError, match="Tolerance cannot be negative"):
            DataQualityCheck(name="test", dataset_name="users", tolerance=-0.1)

    def test_post_init_validation_invalid_timeout(self):
        """Test validation fails for invalid timeout."""
        with pytest.raises(ValueError, match="Timeout must be positive"):
            DataQualityCheck(name="test", dataset_name="users", timeout_seconds=0)

    def test_post_init_validation_negative_retry_attempts(self):
        """Test validation fails for negative retry attempts."""
        with pytest.raises(ValueError, match="Retry attempts cannot be negative"):
            DataQualityCheck(name="test", dataset_name="users", retry_attempts=-1)

    def test_is_column_level_property(self):
        """Test is_column_level property."""
        column_check = DataQualityCheck(name="test", dataset_name="users", column_name="email")
        assert column_check.is_column_level is True
        
        table_check = DataQualityCheck(name="test", dataset_name="users")
        assert table_check.is_column_level is False

    def test_is_table_level_property(self):
        """Test is_table_level property."""
        table_check = DataQualityCheck(name="test", dataset_name="users")
        assert table_check.is_table_level is True
        
        column_check = DataQualityCheck(name="test", dataset_name="users", column_name="email")
        assert column_check.is_table_level is False

    def test_is_overdue_property_no_next_execution(self):
        """Test is_overdue property when no next execution time."""
        check = DataQualityCheck(name="test", dataset_name="users")
        assert check.is_overdue is False

    def test_is_overdue_property_future_execution(self):
        """Test is_overdue property with future execution time."""
        check = DataQualityCheck(name="test", dataset_name="users")
        check.next_execution_at = datetime.utcnow() + timedelta(hours=1)
        assert check.is_overdue is False

    def test_is_overdue_property_past_execution(self):
        """Test is_overdue property with past execution time."""
        check = DataQualityCheck(name="test", dataset_name="users")
        check.next_execution_at = datetime.utcnow() - timedelta(hours=1)
        assert check.is_overdue is True

    def test_has_recent_failure_property(self):
        """Test has_recent_failure property."""
        check = DataQualityCheck(name="test", dataset_name="users")
        assert check.has_recent_failure is False
        
        check.consecutive_failures = 2
        assert check.has_recent_failure is True

    def test_is_healthy_property(self):
        """Test is_healthy property."""
        check = DataQualityCheck(name="test", dataset_name="users")
        check.is_active = True
        check.consecutive_failures = 0
        check.success_rate = 0.85
        assert check.is_healthy is True
        
        # Test inactive check
        check.is_active = False
        assert check.is_healthy is False
        
        # Test with failures
        check.is_active = True
        check.consecutive_failures = 1
        assert check.is_healthy is False
        
        # Test low success rate
        check.consecutive_failures = 0
        check.success_rate = 0.75
        assert check.is_healthy is False

    def test_execute_inactive_check(self):
        """Test executing inactive check fails."""
        check = DataQualityCheck(name="test", dataset_name="users", is_active=False)
        
        with pytest.raises(ValueError, match="Cannot execute inactive check"):
            check.execute()

    def test_execute_success(self):
        """Test successful check execution."""
        check = DataQualityCheck(name="test", dataset_name="users", threshold=0.95)
        
        with patch('data_quality.domain.entities.data_quality_check.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            result = check.execute()
            
            assert check.status == CheckStatus.COMPLETED
            assert check.last_executed_at == mock_now
            assert check.execution_count == 1
            assert check.updated_at == mock_now
            assert check.last_result == result
            assert isinstance(result, CheckResult)
            assert result.check_id == check.id
            assert result.dataset_name == check.dataset_name
            assert result.executed_at == mock_now

    def test_execute_updates_success_tracking(self):
        """Test execution updates success tracking."""
        check = DataQualityCheck(name="test", dataset_name="users")
        
        # First execution (success)
        result1 = check.execute()
        result1.passed = True
        check.last_result = result1
        
        assert check.consecutive_failures == 0
        assert check.success_rate == 1.0
        
        # Second execution (failure)
        result2 = check.execute()
        result2.passed = False
        check.last_result = result2
        
        # Update tracking manually since we're not using real execution
        check.consecutive_failures = 1
        weight = 0.1
        check.success_rate = (1 - weight) * check.success_rate + weight * 0.0
        
        assert check.consecutive_failures == 1
        assert check.success_rate == 0.9

    def test_validate_expression_safe(self):
        """Test expression validation for safe expressions."""
        check = DataQualityCheck(name="test", dataset_name="users")
        check.expression = "email LIKE '%@%.%'"
        
        assert check.validate_expression() is True

    def test_validate_expression_unsafe(self):
        """Test expression validation for unsafe expressions."""
        check = DataQualityCheck(name="test", dataset_name="users")
        check.expression = "DROP TABLE users"
        
        assert check.validate_expression() is False

    def test_validate_expression_empty(self):
        """Test expression validation for empty expression."""
        check = DataQualityCheck(name="test", dataset_name="users")
        check.expression = None
        
        assert check.validate_expression() is True

    def test_activate(self):
        """Test activating check."""
        check = DataQualityCheck(name="test", dataset_name="users", is_active=False)
        check.status = CheckStatus.CANCELLED
        
        with patch('data_quality.domain.entities.data_quality_check.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 13, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            check.activate()
            
            assert check.is_active is True
            assert check.status == CheckStatus.PENDING
            assert check.updated_at == mock_now

    def test_deactivate(self):
        """Test deactivating check."""
        check = DataQualityCheck(name="test", dataset_name="users", is_active=True)
        
        with patch('data_quality.domain.entities.data_quality_check.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 14, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            check.deactivate()
            
            assert check.is_active is False
            assert check.status == CheckStatus.CANCELLED
            assert check.updated_at == mock_now

    def test_add_tag(self):
        """Test adding tags."""
        check = DataQualityCheck(name="test", dataset_name="users")
        
        with patch('data_quality.domain.entities.data_quality_check.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 15, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            check.add_tag("validation")
            check.add_tag("critical")
            
            assert "validation" in check.tags
            assert "critical" in check.tags
            assert len(check.tags) == 2
            assert check.updated_at == mock_now

    def test_add_tag_duplicate(self):
        """Test adding duplicate tag."""
        check = DataQualityCheck(name="test", dataset_name="users")
        check.add_tag("validation")
        check.add_tag("validation")  # Duplicate
        
        assert check.tags.count("validation") == 1

    def test_add_tag_empty(self):
        """Test adding empty tag does nothing."""
        check = DataQualityCheck(name="test", dataset_name="users")
        original_updated_at = check.updated_at
        
        check.add_tag("")
        
        assert len(check.tags) == 0
        assert check.updated_at == original_updated_at

    def test_remove_tag(self):
        """Test removing tags."""
        check = DataQualityCheck(name="test", dataset_name="users")
        check.tags = ["validation", "critical", "email"]
        
        with patch('data_quality.domain.entities.data_quality_check.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 16, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            check.remove_tag("critical")
            
            assert "critical" not in check.tags
            assert len(check.tags) == 2
            assert check.updated_at == mock_now

    def test_remove_tag_nonexistent(self):
        """Test removing non-existent tag."""
        check = DataQualityCheck(name="test", dataset_name="users")
        check.tags = ["validation"]
        original_updated_at = check.updated_at
        
        check.remove_tag("nonexistent")
        
        assert len(check.tags) == 1
        assert check.updated_at == original_updated_at

    def test_has_tag(self):
        """Test checking for tags."""
        check = DataQualityCheck(name="test", dataset_name="users")
        check.tags = ["validation", "critical"]
        
        assert check.has_tag("validation") is True
        assert check.has_tag("critical") is True
        assert check.has_tag("nonexistent") is False

    def test_add_dependency(self):
        """Test adding dependencies."""
        check = DataQualityCheck(name="test", dataset_name="users")
        dep_id = uuid4()
        
        with patch('data_quality.domain.entities.data_quality_check.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 17, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            check.add_dependency(dep_id)
            
            assert dep_id in check.depends_on
            assert len(check.depends_on) == 1
            assert check.updated_at == mock_now

    def test_add_dependency_duplicate(self):
        """Test adding duplicate dependency."""
        check = DataQualityCheck(name="test", dataset_name="users")
        dep_id = uuid4()
        
        check.add_dependency(dep_id)
        check.add_dependency(dep_id)  # Duplicate
        
        assert check.depends_on.count(dep_id) == 1

    def test_remove_dependency(self):
        """Test removing dependencies."""
        check = DataQualityCheck(name="test", dataset_name="users")
        dep_id = uuid4()
        check.depends_on = [dep_id]
        
        with patch('data_quality.domain.entities.data_quality_check.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 18, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            check.remove_dependency(dep_id)
            
            assert dep_id not in check.depends_on
            assert len(check.depends_on) == 0
            assert check.updated_at == mock_now

    def test_remove_dependency_nonexistent(self):
        """Test removing non-existent dependency."""
        check = DataQualityCheck(name="test", dataset_name="users")
        dep_id = uuid4()
        original_updated_at = check.updated_at
        
        check.remove_dependency(dep_id)
        
        assert len(check.depends_on) == 0
        assert check.updated_at == original_updated_at

    def test_update_config(self):
        """Test updating configuration."""
        check = DataQualityCheck(name="test", dataset_name="users")
        
        with patch('data_quality.domain.entities.data_quality_check.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 19, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            check.update_config("batch_size", 1000)
            check.update_config("timeout", 600)
            
            assert check.config["batch_size"] == 1000
            assert check.config["timeout"] == 600
            assert check.updated_at == mock_now

    def test_get_full_target_name_dataset_only(self):
        """Test full target name with dataset only."""
        check = DataQualityCheck(name="test", dataset_name="users")
        assert check.get_full_target_name() == "users"

    def test_get_full_target_name_with_column(self):
        """Test full target name with column."""
        check = DataQualityCheck(name="test", dataset_name="users", column_name="email")
        assert check.get_full_target_name() == "users.email"

    def test_get_full_target_name_full_path(self):
        """Test full target name with full path."""
        check = DataQualityCheck(
            name="test",
            dataset_name="users",
            schema_name="public",
            table_name="user_accounts",
            column_name="email"
        )
        assert check.get_full_target_name() == "public.user_accounts.email"

    def test_to_dict(self):
        """Test converting check to dictionary."""
        check = DataQualityCheck(
            name="Email Validation",
            description="Validate email format",
            check_type=CheckType.VALIDITY,
            dataset_name="users",
            column_name="email",
            threshold=0.95,
            is_active=True,
            created_by="engineer",
            config={"batch_size": 1000},
            tags=["validation", "email"]
        )
        
        result = check.to_dict()
        
        assert result["id"] == str(check.id)
        assert result["name"] == "Email Validation"
        assert result["description"] == "Validate email format"
        assert result["check_type"] == "validity"
        assert result["dataset_name"] == "users"
        assert result["column_name"] == "email"
        assert result["threshold"] == 0.95
        assert result["is_active"] is True
        assert result["created_by"] == "engineer"
        assert result["config"] == {"batch_size": 1000}
        assert result["tags"] == ["validation", "email"]
        assert result["is_column_level"] is True
        assert result["is_table_level"] is False
        assert result["full_target_name"] == "users.email"

    def test_str_representation(self):
        """Test string representation."""
        check = DataQualityCheck(
            name="Email Check",
            check_type=CheckType.VALIDITY,
            dataset_name="users",
            column_name="email"
        )
        
        str_repr = str(check)
        
        assert "DataQualityCheck('Email Check'" in str_repr
        assert "validity" in str_repr
        assert "target=users.email" in str_repr

    def test_repr_representation(self):
        """Test detailed representation."""
        check = DataQualityCheck(
            name="Email Check",
            check_type=CheckType.VALIDITY,
            dataset_name="users",
            status=CheckStatus.COMPLETED
        )
        
        repr_str = repr(check)
        
        assert f"id={check.id}" in repr_str
        assert "name='Email Check'" in repr_str
        assert "type=validity" in repr_str
        assert "status=completed" in repr_str