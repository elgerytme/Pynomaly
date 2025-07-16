"""Comprehensive tests for domain exceptions.

This module provides comprehensive test coverage for all domain exceptions
including base exceptions, entity-specific exceptions, validation errors,
and business rule violations.
"""

from uuid import uuid4

from monorepo.domain.exceptions.base import (
    BusinessRuleViolation,
    DomainException,
    ValidationError,
)
from monorepo.domain.exceptions.dataset_exceptions import (
    DatasetCorruptionError,
    DatasetFormatError,
    DatasetSchemaError,
    DatasetSizeError,
    DatasetValidationError,
)
from monorepo.domain.exceptions.detector_exceptions import (
    DetectorConfigurationError,
    DetectorNotTrainedError,
    DetectorStateError,
    DetectorValidationError,
)
from monorepo.domain.exceptions.entity_exceptions import (
    EntityConstraintViolationError,
    EntityNotFoundError,
    EntityStateError,
    EntityValidationError,
)
from monorepo.domain.exceptions.result_exceptions import (
    ResultInconsistencyError,
    ResultProcessingError,
    ResultStorageError,
    ResultValidationError,
)


class TestBaseDomainExceptions:
    """Comprehensive tests for base domain exceptions."""

    def test_domain_exception_creation(self):
        """Test creating base domain exception."""
        exception = DomainException(
            message="Something went wrong in the domain",
            error_code="DOMAIN_ERROR_001",
            context={"operation": "validation", "entity": "detector"},
        )

        assert str(exception) == "Something went wrong in the domain"
        assert exception.error_code == "DOMAIN_ERROR_001"
        assert exception.context["operation"] == "validation"
        assert exception.context["entity"] == "detector"

    def test_domain_exception_with_cause(self):
        """Test domain exception with underlying cause."""
        underlying_error = ValueError("Invalid input")

        domain_exception = DomainException(
            message="Domain validation failed",
            error_code="VALIDATION_FAILED",
            cause=underlying_error,
            context={"input_value": "invalid_data"},
        )

        assert domain_exception.cause == underlying_error
        assert "Invalid input" in str(domain_exception.cause)

    def test_validation_error(self):
        """Test validation error specific functionality."""
        validation_error = ValidationError(
            message="Field validation failed",
            field_name="contamination_rate",
            invalid_value=1.5,
            validation_rule="must_be_between_0_and_1",
            context={"min_value": 0.0, "max_value": 1.0},
        )

        assert validation_error.field_name == "contamination_rate"
        assert validation_error.invalid_value == 1.5
        assert validation_error.validation_rule == "must_be_between_0_and_1"
        assert validation_error.context["max_value"] == 1.0

    def test_business_rule_violation(self):
        """Test business rule violation functionality."""
        rule_violation = BusinessRuleViolation(
            message="Cannot deploy untrained model",
            rule_name="trained_model_deployment_rule",
            violated_constraints=["model_must_be_trained", "validation_score_minimum"],
            context={
                "model_id": str(uuid4()),
                "is_trained": False,
                "validation_score": None,
            },
        )

        assert rule_violation.rule_name == "trained_model_deployment_rule"
        assert "model_must_be_trained" in rule_violation.violated_constraints
        assert "validation_score_minimum" in rule_violation.violated_constraints
        assert rule_violation.context["is_trained"] is False

    def test_exception_hierarchy(self):
        """Test exception inheritance hierarchy."""
        # All domain exceptions should inherit from DomainException
        validation_error = ValidationError("Validation failed")
        business_rule_violation = BusinessRuleViolation("Rule violated")

        assert isinstance(validation_error, DomainException)
        assert isinstance(business_rule_violation, DomainException)

        # Should also be instances of standard Exception
        assert isinstance(validation_error, Exception)
        assert isinstance(business_rule_violation, Exception)

    def test_exception_serialization(self):
        """Test exception serialization for logging/storage."""
        exception = ValidationError(
            message="Invalid parameter",
            field_name="n_estimators",
            invalid_value=-5,
            validation_rule="positive_integer",
            context={"min_value": 1},
        )

        serialized = exception.to_dict()

        assert serialized["message"] == "Invalid parameter"
        assert serialized["field_name"] == "n_estimators"
        assert serialized["invalid_value"] == -5
        assert serialized["validation_rule"] == "positive_integer"
        assert serialized["context"]["min_value"] == 1
        assert "timestamp" in serialized
        assert "exception_type" in serialized


class TestEntityExceptions:
    """Comprehensive tests for entity-specific exceptions."""

    def test_entity_not_found_error(self):
        """Test entity not found error."""
        entity_id = uuid4()

        not_found_error = EntityNotFoundError(
            entity_type="Detector",
            entity_id=entity_id,
            search_criteria={"name": "NonExistentDetector", "status": "active"},
            context={"repository": "DetectorRepository", "operation": "find_by_name"},
        )

        assert not_found_error.entity_type == "Detector"
        assert not_found_error.entity_id == entity_id
        assert not_found_error.search_criteria["name"] == "NonExistentDetector"
        assert not_found_error.context["repository"] == "DetectorRepository"

    def test_entity_validation_error(self):
        """Test entity validation error."""
        validation_errors = [
            {"field": "name", "error": "cannot_be_empty"},
            {"field": "algorithm_name", "error": "must_be_supported_algorithm"},
            {
                "field": "parameters",
                "error": "missing_required_parameter_contamination",
            },
        ]

        entity_validation_error = EntityValidationError(
            entity_type="Detector",
            entity_data={"name": "", "algorithm_name": "UnsupportedAlgorithm"},
            validation_errors=validation_errors,
            context={"operation": "create_detector"},
        )

        assert entity_validation_error.entity_type == "Detector"
        assert len(entity_validation_error.validation_errors) == 3
        assert entity_validation_error.validation_errors[0]["field"] == "name"
        assert (
            entity_validation_error.validation_errors[0]["error"] == "cannot_be_empty"
        )

    def test_entity_state_error(self):
        """Test entity state error."""
        state_error = EntityStateError(
            entity_type="Detector",
            entity_id=uuid4(),
            current_state="untrained",
            required_state="trained",
            operation="detect_anomalies",
            context={"training_status": "not_started", "last_training_attempt": None},
        )

        assert state_error.entity_type == "Detector"
        assert state_error.current_state == "untrained"
        assert state_error.required_state == "trained"
        assert state_error.operation == "detect_anomalies"

    def test_entity_constraint_violation_error(self):
        """Test entity constraint violation error."""
        constraints = [
            {
                "constraint": "unique_name_per_user",
                "violated_by": {"name": "MyDetector", "user_id": "user123"},
                "existing_entity_id": str(uuid4()),
            },
            {
                "constraint": "max_detectors_per_user",
                "violated_by": {"user_id": "user123", "current_count": 10},
                "limit": 10,
            },
        ]

        constraint_error = EntityConstraintViolationError(
            entity_type="Detector",
            violated_constraints=constraints,
            context={"user_id": "user123", "operation": "create_detector"},
        )

        assert constraint_error.entity_type == "Detector"
        assert len(constraint_error.violated_constraints) == 2
        assert (
            constraint_error.violated_constraints[0]["constraint"]
            == "unique_name_per_user"
        )


class TestDetectorExceptions:
    """Comprehensive tests for detector-specific exceptions."""

    def test_detector_not_trained_error(self):
        """Test detector not trained error."""
        detector_id = uuid4()

        not_trained_error = DetectorNotTrainedError(
            detector_id=detector_id,
            detector_name="TestDetector",
            attempted_operation="detect",
            context={
                "created_at": "2023-10-01T10:00:00Z",
                "training_attempts": 0,
                "algorithm": "IsolationForest",
            },
        )

        assert not_trained_error.detector_id == detector_id
        assert not_trained_error.detector_name == "TestDetector"
        assert not_trained_error.attempted_operation == "detect"
        assert not_trained_error.context["training_attempts"] == 0

    def test_detector_configuration_error(self):
        """Test detector configuration error."""
        invalid_config = {
            "contamination": 1.5,  # Invalid: > 1.0
            "n_estimators": 0,  # Invalid: must be > 0
            "unknown_param": "value",  # Invalid: unknown parameter
        }

        config_errors = [
            {
                "parameter": "contamination",
                "error": "value_out_of_range",
                "valid_range": [0.0, 1.0],
            },
            {
                "parameter": "n_estimators",
                "error": "must_be_positive_integer",
                "minimum": 1,
            },
            {
                "parameter": "unknown_param",
                "error": "unknown_parameter",
                "valid_params": ["contamination", "n_estimators"],
            },
        ]

        config_error = DetectorConfigurationError(
            detector_id=uuid4(),
            algorithm_name="IsolationForest",
            invalid_configuration=invalid_config,
            configuration_errors=config_errors,
            context={"operation": "create_detector"},
        )

        assert config_error.algorithm_name == "IsolationForest"
        assert config_error.invalid_configuration["contamination"] == 1.5
        assert len(config_error.configuration_errors) == 3
        assert config_error.configuration_errors[0]["parameter"] == "contamination"

    def test_detector_validation_error(self):
        """Test detector validation error."""
        validation_issues = [
            {
                "category": "algorithm_compatibility",
                "issue": "algorithm_not_suitable_for_data_type",
                "details": "LocalOutlierFactor requires dense data, but sparse data provided",
            },
            {
                "category": "parameter_consistency",
                "issue": "incompatible_parameter_combination",
                "details": "n_neighbors cannot be greater than number of samples",
            },
        ]

        detector_validation_error = DetectorValidationError(
            detector_id=uuid4(),
            validation_issues=validation_issues,
            context={
                "algorithm": "LocalOutlierFactor",
                "data_type": "sparse",
                "n_samples": 50,
                "n_neighbors": 100,
            },
        )

        assert len(detector_validation_error.validation_issues) == 2
        assert (
            detector_validation_error.validation_issues[0]["category"]
            == "algorithm_compatibility"
        )
        assert detector_validation_error.context["n_neighbors"] == 100

    def test_detector_state_error(self):
        """Test detector state error."""
        detector_state_error = DetectorStateError(
            detector_id=uuid4(),
            current_state="training",
            requested_operation="delete",
            state_transition_error="cannot_delete_while_training",
            context={
                "training_started_at": "2023-10-01T10:00:00Z",
                "training_progress": 0.45,
                "estimated_completion": "2023-10-01T10:30:00Z",
            },
        )

        assert detector_state_error.current_state == "training"
        assert detector_state_error.requested_operation == "delete"
        assert (
            detector_state_error.state_transition_error
            == "cannot_delete_while_training"
        )
        assert detector_state_error.context["training_progress"] == 0.45


class TestDatasetExceptions:
    """Comprehensive tests for dataset-specific exceptions."""

    def test_dataset_validation_error(self):
        """Test dataset validation error."""
        validation_issues = [
            {
                "column": "feature1",
                "issue": "contains_null_values",
                "null_count": 150,
                "total_count": 1000,
            },
            {
                "column": "feature2",
                "issue": "contains_infinite_values",
                "infinite_count": 25,
            },
            {"column": "feature3", "issue": "single_unique_value", "unique_value": 0.0},
        ]

        dataset_validation_error = DatasetValidationError(
            dataset_id=uuid4(),
            dataset_name="TestDataset",
            validation_issues=validation_issues,
            context={
                "total_rows": 1000,
                "total_columns": 5,
                "validation_rules": [
                    "no_null_values",
                    "no_infinite_values",
                    "min_variance",
                ],
            },
        )

        assert dataset_validation_error.dataset_name == "TestDataset"
        assert len(dataset_validation_error.validation_issues) == 3
        assert dataset_validation_error.validation_issues[0]["null_count"] == 150

    def test_dataset_format_error(self):
        """Test dataset format error."""
        format_error = DatasetFormatError(
            dataset_id=uuid4(),
            expected_format="CSV",
            actual_format="JSON",
            format_issues=[
                "missing_header_row",
                "inconsistent_column_count",
                "invalid_delimiter",
            ],
            context={
                "file_path": "/data/invalid_dataset.csv",
                "file_size_bytes": 1048576,
                "detected_delimiter": ";",
                "expected_delimiter": ",",
            },
        )

        assert format_error.expected_format == "CSV"
        assert format_error.actual_format == "JSON"
        assert "missing_header_row" in format_error.format_issues
        assert format_error.context["detected_delimiter"] == ";"

    def test_dataset_size_error(self):
        """Test dataset size error."""
        size_error = DatasetSizeError(
            dataset_id=uuid4(),
            actual_size=50,
            minimum_required_size=100,
            size_type="rows",
            operation="train_detector",
            context={
                "algorithm": "IsolationForest",
                "recommended_minimum": 1000,
                "performance_impact": "high",
            },
        )

        assert size_error.actual_size == 50
        assert size_error.minimum_required_size == 100
        assert size_error.size_type == "rows"
        assert size_error.operation == "train_detector"
        assert size_error.context["recommended_minimum"] == 1000

    def test_dataset_schema_error(self):
        """Test dataset schema error."""
        schema_mismatches = [
            {
                "column": "temperature",
                "expected_type": "float64",
                "actual_type": "object",
                "convertible": False,
            },
            {
                "column": "pressure",
                "expected_range": [900, 1100],
                "actual_range": [0, 10000],
                "outlier_count": 150,
            },
        ]

        schema_error = DatasetSchemaError(
            dataset_id=uuid4(),
            expected_schema={
                "temperature": {"type": "float64", "range": [0, 50]},
                "pressure": {"type": "float64", "range": [900, 1100]},
            },
            actual_schema={
                "temperature": {"type": "object"},
                "pressure": {"type": "float64", "range": [0, 10000]},
            },
            schema_mismatches=schema_mismatches,
            context={"schema_version": "v1.2", "validation_strict": True},
        )

        assert len(schema_error.schema_mismatches) == 2
        assert schema_error.schema_mismatches[0]["column"] == "temperature"
        assert schema_error.schema_mismatches[0]["convertible"] is False

    def test_dataset_corruption_error(self):
        """Test dataset corruption error."""
        corruption_indicators = [
            {
                "type": "checksum_mismatch",
                "expected_checksum": "abc123def456",
                "actual_checksum": "xyz789uvw012",
                "severity": "high",
            },
            {
                "type": "partial_data_loss",
                "expected_rows": 10000,
                "actual_rows": 8500,
                "missing_percentage": 15.0,
            },
            {
                "type": "encoding_corruption",
                "affected_columns": ["description", "comments"],
                "corruption_pattern": "invalid_utf8_sequences",
            },
        ]

        corruption_error = DatasetCorruptionError(
            dataset_id=uuid4(),
            corruption_indicators=corruption_indicators,
            recovery_possible=True,
            context={
                "last_valid_backup": "2023-10-01T09:00:00Z",
                "corruption_detected_at": "2023-10-01T10:30:00Z",
                "recovery_strategy": "restore_from_backup",
            },
        )

        assert len(corruption_error.corruption_indicators) == 3
        assert corruption_error.recovery_possible is True
        assert corruption_error.corruption_indicators[0]["type"] == "checksum_mismatch"
        assert corruption_error.context["recovery_strategy"] == "restore_from_backup"


class TestResultExceptions:
    """Comprehensive tests for result-specific exceptions."""

    def test_result_validation_error(self):
        """Test result validation error."""
        validation_failures = [
            {
                "field": "anomaly_scores",
                "issue": "scores_out_of_range",
                "invalid_values": [1.5, -0.2, 2.0],
                "valid_range": [0.0, 1.0],
            },
            {
                "field": "labels",
                "issue": "invalid_label_values",
                "invalid_values": [2, 3, -1],
                "valid_values": [0, 1],
            },
        ]

        result_validation_error = ResultValidationError(
            result_id=uuid4(),
            result_type="DetectionResult",
            validation_failures=validation_failures,
            context={
                "detector_id": str(uuid4()),
                "dataset_id": str(uuid4()),
                "validation_timestamp": "2023-10-01T10:00:00Z",
            },
        )

        assert result_validation_error.result_type == "DetectionResult"
        assert len(result_validation_error.validation_failures) == 2
        assert (
            result_validation_error.validation_failures[0]["field"] == "anomaly_scores"
        )

    def test_result_processing_error(self):
        """Test result processing error."""
        processing_error = ResultProcessingError(
            result_id=uuid4(),
            processing_stage="score_normalization",
            error_details={
                "error_type": "numerical_overflow",
                "affected_samples": [1500, 2300, 4500],
                "original_scores": [999999.9, 1000000.1, 9999999.0],
            },
            recovery_attempted=True,
            context={
                "processor": "ScoreNormalizer",
                "normalization_method": "min_max_scaling",
                "max_score_threshold": 1000000.0,
            },
        )

        assert processing_error.processing_stage == "score_normalization"
        assert processing_error.recovery_attempted is True
        assert processing_error.error_details["error_type"] == "numerical_overflow"
        assert len(processing_error.error_details["affected_samples"]) == 3

    def test_result_storage_error(self):
        """Test result storage error."""
        storage_error = ResultStorageError(
            result_id=uuid4(),
            storage_backend="postgresql",
            storage_operation="insert",
            error_details={
                "error_code": "23505",  # PostgreSQL unique violation
                "error_message": "duplicate key value violates unique constraint",
                "constraint_name": "detection_results_pkey",
            },
            retry_attempted=True,
            context={
                "table_name": "detection_results",
                "connection_pool": "primary",
                "transaction_id": "tx_123456",
            },
        )

        assert storage_error.storage_backend == "postgresql"
        assert storage_error.storage_operation == "insert"
        assert storage_error.retry_attempted is True
        assert storage_error.error_details["error_code"] == "23505"

    def test_result_inconsistency_error(self):
        """Test result inconsistency error."""
        inconsistencies = [
            {
                "type": "score_label_mismatch",
                "description": "High anomaly scores with normal labels",
                "affected_indices": [100, 250, 890],
                "severity": "medium",
            },
            {
                "type": "threshold_violation",
                "description": "Labels don't match threshold-based classification",
                "threshold": 0.5,
                "violations": 45,
                "severity": "high",
            },
        ]

        inconsistency_error = ResultInconsistencyError(
            result_id=uuid4(),
            inconsistencies=inconsistencies,
            consistency_score=0.65,
            context={
                "detector_algorithm": "IsolationForest",
                "threshold_method": "contamination_based",
                "total_samples": 1000,
            },
        )

        assert len(inconsistency_error.inconsistencies) == 2
        assert inconsistency_error.consistency_score == 0.65
        assert inconsistency_error.inconsistencies[0]["type"] == "score_label_mismatch"
        assert inconsistency_error.inconsistencies[1]["violations"] == 45


class TestExceptionHandling:
    """Test exception handling patterns and utilities."""

    def test_exception_context_propagation(self):
        """Test context propagation through exception chain."""
        # Original context
        original_context = {"user_id": "user123", "operation": "train_model"}

        # Base exception with context
        base_exception = ValidationError(
            message="Invalid parameter",
            field_name="contamination",
            invalid_value=1.5,
            context=original_context,
        )

        # Wrapped exception that adds context
        wrapped_exception = DetectorConfigurationError(
            detector_id=uuid4(),
            algorithm_name="IsolationForest",
            invalid_configuration={"contamination": 1.5},
            configuration_errors=[
                {"parameter": "contamination", "error": "out_of_range"}
            ],
            cause=base_exception,
            context={"detector_name": "MyDetector", **original_context},
        )

        # Verify context propagation
        assert wrapped_exception.context["user_id"] == "user123"
        assert wrapped_exception.context["operation"] == "train_model"
        assert wrapped_exception.context["detector_name"] == "MyDetector"
        assert wrapped_exception.cause == base_exception

    def test_exception_aggregation(self):
        """Test aggregating multiple exceptions."""
        validation_errors = [
            ValidationError("Invalid contamination", field_name="contamination"),
            ValidationError("Invalid n_estimators", field_name="n_estimators"),
            ValidationError("Invalid random_state", field_name="random_state"),
        ]

        # Aggregate into single exception
        aggregated_exception = EntityValidationError(
            entity_type="Detector",
            entity_data={"contamination": 1.5, "n_estimators": 0},
            validation_errors=[
                {"field": "contamination", "error": "out_of_range"},
                {"field": "n_estimators", "error": "must_be_positive"},
                {"field": "random_state", "error": "invalid_type"},
            ],
            underlying_exceptions=validation_errors,
        )

        assert len(aggregated_exception.underlying_exceptions) == 3
        assert all(
            isinstance(exc, ValidationError)
            for exc in aggregated_exception.underlying_exceptions
        )

    def test_exception_retry_information(self):
        """Test exception with retry information."""
        retry_info = {
            "max_retries": 3,
            "current_attempt": 2,
            "retry_delay_seconds": 5.0,
            "exponential_backoff": True,
            "last_retry_at": "2023-10-01T10:05:00Z",
        }

        storage_error = ResultStorageError(
            result_id=uuid4(),
            storage_backend="redis",
            storage_operation="cache_result",
            error_details={"connection_timeout": True},
            retry_attempted=True,
            retry_info=retry_info,
            context={"cache_key": "result_12345"},
        )

        assert storage_error.retry_info["max_retries"] == 3
        assert storage_error.retry_info["current_attempt"] == 2
        assert storage_error.retry_info["exponential_backoff"] is True

        # Check if should retry
        should_retry = storage_error.should_retry()
        assert should_retry is True  # current_attempt < max_retries
