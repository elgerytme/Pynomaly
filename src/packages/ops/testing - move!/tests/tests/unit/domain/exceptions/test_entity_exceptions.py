"""Tests for entity domain exceptions."""

from uuid import UUID, uuid4

import pytest

from monorepo.domain.exceptions.base import DomainError, PynamolyError
from monorepo.domain.exceptions.entity_exceptions import (
    AlertNotFoundError,
    EntityNotFoundError,
    ExperimentNotFoundError,
    InvalidAlertStateError,
    InvalidEntityStateError,
    InvalidExperimentStateError,
    InvalidModelStateError,
    InvalidPipelineStateError,
    ModelNotFoundError,
    PipelineNotFoundError,
)


class TestEntityNotFoundError:
    """Test suite for EntityNotFoundError base class."""

    def test_inheritance(self):
        """Test EntityNotFoundError inheritance."""
        entity_id = uuid4()
        error = EntityNotFoundError("TestEntity", entity_id)
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)
        assert isinstance(error, EntityNotFoundError)

    def test_basic_creation(self):
        """Test basic entity not found error creation."""
        entity_id = uuid4()
        error = EntityNotFoundError("TestEntity", entity_id)
        expected_message = f"TestEntity with ID {entity_id} not found"
        assert error.message == expected_message
        assert error.entity_type == "TestEntity"
        assert error.entity_id == entity_id

    def test_creation_with_custom_message(self):
        """Test creation with custom message."""
        entity_id = uuid4()
        custom_message = "Custom not found message"
        error = EntityNotFoundError("TestEntity", entity_id, custom_message)
        assert error.message == custom_message
        assert error.entity_type == "TestEntity"
        assert error.entity_id == entity_id

    def test_creation_with_none_message(self):
        """Test creation with None message."""
        entity_id = uuid4()
        error = EntityNotFoundError("TestEntity", entity_id, None)
        expected_message = f"TestEntity with ID {entity_id} not found"
        assert error.message == expected_message

    def test_creation_with_empty_message(self):
        """Test creation with empty message."""
        entity_id = uuid4()
        error = EntityNotFoundError("TestEntity", entity_id, "")
        assert error.message == ""

    def test_entity_type_variations(self):
        """Test with different entity types."""
        entity_id = uuid4()

        # Test with different entity types
        error1 = EntityNotFoundError("Model", entity_id)
        assert error1.entity_type == "Model"
        assert f"Model with ID {entity_id} not found" in error1.message

        error2 = EntityNotFoundError("Experiment", entity_id)
        assert error2.entity_type == "Experiment"
        assert f"Experiment with ID {entity_id} not found" in error2.message

    def test_string_representation(self):
        """Test string representation."""
        entity_id = uuid4()
        error = EntityNotFoundError("TestEntity", entity_id)
        result = str(error)
        assert f"TestEntity with ID {entity_id} not found" in result

    def test_exception_raising(self):
        """Test that error can be raised."""
        entity_id = uuid4()
        with pytest.raises(EntityNotFoundError):
            raise EntityNotFoundError("TestEntity", entity_id)

    def test_uuid_handling(self):
        """Test proper UUID handling."""
        entity_id = uuid4()
        error = EntityNotFoundError("TestEntity", entity_id)
        assert isinstance(error.entity_id, UUID)
        assert error.entity_id == entity_id


class TestInvalidEntityStateError:
    """Test suite for InvalidEntityStateError base class."""

    def test_inheritance(self):
        """Test InvalidEntityStateError inheritance."""
        entity_id = uuid4()
        error = InvalidEntityStateError(
            "TestEntity", entity_id, "update", "entity is locked"
        )
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)
        assert isinstance(error, InvalidEntityStateError)

    def test_basic_creation(self):
        """Test basic invalid entity state error creation."""
        entity_id = uuid4()
        error = InvalidEntityStateError(
            "TestEntity", entity_id, "update", "entity is locked"
        )
        expected_message = (
            f"Cannot perform 'update' on TestEntity {entity_id}: entity is locked"
        )
        assert error.message == expected_message
        assert error.entity_type == "TestEntity"
        assert error.entity_id == entity_id
        assert error.operation == "update"
        assert error.reason == "entity is locked"

    def test_different_operations(self):
        """Test with different operations."""
        entity_id = uuid4()

        error1 = InvalidEntityStateError(
            "Model", entity_id, "delete", "model is in use"
        )
        assert error1.operation == "delete"
        assert "delete" in error1.message

        error2 = InvalidEntityStateError(
            "Model", entity_id, "train", "model is not configured"
        )
        assert error2.operation == "train"
        assert "train" in error2.message

    def test_different_reasons(self):
        """Test with different reasons."""
        entity_id = uuid4()

        error1 = InvalidEntityStateError(
            "Model", entity_id, "update", "entity is locked"
        )
        assert error1.reason == "entity is locked"
        assert "entity is locked" in error1.message

        error2 = InvalidEntityStateError(
            "Model", entity_id, "update", "insufficient permissions"
        )
        assert error2.reason == "insufficient permissions"
        assert "insufficient permissions" in error2.message

    def test_empty_operation(self):
        """Test with empty operation."""
        entity_id = uuid4()
        error = InvalidEntityStateError("TestEntity", entity_id, "", "reason")
        assert error.operation == ""
        assert "Cannot perform '' on TestEntity" in error.message

    def test_empty_reason(self):
        """Test with empty reason."""
        entity_id = uuid4()
        error = InvalidEntityStateError("TestEntity", entity_id, "update", "")
        assert error.reason == ""
        assert f"Cannot perform 'update' on TestEntity {entity_id}: " in error.message

    def test_string_representation(self):
        """Test string representation."""
        entity_id = uuid4()
        error = InvalidEntityStateError(
            "TestEntity", entity_id, "update", "entity is locked"
        )
        result = str(error)
        assert (
            f"Cannot perform 'update' on TestEntity {entity_id}: entity is locked"
            in result
        )

    def test_exception_raising(self):
        """Test that error can be raised."""
        entity_id = uuid4()
        with pytest.raises(InvalidEntityStateError):
            raise InvalidEntityStateError(
                "TestEntity", entity_id, "update", "entity is locked"
            )


class TestModelNotFoundError:
    """Test suite for ModelNotFoundError."""

    def test_inheritance(self):
        """Test ModelNotFoundError inheritance."""
        model_id = uuid4()
        error = ModelNotFoundError(model_id)
        assert isinstance(error, EntityNotFoundError)
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)

    def test_basic_creation(self):
        """Test basic model not found error creation."""
        model_id = uuid4()
        error = ModelNotFoundError(model_id)
        expected_message = f"Model with ID {model_id} not found"
        assert error.message == expected_message
        assert error.entity_type == "Model"
        assert error.entity_id == model_id

    def test_creation_with_custom_message(self):
        """Test creation with custom message."""
        model_id = uuid4()
        custom_message = "Custom model not found message"
        error = ModelNotFoundError(model_id, custom_message)
        assert error.message == custom_message
        assert error.entity_type == "Model"
        assert error.entity_id == model_id

    def test_creation_with_none_message(self):
        """Test creation with None message."""
        model_id = uuid4()
        error = ModelNotFoundError(model_id, None)
        expected_message = f"Model with ID {model_id} not found"
        assert error.message == expected_message

    def test_string_representation(self):
        """Test string representation."""
        model_id = uuid4()
        error = ModelNotFoundError(model_id)
        result = str(error)
        assert f"Model with ID {model_id} not found" in result

    def test_exception_raising(self):
        """Test that error can be raised."""
        model_id = uuid4()
        with pytest.raises(ModelNotFoundError):
            raise ModelNotFoundError(model_id)


class TestInvalidModelStateError:
    """Test suite for InvalidModelStateError."""

    def test_inheritance(self):
        """Test InvalidModelStateError inheritance."""
        model_id = uuid4()
        error = InvalidModelStateError(model_id, "train", "model is not configured")
        assert isinstance(error, InvalidEntityStateError)
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)

    def test_basic_creation(self):
        """Test basic invalid model state error creation."""
        model_id = uuid4()
        error = InvalidModelStateError(model_id, "train", "model is not configured")
        expected_message = (
            f"Cannot perform 'train' on Model {model_id}: model is not configured"
        )
        assert error.message == expected_message
        assert error.entity_type == "Model"
        assert error.entity_id == model_id
        assert error.operation == "train"
        assert error.reason == "model is not configured"

    def test_different_operations(self):
        """Test with different operations."""
        model_id = uuid4()

        error1 = InvalidModelStateError(model_id, "predict", "model is not fitted")
        assert error1.operation == "predict"
        assert "predict" in error1.message

        error2 = InvalidModelStateError(model_id, "export", "model is corrupted")
        assert error2.operation == "export"
        assert "export" in error2.message

    def test_string_representation(self):
        """Test string representation."""
        model_id = uuid4()
        error = InvalidModelStateError(model_id, "train", "model is not configured")
        result = str(error)
        assert (
            f"Cannot perform 'train' on Model {model_id}: model is not configured"
            in result
        )

    def test_exception_raising(self):
        """Test that error can be raised."""
        model_id = uuid4()
        with pytest.raises(InvalidModelStateError):
            raise InvalidModelStateError(model_id, "train", "model is not configured")


class TestExperimentNotFoundError:
    """Test suite for ExperimentNotFoundError."""

    def test_inheritance(self):
        """Test ExperimentNotFoundError inheritance."""
        experiment_id = uuid4()
        error = ExperimentNotFoundError(experiment_id)
        assert isinstance(error, EntityNotFoundError)
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)

    def test_basic_creation(self):
        """Test basic experiment not found error creation."""
        experiment_id = uuid4()
        error = ExperimentNotFoundError(experiment_id)
        expected_message = f"Experiment with ID {experiment_id} not found"
        assert error.message == expected_message
        assert error.entity_type == "Experiment"
        assert error.entity_id == experiment_id

    def test_creation_with_custom_message(self):
        """Test creation with custom message."""
        experiment_id = uuid4()
        custom_message = "Custom experiment not found message"
        error = ExperimentNotFoundError(experiment_id, custom_message)
        assert error.message == custom_message
        assert error.entity_type == "Experiment"
        assert error.entity_id == experiment_id

    def test_string_representation(self):
        """Test string representation."""
        experiment_id = uuid4()
        error = ExperimentNotFoundError(experiment_id)
        result = str(error)
        assert f"Experiment with ID {experiment_id} not found" in result

    def test_exception_raising(self):
        """Test that error can be raised."""
        experiment_id = uuid4()
        with pytest.raises(ExperimentNotFoundError):
            raise ExperimentNotFoundError(experiment_id)


class TestInvalidExperimentStateError:
    """Test suite for InvalidExperimentStateError."""

    def test_inheritance(self):
        """Test InvalidExperimentStateError inheritance."""
        experiment_id = uuid4()
        error = InvalidExperimentStateError(
            experiment_id, "start", "experiment is already running"
        )
        assert isinstance(error, InvalidEntityStateError)
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)

    def test_basic_creation(self):
        """Test basic invalid experiment state error creation."""
        experiment_id = uuid4()
        error = InvalidExperimentStateError(
            experiment_id, "start", "experiment is already running"
        )
        expected_message = f"Cannot perform 'start' on Experiment {experiment_id}: experiment is already running"
        assert error.message == expected_message
        assert error.entity_type == "Experiment"
        assert error.entity_id == experiment_id
        assert error.operation == "start"
        assert error.reason == "experiment is already running"

    def test_different_operations(self):
        """Test with different operations."""
        experiment_id = uuid4()

        error1 = InvalidExperimentStateError(
            experiment_id, "stop", "experiment is not running"
        )
        assert error1.operation == "stop"
        assert "stop" in error1.message

        error2 = InvalidExperimentStateError(
            experiment_id, "resume", "experiment is completed"
        )
        assert error2.operation == "resume"
        assert "resume" in error2.message

    def test_string_representation(self):
        """Test string representation."""
        experiment_id = uuid4()
        error = InvalidExperimentStateError(
            experiment_id, "start", "experiment is already running"
        )
        result = str(error)
        assert (
            f"Cannot perform 'start' on Experiment {experiment_id}: experiment is already running"
            in result
        )

    def test_exception_raising(self):
        """Test that error can be raised."""
        experiment_id = uuid4()
        with pytest.raises(InvalidExperimentStateError):
            raise InvalidExperimentStateError(
                experiment_id, "start", "experiment is already running"
            )


class TestPipelineNotFoundError:
    """Test suite for PipelineNotFoundError."""

    def test_inheritance(self):
        """Test PipelineNotFoundError inheritance."""
        pipeline_id = uuid4()
        error = PipelineNotFoundError(pipeline_id)
        assert isinstance(error, EntityNotFoundError)
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)

    def test_basic_creation(self):
        """Test basic pipeline not found error creation."""
        pipeline_id = uuid4()
        error = PipelineNotFoundError(pipeline_id)
        expected_message = f"Pipeline with ID {pipeline_id} not found"
        assert error.message == expected_message
        assert error.entity_type == "Pipeline"
        assert error.entity_id == pipeline_id

    def test_creation_with_custom_message(self):
        """Test creation with custom message."""
        pipeline_id = uuid4()
        custom_message = "Custom pipeline not found message"
        error = PipelineNotFoundError(pipeline_id, custom_message)
        assert error.message == custom_message
        assert error.entity_type == "Pipeline"
        assert error.entity_id == pipeline_id

    def test_string_representation(self):
        """Test string representation."""
        pipeline_id = uuid4()
        error = PipelineNotFoundError(pipeline_id)
        result = str(error)
        assert f"Pipeline with ID {pipeline_id} not found" in result

    def test_exception_raising(self):
        """Test that error can be raised."""
        pipeline_id = uuid4()
        with pytest.raises(PipelineNotFoundError):
            raise PipelineNotFoundError(pipeline_id)


class TestInvalidPipelineStateError:
    """Test suite for InvalidPipelineStateError."""

    def test_inheritance(self):
        """Test InvalidPipelineStateError inheritance."""
        pipeline_id = uuid4()
        error = InvalidPipelineStateError(
            pipeline_id, "execute", "pipeline is not validated"
        )
        assert isinstance(error, InvalidEntityStateError)
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)

    def test_basic_creation(self):
        """Test basic invalid pipeline state error creation."""
        pipeline_id = uuid4()
        error = InvalidPipelineStateError(
            pipeline_id, "execute", "pipeline is not validated"
        )
        expected_message = f"Cannot perform 'execute' on Pipeline {pipeline_id}: pipeline is not validated"
        assert error.message == expected_message
        assert error.entity_type == "Pipeline"
        assert error.entity_id == pipeline_id
        assert error.operation == "execute"
        assert error.reason == "pipeline is not validated"

    def test_different_operations(self):
        """Test with different operations."""
        pipeline_id = uuid4()

        error1 = InvalidPipelineStateError(pipeline_id, "deploy", "pipeline has errors")
        assert error1.operation == "deploy"
        assert "deploy" in error1.message

        error2 = InvalidPipelineStateError(
            pipeline_id, "pause", "pipeline is not running"
        )
        assert error2.operation == "pause"
        assert "pause" in error2.message

    def test_string_representation(self):
        """Test string representation."""
        pipeline_id = uuid4()
        error = InvalidPipelineStateError(
            pipeline_id, "execute", "pipeline is not validated"
        )
        result = str(error)
        assert (
            f"Cannot perform 'execute' on Pipeline {pipeline_id}: pipeline is not validated"
            in result
        )

    def test_exception_raising(self):
        """Test that error can be raised."""
        pipeline_id = uuid4()
        with pytest.raises(InvalidPipelineStateError):
            raise InvalidPipelineStateError(
                pipeline_id, "execute", "pipeline is not validated"
            )


class TestAlertNotFoundError:
    """Test suite for AlertNotFoundError."""

    def test_inheritance(self):
        """Test AlertNotFoundError inheritance."""
        alert_id = uuid4()
        error = AlertNotFoundError(alert_id)
        assert isinstance(error, EntityNotFoundError)
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)

    def test_basic_creation(self):
        """Test basic alert not found error creation."""
        alert_id = uuid4()
        error = AlertNotFoundError(alert_id)
        expected_message = f"Alert with ID {alert_id} not found"
        assert error.message == expected_message
        assert error.entity_type == "Alert"
        assert error.entity_id == alert_id

    def test_creation_with_custom_message(self):
        """Test creation with custom message."""
        alert_id = uuid4()
        custom_message = "Custom alert not found message"
        error = AlertNotFoundError(alert_id, custom_message)
        assert error.message == custom_message
        assert error.entity_type == "Alert"
        assert error.entity_id == alert_id

    def test_string_representation(self):
        """Test string representation."""
        alert_id = uuid4()
        error = AlertNotFoundError(alert_id)
        result = str(error)
        assert f"Alert with ID {alert_id} not found" in result

    def test_exception_raising(self):
        """Test that error can be raised."""
        alert_id = uuid4()
        with pytest.raises(AlertNotFoundError):
            raise AlertNotFoundError(alert_id)


class TestInvalidAlertStateError:
    """Test suite for InvalidAlertStateError."""

    def test_inheritance(self):
        """Test InvalidAlertStateError inheritance."""
        alert_id = uuid4()
        error = InvalidAlertStateError(
            alert_id, "acknowledge", "alert is already acknowledged"
        )
        assert isinstance(error, InvalidEntityStateError)
        assert isinstance(error, DomainError)
        assert isinstance(error, PynamolyError)

    def test_basic_creation(self):
        """Test basic invalid alert state error creation."""
        alert_id = uuid4()
        error = InvalidAlertStateError(
            alert_id, "acknowledge", "alert is already acknowledged"
        )
        expected_message = f"Cannot perform 'acknowledge' on Alert {alert_id}: alert is already acknowledged"
        assert error.message == expected_message
        assert error.entity_type == "Alert"
        assert error.entity_id == alert_id
        assert error.operation == "acknowledge"
        assert error.reason == "alert is already acknowledged"

    def test_different_operations(self):
        """Test with different operations."""
        alert_id = uuid4()

        error1 = InvalidAlertStateError(alert_id, "dismiss", "alert is critical")
        assert error1.operation == "dismiss"
        assert "dismiss" in error1.message

        error2 = InvalidAlertStateError(
            alert_id, "escalate", "alert is already escalated"
        )
        assert error2.operation == "escalate"
        assert "escalate" in error2.message

    def test_string_representation(self):
        """Test string representation."""
        alert_id = uuid4()
        error = InvalidAlertStateError(
            alert_id, "acknowledge", "alert is already acknowledged"
        )
        result = str(error)
        assert (
            f"Cannot perform 'acknowledge' on Alert {alert_id}: alert is already acknowledged"
            in result
        )

    def test_exception_raising(self):
        """Test that error can be raised."""
        alert_id = uuid4()
        with pytest.raises(InvalidAlertStateError):
            raise InvalidAlertStateError(
                alert_id, "acknowledge", "alert is already acknowledged"
            )


class TestEntityExceptionIntegration:
    """Test integration scenarios for entity exceptions."""

    def test_exception_hierarchy(self):
        """Test entity exception hierarchy relationships."""
        entity_id = uuid4()

        # Base entity exceptions inherit from DomainError
        entity_not_found = EntityNotFoundError("TestEntity", entity_id)
        assert isinstance(entity_not_found, DomainError)
        assert isinstance(entity_not_found, PynamolyError)

        entity_state_error = InvalidEntityStateError(
            "TestEntity", entity_id, "update", "locked"
        )
        assert isinstance(entity_state_error, DomainError)
        assert isinstance(entity_state_error, PynamolyError)

        # Specific entity exceptions inherit from base entity exceptions
        model_not_found = ModelNotFoundError(entity_id)
        assert isinstance(model_not_found, EntityNotFoundError)
        assert isinstance(model_not_found, DomainError)
        assert isinstance(model_not_found, PynamolyError)

        model_state_error = InvalidModelStateError(entity_id, "train", "not configured")
        assert isinstance(model_state_error, InvalidEntityStateError)
        assert isinstance(model_state_error, DomainError)
        assert isinstance(model_state_error, PynamolyError)

    def test_context_building(self):
        """Test building context across entity exception handling."""
        entity_id = uuid4()
        error = ModelNotFoundError(entity_id)
        error.with_context(user_id="123", timestamp="2023-01-01")
        error.with_context(operation="model_training")

        assert error.details["user_id"] == "123"
        assert error.details["timestamp"] == "2023-01-01"
        assert error.details["operation"] == "model_training"

    def test_exception_chaining(self):
        """Test exception chaining with entity errors."""
        entity_id = uuid4()
        original_error = ValueError("Database connection failed")
        entity_error = ModelNotFoundError(entity_id, cause=original_error)

        assert entity_error.cause == original_error
        assert "Caused by: ValueError: Database connection failed" in str(entity_error)

    def test_complex_error_scenario(self):
        """Test complex error scenario with multiple details."""
        model_id = uuid4()
        error = InvalidModelStateError(model_id, "predict", "model is not fitted")
        error.with_context(
            dataset_name="test_data",
            model_version="v1.0",
            user_id="user123",
            timestamp="2023-01-01T12:00:00Z",
        )

        result = str(error)
        assert (
            f"Cannot perform 'predict' on Model {model_id}: model is not fitted"
            in result
        )
        assert "dataset_name=test_data" in result
        assert "model_version=v1.0" in result
        assert "user_id=user123" in result

    def test_nested_exception_handling(self):
        """Test nested exception handling with entity errors."""
        entity_id = uuid4()
        try:
            try:
                raise ValueError("Database error")
            except ValueError as e:
                raise ModelNotFoundError(entity_id, cause=e)
        except ModelNotFoundError as e:
            assert e.cause is not None
            assert isinstance(e.cause, ValueError)
            assert str(e.cause) == "Database error"

    def test_exception_in_try_catch(self):
        """Test entity exceptions in try-catch blocks."""
        experiment_id = uuid4()
        with pytest.raises(InvalidExperimentStateError) as exc_info:
            raise InvalidExperimentStateError(
                experiment_id, "start", "experiment is already running"
            )

        error = exc_info.value
        assert (
            f"Cannot perform 'start' on Experiment {experiment_id}: experiment is already running"
            in error.message
        )
        assert error.entity_type == "Experiment"
        assert error.entity_id == experiment_id
        assert error.operation == "start"
        assert error.reason == "experiment is already running"

    def test_uuid_consistency(self):
        """Test UUID consistency across different entity types."""
        entity_id = uuid4()

        # Test that the same UUID is preserved across different entity types
        model_error = ModelNotFoundError(entity_id)
        experiment_error = ExperimentNotFoundError(entity_id)
        pipeline_error = PipelineNotFoundError(entity_id)
        alert_error = AlertNotFoundError(entity_id)

        assert model_error.entity_id == entity_id
        assert experiment_error.entity_id == entity_id
        assert pipeline_error.entity_id == entity_id
        assert alert_error.entity_id == entity_id

        # Test that all are UUID instances
        assert isinstance(model_error.entity_id, UUID)
        assert isinstance(experiment_error.entity_id, UUID)
        assert isinstance(pipeline_error.entity_id, UUID)
        assert isinstance(alert_error.entity_id, UUID)

    def test_error_serialization_compatibility(self):
        """Test entity error compatibility with serialization."""
        model_id = uuid4()
        error = InvalidModelStateError(model_id, "train", "model is not configured")
        error.with_context(user_id="user123", dataset_name="training_data")

        # Should be able to extract serializable data
        error_data = {
            "message": error.message,
            "details": error.details,
            "type": type(error).__name__,
            "entity_type": error.entity_type,
            "entity_id": str(error.entity_id),  # UUID needs to be converted to string
            "operation": error.operation,
            "reason": error.reason,
        }

        assert (
            f"Cannot perform 'train' on Model {model_id}: model is not configured"
            in error_data["message"]
        )
        assert error_data["entity_type"] == "Model"
        assert error_data["entity_id"] == str(model_id)
        assert error_data["operation"] == "train"
        assert error_data["reason"] == "model is not configured"
        assert error_data["details"]["user_id"] == "user123"
        assert error_data["details"]["dataset_name"] == "training_data"
        assert error_data["type"] == "InvalidModelStateError"

    def test_error_message_construction(self):
        """Test error message construction in various scenarios."""
        entity_id = uuid4()

        # Test EntityNotFoundError message construction
        error1 = EntityNotFoundError("TestEntity", entity_id)
        assert f"TestEntity with ID {entity_id} not found" in error1.message

        error2 = EntityNotFoundError("TestEntity", entity_id, "Custom message")
        assert error2.message == "Custom message"

        # Test InvalidEntityStateError message construction
        error3 = InvalidEntityStateError("TestEntity", entity_id, "update", "locked")
        assert (
            f"Cannot perform 'update' on TestEntity {entity_id}: locked"
            in error3.message
        )

        # Test specific entity error message construction
        error4 = ModelNotFoundError(entity_id)
        assert f"Model with ID {entity_id} not found" in error4.message

        error5 = InvalidModelStateError(entity_id, "predict", "not fitted")
        assert (
            f"Cannot perform 'predict' on Model {entity_id}: not fitted"
            in error5.message
        )

    def test_entity_type_consistency(self):
        """Test entity type consistency across related exceptions."""
        entity_id = uuid4()

        # Test Model entity type consistency
        model_not_found = ModelNotFoundError(entity_id)
        model_state_error = InvalidModelStateError(entity_id, "train", "not configured")
        assert model_not_found.entity_type == "Model"
        assert model_state_error.entity_type == "Model"

        # Test Experiment entity type consistency
        experiment_not_found = ExperimentNotFoundError(entity_id)
        experiment_state_error = InvalidExperimentStateError(
            entity_id, "start", "running"
        )
        assert experiment_not_found.entity_type == "Experiment"
        assert experiment_state_error.entity_type == "Experiment"

        # Test Pipeline entity type consistency
        pipeline_not_found = PipelineNotFoundError(entity_id)
        pipeline_state_error = InvalidPipelineStateError(
            entity_id, "execute", "not validated"
        )
        assert pipeline_not_found.entity_type == "Pipeline"
        assert pipeline_state_error.entity_type == "Pipeline"

        # Test Alert entity type consistency
        alert_not_found = AlertNotFoundError(entity_id)
        alert_state_error = InvalidAlertStateError(
            entity_id, "acknowledge", "already acknowledged"
        )
        assert alert_not_found.entity_type == "Alert"
        assert alert_state_error.entity_type == "Alert"

    def test_entity_workflow_errors(self):
        """Test entity errors in typical workflow scenarios."""
        model_id = uuid4()
        experiment_id = uuid4()
        pipeline_id = uuid4()
        alert_id = uuid4()

        # Test model workflow errors
        with pytest.raises(ModelNotFoundError):
            raise ModelNotFoundError(model_id)

        with pytest.raises(InvalidModelStateError):
            raise InvalidModelStateError(model_id, "predict", "model is not fitted")

        # Test experiment workflow errors
        with pytest.raises(ExperimentNotFoundError):
            raise ExperimentNotFoundError(experiment_id)

        with pytest.raises(InvalidExperimentStateError):
            raise InvalidExperimentStateError(
                experiment_id, "start", "experiment is already running"
            )

        # Test pipeline workflow errors
        with pytest.raises(PipelineNotFoundError):
            raise PipelineNotFoundError(pipeline_id)

        with pytest.raises(InvalidPipelineStateError):
            raise InvalidPipelineStateError(
                pipeline_id, "execute", "pipeline is not validated"
            )

        # Test alert workflow errors
        with pytest.raises(AlertNotFoundError):
            raise AlertNotFoundError(alert_id)

        with pytest.raises(InvalidAlertStateError):
            raise InvalidAlertStateError(
                alert_id, "acknowledge", "alert is already acknowledged"
            )

    def test_error_context_propagation(self):
        """Test error context propagation through entity operations."""
        model_id = uuid4()

        # Create initial error with context
        error = InvalidModelStateError(model_id, "train", "model is not configured")
        error.with_context(
            user_id="user123",
            operation="batch_training",
            timestamp="2023-01-01T12:00:00Z",
        )

        # Add additional context
        error.with_context(
            dataset_name="training_data", model_version="v1.0", expected_samples=1000
        )

        # Verify all context is preserved
        assert error.details["user_id"] == "user123"
        assert error.details["operation"] == "batch_training"
        assert error.details["timestamp"] == "2023-01-01T12:00:00Z"
        assert error.details["dataset_name"] == "training_data"
        assert error.details["model_version"] == "v1.0"
        assert error.details["expected_samples"] == 1000

        # Test that context is properly formatted in string representation
        result = str(error)
        assert "user_id=user123" in result
        assert "dataset_name=training_data" in result
        assert "model_version=v1.0" in result
        assert "expected_samples=1000" in result

    def test_all_entity_types_coverage(self):
        """Test that all entity types are properly covered."""
        entity_id = uuid4()

        # Test all not found errors
        model_not_found = ModelNotFoundError(entity_id)
        experiment_not_found = ExperimentNotFoundError(entity_id)
        pipeline_not_found = PipelineNotFoundError(entity_id)
        alert_not_found = AlertNotFoundError(entity_id)

        assert model_not_found.entity_type == "Model"
        assert experiment_not_found.entity_type == "Experiment"
        assert pipeline_not_found.entity_type == "Pipeline"
        assert alert_not_found.entity_type == "Alert"

        # Test all invalid state errors
        model_state_error = InvalidModelStateError(entity_id, "train", "not configured")
        experiment_state_error = InvalidExperimentStateError(
            entity_id, "start", "running"
        )
        pipeline_state_error = InvalidPipelineStateError(
            entity_id, "execute", "not validated"
        )
        alert_state_error = InvalidAlertStateError(
            entity_id, "acknowledge", "already acknowledged"
        )

        assert model_state_error.entity_type == "Model"
        assert experiment_state_error.entity_type == "Experiment"
        assert pipeline_state_error.entity_type == "Pipeline"
        assert alert_state_error.entity_type == "Alert"
