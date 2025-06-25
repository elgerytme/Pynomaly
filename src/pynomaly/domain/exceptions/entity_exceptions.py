"""Entity-specific exceptions for domain entities."""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from .base import DomainError


class EntityNotFoundError(DomainError):
    """Base exception for entity not found errors."""
    
    def __init__(self, entity_type: str, entity_id: UUID, message: Optional[str] = None):
        self.entity_type = entity_type
        self.entity_id = entity_id
        
        if message is None:
            message = f"{entity_type} with ID {entity_id} not found"
        
        super().__init__(message)


class InvalidEntityStateError(DomainError):
    """Base exception for invalid entity state errors."""
    
    def __init__(self, entity_type: str, entity_id: UUID, operation: str, reason: str):
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.operation = operation
        self.reason = reason
        
        message = f"Cannot perform '{operation}' on {entity_type} {entity_id}: {reason}"
        super().__init__(message)


# Model-specific exceptions
class ModelNotFoundError(EntityNotFoundError):
    """Raised when a model is not found."""
    
    def __init__(self, model_id: UUID, message: Optional[str] = None):
        super().__init__("Model", model_id, message)


class InvalidModelStateError(InvalidEntityStateError):
    """Raised when trying to perform an invalid operation on a model."""
    
    def __init__(self, model_id: UUID, operation: str, reason: str):
        super().__init__("Model", model_id, operation, reason)


# Experiment-specific exceptions
class ExperimentNotFoundError(EntityNotFoundError):
    """Raised when an experiment is not found."""
    
    def __init__(self, experiment_id: UUID, message: Optional[str] = None):
        super().__init__("Experiment", experiment_id, message)


class InvalidExperimentStateError(InvalidEntityStateError):
    """Raised when trying to perform an invalid operation on an experiment."""
    
    def __init__(self, experiment_id: UUID, operation: str, reason: str):
        super().__init__("Experiment", experiment_id, operation, reason)


# Pipeline-specific exceptions
class PipelineNotFoundError(EntityNotFoundError):
    """Raised when a pipeline is not found."""
    
    def __init__(self, pipeline_id: UUID, message: Optional[str] = None):
        super().__init__("Pipeline", pipeline_id, message)


class InvalidPipelineStateError(InvalidEntityStateError):
    """Raised when trying to perform an invalid operation on a pipeline."""
    
    def __init__(self, pipeline_id: UUID, operation: str, reason: str):
        super().__init__("Pipeline", pipeline_id, operation, reason)


# Alert-specific exceptions
class AlertNotFoundError(EntityNotFoundError):
    """Raised when an alert is not found."""
    
    def __init__(self, alert_id: UUID, message: Optional[str] = None):
        super().__init__("Alert", alert_id, message)


class InvalidAlertStateError(InvalidEntityStateError):
    """Raised when trying to perform an invalid operation on an alert."""
    
    def __init__(self, alert_id: UUID, operation: str, reason: str):
        super().__init__("Alert", alert_id, operation, reason)