"""Application use cases for software core."""

from .evaluate_model import (
    EvaluateModelRequest,
    EvaluateModelResponse,
    EvaluateModelUseCase,
)
from .generic_detection import (
    GenericDetectionRequest,
    GenericDetectionResponse,
    GenericDetectionUseCase,
)

__all__ = [
    "EvaluateModelUseCase",
    "EvaluateModelRequest",
    "EvaluateModelResponse",
    "GenericDetectionUseCase",
    "GenericDetectionRequest",
    "GenericDetectionResponse",
]
