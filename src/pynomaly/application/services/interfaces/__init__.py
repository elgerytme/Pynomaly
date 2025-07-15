"""Service interfaces for dependency injection."""

from .detection_service_interface import DetectionServiceInterface
from .data_service_interface import DataServiceInterface

__all__ = [
    "DetectionServiceInterface",
    "DataServiceInterface", 
]