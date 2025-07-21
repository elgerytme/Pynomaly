"""
Data Ingestion Package

A comprehensive, self-contained package for data ingestion and collection capabilities.
"""

__version__ = "0.1.0"

from .application.services.ingestion_service import DataIngestionService
from .domain.entities.ingestion_request import IngestionRequest
from .domain.entities.ingestion_result import IngestionResult
from .domain.value_objects.data_source import DataSource

__all__ = [
    "DataIngestionService",
    "IngestionRequest", 
    "IngestionResult",
    "DataSource",
    "__version__",
]