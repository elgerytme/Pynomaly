"""
Import Protocol for Software

Defines the interface for importing datasets from various sources.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ...domain.entities.dataset import Dataset


class ImportProtocol(ABC):
    """Protocol for importing datasets from various sources."""

    @abstractmethod
    def import_dataset(
        self, file_path: str | Path, options: dict[str, Any] = None
    ) -> DataCollection:
        """
        Import data_collection from a file.

        Args:
            file_path: Path to the file to import
            options: Import configuration options

        Returns:
            DataCollection object containing the imported data
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> list[str]:
        """
        Return list of supported file formats.

        Returns:
            List of supported file extensions (e.g., ['.xlsx', '.csv'])
        """
        pass

    @abstractmethod
    def validate_file(self, file_path: str | Path) -> bool:
        """
        Validate if the file can be imported by this adapter.

        Args:
            file_path: File path to validate

        Returns:
            True if file can be imported, False otherwise
        """
        pass
