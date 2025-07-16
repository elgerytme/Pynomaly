"""
Export Protocol for Pynomaly

Defines the interface for exporting anomaly detection results to various formats.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ...application.dto.export_options import ExportOptions
from ...domain.entities.detection_result import DetectionResult


class ExportProtocol(ABC):
    """Protocol for exporting anomaly detection results."""

    @abstractmethod
    def export_results(
        self,
        results: DetectionResult,
        file_path: str | Path,
        options: ExportOptions = None,
    ) -> dict[str, Any]:
        """
        Export anomaly detection results to a file.

        Args:
            results: Detection results to export
            file_path: Output file path
            options: Export configuration options

        Returns:
            Dictionary containing export metadata and statistics
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
        Validate if the file path is valid for this exporter.

        Args:
            file_path: File path to validate

        Returns:
            True if file can be exported to, False otherwise
        """
        pass
