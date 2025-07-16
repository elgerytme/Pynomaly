"""
Export Service for Pynomaly

Central service for exporting anomaly detection results to various formats.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ...domain.entities.detection_result import DetectionResult
from ...infrastructure.adapters.excel_adapter import ExcelAdapter
from ...shared.protocols.export_protocol import ExportProtocol
from ..dto.export_options import ExportFormat, ExportOptions

logger = logging.getLogger(__name__)


class ExportService:
    """
    Service for exporting anomaly detection results to various formats.

    This service acts as a facade for different export adapters and handles
    the routing of export requests to the appropriate adapter based on the
    requested format.
    """

    def __init__(self):
        """Initialize the export service with available adapters."""
        self._adapters: dict[ExportFormat, ExportProtocol] = {}
        self._register_adapters()

    def _register_adapters(self) -> None:
        """Register available export adapters."""
        try:
            excel_adapter = ExcelAdapter()
            self._adapters[ExportFormat.EXCEL] = excel_adapter
            logger.info("Registered Excel adapter")
        except ImportError as e:
            logger.warning(f"Excel adapter not available: {e}")

        # Basic CSV and JSON adapters (built-in)
        logger.info("Basic export formats (CSV, JSON) available")

    def export_results(
        self,
        results: DetectionResult,
        file_path: str | Path,
        options: ExportOptions | None = None,
    ) -> dict[str, Any]:
        """
        Export anomaly detection results to the specified format.

        Args:
            results: Detection results to export
            file_path: Output file path
            options: Export configuration options

        Returns:
            Dictionary containing export metadata and statistics

        Raises:
            ValueError: If the requested format is not supported
            RuntimeError: If export fails
        """
        if options is None:
            options = ExportOptions()

        # Get the appropriate adapter
        adapter = self._get_adapter(options.format)

        # Validate file path
        file_path = Path(file_path)
        if not adapter.validate_file(file_path):
            raise ValueError(
                f"Invalid file path for {options.format.value} export: {file_path}"
            )

        try:
            # Perform the export
            logger.info(
                f"Exporting results to {options.format.value} format: {file_path}"
            )
            export_result = adapter.export_results(results, file_path, options)

            # Add service-level metadata
            export_result.update(
                {
                    "service": "ExportService",
                    "format": options.format.value,
                    "destination": options.destination.value,
                    "service_export_time": datetime.now().isoformat(),
                }
            )

            logger.info(f"Successfully exported results to {file_path}")
            return export_result

        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            raise RuntimeError(f"Export failed: {e}")

    def _get_adapter(self, format: ExportFormat) -> ExportProtocol:
        """
        Get the appropriate adapter for the specified format.

        Args:
            format: The export format

        Returns:
            The export adapter

        Raises:
            ValueError: If format is not supported
        """
        if format not in self._adapters:
            available_formats = list(self._adapters.keys())
            raise ValueError(
                f"Export format '{format.value}' is not supported. "
                f"Available formats: {[f.value for f in available_formats]}"
            )

        return self._adapters[format]

    def get_supported_formats(self) -> list[ExportFormat]:
        """
        Get list of supported export formats.

        Returns:
            List of supported export formats
        """
        return list(self._adapters.keys())

    def get_supported_file_extensions(self, format: ExportFormat) -> list[str]:
        """
        Get supported file extensions for a specific format.

        Args:
            format: The export format

        Returns:
            List of supported file extensions

        Raises:
            ValueError: If format is not supported
        """
        adapter = self._get_adapter(format)
        return adapter.get_supported_formats()

    def validate_export_request(
        self,
        format: ExportFormat,
        file_path: str | Path,
        options: ExportOptions | None = None,
    ) -> dict[str, Any]:
        """
        Validate an export request without performing the actual export.

        Args:
            format: The export format
            file_path: Output file path
            options: Export configuration options

        Returns:
            Dictionary containing validation results
        """
        validation_result = {
            "valid": True,
            "format": format.value,
            "file_path": str(file_path),
            "errors": [],
            "warnings": [],
        }

        try:
            # Check if format is supported
            adapter = self._get_adapter(format)

            # Validate file path
            file_path = Path(file_path)
            if not adapter.validate_file(file_path):
                validation_result["valid"] = False
                validation_result["errors"].append(
                    f"Invalid file path for {format.value} export"
                )

            # Check file extension
            supported_extensions = adapter.get_supported_formats()
            if file_path.suffix.lower() not in supported_extensions:
                validation_result["warnings"].append(
                    f"File extension '{file_path.suffix}' may not be optimal for {format.value}. "
                    f"Recommended: {supported_extensions}"
                )

            # Validate options if provided
            if options:
                if options.format != format:
                    validation_result["warnings"].append(
                        f"Format mismatch: request={format.value}, options={options.format.value}"
                    )

        except ValueError as e:
            validation_result["valid"] = False
            validation_result["errors"].append(str(e))
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {e}")

        return validation_result

    def create_export_options(self, format: ExportFormat, **kwargs) -> ExportOptions:
        """
        Create optimized export options for a specific format.

        Args:
            format: The export format
            **kwargs: Additional options

        Returns:
            Optimized export options
        """
        options = ExportOptions(**kwargs)

        # Apply format-specific optimizations
        if format == ExportFormat.EXCEL:
            return options.for_excel()
        elif format == ExportFormat.CSV:
            return options.for_csv()
        elif format == ExportFormat.JSON:
            return options.for_json()
        elif format == ExportFormat.PARQUET:
            return options.for_parquet()

        return options

    def get_export_statistics(self) -> dict[str, Any]:
        """
        Get statistics about available export capabilities.

        Returns:
            Dictionary containing export statistics
        """
        stats = {
            "total_formats": len(self._adapters),
            "supported_formats": [f.value for f in self._adapters.keys()],
            "adapters": {},
        }

        for format, adapter in self._adapters.items():
            stats["adapters"][format.value] = {
                "class": adapter.__class__.__name__,
                "supported_extensions": adapter.get_supported_formats(),
            }

        return stats

    def export_multiple_formats(
        self,
        results: DetectionResult,
        base_path: str | Path,
        formats: list[ExportFormat],
        options_map: dict[ExportFormat, ExportOptions] | None = None,
    ) -> dict[ExportFormat, dict[str, Any]]:
        """
        Export results to multiple formats simultaneously.

        Args:
            results: Detection results to export
            base_path: Base path for output files (format-specific extensions will be added)
            formats: List of formats to export to
            options_map: Optional mapping of format-specific options

        Returns:
            Dictionary mapping formats to their export results
        """
        base_path = Path(base_path)
        options_map = options_map or {}
        export_results = {}

        for format in formats:
            try:
                # Create format-specific file path
                adapter = self._get_adapter(format)
                extensions = adapter.get_supported_formats()
                extension = extensions[0] if extensions else ".out"

                file_path = base_path.with_suffix(extension)
                if format != ExportFormat.EXCEL:
                    # Add format identifier to filename for non-Excel formats
                    file_path = base_path.with_name(
                        f"{base_path.stem}_{format.value}{extension}"
                    )

                # Get options for this format
                options = options_map.get(format, self.create_export_options(format))

                # Export
                result = self.export_results(results, file_path, options)
                export_results[format] = result

            except Exception as e:
                logger.error(f"Failed to export to {format.value}: {e}")
                export_results[format] = {"error": str(e), "success": False}

        return export_results
