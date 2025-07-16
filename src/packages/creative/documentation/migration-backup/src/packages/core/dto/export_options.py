"""
Export Options DTO for Pynomaly

Data Transfer Object for export configuration options.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExportFormat(Enum):
    """Supported export formats (core only)."""

    EXCEL = "excel"
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"


class ExportDestination(Enum):
    """Export destination types."""

    LOCAL_FILE = "local_file"
    CLOUD_STORAGE = "cloud_storage"
    API_ENDPOINT = "api_endpoint"
    EMAIL = "email"


@dataclass
class ExportOptions:
    """
    Configuration options for exporting anomaly detection results.
    """

    # Basic export settings
    format: ExportFormat = ExportFormat.EXCEL
    destination: ExportDestination = ExportDestination.LOCAL_FILE
    include_charts: bool = True
    include_summary: bool = True
    include_metadata: bool = True
    use_advanced_formatting: bool = True

    # Excel-specific options
    create_multiple_sheets: bool = True
    highlight_anomalies: bool = True
    add_conditional_formatting: bool = True
    include_formulas: bool = False
    sheet_names: list[str] | None = None

    # Additional format options (simplified)
    parquet_compression: str = "snappy"
    json_indent: int | None = 2

    # Data filtering and selection
    include_normal_samples: bool = True
    include_anomaly_samples: bool = True
    max_samples: int | None = None
    sample_columns: list[str] | None = None

    # Visualization options
    chart_types: list[str] = field(default_factory=lambda: ["scatter", "histogram"])
    color_scheme: str = "default"
    chart_size: tuple[int, int] = (640, 480)

    # Authentication and security
    credentials: dict[str, Any] | None = None
    api_key: str | None = None
    oauth_token: str | None = None

    # Performance options
    batch_size: int = 1000
    compression: bool = False
    parallel_export: bool = False

    # Notification options
    notify_on_completion: bool = False
    notification_emails: list[str] | None = None
    webhook_url: str | None = None

    # Custom options for extensibility
    custom_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate export options after initialization."""
        # Set default sheet names if not provided
        if self.sheet_names is None:
            self.sheet_names = ["Results", "Summary", "Charts", "Metadata"]

        # Validate chart types
        valid_charts = ["scatter", "histogram", "line", "bar", "pie"]
        self.chart_types = [ct for ct in self.chart_types if ct in valid_charts]

        # Validate parquet compression
        valid_compression = ["snappy", "gzip", "brotli", "lz4", "zstd"]
        if self.parquet_compression not in valid_compression:
            self.parquet_compression = "snappy"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExportOptions":
        """Create ExportOptions from dictionary."""
        # Handle enum conversions
        if "format" in data and isinstance(data["format"], str):
            data["format"] = ExportFormat(data["format"])
        if "destination" in data and isinstance(data["destination"], str):
            data["destination"] = ExportDestination(data["destination"])

        return cls(**data)

    def for_excel(self) -> "ExportOptions":
        """Create Excel-optimized export options."""
        self.format = ExportFormat.EXCEL
        self.use_advanced_formatting = True
        self.highlight_anomalies = True
        self.add_conditional_formatting = True
        self.include_charts = True
        return self

    def for_csv(self) -> "ExportOptions":
        """Create CSV-optimized export options."""
        self.format = ExportFormat.CSV
        self.include_charts = False
        self.use_advanced_formatting = False
        return self

    def for_json(self) -> "ExportOptions":
        """Create JSON-optimized export options."""
        self.format = ExportFormat.JSON
        self.include_charts = False
        self.use_advanced_formatting = False
        self.json_indent = 2
        return self

    def for_parquet(self) -> "ExportOptions":
        """Create Parquet-optimized export options."""
        self.format = ExportFormat.PARQUET
        self.include_charts = False
        self.use_advanced_formatting = False
        self.parquet_compression = "snappy"
        return self
