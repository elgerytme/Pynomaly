"""
Export Options DTO for Pynomaly

Data Transfer Object for export configuration options.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class ExportFormat(Enum):
    """Supported export formats."""
    EXCEL = "excel"
    POWERBI = "powerbi"
    GSHEETS = "gsheets"
    SMARTSHEET = "smartsheet"
    CSV = "csv"
    JSON = "json"


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
    sheet_names: Optional[List[str]] = None
    
    # Power BI options
    workspace_id: Optional[str] = None
    dataset_name: Optional[str] = None
    table_name: Optional[str] = None
    refresh_schedule: Optional[str] = None
    streaming_dataset: bool = False
    
    # Google Sheets options
    spreadsheet_id: Optional[str] = None
    sheet_id: Optional[str] = None
    share_with_emails: Optional[List[str]] = None
    permissions: str = "view"  # view, edit, comment
    
    # Smartsheet options
    sheet_template_id: Optional[str] = None
    folder_id: Optional[str] = None
    workspace_name: Optional[str] = None
    
    # Data filtering and selection
    include_normal_samples: bool = True
    include_anomaly_samples: bool = True
    max_samples: Optional[int] = None
    sample_columns: Optional[List[str]] = None
    
    # Visualization options
    chart_types: List[str] = field(default_factory=lambda: ["scatter", "histogram"])
    color_scheme: str = "default"
    chart_size: tuple = (640, 480)
    
    # Authentication and security
    credentials: Optional[Dict[str, Any]] = None
    api_key: Optional[str] = None
    oauth_token: Optional[str] = None
    
    # Performance options
    batch_size: int = 1000
    compression: bool = False
    parallel_export: bool = False
    
    # Notification options
    notify_on_completion: bool = False
    notification_emails: Optional[List[str]] = None
    webhook_url: Optional[str] = None
    
    # Custom options for extensibility
    custom_options: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate export options after initialization."""
        # Set default sheet names if not provided
        if self.sheet_names is None:
            self.sheet_names = ["Results", "Summary", "Charts", "Metadata"]
        
        # Validate chart types
        valid_charts = ["scatter", "histogram", "line", "bar", "pie"]
        self.chart_types = [ct for ct in self.chart_types if ct in valid_charts]
        
        # Validate permissions
        if self.permissions not in ["view", "edit", "comment"]:
            self.permissions = "view"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExportOptions':
        """Create ExportOptions from dictionary."""
        # Handle enum conversions
        if 'format' in data and isinstance(data['format'], str):
            data['format'] = ExportFormat(data['format'])
        if 'destination' in data and isinstance(data['destination'], str):
            data['destination'] = ExportDestination(data['destination'])
        
        return cls(**data)
    
    def for_excel(self) -> 'ExportOptions':
        """Create Excel-optimized export options."""
        self.format = ExportFormat.EXCEL
        self.use_advanced_formatting = True
        self.highlight_anomalies = True
        self.add_conditional_formatting = True
        self.include_charts = True
        return self
    
    def for_powerbi(self, workspace_id: str, dataset_name: str) -> 'ExportOptions':
        """Create Power BI-optimized export options."""
        self.format = ExportFormat.POWERBI
        self.workspace_id = workspace_id
        self.dataset_name = dataset_name
        self.destination = ExportDestination.API_ENDPOINT
        return self
    
    def for_gsheets(self, spreadsheet_id: Optional[str] = None) -> 'ExportOptions':
        """Create Google Sheets-optimized export options."""
        self.format = ExportFormat.GSHEETS
        self.spreadsheet_id = spreadsheet_id
        self.destination = ExportDestination.CLOUD_STORAGE
        self.include_charts = True
        return self
    
    def for_smartsheet(self, workspace_name: Optional[str] = None) -> 'ExportOptions':
        """Create Smartsheet-optimized export options."""
        self.format = ExportFormat.SMARTSHEET
        self.workspace_name = workspace_name
        self.destination = ExportDestination.API_ENDPOINT
        self.include_charts = False  # Smartsheet has limited chart support
        return self