# Advanced Export Features - Issue #103 Implementation

This document describes the advanced export features implemented for Issue #103, providing business intelligence integrations, custom reporting, automated scheduling, and email delivery capabilities.

## Overview

The advanced export system extends Pynomaly's export capabilities with enterprise-grade features including:

- **Power BI Integration**: Stream data to Microsoft Power BI datasets and create real-time dashboards
- **Google Sheets Export**: Export to Google Sheets with formatting, charts, and collaboration features
- **Smartsheet Integration**: Project management workflows with automated alerts and dashboards
- **Custom Report Templates**: Configurable report layouts with dynamic sections and styling
- **Automated Reporting**: Cron-based scheduling for automated report generation and distribution
- **Email Delivery**: Template-based email notifications with attachments and rich formatting

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Advanced Export System                       │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐    │
│ │  Export Service │ │ Template Service│ │Scheduler Service│    │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐    │
│ │  PowerBI        │ │ Google Sheets   │ │  Smartsheet     │    │
│ │  Adapter        │ │ Adapter         │ │  Adapter        │    │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────┐                        │
│ │  Email Service  │ │ Notification    │                        │
│ │                 │ │ Service         │                        │
│ └─────────────────┘ └─────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

### Key Files

- **Adapters**:
  - `src/pynomaly/infrastructure/adapters/powerbi_adapter.py` - Power BI integration
  - `src/pynomaly/infrastructure/adapters/google_sheets_adapter.py` - Google Sheets export
  - `src/pynomaly/infrastructure/adapters/smartsheet_adapter.py` - Smartsheet integration

- **Services**:
  - `src/pynomaly/application/services/template_service.py` - Report templates
  - `src/pynomaly/application/services/scheduler_service.py` - Automated scheduling
  - `src/pynomaly/infrastructure/notifications/email_service.py` - Email delivery

- **CLI**:
  - `src/pynomaly/presentation/cli/export.py` - Enhanced CLI commands

## Power BI Integration

### Features

- Azure AD authentication with service principals
- Streaming datasets for real-time updates
- Workspace integration
- Automatic schema generation
- Batch data processing (handles 10MB limits)

### Configuration

```python
from pynomaly.infrastructure.adapters.powerbi_adapter import PowerBIConfig, PowerBIAdapter

config = PowerBIConfig(
    client_id="your-client-id",
    client_secret="your-client-secret", 
    tenant_id="your-tenant-id",
    workspace_id="workspace-id",  # Optional
    dataset_id="dataset-id"       # Optional
)

adapter = PowerBIAdapter(config)
```

### Environment Variables

```bash
export POWERBI_CLIENT_ID="your-client-id"
export POWERBI_CLIENT_SECRET="your-client-secret"
export POWERBI_TENANT_ID="your-tenant-id"
export POWERBI_WORKSPACE_ID="workspace-id"    # Optional
export POWERBI_DATASET_ID="dataset-id"        # Optional
```

### Usage

```bash
# CLI export to Power BI
pynomaly export powerbi results.json --workspace-id <id> --dataset-name "Anomaly Results"

# With streaming
pynomaly export powerbi results.json --streaming --workspace-id <id>
```

## Google Sheets Export

### Features

- Service account and OAuth authentication
- Automatic spreadsheet creation
- Conditional formatting for anomaly scores
- Chart generation (histograms, timelines)
- Real-time collaboration and sharing
- Auto-resizing and formatting

### Configuration

```python
from pynomaly.infrastructure.adapters.google_sheets_adapter import GoogleSheetsConfig, GoogleSheetsAdapter

config = GoogleSheetsConfig(
    service_account_file="path/to/credentials.json",
    spreadsheet_id="existing-sheet-id",  # Optional
    share_with=["user@example.com"],
    conditional_formatting=True,
    add_charts=True
)

adapter = GoogleSheetsAdapter(config)
```

### Environment Variables

```bash
export GOOGLE_SHEETS_SERVICE_ACCOUNT_FILE="path/to/credentials.json"
export GOOGLE_SHEETS_SPREADSHEET_ID="sheet-id"     # Optional
export GOOGLE_SHEETS_NAME="Pynomaly Results"       # Optional
```

### Usage

```bash
# CLI export to Google Sheets
pynomaly export gsheets results.json --credentials-file credentials.json

# With sharing
pynomaly export gsheets results.json --share-emails user1@example.com user2@example.com
```

## Smartsheet Integration

### Features

- API token authentication
- Project management workflows
- Automated alerts and notifications
- Dashboard creation with metrics
- Template-based sheet creation
- Collaboration and sharing

### Configuration

```python
from pynomaly.infrastructure.adapters.smartsheet_adapter import SmartsheetConfig, SmartsheetAdapter

config = SmartsheetConfig(
    access_token="your-api-token",
    sheet_id=123456789,  # Optional
    workspace_id=987654321,  # Optional
    enable_alerts=True,
    create_dashboard=True
)

adapter = SmartsheetAdapter(config)
```

### Environment Variables

```bash
export SMARTSHEET_ACCESS_TOKEN="your-api-token"
export SMARTSHEET_SHEET_ID="123456789"        # Optional
export SMARTSHEET_WORKSPACE_ID="987654321"    # Optional
export SMARTSHEET_NAME="Pynomaly Results"     # Optional
```

### Usage

```bash
# CLI export to Smartsheet
pynomaly export smartsheet results.json --access-token <token>

# With workspace
pynomaly export smartsheet results.json --workspace-name "Anomaly Detection"
```

## Custom Report Templates

### Features

- Jinja2-based templating engine
- Configurable layouts and sections
- Dynamic data filtering and transformations
- Multiple output formats (Excel, PDF, HTML, etc.)
- Styling and formatting options
- Template versioning and management

### Template Structure

```python
from pynomaly.application.services.template_service import TemplateService, TemplateConfig

template_config = TemplateConfig(
    template_id="custom-report",
    name="Custom Anomaly Report",
    description="Detailed anomaly analysis report",
    format_type="excel",
    sections=[
        {
            "type": "header",
            "title": "Anomaly Detection Report",
            "include_timestamp": True
        },
        {
            "type": "summary", 
            "title": "Executive Summary",
            "include_charts": True
        },
        {
            "type": "data_table",
            "title": "Anomaly Results",
            "columns": ["timestamp", "anomaly_score", "is_anomaly"],
            "sort_by": "anomaly_score",
            "sort_order": "desc"
        },
        {
            "type": "visualizations",
            "charts": [
                {"type": "histogram", "column": "anomaly_score"},
                {"type": "timeline", "x": "timestamp", "y": "anomaly_score"}
            ]
        }
    ],
    styling={
        "colors": {
            "primary": "#2E86AB",
            "anomaly_high": "#FF6B6B",
            "anomaly_medium": "#FFE66D"
        }
    }
)
```

### Usage

```python
service = TemplateService()

# Create template
template = service.create_template(
    name="Executive Report",
    description="High-level summary for executives", 
    format_type="pdf"
)

# Render with data
result = service.render_template(
    template_id=template.template_id,
    data=anomaly_results_df
)
```

## Automated Reporting and Scheduling

### Features

- Cron-based scheduling
- Multiple task types (export, report, notification)
- Data source integration
- Email notifications on success/failure
- Execution history and monitoring
- Real-time status tracking

### Schedule Configuration

```python
from pynomaly.application.services.scheduler_service import SchedulerService, ScheduleConfig

scheduler = SchedulerService(export_service, template_service, email_service)

# Create daily report schedule
schedule = scheduler.create_schedule(
    name="Daily Anomaly Report",
    cron_expression="0 9 * * *",  # Daily at 9 AM
    task_type="report",
    template_id="executive-summary",
    export_format="excel",
    destination={
        "type": "email",
        "recipients": ["admin@company.com"]
    },
    data_source={
        "type": "api", 
        "endpoint": "/api/v1/anomalies"
    }
)

# Start scheduler
await scheduler.start_scheduler()
```

### Common Schedule Patterns

```python
# Daily reports
"0 9 * * *"          # Daily at 9 AM
"0 18 * * MON-FRI"   # Weekdays at 6 PM

# Weekly reports  
"0 10 * * 1"         # Monday at 10 AM
"0 9 * * SUN"        # Sunday at 9 AM

# Hourly monitoring
"0 * * * *"          # Every hour
"*/15 * * * *"       # Every 15 minutes
```

## Email Delivery System

### Features

- SMTP integration with TLS/SSL support
- HTML and text email templates
- File attachments
- Bulk email sending with rate limiting
- Delivery tracking and statistics
- Template rendering with Jinja2

### Configuration

```python
from pynomaly.infrastructure.notifications.email_service import EmailConfig, EmailService

config = EmailConfig(
    smtp_server="smtp.gmail.com",
    smtp_port=587,
    username="your-email@gmail.com",
    password="your-password",
    from_email="noreply@yourcompany.com",
    from_name="Pynomaly System"
)

service = EmailService(config)
```

### Usage

```python
# Send report email
await service.send_report_email(
    recipients=["admin@company.com"],
    subject="Daily Anomaly Report",
    report_data={
        "timestamp": datetime.now(),
        "metrics": {...},
        "anomalies": [...]
    },
    attachments=["report.xlsx"]
)

# Send alert email
await service.send_alert_email(
    recipients=["alerts@company.com"],
    alert_data={
        "alert_type": "High Priority Anomaly",
        "timestamp": datetime.now(),
        "anomaly_score": 0.95,
        "details": "Critical system anomaly detected"
    },
    priority="high"
)
```

### Email Templates

Built-in templates include:

- `anomaly_report` - Detailed anomaly detection reports
- `anomaly_alert` - High-priority anomaly alerts
- `schedule_success` - Successful schedule execution notifications
- `schedule_failure` - Failed schedule execution notifications

## CLI Commands

### List Available Formats

```bash
pynomaly export list-formats
```

### Export Commands

```bash
# Excel export
pynomaly export excel results.json output.xlsx --include-charts --highlight-anomalies

# Power BI export
pynomaly export powerbi results.json --workspace-id <id> --dataset-name "Results"

# Google Sheets export  
pynomaly export gsheets results.json --credentials-file creds.json --include-charts

# Smartsheet export
pynomaly export smartsheet results.json --access-token <token> --workspace-name "Monitoring"

# Multi-format export
pynomaly export multi results.json --formats excel powerbi gsheets --output-dir ./exports

# Validate configuration
pynomaly export validate excel --output-file test.xlsx
```

## Installation and Dependencies

### Required Dependencies

```bash
# Power BI
pip install requests msal

# Google Sheets  
pip install gspread google-api-python-client google-auth-httplib2 google-auth-oauthlib

# Smartsheet
pip install smartsheet-python-sdk

# Email and scheduling
pip install aiosmtplib croniter jinja2

# Core dependencies (already included)
pip install pandas openpyxl xlsxwriter
```

### Optional Dependencies

```bash
# For PDF generation
pip install reportlab weasyprint

# For advanced charts
pip install plotly matplotlib seaborn
```

## Authentication Setup

### Power BI

1. Register application in Azure AD
2. Grant Power BI API permissions
3. Create client secret
4. Configure workspace access

### Google Sheets

1. Create Google Cloud project
2. Enable Sheets and Drive APIs
3. Create service account credentials
4. Download JSON key file

### Smartsheet

1. Create Smartsheet developer account
2. Generate API access token
3. Configure workspace permissions

## Configuration Examples

### Environment Configuration

```bash
# .env file
POWERBI_CLIENT_ID=your-client-id
POWERBI_CLIENT_SECRET=your-client-secret
POWERBI_TENANT_ID=your-tenant-id

GOOGLE_SHEETS_SERVICE_ACCOUNT_FILE=/path/to/credentials.json

SMARTSHEET_ACCESS_TOKEN=your-token

# Email configuration
SMTP_SERVER=smtp.gmail.com
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-password
```

### Multi-Format Configuration

```json
{
  "powerbi": {
    "workspace_id": "workspace-id",
    "dataset_name": "Pynomaly Results",
    "streaming": true
  },
  "google_sheets": {
    "spreadsheet_name": "Anomaly Detection Results",
    "share_with": ["team@company.com"],
    "include_charts": true
  },
  "smartsheet": {
    "workspace_name": "Monitoring", 
    "enable_alerts": true,
    "create_dashboard": true
  }
}
```

## Performance Considerations

### Batch Processing

- Power BI: 1000 rows per batch (10MB limit)
- Google Sheets: Batch updates for better performance
- Smartsheet: 100 rows per batch insert

### Rate Limiting

- Email: 100 emails per hour by default
- API calls: Automatic retry with exponential backoff
- Concurrent exports: Configurable batch sizes

### Memory Management

- Streaming data processing for large datasets
- Chunked file uploads
- Automatic cleanup of temporary files

## Monitoring and Troubleshooting

### Logging

All services provide comprehensive logging:

```python
import logging
logging.getLogger('pynomaly.infrastructure.adapters').setLevel(logging.DEBUG)
```

### Health Checks

```python
# Validate configurations
powerbi_adapter.validate_file("test")
gsheets_adapter.validate_file("test") 
smartsheet_adapter.validate_file("test")

# Check service status
scheduler.get_execution_history()
email_service.get_delivery_stats()
```

### Common Issues

1. **Authentication failures**: Check credentials and permissions
2. **Rate limiting**: Adjust batch sizes and delays
3. **Large datasets**: Use streaming and chunked processing
4. **Template errors**: Validate template syntax and data structure

## Security Best Practices

1. **Credentials Management**:
   - Use environment variables for sensitive data
   - Rotate API tokens regularly
   - Use service accounts with minimal permissions

2. **Data Protection**:
   - Encrypt credentials in transit and at rest
   - Implement data retention policies
   - Audit export activities

3. **Access Control**:
   - Limit workspace and sheet permissions
   - Use role-based access controls
   - Monitor export activities

## Migration Guide

### From Basic Export

```python
# Before (basic export)
export_service.export_results(results, "output.xlsx", ExportOptions())

# After (advanced export with template)
template_service.render_template("executive-summary", results)
scheduler.create_schedule("daily-report", "0 9 * * *", ...)
```

### Adding New Export Formats

1. Implement `ExportProtocol` interface
2. Add adapter to `ExportService._register_adapters()`
3. Update CLI commands
4. Add configuration methods

## API Reference

### ExportProtocol Interface

```python
class ExportProtocol:
    def get_supported_formats(self) -> List[str]: ...
    async def export_results(self, data, format_type, **kwargs) -> Dict[str, Any]: ...
    async def validate_file(self, file_path: str) -> bool: ...
```

### Template Configuration

```python
@dataclass
class TemplateConfig:
    template_id: str
    name: str
    description: str
    format_type: str
    sections: List[Dict[str, Any]]
    styling: Dict[str, Any]
    # ... additional fields
```

### Schedule Configuration  

```python
@dataclass
class ScheduleConfig:
    schedule_id: str
    name: str
    cron_expression: str
    task_type: str
    export_format: str
    destination: Dict[str, Any]
    # ... additional fields
```

## Examples and Use Cases

### Executive Dashboard

```python
# Create real-time Power BI dashboard
powerbi_config = PowerBIConfig(...)
adapter = PowerBIAdapter(powerbi_config)

# Stream daily anomaly data
await adapter.export_results(
    data=daily_anomalies,
    format_type="powerbi_streaming"
)
```

### Collaborative Analysis

```python
# Export to Google Sheets for team collaboration
gsheets_config = GoogleSheetsConfig(
    share_with=["analyst1@company.com", "analyst2@company.com"],
    conditional_formatting=True,
    add_charts=True
)

adapter = GoogleSheetsAdapter(gsheets_config)
await adapter.export_results(anomaly_data)
```

### Project Management

```python
# Create Smartsheet for anomaly investigation
smartsheet_config = SmartsheetConfig(
    enable_alerts=True,
    create_dashboard=True,
    share_with=["investigator@company.com"]
)

adapter = SmartsheetAdapter(smartsheet_config)
await adapter.export_results(high_priority_anomalies)
```

This completes the implementation of Issue #103 - Advanced Export Formats. The system now provides comprehensive business intelligence integrations, automated reporting, and email delivery capabilities as specified in the acceptance criteria.
