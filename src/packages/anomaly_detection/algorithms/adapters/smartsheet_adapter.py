"""
Smartsheet adapter for exporting anomaly detection results to Smartsheet.

This adapter handles:
- Authentication with Smartsheet API
- Creating and updating sheets
- Project management workflows
- Automated alerts and notifications
- Collaboration features
- Dashboard integration
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import smartsheet

from ...shared.exceptions import AuthenticationError, ExportError
from ...shared.protocols.export_protocol import ExportProtocol


@dataclass
class SmartsheetConfig:
    """Configuration for Smartsheet integration."""

    # Authentication
    access_token: str

    # Sheet configuration
    sheet_id: int | None = None
    sheet_name: str = "Pynomaly Anomaly Results"
    workspace_id: int | None = None

    # Column configuration
    primary_column_title: str = "Timestamp"
    include_attachments: bool = False

    # Project management features
    enable_alerts: bool = True
    enable_automations: bool = True
    create_dashboard: bool = True

    # Sharing configuration
    share_with: list[str] | None = None
    share_level: str = "VIEWER"  # VIEWER, EDITOR, ADMIN

    # API configuration
    api_base_url: str = "https://api.smartsheet.com/2.0"


class SmartsheetAdapter(ExportProtocol):
    """Adapter for exporting data to Smartsheet."""

    def __init__(self, config: SmartsheetConfig):
        """Initialize Smartsheet adapter with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._client: smartsheet.Smartsheet | None = None

    def get_supported_formats(self) -> list[str]:
        """Return supported export formats."""
        return ["smartsheet", "smartsheets", "sheet"]

    async def export_results(
        self,
        data: pd.DataFrame | list[dict[str, Any]],
        format_type: str = "smartsheet",
        **kwargs,
    ) -> dict[str, Any]:
        """Export anomaly detection results to Smartsheet."""
        try:
            self.logger.info(f"Starting Smartsheet export with format: {format_type}")

            # Initialize Smartsheet client
            self._initialize_client()

            # Convert data to DataFrame if needed
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()

            # Prepare data for Smartsheet
            processed_data = self._prepare_data_for_smartsheet(df)

            # Get or create sheet
            sheet = await self._get_or_create_sheet(kwargs)

            # Export data to sheet
            await self._export_to_sheet(sheet, processed_data, **kwargs)

            # Set up project management features
            if self.config.enable_alerts:
                await self._setup_alerts(sheet, df)

            if self.config.enable_automations:
                await self._setup_automations(sheet)

            # Create dashboard if enabled
            if self.config.create_dashboard:
                dashboard = await self._create_dashboard(sheet, df)

            # Share sheet if configured
            if self.config.share_with:
                await self._share_sheet(sheet)

            result = {
                "export_type": "smartsheet",
                "sheet_id": sheet.id,
                "sheet_url": sheet.permalink,
                "sheet_name": sheet.name,
                "rows_exported": len(processed_data),
                "timestamp": datetime.now().isoformat(),
            }

            if self.config.create_dashboard:
                result["dashboard_id"] = dashboard.id if dashboard else None

            self.logger.info("Smartsheet export completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Smartsheet export failed: {str(e)}")
            raise ExportError(f"Failed to export to Smartsheet: {str(e)}") from e

    async def validate_file(self, file_path: str) -> bool:
        """Validate Smartsheet export configuration."""
        try:
            # Test authentication
            self._initialize_client()

            # Test API access
            user_info = self._client.Users.get_current_user()
            if not user_info.result:
                return False

            # Test sheet access if specified
            if self.config.sheet_id:
                return await self._validate_sheet_access()

            return True

        except Exception as e:
            self.logger.error(f"Smartsheet validation failed: {str(e)}")
            return False

    def _initialize_client(self) -> None:
        """Initialize Smartsheet client with authentication."""
        if self._client is not None:
            return

        try:
            self._client = smartsheet.Smartsheet(self.config.access_token)
            self._client.errors_as_exceptions(True)

            self.logger.info("Successfully authenticated with Smartsheet")

        except Exception as e:
            raise AuthenticationError(
                f"Failed to authenticate with Smartsheet: {str(e)}"
            ) from e

    def _prepare_data_for_smartsheet(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Prepare data for Smartsheet consumption."""
        # Convert timestamp columns to string format
        for col in df.columns:
            if df[col].dtype == "datetime64[ns]" or "time" in col.lower():
                df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d %H:%M:%S")

        # Replace NaN values with empty strings
        df = df.fillna("")

        # Convert to list of dictionaries
        records = df.to_dict("records")

        # Add metadata for project management
        for i, record in enumerate(records):
            record["row_id"] = i + 1
            record["export_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            record["source"] = "pynomaly"

            # Add status based on anomaly score
            if "anomaly_score" in record:
                score = (
                    float(record["anomaly_score"]) if record["anomaly_score"] else 0.0
                )
                if score > 0.8:
                    record["status"] = "High Risk"
                    record["priority"] = "High"
                elif score > 0.5:
                    record["status"] = "Medium Risk"
                    record["priority"] = "Medium"
                else:
                    record["status"] = "Normal"
                    record["priority"] = "Low"

        return records

    async def _get_or_create_sheet(
        self, kwargs: dict[str, Any]
    ) -> smartsheet.models.Sheet:
        """Get existing sheet or create a new one."""
        sheet_name = kwargs.get("sheet_name", self.config.sheet_name)

        if self.config.sheet_id:
            try:
                sheet = self._client.Sheets.get_sheet(self.config.sheet_id)
                self.logger.info(f"Using existing sheet: {sheet.name}")
                return sheet
            except Exception:
                self.logger.warning(
                    f"Sheet {self.config.sheet_id} not accessible, creating new one"
                )

        # Create new sheet
        column_specs = self._create_column_specs(kwargs.get("sample_data"))

        sheet_spec = smartsheet.models.Sheet()
        sheet_spec.name = sheet_name
        sheet_spec.columns = column_specs

        if self.config.workspace_id:
            result = self._client.Workspaces.create_sheet_in_workspace(
                self.config.workspace_id, sheet_spec
            )
        else:
            result = self._client.Sheets.create_sheet(sheet_spec)

        sheet = result.result
        self.logger.info(f"Created new sheet: {sheet.name}")

        return sheet

    def _create_column_specs(
        self, sample_data: dict[str, Any] | None = None
    ) -> list[smartsheet.models.Column]:
        """Create column specifications for the sheet."""
        columns = []

        # Primary column (timestamp)
        primary_column = smartsheet.models.Column()
        primary_column.title = self.config.primary_column_title
        primary_column.type = "TEXT_NUMBER"
        primary_column.primary = True
        columns.append(primary_column)

        # Standard anomaly detection columns
        standard_columns = [
            ("anomaly_score", "TEXT_NUMBER"),
            ("is_anomaly", "CHECKBOX"),
            ("status", "PICKLIST"),
            ("priority", "PICKLIST"),
            ("notes", "TEXT_NUMBER"),
            ("assigned_to", "CONTACT_LIST"),
            ("due_date", "DATE"),
            ("export_timestamp", "TEXT_NUMBER"),
            ("source", "TEXT_NUMBER"),
        ]

        for col_name, col_type in standard_columns:
            column = smartsheet.models.Column()
            column.title = col_name
            column.type = col_type

            # Set options for picklist columns
            if col_type == "PICKLIST":
                if col_name == "status":
                    column.options = [
                        "Normal",
                        "Medium Risk",
                        "High Risk",
                        "Investigating",
                        "Resolved",
                    ]
                elif col_name == "priority":
                    column.options = ["Low", "Medium", "High", "Critical"]

            columns.append(column)

        # Add dynamic columns from sample data
        if sample_data:
            for key, value in sample_data.items():
                if key not in [col.title for col in columns]:
                    column = smartsheet.models.Column()
                    column.title = key

                    # Determine column type based on value
                    if isinstance(value, bool):
                        column.type = "CHECKBOX"
                    elif isinstance(value, (int, float)):
                        column.type = "TEXT_NUMBER"
                    else:
                        column.type = "TEXT_NUMBER"

                    columns.append(column)

        return columns

    async def _export_to_sheet(
        self, sheet: smartsheet.models.Sheet, data: list[dict[str, Any]], **kwargs
    ) -> None:
        """Export data to the sheet."""
        if not data:
            return

        # Get column map
        column_map = {col.title: col.id for col in sheet.columns}

        # Clear existing data if specified
        if kwargs.get("clear_existing", False):
            await self._clear_sheet_data(sheet)

        # Create rows for batch insert
        rows_to_add = []

        for record in data:
            row = smartsheet.models.Row()
            row.to_top = True

            cells = []
            for key, value in record.items():
                if key in column_map:
                    cell = smartsheet.models.Cell()
                    cell.column_id = column_map[key]

                    # Handle different value types
                    if key == "is_anomaly":
                        cell.value = bool(value) if value else False
                    elif key in ["due_date"] and value:
                        # Format date for Smartsheet
                        cell.value = str(value)
                    else:
                        cell.value = str(value) if value is not None else ""

                    cells.append(cell)

            row.cells = cells
            rows_to_add.append(row)

            # Batch insert every 100 rows
            if len(rows_to_add) >= 100:
                self._client.Sheets.add_rows(sheet.id, rows_to_add)
                rows_to_add = []

        # Insert remaining rows
        if rows_to_add:
            self._client.Sheets.add_rows(sheet.id, rows_to_add)

        self.logger.info(f"Exported {len(data)} rows to sheet")

    async def _clear_sheet_data(self, sheet: smartsheet.models.Sheet) -> None:
        """Clear existing data from the sheet."""
        try:
            # Get all rows
            sheet_data = self._client.Sheets.get_sheet(sheet.id)

            if sheet_data.rows:
                row_ids = [row.id for row in sheet_data.rows]
                self._client.Sheets.delete_rows(sheet.id, row_ids)

        except Exception as e:
            self.logger.warning(f"Failed to clear sheet data: {str(e)}")

    async def _setup_alerts(
        self, sheet: smartsheet.models.Sheet, df: pd.DataFrame
    ) -> None:
        """Set up automated alerts for high-priority anomalies."""
        try:
            # Create alert rule for high anomaly scores
            if "anomaly_score" in df.columns:
                alert_rule = {
                    "name": "High Anomaly Alert",
                    "criteria": [
                        {
                            "columnId": next(
                                col.id
                                for col in sheet.columns
                                if col.title == "priority"
                            ),
                            "operator": "EQUAL",
                            "value": "High",
                        }
                    ],
                    "recipients": [
                        {"type": "CONTACT", "email": email}
                        for email in (self.config.share_with or [])
                    ],
                    "message": "High priority anomaly detected in Pynomaly results",
                    "frequency": "IMMEDIATELY",
                }

                # Note: Smartsheet API doesn't support creating alert rules programmatically
                # This would need to be done through the UI or webhooks
                self.logger.info("Alert configuration prepared (manual setup required)")

        except Exception as e:
            self.logger.warning(f"Failed to setup alerts: {str(e)}")

    async def _setup_automations(self, sheet: smartsheet.models.Sheet) -> None:
        """Set up automated workflows."""
        try:
            # Note: Smartsheet API has limited automation support
            # Most automations need to be configured through the UI

            # We can set up basic conditional formatting
            formatting_rules = [
                {
                    "criteria": {"priority": "High"},
                    "format": {"backgroundColor": "#FF6B6B"},
                },
                {
                    "criteria": {"priority": "Medium"},
                    "format": {"backgroundColor": "#FFE66D"},
                },
                {
                    "criteria": {"status": "Resolved"},
                    "format": {"backgroundColor": "#4ECDC4"},
                },
            ]

            self.logger.info("Automation configuration prepared")

        except Exception as e:
            self.logger.warning(f"Failed to setup automations: {str(e)}")

    async def _create_dashboard(
        self, sheet: smartsheet.models.Sheet, df: pd.DataFrame
    ) -> smartsheet.models.Dashboard | None:
        """Create a dashboard with anomaly metrics."""
        try:
            dashboard_spec = smartsheet.models.Dashboard()
            dashboard_spec.name = f"{sheet.name} Dashboard"

            # Create widgets for the dashboard
            widgets = []

            # Summary metrics widget
            if "anomaly_score" in df.columns:
                metrics_widget = smartsheet.models.Widget()
                metrics_widget.type = "METRIC"
                metrics_widget.title = "Anomaly Detection Summary"

                # Calculate metrics
                total_records = len(df)
                anomalies = (
                    len(df[df["anomaly_score"] > 0.5])
                    if "anomaly_score" in df.columns
                    else 0
                )
                anomaly_rate = (
                    (anomalies / total_records * 100) if total_records > 0 else 0
                )

                metrics_widget.contents = {
                    "metrics": [
                        {"label": "Total Records", "value": total_records},
                        {"label": "Anomalies Detected", "value": anomalies},
                        {"label": "Anomaly Rate %", "value": f"{anomaly_rate:.2f}"},
                    ]
                }

                widgets.append(metrics_widget)

            # Chart widget
            chart_widget = smartsheet.models.Widget()
            chart_widget.type = "CHART"
            chart_widget.title = "Anomaly Trends"
            chart_widget.contents = {
                "sheetId": sheet.id,
                "chartType": "LINE",
                "axes": [
                    {"title": "Time", "columnId": sheet.columns[0].id},
                    {
                        "title": "Anomaly Score",
                        "columnId": next(
                            col.id
                            for col in sheet.columns
                            if col.title == "anomaly_score"
                        ),
                    },
                ],
            }

            widgets.append(chart_widget)

            dashboard_spec.widgets = widgets

            # Create dashboard
            result = self._client.Dashboards.create_dashboard(dashboard_spec)
            dashboard = result.result

            self.logger.info(f"Created dashboard: {dashboard.name}")
            return dashboard

        except Exception as e:
            self.logger.warning(f"Failed to create dashboard: {str(e)}")
            return None

    async def _share_sheet(self, sheet: smartsheet.models.Sheet) -> None:
        """Share sheet with specified users."""
        try:
            for email in self.config.share_with:
                share_spec = smartsheet.models.Share()
                share_spec.email = email
                share_spec.access_level = self.config.share_level

                self._client.Sheets.share_sheet(sheet.id, [share_spec])
                self.logger.info(
                    f"Shared sheet with {email} as {self.config.share_level}"
                )

        except Exception as e:
            self.logger.warning(f"Failed to share sheet: {str(e)}")

    async def _validate_sheet_access(self) -> bool:
        """Validate access to the specified sheet."""
        try:
            sheet = self._client.Sheets.get_sheet(self.config.sheet_id)
            return sheet is not None
        except Exception:
            return False

    async def create_project_template(
        self, template_name: str = "Anomaly Investigation Template"
    ) -> smartsheet.models.Sheet | None:
        """Create a project template for anomaly investigation workflows."""
        try:
            # Define template columns
            template_columns = [
                ("Task", "TEXT_NUMBER", True),  # Primary column
                ("Assigned To", "CONTACT_LIST", False),
                ("Status", "PICKLIST", False),
                ("Priority", "PICKLIST", False),
                ("Due Date", "DATE", False),
                ("Progress", "TEXT_NUMBER", False),
                ("Notes", "TEXT_NUMBER", False),
                ("Anomaly Reference", "TEXT_NUMBER", False),
            ]

            columns = []
            for title, col_type, is_primary in template_columns:
                column = smartsheet.models.Column()
                column.title = title
                column.type = col_type
                column.primary = is_primary

                if col_type == "PICKLIST":
                    if title == "Status":
                        column.options = [
                            "Not Started",
                            "In Progress",
                            "Completed",
                            "Blocked",
                        ]
                    elif title == "Priority":
                        column.options = ["Low", "Medium", "High", "Critical"]

                columns.append(column)

            # Create template sheet
            template_spec = smartsheet.models.Sheet()
            template_spec.name = template_name
            template_spec.columns = columns

            result = self._client.Sheets.create_sheet(template_spec)
            template_sheet = result.result

            # Add template tasks
            template_tasks = [
                "Initial anomaly assessment",
                "Data validation and verification",
                "Root cause analysis",
                "Impact assessment",
                "Mitigation strategy development",
                "Implementation of fixes",
                "Testing and validation",
                "Documentation and reporting",
            ]

            rows_to_add = []
            for i, task in enumerate(template_tasks):
                row = smartsheet.models.Row()
                row.to_bottom = True

                cell = smartsheet.models.Cell()
                cell.column_id = columns[0].id  # Task column
                cell.value = task

                status_cell = smartsheet.models.Cell()
                status_cell.column_id = columns[2].id  # Status column
                status_cell.value = "Not Started"

                row.cells = [cell, status_cell]
                rows_to_add.append(row)

            if rows_to_add:
                self._client.Sheets.add_rows(template_sheet.id, rows_to_add)

            self.logger.info(f"Created project template: {template_name}")
            return template_sheet

        except Exception as e:
            self.logger.error(f"Failed to create project template: {str(e)}")
            return None
