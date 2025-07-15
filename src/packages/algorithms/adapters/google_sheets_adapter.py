"""
Google Sheets adapter for exporting anomaly detection results to Google Sheets.

This adapter handles:
- Authentication with Google Sheets API
- Creating and updating spreadsheets
- Worksheet management
- Data formatting and visualization
- Sharing and permissions
- Real-time collaboration features
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import gspread
import pandas as pd
from google.oauth2.credentials import Credentials
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from googleapiclient.discovery import build

from ...shared.exceptions import AuthenticationError, ExportError
from ...shared.protocols.export_protocol import ExportProtocol


@dataclass
class GoogleSheetsConfig:
    """Configuration for Google Sheets integration."""

    # Authentication options
    service_account_file: str | None = None
    service_account_info: dict[str, Any] | None = None
    oauth_credentials: str | None = None

    # Spreadsheet configuration
    spreadsheet_id: str | None = None
    spreadsheet_name: str = "Pynomaly Anomaly Results"
    worksheet_name: str = "Results"

    # Formatting options
    auto_resize: bool = True
    freeze_header: bool = True
    add_charts: bool = True
    conditional_formatting: bool = True

    # Sharing options
    share_with: list[str] | None = None
    share_type: str = "reader"  # reader, writer, owner

    # API configuration
    scopes: list[str] = None

    def __post_init__(self):
        if self.scopes is None:
            self.scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ]


class GoogleSheetsAdapter(ExportProtocol):
    """Adapter for exporting data to Google Sheets."""

    def __init__(self, config: GoogleSheetsConfig):
        """Initialize Google Sheets adapter with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._client: gspread.Client | None = None
        self._service = None

    def get_supported_formats(self) -> list[str]:
        """Return supported export formats."""
        return ["google_sheets", "gsheets", "spreadsheet"]

    async def export_results(
        self,
        data: pd.DataFrame | list[dict[str, Any]],
        format_type: str = "google_sheets",
        **kwargs,
    ) -> dict[str, Any]:
        """Export anomaly detection results to Google Sheets."""
        try:
            self.logger.info(
                f"Starting Google Sheets export with format: {format_type}"
            )

            # Initialize Google Sheets client
            await self._initialize_client()

            # Convert data to DataFrame if needed
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()

            # Prepare data for Google Sheets
            processed_data = self._prepare_data_for_sheets(df)

            # Get or create spreadsheet
            spreadsheet = await self._get_or_create_spreadsheet(kwargs)

            # Get or create worksheet
            worksheet = await self._get_or_create_worksheet(spreadsheet, kwargs)

            # Export data to worksheet
            await self._export_to_worksheet(worksheet, processed_data, **kwargs)

            # Apply formatting if enabled
            if self.config.conditional_formatting or self.config.add_charts:
                await self._apply_formatting(spreadsheet, worksheet, df, **kwargs)

            # Share spreadsheet if configured
            if self.config.share_with:
                await self._share_spreadsheet(spreadsheet)

            result = {
                "export_type": "google_sheets",
                "spreadsheet_id": spreadsheet.id,
                "spreadsheet_url": spreadsheet.url,
                "worksheet_name": worksheet.title,
                "rows_exported": len(processed_data),
                "timestamp": datetime.now().isoformat(),
            }

            self.logger.info("Google Sheets export completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Google Sheets export failed: {str(e)}")
            raise ExportError(f"Failed to export to Google Sheets: {str(e)}") from e

    async def validate_file(self, file_path: str) -> bool:
        """Validate Google Sheets export configuration."""
        try:
            # Test authentication
            await self._initialize_client()

            # Test spreadsheet access if specified
            if self.config.spreadsheet_id:
                await self._validate_spreadsheet_access()

            return True

        except Exception as e:
            self.logger.error(f"Google Sheets validation failed: {str(e)}")
            return False

    async def _initialize_client(self) -> None:
        """Initialize Google Sheets client with authentication."""
        if self._client is not None:
            return

        try:
            creds = None

            if self.config.service_account_file:
                creds = ServiceAccountCredentials.from_service_account_file(
                    self.config.service_account_file, scopes=self.config.scopes
                )
            elif self.config.service_account_info:
                creds = ServiceAccountCredentials.from_service_account_info(
                    self.config.service_account_info, scopes=self.config.scopes
                )
            elif self.config.oauth_credentials:
                # Load OAuth credentials from file
                with open(self.config.oauth_credentials) as f:
                    creds_data = json.load(f)
                creds = Credentials.from_authorized_user_info(
                    creds_data, self.config.scopes
                )
            else:
                raise AuthenticationError(
                    "No authentication method configured for Google Sheets"
                )

            self._client = gspread.authorize(creds)
            self._service = build("sheets", "v4", credentials=creds)

            self.logger.info("Successfully authenticated with Google Sheets")

        except Exception as e:
            raise AuthenticationError(
                f"Failed to authenticate with Google Sheets: {str(e)}"
            ) from e

    def _prepare_data_for_sheets(self, df: pd.DataFrame) -> list[list[Any]]:
        """Prepare data for Google Sheets consumption."""
        # Convert timestamp columns to string format
        for col in df.columns:
            if df[col].dtype == "datetime64[ns]" or "time" in col.lower():
                df[col] = df[col].astype(str)

        # Replace NaN values with empty strings
        df = df.fillna("")

        # Convert to list of lists (including headers)
        data = [df.columns.tolist()] + df.values.tolist()

        return data

    async def _get_or_create_spreadsheet(
        self, kwargs: dict[str, Any]
    ) -> gspread.Spreadsheet:
        """Get existing spreadsheet or create a new one."""
        spreadsheet_name = kwargs.get("spreadsheet_name", self.config.spreadsheet_name)

        if self.config.spreadsheet_id:
            try:
                spreadsheet = self._client.open_by_key(self.config.spreadsheet_id)
                self.logger.info(f"Opened existing spreadsheet: {spreadsheet.title}")
                return spreadsheet
            except gspread.SpreadsheetNotFound:
                self.logger.warning(
                    f"Spreadsheet {self.config.spreadsheet_id} not found, "
                    f"creating new one"
                )

        # Create new spreadsheet
        spreadsheet = self._client.create(spreadsheet_name)
        self.logger.info(f"Created new spreadsheet: {spreadsheet.title}")

        return spreadsheet

    async def _get_or_create_worksheet(
        self, spreadsheet: gspread.Spreadsheet, kwargs: dict[str, Any]
    ) -> gspread.Worksheet:
        """Get existing worksheet or create a new one."""
        worksheet_name = kwargs.get("worksheet_name", self.config.worksheet_name)

        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
            self.logger.info(f"Using existing worksheet: {worksheet_name}")
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(
                title=worksheet_name, rows="1000", cols="20"
            )
            self.logger.info(f"Created new worksheet: {worksheet_name}")

        return worksheet

    async def _export_to_worksheet(
        self, worksheet: gspread.Worksheet, data: list[list[Any]], **kwargs
    ) -> None:
        """Export data to the worksheet."""
        # Clear existing data if specified
        if kwargs.get("clear_existing", True):
            worksheet.clear()

        # Update worksheet with data
        if data:
            # Use batch update for better performance
            range_name = f"A1:{chr(65 + len(data[0]) - 1)}{len(data)}"
            worksheet.update(range_name, data)

            # Freeze header row if enabled
            if self.config.freeze_header and len(data) > 1:
                worksheet.freeze(rows=1)

            # Auto-resize columns if enabled
            if self.config.auto_resize:
                self._auto_resize_columns(worksheet, len(data[0]))

        self.logger.info(f"Exported {len(data)} rows to worksheet")

    def _auto_resize_columns(self, worksheet: gspread.Worksheet, num_cols: int) -> None:
        """Auto-resize columns to fit content."""
        try:
            requests = []
            for i in range(num_cols):
                requests.append(
                    {
                        "autoResizeDimensions": {
                            "dimensions": {
                                "sheetId": worksheet.id,
                                "dimension": "COLUMNS",
                                "startIndex": i,
                                "endIndex": i + 1,
                            }
                        }
                    }
                )

            if requests:
                body = {"requests": requests}
                self._service.spreadsheets().batchUpdate(
                    spreadsheetId=worksheet.spreadsheet.id, body=body
                ).execute()

        except Exception as e:
            self.logger.warning(f"Failed to auto-resize columns: {str(e)}")

    async def _apply_formatting(
        self,
        spreadsheet: gspread.Spreadsheet,
        worksheet: gspread.Worksheet,
        df: pd.DataFrame,
        **kwargs,
    ) -> None:
        """Apply conditional formatting and charts."""
        requests = []

        # Apply conditional formatting for anomaly scores
        if self.config.conditional_formatting and "anomaly_score" in df.columns:
            anomaly_col_idx = df.columns.get_loc("anomaly_score")

            # High anomaly scores in red
            requests.append(
                {
                    "addConditionalFormatRule": {
                        "rule": {
                            "ranges": [
                                {
                                    "sheetId": worksheet.id,
                                    "startRowIndex": 1,
                                    "endRowIndex": len(df) + 1,
                                    "startColumnIndex": anomaly_col_idx,
                                    "endColumnIndex": anomaly_col_idx + 1,
                                }
                            ],
                            "gradientRule": {
                                "minpoint": {
                                    "color": {"red": 1, "green": 1, "blue": 1},
                                    "type": "MIN",
                                },
                                "maxpoint": {
                                    "color": {"red": 1, "green": 0.4, "blue": 0.4},
                                    "type": "MAX",
                                },
                            },
                        },
                        "index": 0,
                    }
                }
            )

        # Format header row
        requests.append(
            {
                "repeatCell": {
                    "range": {
                        "sheetId": worksheet.id,
                        "startRowIndex": 0,
                        "endRowIndex": 1,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9},
                            "textFormat": {"bold": True},
                            "borders": {"bottom": {"style": "SOLID", "width": 2}},
                        }
                    },
                    "fields": "userEnteredFormat(backgroundColor,textFormat,borders)",
                }
            }
        )

        # Execute formatting requests
        if requests:
            try:
                body = {"requests": requests}
                self._service.spreadsheets().batchUpdate(
                    spreadsheetId=spreadsheet.id, body=body
                ).execute()

                self.logger.info("Applied formatting to spreadsheet")

            except Exception as e:
                self.logger.warning(f"Failed to apply formatting: {str(e)}")

        # Add charts if enabled
        if self.config.add_charts and "anomaly_score" in df.columns:
            await self._add_charts(spreadsheet, worksheet, df)

    async def _add_charts(
        self,
        spreadsheet: gspread.Spreadsheet,
        worksheet: gspread.Worksheet,
        df: pd.DataFrame,
    ) -> None:
        """Add charts to visualize anomaly data."""
        try:
            # Create anomaly score distribution chart
            chart_request = {
                "addChart": {
                    "chart": {
                        "spec": {
                            "title": "Anomaly Score Distribution",
                            "basicChart": {
                                "chartType": "HISTOGRAM",
                                "legendPosition": "BOTTOM_LEGEND",
                                "axis": [
                                    {
                                        "position": "BOTTOM_AXIS",
                                        "title": "Anomaly Score",
                                    },
                                    {"position": "LEFT_AXIS", "title": "Frequency"},
                                ],
                                "series": [
                                    {
                                        "series": {
                                            "sourceRange": {
                                                "sources": [
                                                    {
                                                        "sheetId": worksheet.id,
                                                        "startRowIndex": 1,
                                                        "endRowIndex": len(df) + 1,
                                                        "startColumnIndex": df.columns.get_loc(
                                                            "anomaly_score"
                                                        ),
                                                        "endColumnIndex": (
                                                            df.columns.get_loc("anomaly_score") + 1
                                                        ),
                                                    }
                                                ]
                                            }
                                        }
                                    }
                                ],
                            },
                        },
                        "position": {
                            "overlayPosition": {
                                "anchorCell": {
                                    "sheetId": worksheet.id,
                                    "rowIndex": len(df) + 3,
                                    "columnIndex": 0,
                                }
                            }
                        },
                    }
                }
            }

            body = {"requests": [chart_request]}
            self._service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet.id, body=body
            ).execute()

            self.logger.info("Added charts to spreadsheet")

        except Exception as e:
            self.logger.warning(f"Failed to add charts: {str(e)}")

    async def _share_spreadsheet(self, spreadsheet: gspread.Spreadsheet) -> None:
        """Share spreadsheet with specified users."""
        try:
            for email in self.config.share_with:
                spreadsheet.share(email, perm_type="user", role=self.config.share_type)
                self.logger.info(
                    f"Shared spreadsheet with {email} as {self.config.share_type}"
                )

        except Exception as e:
            self.logger.warning(f"Failed to share spreadsheet: {str(e)}")

    async def _validate_spreadsheet_access(self) -> bool:
        """Validate access to the specified spreadsheet."""
        try:
            spreadsheet = self._client.open_by_key(self.config.spreadsheet_id)
            return True
        except Exception:
            return False

    async def create_dashboard_sheet(
        self, spreadsheet: gspread.Spreadsheet, df: pd.DataFrame
    ) -> gspread.Worksheet:
        """Create a dashboard worksheet with summary statistics."""
        try:
            dashboard = spreadsheet.add_worksheet(
                title="Dashboard", rows="100", cols="10"
            )

            # Calculate summary statistics
            stats = []
            if "anomaly_score" in df.columns:
                stats.extend(
                    [
                        ["Metric", "Value"],
                        ["Total Records", len(df)],
                        ["Anomalies Detected", len(df[df["anomaly_score"] > 0.5])],
                        [
                            "Anomaly Rate",
                            f"{len(df[df['anomaly_score'] > 0.5]) / len(df) * 100:.2f}%",
                        ],
                        ["Average Anomaly Score", f"{df['anomaly_score'].mean():.4f}"],
                        ["Max Anomaly Score", f"{df['anomaly_score'].max():.4f}"],
                        ["Min Anomaly Score", f"{df['anomaly_score'].min():.4f}"],
                    ]
                )

            if stats:
                dashboard.update("A1:B7", stats)

                # Format dashboard
                dashboard.format(
                    "A1:B1",
                    {
                        "backgroundColor": {"red": 0.2, "green": 0.6, "blue": 0.9},
                        "textFormat": {
                            "bold": True,
                            "foregroundColor": {"red": 1, "green": 1, "blue": 1},
                        },
                    },
                )

            self.logger.info("Created dashboard worksheet")
            return dashboard

        except Exception as e:
            self.logger.warning(f"Failed to create dashboard: {str(e)}")
            return None
