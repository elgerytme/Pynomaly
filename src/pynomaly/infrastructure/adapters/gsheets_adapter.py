"""
Google Sheets Integration Adapter for Pynomaly

This module provides Google Sheets integration capabilities for creating
and updating spreadsheets with anomaly detection results.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import json

try:
    from googleapiclient.discovery import build
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    GSHEETS_AVAILABLE = True
except ImportError:
    GSHEETS_AVAILABLE = False

from ...domain.entities.detection_result import DetectionResult
from ...shared.protocols.export_protocol import ExportProtocol
from ...application.dto.export_options import ExportOptions


logger = logging.getLogger(__name__)


class GoogleSheetsAdapter(ExportProtocol):
    """
    Google Sheets adapter for creating and updating spreadsheets with anomaly detection results.
    
    Supports:
    - Service account authentication
    - OAuth 2.0 authentication
    - Spreadsheet creation and updates
    - Real-time collaborative features
    - Charts and formatting
    """
    
    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    
    def __init__(
        self,
        credentials_path: Optional[str] = None,
        service_account_path: Optional[str] = None,
        credentials_json: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Google Sheets adapter.
        
        Args:
            credentials_path: Path to OAuth2 credentials JSON file
            service_account_path: Path to service account JSON file
            credentials_json: Service account credentials as dictionary
        """
        if not GSHEETS_AVAILABLE:
            raise ImportError(
                "Google Sheets adapter requires Google API libraries. "
                "Install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
            )
        
        self.credentials_path = credentials_path
        self.service_account_path = service_account_path
        self.credentials_json = credentials_json
        
        self._service = None
        self._drive_service = None
        
        logger.info("Google Sheets adapter initialized")
    
    def _authenticate(self):
        """Authenticate with Google APIs."""
        creds = None
        
        try:
            if self.service_account_path:
                # Use service account authentication
                creds = ServiceAccountCredentials.from_service_account_file(
                    self.service_account_path, scopes=self.SCOPES
                )
                logger.info("Authenticated with service account")
                
            elif self.credentials_json:
                # Use service account from JSON dict
                creds = ServiceAccountCredentials.from_service_account_info(
                    self.credentials_json, scopes=self.SCOPES
                )
                logger.info("Authenticated with service account from JSON")
                
            elif self.credentials_path:
                # Use OAuth2 flow
                if Path(self.credentials_path).exists():
                    creds = Credentials.from_authorized_user_file(self.credentials_path, self.SCOPES)
                
                if not creds or not creds.valid:
                    if creds and creds.expired and creds.refresh_token:
                        creds.refresh(Request())
                    else:
                        flow = InstalledAppFlow.from_client_secrets_file(
                            self.credentials_path, self.SCOPES
                        )
                        creds = flow.run_local_server(port=0)
                    
                    # Save credentials for future use
                    with open(self.credentials_path.replace('.json', '_token.json'), 'w') as token:
                        token.write(creds.to_json())
                
                logger.info("Authenticated with OAuth2")
            else:
                raise ValueError("No authentication credentials provided")
            
            # Build services
            self._service = build('sheets', 'v4', credentials=creds)
            self._drive_service = build('drive', 'v3', credentials=creds)
            
        except Exception as e:
            logger.error(f"Google Sheets authentication failed: {e}")
            raise RuntimeError(f"Google Sheets authentication failed: {e}")
    
    def export_results(
        self,
        results: DetectionResult,
        file_path: Union[str, Path],
        options: Optional[ExportOptions] = None
    ) -> Dict[str, Any]:
        """
        Export anomaly detection results to Google Sheets.
        
        Args:
            results: Detection results to export
            file_path: Not used for Google Sheets (spreadsheet info from options)
            options: Export configuration options with Google Sheets settings
            
        Returns:
            Dictionary containing export metadata and statistics
        """
        if not self._service:
            self._authenticate()
        
        if options is None:
            options = ExportOptions()
        
        try:
            # Convert results to DataFrame
            df = self._results_to_dataframe(results)
            
            # Create or get spreadsheet
            if options.spreadsheet_id:
                spreadsheet_id = options.spreadsheet_id
                logger.info(f"Using existing spreadsheet: {spreadsheet_id}")
            else:
                spreadsheet_id = self._create_spreadsheet(
                    title=f"Anomaly Detection Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                logger.info(f"Created new spreadsheet: {spreadsheet_id}")
            
            # Create worksheets and populate data
            worksheets_created = []
            
            # Main results sheet
            if options.include_summary or 'Results' in options.sheet_names:
                self._populate_results_sheet(spreadsheet_id, df, results)
                worksheets_created.append('Results')
            
            # Summary sheet
            if options.include_summary or 'Summary' in options.sheet_names:
                self._create_summary_sheet(spreadsheet_id, results)
                worksheets_created.append('Summary')
            
            # Charts sheet
            if options.include_charts or 'Charts' in options.sheet_names:
                self._create_charts_sheet(spreadsheet_id, df)
                worksheets_created.append('Charts')
            
            # Metadata sheet
            if options.include_metadata or 'Metadata' in options.sheet_names:
                self._create_metadata_sheet(spreadsheet_id, results, options)
                worksheets_created.append('Metadata')
            
            # Apply formatting
            if options.highlight_anomalies:
                self._apply_anomaly_formatting(spreadsheet_id, df)
            
            # Share spreadsheet if emails provided
            if options.share_with_emails:
                self._share_spreadsheet(spreadsheet_id, options.share_with_emails, options.permissions)
            
            # Get spreadsheet URL
            spreadsheet_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
            
            return {
                'spreadsheet_id': spreadsheet_id,
                'spreadsheet_url': spreadsheet_url,
                'export_time': datetime.now().isoformat(),
                'total_samples': len(results.scores),
                'anomalies_count': sum(results.labels),
                'worksheets_created': worksheets_created,
                'shared_with': options.share_with_emails or [],
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Failed to export results to Google Sheets: {e}")
            raise RuntimeError(f"Google Sheets export failed: {e}")
    
    def _results_to_dataframe(self, results: DetectionResult) -> pd.DataFrame:
        """Convert detection results to DataFrame."""
        data = {
            'Index': list(range(len(results.scores))),
            'Anomaly Score': [float(score.value) for score in results.scores],
            'Is Anomaly': ['YES' if bool(label) else 'NO' for label in results.labels],
            'Detector ID': [str(results.detector_id)] * len(results.scores),
            'Dataset ID': [str(results.dataset_id)] * len(results.scores),
            'Timestamp': [results.timestamp.strftime('%Y-%m-%d %H:%M:%S')] * len(results.scores),
            'Threshold': [results.threshold] * len(results.scores)
        }
        
        # Add confidence intervals if available
        if results.confidence_intervals:
            data['Confidence Lower'] = [ci.lower for ci in results.confidence_intervals]
            data['Confidence Upper'] = [ci.upper for ci in results.confidence_intervals]
        
        return pd.DataFrame(data)
    
    def _create_spreadsheet(self, title: str) -> str:
        """Create a new Google Spreadsheet."""
        spreadsheet = {
            'properties': {
                'title': title
            }
        }
        
        response = self._service.spreadsheets().create(body=spreadsheet).execute()
        return response['spreadsheetId']
    
    def _populate_results_sheet(
        self,
        spreadsheet_id: str,
        df: pd.DataFrame,
        results: DetectionResult
    ):
        """Populate the main results sheet."""
        
        # Prepare data for Google Sheets (header + data rows)
        values = [df.columns.tolist()] + df.values.tolist()
        
        # Clear and update the sheet
        self._service.spreadsheets().values().clear(
            spreadsheetId=spreadsheet_id,
            range='Sheet1!A:Z'
        ).execute()
        
        self._service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range='Sheet1!A1',
            valueInputOption='RAW',
            body={'values': values}
        ).execute()
        
        # Rename the sheet
        self._rename_sheet(spreadsheet_id, 0, 'Results')
        
        logger.info(f"Populated results sheet with {len(df)} rows")
    
    def _create_summary_sheet(self, spreadsheet_id: str, results: DetectionResult):
        """Create summary statistics sheet."""
        
        # Calculate statistics
        scores = [float(score.value) for score in results.scores]
        anomalies = [bool(label) for label in results.labels]
        
        total_samples = len(scores)
        anomaly_count = sum(anomalies)
        normal_count = total_samples - anomaly_count
        anomaly_rate = (anomaly_count / total_samples) * 100 if total_samples > 0 else 0
        
        avg_score = sum(scores) / len(scores) if scores else 0
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 0
        
        # Create summary data
        summary_data = [
            ['Metric', 'Value'],
            ['Total Samples', total_samples],
            ['Anomalies Detected', anomaly_count],
            ['Normal Samples', normal_count],
            ['Anomaly Rate (%)', f'{anomaly_rate:.2f}'],
            ['Average Score', f'{avg_score:.4f}'],
            ['Minimum Score', f'{min_score:.4f}'],
            ['Maximum Score', f'{max_score:.4f}'],
            ['Export Time', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Detector ID', str(results.detector_id)],
            ['Dataset ID', str(results.dataset_id)],
            ['Threshold', results.threshold]
        ]
        
        # Add new sheet
        sheet_id = self._add_sheet(spreadsheet_id, 'Summary')
        
        # Populate data
        self._service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range='Summary!A1',
            valueInputOption='RAW',
            body={'values': summary_data}
        ).execute()
        
        # Format header row
        self._format_header_row(spreadsheet_id, sheet_id, 0, 1)
        
        logger.info("Created summary sheet")
    
    def _create_charts_sheet(self, spreadsheet_id: str, df: pd.DataFrame):
        """Create charts and visualizations sheet."""
        
        # Add new sheet for charts
        sheet_id = self._add_sheet(spreadsheet_id, 'Charts')
        
        # Create scatter plot of anomaly scores
        requests = [{
            'addChart': {
                'chart': {
                    'spec': {
                        'title': 'Anomaly Score Distribution',
                        'basicChart': {
                            'chartType': 'SCATTER',
                            'legendPosition': 'RIGHT_LEGEND',
                            'axis': [
                                {
                                    'position': 'BOTTOM_AXIS',
                                    'title': 'Sample Index'
                                },
                                {
                                    'position': 'LEFT_AXIS',
                                    'title': 'Anomaly Score'
                                }
                            ],
                            'domains': [
                                {
                                    'domain': {
                                        'sourceRange': {
                                            'sources': [
                                                {
                                                    'sheetId': 0,  # Results sheet
                                                    'startRowIndex': 1,
                                                    'endRowIndex': len(df) + 1,
                                                    'startColumnIndex': 0,
                                                    'endColumnIndex': 1
                                                }
                                            ]
                                        }
                                    }
                                }
                            ],
                            'series': [
                                {
                                    'series': {
                                        'sourceRange': {
                                            'sources': [
                                                {
                                                    'sheetId': 0,  # Results sheet
                                                    'startRowIndex': 1,
                                                    'endRowIndex': len(df) + 1,
                                                    'startColumnIndex': 1,
                                                    'endColumnIndex': 2
                                                }
                                            ]
                                        }
                                    },
                                    'targetAxis': 'LEFT_AXIS'
                                }
                            ]
                        }
                    },
                    'position': {
                        'overlayPosition': {
                            'anchorCell': {
                                'sheetId': sheet_id,
                                'rowIndex': 1,
                                'columnIndex': 1
                            }
                        }
                    }
                }
            }
        }]
        
        # Execute chart creation
        self._service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={'requests': requests}
        ).execute()
        
        logger.info("Created charts sheet")
    
    def _create_metadata_sheet(
        self,
        spreadsheet_id: str,
        results: DetectionResult,
        options: ExportOptions
    ):
        """Create metadata sheet with export information."""
        
        metadata_data = [
            ['Property', 'Value'],
            ['Export Time', datetime.now().isoformat()],
            ['Detector Type', getattr(results, 'detector_name', 'Unknown')],
            ['Total Samples', len(results.scores)],
            ['Anomaly Threshold', results.threshold],
            ['Execution Time (ms)', results.execution_time_ms or 'N/A'],
            ['Export Format', 'Google Sheets'],
            ['Pynomaly Version', '1.0.0'],  # TODO: Get from package
            ['Spreadsheet ID', spreadsheet_id],
            ['Shared With', ', '.join(options.share_with_emails or [])]
        ]
        
        # Add metadata if available
        if results.metadata:
            for key, value in results.metadata.items():
                metadata_data.append([f'Metadata: {key}', str(value)])
        
        # Add new sheet
        sheet_id = self._add_sheet(spreadsheet_id, 'Metadata')
        
        # Populate data
        self._service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range='Metadata!A1',
            valueInputOption='RAW',
            body={'values': metadata_data}
        ).execute()
        
        # Format header row
        self._format_header_row(spreadsheet_id, sheet_id, 0, 1)
        
        logger.info("Created metadata sheet")
    
    def _add_sheet(self, spreadsheet_id: str, title: str) -> int:
        """Add a new sheet to the spreadsheet."""
        requests = [{
            'addSheet': {
                'properties': {
                    'title': title
                }
            }
        }]
        
        response = self._service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={'requests': requests}
        ).execute()
        
        return response['replies'][0]['addSheet']['properties']['sheetId']
    
    def _rename_sheet(self, spreadsheet_id: str, sheet_id: int, new_title: str):
        """Rename an existing sheet."""
        requests = [{
            'updateSheetProperties': {
                'properties': {
                    'sheetId': sheet_id,
                    'title': new_title
                },
                'fields': 'title'
            }
        }]
        
        self._service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={'requests': requests}
        ).execute()
    
    def _format_header_row(
        self,
        spreadsheet_id: str,
        sheet_id: int,
        start_row: int,
        end_row: int
    ):
        """Format header row with bold text and background color."""
        requests = [{
            'repeatCell': {
                'range': {
                    'sheetId': sheet_id,
                    'startRowIndex': start_row,
                    'endRowIndex': end_row
                },
                'cell': {
                    'userEnteredFormat': {
                        'backgroundColor': {
                            'red': 0.2,
                            'green': 0.4,
                            'blue': 0.8
                        },
                        'textFormat': {
                            'foregroundColor': {
                                'red': 1.0,
                                'green': 1.0,
                                'blue': 1.0
                            },
                            'bold': True
                        }
                    }
                },
                'fields': 'userEnteredFormat(backgroundColor,textFormat)'
            }
        }]
        
        self._service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={'requests': requests}
        ).execute()
    
    def _apply_anomaly_formatting(self, spreadsheet_id: str, df: pd.DataFrame):
        """Apply conditional formatting to highlight anomalies."""
        
        # Find 'Is Anomaly' column index
        anomaly_col_index = df.columns.get_loc('Is Anomaly')
        
        requests = [{
            'addConditionalFormatRule': {
                'rule': {
                    'ranges': [{
                        'sheetId': 0,  # Results sheet
                        'startRowIndex': 1,
                        'endRowIndex': len(df) + 1,
                        'startColumnIndex': anomaly_col_index,
                        'endColumnIndex': anomaly_col_index + 1
                    }],
                    'booleanRule': {
                        'condition': {
                            'type': 'TEXT_EQ',
                            'values': [{'userEnteredValue': 'YES'}]
                        },
                        'format': {
                            'backgroundColor': {
                                'red': 1.0,
                                'green': 0.9,
                                'blue': 0.9
                            },
                            'textFormat': {
                                'foregroundColor': {
                                    'red': 0.8,
                                    'green': 0.0,
                                    'blue': 0.0
                                },
                                'bold': True
                            }
                        }
                    }
                },
                'index': 0
            }
        }]
        
        self._service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={'requests': requests}
        ).execute()
        
        logger.info("Applied anomaly highlighting")
    
    def _share_spreadsheet(
        self,
        spreadsheet_id: str,
        emails: List[str],
        permission: str = 'view'
    ):
        """Share spreadsheet with specified emails."""
        
        # Map permission levels
        permission_mapping = {
            'view': 'reader',
            'edit': 'writer',
            'comment': 'commenter'
        }
        
        role = permission_mapping.get(permission, 'reader')
        
        for email in emails:
            permission_body = {
                'type': 'user',
                'role': role,
                'emailAddress': email
            }
            
            try:
                self._drive_service.permissions().create(
                    fileId=spreadsheet_id,
                    body=permission_body,
                    sendNotificationEmail=True
                ).execute()
                
                logger.info(f"Shared spreadsheet with {email} ({role})")
                
            except Exception as e:
                logger.warning(f"Failed to share with {email}: {e}")
    
    def get_supported_formats(self) -> List[str]:
        """Return list of supported formats (Google Sheets doesn't use file extensions)."""
        return ['.gsheets']  # Virtual extension for validation
    
    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """Validate Google Sheets export request (always true for Google Sheets)."""
        return True  # Google Sheets doesn't use file paths
    
    def list_spreadsheets(self) -> List[Dict[str, Any]]:
        """List user's Google Sheets spreadsheets."""
        if not self._drive_service:
            self._authenticate()
        
        try:
            results = self._drive_service.files().list(
                q="mimeType='application/vnd.google-apps.spreadsheet'",
                pageSize=100,
                fields="nextPageToken, files(id, name, createdTime, modifiedTime)"
            ).execute()
            
            return results.get('files', [])
            
        except Exception as e:
            logger.error(f"Failed to list spreadsheets: {e}")
            return []
    
    def get_spreadsheet_info(self, spreadsheet_id: str) -> Dict[str, Any]:
        """Get information about a specific spreadsheet."""
        if not self._service:
            self._authenticate()
        
        try:
            response = self._service.spreadsheets().get(
                spreadsheetId=spreadsheet_id
            ).execute()
            
            return {
                'id': response['spreadsheetId'],
                'title': response['properties']['title'],
                'sheets': [
                    {
                        'id': sheet['properties']['sheetId'],
                        'title': sheet['properties']['title'],
                        'gridProperties': sheet['properties'].get('gridProperties', {})
                    }
                    for sheet in response['sheets']
                ],
                'url': f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
            }
            
        except Exception as e:
            logger.error(f"Failed to get spreadsheet info: {e}")
            return {}
    
    def update_spreadsheet(
        self,
        spreadsheet_id: str,
        results: DetectionResult,
        sheet_name: str = 'Results',
        append: bool = False
    ) -> Dict[str, Any]:
        """Update existing spreadsheet with new results."""
        if not self._service:
            self._authenticate()
        
        try:
            df = self._results_to_dataframe(results)
            values = [df.columns.tolist()] + df.values.tolist()
            
            if append:
                # Append to existing data
                self._service.spreadsheets().values().append(
                    spreadsheetId=spreadsheet_id,
                    range=f'{sheet_name}!A:Z',
                    valueInputOption='RAW',
                    insertDataOption='INSERT_ROWS',
                    body={'values': values[1:]}  # Skip header
                ).execute()
            else:
                # Replace existing data
                self._service.spreadsheets().values().update(
                    spreadsheetId=spreadsheet_id,
                    range=f'{sheet_name}!A1',
                    valueInputOption='RAW',
                    body={'values': values}
                ).execute()
            
            logger.info(f"Updated spreadsheet {spreadsheet_id}")
            
            return {
                'spreadsheet_id': spreadsheet_id,
                'sheet_name': sheet_name,
                'update_time': datetime.now().isoformat(),
                'rows_updated': len(df),
                'append_mode': append,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Failed to update spreadsheet: {e}")
            raise RuntimeError(f"Spreadsheet update failed: {e}")