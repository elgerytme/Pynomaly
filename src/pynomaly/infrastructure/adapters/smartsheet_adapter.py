"""
Smartsheet Integration Adapter for Pynomaly

This module provides Smartsheet integration capabilities for creating
and managing sheets with anomaly detection results and workflow automation.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

try:
    import smartsheet
    SMARTSHEET_AVAILABLE = True
except ImportError:
    SMARTSHEET_AVAILABLE = False

from ...domain.entities.detection_result import DetectionResult
from ...shared.protocols.export_protocol import ExportProtocol
from ...application.dto.export_options import ExportOptions


logger = logging.getLogger(__name__)


class SmartsheetAdapter(ExportProtocol):
    """
    Smartsheet adapter for creating and managing sheets with anomaly detection results.
    
    Supports:
    - Sheet creation and updates
    - Workflow automation
    - Project tracking integration
    - Automated notifications
    - Dashboard creation
    """
    
    def __init__(self, access_token: Optional[str] = None):
        """
        Initialize Smartsheet adapter.
        
        Args:
            access_token: Smartsheet API access token
        """
        if not SMARTSHEET_AVAILABLE:
            raise ImportError(
                "Smartsheet adapter requires Smartsheet SDK. "
                "Install with: pip install smartsheet-python-sdk"
            )
        
        self.access_token = access_token
        self._client = None
        
        logger.info("Smartsheet adapter initialized")
    
    def _authenticate(self):
        """Initialize Smartsheet client with authentication."""
        if not self.access_token:
            raise ValueError("Smartsheet access token is required")
        
        try:
            self._client = smartsheet.Smartsheet(self.access_token)
            # Test authentication by getting user info
            user_info = self._client.Users.get_current_user()
            logger.info(f"Authenticated as: {user_info.email}")
            
        except Exception as e:
            logger.error(f"Smartsheet authentication failed: {e}")
            raise RuntimeError(f"Smartsheet authentication failed: {e}")
    
    def export_results(
        self,
        results: DetectionResult,
        file_path: Union[str, Path],
        options: Optional[ExportOptions] = None
    ) -> Dict[str, Any]:
        """
        Export anomaly detection results to Smartsheet.
        
        Args:
            results: Detection results to export
            file_path: Not used for Smartsheet (sheet info from options)
            options: Export configuration options with Smartsheet settings
            
        Returns:
            Dictionary containing export metadata and statistics
        """
        if not self._client:
            self._authenticate()
        
        if options is None:
            options = ExportOptions()
        
        try:
            # Convert results to DataFrame
            df = self._results_to_dataframe(results)
            
            # Create or get sheet
            if options.sheet_template_id:
                sheet_id = self._create_sheet_from_template(
                    template_id=options.sheet_template_id,
                    name=f"Anomaly Detection Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
            else:
                sheet_id = self._create_sheet(
                    name=f"Anomaly Detection Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    folder_id=options.folder_id,
                    workspace_name=options.workspace_name
                )
            
            # Populate sheet with data
            self._populate_sheet(sheet_id, df, results)
            
            # Apply formatting and rules
            self._apply_formatting(sheet_id, df)
            
            # Set up automation rules if requested
            if options.notify_on_completion and options.notification_emails:
                self._setup_notification_rules(sheet_id, options.notification_emails)
            
            # Get sheet URL
            sheet_info = self._client.Sheets.get_sheet(sheet_id)
            sheet_url = sheet_info.permalink
            
            return {
                'sheet_id': sheet_id,
                'sheet_name': sheet_info.name,
                'sheet_url': sheet_url,
                'export_time': datetime.now().isoformat(),
                'total_samples': len(results.scores),
                'anomalies_count': sum(results.labels),
                'workspace': options.workspace_name,
                'folder_id': options.folder_id,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Failed to export results to Smartsheet: {e}")
            raise RuntimeError(f"Smartsheet export failed: {e}")
    
    def _results_to_dataframe(self, results: DetectionResult) -> pd.DataFrame:
        """Convert detection results to DataFrame."""
        data = {
            'Index': list(range(len(results.scores))),
            'Anomaly Score': [float(score.value) for score in results.scores],
            'Is Anomaly': ['YES' if bool(label) else 'NO' for label in results.labels],
            'Status': ['Review Required' if bool(label) else 'Normal' for label in results.labels],
            'Detector ID': [str(results.detector_id)] * len(results.scores),
            'Dataset ID': [str(results.dataset_id)] * len(results.scores),
            'Detection Date': [results.timestamp.strftime('%Y-%m-%d')] * len(results.scores),
            'Threshold': [results.threshold] * len(results.scores),
            'Action Required': ['Investigate' if bool(label) else 'None' for label in results.labels]
        }
        
        # Add confidence intervals if available
        if results.confidence_intervals:
            data['Confidence Lower'] = [ci.lower for ci in results.confidence_intervals]
            data['Confidence Upper'] = [ci.upper for ci in results.confidence_intervals]
        
        return pd.DataFrame(data)
    
    def _create_sheet(
        self,
        name: str,
        folder_id: Optional[str] = None,
        workspace_name: Optional[str] = None
    ) -> int:
        """Create a new Smartsheet."""
        
        # Define column specifications
        columns = [
            smartsheet.models.Column({
                'title': 'Index',
                'type': 'TEXT_NUMBER',
                'primary': True
            }),
            smartsheet.models.Column({
                'title': 'Anomaly Score',
                'type': 'TEXT_NUMBER'
            }),
            smartsheet.models.Column({
                'title': 'Is Anomaly',
                'type': 'PICKLIST',
                'options': ['YES', 'NO']
            }),
            smartsheet.models.Column({
                'title': 'Status',
                'type': 'PICKLIST',
                'options': ['Normal', 'Review Required', 'Investigated', 'Resolved']
            }),
            smartsheet.models.Column({
                'title': 'Detection Date',
                'type': 'DATE'
            }),
            smartsheet.models.Column({
                'title': 'Action Required',
                'type': 'PICKLIST',
                'options': ['None', 'Investigate', 'Alert Team', 'Emergency Response']
            }),
            smartsheet.models.Column({
                'title': 'Assigned To',
                'type': 'CONTACT_LIST'
            }),
            smartsheet.models.Column({
                'title': 'Comments',
                'type': 'TEXT_NUMBER'
            })
        ]
        
        # Create sheet specification
        sheet_spec = smartsheet.models.Sheet({
            'name': name,
            'columns': columns
        })
        
        # Create sheet in specified location
        if folder_id:
            response = self._client.Folders.create_sheet_in_folder(folder_id, sheet_spec)
        elif workspace_name:
            # Find workspace by name
            workspace_id = self._find_workspace_id(workspace_name)
            if workspace_id:
                response = self._client.Workspaces.create_sheet_in_workspace(workspace_id, sheet_spec)
            else:
                # Create in home
                response = self._client.Sheets.create_sheet(sheet_spec)
        else:
            # Create in home
            response = self._client.Sheets.create_sheet(sheet_spec)
        
        sheet_id = response.result.id
        logger.info(f"Created Smartsheet: {name} (ID: {sheet_id})")
        
        return sheet_id
    
    def _create_sheet_from_template(self, template_id: str, name: str) -> int:
        """Create sheet from existing template."""
        
        sheet_spec = smartsheet.models.Sheet({
            'name': name,
            'from_id': int(template_id)
        })
        
        response = self._client.Sheets.create_sheet_from_template(sheet_spec)
        sheet_id = response.result.id
        
        logger.info(f"Created sheet from template: {name} (ID: {sheet_id})")
        return sheet_id
    
    def _populate_sheet(self, sheet_id: int, df: pd.DataFrame, results: DetectionResult):
        """Populate sheet with detection results data."""
        
        # Get sheet to get column IDs
        sheet = self._client.Sheets.get_sheet(sheet_id)
        
        # Create column ID mapping
        column_map = {col.title: col.id for col in sheet.columns}
        
        # Prepare rows for insertion
        rows_to_add = []
        
        for _, row_data in df.iterrows():
            cells = []
            
            # Map DataFrame columns to Smartsheet columns
            for col_name, value in row_data.items():
                if col_name in column_map:
                    cell = smartsheet.models.Cell()
                    cell.column_id = column_map[col_name]
                    
                    # Handle different data types
                    if pd.isna(value):
                        cell.value = None
                    elif col_name == 'Detection Date':
                        # Convert to date format
                        cell.value = datetime.strptime(str(value), '%Y-%m-%d').strftime('%Y-%m-%d')
                    elif col_name in ['Anomaly Score', 'Threshold']:
                        cell.value = float(value)
                    elif col_name == 'Index':
                        cell.value = int(value)
                    else:
                        cell.value = str(value)
                    
                    cells.append(cell)
            
            row = smartsheet.models.Row()
            row.cells = cells
            rows_to_add.append(row)
        
        # Add rows in batches (Smartsheet limit is 5000 rows per request)
        batch_size = 1000
        for i in range(0, len(rows_to_add), batch_size):
            batch = rows_to_add[i:i + batch_size]
            self._client.Sheets.add_rows(sheet_id, batch)
            logger.info(f"Added batch {i//batch_size + 1}: {len(batch)} rows")
        
        logger.info(f"Populated sheet with {len(df)} rows")
    
    def _apply_formatting(self, sheet_id: int, df: pd.DataFrame):
        """Apply conditional formatting and rules to the sheet."""
        
        # Get sheet to access rows and columns
        sheet = self._client.Sheets.get_sheet(sheet_id, include='format')
        
        # Find anomaly-related columns
        anomaly_col_id = None
        status_col_id = None
        
        for col in sheet.columns:
            if col.title == 'Is Anomaly':
                anomaly_col_id = col.id
            elif col.title == 'Status':
                status_col_id = col.id
        
        if not anomaly_col_id:
            logger.warning("Could not find 'Is Anomaly' column for formatting")
            return
        
        # Apply formatting rules using cell formatting
        # Note: Smartsheet's conditional formatting is limited via API
        # We'll format cells directly based on their values
        
        try:
            # Get all rows to apply formatting
            sheet_rows = sheet.rows
            rows_to_update = []
            
            for row in sheet_rows:
                cells_to_update = []
                anomaly_value = None
                
                # Find the anomaly cell value
                for cell in row.cells:
                    if cell.column_id == anomaly_col_id:
                        anomaly_value = cell.value
                        break
                
                # Apply formatting based on anomaly status
                if anomaly_value == 'YES':
                    for cell in row.cells:
                        if cell.column_id in [anomaly_col_id, status_col_id]:
                            # Create formatted cell
                            formatted_cell = smartsheet.models.Cell()
                            formatted_cell.column_id = cell.column_id
                            formatted_cell.value = cell.value
                            formatted_cell.format = ',,,,,,,,,18,,,,,,'  # Red background
                            cells_to_update.append(formatted_cell)
                
                if cells_to_update:
                    update_row = smartsheet.models.Row()
                    update_row.id = row.id
                    update_row.cells = cells_to_update
                    rows_to_update.append(update_row)
            
            # Update rows with formatting
            if rows_to_update:
                self._client.Sheets.update_rows(sheet_id, rows_to_update)
                logger.info("Applied anomaly highlighting")
                
        except Exception as e:
            logger.warning(f"Could not apply formatting: {e}")
    
    def _setup_notification_rules(self, sheet_id: int, emails: List[str]):
        """Set up automation rules for notifications."""
        
        try:
            # Create automation rule for anomaly detection
            automation_rule = smartsheet.models.AutomationRule()
            automation_rule.name = "Anomaly Detection Alert"
            automation_rule.enabled = True
            
            # Define trigger: when a row is added or changed
            trigger = smartsheet.models.WebhookSharedSecret()
            # Note: Full automation rule setup requires more complex API calls
            # This is a simplified example
            
            logger.info(f"Notification setup initiated for {len(emails)} recipients")
            
        except Exception as e:
            logger.warning(f"Could not set up notifications: {e}")
    
    def _find_workspace_id(self, workspace_name: str) -> Optional[int]:
        """Find workspace ID by name."""
        try:
            workspaces = self._client.Workspaces.list_workspaces(include_all=True)
            for workspace in workspaces.data:
                if workspace.name == workspace_name:
                    return workspace.id
            return None
        except Exception as e:
            logger.warning(f"Could not find workspace '{workspace_name}': {e}")
            return None
    
    def get_supported_formats(self) -> List[str]:
        """Return list of supported formats (Smartsheet doesn't use file extensions)."""
        return ['.smartsheet']  # Virtual extension for validation
    
    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """Validate Smartsheet export request (always true for Smartsheet)."""
        return True  # Smartsheet doesn't use file paths
    
    def list_sheets(self) -> List[Dict[str, Any]]:
        """List user's Smartsheet sheets."""
        if not self._client:
            self._authenticate()
        
        try:
            response = self._client.Sheets.list_sheets(include_all=True)
            
            sheets = []
            for sheet in response.data:
                sheets.append({
                    'id': sheet.id,
                    'name': sheet.name,
                    'created_at': sheet.created_at.isoformat() if sheet.created_at else None,
                    'modified_at': sheet.modified_at.isoformat() if sheet.modified_at else None,
                    'permalink': sheet.permalink
                })
            
            return sheets
            
        except Exception as e:
            logger.error(f"Failed to list sheets: {e}")
            return []
    
    def list_workspaces(self) -> List[Dict[str, Any]]:
        """List user's workspaces."""
        if not self._client:
            self._authenticate()
        
        try:
            response = self._client.Workspaces.list_workspaces(include_all=True)
            
            workspaces = []
            for workspace in response.data:
                workspaces.append({
                    'id': workspace.id,
                    'name': workspace.name,
                    'permalink': workspace.permalink
                })
            
            return workspaces
            
        except Exception as e:
            logger.error(f"Failed to list workspaces: {e}")
            return []
    
    def create_dashboard(
        self,
        name: str,
        sheet_ids: List[int],
        workspace_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a dashboard with anomaly metrics."""
        
        try:
            # Create dashboard specification
            dashboard_spec = smartsheet.models.Dashboard()
            dashboard_spec.name = name
            
            # Add widgets (simplified example)
            widgets = []
            
            # Chart widget for anomaly distribution
            chart_widget = smartsheet.models.Widget()
            chart_widget.type = 'CHART'
            chart_widget.title = 'Anomaly Distribution'
            # Note: Full widget configuration requires more detailed setup
            widgets.append(chart_widget)
            
            dashboard_spec.widgets = widgets
            
            # Create dashboard
            if workspace_id:
                response = self._client.Dashboards.create_dashboard_in_workspace(
                    workspace_id, dashboard_spec
                )
            else:
                response = self._client.Dashboards.create_dashboard(dashboard_spec)
            
            dashboard_id = response.result.id
            
            logger.info(f"Created dashboard: {name} (ID: {dashboard_id})")
            
            return {
                'dashboard_id': dashboard_id,
                'name': name,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
            return {'success': False, 'error': str(e)}
    
    def update_sheet(
        self,
        sheet_id: int,
        results: DetectionResult,
        append: bool = True
    ) -> Dict[str, Any]:
        """Update existing sheet with new results."""
        if not self._client:
            self._authenticate()
        
        try:
            df = self._results_to_dataframe(results)
            
            if append:
                # Add new rows to existing sheet
                self._populate_sheet(sheet_id, df, results)
            else:
                # Clear and replace sheet content
                # Note: Smartsheet doesn't have a direct "clear all" operation
                # This would require deleting rows and adding new ones
                logger.warning("Replace mode not fully implemented for Smartsheet")
                self._populate_sheet(sheet_id, df, results)
            
            logger.info(f"Updated sheet {sheet_id}")
            
            return {
                'sheet_id': sheet_id,
                'update_time': datetime.now().isoformat(),
                'rows_updated': len(df),
                'append_mode': append,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Failed to update sheet: {e}")
            raise RuntimeError(f"Sheet update failed: {e}")
    
    def get_sheet_info(self, sheet_id: int) -> Dict[str, Any]:
        """Get information about a specific sheet."""
        if not self._client:
            self._authenticate()
        
        try:
            sheet = self._client.Sheets.get_sheet(sheet_id)
            
            return {
                'id': sheet.id,
                'name': sheet.name,
                'permalink': sheet.permalink,
                'created_at': sheet.created_at.isoformat() if sheet.created_at else None,
                'modified_at': sheet.modified_at.isoformat() if sheet.modified_at else None,
                'column_count': len(sheet.columns) if sheet.columns else 0,
                'row_count': sheet.total_row_count if hasattr(sheet, 'total_row_count') else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get sheet info: {e}")
            return {}
    
    def share_sheet(
        self,
        sheet_id: int,
        emails: List[str],
        access_level: str = 'VIEWER'
    ) -> Dict[str, Any]:
        """Share sheet with specified users."""
        
        try:
            # Validate access level
            valid_levels = ['VIEWER', 'EDITOR', 'ADMIN', 'OWNER']
            if access_level not in valid_levels:
                access_level = 'VIEWER'
            
            shared_users = []
            
            for email in emails:
                share_spec = smartsheet.models.Share()
                share_spec.email = email
                share_spec.access_level = access_level
                share_spec.subject = f"Anomaly Detection Results Sheet"
                share_spec.message = "You have been granted access to anomaly detection results."
                
                try:
                    response = self._client.Sheets.share_sheet(sheet_id, [share_spec])
                    shared_users.append({
                        'email': email,
                        'access_level': access_level,
                        'success': True
                    })
                    logger.info(f"Shared sheet with {email} ({access_level})")
                    
                except Exception as e:
                    shared_users.append({
                        'email': email,
                        'access_level': access_level,
                        'success': False,
                        'error': str(e)
                    })
                    logger.warning(f"Failed to share with {email}: {e}")
            
            return {
                'sheet_id': sheet_id,
                'shared_users': shared_users,
                'total_shared': len([u for u in shared_users if u['success']]),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Failed to share sheet: {e}")
            return {'success': False, 'error': str(e)}