"""
Power BI Integration Adapter for Pynomaly

This module provides Power BI integration capabilities for pushing anomaly detection
results to Power BI workspaces, creating datasets, and generating reports.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import json
import asyncio

try:
    import msal
    from azure.identity import ClientSecretCredential, DefaultAzureCredential
    import requests
    POWERBI_AVAILABLE = True
except ImportError:
    POWERBI_AVAILABLE = False

from ...domain.entities.detection_result import DetectionResult
from ...shared.protocols.export_protocol import ExportProtocol
from ...application.dto.export_options import ExportOptions


logger = logging.getLogger(__name__)


class PowerBIAdapter(ExportProtocol):
    """
    Power BI adapter for pushing anomaly detection results to Power BI.
    
    Supports:
    - Dataset creation and management
    - Real-time data streaming
    - Report generation
    - Azure AD authentication
    """
    
    POWERBI_API_URL = "https://api.powerbi.com/v1.0/myorg"
    AUTHORITY = "https://login.microsoftonline.com/common"
    SCOPE = ["https://analysis.windows.net/powerbi/api/.default"]
    
    def __init__(
        self,
        tenant_id: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        use_default_credential: bool = False
    ):
        """
        Initialize Power BI adapter.
        
        Args:
            tenant_id: Azure AD tenant ID
            client_id: Azure AD client (application) ID
            client_secret: Azure AD client secret
            use_default_credential: Use DefaultAzureCredential for authentication
        """
        if not POWERBI_AVAILABLE:
            raise ImportError(
                "Power BI adapter requires Azure libraries. "
                "Install with: pip install msal azure-identity requests"
            )
        
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.use_default_credential = use_default_credential
        
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        
        logger.info("Power BI adapter initialized")
    
    async def _authenticate(self) -> str:
        """
        Authenticate with Azure AD and get access token.
        
        Returns:
            Access token for Power BI API
        """
        if self._access_token and self._token_expiry:
            # Check if token is still valid (with 5-minute buffer)
            if datetime.now() < self._token_expiry.replace(minute=self._token_expiry.minute - 5):
                return self._access_token
        
        try:
            if self.use_default_credential:
                # Use DefaultAzureCredential (works with managed identity, CLI, etc.)
                credential = DefaultAzureCredential()
                token = credential.get_token(*self.SCOPE)
                self._access_token = token.token
                self._token_expiry = datetime.fromtimestamp(token.expires_on)
                
            elif self.tenant_id and self.client_id and self.client_secret:
                # Use client secret authentication
                credential = ClientSecretCredential(
                    tenant_id=self.tenant_id,
                    client_id=self.client_id,
                    client_secret=self.client_secret
                )
                token = credential.get_token(*self.SCOPE)
                self._access_token = token.token
                self._token_expiry = datetime.fromtimestamp(token.expires_on)
                
            else:
                # Use MSAL for interactive authentication
                app = msal.PublicClientApplication(
                    self.client_id or "your-client-id",
                    authority=self.AUTHORITY
                )
                
                # Try to get token silently first
                accounts = app.get_accounts()
                if accounts:
                    result = app.acquire_token_silent(self.SCOPE, account=accounts[0])
                else:
                    result = None
                
                # If silent acquisition fails, use interactive flow
                if not result:
                    result = app.acquire_token_interactive(scopes=self.SCOPE)
                
                if "access_token" in result:
                    self._access_token = result["access_token"]
                    # MSAL doesn't provide direct expiry, estimate 1 hour
                    self._token_expiry = datetime.now().replace(hour=datetime.now().hour + 1)
                else:
                    raise RuntimeError(f"Authentication failed: {result.get('error_description', 'Unknown error')}")
            
            logger.info("Successfully authenticated with Power BI")
            return self._access_token
            
        except Exception as e:
            logger.error(f"Power BI authentication failed: {e}")
            raise RuntimeError(f"Power BI authentication failed: {e}")
    
    def _make_api_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make authenticated API request to Power BI.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (relative to base URL)
            data: Request body data
            params: Query parameters
            
        Returns:
            API response as dictionary
        """
        # Ensure we have a valid token
        token = asyncio.run(self._authenticate())
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        url = f"{self.POWERBI_API_URL}/{endpoint.lstrip('/')}"
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                params=params,
                timeout=30
            )
            
            response.raise_for_status()
            
            if response.content:
                return response.json()
            else:
                return {"success": True}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Power BI API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"Error details: {error_detail}")
                except:
                    logger.error(f"Response content: {e.response.text}")
            raise RuntimeError(f"Power BI API request failed: {e}")
    
    def export_results(
        self,
        results: DetectionResult,
        file_path: Union[str, Path],
        options: Optional[ExportOptions] = None
    ) -> Dict[str, Any]:
        """
        Export anomaly detection results to Power BI.
        
        Args:
            results: Detection results to export
            file_path: Not used for Power BI (workspace/dataset info from options)
            options: Export configuration options with Power BI settings
            
        Returns:
            Dictionary containing export metadata and statistics
        """
        if options is None:
            raise ValueError("ExportOptions with Power BI settings are required")
        
        if not options.workspace_id:
            raise ValueError("workspace_id is required in ExportOptions for Power BI export")
        
        if not options.dataset_name:
            raise ValueError("dataset_name is required in ExportOptions for Power BI export")
        
        try:
            # Convert results to DataFrame
            df = self._results_to_dataframe(results)
            
            # Create or update dataset
            dataset_id = self._create_or_update_dataset(
                workspace_id=options.workspace_id,
                dataset_name=options.dataset_name,
                df=df,
                streaming=options.streaming_dataset
            )
            
            # Push data to dataset
            if options.streaming_dataset:
                push_result = self._push_streaming_data(
                    workspace_id=options.workspace_id,
                    dataset_id=dataset_id,
                    table_name=options.table_name or "AnomalyResults",
                    df=df
                )
            else:
                push_result = self._push_dataset_data(
                    workspace_id=options.workspace_id,
                    dataset_id=dataset_id,
                    table_name=options.table_name or "AnomalyResults",
                    df=df
                )
            
            return {
                'workspace_id': options.workspace_id,
                'dataset_id': dataset_id,
                'dataset_name': options.dataset_name,
                'table_name': options.table_name or "AnomalyResults",
                'export_time': datetime.now().isoformat(),
                'total_samples': len(results.scores),
                'anomalies_count': sum(results.labels),
                'streaming': options.streaming_dataset,
                'push_result': push_result,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Failed to export results to Power BI: {e}")
            raise RuntimeError(f"Power BI export failed: {e}")
    
    def _results_to_dataframe(self, results: DetectionResult) -> pd.DataFrame:
        """Convert detection results to DataFrame."""
        data = {
            'Index': list(range(len(results.scores))),
            'AnomalyScore': [float(score.value) for score in results.scores],
            'IsAnomaly': [bool(label) for label in results.labels],
            'DetectorId': [str(results.detector_id)] * len(results.scores),
            'DatasetId': [str(results.dataset_id)] * len(results.scores),
            'Timestamp': [results.timestamp] * len(results.scores),
            'Threshold': [results.threshold] * len(results.scores)
        }
        
        # Add confidence intervals if available
        if results.confidence_intervals:
            data['ConfidenceLower'] = [ci.lower for ci in results.confidence_intervals]
            data['ConfidenceUpper'] = [ci.upper for ci in results.confidence_intervals]
        
        # Add execution metadata
        if results.execution_time_ms:
            data['ExecutionTimeMs'] = [results.execution_time_ms] * len(results.scores)
        
        return pd.DataFrame(data)
    
    def _create_or_update_dataset(
        self,
        workspace_id: str,
        dataset_name: str,
        df: pd.DataFrame,
        streaming: bool = False
    ) -> str:
        """Create or update Power BI dataset."""
        
        # Check if dataset already exists
        datasets = self._make_api_request(
            "GET",
            f"groups/{workspace_id}/datasets"
        )
        
        existing_dataset = None
        for dataset in datasets.get("value", []):
            if dataset["name"] == dataset_name:
                existing_dataset = dataset
                break
        
        if existing_dataset:
            logger.info(f"Using existing dataset: {dataset_name}")
            return existing_dataset["id"]
        
        # Create new dataset
        schema = self._create_dataset_schema(df, dataset_name, streaming)
        
        if streaming:
            endpoint = f"groups/{workspace_id}/datasets"
            response = self._make_api_request("POST", endpoint, data=schema)
        else:
            endpoint = f"groups/{workspace_id}/datasets"
            response = self._make_api_request("POST", endpoint, data=schema)
        
        dataset_id = response["id"]
        logger.info(f"Created new dataset: {dataset_name} (ID: {dataset_id})")
        
        return dataset_id
    
    def _create_dataset_schema(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        streaming: bool = False
    ) -> Dict[str, Any]:
        """Create Power BI dataset schema from DataFrame."""
        
        # Map pandas dtypes to Power BI data types
        dtype_mapping = {
            'int64': 'Int64',
            'float64': 'Double',
            'bool': 'Boolean',
            'object': 'String',
            'datetime64[ns]': 'DateTime'
        }
        
        columns = []
        for col_name, dtype in df.dtypes.items():
            powerbi_type = dtype_mapping.get(str(dtype), 'String')
            columns.append({
                "name": col_name,
                "dataType": powerbi_type
            })
        
        table_schema = {
            "name": "AnomalyResults",
            "columns": columns
        }
        
        schema = {
            "name": dataset_name,
            "tables": [table_schema]
        }
        
        if streaming:
            schema["defaultMode"] = "Streaming"
        
        return schema
    
    def _push_streaming_data(
        self,
        workspace_id: str,
        dataset_id: str,
        table_name: str,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Push data to streaming dataset."""
        
        # Convert DataFrame to rows
        rows = []
        for _, row in df.iterrows():
            row_data = {}
            for col, value in row.items():
                # Convert numpy types to Python types
                if pd.isna(value):
                    row_data[col] = None
                elif isinstance(value, (pd.Timestamp, datetime)):
                    row_data[col] = value.isoformat()
                else:
                    row_data[col] = value
            rows.append(row_data)
        
        # Push data in batches (Power BI limit is 10,000 rows per request)
        batch_size = 1000
        push_results = []
        
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            
            endpoint = f"groups/{workspace_id}/datasets/{dataset_id}/tables/{table_name}/rows"
            result = self._make_api_request("POST", endpoint, data={"rows": batch})
            push_results.append(result)
            
            logger.info(f"Pushed batch {i//batch_size + 1}: {len(batch)} rows")
        
        return {
            'total_rows': len(rows),
            'batches': len(push_results),
            'batch_results': push_results
        }
    
    def _push_dataset_data(
        self,
        workspace_id: str,
        dataset_id: str,
        table_name: str,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Push data to regular dataset (replace existing data)."""
        
        # For regular datasets, we typically replace all data
        # First, delete existing rows
        try:
            endpoint = f"groups/{workspace_id}/datasets/{dataset_id}/tables/{table_name}/rows"
            self._make_api_request("DELETE", endpoint)
            logger.info(f"Cleared existing data from table {table_name}")
        except Exception as e:
            logger.warning(f"Could not clear existing data: {e}")
        
        # Then push new data
        return self._push_streaming_data(workspace_id, dataset_id, table_name, df)
    
    def get_supported_formats(self) -> List[str]:
        """Return list of supported formats (Power BI doesn't use file extensions)."""
        return ['.powerbi']  # Virtual extension for validation
    
    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """Validate Power BI export request (always true for Power BI)."""
        return True  # Power BI doesn't use file paths
    
    def list_workspaces(self) -> List[Dict[str, Any]]:
        """List available Power BI workspaces."""
        try:
            response = self._make_api_request("GET", "groups")
            return response.get("value", [])
        except Exception as e:
            logger.error(f"Failed to list workspaces: {e}")
            return []
    
    def list_datasets(self, workspace_id: str) -> List[Dict[str, Any]]:
        """List datasets in a workspace."""
        try:
            response = self._make_api_request("GET", f"groups/{workspace_id}/datasets")
            return response.get("value", [])
        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")
            return []
    
    def create_report_from_dataset(
        self,
        workspace_id: str,
        dataset_id: str,
        report_name: str
    ) -> Dict[str, Any]:
        """Create a basic report from the anomaly dataset."""
        
        # Basic report configuration
        report_config = {
            "name": report_name,
            "datasetId": dataset_id
        }
        
        try:
            response = self._make_api_request(
                "POST",
                f"groups/{workspace_id}/reports",
                data=report_config
            )
            
            logger.info(f"Created report: {report_name}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to create report: {e}")
            raise RuntimeError(f"Report creation failed: {e}")
    
    def refresh_dataset(self, workspace_id: str, dataset_id: str) -> Dict[str, Any]:
        """Trigger dataset refresh."""
        try:
            response = self._make_api_request(
                "POST",
                f"groups/{workspace_id}/datasets/{dataset_id}/refreshes"
            )
            
            logger.info(f"Triggered dataset refresh for {dataset_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to refresh dataset: {e}")
            raise RuntimeError(f"Dataset refresh failed: {e}")
    
    def get_dataset_refresh_history(
        self,
        workspace_id: str,
        dataset_id: str
    ) -> List[Dict[str, Any]]:
        """Get dataset refresh history."""
        try:
            response = self._make_api_request(
                "GET",
                f"groups/{workspace_id}/datasets/{dataset_id}/refreshes"
            )
            
            return response.get("value", [])
            
        except Exception as e:
            logger.error(f"Failed to get refresh history: {e}")
            return []