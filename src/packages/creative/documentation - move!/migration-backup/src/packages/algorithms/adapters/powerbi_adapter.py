"""
Power BI adapter for exporting anomaly detection results to Microsoft Power BI.

This adapter handles:
- Dataset creation and management
- Streaming data to Power BI datasets
- Workspace integration
- Authentication with Azure AD
- Real-time dashboard updates
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import requests

from ...shared.exceptions import AuthenticationError, ExportError
from ...shared.protocols.export_protocol import ExportProtocol


@dataclass
class PowerBIConfig:
    """Configuration for Power BI integration."""

    client_id: str
    client_secret: str
    tenant_id: str
    workspace_id: str | None = None
    dataset_id: str | None = None
    table_name: str = "AnomalyResults"
    streaming_endpoint: str | None = None
    api_version: str = "v1.0"
    base_url: str = "https://api.powerbi.com"


class PowerBIAdapter(ExportProtocol):
    """Adapter for exporting data to Microsoft Power BI."""

    def __init__(self, config: PowerBIConfig):
        """Initialize Power BI adapter with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._access_token: str | None = None
        self._token_expires_at: datetime | None = None

    def get_supported_formats(self) -> list[str]:
        """Return supported export formats."""
        return ["powerbi", "powerbi_streaming", "powerbi_dataset"]

    async def export_results(
        self,
        data: pd.DataFrame | list[dict[str, Any]],
        format_type: str = "powerbi",
        **kwargs,
    ) -> dict[str, Any]:
        """Export anomaly detection results to Power BI."""
        try:
            self.logger.info(f"Starting Power BI export with format: {format_type}")

            # Ensure we have a valid access token
            await self._ensure_valid_token()

            # Convert data to DataFrame if needed
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()

            # Prepare data for Power BI
            processed_data = self._prepare_data_for_powerbi(df)

            if format_type == "powerbi_streaming":
                result = await self._push_to_streaming_dataset(processed_data, **kwargs)
            elif format_type == "powerbi_dataset":
                result = await self._create_or_update_dataset(processed_data, **kwargs)
            else:
                # Default to streaming if dataset exists, otherwise create dataset
                if self.config.dataset_id:
                    result = await self._push_to_streaming_dataset(
                        processed_data, **kwargs
                    )
                else:
                    result = await self._create_or_update_dataset(
                        processed_data, **kwargs
                    )

            self.logger.info("Power BI export completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Power BI export failed: {str(e)}")
            raise ExportError(f"Failed to export to Power BI: {str(e)}") from e

    async def validate_file(self, file_path: str) -> bool:
        """Validate Power BI export configuration."""
        try:
            # Check if we can authenticate
            await self._ensure_valid_token()

            # Check workspace access if specified
            if self.config.workspace_id:
                await self._validate_workspace_access()

            # Check dataset access if specified
            if self.config.dataset_id:
                await self._validate_dataset_access()

            return True

        except Exception as e:
            self.logger.error(f"Power BI validation failed: {str(e)}")
            return False

    async def _ensure_valid_token(self) -> None:
        """Ensure we have a valid access token."""
        if (
            self._access_token is None
            or self._token_expires_at is None
            or datetime.now() >= self._token_expires_at
        ):
            await self._authenticate()

    async def _authenticate(self) -> None:
        """Authenticate with Azure AD to get Power BI access token."""
        try:
            auth_url = f"https://login.microsoftonline.com/{self.config.tenant_id}/oauth2/v2.0/token"

            data = {
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
                "scope": "https://analysis.windows.net/powerbi/api/.default",
                "grant_type": "client_credentials",
            }

            response = requests.post(auth_url, data=data)
            response.raise_for_status()

            token_data = response.json()
            self._access_token = token_data["access_token"]

            # Token expires in seconds, subtract 5 minutes for safety
            expires_in = token_data.get("expires_in", 3600) - 300
            self._token_expires_at = datetime.now().timestamp() + expires_in

            self.logger.info("Successfully authenticated with Power BI")

        except Exception as e:
            raise AuthenticationError(
                f"Failed to authenticate with Power BI: {str(e)}"
            ) from e

    def _get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers for API requests."""
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

    def _prepare_data_for_powerbi(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Prepare data for Power BI consumption."""
        # Ensure timestamp column is in the right format
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            )

        # Convert NaN values to None for JSON serialization
        df = df.where(pd.notnull(df), None)

        # Convert to list of dictionaries
        records = df.to_dict("records")

        # Add metadata
        for record in records:
            record["export_timestamp"] = datetime.now().isoformat()
            record["source"] = "pynomaly"

        return records

    async def _push_to_streaming_dataset(
        self, data: list[dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        """Push data to a Power BI streaming dataset."""
        if not self.config.streaming_endpoint and not self.config.dataset_id:
            raise ExportError(
                "Streaming endpoint or dataset ID required for streaming export"
            )

        if self.config.streaming_endpoint:
            # Use direct streaming endpoint
            url = self.config.streaming_endpoint
            headers = {"Content-Type": "application/json"}
        else:
            # Use dataset streaming API
            base_url = f"{self.config.base_url}/{self.config.api_version}"
            if self.config.workspace_id:
                url = f"{base_url}/groups/{self.config.workspace_id}/datasets/{self.config.dataset_id}/rows"
            else:
                url = f"{base_url}/datasets/{self.config.dataset_id}/rows"
            headers = self._get_auth_headers()

        # Split data into batches (Power BI has a 10MB limit per request)
        batch_size = kwargs.get("batch_size", 1000)
        results = []

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]

            payload = {"rows": batch}

            try:
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()

                results.append(
                    {
                        "batch": i // batch_size + 1,
                        "rows_sent": len(batch),
                        "status": "success",
                    }
                )

            except requests.exceptions.RequestException as e:
                self.logger.error(
                    f"Failed to send batch {i // batch_size + 1}: {str(e)}"
                )
                results.append(
                    {
                        "batch": i // batch_size + 1,
                        "rows_sent": len(batch),
                        "status": "failed",
                        "error": str(e),
                    }
                )

        return {
            "export_type": "streaming",
            "total_rows": len(data),
            "batches": results,
            "dataset_id": self.config.dataset_id,
            "timestamp": datetime.now().isoformat(),
        }

    async def _create_or_update_dataset(
        self, data: list[dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        """Create or update a Power BI dataset."""
        dataset_name = kwargs.get("dataset_name", "Pynomaly Anomaly Results")

        # Create dataset schema from data
        if not data:
            raise ExportError("No data provided for dataset creation")

        schema = self._create_dataset_schema(data[0], dataset_name)

        # Create or get dataset
        dataset_id = await self._ensure_dataset_exists(schema, dataset_name)

        # Push data to dataset
        result = await self._push_to_streaming_dataset(data, **kwargs)
        result.update(
            {
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "export_type": "dataset",
            }
        )

        return result

    def _create_dataset_schema(
        self, sample_record: dict[str, Any], dataset_name: str
    ) -> dict[str, Any]:
        """Create Power BI dataset schema from sample data."""
        columns = []

        for key, value in sample_record.items():
            if isinstance(value, bool):
                data_type = "bool"
            elif isinstance(value, int):
                data_type = "Int64"
            elif isinstance(value, float):
                data_type = "Double"
            elif isinstance(value, str):
                if key in ["timestamp", "export_timestamp"] or "time" in key.lower():
                    data_type = "DateTime"
                else:
                    data_type = "string"
            else:
                data_type = "string"

            columns.append({"name": key, "dataType": data_type})

        return {
            "name": dataset_name,
            "tables": [{"name": self.config.table_name, "columns": columns}],
        }

    async def _ensure_dataset_exists(
        self, schema: dict[str, Any], dataset_name: str
    ) -> str:
        """Ensure dataset exists, create if it doesn't."""
        # Check if dataset already exists
        if self.config.dataset_id:
            if await self._validate_dataset_access():
                return self.config.dataset_id

        # Create new dataset
        base_url = f"{self.config.base_url}/{self.config.api_version}"
        if self.config.workspace_id:
            url = f"{base_url}/groups/{self.config.workspace_id}/datasets"
        else:
            url = f"{base_url}/datasets"

        headers = self._get_auth_headers()

        try:
            response = requests.post(url, headers=headers, json=schema)
            response.raise_for_status()

            dataset_info = response.json()
            dataset_id = dataset_info["id"]

            self.logger.info(f"Created new Power BI dataset: {dataset_id}")
            return dataset_id

        except requests.exceptions.RequestException as e:
            raise ExportError(f"Failed to create Power BI dataset: {str(e)}") from e

    async def _validate_workspace_access(self) -> bool:
        """Validate access to the specified workspace."""
        base_url = f"{self.config.base_url}/{self.config.api_version}"
        url = f"{base_url}/groups/{self.config.workspace_id}"
        headers = self._get_auth_headers()

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException:
            return False

    async def _validate_dataset_access(self) -> bool:
        """Validate access to the specified dataset."""
        base_url = f"{self.config.base_url}/{self.config.api_version}"
        if self.config.workspace_id:
            url = f"{base_url}/groups/{self.config.workspace_id}/datasets/{self.config.dataset_id}"
        else:
            url = f"{base_url}/datasets/{self.config.dataset_id}"

        headers = self._get_auth_headers()

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException:
            return False

    async def create_streaming_dataset(
        self, dataset_name: str, table_schema: dict[str, str]
    ) -> dict[str, str]:
        """Create a streaming dataset in Power BI."""
        columns = [
            {"name": name, "dataType": data_type}
            for name, data_type in table_schema.items()
        ]

        schema = {
            "name": dataset_name,
            "defaultMode": "Streaming",
            "tables": [{"name": self.config.table_name, "columns": columns}],
        }

        dataset_id = await self._ensure_dataset_exists(schema, dataset_name)

        # Get streaming URL
        base_url = f"{self.config.base_url}/{self.config.api_version}"
        if self.config.workspace_id:
            url = f"{base_url}/groups/{self.config.workspace_id}/datasets/{dataset_id}/Default.GetBoundGatewayDataSources"
        else:
            url = f"{base_url}/datasets/{dataset_id}/Default.GetBoundGatewayDataSources"

        return {
            "dataset_id": dataset_id,
            "streaming_endpoint": f"{base_url}/datasets/{dataset_id}/rows?noSignUpCheck=1",
        }
