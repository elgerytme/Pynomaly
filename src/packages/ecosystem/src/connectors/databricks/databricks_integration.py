"""
Databricks integration connector.

This module provides comprehensive integration with Databricks for
unified analytics, data processing, and ML model management.
"""

import asyncio
import json
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncIterator
from urllib.parse import urljoin

import httpx
import structlog

from ...core.interfaces import IntegrationConfig, ConnectionHealth, AuthenticationMethod
from ...templates.mlops_platform.base_mlops_integration import MLOpsIntegrationTemplate

logger = structlog.get_logger(__name__)


class DatabricksIntegration(MLOpsIntegrationTemplate):
    """
    Databricks platform integration.
    
    Provides integration with Databricks for unified analytics platform
    including data processing, ML training, and model deployment.
    """
    
    def __init__(self, config: IntegrationConfig):
        """Initialize Databricks integration."""
        super().__init__(config)
        
        # Databricks-specific configuration
        self.workspace_url = config.endpoint
        self.access_token = config.credentials.get("access_token")
        self.cluster_id = config.credentials.get("cluster_id")
        
        # HTTP client for API calls
        self._http_client: Optional[httpx.AsyncClient] = None
        
        self.logger.info(
            "Databricks integration initialized",
            workspace_url=self.workspace_url
        )
    
    # Core connection methods
    
    async def _create_platform_client(self) -> httpx.AsyncClient:
        """Create Databricks HTTP client."""
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        self._http_client = httpx.AsyncClient(
            base_url=self.workspace_url,
            headers=headers,
            timeout=self.config.timeout_seconds
        )
        
        return self._http_client
    
    async def _authenticate_client(self) -> bool:
        """Authenticate with Databricks."""
        try:
            # Test authentication by getting workspace info
            response = await self._http_client.get("/api/2.0/workspace/get-status")
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error("Databricks authentication failed", error=str(e))
            return False
    
    async def _validate_platform_config(self, config: IntegrationConfig) -> bool:
        """Validate Databricks-specific configuration."""
        if not config.endpoint:
            self.logger.error("Databricks workspace URL is required")
            return False
        
        if not config.credentials.get("access_token"):
            self.logger.error("Databricks access token is required")
            return False
        
        if config.auth_method != AuthenticationMethod.API_KEY:
            self.logger.warning(
                "Databricks integration only supports API key authentication"
            )
        
        return True
    
    async def test_connection(self) -> ConnectionHealth:
        """Test Databricks connection health."""
        try:
            if not await self._ensure_authenticated():
                return ConnectionHealth.UNHEALTHY
            
            # Test workspace API
            response = await self._http_client.get("/api/2.0/workspace/list")
            if response.status_code != 200:
                return ConnectionHealth.DEGRADED
            
            # Test clusters API if cluster_id provided
            if self.cluster_id:
                cluster_response = await self._http_client.get(
                    f"/api/2.0/clusters/get?cluster_id={self.cluster_id}"
                )
                if cluster_response.status_code != 200:
                    return ConnectionHealth.DEGRADED
            
            return ConnectionHealth.HEALTHY
            
        except Exception as e:
            self.logger.error("Connection health check failed", error=str(e))
            return ConnectionHealth.UNHEALTHY
    
    # Data operations
    
    async def send_data(
        self,
        data: Any,
        destination: str,
        format_type: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send data to Databricks."""
        try:
            if not await self._ensure_authenticated():
                return False
            
            # Parse destination (table path or DBFS path)
            if destination.startswith("dbfs:/"):
                return await self._upload_to_dbfs(data, destination, format_type)
            else:
                return await self._write_to_table(data, destination, format_type, options)
                
        except Exception as e:
            await self._handle_api_error(e, "send_data")
            return False
    
    async def receive_data(
        self,
        source: str,
        format_type: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Receive data from Databricks."""
        try:
            if not await self._ensure_authenticated():
                return None
            
            # Parse source (table name or DBFS path)
            if source.startswith("dbfs:/"):
                return await self._download_from_dbfs(source)
            else:
                return await self._read_from_table(source, options)
                
        except Exception as e:
            await self._handle_api_error(e, "receive_data")
            return None
    
    # Experiment management
    
    async def create_experiment(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Create MLflow experiment in Databricks."""
        try:
            if not await self._ensure_authenticated():
                raise RuntimeError("Not authenticated")
            
            payload = {
                "name": name,
                "artifact_location": f"dbfs:/databricks/mlflow-tracking/{name}"
            }
            
            if tags:
                payload["tags"] = [{"key": k, "value": v} for k, v in tags.items()]
            
            response = await self._http_client.post(
                "/api/2.0/mlflow/experiments/create",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                experiment_id = result["experiment_id"]
                
                self.logger.info(
                    "Experiment created",
                    name=name,
                    experiment_id=experiment_id
                )
                
                return experiment_id
            else:
                raise RuntimeError(f"Create experiment failed: {response.text}")
                
        except Exception as e:
            await self._handle_api_error(e, "create_experiment")
            raise
    
    async def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment details."""
        try:
            if not await self._ensure_authenticated():
                return None
            
            response = await self._http_client.get(
                f"/api/2.0/mlflow/experiments/get?experiment_id={experiment_id}"
            )
            
            if response.status_code == 200:
                return response.json()["experiment"]
            else:
                return None
                
        except Exception as e:
            await self._handle_api_error(e, "get_experiment")
            return None
    
    async def list_experiments(
        self,
        filter_expr: Optional[str] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """List experiments."""
        try:
            if not await self._ensure_authenticated():
                return []
            
            params = {"max_results": max_results}
            if filter_expr:
                params["filter"] = filter_expr
            
            response = await self._http_client.get(
                "/api/2.0/mlflow/experiments/search",
                params=params
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("experiments", [])
            else:
                return []
                
        except Exception as e:
            await self._handle_api_error(e, "list_experiments")
            return []
    
    async def delete_experiment(self, experiment_id: str) -> bool:
        """Delete experiment."""
        try:
            if not await self._ensure_authenticated():
                return False
            
            response = await self._http_client.post(
                "/api/2.0/mlflow/experiments/delete",
                json={"experiment_id": experiment_id}
            )
            
            return response.status_code == 200
            
        except Exception as e:
            await self._handle_api_error(e, "delete_experiment")
            return False
    
    # Run management
    
    async def create_run(
        self,
        experiment_id: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Create run in experiment."""
        try:
            if not await self._ensure_authenticated():
                raise RuntimeError("Not authenticated")
            
            payload = {"experiment_id": experiment_id}
            
            if tags:
                payload["tags"] = [{"key": k, "value": v} for k, v in tags.items()]
            
            response = await self._http_client.post(
                "/api/2.0/mlflow/runs/create",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                run_id = result["run"]["info"]["run_id"]
                
                # Update run name if provided
                if run_name:
                    await self._update_run_name(run_id, run_name)
                
                self.logger.info(
                    "Run created",
                    experiment_id=experiment_id,
                    run_id=run_id,
                    run_name=run_name
                )
                
                return run_id
            else:
                raise RuntimeError(f"Create run failed: {response.text}")
                
        except Exception as e:
            await self._handle_api_error(e, "create_run")
            raise
    
    async def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run details."""
        try:
            if not await self._ensure_authenticated():
                return None
            
            response = await self._http_client.get(
                f"/api/2.0/mlflow/runs/get?run_id={run_id}"
            )
            
            if response.status_code == 200:
                return response.json()["run"]
            else:
                return None
                
        except Exception as e:
            await self._handle_api_error(e, "get_run")
            return None
    
    async def list_runs(
        self,
        experiment_id: str,
        filter_expr: Optional[str] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """List runs for experiment."""
        try:
            if not await self._ensure_authenticated():
                return []
            
            payload = {
                "experiment_ids": [experiment_id],
                "max_results": max_results
            }
            
            if filter_expr:
                payload["filter"] = filter_expr
            
            response = await self._http_client.post(
                "/api/2.0/mlflow/runs/search",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("runs", [])
            else:
                return []
                
        except Exception as e:
            await self._handle_api_error(e, "list_runs")
            return []
    
    async def update_run(
        self,
        run_id: str,
        status: Optional[str] = None,
        end_time: Optional[datetime] = None
    ) -> bool:
        """Update run status."""
        try:
            if not await self._ensure_authenticated():
                return False
            
            payload = {"run_id": run_id}
            
            if status:
                payload["status"] = status
            
            if end_time:
                payload["end_time"] = int(end_time.timestamp() * 1000)
            
            response = await self._http_client.post(
                "/api/2.0/mlflow/runs/update",
                json=payload
            )
            
            return response.status_code == 200
            
        except Exception as e:
            await self._handle_api_error(e, "update_run")
            return False
    
    async def delete_run(self, run_id: str) -> bool:
        """Delete run."""
        try:
            if not await self._ensure_authenticated():
                return False
            
            response = await self._http_client.post(
                "/api/2.0/mlflow/runs/delete",
                json={"run_id": run_id}
            )
            
            return response.status_code == 200
            
        except Exception as e:
            await self._handle_api_error(e, "delete_run")
            return False
    
    # Metrics and parameters
    
    async def log_metrics(
        self,
        run_id: str,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Log metrics for run."""
        try:
            if not await self._ensure_authenticated():
                return False
            
            timestamp_ms = int((timestamp or datetime.utcnow()).timestamp() * 1000)
            
            metric_list = []
            for key, value in metrics.items():
                metric_entry = {
                    "key": key,
                    "value": value,
                    "timestamp": timestamp_ms
                }
                if step is not None:
                    metric_entry["step"] = step
                
                metric_list.append(metric_entry)
            
            payload = {
                "run_id": run_id,
                "metrics": metric_list
            }
            
            response = await self._http_client.post(
                "/api/2.0/mlflow/runs/log-batch",
                json=payload
            )
            
            return response.status_code == 200
            
        except Exception as e:
            await self._handle_api_error(e, "log_metrics")
            return False
    
    async def log_parameters(
        self,
        run_id: str,
        parameters: Dict[str, Any]
    ) -> bool:
        """Log parameters for run."""
        try:
            if not await self._ensure_authenticated():
                return False
            
            param_list = [
                {"key": key, "value": str(value)}
                for key, value in parameters.items()
            ]
            
            payload = {
                "run_id": run_id,
                "params": param_list
            }
            
            response = await self._http_client.post(
                "/api/2.0/mlflow/runs/log-batch",
                json=payload
            )
            
            return response.status_code == 200
            
        except Exception as e:
            await self._handle_api_error(e, "log_parameters")
            return False
    
    async def get_metrics(
        self,
        run_id: str,
        metric_names: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get metrics for run."""
        try:
            if not await self._ensure_authenticated():
                return {}
            
            # Get run details to access metrics
            run_data = await self.get_run(run_id)
            if not run_data:
                return {}
            
            metrics_data = run_data.get("data", {}).get("metrics", {})
            
            # Filter by metric names if specified
            if metric_names:
                metrics_data = {
                    k: v for k, v in metrics_data.items()
                    if k in metric_names
                }
            
            # For detailed metric history, we'd need to call metric-history API
            result = {}
            for metric_name, latest_value in metrics_data.items():
                result[metric_name] = [
                    {
                        "value": latest_value,
                        "timestamp": datetime.utcnow().isoformat(),
                        "step": 0
                    }
                ]
            
            return result
            
        except Exception as e:
            await self._handle_api_error(e, "get_metrics")
            return {}
    
    async def get_parameters(self, run_id: str) -> Dict[str, Any]:
        """Get parameters for run."""
        try:
            if not await self._ensure_authenticated():
                return {}
            
            run_data = await self.get_run(run_id)
            if not run_data:
                return {}
            
            params = run_data.get("data", {}).get("params", {})
            return params
            
        except Exception as e:
            await self._handle_api_error(e, "get_parameters")
            return {}
    
    # Artifact management
    
    async def log_artifact(
        self,
        run_id: str,
        artifact_path: str,
        artifact_data: Any,
        artifact_type: Optional[str] = None
    ) -> bool:
        """Log artifact for run."""
        try:
            if not await self._ensure_authenticated():
                return False
            
            # Convert artifact data to bytes
            if isinstance(artifact_data, str):
                artifact_bytes = artifact_data.encode('utf-8')
            elif isinstance(artifact_data, dict):
                artifact_bytes = json.dumps(artifact_data).encode('utf-8')
            else:
                # For other types, assume it's already bytes or can be pickled
                import pickle
                artifact_bytes = pickle.dumps(artifact_data)
            
            # Upload to DBFS first
            dbfs_path = f"dbfs:/databricks/mlflow-artifacts/{run_id}/{artifact_path}"
            upload_success = await self._upload_to_dbfs(artifact_bytes, dbfs_path)
            
            if upload_success:
                self.logger.info(
                    "Artifact logged",
                    run_id=run_id,
                    artifact_path=artifact_path,
                    dbfs_path=dbfs_path
                )
                return True
            else:
                return False
                
        except Exception as e:
            await self._handle_api_error(e, "log_artifact")
            return False
    
    async def get_artifact(
        self,
        run_id: str,
        artifact_path: str
    ) -> Optional[Any]:
        """Get artifact from run."""
        try:
            if not await self._ensure_authenticated():
                return None
            
            # Download from DBFS
            dbfs_path = f"dbfs:/databricks/mlflow-artifacts/{run_id}/{artifact_path}"
            return await self._download_from_dbfs(dbfs_path)
            
        except Exception as e:
            await self._handle_api_error(e, "get_artifact")
            return None
    
    async def list_artifacts(
        self,
        run_id: str,
        path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List artifacts for run."""
        try:
            if not await self._ensure_authenticated():
                return []
            
            params = {"run_id": run_id}
            if path:
                params["path"] = path
            
            response = await self._http_client.get(
                "/api/2.0/mlflow/artifacts/list",
                params=params
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("files", [])
            else:
                return []
                
        except Exception as e:
            await self._handle_api_error(e, "list_artifacts")
            return []
    
    # Model management
    
    async def register_model(
        self,
        model_name: str,
        run_id: str,
        artifact_path: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Register model in model registry."""
        try:
            if not await self._ensure_authenticated():
                raise RuntimeError("Not authenticated")
            
            payload = {
                "name": model_name,
                "source": f"runs:/{run_id}/{artifact_path}"
            }
            
            if description:
                payload["description"] = description
            
            if tags:
                payload["tags"] = [{"key": k, "value": v} for k, v in tags.items()]
            
            response = await self._http_client.post(
                "/api/2.0/mlflow/model-versions/create",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                version = result["model_version"]["version"]
                
                self.logger.info(
                    "Model registered",
                    model_name=model_name,
                    version=version,
                    run_id=run_id
                )
                
                return version
            else:
                raise RuntimeError(f"Register model failed: {response.text}")
                
        except Exception as e:
            await self._handle_api_error(e, "register_model")
            raise
    
    async def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get registered model details."""
        try:
            if not await self._ensure_authenticated():
                return None
            
            response = await self._http_client.get(
                f"/api/2.0/mlflow/registered-models/get?name={model_name}"
            )
            
            if response.status_code == 200:
                return response.json()["registered_model"]
            else:
                return None
                
        except Exception as e:
            await self._handle_api_error(e, "get_model")
            return None
    
    async def get_model_version(
        self,
        model_name: str,
        version: str
    ) -> Optional[Dict[str, Any]]:
        """Get model version details."""
        try:
            if not await self._ensure_authenticated():
                return None
            
            response = await self._http_client.get(
                f"/api/2.0/mlflow/model-versions/get",
                params={"name": model_name, "version": version}
            )
            
            if response.status_code == 200:
                return response.json()["model_version"]
            else:
                return None
                
        except Exception as e:
            await self._handle_api_error(e, "get_model_version")
            return None
    
    async def update_model_version_stage(
        self,
        model_name: str,
        version: str,
        stage: str
    ) -> bool:
        """Update model version stage."""
        try:
            if not await self._ensure_authenticated():
                return False
            
            payload = {
                "name": model_name,
                "version": version,
                "stage": stage
            }
            
            response = await self._http_client.post(
                "/api/2.0/mlflow/model-versions/transition-stage",
                json=payload
            )
            
            return response.status_code == 200
            
        except Exception as e:
            await self._handle_api_error(e, "update_model_version_stage")
            return False
    
    async def list_model_versions(
        self,
        model_name: str,
        stages: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List model versions."""
        try:
            if not await self._ensure_authenticated():
                return []
            
            params = {"name": model_name}
            
            response = await self._http_client.get(
                "/api/2.0/mlflow/model-versions/search",
                params=params
            )
            
            if response.status_code == 200:
                result = response.json()
                versions = result.get("model_versions", [])
                
                # Filter by stages if specified
                if stages:
                    versions = [
                        v for v in versions
                        if v.get("current_stage") in stages
                    ]
                
                return versions
            else:
                return []
                
        except Exception as e:
            await self._handle_api_error(e, "list_model_versions")
            return []
    
    # Event handling
    
    async def subscribe_to_events(
        self,
        event_types: List[str],
        callback: Callable[[str, Dict[str, Any]], None]
    ) -> str:
        """Subscribe to Databricks events (webhook-based)."""
        # Databricks doesn't have native event streaming
        # This would need to be implemented using webhooks or polling
        subscription_id = str(asyncio.current_task())
        
        self.logger.info(
            "Event subscription created (polling-based)",
            subscription_id=subscription_id,
            event_types=event_types
        )
        
        return subscription_id
    
    async def unsubscribe_from_events(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        self.logger.info(
            "Event subscription cancelled",
            subscription_id=subscription_id
        )
        return True
    
    # Helper methods
    
    async def _upload_to_dbfs(
        self,
        data: bytes,
        dbfs_path: str,
        format_type: Optional[str] = None
    ) -> bool:
        """Upload data to DBFS."""
        try:
            # Create/overwrite file in DBFS
            payload = {
                "path": dbfs_path,
                "contents": base64.b64encode(data).decode('utf-8'),
                "overwrite": True
            }
            
            response = await self._http_client.post(
                "/api/2.0/dbfs/put",
                json=payload
            )
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error("DBFS upload failed", error=str(e), path=dbfs_path)
            return False
    
    async def _download_from_dbfs(self, dbfs_path: str) -> Optional[bytes]:
        """Download data from DBFS."""
        try:
            response = await self._http_client.get(
                "/api/2.0/dbfs/read",
                params={"path": dbfs_path}
            )
            
            if response.status_code == 200:
                result = response.json()
                return base64.b64decode(result["data"])
            else:
                return None
                
        except Exception as e:
            self.logger.error("DBFS download failed", error=str(e), path=dbfs_path)
            return None
    
    async def _write_to_table(
        self,
        data: Any,
        table_name: str,
        format_type: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Write data to Databricks table (requires cluster)."""
        # This would require executing SQL/Spark commands on a cluster
        # Implementation depends on specific data format and requirements
        self.logger.warning(
            "Table write not implemented",
            table_name=table_name,
            format_type=format_type
        )
        return False
    
    async def _read_from_table(
        self,
        table_name: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Read data from Databricks table (requires cluster)."""
        # This would require executing SQL/Spark commands on a cluster
        # Implementation depends on specific data format and requirements
        self.logger.warning(
            "Table read not implemented",
            table_name=table_name
        )
        return None
    
    async def _update_run_name(self, run_id: str, run_name: str) -> bool:
        """Update run name by setting a tag."""
        try:
            payload = {
                "run_id": run_id,
                "tags": [{"key": "mlflow.runName", "value": run_name}]
            }
            
            response = await self._http_client.post(
                "/api/2.0/mlflow/runs/log-batch",
                json=payload
            )
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error("Failed to update run name", error=str(e))
            return False
    
    async def disconnect(self) -> bool:
        """Close connection to Databricks."""
        try:
            if self._http_client:
                await self._http_client.aclose()
                self._http_client = None
            
            return await super().disconnect()
            
        except Exception as e:
            self.logger.error("Databricks disconnection failed", error=str(e))
            return False
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DatabricksIntegration(workspace={self.workspace_url}, "
            f"authenticated={self._authenticated})"
        )