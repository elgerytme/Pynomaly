"""
Base MLOps platform integration template.

This module provides a standardized template for integrating with
MLOps platforms like MLflow, Kubeflow, Weights & Biases, etc.
"""

import asyncio
from abc import abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncIterator
from uuid import UUID

import structlog

from ...core.interfaces import IntegrationInterface, IntegrationConfig, DataFlowCapability
from ...core.interfaces import ConnectionHealth, IntegrationStatus, AuthenticationMethod

logger = structlog.get_logger(__name__)


class MLOpsIntegrationTemplate(IntegrationInterface):
    """
    Template for MLOps platform integrations.
    
    This template provides common functionality and patterns for
    integrating with ML lifecycle management platforms.
    """
    
    def __init__(self, config: IntegrationConfig):
        """Initialize MLOps integration."""
        super().__init__(config)
        
        # MLOps-specific configuration validation
        self._validate_mlops_config()
        
        # Initialize platform-specific clients
        self._client = None
        self._authenticated = False
        
        self.logger.info("MLOps integration template initialized")
    
    # Abstract methods that must be implemented by concrete integrations
    
    @abstractmethod
    async def _create_platform_client(self) -> Any:
        """Create platform-specific client."""
        pass
    
    @abstractmethod
    async def _authenticate_client(self) -> bool:
        """Authenticate with the platform."""
        pass
    
    @abstractmethod
    async def _validate_platform_config(self, config: IntegrationConfig) -> bool:
        """Validate platform-specific configuration."""
        pass
    
    # Experiment management
    
    @abstractmethod
    async def create_experiment(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Create new experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            tags: Experiment tags
            
        Returns:
            str: Experiment ID
        """
        pass
    
    @abstractmethod
    async def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment details.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Dict[str, Any]: Experiment details, None if not found
        """
        pass
    
    @abstractmethod
    async def list_experiments(
        self,
        filter_expr: Optional[str] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List experiments.
        
        Args:
            filter_expr: Filter expression
            max_results: Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: List of experiments
        """
        pass
    
    @abstractmethod
    async def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            bool: True if deletion successful, False otherwise
        """
        pass
    
    # Run management
    
    @abstractmethod
    async def create_run(
        self,
        experiment_id: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Create new run.
        
        Args:
            experiment_id: Parent experiment ID
            run_name: Run name
            tags: Run tags
            
        Returns:
            str: Run ID
        """
        pass
    
    @abstractmethod
    async def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get run details.
        
        Args:
            run_id: Run ID
            
        Returns:
            Dict[str, Any]: Run details, None if not found
        """
        pass
    
    @abstractmethod
    async def list_runs(
        self,
        experiment_id: str,
        filter_expr: Optional[str] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List runs for experiment.
        
        Args:
            experiment_id: Experiment ID
            filter_expr: Filter expression
            max_results: Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: List of runs
        """
        pass
    
    @abstractmethod
    async def update_run(
        self,
        run_id: str,
        status: Optional[str] = None,
        end_time: Optional[datetime] = None
    ) -> bool:
        """
        Update run status.
        
        Args:
            run_id: Run ID
            status: Run status
            end_time: Run end time
            
        Returns:
            bool: True if update successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def delete_run(self, run_id: str) -> bool:
        """
        Delete run.
        
        Args:
            run_id: Run ID
            
        Returns:
            bool: True if deletion successful, False otherwise
        """
        pass
    
    # Metrics and parameters
    
    @abstractmethod
    async def log_metrics(
        self,
        run_id: str,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Log metrics for run.
        
        Args:
            run_id: Run ID
            metrics: Metrics dictionary
            step: Step number
            timestamp: Timestamp
            
        Returns:
            bool: True if logging successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def log_parameters(
        self,
        run_id: str,
        parameters: Dict[str, Any]
    ) -> bool:
        """
        Log parameters for run.
        
        Args:
            run_id: Run ID
            parameters: Parameters dictionary
            
        Returns:
            bool: True if logging successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_metrics(
        self,
        run_id: str,
        metric_names: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get metrics for run.
        
        Args:
            run_id: Run ID
            metric_names: Specific metric names to retrieve
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Metrics data
        """
        pass
    
    @abstractmethod
    async def get_parameters(self, run_id: str) -> Dict[str, Any]:
        """
        Get parameters for run.
        
        Args:
            run_id: Run ID
            
        Returns:
            Dict[str, Any]: Parameters dictionary
        """
        pass
    
    # Artifact management
    
    @abstractmethod
    async def log_artifact(
        self,
        run_id: str,
        artifact_path: str,
        artifact_data: Any,
        artifact_type: Optional[str] = None
    ) -> bool:
        """
        Log artifact for run.
        
        Args:
            run_id: Run ID
            artifact_path: Artifact path/name
            artifact_data: Artifact data
            artifact_type: Artifact type
            
        Returns:
            bool: True if logging successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_artifact(
        self,
        run_id: str,
        artifact_path: str
    ) -> Optional[Any]:
        """
        Get artifact from run.
        
        Args:
            run_id: Run ID
            artifact_path: Artifact path/name
            
        Returns:
            Any: Artifact data, None if not found
        """
        pass
    
    @abstractmethod
    async def list_artifacts(
        self,
        run_id: str,
        path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List artifacts for run.
        
        Args:
            run_id: Run ID
            path: Artifact path filter
            
        Returns:
            List[Dict[str, Any]]: List of artifacts
        """
        pass
    
    # Model management
    
    @abstractmethod
    async def register_model(
        self,
        model_name: str,
        run_id: str,
        artifact_path: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Register model in model registry.
        
        Args:
            model_name: Model name
            run_id: Source run ID
            artifact_path: Model artifact path
            description: Model description
            tags: Model tags
            
        Returns:
            str: Model version ID
        """
        pass
    
    @abstractmethod
    async def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get registered model details.
        
        Args:
            model_name: Model name
            
        Returns:
            Dict[str, Any]: Model details, None if not found
        """
        pass
    
    @abstractmethod
    async def get_model_version(
        self,
        model_name: str,
        version: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get model version details.
        
        Args:
            model_name: Model name
            version: Model version
            
        Returns:
            Dict[str, Any]: Model version details, None if not found
        """
        pass
    
    @abstractmethod
    async def update_model_version_stage(
        self,
        model_name: str,
        version: str,
        stage: str
    ) -> bool:
        """
        Update model version stage.
        
        Args:
            model_name: Model name
            version: Model version
            stage: New stage (e.g., 'Production', 'Staging')
            
        Returns:
            bool: True if update successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def list_model_versions(
        self,
        model_name: str,
        stages: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List model versions.
        
        Args:
            model_name: Model name
            stages: Filter by stages
            
        Returns:
            List[Dict[str, Any]]: List of model versions
        """
        pass
    
    # Common implementation methods
    
    async def connect(self) -> bool:
        """Establish connection to MLOps platform."""
        try:
            await self._set_status(IntegrationStatus.CONNECTING)
            
            # Create platform client
            self._client = await self._create_platform_client()
            if not self._client:
                await self._set_status(IntegrationStatus.ERROR)
                return False
            
            # Authenticate
            auth_success = await self._authenticate_client()
            if not auth_success:
                await self._set_status(IntegrationStatus.ERROR)
                return False
            
            self._authenticated = True
            
            # Test connection
            health = await self.test_connection()
            await self._set_health(health)
            
            if health == ConnectionHealth.HEALTHY:
                await self._set_status(IntegrationStatus.CONNECTED)
                return True
            else:
                await self._set_status(IntegrationStatus.ERROR)
                return False
                
        except Exception as e:
            self.logger.error("Connection failed", error=str(e))
            await self._set_status(IntegrationStatus.ERROR)
            await self._set_health(ConnectionHealth.UNHEALTHY)
            return False
    
    async def disconnect(self) -> bool:
        """Close connection to MLOps platform."""
        try:
            self._client = None
            self._authenticated = False
            await self._set_status(IntegrationStatus.DISCONNECTED)
            await self._set_health(ConnectionHealth.UNKNOWN)
            
            self.logger.info("Disconnected from MLOps platform")
            return True
            
        except Exception as e:
            self.logger.error("Disconnection failed", error=str(e))
            return False
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get MLOps integration capabilities."""
        return {
            "platform_type": "mlops",
            "supported_operations": [
                "experiment_management",
                "run_tracking",
                "metrics_logging",
                "parameter_logging",
                "artifact_management",
                "model_registry"
            ],
            "data_capabilities": [
                DataFlowCapability.BIDIRECTIONAL.value,
                DataFlowCapability.BATCH.value
            ],
            "supported_formats": [
                "json", "pickle", "joblib", "pytorch", "tensorflow",
                "onnx", "pmml", "h5", "parquet", "csv"
            ],
            "authentication_methods": [
                AuthenticationMethod.API_KEY.value,
                AuthenticationMethod.OAUTH2.value,
                AuthenticationMethod.SERVICE_ACCOUNT.value
            ]
        }
    
    # Helper methods
    
    def _validate_mlops_config(self) -> None:
        """Validate MLOps-specific configuration."""
        required_capabilities = [
            DataFlowCapability.BIDIRECTIONAL,
            DataFlowCapability.BATCH
        ]
        
        missing_capabilities = [
            cap for cap in required_capabilities
            if cap not in self.config.data_capabilities
        ]
        
        if missing_capabilities:
            self.logger.warning(
                "MLOps integration missing required capabilities",
                missing=missing_capabilities
            )
    
    async def _ensure_authenticated(self) -> bool:
        """Ensure client is authenticated."""
        if not self._authenticated or not self._client:
            return await self.connect()
        return True
    
    async def _handle_api_error(self, error: Exception, operation: str) -> None:
        """Handle API errors with consistent logging and metrics."""
        self.logger.error(
            f"MLOps API error in {operation}",
            error=str(error),
            error_type=type(error).__name__
        )
        
        # Update metrics
        await self._update_metrics(False, 0.0)
        
        # Check if we need to reconnect
        if "auth" in str(error).lower() or "unauthorized" in str(error).lower():
            self._authenticated = False
            await self._set_status(IntegrationStatus.ERROR)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MLOpsIntegration(name={self.config.name}, "
            f"platform={self.config.platform}, "
            f"authenticated={self._authenticated})"
        )