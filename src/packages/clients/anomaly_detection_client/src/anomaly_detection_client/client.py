"""Anomaly detection client implementation."""

import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from sdk_core import BaseClient, SyncClient, ClientConfig, Environment, JWTAuth
from anomaly_detection_client.models import (
    DetectionRequest,
    DetectionResponse,
    EnsembleDetectionRequest,
    EnsembleDetectionResponse,
    ModelInfo,
    TrainingRequest,
    TrainingResponse,
    PredictionRequest,
    PredictionResponse,
    AlgorithmInfo,
    BatchDetectionRequest,
    BatchDetectionResponse,
)


class AnomalyDetectionClient:
    """Async client for anomaly detection service."""
    
    def __init__(
        self,
        config: Optional[ClientConfig] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        environment: Optional[Environment] = None,
    ):
        # Create config if not provided
        if config is None:
            api_key = api_key or os.getenv("ANOMALY_DETECTION_API_KEY")
            base_url = base_url or os.getenv("ANOMALY_DETECTION_BASE_URL")
            environment = environment or Environment.PRODUCTION
            
            if not api_key:
                raise ValueError("API key is required. Provide via api_key parameter or ANOMALY_DETECTION_API_KEY env var.")
            
            if base_url:
                config = ClientConfig(base_url=base_url, api_key=api_key)
            else:
                config = ClientConfig.for_environment(environment, api_key=api_key)
        
        # Set up authentication
        if config.api_key:
            auth = JWTAuth(api_key=config.api_key, base_url=config.base_url)
        else:
            auth = None
        
        self._client = BaseClient(config, auth)
        self._config = config
    
    async def detect(
        self,
        data: Union[List[List[float]], np.ndarray, pd.DataFrame],
        algorithm: str = "isolation_forest",
        contamination: float = 0.1,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> DetectionResponse:
        """Detect anomalies in data using specified algorithm."""
        
        # Convert data to proper format
        data_array = self._prepare_data(data)
        
        request = DetectionRequest(
            data=data_array,
            algorithm=algorithm,
            contamination=contamination,
            parameters=parameters or {},
        )
        
        response = await self._client.post(
            f"{self._config.api_base_url}detect",
            json_data=request.dict(),
        )
        
        return DetectionResponse(**response)
    
    async def detect_ensemble(
        self,
        data: Union[List[List[float]], np.ndarray, pd.DataFrame],
        algorithms: List[str],
        voting_strategy: str = "majority",
        contamination: float = 0.1,
        algorithm_parameters: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> EnsembleDetectionResponse:
        """Detect anomalies using ensemble of algorithms."""
        
        data_array = self._prepare_data(data)
        
        request = EnsembleDetectionRequest(
            data=data_array,
            algorithms=algorithms,
            voting_strategy=voting_strategy,
            contamination=contamination,
            algorithm_parameters=algorithm_parameters or {},
        )
        
        response = await self._client.post(
            f"{self._config.api_base_url}ensemble",
            json_data=request.dict(),
        )
        
        return EnsembleDetectionResponse(**response)
    
    async def train_model(
        self,
        data: Union[List[List[float]], np.ndarray, pd.DataFrame],
        algorithm: str,
        name: str,
        description: Optional[str] = None,
        contamination: float = 0.1,
        parameters: Optional[Dict[str, Any]] = None,
        validation_split: float = 0.2,
    ) -> TrainingResponse:
        """Train a new anomaly detection model."""
        
        data_array = self._prepare_data(data)
        
        request = TrainingRequest(
            data=data_array,
            algorithm=algorithm,
            name=name,
            description=description,
            contamination=contamination,
            parameters=parameters or {},
            validation_split=validation_split,
        )
        
        response = await self._client.post(
            f"{self._config.api_base_url}models/train",
            json_data=request.dict(),
        )
        
        return TrainingResponse(**response)
    
    async def predict(
        self,
        data: Union[List[List[float]], np.ndarray, pd.DataFrame],
        model_id: str,
    ) -> PredictionResponse:
        """Make predictions using a trained model."""
        
        data_array = self._prepare_data(data)
        
        request = PredictionRequest(
            data=data_array,
            model_id=model_id,
        )
        
        response = await self._client.post(
            f"{self._config.api_base_url}predict",
            json_data=request.dict(),
        )
        
        return PredictionResponse(**response)
    
    async def get_model(self, model_id: str) -> ModelInfo:
        """Get information about a specific model."""
        response = await self._client.get(f"{self._config.api_base_url}models/{model_id}")
        return ModelInfo(**response["data"])
    
    async def list_models(
        self,
        algorithm: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Dict[str, Any]:
        """List available models."""
        params = {"page": page, "page_size": page_size}
        if algorithm:
            params["algorithm"] = algorithm
        if status:
            params["status"] = status
        
        response = await self._client.get(
            f"{self._config.api_base_url}models",
            params=params,
        )
        
        return response
    
    async def delete_model(self, model_id: str) -> Dict[str, Any]:
        """Delete a model."""
        response = await self._client.delete(f"{self._config.api_base_url}models/{model_id}")
        return response
    
    async def get_algorithms(self) -> List[AlgorithmInfo]:
        """Get list of available algorithms."""
        response = await self._client.get(f"{self._config.api_base_url}algorithms")
        return [AlgorithmInfo(**algo) for algo in response["data"]]
    
    async def batch_detect(
        self,
        datasets: List[Dict[str, Any]],
        algorithm: str = "isolation_forest",
        contamination: float = 0.1,
        parameters: Optional[Dict[str, Any]] = None,
        parallel_processing: bool = True,
    ) -> BatchDetectionResponse:
        """Process multiple datasets in batch."""
        
        request = BatchDetectionRequest(
            datasets=datasets,
            algorithm=algorithm,
            contamination=contamination,
            parameters=parameters or {},
            parallel_processing=parallel_processing,
        )
        
        response = await self._client.post(
            f"{self._config.api_base_url}batch/detect",
            json_data=request.dict(),
        )
        
        return BatchDetectionResponse(**response)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        return await self._client.health_check()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        return await self._client.get(f"{self._config.api_base_url}metrics")
    
    def _prepare_data(self, data: Union[List[List[float]], np.ndarray, pd.DataFrame]) -> List[List[float]]:
        """Convert input data to the expected format."""
        
        if isinstance(data, pd.DataFrame):
            return data.values.tolist()
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, list):
            # Ensure it's a 2D list
            if data and not isinstance(data[0], list):
                # Convert 1D to 2D
                return [[x] for x in data]
            return data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    async def close(self):
        """Close the client."""
        await self._client.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class AnomalyDetectionSyncClient:
    """Synchronous client for anomaly detection service."""
    
    def __init__(self, **kwargs):
        self._async_client = AnomalyDetectionClient(**kwargs)
        self._sync_client = SyncClient(
            self._async_client._config,
            self._async_client._client.auth,
        )
    
    def detect(self, **kwargs) -> DetectionResponse:
        """Synchronous version of detect."""
        import asyncio
        return asyncio.run(self._async_client.detect(**kwargs))
    
    def detect_ensemble(self, **kwargs) -> EnsembleDetectionResponse:
        """Synchronous version of detect_ensemble."""
        import asyncio
        return asyncio.run(self._async_client.detect_ensemble(**kwargs))
    
    def train_model(self, **kwargs) -> TrainingResponse:
        """Synchronous version of train_model."""
        import asyncio
        return asyncio.run(self._async_client.train_model(**kwargs))
    
    def predict(self, **kwargs) -> PredictionResponse:
        """Synchronous version of predict."""
        import asyncio
        return asyncio.run(self._async_client.predict(**kwargs))
    
    def get_model(self, model_id: str) -> ModelInfo:
        """Synchronous version of get_model."""
        import asyncio
        return asyncio.run(self._async_client.get_model(model_id))
    
    def list_models(self, **kwargs) -> Dict[str, Any]:
        """Synchronous version of list_models."""
        import asyncio
        return asyncio.run(self._async_client.list_models(**kwargs))
    
    def delete_model(self, model_id: str) -> Dict[str, Any]:
        """Synchronous version of delete_model."""
        import asyncio
        return asyncio.run(self._async_client.delete_model(model_id))
    
    def get_algorithms(self) -> List[AlgorithmInfo]:
        """Synchronous version of get_algorithms."""
        import asyncio
        return asyncio.run(self._async_client.get_algorithms())
    
    def batch_detect(self, **kwargs) -> BatchDetectionResponse:
        """Synchronous version of batch_detect."""
        import asyncio
        return asyncio.run(self._async_client.batch_detect(**kwargs))
    
    def health_check(self) -> Dict[str, Any]:
        """Synchronous version of health_check."""
        return self._sync_client.health_check()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Synchronous version of get_metrics."""
        return self._sync_client.get(f"{self._async_client._config.api_base_url}metrics")
    
    def close(self):
        """Close the client."""
        self._sync_client.close()
    
    def __enter__(self):
        """Sync context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit."""
        self.close()