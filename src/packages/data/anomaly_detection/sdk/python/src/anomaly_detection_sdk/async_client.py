"""Asynchronous client for Anomaly Detection API."""

import json
from typing import Dict, List, Optional, Any, Union

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .models import (
    DetectionResult,
    ModelInfo,
    ExplanationResult,
    HealthStatus,
    AlgorithmType,
    BatchProcessingRequest,
    TrainingRequest,
    TrainingResult,
)
from .exceptions import (
    APIError,
    ValidationError,
    ConnectionError,
    TimeoutError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
)


class AsyncAnomalyDetectionClient:
    """Asynchronous client for the Anomaly Detection service."""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        headers: Optional[Dict[str, str]] = None,
    ):
        """Initialize the async client.
        
        Args:
            base_url: Base URL of the anomaly detection service
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            headers: Additional headers to send with requests
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Setup headers
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        if headers:
            self.headers.update(headers)
        
        # Initialize HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=timeout,
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request with error handling and retries."""
        try:
            response = await self.client.request(
                method=method,
                url=endpoint,
                json=data,
                params=params,
            )
            
            # Handle HTTP errors
            if response.status_code == 401:
                raise AuthenticationError("Invalid API key or authentication failed")
            elif response.status_code == 404:
                raise ModelNotFoundError("Resource not found")
            elif response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                raise RateLimitError(
                    "Rate limit exceeded",
                    retry_after=int(retry_after) if retry_after else None
                )
            elif response.status_code >= 400:
                try:
                    error_data = response.json()
                except (json.JSONDecodeError, ValueError):
                    error_data = {"detail": response.text}
                
                raise APIError(
                    error_data.get("detail", f"HTTP {response.status_code}"),
                    response.status_code,
                    error_data
                )
            
            response.raise_for_status()
            return response.json()
            
        except httpx.TimeoutException:
            raise TimeoutError(f"Request timed out after {self.timeout} seconds")
        except httpx.ConnectError:
            raise ConnectionError(f"Failed to connect to {self.base_url}")
        except httpx.RequestError as e:
            raise ConnectionError(f"Request failed: {str(e)}")
    
    async def detect_anomalies(
        self,
        data: List[List[float]],
        algorithm: Union[str, AlgorithmType] = AlgorithmType.ISOLATION_FOREST,
        parameters: Optional[Dict[str, Any]] = None,
        return_explanations: bool = False,
    ) -> DetectionResult:
        """Detect anomalies in the provided data.
        
        Args:
            data: List of data points (each point is a list of features)
            algorithm: Algorithm to use for detection
            parameters: Optional algorithm parameters
            return_explanations: Whether to include explanations
            
        Returns:
            DetectionResult: Results of anomaly detection
        """
        if not data:
            raise ValidationError("Data cannot be empty", "data", data)
        
        if not all(isinstance(point, list) for point in data):
            raise ValidationError("All data points must be lists", "data", data)
        
        request_data = {
            "data": data,
            "algorithm": str(algorithm),
            "parameters": parameters or {},
            "return_explanations": return_explanations,
        }
        
        response = await self._make_request("POST", "/api/v1/detect", data=request_data)
        return DetectionResult(**response)
    
    async def batch_detect(self, request: BatchProcessingRequest) -> DetectionResult:
        """Process a batch detection request.
        
        Args:
            request: Batch processing request
            
        Returns:
            DetectionResult: Results of batch detection
        """
        response = await self._make_request(
            "POST", 
            "/api/v1/batch/detect", 
            data=request.model_dump()
        )
        return DetectionResult(**response)
    
    async def train_model(self, request: TrainingRequest) -> TrainingResult:
        """Train a new anomaly detection model.
        
        Args:
            request: Training request with data and parameters
            
        Returns:
            TrainingResult: Results of model training
        """
        response = await self._make_request(
            "POST",
            "/api/v1/models/train",
            data=request.model_dump()
        )
        return TrainingResult(**response)
    
    async def get_model(self, model_id: str) -> ModelInfo:
        """Get information about a specific model.
        
        Args:
            model_id: ID of the model to retrieve
            
        Returns:
            ModelInfo: Model information
        """
        response = await self._make_request("GET", f"/api/v1/models/{model_id}")
        return ModelInfo(**response)
    
    async def list_models(self) -> List[ModelInfo]:
        """List all available models.
        
        Returns:
            List[ModelInfo]: List of available models
        """
        response = await self._make_request("GET", "/api/v1/models")
        return [ModelInfo(**model) for model in response.get("models", [])]
    
    async def delete_model(self, model_id: str) -> Dict[str, str]:
        """Delete a model.
        
        Args:
            model_id: ID of the model to delete
            
        Returns:
            Dict[str, str]: Deletion confirmation
        """
        return await self._make_request("DELETE", f"/api/v1/models/{model_id}")
    
    async def explain_anomaly(
        self,
        data_point: List[float],
        model_id: Optional[str] = None,
        algorithm: Union[str, AlgorithmType] = AlgorithmType.ISOLATION_FOREST,
        method: str = "shap",
    ) -> ExplanationResult:
        """Get explanation for why a data point is anomalous.
        
        Args:
            data_point: The data point to explain
            model_id: Optional model ID to use for explanation
            algorithm: Algorithm to use if no model_id provided
            method: Explanation method ('shap' or 'lime')
            
        Returns:
            ExplanationResult: Explanation of the anomaly
        """
        request_data = {
            "data_point": data_point,
            "method": method,
        }
        
        if model_id:
            request_data["model_id"] = model_id
        else:
            request_data["algorithm"] = str(algorithm)
        
        response = await self._make_request("POST", "/api/v1/explain", data=request_data)
        return ExplanationResult(**response)
    
    async def get_health(self) -> HealthStatus:
        """Get service health status.
        
        Returns:
            HealthStatus: Current health status
        """
        response = await self._make_request("GET", "/api/v1/health")
        return HealthStatus(**response)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics.
        
        Returns:
            Dict[str, Any]: Service metrics
        """
        return await self._make_request("GET", "/api/v1/metrics")
    
    async def upload_data(
        self,
        data: List[List[float]],
        dataset_name: str,
        description: Optional[str] = None,
    ) -> Dict[str, str]:
        """Upload training data to the service.
        
        Args:
            data: Training data to upload
            dataset_name: Name for the dataset
            description: Optional description
            
        Returns:
            Dict[str, str]: Upload confirmation with dataset ID
        """
        request_data = {
            "data": data,
            "name": dataset_name,
            "description": description,
        }
        
        return await self._make_request("POST", "/api/v1/data/upload", data=request_data)