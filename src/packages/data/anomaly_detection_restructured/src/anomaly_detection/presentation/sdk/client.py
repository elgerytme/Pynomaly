"""
Pynomaly Python SDK Client

Main client class for interacting with Pynomaly data science services.
Provides high-level API with async support and integration with popular libraries.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp
import json
from datetime import datetime
import time

# Remove services dependency - interfaces should not depend on services
# from .application.services.detection_service import DetectionService
from .application.dto.detection_dto import DetectionRequestDTO, DetectionResponseDTO
from .domain.value_objects.algorithm_config import AlgorithmConfig
from .domain.value_objects.detection_metadata import DetectionMetadata
from .infrastructure.adapters.pyod_algorithm_adapter import PyODAlgorithmAdapter
from .domain.exceptions.validation_exceptions import ValidationError

logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Configuration for Pynomaly SDK client."""
    base_url: str = "https://api.example.com/v1"
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    backoff_factor: float = 2.0
    verify_ssl: bool = True
    connection_pool_size: int = 10
    max_concurrent_requests: int = 10
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    debug: bool = False
    custom_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    backoff_factor: float = 2.0
    max_backoff: float = 60.0
    retry_on_status: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])
    retry_on_exceptions: List[type] = field(default_factory=lambda: [aiohttp.ClientError, asyncio.TimeoutError])


class PynomagyClient:
    """
    Main client for Pynomaly data science services.
    
    Provides high-level API with async support, connection management,
    error handling, and integration with popular data science libraries.
    """
    
    def __init__(self, config: Optional[ClientConfig] = None):
        """Initialize the Pynomaly client."""
        self.config = config or ClientConfig()
        self.retry_config = RetryConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._detection_service: Optional[DetectionService] = None
        self._pyod_adapter: Optional[PyODAlgorithmAdapter] = None
        self._cache: Dict[str, Any] = {}
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests)
        
        # Setup logging with security configuration
        from .core.security_configuration import get_security_config, configure_secure_logging
        
        security_config = get_security_config()
        
        # Only enable debug logging in development environment
        if self.config.debug and security_config.is_development():
            configure_secure_logging()
        elif security_config.is_production():
            # Force secure logging in production regardless of debug flag
            configure_secure_logging()
        
        logger.info(f"Pynomaly client initialized with base URL: {self.config.base_url}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._cleanup()
    
    async def _initialize_session(self):
        """Initialize aiohttp session with proper configuration."""
        connector = aiohttp.TCPConnector(
            limit=self.config.connection_pool_size,
            limit_per_host=self.config.connection_pool_size,
            verify_ssl=self.config.verify_ssl
        )
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        headers = {
            "User-Agent": "Pynomaly-Python-SDK/1.0.0",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        headers.update(self.config.custom_headers)
        
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        )
        
        # Initialize services
        self._detection_service = DetectionService()
        self._pyod_adapter = PyODAlgorithmAdapter()
        
        logger.debug("HTTP session and services initialized")
    
    async def _cleanup(self):
        """Clean up resources."""
        if self._session:
            await self._session.close()
        
        if self._executor:
            self._executor.shutdown(wait=True)
        
        logger.debug("Client resources cleaned up")
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        retry_count: int = 0
    ) -> Dict:
        """Make HTTP request with retry logic."""
        if not self._session:
            await self._initialize_session()
        
        url = f"{self.config.base_url}/{endpoint.lstrip('/')}"
        
        try:
            async with self._session.request(
                method,
                url,
                json=data,
                params=params
            ) as response:
                
                if response.status == 200:
                    return await response.json()
                elif response.status in self.retry_config.retry_on_status:
                    if retry_count < self.retry_config.max_attempts - 1:
                        delay = min(
                            self.retry_config.backoff_factor ** retry_count,
                            self.retry_config.max_backoff
                        )
                        logger.warning(f"Request failed with status {response.status}, retrying in {delay}s")
                        await asyncio.sleep(delay)
                        return await self._make_request(method, endpoint, data, params, retry_count + 1)
                    else:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status
                        )
                else:
                    response.raise_for_status()
                    
        except Exception as e:
            if (isinstance(e, tuple(self.retry_config.retry_on_exceptions)) and 
                retry_count < self.retry_config.max_attempts - 1):
                delay = min(
                    self.retry_config.backoff_factor ** retry_count,
                    self.retry_config.max_backoff
                )
                logger.warning(f"Request failed with exception {type(e).__name__}, retrying in {delay}s")
                await asyncio.sleep(delay)
                return await self._make_request(method, endpoint, data, params, retry_count + 1)
            else:
                raise
    
    def _get_cache_key(self, method: str, endpoint: str, data: Optional[Dict] = None) -> str:
        """Generate cache key for request."""
        key_parts = [method, endpoint]
        if data:
            key_parts.append(json.dumps(data, sort_keys=True))
        return ":".join(key_parts)
    
    async def _cached_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        use_cache: bool = True
    ) -> Dict:
        """Make request with caching support."""
        if not use_cache or not self.config.enable_caching:
            return await self._make_request(method, endpoint, data, params)
        
        cache_key = self._get_cache_key(method, endpoint, data)
        
        # Check cache
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.config.cache_ttl:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_data
        
        # Make request and cache result
        result = await self._make_request(method, endpoint, data, params)
        self._cache[cache_key] = (result, time.time())
        logger.debug(f"Cached result for {cache_key}")
        
        return result
    
    # Anomaly Detection Methods
    
    async def detect_anomalies(
        self,
        data: Union[pd.DataFrame, np.ndarray, List[List[float]]],
        algorithm: str = "isolation_forest",
        algorithm_params: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        use_cache: bool = True
    ) -> DetectionResponseDTO:
        """
        Detect anomalies in data using specified algorithm.
        
        Args:
            data: Input data as DataFrame, numpy array, or list of lists
            algorithm: Algorithm to use for detection
            algorithm_params: Parameters for the algorithm
            metadata: Additional metadata for the detection
            use_cache: Whether to use caching for the request
            
        Returns:
            DetectionResponseDTO with anomaly detection results
        """
        logger.info(f"Starting anomaly detection with algorithm: {algorithm}")
        
        # Convert data to appropriate format
        if isinstance(data, pd.DataFrame):
            data_array = data.values.tolist()
        elif isinstance(data, np.ndarray):
            data_array = data.tolist()
        else:
            data_array = data
        
        # Create algorithm configuration
        algo_config = AlgorithmConfig(
            name=algorithm,
            parameters=algorithm_params or {},
            version="1.0.0"
        )
        
        # Create detection metadata
        detection_metadata = DetectionMetadata(
            timestamp=datetime.now(),
            source="python_sdk",
            additional_info=metadata or {}
        )
        
        # Create detection request
        request_dto = DetectionRequestDTO(
            data=data_array,
            algorithm_config=algo_config,
            metadata=detection_metadata
        )
        
        # Use local detection service if available, otherwise make API call
        if self._detection_service:
            try:
                response = await self._detection_service.detect_anomalies(request_dto)
                logger.info("Local detection completed successfully")
                return response
            except Exception as e:
                logger.warning(f"Local detection failed: {e}, falling back to API")
        
        # Make API call
        request_data = {
            "data": data_array,
            "algorithm": algorithm,
            "algorithm_params": algorithm_params or {},
            "metadata": metadata or {}
        }
        
        result = await self._cached_request(
            "POST",
            "/anomaly-detection/detect",
            data=request_data,
            use_cache=use_cache
        )
        
        return DetectionResponseDTO(
            request_id=result.get("request_id"),
            anomaly_scores=result.get("anomaly_scores", []),
            anomaly_labels=result.get("anomaly_labels", []),
            algorithm_used=result.get("algorithm_used", algorithm),
            execution_time=result.get("execution_time", 0.0),
            metadata=result.get("metadata", {})
        )
    
    async def detect_anomalies_batch(
        self,
        datasets: List[Dict],
        max_concurrent: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[DetectionResponseDTO]:
        """
        Detect anomalies in multiple datasets concurrently.
        
        Args:
            datasets: List of dataset configurations
            max_concurrent: Maximum concurrent requests
            progress_callback: Callback for progress updates
            
        Returns:
            List of DetectionResponseDTO objects
        """
        max_concurrent = max_concurrent or self.config.max_concurrent_requests
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_dataset(dataset_config: Dict) -> DetectionResponseDTO:
            async with semaphore:
                return await self.detect_anomalies(**dataset_config)
        
        tasks = [process_dataset(config) for config in datasets]
        results = []
        
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(datasets), result)
        
        return results
    
    # Data Quality Methods
    
    async def assess_data_quality(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        quality_metrics: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> Dict:
        """
        Assess data quality using specified metrics.
        
        Args:
            data: Input data to assess
            quality_metrics: List of quality metrics to compute
            use_cache: Whether to use caching
            
        Returns:
            Dictionary containing quality assessment results
        """
        logger.info("Starting data quality assessment")
        
        # Convert data to appropriate format
        if isinstance(data, pd.DataFrame):
            data_dict = data.to_dict('records')
        elif isinstance(data, np.ndarray):
            data_dict = pd.DataFrame(data).to_dict('records')
        else:
            raise ValidationError("Data must be pandas DataFrame or numpy array")
        
        request_data = {
            "data": data_dict,
            "quality_metrics": quality_metrics or ["completeness", "uniqueness", "validity", "consistency"]
        }
        
        result = await self._cached_request(
            "POST",
            "/data-quality/assess",
            data=request_data,
            use_cache=use_cache
        )
        
        return result
    
    # Model Performance Methods
    
    async def evaluate_model_performance(
        self,
        model_id: str,
        test_data: Union[pd.DataFrame, np.ndarray],
        metrics: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> Dict:
        """
        Evaluate model performance on test data.
        
        Args:
            model_id: ID of the model to evaluate
            test_data: Test data for evaluation
            metrics: List of metrics to compute
            use_cache: Whether to use caching
            
        Returns:
            Dictionary containing performance metrics
        """
        logger.info(f"Evaluating model performance for model: {model_id}")
        
        # Convert data to appropriate format
        if isinstance(test_data, pd.DataFrame):
            data_dict = test_data.to_dict('records')
        elif isinstance(test_data, np.ndarray):
            data_dict = pd.DataFrame(test_data).to_dict('records')
        else:
            raise ValidationError("Test data must be pandas DataFrame or numpy array")
        
        request_data = {
            "model_id": model_id,
            "test_data": data_dict,
            "metrics": metrics or ["accuracy", "precision", "recall", "f1_score"]
        }
        
        result = await self._cached_request(
            "POST",
            "/model-performance/evaluate",
            data=request_data,
            use_cache=use_cache
        )
        
        return result
    
    # Utility Methods
    
    async def list_available_algorithms(self) -> List[Dict]:
        """Get list of available algorithms."""
        result = await self._cached_request("GET", "/algorithms", use_cache=True)
        return result.get("algorithms", [])
    
    async def get_algorithm_info(self, algorithm_name: str) -> Dict:
        """Get information about a specific algorithm."""
        result = await self._cached_request(
            "GET",
            f"/algorithms/{algorithm_name}",
            use_cache=True
        )
        return result
    
    async def health_check(self) -> Dict:
        """Check API health status."""
        result = await self._cached_request("GET", "/health", use_cache=False)
        return result
    
    def clear_cache(self):
        """Clear the request cache."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_enabled": self.config.enable_caching,
            "cache_ttl": self.config.cache_ttl
        }


# Convenience function for creating client
def create_client(
    api_key: Optional[str] = None,
    base_url: str = "https://api.example.com/v1",
    **kwargs
) -> PynomagyClient:
    """
    Create a Pynomaly client with convenient configuration.
    
    Args:
        api_key: API key for authentication
        base_url: Base URL for the API
        **kwargs: Additional configuration options
        
    Returns:
        Configured PynomagyClient instance
    """
    config = ClientConfig(
        api_key=api_key,
        base_url=base_url,
        **kwargs
    )
    return PynomagyClient(config)