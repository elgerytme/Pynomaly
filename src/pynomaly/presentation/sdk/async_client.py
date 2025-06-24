"""
Pynomaly Asynchronous SDK Client

High-performance asynchronous client for the Pynomaly API.
Provides async/await support for all operations with concurrent processing capabilities.
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional, Union, AsyncIterator
from pathlib import Path
import logging

import aiohttp
import aiofiles
from aiohttp import ClientSession, ClientTimeout, TCPConnector

from .config import SDKConfig, load_config
from .models import (
    Dataset, Detector, DetectionResult, TrainingJob, ExperimentResult,
    AnomalyScore, PerformanceMetrics, BatchDetectionRequest,
    AlgorithmType, DataFormat, TaskStatus, DetectorStatus,
    validate_data_shape, numpy_to_list
)
from .exceptions import (
    PynomaliSDKError, AuthenticationError, ValidationError,
    ResourceNotFoundError, ServerError, TimeoutError, NetworkError,
    map_http_error
)


logger = logging.getLogger(__name__)


class AsyncPynomaliClient:
    """
    Asynchronous client for the Pynomaly anomaly detection platform.
    
    Provides high-performance async/await interfaces for all operations
    with support for concurrent processing and streaming.
    
    Examples:
        Basic usage:
        ```python
        import asyncio
        from pynomaly.presentation.sdk import AsyncPynomaliClient
        
        async def main():
            async with AsyncPynomaliClient(base_url="http://localhost:8000", 
                                         api_key="your-key") as client:
                # Upload dataset
                dataset = await client.create_dataset("data.csv", name="My Dataset")
                
                # Train detector
                detector = await client.train_detector(
                    dataset_id=dataset.id,
                    algorithm=AlgorithmType.ISOLATION_FOREST
                )
                
                # Concurrent detection on multiple datasets
                data_batches = [[[1, 2]], [[3, 4]], [[5, 6]]]
                results = await client.batch_detect_concurrent(detector.id, data_batches)
        
        asyncio.run(main())
        ```
    """
    
    def __init__(self,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 config: Optional[SDKConfig] = None,
                 config_path: Optional[str] = None):
        """
        Initialize the async Pynomaly client.
        
        Args:
            base_url: Base URL of the Pynomaly API
            api_key: API key for authentication
            config: SDKConfig instance
            config_path: Path to configuration file
        """
        
        # Load configuration
        if config:
            self.config = config
        elif config_path:
            self.config = SDKConfig.from_file(config_path)
        else:
            self.config = load_config()
        
        # Override with provided values
        if base_url:
            self.config.base_url = base_url
        if api_key:
            self.config.api_key = api_key
        
        # Validate configuration
        self.config.validate()
        self.config.setup_logging()
        
        # Session will be created when needed
        self._session: Optional[ClientSession] = None
        self._closed = False
        
        logger.info(f"Async Pynomaly client initialized for {self.config.base_url}")
    
    async def _ensure_session(self) -> ClientSession:
        """Ensure HTTP session is created."""
        
        if self._session is None or self._session.closed:
            self._session = await self._create_session()
        
        return self._session
    
    async def _create_session(self) -> ClientSession:
        """Create configured HTTP session."""
        
        # Configure connector
        connector = TCPConnector(
            limit=self.config.client.max_connections,
            limit_per_host=self.config.client.max_connections_per_host,
            verify_ssl=self.config.client.verify_ssl,
            enable_cleanup_closed=True
        )
        
        # Configure timeout
        timeout = ClientTimeout(total=self.config.client.timeout)
        
        # Create session
        session = ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "User-Agent": self.config.client.user_agent,
                "Content-Type": "application/json",
                "Accept": "application/json",
                **self.config.get_auth_headers()
            }
        )
        
        return session
    
    async def _make_request(self,
                           method: str,
                           endpoint: str,
                           data: Optional[Dict] = None,
                           params: Optional[Dict] = None,
                           timeout: Optional[float] = None) -> aiohttp.ClientResponse:
        """Make async HTTP request with error handling."""
        
        session = await self._ensure_session()
        url = f"{self.config.api_base_url}/{endpoint.lstrip('/')}"
        
        # Override timeout if specified
        if timeout:
            timeout_obj = ClientTimeout(total=timeout)
        else:
            timeout_obj = None
        
        try:
            # Log request if enabled
            if self.config.log_requests:
                logger.debug(f"{method} {url} - Data: {data} - Params: {params}")
            
            async with session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                timeout=timeout_obj
            ) as response:
                
                # Log response if enabled
                if self.config.log_responses:
                    text = await response.text()
                    logger.debug(f"Response {response.status}: {text[:500]}")
                
                # Handle HTTP errors
                if not response.ok:
                    await self._handle_error_response(response)
                
                return response
                
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request to {url} timed out")
        
        except aiohttp.ClientConnectionError as e:
            raise NetworkError(f"Connection error: {e}")
        
        except aiohttp.ClientError as e:
            raise PynomaliSDKError(f"Request failed: {e}")
    
    async def _handle_error_response(self, response: aiohttp.ClientResponse) -> None:
        """Handle HTTP error responses."""
        
        try:
            error_data = await response.json()
            message = error_data.get("message", response.reason)
            details = error_data.get("details", {})
        except (ValueError, KeyError):
            message = response.reason or f"HTTP {response.status}"
            details = {}
        
        raise map_http_error(response.status, message, details)
    
    async def _parse_response(self, response: aiohttp.ClientResponse, model_class=None):
        """Parse response JSON and optionally convert to model."""
        
        try:
            data = await response.json()
        except ValueError:
            raise PynomaliSDKError("Invalid JSON response")
        
        if model_class:
            if isinstance(data, list):
                return [model_class(**item) for item in data]
            else:
                return model_class(**data)
        
        return data
    
    # Dataset Management
    
    async def create_dataset(self,
                           data_source: Union[str, Path, List, Dict],
                           name: str,
                           description: Optional[str] = None,
                           format: DataFormat = DataFormat.CSV,
                           feature_names: Optional[List[str]] = None) -> Dataset:
        """Create a new dataset asynchronously."""
        
        logger.info(f"Creating dataset '{name}' from {data_source}")
        
        session = await self._ensure_session()
        
        # Handle different data source types
        if isinstance(data_source, (str, Path)):
            # File upload
            file_path = Path(data_source)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            async with aiofiles.open(file_path, 'rb') as f:
                file_content = await f.read()
            
            # Create multipart form data
            data = aiohttp.FormData()
            data.add_field('name', name)
            data.add_field('description', description or '')
            data.add_field('format', format.value)
            if feature_names:
                data.add_field('feature_names', json.dumps(feature_names))
            data.add_field('file', file_content, filename=file_path.name)
            
            url = f"{self.config.api_base_url}/datasets"
            
            async with session.post(url, data=data) as response:
                if not response.ok:
                    await self._handle_error_response(response)
                dataset_data = await response.json()
        
        else:
            # Direct data upload
            if isinstance(data_source, list):
                validate_data_shape(data_source)
                
                # Convert numpy arrays if needed
                try:
                    import numpy as np
                    if isinstance(data_source, np.ndarray):
                        data_source = numpy_to_list(data_source)
                except ImportError:
                    pass
            
            request_data = {
                'name': name,
                'description': description or '',
                'format': format.value,
                'data': data_source
            }
            if feature_names:
                request_data['feature_names'] = feature_names
            
            response = await self._make_request('POST', 'datasets', data=request_data)
            dataset_data = await self._parse_response(response)
        
        dataset = Dataset(**dataset_data)
        logger.info(f"Dataset created with ID: {dataset.id}")
        return dataset
    
    async def get_dataset(self, dataset_id: str) -> Dataset:
        """Get dataset by ID."""
        
        response = await self._make_request('GET', f'datasets/{dataset_id}')
        return await self._parse_response(response, Dataset)
    
    async def list_datasets(self,
                          limit: int = 100,
                          offset: int = 0,
                          search: Optional[str] = None) -> List[Dataset]:
        """List datasets with optional filtering."""
        
        params = {'limit': limit, 'offset': offset}
        if search:
            params['search'] = search
        
        response = await self._make_request('GET', 'datasets', params=params)
        return await self._parse_response(response, Dataset)
    
    async def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset."""
        
        response = await self._make_request('DELETE', f'datasets/{dataset_id}')
        return response.status == 204
    
    async def download_dataset(self,
                             dataset_id: str,
                             format: DataFormat = DataFormat.CSV,
                             output_path: Optional[Union[str, Path]] = None) -> Union[bytes, str]:
        """Download dataset data."""
        
        params = {'format': format.value}
        response = await self._make_request('GET', f'datasets/{dataset_id}/download', params=params)
        
        content = await response.read()
        
        if output_path:
            async with aiofiles.open(output_path, 'wb') as f:
                await f.write(content)
            return str(output_path)
        
        return content
    
    # Detector Management
    
    async def create_detector(self,
                            name: str,
                            algorithm: AlgorithmType,
                            description: Optional[str] = None,
                            parameters: Optional[Dict[str, Any]] = None,
                            contamination_rate: float = 0.1) -> Detector:
        """Create a new detector."""
        
        data = {
            'name': name,
            'algorithm': algorithm.value,
            'description': description or '',
            'parameters': parameters or {},
            'contamination_rate': contamination_rate
        }
        
        response = await self._make_request('POST', 'detectors', data=data)
        detector = await self._parse_response(response, Detector)
        logger.info(f"Detector created with ID: {detector.id}")
        return detector
    
    async def train_detector(self,
                           dataset_id: str,
                           algorithm: AlgorithmType,
                           name: Optional[str] = None,
                           parameters: Optional[Dict[str, Any]] = None,
                           contamination_rate: float = 0.1,
                           wait_for_completion: bool = True,
                           timeout: Optional[float] = None) -> Union[Detector, TrainingJob]:
        """Train a detector on a dataset."""
        
        detector_name = name or f"{algorithm.value}_detector_{int(time.time())}"
        
        # Create detector first
        detector = await self.create_detector(
            name=detector_name,
            algorithm=algorithm,
            parameters=parameters,
            contamination_rate=contamination_rate
        )
        
        # Start training
        training_data = {
            'dataset_id': dataset_id,
            'parameters': parameters or {}
        }
        
        response = await self._make_request('POST', f'detectors/{detector.id}/train', data=training_data)
        training_job = await self._parse_response(response, TrainingJob)
        
        logger.info(f"Training started for detector {detector.id}")
        
        if wait_for_completion:
            # Wait for training to complete
            detector = await self._wait_for_training(detector.id, timeout)
            return detector
        
        return training_job
    
    async def _wait_for_training(self, detector_id: str, timeout: Optional[float] = None) -> Detector:
        """Wait for detector training to complete."""
        
        start_time = time.time()
        timeout = timeout or 3600  # Default 1 hour timeout
        
        while True:
            detector = await self.get_detector(detector_id)
            
            if detector.status == DetectorStatus.TRAINED:
                logger.info(f"Training completed for detector {detector_id}")
                return detector
            
            elif detector.status == DetectorStatus.FAILED:
                raise PynomaliSDKError(f"Training failed for detector {detector_id}")
            
            elif time.time() - start_time > timeout:
                raise TimeoutError(f"Training timeout for detector {detector_id}")
            
            await asyncio.sleep(5)  # Check every 5 seconds
    
    async def get_detector(self, detector_id: str) -> Detector:
        """Get detector by ID."""
        
        response = await self._make_request('GET', f'detectors/{detector_id}')
        return await self._parse_response(response, Detector)
    
    async def list_detectors(self,
                           limit: int = 100,
                           offset: int = 0,
                           status: Optional[DetectorStatus] = None,
                           algorithm: Optional[AlgorithmType] = None) -> List[Detector]:
        """List detectors with optional filtering."""
        
        params = {'limit': limit, 'offset': offset}
        if status:
            params['status'] = status.value
        if algorithm:
            params['algorithm'] = algorithm.value
        
        response = await self._make_request('GET', 'detectors', params=params)
        return await self._parse_response(response, Detector)
    
    async def delete_detector(self, detector_id: str) -> bool:
        """Delete a detector."""
        
        response = await self._make_request('DELETE', f'detectors/{detector_id}')
        return response.status == 204
    
    async def deploy_detector(self, detector_id: str) -> Detector:
        """Deploy a trained detector."""
        
        response = await self._make_request('POST', f'detectors/{detector_id}/deploy')
        detector = await self._parse_response(response, Detector)
        logger.info(f"Detector {detector_id} deployed")
        return detector
    
    # Anomaly Detection
    
    async def detect_anomalies(self,
                             detector_id: str,
                             data: Union[List, str],
                             return_scores: bool = True,
                             return_explanations: bool = False,
                             batch_size: Optional[int] = None) -> DetectionResult:
        """Detect anomalies in data."""
        
        # Prepare detection request
        if isinstance(data, str):
            # Dataset ID
            request_data = {
                'detector_id': detector_id,
                'dataset_id': data,
                'return_scores': return_scores,
                'return_explanations': return_explanations
            }
        else:
            # Direct data
            validate_data_shape(data)
            
            # Convert numpy arrays if needed
            try:
                import numpy as np
                if isinstance(data, np.ndarray):
                    data = numpy_to_list(data)
            except ImportError:
                pass
            
            request_data = {
                'detector_id': detector_id,
                'data': data,
                'return_scores': return_scores,
                'return_explanations': return_explanations
            }
        
        if batch_size:
            request_data['batch_size'] = batch_size
        
        response = await self._make_request('POST', 'detection/detect', data=request_data)
        result = await self._parse_response(response, DetectionResult)
        
        logger.info(f"Detection completed: {result.num_anomalies}/{result.num_samples} anomalies")
        return result
    
    async def batch_detect_concurrent(self,
                                    detector_id: str,
                                    data_sources: List[Union[str, List]],
                                    return_scores: bool = True,
                                    return_explanations: bool = False,
                                    max_concurrent: int = 10) -> List[DetectionResult]:
        """Perform concurrent batch detection on multiple data sources."""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def detect_single(data_source):
            async with semaphore:
                return await self.detect_anomalies(
                    detector_id=detector_id,
                    data=data_source,
                    return_scores=return_scores,
                    return_explanations=return_explanations
                )
        
        tasks = [detect_single(data_source) for data_source in data_sources]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def stream_detection(self,
                             detector_id: str,
                             data_stream: AsyncIterator[List],
                             buffer_size: int = 100,
                             return_scores: bool = True) -> AsyncIterator[DetectionResult]:
        """Stream detection results for continuous data processing."""
        
        buffer = []
        
        async for data_batch in data_stream:
            buffer.extend(data_batch)
            
            if len(buffer) >= buffer_size:
                # Process buffer
                result = await self.detect_anomalies(
                    detector_id=detector_id,
                    data=buffer,
                    return_scores=return_scores
                )
                yield result
                buffer = []
        
        # Process remaining data in buffer
        if buffer:
            result = await self.detect_anomalies(
                detector_id=detector_id,
                data=buffer,
                return_scores=return_scores
            )
            yield result
    
    async def get_detection_result(self, result_id: str) -> DetectionResult:
        """Get detection result by ID."""
        
        response = await self._make_request('GET', f'detection/results/{result_id}')
        return await self._parse_response(response, DetectionResult)
    
    async def list_detection_results(self,
                                   detector_id: Optional[str] = None,
                                   limit: int = 100,
                                   offset: int = 0) -> List[DetectionResult]:
        """List detection results."""
        
        params = {'limit': limit, 'offset': offset}
        if detector_id:
            params['detector_id'] = detector_id
        
        response = await self._make_request('GET', 'detection/results', params=params)
        return await self._parse_response(response, DetectionResult)
    
    # Experiment Management
    
    async def create_experiment(self,
                              name: str,
                              dataset_id: str,
                              algorithms: List[AlgorithmType],
                              description: Optional[str] = None,
                              parameters: Optional[Dict[str, Dict[str, Any]]] = None) -> ExperimentResult:
        """Create and run an experiment comparing multiple algorithms."""
        
        data = {
            'name': name,
            'dataset_id': dataset_id,
            'algorithms': [alg.value for alg in algorithms],
            'description': description or '',
            'parameters': parameters or {}
        }
        
        response = await self._make_request('POST', 'experiments', data=data)
        experiment = await self._parse_response(response, ExperimentResult)
        logger.info(f"Experiment created with ID: {experiment.id}")
        return experiment
    
    async def get_experiment(self, experiment_id: str) -> ExperimentResult:
        """Get experiment by ID."""
        
        response = await self._make_request('GET', f'experiments/{experiment_id}')
        return await self._parse_response(response, ExperimentResult)
    
    async def list_experiments(self,
                             dataset_id: Optional[str] = None,
                             limit: int = 100,
                             offset: int = 0) -> List[ExperimentResult]:
        """List experiments."""
        
        params = {'limit': limit, 'offset': offset}
        if dataset_id:
            params['dataset_id'] = dataset_id
        
        response = await self._make_request('GET', 'experiments', params=params)
        return await self._parse_response(response, ExperimentResult)
    
    # Health and Monitoring
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        
        response = await self._make_request('GET', 'health')
        return await self._parse_response(response)
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        
        response = await self._make_request('GET', 'health/metrics')
        return await self._parse_response(response)
    
    # Utility Methods
    
    async def validate_connection(self) -> bool:
        """Validate connection to the API."""
        
        try:
            await self.health_check()
            return True
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False
    
    async def close(self) -> None:
        """Close the client session."""
        
        if self._session and not self._session.closed:
            await self._session.close()
            self._closed = True
            logger.info("Async client session closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()