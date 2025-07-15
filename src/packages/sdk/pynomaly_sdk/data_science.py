"""
Pynomaly SDK Data Science API

High-level API for data science operations including detector management,
anomaly detection, and experiment tracking.
"""

from typing import Optional, Dict, Any, List, Union
from uuid import UUID
import pandas as pd
import numpy as np

from .models import (
    DetectorConfig,
    Dataset,
    DetectionResult,
    ExperimentConfig,
    ModelMetrics,
    TrainingJob,
    PaginatedResponse
)
from .exceptions import ValidationError, APIError, ResourceNotFoundError


class DataScienceAPI:
    """
    High-level API for data science operations.
    
    Provides convenient methods for common anomaly detection workflows
    including detector management, data processing, and experiment tracking.
    """
    
    def __init__(self, client):
        """Initialize with a PynomalyClient instance."""
        self._client = client
    
    # Detector Management
    
    async def create_detector(
        self,
        name: str,
        config: DetectorConfig,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a new anomaly detector.
        
        Args:
            name: Detector name
            config: Detector configuration
            description: Optional description
            tags: Optional tags for organization
            
        Returns:
            Created detector information
            
        Raises:
            ValidationError: If configuration is invalid
            APIError: If creation fails
        """
        payload = {
            "name": name,
            "algorithm_name": config.algorithm_name,
            "hyperparameters": config.hyperparameters,
            "contamination_rate": config.contamination_rate,
            "random_state": config.random_state,
            "n_jobs": config.n_jobs
        }
        
        if description:
            payload["description"] = description
        if tags:
            payload["tags"] = tags
            
        response = await self._client._request("POST", "/detectors", json_data=payload)
        return response.data
    
    async def list_detectors(
        self,
        page: int = 1,
        page_size: int = 20,
        algorithm_name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> PaginatedResponse:
        """
        List available detectors.
        
        Args:
            page: Page number (1-based)
            page_size: Number of detectors per page
            algorithm_name: Filter by algorithm name
            tags: Filter by tags
            
        Returns:
            Paginated list of detectors
        """
        params = {
            "page": page,
            "page_size": page_size
        }
        
        if algorithm_name:
            params["algorithm_name"] = algorithm_name
        if tags:
            params["tags"] = ",".join(tags)
            
        response = await self._client._request("GET", "/detectors", params=params)
        return PaginatedResponse(**response.data)
    
    async def get_detector(self, detector_id: Union[str, UUID]) -> Dict[str, Any]:
        """
        Get detector details by ID.
        
        Args:
            detector_id: Detector ID
            
        Returns:
            Detector information
            
        Raises:
            ResourceNotFoundError: If detector not found
        """
        try:
            response = await self._client._request("GET", f"/detectors/{detector_id}")
            return response.data
        except APIError as e:
            if e.status_code == 404:
                raise ResourceNotFoundError("Detector", str(detector_id))
            raise
    
    async def update_detector(
        self,
        detector_id: Union[str, UUID],
        name: Optional[str] = None,
        config: Optional[DetectorConfig] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Update detector configuration.
        
        Args:
            detector_id: Detector ID
            name: New name (optional)
            config: New configuration (optional)
            description: New description (optional)
            tags: New tags (optional)
            
        Returns:
            Updated detector information
        """
        payload = {}
        
        if name is not None:
            payload["name"] = name
        if config is not None:
            payload.update({
                "algorithm_name": config.algorithm_name,
                "hyperparameters": config.hyperparameters,
                "contamination_rate": config.contamination_rate,
                "random_state": config.random_state,
                "n_jobs": config.n_jobs
            })
        if description is not None:
            payload["description"] = description
        if tags is not None:
            payload["tags"] = tags
            
        response = await self._client._request("PUT", f"/detectors/{detector_id}", json_data=payload)
        return response.data
    
    async def delete_detector(self, detector_id: Union[str, UUID]) -> bool:
        """
        Delete a detector.
        
        Args:
            detector_id: Detector ID
            
        Returns:
            True if deleted successfully
        """
        try:
            await self._client._request("DELETE", f"/detectors/{detector_id}")
            return True
        except APIError as e:
            if e.status_code == 404:
                raise ResourceNotFoundError("Detector", str(detector_id))
            raise
    
    # Training Operations
    
    async def train_detector(
        self,
        detector_id: Union[str, UUID],
        dataset: Union[Dataset, pd.DataFrame, np.ndarray],
        job_name: Optional[str] = None
    ) -> TrainingJob:
        """
        Train a detector with provided data.
        
        Args:
            detector_id: Detector ID to train
            dataset: Training dataset
            job_name: Optional job name
            
        Returns:
            Training job information
        """
        # Convert dataset to API format
        if isinstance(dataset, (pd.DataFrame, np.ndarray)):
            dataset = Dataset.from_dataframe("training_data", dataset) if isinstance(dataset, pd.DataFrame) else Dataset.from_numpy("training_data", dataset)
        
        payload = {
            "detector_id": str(detector_id),
            "dataset": dataset.to_dict()
        }
        
        if job_name:
            payload["job_name"] = job_name
            
        response = await self._client._request("POST", "/training/jobs", json_data=payload)
        return TrainingJob(**response.data)
    
    async def get_training_job(self, job_id: Union[str, UUID]) -> TrainingJob:
        """
        Get training job status and results.
        
        Args:
            job_id: Training job ID
            
        Returns:
            Training job information
        """
        try:
            response = await self._client._request("GET", f"/training/jobs/{job_id}")
            return TrainingJob(**response.data)
        except APIError as e:
            if e.status_code == 404:
                raise ResourceNotFoundError("Training Job", str(job_id))
            raise
    
    async def list_training_jobs(
        self,
        detector_id: Optional[Union[str, UUID]] = None,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> PaginatedResponse:
        """
        List training jobs with optional filtering.
        
        Args:
            detector_id: Filter by detector ID
            status: Filter by job status
            page: Page number
            page_size: Jobs per page
            
        Returns:
            Paginated list of training jobs
        """
        params = {
            "page": page,
            "page_size": page_size
        }
        
        if detector_id:
            params["detector_id"] = str(detector_id)
        if status:
            params["status"] = status
            
        response = await self._client._request("GET", "/training/jobs", params=params)
        return PaginatedResponse(**response.data)
    
    # Detection Operations
    
    async def detect_anomalies(
        self,
        detector_id: Union[str, UUID],
        data: Union[Dataset, pd.DataFrame, np.ndarray],
        return_scores: bool = True,
        threshold: Optional[float] = None
    ) -> DetectionResult:
        """
        Detect anomalies in data using a trained detector.
        
        Args:
            detector_id: ID of trained detector
            data: Data to analyze
            return_scores: Whether to return anomaly scores
            threshold: Custom decision threshold
            
        Returns:
            Detection results
            
        Raises:
            ValidationError: If data format is invalid
            ResourceNotFoundError: If detector not found
        """
        # Convert data to API format
        if isinstance(data, (pd.DataFrame, np.ndarray)):
            data = Dataset.from_dataframe("detection_data", data) if isinstance(data, pd.DataFrame) else Dataset.from_numpy("detection_data", data)
        
        payload = {
            "detector_id": str(detector_id),
            "dataset": data.to_dict(),
            "return_scores": return_scores
        }
        
        if threshold is not None:
            payload["threshold"] = threshold
            
        try:
            response = await self._client._request("POST", "/detection/predict", json_data=payload)
            return DetectionResult(**response.data)
        except APIError as e:
            if e.status_code == 404:
                raise ResourceNotFoundError("Detector", str(detector_id))
            raise
    
    async def batch_detect(
        self,
        detector_id: Union[str, UUID],
        datasets: List[Union[Dataset, pd.DataFrame, np.ndarray]],
        job_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit batch detection job for multiple datasets.
        
        Args:
            detector_id: ID of trained detector
            datasets: List of datasets to process
            job_name: Optional job name
            
        Returns:
            Batch job information
        """
        # Convert datasets to API format
        converted_datasets = []
        for i, dataset in enumerate(datasets):
            if isinstance(dataset, (pd.DataFrame, np.ndarray)):
                dataset = Dataset.from_dataframe(f"batch_data_{i}", dataset) if isinstance(dataset, pd.DataFrame) else Dataset.from_numpy(f"batch_data_{i}", dataset)
            converted_datasets.append(dataset.to_dict())
        
        payload = {
            "detector_id": str(detector_id),
            "datasets": converted_datasets
        }
        
        if job_name:
            payload["job_name"] = job_name
            
        response = await self._client._request("POST", "/detection/batch", json_data=payload)
        return response.data
    
    # Experiment Management
    
    async def create_experiment(
        self,
        config: ExperimentConfig,
        dataset: Union[Dataset, pd.DataFrame, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Create and run a new experiment.
        
        Args:
            config: Experiment configuration
            dataset: Dataset for the experiment
            
        Returns:
            Experiment information
        """
        # Convert dataset to API format
        if isinstance(dataset, (pd.DataFrame, np.ndarray)):
            dataset = Dataset.from_dataframe("experiment_data", dataset) if isinstance(dataset, pd.DataFrame) else Dataset.from_numpy("experiment_data", dataset)
        
        payload = {
            "name": config.name,
            "description": config.description,
            "algorithm_configs": [conf.dict() for conf in config.algorithm_configs],
            "evaluation_metrics": config.evaluation_metrics,
            "cross_validation_folds": config.cross_validation_folds,
            "random_state": config.random_state,
            "parallel_jobs": config.parallel_jobs,
            "optimization_enabled": config.optimization_enabled,
            "optimization_trials": config.optimization_trials,
            "optimization_timeout": config.optimization_timeout,
            "dataset": dataset.to_dict()
        }
        
        response = await self._client._request("POST", "/experiments", json_data=payload)
        return response.data
    
    async def get_experiment(self, experiment_id: Union[str, UUID]) -> Dict[str, Any]:
        """
        Get experiment details and results.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment information
        """
        try:
            response = await self._client._request("GET", f"/experiments/{experiment_id}")
            return response.data
        except APIError as e:
            if e.status_code == 404:
                raise ResourceNotFoundError("Experiment", str(experiment_id))
            raise
    
    async def list_experiments(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None
    ) -> PaginatedResponse:
        """
        List experiments with optional filtering.
        
        Args:
            page: Page number
            page_size: Experiments per page
            status: Filter by status
            
        Returns:
            Paginated list of experiments
        """
        params = {
            "page": page,
            "page_size": page_size
        }
        
        if status:
            params["status"] = status
            
        response = await self._client._request("GET", "/experiments", params=params)
        return PaginatedResponse(**response.data)
    
    # Utility Methods
    
    async def validate_dataset(
        self,
        dataset: Union[Dataset, pd.DataFrame, np.ndarray],
        algorithm_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate dataset format and compatibility.
        
        Args:
            dataset: Dataset to validate
            algorithm_name: Optional algorithm to check compatibility
            
        Returns:
            Validation results
        """
        # Convert dataset to API format
        if isinstance(dataset, (pd.DataFrame, np.ndarray)):
            dataset = Dataset.from_dataframe("validation_data", dataset) if isinstance(dataset, pd.DataFrame) else Dataset.from_numpy("validation_data", dataset)
        
        payload = {
            "dataset": dataset.to_dict()
        }
        
        if algorithm_name:
            payload["algorithm_name"] = algorithm_name
            
        response = await self._client._request("POST", "/validation/dataset", json_data=payload)
        return response.data
    
    async def get_supported_algorithms(self) -> List[Dict[str, Any]]:
        """
        Get list of supported algorithms and their configurations.
        
        Returns:
            List of algorithm information
        """
        response = await self._client._request("GET", "/algorithms")
        return response.data.get("algorithms", [])
    
    async def get_algorithm_info(self, algorithm_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Algorithm details and parameter information
        """
        try:
            response = await self._client._request("GET", f"/algorithms/{algorithm_name}")
            return response.data
        except APIError as e:
            if e.status_code == 404:
                raise ResourceNotFoundError("Algorithm", algorithm_name)
            raise