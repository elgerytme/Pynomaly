"""
Pynomaly Synchronous SDK Client

Main synchronous client for interacting with the Pynomaly API.
Provides methods for all major operations including dataset management,
detector training, anomaly detection, and experiment tracking.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import security hardening
try:
    from ...infrastructure.security.security_hardening import (
        SecurityHardeningService,
        SecurityHardeningConfig,
        TLSVersion,
        ChecksumAlgorithm
    )
    SECURITY_HARDENING_AVAILABLE = True
except ImportError:
    SECURITY_HARDENING_AVAILABLE = False

from .config import SDKConfig, load_config
from .exceptions import NetworkError, PynomaliSDKError, TimeoutError, map_http_error
from .models import (
    AlgorithmType,
    DataFormat,
    Dataset,
    DetectionResult,
    Detector,
    DetectorStatus,
    ExperimentResult,
    TrainingJob,
    numpy_to_list,
    validate_data_shape,
)

logger = logging.getLogger(__name__)


class PynomaliClient:
    """
    Synchronous client for the Pynomaly anomaly detection platform.

    Provides a comprehensive interface for dataset management, detector training,
    anomaly detection, and experiment tracking.

    Examples:
        Basic usage:
        ```python
        from pynomaly.presentation.sdk import PynomaliClient

        client = PynomaliClient(base_url="http://localhost:8000", api_key="your-key")

        # Upload a dataset
        dataset = client.create_dataset("my_data.csv", name="My Dataset")

        # Train a detector
        detector = client.train_detector(
            dataset_id=dataset.id,
            algorithm=AlgorithmType.ISOLATION_FOREST,
            parameters={"n_estimators": 100}
        )

        # Detect anomalies
        result = client.detect_anomalies(detector.id, [[1, 2, 3], [4, 5, 6]])
        ```
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        config: SDKConfig | None = None,
        config_path: str | None = None,
    ):
        """
        Initialize the Pynomaly client.

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

        # Initialize HTTP session
        self._session = self._create_session()
        
        # Initialize security hardening if available
        self._security_service = None
        if SECURITY_HARDENING_AVAILABLE and self.config.client.enforce_tls:
            security_config = SecurityHardeningConfig(
                enforce_tls=self.config.client.enforce_tls,
                minimum_tls_version=TLSVersion.TLS_1_2 if self.config.client.minimum_tls_version == "TLSv1.2" else TLSVersion.TLS_1_3,
                enable_checksum_validation=self.config.client.enable_checksum_validation,
                enable_client_side_encryption=self.config.client.enable_client_side_encryption
            )
            self._security_service = SecurityHardeningService(security_config)
            logger.info("Security hardening enabled")

        logger.info(f"Pynomaly client initialized for {self.config.base_url}")

    def _create_session(self) -> requests.Session:
        """Create configured HTTP session."""

        session = requests.Session()

        # Set up retries
        retry_strategy = Retry(
            total=self.config.client.max_retries,
            backoff_factor=self.config.client.retry_backoff_factor,
            status_forcelist=self.config.client.retry_on_status,
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.config.client.max_connections,
            pool_maxsize=self.config.client.max_connections_per_host,
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set default headers
        session.headers.update(
            {
                "User-Agent": self.config.client.user_agent,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

        # Add authentication headers
        session.headers.update(self.config.get_auth_headers())

        return session

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: dict | None = None,
        params: dict | None = None,
        files: dict | None = None,
        timeout: float | None = None,
    ) -> requests.Response:
        """Make HTTP request with error handling."""

        url = f"{self.config.api_base_url}/{endpoint.lstrip('/')}"
        timeout = timeout or self.config.client.timeout

        try:
            # Log request if enabled
            if self.config.log_requests:
                logger.debug(f"{method} {url} - Data: {data} - Params: {params}")

            # Prepare request data
            json_data = None
            if data and not files:
                json_data = data
                data = None

            response = self._session.request(
                method=method,
                url=url,
                json=json_data,
                data=data,
                params=params,
                files=files,
                timeout=timeout,
                verify=self.config.client.verify_ssl,
            )

            # Log response if enabled
            if self.config.log_responses:
                logger.debug(f"Response {response.status_code}: {response.text[:500]}")

            # Handle HTTP errors
            if not response.ok:
                self._handle_error_response(response)

            return response

        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request to {url} timed out after {timeout} seconds")

        except requests.exceptions.ConnectionError as e:
            raise NetworkError(f"Connection error: {e}")

        except requests.exceptions.RequestException as e:
            raise PynomaliSDKError(f"Request failed: {e}")

    def _handle_error_response(self, response: requests.Response) -> None:
        """Handle HTTP error responses."""

        try:
            error_data = response.json()
            message = error_data.get("message", response.reason)
            details = error_data.get("details", {})
        except (ValueError, KeyError):
            message = response.reason or f"HTTP {response.status_code}"
            details = {}

        raise map_http_error(response.status_code, message, details)

    def _parse_response(self, response: requests.Response, model_class=None):
        """Parse response JSON and optionally convert to model."""

        try:
            data = response.json()
        except ValueError:
            raise PynomaliSDKError("Invalid JSON response")

        if model_class:
            if isinstance(data, list):
                return [model_class(**item) for item in data]
            else:
                return model_class(**data)

        return data

    # Dataset Management

    def create_dataset(
        self,
        data_source: str | Path | list | dict,
        name: str,
        description: str | None = None,
        format: DataFormat = DataFormat.CSV,
        feature_names: list[str] | None = None,
    ) -> Dataset:
        """
        Create a new dataset.

        Args:
            data_source: File path, URL, or data array/dict
            name: Dataset name
            description: Optional description
            format: Data format
            feature_names: Optional feature names

        Returns:
            Created dataset object
        """

        logger.info(f"Creating dataset '{name}' from {data_source}")

        # Handle different data source types
        if isinstance(data_source, str | Path):
            # File upload
            file_path = Path(data_source)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(file_path, "rb") as f:
                file_data = f.read()
                
                # Apply security hardening if available
                if self._security_service:
                    client_info = {
                        'client_ip': '127.0.0.1',  # Would be detected from actual request
                        'user_agent': self.config.client.user_agent,
                        'sdk_version': '1.0.0',
                        'python_version': '3.8+'
                    }
                    
                    # Secure upload with encryption and checksum
                    upload_metadata = self._security_service.secure_upload(
                        file_data=file_data,
                        file_name=file_path.name,
                        content_type="application/octet-stream",
                        client_info=client_info,
                        encryption_key=self.config.client.encryption_key
                    )
                    
                    # Add security metadata to upload
                    data = {
                        "name": name,
                        "description": description or "",
                        "format": format.value,
                        "checksum": upload_metadata.checksum,
                        "checksum_algorithm": upload_metadata.checksum_algorithm.value,
                        "client_side_encrypted": upload_metadata.client_side_encrypted,
                        "encryption_key_id": upload_metadata.encryption_key_id
                    }
                else:
                    data = {
                        "name": name,
                        "description": description or "",
                        "format": format.value,
                    }
                    
                if feature_names:
                    data["feature_names"] = json.dumps(feature_names)

                files = {"file": (file_path.name, file_data, "application/octet-stream")}
                response = self._make_request(
                    "POST", "datasets", data=data, files=files
                )

        else:
            # Direct data upload
            if isinstance(data_source, list):
                # Validate data shape
                validate_data_shape(data_source)

                # Convert numpy arrays if needed
                try:
                    import numpy as np

                    if isinstance(data_source, np.ndarray):
                        data_source = numpy_to_list(data_source)
                except ImportError:
                    pass

            data = {
                "name": name,
                "description": description or "",
                "format": format.value,
                "data": data_source,
            }
            if feature_names:
                data["feature_names"] = feature_names

            response = self._make_request("POST", "datasets", data=data)

        dataset = self._parse_response(response, Dataset)
        logger.info(f"Dataset created with ID: {dataset.id}")
        return dataset

    def get_dataset(self, dataset_id: str) -> Dataset:
        """Get dataset by ID."""

        response = self._make_request("GET", f"datasets/{dataset_id}")
        return self._parse_response(response, Dataset)

    def list_datasets(
        self, limit: int = 100, offset: int = 0, search: str | None = None
    ) -> list[Dataset]:
        """List datasets with optional filtering."""

        params = {"limit": limit, "offset": offset}
        if search:
            params["search"] = search

        response = self._make_request("GET", "datasets", params=params)
        return self._parse_response(response, Dataset)

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset."""

        response = self._make_request("DELETE", f"datasets/{dataset_id}")
        return response.status_code == 204

    def download_dataset(
        self,
        dataset_id: str,
        format: DataFormat = DataFormat.CSV,
        output_path: str | Path | None = None,
    ) -> bytes | str:
        """Download dataset data."""

        params = {"format": format.value}
        response = self._make_request(
            "GET", f"datasets/{dataset_id}/download", params=params
        )

        if output_path:
            with open(output_path, "wb") as f:
                f.write(response.content)
            return str(output_path)

        return response.content

    # Detector Management

    def create_detector(
        self,
        name: str,
        algorithm: AlgorithmType,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        contamination_rate: float = 0.1,
    ) -> Detector:
        """Create a new detector."""

        data = {
            "name": name,
            "algorithm": algorithm.value,
            "description": description or "",
            "parameters": parameters or {},
            "contamination_rate": contamination_rate,
        }

        response = self._make_request("POST", "detectors", data=data)
        detector = self._parse_response(response, Detector)
        logger.info(f"Detector created with ID: {detector.id}")
        return detector

    def train_detector(
        self,
        dataset_id: str,
        algorithm: AlgorithmType,
        name: str | None = None,
        parameters: dict[str, Any] | None = None,
        contamination_rate: float = 0.1,
        wait_for_completion: bool = True,
        timeout: float | None = None,
    ) -> Detector | TrainingJob:
        """Train a detector on a dataset."""

        detector_name = name or f"{algorithm.value}_detector_{int(time.time())}"

        # Create detector first
        detector = self.create_detector(
            name=detector_name,
            algorithm=algorithm,
            parameters=parameters,
            contamination_rate=contamination_rate,
        )

        # Start training
        training_data = {"dataset_id": dataset_id, "parameters": parameters or {}}

        response = self._make_request(
            "POST", f"detectors/{detector.id}/train", data=training_data
        )
        training_job = self._parse_response(response, TrainingJob)

        logger.info(f"Training started for detector {detector.id}")

        if wait_for_completion:
            # Wait for training to complete
            detector = self._wait_for_training(detector.id, timeout)
            return detector

        return training_job

    def _wait_for_training(
        self, detector_id: str, timeout: float | None = None
    ) -> Detector:
        """Wait for detector training to complete."""

        start_time = time.time()
        timeout = timeout or 3600  # Default 1 hour timeout

        while True:
            detector = self.get_detector(detector_id)

            if detector.status == DetectorStatus.TRAINED:
                logger.info(f"Training completed for detector {detector_id}")
                return detector

            elif detector.status == DetectorStatus.FAILED:
                raise PynomaliSDKError(f"Training failed for detector {detector_id}")

            elif time.time() - start_time > timeout:
                raise TimeoutError(f"Training timeout for detector {detector_id}")

            time.sleep(5)  # Check every 5 seconds

    def get_detector(self, detector_id: str) -> Detector:
        """Get detector by ID."""

        response = self._make_request("GET", f"detectors/{detector_id}")
        return self._parse_response(response, Detector)

    def list_detectors(
        self,
        limit: int = 100,
        offset: int = 0,
        status: DetectorStatus | None = None,
        algorithm: AlgorithmType | None = None,
    ) -> list[Detector]:
        """List detectors with optional filtering."""

        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status.value
        if algorithm:
            params["algorithm"] = algorithm.value

        response = self._make_request("GET", "detectors", params=params)
        return self._parse_response(response, Detector)

    def delete_detector(self, detector_id: str) -> bool:
        """Delete a detector."""

        response = self._make_request("DELETE", f"detectors/{detector_id}")
        return response.status_code == 204

    def deploy_detector(self, detector_id: str) -> Detector:
        """Deploy a trained detector."""

        response = self._make_request("POST", f"detectors/{detector_id}/deploy")
        detector = self._parse_response(response, Detector)
        logger.info(f"Detector {detector_id} deployed")
        return detector

    # Anomaly Detection

    def detect_anomalies(
        self,
        detector_id: str,
        data: list | str,
        return_scores: bool = True,
        return_explanations: bool = False,
        batch_size: int | None = None,
    ) -> DetectionResult:
        """Detect anomalies in data."""

        # Prepare detection request
        if isinstance(data, str):
            # Dataset ID
            request_data = {
                "detector_id": detector_id,
                "dataset_id": data,
                "return_scores": return_scores,
                "return_explanations": return_explanations,
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
                "detector_id": detector_id,
                "data": data,
                "return_scores": return_scores,
                "return_explanations": return_explanations,
            }

        if batch_size:
            request_data["batch_size"] = batch_size

        response = self._make_request("POST", "detection/detect", data=request_data)
        result = self._parse_response(response, DetectionResult)

        logger.info(
            f"Detection completed: {result.num_anomalies}/{result.num_samples} anomalies"
        )
        return result

    def batch_detect(
        self,
        detector_id: str,
        data_sources: list[str | list],
        return_scores: bool = True,
        return_explanations: bool = False,
    ) -> list[DetectionResult]:
        """Perform batch detection on multiple data sources."""

        results = []
        for data_source in data_sources:
            result = self.detect_anomalies(
                detector_id=detector_id,
                data=data_source,
                return_scores=return_scores,
                return_explanations=return_explanations,
            )
            results.append(result)

        return results

    def get_detection_result(self, result_id: str) -> DetectionResult:
        """Get detection result by ID."""

        response = self._make_request("GET", f"detection/results/{result_id}")
        return self._parse_response(response, DetectionResult)

    def list_detection_results(
        self, detector_id: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[DetectionResult]:
        """List detection results."""

        params = {"limit": limit, "offset": offset}
        if detector_id:
            params["detector_id"] = detector_id

        response = self._make_request("GET", "detection/results", params=params)
        return self._parse_response(response, DetectionResult)

    # Experiment Management

    def create_experiment(
        self,
        name: str,
        dataset_id: str,
        algorithms: list[AlgorithmType],
        description: str | None = None,
        parameters: dict[str, dict[str, Any]] | None = None,
    ) -> ExperimentResult:
        """Create and run an experiment comparing multiple algorithms."""

        data = {
            "name": name,
            "dataset_id": dataset_id,
            "algorithms": [alg.value for alg in algorithms],
            "description": description or "",
            "parameters": parameters or {},
        }

        response = self._make_request("POST", "experiments", data=data)
        experiment = self._parse_response(response, ExperimentResult)
        logger.info(f"Experiment created with ID: {experiment.id}")
        return experiment

    def get_experiment(self, experiment_id: str) -> ExperimentResult:
        """Get experiment by ID."""

        response = self._make_request("GET", f"experiments/{experiment_id}")
        return self._parse_response(response, ExperimentResult)

    def list_experiments(
        self, dataset_id: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[ExperimentResult]:
        """List experiments."""

        params = {"limit": limit, "offset": offset}
        if dataset_id:
            params["dataset_id"] = dataset_id

        response = self._make_request("GET", "experiments", params=params)
        return self._parse_response(response, ExperimentResult)

    # Health and Monitoring

    def health_check(self) -> dict[str, Any]:
        """Check API health status."""

        response = self._make_request("GET", "health")
        return self._parse_response(response)

    def get_system_metrics(self) -> dict[str, Any]:
        """Get system performance metrics."""

        response = self._make_request("GET", "health/metrics")
        return self._parse_response(response)

    # Utility Methods

    def validate_connection(self) -> bool:
        """Validate connection to the API."""

        try:
            self.health_check()
            return True
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False

    def close(self) -> None:
        """Close the client session."""

        if self._session:
            self._session.close()
            logger.info("Client session closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
