"""
Pynomaly Python Client Implementation

This module provides the main client classes for interacting with the anomaly detection API.
Includes both synchronous and asynchronous clients with comprehensive functionality.
"""

import json
import logging
from typing import Any
from urllib.parse import urljoin

import aiohttp
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from .auth import AuthManager
from .exceptions import (
    AuthenticationError,
    AuthorizationError,
    NetworkError,
    PynomaliException,
    RateLimitError,
    ServerError,
    ValidationError,
)
from .models import *
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class BaseClient:
    """Base client with common functionality."""

    DEFAULT_BASE_URL = "https://api.pynomaly.com"
    DEFAULT_TIMEOUT = 30
    DEFAULT_MAX_RETRIES = 3

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        timeout: int = None,
        max_retries: int = None,
        rate_limit_requests: int = 100,
        rate_limit_period: int = 60,
        user_agent: str = None,
    ):
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.max_retries = max_retries or self.DEFAULT_MAX_RETRIES

        # Authentication
        self.auth_manager = AuthManager(api_key)

        # Rate limiting
        self.rate_limiter = RateLimiter(rate_limit_requests, rate_limit_period)

        # User agent
        self.user_agent = user_agent or f"pynomaly-python-sdk/{__version__}"

        # Request session (will be set in subclasses)
        self.session = None

    def _get_headers(self, additional_headers: dict[str, str] = None) -> dict[str, str]:
        """Get request headers with authentication."""
        headers = {
            "User-Agent": self.user_agent,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Add authentication headers
        auth_headers = self.auth_manager.get_auth_headers()
        headers.update(auth_headers)

        # Add any additional headers
        if additional_headers:
            headers.update(additional_headers)

        return headers

    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint."""
        return urljoin(f"{self.base_url}/", endpoint.lstrip("/"))

    def _handle_response(self, response, expected_status: int = 200):
        """Handle HTTP response and raise appropriate exceptions."""
        try:
            if response.status_code == 401:
                raise AuthenticationError("Authentication failed")
            elif response.status_code == 403:
                raise AuthorizationError("Access forbidden")
            elif response.status_code == 400:
                error_data = response.json() if response.content else {}
                raise ValidationError(error_data.get("message", "Validation error"))
            elif response.status_code == 429:
                retry_after = response.headers.get("Retry-After", 60)
                raise RateLimitError(
                    f"Rate limit exceeded. Retry after {retry_after} seconds"
                )
            elif response.status_code >= 500:
                raise ServerError(f"Server error: {response.status_code}")
            elif response.status_code != expected_status:
                raise PynomaliException(
                    f"Unexpected status code: {response.status_code}"
                )

            # Parse JSON response
            if response.content:
                return response.json()
            return None

        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Network error: {str(e)}")
        except json.JSONDecodeError as e:
            raise PynomaliException(f"Invalid JSON response: {str(e)}")


class PynomaliClient(BaseClient):
    """Synchronous client for the anomaly detection API."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Setup requests session with retry strategy
        self.session = requests.Session()

        retry_strategy = Retry(
            total=self.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE"],
            backoff_factor=1,
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Initialize API modules
        self.auth = AuthAPI(self)
        self.detection = DetectionAPI(self)
        self.training = TrainingAPI(self)
        self.datasets = DatasetsAPI(self)
        self.models = ModelsAPI(self)
        self.streaming = StreamingAPI(self)
        self.explainability = ExplainabilityAPI(self)
        self.health = HealthAPI(self)

    def request(
        self,
        method: str,
        endpoint: str,
        data: Any = None,
        params: dict[str, Any] = None,
        headers: dict[str, str] = None,
        timeout: int = None,
    ) -> Any:
        """Make HTTP request with error handling and rate limiting."""
        # Rate limiting
        self.rate_limiter.wait_if_needed()

        url = self._build_url(endpoint)
        request_headers = self._get_headers(headers)
        request_timeout = timeout or self.timeout

        logger.debug(f"Making {method} request to {url}")

        try:
            if method.upper() == "GET":
                response = self.session.get(
                    url, params=params, headers=request_headers, timeout=request_timeout
                )
            elif method.upper() == "POST":
                response = self.session.post(
                    url,
                    json=data,
                    params=params,
                    headers=request_headers,
                    timeout=request_timeout,
                )
            elif method.upper() == "PUT":
                response = self.session.put(
                    url,
                    json=data,
                    params=params,
                    headers=request_headers,
                    timeout=request_timeout,
                )
            elif method.upper() == "DELETE":
                response = self.session.delete(
                    url, params=params, headers=request_headers, timeout=request_timeout
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            return self._handle_response(response)

        except requests.exceptions.Timeout:
            raise NetworkError(f"Request timeout after {request_timeout} seconds")
        except requests.exceptions.ConnectionError:
            raise NetworkError("Connection error")

    def close(self):
        """Close the client session."""
        if self.session:
            self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AsyncPynomaliClient(BaseClient):
    """Asynchronous client for the anomaly detection API."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Setup aiohttp session
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=30),
        )

        # Initialize API modules
        self.auth = AsyncAuthAPI(self)
        self.detection = AsyncDetectionAPI(self)
        self.training = AsyncTrainingAPI(self)
        self.datasets = AsyncDatasetsAPI(self)
        self.models = AsyncModelsAPI(self)
        self.streaming = AsyncStreamingAPI(self)
        self.explainability = AsyncExplainabilityAPI(self)
        self.health = AsyncHealthAPI(self)

    async def request(
        self,
        method: str,
        endpoint: str,
        data: Any = None,
        params: dict[str, Any] = None,
        headers: dict[str, str] = None,
        timeout: int = None,
    ) -> Any:
        """Make async HTTP request with error handling and rate limiting."""
        # Rate limiting
        await self.rate_limiter.wait_if_needed_async()

        url = self._build_url(endpoint)
        request_headers = self._get_headers(headers)
        request_timeout = timeout or self.timeout

        logger.debug(f"Making async {method} request to {url}")

        try:
            async with self.session.request(
                method,
                url,
                json=data,
                params=params,
                headers=request_headers,
                timeout=aiohttp.ClientTimeout(total=request_timeout),
            ) as response:
                return await self._handle_response_async(response)

        except TimeoutError:
            raise NetworkError(f"Request timeout after {request_timeout} seconds")
        except aiohttp.ClientError as e:
            raise NetworkError(f"Network error: {str(e)}")

    async def _handle_response_async(self, response, expected_status: int = 200):
        """Handle async HTTP response and raise appropriate exceptions."""
        try:
            if response.status == 401:
                raise AuthenticationError("Authentication failed")
            elif response.status == 403:
                raise AuthorizationError("Access forbidden")
            elif response.status == 400:
                error_data = await response.json() if response.content_length else {}
                raise ValidationError(error_data.get("message", "Validation error"))
            elif response.status == 429:
                retry_after = response.headers.get("Retry-After", 60)
                raise RateLimitError(
                    f"Rate limit exceeded. Retry after {retry_after} seconds"
                )
            elif response.status >= 500:
                raise ServerError(f"Server error: {response.status}")
            elif response.status != expected_status:
                raise PynomaliException(f"Unexpected status code: {response.status}")

            # Parse JSON response
            if response.content_length:
                return await response.json()
            return None

        except aiohttp.ContentTypeError as e:
            raise PynomaliException(f"Invalid JSON response: {str(e)}")

    async def close(self):
        """Close the async client session."""
        if self.session:
            await self.session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# API Module Classes
class AuthAPI:
    """Authentication API methods."""

    def __init__(self, client: PynomaliClient):
        self.client = client

    def login(self, username: str, password: str) -> AuthToken:
        """Authenticate with username and password."""
        data = {"username": username, "password": password}
        response = self.client.request("POST", "/auth/login", data)

        # Store token in auth manager
        token = AuthToken(**response)
        self.client.auth_manager.set_jwt_token(token.access_token)

        return token

    def refresh_token(self, refresh_token: str) -> AuthToken:
        """Refresh authentication token."""
        data = {"refresh_token": refresh_token}
        response = self.client.request("POST", "/auth/refresh", data)

        token = AuthToken(**response)
        self.client.auth_manager.set_jwt_token(token.access_token)

        return token

    def logout(self) -> None:
        """Logout and invalidate token."""
        self.client.request("POST", "/auth/logout")
        self.client.auth_manager.clear_token()


class DetectionAPI:
    """Anomaly detection API methods."""

    def __init__(self, client: PynomaliClient):
        self.client = client

    def detect(
        self,
        data: list[float],
        algorithm: str = "isolation_forest",
        parameters: dict[str, Any] = None,
    ) -> DetectionResponse:
        """Detect anomalies in data."""
        request_data = DetectionRequest(
            data=data, algorithm=algorithm, parameters=parameters or {}
        )

        response = self.client.request("POST", "/detection/detect", request_data.dict())
        return DetectionResponse(**response)

    def batch_detect(
        self,
        datasets: list[list[float]],
        algorithm: str = "isolation_forest",
        parameters: dict[str, Any] = None,
    ) -> list[DetectionResponse]:
        """Detect anomalies in multiple datasets."""
        data = {
            "datasets": datasets,
            "algorithm": algorithm,
            "parameters": parameters or {},
        }

        response = self.client.request("POST", "/detection/batch", data)
        return [DetectionResponse(**r) for r in response["results"]]


class TrainingAPI:
    """Model training API methods."""

    def __init__(self, client: PynomaliClient):
        self.client = client

    def train_model(
        self,
        data: list[float],
        algorithm: str = "isolation_forest",
        parameters: dict[str, Any] = None,
        model_name: str = None,
    ) -> TrainingResponse:
        """Train a new anomaly detection model."""
        request_data = TrainingRequest(
            data=data,
            algorithm=algorithm,
            parameters=parameters or {},
            model_name=model_name,
        )

        response = self.client.request("POST", "/training/train", request_data.dict())
        return TrainingResponse(**response)


# Add async versions
class AsyncAuthAPI:
    def __init__(self, client: AsyncPynomaliClient):
        self.client = client

    async def login(self, username: str, password: str) -> AuthToken:
        data = {"username": username, "password": password}
        response = await self.client.request("POST", "/auth/login", data)

        token = AuthToken(**response)
        self.client.auth_manager.set_jwt_token(token.access_token)

        return token


class AsyncDetectionAPI:
    def __init__(self, client: AsyncPynomaliClient):
        self.client = client

    async def detect(
        self,
        data: list[float],
        algorithm: str = "isolation_forest",
        parameters: dict[str, Any] = None,
    ) -> DetectionResponse:
        request_data = DetectionRequest(
            data=data, algorithm=algorithm, parameters=parameters or {}
        )

        response = await self.client.request(
            "POST", "/detection/detect", request_data.dict()
        )
        return DetectionResponse(**response)


# Placeholder for other API classes
class AsyncTrainingAPI:
    def __init__(self, client):
        self.client = client


class DatasetsAPI:
    def __init__(self, client):
        self.client = client


class AsyncDatasetsAPI:
    def __init__(self, client):
        self.client = client


class ModelsAPI:
    def __init__(self, client):
        self.client = client


class AsyncModelsAPI:
    def __init__(self, client):
        self.client = client


class StreamingAPI:
    def __init__(self, client):
        self.client = client


class AsyncStreamingAPI:
    def __init__(self, client):
        self.client = client


class ExplainabilityAPI:
    def __init__(self, client):
        self.client = client


class AsyncExplainabilityAPI:
    def __init__(self, client):
        self.client = client


class HealthAPI:
    def __init__(self, client):
        self.client = client


class AsyncHealthAPI:
    def __init__(self, client):
        self.client = client
