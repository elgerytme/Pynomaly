"""
Pynomaly SDK Core Client

Main client class for interacting with Pynomaly API services.
Handles authentication, session management, and high-level API access.
"""

import asyncio
import json
from typing import Optional, Dict, Any, List, Union
from urllib.parse import urljoin
import httpx

from .exceptions import PynomalySDKError, AuthenticationError, APIError
from .data_science import DataScienceAPI
from .models import APIResponse


class PynomalyClient:
    """
    Main Pynomaly SDK client for data science operations.
    
    Provides authenticated access to Pynomaly services with high-level APIs
    for common data science workflows.
    
    Example:
        >>> client = PynomalyClient(
        ...     base_url="https://api.pynomaly.com",
        ...     api_key="your-api-key"
        ... )
        >>> async with client:
        ...     detectors = await client.data_science.list_detectors()
        ...     result = await client.data_science.detect_anomalies(
        ...         detector_id="detector-123",
        ...         data=your_dataframe
        ...     )
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        verify_ssl: bool = True,
        **kwargs
    ):
        """
        Initialize the Pynomaly client.
        
        Args:
            base_url: Base URL for the Pynomaly API
            api_key: API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            verify_ssl: Whether to verify SSL certificates
            **kwargs: Additional httpx client parameters
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        
        # HTTP client configuration
        headers = {
            "User-Agent": f"pynomaly-sdk/0.1.0",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
            verify=verify_ssl,
            **kwargs
        )
        
        # API interfaces
        self._data_science = None
    
    @property
    def data_science(self) -> DataScienceAPI:
        """Access to data science API operations."""
        if self._data_science is None:
            self._data_science = DataScienceAPI(self)
        return self._data_science
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check API health status.
        
        Returns:
            Dictionary containing health status information
            
        Raises:
            APIError: If health check fails
        """
        try:
            response = await self._request("GET", "/health")
            return response.data
        except Exception as e:
            raise APIError(f"Health check failed: {str(e)}") from e
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> APIResponse:
        """
        Make an authenticated API request.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            endpoint: API endpoint path
            json_data: JSON payload for request body
            params: Query parameters
            files: Files to upload
            **kwargs: Additional request parameters
            
        Returns:
            APIResponse object containing response data
            
        Raises:
            AuthenticationError: If authentication fails
            APIError: If API request fails
        """
        url = endpoint if endpoint.startswith('http') else f"/api/v1{endpoint}"
        
        for attempt in range(self.max_retries + 1):
            try:
                # Prepare request kwargs
                request_kwargs = kwargs.copy()
                if json_data is not None:
                    request_kwargs["json"] = json_data
                if params is not None:
                    request_kwargs["params"] = params
                if files is not None:
                    request_kwargs["files"] = files
                
                response = await self._client.request(method, url, **request_kwargs)
                
                # Handle authentication errors
                if response.status_code == 401:
                    raise AuthenticationError("Invalid API key or authentication failed")
                
                # Handle client errors
                if 400 <= response.status_code < 500:
                    error_detail = "Unknown client error"
                    try:
                        error_data = response.json()
                        error_detail = error_data.get("detail", error_detail)
                    except (json.JSONDecodeError, AttributeError):
                        error_detail = response.text or error_detail
                    
                    raise APIError(f"Client error ({response.status_code}): {error_detail}")
                
                # Handle server errors with retry
                if response.status_code >= 500:
                    if attempt < self.max_retries:
                        wait_time = 2 ** attempt  # Exponential backoff
                        await asyncio.sleep(wait_time)
                        continue
                    
                    raise APIError(f"Server error ({response.status_code}): {response.text}")
                
                # Parse successful response
                try:
                    data = response.json() if response.content else {}
                except json.JSONDecodeError:
                    data = {"raw_response": response.text}
                
                return APIResponse(
                    status_code=response.status_code,
                    data=data,
                    headers=dict(response.headers),
                    success=200 <= response.status_code < 300
                )
                
            except httpx.TimeoutException as e:
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise APIError(f"Request timeout after {self.timeout}s") from e
            
            except httpx.ConnectError as e:
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise APIError(f"Connection error: {str(e)}") from e
            
            except (AuthenticationError, APIError):
                # Don't retry authentication or client errors
                raise
            
            except Exception as e:
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                raise PynomalySDKError(f"Unexpected error: {str(e)}") from e
        
        # Should never reach here, but just in case
        raise APIError(f"Max retries ({self.max_retries}) exceeded")
    
    def set_api_key(self, api_key: str):
        """
        Update the API key for authentication.
        
        Args:
            api_key: New API key
        """
        self.api_key = api_key
        self._client.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def get_api_info(self) -> Dict[str, Any]:
        """
        Get client configuration information.
        
        Returns:
            Dictionary containing client configuration
        """
        return {
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "has_api_key": bool(self.api_key),
            "sdk_version": "0.1.0"
        }


# Convenience function for quick client creation
def create_client(
    base_url: str = "http://localhost:8000",
    api_key: Optional[str] = None,
    **kwargs
) -> PynomalyClient:
    """
    Create a Pynomaly client with default configuration.
    
    Args:
        base_url: Base URL for the Pynomaly API
        api_key: API key for authentication  
        **kwargs: Additional client parameters
        
    Returns:
        Configured PynomalyClient instance
        
    Example:
        >>> client = create_client(api_key="your-key")
        >>> async with client:
        ...     detectors = await client.data_science.list_detectors()
    """
    return PynomalyClient(base_url=base_url, api_key=api_key, **kwargs)