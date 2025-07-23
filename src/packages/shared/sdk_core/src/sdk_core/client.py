"""HTTP client implementation for SDK."""

import asyncio
import json
from typing import Any, Dict, List, Optional, Union

import httpx
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from sdk_core.auth import AuthHandler
from sdk_core.config import ClientConfig
from sdk_core.exceptions import (
    ConnectionError,
    TimeoutError,
    create_exception_from_response,
)
from sdk_core.models import BaseResponse, ErrorResponse
from sdk_core.utils import RateLimiter


logger = structlog.get_logger(__name__)


class BaseClient:
    """Base HTTP client for SDK operations."""
    
    def __init__(self, config: ClientConfig, auth: Optional[AuthHandler] = None):
        self.config = config
        self.auth = auth
        self._client: Optional[httpx.AsyncClient] = None
        self._rate_limiter = RateLimiter(
            requests=config.rate_limit_requests,
            period=config.rate_limit_period,
        )
        self._logger = logger.bind(
            client=self.__class__.__name__,
            base_url=config.base_url,
        )
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            limits = httpx.Limits(
                max_connections=self.config.connection_pool_size,
                max_keepalive_connections=self.config.max_keepalive_connections,
            )
            
            timeout = httpx.Timeout(timeout=self.config.timeout)
            
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=self.config.default_headers,
                timeout=timeout,
                limits=limits,
                verify=self.config.verify_ssl,
            )
        
        return self._client
    
    async def _prepare_headers(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Prepare request headers with authentication."""
        final_headers = dict(self.config.default_headers)
        
        if headers:
            final_headers.update(headers)
        
        if self.auth:
            auth_headers = await self.auth.get_auth_headers()
            final_headers.update(auth_headers)
        
        return final_headers
    
    async def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle HTTP response and convert to dict."""
        # Log response if configured
        if self.config.log_responses:
            self._logger.info(
                "HTTP response",
                status_code=response.status_code,
                headers=dict(response.headers),
                url=str(response.url),
            )
        
        # Handle successful responses
        if response.is_success:
            try:
                return response.json()
            except json.JSONDecodeError:
                return {"data": response.text}
        
        # Handle authentication errors
        if response.status_code == 401 and self.auth:
            if await self.auth.handle_auth_error(response):
                # Auth was refreshed, caller should retry
                raise AuthRetryException()
        
        # Parse error response
        try:
            error_data = response.json()
        except json.JSONDecodeError:
            error_data = {"error_message": response.text}
        
        # Create appropriate exception
        error_message = error_data.get("error_message") or error_data.get("message", "Unknown error")
        exception = create_exception_from_response(
            status_code=response.status_code,
            message=error_message,
            details=error_data,
        )
        
        raise exception
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=1, max=10),
        retry=retry_if_exception_type((AuthRetryException, httpx.RequestError)),
        reraise=True,
    )
    async def _make_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Union[str, bytes]] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic."""
        
        # Rate limiting
        await self._rate_limiter.acquire()
        
        # Prepare headers
        final_headers = await self._prepare_headers(headers)
        
        # Log request if configured
        if self.config.log_requests:
            self._logger.info(
                "HTTP request",
                method=method,
                url=url,
                headers=final_headers,
                params=params,
            )
        
        try:
            response = await self.client.request(
                method=method,
                url=url,
                headers=final_headers,
                params=params,
                json=json_data,
                data=data,
                files=files,
            )
            
            return await self._handle_response(response)
            
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timed out: {e}")
        except httpx.ConnectError as e:
            raise ConnectionError(f"Connection failed: {e}")
        except httpx.RequestError as e:
            raise ConnectionError(f"Request failed: {e}")
    
    async def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make a GET request."""
        return await self._make_request("GET", url, headers=headers, params=params)
    
    async def post(
        self,
        url: str,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Union[str, bytes]] = None,
        files: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make a POST request."""
        return await self._make_request(
            "POST",
            url,
            headers=headers,
            params=params,
            json_data=json_data,
            data=data,
            files=files,
        )
    
    async def put(
        self,
        url: str,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Union[str, bytes]] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make a PUT request."""
        return await self._make_request(
            "PUT", url, headers=headers, params=params, json_data=json_data, data=data
        )
    
    async def patch(
        self,
        url: str,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Union[str, bytes]] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make a PATCH request."""
        return await self._make_request(
            "PATCH", url, headers=headers, params=params, json_data=json_data, data=data
        )
    
    async def delete(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make a DELETE request."""
        return await self._make_request("DELETE", url, headers=headers, params=params)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return await self.get("/health")
    
    async def close(self):
        """Close the client and cleanup resources."""
        if self._client:
            await self._client.aclose()
        
        if self.auth:
            await self.auth.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class SyncClient:
    """Synchronous wrapper for BaseClient."""
    
    def __init__(self, config: ClientConfig, auth: Optional[AuthHandler] = None):
        self._async_client = BaseClient(config, auth)
    
    def _run_async(self, coro):
        """Run async coroutine in sync context."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(coro)
    
    def get(self, url: str, **kwargs) -> Dict[str, Any]:
        """Make a synchronous GET request."""
        return self._run_async(self._async_client.get(url, **kwargs))
    
    def post(self, url: str, **kwargs) -> Dict[str, Any]:
        """Make a synchronous POST request."""
        return self._run_async(self._async_client.post(url, **kwargs))
    
    def put(self, url: str, **kwargs) -> Dict[str, Any]:
        """Make a synchronous PUT request."""
        return self._run_async(self._async_client.put(url, **kwargs))
    
    def patch(self, url: str, **kwargs) -> Dict[str, Any]:
        """Make a synchronous PATCH request."""
        return self._run_async(self._async_client.patch(url, **kwargs))
    
    def delete(self, url: str, **kwargs) -> Dict[str, Any]:
        """Make a synchronous DELETE request."""
        return self._run_async(self._async_client.delete(url, **kwargs))
    
    def health_check(self) -> Dict[str, Any]:
        """Perform synchronous health check."""
        return self._run_async(self._async_client.health_check())
    
    def close(self):
        """Close the client."""
        self._run_async(self._async_client.close())
    
    def __enter__(self):
        """Sync context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit."""
        self.close()


class AuthRetryException(Exception):
    """Internal exception to trigger auth retry."""
    pass