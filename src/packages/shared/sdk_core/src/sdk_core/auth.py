"""Authentication components for SDK clients."""

import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Optional

import httpx
from jose import jwt, JWTError
from pydantic import BaseModel

from sdk_core.exceptions import AuthenticationError


class TokenInfo(BaseModel):
    """Token information."""
    
    access_token: str
    token_type: str = "bearer"
    expires_in: Optional[int] = None
    refresh_token: Optional[str] = None
    scope: Optional[str] = None


class AuthHandler(ABC):
    """Abstract base class for authentication handlers."""
    
    @abstractmethod
    async def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for requests."""
        pass
    
    @abstractmethod
    async def handle_auth_error(self, response: httpx.Response) -> bool:
        """Handle authentication errors. Return True if auth was refreshed."""
        pass


class TokenAuth(AuthHandler):
    """Simple token-based authentication."""
    
    def __init__(self, token: str, token_type: str = "Bearer"):
        self.token = token
        self.token_type = token_type
    
    async def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        return {"Authorization": f"{self.token_type} {self.token}"}
    
    async def handle_auth_error(self, response: httpx.Response) -> bool:
        """Token auth doesn't support refresh."""
        return False


class JWTAuth(AuthHandler):
    """JWT-based authentication with automatic refresh."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        token_endpoint: str = "/auth/token",
        refresh_endpoint: str = "/auth/refresh",
        client: Optional[httpx.AsyncClient] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.token_endpoint = token_endpoint
        self.refresh_endpoint = refresh_endpoint
        self._client = client
        self._token_info: Optional[TokenInfo] = None
        self._token_expires_at: Optional[datetime] = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client
    
    async def _fetch_token(self) -> TokenInfo:
        """Fetch a new access token."""
        url = f"{self.base_url}{self.token_endpoint}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            response = await self.client.post(url, headers=headers)
            response.raise_for_status()
            token_data = response.json()
            
            token_info = TokenInfo(**token_data)
            
            # Calculate expiration time
            if token_info.expires_in:
                self._token_expires_at = datetime.utcnow() + timedelta(
                    seconds=token_info.expires_in - 60  # Refresh 1 minute early
                )
            
            return token_info
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise AuthenticationError("Invalid API key")
            raise AuthenticationError(f"Token fetch failed: {e}")
        except Exception as e:
            raise AuthenticationError(f"Token fetch failed: {e}")
    
    async def _refresh_token(self) -> TokenInfo:
        """Refresh the access token."""
        if not self._token_info or not self._token_info.refresh_token:
            return await self._fetch_token()
        
        url = f"{self.base_url}{self.refresh_endpoint}"
        data = {"refresh_token": self._token_info.refresh_token}
        
        try:
            response = await self.client.post(url, json=data)
            response.raise_for_status()
            token_data = response.json()
            
            token_info = TokenInfo(**token_data)
            
            if token_info.expires_in:
                self._token_expires_at = datetime.utcnow() + timedelta(
                    seconds=token_info.expires_in - 60
                )
            
            return token_info
            
        except httpx.HTTPStatusError:
            # If refresh fails, fetch a new token
            return await self._fetch_token()
        except Exception as e:
            raise AuthenticationError(f"Token refresh failed: {e}")
    
    async def _ensure_valid_token(self) -> TokenInfo:
        """Ensure we have a valid access token."""
        now = datetime.utcnow()
        
        # Check if we need a new token
        if (
            self._token_info is None
            or self._token_expires_at is None
            or now >= self._token_expires_at
        ):
            if self._token_info is None:
                # First time - fetch token
                self._token_info = await self._fetch_token()
            else:
                # Token expired - try to refresh
                self._token_info = await self._refresh_token()
        
        return self._token_info
    
    async def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers with valid token."""
        token_info = await self._ensure_valid_token()
        return {"Authorization": f"{token_info.token_type} {token_info.access_token}"}
    
    async def handle_auth_error(self, response: httpx.Response) -> bool:
        """Handle authentication errors by refreshing token."""
        if response.status_code == 401:
            try:
                # Force token refresh
                self._token_info = await self._refresh_token()
                return True
            except AuthenticationError:
                # Refresh failed, let the error bubble up
                pass
        return False
    
    def decode_token(self, token: Optional[str] = None) -> Dict:
        """Decode JWT token without verification (for inspection)."""
        if token is None:
            if self._token_info is None:
                raise AuthenticationError("No token available")
            token = self._token_info.access_token
        
        try:
            # Decode without verification (just for inspection)
            return jwt.get_unverified_claims(token)
        except JWTError as e:
            raise AuthenticationError(f"Token decode failed: {e}")
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()


class ApiKeyAuth(AuthHandler):
    """API key authentication."""
    
    def __init__(self, api_key: str, header_name: str = "X-API-Key"):
        self.api_key = api_key
        self.header_name = header_name
    
    async def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        return {self.header_name: self.api_key}
    
    async def handle_auth_error(self, response: httpx.Response) -> bool:
        """API key auth doesn't support refresh."""
        return False