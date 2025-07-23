"""Utility functions and classes for SDK."""

import asyncio
import time
from collections import deque
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, requests: int, period: int):
        self.requests = requests
        self.period = period
        self.tokens = requests
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire a token, blocking if necessary."""
        async with self._lock:
            now = time.time()
            time_passed = now - self.last_update
            
            # Add tokens based on time passed
            self.tokens = min(
                self.requests,
                self.tokens + (time_passed * self.requests / self.period)
            )
            self.last_update = now
            
            # If no tokens available, wait
            if self.tokens < 1:
                wait_time = (1 - self.tokens) * self.period / self.requests
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class RetryManager:
    """Manages retry logic with exponential backoff."""
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        max_backoff: float = 60.0,
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.backoff_factor * (2 ** attempt)
        return min(delay, self.max_backoff)


class UrlBuilder:
    """Helper for building API URLs."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
    
    def build(self, *path_parts: str, **query_params: Any) -> str:
        """Build URL from path parts and query parameters."""
        path = "/".join(str(part).strip("/") for part in path_parts if part)
        url = urljoin(f"{self.base_url}/", path)
        
        if query_params:
            # Filter out None values
            params = {k: v for k, v in query_params.items() if v is not None}
            if params:
                from urllib.parse import urlencode
                url += "?" + urlencode(params)
        
        return url


class ResponseCache:
    """Simple in-memory response cache."""
    
    def __init__(self, max_size: int = 100, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, tuple] = {}
        self._access_order = deque()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl:
                # Move to end (most recently used)
                self._access_order.remove(key)
                self._access_order.append(key)
                return value
            else:
                # Expired, remove from cache
                del self._cache[key]
                self._access_order.remove(key)
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value."""
        # Remove if already exists
        if key in self._cache:
            self._access_order.remove(key)
        
        # Add new value
        self._cache[key] = (value, time.time())
        self._access_order.append(key)
        
        # Evict oldest if over max size
        while len(self._cache) > self.max_size:
            oldest_key = self._access_order.popleft()
            del self._cache[oldest_key]


def format_error_message(error_data: Dict[str, Any]) -> str:
    """Format error data into human-readable message."""
    if isinstance(error_data.get("details"), list):
        details = error_data["details"]
        if details:
            field_errors = []
            for detail in details:
                if isinstance(detail, dict):
                    field = detail.get("field", "unknown")
                    message = detail.get("message", "unknown error")
                    field_errors.append(f"{field}: {message}")
            
            if field_errors:
                base_message = error_data.get("error_message", "Validation failed")
                return f"{base_message}. Details: {'; '.join(field_errors)}"
    
    return error_data.get("error_message") or error_data.get("message", "Unknown error")


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size."""
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """Safely parse JSON, returning None on error."""
    try:
        import json
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None