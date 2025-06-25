"""Web API middleware for automatic configuration capture.

This middleware automatically captures configuration data from API requests
and responses, enabling seamless configuration management for web-based
anomaly detection workflows.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from starlette.responses import Response as StarletteResponse

from pynomaly.application.dto.configuration_dto import (
    ConfigurationCaptureRequestDTO, ConfigurationSource, 
    WebAPIContextDTO, RequestConfigurationDTO, ResponseConfigurationDTO
)
from pynomaly.application.services.configuration_capture_service import ConfigurationCaptureService
from pynomaly.infrastructure.config.feature_flags import require_feature

logger = logging.getLogger(__name__)


class ConfigurationCaptureMiddleware(BaseHTTPMiddleware):
    """Middleware for capturing API configurations automatically."""
    
    def __init__(
        self,
        app: ASGIApp,
        configuration_service: ConfigurationCaptureService,
        enable_capture: bool = True,
        capture_successful_only: bool = True,
        capture_threshold_ms: float = 100.0,
        excluded_paths: Optional[List[str]] = None,
        capture_request_body: bool = True,
        capture_response_body: bool = False,
        max_body_size: int = 1024 * 1024,  # 1MB
        anonymize_sensitive_data: bool = True
    ):
        """Initialize configuration capture middleware.
        
        Args:
            app: ASGI application
            configuration_service: Configuration capture service
            enable_capture: Enable configuration capture
            capture_successful_only: Only capture successful requests (2xx status)
            capture_threshold_ms: Minimum request duration to capture (ms)
            excluded_paths: Paths to exclude from capture
            capture_request_body: Capture request body data
            capture_response_body: Capture response body data
            max_body_size: Maximum body size to capture
            anonymize_sensitive_data: Anonymize sensitive data fields
        """
        super().__init__(app)
        self.configuration_service = configuration_service
        self.enable_capture = enable_capture
        self.capture_successful_only = capture_successful_only
        self.capture_threshold_ms = capture_threshold_ms
        self.excluded_paths = excluded_paths or [
            "/health", "/metrics", "/docs", "/redoc", "/openapi.json"
        ]
        self.capture_request_body = capture_request_body
        self.capture_response_body = capture_response_body
        self.max_body_size = max_body_size
        self.anonymize_sensitive_data = anonymize_sensitive_data
        
        # Capture statistics
        self.capture_stats = {
            "total_requests": 0,
            "captured_requests": 0,
            "skipped_requests": 0,
            "capture_errors": 0,
            "last_capture_time": None
        }
        
        # Sensitive field patterns
        self.sensitive_patterns = [
            "password", "token", "key", "secret", "auth", "credential",
            "api_key", "access_token", "refresh_token", "session_id"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and capture configuration if applicable."""
        if not self.enable_capture:
            return await call_next(request)
        
        self.capture_stats["total_requests"] += 1
        
        # Check if path should be excluded
        if self._should_exclude_path(request.url.path):
            self.capture_stats["skipped_requests"] += 1
            return await call_next(request)
        
        # Start timing
        start_time = time.time()
        
        # Extract request data
        request_data = await self._extract_request_data(request)
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Extract response data
        response_data = await self._extract_response_data(response, duration_ms)
        
        # Check if should capture
        if self._should_capture_request(response, duration_ms):
            try:
                await self._capture_configuration(request_data, response_data, request)
                self.capture_stats["captured_requests"] += 1
                self.capture_stats["last_capture_time"] = datetime.now().isoformat()
            except Exception as e:
                logger.warning(f"Configuration capture failed: {e}")
                self.capture_stats["capture_errors"] += 1
        else:
            self.capture_stats["skipped_requests"] += 1
        
        return response
    
    def _should_exclude_path(self, path: str) -> bool:
        """Check if path should be excluded from capture."""
        return any(excluded in path for excluded in self.excluded_paths)
    
    def _should_capture_request(self, response: Response, duration_ms: float) -> bool:
        """Determine if request should be captured."""
        # Check duration threshold
        if duration_ms < self.capture_threshold_ms:
            return False
        
        # Check status code
        if self.capture_successful_only and response.status_code >= 400:
            return False
        
        return True
    
    async def _extract_request_data(self, request: Request) -> RequestConfigurationDTO:
        """Extract configuration data from request."""
        # Get request body if enabled
        body_data = None
        if self.capture_request_body:
            try:
                body = await request.body()
                if len(body) <= self.max_body_size:
                    # Try to parse as JSON
                    try:
                        body_data = json.loads(body.decode('utf-8'))
                        if self.anonymize_sensitive_data:
                            body_data = self._anonymize_data(body_data)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # Store as string if not JSON
                        body_data = body.decode('utf-8', errors='ignore')[:1000]
            except Exception as e:
                logger.debug(f"Failed to extract request body: {e}")
        
        # Extract query parameters
        query_params = dict(request.query_params)
        if self.anonymize_sensitive_data:
            query_params = self._anonymize_data(query_params)
        
        # Extract headers (excluding sensitive ones)
        headers = dict(request.headers)
        if self.anonymize_sensitive_data:
            headers = self._anonymize_headers(headers)
        
        return RequestConfigurationDTO(
            method=request.method,
            path=request.url.path,
            query_parameters=query_params,
            headers=headers,
            body=body_data,
            client_ip=self._get_client_ip(request),
            user_agent=headers.get("user-agent"),
            content_type=headers.get("content-type"),
            content_length=headers.get("content-length")
        )
    
    async def _extract_response_data(
        self, 
        response: Response, 
        duration_ms: float
    ) -> ResponseConfigurationDTO:
        """Extract configuration data from response."""
        # Get response body if enabled and successful
        body_data = None
        if (self.capture_response_body and 
            response.status_code < 400 and 
            hasattr(response, 'body')):
            try:
                if hasattr(response.body, 'decode'):
                    body_size = len(response.body)
                    if body_size <= self.max_body_size:
                        body_str = response.body.decode('utf-8')
                        try:
                            body_data = json.loads(body_str)
                            if self.anonymize_sensitive_data:
                                body_data = self._anonymize_data(body_data)
                        except json.JSONDecodeError:
                            body_data = body_str[:1000]
            except Exception as e:
                logger.debug(f"Failed to extract response body: {e}")
        
        # Extract response headers
        headers = dict(response.headers) if hasattr(response, 'headers') else {}
        
        return ResponseConfigurationDTO(
            status_code=response.status_code,
            headers=headers,
            body=body_data,
            processing_time_ms=duration_ms,
            content_type=headers.get("content-type"),
            content_length=headers.get("content-length")
        )
    
    async def _capture_configuration(
        self,
        request_data: RequestConfigurationDTO,
        response_data: ResponseConfigurationDTO,
        original_request: Request
    ) -> None:
        """Capture configuration from request/response data."""
        # Extract algorithm and anomaly detection parameters
        raw_parameters = self._extract_parameters(request_data, response_data)
        
        if not raw_parameters:
            return  # No relevant parameters found
        
        # Build web API context
        web_api_context = WebAPIContextDTO(
            request_config=request_data,
            response_config=response_data,
            endpoint=f"{request_data.method} {request_data.path}",
            api_version=self._extract_api_version(original_request),
            client_info=self._extract_client_info(request_data),
            session_id=self._extract_session_id(request_data)
        )
        
        # Create capture request
        capture_request = ConfigurationCaptureRequestDTO(
            source=ConfigurationSource.WEB_API,
            raw_parameters=raw_parameters,
            execution_results=self._extract_execution_results(response_data),
            source_context={"web_api_context": web_api_context.dict()},
            metadata={
                "endpoint": web_api_context.endpoint,
                "processing_time_ms": response_data.processing_time_ms,
                "status_code": response_data.status_code,
                "client_ip": request_data.client_ip,
                "user_agent": request_data.user_agent
            },
            auto_save=True,
            tags=self._generate_tags(request_data, response_data)
        )
        
        # Capture configuration asynchronously
        try:
            await self.configuration_service.capture_configuration(capture_request)
            logger.debug(f"Captured configuration for {web_api_context.endpoint}")
        except Exception as e:
            logger.error(f"Failed to capture configuration: {e}")
            raise
    
    def _extract_parameters(
        self, 
        request_data: RequestConfigurationDTO, 
        response_data: ResponseConfigurationDTO
    ) -> Dict[str, Any]:
        """Extract anomaly detection parameters from request/response."""
        parameters = {}
        
        # Check request body for parameters
        if request_data.body and isinstance(request_data.body, dict):
            parameters.update(self._extract_anomaly_params(request_data.body))
        
        # Check query parameters
        if request_data.query_parameters:
            parameters.update(self._extract_anomaly_params(request_data.query_parameters))
        
        # Extract algorithm from path
        if "/detect" in request_data.path or "/anomaly" in request_data.path:
            # Add default detection parameters
            parameters.setdefault("detection_endpoint", True)
            parameters.setdefault("endpoint_type", "anomaly_detection")
        
        return parameters
    
    def _extract_anomaly_params(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract anomaly detection specific parameters."""
        anomaly_params = {}
        
        # Common anomaly detection parameter names
        param_patterns = [
            "algorithm", "detector", "contamination", "n_estimators",
            "n_neighbors", "threshold", "outlier", "anomaly", "score",
            "features", "dataset", "model", "training", "prediction",
            "cross_validation", "cv", "folds", "metric", "accuracy",
            "precision", "recall", "f1", "auc", "roc"
        ]
        
        for key, value in data.items():
            key_lower = key.lower()
            
            # Check if key matches anomaly detection patterns
            if any(pattern in key_lower for pattern in param_patterns):
                anomaly_params[key] = value
            
            # Recursively check nested dictionaries
            elif isinstance(value, dict):
                nested_params = self._extract_anomaly_params(value)
                if nested_params:
                    anomaly_params[key] = nested_params
        
        return anomaly_params
    
    def _extract_execution_results(
        self, 
        response_data: ResponseConfigurationDTO
    ) -> Dict[str, Any]:
        """Extract execution results from response."""
        results = {
            "processing_time_ms": response_data.processing_time_ms,
            "status_code": response_data.status_code,
            "success": response_data.status_code < 400
        }
        
        # Extract metrics from response body
        if response_data.body and isinstance(response_data.body, dict):
            # Look for common result patterns
            result_patterns = [
                "accuracy", "precision", "recall", "f1_score", "auc_roc", "auc_pr",
                "anomaly_count", "outlier_count", "score", "scores", "predictions",
                "training_time", "inference_time", "model_size", "metrics"
            ]
            
            for key, value in response_data.body.items():
                key_lower = key.lower()
                if any(pattern in key_lower for pattern in result_patterns):
                    results[key] = value
        
        return results
    
    def _extract_api_version(self, request: Request) -> Optional[str]:
        """Extract API version from request."""
        # Check headers
        if "api-version" in request.headers:
            return request.headers["api-version"]
        
        # Check path
        path_parts = request.url.path.split("/")
        for part in path_parts:
            if part.startswith("v") and part[1:].isdigit():
                return part
        
        return None
    
    def _extract_client_info(self, request_data: RequestConfigurationDTO) -> Dict[str, Any]:
        """Extract client information."""
        return {
            "ip": request_data.client_ip,
            "user_agent": request_data.user_agent,
            "platform": self._parse_platform(request_data.user_agent),
            "client_type": self._determine_client_type(request_data.user_agent)
        }
    
    def _extract_session_id(self, request_data: RequestConfigurationDTO) -> Optional[str]:
        """Extract session ID from request."""
        # Check headers
        for key, value in request_data.headers.items():
            if "session" in key.lower() or "x-session" in key.lower():
                return str(value)
        
        # Check query parameters
        for key, value in request_data.query_parameters.items():
            if "session" in key.lower():
                return str(value)
        
        return None
    
    def _generate_tags(
        self, 
        request_data: RequestConfigurationDTO, 
        response_data: ResponseConfigurationDTO
    ) -> List[str]:
        """Generate tags for configuration."""
        tags = ["web_api", request_data.method.lower()]
        
        # Add endpoint-based tags
        path_parts = request_data.path.strip("/").split("/")
        for part in path_parts[:3]:  # Limit to first 3 path parts
            if part and part.isalpha():
                tags.append(part)
        
        # Add status-based tags
        if response_data.status_code < 300:
            tags.append("successful")
        elif response_data.status_code < 400:
            tags.append("redirect")
        elif response_data.status_code < 500:
            tags.append("client_error")
        else:
            tags.append("server_error")
        
        # Add performance-based tags
        if response_data.processing_time_ms < 100:
            tags.append("fast")
        elif response_data.processing_time_ms < 1000:
            tags.append("moderate")
        else:
            tags.append("slow")
        
        return tags
    
    def _anonymize_data(self, data: Any) -> Any:
        """Recursively anonymize sensitive data."""
        if isinstance(data, dict):
            return {
                key: "***REDACTED***" if self._is_sensitive_field(key) 
                else self._anonymize_data(value)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self._anonymize_data(item) for item in data]
        else:
            return data
    
    def _anonymize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Anonymize sensitive headers."""
        anonymized = {}
        for key, value in headers.items():
            if self._is_sensitive_field(key):
                anonymized[key] = "***REDACTED***"
            else:
                anonymized[key] = value
        return anonymized
    
    def _is_sensitive_field(self, field_name: str) -> bool:
        """Check if field contains sensitive data."""
        field_lower = field_name.lower()
        return any(pattern in field_lower for pattern in self.sensitive_patterns)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to client host
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"
    
    def _parse_platform(self, user_agent: Optional[str]) -> str:
        """Parse platform from user agent."""
        if not user_agent:
            return "unknown"
        
        user_agent_lower = user_agent.lower()
        
        if "windows" in user_agent_lower:
            return "windows"
        elif "mac" in user_agent_lower or "darwin" in user_agent_lower:
            return "macos"
        elif "linux" in user_agent_lower:
            return "linux"
        elif "android" in user_agent_lower:
            return "android"
        elif "iphone" or "ipad" in user_agent_lower:
            return "ios"
        else:
            return "unknown"
    
    def _determine_client_type(self, user_agent: Optional[str]) -> str:
        """Determine client type from user agent."""
        if not user_agent:
            return "unknown"
        
        user_agent_lower = user_agent.lower()
        
        # Check for common API clients
        api_clients = ["curl", "wget", "httpie", "postman", "insomnia", "requests", "urllib"]
        for client in api_clients:
            if client in user_agent_lower:
                return "api_client"
        
        # Check for browsers
        browsers = ["chrome", "firefox", "safari", "edge", "opera"]
        for browser in browsers:
            if browser in user_agent_lower:
                return "browser"
        
        return "unknown"
    
    def get_capture_statistics(self) -> Dict[str, Any]:
        """Get middleware capture statistics."""
        return {
            "capture_stats": self.capture_stats.copy(),
            "configuration": {
                "enabled": self.enable_capture,
                "capture_successful_only": self.capture_successful_only,
                "capture_threshold_ms": self.capture_threshold_ms,
                "excluded_paths": self.excluded_paths,
                "capture_request_body": self.capture_request_body,
                "capture_response_body": self.capture_response_body,
                "max_body_size": self.max_body_size,
                "anonymize_sensitive_data": self.anonymize_sensitive_data
            }
        }


class ConfigurationAPIMiddleware:
    """Factory for creating configuration capture middleware."""
    
    @staticmethod
    def create_middleware(
        configuration_service: ConfigurationCaptureService,
        **kwargs
    ) -> ConfigurationCaptureMiddleware:
        """Create configuration capture middleware.
        
        Args:
            configuration_service: Configuration capture service
            **kwargs: Additional middleware configuration
            
        Returns:
            Configured middleware instance
        """
        return lambda app: ConfigurationCaptureMiddleware(
            app=app,
            configuration_service=configuration_service,
            **kwargs
        )
    
    @staticmethod
    @require_feature("advanced_automl")
    def create_production_middleware(
        configuration_service: ConfigurationCaptureService
    ) -> ConfigurationCaptureMiddleware:
        """Create production-ready middleware with security considerations.
        
        Args:
            configuration_service: Configuration capture service
            
        Returns:
            Production-configured middleware instance
        """
        return lambda app: ConfigurationCaptureMiddleware(
            app=app,
            configuration_service=configuration_service,
            enable_capture=True,
            capture_successful_only=True,
            capture_threshold_ms=200.0,  # Higher threshold for production
            excluded_paths=[
                "/health", "/metrics", "/docs", "/redoc", "/openapi.json",
                "/admin", "/internal", "/debug"
            ],
            capture_request_body=True,
            capture_response_body=False,  # Disabled for security
            max_body_size=512 * 1024,  # 512KB limit
            anonymize_sensitive_data=True
        )
    
    @staticmethod
    def create_development_middleware(
        configuration_service: ConfigurationCaptureService
    ) -> ConfigurationCaptureMiddleware:
        """Create development-friendly middleware with verbose capture.
        
        Args:
            configuration_service: Configuration capture service
            
        Returns:
            Development-configured middleware instance
        """
        return lambda app: ConfigurationCaptureMiddleware(
            app=app,
            configuration_service=configuration_service,
            enable_capture=True,
            capture_successful_only=False,  # Capture all requests
            capture_threshold_ms=50.0,  # Lower threshold
            excluded_paths=["/health", "/metrics"],  # Minimal exclusions
            capture_request_body=True,
            capture_response_body=True,  # Enabled for debugging
            max_body_size=2 * 1024 * 1024,  # 2MB limit
            anonymize_sensitive_data=False  # Disabled for debugging
        )