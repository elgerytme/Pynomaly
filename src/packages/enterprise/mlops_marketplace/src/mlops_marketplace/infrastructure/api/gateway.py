"""
API Gateway for the MLOps Marketplace.

Provides centralized API management, routing, authentication, rate limiting,
and monitoring for all marketplace APIs.
"""

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from mlops_marketplace.infrastructure.api.middleware import (
    AuthenticationMiddleware,
    RateLimitingMiddleware,
    LoggingMiddleware,
)
from mlops_marketplace.infrastructure.api.rate_limiter import RateLimiter
from mlops_marketplace.infrastructure.monitoring import (
    PrometheusMetrics,
    OpenTelemetryTracer,
)
from mlops_marketplace.domain.interfaces import CacheService


logger = structlog.get_logger()


class APIGatewayConfig:
    """Configuration for the API Gateway."""
    
    def __init__(
        self,
        title: str = "MLOps Marketplace API Gateway",
        version: str = "1.0.0",
        description: str = "Enterprise MLOps Marketplace API Gateway",
        cors_origins: List[str] = None,
        rate_limit_requests_per_minute: int = 1000,
        rate_limit_burst_size: int = 100,
        enable_gzip: bool = True,
        enable_authentication: bool = True,
        enable_rate_limiting: bool = True,
        enable_logging: bool = True,
        enable_metrics: bool = True,
        enable_tracing: bool = True,
        jwt_secret_key: str = "your-secret-key",
        jwt_algorithm: str = "HS256",
        jwt_expiration_seconds: int = 3600,
    ):
        """Initialize API Gateway configuration."""
        self.title = title
        self.version = version
        self.description = description
        self.cors_origins = cors_origins or ["*"]
        self.rate_limit_requests_per_minute = rate_limit_requests_per_minute
        self.rate_limit_burst_size = rate_limit_burst_size
        self.enable_gzip = enable_gzip
        self.enable_authentication = enable_authentication
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_logging = enable_logging
        self.enable_metrics = enable_metrics
        self.enable_tracing = enable_tracing
        self.jwt_secret_key = jwt_secret_key
        self.jwt_algorithm = jwt_algorithm
        self.jwt_expiration_seconds = jwt_expiration_seconds


class RequestMetrics:
    """Request metrics tracking."""
    
    def __init__(self):
        """Initialize request metrics."""
        self.request_count = 0
        self.total_duration = 0.0
        self.error_count = 0
        self.active_requests = 0


class APIGateway:
    """
    Centralized API Gateway for the MLOps Marketplace.
    
    Provides:
    - Request routing and load balancing
    - Authentication and authorization
    - Rate limiting and throttling
    - Request/response logging
    - Metrics collection and monitoring
    - Circuit breaker pattern
    - Request/response transformation
    """
    
    def __init__(
        self,
        config: APIGatewayConfig,
        cache_service: CacheService,
        rate_limiter: RateLimiter,
        metrics: Optional[PrometheusMetrics] = None,
        tracer: Optional[OpenTelemetryTracer] = None,
    ):
        """Initialize the API Gateway."""
        self.config = config
        self.cache_service = cache_service
        self.rate_limiter = rate_limiter
        self.metrics = metrics
        self.tracer = tracer
        
        # Create FastAPI app
        self.app = FastAPI(
            title=config.title,
            version=config.version,
            description=config.description,
        )
        
        # Service registry for routing
        self.service_registry: Dict[str, Dict[str, Any]] = {}
        
        # Request metrics
        self.request_metrics = RequestMetrics()
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_error_handlers()
        self._setup_health_endpoints()
    
    def _setup_middleware(self) -> None:
        """Setup middleware stack."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # GZip compression
        if self.config.enable_gzip:
            self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Custom middleware
        if self.config.enable_logging:
            self.app.add_middleware(LoggingMiddleware)
        
        if self.config.enable_rate_limiting:
            self.app.add_middleware(
                RateLimitingMiddleware,
                rate_limiter=self.rate_limiter,
            )
        
        if self.config.enable_authentication:
            self.app.add_middleware(
                AuthenticationMiddleware,
                jwt_secret_key=self.config.jwt_secret_key,
                jwt_algorithm=self.config.jwt_algorithm,
            )
        
        # Request tracking middleware
        self.app.add_middleware(RequestTrackingMiddleware, gateway=self)
    
    def _setup_error_handlers(self) -> None:
        """Setup global error handlers."""
        
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            """Handle HTTP exceptions."""
            self.request_metrics.error_count += 1
            
            if self.metrics:
                self.metrics.increment_counter(
                    "api_gateway_errors_total",
                    labels={"status_code": str(exc.status_code), "path": request.url.path}
                )
            
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "code": exc.status_code,
                        "message": exc.detail,
                        "request_id": getattr(request.state, "request_id", str(uuid4())),
                        "timestamp": time.time(),
                    }
                },
            )
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """Handle general exceptions."""
            self.request_metrics.error_count += 1
            
            logger.error(
                "Unhandled exception in API Gateway",
                exception=str(exc),
                path=request.url.path,
                method=request.method,
            )
            
            if self.metrics:
                self.metrics.increment_counter(
                    "api_gateway_errors_total",
                    labels={"status_code": "500", "path": request.url.path}
                )
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": 500,
                        "message": "Internal server error",
                        "request_id": getattr(request.state, "request_id", str(uuid4())),
                        "timestamp": time.time(),
                    }
                },
            )
    
    def _setup_health_endpoints(self) -> None:
        """Setup health check and monitoring endpoints."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "version": self.config.version,
                "services": await self._check_service_health(),
            }
        
        @self.app.get("/metrics")
        async def metrics_endpoint():
            """Metrics endpoint."""
            return {
                "request_count": self.request_metrics.request_count,
                "error_count": self.request_metrics.error_count,
                "active_requests": self.request_metrics.active_requests,
                "average_response_time": (
                    self.request_metrics.total_duration / self.request_metrics.request_count
                    if self.request_metrics.request_count > 0 else 0
                ),
                "registered_services": len(self.service_registry),
            }
        
        @self.app.get("/services")
        async def services_endpoint():
            """Service registry endpoint."""
            return {
                "services": {
                    name: {
                        "url": service["url"],
                        "health_check_url": service.get("health_check_url"),
                        "status": service.get("status", "unknown"),
                        "last_health_check": service.get("last_health_check"),
                    }
                    for name, service in self.service_registry.items()
                }
            }
    
    def register_service(
        self,
        name: str,
        base_url: str,
        health_check_url: Optional[str] = None,
        routes: Optional[List[str]] = None,
        load_balancer_strategy: str = "round_robin",
        circuit_breaker_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a service with the gateway."""
        self.service_registry[name] = {
            "url": base_url,
            "health_check_url": health_check_url,
            "routes": routes or [],
            "load_balancer_strategy": load_balancer_strategy,
            "circuit_breaker_config": circuit_breaker_config or {},
            "status": "unknown",
            "last_health_check": None,
            "instances": [{"url": base_url, "healthy": True}],
        }
        
        logger.info(f"Registered service: {name} at {base_url}")
    
    def add_service_instance(
        self,
        service_name: str,
        instance_url: str,
    ) -> None:
        """Add a new instance to an existing service."""
        if service_name not in self.service_registry:
            raise ValueError(f"Service {service_name} not registered")
        
        self.service_registry[service_name]["instances"].append({
            "url": instance_url,
            "healthy": True,
        })
        
        logger.info(f"Added instance {instance_url} to service {service_name}")
    
    def create_route_handler(
        self,
        service_name: str,
        path_prefix: str = "",
    ) -> Callable:
        """Create a route handler for a service."""
        
        async def route_handler(request: Request):
            """Handle requests to the service."""
            if service_name not in self.service_registry:
                raise HTTPException(
                    status_code=404,
                    detail=f"Service {service_name} not found"
                )
            
            # Get healthy instance
            instance = await self._get_healthy_instance(service_name)
            if not instance:
                raise HTTPException(
                    status_code=503,
                    detail=f"No healthy instances available for service {service_name}"
                )
            
            # Forward request
            return await self._forward_request(request, instance, path_prefix)
        
        return route_handler
    
    async def _get_healthy_instance(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get a healthy instance using load balancing."""
        service = self.service_registry[service_name]
        healthy_instances = [
            instance for instance in service["instances"]
            if instance["healthy"]
        ]
        
        if not healthy_instances:
            return None
        
        # Simple round-robin for now
        # In production, you'd implement more sophisticated load balancing
        strategy = service.get("load_balancer_strategy", "round_robin")
        
        if strategy == "round_robin":
            # Get next instance in rotation
            current_index = getattr(service, "_current_instance_index", 0)
            instance = healthy_instances[current_index % len(healthy_instances)]
            service["_current_instance_index"] = (current_index + 1) % len(healthy_instances)
            return instance
        
        # Default to first healthy instance
        return healthy_instances[0]
    
    async def _forward_request(
        self,
        request: Request,
        instance: Dict[str, Any],
        path_prefix: str = "",
    ) -> Response:
        """Forward request to service instance."""
        import httpx
        
        # Build target URL
        path = str(request.url.path)
        if path_prefix and path.startswith(path_prefix):
            path = path[len(path_prefix):]
        
        target_url = f"{instance['url']}{path}"
        if request.url.query:
            target_url += f"?{request.url.query}"
        
        # Prepare headers (remove hop-by-hop headers)
        headers = dict(request.headers)
        hop_by_hop_headers = {
            "connection", "keep-alive", "proxy-authenticate",
            "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade"
        }
        headers = {k: v for k, v in headers.items() if k.lower() not in hop_by_hop_headers}
        
        # Add forwarding headers
        headers["X-Forwarded-For"] = request.client.host if request.client else "unknown"
        headers["X-Forwarded-Proto"] = request.url.scheme
        headers["X-Request-ID"] = getattr(request.state, "request_id", str(uuid4()))
        
        try:
            # Forward request
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    content=await request.body(),
                )
                
                # Return response
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )
        
        except httpx.RequestError as e:
            logger.error(f"Request failed to {target_url}: {e}")
            # Mark instance as unhealthy
            instance["healthy"] = False
            
            raise HTTPException(
                status_code=503,
                detail=f"Service temporarily unavailable: {str(e)}"
            )
    
    async def _check_service_health(self) -> Dict[str, str]:
        """Check health of all registered services."""
        health_status = {}
        
        for name, service in self.service_registry.items():
            try:
                if service.get("health_check_url"):
                    import httpx
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.get(service["health_check_url"])
                        if response.status_code == 200:
                            health_status[name] = "healthy"
                            service["status"] = "healthy"
                        else:
                            health_status[name] = "unhealthy"
                            service["status"] = "unhealthy"
                else:
                    health_status[name] = "unknown"
                    service["status"] = "unknown"
                
                service["last_health_check"] = time.time()
            
            except Exception as e:
                logger.warning(f"Health check failed for {name}: {e}")
                health_status[name] = "unhealthy"
                service["status"] = "unhealthy"
                service["last_health_check"] = time.time()
        
        return health_status
    
    def start_health_checker(self, interval_seconds: int = 30) -> None:
        """Start background health checker."""
        
        async def health_checker():
            """Background health checker task."""
            while True:
                try:
                    await self._check_service_health()
                    await asyncio.sleep(interval_seconds)
                except Exception as e:
                    logger.error(f"Health checker error: {e}")
                    await asyncio.sleep(interval_seconds)
        
        # Start background task
        asyncio.create_task(health_checker())
        logger.info(f"Started health checker with {interval_seconds}s interval")


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware for tracking request metrics."""
    
    def __init__(self, app, gateway: APIGateway):
        super().__init__(app)
        self.gateway = gateway
    
    async def dispatch(self, request: Request, call_next):
        """Process request and track metrics."""
        # Generate request ID
        request_id = str(uuid4())
        request.state.request_id = request_id
        
        # Track request start
        start_time = time.time()
        self.gateway.request_metrics.request_count += 1
        self.gateway.request_metrics.active_requests += 1
        
        try:
            # Process request
            response = await call_next(request)
            
            # Track completion
            duration = time.time() - start_time
            self.gateway.request_metrics.total_duration += duration
            
            # Add response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            
            # Record metrics
            if self.gateway.metrics:
                self.gateway.metrics.record_histogram(
                    "api_gateway_request_duration_seconds",
                    duration,
                    labels={
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": str(response.status_code),
                    }
                )
            
            return response
        
        finally:
            self.gateway.request_metrics.active_requests -= 1