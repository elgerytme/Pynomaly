"""Monitoring middleware for automatic metrics collection and tracing.

This module provides middleware for automatically collecting metrics and traces
from HTTP requests, database operations, and other system activities.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint

from pynomaly.infrastructure.monitoring.prometheus_metrics import (
    get_metrics_service, PrometheusMetricsService
)
from pynomaly.infrastructure.monitoring.telemetry import get_telemetry

logger = logging.getLogger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting HTTP request metrics."""
    
    def __init__(
        self, 
        app: FastAPI, 
        metrics_service: Optional[PrometheusMetricsService] = None,
        exclude_paths: Optional[list[str]] = None
    ):
        """Initialize metrics middleware.
        
        Args:
            app: FastAPI application
            metrics_service: Prometheus metrics service
            exclude_paths: Paths to exclude from metrics collection
        """
        super().__init__(app)
        self.metrics_service = metrics_service or get_metrics_service()
        self.exclude_paths = set(exclude_paths or ["/health", "/metrics", "/docs", "/openapi.json"])
        
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process HTTP request and collect metrics.
        
        Args:
            request: HTTP request
            call_next: Next middleware in chain
            
        Returns:
            HTTP response
        """
        # Skip metrics collection for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Record request start time
        start_time = time.time()
        
        # Get telemetry service for tracing
        telemetry = get_telemetry()
        
        # Create trace span for request
        span_name = f"{request.method} {request.url.path}"
        
        if telemetry:
            with telemetry.trace_span(span_name, {
                "http.method": request.method,
                "http.url": str(request.url),
                "http.route": request.url.path,
                "http.user_agent": request.headers.get("user-agent", ""),
                "http.client_ip": request.client.host if request.client else "unknown"
            }) as span:
                try:
                    # Process request
                    response = await call_next(request)
                    
                    # Calculate processing time
                    duration = time.time() - start_time
                    
                    # Record metrics
                    if self.metrics_service:
                        self.metrics_service.record_http_request(
                            method=request.method,
                            endpoint=request.url.path,
                            status_code=response.status_code,
                            duration=duration
                        )
                        
                        # Record API response size if available
                        if hasattr(response, 'body') and response.body:
                            self.metrics_service.record_api_response(
                                endpoint=request.url.path,
                                response_size_bytes=len(response.body)
                            )
                    
                    # Add span attributes
                    if span:
                        span.set_attributes({
                            "http.status_code": response.status_code,
                            "http.response_size": len(response.body) if hasattr(response, 'body') and response.body else 0,
                            "duration_seconds": duration
                        })
                    
                    return response
                    
                except Exception as e:
                    # Record error metrics
                    duration = time.time() - start_time
                    
                    if self.metrics_service:
                        self.metrics_service.record_http_request(
                            method=request.method,
                            endpoint=request.url.path,
                            status_code=500,
                            duration=duration
                        )
                        
                        self.metrics_service.record_error(
                            error_type=type(e).__name__,
                            component="http_middleware",
                            severity="error"
                        )
                    
                    # Record error in span
                    if span:
                        span.set_attributes({
                            "http.status_code": 500,
                            "error": True,
                            "error.type": type(e).__name__,
                            "error.message": str(e),
                            "duration_seconds": duration
                        })
                    
                    raise
        else:
            # No telemetry available, just collect basic metrics
            try:
                response = await call_next(request)
                duration = time.time() - start_time
                
                if self.metrics_service:
                    self.metrics_service.record_http_request(
                        method=request.method,
                        endpoint=request.url.path,
                        status_code=response.status_code,
                        duration=duration
                    )
                
                return response
                
            except Exception as e:
                duration = time.time() - start_time
                
                if self.metrics_service:
                    self.metrics_service.record_http_request(
                        method=request.method,
                        endpoint=request.url.path,
                        status_code=500,
                        duration=duration
                    )
                    
                    self.metrics_service.record_error(
                        error_type=type(e).__name__,
                        component="http_middleware"
                    )
                
                raise


class DatabaseMetricsMiddleware:
    """Middleware for collecting database operation metrics."""
    
    def __init__(self, metrics_service: Optional[PrometheusMetricsService] = None):
        """Initialize database metrics middleware.
        
        Args:
            metrics_service: Prometheus metrics service
        """
        self.metrics_service = metrics_service or get_metrics_service()
    
    def record_query(
        self, 
        operation: str, 
        table: str, 
        duration: float, 
        success: bool,
        rows_affected: int = 0
    ):
        """Record database query metrics.
        
        Args:
            operation: Database operation (SELECT, INSERT, UPDATE, DELETE)
            table: Database table name
            duration: Query duration in seconds
            success: Whether query was successful
            rows_affected: Number of rows affected
        """
        if not self.metrics_service:
            return
        
        # Record custom database metrics (would need to be added to prometheus_metrics.py)
        try:
            # Create database-specific metrics if they don't exist
            if not hasattr(self.metrics_service, '_db_operations_total'):
                # These would be added to the PrometheusMetricsService class
                logger.info("Database metrics not yet implemented in PrometheusMetricsService")
            
        except Exception as e:
            logger.warning(f"Failed to record database metrics: {e}")


class CacheMetricsMiddleware:
    """Middleware for collecting cache operation metrics."""
    
    def __init__(self, metrics_service: Optional[PrometheusMetricsService] = None):
        """Initialize cache metrics middleware.
        
        Args:
            metrics_service: Prometheus metrics service
        """
        self.metrics_service = metrics_service or get_metrics_service()
    
    def record_cache_operation(
        self,
        cache_type: str,
        operation: str,
        hit: bool,
        cache_size: int,
        key: Optional[str] = None
    ):
        """Record cache operation metrics.
        
        Args:
            cache_type: Type of cache (redis, memory, disk)
            operation: Cache operation (get, set, delete)
            hit: Whether operation was a hit
            cache_size: Current cache size
            key: Cache key (optional, for debugging)
        """
        if not self.metrics_service:
            return
        
        try:
            self.metrics_service.record_cache_metrics(
                cache_type=cache_type,
                operation=operation,
                hit=hit,
                cache_size=cache_size
            )
            
        except Exception as e:
            logger.warning(f"Failed to record cache metrics: {e}")


class DetectionMetricsMiddleware:
    """Middleware for collecting anomaly detection metrics."""
    
    def __init__(self, metrics_service: Optional[PrometheusMetricsService] = None):
        """Initialize detection metrics middleware.
        
        Args:
            metrics_service: Prometheus metrics service
        """
        self.metrics_service = metrics_service or get_metrics_service()
    
    def record_detection(
        self,
        algorithm: str,
        dataset_type: str,
        dataset_size: int,
        duration: float,
        anomalies_found: int,
        success: bool,
        accuracy: Optional[float] = None,
        confidence: Optional[float] = None
    ):
        """Record anomaly detection metrics.
        
        Args:
            algorithm: Algorithm used
            dataset_type: Type of dataset
            dataset_size: Size of dataset
            duration: Detection duration
            anomalies_found: Number of anomalies found
            success: Whether detection was successful
            accuracy: Detection accuracy if available
            confidence: Prediction confidence if available
        """
        if not self.metrics_service:
            return
        
        try:
            self.metrics_service.record_detection(
                algorithm=algorithm,
                dataset_type=dataset_type,
                dataset_size=dataset_size,
                duration=duration,
                anomalies_found=anomalies_found,
                success=success,
                accuracy=accuracy
            )
            
            # Record confidence if available
            if confidence is not None:
                self.metrics_service.update_quality_metrics(
                    dataset_id=f"detection_{int(time.time())}",
                    quality_scores={"detection_confidence": confidence},
                    prediction_confidence=confidence,
                    algorithm=algorithm
                )
            
        except Exception as e:
            logger.warning(f"Failed to record detection metrics: {e}")
    
    def record_training(
        self,
        algorithm: str,
        dataset_size: int,
        duration: float,
        model_size_bytes: int,
        success: bool,
        validation_score: Optional[float] = None
    ):
        """Record model training metrics.
        
        Args:
            algorithm: Algorithm trained
            dataset_size: Training dataset size
            duration: Training duration
            model_size_bytes: Size of trained model
            success: Whether training was successful
            validation_score: Validation score if available
        """
        if not self.metrics_service:
            return
        
        try:
            self.metrics_service.record_training(
                algorithm=algorithm,
                dataset_size=dataset_size,
                duration=duration,
                model_size_bytes=model_size_bytes,
                success=success
            )
            
            # Record validation score if available
            if validation_score is not None:
                self.metrics_service.update_quality_metrics(
                    dataset_id=f"training_{int(time.time())}",
                    quality_scores={"validation_score": validation_score},
                    algorithm=algorithm
                )
            
        except Exception as e:
            logger.warning(f"Failed to record training metrics: {e}")


@asynccontextmanager
async def monitor_operation(
    operation_name: str,
    component: str,
    metrics_service: Optional[PrometheusMetricsService] = None,
    **attributes
):
    """Context manager for monitoring operations with metrics and tracing.
    
    Args:
        operation_name: Name of the operation
        component: Component performing the operation
        metrics_service: Prometheus metrics service
        **attributes: Additional attributes for tracing
    """
    start_time = time.time()
    telemetry = get_telemetry()
    metrics = metrics_service or get_metrics_service()
    
    # Create trace span
    if telemetry:
        with telemetry.trace_span(operation_name, attributes) as span:
            try:
                yield
                
                # Record success metrics
                duration = time.time() - start_time
                
                if span:
                    span.set_attributes({
                        "operation.success": True,
                        "operation.duration_seconds": duration
                    })
                
            except Exception as e:
                # Record error metrics
                duration = time.time() - start_time
                
                if metrics:
                    metrics.record_error(
                        error_type=type(e).__name__,
                        component=component,
                        severity="error"
                    )
                
                if span:
                    span.set_attributes({
                        "operation.success": False,
                        "operation.duration_seconds": duration,
                        "error.type": type(e).__name__,
                        "error.message": str(e)
                    })
                
                raise
    else:
        # No telemetry, just basic error tracking
        try:
            yield
        except Exception as e:
            if metrics:
                metrics.record_error(
                    error_type=type(e).__name__,
                    component=component
                )
            raise


# Global middleware instances
_metrics_middleware: Optional[MetricsMiddleware] = None
_db_middleware: Optional[DatabaseMetricsMiddleware] = None
_cache_middleware: Optional[CacheMetricsMiddleware] = None
_detection_middleware: Optional[DetectionMetricsMiddleware] = None


def get_metrics_middleware() -> Optional[MetricsMiddleware]:
    """Get global metrics middleware instance."""
    return _metrics_middleware


def get_db_middleware() -> Optional[DatabaseMetricsMiddleware]:
    """Get global database middleware instance."""
    return _db_middleware


def get_cache_middleware() -> Optional[CacheMetricsMiddleware]:
    """Get global cache middleware instance."""
    return _cache_middleware


def get_detection_middleware() -> Optional[DetectionMetricsMiddleware]:
    """Get global detection middleware instance."""
    return _detection_middleware


def setup_monitoring_middleware(
    app: FastAPI,
    metrics_service: Optional[PrometheusMetricsService] = None,
    exclude_paths: Optional[list[str]] = None
):
    """Set up monitoring middleware for the application.
    
    Args:
        app: FastAPI application
        metrics_service: Prometheus metrics service
        exclude_paths: Paths to exclude from metrics collection
    """
    global _metrics_middleware, _db_middleware, _cache_middleware, _detection_middleware
    
    # Initialize middleware instances
    _metrics_middleware = MetricsMiddleware(
        app=app,
        metrics_service=metrics_service,
        exclude_paths=exclude_paths
    )
    
    _db_middleware = DatabaseMetricsMiddleware(metrics_service)
    _cache_middleware = CacheMetricsMiddleware(metrics_service)
    _detection_middleware = DetectionMetricsMiddleware(metrics_service)
    
    # Add HTTP metrics middleware to FastAPI app
    app.add_middleware(MetricsMiddleware, 
                      metrics_service=metrics_service,
                      exclude_paths=exclude_paths)
    
    logger.info("Monitoring middleware setup completed")