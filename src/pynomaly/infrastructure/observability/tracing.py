"""Distributed tracing configuration with OpenTelemetry."""

import logging
from typing import Any, Dict, Optional

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = logging.getLogger(__name__)


class TracingConfig:
    """Configuration for distributed tracing."""
    
    def __init__(
        self,
        service_name: str = "pynomaly",
        service_version: str = "1.0.0",
        environment: str = "development",
        jaeger_endpoint: Optional[str] = None,
        otlp_endpoint: Optional[str] = None,
        console_export: bool = False,
        sample_rate: float = 1.0,
    ):
        """Initialize tracing configuration.
        
        Args:
            service_name: Name of the service
            service_version: Version of the service
            environment: Environment (development, staging, production)
            jaeger_endpoint: Jaeger collector endpoint
            otlp_endpoint: OTLP collector endpoint
            console_export: Enable console span export for debugging
            sample_rate: Sampling rate (0.0 to 1.0)
        """
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.jaeger_endpoint = jaeger_endpoint
        self.otlp_endpoint = otlp_endpoint
        self.console_export = console_export
        self.sample_rate = sample_rate


def setup_tracing(config: TracingConfig) -> TracerProvider:
    """Set up distributed tracing with OpenTelemetry.
    
    Args:
        config: Tracing configuration
        
    Returns:
        Configured tracer provider
    """
    # Create resource with service information
    resource = Resource.create({
        "service.name": config.service_name,
        "service.version": config.service_version,
        "deployment.environment": config.environment,
    })
    
    # Create tracer provider
    tracer_provider = TracerProvider(resource=resource)
    
    # Set up span processors and exporters
    span_processors = []
    
    # Console exporter for debugging
    if config.console_export:
        console_exporter = ConsoleSpanExporter()
        console_processor = BatchSpanProcessor(console_exporter)
        span_processors.append(console_processor)
    
    # Jaeger exporter
    if config.jaeger_endpoint:
        try:
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
                collector_endpoint=config.jaeger_endpoint,
            )
            jaeger_processor = BatchSpanProcessor(jaeger_exporter)
            span_processors.append(jaeger_processor)
            logger.info(f"Configured Jaeger tracing: {config.jaeger_endpoint}")
        except Exception as e:
            logger.warning(f"Failed to configure Jaeger exporter: {e}")
    
    # OTLP exporter
    if config.otlp_endpoint:
        try:
            otlp_exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint)
            otlp_processor = BatchSpanProcessor(otlp_exporter)
            span_processors.append(otlp_processor)
            logger.info(f"Configured OTLP tracing: {config.otlp_endpoint}")
        except Exception as e:
            logger.warning(f"Failed to configure OTLP exporter: {e}")
    
    # Add span processors to tracer provider
    for processor in span_processors:
        tracer_provider.add_span_processor(processor)
    
    # Set the global tracer provider
    trace.set_tracer_provider(tracer_provider)
    
    # Set up propagators for distributed tracing
    from opentelemetry.propagate import set_global_textmap
    set_global_textmap(B3MultiFormat())
    
    logger.info(f"Tracing configured for service: {config.service_name}")
    return tracer_provider


def instrument_application(app: Any, config: TracingConfig) -> None:
    """Instrument application with automatic tracing.
    
    Args:
        app: FastAPI application instance
        config: Tracing configuration
    """
    try:
        # Instrument FastAPI
        FastAPIInstrumentor.instrument_app(
            app,
            server_request_hook=_server_request_hook,
            client_request_hook=_client_request_hook,
        )
        logger.info("FastAPI instrumentation enabled")
        
        # Instrument HTTP clients
        RequestsInstrumentor().instrument()
        HTTPXClientInstrumentor().instrument()
        logger.info("HTTP client instrumentation enabled")
        
        # Instrument database
        try:
            SQLAlchemyInstrumentor().instrument()
            logger.info("SQLAlchemy instrumentation enabled")
        except Exception as e:
            logger.warning(f"SQLAlchemy instrumentation failed: {e}")
        
        try:
            Psycopg2Instrumentor().instrument()
            logger.info("PostgreSQL instrumentation enabled")
        except Exception as e:
            logger.warning(f"PostgreSQL instrumentation failed: {e}")
        
        # Instrument Redis
        try:
            RedisInstrumentor().instrument()
            logger.info("Redis instrumentation enabled")
        except Exception as e:
            logger.warning(f"Redis instrumentation failed: {e}")
        
        # Instrument logging
        LoggingInstrumentor().instrument()
        logger.info("Logging instrumentation enabled")
        
    except Exception as e:
        logger.error(f"Failed to instrument application: {e}")


def _server_request_hook(span: trace.Span, scope: Dict[str, Any]) -> None:
    """Hook for server requests to add custom attributes.
    
    Args:
        span: OpenTelemetry span
        scope: ASGI scope
    """
    # Add custom attributes to spans
    if "path" in scope:
        span.set_attribute("http.route", scope["path"])
    
    # Add user information if available
    if "user" in scope:
        span.set_attribute("user.id", scope["user"].get("id", "anonymous"))


def _client_request_hook(span: trace.Span, request: Any) -> None:
    """Hook for client requests to add custom attributes.
    
    Args:
        span: OpenTelemetry span
        request: HTTP request object
    """
    # Add custom attributes for outgoing requests
    span.set_attribute("http.client.name", "pynomaly")


def get_tracer(name: str = __name__) -> trace.Tracer:
    """Get a tracer instance.
    
    Args:
        name: Tracer name (typically module name)
        
    Returns:
        OpenTelemetry tracer instance
    """
    return trace.get_tracer(name)


def create_span(
    tracer: trace.Tracer,
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
) -> trace.Span:
    """Create a new span with attributes.
    
    Args:
        tracer: Tracer instance
        name: Span name
        attributes: Optional span attributes
        kind: Span kind
        
    Returns:
        New span instance
    """
    span = tracer.start_span(name, kind=kind)
    
    if attributes:
        for key, value in attributes.items():
            span.set_attribute(key, value)
    
    return span


# Pre-configured tracing setups for different environments
TRACING_CONFIGS = {
    "development": TracingConfig(
        environment="development",
        console_export=True,
        sample_rate=1.0,
    ),
    "staging": TracingConfig(
        environment="staging",
        jaeger_endpoint="http://jaeger-collector:14268/api/traces",
        sample_rate=0.1,
    ),
    "production": TracingConfig(
        environment="production",
        otlp_endpoint="http://otel-collector:4317",
        sample_rate=0.01,
    ),
}


def setup_tracing_for_environment(
    environment: str = "development",
    service_name: str = "pynomaly",
    service_version: str = "1.0.0",
) -> TracerProvider:
    """Set up tracing for a specific environment.
    
    Args:
        environment: Environment name
        service_name: Service name
        service_version: Service version
        
    Returns:
        Configured tracer provider
    """
    config = TRACING_CONFIGS.get(environment, TRACING_CONFIGS["development"])
    config.service_name = service_name
    config.service_version = service_version
    
    return setup_tracing(config)