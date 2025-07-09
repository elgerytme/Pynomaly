# Resilience Patterns

This document describes the comprehensive resilience layer implementation in Pynomaly, which provides circuit breakers, retries, and timeouts for outbound I/O operations.

## Overview

The resilience layer protects your application from cascading failures by implementing three key patterns:

1. **Circuit Breakers**: Prevent calls to failing services
2. **Retries**: Automatically retry failed operations with exponential backoff
3. **Timeouts**: Prevent operations from hanging indefinitely

## Features

- **Multiple I/O Types**: Database, Redis, HTTP, S3, and ML operations
- **Configurable Defaults**: Sane defaults for each operation type
- **Settings Integration**: Configuration via environment variables and settings
- **Async Support**: Full async/await support for FastAPI endpoints
- **Health Monitoring**: Circuit breaker health checks and metrics
- **Fallback Mechanisms**: Graceful degradation when services are unavailable

## Quick Start

### Basic Usage

```python
from pynomaly.infrastructure.resilience import database_resilient, api_resilient

# Database operation with default resilience
@database_resilient(operation_name="get_user")
def get_user(user_id: int):
    return query("SELECT * FROM users WHERE id = %s", (user_id,))

# API operation with custom settings
@api_resilient(
    operation_name="payment_api",
    timeout_seconds=30.0,
    max_attempts=3,
)
def process_payment(payment_data: dict):
    return http_post("https://api.payment.com/charge", json=payment_data)
```

### Async Operations

```python
@database_resilient(operation_name="async_get_user")
async def async_get_user(user_id: int):
    return await async_query("SELECT * FROM users WHERE id = %s", (user_id,))

@api_resilient(operation_name="async_payment")
async def async_process_payment(payment_data: dict):
    return await async_http_post("https://api.payment.com/charge", json=payment_data)
```

## Operation Types

### Database Operations

```python
from pynomaly.infrastructure.resilience import database_resilient

@database_resilient(operation_name="user_lookup")
def get_user(user_id: int):
    # Automatic retry with exponential backoff
    # Circuit breaker protection
    # 30-second timeout
    return query("SELECT * FROM users WHERE id = %s", (user_id,))
```

**Default Settings:**
- Timeout: 30 seconds
- Max attempts: 3
- Base delay: 0.5 seconds
- Max delay: 10 seconds
- Failure threshold: 3
- Recovery timeout: 30 seconds

### Cache Operations

```python
from pynomaly.infrastructure.resilience import cache_resilient

@cache_resilient(operation_name="user_cache")
async def cache_user(user_id: int, user_data: dict):
    # Fast fail for cache operations
    # Short timeout and minimal retries
    return await cache_set(f"user:{user_id}", user_data, ttl=3600)
```

**Default Settings:**
- Timeout: 5 seconds
- Max attempts: 2
- Base delay: 0.1 seconds
- Max delay: 1 second
- Failure threshold: 3
- Recovery timeout: 15 seconds

### HTTP API Operations

```python
from pynomaly.infrastructure.resilience import api_resilient

@api_resilient(operation_name="external_api")
def call_external_api(data: dict):
    # Longer timeout for external APIs
    # More retry attempts
    return http_post("https://api.external.com/endpoint", json=data)
```

**Default Settings:**
- Timeout: 60 seconds
- Max attempts: 5
- Base delay: 1 second
- Max delay: 30 seconds
- Failure threshold: 5
- Recovery timeout: 60 seconds

### S3 File Operations

```python
from pynomaly.infrastructure.resilience import file_resilient

@file_resilient(operation_name="model_upload")
def upload_model(file_path: str, bucket: str, key: str):
    # Optimized for file operations
    # Moderate timeout and retry settings
    return s3_upload(file_path, bucket, key)
```

**Default Settings:**
- Timeout: 10 seconds
- Max attempts: 3
- Base delay: 0.2 seconds
- Max delay: 5 seconds
- Failure threshold: 3
- Recovery timeout: 30 seconds

### ML Operations

```python
from pynomaly.infrastructure.resilience import ml_resilient

@ml_resilient(operation_name="model_training")
def train_model(dataset_path: str, config: dict):
    # Long timeout for training
    # Minimal retries for expensive operations
    return perform_training(dataset_path, config)
```

**Default Settings:**
- Timeout: 300 seconds (5 minutes)
- Max attempts: 2
- Base delay: 5 seconds
- Max delay: 30 seconds
- Failure threshold: 2
- Recovery timeout: 120 seconds

## Configuration

### Environment Variables

You can configure resilience settings via environment variables:

```bash
# Global settings
PYNOMALY_RESILIENCE_ENABLED=true
PYNOMALY_DEFAULT_TIMEOUT=5.0
PYNOMALY_DEFAULT_MAX_ATTEMPTS=3

# Database settings
PYNOMALY_DATABASE_TIMEOUT=30.0
PYNOMALY_DATABASE_MAX_ATTEMPTS=3
PYNOMALY_DATABASE_BASE_DELAY=0.5
PYNOMALY_DATABASE_FAILURE_THRESHOLD=3

# API settings
PYNOMALY_API_TIMEOUT=60.0
PYNOMALY_API_MAX_ATTEMPTS=5
PYNOMALY_API_BASE_DELAY=1.0
PYNOMALY_API_FAILURE_THRESHOLD=5

# Cache settings
PYNOMALY_CACHE_TIMEOUT=5.0
PYNOMALY_CACHE_MAX_ATTEMPTS=2
PYNOMALY_CACHE_BASE_DELAY=0.1
PYNOMALY_CACHE_FAILURE_THRESHOLD=3

# File settings
PYNOMALY_FILE_TIMEOUT=10.0
PYNOMALY_FILE_MAX_ATTEMPTS=3
PYNOMALY_FILE_BASE_DELAY=0.2
PYNOMALY_FILE_FAILURE_THRESHOLD=3

# ML settings
PYNOMALY_ML_TIMEOUT=300.0
PYNOMALY_ML_MAX_ATTEMPTS=2
PYNOMALY_ML_BASE_DELAY=5.0
PYNOMALY_ML_FAILURE_THRESHOLD=2
```

### Custom Configuration

```python
from pynomaly.infrastructure.resilience import resilient

@resilient(
    operation_type="custom",
    operation_name="critical_operation",
    timeout_seconds=15.0,
    max_attempts=5,
    base_delay=2.0,
    max_delay=30.0,
    exponential_base=1.5,
    jitter=True,
    failure_threshold=3,
    recovery_timeout=45.0,
    retry_exceptions=(ConnectionError, TimeoutError),
    enable_timeout=True,
    enable_retry=True,
    enable_circuit_breaker=True,
)
def critical_operation(data: dict):
    return perform_critical_operation(data)
```

## Resilient Clients

### Database Client

```python
from pynomaly.infrastructure.resilience import get_database_client

# Get resilient database client
client = get_database_client()

# Use context manager for transactions
with client.session() as session:
    result = session.execute("SELECT * FROM users")
    users = result.fetchall()

# Async operations
async with client.async_session() as session:
    result = await session.execute("SELECT * FROM users")
    users = result.fetchall()
```

### Redis Client

```python
from pynomaly.infrastructure.resilience import get_redis_client

# Get resilient Redis client (with fallback)
client = get_redis_client()

# Cache operations
await client.set("key", "value", ttl=3600)
value = await client.get("key")

# Hash operations
await client.hset("user:1", "name", "John")
name = await client.hget("user:1", "name")
```

### HTTP Client

```python
from pynomaly.infrastructure.resilience import get_http_client

# Get resilient HTTP client
client = get_http_client(base_url="https://api.example.com")

# HTTP operations
response = client.get("/users")
result = await client.async_post("/users", json={"name": "John"})
```

### S3 Client

```python
from pynomaly.infrastructure.resilience import get_s3_client

# Get resilient S3 client
client = get_s3_client()

# S3 operations
client.upload_file("local_file.txt", "bucket", "key")
await client.async_download_file("bucket", "key", "local_file.txt")
```

## Context Managers

For multiple operations that should share the same resilience configuration:

```python
from pynomaly.infrastructure.resilience import resilient_context

# Sync context
with resilient_context(
    operation_type="database",
    operation_name="bulk_operations",
    timeout_seconds=30.0,
    max_attempts=2,
) as ctx:
    result1 = ctx.call(get_users)
    result2 = ctx.call(update_user_status, user_id=1, status="active")

# Async context
async with resilient_context(
    operation_type="api",
    operation_name="bulk_api_calls",
) as ctx:
    result1 = await ctx.acall(fetch_user_data, user_id=1)
    result2 = await ctx.acall(update_user_profile, user_id=1, data={"name": "John"})
```

## Health Monitoring

### Circuit Breaker Health

```python
from pynomaly.infrastructure.resilience import get_resilience_health

# Get overall health status
health = get_resilience_health()
print(f"Overall health: {health['overall_health']}")

# Check for problems
if health['open_breakers']:
    print(f"Open circuit breakers: {health['open_breakers']}")

if health['degraded_breakers']:
    print(f"Degraded circuit breakers: {health['degraded_breakers']}")
```

### Client Health Checks

```python
# Database health
db_client = get_database_client()
db_health = db_client.health_check()
print(f"Database status: {db_health['status']}")

# Redis health
redis_client = get_redis_client()
redis_health = await redis_client.health_check()
print(f"Redis status: {redis_health['status']}")

# HTTP health
http_client = get_http_client()
http_health = http_client.health_check("/health")
print(f"HTTP API status: {http_health['status']}")

# S3 health
s3_client = get_s3_client()
s3_health = s3_client.health_check("my-bucket")
print(f"S3 status: {s3_health['status']}")
```

## FastAPI Integration

### Endpoint Decoration

```python
from fastapi import FastAPI
from pynomaly.infrastructure.resilience import api_resilient, database_resilient

app = FastAPI()

@app.get("/users/{user_id}")
@database_resilient(operation_name="get_user_endpoint")
async def get_user(user_id: int):
    # Database operation with resilience
    return await async_query("SELECT * FROM users WHERE id = %s", (user_id,))

@app.post("/users")
@api_resilient(operation_name="create_user_endpoint")
async def create_user(user_data: dict):
    # External API call with resilience
    return await async_http_post("https://api.external.com/users", json=user_data)
```

### Middleware Integration

```python
from fastapi import FastAPI, Request
from pynomaly.infrastructure.resilience import get_resilience_health

app = FastAPI()

@app.middleware("http")
async def resilience_middleware(request: Request, call_next):
    # Add resilience health to response headers
    response = await call_next(request)
    
    health = get_resilience_health()
    response.headers["X-Resilience-Status"] = health["overall_health"]
    
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint including resilience status."""
    return {
        "status": "healthy",
        "resilience": get_resilience_health(),
        "clients": {
            "database": get_database_client().health_check(),
            "redis": await get_redis_client().health_check(),
            "http": get_http_client().health_check(),
            "s3": get_s3_client().health_check(),
        }
    }
```

## Best Practices

### 1. Choose Appropriate Operation Types

- Use `database_resilient` for database operations
- Use `api_resilient` for external HTTP APIs
- Use `cache_resilient` for Redis/cache operations
- Use `file_resilient` for S3/file operations
- Use `ml_resilient` for machine learning operations

### 2. Configure Timeouts Appropriately

```python
# Short timeout for cache operations
@cache_resilient(timeout_seconds=2.0)
async def get_cached_data(key: str):
    return await cache_get(key)

# Long timeout for ML operations
@ml_resilient(timeout_seconds=600.0)  # 10 minutes
def train_model(data):
    return perform_training(data)
```

### 3. Use Appropriate Retry Settings

```python
# Fewer retries for expensive operations
@ml_resilient(max_attempts=1)
def expensive_operation():
    return perform_expensive_computation()

# More retries for critical operations
@api_resilient(max_attempts=5)
def critical_api_call():
    return call_critical_api()
```

### 4. Monitor Circuit Breaker Health

```python
import logging
from pynomaly.infrastructure.resilience import get_resilience_health

logger = logging.getLogger(__name__)

def monitor_resilience_health():
    health = get_resilience_health()
    
    if health["overall_health"] != "healthy":
        logger.warning(f"Resilience health degraded: {health}")
        
        # Alert on open circuit breakers
        if health["open_breakers"]:
            logger.error(f"Circuit breakers open: {health['open_breakers']}")
            # Send alert to monitoring system
```

### 5. Handle Exceptions Gracefully

```python
from pynomaly.infrastructure.resilience import CircuitBreakerError
from tenacity import RetryError

@api_resilient(operation_name="user_service")
def get_user_data(user_id: int):
    try:
        return call_user_service(user_id)
    except CircuitBreakerError:
        # Circuit breaker is open, use fallback
        return get_cached_user_data(user_id)
    except RetryError:
        # All retries exhausted
        logger.error(f"Failed to get user data for {user_id}")
        raise
```

## Error Handling

### Exception Types

The resilience layer handles several exception types:

- `CircuitBreakerError`: Circuit breaker is open
- `RetryError`: All retry attempts exhausted
- `TimeoutError`: Operation timed out
- Original exceptions: Database, HTTP, Redis errors

### Fallback Patterns

```python
@api_resilient(operation_name="user_enrichment")
def enrich_user_data(user: dict):
    try:
        # Try external enrichment service
        return call_enrichment_api(user)
    except (CircuitBreakerError, RetryError):
        # Fallback to basic data
        return {"id": user["id"], "name": user["name"]}

@cache_resilient(operation_name="user_cache")
async def get_user_from_cache(user_id: int):
    try:
        return await cache_get(f"user:{user_id}")
    except Exception:
        # Cache miss or failure, fetch from database
        return await get_user_from_database(user_id)
```

## Testing

### Unit Testing

```python
import pytest
from pynomaly.infrastructure.resilience import database_resilient, reset_resilience_state

@pytest.fixture
def reset_circuits():
    """Reset circuit breakers before each test."""
    reset_resilience_state()
    yield
    reset_resilience_state()

def test_database_operation_with_retry(reset_circuits):
    attempt_count = 0
    
    @database_resilient(operation_name="test_operation")
    def flaky_operation():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError("Database connection failed")
        return "success"
    
    result = flaky_operation()
    assert result == "success"
    assert attempt_count == 3
```

### Integration Testing

```python
import asyncio
from pynomaly.infrastructure.resilience import get_redis_client

@pytest.mark.asyncio
async def test_redis_fallback():
    """Test Redis client with fallback cache."""
    client = get_redis_client()
    
    # Should work even without Redis server
    await client.set("test_key", "test_value")
    value = await client.get("test_key")
    assert value == "test_value"
```

## Monitoring and Observability

### Metrics

The resilience layer provides metrics for monitoring:

- Circuit breaker state (open/closed/half-open)
- Success/failure rates
- Retry attempts
- Timeout occurrences
- Response times

### Logging

All resilience operations are logged with structured information:

```python
import logging

# Configure logging to capture resilience events
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Resilience events will be logged automatically
logger = logging.getLogger("pynomaly.infrastructure.resilience")
```

### Health Endpoints

```python
from fastapi import FastAPI
from pynomaly.infrastructure.resilience import get_resilience_health

app = FastAPI()

@app.get("/health/resilience")
async def resilience_health():
    """Dedicated resilience health endpoint."""
    return get_resilience_health()

@app.get("/metrics/resilience")
async def resilience_metrics():
    """Resilience metrics for monitoring systems."""
    health = get_resilience_health()
    
    return {
        "circuit_breaker_states": {
            name: stats["state"] 
            for name, stats in health["circuit_breakers"].items()
        },
        "failure_rates": {
            name: stats["failure_rate"] 
            for name, stats in health["circuit_breakers"].items()
        },
        "total_calls": {
            name: stats["total_calls"] 
            for name, stats in health["circuit_breakers"].items()
        },
        "overall_health": health["overall_health"],
    }
```

## Conclusion

The resilience layer provides comprehensive protection for outbound I/O operations with:

- **Automatic Retry**: Exponential backoff with jitter
- **Circuit Breakers**: Fail-fast when services are down
- **Timeouts**: Prevent hanging operations
- **Fallback Support**: Graceful degradation
- **Health Monitoring**: Observability and alerting
- **Async Support**: Full async/await compatibility

This ensures your application remains stable and responsive even when external dependencies fail or become slow.
