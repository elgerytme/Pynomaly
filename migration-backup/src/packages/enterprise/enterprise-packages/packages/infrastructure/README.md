# Enterprise Infrastructure

Production infrastructure patterns for monitoring, security, performance, and reliability in enterprise applications.

## Features

### Monitoring & Observability

- **Metrics Collection**: Prometheus, OpenTelemetry, custom backends
- **Distributed Tracing**: OpenTelemetry with Jaeger integration
- **Health Checks**: System and application health monitoring
- **Performance Monitoring**: Request timing, resource usage, SLA tracking

### Security

- **Authentication**: JWT tokens, password policies, account lockout
- **Authorization**: Role-based access control (RBAC) with permissions
- **Encryption**: Data encryption, password hashing, secure tokens
- **Security Middleware**: CORS, rate limiting, input validation

### Resilience Patterns

- **Circuit Breaker**: Prevent cascade failures
- **Rate Limiting**: Protect against abuse and overload
- **Retry Logic**: Exponential backoff with jitter
- **Bulkhead Isolation**: Resource isolation patterns
- **Timeout Management**: Request and operation timeouts

### Performance Optimization

- **Caching**: Multi-level caching strategies
- **Connection Pooling**: Database and HTTP connection management
- **Resource Optimization**: Memory and CPU optimization
- **Performance Profiling**: Bottleneck identification and analysis

## Quick Start

### Installation

```bash
# Base package
pip install enterprise-infrastructure

# With specific features
pip install enterprise-infrastructure[monitoring,security,rate_limiting]

# All features
pip install enterprise-infrastructure[all]
```

### Monitoring Setup

```python
from enterprise_infrastructure import (
    MetricsCollector,
    PrometheusMetrics,
    HealthCheckManager,
    SystemHealthCheck,
    PerformanceMonitor,
)

# Set up metrics collection
prometheus_backend = PrometheusMetrics(namespace="myapp")
metrics = MetricsCollector(backend=prometheus_backend)

# Record metrics
metrics.counter("requests.total")
metrics.gauge("active_connections", 42)
metrics.histogram("request.duration", 0.250)

# Use timer context
with metrics.timer("database.query"):
    # Database operation here
    pass

# Set up health checks
health_manager = HealthCheckManager()
health_manager.register_check("system", SystemHealthCheck())

# Check health
health_status = await health_manager.get_overall_health()
print(f"Health: {health_status.status} - {health_status.message}")

# Performance monitoring
perf_monitor = PerformanceMonitor(metrics)

async with perf_monitor.monitor_request("api.users.get"):
    # API operation here
    pass

stats = perf_monitor.get_performance_stats()
print(f"Average response time: {stats['avg_response_time']:.3f}s")
```

### Security Implementation

```python
from enterprise_infrastructure import (
    SecurityManager,
    SecurityConfiguration,
    User,
    Permission,
    Role,
)

# Configure security
config = SecurityConfiguration(
    secret_key="your-secret-key-here",
    access_token_expire_minutes=30,
    password_min_length=8,
    require_mfa=False,
)

# Create security manager
security = SecurityManager(config)
security.setup_default_roles()

# Define custom roles
admin_role = Role(
    name="admin",
    permissions=[Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN],
    description="Administrator with full access"
)
security.authz_manager.define_role(admin_role)

# Authenticate user
user = User(
    id="user123",
    username="john.doe",
    email="john@example.com",
    roles=["admin"]
)

# Create session
session = security.create_session(user)
print(f"Access token: {session['access_token']}")

# Verify permissions
security.authz_manager.require_permission(user.id, Permission.WRITE, "users")

# Token management
token_claims = security.token_manager.verify_token(session['access_token'])
print(f"Token valid for user: {token_claims.sub}")
```

### Resilience Patterns

```python
from enterprise_infrastructure import (
    CircuitBreaker,
    RateLimiter,
    RetryManager,
    TimeoutManager,
)

# Circuit breaker
circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=Exception
)

@circuit_breaker
async def external_service_call():
    # External service call that might fail
    pass

# Rate limiting
rate_limiter = RateLimiter(
    max_requests=100,
    window_seconds=60,
    burst_limit=10
)

if await rate_limiter.is_allowed("user123"):
    # Process request
    pass
else:
    # Rate limit exceeded
    pass

# Retry with exponential backoff
retry_manager = RetryManager(
    max_attempts=3,
    base_delay=1.0,
    max_delay=60.0,
    exponential_base=2
)

@retry_manager.retry
async def unreliable_operation():
    # Operation that might fail
    pass

# Timeout management
timeout_manager = TimeoutManager(default_timeout=30.0)

async with timeout_manager.timeout(10.0):
    # Operation with 10 second timeout
    pass
```

### Middleware Integration

```python
from enterprise_infrastructure import (
    RequestLoggingMiddleware,
    SecurityMiddleware,
    CORSMiddleware,
    ErrorHandlingMiddleware,
)

# FastAPI integration
from fastapi import FastAPI

app = FastAPI()

# Add middleware (order matters)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RequestLoggingMiddleware, logger=logger)
app.add_middleware(SecurityMiddleware, security_manager=security)
app.add_middleware(CORSMiddleware, 
    allow_origins=["https://example.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

@app.get("/api/users")
async def get_users():
    # Protected endpoint
    return {"users": []}
```

## Advanced Features

### Custom Metrics Backend

```python
from enterprise_infrastructure import MetricsBackend, MetricPoint

class CustomMetricsBackend(MetricsBackend):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
    
    def record_metric(self, metric: MetricPoint) -> None:
        # Send metric to custom backend
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        # Retrieve metrics from custom backend
        return {}

# Use custom backend
custom_backend = CustomMetricsBackend("https://metrics.example.com")
metrics = MetricsCollector(backend=custom_backend)
```

### Distributed Tracing

```python
from enterprise_infrastructure import OpenTelemetryTracing

# Set up tracing
tracing = OpenTelemetryTracing(
    service_name="my-service",
    jaeger_endpoint="http://localhost:14268"
)
tracing.initialize()

# Trace operations
async with tracing.trace_async("database.query", user_id="123") as span:
    # Database operation
    span.set_attribute("query.type", "SELECT")
    span.set_attribute("table.name", "users")
    
    result = await db.query("SELECT * FROM users")
    span.set_attribute("result.count", len(result))
```

### Advanced Security Features

```python
# Multi-factor authentication
from enterprise_infrastructure import MFAManager

mfa = MFAManager(security.encryption)

# Generate TOTP secret for user
secret = mfa.generate_totp_secret(user.id)
qr_url = mfa.get_qr_code_url(user.email, secret)

# Verify TOTP token
is_valid = mfa.verify_totp_token(user.id, "123456")

# Backup codes
backup_codes = mfa.generate_backup_codes(user.id)
is_backup_valid = mfa.verify_backup_code(user.id, "abc123def")
```

### Performance Profiling

```python
from enterprise_infrastructure import PerformanceProfiler

profiler = PerformanceProfiler()

# Profile function execution
@profiler.profile
async def slow_function():
    # Potentially slow operation
    pass

# Get profiling results
results = profiler.get_results()
for func_name, stats in results.items():
    print(f"{func_name}: {stats['avg_time']:.3f}s avg, {stats['call_count']} calls")
```

### Custom Health Checks

```python
from enterprise_infrastructure import HealthCheck, HealthStatus

class DatabaseHealthCheck(HealthCheck):
    def __init__(self, db_connection):
        self.db = db_connection
    
    async def check(self) -> HealthStatus:
        try:
            # Test database connection
            await self.db.execute("SELECT 1")
            
            # Check connection pool
            pool_size = self.db.get_pool_size()
            active_connections = self.db.get_active_connections()
            
            if active_connections / pool_size > 0.9:
                return HealthStatus(
                    status="degraded",
                    message="Database connection pool nearly exhausted",
                    details={
                        "active_connections": active_connections,
                        "pool_size": pool_size,
                        "utilization": active_connections / pool_size
                    }
                )
            
            return HealthStatus(
                status="healthy",
                message="Database is healthy",
                details={
                    "active_connections": active_connections,
                    "pool_size": pool_size
                }
            )
            
        except Exception as e:
            return HealthStatus(
                status="unhealthy",
                message=f"Database check failed: {e}",
                details={"error": str(e)}
            )

# Register custom health check
health_manager.register_check("database", DatabaseHealthCheck(db))
```

### Rate Limiting Strategies

```python
from enterprise_infrastructure import RateLimiter

# Per-user rate limiting
user_limiter = RateLimiter(
    max_requests=1000,
    window_seconds=3600,  # 1 hour window
    key_prefix="user"
)

# Per-IP rate limiting
ip_limiter = RateLimiter(
    max_requests=100,
    window_seconds=60,    # 1 minute window
    key_prefix="ip"
)

# API endpoint rate limiting
api_limiter = RateLimiter(
    max_requests=10000,
    window_seconds=60,
    key_prefix="api"
)

# Check multiple rate limits
async def check_rate_limits(user_id: str, ip_address: str, endpoint: str) -> bool:
    checks = [
        await user_limiter.is_allowed(user_id),
        await ip_limiter.is_allowed(ip_address),
        await api_limiter.is_allowed(endpoint)
    ]
    return all(checks)
```

### Circuit Breaker Configuration

```python
from enterprise_infrastructure import CircuitBreaker

# Database circuit breaker
db_breaker = CircuitBreaker(
    failure_threshold=5,      # Open after 5 failures
    recovery_timeout=30,      # Try recovery after 30 seconds
    expected_exception=DatabaseError,
    half_open_max_calls=3     # Max calls in half-open state
)

# External API circuit breaker
api_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=60,
    expected_exception=(TimeoutError, ConnectionError),
    success_threshold=2       # Require 2 successes to close
)

@db_breaker
async def database_operation():
    # Database operation
    pass

@api_breaker
async def external_api_call():
    # External API call
    pass

# Monitor circuit breaker state
if db_breaker.state == "open":
    logger.warning("Database circuit breaker is open")
```

## Configuration

### Environment Variables

```bash
# Security
SECURITY_SECRET_KEY=your-secret-key-here
SECURITY_ACCESS_TOKEN_EXPIRE_MINUTES=30
SECURITY_PASSWORD_MIN_LENGTH=8
SECURITY_MAX_LOGIN_ATTEMPTS=5

# Monitoring
METRICS_BACKEND=prometheus
METRICS_NAMESPACE=myapp
TRACING_SERVICE_NAME=my-service
TRACING_JAEGER_ENDPOINT=http://localhost:14268

# Rate Limiting
RATE_LIMIT_MAX_REQUESTS=100
RATE_LIMIT_WINDOW_SECONDS=60

# Circuit Breaker
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT=30
```

### Configuration File

```yaml
# infrastructure.yaml
security:
  secret_key: "${SECRET_KEY}"
  access_token_expire_minutes: 30
  password_min_length: 8
  require_mfa: false
  max_login_attempts: 5
  lockout_duration_minutes: 15

monitoring:
  metrics_backend: "prometheus"
  namespace: "myapp"
  tracing:
    service_name: "my-service"
    jaeger_endpoint: "http://localhost:14268"

rate_limiting:
  default_max_requests: 100
  default_window_seconds: 60
  burst_limit: 10

circuit_breaker:
  default_failure_threshold: 5
  default_recovery_timeout: 30
  default_success_threshold: 2
```

## Testing

Enterprise Infrastructure includes comprehensive testing utilities:

```python
from enterprise_infrastructure.testing import (
    MockMetricsCollector,
    MockSecurityManager,
    MockHealthCheck,
)

def test_metrics_collection():
    metrics = MockMetricsCollector()
    
    metrics.counter("test.counter")
    metrics.gauge("test.gauge", 42)
    
    assert metrics.get_counter("test.counter") == 1
    assert metrics.get_gauge("test.gauge") == 42

def test_security():
    security = MockSecurityManager()
    
    user = User(id="test", username="test", email="test@example.com")
    session = security.create_session(user)
    
    assert session["access_token"]
    assert security.validate_session(session["session_id"])
```

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines on contributing to this package.

## License

MIT License - see [LICENSE](../../LICENSE) file for details.
