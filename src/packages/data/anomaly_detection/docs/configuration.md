# Configuration Guide

This guide covers all configuration options for the Anomaly Detection package, including environment variables, configuration files, and runtime settings.

## Table of Contents

1. [Configuration Overview](#configuration-overview)
2. [Environment Variables](#environment-variables)
3. [Configuration Files](#configuration-files)
4. [Algorithm Configuration](#algorithm-configuration)
5. [Service Configuration](#service-configuration)
6. [Logging Configuration](#logging-configuration)
7. [Performance Tuning](#performance-tuning)
8. [Security Configuration](#security-configuration)
9. [Monitoring Configuration](#monitoring-configuration)

## Configuration Overview

The Anomaly Detection package supports multiple configuration methods:

1. **Environment Variables**: For deployment-specific settings
2. **Configuration Files**: YAML/JSON files for complex configurations
3. **Runtime Parameters**: Programmatic configuration via code
4. **CLI Arguments**: Command-line parameter overrides

### Configuration Precedence

Configuration sources are applied in the following order (highest to lowest priority):

1. CLI arguments
2. Environment variables
3. Configuration files
4. Default values

## Environment Variables

### Core Settings

```bash
# Application environment
export ANOMALY_DETECTION_ENV=production  # development, staging, production

# API server settings
export ANOMALY_DETECTION_HOST=0.0.0.0
export ANOMALY_DETECTION_PORT=8001
export ANOMALY_DETECTION_WORKERS=4

# Paths
export ANOMALY_DETECTION_DATA_DIR=/var/lib/anomaly_detection/data
export ANOMALY_DETECTION_MODELS_DIR=/var/lib/anomaly_detection/models
export ANOMALY_DETECTION_LOGS_DIR=/var/log/anomaly_detection

# Database (if using persistence)
export ANOMALY_DETECTION_DATABASE_URL=postgresql://user:pass@localhost/anomaly_db
export ANOMALY_DETECTION_REDIS_URL=redis://localhost:6379/0
```

### Algorithm Settings

```bash
# Default algorithm
export ANOMALY_DETECTION_DEFAULT_ALGORITHM=isolation_forest

# Default contamination rate
export ANOMALY_DETECTION_DEFAULT_CONTAMINATION=0.1

# Algorithm timeouts (seconds)
export ANOMALY_DETECTION_ALGORITHM_TIMEOUT=300

# Memory limits (MB)
export ANOMALY_DETECTION_MAX_MEMORY_PER_JOB=2048
```

### Performance Settings

```bash
# Worker configuration
export ANOMALY_DETECTION_MAX_WORKERS=8
export ANOMALY_DETECTION_WORKER_TIMEOUT=600
export ANOMALY_DETECTION_JOB_QUEUE_SIZE=1000

# Caching
export ANOMALY_DETECTION_ENABLE_CACHE=true
export ANOMALY_DETECTION_CACHE_TTL=3600

# Batch processing
export ANOMALY_DETECTION_BATCH_SIZE=1000
export ANOMALY_DETECTION_MAX_BATCH_SIZE=10000
```

### Security Settings

```bash
# API security
export ANOMALY_DETECTION_API_KEY=your-secret-api-key
export ANOMALY_DETECTION_ENABLE_AUTH=true
export ANOMALY_DETECTION_JWT_SECRET=your-jwt-secret

# CORS settings
export ANOMALY_DETECTION_CORS_ORIGINS=http://localhost:3000,https://app.example.com
export ANOMALY_DETECTION_CORS_ALLOW_CREDENTIALS=true

# Rate limiting
export ANOMALY_DETECTION_RATE_LIMIT_ENABLED=true
export ANOMALY_DETECTION_RATE_LIMIT_PER_MINUTE=100
```

## Configuration Files

### Main Configuration File

Create `config.yaml` in your project root or specify via `--config` flag:

```yaml
# config.yaml
anomaly_detection:
  environment: production
  
  server:
    host: 0.0.0.0
    port: 8001
    workers: 4
    reload: false
    cors:
      enabled: true
      origins:
        - http://localhost:3000
        - https://app.example.com
      allow_credentials: true
      allow_methods:
        - GET
        - POST
        - PUT
        - DELETE
      allow_headers:
        - Content-Type
        - Authorization
  
  algorithms:
    default: isolation_forest
    defaults:
      contamination: 0.1
      random_state: 42
    
    isolation_forest:
      n_estimators: 100
      max_samples: auto
      max_features: 1.0
      bootstrap: false
      n_jobs: -1
    
    local_outlier_factor:
      n_neighbors: 20
      metric: minkowski
      p: 2
      metric_params: null
      contamination: auto
      novelty: true
    
    one_class_svm:
      kernel: rbf
      gamma: auto
      nu: 0.1
      shrinking: true
      cache_size: 200
      tol: 0.001
  
  data:
    max_file_size: 100MB
    allowed_formats:
      - csv
      - json
      - parquet
      - feather
    validation:
      check_finite: true
      force_all_finite: true
      ensure_2d: true
      allow_nd: false
      ensure_min_samples: 2
      ensure_min_features: 1
  
  logging:
    level: INFO
    format: json
    file:
      enabled: true
      path: /var/log/anomaly_detection/app.log
      max_size: 100MB
      backup_count: 5
    console:
      enabled: true
      colorize: true
  
  monitoring:
    enabled: true
    metrics:
      enabled: true
      export_interval: 60
      exporters:
        - prometheus
        - statsd
    tracing:
      enabled: true
      sampling_rate: 0.1
      exporter: jaeger
      endpoint: http://localhost:14268/api/traces
  
  storage:
    models:
      backend: filesystem  # filesystem, s3, gcs, azure
      path: /var/lib/anomaly_detection/models
      s3:
        bucket: anomaly-detection-models
        region: us-east-1
        prefix: models/
    
    results:
      backend: database  # database, filesystem, redis
      retention_days: 30
      compression: true
  
  performance:
    cache:
      enabled: true
      backend: redis  # redis, memory
      ttl: 3600
      max_size: 1000
    
    workers:
      max_concurrent_jobs: 10
      job_timeout: 600
      queue_size: 1000
      priority_queues:
        - name: critical
          workers: 2
        - name: high
          workers: 3
        - name: normal
          workers: 4
        - name: low
          workers: 1
  
  security:
    api_keys:
      enabled: true
      header_name: X-API-Key
    
    jwt:
      enabled: false
      algorithm: HS256
      expiry_hours: 24
    
    rate_limiting:
      enabled: true
      backend: redis
      limits:
        - path: /api/v1/detect
          max_requests: 100
          window_seconds: 60
        - path: /api/v1/ensemble
          max_requests: 50
          window_seconds: 60
```

### Algorithm-Specific Configuration

Create separate configuration files for complex algorithm setups:

```yaml
# algorithms/deep_learning.yaml
deep_learning:
  autoencoder:
    architecture:
      input_dim: null  # Inferred from data
      encoding_dim: 32
      hidden_layers:
        - 128
        - 64
        - 32
      activation: relu
      output_activation: sigmoid
    
    training:
      epochs: 100
      batch_size: 32
      validation_split: 0.1
      early_stopping:
        enabled: true
        patience: 10
        monitor: val_loss
      
      optimizer:
        name: adam
        learning_rate: 0.001
        beta_1: 0.9
        beta_2: 0.999
    
    anomaly_threshold:
      method: percentile  # percentile, std, fixed
      value: 95  # 95th percentile
  
  variational_autoencoder:
    architecture:
      latent_dim: 20
      intermediate_dim: 64
      epsilon_std: 1.0
    
    loss:
      reconstruction_weight: 1.0
      kl_weight: 0.1
```

### Environment-Specific Configurations

```yaml
# config.development.yaml
anomaly_detection:
  environment: development
  
  server:
    reload: true
    debug: true
    workers: 1
  
  logging:
    level: DEBUG
    console:
      colorize: true
  
  monitoring:
    enabled: false

---
# config.production.yaml
anomaly_detection:
  environment: production
  
  server:
    reload: false
    debug: false
    workers: 8
  
  logging:
    level: INFO
    format: json
  
  monitoring:
    enabled: true
    sampling_rate: 0.01
```

## Algorithm Configuration

### Isolation Forest

```python
from anomaly_detection import DetectionService

# Configure via parameters
service = DetectionService()
result = service.detect_anomalies(
    data=data,
    algorithm='iforest',
    n_estimators=200,
    max_samples='auto',
    contamination=0.1,
    max_features=1.0,
    bootstrap=False,
    n_jobs=-1,
    random_state=42,
    verbose=0
)

# Or via configuration dict
config = {
    'algorithm': 'iforest',
    'params': {
        'n_estimators': 200,
        'max_samples': 256,
        'contamination': 0.1
    }
}
result = service.detect_anomalies(data, **config)
```

### Local Outlier Factor

```python
# LOF configuration
lof_config = {
    'n_neighbors': 20,
    'algorithm': 'auto',  # auto, ball_tree, kd_tree, brute
    'leaf_size': 30,
    'metric': 'minkowski',
    'p': 2,
    'metric_params': None,
    'contamination': 0.1,
    'novelty': True,  # True for predict, False for fit_predict
    'n_jobs': -1
}

result = service.detect_anomalies(data, algorithm='lof', **lof_config)
```

### Ensemble Configuration

```python
from anomaly_detection import EnsembleService

ensemble = EnsembleService()

# Configure ensemble
ensemble_config = {
    'algorithms': [
        {'name': 'iforest', 'weight': 0.4, 'params': {'n_estimators': 100}},
        {'name': 'lof', 'weight': 0.3, 'params': {'n_neighbors': 20}},
        {'name': 'ocsvm', 'weight': 0.3, 'params': {'nu': 0.1}}
    ],
    'combination_method': 'weighted_vote',
    'require_unanimous': False,
    'confidence_threshold': 0.7
}

result = ensemble.detect_with_config(data, ensemble_config)
```

## Service Configuration

### API Server Configuration

```python
from anomaly_detection.server import create_app
from anomaly_detection.infrastructure.config import Settings

# Configure programmatically
settings = Settings(
    host="0.0.0.0",
    port=8001,
    workers=4,
    reload=False,
    cors_enabled=True,
    cors_origins=["https://app.example.com"],
    rate_limit_enabled=True,
    rate_limit_per_minute=100
)

app = create_app(settings)
```

### Worker Configuration

```python
from anomaly_detection.worker import AnomalyDetectionWorker

# Configure worker
worker = AnomalyDetectionWorker(
    models_dir="/var/lib/models",
    max_concurrent_jobs=5,
    job_timeout=600,
    enable_monitoring=True,
    redis_url="redis://localhost:6379/0"
)

# Start with custom configuration
worker.start(
    queue_names=["critical", "high", "normal"],
    poll_interval=1.0,
    batch_size=10
)
```

## Logging Configuration

### Structured Logging Setup

```python
import structlog
from anomaly_detection.infrastructure.logging import configure_logging

# Configure structured logging
configure_logging(
    level="INFO",
    format="json",  # json, console, logfmt
    add_timestamp=True,
    add_caller_info=True,
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ]
)
```

### Log Levels by Component

```yaml
logging:
  root_level: INFO
  loggers:
    anomaly_detection.domain: INFO
    anomaly_detection.infrastructure: DEBUG
    anomaly_detection.algorithms: WARNING
    anomaly_detection.api: INFO
    anomaly_detection.worker: DEBUG
    uvicorn: WARNING
    fastapi: INFO
```

## Performance Tuning

### Memory Configuration

```yaml
performance:
  memory:
    max_memory_per_job: 2GB
    max_memory_per_worker: 4GB
    garbage_collection:
      enabled: true
      threshold: 80  # Trigger GC at 80% memory usage
    
    data_loading:
      chunk_size: 10000
      use_memory_map: true
      dtype_optimization: true  # Use optimal dtypes
```

### Parallel Processing

```yaml
performance:
  parallel:
    n_jobs: -1  # Use all CPU cores
    backend: threading  # threading, multiprocessing, loky
    batch_size: auto
    pre_dispatch: 2  # 2 * n_jobs
    
    algorithms:
      isolation_forest:
        n_jobs: -1
        parallel_backend: threading
      
      local_outlier_factor:
        n_jobs: -1
        parallel_backend: loky
```

### Caching Configuration

```yaml
performance:
  cache:
    enabled: true
    backend: redis
    
    levels:
      - name: memory
        max_size: 100MB
        ttl: 300
      
      - name: redis
        max_size: 1GB
        ttl: 3600
        eviction_policy: lru
    
    key_patterns:
      - pattern: "detection:*"
        ttl: 1800
      - pattern: "model:*"
        ttl: 86400
```

## Security Configuration

### API Security

```yaml
security:
  authentication:
    enabled: true
    providers:
      - type: api_key
        header_name: X-API-Key
        query_param: api_key
      
      - type: jwt
        algorithm: RS256
        public_key_path: /etc/anomaly_detection/jwt_public.pem
        issuer: https://auth.example.com
        audience: anomaly-detection-api
  
  authorization:
    enabled: true
    rbac:
      roles:
        - name: admin
          permissions: ["*"]
        - name: analyst
          permissions: ["detect", "read"]
        - name: viewer
          permissions: ["read"]
  
  encryption:
    data_at_rest: true
    algorithm: AES-256-GCM
    key_rotation_days: 90
```

### Input Validation

```yaml
security:
  validation:
    max_request_size: 100MB
    max_array_length: 1000000
    max_features: 10000
    
    sanitization:
      remove_nan: true
      remove_inf: true
      clip_outliers: true
      outlier_std_threshold: 10
```

## Monitoring Configuration

### Metrics Export

```yaml
monitoring:
  metrics:
    enabled: true
    
    prometheus:
      enabled: true
      port: 9090
      path: /metrics
      
      metrics:
        - name: anomaly_detection_requests_total
          type: counter
          labels: ["method", "endpoint", "status"]
        
        - name: anomaly_detection_processing_duration_seconds
          type: histogram
          labels: ["algorithm"]
          buckets: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        - name: anomaly_detection_active_jobs
          type: gauge
          labels: ["priority", "status"]
    
    statsd:
      enabled: true
      host: localhost
      port: 8125
      prefix: anomaly_detection
```

### Distributed Tracing

```yaml
monitoring:
  tracing:
    enabled: true
    
    opentelemetry:
      service_name: anomaly-detection
      sampling_rate: 0.1
      
      exporters:
        - type: jaeger
          endpoint: http://jaeger:14268/api/traces
          
        - type: zipkin
          endpoint: http://zipkin:9411/api/v2/spans
      
      propagators:
        - tracecontext
        - baggage
```

### Health Checks

```yaml
monitoring:
  health_checks:
    enabled: true
    
    checks:
      - name: database
        type: sql
        connection_string: ${DATABASE_URL}
        query: "SELECT 1"
        timeout: 5s
        
      - name: redis
        type: redis
        url: ${REDIS_URL}
        timeout: 5s
        
      - name: disk_space
        type: disk
        path: /var/lib/anomaly_detection
        min_free_bytes: 1GB
        
      - name: memory
        type: memory
        max_usage_percent: 90
```

## Loading Configuration

### In Python

```python
from anomaly_detection.infrastructure.config import load_config

# Load from file
config = load_config("config.yaml")

# Load with environment overrides
config = load_config("config.yaml", use_env=True)

# Load environment-specific
config = load_config(f"config.{os.getenv('ENV', 'development')}.yaml")
```

### Via CLI

```bash
# Specify config file
anomaly-detection --config /etc/anomaly_detection/config.yaml detect ...

# Override specific values
anomaly-detection \
  --set server.port=8002 \
  --set algorithms.default=lof \
  detect ...
```

## Best Practices

1. **Use Environment Variables** for secrets and deployment-specific values
2. **Use Configuration Files** for complex, structured configuration
3. **Implement Validation** for all configuration values
4. **Document Defaults** clearly in code and configuration
5. **Version Configuration** files alongside code
6. **Monitor Configuration** changes in production
7. **Test Configurations** in staging before production
8. **Secure Sensitive Data** using encryption or secret management tools

## Configuration Validation

The package validates configuration on startup:

```python
from anomaly_detection.infrastructure.config import validate_config

# Validate configuration
errors = validate_config(config)
if errors:
    for error in errors:
        print(f"Configuration error: {error}")
    sys.exit(1)
```

## Migration from Previous Versions

If migrating from an older version:

```bash
# Generate migration report
anomaly-detection config migrate --from 0.x --to 1.0 --dry-run

# Apply migration
anomaly-detection config migrate --from 0.x --to 1.0 --apply
```