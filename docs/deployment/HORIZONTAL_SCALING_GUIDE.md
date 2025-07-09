# Horizontal Scaling Guide for Pynomaly

This guide covers horizontal scaling strategies for Pynomaly using Gunicorn workers and Kubernetes Horizontal Pod Autoscaler (HPA).

## Table of Contents

1. [Overview](#overview)
2. [Gunicorn Worker Scaling](#gunicorn-worker-scaling)
3. [Kubernetes HPA Configuration](#kubernetes-hpa-configuration)
4. [Performance Monitoring](#performance-monitoring)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

## Overview

Pynomaly supports horizontal scaling to handle increased load and improve performance:

- **Gunicorn Workers**: Scale API server processes
- **Celery Workers**: Scale background task processing
- **Kubernetes HPA**: Automatically scale pods based on metrics
- **Load Balancing**: Distribute requests across instances

## Gunicorn Worker Scaling

### Basic Configuration

```bash
# Start with multiple workers
gunicorn --workers 4 --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --worker-connections 1000 \
  --max-requests 1000 \
  --max-requests-jitter 50 \
  --preload \
  --timeout 30 \
  --keepalive 2 \
  pynomaly.presentation.api.app:create_app
```

### Dynamic Worker Calculation

```python
# Calculate optimal worker count
import multiprocessing

def calculate_workers():
    """Calculate optimal number of workers based on CPU cores."""
    cpu_count = multiprocessing.cpu_count()
    return min(max(2, cpu_count * 2 + 1), 12)

workers = calculate_workers()
```

### Gunicorn Configuration File

Create `gunicorn.conf.py`:

```python
import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
preload_app = True
timeout = 30
keepalive = 2

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "pynomaly-api"

# Security
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# Performance
worker_tmp_dir = "/dev/shm"  # Use RAM for worker tmp files
```

### Environment-Specific Scaling

```bash
# Development (single worker)
gunicorn --workers 1 --reload pynomaly.presentation.api.app:create_app

# Production (multiple workers)
gunicorn --config gunicorn.conf.py pynomaly.presentation.api.app:create_app

# High load (optimized)
gunicorn --workers 8 --worker-class uvicorn.workers.UvicornWorker \
  --worker-connections 2000 \
  --max-requests 2000 \
  --preload \
  --bind 0.0.0.0:8000 \
  pynomaly.presentation.api.app:create_app
```

## Kubernetes HPA Configuration

### Prerequisites

1. Metrics Server installed in cluster
2. Resource requests defined in deployment
3. Prometheus (optional, for custom metrics)

### Basic HPA Configuration

```yaml
# hpa-basic.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pynomaly-api-hpa
  namespace: pynomaly
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pynomaly-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 2
        periodSeconds: 60
```

### Advanced HPA with Custom Metrics

```yaml
# hpa-advanced.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pynomaly-api-hpa-advanced
  namespace: pynomaly
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pynomaly-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  # CPU utilization
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  # Memory utilization
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  # Request rate (custom metric)
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  # Response time (custom metric)
  - type: Pods
    pods:
      metric:
        name: http_request_duration_seconds
      target:
        type: AverageValue
        averageValue: "500m"
  # Queue length (for Celery workers)
  - type: External
    external:
      metric:
        name: celery_queue_length
      target:
        type: AverageValue
        averageValue: "50"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 25
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 3
        periodSeconds: 60
```

### Deployment with Resource Requests

```yaml
# deployment-scalable.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pynomaly-api
  namespace: pynomaly
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pynomaly-api
  template:
    metadata:
      labels:
        app: pynomaly-api
    spec:
      containers:
      - name: pynomaly-api
        image: pynomaly:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        env:
        - name: WORKERS
          value: "4"
        - name: WORKER_CLASS
          value: "uvicorn.workers.UvicornWorker"
        - name: WORKER_CONNECTIONS
          value: "1000"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Celery Worker HPA

```yaml
# celery-worker-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pynomaly-celery-hpa
  namespace: pynomaly
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pynomaly-celery-worker
  minReplicas: 2
  maxReplicas: 15
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 80
  - type: External
    external:
      metric:
        name: celery_queue_length
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
      - type: Pods
        value: 2
        periodSeconds: 30
```

## Performance Monitoring

### Monitoring Setup

```python
# monitoring.py
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections')
WORKER_MEMORY = Gauge('worker_memory_usage_bytes', 'Worker memory usage')
WORKER_CPU = Gauge('worker_cpu_usage_percent', 'Worker CPU usage')

def monitor_performance():
    """Monitor system performance metrics."""
    while True:
        # Update system metrics
        WORKER_MEMORY.set(psutil.virtual_memory().used)
        WORKER_CPU.set(psutil.cpu_percent())
        
        time.sleep(5)

# Start metrics server
start_http_server(8001)
```

### Grafana Dashboard

Key metrics to monitor:

1. **Request Rate**: Requests per second
2. **Response Time**: Average and percentiles
3. **Error Rate**: HTTP 5xx errors
4. **Resource Usage**: CPU, memory, disk
5. **Queue Length**: Celery task queue
6. **Worker Status**: Active/idle workers

### Alerting Rules

```yaml
# alerts.yaml
groups:
- name: pynomaly-scaling
  rules:
  - alert: HighRequestRate
    expr: rate(http_requests_total[5m]) > 1000
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High request rate detected"
      
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High response time detected"
      
  - alert: LongQueueLength
    expr: celery_queue_length > 500
    for: 3m
    labels:
      severity: warning
    annotations:
      summary: "Long Celery queue detected"
```

## Best Practices

### 1. Resource Planning

```python
# Resource calculation
def calculate_resources(expected_rps, avg_response_time_ms):
    """Calculate required resources based on expected load."""
    
    # Rule of thumb: 1 worker can handle ~100 RPS
    workers_needed = max(2, int(expected_rps / 100))
    
    # Memory: ~500MB base + 200MB per worker
    memory_gb = 0.5 + (workers_needed * 0.2)
    
    # CPU: ~0.5 core base + 0.2 cores per worker
    cpu_cores = 0.5 + (workers_needed * 0.2)
    
    return {
        'workers': workers_needed,
        'memory_gb': memory_gb,
        'cpu_cores': cpu_cores,
        'max_replicas': min(workers_needed * 2, 20)
    }
```

### 2. Load Testing

```bash
# Load test with Apache Bench
ab -n 10000 -c 100 http://localhost:8000/health

# Load test with wrk
wrk -t12 -c400 -d30s http://localhost:8000/api/v1/detectors

# Load test with hey
hey -n 10000 -c 100 -t 30 http://localhost:8000/health
```

### 3. Optimization Strategies

1. **Connection Pooling**: Use connection pools for databases
2. **Caching**: Implement Redis caching for frequent queries
3. **Async Processing**: Use Celery for heavy computations
4. **CDN**: Use CDN for static assets
5. **Database Optimization**: Index frequently queried fields

### 4. Scaling Patterns

```python
# Gradual scaling
def gradual_scale(current_load, target_load, max_step=2):
    """Gradually scale up to avoid overwhelming the system."""
    if target_load > current_load:
        return min(current_load + max_step, target_load)
    return target_load

# Predictive scaling
def predictive_scale(historical_data, time_ahead_minutes=15):
    """Predict future load and scale proactively."""
    # Simple moving average prediction
    recent_values = historical_data[-5:]
    predicted_load = sum(recent_values) / len(recent_values)
    
    # Add buffer for peak hours
    if is_peak_hour():
        predicted_load *= 1.5
        
    return predicted_load
```

## Troubleshooting

### Common Issues

1. **HPA Not Scaling**
   - Check metrics server is running
   - Verify resource requests are set
   - Check HPA status: `kubectl describe hpa`

2. **Workers Not Starting**
   - Check resource limits
   - Verify container image
   - Check logs: `kubectl logs -f deployment/pynomaly-api`

3. **High Memory Usage**
   - Check for memory leaks
   - Tune worker recycling
   - Monitor object pools

4. **Slow Response Times**
   - Check database connections
   - Monitor queue lengths
   - Verify network latency

### Debugging Commands

```bash
# Check HPA status
kubectl get hpa -n pynomaly

# View HPA events
kubectl describe hpa pynomaly-api-hpa -n pynomaly

# Check pod resources
kubectl top pods -n pynomaly

# View metrics
kubectl get --raw /apis/metrics.k8s.io/v1beta1/namespaces/pynomaly/pods

# Check Celery workers
celery -A pynomaly.infrastructure.async_tasks.celery_tasks inspect active

# Monitor queue length
celery -A pynomaly.infrastructure.async_tasks.celery_tasks flower
```

### Performance Tuning

1. **Worker Configuration**
   ```python
   # Optimize worker settings
   worker_connections = 1000
   max_requests = 1000
   max_requests_jitter = 50
   preload_app = True
   ```

2. **Database Optimization**
   ```python
   # Connection pooling
   DATABASE_URL = "postgresql://user:pass@host:5432/db?pool_size=20&max_overflow=0"
   ```

3. **Caching Strategy**
   ```python
   # Redis configuration
   REDIS_URL = "redis://localhost:6379/0"
   CACHE_TTL = 300  # 5 minutes
   ```

## Conclusion

Horizontal scaling in Pynomaly involves multiple components working together:

- **Gunicorn**: Scales API server processes
- **Celery**: Scales background task processing
- **Kubernetes HPA**: Automatically scales pods
- **Monitoring**: Provides visibility into performance

Follow the guidelines in this document to implement effective horizontal scaling for your Pynomaly deployment.

For more information, see:
- [Kubernetes HPA Documentation](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Gunicorn Configuration](https://docs.gunicorn.org/en/stable/configure.html)
- [Celery Monitoring](https://docs.celeryproject.org/en/stable/userguide/monitoring.html)
