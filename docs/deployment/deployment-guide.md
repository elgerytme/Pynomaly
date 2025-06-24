# Pynomaly Deployment Pipeline Guide

## Overview

This guide provides comprehensive documentation for deploying and serving Pynomaly anomaly detection models in production environments. The deployment pipeline supports enterprise-grade MLOps practices with automated deployment strategies, health monitoring, and rollback mechanisms.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Deployment Strategies](#deployment-strategies)
3. [Environment Management](#environment-management)
4. [Model Serving API](#model-serving-api)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [CLI Commands](#cli-commands)
7. [Monitoring and Health Checks](#monitoring-and-health-checks)
8. [Security and RBAC](#security-and-rbac)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

## Quick Start

### 1. Deploy Your First Model

```bash
# Deploy a model to staging environment
pynomaly deploy deploy <model-version-id> \
  --env staging \
  --strategy rolling \
  --replicas 2

# Check deployment status
pynomaly deploy status <deployment-id>

# Promote to production
pynomaly deploy promote <deployment-id> \
  --notes "Performance validated in staging"
```

### 2. Start Model Serving API

```bash
# Start local development server
pynomaly deploy serve --host 0.0.0.0 --port 8080

# Start production server with multiple workers
pynomaly deploy serve \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 4 \
  --env production
```

### 3. Make Predictions

```bash
# Single prediction
curl -X POST "http://localhost:8080/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0},
    "return_confidence": true
  }'

# Batch predictions
curl -X POST "http://localhost:8080/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"feature1": 1.0, "feature2": 2.0},
      {"feature1": 2.0, "feature2": 3.0}
    ],
    "batch_size": 1000
  }'
```

## Deployment Strategies

### Rolling Deployment

Gradually replaces instances one by one with zero downtime.

```bash
pynomaly deploy deploy <model-version-id> \
  --strategy rolling \
  --env production
```

**Characteristics:**
- ✅ Zero downtime
- ✅ Resource efficient
- ✅ Gradual rollout
- ⚠️ Mixed versions during deployment

### Blue-Green Deployment

Maintains two identical environments and switches traffic instantly.

```bash
pynomaly deploy deploy <model-version-id> \
  --strategy blue_green \
  --env production
```

**Characteristics:**
- ✅ Instant rollback
- ✅ Zero downtime
- ✅ Full testing before switch
- ⚠️ Requires double resources

### Canary Deployment

Routes a small percentage of traffic to the new version gradually.

```bash
pynomaly deploy deploy <model-version-id> \
  --strategy canary \
  --env production
```

**Characteristics:**
- ✅ Risk mitigation
- ✅ Gradual validation
- ✅ A/B testing capability
- ⚠️ Complex traffic management

### Direct Deployment

Replaces the existing deployment immediately.

```bash
pynomaly deploy deploy <model-version-id> \
  --strategy direct \
  --env development
```

**Characteristics:**
- ✅ Simple and fast
- ✅ Resource efficient
- ⚠️ Brief downtime
- ⚠️ No gradual validation

## Environment Management

### Environment Types

#### Development Environment
- **Purpose**: Local development and initial testing
- **Resources**: Minimal (1 replica, 256Mi memory)
- **Monitoring**: Basic health checks
- **Access**: Open to development team

```bash
pynomaly deploy deploy <model-version-id> \
  --env development \
  --replicas 1 \
  --memory-request 256Mi \
  --cpu-request 100m
```

#### Staging Environment
- **Purpose**: Pre-production testing and validation
- **Resources**: Production-like (2-3 replicas, 1Gi memory)
- **Monitoring**: Full monitoring and alerting
- **Access**: QA and staging validation team

```bash
pynomaly deploy deploy <model-version-id> \
  --env staging \
  --replicas 2 \
  --memory-request 1Gi \
  --cpu-request 500m
```

#### Production Environment
- **Purpose**: Live model serving
- **Resources**: High availability (5+ replicas, 4Gi memory)
- **Monitoring**: Comprehensive monitoring and alerting
- **Access**: Restricted to ops team

```bash
pynomaly deploy deploy <model-version-id> \
  --env production \
  --replicas 5 \
  --memory-request 2Gi \
  --cpu-request 1000m
```

### Environment Promotion Workflow

```bash
# 1. Deploy to development
pynomaly deploy deploy <model-version-id> --env development

# 2. Validate and deploy to staging
pynomaly deploy deploy <model-version-id> --env staging

# 3. Promote to production (requires approval)
pynomaly deploy promote <staging-deployment-id> \
  --notes "Performance metrics validated"
```

### Check Environment Status

```bash
# View all environments
pynomaly deploy environments

# View specific environment
pynomaly deploy environments --env production --json
```

## Model Serving API

### API Endpoints

#### Prediction Endpoints

##### Single Prediction
```http
POST /api/v1/predict
Content-Type: application/json

{
  "data": {
    "feature1": 1.0,
    "feature2": 2.0,
    "feature3": 3.0
  },
  "model_id": "optional-model-id",
  "version": "optional-version",
  "return_confidence": true,
  "return_explanation": false
}
```

**Response:**
```json
{
  "prediction": 0.75,
  "is_anomaly": true,
  "confidence": 0.92,
  "model_id": "model-123",
  "model_version": "1.0.0",
  "prediction_time": "2025-06-24T10:30:00Z",
  "latency_ms": 45.2
}
```

##### Batch Prediction
```http
POST /api/v1/predict/batch
Content-Type: application/json

{
  "data": [
    {"feature1": 1.0, "feature2": 2.0},
    {"feature1": 2.0, "feature2": 3.0}
  ],
  "batch_size": 1000,
  "return_confidence": true
}
```

**Response:**
```json
{
  "predictions": [...],
  "total_processed": 2,
  "total_anomalies": 1,
  "processing_time_ms": 123.4,
  "throughput_per_second": 16.2
}
```

##### Streaming Prediction (WebSocket)
```javascript
// WebSocket connection
const ws = new WebSocket('ws://localhost:8080/api/v1/predict/stream');

// Send prediction request
ws.send(JSON.stringify({
  "data": {"feature1": 1.0, "feature2": 2.0},
  "return_confidence": true
}));

// Receive prediction response
ws.onmessage = function(event) {
  const prediction = JSON.parse(event.data);
  console.log('Anomaly score:', prediction.prediction);
};
```

#### Model Management Endpoints

##### List Models
```http
GET /api/v1/models
```

##### Get Model Info
```http
GET /api/v1/models/{model_id}
```

##### Load Model
```http
POST /api/v1/models/{model_id}/load
```

##### Unload Model
```http
DELETE /api/v1/models/{model_id}/unload
```

#### Health and Monitoring Endpoints

##### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-06-24T10:30:00Z",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "loaded_models": 3,
  "active_connections": 15,
  "memory_usage_mb": 1024.5
}
```

##### Readiness Check
```http
GET /ready
```

##### Prometheus Metrics
```http
GET /metrics
```

### Performance Tuning

#### Model Caching
- **Cache Size**: Configure with `model_cache_size` parameter
- **LRU Eviction**: Automatically removes least recently used models
- **Memory Management**: Monitor memory usage and adjust cache size

#### Batch Processing
- **Optimal Batch Size**: Start with 1000, adjust based on memory and latency
- **Async Processing**: Use async endpoints for better throughput
- **Connection Pooling**: Use HTTP/2 or connection pooling for multiple requests

#### Resource Allocation
```yaml
# Kubernetes resource configuration
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

## Kubernetes Deployment

### Prerequisites

1. **Kubernetes Cluster**: Version 1.20+
2. **kubectl**: Configured and connected to cluster
3. **Container Registry**: Docker images available
4. **Storage Class**: For persistent volumes

### Deployment Steps

#### 1. Create Namespace and Resources

```bash
# Apply complete deployment
kubectl apply -f docker/kubernetes/model-server-deployment.yaml

# Verify deployment
kubectl get pods -n pynomaly-production
kubectl get services -n pynomaly-production
kubectl get ingress -n pynomaly-production
```

#### 2. Configure Environment Variables

```yaml
# ConfigMap example
apiVersion: v1
kind: ConfigMap
metadata:
  name: pynomaly-model-server-config
  namespace: pynomaly-production
data:
  ENVIRONMENT: "production"
  MODEL_SERVER_PORT: "8080"
  LOG_LEVEL: "info"
  WORKERS: "3"
```

#### 3. Configure Secrets

```bash
# Create secret for sensitive data
kubectl create secret generic pynomaly-model-server-secrets \
  --from-literal=database_password=<password> \
  --namespace=pynomaly-production
```

#### 4. Monitor Deployment

```bash
# Check deployment status
kubectl rollout status deployment/pynomaly-model-server -n pynomaly-production

# View logs
kubectl logs -f deployment/pynomaly-model-server -n pynomaly-production

# Check health
kubectl port-forward service/pynomaly-model-server-service 8080:80 -n pynomaly-production
curl http://localhost:8080/health
```

### Auto-Scaling Configuration

```yaml
# HPA configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pynomaly-model-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pynomaly-model-server
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Ingress Configuration

```yaml
# Ingress with SSL
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pynomaly-model-server-ingress
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - pynomaly-api.yourdomain.com
    secretName: pynomaly-tls-secret
  rules:
  - host: pynomaly-api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: pynomaly-model-server-service
            port:
              number: 80
```

## CLI Commands

### Deployment Management

#### List Deployments
```bash
# List all deployments
pynomaly deploy list

# Filter by environment
pynomaly deploy list --env production

# Filter by status
pynomaly deploy list --status deployed

# JSON output
pynomaly deploy list --json
```

#### Deploy Model
```bash
# Basic deployment
pynomaly deploy deploy <model-version-id> --env staging

# Advanced deployment with custom configuration
pynomaly deploy deploy <model-version-id> \
  --env production \
  --strategy canary \
  --replicas 5 \
  --cpu-request 1000m \
  --cpu-limit 2000m \
  --memory-request 2Gi \
  --memory-limit 4Gi \
  --wait
```

#### Check Deployment Status
```bash
# Detailed status
pynomaly deploy status <deployment-id>

# JSON output for automation
pynomaly deploy status <deployment-id> --json
```

#### Rollback Deployment
```bash
# Manual rollback
pynomaly deploy rollback <deployment-id> \
  --reason "Performance regression detected"

# Wait for rollback completion
pynomaly deploy rollback <deployment-id> --wait
```

#### Promote to Production
```bash
# Promote staging to production
pynomaly deploy promote <staging-deployment-id> \
  --notes "All validation tests passed"
```

#### Environment Status
```bash
# View all environments
pynomaly deploy environments

# JSON output
pynomaly deploy environments --json
```

### Model Serving

#### Start Server
```bash
# Development server
pynomaly deploy serve --reload

# Production server
pynomaly deploy serve \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 4 \
  --env production
```

## Monitoring and Health Checks

### Health Check Endpoints

#### Basic Health Check
```bash
curl http://localhost:8080/health
```

#### Readiness Check
```bash
curl http://localhost:8080/ready
```

#### Metrics Collection
```bash
curl http://localhost:8080/metrics
```

### Prometheus Metrics

The model server exposes comprehensive metrics:

```
# Prediction metrics
model_predictions_total{model_id="123", version="1.0.0"} 1500
model_prediction_duration_seconds{model_id="123", version="1.0.0"} 0.045

# Error metrics
model_errors_total{model_id="123", version="1.0.0", error_type="ValidationError"} 5

# System metrics
model_websocket_connections 25
model_load_duration_seconds{model_id="123", version="1.0.0"} 2.3
```

### Custom Alerts

#### High Error Rate Alert
```yaml
# Prometheus alert rule
groups:
- name: pynomaly_alerts
  rules:
  - alert: HighModelErrorRate
    expr: rate(model_errors_total[5m]) > 0.05
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected for model {{ $labels.model_id }}"
```

#### High Latency Alert
```yaml
- alert: HighPredictionLatency
  expr: histogram_quantile(0.95, rate(model_prediction_duration_seconds_bucket[5m])) > 1.0
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "High prediction latency detected"
```

### Log Monitoring

#### Application Logs
```bash
# View live logs
kubectl logs -f deployment/pynomaly-model-server -n pynomaly-production

# Filter error logs
kubectl logs deployment/pynomaly-model-server -n pynomaly-production | grep ERROR
```

#### Structured Logging Format
```json
{
  "timestamp": "2025-06-24T10:30:00Z",
  "level": "INFO",
  "logger": "model_server",
  "message": "Prediction request processed",
  "model_id": "123",
  "latency_ms": 45.2,
  "prediction_score": 0.75
}
```

## Security and RBAC

### Authentication

#### JWT Token Authentication
```python
# Example client authentication
import requests

headers = {
    "Authorization": "Bearer <jwt-token>",
    "Content-Type": "application/json"
}

response = requests.post(
    "https://pynomaly-api.yourdomain.com/api/v1/predict",
    headers=headers,
    json={"data": {"feature1": 1.0}}
)
```

#### API Key Authentication
```bash
curl -X POST "https://pynomaly-api.yourdomain.com/api/v1/predict" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"data": {"feature1": 1.0}}'
```

### Role-Based Access Control

#### Kubernetes RBAC
```yaml
# Service account with limited permissions
apiVersion: v1
kind: ServiceAccount
metadata:
  name: pynomaly-model-server
  namespace: pynomaly-production

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: pynomaly-production
  name: pynomaly-model-server-role
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list", "watch"]
```

#### Application-Level RBAC
```python
# Example role definitions
ROLES = {
    "viewer": ["predict", "health", "metrics"],
    "operator": ["predict", "health", "metrics", "load_model"],
    "admin": ["*"]
}
```

### Input Validation

#### Request Validation
```python
# Pydantic model validation
class PredictionRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Input data")
    model_id: Optional[str] = Field(None, regex=r'^[a-zA-Z0-9\-_]+$')
    return_confidence: bool = Field(True)
    
    @validator('data')
    def validate_data(cls, v):
        if not v:
            raise ValueError('Data cannot be empty')
        return v
```

#### Rate Limiting
```python
# Example rate limiting configuration
RATE_LIMITS = {
    "predict": "100/minute",
    "batch_predict": "10/minute",
    "model_management": "50/hour"
}
```

### Network Security

#### TLS Configuration
```yaml
# TLS secret for HTTPS
apiVersion: v1
kind: Secret
metadata:
  name: pynomaly-tls-secret
  namespace: pynomaly-production
type: kubernetes.io/tls
data:
  tls.crt: <base64-encoded-cert>
  tls.key: <base64-encoded-key>
```

#### Network Policies
```yaml
# Network policy example
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: pynomaly-model-server-netpol
  namespace: pynomaly-production
spec:
  podSelector:
    matchLabels:
      app: pynomaly-model-server
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: pynomaly-frontend
    ports:
    - protocol: TCP
      port: 8080
```

## Troubleshooting

### Common Issues

#### Deployment Failures

**Issue**: Deployment stuck in pending state
```bash
# Check pod events
kubectl describe pod <pod-name> -n pynomaly-production

# Check resource availability
kubectl top nodes
kubectl describe nodes
```

**Solution**: Increase resource requests or add more nodes

**Issue**: Image pull errors
```bash
# Check image availability
docker pull pynomaly/model-server:latest

# Check registry credentials
kubectl get secret regcred -n pynomaly-production -o yaml
```

**Solution**: Verify image tag and registry credentials

#### Model Loading Issues

**Issue**: Model fails to load
```bash
# Check model server logs
kubectl logs -f deployment/pynomaly-model-server -n pynomaly-production | grep "model"

# Check storage access
kubectl exec -it <pod-name> -n pynomaly-production -- ls -la /app/models
```

**Solution**: Verify model file permissions and storage access

**Issue**: Out of memory errors
```bash
# Check memory usage
kubectl top pods -n pynomaly-production

# Check memory limits
kubectl describe pod <pod-name> -n pynomaly-production | grep -A 10 "Limits"
```

**Solution**: Increase memory limits or reduce model cache size

#### Performance Issues

**Issue**: High prediction latency
```bash
# Check metrics
curl http://localhost:8080/metrics | grep prediction_duration

# Check resource utilization
kubectl top pods -n pynomaly-production
```

**Solution**: Increase CPU allocation or add more replicas

**Issue**: Connection timeouts
```bash
# Check service endpoints
kubectl get endpoints -n pynomaly-production

# Check ingress configuration
kubectl describe ingress pynomaly-model-server-ingress -n pynomaly-production
```

**Solution**: Adjust timeout settings and check network connectivity

### Debugging Commands

#### Pod Debugging
```bash
# Get pod shell access
kubectl exec -it <pod-name> -n pynomaly-production -- /bin/bash

# Check processes
kubectl exec -it <pod-name> -n pynomaly-production -- ps aux

# Check disk usage
kubectl exec -it <pod-name> -n pynomaly-production -- df -h
```

#### Network Debugging
```bash
# Test connectivity
kubectl exec -it <pod-name> -n pynomaly-production -- curl http://localhost:8080/health

# Check DNS resolution
kubectl exec -it <pod-name> -n pynomaly-production -- nslookup kubernetes.default
```

#### Log Analysis
```bash
# Filter error logs
kubectl logs deployment/pynomaly-model-server -n pynomaly-production | grep "ERROR\|FATAL"

# Follow logs with timestamps
kubectl logs -f --timestamps deployment/pynomaly-model-server -n pynomaly-production
```

## Best Practices

### Deployment Best Practices

1. **Use Staging Environment**: Always test in staging before production
2. **Gradual Rollouts**: Use canary or blue-green deployments for production
3. **Resource Management**: Set appropriate resource requests and limits
4. **Health Checks**: Configure comprehensive health and readiness probes
5. **Monitoring**: Implement comprehensive monitoring and alerting

### Model Management Best Practices

1. **Version Control**: Use semantic versioning for models
2. **Model Validation**: Validate model performance before deployment
3. **Rollback Strategy**: Always have a tested rollback plan
4. **Performance Monitoring**: Monitor model performance continuously
5. **A/B Testing**: Use canary deployments for A/B testing

### Security Best Practices

1. **Principle of Least Privilege**: Grant minimal required permissions
2. **Input Validation**: Validate all inputs thoroughly
3. **Network Security**: Use network policies and TLS encryption
4. **Regular Updates**: Keep all dependencies and base images updated
5. **Audit Logging**: Log all access and operations for auditing

### Performance Best Practices

1. **Resource Optimization**: Right-size resource allocation
2. **Caching Strategy**: Implement appropriate model and data caching
3. **Load Testing**: Regular load testing to validate performance
4. **Auto-scaling**: Configure auto-scaling for variable loads
5. **Connection Pooling**: Use connection pooling for better throughput

### Operational Best Practices

1. **Documentation**: Maintain up-to-date deployment documentation
2. **Disaster Recovery**: Implement backup and disaster recovery procedures
3. **Change Management**: Use proper change management processes
4. **Team Training**: Ensure team is trained on deployment procedures
5. **Continuous Improvement**: Regularly review and improve processes

This comprehensive guide provides everything needed to deploy and operate Pynomaly models in production environments with enterprise-grade reliability, security, and performance.