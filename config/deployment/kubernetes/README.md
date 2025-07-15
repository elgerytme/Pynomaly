# Pynomaly Data Science Kubernetes Deployment

This directory contains comprehensive Kubernetes manifests for deploying the Pynomaly Data Science infrastructure in production environments.

## Overview

The deployment supports:
- **Horizontal scaling to 100+ nodes** for handling enterprise workloads
- **10,000+ concurrent API requests** with optimized load balancing
- **99.9% uptime** through redundancy and health monitoring
- **Enterprise security** with RBAC, network policies, and encryption
- **Comprehensive monitoring** with Prometheus, Grafana, and InfluxDB
- **High-performance data processing** with dedicated worker nodes

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│  Nginx (3x)     │────│  API (5-100x)   │
│   (External)    │    │  Rate Limiting  │    │  Auto-scaling   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Workers (3-50x)│────│   Message Queue │────│  PostgreSQL     │
│  Background     │    │   Redis Cluster │    │  (HA)           │
│  Processing     │    └─────────────────┘    └─────────────────┘
└─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Monitoring     │────│   Metrics DB    │────│  Storage        │
│  Grafana        │    │   InfluxDB      │    │  Persistent     │
│  Prometheus     │    │   (Time-series) │    │  Volumes        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Deployment Files

### Core Infrastructure
- `data-science-namespace.yaml` - Namespace, resource quotas, and limits
- `data-science-storage.yaml` - Persistent volume claims and storage classes
- `data-science-secrets.yaml` - Secure credential management
- `data-science-configmaps.yaml` - Application configuration

### Database Layer
- `data-science-database.yaml` - PostgreSQL with high availability
- `data-science-redis.yaml` - Redis for caching and sessions
- `data-science-influxdb.yaml` - InfluxDB for metrics and time-series data

### Application Layer
- `data-science-api.yaml` - FastAPI application with auto-scaling (5-100 replicas)
- `data-science-worker.yaml` - Background processing workers (3-50 replicas)

### Monitoring & Observability
- `data-science-monitoring.yaml` - Prometheus and Grafana
- `data-science-nginx.yaml` - Load balancer and reverse proxy

### Deployment Orchestration
- `data-science-deploy.yaml` - Complete deployment with security policies
- `README.md` - This documentation file

## Prerequisites

### Kubernetes Cluster Requirements
```yaml
Minimum Resources:
  - CPU: 32 cores
  - Memory: 128GB RAM
  - Storage: 2TB SSD
  - Nodes: 3 (with anti-affinity)

Recommended for Production:
  - CPU: 100+ cores
  - Memory: 500GB+ RAM
  - Storage: 5TB+ SSD
  - Nodes: 10+ (distributed across availability zones)
```

### Required Kubernetes Features
- **Metrics Server** (for Horizontal Pod Autoscaler)
- **Storage Class** with SSD support
- **Load Balancer Controller** (AWS ALB, GCP, etc.)
- **Ingress Controller** (nginx-ingress recommended)
- **Cert-Manager** (for TLS certificates)

### Security Requirements
- **RBAC enabled** on cluster
- **Pod Security Policies** or **Pod Security Standards**
- **Network Policies** support
- **Secrets encryption at rest**

## Quick Start

### 1. Validate Prerequisites
```bash
# Check cluster resources
kubectl top nodes
kubectl get storageclass

# Verify required controllers
kubectl get pods -n kube-system | grep -E "(metrics-server|ingress|cert-manager)"
```

### 2. Deploy Infrastructure
```bash
# Deploy in order (dependencies matter)
kubectl apply -f data-science-namespace.yaml
kubectl apply -f data-science-storage.yaml
kubectl apply -f data-science-secrets.yaml
kubectl apply -f data-science-configmaps.yaml
```

### 3. Deploy Database Layer
```bash
kubectl apply -f data-science-database.yaml
kubectl apply -f data-science-redis.yaml
kubectl apply -f data-science-influxdb.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=pynomaly,component=postgres -n pynomaly-data-science --timeout=300s
kubectl wait --for=condition=ready pod -l app=pynomaly,component=redis -n pynomaly-data-science --timeout=180s
kubectl wait --for=condition=ready pod -l app=pynomaly,component=influxdb -n pynomaly-data-science --timeout=180s
```

### 4. Deploy Application Layer
```bash
kubectl apply -f data-science-api.yaml
kubectl apply -f data-science-worker.yaml

# Wait for API to be ready
kubectl wait --for=condition=ready pod -l app=pynomaly,component=api -n pynomaly-data-science --timeout=300s
```

### 5. Deploy Monitoring & Load Balancer
```bash
kubectl apply -f data-science-monitoring.yaml
kubectl apply -f data-science-nginx.yaml

# Deploy complete orchestration
kubectl apply -f data-science-deploy.yaml
```

### 6. Verify Deployment
```bash
# Check all pods are running
kubectl get pods -n pynomaly-data-science

# Check services and ingress
kubectl get svc,ingress -n pynomaly-data-science

# Test API health
kubectl port-forward -n pynomaly-data-science svc/pynomaly-nginx-service 8080:80
curl http://localhost:8080/health
```

## Configuration

### Environment-Specific Settings

**Development:**
```yaml
replicas: 1-2
resources: minimal
monitoring: basic
security: relaxed
```

**Staging:**
```yaml
replicas: 2-5
resources: moderate
monitoring: full
security: enforced
```

**Production:**
```yaml
replicas: 5-100 (auto-scaling)
resources: optimized
monitoring: comprehensive
security: strict
```

### Scaling Configuration

The deployment includes Horizontal Pod Autoscalers (HPA) configured for:

**API Scaling:**
- **Min replicas:** 5
- **Max replicas:** 100
- **CPU target:** 70%
- **Memory target:** 80%
- **Scale-up:** Aggressive (100% increase every 15s)
- **Scale-down:** Conservative (10% decrease every 60s)

**Worker Scaling:**
- **Min replicas:** 3
- **Max replicas:** 50
- **CPU target:** 75%
- **Memory target:** 85%
- **Queue-based scaling:** Supported via custom metrics

### Security Configuration

**Network Security:**
- Network policies restrict inter-pod communication
- Ingress with rate limiting (100 req/min per IP)
- TLS termination with cert-manager

**Pod Security:**
- Non-root containers enforced
- Read-only root filesystems where possible
- Capability dropping (ALL capabilities dropped)
- Security contexts with appropriate user/group IDs

**Secret Management:**
- Kubernetes secrets for sensitive data
- Separate secrets per component
- Rotation support through external secret managers

## Monitoring

### Available Endpoints

**Health Monitoring:**
- API Health: `http://api.pynomaly.io/api/v1/health`
- Prometheus: `http://prometheus.pynomaly.io` (internal)
- Grafana: `http://grafana.pynomaly.io`

**Metrics Collection:**
- **Application metrics:** Custom Prometheus metrics
- **Infrastructure metrics:** Node, pod, and container metrics
- **Business metrics:** Request rates, error rates, latencies
- **Resource metrics:** CPU, memory, disk, network usage

### Dashboard Access

**Grafana Dashboards:**
1. **API Performance:** Request rates, response times, error rates
2. **Infrastructure:** Node health, pod status, resource utilization  
3. **Data Science:** ML pipeline metrics, model performance
4. **Security:** Failed authentication, rate limiting, network policies

### Alerting

Critical alerts configured for:
- Pod crash loops or high restart rates
- High memory/CPU usage (>90% for 5 minutes)
- Database connection failures
- API response time degradation (>2s average)
- Storage space exhaustion (>90% full)

## Troubleshooting

### Common Issues

**Pod Scheduling Problems:**
```bash
# Check node resources
kubectl describe nodes

# Check resource quotas
kubectl describe quota -n pynomaly-data-science

# Check pod events
kubectl describe pod <pod-name> -n pynomaly-data-science
```

**Database Connection Issues:**
```bash
# Check database pod logs
kubectl logs -f statefulset/pynomaly-postgres -n pynomaly-data-science

# Test database connectivity
kubectl run -it --rm debug --image=postgres:15-alpine --restart=Never -- psql -h pynomaly-postgres-service.pynomaly-data-science.svc.cluster.local -U pynomaly
```

**Performance Issues:**
```bash
# Check resource usage
kubectl top pods -n pynomaly-data-science

# Check HPA status  
kubectl get hpa -n pynomaly-data-science

# Review API logs for bottlenecks
kubectl logs -f deployment/pynomaly-api -n pynomaly-data-science
```

### Log Analysis

**Centralized Logging:**
```bash
# API logs
kubectl logs -f deployment/pynomaly-api -n pynomaly-data-science

# Worker logs
kubectl logs -f deployment/pynomaly-worker -n pynomaly-data-science

# Database logs
kubectl logs -f statefulset/pynomaly-postgres -n pynomaly-data-science

# Nginx access logs
kubectl logs -f deployment/pynomaly-nginx -n pynomaly-data-science
```

## Performance Optimization

### Database Tuning

**PostgreSQL:**
- Connection pooling: 500 max connections
- Shared buffers: 2GB (25% of RAM)
- Work memory: 32MB per operation
- Maintenance work memory: 512MB

**Redis:**
- Max memory: 8GB with LRU eviction
- Persistence: AOF + RDB snapshots
- TCP keepalive: 60 seconds

**InfluxDB:**
- Retention policy: 30 days for high-resolution data
- Downsampling: Automated for long-term storage
- Query optimization: Indexed tags and fields

### Application Tuning

**API Optimization:**
- Connection pooling for database
- Redis caching for frequent queries
- Async processing for heavy operations
- Request/response compression

**Worker Optimization:**
- Task queuing with Redis
- Parallel processing within workers
- Memory-efficient data processing
- Graceful task cancellation

## Backup and Disaster Recovery

### Backup Strategy

**Database Backups:**
```bash
# PostgreSQL automated backups
kubectl create cronjob postgres-backup \
  --image=postgres:15-alpine \
  --schedule="0 2 * * *" \
  --restart=OnFailure \
  -- pg_dump -h pynomaly-postgres-service -U pynomaly pynomaly > /backup/pynomaly-$(date +%Y%m%d).sql
```

**Persistent Volume Snapshots:**
- Daily snapshots of all PVCs
- Cross-region replication for critical data
- Point-in-time recovery capabilities

### Disaster Recovery

**RTO (Recovery Time Objective):** < 1 hour
**RPO (Recovery Point Objective):** < 15 minutes

**Recovery Procedures:**
1. Deploy infrastructure in DR region
2. Restore database from latest snapshot
3. Update DNS to point to DR cluster
4. Verify application functionality
5. Monitor for performance issues

## Security Hardening

### Network Security
- All inter-service communication encrypted
- Network segmentation via Kubernetes NetworkPolicies
- Ingress with WAF and DDoS protection
- Regular security scanning of container images

### Access Control
- Least privilege RBAC policies
- Service account token auto-mounting disabled
- Pod security contexts with non-root users
- Admission controllers for policy enforcement

### Compliance
- SOC 2 Type II compatible configurations
- GDPR compliance for data handling
- Audit logging for all administrative actions
- Regular vulnerability assessments

## Maintenance

### Regular Tasks

**Daily:**
- Monitor resource usage and scaling events
- Review error logs and alert notifications
- Verify backup completion and integrity

**Weekly:**
- Update security patches for base images
- Review and rotate access credentials
- Analyze performance metrics and trends
- Test disaster recovery procedures

**Monthly:**
- Capacity planning and resource optimization
- Security audit and compliance review
- Update dependencies and libraries
- Performance tuning based on usage patterns

### Upgrade Procedures

**Application Updates:**
1. Deploy to staging environment
2. Run comprehensive test suite
3. Perform rolling update in production
4. Monitor metrics and rollback if needed

**Infrastructure Updates:**
1. Update Kubernetes cluster (one node at a time)
2. Update operators and controllers
3. Verify all workloads remain healthy
4. Update monitoring and logging stack

For additional support and enterprise deployment assistance, contact the Pynomaly team at support@pynomaly.io.