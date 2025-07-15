# Pynomaly Multi-Environment Deployment Guide

🚀 **Comprehensive Production Deployment Guide for Pynomaly**

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Environment Configurations](#environment-configurations)
5. [Deployment Strategies](#deployment-strategies)
6. [Container Deployment](#container-deployment)
7. [Kubernetes Deployment](#kubernetes-deployment)
8. [CI/CD Pipeline](#cicd-pipeline)
9. [Infrastructure as Code](#infrastructure-as-code)
10. [Monitoring & Observability](#monitoring--observability)
11. [Security & Compliance](#security--compliance)
12. [Troubleshooting](#troubleshooting)
13. [Rollback Procedures](#rollback-procedures)

---

## Overview

Pynomaly provides enterprise-grade deployment infrastructure supporting multiple environments and deployment strategies. This guide covers the complete deployment process from development to production.

### Supported Environments
- **Development**: Local development environment
- **Staging**: Pre-production testing environment
- **Production**: Live production environment

### Deployment Strategies
- **Rolling Deployment**: Gradual replacement of instances
- **Blue-Green Deployment**: Zero-downtime deployment with instant rollback
- **Canary Deployment**: Gradual traffic shifting to new version

---

## Prerequisites

### System Requirements
- **Docker**: Version 20.10+ and Docker Compose v2.0+
- **Kubernetes**: Version 1.21+ (for K8s deployment)
- **Terraform**: Version 1.0+ (for infrastructure provisioning)
- **Python**: Version 3.11+ with pip
- **Git**: Version 2.30+

### Cloud Provider Requirements
- **AWS**: EKS cluster, RDS PostgreSQL, ElastiCache Redis
- **Azure**: AKS cluster, Azure Database for PostgreSQL
- **GCP**: GKE cluster, Cloud SQL PostgreSQL

### Access Requirements
- Docker Hub or private registry access
- Kubernetes cluster admin access
- Cloud provider credentials
- CI/CD pipeline access (GitHub Actions)

---

## Quick Start

### 1. Local Development Deployment

```bash
# Clone the repository
git clone https://github.com/elgerytme/Pynomaly.git
cd Pynomaly

# Start development environment
docker-compose up -d

# Verify deployment
curl http://localhost:8000/health
```

### 2. Staging Deployment

```bash
# Deploy to staging
./scripts/deploy.sh staging

# Run health checks
pynomaly health full --detailed
```

### 3. Production Deployment

```bash
# Deploy to production
./scripts/deploy.sh production

# Monitor deployment
kubectl get pods -n pynomaly-production
```

---

## Environment Configurations

### Development Environment
```yaml
# config/deployment/development/config.yaml
environment: development
debug: true
database:
  host: localhost
  port: 5432
  name: pynomaly_dev
redis:
  host: localhost
  port: 6379
```

### Staging Environment
```yaml
# config/deployment/staging/config.yaml
environment: staging
debug: false
database:
  host: pynomaly-staging-db.cluster-xxx.us-east-1.rds.amazonaws.com
  port: 5432
  name: pynomaly_staging
redis:
  host: pynomaly-staging-redis.xxx.cache.amazonaws.com
  port: 6379
```

### Production Environment
```yaml
# config/deployment/production/config.yaml
environment: production
debug: false
database:
  host: pynomaly-prod-db.cluster-xxx.us-east-1.rds.amazonaws.com
  port: 5432
  name: pynomaly_production
redis:
  host: pynomaly-prod-redis.xxx.cache.amazonaws.com
  port: 6379
```

---

## Deployment Strategies

### Rolling Deployment
```bash
# Update deployment with rolling strategy
kubectl set image deployment/pynomaly-api api=pynomaly:v2.0.0

# Monitor rollout
kubectl rollout status deployment/pynomaly-api
```

### Blue-Green Deployment
```bash
# Deploy green environment
kubectl apply -f k8s/blue-green/green-deployment.yaml

# Switch traffic
kubectl patch service pynomaly-api -p '{"spec":{"selector":{"version":"green"}}}'

# Cleanup blue environment
kubectl delete deployment pynomaly-api-blue
```

### Canary Deployment
```bash
# Deploy canary version (10% traffic)
kubectl apply -f k8s/canary/canary-deployment.yaml

# Increase traffic to 50%
kubectl patch virtualservice pynomaly-api --type='json' -p='[{"op": "replace", "path": "/spec/http/0/match/0/weight", "value": 50}]'

# Complete rollout
kubectl apply -f k8s/canary/full-deployment.yaml
```

---

## Container Deployment

### Docker Deployment

#### Single Container
```bash
# Build production image
docker build -f deployment/infrastructure/docker/Dockerfile.production -t pynomaly:latest .

# Run container
docker run -d \
  --name pynomaly-api \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@db:5432/pynomaly \
  -e REDIS_URL=redis://redis:6379/0 \
  pynomaly:latest
```

#### Docker Compose (Full Stack)
```bash
# Start complete stack
docker-compose -f docker-compose.yml up -d

# Scale API service
docker-compose up -d --scale api=3

# View logs
docker-compose logs -f api
```

### Multi-Stage Build
```dockerfile
# deployment/infrastructure/docker/Dockerfile.production
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim as runtime
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
EXPOSE 8000
CMD ["uvicorn", "pynomaly.presentation.web.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Kubernetes Deployment

### Basic Deployment
```bash
# Apply namespace
kubectl apply -f k8s/namespace.yaml

# Apply configurations
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml

# Deploy services
kubectl apply -f k8s/deployments.yaml
kubectl apply -f k8s/services.yaml

# Apply ingress
kubectl apply -f k8s/ingress.yaml
```

### Production Configuration
```yaml
# k8s/deployments.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pynomaly-api
  namespace: pynomaly-production
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
      - name: api
        image: pynomaly:v2.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: pynomaly-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
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

### Auto-scaling Configuration
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pynomaly-api-hpa
  namespace: pynomaly-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pynomaly-api
  minReplicas: 3
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
```

---

## CI/CD Pipeline

### GitHub Actions Workflow
```yaml
# .github/workflows/deployment.yml
name: Deploy to Production

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v1
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Build and push Docker image
      run: |
        docker build -t pynomaly:${{ github.sha }} .
        docker tag pynomaly:${{ github.sha }} ${{ secrets.ECR_REGISTRY }}/pynomaly:${{ github.sha }}
        docker push ${{ secrets.ECR_REGISTRY }}/pynomaly:${{ github.sha }}
    
    - name: Deploy to EKS
      run: |
        aws eks update-kubeconfig --region us-east-1 --name pynomaly-production
        kubectl set image deployment/pynomaly-api api=${{ secrets.ECR_REGISTRY }}/pynomaly:${{ github.sha }}
        kubectl rollout status deployment/pynomaly-api
```

### Deployment Environments
```yaml
# Environment-specific deployments
staging:
  environment: staging
  if: github.ref == 'refs/heads/develop'
  
production:
  environment: production
  if: github.ref == 'refs/heads/main'
  needs: [tests, security-scan]
```

---

## Infrastructure as Code

### Terraform Configuration
```hcl
# config/deployment/terraform/main.tf
provider "aws" {
  region = var.aws_region
}

# EKS Cluster
resource "aws_eks_cluster" "pynomaly" {
  name     = "pynomaly-production"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = "1.24"

  vpc_config {
    subnet_ids = [
      aws_subnet.private_1.id,
      aws_subnet.private_2.id
    ]
    endpoint_private_access = true
    endpoint_public_access  = true
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
    aws_iam_role_policy_attachment.eks_service_policy,
  ]
}

# RDS Database
resource "aws_rds_cluster" "pynomaly" {
  cluster_identifier      = "pynomaly-production"
  engine                  = "aurora-postgresql"
  engine_version          = "13.7"
  database_name           = "pynomaly"
  master_username         = var.db_username
  master_password         = var.db_password
  backup_retention_period = 30
  preferred_backup_window = "07:00-09:00"
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.pynomaly.name
  
  final_snapshot_identifier = "pynomaly-final-snapshot"
  
  tags = {
    Name = "pynomaly-production"
    Environment = "production"
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "pynomaly" {
  name       = "pynomaly-cache-subnet"
  subnet_ids = [aws_subnet.private_1.id, aws_subnet.private_2.id]
}

resource "aws_elasticache_replication_group" "pynomaly" {
  replication_group_id       = "pynomaly-redis"
  description                = "Redis cluster for Pynomaly"
  
  num_cache_clusters         = 2
  node_type                  = "cache.r6g.large"
  engine_version             = "6.2"
  parameter_group_name       = "default.redis6.x"
  port                       = 6379
  
  subnet_group_name          = aws_elasticache_subnet_group.pynomaly.name
  security_group_ids         = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = {
    Name = "pynomaly-redis"
    Environment = "production"
  }
}
```

### Terraform Deployment
```bash
# Initialize Terraform
terraform init

# Plan deployment
terraform plan -var-file="production.tfvars"

# Apply infrastructure
terraform apply -var-file="production.tfvars"

# Destroy infrastructure (if needed)
terraform destroy -var-file="production.tfvars"
```

---

## Monitoring & Observability

### Prometheus Configuration
```yaml
# config/deployment/monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert.rules"

scrape_configs:
  - job_name: 'pynomaly-api'
    static_configs:
      - targets: ['pynomaly-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'postgresql'
    static_configs:
      - targets: ['postgres-exporter:9187']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboards
```json
{
  "dashboard": {
    "title": "Pynomaly Production Metrics",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, pynomaly_request_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Anomaly Detection Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(pynomaly_anomalies_detected_total[5m])",
            "legendFormat": "Anomalies/sec"
          }
        ]
      }
    ]
  }
}
```

### AlertManager Rules
```yaml
# config/deployment/monitoring/alert.rules
groups:
  - name: pynomaly.rules
    rules:
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, pynomaly_request_duration_seconds_bucket) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }}s"

      - alert: DatabaseConnectionFailed
        expr: pynomaly_database_connections_failed_total > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection failures"
          description: "{{ $value }} database connections failed"
```

---

## Security & Compliance

### Security Policies
```yaml
# config/deployment/security/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: pynomaly-network-policy
  namespace: pynomaly-production
spec:
  podSelector:
    matchLabels:
      app: pynomaly-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-system
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: TCP
      port: 443
```

### RBAC Configuration
```yaml
# config/deployment/security/rbac.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: pynomaly-production
  name: pynomaly-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: pynomaly-binding
  namespace: pynomaly-production
subjects:
- kind: ServiceAccount
  name: pynomaly-service-account
  namespace: pynomaly-production
roleRef:
  kind: Role
  name: pynomaly-role
  apiGroup: rbac.authorization.k8s.io
```

---

## Troubleshooting

### Common Issues

#### 1. Database Connection Issues
```bash
# Check database connectivity
kubectl exec -it pynomaly-api-xxx -- pg_isready -h db-host -p 5432

# Check database logs
kubectl logs -f deployment/postgresql

# Test database connection
psql -h db-host -p 5432 -U username -d pynomaly
```

#### 2. Redis Connection Issues
```bash
# Test Redis connectivity
kubectl exec -it pynomaly-api-xxx -- redis-cli -h redis-host -p 6379 ping

# Check Redis logs
kubectl logs -f deployment/redis

# Monitor Redis metrics
redis-cli -h redis-host -p 6379 info
```

#### 3. Pod Startup Issues
```bash
# Check pod status
kubectl get pods -n pynomaly-production

# View pod logs
kubectl logs -f pod/pynomaly-api-xxx -n pynomaly-production

# Describe pod for events
kubectl describe pod pynomaly-api-xxx -n pynomaly-production

# Debug pod issues
kubectl exec -it pynomaly-api-xxx -- /bin/bash
```

### Health Check Commands
```bash
# Application health check
curl -f http://api-endpoint/health || echo "Health check failed"

# Database health check
pynomaly health connectivity --timeout 30

# Full system health check
pynomaly health full --detailed --fix
```

---

## Rollback Procedures

### Kubernetes Rollback
```bash
# View rollout history
kubectl rollout history deployment/pynomaly-api

# Rollback to previous version
kubectl rollout undo deployment/pynomaly-api

# Rollback to specific revision
kubectl rollout undo deployment/pynomaly-api --to-revision=2

# Monitor rollback
kubectl rollout status deployment/pynomaly-api
```

### Database Rollback
```bash
# Restore from backup
pg_restore -h db-host -U username -d pynomaly backup-file.sql

# Point-in-time recovery
aws rds restore-db-cluster-to-point-in-time \
  --db-cluster-identifier pynomaly-restored \
  --source-db-cluster-identifier pynomaly-production \
  --restore-to-time 2024-01-15T10:00:00Z
```

### Application Rollback
```bash
# Rollback using blue-green deployment
kubectl patch service pynomaly-api -p '{"spec":{"selector":{"version":"blue"}}}'

# Rollback Docker deployment
docker-compose down
docker-compose -f docker-compose.previous.yml up -d
```

---

## Performance Optimization

### Resource Limits
```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

### Auto-scaling
```yaml
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pynomaly-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pynomaly-api
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

---

## Best Practices

### 1. Security Best Practices
- Use least privilege access
- Implement network policies
- Enable encryption at rest and in transit
- Regular security scanning
- Secrets management with external systems

### 2. Monitoring Best Practices
- Implement comprehensive logging
- Set up alerting for critical metrics
- Monitor business metrics
- Use distributed tracing
- Regular performance reviews

### 3. Deployment Best Practices
- Use immutable deployments
- Implement health checks
- Use blue-green or canary deployments
- Automate testing in CI/CD
- Regular disaster recovery testing

### 4. Operational Best Practices
- Implement proper backup strategies
- Use Infrastructure as Code
- Automate routine tasks
- Document procedures
- Regular capacity planning

---

## Support & Documentation

### Additional Resources
- **API Documentation**: `/docs/api/`
- **Architecture Guide**: `/docs/architecture/`
- **Security Guide**: `/docs/security/`
- **Monitoring Guide**: `/docs/monitoring/`

### Support Channels
- **GitHub Issues**: https://github.com/elgerytme/Pynomaly/issues
- **Documentation**: https://pynomaly.readthedocs.io
- **Community**: Discord/Slack channels

---

*This deployment guide provides comprehensive coverage of Pynomaly's multi-environment deployment capabilities. For specific deployment scenarios or troubleshooting, refer to the individual component documentation.*