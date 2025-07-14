# Pynomaly Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying Pynomaly in production environments. The deployment strategy includes security hardening, monitoring, and high availability configurations.

## 🔧 Prerequisites

### System Requirements

- **Kubernetes**: v1.24+ (recommended: v1.28+)
- **Docker**: v20.10+ with BuildKit support
- **Docker Compose**: v2.0+ (for standalone deployment)
- **Helm**: v3.8+ (for Kubernetes deployment)
- **kubectl**: v1.24+

### Infrastructure Requirements

#### Minimum Resources

- **CPU**: 8 cores
- **Memory**: 16 GB RAM
- **Storage**: 100 GB SSD
- **Network**: 1 Gbps

#### Recommended Production Resources

- **CPU**: 16 cores
- **Memory**: 32 GB RAM
- **Storage**: 500 GB SSD (with backup storage)
- **Network**: 10 Gbps

### Cloud Provider Support

- ✅ AWS EKS
- ✅ Google GKE
- ✅ Azure AKS
- ✅ DigitalOcean Kubernetes
- ✅ Self-managed Kubernetes

## 🚀 Deployment Methods

### Method 1: Kubernetes Deployment (Recommended)

#### Step 1: Prepare the Cluster

```bash
# Create namespace
kubectl create namespace pynomaly-production

# Label namespace for monitoring
kubectl label namespace pynomaly-production \
  environment=production \
  app=pynomaly \
  monitoring=enabled
```

#### Step 2: Setup Secrets Management

```bash
# Run the secrets setup script
./scripts/secrets-setup.sh

# Verify secrets are created
kubectl get secrets -n pynomaly-production
```

#### Step 3: Deploy Infrastructure Components

```bash
# Deploy PostgreSQL
kubectl apply -f k8s/production/databases.yaml

# Deploy Redis
kubectl apply -f k8s/redis.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod \
  --selector=app=postgres \
  --namespace=pynomaly-production \
  --timeout=300s
```

#### Step 4: Deploy Application

```bash
# Deploy main application
kubectl apply -f k8s/production/enhanced-deployment.yaml

# Deploy monitoring stack
kubectl apply -f k8s/monitoring.yaml

# Deploy ingress
kubectl apply -f k8s/production/ingress.yaml
```

#### Step 5: Verify Deployment

```bash
# Check all pods are running
kubectl get pods -n pynomaly-production

# Check services
kubectl get services -n pynomaly-production

# Check ingress
kubectl get ingress -n pynomaly-production

# Test health endpoint
kubectl port-forward service/pynomaly-api-service 8000:8000 -n pynomaly-production &
curl http://localhost:8000/api/health
```

### Method 2: Docker Compose Deployment

#### Step 1: Prepare Environment

```bash
# Clone repository
git clone https://github.com/pynomaly/pynomaly.git
cd pynomaly

# Copy environment configuration
cp .env.production.example .env.production

# Edit production environment variables
nano .env.production
```

#### Step 2: Configure Secrets

```bash
# Generate secure passwords
openssl rand -base64 32  # For database password
openssl rand -base64 64  # For application secrets

# Update .env.production with generated values
```

#### Step 3: Deploy Stack

```bash
# Build and deploy
docker-compose -f deploy/docker/docker-compose.production.yml up -d

# Check status
docker-compose -f deploy/docker/docker-compose.production.yml ps

# View logs
docker-compose -f deploy/docker/docker-compose.production.yml logs -f pynomaly-api
```

## 🔒 Security Configuration

### SSL/TLS Certificate Setup

#### Using cert-manager (Kubernetes)

```bash
# Install cert-manager
helm repo add jetstack https://charts.jetstack.io
helm repo update
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --version v1.13.0 \
  --set installCRDs=true

# Apply cluster issuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@your-domain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

#### Using Custom Certificates

```bash
# Create TLS secret
kubectl create secret tls pynomaly-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key \
  --namespace=pynomaly-production
```

### Network Security

#### Kubernetes Network Policies

```bash
# Apply network policies
kubectl apply -f k8s/production/network-policies.yaml

# Verify policies
kubectl get networkpolicies -n pynomaly-production
```

#### Firewall Rules (Cloud Provider)

**AWS Security Groups:**

```bash
# Allow HTTPS traffic
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxx \
  --protocol tcp \
  --port 443 \
  --cidr 0.0.0.0/0

# Allow internal cluster communication
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxx \
  --protocol tcp \
  --port 0-65535 \
  --source-group sg-xxxxx
```

## 📊 Monitoring Setup

### Prometheus Configuration

```bash
# Deploy Prometheus stack
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values - <<EOF
prometheus:
  prometheusSpec:
    retention: 30d
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: fast-ssd
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 100Gi
grafana:
  adminPassword: ${GRAFANA_ADMIN_PASSWORD}
  persistence:
    enabled: true
    storageClassName: fast-ssd
    size: 10Gi
EOF
```

### Application Metrics

```bash
# Port-forward to access Grafana
kubectl port-forward service/prometheus-grafana 3000:80 -n monitoring

# Import Pynomaly dashboards
# Open http://localhost:3000 and import dashboards from deploy/monitoring/grafana/
```

## 🔄 Backup and Recovery

### Database Backup

#### Automated Backup (Kubernetes CronJob)

```bash
# Deploy backup CronJob
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
  namespace: pynomaly-production
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: postgres-backup
            image: postgres:16-alpine
            env:
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: pynomaly-production-secrets
                  key: POSTGRES_PASSWORD
            command:
            - /bin/sh
            - -c
            - |
              pg_dump -h postgres-service -U pynomaly pynomaly | \
              gzip > /backup/pynomaly-\$(date +%Y%m%d_%H%M%S).sql.gz
              
              # Upload to S3 (if configured)
              aws s3 cp /backup/pynomaly-\$(date +%Y%m%d_%H%M%S).sql.gz \
                s3://your-backup-bucket/database/
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
EOF
```

#### Manual Backup

```bash
# Create database backup
kubectl exec -it postgres-pod -n pynomaly-production -- \
  pg_dump -U pynomaly pynomaly | gzip > pynomaly-backup-$(date +%Y%m%d).sql.gz

# Backup persistent volumes
kubectl get pv
# Use cloud provider tools to snapshot volumes
```

### Application Data Backup

```bash
# Backup application storage
kubectl cp pynomaly-production/pynomaly-api-pod:/app/storage ./storage-backup

# Backup configuration
kubectl get configmaps -n pynomaly-production -o yaml > configmaps-backup.yaml
kubectl get secrets -n pynomaly-production -o yaml > secrets-backup.yaml
```

## 🔧 Maintenance Operations

### Scaling

#### Horizontal Pod Autoscaling

```bash
# Check HPA status
kubectl get hpa -n pynomaly-production

# Manual scaling
kubectl scale deployment pynomaly-api-production --replicas=10 -n pynomaly-production
```

#### Vertical Scaling

```bash
# Update resource limits
kubectl patch deployment pynomaly-api-production -n pynomaly-production -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "pynomaly-api",
            "resources": {
              "limits": {
                "memory": "4Gi",
                "cpu": "2000m"
              },
              "requests": {
                "memory": "2Gi",
                "cpu": "1000m"
              }
            }
          }
        ]
      }
    }
  }
}'
```

### Rolling Updates

```bash
# Update image version
kubectl set image deployment/pynomaly-api-production \
  pynomaly-api=ghcr.io/pynomaly/pynomaly:v1.2.0 \
  -n pynomaly-production

# Check rollout status
kubectl rollout status deployment/pynomaly-api-production -n pynomaly-production

# Rollback if needed
kubectl rollout undo deployment/pynomaly-api-production -n pynomaly-production
```

### Secret Rotation

```bash
# Trigger manual secret rotation
kubectl create job --from=cronjob/pynomaly-secret-rotation \
  manual-rotation-$(date +%Y%m%d) -n pynomaly-production

# Check rotation status
kubectl logs job/manual-rotation-$(date +%Y%m%d) -n pynomaly-production
```

## 🚨 Troubleshooting

### Common Issues

#### Pod Startup Issues

```bash
# Check pod status
kubectl describe pod pynomaly-api-pod -n pynomaly-production

# Check logs
kubectl logs pynomaly-api-pod -n pynomaly-production --previous

# Check events
kubectl get events -n pynomaly-production --sort-by='.lastTimestamp'
```

#### Database Connection Issues

```bash
# Test database connectivity
kubectl exec -it pynomaly-api-pod -n pynomaly-production -- \
  python -c "
import asyncpg
import asyncio
async def test():
    conn = await asyncpg.connect('postgresql://user:pass@host:5432/db')
    result = await conn.fetchval('SELECT 1')
    print(f'DB test result: {result}')
    await conn.close()
asyncio.run(test())
"
```

#### Performance Issues

```bash
# Check resource usage
kubectl top pods -n pynomaly-production
kubectl top nodes

# Check HPA status
kubectl describe hpa pynomaly-api-hpa -n pynomaly-production

# Review metrics
kubectl port-forward service/prometheus-server 9090:80 -n monitoring
# Open http://localhost:9090
```

### Log Analysis

```bash
# Centralized logging with FluentBit
kubectl logs -l app=fluent-bit -n logging

# Application logs
kubectl logs -l app=pynomaly -n pynomaly-production --tail=100

# Error log analysis
kubectl logs -l app=pynomaly -n pynomaly-production | grep ERROR
```

## 🔐 Security Hardening Checklist

### Container Security

- ✅ Non-root user (UID 65532)
- ✅ Read-only root filesystem
- ✅ No privileged containers
- ✅ Security context constraints
- ✅ Resource limits enforced

### Network Security

- ✅ Network policies implemented
- ✅ TLS encryption for all traffic
- ✅ Service mesh (optional: Istio/Linkerd)
- ✅ Ingress with WAF
- ✅ Internal service communication encrypted

### Secret Management

- ✅ External secrets operator
- ✅ Secret rotation automation
- ✅ Encryption at rest
- ✅ Minimal secret exposure
- ✅ Audit logging

### Image Security

- ✅ Distroless base images
- ✅ Vulnerability scanning
- ✅ Image signing
- ✅ Registry access controls
- ✅ Regular image updates

## 📈 Performance Optimization

### Database Optimization

```sql
-- Add indexes for common queries
CREATE INDEX CONCURRENTLY idx_anomaly_detection_timestamp 
ON anomaly_detections(created_at);

CREATE INDEX CONCURRENTLY idx_model_training_status 
ON model_training(status, created_at);

-- Configure connection pooling
-- Set in postgresql.conf:
-- max_connections = 200
-- shared_buffers = 256MB
-- effective_cache_size = 1GB
```

### Application Optimization

```python
# Configure async connection pools
DATABASE_CONFIG = {
    "pool_size": 20,
    "max_overflow": 30,
    "pool_timeout": 30,
    "pool_recycle": 3600
}

REDIS_CONFIG = {
    "max_connections": 50,
    "retry_on_timeout": True,
    "socket_keepalive": True,
    "socket_keepalive_options": {}
}
```

### Caching Strategy

```bash
# Redis cache configuration
kubectl patch configmap pynomaly-config -n pynomaly-production --patch '{
  "data": {
    "cache_ttl": "3600",
    "cache_max_size": "1000",
    "enable_query_cache": "true"
  }
}'
```

## 🚀 Production Readiness Checklist

### Pre-Deployment

- [ ] All secrets configured
- [ ] SSL certificates installed
- [ ] Database migrations tested
- [ ] Load testing completed
- [ ] Security scan passed
- [ ] Backup strategy tested
- [ ] Monitoring configured
- [ ] Alerting rules set up
- [ ] Documentation reviewed
- [ ] Team training completed

### Post-Deployment

- [ ] Health checks passing
- [ ] Metrics collecting
- [ ] Logs flowing
- [ ] Backups running
- [ ] Alerts configured
- [ ] Performance baseline established
- [ ] Security monitoring active
- [ ] User acceptance testing
- [ ] Support procedures documented
- [ ] Incident response plan ready

## 📞 Support and Maintenance

### Regular Maintenance Tasks

**Daily:**

- Check application health
- Review error logs
- Monitor resource usage
- Verify backup completion

**Weekly:**

- Review security alerts
- Check for updates
- Analyze performance metrics
- Test disaster recovery

**Monthly:**

- Security vulnerability scan
- Capacity planning review
- Update dependencies
- Review and rotate secrets

### Emergency Contacts

- **Platform Team**: <platform@your-company.com>
- **Security Team**: <security@your-company.com>
- **On-Call Engineer**: +1-555-0123

### Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Security Best Practices](https://docs.docker.com/develop/security-best-practices/)
- [Prometheus Monitoring](https://prometheus.io/docs/)
- [PostgreSQL Performance Tuning](https://wiki.postgresql.org/wiki/Performance_Optimization)

---

**Last Updated**: July 11, 2025  
**Version**: 1.0.0  
**Maintainer**: Pynomaly DevOps Team
