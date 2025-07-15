# Pynomaly Production Operations Runbook

## Overview

This runbook provides comprehensive operational procedures for managing Pynomaly in production environments. It covers deployment, monitoring, troubleshooting, and maintenance tasks.

## Table of Contents

1. [Deployment Procedures](#deployment-procedures)
2. [Monitoring and Alerting](#monitoring-and-alerting)
3. [Troubleshooting Guide](#troubleshooting-guide)
4. [Backup and Recovery](#backup-and-recovery)
5. [Performance Optimization](#performance-optimization)
6. [Security Operations](#security-operations)
7. [Maintenance Tasks](#maintenance-tasks)

## Deployment Procedures

### 1. Production Deployment

#### Prerequisites
- Kubernetes cluster with RBAC enabled
- Helm 3.x installed
- kubectl configured for the target cluster
- Docker images pushed to registry

#### Standard Deployment

```bash
# 1. Add the Pynomaly Helm repository (if external)
helm repo add pynomaly https://charts.pynomaly.ai
helm repo update

# 2. Create namespace
kubectl create namespace pynomaly-production

# 3. Install with production values
helm install pynomaly pynomaly/pynomaly \
  --namespace pynomaly-production \
  --values deployment/helm/pynomaly/values-production.yaml \
  --set image.tag="v0.1.3" \
  --wait

# 4. Verify deployment
kubectl get pods -n pynomaly-production
kubectl get services -n pynomaly-production
```

#### Rolling Updates

```bash
# 1. Update image tag
helm upgrade pynomaly pynomaly/pynomaly \
  --namespace pynomaly-production \
  --reuse-values \
  --set image.tag="v0.1.4" \
  --wait

# 2. Monitor rollout status
kubectl rollout status deployment/pynomaly -n pynomaly-production

# 3. Verify new version
kubectl get pods -n pynomaly-production -o jsonpath='{.items[*].spec.containers[*].image}'
```

#### Rollback Procedures

```bash
# 1. List available revisions
helm history pynomaly -n pynomaly-production

# 2. Rollback to previous version
helm rollback pynomaly 1 -n pynomaly-production

# 3. Emergency rollback (immediate)
kubectl rollout undo deployment/pynomaly -n pynomaly-production
```

### 2. Blue-Green Deployment

```bash
# 1. Deploy to green environment
helm install pynomaly-green pynomaly/pynomaly \
  --namespace pynomaly-green \
  --values deployment/helm/pynomaly/values-production.yaml \
  --set image.tag="v0.1.4"

# 2. Run smoke tests
./scripts/smoke-tests.sh pynomaly-green

# 3. Switch traffic (update ingress)
kubectl patch ingress pynomaly-ingress -n pynomaly-production \
  --type='json' \
  -p='[{"op": "replace", "path": "/spec/rules/0/http/paths/0/backend/service/name", "value": "pynomaly-green"}]'

# 4. Monitor and validate
./scripts/validate-deployment.sh

# 5. Clean up blue environment (after validation)
helm uninstall pynomaly -n pynomaly-production
```

## Monitoring and Alerting

### 1. Health Checks

#### Application Health
```bash
# Health endpoint
curl -f https://api.pynomaly.ai/health

# Readiness endpoint
curl -f https://api.pynomaly.ai/ready

# Metrics endpoint
curl https://api.pynomaly.ai/metrics
```

#### Kubernetes Health
```bash
# Pod status
kubectl get pods -n pynomaly-production

# Service status
kubectl get services -n pynomaly-production

# Ingress status
kubectl get ingress -n pynomaly-production

# Events
kubectl get events -n pynomaly-production --sort-by='.lastTimestamp'
```

### 2. Key Metrics to Monitor

#### Application Metrics
- Request rate (requests/second)
- Response time (95th percentile < 500ms)
- Error rate (< 1%)
- Memory usage (< 80% of limit)
- CPU usage (< 70% of limit)

#### Infrastructure Metrics
- Pod availability (> 99%)
- Database connection pool usage
- Redis connection pool usage
- Disk usage (< 80%)
- Network latency

#### Business Metrics
- Anomaly detection requests/hour
- Model training jobs/day
- Active users
- API key usage

### 3. Alert Conditions

#### Critical Alerts (Immediate Response)
- All pods down
- Error rate > 5%
- Response time > 2 seconds
- Database unavailable
- Redis unavailable

#### Warning Alerts (Response within 1 hour)
- Pod count below minimum
- Memory usage > 80%
- CPU usage > 80%
- Disk usage > 80%
- Response time > 1 second

## Troubleshooting Guide

### 1. Common Issues

#### Pods Failing to Start

**Symptoms:**
- Pods stuck in `Pending` or `CrashLoopBackOff` state
- Deployment not progressing

**Diagnosis:**
```bash
# Check pod status and events
kubectl describe pod <pod-name> -n pynomaly-production

# Check logs
kubectl logs <pod-name> -n pynomaly-production

# Check resource constraints
kubectl top pods -n pynomaly-production
kubectl describe nodes
```

**Common Causes & Solutions:**
1. **Insufficient resources:**
   ```bash
   # Scale down temporarily or add nodes
   kubectl scale deployment pynomaly --replicas=2 -n pynomaly-production
   ```

2. **Configuration errors:**
   ```bash
   # Check ConfigMap and Secrets
   kubectl get configmap pynomaly-config -n pynomaly-production -o yaml
   kubectl get secret pynomaly-secret -n pynomaly-production -o yaml
   ```

3. **Image pull errors:**
   ```bash
   # Check image pull secrets
   kubectl get secrets -n pynomaly-production
   ```

#### High Memory Usage

**Symptoms:**
- Pods being OOMKilled
- Slow response times
- Memory usage alerts

**Diagnosis:**
```bash
# Check memory usage
kubectl top pods -n pynomaly-production

# Check memory limits
kubectl describe pod <pod-name> -n pynomaly-production | grep -A 5 -B 5 memory

# Check application logs for memory leaks
kubectl logs <pod-name> -n pynomaly-production | grep -i memory
```

**Solutions:**
1. **Increase memory limits:**
   ```bash
   helm upgrade pynomaly pynomaly/pynomaly \
     --reuse-values \
     --set resources.limits.memory=6Gi
   ```

2. **Scale horizontally:**
   ```bash
   kubectl scale deployment pynomaly --replicas=5 -n pynomaly-production
   ```

3. **Tune garbage collection:**
   ```bash
   # Add environment variables for Python memory management
   helm upgrade pynomaly pynomaly/pynomaly \
     --reuse-values \
     --set env.PYTHONMALLOC=malloc \
     --set env.MALLOC_TRIM_THRESHOLD_=100000
   ```

#### Database Connection Issues

**Symptoms:**
- Connection timeout errors
- "Too many connections" errors
- Slow database queries

**Diagnosis:**
```bash
# Check database connectivity
kubectl exec -it deployment/pynomaly -n pynomaly-production -- \
  psql -h pynomaly-postgresql -U pynomaly -d pynomaly -c "SELECT 1;"

# Check connection pool status
kubectl logs deployment/pynomaly -n pynomaly-production | grep -i "connection pool"

# Check database metrics
kubectl port-forward svc/pynomaly-postgresql 5432:5432 -n pynomaly-production &
psql -h localhost -U pynomaly -d pynomaly -c "SELECT count(*) FROM pg_stat_activity;"
```

**Solutions:**
1. **Increase connection pool size:**
   ```bash
   helm upgrade pynomaly pynomaly/pynomaly \
     --reuse-values \
     --set config.database.pool_size=30 \
     --set config.database.max_overflow=50
   ```

2. **Scale database:**
   ```bash
   helm upgrade pynomaly pynomaly/pynomaly \
     --reuse-values \
     --set postgresql.primary.resources.limits.cpu=4000m \
     --set postgresql.primary.resources.limits.memory=8Gi
   ```

### 2. Performance Issues

#### Slow Response Times

**Investigation Steps:**
1. Check application metrics in Grafana
2. Identify slow endpoints in logs
3. Check database query performance
4. Verify external service dependencies

**Quick Fixes:**
```bash
# Restart deployment (rolling restart)
kubectl rollout restart deployment/pynomaly -n pynomaly-production

# Scale up temporarily
kubectl scale deployment pynomaly --replicas=6 -n pynomaly-production

# Clear Redis cache
kubectl exec -it deployment/pynomaly-redis -n pynomaly-production -- redis-cli FLUSHALL
```

## Backup and Recovery

### 1. Automated Backups

#### Database Backups
```bash
# Manual backup
kubectl exec -it pynomaly-postgresql-0 -n pynomaly-production -- \
  pg_dump -U pynomaly pynomaly > backup-$(date +%Y%m%d-%H%M%S).sql

# Verify backup schedule
kubectl get cronjobs -n pynomaly-production
```

#### Configuration Backups
```bash
# Backup Helm values
helm get values pynomaly -n pynomaly-production > pynomaly-values-backup.yaml

# Backup Kubernetes resources
kubectl get all -n pynomaly-production -o yaml > pynomaly-k8s-backup.yaml
```

### 2. Disaster Recovery

#### Complete Environment Recovery
```bash
# 1. Restore database from backup
kubectl exec -i pynomaly-postgresql-0 -n pynomaly-production -- \
  psql -U pynomaly pynomaly < backup-20240714-120000.sql

# 2. Redeploy application
helm install pynomaly pynomaly/pynomaly \
  --namespace pynomaly-production \
  --values pynomaly-values-backup.yaml

# 3. Verify recovery
./scripts/validate-deployment.sh
```

#### Database Point-in-Time Recovery
```bash
# 1. Stop application
kubectl scale deployment pynomaly --replicas=0 -n pynomaly-production

# 2. Restore database to specific timestamp
kubectl exec -it pynomaly-postgresql-0 -n pynomaly-production -- \
  pg_basebackup -U postgres -D /var/lib/postgresql/data/recovery -P -W

# 3. Restart database and application
kubectl delete pod pynomaly-postgresql-0 -n pynomaly-production
kubectl scale deployment pynomaly --replicas=3 -n pynomaly-production
```

## Performance Optimization

### 1. Auto-scaling Configuration

```bash
# Enable horizontal pod autoscaling
kubectl autoscale deployment pynomaly \
  --cpu-percent=70 \
  --memory-percent=80 \
  --min=3 \
  --max=20 \
  -n pynomaly-production

# Monitor autoscaling
kubectl get hpa -n pynomaly-production
```

### 2. Resource Optimization

#### CPU Optimization
```bash
# Update CPU requests and limits
helm upgrade pynomaly pynomaly/pynomaly \
  --reuse-values \
  --set resources.requests.cpu=750m \
  --set resources.limits.cpu=2000m
```

#### Memory Optimization
```bash
# Update memory requests and limits
helm upgrade pynomaly pynomaly/pynomaly \
  --reuse-values \
  --set resources.requests.memory=2Gi \
  --set resources.limits.memory=6Gi
```

### 3. Database Optimization

```bash
# Tune PostgreSQL for production
helm upgrade pynomaly pynomaly/pynomaly \
  --reuse-values \
  --set postgresql.primary.configuration.max_connections=400 \
  --set postgresql.primary.configuration.shared_buffers=512MB \
  --set postgresql.primary.configuration.effective_cache_size=2GB
```

## Security Operations

### 1. Certificate Management

```bash
# Check certificate expiration
kubectl get certificates -n pynomaly-production

# Renew certificates manually (if needed)
kubectl delete certificate pynomaly-tls -n pynomaly-production
kubectl apply -f deployment/infrastructure/kubernetes/production/certificate.yaml
```

### 2. Security Scanning

```bash
# Run security scans
trivy image ghcr.io/elgerytme/pynomaly:latest

# Check for vulnerabilities in running containers
kubectl exec -it deployment/pynomaly -n pynomaly-production -- \
  apt list --upgradable
```

### 3. Access Control

```bash
# Review RBAC permissions
kubectl auth can-i --list --as=system:serviceaccount:pynomaly-production:pynomaly

# Rotate secrets
kubectl create secret generic pynomaly-secret-new \
  --from-literal=database-password=$(openssl rand -base64 32) \
  --from-literal=redis-password=$(openssl rand -base64 32) \
  -n pynomaly-production
```

## Maintenance Tasks

### 1. Regular Maintenance (Weekly)

```bash
#!/bin/bash
# Weekly maintenance script

# 1. Check resource usage
kubectl top pods -n pynomaly-production
kubectl top nodes

# 2. Review logs for errors
kubectl logs -l app=pynomaly -n pynomaly-production --since=7d | grep -i error

# 3. Check certificate expiration
kubectl get certificates -n pynomaly-production

# 4. Verify backup status
kubectl get cronjobs -n pynomaly-production

# 5. Review metrics and alerts
echo "Review Grafana dashboards and Prometheus alerts"
```

### 2. Regular Maintenance (Monthly)

```bash
#!/bin/bash
# Monthly maintenance script

# 1. Update dependencies
helm repo update

# 2. Plan upgrade strategy
helm diff upgrade pynomaly pynomaly/pynomaly \
  --namespace pynomaly-production \
  --reuse-values

# 3. Review and rotate secrets
kubectl get secrets -n pynomaly-production -o jsonpath='{.items[*].metadata.creationTimestamp}'

# 4. Database maintenance
kubectl exec -it pynomaly-postgresql-0 -n pynomaly-production -- \
  psql -U pynomaly -d pynomaly -c "VACUUM ANALYZE;"

# 5. Review access logs and usage patterns
```

### 3. Emergency Procedures

#### Complete Service Outage
1. **Immediate Response:**
   ```bash
   # Check cluster status
   kubectl get nodes
   kubectl get pods --all-namespaces
   
   # Check critical services
   kubectl get pods -n kube-system
   kubectl get pods -n ingress-nginx
   ```

2. **Service Recovery:**
   ```bash
   # Scale up from zero if needed
   kubectl scale deployment pynomaly --replicas=3 -n pynomaly-production
   
   # Force pod recreation
   kubectl delete pods -l app=pynomaly -n pynomaly-production
   ```

3. **Communication:**
   - Update status page
   - Notify stakeholders
   - Document incident for post-mortem

#### Data Corruption
1. **Immediate Actions:**
   ```bash
   # Stop writes to database
   kubectl scale deployment pynomaly --replicas=0 -n pynomaly-production
   
   # Create emergency backup
   kubectl exec -it pynomaly-postgresql-0 -n pynomaly-production -- \
     pg_dump -U pynomaly pynomaly > emergency-backup-$(date +%Y%m%d-%H%M%S).sql
   ```

2. **Recovery:**
   ```bash
   # Restore from latest clean backup
   kubectl exec -i pynomaly-postgresql-0 -n pynomaly-production -- \
     psql -U pynomaly pynomaly < latest-clean-backup.sql
   
   # Restart services
   kubectl scale deployment pynomaly --replicas=3 -n pynomaly-production
   ```

## Contacts and Escalation

### On-Call Rotation
- **Primary:** DevOps Engineer (24/7)
- **Secondary:** Platform Engineer (Business Hours)
- **Escalation:** Engineering Manager

### Critical Escalation Procedures
1. **Severity 1 (Complete Outage):** Immediate escalation to Engineering Manager
2. **Severity 2 (Degraded Performance):** Escalate within 1 hour
3. **Severity 3 (Minor Issues):** Handle within SLA

### Communication Channels
- **Alerts:** PagerDuty
- **Updates:** Slack #pynomaly-ops
- **Status:** status.pynomaly.ai
- **Documentation:** docs.pynomaly.ai/operations