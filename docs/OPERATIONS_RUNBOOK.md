# Operations Runbook

## Production Operations Guide

This runbook provides comprehensive operational procedures for managing the enterprise platform in production environments.

## System Architecture

### Production Environment Overview
- **Kubernetes Cluster**: EKS on AWS with auto-scaling
- **Databases**: RDS PostgreSQL (Multi-AZ), ElastiCache Redis
- **Monitoring**: Prometheus, Grafana, CloudWatch
- **Security**: WAF, Security Groups, VPC, KMS encryption
- **Load Balancing**: Application Load Balancer with SSL termination
- **CI/CD**: GitHub Actions with GitOps (ArgoCD)

### Critical Services
1. **Anomaly Detection API** - Core ML inference service
2. **Security Scanner** - Vulnerability and threat detection
3. **Analytics Engine** - BI and insights generation
4. **Auto-Scaling Engine** - AI-powered resource management
5. **Predictive Maintenance** - Proactive system health monitoring

## Monitoring and Alerting

### Key Metrics to Monitor

#### System Health Metrics
```yaml
# CPU and Memory Usage
- cpu_usage_percent > 80% (Warning)
- cpu_usage_percent > 90% (Critical)
- memory_usage_percent > 85% (Warning)
- memory_usage_percent > 95% (Critical)

# Request Metrics  
- request_rate (requests/second)
- error_rate > 5% (Warning)
- error_rate > 10% (Critical)
- response_time_p95 > 2s (Warning)
- response_time_p95 > 5s (Critical)
```

#### Application Metrics
```yaml
# Anomaly Detection
- anomaly_detection_accuracy < 0.85 (Warning)
- model_inference_time > 1s (Warning)
- false_positive_rate > 0.1 (Warning)

# Security
- vulnerability_scan_failures > 0 (Critical)
- security_threats_detected > 0 (Critical)
- authentication_failure_rate > 0.05 (Warning)

# Data Quality
- data_quality_score < 0.9 (Warning)
- data_processing_lag > 5min (Warning)
```

### Alert Escalation
1. **P1 (Critical)**: Security breaches, system down - Immediate response
2. **P2 (High)**: Performance degradation, partial outages - 30min response
3. **P3 (Medium)**: Warnings, non-critical issues - 2hr response
4. **P4 (Low)**: Informational, maintenance needs - Next business day

### Monitoring Dashboards
- **System Overview**: `/grafana/d/system-overview`
- **Application Health**: `/grafana/d/app-health`
- **Security Dashboard**: `/grafana/d/security`
- **ML Model Performance**: `/grafana/d/ml-performance`

## Incident Response

### Emergency Procedures

#### System Down (P1)
```bash
# 1. Check system status
kubectl get pods -n production
kubectl get services -n production

# 2. Check recent deployments
kubectl rollout history deployment/anomaly-detection-api -n production

# 3. Check logs
kubectl logs -f deployment/anomaly-detection-api -n production

# 4. If needed, rollback
kubectl rollout undo deployment/anomaly-detection-api -n production

# 5. Scale up if resource issues
kubectl scale deployment/anomaly-detection-api --replicas=10 -n production
```

#### High Error Rate (P2)
```bash
# 1. Check application logs
kubectl logs -f -l app=anomaly-detection-api -n production | grep ERROR

# 2. Check database connectivity
kubectl exec -it deployment/anomaly-detection-api -n production -- \
  python -c "from src.infrastructure.database import test_connection; test_connection()"

# 3. Check external dependencies
curl -I https://external-api.example.com/health

# 4. Restart problematic pods
kubectl delete pod -l app=anomaly-detection-api -n production
```

#### Security Incident (P1)
```bash
# 1. Immediate isolation
kubectl patch service anomaly-detection-api -n production -p '{"spec":{"type":"ClusterIP"}}'

# 2. Check security logs
kubectl logs -f -l app=security-scanner -n production | grep THREAT

# 3. Run emergency security scan
kubectl create job emergency-scan --from=cronjob/security-scan -n production

# 4. Notify security team
# Send alert to security-alerts@company.com
```

### Runbook Procedures

#### Daily Health Checks
```bash
# System health verification script
#!/bin/bash
echo "=== Daily Health Check $(date) ==="

# 1. Check all pods are running
echo "Checking pod status..."
kubectl get pods -n production | grep -v Running | wc -l

# 2. Check API endpoints
echo "Checking API health..."
curl -f http://anomaly-detection-api/health || echo "API DOWN"

# 3. Check database connections
echo "Checking database..."
kubectl exec deployment/anomaly-detection-api -n production -- \
  python -c "from infrastructure.database import health_check; health_check()"

# 4. Check security status
echo "Checking security..."
kubectl logs --tail=10 -l app=security-scanner -n production | grep -i error

# 5. Check auto-scaling status
echo "Checking auto-scaling..."
kubectl get hpa -n production

echo "=== Health Check Complete ==="
```

#### Weekly Maintenance
```bash
# Weekly maintenance script
#!/bin/bash
echo "=== Weekly Maintenance $(date) ==="

# 1. Update security signatures
kubectl create job update-security-db --from=cronjob/security-update -n production

# 2. ML model retraining check
kubectl logs -l app=ml-training-pipeline | grep "Training completed"

# 3. Database maintenance
kubectl exec -it postgresql-0 -n production -- \
  psql -c "VACUUM ANALYZE; REINDEX DATABASE anomaly_detection;"

# 4. Log cleanup
kubectl delete pods -l job-name=log-cleanup -n production
kubectl create job log-cleanup --image=log-cleaner:latest -n production

# 5. Security compliance scan
kubectl create job compliance-scan --from=cronjob/compliance-audit -n production

echo "=== Maintenance Complete ==="
```

## Deployment Procedures

### Standard Deployment
```bash
# 1. Pre-deployment checks
./scripts/pre-deployment-check.sh

# 2. Deploy using GitOps
git tag v1.2.3
git push origin v1.2.3

# 3. Monitor deployment
kubectl rollout status deployment/anomaly-detection-api -n production

# 4. Post-deployment validation
./scripts/post-deployment-validation.sh

# 5. Update monitoring dashboards
# Grafana dashboards will auto-update with new version
```

### Emergency Rollback
```bash
# 1. Immediate rollback
kubectl rollout undo deployment/anomaly-detection-api -n production

# 2. Verify rollback
kubectl rollout status deployment/anomaly-detection-api -n production

# 3. Check application health
curl -f http://anomaly-detection-api/health

# 4. Document incident
# Create post-mortem in incident-reports/
```

### Blue-Green Deployment
```bash
# 1. Deploy to green environment
kubectl apply -f k8s/green-deployment.yaml

# 2. Wait for green to be ready
kubectl wait --for=condition=available deployment/anomaly-detection-api-green -n production

# 3. Run smoke tests on green
./scripts/smoke-tests.sh green

# 4. Switch traffic to green
kubectl patch service/anomaly-detection-api --patch '{"spec":{"selector":{"version":"green"}}}'

# 5. Monitor for issues
./scripts/monitor-deployment.sh

# 6. Clean up blue environment (after 24h)
kubectl delete deployment/anomaly-detection-api-blue -n production
```

## Performance Tuning

### Database Optimization
```sql
-- Regular maintenance queries
VACUUM ANALYZE anomaly_detections;
REINDEX INDEX idx_anomaly_timestamp;

-- Performance monitoring
SELECT * FROM pg_stat_activity WHERE state = 'active';
SELECT * FROM pg_stat_user_tables WHERE relname = 'anomaly_detections';
```

### Kubernetes Resource Optimization
```yaml
# Resource limits optimization
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi" 
    cpu: "1000m"

# HPA configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: anomaly-detection-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### ML Model Performance
```python
# Model performance monitoring
def monitor_model_performance():
    """Monitor ML model accuracy and latency"""
    # Check inference time
    start_time = time.time()
    result = model.predict(sample_data)
    inference_time = time.time() - start_time
    
    if inference_time > 1.0:
        alert("Model inference time too high: {:.2f}s".format(inference_time))
    
    # Check model accuracy
    accuracy = calculate_accuracy(result, ground_truth)
    if accuracy < 0.85:
        alert("Model accuracy below threshold: {:.2f}".format(accuracy))
```

## Security Operations

### Security Monitoring
```bash
# Check for security threats
kubectl logs -l app=threat-detector -n production | grep THREAT_DETECTED

# Vulnerability scan results
kubectl logs -l app=security-scanner -n production | grep VULNERABILITY

# Authentication failures
kubectl logs -l app=auth-service -n production | grep "AUTH_FAILED"

# Compliance status
kubectl create job compliance-check --from=cronjob/compliance-audit -n production
```

### Security Incident Response
1. **Isolate affected systems**
2. **Collect forensic evidence**  
3. **Patch vulnerabilities**
4. **Update security rules**
5. **Monitor for reoccurrence**

## Backup and Recovery

### Database Backup
```bash
# Daily automated backup
kubectl create job db-backup-$(date +%Y%m%d) --from=cronjob/postgres-backup -n production

# Restore from backup
kubectl create job db-restore --from=configmap/restore-config -n production
```

### Configuration Backup
```bash
# Backup Kubernetes configurations
kubectl get all -n production -o yaml > backup-$(date +%Y%m%d).yaml

# Backup secrets (encrypted)
kubectl get secrets -n production -o yaml | gpg --encrypt > secrets-backup.gpg
```

## Contact Information

### On-Call Rotation
- **Primary**: Platform Team (+1-555-0123)
- **Secondary**: DevOps Team (+1-555-0124)  
- **Security**: Security Team (+1-555-0125)

### Escalation Matrix
1. **Platform Engineer** (Response: 15min)
2. **Senior Platform Engineer** (Response: 30min)
3. **Platform Team Lead** (Response: 1hr)
4. **Engineering Manager** (Response: 2hr)

### Emergency Contacts
- **System Down**: Immediately page on-call engineer
- **Security Incident**: Contact security team + engineering manager
- **Data Loss**: Contact DBA + backup team + engineering manager

Remember: When in doubt, escalate early. It's better to have a false alarm than miss a critical issue.