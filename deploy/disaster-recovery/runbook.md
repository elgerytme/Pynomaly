# Pynomaly Disaster Recovery Runbook

## Overview

This runbook provides step-by-step procedures for disaster recovery scenarios in the Pynomaly production environment.

## Emergency Contacts

- **Primary On-Call**: [team@pynomaly.ai](mailto:team@pynomaly.ai)
- **Secondary On-Call**: [infrastructure@pynomaly.ai](mailto:infrastructure@pynomaly.ai)
- **Management Escalation**: [leadership@pynomaly.ai](mailto:leadership@pynomaly.ai)
- **Slack Channel**: #incident-response

## Disaster Scenarios

### 1. Complete Service Outage

#### Symptoms
- API returning 5xx errors or not responding
- Health checks failing
- No traffic reaching the application

#### Immediate Response (0-15 minutes)
1. **Assess the situation**
   ```bash
   # Check pod status
   kubectl get pods -n pynomaly
   
   # Check service status
   kubectl get svc -n pynomaly
   
   # Check ingress
   kubectl get ingress -n pynomaly
   
   # Check events
   kubectl get events -n pynomaly --sort-by='.lastTimestamp'
   ```

2. **Check infrastructure**
   ```bash
   # Check node status
   kubectl get nodes
   
   # Check system pods
   kubectl get pods -n kube-system
   
   # Check AWS EKS cluster
   aws eks describe-cluster --name pynomaly-production
   ```

3. **Quick recovery attempts**
   ```bash
   # Restart deployment
   kubectl rollout restart deployment/pynomaly-api -n pynomaly
   kubectl rollout restart deployment/pynomaly-worker -n pynomaly
   
   # Wait for rollout
   kubectl rollout status deployment/pynomaly-api -n pynomaly --timeout=300s
   ```

#### Escalation (15-30 minutes)
If quick recovery fails:

1. **Rollback to last known good version**
   ```bash
   # Check rollout history
   helm history pynomaly -n pynomaly
   
   # Rollback to previous version
   helm rollback pynomaly -n pynomaly
   
   # Wait for rollback
   kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=pynomaly -n pynomaly --timeout=600s
   ```

2. **Scale up resources**
   ```bash
   # Scale API pods
   kubectl scale deployment pynomaly-api --replicas=10 -n pynomaly
   
   # Scale worker pods
   kubectl scale deployment pynomaly-worker --replicas=5 -n pynomaly
   ```

### 2. Database Failure

#### Symptoms
- Connection errors to PostgreSQL
- Data inconsistency issues
- Database performance degradation

#### Immediate Response
1. **Check database status**
   ```bash
   # Check PostgreSQL pods
   kubectl get pods -n pynomaly -l app.kubernetes.io/name=postgresql
   
   # Check PostgreSQL logs
   kubectl logs -n pynomaly -l app.kubernetes.io/name=postgresql --tail=100
   
   # Check persistent volumes
   kubectl get pv,pvc -n pynomaly
   ```

2. **Database connectivity test**
   ```bash
   # Test connection from API pod
   kubectl exec -it deployment/pynomaly-api -n pynomaly -- \
     psql -h pynomaly-postgresql -U pynomaly -d pynomaly -c "SELECT 1;"
   ```

#### Recovery Steps
1. **Restart database**
   ```bash
   kubectl rollout restart statefulset/pynomaly-postgresql -n pynomaly
   kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=postgresql -n pynomaly --timeout=600s
   ```

2. **Restore from backup** (if restart fails)
   ```bash
   # List available backups
   aws s3 ls s3://pynomaly-backups/postgresql/
   
   # Restore from latest backup
   kubectl exec -it pynomaly-postgresql-0 -n pynomaly -- \
     psql -U pynomaly -d pynomaly -f /backups/latest-backup.sql
   ```

### 3. Redis Cache Failure

#### Symptoms
- Increased API response times
- Cache miss errors in logs
- Redis connection failures

#### Recovery Steps
1. **Check Redis status**
   ```bash
   kubectl get pods -n pynomaly -l app.kubernetes.io/name=redis
   kubectl logs -n pynomaly -l app.kubernetes.io/name=redis --tail=50
   ```

2. **Restart Redis**
   ```bash
   kubectl rollout restart statefulset/pynomaly-redis -n pynomaly
   kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=redis -n pynomaly --timeout=300s
   ```

3. **Verify cache functionality**
   ```bash
   kubectl exec -it pynomaly-redis-master-0 -n pynomaly -- redis-cli ping
   ```

### 4. Storage Issues

#### Symptoms
- Pod evictions due to disk pressure
- Persistent volume claim issues
- File system errors

#### Recovery Steps
1. **Check storage usage**
   ```bash
   # Check PVC status
   kubectl get pvc -n pynomaly
   
   # Check node storage
   kubectl describe nodes
   
   # Check pod disk usage
   kubectl exec -it deployment/pynomaly-api -n pynomaly -- df -h
   ```

2. **Clean up old data**
   ```bash
   # Clean temporary files
   kubectl exec -it deployment/pynomaly-api -n pynomaly -- \
     find /app/temp -type f -mtime +7 -delete
   
   # Clean old logs
   kubectl exec -it deployment/pynomaly-api -n pynomaly -- \
     find /app/logs -name "*.log" -mtime +30 -delete
   ```

3. **Expand storage**
   ```bash
   # Expand PVC (if supported by storage class)
   kubectl patch pvc pynomaly-storage -n pynomaly \
     -p '{"spec":{"resources":{"requests":{"storage":"100Gi"}}}}'
   ```

### 5. Network Connectivity Issues

#### Symptoms
- Intermittent connection failures
- Load balancer issues
- DNS resolution problems

#### Recovery Steps
1. **Check network components**
   ```bash
   # Check ingress controller
   kubectl get pods -n ingress-nginx
   
   # Check CoreDNS
   kubectl get pods -n kube-system -l k8s-app=kube-dns
   
   # Check load balancer
   kubectl describe svc pynomaly -n pynomaly
   ```

2. **Test connectivity**
   ```bash
   # Test internal connectivity
   kubectl exec -it deployment/pynomaly-api -n pynomaly -- \
     nslookup pynomaly-postgresql
   
   # Test external connectivity
   kubectl exec -it deployment/pynomaly-api -n pynomaly -- \
     curl -I https://api.pynomaly.ai/health
   ```

## Complete Environment Recovery

### Scenario: Entire Kubernetes cluster is lost

1. **Prepare new cluster**
   ```bash
   # Create new EKS cluster
   eksctl create cluster --name pynomaly-production-new \
     --region us-west-2 \
     --nodegroup-name workers \
     --node-type m5.xlarge \
     --nodes 3 \
     --nodes-min 3 \
     --nodes-max 10
   ```

2. **Install necessary components**
   ```bash
   # Install ingress controller
   kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/aws/deploy.yaml
   
   # Install cert-manager
   kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
   
   # Install metrics server
   kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
   ```

3. **Restore data**
   ```bash
   # Create namespace
   kubectl create namespace pynomaly
   
   # Restore database from backup
   # (Detailed steps depend on backup strategy)
   
   # Deploy application
   helm install pynomaly ./deploy/helm/pynomaly \
     --namespace pynomaly \
     --set pynomaly.image.tag=latest \
     --values deploy/helm/pynomaly/values-production.yaml
   ```

## Monitoring and Alerting

### Key Metrics to Monitor
- API response time (< 1000ms p95)
- Error rate (< 1%)
- Pod restart count
- Database connection pool usage
- Memory and CPU utilization
- Disk space usage

### Critical Alerts
1. **Service Down**: API health check fails for 3 consecutive minutes
2. **High Error Rate**: Error rate > 5% for 5 minutes
3. **Database Issues**: Connection failures or slow queries
4. **Resource Exhaustion**: CPU > 80% or Memory > 85% for 10 minutes
5. **Storage Issues**: Disk usage > 90%

## Post-Incident Activities

### 1. Immediate (0-2 hours)
- [ ] Confirm service is fully restored
- [ ] Update status page
- [ ] Notify stakeholders
- [ ] Document timeline of events

### 2. Short-term (2-24 hours)
- [ ] Complete incident report
- [ ] Identify root cause
- [ ] Implement immediate fixes
- [ ] Review monitoring gaps

### 3. Long-term (1-7 days)
- [ ] Conduct post-mortem meeting
- [ ] Update runbooks and procedures
- [ ] Implement preventive measures
- [ ] Test disaster recovery procedures

## Backup and Recovery Strategy

### Automated Backups
- **Database**: Daily PostgreSQL dumps to S3
- **Configuration**: Helm values and Kubernetes manifests
- **Application Data**: User uploads and processed datasets

### Backup Locations
- **Primary**: AWS S3 (us-west-2)
- **Secondary**: AWS S3 (us-east-1)
- **Retention**: 30 days for daily, 12 months for weekly

### Recovery Testing
- Monthly disaster recovery drills
- Quarterly full environment recovery tests
- Annual cross-region failover tests

## Contact Information and Escalation

### Escalation Matrix
1. **Level 1**: On-call engineer (0-30 minutes)
2. **Level 2**: Infrastructure team lead (30-60 minutes)
3. **Level 3**: Engineering manager (1-2 hours)
4. **Level 4**: CTO/VP Engineering (2+ hours)

### External Vendors
- **AWS Support**: Enterprise support plan
- **DNS Provider**: Cloudflare
- **Monitoring**: DataDog, PagerDuty

## Maintenance Windows

### Scheduled Maintenance
- **Frequency**: Monthly (first Sunday, 2-4 AM UTC)
- **Duration**: 2 hours maximum
- **Advance Notice**: 72 hours minimum

### Emergency Maintenance
- **Authorization**: Engineering manager or above
- **Communication**: Immediate notification to all stakeholders
- **Documentation**: Incident ticket and timeline required