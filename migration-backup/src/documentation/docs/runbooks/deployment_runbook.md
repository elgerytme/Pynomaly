# Pynomaly Deployment Runbook

## Overview
This runbook provides step-by-step instructions for deploying Pynomaly to different environments (development, staging, production).

## Prerequisites
- Kubernetes cluster access
- kubectl configured
- Docker registry access
- Access to configuration management systems
- Monitoring and alerting systems configured

## Deployment Environments

### Development Environment
**Purpose**: Feature development and initial testing
**URL**: http://dev.pynomaly.local
**Namespace**: pynomaly-dev

### Staging Environment
**Purpose**: Pre-production testing and validation
**URL**: https://staging.pynomaly.com
**Namespace**: pynomaly-staging

### Production Environment
**Purpose**: Live production workloads
**URL**: https://pynomaly.com
**Namespace**: pynomaly-prod

## Pre-Deployment Checklist

### Code Quality
- [ ] All tests passing (unit, integration, security)
- [ ] Code review completed and approved
- [ ] Security scan completed with no critical issues
- [ ] Performance benchmarks meet requirements
- [ ] Documentation updated

### Infrastructure
- [ ] Kubernetes cluster health verified
- [ ] Database backups completed
- [ ] Monitoring systems operational
- [ ] Alerting rules configured
- [ ] SSL certificates valid

### Configuration
- [ ] Environment variables validated
- [ ] Secrets updated in Kubernetes
- [ ] ConfigMaps updated
- [ ] Resource limits reviewed
- [ ] Scaling policies configured

## Deployment Process

### Step 1: Pre-Deployment Preparation

1. **Verify Cluster Status**
   ```bash
   kubectl cluster-info
   kubectl get nodes
   kubectl get pods --all-namespaces
   ```

2. **Check Resource Availability**
   ```bash
   kubectl top nodes
   kubectl describe nodes
   ```

3. **Verify Docker Image**
   ```bash
   docker pull pynomaly:latest
   docker images | grep pynomaly
   ```

### Step 2: Database Migration (if required)

1. **Backup Current Database**
   ```bash
   kubectl exec -n pynomaly-prod postgres-0 -- pg_dump -U postgres pynomaly > backup_$(date +%Y%m%d_%H%M%S).sql
   ```

2. **Run Database Migrations**
   ```bash
   kubectl apply -f k8s/migrations/
   kubectl wait --for=condition=complete job/migration-job --timeout=300s
   ```

3. **Verify Migration Success**
   ```bash
   kubectl logs job/migration-job
   kubectl exec -n pynomaly-prod postgres-0 -- psql -U postgres -d pynomaly -c "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1;"
   ```

### Step 3: Deploy Application

1. **Apply Kubernetes Manifests**
   ```bash
   # Deploy in order
   kubectl apply -f k8s/prod/namespace.yaml
   kubectl apply -f k8s/prod/secrets.yaml
   kubectl apply -f k8s/prod/configmap.yaml
   kubectl apply -f k8s/prod/databases.yaml
   kubectl apply -f k8s/prod/pynomaly-prod.yaml
   kubectl apply -f k8s/prod/ingress.yaml
   ```

2. **Wait for Deployment to Complete**
   ```bash
   kubectl rollout status deployment/pynomaly-prod-app -n pynomaly-prod --timeout=600s
   ```

3. **Verify Pod Status**
   ```bash
   kubectl get pods -n pynomaly-prod
   kubectl describe pods -n pynomaly-prod
   ```

### Step 4: Post-Deployment Verification

1. **Health Check**
   ```bash
   kubectl exec -n pynomaly-prod deployment/pynomaly-prod-app -- curl -f http://localhost:8000/health
   ```

2. **API Endpoint Test**
   ```bash
   curl -f https://pynomaly.com/api/v1/health
   curl -f https://pynomaly.com/metrics
   ```

3. **Database Connectivity**
   ```bash
   kubectl exec -n pynomaly-prod deployment/pynomaly-prod-app -- python -c "
   import psycopg2
   conn = psycopg2.connect(host='postgres-prod-service', database='pynomaly', user='postgres', password='${DB_PASSWORD}')
   print('Database connection successful')
   conn.close()
   "
   ```

### Step 5: Monitoring and Alerting

1. **Verify Monitoring**
   ```bash
   # Check Prometheus targets
   curl -s http://prometheus:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job=="pynomaly")'
   
   # Check Grafana dashboards
   curl -s http://grafana:3000/api/dashboards/search | jq '.[] | select(.title | contains("Pynomaly"))'
   ```

2. **Test Alerting**
   ```bash
   # Trigger test alert
   kubectl exec -n pynomaly-prod deployment/pynomaly-prod-app -- curl -X POST http://localhost:8000/test/alert
   ```

3. **Check Logs**
   ```bash
   kubectl logs -n pynomaly-prod deployment/pynomaly-prod-app --tail=100
   ```

## Rollback Process

### Immediate Rollback (Emergency)

1. **Rollback Deployment**
   ```bash
   kubectl rollout undo deployment/pynomaly-prod-app -n pynomaly-prod
   kubectl rollout status deployment/pynomaly-prod-app -n pynomaly-prod
   ```

2. **Verify Rollback**
   ```bash
   kubectl get pods -n pynomaly-prod
   curl -f https://pynomaly.com/health
   ```

### Complete Rollback (Planned)

1. **Rollback Database (if migrations were applied)**
   ```bash
   kubectl exec -n pynomaly-prod postgres-0 -- psql -U postgres -d pynomaly < backup_previous.sql
   ```

2. **Rollback Application**
   ```bash
   kubectl apply -f k8s/prod/pynomaly-prod.yaml.backup
   kubectl rollout status deployment/pynomaly-prod-app -n pynomaly-prod
   ```

3. **Verify System Health**
   ```bash
   ./scripts/health_check.sh
   ```

## Common Issues and Troubleshooting

### Issue: Pod Not Starting

**Symptoms**:
- Pods stuck in Pending or CrashLoopBackOff state
- Application health checks failing

**Diagnosis**:
```bash
kubectl describe pod <pod-name> -n pynomaly-prod
kubectl logs <pod-name> -n pynomaly-prod
kubectl get events -n pynomaly-prod --sort-by=.metadata.creationTimestamp
```

**Resolution**:
1. Check resource availability
2. Verify image pull secrets
3. Review configuration and environment variables
4. Check database connectivity

### Issue: Database Connection Failed

**Symptoms**:
- Application logs showing database connection errors
- Health check failing

**Diagnosis**:
```bash
kubectl exec -n pynomaly-prod deployment/pynomaly-prod-app -- nslookup postgres-prod-service
kubectl exec -n pynomaly-prod postgres-0 -- pg_isready
```

**Resolution**:
1. Verify database pod is running
2. Check database service configuration
3. Verify database credentials in secrets
4. Check network policies

### Issue: High Response Time

**Symptoms**:
- API responses taking longer than expected
- Monitoring alerts for response time thresholds

**Diagnosis**:
```bash
kubectl top pods -n pynomaly-prod
kubectl exec -n pynomaly-prod deployment/pynomaly-prod-app -- curl -w "@curl-format.txt" -s http://localhost:8000/api/v1/health
```

**Resolution**:
1. Check resource utilization
2. Scale deployment if needed
3. Verify database performance
4. Check network latency

## Performance Optimization

### Scaling Operations

1. **Manual Scaling**
   ```bash
   kubectl scale deployment/pynomaly-prod-app --replicas=5 -n pynomaly-prod
   ```

2. **Horizontal Pod Autoscaler**
   ```bash
   kubectl autoscale deployment/pynomaly-prod-app --cpu-percent=70 --min=3 --max=10 -n pynomaly-prod
   ```

3. **Verify Scaling**
   ```bash
   kubectl get hpa -n pynomaly-prod
   kubectl get pods -n pynomaly-prod
   ```

### Resource Optimization

1. **Update Resource Limits**
   ```yaml
   resources:
     requests:
       memory: "512Mi"
       cpu: "500m"
     limits:
       memory: "1Gi"
       cpu: "1000m"
   ```

2. **Apply Changes**
   ```bash
   kubectl apply -f k8s/prod/pynomaly-prod.yaml
   kubectl rollout status deployment/pynomaly-prod-app -n pynomaly-prod
   ```

## Security Considerations

### SSL Certificate Management

1. **Check Certificate Expiry**
   ```bash
   kubectl get certificates -n pynomaly-prod
   kubectl describe certificate pynomaly-tls -n pynomaly-prod
   ```

2. **Renew Certificate**
   ```bash
   kubectl delete secret pynomaly-tls -n pynomaly-prod
   kubectl apply -f k8s/prod/certificates.yaml
   ```

### Secret Rotation

1. **Update Database Password**
   ```bash
   kubectl create secret generic pynomaly-prod-secrets \
     --from-literal=POSTGRES_PASSWORD=new_password \
     --dry-run=client -o yaml | kubectl replace -f -
   ```

2. **Restart Pods**
   ```bash
   kubectl rollout restart deployment/pynomaly-prod-app -n pynomaly-prod
   ```

## Monitoring and Alerting

### Key Metrics to Monitor

1. **Application Metrics**
   - Response time (P95, P99)
   - Request rate
   - Error rate
   - Active connections

2. **Infrastructure Metrics**
   - CPU utilization
   - Memory usage
   - Disk I/O
   - Network throughput

3. **Database Metrics**
   - Connection pool utilization
   - Query performance
   - Cache hit ratio
   - Replication lag

### Alert Rules

1. **High Error Rate**
   ```
   rate(http_requests_total{status=~"5.."}[5m]) > 0.05
   ```

2. **High Response Time**
   ```
   histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
   ```

3. **High CPU Usage**
   ```
   rate(container_cpu_usage_seconds_total[5m]) > 0.8
   ```

## Contact Information

### On-Call Team
- **Primary**: DevOps Team (+1-555-0123)
- **Secondary**: Platform Team (+1-555-0124)
- **Escalation**: Engineering Manager (+1-555-0125)

### Communication Channels
- **Slack**: #pynomaly-alerts
- **Email**: pynomaly-ops@company.com
- **PagerDuty**: https://company.pagerduty.com

## References
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Pynomaly Architecture Guide](../architecture/system_design.md)
- [Security Guidelines](../security/security_guidelines.md)
- [Performance Tuning Guide](../performance/optimization_guide.md)