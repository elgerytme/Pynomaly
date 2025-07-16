# Incident Response Playbook

## Overview
This document provides step-by-step procedures for responding to production incidents in the Pynomaly system.

## Incident Severity Levels

### Critical (P0)
- Complete system outage
- Data loss or corruption
- Security breach
- Revenue-impacting issues

**Response Time**: Immediate (< 15 minutes)
**Escalation**: Page on-call engineer immediately

### High (P1)
- Significant feature degradation
- High error rates (> 5%)
- Performance degradation (> 2s response time)
- Partial system outage

**Response Time**: < 1 hour
**Escalation**: Notify on-call engineer within 30 minutes

### Medium (P2)
- Minor feature issues
- Non-critical service degradation
- Monitoring alerts

**Response Time**: < 4 hours
**Escalation**: Standard business hours response

### Low (P3)
- Cosmetic issues
- Documentation updates
- Non-urgent improvements

**Response Time**: Next business day
**Escalation**: Regular development workflow

## Incident Response Process

### 1. Initial Response (0-15 minutes)

#### Immediate Actions
1. **Acknowledge the alert** in monitoring system
2. **Join incident channel**: `#incident-response`
3. **Declare incident** with severity level
4. **Assign incident commander** (on-call engineer or team lead)

#### Assessment Commands
```bash
# Check system health
kubectl get pods -n pynomaly-production
kubectl get nodes
kubectl top nodes

# Check recent deployments
kubectl rollout history deployment/pynomaly-api -n pynomaly-production

# Check application logs
kubectl logs -l app.kubernetes.io/name=pynomaly-api -n pynomaly-production --tail=100

# Check monitoring dashboards
# - Grafana: https://monitoring.pynomaly.com/grafana
# - Prometheus: https://monitoring.pynomaly.com/prometheus
```

### 2. Investigation (15-60 minutes)

#### System Health Checks
```bash
# Check database connectivity
kubectl exec -it deployment/pynomaly-api -n pynomaly-production -- \
  python -c "from pynomaly.infrastructure.database import get_database; print('DB:', get_database().is_connected())"

# Check Redis connectivity
kubectl exec -it deployment/pynomaly-api -n pynomaly-production -- \
  python -c "import redis; r=redis.from_url('redis://redis:6379'); print('Redis:', r.ping())"

# Check external dependencies
kubectl exec -it deployment/pynomaly-api -n pynomaly-production -- \
  curl -s -o /dev/null -w "%{http_code}" https://api.external-service.com/health
```

#### Performance Diagnostics
```bash
# Check resource usage
kubectl top pods -n pynomaly-production

# Check API response times
kubectl exec -it deployment/pynomaly-api -n pynomaly-production -- \
  curl -w "@curl-format.txt" -s -o /dev/null https://api.pynomaly.com/health

# Check error rates
kubectl logs -l app.kubernetes.io/name=pynomaly-api -n pynomaly-production | \
  grep -E "(ERROR|CRITICAL)" | tail -20
```

### 3. Containment and Resolution

#### Database Issues
```bash
# Check database connections
kubectl exec -it postgres-0 -n pynomaly-production -- \
  psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"

# Check database locks
kubectl exec -it postgres-0 -n pynomaly-production -- \
  psql -U postgres -c "SELECT * FROM pg_locks WHERE NOT granted;"

# Restart database connection pool
kubectl rollout restart deployment/pynomaly-api -n pynomaly-production
```

#### High Memory Usage
```bash
# Check memory usage
kubectl top pods -n pynomaly-production

# Scale up replicas temporarily
kubectl scale deployment/pynomaly-api --replicas=6 -n pynomaly-production

# Monitor memory leaks
kubectl exec -it deployment/pynomaly-api -n pynomaly-production -- \
  python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

#### High CPU Usage
```bash
# Check CPU usage
kubectl top pods -n pynomaly-production

# Scale horizontally
kubectl scale deployment/pynomaly-api --replicas=8 -n pynomaly-production

# Check for CPU-intensive processes
kubectl exec -it deployment/pynomaly-api -n pynomaly-production -- top
```

#### Application Errors
```bash
# Check recent error logs
kubectl logs -l app.kubernetes.io/name=pynomaly-api -n pynomaly-production --since=1h | \
  grep -E "(ERROR|CRITICAL|FATAL)"

# Check application health endpoint
curl -f https://api.pynomaly.com/health || echo "Health check failed"

# Restart application if needed
kubectl rollout restart deployment/pynomaly-api -n pynomaly-production
```

### 4. Rollback Procedures

#### Application Rollback
```bash
# Check rollout history
kubectl rollout history deployment/pynomaly-api -n pynomaly-production

# Rollback to previous version
kubectl rollout undo deployment/pynomaly-api -n pynomaly-production

# Rollback to specific revision
kubectl rollout undo deployment/pynomaly-api -n pynomaly-production --to-revision=<revision>

# Monitor rollback status
kubectl rollout status deployment/pynomaly-api -n pynomaly-production
```

#### Database Rollback
```bash
# Restore from latest backup
python scripts/deployment/backup_deployment.py --type database --restore-latest

# Point-in-time recovery
kubectl exec -it postgres-0 -n pynomaly-production -- \
  pg_restore -d pynomaly_prod /backups/latest_backup.sql
```

### 5. Communication

#### Incident Updates
- Update `#incident-response` channel every 30 minutes
- Post to status page: https://status.pynomaly.com
- Notify stakeholders via pre-defined communication plan

#### Status Page Updates
```bash
# Update status page
curl -X POST "https://api.statuspage.io/v1/pages/PAGE_ID/incidents" \
  -H "Authorization: OAuth TOKEN" \
  -d '{
    "incident": {
      "name": "API Performance Issues",
      "status": "investigating",
      "impact_override": "minor"
    }
  }'
```

## Common Incident Scenarios

### API Downtime

#### Symptoms
- Health check failures
- 5xx error responses
- Pod restart loops

#### Investigation
```bash
# Check pod status
kubectl get pods -l app.kubernetes.io/name=pynomaly-api -n pynomaly-production

# Check pod logs
kubectl logs -l app.kubernetes.io/name=pynomaly-api -n pynomaly-production --tail=50

# Check resource limits
kubectl describe pods -l app.kubernetes.io/name=pynomaly-api -n pynomaly-production
```

#### Resolution
1. Check if it's a deployment issue - rollback if needed
2. Check resource constraints - scale up if needed
3. Check dependencies - database, Redis, external APIs
4. Restart pods if application is stuck

### Database Performance Issues

#### Symptoms
- Slow query responses
- Connection timeouts
- High database CPU/memory

#### Investigation
```bash
# Check active connections
kubectl exec -it postgres-0 -n pynomaly-production -- \
  psql -U postgres -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';"

# Check slow queries
kubectl exec -it postgres-0 -n pynomaly-production -- \
  psql -U postgres -c "SELECT query, query_start FROM pg_stat_activity WHERE state = 'active' ORDER BY query_start;"

# Check database locks
kubectl exec -it postgres-0 -n pynomaly-production -- \
  psql -U postgres -c "SELECT * FROM pg_locks WHERE NOT granted;"
```

#### Resolution
1. Kill long-running queries if identified
2. Restart application to reset connection pool
3. Scale up database resources if needed
4. Implement query optimizations

### High Error Rates

#### Symptoms
- Error rate > 5%
- Increased 4xx/5xx responses
- User complaints

#### Investigation
```bash
# Check error distribution
kubectl logs -l app.kubernetes.io/name=pynomaly-api -n pynomaly-production --since=1h | \
  grep -E "HTTP/[0-9\.]+ [45][0-9][0-9]" | \
  awk '{print $6}' | sort | uniq -c | sort -nr

# Check specific error patterns
kubectl logs -l app.kubernetes.io/name=pynomaly-api -n pynomaly-production --since=1h | \
  grep "ERROR" | head -20
```

#### Resolution
1. Identify error patterns and root cause
2. Check if it's related to recent deployment
3. Implement quick fixes or rollback
4. Scale up if it's a capacity issue

## Post-Incident Activities

### 1. Incident Resolution
- Confirm all systems are operational
- Remove any temporary fixes
- Update monitoring thresholds if needed

### 2. Communication
- Update incident channel with resolution
- Close status page incident
- Notify stakeholders of resolution

### 3. Post-Mortem
- Schedule post-mortem meeting within 48 hours
- Document timeline, root cause, and action items
- Update runbooks based on lessons learned

## Emergency Contacts

### On-Call Engineers
- Primary: [On-call rotation via PagerDuty]
- Secondary: [Backup engineer contact]

### Escalation Path
1. On-call Engineer
2. Team Lead
3. Engineering Manager
4. VP of Engineering

### External Contacts
- Cloud Provider Support: [AWS Support case]
- CDN Provider: [CloudFlare support]
- Monitoring Vendor: [DataDog support]

## Tools and Access

### Required Access
- Kubernetes cluster access via kubectl
- Monitoring dashboards (Grafana, Prometheus)
- Log aggregation (CloudWatch, ELK)
- Status page admin access

### Emergency Access
- Break-glass admin access procedures
- Production database direct access (emergency only)
- Cloud provider root account access

## Recovery Time Objectives (RTO) and Recovery Point Objectives (RPO)

| Service Component | RTO | RPO |
|------------------|-----|-----|
| API Service | < 15 minutes | < 5 minutes |
| Database | < 30 minutes | < 15 minutes |
| ML Models | < 1 hour | < 1 hour |
| File Storage | < 2 hours | < 1 hour |

## Monitoring and Alerting

### Critical Alerts
- API health check failures
- Database connection issues
- High error rates (> 5%)
- Resource exhaustion (CPU > 80%, Memory > 90%)

### Alert Channels
- PagerDuty for critical alerts
- Slack `#alerts` for warnings
- Email for informational alerts

## Backup and Recovery

### Automated Backups
- Database: Every 6 hours, retained for 30 days
- Configuration: Daily, retained for 90 days
- ML Models: Weekly, retained for 180 days

### Manual Backup
```bash
# Create emergency backup
python scripts/deployment/backup_deployment.py --type full

# Restore from backup
python scripts/deployment/backup_deployment.py --restore --backup-id <backup_id>
```

## Security Incident Response

### Suspected Security Breach
1. **Immediate**: Isolate affected systems
2. **Alert**: Security team and management
3. **Preserve**: Evidence and logs
4. **Investigate**: Scope and impact
5. **Remediate**: Close vulnerabilities
6. **Communicate**: Internal and external stakeholders

### Commands for Security Investigation
```bash
# Check for unusual network activity
kubectl logs -l app.kubernetes.io/name=pynomaly-api -n pynomaly-production | \
  grep -E "failed.*login|unauthorized|403|401"

# Check for privilege escalation
kubectl get pods -o wide | grep -v Running

# Review recent changes
kubectl rollout history deployment/pynomaly-api -n pynomaly-production
```

---

**Last Updated**: 2024-07-10
**Version**: 1.0
**Owner**: Platform Engineering Team
**Review Cycle**: Monthly