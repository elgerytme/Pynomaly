# Operational Monitoring Guide

## Overview
This guide provides comprehensive monitoring procedures for the Pynomaly production environment, including metrics, alerting, and troubleshooting.

## Monitoring Stack Architecture

### Core Components
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Alertmanager**: Alert routing and notification
- **Jaeger**: Distributed tracing
- **Fluentd/ELK**: Log aggregation and analysis

### Access Points
- **Grafana**: https://monitoring.pynomaly.com/grafana
- **Prometheus**: https://monitoring.pynomaly.com/prometheus
- **Alertmanager**: https://monitoring.pynomaly.com/alertmanager
- **Kibana**: https://monitoring.pynomaly.com/kibana

## Key Performance Indicators (KPIs)

### Application Health
- **Service Availability**: > 99.9% uptime
- **API Response Time**: < 500ms (P95)
- **Error Rate**: < 1% of total requests
- **Throughput**: Monitor requests per second

### Infrastructure Health
- **CPU Utilization**: < 70% average
- **Memory Usage**: < 80% of allocated
- **Disk I/O**: Monitor read/write latency
- **Network**: Monitor bandwidth and packet loss

### Business Metrics
- **Active Users**: Daily and monthly active users
- **Model Predictions**: Predictions per hour
- **Data Processing**: Volume of processed data
- **Feature Usage**: Track feature adoption

## Dashboard Overview

### 1. System Overview Dashboard
**Purpose**: High-level system health and performance
**Key Metrics**:
- Service status indicators
- Request rate and response times
- Error rates and alerts
- Resource utilization

**Access**: Grafana → Dashboards → System Overview

### 2. Application Performance Dashboard
**Purpose**: Detailed application metrics
**Key Metrics**:
- API endpoint performance
- Database query performance
- Queue processing rates
- Cache hit ratios

**Access**: Grafana → Dashboards → Application Performance

### 3. Infrastructure Dashboard
**Purpose**: Infrastructure monitoring
**Key Metrics**:
- Kubernetes cluster health
- Node resource usage
- Pod status and restarts
- Storage utilization

**Access**: Grafana → Dashboards → Infrastructure

### 4. Business Metrics Dashboard
**Purpose**: Business KPIs and usage analytics
**Key Metrics**:
- User engagement metrics
- Model prediction accuracy
- Data processing volumes
- Revenue-related metrics

**Access**: Grafana → Dashboards → Business Metrics

## Alert Hierarchy and Response

### Critical Alerts (P0)
**Response Time**: Immediate (< 5 minutes)
**Escalation**: Page on-call engineer

#### Alert Types:
- Service completely down
- Database unavailable
- Security incidents
- Data corruption

#### Response Actions:
1. Acknowledge alert immediately
2. Join incident response channel
3. Begin initial assessment
4. Escalate if needed

### High Priority Alerts (P1)
**Response Time**: < 30 minutes
**Escalation**: Notify on-call engineer

#### Alert Types:
- High error rates (> 5%)
- Performance degradation
- Partial service outage
- Resource exhaustion

#### Response Actions:
1. Acknowledge alert
2. Investigate root cause
3. Implement quick fixes
4. Monitor for resolution

### Medium Priority Alerts (P2)
**Response Time**: < 2 hours
**Escalation**: Standard business hours

#### Alert Types:
- Minor performance issues
- Non-critical service degradation
- Resource warnings

#### Response Actions:
1. Log and prioritize
2. Investigate during business hours
3. Plan remediation

### Low Priority Alerts (P3)
**Response Time**: Next business day
**Escalation**: Regular development workflow

#### Alert Types:
- Information alerts
- Trend notifications
- Capacity planning alerts

## Monitoring Queries and Commands

### Prometheus Queries

#### Application Metrics
```promql
# Request rate
rate(http_requests_total[5m])

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# Response time P95
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Active connections
database_connections_active

# Queue depth
queue_depth_total
```

#### Infrastructure Metrics
```promql
# CPU usage
rate(container_cpu_usage_seconds_total[5m])

# Memory usage
container_memory_usage_bytes / container_spec_memory_limit_bytes

# Disk usage
(node_filesystem_size_bytes - node_filesystem_free_bytes) / node_filesystem_size_bytes

# Network I/O
rate(container_network_receive_bytes_total[5m])
```

#### Business Metrics
```promql
# Active users
increase(user_sessions_total[1h])

# Model predictions
rate(ml_predictions_total[5m])

# Data processing volume
increase(data_processed_bytes_total[1h])
```

### Kubernetes Monitoring Commands

#### Cluster Health
```bash
# Check cluster status
kubectl cluster-info
kubectl get nodes
kubectl get pods --all-namespaces | grep -v Running

# Check resource usage
kubectl top nodes
kubectl top pods --all-namespaces --sort-by=memory

# Check events
kubectl get events --all-namespaces --sort-by=.metadata.creationTimestamp
```

#### Application Health
```bash
# Check pod status
kubectl get pods -n pynomaly-production
kubectl describe pods -n pynomaly-production

# Check logs
kubectl logs -f deployment/pynomaly-api -n pynomaly-production
kubectl logs --previous -n pynomaly-production deployment/pynomaly-api

# Check services
kubectl get services -n pynomaly-production
kubectl describe service pynomaly-api -n pynomaly-production
```

### Database Monitoring

#### PostgreSQL Queries
```sql
-- Check active connections
SELECT count(*) FROM pg_stat_activity WHERE state = 'active';

-- Check long-running queries
SELECT now() - query_start as duration, query 
FROM pg_stat_activity 
WHERE state = 'active' 
ORDER BY duration DESC;

-- Check database size
SELECT pg_size_pretty(pg_database_size('pynomaly'));

-- Check table sizes
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) 
FROM pg_tables 
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

#### Redis Monitoring
```bash
# Connect to Redis
kubectl exec -it redis-0 -n pynomaly-production -- redis-cli

# Check memory usage
INFO memory

# Check connected clients
INFO clients

# Check keyspace
INFO keyspace

# Monitor commands
MONITOR
```

## Log Analysis and Troubleshooting

### Log Locations
```bash
# Application logs
kubectl logs -n pynomaly-production deployment/pynomaly-api

# System logs
kubectl logs -n kube-system

# Nginx/Ingress logs
kubectl logs -n ingress-nginx

# Database logs
kubectl logs -n pynomaly-production postgres-0
```

### Common Log Patterns

#### Error Investigation
```bash
# Search for errors in application logs
kubectl logs -n pynomaly-production deployment/pynomaly-api | grep -E "(ERROR|CRITICAL|FATAL)"

# Search for specific error patterns
kubectl logs -n pynomaly-production deployment/pynomaly-api | grep "database connection"

# Search for HTTP errors
kubectl logs -n pynomaly-production deployment/pynomaly-api | grep -E "HTTP/[0-9\.]+ [45][0-9][0-9]"
```

#### Performance Investigation
```bash
# Search for slow queries
kubectl logs -n pynomaly-production deployment/pynomaly-api | grep "slow query"

# Search for timeout errors
kubectl logs -n pynomaly-production deployment/pynomaly-api | grep -i timeout

# Search for memory issues
kubectl logs -n pynomaly-production deployment/pynomaly-api | grep -i "out of memory"
```

### ELK Stack Queries (Kibana)

#### Application Logs
```
# High-level error search
level:ERROR OR level:CRITICAL

# API errors
message:"HTTP 5*" AND kubernetes.namespace:"pynomaly-production"

# Database errors
message:"database" AND level:ERROR

# Performance issues
message:"slow" OR message:"timeout" OR response_time:>2000
```

#### Infrastructure Logs
```
# Pod restart events
kubernetes.event.reason:"Failed" OR kubernetes.event.reason:"FailedScheduling"

# Resource issues
message:"insufficient resources" OR message:"evicted"

# Network issues
message:"connection refused" OR message:"network timeout"
```

## Performance Tuning Guidelines

### Application Performance

#### Response Time Optimization
```bash
# Check slow endpoints
kubectl exec -it deployment/prometheus -n monitoring -- \
  promtool query instant 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) by (endpoint)'

# Identify bottlenecks
kubectl logs -n pynomaly-production deployment/pynomaly-api | grep "slow query" | tail -20
```

#### Resource Optimization
```bash
# Check resource usage
kubectl top pods -n pynomaly-production

# Identify resource-constrained pods
kubectl describe pods -n pynomaly-production | grep -A 5 -B 5 "cpu\|memory"

# Scale up if needed
kubectl scale deployment/pynomaly-api --replicas=5 -n pynomaly-production
```

### Database Performance

#### Query Optimization
```sql
-- Identify slow queries
SELECT query, mean_exec_time, calls 
FROM pg_stat_statements 
ORDER BY mean_exec_time DESC 
LIMIT 10;

-- Check index usage
SELECT schemaname, tablename, attname, n_distinct, correlation 
FROM pg_stats 
WHERE schemaname = 'public';
```

#### Connection Pool Tuning
```bash
# Check connection pool status
kubectl exec -it deployment/pynomaly-api -n pynomaly-production -- \
  python -c "from sqlalchemy import create_engine; engine = create_engine('postgresql://...'); print(engine.pool.status())"
```

## Capacity Planning

### Resource Monitoring
```bash
# CPU trend analysis
kubectl exec -it deployment/prometheus -n monitoring -- \
  promtool query instant 'avg_over_time(rate(container_cpu_usage_seconds_total[5m])[7d])'

# Memory trend analysis
kubectl exec -it deployment/prometheus -n monitoring -- \
  promtool query instant 'avg_over_time(container_memory_usage_bytes[7d])'

# Storage growth analysis
kubectl exec -it deployment/prometheus -n monitoring -- \
  promtool query instant 'increase(node_filesystem_size_bytes[7d])'
```

### Scaling Triggers
- **CPU**: Scale up when average CPU > 70% for 5 minutes
- **Memory**: Scale up when memory usage > 80% for 3 minutes
- **Response Time**: Scale up when P95 > 1 second for 2 minutes
- **Error Rate**: Investigate when error rate > 5% for 1 minute

## Health Check Procedures

### Application Health Checks
```bash
# HTTP health check
curl -f https://api.pynomaly.com/health

# Detailed health check
curl -s https://api.pynomaly.com/health | jq '.'

# Database connectivity check
kubectl exec -it deployment/pynomaly-api -n pynomaly-production -- \
  python -c "from pynomaly.infrastructure.database import get_database; print(get_database().is_connected())"
```

### Infrastructure Health Checks
```bash
# Kubernetes health
kubectl get componentstatuses

# Node health
kubectl get nodes
kubectl describe nodes | grep -E "(Ready|OutOfDisk|MemoryPressure|DiskPressure)"

# Pod health
kubectl get pods -n pynomaly-production
kubectl get pods --all-namespaces | grep -v Running
```

## Alerting Configuration

### Alert Rules Configuration
The alert rules are managed via the monitoring configuration:

```bash
# Apply alert rules
kubectl apply -f config/monitoring/prometheus_alert_rules.yml

# Reload Prometheus configuration
kubectl exec -it deployment/prometheus -n monitoring -- \
  curl -X POST http://localhost:9090/-/reload
```

### Notification Channels

#### Slack Integration
```bash
# Test Slack notification
curl -X POST "SLACK_WEBHOOK_URL" \
  -H 'Content-type: application/json' \
  --data '{"text":"Test alert from Pynomaly monitoring"}'
```

#### PagerDuty Integration
```bash
# Test PagerDuty notification
curl -X POST "https://events.pagerduty.com/v2/enqueue" \
  -H 'Content-Type: application/json' \
  -d '{
    "routing_key": "PAGERDUTY_INTEGRATION_KEY",
    "event_action": "trigger",
    "payload": {
      "summary": "Test alert from Pynomaly",
      "severity": "critical",
      "source": "monitoring"
    }
  }'
```

## Backup Monitoring

### Backup Health Checks
```bash
# Check recent backups
python scripts/deployment/backup_deployment.py --list-recent

# Verify backup integrity
python scripts/deployment/backup_deployment.py --verify-latest

# Check backup storage
aws s3 ls s3://pynomaly-backups/backups/ --recursive --human-readable
```

### Backup Alerting
Monitor backup job completion and failures:
```promql
# Backup success rate
rate(backup_jobs_total{status="success"}[24h]) / rate(backup_jobs_total[24h])

# Backup failures
increase(backup_jobs_total{status="failed"}[24h])

# Time since last successful backup
time() - backup_last_success_timestamp
```

## Security Monitoring

### Security Metrics
```promql
# Failed authentication attempts
rate(authentication_failed_total[5m])

# Blocked requests by WAF
rate(waf_blocked_requests_total[5m])

# Suspicious network activity
rate(network_connections_refused_total[5m])
```

### Security Log Analysis
```bash
# Search for security events
kubectl logs -n pynomaly-production deployment/pynomaly-api | grep -i "security\|auth\|login\|unauthorized"

# Check for failed login attempts
kubectl logs -n pynomaly-production deployment/pynomaly-api | grep "authentication failed"

# Monitor for SQL injection attempts
kubectl logs -n pynomaly-production deployment/pynomaly-api | grep -i "sql injection"
```

## Troubleshooting Playbooks

### High Response Time
1. Check current load and resource usage
2. Identify slow database queries
3. Review recent deployments
4. Scale up if needed
5. Implement caching if appropriate

### High Error Rate
1. Check application logs for error patterns
2. Verify external service availability
3. Check database connectivity
4. Review recent configuration changes
5. Consider rollback if deployment-related

### Memory Issues
1. Check for memory leaks in application logs
2. Review memory usage trends
3. Identify resource-intensive operations
4. Scale up or optimize code
5. Restart services if necessary

### Database Performance Issues
1. Check for long-running queries
2. Identify blocking locks
3. Review connection pool status
4. Check for index optimization opportunities
5. Consider read replica scaling

## Maintenance and Updates

### Regular Maintenance Tasks
- **Daily**: Review dashboard alerts and metrics
- **Weekly**: Analyze performance trends and capacity
- **Monthly**: Update monitoring rules and thresholds
- **Quarterly**: Review and update alert playbooks

### Monitoring System Updates
```bash
# Update Prometheus configuration
kubectl apply -f config/monitoring/prometheus-config.yml

# Update Grafana dashboards
kubectl apply -f config/monitoring/grafana-dashboards/

# Update alert rules
kubectl apply -f config/monitoring/prometheus_alert_rules.yml

# Restart monitoring services
kubectl rollout restart deployment/prometheus -n monitoring
kubectl rollout restart deployment/grafana -n monitoring
```

## Contact Information

### Monitoring Team
- **Primary**: Platform Engineering Team
- **Secondary**: DevOps Engineer
- **Escalation**: Engineering Manager

### Alert Response Team
- **On-Call Engineer**: [PagerDuty rotation]
- **Database Expert**: [DBA on-call]
- **Security Contact**: [Security team lead]

### Communication Channels
- **Critical Alerts**: #incident-response
- **General Alerts**: #alerts
- **Monitoring Issues**: #monitoring

---

**Last Updated**: 2024-07-10
**Version**: 1.0
**Owner**: Platform Engineering Team
**Review Cycle**: Monthly