# Incident Response Playbook

## Overview

This playbook provides comprehensive procedures for responding to incidents in the enterprise anomaly detection platform. It covers incident classification, response procedures, communication protocols, and post-incident activities.

## Incident Classification

### Severity Levels

#### P0 - Critical (Immediate Response)
**Response Time: 15 minutes**
- Complete service outage affecting all users
- Data loss or corruption
- Security breach with active exploitation
- Legal/regulatory compliance violation
- Financial impact >$10k/hour

#### P1 - High (Urgent Response)
**Response Time: 30 minutes**
- Partial service outage affecting >50% of users
- Performance degradation >50% below SLA
- Security vulnerability requiring immediate action
- Critical feature completely unavailable
- Financial impact >$1k/hour

#### P2 - Medium (Standard Response)
**Response Time: 2 hours**
- Performance degradation 20-50% below SLA
- Non-critical feature unavailable
- Monitoring/alerting system issues
- Third-party service dependencies affected
- Financial impact <$1k/hour

#### P3 - Low (Planned Response)
**Response Time: 24 hours**
- Minor performance issues
- Documentation or UI inconsistencies
- Non-urgent maintenance items
- Feature requests or enhancements

## Incident Response Team Structure

### Core Response Team
- **Incident Commander (IC)**: Overall incident coordination
- **Technical Lead**: Primary technical troubleshooting
- **Communications Lead**: Stakeholder communication
- **Customer Success Lead**: Customer impact assessment

### Extended Team (As Needed)
- **Security Lead**: Security-related incidents
- **Database Engineer**: Database-related issues
- **Infrastructure Engineer**: Infrastructure problems
- **Product Manager**: Feature/business impact decisions
- **Legal/Compliance**: Regulatory or legal issues

## Incident Response Process

### 1. Detection and Alert (0-5 minutes)

#### Automated Detection
- Monitoring alerts via PagerDuty/AlertManager
- Health check failures
- Performance threshold breaches
- Security event triggers

#### Manual Detection
- Customer reports
- Team member observations
- Third-party notifications

#### Initial Actions
```bash
# Check monitoring dashboards
open https://grafana.detection-platform.io/dashboards

# Check system status
kubectl get pods --all-namespaces | grep -v Running
kubectl top nodes

# Check recent deployments
kubectl rollout history deployment/anomaly-detection-green -n production
```

### 2. Initial Assessment (5-15 minutes)

#### Triage Questions
1. **What is the impact?**
   - How many users are affected?
   - Which services/features are impacted?
   - What is the business impact?

2. **What is the severity?**
   - Use severity classification above
   - Consider customer tier (enterprise vs. standard)
   - Time sensitivity (business hours vs. off-hours)

3. **What is the scope?**
   - Single service or multiple services?
   - Specific region or global?
   - All users or specific subset?

#### Assessment Commands
```bash
# Check service health
curl -f https://api.detection-platform.io/health/ready
curl -f https://api.detection-platform.io/health/live

# Check error rates
curl -s "http://prometheus:9090/api/v1/query?query=rate(http_requests_total{status=~\"5..\"}[5m])"

# Check response times
curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,rate(http_request_duration_seconds_bucket[5m]))"

# Check database status
kubectl exec -it deployment/anomaly-detection-green -n production -- \
  python -c "import psycopg2; print('DB OK')"
```

### 3. Escalation and Team Assembly (15-30 minutes)

#### Escalation Matrix
| Severity | Immediate Notification | Escalation Time |
|----------|----------------------|-----------------|
| P0 | IC, Technical Lead, Communications Lead | 15 minutes |
| P1 | IC, Technical Lead | 30 minutes |
| P2 | Technical Lead | 2 hours |
| P3 | Technical Lead | 24 hours |

#### Notification Channels
```bash
# PagerDuty escalation
curl -X POST https://api.pagerduty.com/incidents \
  -H "Authorization: Token $PAGERDUTY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "incident": {
      "type": "incident",
      "title": "Production Incident: [Description]",
      "service": {"id": "$SERVICE_ID", "type": "service_reference"},
      "urgency": "high",
      "body": {"type": "incident_body", "details": "Detailed description"}
    }
  }'

# Slack notifications
curl -X POST https://hooks.slack.com/services/$SLACK_WEBHOOK \
  -H "Content-Type: application/json" \
  -d '{
    "text": ":rotating_light: *P1 INCIDENT* :rotating_light:",
    "attachments": [{
      "color": "danger",
      "fields": [
        {"title": "Severity", "value": "P1", "short": true},
        {"title": "Impact", "value": "Service degradation", "short": true},
        {"title": "Incident Commander", "value": "@on-call-engineer", "short": true}
      ]
    }]
  }'
```

### 4. Investigation and Diagnosis

#### Common Investigation Steps

##### Application Issues
```bash
# Check pod status and logs
kubectl get pods -n production -l app=anomaly-detection
kubectl logs -f deployment/anomaly-detection-green -n production --tail=100

# Check recent changes
kubectl describe deployment/anomaly-detection-green -n production
kubectl rollout history deployment/anomaly-detection-green -n production

# Check resource usage
kubectl top pods -n production
kubectl describe hpa anomaly-detection-hpa -n production
```

##### Database Issues
```bash
# Check database connections
kubectl exec -it deployment/anomaly-detection-green -n production -- \
  python -c "
import psycopg2
import time
for i in range(5):
    try:
        conn = psycopg2.connect(host='$DB_HOST', database='anomaly_detection', user='app_user', password='$DB_PASSWORD')
        print(f'Connection {i+1}: SUCCESS')
        conn.close()
    except Exception as e:
        print(f'Connection {i+1}: FAILED - {e}')
    time.sleep(1)
"

# Check database performance
kubectl exec -it deployment/anomaly-detection-green -n production -- \
  psql -h $DB_HOST -U app_user -d anomaly_detection -c "
SELECT query, calls, total_time, mean_time 
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 10;"
```

##### Infrastructure Issues
```bash
# Check node status
kubectl get nodes
kubectl describe nodes | grep -A 10 "Conditions:"

# Check cluster events
kubectl get events --all-namespaces --sort-by='.lastTimestamp' | head -20

# Check AWS resources
aws elbv2 describe-target-health --target-group-arn $TARGET_GROUP_ARN
aws rds describe-db-instances --db-instance-identifier anomaly-detection-production-postgresql
```

#### Performance Issues
```bash
# Check system metrics
kubectl top nodes
kubectl top pods -n production

# Check application metrics
curl -s "http://prometheus:9090/api/v1/query?query=rate(http_requests_total[5m])"
curl -s "http://prometheus:9090/api/v1/query?query=process_resident_memory_bytes"
curl -s "http://prometheus:9090/api/v1/query?query=go_goroutines"
```

#### Security Issues
```bash
# Check Falco alerts
kubectl logs -n security -l app=falco --tail=50

# Check authentication logs
kubectl logs -f deployment/anomaly-detection-green -n production | grep -i "auth\|login\|failed"

# Check for suspicious activity
kubectl exec -it deployment/anomaly-detection-green -n production -- \
  tail -f /var/log/audit.log | grep -i "denied\|failed\|unauthorized"
```

### 5. Immediate Response Actions

#### Service Restoration Priority
1. **Stop the bleeding** - Prevent further damage
2. **Restore service** - Get users back online
3. **Investigate root cause** - Understand what happened
4. **Implement permanent fix** - Prevent recurrence

#### Common Response Actions

##### Rollback Deployment
```bash
# Quick rollback to previous version
kubectl patch service anomaly-detection-active -n production \
  -p '{"spec":{"selector":{"version":"blue"}}}'

# Verify rollback
kubectl get endpoints anomaly-detection-active -n production

# Scale up previous version if needed
kubectl scale deployment anomaly-detection-blue --replicas=5 -n production
```

##### Scale Resources
```bash
# Horizontal scaling
kubectl scale deployment anomaly-detection-green --replicas=10 -n production

# Vertical scaling (update resource limits)
kubectl patch deployment anomaly-detection-green -n production -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "anomaly-detection",
          "resources": {
            "requests": {"memory": "2Gi", "cpu": "1000m"},
            "limits": {"memory": "4Gi", "cpu": "2000m"}
          }
        }]
      }
    }
  }
}'
```

##### Database Issues
```bash
# Restart database connections
kubectl rollout restart deployment/anomaly-detection-green -n production

# Scale database (if using read replicas)
# aws rds create-db-instance-read-replica ...

# Switch to maintenance mode
kubectl patch configmap app-config -n production \
  -p '{"data":{"maintenance_mode":"true"}}'
```

##### Infrastructure Issues
```bash
# Drain and replace nodes
kubectl drain $NODE_NAME --ignore-daemonsets --delete-emptydir-data
kubectl delete node $NODE_NAME

# Update Auto Scaling Group
aws autoscaling set-desired-capacity \
  --auto-scaling-group-name $ASG_NAME \
  --desired-capacity 6
```

### 6. Communication Procedures

#### Internal Communication

##### Status Page Updates
```bash
# Update status page (example with Statuspage.io)
curl -X PATCH https://api.statuspage.io/v1/pages/$PAGE_ID/incidents/$INCIDENT_ID \
  -H "Authorization: OAuth $STATUSPAGE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "incident": {
      "status": "investigating",
      "body": "We are investigating reports of service degradation and will provide updates as we learn more."
    }
  }'
```

##### Internal Updates Template
```
🚨 INCIDENT UPDATE - [TIMESTAMP]

**Incident:** [Brief description]
**Severity:** [P0/P1/P2/P3]
**Status:** [Investigating/Identified/Monitoring/Resolved]
**Impact:** [Description of customer impact]
**ETA:** [Expected resolution time]

**Current Actions:**
- [Action 1]
- [Action 2]

**Next Update:** [Time]

**Incident Commander:** [Name]
```

#### External Communication

##### Customer Communication Template
```
Subject: [Action Required/Service Alert] - Anomaly Detection Platform

Dear Valued Customer,

We are currently experiencing [brief description of issue] with our Anomaly Detection Platform. This may affect [specific functionality/features].

**What we know:**
- Issue started at approximately [time]
- Impact: [description]
- Current status: [investigating/identified/monitoring]

**What we're doing:**
- [Action 1]
- [Action 2]

**What you can do:**
- [Any workarounds if available]
- [Contact information for support]

We will provide updates every [frequency] until resolved. We apologize for any inconvenience.

Best regards,
Platform Operations Team
```

### 7. Resolution and Recovery

#### Verification Steps
```bash
# Health checks
curl -f https://api.detection-platform.io/health/ready
curl -f https://api.detection-platform.io/health/live

# End-to-end testing
curl -X POST https://api.detection-platform.io/api/v1/detect \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TEST_TOKEN" \
  -d '{"data": [[1,2,3,4], [5,6,7,8]]}'

# Performance validation
curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,rate(http_request_duration_seconds_bucket[5m]))"

# Error rate check
curl -s "http://prometheus:9090/api/v1/query?query=rate(http_requests_total{status=~\"5..\"}[5m])"
```

#### Service Monitoring Period
- **P0/P1**: Monitor for 4 hours after resolution
- **P2**: Monitor for 2 hours after resolution
- **P3**: Monitor for 1 hour after resolution

### 8. Post-Incident Activities

#### Immediate Actions (Within 2 hours)
- [ ] Update status page to "Resolved"
- [ ] Send final customer communication
- [ ] Document timeline in incident ticket
- [ ] Preserve logs and evidence
- [ ] Create follow-up tickets for permanent fixes

#### Post-Incident Review (Within 48 hours)
- [ ] Schedule post-mortem meeting
- [ ] Gather timeline and evidence
- [ ] Identify root cause
- [ ] Document lessons learned
- [ ] Create action items for prevention

#### Post-Mortem Template
```markdown
# Post-Incident Review - [Date]

## Summary
[Brief description of what happened]

## Timeline
| Time | Action/Event |
|------|-------------|
| [HH:MM] | [Event] |
| [HH:MM] | [Event] |

## Root Cause
[Detailed analysis of what caused the incident]

## Impact
- **Duration:** [Total time]
- **Users Affected:** [Number/percentage]
- **Revenue Impact:** [If applicable]
- **SLA Impact:** [If applicable]

## What Went Well
- [Item 1]
- [Item 2]

## What Could Be Improved
- [Item 1]
- [Item 2]

## Action Items
| Action | Owner | Due Date | Priority |
|--------|-------|----------|----------|
| [Action] | [Name] | [Date] | [High/Medium/Low] |
```

## Incident Response Runbooks

### Service Outage
1. Check load balancer health
2. Verify pod status and logs
3. Check database connectivity
4. Verify external dependencies
5. Consider rollback if recent deployment
6. Scale resources if capacity issue

### Performance Degradation
1. Check resource utilization (CPU, memory, network)
2. Identify performance bottlenecks
3. Check database query performance
4. Verify caching effectiveness
5. Scale horizontally/vertically as needed
6. Implement circuit breakers if dependency issue

### Security Incident
1. **DO NOT** shut down systems (preserve evidence)
2. Isolate affected systems
3. Notify security team immediately
4. Document everything
5. Preserve logs and evidence
6. Follow legal/compliance requirements

### Database Issues
1. Check database connectivity
2. Verify query performance
3. Check connection pool status
4. Monitor transaction locks
5. Consider read replica failover
6. Prepare for point-in-time recovery if needed

## Contact Information

### On-Call Rotation
| Role | Primary | Secondary | Escalation |
|------|---------|-----------|------------|
| Incident Commander | [Name/Contact] | [Name/Contact] | [Manager] |
| Technical Lead | [Name/Contact] | [Name/Contact] | [Senior Engineer] |
| Infrastructure | [Name/Contact] | [Name/Contact] | [Infra Manager] |
| Security | [Name/Contact] | [Name/Contact] | [Security Manager] |

### External Contacts
- **AWS Support**: Case Priority [High/Urgent]
- **Third-party Vendors**: [Contact information]
- **Legal Team**: [Contact for compliance/legal issues]
- **PR Team**: [For public communications]

### Communication Channels
- **Primary**: Slack #incident-response
- **Secondary**: Conference bridge: [Number]
- **War Room**: [Physical/virtual location]
- **Status Page**: https://status.detection-platform.io

## Quick Reference Commands

### Health Checks
```bash
# Application health
curl -f https://api.detection-platform.io/health/ready

# Database health
kubectl exec -it deployment/anomaly-detection-green -n production -- \
  python -c "import psycopg2; print('DB OK')"

# Redis health
kubectl exec -it deployment/anomaly-detection-green -n production -- \
  redis-cli -h $REDIS_HOST ping
```

### System Status
```bash
# Pod status
kubectl get pods -n production | grep -v Running

# Node status
kubectl get nodes | grep -v Ready

# Recent events
kubectl get events --all-namespaces --sort-by='.lastTimestamp' | head -10
```

### Rollback Commands
```bash
# Immediate traffic switch
kubectl patch service anomaly-detection-active -n production \
  -p '{"spec":{"selector":{"version":"blue"}}}'

# Database rollback (if needed)
aws rds restore-db-instance-to-point-in-time \
  --source-db-instance-identifier $DB_ID \
  --target-db-instance-identifier $DB_ID-rollback \
  --restore-time $(date -d '1 hour ago' -u +%Y-%m-%dT%H:%M:%S.000Z)
```

---

**Playbook Version:** 1.0  
**Last Updated:** January 1, 2025  
**Next Review:** February 1, 2025

*This playbook should be regularly updated based on incident learnings and system changes.*