# Pynomaly Incident Response Runbook

## Overview
This runbook provides standardized procedures for responding to incidents in the Pynomaly anomaly detection platform.

## Incident Classification

### Severity Levels

#### P0 - Critical
- Complete system outage
- Data loss or corruption
- Security breach
- **Response Time**: 15 minutes
- **Resolution Time**: 2 hours

#### P1 - High
- Major feature unavailable
- Significant performance degradation
- Partial system outage
- **Response Time**: 30 minutes
- **Resolution Time**: 4 hours

#### P2 - Medium
- Minor feature issues
- Moderate performance impact
- Non-critical alerts
- **Response Time**: 2 hours
- **Resolution Time**: 24 hours

#### P3 - Low
- Cosmetic issues
- Documentation problems
- Non-urgent improvements
- **Response Time**: 1 business day
- **Resolution Time**: 1 week

## Incident Response Process

### Step 1: Detection and Alerting

#### Automated Detection
- Prometheus alerts
- Grafana dashboards
- Application health checks
- Third-party monitoring services

#### Manual Detection
- User reports
- Support tickets
- Manual system checks
- Routine monitoring

### Step 2: Initial Response

#### Immediate Actions (0-5 minutes)
1. **Acknowledge Alert**
   - Acknowledge in PagerDuty
   - Update Slack channel (#pynomaly-alerts)
   - Assign incident commander

2. **Assess Severity**
   - Determine incident severity level
   - Escalate if necessary
   - Document initial assessment

3. **Form Response Team**
   - Incident Commander
   - Technical Lead
   - Subject Matter Expert
   - Communications Lead (for P0/P1)

### Step 3: Investigation and Diagnosis

#### System Health Assessment
```bash
# Check overall system status
kubectl get pods --all-namespaces
kubectl get nodes
kubectl top nodes

# Check application health
curl -f https://pynomaly.com/health
kubectl logs -n pynomaly-prod deployment/pynomaly-prod-app --tail=100

# Check database status
kubectl exec -n pynomaly-prod postgres-0 -- pg_isready
kubectl exec -n pynomaly-prod postgres-0 -- psql -U postgres -c "SELECT version();"
```

#### Performance Metrics
```bash
# Check resource usage
kubectl top pods -n pynomaly-prod
kubectl describe nodes

# Check application metrics
curl -s https://pynomaly.com/metrics | grep -E "(response_time|error_rate|throughput)"
```

#### Log Analysis
```bash
# Application logs
kubectl logs -n pynomaly-prod deployment/pynomaly-prod-app --since=10m

# Database logs
kubectl logs -n pynomaly-prod postgres-0 --since=10m

# Ingress logs
kubectl logs -n pynomaly-prod deployment/nginx-ingress-controller --since=10m
```

### Step 4: Containment and Mitigation

#### Immediate Containment
1. **If Database Issue**
   ```bash
   # Scale down application to prevent further damage
   kubectl scale deployment/pynomaly-prod-app --replicas=0 -n pynomaly-prod
   
   # Isolate problematic database
   kubectl patch service postgres-prod-service -p '{"spec":{"selector":{"app":"maintenance"}}}'
   ```

2. **If Application Issue**
   ```bash
   # Rollback to previous version
   kubectl rollout undo deployment/pynomaly-prod-app -n pynomaly-prod
   
   # Or scale up healthy pods
   kubectl scale deployment/pynomaly-prod-app --replicas=5 -n pynomaly-prod
   ```

3. **If Infrastructure Issue**
   ```bash
   # Drain problematic nodes
   kubectl drain <node-name> --ignore-daemonsets --force
   
   # Reschedule pods
   kubectl delete pod <pod-name> -n pynomaly-prod
   ```

#### Traffic Management
```bash
# Implement rate limiting
kubectl apply -f k8s/emergency/rate-limit.yaml

# Redirect traffic to maintenance page
kubectl apply -f k8s/emergency/maintenance-mode.yaml

# Enable circuit breaker
kubectl patch deployment pynomaly-prod-app -p '{"spec":{"template":{"spec":{"containers":[{"name":"pynomaly-app","env":[{"name":"CIRCUIT_BREAKER_ENABLED","value":"true"}]}]}}}}'
```

### Step 5: Resolution and Recovery

#### System Recovery
1. **Verify Fix**
   ```bash
   # Health checks
   curl -f https://pynomaly.com/health
   kubectl get pods -n pynomaly-prod
   
   # Run smoke tests
   ./scripts/smoke_tests.sh
   ```

2. **Gradual Traffic Restoration**
   ```bash
   # Remove maintenance mode
   kubectl delete -f k8s/emergency/maintenance-mode.yaml
   
   # Gradually increase replicas
   kubectl scale deployment/pynomaly-prod-app --replicas=2 -n pynomaly-prod
   # Wait and monitor
   kubectl scale deployment/pynomaly-prod-app --replicas=5 -n pynomaly-prod
   ```

3. **Monitor Recovery**
   ```bash
   # Watch metrics
   watch -n 5 'curl -s https://pynomaly.com/metrics | grep -E "(response_time|error_rate)"'
   
   # Monitor logs
   kubectl logs -n pynomaly-prod deployment/pynomaly-prod-app -f
   ```

### Step 6: Post-Incident Review

#### Immediate Post-Incident (Within 24 hours)
1. **Document Timeline**
   - Incident start time
   - Detection time
   - Response time
   - Resolution time
   - All actions taken

2. **Assess Impact**
   - Affected users
   - Data integrity
   - Service availability
   - Financial impact

3. **Preliminary Root Cause**
   - Initial findings
   - Contributing factors
   - Immediate fixes applied

#### Formal Post-Incident Review (Within 1 week)
1. **Root Cause Analysis**
   - Detailed investigation
   - Contributing factors
   - Timeline analysis
   - Impact assessment

2. **Action Items**
   - Preventive measures
   - Process improvements
   - Monitoring enhancements
   - Training needs

## Common Incident Scenarios

### Scenario 1: Application Pods Crashing

#### Symptoms
- Pods in CrashLoopBackOff state
- Health check failures
- High error rates

#### Diagnosis
```bash
kubectl describe pod <pod-name> -n pynomaly-prod
kubectl logs <pod-name> -n pynomaly-prod --previous
kubectl get events -n pynomaly-prod --sort-by=.metadata.creationTimestamp
```

#### Resolution
1. Check resource limits
2. Verify configuration
3. Review recent changes
4. Rollback if necessary

### Scenario 2: Database Connection Issues

#### Symptoms
- Database connection errors
- Timeout errors
- Connection pool exhaustion

#### Diagnosis
```bash
kubectl exec -n pynomaly-prod postgres-0 -- psql -U postgres -c "SELECT * FROM pg_stat_activity;"
kubectl exec -n pynomaly-prod postgres-0 -- psql -U postgres -c "SELECT * FROM pg_stat_database;"
```

#### Resolution
1. Check database server status
2. Verify connection strings
3. Restart database if needed
4. Scale application pods

### Scenario 3: High Memory Usage

#### Symptoms
- OOMKilled pods
- High memory alerts
- Slow response times

#### Diagnosis
```bash
kubectl top pods -n pynomaly-prod
kubectl describe pod <pod-name> -n pynomaly-prod
kubectl exec -n pynomaly-prod <pod-name> -- ps aux --sort=-%mem
```

#### Resolution
1. Increase memory limits
2. Identify memory leaks
3. Optimize application
4. Scale horizontally

### Scenario 4: Network Connectivity Issues

#### Symptoms
- Intermittent failures
- Timeout errors
- DNS resolution failures

#### Diagnosis
```bash
kubectl exec -n pynomaly-prod <pod-name> -- nslookup kubernetes.default.svc.cluster.local
kubectl exec -n pynomaly-prod <pod-name> -- ping google.com
kubectl get networkpolicies -n pynomaly-prod
```

#### Resolution
1. Check network policies
2. Verify DNS configuration
3. Review firewall rules
4. Restart network components

## Emergency Contacts

### On-Call Rotation
- **Primary**: DevOps Engineer
- **Secondary**: Platform Engineer  
- **Escalation**: Engineering Manager
- **Executive**: CTO

### Contact Information
- **DevOps Team**: +1-555-0123
- **Platform Team**: +1-555-0124
- **Security Team**: +1-555-0125
- **Engineering Manager**: +1-555-0126

### Communication Channels
- **Slack**: #pynomaly-alerts
- **Email**: pynomaly-ops@company.com
- **PagerDuty**: https://company.pagerduty.com
- **Status Page**: https://status.pynomaly.com

## Tools and Resources

### Monitoring and Alerting
- **Prometheus**: http://prometheus.company.com
- **Grafana**: http://grafana.company.com
- **PagerDuty**: https://company.pagerduty.com
- **Slack**: #pynomaly-alerts

### Infrastructure
- **Kubernetes Dashboard**: https://k8s.company.com
- **AWS Console**: https://console.aws.amazon.com
- **Docker Registry**: https://registry.company.com

### Documentation
- **Runbooks**: /docs/runbooks/
- **Architecture**: /docs/architecture/
- **API Documentation**: https://api.pynomaly.com/docs

## Incident Templates

### Incident Report Template
```markdown
# Incident Report: [Brief Description]

## Summary
- **Incident ID**: INC-YYYY-MM-DD-###
- **Severity**: P0/P1/P2/P3
- **Start Time**: YYYY-MM-DD HH:MM UTC
- **End Time**: YYYY-MM-DD HH:MM UTC
- **Duration**: X hours Y minutes
- **Incident Commander**: [Name]

## Impact
- **Affected Services**: [List]
- **Affected Users**: [Number/Percentage]
- **Data Impact**: [Yes/No - Details]

## Timeline
- **HH:MM** - [Event description]
- **HH:MM** - [Event description]

## Root Cause
[Detailed description of root cause]

## Resolution
[Description of how the incident was resolved]

## Action Items
- [ ] [Action item 1] - [Owner] - [Due Date]
- [ ] [Action item 2] - [Owner] - [Due Date]

## Lessons Learned
[Key takeaways and improvements]
```

### Communication Template
```markdown
# Incident Update: [Brief Description]

**Status**: [INVESTIGATING/IDENTIFIED/MONITORING/RESOLVED]
**Severity**: P0/P1/P2/P3
**Impact**: [Brief impact description]
**ETA**: [Estimated time to resolution]

## Current Status
[Current status and actions being taken]

## Next Update
[When next update will be provided]

## Contact
For questions: #pynomaly-alerts or pynomaly-ops@company.com
```

## Training and Drills

### Monthly Incident Response Drills
- **Tabletop Exercises**: Practice incident response procedures
- **Failure Injection**: Test system resilience
- **Communication Drills**: Practice stakeholder communication
- **Recovery Drills**: Test backup and recovery procedures

### Training Requirements
- **New Team Members**: Complete incident response training within 30 days
- **Annual Refresh**: All team members complete annual training
- **Specialized Training**: Role-specific training for incident commanders

## Continuous Improvement

### Metrics to Track
- **MTTD** (Mean Time To Detect): How quickly incidents are detected
- **MTTR** (Mean Time To Resolve): How quickly incidents are resolved
- **Incident Frequency**: Number of incidents per time period
- **Severity Distribution**: Breakdown of incidents by severity

### Review Process
- **Weekly**: Review open incidents and action items
- **Monthly**: Analyze incident trends and metrics
- **Quarterly**: Review and update runbooks and procedures
- **Annually**: Comprehensive review of incident response program

## References
- [Kubernetes Troubleshooting Guide](https://kubernetes.io/docs/tasks/debug-application-cluster/)
- [Prometheus Alerting Rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/)
- [Site Reliability Engineering Handbook](https://landing.google.com/sre/books/)
- [Incident Response Best Practices](https://www.pagerduty.com/resources/learn/incident-response/)