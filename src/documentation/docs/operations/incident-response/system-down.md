# System Down - Incident Response Runbook

## 🚨 Overview

This runbook provides step-by-step instructions for responding to complete system outages where Pynomaly is completely unavailable to users.

**Severity**: P0 (Critical)  
**Response Time**: Immediate (< 5 minutes)  
**Communication**: #pynomaly-incidents Slack channel

## 🔍 Detection

### Symptoms

- Health check endpoints returning HTTP 5xx errors or timing out
- User reports of "Service Unavailable" errors
- Monitoring alerts for service availability
- Load balancer showing all targets unhealthy

### Verification Commands

```bash
# Check external service availability
curl -f https://api.pynomaly.com/health
curl -f https://app.pynomaly.com/health

# Check load balancer status
aws elbv2 describe-target-health --target-group-arn arn:aws:elasticloadbalancing:...

# Check application server status
ansible production -m shell -a "systemctl is-active pynomaly"
```

## 🚀 Immediate Response (0-5 minutes)

### 1. Acknowledge and Communicate

```bash
# Post to incident channel
echo "🚨 SYSTEM DOWN - P0 Incident detected at $(date). Investigating..." | slack-cli -c pynomaly-incidents

# Update status page
curl -X POST https://api.statuspage.io/v1/pages/PAGE_ID/incidents \
  -H "Authorization: OAuth TOKEN" \
  -d "incident[name]=System Outage" \
  -d "incident[status]=investigating"
```

### 2. Quick Health Assessment

```bash
# Check all critical services
./scripts/monitoring/health-check-all.sh

# Check AWS service health
aws health describe-events --filter eventTypeCategories=issue,eventStatusCodes=open

# Check recent deployments
git log --oneline -5
kubectl get pods -n production
```

### 3. Immediate Triage

```bash
# Check application logs for errors
tail -100 /var/log/pynomaly/application.log | grep -i error

# Check database connectivity
psql -h prod-db.amazonaws.com -U pynomaly -c "SELECT 1;"

# Check Redis connectivity
redis-cli -h prod-redis.amazonaws.com ping
```

## 🔧 Diagnosis and Resolution (5-15 minutes)

### Application Server Issues

#### Check Process Status

```bash
# Check if application is running
sudo systemctl status pynomaly
ps aux | grep pynomaly

# Check resource usage
htop
df -h
free -h
```

#### Restart Application

```bash
# Restart application service
sudo systemctl restart pynomaly

# Check logs for startup errors
journalctl -u pynomaly -f

# Verify application started
curl http://localhost:8000/health
```

### Database Issues

#### Check Database Status

```bash
# Connect to database
psql -h prod-db.amazonaws.com -U pynomaly -d pynomaly_prod

# Check database locks
SELECT * FROM pg_locks WHERE NOT granted;

# Check active connections
SELECT count(*) FROM pg_stat_activity WHERE state = 'active';

# Check database size and space
SELECT pg_size_pretty(pg_database_size('pynomaly_prod'));
```

#### Database Recovery

```bash
# If database is down, check AWS RDS status
aws rds describe-db-instances --db-instance-identifier pynomaly-prod

# Force failover to read replica if needed
aws rds failover-db-cluster --db-cluster-identifier pynomaly-cluster

# Restart database connections
sudo systemctl restart pynomaly
```

### Load Balancer Issues

#### Check Load Balancer

```bash
# Check target group health
aws elbv2 describe-target-health --target-group-arn $TARGET_GROUP_ARN

# Check load balancer logs
aws logs filter-log-events --log-group-name /aws/applicationloadbalancer/pynomaly-alb

# Manually register healthy targets
aws elbv2 register-targets --target-group-arn $TARGET_GROUP_ARN --targets Id=i-1234567890abcdef0
```

### Infrastructure Issues

#### Check AWS Services

```bash
# Check EC2 instance status
aws ec2 describe-instance-status --instance-ids i-1234567890abcdef0

# Check security groups
aws ec2 describe-security-groups --group-ids sg-12345678

# Check network ACLs
aws ec2 describe-network-acls --filters "Name=association.subnet-id,Values=subnet-12345678"
```

#### Auto Scaling Response

```bash
# Scale up immediately if needed
aws autoscaling set-desired-capacity --auto-scaling-group-name pynomaly-asg --desired-capacity 6

# Check new instance status
aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names pynomaly-asg
```

## 🔄 Recovery Procedures

### Scenario 1: Application Crash

```bash
# Check application logs for crash reason
tail -500 /var/log/pynomaly/application.log | grep -E "(error|exception|fatal)"

# Check for memory issues
dmesg | grep -i "killed process"

# Restart with increased resources if needed
sudo systemctl edit pynomaly
# Add:
# [Service]
# MemoryLimit=4G
# CPUQuota=200%

sudo systemctl restart pynomaly
```

### Scenario 2: Database Connection Issues

```bash
# Check connection pool status
psql -c "SELECT count(*) FROM pg_stat_activity WHERE application_name='pynomaly';"

# Reset connection pool
sudo systemctl restart pynomaly

# Check for connection leaks
grep -i "connection" /var/log/pynomaly/application.log | tail -50
```

### Scenario 3: Disk Space Issues

```bash
# Check disk usage
df -h

# Clean up logs if needed
sudo find /var/log -name "*.log" -type f -mtime +7 -delete
sudo journalctl --vacuum-time=24h

# Clean up temporary files
sudo rm -rf /tmp/pynomaly_*
```

### Scenario 4: Network Connectivity

```bash
# Check network connectivity
ping 8.8.8.8
curl -I https://www.google.com

# Check DNS resolution
nslookup prod-db.amazonaws.com
dig prod-db.amazonaws.com

# Check security group rules
aws ec2 describe-security-groups --group-ids sg-12345678
```

## ✅ Verification Steps

### 1. Service Availability

```bash
# Check all health endpoints
curl -f https://api.pynomaly.com/health
curl -f https://app.pynomaly.com/health

# Run synthetic tests
./scripts/testing/smoke-test.sh

# Check user-facing functionality
./scripts/testing/user-journey-test.sh
```

### 2. Performance Verification

```bash
# Check response times
for i in {1..10}; do
  curl -w "%{time_total}\n" -o /dev/null -s https://api.pynomaly.com/health
done

# Check error rates
grep -c "5xx" /var/log/nginx/access.log | tail -100
```

### 3. Monitoring Recovery

```bash
# Check all monitoring alerts are cleared
./scripts/monitoring/check-alerts.sh

# Verify metrics are being collected
curl -s http://localhost:9090/api/v1/query?query=up | jq .
```

## 📊 Post-Incident Actions

### 1. Update Status Page

```bash
curl -X PATCH https://api.statuspage.io/v1/pages/PAGE_ID/incidents/INCIDENT_ID \
  -H "Authorization: OAuth TOKEN" \
  -d "incident[status]=resolved" \
  -d "incident[message]=System fully restored at $(date)"
```

### 2. Communication

```bash
# Notify stakeholders
echo "✅ System restored at $(date). All services operational." | slack-cli -c pynomaly-incidents

# Send customer communication
./scripts/notifications/send-incident-resolved.sh
```

### 3. Data Collection

```bash
# Collect logs for analysis
mkdir -p /tmp/incident-$(date +%Y%m%d%H%M)
cp /var/log/pynomaly/application.log /tmp/incident-$(date +%Y%m%d%H%M)/
cp /var/log/nginx/error.log /tmp/incident-$(date +%Y%m%d%H%M)/
journalctl -u pynomaly --since "1 hour ago" > /tmp/incident-$(date +%Y%m%d%H%M)/systemd.log

# Export monitoring data
curl "http://prometheus:9090/api/v1/query_range?query=up&start=$(date -d '1 hour ago' +%s)&end=$(date +%s)&step=60" > /tmp/incident-$(date +%Y%m%d%H%M)/metrics.json
```

## 📋 Post-Mortem Checklist

- [ ] Incident timeline documented
- [ ] Root cause identified
- [ ] Impact assessment completed
- [ ] Customer communication sent
- [ ] Action items created to prevent recurrence
- [ ] Monitoring and alerting improvements identified
- [ ] Runbook updates needed
- [ ] Team debrief scheduled

## 🔗 Related Runbooks

- [High CPU Usage](high-cpu.md)
- [Memory Issues](memory-issues.md)
- [Database Issues](database-issues.md)
- [Network Issues](network-issues.md)

## 📞 Escalation

- **15 minutes**: Escalate to Engineering Lead
- **30 minutes**: Escalate to VP Engineering
- **45 minutes**: Escalate to CTO
- **60 minutes**: Executive communication required

---

**Last Updated**: 2024-12-10  
**Next Review**: 2025-01-10  
**Document Owner**: DevOps Team
