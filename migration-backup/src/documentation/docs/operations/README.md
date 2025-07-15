# Pynomaly Operational Runbooks

Comprehensive operational documentation for Pynomaly production environments.

## 📋 Overview

This directory contains operational runbooks, troubleshooting guides, and standard operating procedures (SOPs) for maintaining and operating Pynomaly in production environments.

## 🗂️ Runbook Categories

### 🚨 [Incident Response](incident-response/)
- [System Down](incident-response/system-down.md)
- [High CPU Usage](incident-response/high-cpu.md)
- [Memory Issues](incident-response/memory-issues.md)
- [Database Problems](incident-response/database-issues.md)
- [Network Connectivity](incident-response/network-issues.md)
- [Security Incidents](incident-response/security-incidents.md)

### 🔧 [Maintenance Procedures](maintenance/)
- [System Updates](maintenance/system-updates.md)
- [Database Maintenance](maintenance/database-maintenance.md)
- [Backup Procedures](maintenance/backup-procedures.md)
- [Log Rotation](maintenance/log-rotation.md)
- [Certificate Renewal](maintenance/certificate-renewal.md)

### 📊 [Monitoring & Alerting](monitoring/)
- [Alert Escalation](monitoring/alert-escalation.md)
- [Performance Monitoring](monitoring/performance-monitoring.md)
- [Health Checks](monitoring/health-checks.md)
- [Metrics Analysis](monitoring/metrics-analysis.md)

### 🚀 [Deployment & Release](deployment/)
- [Production Deployment](deployment/production-deployment.md)
- [Rollback Procedures](deployment/rollback-procedures.md)
- [Blue-Green Deployment](deployment/blue-green-deployment.md)
- [Hotfix Deployment](deployment/hotfix-deployment.md)

### 🔒 [Security Operations](security/)
- [Access Management](security/access-management.md)
- [Key Rotation](security/key-rotation.md)
- [Vulnerability Response](security/vulnerability-response.md)
- [Compliance Checks](security/compliance-checks.md)

### 💾 [Backup & Recovery](backup-recovery/)
- [Backup Verification](backup-recovery/backup-verification.md)
- [Disaster Recovery](backup-recovery/disaster-recovery.md)
- [Point-in-Time Recovery](backup-recovery/point-in-time-recovery.md)
- [Data Migration](backup-recovery/data-migration.md)

## 📞 Emergency Contacts

### Primary On-Call
- **Phone**: +1-XXX-XXX-XXXX
- **Email**: oncall@pynomaly.com
- **Slack**: #pynomaly-oncall

### Engineering Team
- **Lead Engineer**: engineer-lead@pynomaly.com
- **DevOps Team**: devops@pynomaly.com
- **Security Team**: security@pynomaly.com

### External Support
- **Cloud Provider**: AWS Support (Premium)
- **Database Support**: PostgreSQL Enterprise
- **Monitoring**: DataDog Support

## 🔍 Quick Reference

### System Status Dashboard
- **Primary**: https://status.pynomaly.com
- **Internal**: https://monitoring.pynomaly.internal
- **Grafana**: https://grafana.pynomaly.internal

### Key System Components
```
Production Infrastructure:
├── Load Balancer (AWS ALB)
├── Application Servers (3x EC2 instances)
├── Database (RDS PostgreSQL)
├── Cache (ElastiCache Redis)
├── Message Queue (AWS SQS)
├── Object Storage (S3)
└── Monitoring (CloudWatch + Grafana)
```

### Critical Thresholds
- **CPU Usage**: > 80% (Warning), > 95% (Critical)
- **Memory Usage**: > 85% (Warning), > 95% (Critical)
- **Disk Usage**: > 85% (Warning), > 95% (Critical)
- **Response Time**: > 500ms (Warning), > 1000ms (Critical)
- **Error Rate**: > 1% (Warning), > 5% (Critical)

## 🚨 Emergency Procedures

### Immediate Actions for Critical Issues

1. **System Down**:
   ```bash
   # Check system status
   ./scripts/monitoring/monitoring-status.sh
   
   # Check application health
   curl -f https://api.pynomaly.com/health
   
   # Review recent deployments
   git log --oneline -10
   ```

2. **High Load**:
   ```bash
   # Scale application instances
   aws autoscaling set-desired-capacity --auto-scaling-group-name pynomaly-asg --desired-capacity 6
   
   # Check resource usage
   htop
   iostat -x 1
   ```

3. **Database Issues**:
   ```bash
   # Check database connections
   psql -h prod-db.amazonaws.com -U pynomaly -c "SELECT * FROM pg_stat_activity;"
   
   # Check replication lag
   psql -h prod-db-replica.amazonaws.com -U pynomaly -c "SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) AS lag_seconds;"
   ```

## 📈 Performance Baselines

### Normal Operating Conditions
- **CPU**: 20-40% average
- **Memory**: 40-60% usage
- **Disk I/O**: < 80% utilization
- **Network**: < 70% bandwidth
- **Response Time**: < 200ms average
- **Error Rate**: < 0.1%

### Traffic Patterns
- **Peak Hours**: 9 AM - 5 PM EST (3x normal load)
- **Weekend Traffic**: 40% of weekday average
- **Holiday Traffic**: 20% of normal load

## 🔧 Common Commands

### System Administration
```bash
# Check system resources
htop
df -h
free -h

# View system logs
journalctl -u pynomaly -f
tail -f /var/log/pynomaly/application.log

# Check service status
systemctl status pynomaly
systemctl status nginx
systemctl status postgresql
```

### Application Management
```bash
# Restart application
sudo systemctl restart pynomaly

# View application logs
tail -f /var/log/pynomaly/app.log

# Check configuration
python -m pynomaly.config.validate

# Run health check
curl -f http://localhost:8000/health
```

### Database Operations
```bash
# Connect to database
psql -h localhost -U pynomaly -d pynomaly_prod

# Check database size
psql -c "SELECT pg_size_pretty(pg_database_size('pynomaly_prod'));"

# Check active connections
psql -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';"

# Backup database
pg_dump -h localhost -U pynomaly pynomaly_prod > backup_$(date +%Y%m%d_%H%M%S).sql
```

## 📚 Documentation Links

- [Architecture Overview](../architecture/README.md)
- [API Documentation](../api/README.md)
- [Configuration Guide](../configuration/README.md)
- [Troubleshooting Guide](../troubleshooting/README.md)
- [Security Guide](../security/README.md)

## 🔄 Change Management

### Pre-Change Checklist
- [ ] Change approved by engineering lead
- [ ] Backup created and verified
- [ ] Rollback plan documented
- [ ] Monitoring alerts configured
- [ ] Team notified of change window

### Post-Change Checklist
- [ ] Application functionality verified
- [ ] Performance metrics reviewed
- [ ] Error rates checked
- [ ] Monitoring alerts cleared
- [ ] Change documentation updated

## 📝 Incident Documentation

All incidents must be documented with:
- **Incident ID**: Generated automatically
- **Start Time**: When issue was first detected
- **End Time**: When issue was fully resolved
- **Impact**: User-facing impact description
- **Root Cause**: Technical root cause analysis
- **Resolution**: Steps taken to resolve
- **Prevention**: Measures to prevent recurrence

### Incident Severity Levels

- **Critical (P0)**: Complete system outage, data loss
- **High (P1)**: Major functionality unavailable, significant performance degradation
- **Medium (P2)**: Minor functionality issues, moderate performance impact
- **Low (P3)**: Cosmetic issues, minimal impact

## 📞 Escalation Matrix

| Severity | Initial Response | Escalation (30min) | Management (1hr) |
|----------|------------------|-------------------|------------------|
| P0       | On-call Engineer | Engineering Lead  | VP Engineering   |
| P1       | On-call Engineer | Engineering Lead  | Engineering Lead |
| P2       | On-call Engineer | Team Lead         | Team Lead        |
| P3       | Assigned Engineer| Team Lead         | N/A              |

---

**Last Updated**: 2024-12-10  
**Next Review**: 2025-01-10  
**Document Owner**: DevOps Team