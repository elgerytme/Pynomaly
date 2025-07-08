# Comprehensive Disaster Recovery Plan
# Pynomaly Production System

## Overview

This document outlines the comprehensive disaster recovery (DR) plan for the Pynomaly production system. The plan is designed to ensure business continuity with minimal downtime and data loss.

### Recovery Objectives

- **Recovery Time Objective (RTO)**: 30 minutes
- **Recovery Point Objective (RPO)**: 5 minutes
- **Maximum Acceptable Outage**: 4 hours
- **Data Loss Tolerance**: < 5 minutes of data

## Architecture Overview

### Primary Production Environment
- **Region**: us-west-2 (Oregon)
- **Availability Zones**: us-west-2a, us-west-2b, us-west-2c
- **Kubernetes Cluster**: pynomaly-production
- **Database**: Amazon RDS PostgreSQL Multi-AZ
- **Cache**: Amazon ElastiCache Redis
- **Storage**: Amazon EFS, Amazon S3

### Disaster Recovery Environment
- **Region**: us-east-1 (N. Virginia)
- **Availability Zones**: us-east-1a, us-east-1b, us-east-1c
- **Kubernetes Cluster**: pynomaly-dr
- **Database**: RDS Cross-Region Read Replica
- **Cache**: ElastiCache Redis (warm standby)
- **Storage**: S3 Cross-Region Replication

## Backup Strategy

### Database Backups

#### Automated Backups
```yaml
# RDS Automated Backup Configuration
BackupRetentionPeriod: 30 days
BackupWindow: "03:00-04:00 UTC"
MaintenanceWindow: "sun:04:00-sun:05:00 UTC"
DeletionProtection: true
MultiAZ: true
```

#### Point-in-Time Recovery
- **Retention**: 30 days
- **Granularity**: 1 second
- **Cross-Region**: Automated snapshots replicated to DR region

#### Manual Snapshots
```bash
# Create manual snapshot before major deployments
aws rds create-db-snapshot \
  --db-instance-identifier pynomaly-prod \
  --db-snapshot-identifier pynomaly-$(date +%Y%m%d-%H%M%S) \
  --tags Key=Environment,Value=production Key=Purpose,Value=pre-deployment
```

### Application Data Backups

#### File System Backups (EFS)
```yaml
# AWS Backup Plan for EFS
BackupPlan:
  BackupPlanName: pynomaly-efs-backup
  BackupPlanRule:
    - RuleName: DailyBackups
      TargetBackupVault: pynomaly-backup-vault
      ScheduleExpression: "cron(0 2 * * ? *)"
      Lifecycle:
        DeleteAfterDays: 90
        MoveToColdStorageAfterDays: 30
```

#### Model Artifacts Backup
```bash
#!/bin/bash
# Backup trained models to S3
aws s3 sync /app/storage/models/ s3://pynomaly-model-backups/$(date +%Y%m%d)/ \
  --delete \
  --storage-class STANDARD_IA \
  --metadata backup-date=$(date -Iseconds)
```

#### Configuration Backups
```bash
#!/bin/bash
# Backup Kubernetes configurations
kubectl get all,configmaps,secrets -n pynomaly-production -o yaml > \
  k8s-backup-$(date +%Y%m%d-%H%M%S).yaml

# Upload to S3
aws s3 cp k8s-backup-*.yaml s3://pynomaly-config-backups/kubernetes/
```

### Log Backups
```yaml
# CloudWatch Logs Retention
LogGroups:
  - LogGroupName: /aws/eks/pynomaly-production/cluster
    RetentionInDays: 365
  - LogGroupName: /aws/rds/instance/pynomaly-prod/postgresql
    RetentionInDays: 90
  - LogGroupName: /pynomaly/application
    RetentionInDays: 90
```

## Disaster Scenarios and Response Procedures

### Scenario 1: Single AZ Failure

**Impact**: Partial service degradation
**RTO**: 5 minutes
**Automated Response**: Yes

#### Detection
- CloudWatch alarms for AZ-specific metrics
- Kubernetes health checks fail for nodes in affected AZ
- Load balancer health checks detect unhealthy targets

#### Automatic Response
1. **Kubernetes Auto-Recovery**
   ```yaml
   # Pod disruption budgets ensure availability
   apiVersion: policy/v1
   kind: PodDisruptionBudget
   metadata:
     name: pynomaly-api-pdb
   spec:
     minAvailable: 2
     selector:
       matchLabels:
         app: pynomaly-api
   ```

2. **Database Failover**
   - RDS Multi-AZ automatically fails over to standby
   - Failover typically completes in 60-120 seconds

3. **Load Balancer Adjustment**
   - ALB/NLB automatically routes traffic away from failed AZ
   - Health check interval: 30 seconds

#### Manual Verification
```bash
# Check cluster node status
kubectl get nodes -o wide

# Verify pod distribution
kubectl get pods -n pynomaly-production -o wide

# Check RDS status
aws rds describe-db-instances --db-instance-identifier pynomaly-prod
```

### Scenario 2: Complete Regional Failure

**Impact**: Complete service outage
**RTO**: 30 minutes
**Automated Response**: Partial

#### Detection
- Multi-AZ health check failures
- Regional CloudWatch metrics unavailable
- External monitoring alerts from different region

#### Response Procedure

1. **Assess Scope** (2 minutes)
   ```bash
   # Check AWS Service Health Dashboard
   curl -s https://status.aws.amazon.com/data.json | jq '.page.status'
   
   # Verify cross-region connectivity
   aws ec2 describe-regions --region us-east-1
   ```

2. **Activate DR Site** (5 minutes)
   ```bash
   # Switch to DR region
   export AWS_DEFAULT_REGION=us-east-1
   
   # Update kubeconfig for DR cluster
   aws eks update-kubeconfig --region us-east-1 --name pynomaly-dr
   ```

3. **Promote Read Replica** (5 minutes)
   ```bash
   # Promote read replica to master
   aws rds promote-read-replica \
     --db-instance-identifier pynomaly-dr-replica \
     --backup-retention-period 30
   ```

4. **Deploy Application** (10 minutes)
   ```bash
   # Deploy to DR environment
   kubectl apply -f deploy/kubernetes/dr-deployment.yaml
   
   # Update DNS to point to DR region
   aws route53 change-resource-record-sets \
     --hosted-zone-id Z123456789 \
     --change-batch file://dns-failover.json
   ```

5. **Verify Service** (5 minutes)
   ```bash
   # Health checks
   curl -f https://api.pynomaly.io/health
   
   # Functional tests
   python scripts/testing/test_dr_functionality.py
   ```

6. **Notify Stakeholders** (3 minutes)
   ```bash
   # Send notifications
   curl -X POST $SLACK_WEBHOOK \
     -d '{"text":"🔴 DR Activated: Service restored in us-east-1"}'
   ```

### Scenario 3: Database Corruption

**Impact**: Data integrity issues
**RTO**: 60 minutes
**Automated Response**: No

#### Detection
- Data validation checks fail
- Checksum mismatches
- Application errors indicating data corruption

#### Response Procedure

1. **Immediate Isolation** (5 minutes)
   ```bash
   # Stop all write operations
   kubectl scale deployment pynomaly-api --replicas=0
   kubectl scale deployment pynomaly-worker --replicas=0
   
   # Enable read-only mode
   aws rds modify-db-instance \
     --db-instance-identifier pynomaly-prod \
     --apply-immediately \
     --publicly-accessible false
   ```

2. **Assess Corruption Extent** (15 minutes)
   ```sql
   -- Check database integrity
   SELECT pg_database.datname, pg_database_size(pg_database.datname)
   FROM pg_database;
   
   -- Run corruption checks
   SELECT schemaname, tablename, attname, n_distinct, correlation
   FROM pg_stats WHERE schemaname = 'public';
   ```

3. **Restore from Backup** (30 minutes)
   ```bash
   # Identify last known good backup
   aws rds describe-db-snapshots \
     --db-instance-identifier pynomaly-prod \
     --snapshot-type automated \
     --max-items 10
   
   # Restore to new instance
   aws rds restore-db-instance-from-db-snapshot \
     --db-instance-identifier pynomaly-prod-restored \
     --db-snapshot-identifier rds:pynomaly-prod-2024-01-15-03-15
   ```

4. **Data Validation** (10 minutes)
   ```python
   # Run data integrity checks
   python scripts/validation/check_data_integrity.py \
     --database pynomaly-prod-restored \
     --full-scan
   ```

### Scenario 4: Security Breach

**Impact**: Potential data compromise
**RTO**: Variable (depends on breach scope)
**Automated Response**: Partial

#### Detection
- Security monitoring alerts
- Unusual access patterns
- Failed authentication attempts
- Data exfiltration indicators

#### Response Procedure

1. **Immediate Containment** (5 minutes)
   ```bash
   # Isolate affected systems
   kubectl delete networkpolicy --all -n pynomaly-production
   kubectl apply -f security/incident-isolation-policy.yaml
   
   # Revoke all active sessions
   kubectl delete secret pynomaly-jwt-secret
   kubectl create secret generic pynomaly-jwt-secret --from-literal=key=$(openssl rand -base64 32)
   ```

2. **Forensic Preservation** (15 minutes)
   ```bash
   # Create forensic snapshots
   aws rds create-db-snapshot \
     --db-instance-identifier pynomaly-prod \
     --db-snapshot-identifier forensic-$(date +%Y%m%d-%H%M%S)
   
   # Capture system state
   kubectl get events --all-namespaces > incident-events.log
   kubectl logs -n pynomaly-production --all-containers=true > incident-logs.log
   ```

3. **Assessment and Recovery** (Variable)
   - Engage incident response team
   - Coordinate with security team
   - Follow incident response playbook
   - Restore from clean backup if necessary

## Monitoring and Alerting

### DR Health Monitoring

```yaml
# CloudWatch Alarms for DR Readiness
alarms:
  - name: "DR-Database-Replica-Lag"
    metric: "DatabaseConnections"
    threshold: 300  # 5 minutes
    comparison: "GreaterThanThreshold"
    
  - name: "DR-Cross-Region-Replication"
    metric: "ReplicationLag"
    threshold: 600  # 10 minutes
    comparison: "GreaterThanThreshold"
    
  - name: "DR-Backup-Failure"
    metric: "BackupJobFailed"
    threshold: 1
    comparison: "GreaterThanOrEqualToThreshold"
```

### Automated DR Testing

```yaml
# Monthly DR drill automation
apiVersion: batch/v1
kind: CronJob
metadata:
  name: dr-drill
spec:
  schedule: "0 6 1 * *"  # First day of month at 6 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: dr-test
            image: pynomaly/dr-tester:latest
            command: ["python3", "/scripts/dr_drill.py"]
            env:
            - name: DR_REGION
              value: "us-east-1"
            - name: NOTIFICATION_WEBHOOK
              value: "https://hooks.slack.com/services/..."
```

## Communication Plan

### Stakeholder Notification Matrix

| Scenario | Primary | Secondary | Communication Method | Timeline |
|----------|---------|-----------|---------------------|----------|
| AZ Failure | Engineering | Product | Slack, Email | Immediate |
| Regional Failure | CEO, CTO | All Staff | All channels | <5 minutes |
| Data Corruption | DPO, Legal | Engineering | Secure channels | <15 minutes |
| Security Breach | CISO, Legal | Board | Encrypted email | <10 minutes |

### Communication Templates

#### Regional Failover Notification
```text
🔴 INCIDENT ALERT - Pynomaly Production

Status: Service disruption detected in primary region (us-west-2)
Action: Activating disaster recovery site (us-east-1)
ETA: Service restoration expected within 30 minutes
Impact: Temporary service unavailability

Next Update: In 15 minutes
Incident Commander: [Name]
War Room: #incident-response
```

#### Service Restoration Notification
```text
✅ INCIDENT RESOLVED - Pynomaly Production

Status: Service fully restored
Resolution: DR site activated successfully
Duration: XX minutes total outage
Impact: No data loss confirmed

Post-Incident: Full review scheduled within 24 hours
Lessons Learned: [Link to be provided]
```

## Testing and Validation

### Monthly DR Drills

#### Scope
1. **Database Failover Test**
   - Promote read replica
   - Validate data consistency
   - Measure failover time

2. **Application Recovery Test**
   - Deploy to DR region
   - Run functional tests
   - Validate performance

3. **DNS Failover Test**
   - Switch DNS records
   - Verify global propagation
   - Test SSL certificate validity

#### Success Criteria
- RTO < 30 minutes
- RPO < 5 minutes
- 100% functional test pass rate
- No data corruption
- All monitoring alerts working

### Quarterly Full DR Exercise

#### Scope
- Complete regional failover simulation
- Full application stack testing
- End-to-end user scenario validation
- Communication plan execution
- Documentation review and update

## Recovery Procedures

### Database Recovery Procedures

#### Point-in-Time Recovery
```bash
#!/bin/bash
# Restore database to specific point in time

RESTORE_TIME="2024-01-15T10:30:00.000Z"
SOURCE_DB="pynomaly-prod"
TARGET_DB="pynomaly-pitr-$(date +%Y%m%d)"

aws rds restore-db-instance-to-point-in-time \
  --source-db-instance-identifier $SOURCE_DB \
  --target-db-instance-identifier $TARGET_DB \
  --restore-time $RESTORE_TIME \
  --db-instance-class db.r5.xlarge \
  --multi-az \
  --publicly-accessible false \
  --storage-encrypted \
  --copy-tags-to-snapshot
```

#### Cross-Region Recovery
```bash
#!/bin/bash
# Restore database from cross-region snapshot

SNAPSHOT_ID="arn:aws:rds:us-west-2:123456789012:snapshot:pynomaly-prod-final"
TARGET_DB="pynomaly-xregion-restore"

aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier $TARGET_DB \
  --db-snapshot-identifier $SNAPSHOT_ID \
  --db-instance-class db.r5.xlarge \
  --multi-az \
  --storage-encrypted \
  --kms-key-id alias/pynomaly-encryption-key
```

### Application Recovery Procedures

#### Kubernetes Deployment Recovery
```bash
#!/bin/bash
# Restore Kubernetes deployment from backup

# Apply saved configurations
kubectl apply -f /backups/kubernetes/latest/

# Verify deployment status
kubectl rollout status deployment/pynomaly-api -n pynomaly-production

# Run health checks
python scripts/testing/test_deployment_health.py
```

#### Configuration Recovery
```bash
#!/bin/bash
# Restore application configuration

# Restore secrets from AWS Secrets Manager
aws secretsmanager get-secret-value \
  --secret-id pynomaly/production/database \
  --region us-east-1 | jq -r .SecretString | kubectl apply -f -

# Restore ConfigMaps
kubectl apply -f /backups/config/latest/configmaps.yaml
```

## Documentation and Runbooks

### Runbook Links
- [Database Failover Runbook](runbooks/database-failover.md)
- [Application Recovery Runbook](runbooks/application-recovery.md)
- [DNS Failover Runbook](runbooks/dns-failover.md)
- [Security Incident Response](runbooks/security-incident.md)

### Required Documentation Updates
- Monthly: Test results and procedure refinements
- Quarterly: Full plan review and architecture updates
- Annually: Complete disaster recovery strategy review

## Compliance and Audit

### Regulatory Requirements
- **SOC 2 Type II**: Documented recovery procedures and testing
- **ISO 27001**: Information security management
- **GDPR**: Data protection and breach notification

### Audit Trail
- All DR activities logged in CloudTrail
- Recovery procedures documented in incident management system
- Test results archived for compliance review

## Continuous Improvement

### Metrics and KPIs
- Mean Time to Detection (MTTD)
- Mean Time to Resolution (MTTR)
- Recovery Time Actual vs. RTO
- Data Loss Actual vs. RPO
- DR Drill Success Rate

### Review Schedule
- **Weekly**: Monitoring and alerting review
- **Monthly**: DR drill execution and analysis
- **Quarterly**: Full plan review and updates
- **Annually**: Strategic DR assessment and planning

---

**Document Version**: 1.0
**Last Updated**: January 2024
**Next Review**: March 2024
**Owner**: Infrastructure Team
**Approved By**: CTO, CISO