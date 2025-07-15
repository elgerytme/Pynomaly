# Pynomaly Disaster Recovery Runbook

## Overview
This runbook provides comprehensive procedures for disaster recovery, business continuity, and system restoration for the Pynomaly anomaly detection platform.

## Disaster Recovery Objectives

### Recovery Time Objective (RTO)
- **Critical Services**: 2 hours
- **Non-Critical Services**: 8 hours
- **Full System Recovery**: 24 hours

### Recovery Point Objective (RPO)
- **Database**: 15 minutes
- **Application State**: 1 hour
- **Configuration**: 24 hours

## Disaster Scenarios

### 1. Database Failure
**Impact**: Complete loss of data access
**Probability**: Medium
**Detection**: Database health checks, application errors

### 2. Application Server Failure
**Impact**: Service unavailability
**Probability**: High
**Detection**: Health checks, monitoring alerts

### 3. Network Partition
**Impact**: Service isolation
**Probability**: Medium
**Detection**: Network monitoring, connectivity tests

### 4. Data Center Failure
**Impact**: Complete service outage
**Probability**: Low
**Detection**: Infrastructure monitoring, external probes

### 5. Security Breach
**Impact**: Data compromise, service disruption
**Probability**: Low
**Detection**: Security monitoring, intrusion detection

## Backup Strategy

### Backup Types

#### 1. Database Backups
- **Full Backup**: Daily at 2:00 AM UTC
- **Incremental Backup**: Every 15 minutes
- **Transaction Log Backup**: Continuous

#### 2. Application Backups
- **Configuration Backup**: Daily
- **Code Backup**: Git repository (multiple locations)
- **Log Backup**: Continuous to centralized logging

#### 3. Infrastructure Backups
- **Kubernetes Configuration**: Daily
- **Container Images**: Multi-region registry
- **Infrastructure as Code**: Git repository

### Backup Locations
- **Primary**: AWS S3 (us-west-2)
- **Secondary**: AWS S3 (us-east-1)
- **Tertiary**: Google Cloud Storage (us-central1)

### Backup Retention
- **Daily Backups**: 30 days
- **Weekly Backups**: 12 weeks
- **Monthly Backups**: 12 months
- **Yearly Backups**: 7 years

## Recovery Procedures

### Database Recovery

#### Scenario 1: Database Corruption
1. **Assess Damage**
   ```bash
   kubectl exec -n pynomaly-prod postgres-0 -- pg_dump --schema-only pynomaly > schema_check.sql
   kubectl exec -n pynomaly-prod postgres-0 -- psql -U postgres -c "SELECT pg_database_size('pynomaly');"
   ```

2. **Stop Application**
   ```bash
   kubectl scale deployment pynomaly-prod-app --replicas=0 -n pynomaly-prod
   ```

3. **Restore Database**
   ```bash
   # Find latest backup
   aws s3 ls s3://pynomaly-backups/database/ --recursive | grep "$(date +%Y-%m-%d)"
   
   # Download backup
   aws s3 cp s3://pynomaly-backups/database/pynomaly_backup_20231201_140000.sql ./
   
   # Restore database
   kubectl exec -i -n pynomaly-prod postgres-0 -- psql -U postgres -d pynomaly < pynomaly_backup_20231201_140000.sql
   ```

4. **Verify Restoration**
   ```bash
   kubectl exec -n pynomaly-prod postgres-0 -- psql -U postgres -c "SELECT COUNT(*) FROM users;"
   kubectl exec -n pynomaly-prod postgres-0 -- psql -U postgres -c "SELECT version();"
   ```

5. **Restart Application**
   ```bash
   kubectl scale deployment pynomaly-prod-app --replicas=3 -n pynomaly-prod
   kubectl rollout status deployment pynomaly-prod-app -n pynomaly-prod
   ```

#### Scenario 2: Point-in-Time Recovery
1. **Identify Recovery Point**
   ```bash
   # Check transaction logs
   kubectl exec -n pynomaly-prod postgres-0 -- psql -U postgres -c "SELECT * FROM pg_stat_archiver;"
   ```

2. **Restore to Point in Time**
   ```bash
   # Create recovery configuration
   kubectl exec -n pynomaly-prod postgres-0 -- bash -c "
   echo \"restore_command = 'cp /archive/%f %p'\" >> /var/lib/postgresql/data/recovery.conf
   echo \"recovery_target_time = '2023-12-01 14:30:00'\" >> /var/lib/postgresql/data/recovery.conf
   "
   
   # Restart database
   kubectl delete pod postgres-0 -n pynomaly-prod
   kubectl wait --for=condition=ready pod postgres-0 -n pynomaly-prod
   ```

### Application Recovery

#### Scenario 1: Application Deployment Failure
1. **Rollback to Previous Version**
   ```bash
   kubectl rollout undo deployment pynomaly-prod-app -n pynomaly-prod
   kubectl rollout status deployment pynomaly-prod-app -n pynomaly-prod
   ```

2. **Verify Rollback**
   ```bash
   kubectl get pods -n pynomaly-prod
   curl -f https://pynomaly.com/health
   ```

#### Scenario 2: Configuration Corruption
1. **Restore Configuration**
   ```bash
   # Download configuration backup
   aws s3 cp s3://pynomaly-backups/config/configmap_20231201.yaml ./
   
   # Apply configuration
   kubectl apply -f configmap_20231201.yaml
   ```

2. **Restart Application**
   ```bash
   kubectl rollout restart deployment pynomaly-prod-app -n pynomaly-prod
   ```

### Infrastructure Recovery

#### Scenario 1: Kubernetes Cluster Failure
1. **Assess Cluster State**
   ```bash
   kubectl cluster-info
   kubectl get nodes
   kubectl get pods --all-namespaces
   ```

2. **Restore Cluster Configuration**
   ```bash
   # Download cluster backup
   aws s3 cp s3://pynomaly-backups/k8s/cluster_backup_20231201.tar.gz ./
   
   # Extract and apply
   tar -xzf cluster_backup_20231201.tar.gz
   kubectl apply -f cluster_backup/
   ```

3. **Verify Cluster Recovery**
   ```bash
   kubectl get all -n pynomaly-prod
   kubectl get pvc -n pynomaly-prod
   ```

#### Scenario 2: Node Failure
1. **Drain Failed Node**
   ```bash
   kubectl drain <node-name> --ignore-daemonsets --force
   kubectl delete node <node-name>
   ```

2. **Add New Node**
   ```bash
   # Add new node to cluster (AWS specific)
   aws eks update-nodegroup-config --cluster-name pynomaly-prod --nodegroup-name workers --scaling-config minSize=3,maxSize=10,desiredSize=4
   ```

3. **Verify Node Addition**
   ```bash
   kubectl get nodes
   kubectl get pods -n pynomaly-prod -o wide
   ```

## Network Recovery

### Network Partition Recovery
1. **Identify Network Issues**
   ```bash
   kubectl exec -n pynomaly-prod <pod-name> -- ping <target-ip>
   kubectl exec -n pynomaly-prod <pod-name> -- nslookup <service-name>
   ```

2. **Check Network Policies**
   ```bash
   kubectl get networkpolicies -n pynomaly-prod
   kubectl describe networkpolicy <policy-name> -n pynomaly-prod
   ```

3. **Restore Network Connectivity**
   ```bash
   # Remove problematic network policies
   kubectl delete networkpolicy <policy-name> -n pynomaly-prod
   
   # Restart network components
   kubectl rollout restart daemonset kube-proxy -n kube-system
   ```

### DNS Issues
1. **Check DNS Resolution**
   ```bash
   kubectl exec -n pynomaly-prod <pod-name> -- nslookup kubernetes.default.svc.cluster.local
   ```

2. **Restart DNS Services**
   ```bash
   kubectl rollout restart deployment coredns -n kube-system
   ```

## Security Incident Response

### Security Breach Recovery
1. **Immediate Containment**
   ```bash
   # Isolate affected pods
   kubectl apply -f - <<EOF
   apiVersion: networking.k8s.io/v1
   kind: NetworkPolicy
   metadata:
     name: isolate-compromised
     namespace: pynomaly-prod
   spec:
     podSelector:
       matchLabels:
         app: pynomaly
     policyTypes:
     - Ingress
     - Egress
   EOF
   ```

2. **Assess Damage**
   ```bash
   # Check for unauthorized access
   kubectl logs -n pynomaly-prod deployment/pynomaly-prod-app | grep -i "unauthorized\|breach\|attack"
   
   # Check database for unauthorized changes
   kubectl exec -n pynomaly-prod postgres-0 -- psql -U postgres -c "SELECT * FROM audit_log WHERE timestamp > NOW() - INTERVAL '1 hour';"
   ```

3. **Restore from Clean Backup**
   ```bash
   # Use backup from before incident
   aws s3 cp s3://pynomaly-backups/database/pynomaly_backup_20231201_120000.sql ./
   kubectl exec -i -n pynomaly-prod postgres-0 -- psql -U postgres -d pynomaly < pynomaly_backup_20231201_120000.sql
   ```

4. **Implement Security Measures**
   ```bash
   # Update all secrets
   kubectl delete secret pynomaly-prod-secrets -n pynomaly-prod
   kubectl create secret generic pynomaly-prod-secrets --from-literal=DB_PASSWORD=new_password
   
   # Apply security patches
   kubectl apply -f k8s/security/hardened-deployment.yaml
   ```

## Testing and Validation

### Disaster Recovery Testing Schedule
- **Monthly**: Database backup/restore testing
- **Quarterly**: Application recovery testing
- **Semi-annually**: Full disaster recovery exercise
- **Annually**: Cross-region failover testing

### Automated Testing
```bash
# Run disaster recovery tests
python scripts/disaster_recovery_test.py --config config/dr_config.yaml

# Validate backup integrity
python scripts/validate_backups.py --backup-type database --date 2023-12-01

# Test recovery procedures
python scripts/test_recovery.py --scenario database_failure --dry-run
```

### Manual Testing Procedures

#### Database Recovery Test
1. **Create Test Database**
   ```bash
   kubectl exec -n pynomaly-staging postgres-0 -- createdb test_recovery
   ```

2. **Populate Test Data**
   ```bash
   kubectl exec -n pynomaly-staging postgres-0 -- psql -U postgres -d test_recovery -c "
   CREATE TABLE test_data (id SERIAL PRIMARY KEY, data VARCHAR(100));
   INSERT INTO test_data (data) VALUES ('test1'), ('test2'), ('test3');
   "
   ```

3. **Create Backup**
   ```bash
   kubectl exec -n pynomaly-staging postgres-0 -- pg_dump -U postgres test_recovery > test_backup.sql
   ```

4. **Simulate Failure**
   ```bash
   kubectl exec -n pynomaly-staging postgres-0 -- dropdb test_recovery
   ```

5. **Restore from Backup**
   ```bash
   kubectl exec -n pynomaly-staging postgres-0 -- createdb test_recovery
   kubectl exec -i -n pynomaly-staging postgres-0 -- psql -U postgres -d test_recovery < test_backup.sql
   ```

6. **Verify Recovery**
   ```bash
   kubectl exec -n pynomaly-staging postgres-0 -- psql -U postgres -d test_recovery -c "SELECT COUNT(*) FROM test_data;"
   ```

## Monitoring and Alerting

### Recovery Monitoring
- **Database Recovery**: Monitor replication lag, backup success
- **Application Recovery**: Monitor pod status, health checks
- **Network Recovery**: Monitor connectivity, DNS resolution
- **Storage Recovery**: Monitor disk usage, backup completion

### Alert Conditions
- **Backup Failure**: Immediate alert
- **High Recovery Time**: Alert if RTO exceeded
- **Data Loss**: Critical alert for RPO violation
- **Security Incident**: Immediate escalation

### Metrics to Track
- **Backup Success Rate**: Should be 100%
- **Recovery Time**: Should meet RTO objectives
- **Data Loss**: Should meet RPO objectives
- **Test Success Rate**: Should be >95%

## Communication Plan

### Internal Communication
- **Incident Commander**: Coordinates recovery efforts
- **Technical Team**: Executes recovery procedures
- **Management**: Receives status updates
- **Legal/Compliance**: Notified of data incidents

### External Communication
- **Customers**: Service status updates
- **Partners**: API availability notifications
- **Regulators**: Compliance incident reporting
- **Media**: Public relations as needed

### Communication Channels
- **Slack**: #incident-response
- **Email**: disaster-recovery@company.com
- **Phone**: Emergency contact list
- **Status Page**: https://status.pynomaly.com

## Post-Incident Procedures

### Immediate Post-Incident (0-24 hours)
1. **Verify Full Recovery**
   - All systems operational
   - Data integrity confirmed
   - Performance metrics normal

2. **Document Timeline**
   - Incident start time
   - Detection time
   - Response actions
   - Recovery completion

3. **Assess Impact**
   - Service availability
   - Data loss (if any)
   - Customer impact
   - Financial impact

### Short-term Post-Incident (1-7 days)
1. **Conduct Post-Incident Review**
   - Root cause analysis
   - Timeline analysis
   - Response effectiveness
   - Improvement opportunities

2. **Update Procedures**
   - Refine recovery procedures
   - Update documentation
   - Improve monitoring
   - Enhance automation

3. **Communicate Lessons Learned**
   - Share findings with team
   - Update training materials
   - Inform stakeholders
   - Document best practices

### Long-term Post-Incident (1-4 weeks)
1. **Implement Improvements**
   - Infrastructure changes
   - Process improvements
   - Tool enhancements
   - Training updates

2. **Validate Changes**
   - Test new procedures
   - Verify improvements
   - Update documentation
   - Train team members

## Training and Preparedness

### Team Training Requirements
- **Disaster Recovery Procedures**: Quarterly training
- **Backup and Restore**: Monthly hands-on practice
- **Incident Response**: Bi-annual simulation
- **Security Response**: Annual security drills

### Training Materials
- **Runbooks**: Detailed procedures
- **Video Tutorials**: Step-by-step guides
- **Simulation Exercises**: Practice scenarios
- **Documentation**: Reference materials

### Certification Requirements
- **Disaster Recovery Specialist**: Lead responders
- **Backup Administrator**: Backup team members
- **Security Responder**: Security team members
- **Incident Commander**: Senior team members

## Continuous Improvement

### Regular Reviews
- **Monthly**: Review backup success rates
- **Quarterly**: Review recovery procedures
- **Semi-annually**: Full disaster recovery plan review
- **Annually**: Complete plan overhaul

### Improvement Areas
- **Automation**: Increase recovery automation
- **Monitoring**: Enhance detection capabilities
- **Documentation**: Improve procedure clarity
- **Training**: Enhance team preparedness

### Metrics and KPIs
- **Recovery Time**: Actual vs. target RTO
- **Recovery Point**: Actual vs. target RPO
- **Test Success Rate**: DR test success percentage
- **Mean Time to Recovery**: Average recovery time

## Contact Information

### Disaster Recovery Team
- **DR Coordinator**: dr-coordinator@company.com
- **Database Administrator**: dba@company.com
- **Security Team**: security@company.com
- **Infrastructure Team**: infra@company.com

### Emergency Contacts
- **Primary On-Call**: +1-555-0123
- **Secondary On-Call**: +1-555-0124
- **Manager Escalation**: +1-555-0125
- **Executive Escalation**: +1-555-0126

### Vendor Support
- **AWS Support**: Enterprise support case
- **Kubernetes Support**: Support subscription
- **Database Support**: PostgreSQL support
- **Security Support**: Security vendor support

## References
- [Business Continuity Plan](business_continuity_plan.md)
- [Incident Response Runbook](incident_response_runbook.md)
- [Security Incident Response](security_incident_response.md)
- [Backup and Recovery Procedures](backup_recovery_procedures.md)
- [AWS Disaster Recovery Guide](https://docs.aws.amazon.com/whitepapers/latest/disaster-recovery-workloads-on-aws/disaster-recovery-workloads-on-aws.html)
- [Kubernetes Disaster Recovery](https://kubernetes.io/docs/tasks/administer-cluster/configure-upgrade-etcd/#backing-up-an-etcd-cluster)