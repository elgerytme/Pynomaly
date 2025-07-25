# MLOps Platform Production Deployment Guide

## 📋 **Overview**

This guide provides comprehensive instructions for deploying the MLOps platform to production environments. The platform uses a microservices architecture with Kubernetes orchestration, Istio service mesh, and comprehensive monitoring.

## 🏗️ **Architecture Overview**

### Core Components
- **API Gateway**: Istio Ingress Gateway with SSL termination
- **Application Services**: API Server, Model Server, Web UI
- **Data Layer**: PostgreSQL (primary), Redis (cache/sessions)
- **Monitoring Stack**: Prometheus, Grafana, AlertManager
- **Security**: Istio mTLS, OPA Gatekeeper, Falco runtime security

### Infrastructure Requirements
- **Kubernetes Cluster**: v1.24+ with at least 6 worker nodes
- **Node Specifications**: 8 CPU cores, 32GB RAM, 500GB storage per node
- **Network**: Load balancer, SSL certificates, DNS configuration
- **External Dependencies**: Object storage (S3), external database (optional)

---

## 🚀 **Pre-Deployment Checklist**

### Infrastructure Prerequisites
- [ ] Kubernetes cluster provisioned and accessible
- [ ] kubectl configured with cluster admin access
- [ ] Helm 3.x installed
- [ ] Istio 1.15+ installed and configured
- [ ] Cert-manager installed for SSL certificate management
- [ ] External secrets operator configured (optional)

### Security Requirements
- [ ] SSL certificates obtained for domain names
- [ ] Secrets management system configured (Vault/AWS Secrets Manager)
- [ ] Database credentials generated and stored securely
- [ ] API keys for external services obtained
- [ ] Network security groups/firewalls configured

### DNS Configuration
- [ ] `api.mlops-platform.com` → Load balancer IP
- [ ] `app.mlops-platform.com` → Load balancer IP  
- [ ] `monitoring.mlops-platform.com` → Load balancer IP

---

## 📦 **Deployment Steps**

### Step 1: Prepare Configuration Files

```bash
# Clone the repository
git clone https://github.com/company/mlops-platform.git
cd mlops-platform/infrastructure/production

# Copy and customize configuration templates
cp config/application.yml.template config/application.yml
cp config/secrets.yaml.template config/secrets.yaml

# Edit configuration files with production values
vi config/application.yml
vi config/secrets.yaml
```

### Step 2: Create Kubernetes Namespace and Base Resources

```bash
# Create production namespace
kubectl create namespace mlops-production

# Label namespace for Istio injection
kubectl label namespace mlops-production istio-injection=enabled

# Apply base configuration
kubectl apply -f config/kubernetes/namespace.yaml
kubectl apply -f config/kubernetes/rbac.yaml
```

### Step 3: Deploy Secrets and ConfigMaps

```bash
# Create secrets (use external secrets operator in production)
kubectl apply -f config/kubernetes/secrets.yaml

# Create configuration maps
kubectl apply -f config/kubernetes/configmaps.yaml

# Verify secrets are created
kubectl get secrets -n mlops-production
```

### Step 4: Deploy Data Layer

```bash
# Deploy PostgreSQL
kubectl apply -f config/kubernetes/postgres.yaml

# Wait for PostgreSQL to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n mlops-production --timeout=300s

# Deploy Redis instances
kubectl apply -f config/kubernetes/redis.yaml

# Verify data layer deployment
kubectl get pods -n mlops-production -l tier=database
```

### Step 5: Deploy Application Services

```bash
# Deploy API server
kubectl apply -f config/kubernetes/api-server.yaml

# Deploy model server
kubectl apply -f config/kubernetes/model-server.yaml

# Deploy web UI
kubectl apply -f config/kubernetes/web-ui.yaml

# Deploy background workers
kubectl apply -f config/kubernetes/workers.yaml

# Verify application deployment
kubectl get pods -n mlops-production -l tier=application
```

### Step 6: Configure Service Mesh

```bash
# Apply Istio gateway configuration
kubectl apply -f gateway/istio-gateway.yaml

# Apply virtual services and destination rules
kubectl apply -f gateway/virtual-services.yaml
kubectl apply -f gateway/destination-rules.yaml

# Verify Istio configuration
istioctl analyze -n mlops-production
```

### Step 7: Deploy Monitoring Stack

```bash
# Deploy Prometheus
kubectl apply -f monitoring/prometheus.yaml

# Deploy Grafana
kubectl apply -f monitoring/grafana.yaml

# Deploy AlertManager
kubectl apply -f monitoring/alertmanager.yaml

# Verify monitoring deployment
kubectl get pods -n mlops-production -l tier=monitoring
```

### Step 8: Apply Security Policies

```bash
# Apply network policies
kubectl apply -f security/network-policies.yaml

# Apply pod security policies
kubectl apply -f security/pod-security-policies.yaml

# Deploy OPA Gatekeeper policies
kubectl apply -f security/gatekeeper-policies.yaml

# Deploy Falco security monitoring
kubectl apply -f security/falco-config.yaml
```

### Step 9: Configure SSL and Ingress

```bash
# Apply cert-manager certificate requests
kubectl apply -f security/certificates.yaml

# Wait for certificates to be issued
kubectl wait --for=condition=ready certificate -n mlops-production --all --timeout=300s

# Verify SSL certificates
kubectl get certificates -n mlops-production
```

### Step 10: Deploy Autoscaling Configuration

```bash
# Apply horizontal pod autoscalers
kubectl apply -f config/kubernetes/hpa.yaml

# Apply vertical pod autoscalers (if enabled)
kubectl apply -f config/kubernetes/vpa.yaml

# Verify autoscaler configuration
kubectl get hpa -n mlops-production
```

---

## 🔍 **Post-Deployment Verification**

### Health Checks

```bash
# Check all pods are running
kubectl get pods -n mlops-production

# Verify services are accessible
kubectl get services -n mlops-production

# Check ingress status
kubectl get ingress -n mlops-production

# Test API endpoints
curl -k https://api.mlops-platform.com/health
curl -k https://app.mlops-platform.com/health
```

### Functional Testing

```bash
# Run deployment verification tests
cd tests/deployment
python verify_deployment.py --environment production

# Test authentication
curl -X POST https://api.mlops-platform.com/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'

# Test model prediction
curl -X POST https://api.mlops-platform.com/api/v1/models/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"data": [[1,2,3,4,5]]}'
```

### Performance Testing

```bash
# Run load tests
cd tests/performance
python load_test.py --target https://api.mlops-platform.com --concurrent 50 --duration 300

# Monitor resource usage
kubectl top nodes
kubectl top pods -n mlops-production
```

---

## 📊 **Monitoring and Alerting**

### Accessing Monitoring Dashboards

- **Grafana**: https://monitoring.mlops-platform.com/grafana
  - Username: `admin`
  - Password: Check secret `grafana-admin-password`

- **Prometheus**: https://monitoring.mlops-platform.com/prometheus
- **AlertManager**: https://monitoring.mlops-platform.com/alertmanager

### Key Metrics to Monitor

| Metric | Threshold | Description |
|--------|-----------|-------------|
| API Response Time | < 100ms p95 | API performance |
| Error Rate | < 1% | Service reliability |
| CPU Utilization | < 70% | Resource usage |
| Memory Utilization | < 80% | Memory consumption |
| Database Connections | < 80% of pool | DB connection health |
| Model Latency | < 50ms p95 | ML model performance |

### Alert Configuration

Critical alerts are sent to:
- **PagerDuty**: For immediate response
- **Slack**: `#mlops-critical` channel
- **Email**: `ops-team@company.com`

---

## 🔧 **Maintenance and Operations**

### Regular Maintenance Tasks

#### Daily
- [ ] Check system health dashboards
- [ ] Monitor alert notifications
- [ ] Review error logs for anomalies
- [ ] Verify backup completion

#### Weekly  
- [ ] Review resource utilization trends
- [ ] Update security patches (if available)
- [ ] Test disaster recovery procedures
- [ ] Review and update monitoring thresholds

#### Monthly
- [ ] Security audit and vulnerability assessment
- [ ] Performance optimization review
- [ ] Capacity planning assessment
- [ ] Update documentation

### Backup and Recovery

```bash
# Manual database backup
kubectl exec -n mlops-production postgres-0 -- pg_dump -U mlops mlops_prod > backup.sql

# Restore from backup
kubectl exec -i -n mlops-production postgres-0 -- psql -U mlops mlops_prod < backup.sql

# Backup persistent volumes
kubectl get pv -o yaml > pv-backup.yaml
```

### Scaling Operations

```bash
# Scale API servers
kubectl scale deployment api-server -n mlops-production --replicas=5

# Scale model servers
kubectl scale deployment model-server -n mlops-production --replicas=4

# Update resource limits
kubectl patch deployment api-server -n mlops-production -p '{"spec":{"template":{"spec":{"containers":[{"name":"api-server","resources":{"limits":{"cpu":"4","memory":"8Gi"}}}]}}}}'
```

---

## 🚨 **Troubleshooting Guide**

### Common Issues and Solutions

#### Pod Startup Issues
```bash
# Check pod status and events
kubectl describe pod <pod-name> -n mlops-production

# Check logs
kubectl logs <pod-name> -n mlops-production --previous

# Check resource constraints
kubectl top pod <pod-name> -n mlops-production
```

#### Database Connection Issues
```bash
# Test database connectivity
kubectl exec -n mlops-production <api-pod> -- pg_isready -h postgres -p 5432

# Check database logs
kubectl logs postgres-0 -n mlops-production

# Verify credentials
kubectl get secret mlops-database-credentials -n mlops-production -o yaml
```

#### SSL Certificate Issues
```bash
# Check certificate status
kubectl describe certificate mlops-platform-cert -n mlops-production

# Check cert-manager logs
kubectl logs -n cert-manager deployment/cert-manager

# Manual certificate renewal
kubectl delete certificate mlops-platform-cert -n mlops-production
kubectl apply -f security/certificates.yaml
```

#### Performance Issues
```bash
# Check resource usage
kubectl top nodes
kubectl top pods -n mlops-production

# Check horizontal pod autoscaler
kubectl describe hpa -n mlops-production

# Check Istio service mesh metrics
istioctl dashboard kiali
```

### Emergency Procedures

#### Service Outage Response
1. **Immediate**: Check system status dashboard
2. **Assess**: Identify affected services and impact
3. **Communicate**: Notify stakeholders via status page
4. **Investigate**: Review logs and metrics
5. **Resolve**: Apply fixes or rollback changes
6. **Monitor**: Verify service restoration
7. **Document**: Post-incident review and documentation

#### Data Corruption Response
1. **Isolate**: Stop affected services immediately
2. **Assess**: Determine extent of data corruption
3. **Restore**: Restore from latest clean backup
4. **Verify**: Validate data integrity
5. **Resume**: Restart services with verified data
6. **Investigate**: Root cause analysis and prevention

---

## 🔄 **CI/CD Integration**

### Automated Deployment Pipeline

```yaml
# .github/workflows/production-deploy.yml
name: Production Deployment
on:
  push:
    tags: ['v*']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure kubectl
      uses: azure/k8s-set-context@v1
      with:
        kubeconfig: ${{ secrets.KUBE_CONFIG }}
    
    - name: Deploy to production
      run: |
        kubectl set image deployment/api-server api-server=${{ env.IMAGE_TAG }} -n mlops-production
        kubectl rollout status deployment/api-server -n mlops-production
    
    - name: Run smoke tests
      run: |
        cd tests/deployment
        python smoke_tests.py --environment production
```

### Blue-Green Deployment

```bash
# Deploy new version to staging slot
kubectl apply -f config/kubernetes/api-server-blue.yaml

# Test new version
curl -H "Host: api-staging.mlops-platform.com" https://api.mlops-platform.com/health

# Switch traffic to new version
kubectl patch virtualservice mlops-api-vs -n mlops-production --type='json' \
  -p='[{"op": "replace", "path": "/spec/http/0/route/0/destination/subset", "value": "blue"}]'

# Monitor and rollback if needed
kubectl patch virtualservice mlops-api-vs -n mlops-production --type='json' \
  -p='[{"op": "replace", "path": "/spec/http/0/route/0/destination/subset", "value": "green"}]'
```

---

## 📚 **Additional Resources**

### Documentation Links
- [Architecture Decision Records](../docs/architecture/)
- [API Documentation](../docs/api/)
- [Security Policies](../docs/security/)
- [Monitoring Runbooks](../docs/monitoring/)

### External Dependencies
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Istio Service Mesh](https://istio.io/docs/)
- [Prometheus Monitoring](https://prometheus.io/docs/)
- [Grafana Dashboards](https://grafana.com/docs/)

### Support Contacts
- **Platform Team**: platform-team@company.com
- **Security Team**: security@company.com
- **On-Call Engineer**: +1-555-ON-CALL (665-2255)

---

## 📋 **Deployment Checklist Summary**

### Pre-Deployment
- [ ] Infrastructure provisioned
- [ ] DNS configured
- [ ] SSL certificates ready
- [ ] Secrets configured
- [ ] Configuration files customized

### Deployment
- [ ] Namespace and RBAC created
- [ ] Secrets and ConfigMaps applied
- [ ] Data layer deployed
- [ ] Application services deployed
- [ ] Service mesh configured
- [ ] Monitoring stack deployed
- [ ] Security policies applied
- [ ] SSL and ingress configured
- [ ] Autoscaling configured

### Post-Deployment
- [ ] Health checks passed
- [ ] Functional tests completed
- [ ] Performance tests executed
- [ ] Monitoring configured
- [ ] Alerts tested
- [ ] Documentation updated
- [ ] Team notified

---

**🎉 Deployment Complete!**

Your MLOps platform is now running in production with enterprise-grade security, monitoring, and scalability features. 

For ongoing support, refer to the [Operations Runbook](./operations-runbook.md) and [Incident Response Guide](./incident-response.md).