# Pynomaly Kubernetes Deployment

This directory contains comprehensive Kubernetes manifests for deploying Pynomaly in production environments. The deployment includes the complete anomaly detection stack with monitoring, security, and scalability features.

## 🏗️ Architecture Overview

The Kubernetes deployment provides:

- **API Server**: FastAPI-based REST API with auto-scaling
- **Worker Services**: Distributed task processing (training, drift detection, scheduling)
- **Database Stack**: PostgreSQL with Redis for caching and queuing
- **Monitoring Stack**: Prometheus, Grafana, OpenTelemetry, and Alertmanager
- **Security**: RBAC, network policies, and security contexts
- **Storage**: Persistent volumes for data, logs, and backups
- **Ingress**: SSL termination, rate limiting, and routing

## 📋 Prerequisites

### Required Tools
- **kubectl** >= 1.25
- **kustomize** >= 4.5
- **helm** >= 3.10 (optional, for Helm deployment)

### Optional Tools (for enhanced functionality)
- **kubeval** - YAML validation
- **kube-score** - Security and best practices analysis
- **kube-hunter** - Security scanning

### Cluster Requirements
- **Kubernetes**: 1.25+ with RBAC enabled
- **Storage**: Dynamic provisioning with gp3/fast-ssd support
- **Ingress Controller**: NGINX Ingress Controller
- **Cert Manager**: For SSL certificate management
- **Resources**: Minimum 4 CPU cores, 8GB RAM per node

## 🚀 Quick Start

### 1. Prepare Environment

```bash
# Clone repository and navigate to Kubernetes directory
cd deploy/kubernetes

# Verify prerequisites
make check-prerequisites

# Validate manifests
make validate
```

### 2. Configure Secrets

Update the secrets template with your production values:

```bash
# Edit secrets (use base64 encoded values)
kubectl create secret generic pynomaly-secrets \
  --from-literal=postgres-password="your-secure-password" \
  --from-literal=redis-password="your-redis-password" \
  --from-literal=secret-key="your-secret-key" \
  --from-literal=jwt-secret="your-jwt-secret" \
  --dry-run=client -o yaml > secrets.yaml
```

### 3. Deploy to Kubernetes

```bash
# Deploy the complete stack
make deploy

# Wait for deployment to be ready
make wait

# Check deployment status
make status
```

### 4. Access Services

```bash
# View service URLs
make urls

# Port forward for local access
make port-forward-api      # API at http://localhost:8000
make port-forward-grafana  # Grafana at http://localhost:3000
```

## 📁 File Structure

```
deploy/kubernetes/
├── README.md                 # This file
├── Makefile                  # Automation commands
├── kustomization.yaml        # Kustomize configuration
├── namespace.yaml            # Namespace and resource quotas
├── rbac.yaml                 # Service accounts and permissions
├── secrets.yaml              # Secret management template
├── config-maps.yaml          # Configuration management
├── persistent-volumes.yaml   # Storage configuration
├── database-statefulset.yaml # PostgreSQL and Redis
├── api-deployment.yaml       # API server deployment
├── worker-deployment.yaml    # Worker services
├── monitoring-deployment.yaml # Monitoring stack
└── ingress.yaml              # External access configuration
```

## 🛠️ Available Commands

### Deployment Operations
```bash
make deploy        # Deploy complete stack
make upgrade       # Upgrade existing deployment
make rollback      # Rollback to previous version
make wait          # Wait for deployment readiness
```

### Status and Monitoring
```bash
make status        # Show deployment status
make health        # Check application health
make pods          # Show pod details
make events        # Show recent events
```

### Logging and Debugging
```bash
make logs          # Show application logs
make logs-api      # Show API server logs
make logs-workers  # Show worker logs
make logs-database # Show database logs
make describe-failed # Describe failed pods
```

### Access and Port Forwarding
```bash
make port-forward-api       # Forward API port (8000)
make port-forward-grafana   # Forward Grafana port (3000)
make port-forward-prometheus # Forward Prometheus port (9090)
make shell-api             # Open shell in API pod
make shell-database        # Open database shell
```

### Scaling Operations
```bash
make scale-api REPLICAS=5      # Scale API deployment
make scale-workers REPLICAS=3  # Scale worker deployments
make autoscale-api            # Enable auto-scaling
```

### Testing and Validation
```bash
make test          # Run deployment tests
make load-test     # Run basic load test
make validate      # Validate manifests
make lint          # Lint Kubernetes files
```

### Backup and Restore
```bash
make backup        # Create database backup
make list-backups  # List available backups
make restore BACKUP_FILE=backup.sql # Restore from backup
```

### Security and Maintenance
```bash
make security-scan # Run security analysis
make update-secrets # Update secret values
make restart-all   # Restart all deployments
```

### Cleanup
```bash
make clean         # Remove deployment (keep data)
make clean-all     # Remove everything including data
```

## 🔧 Configuration

### Environment Variables

The deployment supports these environment variables:

- **NAMESPACE**: Kubernetes namespace (default: `pynomaly`)
- **ENVIRONMENT**: Deployment environment (default: `production`)
- **KUBECONFIG**: Path to kubeconfig file
- **TIMEOUT**: Deployment timeout (default: `600s`)

### Storage Configuration

The deployment creates several persistent volumes:

- **Application Storage**: 50Gi for models and data
- **Logs Storage**: 20Gi for application logs
- **Prometheus Storage**: 100Gi for metrics retention
- **Grafana Storage**: 10Gi for dashboards and configs
- **Backup Storage**: 200Gi for backup retention

### Resource Requirements

**API Deployment (3 replicas)**:
- CPU: 500m request, 2 limit
- Memory: 1Gi request, 4Gi limit

**Worker Deployments (2 replicas each)**:
- CPU: 1 request, 4 limit
- Memory: 2Gi request, 8Gi limit

**Database Services**:
- PostgreSQL: 500m CPU, 2Gi memory
- Redis: 250m CPU, 512Mi memory

## 🌐 Ingress Configuration

The deployment creates three ingress resources:

### Main Application (`pynomaly.local`)
- Main API and web interface
- SSL termination with Let's Encrypt
- Rate limiting: 100 requests/minute
- Security headers and CORS

### Monitoring (`grafana.pynomaly.local`, `prometheus.pynomaly.local`)
- Basic authentication required
- SSL termination
- Rate limiting: 50 requests/minute

### Admin (`flower.pynomaly.local`)
- Strong authentication
- IP whitelist for admin access
- Rate limiting: 20 requests/minute

## 🔒 Security Features

### Network Security
- **Network Policies**: Restrict inter-pod communication
- **RBAC**: Least-privilege access control
- **Security Contexts**: Non-root execution, read-only filesystems
- **Pod Security Policies**: Enhanced security constraints

### Data Security
- **Encrypted Storage**: All persistent volumes encrypted
- **Secret Management**: Kubernetes secrets for sensitive data
- **SSL/TLS**: End-to-end encryption with cert-manager
- **Input Validation**: Request size limits and validation

### Authentication
- **Basic Auth**: For monitoring and admin interfaces
- **JWT Tokens**: For API authentication
- **Service Accounts**: For inter-service communication

## 📊 Monitoring and Observability

### Metrics Collection
- **Prometheus**: System and application metrics
- **OpenTelemetry**: Distributed tracing
- **Node Exporter**: System-level metrics
- **PostgreSQL Exporter**: Database metrics

### Visualization
- **Grafana**: Dashboards for system and application metrics
- **Alertmanager**: Alert routing and notification
- **Flower**: Celery task monitoring

### Logging
- **Structured Logging**: JSON-formatted logs
- **Log Aggregation**: Centralized log collection
- **Log Retention**: Configurable retention policies

## 🚨 Troubleshooting

### Common Issues

**Pods in Pending State**:
```bash
make describe-failed  # Check resource constraints
kubectl describe nodes # Check node capacity
```

**ImagePullBackOff Errors**:
```bash
kubectl describe pod <pod-name> -n pynomaly
# Check registry credentials and image availability
```

**Database Connection Issues**:
```bash
make health           # Check service health
make logs-database    # Check database logs
make shell-database   # Connect to database directly
```

**SSL Certificate Issues**:
```bash
kubectl describe certificate -n pynomaly
kubectl logs -n cert-manager deployment/cert-manager
```

### Health Checks

The deployment includes comprehensive health checks:

- **API Health**: `GET /api/health`
- **Database Health**: PostgreSQL `pg_isready`
- **Cache Health**: Redis `PING` command
- **Monitoring Health**: Prometheus `/healthy` endpoint

### Performance Tuning

**Scaling Recommendations**:
- Start with default replica counts
- Monitor resource usage with Grafana
- Scale based on CPU/memory utilization
- Use HPA for automatic scaling

**Resource Optimization**:
- Adjust CPU/memory limits based on workload
- Use node affinity for database placement
- Configure storage classes for performance

## 🔄 GitOps Integration

The Kustomization configuration supports GitOps workflows:

### ArgoCD Integration
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: pynomaly
spec:
  source:
    repoURL: https://github.com/your-org/pynomaly
    path: deploy/kubernetes
    targetRevision: main
  destination:
    server: https://kubernetes.default.svc
    namespace: pynomaly
```

### FluxCD Integration
```yaml
apiVersion: kustomize.toolkit.fluxcd.io/v1beta1
kind: Kustomization
metadata:
  name: pynomaly
spec:
  sourceRef:
    kind: GitRepository
    name: pynomaly
  path: "./deploy/kubernetes"
  interval: 10m
```

## 📈 Scaling and Performance

### Auto-scaling Configuration

**Horizontal Pod Autoscaler (HPA)**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pynomaly-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pynomaly-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

**Vertical Pod Autoscaler (VPA)**:
```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: pynomaly-api-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pynomaly-api
  updatePolicy:
    updateMode: "Auto"
```

### Performance Monitoring

Key metrics to monitor:
- **API Response Time**: p95 < 500ms
- **Throughput**: Requests per second
- **Error Rate**: < 1% error rate
- **Resource Utilization**: CPU < 80%, Memory < 80%
- **Database Performance**: Query time, connection pool usage

## 🛡️ Disaster Recovery

### Backup Strategy

**Automated Backups**:
- Database: Daily PostgreSQL dumps
- Configuration: ConfigMaps and Secrets backup
- Persistent Volumes: Snapshot-based backups

**Backup Verification**:
```bash
make backup           # Create backup
make list-backups     # Verify backup creation
make test-restore     # Test restore process
```

### Recovery Procedures

**Database Recovery**:
```bash
# Restore from specific backup
make restore BACKUP_FILE=/backup/pynomaly-20240101-120000.sql

# Verify data integrity
make health
```

**Complete Environment Recreation**:
```bash
# Save current state
kubectl get all -n pynomaly -o yaml > pynomaly-backup.yaml

# Clean and redeploy
make clean-all
make deploy
make restore BACKUP_FILE=latest-backup.sql
```

## 📞 Support and Maintenance

### Regular Maintenance Tasks

**Weekly**:
- Review resource utilization
- Check security vulnerability scans
- Validate backup integrity
- Update monitoring dashboards

**Monthly**:
- Update container images
- Review and rotate secrets
- Perform disaster recovery tests
- Capacity planning review

### Support Contacts

For issues and support:
- **Technical Issues**: Create GitHub issue
- **Security Concerns**: security@pynomaly.io
- **Documentation**: docs@pynomaly.io

## 📚 Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Kustomize Guide](https://kustomize.io/)
- [Prometheus Operator](https://prometheus-operator.dev/)
- [Grafana Documentation](https://grafana.com/docs/)
- [cert-manager Documentation](https://cert-manager.io/docs/)

---

**Note**: This deployment configuration is designed for production use with security, monitoring, and scalability in mind. Always review and customize the configuration for your specific environment and requirements.
