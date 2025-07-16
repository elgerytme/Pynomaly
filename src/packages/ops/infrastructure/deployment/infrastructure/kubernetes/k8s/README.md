# Pynomaly Kubernetes Deployment

This directory contains comprehensive Kubernetes deployment configurations for the Pynomaly anomaly detection platform.

## Overview

The deployment includes:

- **Application**: Pynomaly API and web interface
- **Databases**: PostgreSQL, Redis, MongoDB
- **Monitoring**: Prometheus, Grafana, Alertmanager
- **Ingress**: NGINX ingress controller
- **Security**: RBAC, Network policies, Pod security
- **Autoscaling**: Horizontal Pod Autoscaler
- **Storage**: Persistent volumes for data

## Prerequisites

- Kubernetes cluster (v1.20+)
- kubectl configured for your cluster
- Docker registry access
- Helm (optional, for package management)

## Quick Start

### 1. Build and Push Images

```bash
# Build production image
docker build -t pynomaly:latest -f Dockerfile.production .

# Tag and push to registry
docker tag pynomaly:latest your-registry.com/pynomaly:latest
docker push your-registry.com/pynomaly:latest
```

### 2. Update Configuration

Edit the following files with your specific values:

**secrets.yaml**

```yaml
# Update with your actual base64-encoded secrets
data:
  SECRET_KEY: <base64-encoded-secret>
  JWT_SECRET_KEY: <base64-encoded-jwt-secret>
  POSTGRES_PASSWORD: <base64-encoded-password>
```

**configmap.yaml**

```yaml
# Update with your environment-specific values
data:
  PYNOMALY_ENV: "production"
  PYNOMALY_LOG_LEVEL: "INFO"
```

### 3. Deploy to Kubernetes

```bash
# Run the deployment script
./scripts/deploy_kubernetes.sh deploy

# Or deploy manually
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/mongodb.yaml
kubectl apply -f k8s/pynomaly-app.yaml
kubectl apply -f k8s/monitoring.yaml
kubectl apply -f k8s/nginx-ingress.yaml
```

### 4. Verify Deployment

```bash
# Check deployment status
./scripts/validate_production.sh

# Check pods
kubectl get pods -n pynomaly-production

# Check services
kubectl get services -n pynomaly-production
```

## Configuration Files

### Core Application

- **namespace.yaml**: Kubernetes namespace for isolation
- **secrets.yaml**: Sensitive configuration (passwords, keys)
- **configmap.yaml**: Application configuration
- **pynomaly-app.yaml**: Main application deployment and service

### Data Layer

- **postgres.yaml**: PostgreSQL database (primary storage)
- **redis.yaml**: Redis cache and session store
- **mongodb.yaml**: MongoDB document storage

### Monitoring

- **monitoring.yaml**: Prometheus metrics collection and Grafana dashboards

### Networking

- **nginx-ingress.yaml**: NGINX ingress controller and SSL termination

## Security Configuration

### Secrets Management

The deployment uses Kubernetes secrets for sensitive data:

```bash
# Create secrets manually
kubectl create secret generic pynomaly-secrets \
  --from-literal=SECRET_KEY="$(openssl rand -base64 32)" \
  --from-literal=JWT_SECRET_KEY="$(openssl rand -base64 32)" \
  --from-literal=POSTGRES_PASSWORD="$(openssl rand -base64 16)" \
  -n pynomaly-production

# Create TLS secret
kubectl create secret tls pynomaly-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key \
  -n pynomaly-production
```

### Network Security

```yaml
# Network policy example
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: pynomaly-network-policy
spec:
  podSelector:
    matchLabels:
      app: pynomaly
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: pynomaly
    ports:
    - protocol: TCP
      port: 8000
```

### Pod Security

```yaml
# Pod security context
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  seccompProfile:
    type: RuntimeDefault
```

## Scaling and Performance

### Horizontal Pod Autoscaler

The application automatically scales based on CPU and memory usage:

```yaml
# HPA configuration
spec:
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Resource Limits

```yaml
# Resource configuration
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

## Monitoring and Observability

### Metrics Collection

Prometheus collects metrics from:

- Application metrics (`/metrics` endpoint)
- System metrics (CPU, memory, disk)
- Custom business metrics

### Dashboards

Grafana provides dashboards for:

- Application performance
- System resource usage
- Business KPIs
- Security events

### Alerting

Configured alerts for:

- High error rates
- Resource exhaustion
- Service unavailability
- Security threats

## Storage

### Persistent Volumes

The deployment uses persistent volumes for:

- Database data (PostgreSQL, MongoDB)
- Cache data (Redis)
- Application logs
- Model artifacts

### Storage Classes

```yaml
# Fast SSD storage class
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/gce-pd
parameters:
  type: pd-ssd
  zone: us-central1-a
```

## Backup and Recovery

### Database Backups

```yaml
# Backup CronJob
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:14
            command:
            - /bin/bash
            - -c
            - pg_dump $DATABASE_URL > /backup/backup-$(date +%Y%m%d).sql
```

### Recovery Procedures

1. **Database Recovery**

   ```bash
   # Restore from backup
   kubectl exec -it postgres-pod -- psql $DATABASE_URL < backup.sql
   ```

2. **Application Recovery**

   ```bash
   # Restart application
   kubectl rollout restart deployment/pynomaly-app -n pynomaly-production
   ```

## Troubleshooting

### Common Issues

1. **Pods Not Starting**

   ```bash
   # Check pod logs
   kubectl logs -f deployment/pynomaly-app -n pynomaly-production
   
   # Check events
   kubectl get events -n pynomaly-production --sort-by=.metadata.creationTimestamp
   ```

2. **Database Connection Issues**

   ```bash
   # Test database connectivity
   kubectl exec -it pynomaly-app-pod -- python -c "import psycopg2; psycopg2.connect(os.environ['DATABASE_URL'])"
   ```

3. **Storage Issues**

   ```bash
   # Check PVC status
   kubectl get pvc -n pynomaly-production
   
   # Check storage class
   kubectl get storageclass
   ```

### Performance Tuning

1. **Resource Optimization**

   ```yaml
   # Increase resources for heavy workloads
   resources:
     requests:
       memory: "1Gi"
       cpu: "500m"
     limits:
       memory: "2Gi"
       cpu: "1000m"
   ```

2. **Database Tuning**

   ```sql
   -- PostgreSQL optimization
   ALTER SYSTEM SET shared_buffers = '256MB';
   ALTER SYSTEM SET effective_cache_size = '1GB';
   ```

## Maintenance

### Rolling Updates

```bash
# Update application image
kubectl set image deployment/pynomaly-app pynomaly-app=your-registry.com/pynomaly:v2.0.0 -n pynomaly-production

# Check rollout status
kubectl rollout status deployment/pynomaly-app -n pynomaly-production
```

### Health Checks

```bash
# Run comprehensive validation
./scripts/validate_production.sh

# Check specific components
kubectl get pods -n pynomaly-production -l component=app
kubectl get services -n pynomaly-production
```

### Scaling Operations

```bash
# Manual scaling
kubectl scale deployment pynomaly-app --replicas=5 -n pynomaly-production

# Update HPA
kubectl patch hpa pynomaly-app-hpa -n pynomaly-production -p '{"spec":{"maxReplicas":20}}'
```

## Security Best Practices

1. **Regular Security Updates**
   - Keep base images updated
   - Apply security patches promptly
   - Monitor for vulnerabilities

2. **Access Control**
   - Use RBAC for fine-grained permissions
   - Implement network policies
   - Regular access reviews

3. **Data Protection**
   - Encrypt data at rest
   - Use TLS for data in transit
   - Implement backup encryption

## Support

For deployment support:

- Check the deployment logs
- Review the validation report
- Consult the troubleshooting guide
- Contact the development team

## Contributing

When adding new components:

1. Update the deployment configurations
2. Add monitoring and alerting
3. Include security configurations
4. Update documentation
5. Test thoroughly before deployment
