# Kubernetes Deployment Guide

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 🚀 [Deployment](README.md) > 📄 Kubernetes

---


## Overview

This guide provides comprehensive instructions for deploying Pynomaly in production Kubernetes environments. It covers everything from basic single-node deployments to highly available, multi-region setups with auto-scaling and monitoring.

## Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured and connected to your cluster
- Helm 3.0+ (optional but recommended)
- Docker registry access for pulling images
- Persistent storage class available in your cluster

## Quick Start

### 1. Create Namespace

```bash
kubectl create namespace pynomaly
kubectl config set-context --current --namespace=pynomaly
```

### 2. Deploy Basic Configuration

```bash
# Apply all manifests
kubectl apply -f k8s/base/

# Or using Kustomize
kubectl apply -k k8s/overlays/production/
```

### 3. Verify Deployment

```bash
# Check pod status
kubectl get pods

# Check service endpoints
kubectl get services

# View logs
kubectl logs -l app=pynomaly-api
```

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│   Ingress       │────│   Services      │
│   (External)    │    │   Controller    │    │   (ClusterIP)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Kubernetes Cluster                       │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ API Pods    │  │ Worker Pods │  │ Web UI Pods │             │
│  │ (3 replicas)│  │ (5 replicas)│  │ (2 replicas)│             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ PostgreSQL  │  │   Redis     │  │ Monitoring  │             │
│  │ (StatefulSet│  │ (StatefulSet│  │ (Prometheus)│             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration Files

### Deployment Manifest

Create `k8s/base/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pynomaly-api
  labels:
    app: pynomaly-api
    component: api
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: pynomaly-api
  template:
    metadata:
      labels:
        app: pynomaly-api
        component: api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: pynomaly-api
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 2000
      containers:
      - name: api
        image: pynomaly/api:latest
        imagePullPolicy: Always
        ports:
        - name: http
          containerPort: 8000
          protocol: TCP
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: pynomaly-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: pynomaly-secrets
              key: redis-url
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: pynomaly-secrets
              key: jwt-secret
        envFrom:
        - configMapRef:
            name: pynomaly-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "200m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /api/health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /api/health
            port: http
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
        - name: temp-storage
          mountPath: /tmp
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: pynomaly-models-pvc
      - name: temp-storage
        emptyDir: {}
      nodeSelector:
        kubernetes.io/os: linux
      tolerations:
      - key: "node.kubernetes.io/not-ready"
        operator: "Exists"
        effect: "NoExecute"
        tolerationSeconds: 300
      - key: "node.kubernetes.io/unreachable"
        operator: "Exists"
        effect: "NoExecute"
        tolerationSeconds: 300
```

### Service Configuration

Create `k8s/base/service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: pynomaly-api
  labels:
    app: pynomaly-api
    component: api
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8000"
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: http
    protocol: TCP
    name: http
  selector:
    app: pynomaly-api
---
apiVersion: v1
kind: Service
metadata:
  name: pynomaly-api-headless
  labels:
    app: pynomaly-api
    component: api
spec:
  type: ClusterIP
  clusterIP: None
  ports:
  - port: 8000
    targetPort: http
    protocol: TCP
    name: http
  selector:
    app: pynomaly-api
```

### ConfigMap

Create `k8s/base/configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pynomaly-config
data:
  # Application Configuration
  APP_NAME: "Pynomaly"
  APP_VERSION: "1.0.0"
  DEBUG: "false"
  
  # API Configuration
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  API_WORKERS: "4"
  API_TIMEOUT: "300"
  
  # Cache Configuration
  CACHE_ENABLED: "true"
  CACHE_TTL: "3600"
  
  # Authentication
  AUTH_ENABLED: "true"
  JWT_ALGORITHM: "HS256"
  JWT_EXPIRATION: "3600"
  
  # Rate Limiting
  API_RATE_LIMIT: "100"
  
  # Monitoring
  METRICS_ENABLED: "true"
  TRACING_ENABLED: "true"
  PROMETHEUS_ENABLED: "true"
  
  # Logging
  LOG_LEVEL: "INFO"
  LOG_FORMAT: "json"
  
  # Performance
  MAX_DATASET_SIZE_MB: "1000"
  GPU_ENABLED: "false"
  
  # Storage
  MODEL_STORAGE_PATH: "/app/models"
  EXPERIMENT_STORAGE_PATH: "/app/experiments"
```

### Secrets

Create `k8s/base/secrets.yaml`:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: pynomaly-secrets
type: Opaque
data:
  # Base64 encoded values - replace with actual values
  database-url: cG9zdGdyZXNxbDovL3VzZXI6cGFzc0BkYi9weW5vbWFseQ==
  redis-url: cmVkaXM6Ly9yZWRpcy1zZXJ2aWNlOjYzNzk=
  jwt-secret: eW91ci1zdXBlci1zZWNyZXQta2V5LWhhdmUtYS1nb29kLW9uZQ==
  
  # Optional: API keys for external services
  openai-api-key: ""
  huggingface-token: ""
```

### Persistent Volume Claims

Create `k8s/base/pvc.yaml`:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pynomaly-models-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pynomaly-data-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 500Gi
  storageClassName: standard
```

### Horizontal Pod Autoscaler

Create `k8s/base/hpa.yaml`:

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
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 20
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 30
      - type: Pods
        value: 4
        periodSeconds: 60
      selectPolicy: Max
```

### Ingress Configuration

Create `k8s/base/ingress.yaml`:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pynomaly-ingress
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "1000m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.pynomaly.io
    - pynomaly.io
    secretName: pynomaly-tls
  rules:
  - host: api.pynomaly.io
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: pynomaly-api
            port:
              number: 80
  - host: pynomaly.io
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: pynomaly-api
            port:
              number: 80
```

### Service Account and RBAC

Create `k8s/base/rbac.yaml`:

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: pynomaly-api
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::ACCOUNT:role/pynomaly-api-role
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pynomaly-api
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: pynomaly-api
subjects:
- kind: ServiceAccount
  name: pynomaly-api
  namespace: pynomaly
roleRef:
  kind: Role
  name: pynomaly-api
  apiGroup: rbac.authorization.k8s.io
```

## Database Setup

### PostgreSQL StatefulSet

Create `k8s/base/postgres.yaml`:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:14
        env:
        - name: POSTGRES_DB
          value: pynomaly
        - name: POSTGRES_USER
          value: pynomaly
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "1Gi"
            cpu: "500m"
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 20Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
spec:
  ports:
  - port: 5432
  clusterIP: None
  selector:
    app: postgres
```

### Redis StatefulSet

Create `k8s/base/redis.yaml`:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
spec:
  serviceName: redis
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-storage
          mountPath: /data
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"
  volumeClaimTemplates:
  - metadata:
      name: redis-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: redis
spec:
  ports:
  - port: 6379
  clusterIP: None
  selector:
    app: redis
```

## Environment-Specific Deployments

### Development Environment

Create `k8s/overlays/development/kustomization.yaml`:

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: pynomaly-dev

resources:
- ../../base

patchesStrategicMerge:
- deployment-patch.yaml
- configmap-patch.yaml

images:
- name: pynomaly/api
  newTag: dev-latest

replicas:
- name: pynomaly-api
  count: 1
```

Create `k8s/overlays/development/deployment-patch.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pynomaly-api
spec:
  template:
    spec:
      containers:
      - name: api
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

### Production Environment

Create `k8s/overlays/production/kustomization.yaml`:

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: pynomaly

resources:
- ../../base

patchesStrategicMerge:
- deployment-patch.yaml
- configmap-patch.yaml

images:
- name: pynomaly/api
  newTag: v1.0.0

replicas:
- name: pynomaly-api
  count: 3
```

## Monitoring and Observability

### ServiceMonitor for Prometheus

Create `k8s/monitoring/servicemonitor.yaml`:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: pynomaly-api
  labels:
    app: pynomaly-api
spec:
  selector:
    matchLabels:
      app: pynomaly-api
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
    scrapeTimeout: 10s
```

### Grafana Dashboard ConfigMap

Create `k8s/monitoring/grafana-dashboard.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pynomaly-dashboard
  labels:
    grafana_dashboard: "1"
data:
  pynomaly.json: |
    {
      "dashboard": {
        "id": null,
        "title": "Pynomaly Metrics",
        "tags": ["pynomaly"],
        "style": "dark",
        "timezone": "browser",
        "panels": [
          {
            "title": "Request Rate",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(http_requests_total{job=\"pynomaly-api\"}[5m])",
                "legendFormat": "{{method}} {{status}}"
              }
            ]
          },
          {
            "title": "Response Time",
            "type": "graph",
            "targets": [
              {
                "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job=\"pynomaly-api\"}[5m]))",
                "legendFormat": "95th percentile"
              }
            ]
          }
        ]
      }
    }
```

## Deployment Commands

### Initial Deployment

```bash
# Create namespace
kubectl create namespace pynomaly

# Apply secrets (update with real values first!)
kubectl apply -f k8s/base/secrets.yaml

# Apply database
kubectl apply -f k8s/base/postgres.yaml
kubectl apply -f k8s/base/redis.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=postgres --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis --timeout=300s

# Apply main application
kubectl apply -f k8s/base/

# Or using Kustomize for specific environment
kubectl apply -k k8s/overlays/production/
```

### Rolling Updates

```bash
# Update image tag
kubectl set image deployment/pynomaly-api api=pynomaly/api:v1.1.0

# Monitor rollout
kubectl rollout status deployment/pynomaly-api

# Rollback if needed
kubectl rollout undo deployment/pynomaly-api
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment pynomaly-api --replicas=5

# Check HPA status
kubectl get hpa pynomaly-api-hpa

# View HPA events
kubectl describe hpa pynomaly-api-hpa
```

## Health Checks and Troubleshooting

### Health Check Commands

```bash
# Check pod health
kubectl get pods -l app=pynomaly-api

# Check pod logs
kubectl logs -l app=pynomaly-api --tail=100

# Check service endpoints
kubectl get endpoints pynomaly-api

# Test health endpoint
kubectl port-forward svc/pynomaly-api 8080:80
curl http://localhost:8080/api/health
```

### Common Issues

#### Pod CrashLoopBackOff

```bash
# Check pod events
kubectl describe pod <pod-name>

# Check logs for errors
kubectl logs <pod-name> --previous

# Common causes:
# - Database connection issues
# - Missing environment variables
# - Resource limits too low
```

#### Service Unavailable

```bash
# Check service configuration
kubectl describe service pynomaly-api

# Check endpoints
kubectl get endpoints pynomaly-api

# Verify pod labels match service selector
kubectl get pods --show-labels
```

#### High Memory Usage

```bash
# Check resource usage
kubectl top pods

# Increase memory limits in deployment
kubectl patch deployment pynomaly-api -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","resources":{"limits":{"memory":"4Gi"}}}]}}}}'
```

### Debugging Commands

```bash
# Access pod shell
kubectl exec -it <pod-name> -- /bin/bash

# View configuration
kubectl exec -it <pod-name> -- env | grep PYNOMALY

# Check file permissions
kubectl exec -it <pod-name> -- ls -la /app/models

# Test database connection
kubectl exec -it <pod-name> -- python -c "
import os
import psycopg2
conn = psycopg2.connect(os.environ['DATABASE_URL'])
print('Database connection successful')
"
```

## Performance Optimization

### Resource Allocation

```yaml
# Optimized resource configuration
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

### Node Affinity

```yaml
# Prefer nodes with SSD storage
affinity:
  nodeAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      preference:
        matchExpressions:
        - key: storage-type
          operator: In
          values:
          - ssd
```

### Pod Disruption Budget

Create `k8s/base/pdb.yaml`:

```yaml
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

## Security Considerations

### Network Policies

Create `k8s/security/network-policy.yaml`:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: pynomaly-api-netpol
spec:
  podSelector:
    matchLabels:
      app: pynomaly-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

### Pod Security Policy

Create `k8s/security/pod-security-policy.yaml`:

```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: pynomaly-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

## Backup and Disaster Recovery

### Database Backup CronJob

Create `k8s/backup/postgres-backup.yaml`:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: postgres-backup
            image: postgres:14
            env:
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: postgres-secret
                  key: password
            command:
            - /bin/bash
            - -c
            - |
              pg_dump -h postgres -U pynomaly pynomaly | gzip > /backup/pynomaly-$(date +%Y%m%d-%H%M%S).sql.gz
              # Upload to S3 or your backup storage
              aws s3 cp /backup/pynomaly-*.sql.gz s3://your-backup-bucket/postgres/
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            emptyDir: {}
          restartPolicy: OnFailure
```

## Best Practices

### 1. Resource Management
- Set appropriate resource requests and limits
- Use HPA for automatic scaling
- Monitor resource usage regularly

### 2. High Availability
- Run multiple replicas across different nodes
- Use pod anti-affinity rules
- Implement proper health checks

### 3. Security
- Use non-root containers
- Implement network policies
- Regularly rotate secrets
- Scan images for vulnerabilities

### 4. Monitoring
- Set up comprehensive monitoring
- Create alerts for critical metrics
- Use distributed tracing

### 5. Updates
- Use rolling updates for zero-downtime deployments
- Test updates in staging environment first
- Have rollback procedures ready

This comprehensive Kubernetes deployment guide provides everything needed to run Pynomaly in production environments with high availability, security, and observability.

---

## 🔗 **Related Documentation**

### **Getting Started**
- **[Installation Guide](../../getting-started/installation.md)** - Setup and installation
- **[Quick Start](../../getting-started/quickstart.md)** - Your first detection
- **[Platform Setup](../../getting-started/platform-specific/)** - Platform-specific guides

### **User Guides**
- **[Basic Usage](../basic-usage/README.md)** - Essential functionality
- **[Advanced Features](../advanced-features/README.md)** - Sophisticated capabilities  
- **[Troubleshooting](../troubleshooting/README.md)** - Problem solving

### **Reference**
- **[Algorithm Reference](../../reference/algorithms/README.md)** - Algorithm documentation
- **[API Documentation](../../developer-guides/api-integration/README.md)** - Programming interfaces
- **[Configuration](../../reference/configuration/)** - System configuration

### **Examples**
- **[Examples & Tutorials](../../examples/README.md)** - Real-world use cases
- **[Banking Examples](../../examples/banking/)** - Financial fraud detection
- **[Notebooks](../../examples/notebooks/)** - Interactive examples

---

## 🆘 **Getting Help**

- **[Troubleshooting Guide](../troubleshooting/troubleshooting.md)** - Common issues and solutions
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs and request features
- **[GitHub Discussions](https://github.com/your-org/pynomaly/discussions)** - Ask questions and share ideas
- **[Security Issues](mailto:security@pynomaly.org)** - Report security vulnerabilities
