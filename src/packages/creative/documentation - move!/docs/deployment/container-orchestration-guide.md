# Container Orchestration Guide for Pynomaly

This guide covers the complete container orchestration setup for Pynomaly, including Kubernetes and Docker Swarm deployment options.

## Overview

Pynomaly supports two main container orchestration platforms:

1. **Kubernetes** - Full-featured orchestration for production environments
2. **Docker Swarm** - Simpler alternative for smaller deployments

Both options include:
- Complete application stack (API, workers, scheduler)
- Monitoring infrastructure (Prometheus, Grafana)
- Distributed tracing (Jaeger, OpenTelemetry)
- Logging infrastructure (ELK stack)
- Database services (PostgreSQL, Redis)
- Load balancing and ingress

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Container Orchestration                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Kubernetes  │  │ Docker      │  │ Helm        │          │
│  │ Manifests   │  │ Swarm       │  │ Charts      │          │
│  │             │  │ Stacks      │  │             │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                   Application Services                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Pynomaly    │  │ Training    │  │ Drift       │          │
│  │ API         │  │ Workers     │  │ Workers     │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Celery      │  │ Flower      │  │ Nginx       │          │
│  │ Scheduler   │  │ Monitor     │  │ Proxy       │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                  Infrastructure Services                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ PostgreSQL  │  │ Redis       │  │ Prometheus  │          │
│  │ Database    │  │ Cache       │  │ Monitoring  │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Grafana     │  │ Jaeger      │  │ Elasticsearch│         │
│  │ Dashboards  │  │ Tracing     │  │ Logging     │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured
- Helm 3.0+
- Storage class for persistent volumes
- Ingress controller (NGINX recommended)

### Quick Start

1. **Setup Kubernetes infrastructure:**
   ```bash
   cd /mnt/c/Users/andre/Pynomaly
   python3 scripts/orchestration/setup_kubernetes.py
   ```

2. **Deploy using Helm:**
   ```bash
   helm install pynomaly deploy/helm/pynomaly-complete \
     --namespace pynomaly-production \
     --create-namespace \
     --values deploy/helm/pynomaly-complete/values.yaml
   ```

3. **Or deploy using kubectl:**
   ```bash
   kubectl apply -f deploy/kubernetes/production-complete.yaml
   ```

### Configuration Options

#### Production Values (values.yaml)
- High availability with 3+ replicas
- Resource limits and requests
- Persistent storage for all data
- Full monitoring and logging stack
- TLS termination and security policies

#### Development Values (values.development.yaml)
- Single replicas for testing
- Reduced resource requirements
- Optional monitoring components
- Local storage options

### Scaling

```bash
# Scale API pods
kubectl scale deployment pynomaly-api --replicas=5 -n pynomaly-production

# Scale training workers
kubectl scale deployment pynomaly-worker-training --replicas=3 -n pynomaly-production

# Or use HPA (Horizontal Pod Autoscaler)
kubectl autoscale deployment pynomaly-api --cpu-percent=70 --min=3 --max=10 -n pynomaly-production
```

### Monitoring

```bash
# Check deployment status
kubectl get pods -n pynomaly-production

# View logs
kubectl logs -f deployment/pynomaly-api -n pynomaly-production

# Check resource usage
kubectl top pods -n pynomaly-production
```

## Docker Swarm Deployment

### Prerequisites

- Docker Engine with Swarm mode
- Docker Compose 3.8+
- Manager node with sufficient resources

### Quick Start

1. **Setup Docker Swarm:**
   ```bash
   cd /mnt/c/Users/andre/Pynomaly
   python3 scripts/orchestration/docker-stack-deploy.py
   ```

2. **Deploy the stack:**
   ```bash
   docker stack deploy --compose-file deploy/docker-swarm/docker-stack.yml pynomaly
   ```

### Configuration

The Docker Swarm deployment uses:
- Docker secrets for sensitive data
- Docker configs for configuration files
- Overlay networks for service communication
- Volume mounts for persistent data

### Scaling

```bash
# Scale API service
docker service scale pynomaly_pynomaly-api=5

# Scale training workers
docker service scale pynomaly_pynomaly-worker-training=3
```

### Monitoring

```bash
# Check stack status
docker stack ps pynomaly

# View service logs
docker service logs pynomaly_pynomaly-api

# Check resource usage
docker system df
```

## Service Discovery and Networking

### Kubernetes

- **Service Discovery**: Kubernetes DNS (cluster.local)
- **Load Balancing**: Service LoadBalancer type
- **Ingress**: NGINX Ingress Controller
- **Network Policies**: Pod-to-pod communication rules

### Docker Swarm

- **Service Discovery**: Docker DNS (tasks.service-name)
- **Load Balancing**: Built-in round-robin
- **Ingress**: Nginx reverse proxy
- **Networks**: Overlay networks with attachable option

## Storage Configuration

### Persistent Volumes

| Service | Size | Access Mode | Storage Class |
|---------|------|-------------|---------------|
| PostgreSQL | 20Gi | ReadWriteOnce | fast-ssd |
| Redis | 5Gi | ReadWriteOnce | fast-ssd |
| Prometheus | 50Gi | ReadWriteOnce | fast-ssd |
| Grafana | 10Gi | ReadWriteOnce | fast-ssd |
| Elasticsearch | 100Gi | ReadWriteOnce | fast-ssd |
| Jaeger | 30Gi | ReadWriteOnce | fast-ssd |

### Storage Classes

- **fast-ssd**: High-performance SSD storage for databases
- **standard**: Standard storage for logs and temporary data

## Security Configuration

### Network Security

- Network policies restricting pod-to-pod communication
- TLS termination at ingress level
- Service mesh (optional) for additional security

### Secrets Management

- Kubernetes secrets or Docker secrets for sensitive data
- External secret management integration (Vault, AWS Secrets Manager)
- Automatic secret rotation capabilities

### RBAC (Role-Based Access Control)

```yaml
# Example RBAC configuration
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: pynomaly-production
  name: pynomaly-operator
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch", "update"]
```

## High Availability Setup

### Database HA

- PostgreSQL with streaming replication
- Redis Sentinel for high availability
- Automated failover and backup

### Application HA

- Multiple replicas across availability zones
- Pod disruption budgets
- Rolling updates with zero downtime

### Infrastructure HA

- Multi-node cluster setup
- Load balancer redundancy
- Geographic distribution (if required)

## Backup and Disaster Recovery

### Automated Backups

```bash
# PostgreSQL backup
kubectl create job --from=cronjob/postgres-backup postgres-backup-manual -n pynomaly-production

# Volume snapshots
kubectl create volumesnapshot postgres-snapshot --volume-snapshot-class=csi-hostpath-snapclass
```

### Disaster Recovery

1. **Data Recovery**: Restore from automated backups
2. **Application Recovery**: Redeploy from Git/registry
3. **Configuration Recovery**: Infrastructure as Code (IaC)

## Performance Optimization

### Resource Optimization

- CPU and memory limits/requests tuning
- JVM heap size optimization for Java services
- Connection pooling configuration

### Caching Strategy

- Redis for application caching
- CDN for static assets
- Database query optimization

### Auto-scaling

```yaml
# Horizontal Pod Autoscaler example
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
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Troubleshooting

### Common Issues

1. **Pod Stuck in Pending State**
   ```bash
   kubectl describe pod <pod-name> -n pynomaly-production
   # Check resource constraints and node availability
   ```

2. **Service Not Accessible**
   ```bash
   kubectl get endpoints <service-name> -n pynomaly-production
   # Verify pod selectors and labels
   ```

3. **Storage Issues**
   ```bash
   kubectl get pv,pvc -n pynomaly-production
   # Check storage class and provisioner
   ```

### Debugging Commands

```bash
# Get detailed cluster information
kubectl cluster-info dump

# Check resource usage
kubectl top nodes
kubectl top pods -n pynomaly-production

# View events
kubectl get events -n pynomaly-production --sort-by='.lastTimestamp'

# Port forwarding for debugging
kubectl port-forward deployment/pynomaly-api 8000:8000 -n pynomaly-production
```

## Migration Guide

### From Docker Compose to Kubernetes

1. **Export data** from existing containers
2. **Create persistent volumes** in Kubernetes
3. **Import data** to new deployments
4. **Update DNS** and routing

### From Docker Swarm to Kubernetes

1. **Backup Docker volumes**
2. **Convert configurations** to Kubernetes format
3. **Deploy and verify** services
4. **Switch traffic** to new deployment

## Best Practices

### Development

- Use local development with Kind or minikube
- Implement proper health checks
- Use resource quotas to prevent resource exhaustion

### Staging

- Mirror production configuration
- Test disaster recovery procedures
- Validate monitoring and alerting

### Production

- Implement proper backup strategies
- Use blue-green or rolling deployments
- Monitor resource usage and performance
- Implement proper logging and tracing

## Scripts Reference

### Setup Scripts

- `scripts/orchestration/setup_kubernetes.py` - Kubernetes cluster setup
- `scripts/orchestration/docker-stack-deploy.py` - Docker Swarm setup

### Management Scripts

- `scripts/orchestration/deploy.sh` - Deploy applications
- `scripts/orchestration/monitor.sh` - Monitor deployments
- `scripts/orchestration/scale.sh` - Scale services
- `scripts/orchestration/cleanup.sh` - Clean up resources

### Configuration Files

- `deploy/kubernetes/production-complete.yaml` - Complete Kubernetes manifests
- `deploy/helm/pynomaly-complete/` - Helm chart for flexible deployments
- `deploy/docker-swarm/docker-stack.yml` - Docker Swarm stack definition

## Support and Resources

### Documentation

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Swarm Documentation](https://docs.docker.com/engine/swarm/)
- [Helm Documentation](https://helm.sh/docs/)

### Community

- Kubernetes Slack community
- Docker Community Forums
- Pynomaly GitHub Issues

### Professional Support

- Cloud provider managed services (EKS, GKE, AKS)
- Container orchestration consulting
- 24/7 monitoring and support services