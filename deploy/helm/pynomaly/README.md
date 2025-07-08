# Pynomaly Helm Chart

This Helm chart deploys the Pynomaly anomaly detection platform on Kubernetes.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+
- PV provisioner support in the underlying infrastructure
- Metrics server installed (for HPA)

## Installing the Chart

To install the chart with the release name `pynomaly`:

```bash
helm repo add pynomaly https://charts.pynomaly.ai
helm install pynomaly pynomaly/pynomaly
```

Or install from source:

```bash
git clone https://github.com/pynomaly/pynomaly.git
cd pynomaly/deploy/helm
helm install pynomaly ./pynomaly
```

## Uninstalling the Chart

To uninstall/delete the `pynomaly` deployment:

```bash
helm delete pynomaly
```

## Configuration

The following table lists the configurable parameters of the Pynomaly chart and their default values.

### Global Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `global.imageRegistry` | Global Docker image registry | `""` |
| `global.imagePullSecrets` | Global Docker registry secret names | `[]` |
| `global.storageClass` | Global StorageClass for Persistent Volume(s) | `""` |

### Pynomaly Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `pynomaly.image.registry` | Pynomaly image registry | `docker.io` |
| `pynomaly.image.repository` | Pynomaly image repository | `pynomaly/pynomaly` |
| `pynomaly.image.tag` | Pynomaly image tag | `1.0.0` |
| `pynomaly.image.pullPolicy` | Pynomaly image pull policy | `IfNotPresent` |
| `pynomaly.environment` | Application environment | `production` |
| `pynomaly.logLevel` | Application log level | `INFO` |

### API Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `api.replicaCount` | Number of API replicas | `3` |
| `api.resources.requests.memory` | API memory request | `512Mi` |
| `api.resources.requests.cpu` | API CPU request | `250m` |
| `api.resources.limits.memory` | API memory limit | `2Gi` |
| `api.resources.limits.cpu` | API CPU limit | `1000m` |
| `api.autoscaling.enabled` | Enable horizontal pod autoscaling | `true` |
| `api.autoscaling.minReplicas` | Minimum number of replicas | `3` |
| `api.autoscaling.maxReplicas` | Maximum number of replicas | `10` |

### Worker Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `worker.replicaCount` | Number of worker replicas | `2` |
| `worker.concurrency` | Worker concurrency | `2` |
| `worker.resources.requests.memory` | Worker memory request | `1Gi` |
| `worker.resources.requests.cpu` | Worker CPU request | `500m` |
| `worker.resources.limits.memory` | Worker memory limit | `4Gi` |
| `worker.resources.limits.cpu` | Worker CPU limit | `2000m` |

### Service Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `service.type` | Kubernetes service type | `LoadBalancer` |
| `service.port` | Service port | `80` |
| `service.targetPort` | Target port | `8000` |

### Ingress Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ingress.enabled` | Enable ingress controller resource | `true` |
| `ingress.className` | Ingress class name | `nginx` |
| `ingress.hosts[0].host` | Hostname | `api.pynomaly.ai` |
| `ingress.hosts[0].paths[0].path` | Path | `/` |
| `ingress.hosts[0].paths[0].pathType` | Path type | `Prefix` |
| `ingress.tls[0].secretName` | TLS secret name | `pynomaly-tls` |

### PostgreSQL Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `postgresql.enabled` | Enable PostgreSQL deployment | `true` |
| `postgresql.auth.database` | Database name | `pynomaly` |
| `postgresql.auth.username` | Database username | `pynomaly` |
| `postgresql.auth.password` | Database password | `""` |
| `postgresql.primary.persistence.enabled` | Enable persistence | `true` |
| `postgresql.primary.persistence.size` | Storage size | `20Gi` |

### Redis Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `redis.enabled` | Enable Redis deployment | `true` |
| `redis.auth.enabled` | Enable Redis authentication | `true` |
| `redis.auth.password` | Redis password | `""` |
| `redis.master.persistence.enabled` | Enable persistence | `true` |
| `redis.master.persistence.size` | Storage size | `5Gi` |

### Monitoring Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `monitoring.metrics.enabled` | Enable Prometheus metrics | `true` |
| `monitoring.metrics.port` | Metrics port | `9090` |
| `monitoring.metrics.path` | Metrics path | `/metrics` |
| `monitoring.metrics.serviceMonitor.enabled` | Enable ServiceMonitor | `true` |
| `monitoring.metrics.serviceMonitor.interval` | Scrape interval | `30s` |

### Security Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `serviceAccount.create` | Create service account | `true` |
| `serviceAccount.automountServiceAccountToken` | Automount service account token | `false` |
| `rbac.create` | Create RBAC resources | `true` |
| `securityPolicies.networkPolicy.enabled` | Enable network policy | `true` |
| `securityPolicies.podSecurityStandards.enforce` | Pod security standards | `restricted` |

## Examples

### Basic Installation

```bash
helm install pynomaly pynomaly/pynomaly
```

### Production Installation with Custom Values

```bash
helm install pynomaly pynomaly/pynomaly -f values-production.yaml
```

Example `values-production.yaml`:

```yaml
pynomaly:
  image:
    tag: "1.0.0"
  environment: production
  logLevel: INFO

api:
  replicaCount: 5
  resources:
    requests:
      memory: "1Gi"
      cpu: "500m"
    limits:
      memory: "4Gi"
      cpu: "2000m"
  autoscaling:
    minReplicas: 5
    maxReplicas: 20

worker:
  replicaCount: 3
  resources:
    requests:
      memory: "2Gi"
      cpu: "1000m"
    limits:
      memory: "8Gi"
      cpu: "4000m"

postgresql:
  primary:
    persistence:
      size: 100Gi
    resources:
      requests:
        memory: "2Gi"
        cpu: "1000m"
      limits:
        memory: "8Gi"
        cpu: "4000m"

redis:
  master:
    persistence:
      size: 20Gi
    resources:
      requests:
        memory: "1Gi"
        cpu: "500m"
      limits:
        memory: "4Gi"
        cpu: "2000m"

ingress:
  enabled: true
  className: "nginx"
  hosts:
    - host: api.yourcompany.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: pynomaly-tls
      hosts:
        - api.yourcompany.com
```

### Development Installation

```bash
helm install pynomaly pynomaly/pynomaly \
  --set pynomaly.environment=development \
  --set pynomaly.logLevel=DEBUG \
  --set api.replicaCount=1 \
  --set worker.replicaCount=1 \
  --set postgresql.primary.persistence.size=10Gi \
  --set redis.master.persistence.size=2Gi
```

### Installation with External Database

```bash
helm install pynomaly pynomaly/pynomaly \
  --set postgresql.enabled=false \
  --set externalDatabase.host=postgres.example.com \
  --set externalDatabase.port=5432 \
  --set externalDatabase.database=pynomaly \
  --set externalDatabase.username=pynomaly \
  --set externalDatabase.password=secret
```

## Upgrading

To upgrade the Pynomaly deployment:

```bash
helm upgrade pynomaly pynomaly/pynomaly
```

## Troubleshooting

### Common Issues

1. **Pods not starting**: Check resource limits and node capacity
2. **Database connection errors**: Verify PostgreSQL is running and accessible
3. **Persistent volume issues**: Check storage class and PV availability
4. **Ingress not working**: Verify ingress controller is installed and configured

### Debugging Commands

```bash
# Check pod status
kubectl get pods -l app.kubernetes.io/name=pynomaly

# Check logs
kubectl logs -l app.kubernetes.io/name=pynomaly -f

# Check services
kubectl get svc -l app.kubernetes.io/name=pynomaly

# Check ingress
kubectl get ingress -l app.kubernetes.io/name=pynomaly

# Run tests
helm test pynomaly
```

## Support

For support, please:

1. Check the [documentation](https://docs.pynomaly.ai)
2. Search [existing issues](https://github.com/pynomaly/pynomaly/issues)
3. Create a [new issue](https://github.com/pynomaly/pynomaly/issues/new)
4. Join our [Discord community](https://discord.gg/pynomaly)

## License

This chart is licensed under the MIT License. See [LICENSE](LICENSE) for details.
