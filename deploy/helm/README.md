# Anomaly Detection Helm Chart

This Helm chart deploys the Anomaly Detection microservice platform on Kubernetes.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.2.0+
- PV provisioner support in the underlying infrastructure (if persistence is enabled)

## Installing the Chart

To install the chart with the release name `my-release`:

```bash
helm install my-release ./deploy/helm
```

The command deploys the anomaly detection service on the Kubernetes cluster in the default configuration.

## Environment-Specific Deployments

### Development
```bash
helm install anomaly-detection-dev ./deploy/helm -f ./deploy/helm/values-development.yaml
```

### Staging
```bash
helm install anomaly-detection-staging ./deploy/helm -f ./deploy/helm/values-staging.yaml
```

### Production
```bash
helm install anomaly-detection-prod ./deploy/helm -f ./deploy/helm/values-production.yaml
```

## Uninstalling the Chart

To uninstall/delete the `my-release` deployment:

```bash
helm delete my-release
```

## Configuration

The following table lists the configurable parameters and their default values:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image.registry` | Container image registry | `ghcr.io` |
| `image.repository` | Container image repository | `your-org/anomaly-detection` |
| `image.tag` | Container image tag | `""` (uses Chart.AppVersion) |
| `image.pullPolicy` | Image pull policy | `IfNotPresent` |
| `replicaCount` | Number of replicas | `2` |
| `resources.limits.cpu` | CPU limit | `1000m` |
| `resources.limits.memory` | Memory limit | `1Gi` |
| `resources.requests.cpu` | CPU request | `500m` |
| `resources.requests.memory` | Memory request | `512Mi` |
| `autoscaling.enabled` | Enable horizontal pod autoscaler | `true` |
| `autoscaling.minReplicas` | Minimum number of replicas | `2` |
| `autoscaling.maxReplicas` | Maximum number of replicas | `10` |
| `service.type` | Kubernetes service type | `ClusterIP` |
| `service.port` | Service port | `8000` |
| `ingress.enabled` | Enable ingress | `false` |
| `persistence.enabled` | Enable persistent storage | `true` |
| `persistence.size` | Storage size | `10Gi` |
| `postgresql.enabled` | Deploy PostgreSQL | `true` |
| `redis.enabled` | Deploy Redis | `true` |
| `monitoring.enabled` | Enable monitoring | `true` |
| `networkPolicy.enabled` | Enable network policies | `true` |

### Complete Parameter List

For a complete list of configurable parameters, see the [values.yaml](values.yaml) file.

## Dependencies

This chart has optional dependencies on:

- **PostgreSQL** (Bitnami): Database storage
- **Redis** (Bitnami): Caching and session storage  
- **Kafka** (Bitnami): Message streaming (optional)

To update dependencies:

```bash
helm dependency update ./deploy/helm
```

## Health Checks

The chart includes comprehensive health checks:

- **Liveness Probe**: `/api/health/live` - Checks if the application is running
- **Readiness Probe**: `/api/health/ready` - Checks if the application is ready to serve traffic

## Security Features

- Non-root container execution
- Read-only root filesystem
- Security contexts with dropped capabilities
- Network policies for traffic isolation
- Pod security standards compliance

## Monitoring

When monitoring is enabled, the chart creates:

- ServiceMonitor for Prometheus scraping
- Custom metrics endpoint at `/metrics`
- Health check endpoints for external monitoring

## High Availability

Production deployments include:

- Pod anti-affinity rules
- Pod disruption budgets
- Horizontal pod autoscaling
- Multi-zone deployment support

## Backup and Recovery

For production deployments:

1. Database backups are handled by external managed services
2. Persistent volume snapshots should be configured at the infrastructure level
3. Application state is designed to be stateless where possible

## Troubleshooting

### Common Issues

1. **Pod fails to start**: Check resource constraints and node capacity
2. **Database connection issues**: Verify PostgreSQL configuration and secrets
3. **Ingress not accessible**: Check ingress controller and DNS configuration

### Debugging Commands

```bash
# Check pod status
kubectl get pods -l app.kubernetes.io/name=anomaly-detection

# View logs
kubectl logs -l app.kubernetes.io/name=anomaly-detection

# Check events
kubectl get events --sort-by=.metadata.creationTimestamp

# Test health endpoints
kubectl port-forward svc/anomaly-detection 8080:8000
curl http://localhost:8080/api/health/ready
```

## Upgrading

To upgrade an existing release:

```bash
helm upgrade my-release ./deploy/helm
```

For major version upgrades, review the [CHANGELOG.md](../../CHANGELOG.md) for breaking changes.

## Contributing

When modifying this chart:

1. Update the version in `Chart.yaml`
2. Document new parameters in this README
3. Test with all three environment configurations
4. Run `helm lint` to validate the chart