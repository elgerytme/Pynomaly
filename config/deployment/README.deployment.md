# Pynomaly Deployment Configuration

This directory contains all deployment-related configuration files for the Pynomaly project, organized by deployment type and environment.

## Directory Structure

```
config/deployment/
├── README.deployment.md                # This file
├── MANIFEST.in                         # Package manifest
├── sdk_generator_config.yaml           # SDK generation configuration
├── analytics/                          # Analytics configurations
│   └── analytics.yml
├── backup/                             # Backup and disaster recovery
│   ├── backup_config.yml
│   └── dr_config.yml
├── ci-cd/                              # CI/CD pipeline configurations
│   ├── github_sync_config.yaml
│   ├── pipeline-config.yaml
│   └── github-actions.template.yml     # GitHub Actions template
├── data/                               # Data management configurations
│   ├── schemas/                        # Data schemas
│   │   ├── anomaly_data.json
│   │   ├── config_data.json
│   │   ├── timeseries_data.json
│   │   └── user_data.json
│   └── validation_config.yml
├── docker/                             # Docker deployment configurations
│   ├── docker-compose.yml              # Base Docker Compose
│   ├── docker-compose.development.yml  # Development overrides
│   ├── docker-compose.production.yml   # Production overrides
│   ├── docker-compose.template.yml     # Template for new environments
│   ├── production.yml                  # Production configuration
│   ├── production.yaml                 # Additional production settings
│   └── production/                     # Production-specific configs
│       ├── README.md
│       ├── docker-compose.prod.yml
│       ├── grafana/
│       ├── loki/
│       ├── nginx/
│       ├── prometheus/
│       └── promtail/
├── docs/                               # Documentation configurations
│   └── mkdocs.yml
├── kubernetes/                         # Kubernetes deployment configurations
│   └── deployment.template.yaml        # Kubernetes deployment template
├── logging/                            # Logging configurations
│   └── production_logging.yaml
├── ml/                                 # Machine Learning configurations
│   ├── mlflow.yml
│   ├── mlops.yml
│   └── model_governance.yml
├── monitoring/                         # Monitoring stack configurations
│   ├── monitoring-stack.template.yml   # Complete monitoring stack template
│   ├── advanced_monitoring.yml
│   ├── alert_rules.yml
│   ├── alertmanager.yml
│   ├── grafana_dashboards.json
│   ├── performance_alerts.json
│   ├── production.yml
│   ├── prometheus.yml
│   ├── alertmanager/
│   │   └── alertmanager.yml
│   ├── grafana/
│   │   └── provisioning/
│   └── prometheus/
│       └── rules/
├── performance/                        # Performance optimization configurations
│   ├── performance_optimization.yaml
│   ├── caching-strategy.js
│   ├── image-optimizer.js
│   ├── performance-monitor.js
│   └── performance-optimizer.js
├── redis/                              # Redis configurations
│   ├── redis-production.conf
│   └── sentinel.conf
├── security/                           # Security configurations
│   ├── security.yaml
│   ├── keys.json
│   ├── security_checklist.md
│   ├── security_config.py
│   ├── security_policy.yml
│   ├── waf_config.json
│   └── waf_signatures.json
├── systemd/                            # SystemD service configurations
│   ├── pynomaly-alerts.service
│   └── pynomaly-dashboard.service
├── terraform/                          # Infrastructure as Code
│   └── main.template.tf                # Terraform template for AWS/GCP/Azure
├── testing/                            # Testing configurations
│   ├── advanced_testing_config.json
│   ├── coverage_config.yml
│   ├── pytest-bdd.ini
│   ├── quality_gates.yml
│   ├── tdd_config.json
│   └── tox.ini
├── toolchains/                         # Build toolchain configurations
│   └── BUCK
├── tracing/                            # Distributed tracing configurations
│   └── distributed_tracing.yaml
└── web/                                # Web frontend configurations
    ├── nginx.conf
    ├── playwright.config.ts
    ├── lighthouse.config.js
    ├── lighthouserc.js
    ├── postcss.config.js
    └── tailwind.config.js
```

## Usage

### Environment Variables

All deployment configurations support environment variable substitution:

```bash
# Set environment
export PYNOMALY_ENV=production
export VERSION=1.2.0

# Database configuration
export DATABASE_URL="postgresql://user:pass@host:5432/pynomaly"
export REDIS_URL="redis://redis-host:6379"

# Security
export SECRET_KEY="your-secret-key"
export GRAFANA_PASSWORD="secure-password"
```

### Docker Deployment

```bash
# Development
docker-compose -f config/deployment/docker/docker-compose.yml \
               -f config/deployment/docker/docker-compose.development.yml up

# Production
docker-compose -f config/deployment/docker/docker-compose.yml \
               -f config/deployment/docker/docker-compose.production.yml up
```

### Kubernetes Deployment

```bash
# Apply base configuration
envsubst < config/deployment/kubernetes/deployment.template.yaml | kubectl apply -f -

# Scale deployment
kubectl scale deployment pynomaly-api --replicas=5
```

### Terraform Infrastructure

```bash
cd config/deployment/terraform/
terraform init
terraform plan -var="environment=production" -var="region=us-west-2"
terraform apply
```

### Monitoring Stack

```bash
# Start complete monitoring stack
docker-compose -f config/deployment/monitoring/monitoring-stack.template.yml up

# Access dashboards
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
# Alertmanager: http://localhost:9093
```

## Configuration Templates

The following template files are provided for common deployment patterns:

- `docker/docker-compose.template.yml` - Complete Docker stack with all services
- `kubernetes/deployment.template.yaml` - Kubernetes deployment with secrets and services
- `terraform/main.template.tf` - Infrastructure provisioning for cloud providers
- `monitoring/monitoring-stack.template.yml` - Complete observability stack
- `ci-cd/github-actions.template.yml` - CI/CD pipeline template

### Using Templates

1. Copy template to new file:
   ```bash
   cp config/deployment/docker/docker-compose.template.yml \
      config/deployment/docker/docker-compose.staging.yml
   ```

2. Customize for your environment:
   ```bash
   # Edit the file to match your staging environment
   vim config/deployment/docker/docker-compose.staging.yml
   ```

3. Set environment variables and deploy:
   ```bash
   export PYNOMALY_ENV=staging
   docker-compose -f config/deployment/docker/docker-compose.staging.yml up
   ```

## Security Considerations

- **Secrets Management**: Never commit secrets to version control
- **Environment Variables**: Use environment variables for sensitive data
- **Access Controls**: Configure proper RBAC in Kubernetes
- **Network Security**: Use proper firewalls and security groups
- **Encryption**: Enable encryption at rest and in transit

## Configuration Validation

Validate configurations before deployment:

```bash
# Validate Docker Compose
docker-compose -f config/deployment/docker/docker-compose.yml config

# Validate Kubernetes manifests
kubectl --dry-run=client apply -f config/deployment/kubernetes/

# Validate Terraform
cd config/deployment/terraform/
terraform validate
```

## Troubleshooting

### Common Issues

1. **Environment Variables Not Set**:
   ```bash
   # Check required variables
   echo $PYNOMALY_ENV $DATABASE_URL $SECRET_KEY
   ```

2. **Permission Denied**:
   ```bash
   # Fix file permissions
   chmod +x config/deployment/scripts/*.sh
   ```

3. **Port Conflicts**:
   ```bash
   # Check port usage
   netstat -tlnp | grep :8000
   ```

### Debug Mode

Enable debug logging in deployment:

```bash
export PYNOMALY_LOG_LEVEL=DEBUG
export PYNOMALY_DEBUG=true
```

## Migration from Legacy Configuration

This structure consolidates the previous scattered configuration files from `deployment/config_files/config/` (~47 files) into an organized hierarchy. The migration provides:

- **Single Source of Truth**: Each configuration type has a dedicated location
- **Environment Separation**: Clear separation between development, staging, and production
- **Template System**: Reusable templates for common deployment patterns
- **Better Organization**: Logical grouping by deployment concern

### Automated Migration

Use the migration script to update references:

```bash
python scripts/migrate_config.py --project-root .
```

## Contributing

When adding new deployment configurations:

1. Place files in the appropriate category directory
2. Use environment variable substitution for environment-specific values
3. Create templates for reusable patterns
4. Update this README with new configurations
5. Test configurations in development before production deployment

## Support

For deployment issues:
- Check the configuration validation output
- Review logs in `storage/logs/`
- Consult the troubleshooting section above
- Open an issue with deployment details