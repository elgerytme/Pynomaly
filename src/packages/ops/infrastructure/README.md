# Pynomaly Infrastructure Package

A comprehensive infrastructure and operations package for the Pynomaly platform, providing multi-cloud deployment, monitoring, and automation capabilities.

## Features

- **Multi-Cloud Support**: Deploy and manage infrastructure across AWS, Azure, and Google Cloud Platform
- **Kubernetes Orchestration**: Container orchestration and management
- **Infrastructure as Code**: Terraform and CloudFormation templates
- **Monitoring & Observability**: Prometheus, Grafana, and custom metrics
- **Automated Deployment**: CI/CD pipeline integration
- **Configuration Management**: Centralized configuration and secrets management
- **CLI Tools**: Command-line interface for infrastructure operations

## Installation

```bash
pip install pynomaly-infrastructure
```

For development with all extras:
```bash
pip install pynomaly-infrastructure[dev,aws,azure,gcp,kubernetes,monitoring]
```

## Quick Start

### Basic Usage

```python
from infrastructure import InfrastructureManager

# Initialize infrastructure manager
manager = InfrastructureManager()

# Deploy to AWS
manager.deploy_to_aws(
    region='us-east-1',
    instance_type='t3.medium',
    replicas=3
)

# Monitor deployment
status = manager.get_deployment_status()
print(f"Deployment status: {status}")
```

### CLI Usage

```bash
# Deploy infrastructure
pynomaly-infra deploy --provider aws --region us-east-1

# Monitor services
pynomaly-monitor status --service-name pynomaly-api

# Scale deployment
pynomaly-deploy scale --replicas 5 --service pynomaly-worker
```

## Architecture

The infrastructure package follows a modular architecture:

```
infrastructure/
├── providers/          # Cloud provider implementations
│   ├── aws/           # AWS-specific implementations
│   ├── azure/         # Azure-specific implementations
│   └── gcp/           # GCP-specific implementations
├── orchestration/     # Container orchestration
│   ├── kubernetes/    # Kubernetes manifests and operators
│   └── docker/        # Docker compositions
├── monitoring/        # Monitoring and observability
│   ├── prometheus/    # Prometheus configuration
│   ├── grafana/       # Grafana dashboards
│   └── alerting/      # Alert rules and notifications
├── deployment/        # Deployment automation
│   ├── pipelines/     # CI/CD pipeline configurations
│   └── scripts/       # Deployment scripts
└── templates/         # IaC templates
    ├── terraform/     # Terraform modules
    └── cloudformation/ # CloudFormation templates
```

## Configuration

### Environment Variables

```bash
# AWS Configuration
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1

# Azure Configuration
export AZURE_CLIENT_ID=your_client_id
export AZURE_CLIENT_SECRET=your_client_secret
export AZURE_TENANT_ID=your_tenant_id

# GCP Configuration
export GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
export GOOGLE_CLOUD_PROJECT=your-project-id
```

### Configuration File

```yaml
# config.yaml
infrastructure:
  providers:
    aws:
      region: us-east-1
      instance_type: t3.medium
    azure:
      location: East US
      vm_size: Standard_B2s
    gcp:
      zone: us-central1-a
      machine_type: n1-standard-2
  
  monitoring:
    prometheus:
      retention: 30d
      scrape_interval: 15s
    grafana:
      admin_user: admin
      admin_password: ${GRAFANA_PASSWORD}
  
  deployment:
    strategy: rolling
    max_unavailable: 1
    max_surge: 1
```

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/pynomaly/infrastructure.git
cd infrastructure

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .[dev,aws,azure,gcp,kubernetes,monitoring]

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration            # Integration tests only
pytest -m "not slow"             # Skip slow tests

# Run with coverage
pytest --cov=infrastructure --cov-report=html
```

### Code Quality

```bash
# Format code
black src tests
isort src tests

# Lint code
ruff check src tests

# Type checking
mypy src
```

## Deployment

### AWS Deployment

```python
from infrastructure.providers.aws import AWSDeployment

deployment = AWSDeployment(
    region='us-east-1',
    vpc_id='vpc-12345678',
    subnet_ids=['subnet-12345678', 'subnet-87654321']
)

# Deploy ECS cluster
deployment.deploy_ecs_cluster(
    cluster_name='pynomaly-cluster',
    instance_type='t3.medium',
    desired_capacity=3
)

# Deploy RDS database
deployment.deploy_rds_instance(
    db_instance_identifier='pynomaly-db',
    engine='postgres',
    instance_class='db.t3.micro'
)
```

### Kubernetes Deployment

```python
from infrastructure.orchestration.kubernetes import KubernetesDeployment

k8s = KubernetesDeployment(
    cluster_name='pynomaly-cluster',
    namespace='pynomaly'
)

# Deploy application
k8s.deploy_application(
    name='pynomaly-api',
    image='pynomaly/api:latest',
    replicas=3,
    port=8000
)

# Deploy database
k8s.deploy_postgresql(
    name='pynomaly-db',
    storage_size='10Gi',
    password='${DB_PASSWORD}'
)
```

## Monitoring

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'pynomaly-api'
    static_configs:
      - targets: ['localhost:8000']
  
  - job_name: 'pynomaly-worker'
    static_configs:
      - targets: ['localhost:8001']
```

### Grafana Dashboards

Pre-configured dashboards for:
- Application performance metrics
- Infrastructure resource utilization
- Database performance
- Error rates and response times

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- Documentation: https://infrastructure.pynomaly.com
- Issues: https://github.com/pynomaly/infrastructure/issues
- Discussions: https://github.com/pynomaly/infrastructure/discussions

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes and version history.