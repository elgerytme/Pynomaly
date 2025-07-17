# Contributing to Infrastructure Package

Thank you for your interest in contributing to the Infrastructure package! This package provides comprehensive infrastructure and operations capabilities for multi-cloud deployment, monitoring, and automation.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Infrastructure-Specific Guidelines](#infrastructure-specific-guidelines)
- [Testing Requirements](#testing-requirements)
- [Documentation Standards](#documentation-standards)
- [Pull Request Process](#pull-request-process)
- [Security Considerations](#security-considerations)
- [Community](#community)

## Code of Conduct

This project adheres to our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Terraform 1.5+
- kubectl (for Kubernetes operations)
- Cloud CLI tools (aws-cli, azure-cli, gcloud)
- Access to test cloud environments

### Repository Setup

```bash
# Clone the repository
git clone https://github.com/your-org/monorepo.git
cd monorepo/src/packages/ops/infrastructure

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,test,aws,azure,gcp,kubernetes,monitoring]"

# Install pre-commit hooks
pre-commit install
```

### Cloud Provider Setup

**AWS Configuration:**
```bash
# Configure AWS CLI
aws configure

# Set environment variables
export AWS_DEFAULT_REGION=us-east-1
export AWS_PROFILE=infrastructure-dev
```

**Azure Configuration:**
```bash
# Login to Azure
az login

# Set subscription
az account set --subscription "your-subscription-id"

# Create service principal for testing
az ad sp create-for-rbac --name "infrastructure-test" --role contributor
```

**GCP Configuration:**
```bash
# Authenticate with GCP
gcloud auth login
gcloud auth application-default login

# Set project
gcloud config set project your-project-id

# Create service account for testing
gcloud iam service-accounts create infrastructure-test
```

## Development Environment

### IDE Configuration

Recommended VS Code extensions:
- Python
- Docker
- Kubernetes
- Terraform
- Azure Resource Manager Tools
- AWS Toolkit

### Environment Variables

Create a `.env` file for local development:

```bash
# Infrastructure Configuration
INFRA_ENVIRONMENT=development
INFRA_LOG_LEVEL=DEBUG

# Cloud Provider Configuration
AWS_PROFILE=infrastructure-dev
AZURE_SUBSCRIPTION_ID=your-subscription-id
GOOGLE_CLOUD_PROJECT=your-project-id

# Kubernetes Configuration
KUBECONFIG=~/.kube/config
DEFAULT_NAMESPACE=infrastructure-dev

# Monitoring Configuration
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3000
GRAFANA_ADMIN_PASSWORD=admin

# Terraform Configuration
TF_VAR_environment=development
TF_VAR_project_name=infrastructure-test
```

### Local Testing Infrastructure

```bash
# Start local testing stack
docker-compose -f docker/local-stack.yml up -d

# This starts:
# - LocalStack (AWS simulation)
# - Azurite (Azure Storage simulation)
# - Prometheus (monitoring)
# - Grafana (dashboards)
# - MinIO (S3-compatible storage)
```

## Infrastructure-Specific Guidelines

### Architecture Principles

This package follows infrastructure best practices:

1. **Infrastructure as Code**: Everything defined in code (Terraform, Kubernetes manifests)
2. **Multi-Cloud Support**: Provider-agnostic abstractions with cloud-specific implementations
3. **Immutable Infrastructure**: Treat infrastructure as immutable, replace rather than modify
4. **Security by Default**: Secure configurations and least privilege access
5. **Observability**: Comprehensive monitoring, logging, and alerting

### Code Organization

```
infrastructure/
├── providers/              # Cloud provider implementations
│   ├── aws/               # AWS-specific code
│   │   ├── compute/       # EC2, ECS, Lambda
│   │   ├── storage/       # S3, EBS, EFS
│   │   ├── networking/    # VPC, ALB, Route53
│   │   └── security/      # IAM, Security Groups
│   ├── azure/             # Azure-specific code
│   └── gcp/               # GCP-specific code
├── orchestration/         # Container orchestration
│   ├── kubernetes/        # K8s manifests and operators
│   └── docker/            # Docker compositions
├── monitoring/            # Observability stack
│   ├── prometheus/        # Metrics collection
│   ├── grafana/          # Dashboards and visualization
│   └── alerting/         # Alert rules and notifications
├── deployment/           # Deployment automation
├── security/            # Security configurations
└── templates/           # Infrastructure templates
```

### Provider Implementation Pattern

When adding new cloud provider support:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from infrastructure.base import BaseProvider, DeploymentResult

class ComputeProvider(ABC):
    """Abstract base class for compute providers."""
    
    @abstractmethod
    async def create_instance(
        self,
        name: str,
        instance_type: str,
        image_id: str,
        **kwargs
    ) -> DeploymentResult:
        """Create a compute instance."""
        pass
    
    @abstractmethod
    async def delete_instance(self, instance_id: str) -> bool:
        """Delete a compute instance."""
        pass

class AWSComputeProvider(ComputeProvider):
    """AWS EC2 implementation."""
    
    def __init__(self, session: boto3.Session):
        self.ec2 = session.client('ec2')
    
    async def create_instance(
        self,
        name: str,
        instance_type: str,
        image_id: str,
        **kwargs
    ) -> DeploymentResult:
        """Create EC2 instance with proper tagging and security."""
        try:
            response = self.ec2.run_instances(
                ImageId=image_id,
                MinCount=1,
                MaxCount=1,
                InstanceType=instance_type,
                SecurityGroupIds=kwargs.get('security_groups', []),
                SubnetId=kwargs.get('subnet_id'),
                TagSpecifications=[{
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': name},
                        {'Key': 'Environment', 'Value': kwargs.get('environment', 'dev')},
                        {'Key': 'ManagedBy', 'Value': 'infrastructure-package'}
                    ]
                }]
            )
            
            instance_id = response['Instances'][0]['InstanceId']
            
            return DeploymentResult(
                success=True,
                resource_id=instance_id,
                metadata=response['Instances'][0]
            )
            
        except Exception as e:
            return DeploymentResult(
                success=False,
                error=str(e)
            )
```

### Infrastructure as Code

All infrastructure must be defined as code:

**Terraform Modules:**
```hcl
# modules/compute/main.tf
resource "aws_instance" "main" {
  ami           = var.ami_id
  instance_type = var.instance_type
  subnet_id     = var.subnet_id
  
  vpc_security_group_ids = var.security_group_ids
  
  tags = merge(var.tags, {
    Name        = var.instance_name
    Environment = var.environment
    ManagedBy   = "terraform"
  })
  
  root_block_device {
    encrypted             = true
    volume_type          = "gp3"
    volume_size          = var.root_volume_size
    delete_on_termination = true
  }
  
  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"  # Require IMDSv2
  }
}
```

**Kubernetes Manifests:**
```yaml
# templates/kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Values.name }}
  namespace: {{ .Values.namespace }}
  labels:
    app: {{ .Values.name }}
    version: {{ .Values.version }}
spec:
  replicas: {{ .Values.replicas }}
  selector:
    matchLabels:
      app: {{ .Values.name }}
  template:
    metadata:
      labels:
        app: {{ .Values.name }}
        version: {{ .Values.version }}
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 65534
        fsGroup: 65534
      containers:
      - name: {{ .Values.name }}
        image: {{ .Values.image }}
        imagePullPolicy: Always
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          capabilities:
            drop:
            - ALL
        resources:
          limits:
            memory: {{ .Values.resources.limits.memory }}
            cpu: {{ .Values.resources.limits.cpu }}
          requests:
            memory: {{ .Values.resources.requests.memory }}
            cpu: {{ .Values.resources.requests.cpu }}
```

### Security Standards

All infrastructure components must follow security best practices:

**Access Control:**
- Use least privilege principle
- Implement role-based access control (RBAC)
- Regular access reviews and rotation

**Network Security:**
- Network segmentation with security groups/NSGs
- VPN or private connectivity for management
- TLS encryption for all communications

**Data Protection:**
- Encryption at rest and in transit
- Secure key management
- Data classification and handling

**Monitoring and Auditing:**
- Comprehensive audit logging
- Real-time security monitoring
- Incident response procedures

## Testing Requirements

### Test Categories

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test cloud provider integrations
3. **Infrastructure Tests**: Test infrastructure provisioning
4. **Security Tests**: Test security configurations
5. **Performance Tests**: Test deployment speed and resource efficiency

### Test Structure

```bash
tests/
├── unit/                    # Unit tests
│   ├── providers/          # Provider implementations
│   ├── orchestration/      # Container orchestration
│   └── monitoring/         # Monitoring components
├── integration/            # Integration tests
│   ├── aws/               # AWS integration tests
│   ├── azure/             # Azure integration tests
│   └── gcp/               # GCP integration tests
├── infrastructure/        # Infrastructure tests
│   ├── terraform/         # Terraform plan validation
│   └── kubernetes/        # K8s manifest validation
├── security/             # Security tests
└── performance/          # Performance benchmarks
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific provider tests
pytest tests/integration/aws/
pytest tests/integration/azure/
pytest tests/integration/gcp/

# Run infrastructure tests (requires cloud access)
pytest tests/infrastructure/ --cloud-tests

# Run security tests
pytest tests/security/

# Run with coverage
pytest --cov=infrastructure --cov-report=html

# Run performance benchmarks
pytest tests/performance/ --benchmark-only
```

### Test Requirements

- **Coverage**: Minimum 85% code coverage
- **Isolation**: Use mocking for external services when possible
- **Cleanup**: Always clean up created resources
- **Idempotency**: Tests should be idempotent
- **Documentation**: Clear test documentation and assertions

### Mock Guidelines

```python
import pytest
from unittest.mock import AsyncMock, patch
from infrastructure.providers.aws import AWSComputeProvider

@pytest.fixture
def mock_ec2_client():
    with patch('boto3.Session') as mock_session:
        mock_client = AsyncMock()
        mock_session.return_value.client.return_value = mock_client
        
        # Mock successful instance creation
        mock_client.run_instances.return_value = {
            'Instances': [{
                'InstanceId': 'i-1234567890abcdef0',
                'State': {'Name': 'pending'}
            }]
        }
        
        yield mock_client

@pytest.mark.asyncio
async def test_create_instance(mock_ec2_client):
    provider = AWSComputeProvider(session=mock_session)
    
    result = await provider.create_instance(
        name="test-instance",
        instance_type="t3.micro",
        image_id="ami-12345678"
    )
    
    assert result.success
    assert result.resource_id == "i-1234567890abcdef0"
    mock_ec2_client.run_instances.assert_called_once()
```

## Documentation Standards

### Code Documentation

- **Docstrings**: All public functions and classes must have comprehensive docstrings
- **Type Hints**: Use type hints for all function signatures
- **Infrastructure Diagrams**: Include architecture diagrams for complex setups
- **Configuration Examples**: Provide comprehensive configuration examples

```python
from typing import Optional, Dict, Any, List
from infrastructure.types import DeploymentResult, ResourceConfig

class InfrastructureManager:
    """Infrastructure manager for multi-cloud deployments.
    
    The InfrastructureManager provides a unified interface for deploying
    and managing infrastructure across multiple cloud providers. It abstracts
    provider-specific implementations and provides common operations.
    
    Examples:
        Basic deployment:
        
        >>> manager = InfrastructureManager()
        >>> result = await manager.deploy_stack(
        ...     provider="aws",
        ...     region="us-east-1",
        ...     config=stack_config
        ... )
        
        Multi-cloud deployment:
        
        >>> results = await manager.deploy_multi_cloud([
        ...     {"provider": "aws", "region": "us-east-1"},
        ...     {"provider": "azure", "region": "eastus"},
        ... ])
    """
    
    async def deploy_stack(
        self,
        provider: str,
        region: str,
        config: ResourceConfig,
        tags: Optional[Dict[str, str]] = None
    ) -> DeploymentResult:
        """Deploy infrastructure stack to specified cloud provider.
        
        Args:
            provider: Cloud provider name (aws, azure, gcp)
            region: Target deployment region
            config: Infrastructure configuration
            tags: Additional resource tags
            
        Returns:
            DeploymentResult with success status and resource information
            
        Raises:
            ProviderError: If provider is not supported
            DeploymentError: If deployment fails
            ValidationError: If configuration is invalid
        """
        # Implementation here
        pass
```

### Infrastructure Documentation

- **Deployment Guides**: Step-by-step deployment instructions
- **Architecture Decisions**: Document design choices and trade-offs
- **Operational Runbooks**: Troubleshooting and maintenance procedures
- **Security Guidelines**: Security configuration and best practices

## Pull Request Process

### Before Submitting

1. **Run Tests**: Ensure all tests pass locally
2. **Security Check**: Run security scans and validate configurations
3. **Documentation**: Update relevant documentation
4. **Infrastructure Validation**: Validate Terraform plans and K8s manifests
5. **Cost Impact**: Assess cost implications of changes

### Pull Request Template

```markdown
## Description
Brief description of infrastructure changes and motivation.

## Type of Change
- [ ] Bug fix
- [ ] New provider support
- [ ] Infrastructure enhancement
- [ ] Security improvement
- [ ] Documentation update
- [ ] Performance optimization

## Infrastructure Impact
- [ ] New cloud resources
- [ ] Modified existing resources
- [ ] Resource deletion
- [ ] Configuration changes
- [ ] Security policy changes

## Providers Affected
- [ ] AWS
- [ ] Azure
- [ ] Google Cloud Platform
- [ ] Kubernetes
- [ ] Monitoring stack

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests run
- [ ] Infrastructure tests validated
- [ ] Security tests passed
- [ ] Performance impact assessed

## Deployment Considerations
- [ ] Backward compatible
- [ ] Requires downtime
- [ ] Migration required
- [ ] Cost impact assessed
- [ ] Security review needed

## Security Checklist
- [ ] No hardcoded secrets
- [ ] Proper access controls
- [ ] Encryption enabled
- [ ] Network security configured
- [ ] Audit logging enabled

## Documentation
- [ ] Code comments updated
- [ ] Infrastructure diagrams updated
- [ ] Deployment guide updated
- [ ] Configuration examples provided
```

### Review Process

1. **Automated Checks**: CI/CD pipeline validates infrastructure
2. **Code Review**: Technical review by infrastructure team
3. **Security Review**: Security team review for security changes
4. **Cost Review**: Cost optimization team review for resource changes
5. **Final Approval**: Senior infrastructure engineer approval

## Security Considerations

### Secure Development

- **Secrets Management**: Never commit secrets or credentials
- **Least Privilege**: Apply minimal required permissions
- **Security Scanning**: Regular security scans of infrastructure code
- **Compliance**: Follow relevant compliance requirements

### Infrastructure Security

- **Encryption**: Enable encryption for all data at rest and in transit
- **Network Security**: Implement proper network segmentation
- **Access Control**: Use strong authentication and authorization
- **Monitoring**: Enable comprehensive security monitoring

### Security Testing

```python
def test_security_group_no_public_ssh():
    """Test that security groups don't allow public SSH access."""
    for sg in security_groups:
        for rule in sg.ingress_rules:
            if rule.port == 22 and rule.source == "0.0.0.0/0":
                pytest.fail(f"Security group {sg.name} allows public SSH")

def test_s3_bucket_encryption():
    """Test that S3 buckets have encryption enabled."""
    for bucket in s3_buckets:
        assert bucket.encryption_enabled, f"Bucket {bucket.name} not encrypted"
```

## Community

### Communication Channels

- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and ideas
- **Slack**: #infrastructure channel for real-time communication
- **Email**: infrastructure-team@yourorg.com for sensitive issues

### Getting Help

1. **Documentation**: Check package documentation and guides
2. **Examples**: Review example configurations and templates
3. **Issues**: Search existing issues before creating new ones
4. **Community**: Ask questions in GitHub Discussions

### Infrastructure Team

- **Infrastructure Lead**: [Name] - Overall architecture and strategy
- **Cloud Specialists**: Provider-specific expertise
- **Security Engineer**: Security reviews and compliance
- **SRE Team**: Operational support and monitoring

Thank you for contributing to the Infrastructure package! Your contributions help improve infrastructure operations for the entire platform.