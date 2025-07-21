# Configuration-Based Architecture

This directory contains deployment configurations that wire together:
- **Core domain packages** (business logic)
- **Enterprise cross-cutting services** (auth, ops, multi-tenancy)  
- **Platform integrations** (MLflow, Kubeflow, monitoring)

## Architecture Overview

```
packages/
├── core/                    # Domain logic only
│   ├── ai/machine_learning/ # MLOps domain
│   └── data/processing/     # Data domain
│
├── enterprise/              # Cross-cutting enterprise services
│   ├── auth/               # RBAC, SSO, SAML
│   ├── multi_tenancy/      # Tenant isolation
│   ├── operations/         # Monitoring, SRE
│   ├── scalability/        # Distributed computing
│   └── security/           # Enterprise security
│
├── integrations/           # External platform connectors
│   ├── mlops/             # MLflow, Kubeflow, W&B
│   ├── monitoring/        # Datadog, New Relic
│   └── cloud/             # AWS, Azure, GCP
│
└── configurations/        # Application composition
    ├── basic/             # Open source configs
    ├── enterprise/        # Enterprise configs
    └── custom/            # Custom deployments
```

## Configuration Types

### Basic Configurations (`basic/`)
Open source deployments with minimal dependencies:
- Local file storage
- No authentication
- Basic logging
- Single node operation

**Example:**
```python
from configurations.basic.mlops_basic import create_basic_mlops_config

config = create_basic_mlops_config(data_path="./data")
services = config.create_mlops_services()
```

### Enterprise Configurations (`enterprise/`)
Full enterprise feature deployments:
- Multi-tenant architecture
- Enterprise authentication (SAML/SSO)
- Production monitoring & alerting
- Distributed scalability
- Platform integrations

**Example:**
```python
from configurations.enterprise.mlops_enterprise import create_production_config

config = create_production_config(data_path="/opt/mlops")
services = await config.start_services()
```

### Custom Configurations (`custom/`)
Specialized deployment configurations:
- Cloud-specific setups
- Environment-specific configs
- Mixed enterprise/basic features

## Benefits

### 1. Separation of Concerns
- **Domain packages** focus purely on business logic
- **Enterprise services** handle cross-cutting concerns
- **Integrations** are modular and swappable
- **Configurations** compose everything cleanly

### 2. Flexible Deployment
- Choose exactly what features you need
- Easy to upgrade from basic to enterprise
- Mix and match components
- Environment-specific optimizations

### 3. Clear Dependencies
- No confusion about what's "enterprise" vs "domain-specific"
- Optional dependencies are truly optional
- Clean import paths
- Easy to test each layer independently

## Usage Examples

### Basic MLOps (Open Source)
```python
# Simple, no enterprise features
from configurations.basic.mlops_basic import create_basic_mlops_config

config = create_basic_mlops_config()
services = config.create_mlops_services()

experiment_tracker = services["experiment_tracker"]
model_registry = services["model_registry"]
```

### Enterprise MLOps (Full Features)
```python
# Full enterprise stack
from configurations.enterprise.mlops_enterprise import create_enterprise_mlops_config

config = create_enterprise_mlops_config(
    auth_config={"type": "saml", "enable_rbac": True},
    mlflow_config={"enabled": True, "tracking_uri": "https://mlflow.company.com"},
    monitoring_config={"datadog": {"api_key": "..."}}
)

services = await config.start_services()

# Access enterprise services
auth = services["auth"]
tenant_manager = services["multi_tenant"]
operations = services["operations"]

# Access core MLOps with enterprise features
experiment_tracker = services["experiment_tracker"] 
model_registry = services["model_registry"]
mlflow = services["mlflow"]
```

### Custom Deployment
```python
# Mix basic MLOps with some enterprise features
from configurations.basic.mlops_basic import BasicMLOpsConfiguration
from enterprise.operations import EnterpriseOperationsService

class CustomConfig(BasicMLOpsConfiguration):
    def create_mlops_services(self):
        services = super().create_mlops_services()
        
        # Add monitoring but not auth
        services["monitoring"] = EnterpriseOperationsService()
        
        return services
```

## Adding New Configurations

1. **Create configuration package** in appropriate directory
2. **Import required services** from core, enterprise, integrations
3. **Compose services** in configuration class
4. **Provide factory functions** for common setups
5. **Add pyproject.toml** with specific dependencies

Example structure:
```python
class MyCustomConfiguration:
    def __init__(self, **config):
        pass
    
    def create_domain_services(self):
        # Import and configure core domain services
        pass
    
    def create_enterprise_services(self):
        # Import and configure enterprise services
        pass
    
    def create_integrations(self):
        # Import and configure platform integrations
        pass
    
    def create_all_services(self):
        # Compose everything together
        pass
```

This approach provides maximum flexibility while maintaining clear boundaries between different types of functionality.