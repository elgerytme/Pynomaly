# Domain-Based Architecture

## Overview

The anomaly_detection package architecture is organized by business domains with clear boundaries:

1. **Domain Packages** - Business logic organized by domain (ai/, data/)
2. **Enterprise Services** - Cross-cutting enterprise concerns  
3. **Platform Integrations** - External tool connectors
4. **Configurations** - Application composition

## Critical Rule: NO "CORE" PACKAGES

**FORBIDDEN:** `src/packages/core/` - There is no universal "core" across domains. Each domain contains its own business logic.

## Directory Structure

```
src/packages/
├── ai/                             # AI/ML business domain
│   ├── anomaly_detection/          # Anomaly detection domain logic
│   ├── machine_learning/           # General ML domain logic  
│   ├── mlops/                      # MLOps domain logic
│   └── data_science/               # Data science domain logic
├── data/                           # Data business domain
│   ├── quality/                    # Data quality domain logic
│   ├── observability/              # Data observability domain logic
│   ├── profiling/                  # Data profiling domain logic
│   ├── transformation/             # Data transformation domain logic
│   └── lineage/                    # Data lineage domain logic
│
├── enterprise/                     # Cross-cutting enterprise services
│   ├── auth/                      # Authentication & authorization
│   ├── multi_tenancy/             # Multi-tenant architecture
│   ├── operations/                # Monitoring, alerting, SRE
│   ├── scalability/               # Distributed computing
│   ├── governance/                # Audit, compliance
│   └── security/                  # Enterprise security
│
├── integrations/                   # External monorepo connectors
│   ├── mlops/                     # MLOps monorepos
│   │   ├── mlflow_integration.py  # MLflow connector
│   │   ├── kubeflow_integration.py # Kubeflow connector
│   │   └── wandb_integration.py   # W&B connector
│   ├── monitoring/                # Monitoring monorepos
│   │   ├── datadog_integration.py # Datadog connector
│   │   └── newrelic_integration.py # New Relic connector
│   └── cloud/                     # Cloud providers
│       ├── aws/                   # AWS services
│       ├── azure/                 # Azure services
│       └── gcp/                   # GCP services
│
└── configurations/                 # Application composition
    ├── basic/                     # Open source configs
    │   ├── mlops_basic/           # Basic MLOps
    │   └── anomaly_detection_basic/
    ├── enterprise/                # Enterprise configs
    │   ├── mlops_enterprise/      # Enterprise MLOps
    │   └── anomaly_detection_enterprise/
    └── custom/                    # Custom deployments
        ├── mlops_aws_production/
        └── mlops_k8s_staging/
```

## Architecture Principles

### 1. Domain Boundaries

**Domain Packages** (ai/, data/) contain only business logic for that domain:
- Business entities and rules specific to the domain
- Domain services and operations  
- No enterprise features
- No monorepo integrations
- No cross-domain dependencies

**Enterprise Services** handle cross-cutting concerns:
- Authentication & authorization
- Multi-tenancy & tenant isolation
- Monitoring, alerting, SRE
- Audit logging & compliance
- Distributed scalability
- Security & encryption

**Integrations** connect to external monorepos:
- MLOps monorepos (MLflow, Kubeflow)
- Monitoring services (Datadog, New Relic)
- Cloud providers (AWS, Azure, GCP)
- Storage systems
- Message queues

**Configurations** compose everything:
- Wire services together
- Define deployment modes
- Handle environment differences
- Manage dependencies

### 2. Dependency Direction

Dependencies flow in one direction:

```
Configurations → Enterprise Services
Configurations → Integrations  
Configurations → Domain Packages
```

**Never:**
- Domain packages depend on enterprise services
- Domain packages depend on integrations
- Enterprise services depend on integrations
- Integrations depend on each other
- Cross-domain dependencies (ai/ ↔ data/)

### 3. Optional Dependencies

All non-domain dependencies are optional:
- Enterprise features are optional
- Platform integrations are optional
- Missing dependencies are handled gracefully
- Basic domain functionality always works

## Usage Examples

### Basic Open Source Deployment

```python
from configurations.basic.mlops_basic import create_basic_mlops_config

# Simple setup with no enterprise features
config = create_basic_mlops_config(data_path="./data")
services = config.create_mlops_services()

# Use core functionality
experiment_tracker = services["experiment_tracker"]
model_registry = services["model_registry"]
```

### Enterprise Deployment

```python
from configurations.enterprise.mlops_enterprise import create_enterprise_mlops_config

# Full enterprise stack
config = create_enterprise_mlops_config(
    data_path="/opt/mlops",
    auth_config={"type": "saml", "enable_rbac": True},
    mlflow_config={"enabled": True, "tracking_uri": "https://mlflow.company.com"},
    monitoring_config={"datadog": {"api_key": "..."}}
)

services = await config.start_services()

# Access enterprise services
auth = services["auth"] 
tenant_manager = services["multi_tenant"]
operations = services["operations"]

# Access core functionality with enterprise features
experiment_tracker = services["experiment_tracker"]
model_registry = services["model_registry"]
mlflow = services["mlflow"]
```

### Custom Configuration

```python
# Create your own configuration mixing components
from configurations.basic.mlops_basic import BasicMLOpsConfiguration
from enterprise.operations import EnterpriseOperationsService
from integrations.monitoring import PrometheusIntegration

class CustomMLOpsConfig(BasicMLOpsConfiguration):
    def create_mlops_services(self):
        services = super().create_mlops_services()
        
        # Add monitoring but not auth
        services["monitoring"] = EnterpriseOperationsService()
        services["prometheus"] = PrometheusIntegration()
        
        return services
```

## Benefits

### 1. Clear Boundaries
- No confusion about what's "enterprise" vs "domain-specific"
- Platform integrations are clearly separated from business logic
- Enterprise features are cross-cutting concerns, not domain-specific

### 2. Flexible Deployment
- Choose exactly what features you need
- Easy to upgrade from basic to enterprise
- Environment-specific configurations
- Custom deployments possible

### 3. Maintainability
- Each layer has single responsibility
- Dependencies are clearly defined
- Easy to test each layer independently
- Changes isolated to appropriate layers

### 4. Extensibility
- Add new enterprise services without touching core
- Add new monorepo integrations independently
- Create new configurations easily
- Mix and match components as needed

## Migration from Old Architecture

The previous architecture mixed domain logic with enterprise features and monorepo integrations. The new architecture separates these concerns:

**Old:**
```
enterprise_mlops/                   # Mixed everything
├── domain/entities/mlops.py        # Enterprise entities
├── infrastructure/mlops/mlflow/    # Platform integrations
└── infrastructure/monitoring/      # More monorepo integrations
```

**New:**
```
core/ai/machine_learning/mlops/     # Domain logic only
enterprise/                         # Cross-cutting services
integrations/                       # Platform connectors
configurations/                     # Composition layer
```

Use configuration packages to get the same functionality with better separation of concerns.