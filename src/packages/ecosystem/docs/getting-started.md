# Getting Started with Ecosystem Integration Framework

Welcome to the Ecosystem Integration Framework - a comprehensive solution for managing partnerships and integrations across your MLOps platform. This guide will help you get up and running quickly with partner integrations.

## Overview

The Ecosystem Integration Framework provides:

- **Standardized Integration Patterns**: Consistent interfaces for all platform integrations
- **Partner Management**: Tools for managing strategic partnerships and contracts
- **Health Monitoring**: Real-time monitoring of integration health and performance
- **Pluggable Architecture**: Easy addition of new integrations and capabilities
- **Production-Ready**: Enterprise-grade security, monitoring, and governance

## Quick Setup

### 1. Installation

```bash
# Install the base package
pip install ecosystem

# Install with specific integration support
pip install ecosystem[mlops,data,monitoring]

# Install all integrations
pip install ecosystem[all]
```

### 2. Basic Configuration

Create a configuration file `ecosystem_config.yaml`:

```yaml
# Core configuration
ecosystem:
  name: "production-mlops-platform"
  environment: "production"
  
  # Monitoring settings
  monitoring:
    health_check_interval_minutes: 5
    metrics_collection_enabled: true
  
  # Security settings
  security:
    encryption_at_rest: true
    audit_logging: true

# Partner configurations
partners:
  databricks:
    name: "databricks-prod"
    platform: "databricks"
    tier: "enterprise"
    endpoint: "https://your-workspace.cloud.databricks.com"
    credentials:
      access_token: "${DATABRICKS_TOKEN}"
      cluster_id: "${DATABRICKS_CLUSTER_ID}"
    capabilities:
      - "data_processing"
      - "ml_training"
      - "experiment_tracking"
  
  snowflake:
    name: "snowflake-warehouse"
    platform: "snowflake"
    tier: "enterprise"
    credentials:
      account: "${SNOWFLAKE_ACCOUNT}"
      user: "${SNOWFLAKE_USER}"
      password: "${SNOWFLAKE_PASSWORD}"
      warehouse: "COMPUTE_WH"
      database: "MLOPS_DB"
    capabilities:
      - "data_storage"
      - "data_analytics"
      - "feature_store"
```

### 3. Initialize the Registry

```python
import asyncio
from ecosystem.management.registry import PartnerRegistry
from ecosystem.connectors.databricks import DatabricksIntegration
from ecosystem.connectors.snowflake import SnowflakeIntegration
from ecosystem.core.interfaces import IntegrationConfig, PartnerTier

async def setup_ecosystem():
    # Initialize partner registry
    registry = PartnerRegistry()
    
    # Configure Databricks integration
    databricks_config = IntegrationConfig(
        name="databricks-prod",
        platform="databricks",
        endpoint="https://your-workspace.cloud.databricks.com",
        credentials={
            "access_token": "your-token",
            "cluster_id": "your-cluster-id"
        }
    )
    
    databricks_integration = DatabricksIntegration(databricks_config)
    
    # Register partner
    await registry.register_partner(
        name="databricks-prod",
        partner=databricks_integration,  # Will be wrapped in partner interface
        integration=databricks_integration,
        tier=PartnerTier.ENTERPRISE
    )
    
    # Check health
    health = await registry.check_health("databricks-prod")
    print(f"Databricks health: {health}")
    
    return registry

# Run the setup
registry = asyncio.run(setup_ecosystem())
```

## Core Concepts

### 1. Integration Interface

All integrations implement the `IntegrationInterface` which provides:

```python
from ecosystem.core.interfaces import IntegrationInterface

class MyIntegration(IntegrationInterface):
    async def connect(self) -> bool:
        """Establish connection to external platform"""
        pass
    
    async def send_data(self, data, destination, format_type=None, options=None) -> bool:
        """Send data to external platform"""
        pass
    
    async def receive_data(self, source, format_type=None, options=None) -> Any:
        """Receive data from external platform"""
        pass
```

### 2. Partner Management

Partners are managed through contracts and tiers:

```python
from ecosystem.core.interfaces import PartnerContract, PartnerTier, PartnerCapability

contract = PartnerContract(
    contract_id="DATABRICKS-2024-001",
    partner_name="databricks-prod",
    tier=PartnerTier.ENTERPRISE,
    capabilities={
        PartnerCapability.DATA_PROCESSING,
        PartnerCapability.ML_TRAINING,
        PartnerCapability.EXPERIMENT_TRACKING
    },
    monthly_api_limit=1000000,
    uptime_sla_percentage=99.9
)
```

### 3. Health Monitoring

Continuous monitoring of integration health:

```python
# Check individual partner health
health = await registry.check_health("databricks-prod")

# Check all partners
health_summary = await registry.get_health_summary()
print(f"Overall health: {health_summary['health_percentage']:.1f}%")

# Set up health callbacks
def on_health_change(partner_name: str, health: ConnectionHealth):
    print(f"Partner {partner_name} health changed to {health}")

registry.add_health_callback(on_health_change)
```

## Common Use Cases

### 1. Data Pipeline Integration

```python
# Read data from Snowflake
snowflake_integration = registry.get_integration("snowflake-warehouse")
raw_data = await snowflake_integration.receive_data(
    source="raw_events",
    options={"where": "created_at > '2024-01-01'", "limit": 10000}
)

# Process data in Databricks
databricks_integration = registry.get_integration("databricks-prod")
processed_data = await databricks_integration.send_data(
    data=raw_data,
    destination="dbfs:/data/processed/events.parquet",
    format_type="parquet"
)
```

### 2. ML Experiment Tracking

```python
# Create experiment in Databricks MLflow
experiment_id = await databricks_integration.create_experiment(
    name="fraud-detection-v2",
    description="Fraud detection model with new features"
)

# Start training run
run_id = await databricks_integration.create_run(
    experiment_id=experiment_id,
    run_name="xgboost-baseline"
)

# Log metrics and parameters
await databricks_integration.log_parameters(run_id, {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100
})

await databricks_integration.log_metrics(run_id, {
    "accuracy": 0.92,
    "precision": 0.89,
    "recall": 0.94
})
```

### 3. Model Deployment Pipeline

```python
# Register model in MLflow
model_version = await databricks_integration.register_model(
    model_name="fraud-detector",
    run_id=run_id,
    artifact_path="model"
)

# Promote to staging
await databricks_integration.update_model_version_stage(
    model_name="fraud-detector",
    version=model_version,
    stage="Staging"
)

# Deploy to production after validation
await databricks_integration.update_model_version_stage(
    model_name="fraud-detector",
    version=model_version,
    stage="Production"
)
```

## Advanced Features

### 1. Event-Driven Integration

```python
from ecosystem.core.interfaces import Event, EventType, EventPriority

# Create custom event
event = Event(
    name="model_drift_detected",
    type=EventType.MONITORING,
    priority=EventPriority.HIGH,
    payload={
        "model_name": "fraud-detector",
        "drift_score": 0.85,
        "threshold": 0.7
    }
)

# Publish event
await integration.publish_event(event)

# Subscribe to events
def handle_drift_event(event: Event):
    print(f"Model drift detected: {event.payload}")

subscription_id = await integration.subscribe_to_events(
    event_types=[EventType.MONITORING],
    callback=handle_drift_event
)
```

### 2. Custom Integration Development

```python
from ecosystem.templates.data_platform.base_data_integration import DataPlatformTemplate

class CustomIntegration(DataPlatformTemplate):
    async def _create_platform_client(self):
        # Implement platform-specific client creation
        pass
    
    async def _authenticate_client(self) -> bool:
        # Implement authentication logic
        pass
    
    async def _validate_platform_config(self, config: IntegrationConfig) -> bool:
        # Validate configuration
        return True
```

### 3. Partnership Analytics

```python
# Collect metrics across all partners
metrics = await registry.collect_all_metrics()

# Generate usage report
report = await registry.generate_usage_report(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31)
)

print(f"Total API calls: {report['summary']['total_api_calls']}")
print(f"Total cost: ${report['summary']['total_cost_usd']:.2f}")
```

## Best Practices

### 1. Configuration Management

- Use environment variables for sensitive credentials
- Implement configuration validation
- Use different configurations for different environments

### 2. Error Handling

```python
from ecosystem.core.interfaces import IntegrationError

try:
    await integration.send_data(data, "destination")
except IntegrationError as e:
    logger.error(f"Integration failed: {e}")
    # Implement fallback logic
```

### 3. Monitoring and Alerting

```python
# Set up comprehensive monitoring
async def monitor_integrations():
    while True:
        health_summary = await registry.get_health_summary()
        
        if health_summary['health_percentage'] < 95.0:
            # Send alert
            await send_alert(f"Integration health below threshold: {health_summary}")
        
        await asyncio.sleep(300)  # Check every 5 minutes
```

### 4. Security

- Rotate credentials regularly
- Use least-privilege access principles
- Enable audit logging
- Encrypt sensitive data at rest and in transit

## Next Steps

- [Integration Templates Guide](integration-templates.md)
- [Partner Management](partner-management.md)
- [Monitoring and Observability](monitoring.md)
- [Security Best Practices](security.md)
- [API Reference](api-reference.md)

## Support

For questions and support:

- 📚 [Documentation](https://docs.company.com/ecosystem)
- 💬 [Community Forum](https://community.company.com/ecosystem)
- 🐛 [Issue Tracker](https://github.com/company/mlops-platform/issues)
- 📧 [Email Support](mailto:platform-support@company.com)