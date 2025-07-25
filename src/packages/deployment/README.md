# Production Deployment Configurations

This directory contains production-ready deployment configurations for the hexagonal architecture implementation across all packages.

## Architecture Overview

Our hexagonal architecture provides clean separation of concerns and flexible deployment options:

- **Data Quality Package**: Complete hexagonal architecture with 9+ domain interfaces
- **MLOps Package**: Partial implementation with service discovery and configuration management  
- **Machine Learning Package**: Basic structure ready for completion
- **Anomaly Detection Package**: Traditional structure, candidate for hexagonal migration

## Deployment Strategies

### 1. Container-Based Deployment
- Docker configurations for each package
- Docker Compose for local development and testing
- Kubernetes manifests for production orchestration

### 2. Microservices Architecture
- Independent deployment of each package
- Service discovery and configuration management
- Load balancing and health monitoring

### 3. Hexagonal Architecture Benefits
- **Adapter Swapping**: Easy switching between file-based, database, and cloud adapters
- **Environment Configuration**: Different configurations for dev/staging/production
- **Graceful Degradation**: Automatic fallback to stub implementations
- **Independent Scaling**: Each package can scale independently

## Deployment Configurations

```
deployment/
├── docker/                 # Docker configurations
│   ├── data-quality/       # Data Quality service
│   ├── mlops/             # MLOps service
│   ├── machine-learning/  # ML service  
│   └── anomaly-detection/ # Anomaly Detection service
├── kubernetes/            # Kubernetes manifests
│   ├── base/              # Base configurations
│   ├── overlays/          # Environment-specific overlays
│   │   ├── development/
│   │   ├── staging/
│   │   └── production/
├── compose/               # Docker Compose files
│   ├── development.yml
│   ├── staging.yml
│   └── production.yml
├── terraform/             # Infrastructure as Code
│   ├── aws/
│   ├── azure/
│   └── gcp/
└── scripts/              # Deployment scripts
    ├── deploy.sh
    ├── health-check.sh
    └── rollback.sh
```

## Environment-Specific Configurations

### Development
- File-based adapters for all services
- Stub implementations for external dependencies
- Local storage for all data
- Debug logging enabled

### Staging  
- Mix of real and stub adapters
- Shared storage systems
- Integration testing environment
- Performance monitoring

### Production
- Real adapters for all external systems
- Cloud storage and databases
- Comprehensive monitoring and alerting
- High availability and disaster recovery

## Getting Started

1. **Local Development**:
   ```bash
   docker-compose -f compose/development.yml up
   ```

2. **Staging Deployment**:
   ```bash
   kubectl apply -k kubernetes/overlays/staging
   ```

3. **Production Deployment**:
   ```bash
   ./scripts/deploy.sh production
   ```

## Monitoring and Observability

- Health checks for all services
- Metrics collection and alerting
- Distributed tracing
- Log aggregation and analysis
- Performance monitoring

## Security

- Secret management
- Network security policies
- Authentication and authorization
- Audit logging
- Vulnerability scanning