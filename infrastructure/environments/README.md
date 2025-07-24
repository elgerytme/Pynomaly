# ML/MLOps Platform Environment Setup

This directory contains infrastructure configurations for different environments of the ML/MLOps platform.

## Environment Overview

### Development Environment
- **Purpose**: Local development and testing
- **Infrastructure**: Docker Compose
- **Components**: All services running locally
- **Data**: Sample/synthetic data
- **Monitoring**: Basic Prometheus/Grafana setup

### Staging Environment  
- **Purpose**: Pre-production testing and validation
- **Infrastructure**: Kubernetes cluster
- **Components**: Production-like deployment
- **Data**: Anonymized production data subset
- **Monitoring**: Full observability stack

## Quick Start

### Development Environment

1. **Prerequisites**
   ```bash
   # Install Docker and Docker Compose
   docker --version
   docker-compose --version
   ```

2. **Start Development Environment**
   ```bash
   cd infrastructure/environments/development
   docker-compose up -d
   ```

3. **Access Services**
   - Model Server: http://localhost:8000
   - Feature Store: http://localhost:8001  
   - Inference Engine: http://localhost:8002
   - Jupyter Lab: http://localhost:8888 (token: development_token)
   - Grafana: http://localhost:3000 (admin/admin_password)
   - Prometheus: http://localhost:9090

4. **Stop Environment**
   ```bash
   docker-compose down
   ```

### Staging Environment

1. **Prerequisites**
   ```bash
   # Kubernetes cluster access
   kubectl cluster-info
   
   # Helm (for dependency management)
   helm version
   ```

2. **Deploy to Staging**
   ```bash
   cd infrastructure/environments/staging
   
   # Create namespace and configs
   kubectl apply -f kubernetes/namespace.yaml
   
   # Deploy services
   kubectl apply -f kubernetes/deployments.yaml
   kubectl apply -f kubernetes/services.yaml
   
   # Verify deployment
   kubectl get pods -n mlops-staging
   ```

3. **Access Services**
   ```bash
   # Port forward for testing
   kubectl port-forward -n mlops-staging svc/inference-engine-service 8002:8002
   
   # Or via ingress (configure DNS)
   # https://mlops-staging.company.com/api/v1/inference/
   ```

## Environment Configuration

### Development
- **Database**: PostgreSQL (local container)
- **Cache**: Redis (local container)
- **Storage**: MinIO (local container)
- **Monitoring**: Prometheus + Grafana
- **Message Queue**: Kafka (local container)

### Staging
- **Database**: PostgreSQL (Kubernetes deployment)
- **Cache**: Redis (Kubernetes deployment)  
- **Storage**: Cloud storage (AWS S3/GCS/Azure Blob)
- **Monitoring**: Full observability stack
- **Message Queue**: Managed Kafka service

## Service Configuration

### Model Server (Port 8000)
- Model serving and management
- Health checks: `/health`, `/ready`
- Metrics: `/metrics`
- API docs: `/docs`

### Feature Store (Port 8001)  
- Feature engineering and serving
- Real-time and batch features
- Feature lineage and versioning
- API docs: `/docs`

### Inference Engine (Port 8002)
- Real-time model inference
- Batch prediction processing
- A/B testing integration
- API docs: `/docs`

## Monitoring and Observability

### Prometheus Metrics
- Service health and performance
- Business metrics
- Infrastructure metrics
- Custom application metrics

### Grafana Dashboards
- Service overview
- Model performance
- Infrastructure monitoring
- Business KPIs

### Logging
- Structured logging with correlation IDs
- Centralized log aggregation
- Log-based alerting

## Security Configuration

### Development
- Basic authentication
- Local network isolation
- Development certificates

### Staging
- RBAC (Role-Based Access Control)
- Network policies
- Secrets management
- TLS/SSL encryption
- API rate limiting

## Data Configuration

### Development
- Synthetic customer data
- Sample ML datasets
- Fast data generation
- No PII/sensitive data

### Staging
- Anonymized production data
- Real-world data patterns
- Data privacy compliance
- Production-like volumes

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep LISTEN
   
   # Stop conflicting services
   docker-compose down
   ```

2. **Database Connection Issues**
   ```bash
   # Check database status
   docker-compose ps postgres
   
   # View logs
   docker-compose logs postgres
   ```

3. **Memory Issues**
   ```bash
   # Increase Docker memory allocation
   # Docker Desktop: Settings > Resources > Memory
   
   # Monitor resource usage
   docker stats
   ```

### Health Checks

```bash
# Development environment
curl http://localhost:8000/health
curl http://localhost:8001/health  
curl http://localhost:8002/health

# Staging environment
kubectl get pods -n mlops-staging
kubectl describe pod -n mlops-staging <pod-name>
```

## Scaling Configuration

### Development
- Single instance per service
- Local resource limits
- Fast startup time

### Staging  
- Multiple replicas
- Resource requests/limits
- Horizontal Pod Autoscaling
- Load balancing

## Next Steps

1. **Production Environment**: Create production-ready Kubernetes configurations
2. **CI/CD Integration**: Automated deployment pipelines
3. **Advanced Monitoring**: APM and distributed tracing
4. **Security Hardening**: Enhanced security policies
5. **Disaster Recovery**: Backup and restore procedures