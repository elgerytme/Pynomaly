# Infrastructure as Code

This directory contains all infrastructure-related code, configurations, and deployment manifests.

## Infrastructure Structure

```
infra/
├── docker/         # Docker configurations
├── kubernetes/     # Kubernetes manifests
├── terraform/      # Terraform infrastructure code
├── helm/          # Helm charts
└── ci_cd/         # CI/CD pipeline configurations
```

## Infrastructure Components

### Docker (`docker/`)
**Container configurations and Docker-related files**

- `Dockerfile` - Application container definition
- `docker-compose.yml` - Multi-container application setup
- `docker-compose.override.yml` - Development overrides
- `.dockerignore` - Files to exclude from Docker context
- Multi-stage builds for different environments

**Purpose**: Containerize application and its dependencies

### Kubernetes (`kubernetes/`)
**Kubernetes deployment manifests**

- `deployments/` - Application deployment configurations
- `services/` - Service definitions and networking
- `ingress/` - Ingress controllers and routing
- `configmaps/` - Configuration data
- `secrets/` - Sensitive data management
- `rbac/` - Role-based access control
- `monitoring/` - Monitoring and observability

**Purpose**: Orchestrate containers at scale

### Terraform (`terraform/`)
**Infrastructure as Code using Terraform**

- `main.tf` - Main infrastructure configuration
- `variables.tf` - Input variables
- `outputs.tf` - Output values
- `providers.tf` - Provider configurations
- `modules/` - Reusable infrastructure modules
- `environments/` - Environment-specific configurations

**Purpose**: Provision and manage cloud infrastructure

### Helm (`helm/`)
**Kubernetes package manager configurations**

- `Chart.yaml` - Helm chart metadata
- `values.yaml` - Default configuration values
- `templates/` - Kubernetes manifest templates
- `charts/` - Chart dependencies
- `values/` - Environment-specific values

**Purpose**: Package and deploy applications to Kubernetes

### CI/CD (`ci_cd/`)
**Continuous Integration and Deployment configurations**

- `github/` - GitHub Actions workflows
- `gitlab/` - GitLab CI configurations
- `jenkins/` - Jenkins pipeline definitions
- `azure/` - Azure DevOps pipelines
- `scripts/` - Deployment scripts

**Purpose**: Automate build, test, and deployment processes

## Infrastructure Principles

### Infrastructure as Code (IaC)
- Version control all infrastructure
- Declarative configuration
- Immutable infrastructure
- Automated provisioning and deployment

### Environment Management
- **Development**: Local development environment
- **Testing**: Automated testing environment
- **Staging**: Pre-production environment
- **Production**: Live production environment

### Security Best Practices
- Least privilege access
- Secrets management
- Network security
- Regular security updates
- Vulnerability scanning

### Scalability and Reliability
- Horizontal scaling
- Load balancing
- Health checks
- Auto-recovery
- Disaster recovery

## Docker Configuration

### Dockerfile Best Practices
```dockerfile
# Use official base images
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Install dependencies first (layer caching)
COPY package*.json ./
RUN npm ci --only=production

# Copy application code
COPY . .

# Create non-root user
RUN adduser -D -s /bin/sh appuser
USER appuser

# Expose port
EXPOSE 3000

# Define health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Start application
CMD ["npm", "start"]
```

### Docker Compose Structure
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
    depends_on:
      - database
      - cache
    networks:
      - app-network

  database:
    image: postgres:15
    environment:
      - POSTGRES_DB=myapp
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    secrets:
      - db_password

networks:
  app-network:

volumes:
  postgres_data:

secrets:
  db_password:
    file: ./secrets/db_password.txt
```

## Kubernetes Configuration

### Deployment Structure
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: myapp
        image: myapp:latest
        ports:
        - containerPort: 3000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: myapp-secrets
              key: database-url
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Service Configuration
```yaml
apiVersion: v1
kind: Service
metadata:
  name: myapp-service
  namespace: production
spec:
  selector:
    app: myapp
  ports:
  - port: 80
    targetPort: 3000
  type: LoadBalancer
```

## Terraform Configuration

### Main Configuration
```hcl
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket = "my-terraform-state"
    key    = "infrastructure/terraform.tfstate"
    region = "us-west-2"
  }
}

provider "aws" {
  region = var.aws_region
}

module "vpc" {
  source = "./modules/vpc"
  
  vpc_cidr = var.vpc_cidr
  environment = var.environment
}

module "eks" {
  source = "./modules/eks"
  
  cluster_name = var.cluster_name
  vpc_id = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnet_ids
}
```

### Variables Configuration
```hcl
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}
```

## CI/CD Pipeline

### GitHub Actions Example
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
    
    - name: Install dependencies
      run: npm ci
    
    - name: Run tests
      run: npm test
    
    - name: Run security audit
      run: npm audit
  
  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: docker build -t myapp:${{ github.sha }} .
    
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push myapp:${{ github.sha }}
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to staging
      run: |
        kubectl set image deployment/myapp myapp=myapp:${{ github.sha }}
        kubectl rollout status deployment/myapp
```

## Environment Management

### Development Environment
```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  app:
    build:
      context: .
      target: development
    volumes:
      - .:/app
      - /app/node_modules
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=development
      - DEBUG=*
```

### Production Environment
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  app:
    image: myapp:latest
    restart: unless-stopped
    environment:
      - NODE_ENV=production
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
```

## Security Configuration

### Secrets Management
```yaml
# kubernetes/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: myapp-secrets
type: Opaque
data:
  database-url: <base64-encoded-url>
  api-key: <base64-encoded-key>
```

### Network Policies
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: myapp-network-policy
spec:
  podSelector:
    matchLabels:
      app: myapp
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 3000
```

## Monitoring and Observability

### Prometheus Configuration
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'myapp'
      static_configs:
      - targets: ['myapp-service:3000']
```

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Application Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ]
      }
    ]
  }
}
```

## Deployment Strategies

### Blue-Green Deployment
```yaml
# Blue environment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp-blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
      version: blue
```

### Canary Deployment
```yaml
# Canary deployment with traffic splitting
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: myapp-rollout
spec:
  strategy:
    canary:
      steps:
      - setWeight: 10
      - pause: {}
      - setWeight: 50
      - pause: {}
```

## Best Practices

### Infrastructure
1. **Version Control**: All infrastructure as code
2. **Immutable**: Rebuild don't modify
3. **Automated**: Minimize manual processes
4. **Documented**: Clear documentation
5. **Tested**: Test infrastructure changes

### Security
1. **Least Privilege**: Minimal required permissions
2. **Secrets Management**: Never commit secrets
3. **Network Segmentation**: Isolate components
4. **Regular Updates**: Keep dependencies updated
5. **Monitoring**: Continuous security monitoring

### Operations
1. **Observability**: Comprehensive logging and monitoring
2. **Backup**: Regular backups and disaster recovery
3. **Scaling**: Plan for growth and traffic spikes
4. **Cost Optimization**: Monitor and optimize costs
5. **Documentation**: Keep runbooks updated

## Tools and Resources

### Essential Tools
- **Docker**: Container runtime
- **Kubernetes**: Container orchestration
- **Terraform**: Infrastructure as code
- **Helm**: Kubernetes package manager
- **GitHub Actions**: CI/CD automation

### Monitoring Stack
- **Prometheus**: Metrics collection
- **Grafana**: Visualization
- **Jaeger**: Distributed tracing
- **ELK Stack**: Centralized logging

### Security Tools
- **Trivy**: Container vulnerability scanning
- **Falco**: Runtime security monitoring
- **OPA**: Policy enforcement
- **Vault**: Secrets management
