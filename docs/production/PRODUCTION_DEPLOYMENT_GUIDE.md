# Production Deployment Guide

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Infrastructure Deployment](#infrastructure-deployment)
4. [Application Deployment](#application-deployment)
5. [Database Setup](#database-setup)
6. [Monitoring Configuration](#monitoring-configuration)
7. [Security Configuration](#security-configuration)
8. [Post-Deployment Validation](#post-deployment-validation)
9. [Troubleshooting](#troubleshooting)
10. [Rollback Procedures](#rollback-procedures)

## Overview

This guide provides step-by-step instructions for deploying the enterprise anomaly detection platform to production. The deployment follows a blue-green strategy to ensure zero-downtime deployment with the ability to quickly rollback if issues are detected.

### Architecture Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│  Kubernetes     │────│   Database      │
│   (ALB)         │    │   Cluster       │    │   (RDS)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CloudFront    │    │   Monitoring    │    │   Cache         │
│   (CDN)         │    │   (Prometheus)  │    │   (Redis)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Deployment Strategy
- **Blue-Green Deployment**: Zero-downtime deployments with instant rollback capability
- **Infrastructure as Code**: All infrastructure managed via Terraform
- **GitOps**: Application deployments managed via ArgoCD
- **Monitoring**: Comprehensive observability with Prometheus and Grafana

## Prerequisites

### Required Tools
```bash
# Install required tools
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
terraform --version
aws --version
docker --version
```

### Access Requirements
- AWS CLI configured with appropriate permissions
- Kubectl access to production Kubernetes cluster
- Docker registry access (GitHub Container Registry)
- Terraform Cloud/Enterprise access
- VPN access to production networks (if required)

### Environment Variables
```bash
export AWS_REGION=us-west-2
export AWS_PROFILE=production
export CLUSTER_NAME=anomaly-detection-production-cluster
export REGISTRY=ghcr.io/your-org/anomaly-detection
export IMAGE_TAG=v1.0.0
export ENVIRONMENT=production
```

## Infrastructure Deployment

### 1. Terraform Infrastructure Provisioning

#### Initialize Terraform
```bash
cd infrastructure/terraform

# Initialize Terraform with production backend
terraform init \
  -backend-config="bucket=anomaly-detection-terraform-state" \
  -backend-config="key=production/terraform.tfstate" \
  -backend-config="region=us-west-2"
```

#### Plan Infrastructure Changes
```bash
# Review planned changes
terraform plan \
  -var-file="environments/production.tfvars" \
  -out=production.tfplan

# Review the plan carefully
terraform show production.tfplan
```

#### Apply Infrastructure
```bash
# Apply infrastructure changes
terraform apply production.tfplan

# Save outputs for later use
terraform output -json > ../outputs/production-outputs.json
```

#### Verify Infrastructure
```bash
# Verify EKS cluster
aws eks describe-cluster --name $CLUSTER_NAME --region $AWS_REGION

# Verify RDS instance
aws rds describe-db-instances --region $AWS_REGION

# Verify load balancer
aws elbv2 describe-load-balancers --region $AWS_REGION
```

### 2. Kubernetes Cluster Configuration

#### Configure kubectl
```bash
# Update kubeconfig
aws eks update-kubeconfig \
  --region $AWS_REGION \
  --name $CLUSTER_NAME

# Verify connectivity
kubectl cluster-info
kubectl get nodes
```

#### Install Cluster Add-ons
```bash
# Install AWS Load Balancer Controller
helm repo add eks https://aws.github.io/eks-charts
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=$CLUSTER_NAME \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller

# Install Cluster Autoscaler
kubectl apply -f k8s/cluster-autoscaler.yaml

# Install Metrics Server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

## Application Deployment

### 1. Prepare Application Images

#### Build and Push Images
```bash
# Build production image
docker build -t $REGISTRY:$IMAGE_TAG \
  -f deploy/docker/Dockerfile.production \
  --target runtime .

# Push to registry
docker push $REGISTRY:$IMAGE_TAG

# Verify image
docker inspect $REGISTRY:$IMAGE_TAG
```

#### Security Scanning
```bash
# Scan image for vulnerabilities
trivy image $REGISTRY:$IMAGE_TAG

# Ensure no critical vulnerabilities
trivy image --severity HIGH,CRITICAL $REGISTRY:$IMAGE_TAG
```

### 2. Deploy Application Components

#### Create Namespace and Secrets
```bash
# Create production namespace
kubectl create namespace production

# Create image pull secret
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=$GITHUB_USERNAME \
  --docker-password=$GITHUB_TOKEN \
  --namespace=production

# Create application secrets
kubectl create secret generic app-secrets \
  --from-literal=database-password="$(aws secretsmanager get-secret-value --secret-id prod-db-password --query SecretString --output text)" \
  --from-literal=redis-password="$(aws secretsmanager get-secret-value --secret-id prod-redis-password --query SecretString --output text)" \
  --from-literal=jwt-secret="$(aws secretsmanager get-secret-value --secret-id prod-jwt-secret --query SecretString --output text)" \
  --namespace=production
```

#### Deploy Database Migrations
```bash
# Run database migrations
kubectl apply -f k8s/production/migration-job.yaml

# Wait for migration completion
kubectl wait --for=condition=complete job/db-migration \
  --timeout=600s \
  --namespace=production

# Verify migration logs
kubectl logs job/db-migration --namespace=production
```

#### Deploy Application (Blue-Green)
```bash
# Deploy green environment
envsubst < k8s/production/deployment-green.yaml | kubectl apply -f -

# Wait for green deployment to be ready
kubectl rollout status deployment/anomaly-detection-green \
  --namespace=production \
  --timeout=600s

# Verify green deployment health
kubectl get pods -l app=anomaly-detection,version=green \
  --namespace=production
```

### 3. Service and Ingress Configuration

#### Deploy Services
```bash
# Deploy green service
kubectl apply -f k8s/production/service-green.yaml

# Deploy load balancer service
kubectl apply -f k8s/production/service-active.yaml

# Verify services
kubectl get services --namespace=production
```

#### Configure Ingress
```bash
# Deploy ingress with SSL
kubectl apply -f k8s/production/ingress.yaml

# Wait for ALB provisioning
kubectl wait --for=condition=ready \
  ingress/anomaly-detection-ingress \
  --namespace=production \
  --timeout=600s

# Get ALB DNS name
kubectl get ingress anomaly-detection-ingress \
  --namespace=production \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
```

## Database Setup

### 1. Database Configuration

#### Connect to Database
```bash
# Get database endpoint
DB_ENDPOINT=$(aws rds describe-db-instances \
  --db-instance-identifier anomaly-detection-production-postgresql \
  --query 'DBInstances[0].Endpoint.Address' \
  --output text)

# Connect via psql (through bastion host)
psql -h $DB_ENDPOINT -U postgres -d anomaly_detection
```

#### Initialize Database
```sql
-- Create application user
CREATE USER app_user WITH PASSWORD 'secure_password';

-- Grant necessary permissions
GRANT CONNECT ON DATABASE anomaly_detection TO app_user;
GRANT USAGE ON SCHEMA public TO app_user;
GRANT CREATE ON SCHEMA public TO app_user;

-- Create necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Verify setup
\l
\du
```

#### Configure Connection Pooling
```bash
# Deploy PgBouncer
kubectl apply -f k8s/production/pgbouncer.yaml

# Verify PgBouncer deployment
kubectl get pods -l app=pgbouncer --namespace=production
```

### 2. Redis Configuration

#### Verify Redis Cluster
```bash
# Get Redis endpoint
REDIS_ENDPOINT=$(aws elasticache describe-replication-groups \
  --replication-group-id anomaly-detection-production-redis \
  --query 'ReplicationGroups[0].RedisCluster.ClusterEndpoint.Address' \
  --output text)

# Test Redis connectivity
redis-cli -h $REDIS_ENDPOINT -p 6379 ping
```

#### Configure Redis for Application
```bash
# Test Redis operations
redis-cli -h $REDIS_ENDPOINT -p 6379 \
  -a "$(aws secretsmanager get-secret-value --secret-id prod-redis-password --query SecretString --output text)" \
  set test-key "test-value"

redis-cli -h $REDIS_ENDPOINT -p 6379 \
  -a "$(aws secretsmanager get-secret-value --secret-id prod-redis-password --query SecretString --output text)" \
  get test-key
```

## Monitoring Configuration

### 1. Prometheus Deployment

#### Deploy Prometheus Stack
```bash
# Add Prometheus Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install kube-prometheus-stack
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values monitoring/prometheus-values.yaml \
  --wait

# Verify Prometheus deployment
kubectl get pods -n monitoring
```

#### Configure Service Monitors
```bash
# Deploy application service monitor
kubectl apply -f monitoring/service-monitor.yaml

# Deploy custom alert rules
kubectl apply -f monitoring/alert-rules.yaml

# Verify configuration
kubectl get servicemonitors -n monitoring
kubectl get prometheusrules -n monitoring
```

### 2. Grafana Configuration

#### Access Grafana
```bash
# Get Grafana admin password
kubectl get secret prometheus-grafana \
  -n monitoring \
  -o jsonpath="{.data.admin-password}" | base64 --decode

# Port forward to access Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
```

#### Import Dashboards
```bash
# Import application dashboards
curl -X POST \
  http://admin:$GRAFANA_PASSWORD@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @monitoring/dashboards/application-overview.json

# Import infrastructure dashboards
curl -X POST \
  http://admin:$GRAFANA_PASSWORD@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @monitoring/dashboards/infrastructure-overview.json
```

### 3. Logging Configuration

#### Deploy Logging Stack
```bash
# Deploy Elasticsearch (if using ELK stack)
helm install elasticsearch elastic/elasticsearch \
  --namespace logging \
  --create-namespace \
  --values logging/elasticsearch-values.yaml

# Deploy Logstash
helm install logstash elastic/logstash \
  --namespace logging \
  --values logging/logstash-values.yaml

# Deploy Filebeat
kubectl apply -f logging/filebeat.yaml
```

## Security Configuration

### 1. Network Security

#### Configure Network Policies
```bash
# Apply network policies
kubectl apply -f security/network-policies/

# Verify network policies
kubectl get networkpolicies --all-namespaces
```

#### Configure Pod Security Policies
```bash
# Apply pod security policies
kubectl apply -f security/pod-security-policies/

# Verify PSPs
kubectl get podsecuritypolicies
```

### 2. RBAC Configuration

#### Configure Service Accounts
```bash
# Create service accounts
kubectl apply -f security/rbac/service-accounts.yaml

# Apply RBAC policies
kubectl apply -f security/rbac/rbac.yaml

# Verify RBAC
kubectl auth can-i --list --as=system:serviceaccount:production:anomaly-detection
```

### 3. Security Scanning

#### Deploy Security Scanning
```bash
# Deploy Falco for runtime security
helm install falco falcosecurity/falco \
  --namespace security \
  --create-namespace \
  --values security/falco-values.yaml

# Deploy Trivy Operator for vulnerability scanning
kubectl apply -f https://raw.githubusercontent.com/aquasecurity/trivy-operator/main/deploy/static/trivy-operator.yaml
```

## Post-Deployment Validation

### 1. Health Checks

#### Application Health
```bash
# Check application health endpoints
curl -f https://api.detection-platform.io/health/ready
curl -f https://api.detection-platform.io/health/live

# Verify API functionality
curl -X POST https://api.detection-platform.io/api/v1/detect \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_TOKEN" \
  -d '{"data": [[1,2,3,4], [5,6,7,8]]}'
```

#### Database Health
```bash
# Check database connectivity
kubectl exec -it deployment/anomaly-detection-green \
  --namespace=production -- \
  python -c "
import psycopg2
conn = psycopg2.connect(
    host='$DB_ENDPOINT',
    database='anomaly_detection',
    user='app_user',
    password='$DB_PASSWORD'
)
print('Database connection successful')
conn.close()
"
```

#### Cache Health
```bash
# Check Redis connectivity
kubectl exec -it deployment/anomaly-detection-green \
  --namespace=production -- \
  redis-cli -h $REDIS_ENDPOINT -p 6379 ping
```

### 2. Performance Validation

#### Load Testing
```bash
# Run performance tests
cd tests/load_testing
python -m pytest test_platform_load_performance.py -v

# Run specific endpoint tests
k6 run --vus 100 --duration 5m performance-test.js
```

#### Monitor Key Metrics
```bash
# Check response times
curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,rate(http_request_duration_seconds_bucket[5m]))"

# Check error rates
curl -s "http://prometheus:9090/api/v1/query?query=rate(http_requests_total{status=~\"5..\"}[5m])"

# Check throughput
curl -s "http://prometheus:9090/api/v1/query?query=rate(http_requests_total[5m])"
```

### 3. Traffic Switch (Blue-Green)

#### Switch Traffic to Green
```bash
# Update service selector to green
kubectl patch service anomaly-detection-active \
  --namespace=production \
  -p '{"spec":{"selector":{"version":"green"}}}'

# Verify traffic switch
kubectl get endpoints anomaly-detection-active \
  --namespace=production

# Monitor application health after switch
watch kubectl get pods -l app=anomaly-detection \
  --namespace=production
```

#### Validate Traffic Flow
```bash
# Monitor logs for new requests
kubectl logs -f deployment/anomaly-detection-green \
  --namespace=production

# Check metrics for traffic patterns
curl -s "http://prometheus:9090/api/v1/query?query=rate(http_requests_total{version=\"green\"}[1m])"
```

### 4. Cleanup Blue Environment
```bash
# Scale down blue deployment after validation period
kubectl scale deployment anomaly-detection-blue \
  --replicas=0 \
  --namespace=production

# Remove blue service (optional, keep for rollback)
# kubectl delete service anomaly-detection-blue --namespace=production
```

## Troubleshooting

### Common Issues and Solutions

#### Pod Startup Issues
```bash
# Check pod status
kubectl describe pods -l app=anomaly-detection --namespace=production

# Check pod logs
kubectl logs deployment/anomaly-detection-green --namespace=production

# Check events
kubectl get events --namespace=production --sort-by='.lastTimestamp'
```

#### Database Connection Issues
```bash
# Test database connectivity from pod
kubectl exec -it deployment/anomaly-detection-green \
  --namespace=production -- \
  nc -zv $DB_ENDPOINT 5432

# Check database logs
aws rds describe-db-log-files \
  --db-instance-identifier anomaly-detection-production-postgresql
```

#### Load Balancer Issues
```bash
# Check ALB status
aws elbv2 describe-load-balancers \
  --load-balancer-arns $(kubectl get ingress anomaly-detection-ingress \
    --namespace=production \
    -o jsonpath='{.metadata.annotations.alb\.ingress\.kubernetes\.io/load-balancer-arn}')

# Check target group health
aws elbv2 describe-target-health \
  --target-group-arn $TARGET_GROUP_ARN
```

#### Performance Issues
```bash
# Check resource usage
kubectl top pods --namespace=production
kubectl top nodes

# Check HPA status
kubectl get hpa --namespace=production

# Scale manually if needed
kubectl scale deployment anomaly-detection-green \
  --replicas=10 \
  --namespace=production
```

## Rollback Procedures

### Immediate Rollback (Emergency)
```bash
# Switch traffic back to blue environment
kubectl patch service anomaly-detection-active \
  --namespace=production \
  -p '{"spec":{"selector":{"version":"blue"}}}'

# Verify rollback
kubectl get endpoints anomaly-detection-active \
  --namespace=production

# Scale up blue environment if needed
kubectl scale deployment anomaly-detection-blue \
  --replicas=5 \
  --namespace=production
```

### Database Rollback
```bash
# Restore from point-in-time backup (if needed)
aws rds restore-db-instance-to-point-in-time \
  --source-db-instance-identifier anomaly-detection-production-postgresql \
  --target-db-instance-identifier anomaly-detection-production-postgresql-rollback \
  --restore-time 2024-01-01T10:00:00.000Z
```

### Infrastructure Rollback
```bash
# Rollback Terraform changes
terraform plan -destroy -var-file="environments/production.tfvars"
terraform apply -var-file="environments/production.tfvars" -target=specific_resource
```

## Monitoring and Maintenance

### Daily Checks
```bash
# Check application health
curl -f https://api.detection-platform.io/health/ready

# Check key metrics
kubectl top pods --namespace=production
kubectl get hpa --namespace=production

# Check for failed pods
kubectl get pods --field-selector=status.phase=Failed --all-namespaces
```

### Weekly Maintenance
```bash
# Update node images (if needed)
aws eks update-nodegroup-version \
  --cluster-name $CLUSTER_NAME \
  --nodegroup-name production-nodes

# Review and update security patches
kubectl get vulnerabilityreports -A

# Review capacity and scaling
kubectl describe hpa --namespace=production
```

## Contacts and Escalation

### On-Call Contacts
- **Primary On-Call**: [Phone/Slack]
- **Secondary On-Call**: [Phone/Slack]
- **Engineering Manager**: [Phone/Slack]
- **Infrastructure Team**: [Phone/Slack]

### Escalation Matrix
1. **Level 1** (0-15 min): On-call engineer
2. **Level 2** (15-30 min): Team lead
3. **Level 3** (30-60 min): Engineering manager
4. **Level 4** (1+ hour): Director of Engineering

---

**Guide Version:** 1.0  
**Last Updated:** January 1, 2025  
**Next Review:** February 1, 2025

*This guide should be updated after each deployment to reflect any changes or lessons learned.*