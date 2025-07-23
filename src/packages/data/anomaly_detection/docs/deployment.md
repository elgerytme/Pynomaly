# Production Deployment and Scaling Guide

This guide covers production deployment strategies, scaling patterns, and operational considerations for the Anomaly Detection package in enterprise environments.

## Table of Contents

1. [Overview](#overview)
2. [Deployment Architectures](#deployment-architectures)
3. [Containerization](#containerization)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Auto-scaling Strategies](#auto-scaling-strategies)
6. [Load Balancing](#load-balancing)
7. [Service Mesh Integration](#service-mesh-integration)
8. [Database and Storage](#database-and-storage)
9. [Monitoring and Observability](#monitoring-and-observability)
10. [CI/CD Pipelines](#cicd-pipelines)
11. [Multi-region Deployment](#multi-region-deployment)
12. [Best Practices](#best-practices)

## Overview

Production deployment of anomaly detection systems requires careful consideration of scalability, reliability, performance, and operational requirements. This guide provides comprehensive strategies for deploying at scale.

### Deployment Patterns

- **Microservices Architecture**: Decomposed services for different components
- **API Gateway Pattern**: Centralized entry point with routing and security
- **Event-Driven Architecture**: Asynchronous processing with message queues
- **Multi-tenant Architecture**: Serving multiple customers efficiently
- **Edge Deployment**: Processing at edge locations for low latency

### Scaling Dimensions

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class ScalingType(Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    FUNCTIONAL = "functional"

@dataclass
class ScalingConfiguration:
    """Configuration for scaling strategies."""
    
    # Horizontal scaling
    min_replicas: int = 2
    max_replicas: int = 20
    target_cpu_utilization: int = 70
    target_memory_utilization: int = 80
    target_rps: int = 100
    
    # Vertical scaling
    min_cpu_cores: float = 0.5
    max_cpu_cores: float = 4.0
    min_memory_gb: float = 1.0
    max_memory_gb: float = 8.0
    
    # Custom metrics
    target_queue_length: int = 100
    target_latency_p95_ms: float = 200.0
    
    # Scaling behavior
    scale_up_period_seconds: int = 60
    scale_down_period_seconds: int = 300
    scale_up_threshold_breach_count: int = 2
    scale_down_threshold_breach_count: int = 3

# Example deployment configurations
DEPLOYMENT_CONFIGS = {
    'development': ScalingConfiguration(
        min_replicas=1,
        max_replicas=3,
        min_cpu_cores=0.25,
        max_cpu_cores=1.0,
        min_memory_gb=0.5,
        max_memory_gb=2.0
    ),
    'staging': ScalingConfiguration(
        min_replicas=2,
        max_replicas=5,
        target_rps=50,
        target_latency_p95_ms=500.0
    ),
    'production': ScalingConfiguration(
        min_replicas=3,
        max_replicas=50,
        target_rps=1000,
        target_latency_p95_ms=100.0,
        min_cpu_cores=1.0,
        max_cpu_cores=8.0,
        min_memory_gb=2.0,
        max_memory_gb=16.0
    )
}
```

## Deployment Architectures

### Microservices Architecture

```yaml
# docker-compose.yml - Microservices deployment
version: '3.8'

services:
  # API Gateway
  api-gateway:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - anomaly-api
      - model-service
      - batch-processor
    networks:
      - anomaly-network

  # Core Anomaly Detection API
  anomaly-api:
    image: anomaly-detection:latest
    environment:
      - SERVICE_NAME=anomaly-api
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@postgres:5432/anomaly_db
      - MODEL_SERVICE_URL=http://model-service:8080
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '0.5'
          memory: 1G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - anomaly-network

  # Model Management Service
  model-service:
    image: anomaly-detection-models:latest
    environment:
      - SERVICE_NAME=model-service
      - MODEL_STORAGE_PATH=/models
      - S3_BUCKET=anomaly-models-bucket
    volumes:
      - model-storage:/models
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '1.0'
          memory: 2G
    networks:
      - anomaly-network

  # Batch Processing Service
  batch-processor:
    image: anomaly-detection-batch:latest
    environment:
      - SERVICE_NAME=batch-processor
      - CELERY_BROKER_URL=redis://redis:6379
      - CELERY_RESULT_BACKEND=redis://redis:6379
    deploy:
      replicas: 4
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
    networks:
      - anomaly-network

  # Streaming Processing Service
  stream-processor:
    image: anomaly-detection-stream:latest
    environment:
      - SERVICE_NAME=stream-processor
      - KAFKA_BROKERS=kafka:9092
      - KAFKA_TOPIC=sensor-data
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
    networks:
      - anomaly-network

  # Redis for caching and job queue
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    deploy:
      resources:
        limits:
          memory: 2G
    networks:
      - anomaly-network

  # PostgreSQL for metadata and results
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=anomaly_db
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres-data:/var/lib/postgresql/data
    deploy:
      resources:
        limits:
          memory: 4G
    networks:
      - anomaly-network

  # Kafka for streaming data
  kafka:
    image: confluentinc/cp-kafka:latest
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    depends_on:
      - zookeeper
    networks:
      - anomaly-network

  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    networks:
      - anomaly-network

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    networks:
      - anomaly-network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    networks:
      - anomaly-network

volumes:
  model-storage:
  redis-data:
  postgres-data:
  prometheus-data:
  grafana-data:

networks:
  anomaly-network:
    driver: bridge
```

### High-Availability Architecture

```python
# infrastructure/deployment.py
import boto3
from typing import Dict, List
import yaml

class HighAvailabilityDeployment:
    """High-availability deployment configuration for AWS."""
    
    def __init__(self, region: str = 'us-west-2'):
        self.region = region
        self.ec2 = boto3.client('ec2', region_name=region)
        self.ecs = boto3.client('ecs', region_name=region)
        self.elb = boto3.client('elbv2', region_name=region)
        self.rds = boto3.client('rds', region_name=region)
        
    def create_vpc_infrastructure(self) -> Dict:
        """Create VPC with multi-AZ subnets."""
        
        # Create VPC
        vpc_response = self.ec2.create_vpc(
            CidrBlock='10.0.0.0/16',
            TagSpecifications=[{
                'ResourceType': 'vpc',
                'Tags': [{'Key': 'Name', 'Value': 'anomaly-detection-vpc'}]
            }]
        )
        vpc_id = vpc_response['Vpc']['VpcId']
        
        # Get availability zones
        azs = self.ec2.describe_availability_zones()['AvailabilityZones']
        
        # Create subnets in multiple AZs
        public_subnets = []
        private_subnets = []
        
        for i, az in enumerate(azs[:3]):  # Use first 3 AZs
            # Public subnet
            public_subnet = self.ec2.create_subnet(
                VpcId=vpc_id,
                CidrBlock=f'10.0.{i*2+1}.0/24',
                AvailabilityZone=az['ZoneName'],
                TagSpecifications=[{
                    'ResourceType': 'subnet',
                    'Tags': [{'Key': 'Name', 'Value': f'public-subnet-{i+1}'}]
                }]
            )
            public_subnets.append(public_subnet['Subnet']['SubnetId'])
            
            # Private subnet
            private_subnet = self.ec2.create_subnet(
                VpcId=vpc_id,
                CidrBlock=f'10.0.{i*2+2}.0/24',
                AvailabilityZone=az['ZoneName'],
                TagSpecifications=[{
                    'ResourceType': 'subnet',
                    'Tags': [{'Key': 'Name', 'Value': f'private-subnet-{i+1}'}]
                }]
            )
            private_subnets.append(private_subnet['Subnet']['SubnetId'])
        
        return {
            'vpc_id': vpc_id,
            'public_subnets': public_subnets,
            'private_subnets': private_subnets
        }
    
    def create_ecs_cluster(self, vpc_config: Dict) -> str:
        """Create ECS cluster with auto-scaling."""
        
        cluster_response = self.ecs.create_cluster(
            clusterName='anomaly-detection-cluster',
            capacityProviders=['FARGATE', 'FARGATE_SPOT'],
            defaultCapacityProviderStrategy=[
                {
                    'capacityProvider': 'FARGATE',
                    'weight': 1,
                    'base': 2
                },
                {
                    'capacityProvider': 'FARGATE_SPOT',
                    'weight': 4
                }
            ]
        )
        
        return cluster_response['cluster']['clusterArn']
    
    def create_load_balancer(self, vpc_config: Dict) -> Dict:
        """Create Application Load Balancer."""
        
        # Create ALB
        alb_response = self.elb.create_load_balancer(
            Name='anomaly-detection-alb',
            Subnets=vpc_config['public_subnets'],
            SecurityGroups=['sg-xxx'],  # Would create security group first
            Scheme='internet-facing',
            Type='application',
            IpAddressType='ipv4'
        )
        
        alb_arn = alb_response['LoadBalancers'][0]['LoadBalancerArn']
        alb_dns = alb_response['LoadBalancers'][0]['DNSName']
        
        # Create target group
        target_group_response = self.elb.create_target_group(
            Name='anomaly-api-targets',
            Protocol='HTTP',
            Port=8000,
            VpcId=vpc_config['vpc_id'],
            TargetType='ip',
            HealthCheckPath='/health',
            HealthCheckIntervalSeconds=30,
            HealthCheckTimeoutSeconds=5,
            HealthyThresholdCount=2,
            UnhealthyThresholdCount=3
        )
        
        target_group_arn = target_group_response['TargetGroups'][0]['TargetGroupArn']
        
        # Create listener
        self.elb.create_listener(
            LoadBalancerArn=alb_arn,
            Protocol='HTTP',
            Port=80,
            DefaultActions=[{
                'Type': 'forward',
                'TargetGroupArn': target_group_arn
            }]
        )
        
        return {
            'alb_arn': alb_arn,
            'alb_dns': alb_dns,
            'target_group_arn': target_group_arn
        }
    
    def create_rds_cluster(self, vpc_config: Dict) -> Dict:
        """Create RDS Aurora cluster for high availability."""
        
        # Create DB subnet group
        subnet_group_response = self.rds.create_db_subnet_group(
            DBSubnetGroupName='anomaly-db-subnet-group',
            DBSubnetGroupDescription='Subnet group for anomaly detection DB',
            SubnetIds=vpc_config['private_subnets']
        )
        
        # Create Aurora cluster
        cluster_response = self.rds.create_db_cluster(
            DBClusterIdentifier='anomaly-detection-cluster',
            Engine='aurora-postgresql',
            MasterUsername='anomaly_user',
            MasterUserPassword='secure_password',
            DatabaseName='anomaly_db',
            DBSubnetGroupName='anomaly-db-subnet-group',
            VpcSecurityGroupIds=['sg-xxx'],  # Would create security group first
            BackupRetentionPeriod=7,
            PreferredBackupWindow='03:00-04:00',
            PreferredMaintenanceWindow='sun:04:00-sun:05:00',
            EnableCloudwatchLogsExports=['postgresql'],
            DeletionProtection=True
        )
        
        # Create primary instance
        primary_instance = self.rds.create_db_instance(
            DBInstanceIdentifier='anomaly-db-primary',
            DBInstanceClass='db.r5.large',
            Engine='aurora-postgresql',
            DBClusterIdentifier='anomaly-detection-cluster',
            PubliclyAccessible=False,
            PerformanceInsightsEnabled=True
        )
        
        # Create read replica
        replica_instance = self.rds.create_db_instance(
            DBInstanceIdentifier='anomaly-db-replica',
            DBInstanceClass='db.r5.large',
            Engine='aurora-postgresql',
            DBClusterIdentifier='anomaly-detection-cluster',
            PubliclyAccessible=False,
            PerformanceInsightsEnabled=True
        )
        
        return {
            'cluster_arn': cluster_response['DBCluster']['DBClusterArn'],
            'cluster_endpoint': cluster_response['DBCluster']['Endpoint'],
            'reader_endpoint': cluster_response['DBCluster']['ReaderEndpoint']
        }

# Terraform configuration for infrastructure as code
terraform_config = """
# terraform/main.tf
provider "aws" {
  region = var.aws_region
}

# VPC Configuration
resource "aws_vpc" "anomaly_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "anomaly-detection-vpc"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "anomaly_igw" {
  vpc_id = aws_vpc.anomaly_vpc.id

  tags = {
    Name = "anomaly-detection-igw"
  }
}

# Public Subnets
resource "aws_subnet" "public_subnets" {
  count             = 3
  vpc_id            = aws_vpc.anomaly_vpc.id
  cidr_block        = "10.0.${count.index * 2 + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  map_public_ip_on_launch = true

  tags = {
    Name = "public-subnet-${count.index + 1}"
    Type = "public"
  }
}

# Private Subnets
resource "aws_subnet" "private_subnets" {
  count             = 3
  vpc_id            = aws_vpc.anomaly_vpc.id
  cidr_block        = "10.0.${count.index * 2 + 2}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "private-subnet-${count.index + 1}"
    Type = "private"
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "anomaly_cluster" {
  name = "anomaly-detection-cluster"

  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight           = 1
    base            = 2
  }

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    weight           = 4
  }

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# Application Load Balancer
resource "aws_lb" "anomaly_alb" {
  name               = "anomaly-detection-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = aws_subnet.public_subnets[*].id

  enable_deletion_protection = true

  tags = {
    Name = "anomaly-detection-alb"
  }
}

# Auto Scaling Configuration
resource "aws_appautoscaling_target" "anomaly_api_target" {
  max_capacity       = 20
  min_capacity       = 2
  resource_id        = "service/${aws_ecs_cluster.anomaly_cluster.name}/${aws_ecs_service.anomaly_api.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "anomaly_api_cpu_policy" {
  name               = "anomaly-api-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.anomaly_api_target.resource_id
  scalable_dimension = aws_appautoscaling_target.anomaly_api_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.anomaly_api_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 70.0
  }
}

resource "aws_appautoscaling_policy" "anomaly_api_memory_policy" {
  name               = "anomaly-api-memory-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.anomaly_api_target.resource_id
  scalable_dimension = aws_appautoscaling_target.anomaly_api_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.anomaly_api_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageMemoryUtilization"
    }
    target_value = 80.0
  }
}
"""
```

## Containerization

### Multi-stage Docker Build

```dockerfile
# Dockerfile - Multi-stage build for production
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libc6-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN adduser --disabled-password --gecos '' --shell /bin/bash appuser

#############################################
# Development stage
#############################################
FROM base as development

# Install development dependencies
RUN pip install poetry

# Copy poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --with dev,test

# Copy source code
COPY . .

# Set ownership
RUN chown -R appuser:appuser .

USER appuser

CMD ["uvicorn", "anomaly_detection.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

#############################################
# Production builder stage
#############################################
FROM base as builder

# Install poetry
RUN pip install poetry

# Copy poetry files
COPY pyproject.toml poetry.lock ./

# Configure poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-dev

# Copy source code
COPY . .

# Build wheel
RUN poetry build

#############################################
# Production stage
#############################################
FROM python:3.11-slim as production

# Set environment variables for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

# Create app user
RUN adduser --disabled-password --gecos '' --shell /bin/bash appuser

# Set work directory
WORKDIR /app

# Copy built wheel from builder stage
COPY --from=builder /dist/*.whl /tmp/

# Install the package
RUN pip install --no-cache-dir /tmp/*.whl \
    && rm /tmp/*.whl

# Copy configuration files
COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser scripts/ ./scripts/

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/models \
    && chown -R appuser:appuser /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Switch to app user
USER appuser

# Expose port
EXPOSE ${PORT}

# Run the application
CMD ["sh", "-c", "uvicorn anomaly_detection.main:app --host 0.0.0.0 --port ${PORT} --workers 4"]

#############################################
# GPU-enabled production stage
#############################################
FROM nvidia/cuda:11.8-runtime-ubuntu22.04 as gpu-production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000

# Install Python and runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN adduser --disabled-password --gecos '' --shell /bin/bash appuser

# Set work directory
WORKDIR /app

# Copy built wheel from builder stage
COPY --from=builder /dist/*.whl /tmp/

# Install the package with GPU support
RUN pip3.11 install --no-cache-dir /tmp/*.whl[gpu] \
    && rm /tmp/*.whl

# Copy configuration files
COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser scripts/ ./scripts/

# Create directories
RUN mkdir -p /app/data /app/logs /app/models \
    && chown -R appuser:appuser /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

USER appuser
EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn anomaly_detection.main:app --host 0.0.0.0 --port ${PORT} --workers 2"]
```

### Docker Security Best Practices

```dockerfile
# Dockerfile.secure - Security-hardened container
FROM python:3.11-slim

LABEL maintainer="security@company.com" \
      version="1.0" \
      description="Secure anomaly detection service"

# Security: Create non-root user first
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin -c "App User" appuser

# Security: Update system and install only necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

# Security: Set secure permissions
WORKDIR /app
RUN chown appuser:appgroup /app

# Security: Copy files with appropriate permissions
COPY --chown=appuser:appgroup requirements.txt .
COPY --chown=appuser:appgroup . .

# Security: Install Python packages as non-root
USER appuser
RUN pip install --user --no-cache-dir -r requirements.txt

# Security: Remove unnecessary files
RUN find /app -name "*.pyc" -delete \
    && find /app -name "__pycache__" -type d -exec rm -rf {} + \
    && find /app -name ".git" -type d -exec rm -rf {} + || true

# Security: Set read-only filesystem
RUN chmod -R 444 /app \
    && chmod 555 /app \
    && mkdir -p /tmp/app \
    && chmod 755 /tmp/app

# Security: Use specific port and non-root
EXPOSE 8000

# Security: Health check without curl as root
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Security: Read-only root filesystem
VOLUME ["/tmp/app"]

# Security: Drop all capabilities
ENTRYPOINT ["python", "-m", "anomaly_detection.main"]
```

## Kubernetes Deployment

### Complete Kubernetes Configuration

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: anomaly-detection
  labels:
    name: anomaly-detection
    monitoring: enabled

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: anomaly-config
  namespace: anomaly-detection
data:
  config.yaml: |
    server:
      host: 0.0.0.0
      port: 8000
      workers: 4
    
    redis:
      host: redis-service
      port: 6379
      db: 0
    
    database:
      host: postgres-service
      port: 5432
      name: anomaly_db
      
    algorithms:
      default: isolation_forest
      cache_models: true
      
    monitoring:
      metrics_enabled: true
      tracing_enabled: true

---
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: anomaly-secrets
  namespace: anomaly-detection
type: Opaque
data:
  database-password: cGFzc3dvcmQ=  # base64 encoded
  redis-password: cmVkaXNwYXNz      # base64 encoded
  jwt-secret: c2VjcmV0a2V5          # base64 encoded

---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: anomaly-api
  namespace: anomaly-detection
  labels:
    app: anomaly-api
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: anomaly-api
  template:
    metadata:
      labels:
        app: anomaly-api
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: anomaly-service-account
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 2000
      containers:
      - name: anomaly-api
        image: anomaly-detection:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: CONFIG_PATH
          value: "/etc/config/config.yaml"
        - name: DATABASE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: anomaly-secrets
              key: database-password
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: anomaly-secrets
              key: redis-password
        volumeMounts:
        - name: config-volume
          mountPath: /etc/config
        - name: model-storage
          mountPath: /app/models
        - name: tmp-volume
          mountPath: /tmp
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
      volumes:
      - name: config-volume
        configMap:
          name: anomaly-config
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage-pvc
      - name: tmp-volume
        emptyDir: {}
      nodeSelector:
        workload-type: cpu-intensive
      tolerations:
      - key: "workload-type"
        operator: "Equal"
        value: "cpu-intensive"
        effect: "NoSchedule"

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: anomaly-api-service
  namespace: anomaly-detection
  labels:
    app: anomaly-api
spec:
  type: ClusterIP
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  selector:
    app: anomaly-api

---
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: anomaly-api-hpa
  namespace: anomaly-detection
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: anomaly-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      selectPolicy: Min

---
# k8s/vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: anomaly-api-vpa
  namespace: anomaly-detection
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: anomaly-api
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: anomaly-api
      minAllowed:
        cpu: 100m
        memory: 128Mi
      maxAllowed:
        cpu: 4
        memory: 8Gi
      controlledResources: ["cpu", "memory"]

---
# k8s/pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: anomaly-api-pdb
  namespace: anomaly-detection
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: anomaly-api

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: anomaly-api-ingress
  namespace: anomaly-detection
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.anomaly-detection.example.com
    secretName: anomaly-api-tls
  rules:
  - host: api.anomaly-detection.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: anomaly-api-service
            port:
              number: 80

---
# k8s/networkpolicy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: anomaly-api-netpol
  namespace: anomaly-detection
spec:
  podSelector:
    matchLabels:
      app: anomaly-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: istio-system
    - namespaceSelector:
        matchLabels:
          name: anomaly-detection
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: anomaly-detection
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
  - to: []  # Allow external API calls
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 80
```

### Kubernetes Operators

```python
# operators/anomaly_operator.py
import kopf
import kubernetes
import yaml
from typing import Dict, Any

@kopf.on.create('anomaly-detection.io', 'v1', 'anomalydetectors')
async def create_anomaly_detector(spec: Dict[str, Any], name: str, namespace: str, **kwargs):
    """Handle creation of AnomalyDetector custom resource."""
    
    # Extract configuration
    algorithm = spec.get('algorithm', 'isolation_forest')
    replicas = spec.get('replicas', 3)
    resources = spec.get('resources', {})
    
    # Create deployment
    deployment = create_deployment_manifest(name, namespace, algorithm, replicas, resources)
    
    # Apply deployment
    apps_v1 = kubernetes.client.AppsV1Api()
    await apps_v1.create_namespaced_deployment(namespace, deployment)
    
    # Create service
    service = create_service_manifest(name, namespace)
    core_v1 = kubernetes.client.CoreV1Api()
    await core_v1.create_namespaced_service(namespace, service)
    
    # Create HPA
    if spec.get('autoscaling', {}).get('enabled', False):
        hpa = create_hpa_manifest(name, namespace, spec['autoscaling'])
        autoscaling_v2 = kubernetes.client.AutoscalingV2Api()
        await autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(namespace, hpa)
    
    return {'message': f'AnomalyDetector {name} created successfully'}

@kopf.on.update('anomaly-detection.io', 'v1', 'anomalydetectors')
async def update_anomaly_detector(spec: Dict[str, Any], name: str, namespace: str, **kwargs):
    """Handle updates to AnomalyDetector custom resource."""
    
    # Update deployment
    apps_v1 = kubernetes.client.AppsV1Api()
    deployment = await apps_v1.read_namespaced_deployment(name, namespace)
    
    # Update replicas
    if 'replicas' in spec:
        deployment.spec.replicas = spec['replicas']
    
    # Update resources
    if 'resources' in spec:
        container = deployment.spec.template.spec.containers[0]
        container.resources = kubernetes.client.V1ResourceRequirements(**spec['resources'])
    
    await apps_v1.patch_namespaced_deployment(name, namespace, deployment)
    
    return {'message': f'AnomalyDetector {name} updated successfully'}

@kopf.on.delete('anomaly-detection.io', 'v1', 'anomalydetectors')
async def delete_anomaly_detector(name: str, namespace: str, **kwargs):
    """Handle deletion of AnomalyDetector custom resource."""
    
    # Delete deployment
    apps_v1 = kubernetes.client.AppsV1Api()
    await apps_v1.delete_namespaced_deployment(name, namespace)
    
    # Delete service
    core_v1 = kubernetes.client.CoreV1Api()
    await core_v1.delete_namespaced_service(name, namespace)
    
    # Delete HPA if exists
    try:
        autoscaling_v2 = kubernetes.client.AutoscalingV2Api()
        await autoscaling_v2.delete_namespaced_horizontal_pod_autoscaler(name, namespace)
    except kubernetes.client.exceptions.ApiException as e:
        if e.status != 404:
            raise
    
    return {'message': f'AnomalyDetector {name} deleted successfully'}

def create_deployment_manifest(name: str, namespace: str, algorithm: str, 
                             replicas: int, resources: Dict) -> Dict:
    """Create deployment manifest."""
    return {
        'apiVersion': 'apps/v1',
        'kind': 'Deployment',
        'metadata': {
            'name': name,
            'namespace': namespace,
            'labels': {
                'app': name,
                'algorithm': algorithm
            }
        },
        'spec': {
            'replicas': replicas,
            'selector': {
                'matchLabels': {
                    'app': name
                }
            },
            'template': {
                'metadata': {
                    'labels': {
                        'app': name,
                        'algorithm': algorithm
                    }
                },
                'spec': {
                    'containers': [{
                        'name': 'anomaly-detector',
                        'image': 'anomaly-detection:latest',
                        'ports': [{'containerPort': 8000}],
                        'env': [
                            {
                                'name': 'ALGORITHM',
                                'value': algorithm
                            }
                        ],
                        'resources': resources,
                        'livenessProbe': {
                            'httpGet': {
                                'path': '/health/live',
                                'port': 8000
                            },
                            'initialDelaySeconds': 30,
                            'periodSeconds': 10
                        },
                        'readinessProbe': {
                            'httpGet': {
                                'path': '/health/ready',
                                'port': 8000
                            },
                            'initialDelaySeconds': 10,
                            'periodSeconds': 5
                        }
                    }]
                }
            }
        }
    }

# Custom Resource Definition
anomaly_detector_crd = """
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: anomalydetectors.anomaly-detection.io
spec:
  group: anomaly-detection.io
  versions:
  - name: v1
    served: true
    storage: true
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              algorithm:
                type: string
                enum: ["isolation_forest", "local_outlier_factor", "one_class_svm"]
                default: "isolation_forest"
              replicas:
                type: integer
                minimum: 1
                maximum: 50
                default: 3
              resources:
                type: object
                properties:
                  requests:
                    type: object
                    properties:
                      cpu:
                        type: string
                      memory:
                        type: string
                  limits:
                    type: object
                    properties:
                      cpu:
                        type: string
                      memory:
                        type: string
              autoscaling:
                type: object
                properties:
                  enabled:
                    type: boolean
                    default: false
                  minReplicas:
                    type: integer
                    minimum: 1
                    default: 3
                  maxReplicas:
                    type: integer
                    maximum: 100
                    default: 20
                  targetCPU:
                    type: integer
                    minimum: 1
                    maximum: 100
                    default: 70
            required: ["algorithm"]
          status:
            type: object
            properties:
              phase:
                type: string
              replicas:
                type: integer
              readyReplicas:
                type: integer
  scope: Namespaced
  names:
    plural: anomalydetectors
    singular: anomalydetector
    kind: AnomalyDetector
    shortNames:
    - ad
"""
```

## Auto-scaling Strategies

### Custom Metrics Scaling

```python
# scaling/custom_metrics.py
from kubernetes import client, config
import time
import numpy as np
from typing import Dict, List

class CustomMetricsScaler:
    """Custom metrics-based auto-scaler for anomaly detection services."""
    
    def __init__(self, namespace: str = 'anomaly-detection'):
        config.load_incluster_config()  # Load config from within cluster
        self.apps_v1 = client.AppsV1Api()
        self.custom_metrics = client.CustomObjectsApi()
        self.namespace = namespace
        
    def get_anomaly_rate_metric(self, deployment_name: str) -> float:
        """Get current anomaly detection rate."""
        try:
            # Query custom metric from Prometheus adapter
            metric = self.custom_metrics.get_namespaced_custom_object(
                group="custom.metrics.k8s.io",
                version="v1beta1",
                namespace=self.namespace,
                plural="deployments.apps",
                name=f"{deployment_name}/anomaly_rate"
            )
            return float(metric['value'])
        except Exception as e:
            print(f"Error getting anomaly rate metric: {e}")
            return 0.0
    
    def get_queue_length_metric(self, service_name: str) -> int:
        """Get current processing queue length."""
        try:
            metric = self.custom_metrics.get_namespaced_custom_object(
                group="custom.metrics.k8s.io",
                version="v1beta1",
                namespace=self.namespace,
                plural="services",
                name=f"{service_name}/queue_length"
            )
            return int(metric['value'])
        except Exception as e:
            print(f"Error getting queue length metric: {e}")
            return 0
    
    def get_prediction_latency_p95(self, deployment_name: str) -> float:
        """Get 95th percentile prediction latency."""
        try:
            metric = self.custom_metrics.get_namespaced_custom_object(
                group="custom.metrics.k8s.io",
                version="v1beta1",
                namespace=self.namespace,
                plural="deployments.apps",
                name=f"{deployment_name}/prediction_latency_p95"
            )
            return float(metric['value'])
        except Exception as e:
            print(f"Error getting latency metric: {e}")
            return 0.0
    
    def calculate_optimal_replicas(self, deployment_name: str, 
                                 current_replicas: int,
                                 target_metrics: Dict) -> int:
        """Calculate optimal number of replicas based on multiple metrics."""
        
        # Get current metrics
        anomaly_rate = self.get_anomaly_rate_metric(deployment_name)
        queue_length = self.get_queue_length_metric(deployment_name)
        latency_p95 = self.get_prediction_latency_p95(deployment_name)
        
        # Calculate scaling factors for each metric
        scaling_factors = []
        
        # Anomaly rate scaling
        target_anomaly_rate = target_metrics.get('target_anomaly_rate', 0.1)
        if anomaly_rate > target_anomaly_rate * 1.5:
            # High anomaly rate requires more resources
            anomaly_factor = min(2.0, anomaly_rate / target_anomaly_rate)
            scaling_factors.append(anomaly_factor)
        
        # Queue length scaling
        target_queue_length = target_metrics.get('target_queue_length', 10)
        if queue_length > target_queue_length:
            queue_factor = min(3.0, queue_length / target_queue_length)
            scaling_factors.append(queue_factor)
        
        # Latency scaling
        target_latency_p95 = target_metrics.get('target_latency_p95', 100.0)  # ms
        if latency_p95 > target_latency_p95:
            latency_factor = min(2.0, latency_p95 / target_latency_p95)
            scaling_factors.append(latency_factor)
        
        # Calculate optimal replicas
        if scaling_factors:
            # Use the maximum scaling factor (most conservative)
            max_scaling_factor = max(scaling_factors)
            optimal_replicas = int(current_replicas * max_scaling_factor)
        else:
            # Consider scaling down if all metrics are good
            if (anomaly_rate < target_anomaly_rate * 0.5 and 
                queue_length < target_queue_length * 0.5 and
                latency_p95 < target_latency_p95 * 0.7):
                optimal_replicas = max(1, int(current_replicas * 0.8))
            else:
                optimal_replicas = current_replicas
        
        # Apply constraints
        min_replicas = target_metrics.get('min_replicas', 2)
        max_replicas = target_metrics.get('max_replicas', 20)
        optimal_replicas = max(min_replicas, min(max_replicas, optimal_replicas))
        
        return optimal_replicas
    
    def scale_deployment(self, deployment_name: str, target_replicas: int):
        """Scale deployment to target replicas."""
        try:
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            # Update replica count
            deployment.spec.replicas = target_replicas
            
            # Apply update
            self.apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace,
                body=deployment
            )
            
            print(f"Scaled {deployment_name} to {target_replicas} replicas")
            
        except Exception as e:
            print(f"Error scaling deployment {deployment_name}: {e}")
    
    def run_scaling_loop(self, deployments: List[str], 
                        target_metrics: Dict,
                        check_interval: int = 30):
        """Run continuous scaling loop."""
        
        while True:
            for deployment_name in deployments:
                try:
                    # Get current deployment
                    deployment = self.apps_v1.read_namespaced_deployment(
                        name=deployment_name,
                        namespace=self.namespace
                    )
                    current_replicas = deployment.spec.replicas
                    
                    # Calculate optimal replicas
                    optimal_replicas = self.calculate_optimal_replicas(
                        deployment_name, current_replicas, target_metrics
                    )
                    
                    # Scale if needed
                    if optimal_replicas != current_replicas:
                        print(f"Scaling {deployment_name}: {current_replicas} -> {optimal_replicas}")
                        self.scale_deployment(deployment_name, optimal_replicas)
                    
                except Exception as e:
                    print(f"Error processing deployment {deployment_name}: {e}")
            
            time.sleep(check_interval)

# Prometheus configuration for custom metrics
prometheus_rules = """
groups:
- name: anomaly_detection_custom_metrics
  rules:
  - record: anomaly_detection:anomaly_rate
    expr: |
      (
        increase(anomaly_detections_total[5m]) / 
        increase(predictions_total[5m])
      ) or on() vector(0)
    labels:
      service: anomaly-detection
  
  - record: anomaly_detection:queue_length
    expr: |
      anomaly_processing_queue_size
    labels:
      service: anomaly-detection
  
  - record: anomaly_detection:prediction_latency_p95
    expr: |
      histogram_quantile(0.95, 
        rate(prediction_duration_seconds_bucket[5m])
      ) * 1000
    labels:
      service: anomaly-detection
  
  - alert: HighAnomalyRate
    expr: anomaly_detection:anomaly_rate > 0.2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High anomaly detection rate"
      description: "Anomaly rate is {{ $value }} which is above 20%"
  
  - alert: HighProcessingLatency
    expr: anomaly_detection:prediction_latency_p95 > 500
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High prediction latency"
      description: "95th percentile latency is {{ $value }}ms"
  
  - alert: LargeProcessingQueue
    expr: anomaly_detection:queue_length > 100
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "Large processing queue"
      description: "Processing queue has {{ $value }} items"
"""

# Kubernetes HPA with custom metrics
custom_hpa_yaml = """
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: anomaly-api-custom-hpa
  namespace: anomaly-detection
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: anomaly-api
  minReplicas: 3
  maxReplicas: 50
  metrics:
  # CPU utilization
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  
  # Memory utilization
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  
  # Custom metric: Queue length
  - type: Object
    object:
      metric:
        name: queue_length
      target:
        type: Value
        value: "10"
      describedObject:
        apiVersion: v1
        kind: Service
        name: anomaly-api-service
  
  # Custom metric: Anomaly rate
  - type: Object
    object:
      metric:
        name: anomaly_rate
      target:
        type: Value
        value: "0.1"
      describedObject:
        apiVersion: apps/v1
        kind: Deployment
        name: anomaly-api
  
  # Custom metric: Prediction latency P95
  - type: Object
    object:
      metric:
        name: prediction_latency_p95
      target:
        type: Value
        value: "100"
      describedObject:
        apiVersion: apps/v1
        kind: Deployment
        name: anomaly-api
  
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
      - type: Pods
        value: 4
        periodSeconds: 60
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
      selectPolicy: Min
"""
```

This deployment guide provides comprehensive strategies for production deployment, from containerization to Kubernetes orchestration, auto-scaling, and operational best practices. The configurations cover security, high availability, monitoring, and scalability requirements for enterprise anomaly detection systems.