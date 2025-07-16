# Enhanced Production Deployment Guide

## Overview

This comprehensive guide provides enterprise-grade deployment instructions for Pynomaly with complete infrastructure automation, monitoring, security hardening, and operational procedures.

## Quick Reference Checklist

### Pre-Deployment Checklist

- [ ] Infrastructure provisioned and configured
- [ ] SSL certificates obtained and installed
- [ ] Database cluster configured with replication
- [ ] Redis cluster configured for high availability
- [ ] Monitoring stack deployed (Prometheus, Grafana, AlertManager)
- [ ] Security hardening applied
- [ ] Backup systems configured and tested
- [ ] CI/CD pipeline configured
- [ ] Load testing completed
- [ ] Disaster recovery procedures tested

### Production Readiness Checklist

- [ ] All health checks passing
- [ ] SSL/TLS certificates valid and auto-renewing
- [ ] Database performance optimized
- [ ] Application performance tuned
- [ ] Security scans completed with no critical issues
- [ ] Monitoring dashboards configured
- [ ] Alerting rules configured and tested
- [ ] Documentation updated
- [ ] Team trained on operational procedures

## Infrastructure as Code

### Terraform Configuration

#### Main Infrastructure

```hcl
# infrastructure/main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC and Networking
resource "aws_vpc" "pynomaly_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "pynomaly-vpc"
    Environment = var.environment
  }
}

resource "aws_subnet" "public_subnets" {
  count = 2

  vpc_id                  = aws_vpc.pynomaly_vpc.id
  cidr_block              = "10.0.${count.index + 1}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name        = "pynomaly-public-subnet-${count.index + 1}"
    Environment = var.environment
  }
}

resource "aws_subnet" "private_subnets" {
  count = 2

  vpc_id            = aws_vpc.pynomaly_vpc.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name        = "pynomaly-private-subnet-${count.index + 1}"
    Environment = var.environment
  }
}

# Internet Gateway
resource "aws_internet_gateway" "pynomaly_igw" {
  vpc_id = aws_vpc.pynomaly_vpc.id

  tags = {
    Name        = "pynomaly-igw"
    Environment = var.environment
  }
}

# RDS Database
resource "aws_db_subnet_group" "pynomaly_db_subnet_group" {
  name       = "pynomaly-db-subnet-group"
  subnet_ids = aws_subnet.private_subnets[*].id

  tags = {
    Name        = "pynomaly-db-subnet-group"
    Environment = var.environment
  }
}

resource "aws_db_instance" "pynomaly_db" {
  identifier = "pynomaly-db"

  engine         = "postgres"
  engine_version = "14.9"
  instance_class = var.db_instance_class

  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type          = "gp3"
  storage_encrypted     = true

  db_name  = "pynomaly"
  username = "pynomaly_user"
  password = var.db_password

  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.pynomaly_db_subnet_group.name

  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "Sun:04:00-Sun:05:00"

  skip_final_snapshot = false
  final_snapshot_identifier = "pynomaly-db-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"

  tags = {
    Name        = "pynomaly-db"
    Environment = var.environment
  }
}

# ElastiCache Redis Cluster
resource "aws_elasticache_subnet_group" "pynomaly_cache_subnet_group" {
  name       = "pynomaly-cache-subnet-group"
  subnet_ids = aws_subnet.private_subnets[*].id
}

resource "aws_elasticache_replication_group" "pynomaly_redis" {
  replication_group_id       = "pynomaly-redis"
  description                = "Redis cluster for Pynomaly"

  port               = 6379
  parameter_group_name = "default.redis7"
  node_type          = var.redis_node_type
  num_cache_clusters = 2

  subnet_group_name  = aws_elasticache_subnet_group.pynomaly_cache_subnet_group.name
  security_group_ids = [aws_security_group.elasticache_sg.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = var.redis_auth_token

  tags = {
    Name        = "pynomaly-redis"
    Environment = var.environment
  }
}

# Application Load Balancer
resource "aws_lb" "pynomaly_alb" {
  name               = "pynomaly-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = aws_subnet.public_subnets[*].id

  enable_deletion_protection = true

  tags = {
    Name        = "pynomaly-alb"
    Environment = var.environment
  }
}

# Auto Scaling Group
resource "aws_autoscaling_group" "pynomaly_asg" {
  name                = "pynomaly-asg"
  vpc_zone_identifier = aws_subnet.private_subnets[*].id
  target_group_arns   = [aws_lb_target_group.pynomaly_tg.arn]
  health_check_type   = "ELB"

  min_size         = var.min_instances
  max_size         = var.max_instances
  desired_capacity = var.desired_instances

  launch_template {
    id      = aws_launch_template.pynomaly_lt.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "pynomaly-instance"
    propagate_at_launch = true
  }

  tag {
    key                 = "Environment"
    value               = var.environment
    propagate_at_launch = true
  }
}
```

#### Security Groups

```hcl
# infrastructure/security_groups.tf
resource "aws_security_group" "alb_sg" {
  name_prefix = "pynomaly-alb-"
  vpc_id      = aws_vpc.pynomaly_vpc.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "pynomaly-alb-sg"
    Environment = var.environment
  }
}

resource "aws_security_group" "app_sg" {
  name_prefix = "pynomaly-app-"
  vpc_id      = aws_vpc.pynomaly_vpc.id

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb_sg.id]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.management_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "pynomaly-app-sg"
    Environment = var.environment
  }
}

resource "aws_security_group" "rds_sg" {
  name_prefix = "pynomaly-rds-"
  vpc_id      = aws_vpc.pynomaly_vpc.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.app_sg.id]
  }

  tags = {
    Name        = "pynomaly-rds-sg"
    Environment = var.environment
  }
}

resource "aws_security_group" "elasticache_sg" {
  name_prefix = "pynomaly-elasticache-"
  vpc_id      = aws_vpc.pynomaly_vpc.id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.app_sg.id]
  }

  tags = {
    Name        = "pynomaly-elasticache-sg"
    Environment = var.environment
  }
}
```

#### Variables

```hcl
# infrastructure/variables.tf
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.large"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "redis_node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.t3.medium"
}

variable "redis_auth_token" {
  description = "Redis authentication token"
  type        = string
  sensitive   = true
}

variable "min_instances" {
  description = "Minimum number of instances"
  type        = number
  default     = 2
}

variable "max_instances" {
  description = "Maximum number of instances"
  type        = number
  default     = 10
}

variable "desired_instances" {
  description = "Desired number of instances"
  type        = number
  default     = 3
}

variable "management_cidr" {
  description = "CIDR block for management access"
  type        = string
  default     = "10.0.0.0/16"
}
```

### Kubernetes Deployment

#### Namespace and ConfigMap

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: pynomaly
  labels:
    name: pynomaly
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: pynomaly-config
  namespace: pynomaly
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  PROMETHEUS_METRICS_ENABLED: "true"
  RATE_LIMIT_REQUESTS: "1000"
  RATE_LIMIT_WINDOW: "3600"
```

#### Secret Management

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: pynomaly-secrets
  namespace: pynomaly
type: Opaque
data:
  DATABASE_URL: <base64-encoded-database-url>
  REDIS_URL: <base64-encoded-redis-url>
  JWT_SECRET: <base64-encoded-jwt-secret>
  ENCRYPTION_KEY: <base64-encoded-encryption-key>
```

#### Application Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pynomaly-app
  namespace: pynomaly
  labels:
    app: pynomaly
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: pynomaly
  template:
    metadata:
      labels:
        app: pynomaly
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: pynomaly
        image: pynomaly:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: ENVIRONMENT
          valueFrom:
            configMapKeyRef:
              name: pynomaly-config
              key: ENVIRONMENT
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: pynomaly-secrets
              key: DATABASE_URL
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: pynomaly-secrets
              key: REDIS_URL
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: pynomaly-secrets
              key: JWT_SECRET
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
          readOnlyRootFilesystem: true
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: var-log
          mountPath: /var/log
      volumes:
      - name: tmp
        emptyDir: {}
      - name: var-log
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: pynomaly-service
  namespace: pynomaly
spec:
  selector:
    app: pynomaly
  ports:
  - name: http
    port: 80
    targetPort: 8000
  type: ClusterIP
```

#### Ingress Configuration

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pynomaly-ingress
  namespace: pynomaly
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.pynomaly.com
    secretName: pynomaly-tls
  rules:
  - host: api.pynomaly.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: pynomaly-service
            port:
              number: 80
```

## Enhanced Monitoring and Observability

### Prometheus Configuration with Custom Metrics

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "pynomaly_rules.yml"
  - "infrastructure_rules.yml"

scrape_configs:
  - job_name: 'pynomaly-app'
    static_configs:
      - targets: ['pynomaly-service:80']
    metrics_path: /metrics
    scrape_interval: 15s
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
        replacement: 'pynomaly-app'

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'nginx-exporter'
    static_configs:
      - targets: ['nginx-exporter:9113']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Advanced Alerting Rules

```yaml
# monitoring/pynomaly_rules.yml
groups:
  - name: pynomaly.rules
    rules:
      # Application Health
      - alert: Pynomaly_HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
          service: pynomaly
        annotations:
          summary: "High error rate detected in Pynomaly"
          description: "Error rate is {{ $value | humanizePercentage }} for the last 5 minutes"

      - alert: Pynomaly_HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
          service: pynomaly
        annotations:
          summary: "High latency detected in Pynomaly"
          description: "95th percentile latency is {{ $value }}s"

      - alert: Pynomaly_LowThroughput
        expr: rate(http_requests_total[5m]) < 1
        for: 10m
        labels:
          severity: warning
          service: pynomaly
        annotations:
          summary: "Low request throughput in Pynomaly"
          description: "Request rate is {{ $value }} requests/second"

      # Resource Usage
      - alert: Pynomaly_HighMemoryUsage
        expr: (container_memory_usage_bytes{container="pynomaly"} / container_spec_memory_limit_bytes{container="pynomaly"}) > 0.8
        for: 5m
        labels:
          severity: warning
          service: pynomaly
        annotations:
          summary: "High memory usage in Pynomaly container"
          description: "Memory usage is {{ $value | humanizePercentage }}"

      - alert: Pynomaly_HighCPUUsage
        expr: rate(container_cpu_usage_seconds_total{container="pynomaly"}[5m]) > 0.8
        for: 5m
        labels:
          severity: warning
          service: pynomaly
        annotations:
          summary: "High CPU usage in Pynomaly container"
          description: "CPU usage is {{ $value | humanizePercentage }}"

      # Database Alerts
      - alert: Pynomaly_DatabaseConnectionsHigh
        expr: pg_stat_database_numbackends{datname="pynomaly"} > 80
        for: 5m
        labels:
          severity: warning
          service: database
        annotations:
          summary: "High number of database connections"
          description: "Number of connections is {{ $value }}"

      - alert: Pynomaly_DatabaseSlowQueries
        expr: pg_stat_statements_mean_time_ms > 1000
        for: 5m
        labels:
          severity: warning
          service: database
        annotations:
          summary: "Slow database queries detected"
          description: "Average query time is {{ $value }}ms"

      # Redis Alerts
      - alert: Pynomaly_RedisMemoryHigh
        expr: (redis_memory_used_bytes / redis_memory_max_bytes) > 0.8
        for: 5m
        labels:
          severity: warning
          service: redis
        annotations:
          summary: "High Redis memory usage"
          description: "Redis memory usage is {{ $value | humanizePercentage }}"

      - alert: Pynomaly_RedisConnectionsHigh
        expr: redis_connected_clients > 100
        for: 5m
        labels:
          severity: warning
          service: redis
        annotations:
          summary: "High number of Redis connections"
          description: "Number of Redis connections is {{ $value }}"
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "id": null,
    "title": "Pynomaly Production Dashboard",
    "tags": ["pynomaly", "production"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{job=\"pynomaly-app\"}[5m])) by (method, status)",
            "legendFormat": "{{method}} {{status}}"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 0
        }
      },
      {
        "id": 2,
        "title": "Response Time Percentiles",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket{job=\"pynomaly-app\"}[5m]))",
            "legendFormat": "50th percentile"
          },
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job=\"pynomaly-app\"}[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{job=\"pynomaly-app\"}[5m]))",
            "legendFormat": "99th percentile"
          }
        ],
        "yAxes": [
          {
            "label": "Seconds",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 0
        }
      },
      {
        "id": 3,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{job=\"pynomaly-app\",status=~\"4..|5..\"}[5m])) by (status)",
            "legendFormat": "{{status}} errors"
          }
        ],
        "yAxes": [
          {
            "label": "Errors/sec",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 8
        }
      },
      {
        "id": 4,
        "title": "Active Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(pynomaly_active_connections) by (instance)",
            "legendFormat": "{{instance}}"
          }
        ],
        "yAxes": [
          {
            "label": "Connections",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 8
        }
      }
    ]
  }
}
```

## Advanced Security Configuration

### Web Application Firewall (WAF) Rules

```json
{
  "Rules": [
    {
      "Name": "PynomalaySQLInjectionRule",
      "Priority": 1,
      "Statement": {
        "ManagedRuleGroupStatement": {
          "VendorName": "AWS",
          "Name": "AWSManagedRulesSQLiRuleSet"
        }
      },
      "Action": {
        "Block": {}
      },
      "VisibilityConfig": {
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "PynomalaySQLInjectionRule"
      }
    },
    {
      "Name": "PynomayXSSRule",
      "Priority": 2,
      "Statement": {
        "ManagedRuleGroupStatement": {
          "VendorName": "AWS",
          "Name": "AWSManagedRulesCommonRuleSet"
        }
      },
      "Action": {
        "Block": {}
      },
      "VisibilityConfig": {
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "PynomayXSSRule"
      }
    },
    {
      "Name": "PynomalaRateLimitRule",
      "Priority": 3,
      "Statement": {
        "RateBasedStatement": {
          "Limit": 2000,
          "AggregateKeyType": "IP"
        }
      },
      "Action": {
        "Block": {}
      },
      "VisibilityConfig": {
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "PynomalaRateLimitRule"
      }
    }
  ]
}
```

### Security Scanning Automation

```bash
#!/bin/bash
# scripts/security_scan.sh

set -e

echo "Starting comprehensive security scan..."

# Container image scanning
echo "Scanning container images..."
trivy image --severity HIGH,CRITICAL pynomaly:latest

# Dependency vulnerability scanning
echo "Scanning dependencies..."
safety check --file requirements.txt
pip-audit --requirement requirements.txt

# Static code analysis
echo "Running static analysis..."
bandit -r src/pynomaly/ -f json -o security_report.json
semgrep --config=auto src/pynomaly/

# Secret scanning
echo "Scanning for secrets..."
detect-secrets scan --all-files --baseline .secrets.baseline

# Infrastructure scanning
echo "Scanning infrastructure..."
checkov -d infrastructure/ --framework terraform

# SSL/TLS configuration testing
echo "Testing SSL configuration..."
testssl.sh --quiet --jsonfile ssl_report.json https://api.pynomaly.com

# Generate security report
python scripts/generate_security_report.py

echo "Security scan completed. Review reports in security_reports/ directory."
```

## Performance Optimization and Tuning

### Database Performance Tuning

```sql
-- scripts/database_performance_tuning.sql

-- Enable query statistics collection
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Optimize PostgreSQL configuration for production
-- Add to postgresql.conf:

-- Memory Configuration
-- shared_buffers = 25% of RAM (e.g., 8GB for 32GB RAM)
-- effective_cache_size = 75% of RAM (e.g., 24GB for 32GB RAM)
-- work_mem = RAM / max_connections / 4 (e.g., 64MB)
-- maintenance_work_mem = RAM / 16 (e.g., 2GB for 32GB RAM)

-- Connection Configuration
-- max_connections = 200
-- max_prepared_transactions = 200

-- WAL Configuration
-- wal_buffers = 64MB
-- checkpoint_completion_target = 0.9
-- checkpoint_timeout = 15min
-- max_wal_size = 2GB
-- min_wal_size = 1GB

-- Logging Configuration
-- log_min_duration_statement = 1000  -- Log slow queries
-- log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
-- log_checkpoints = on
-- log_connections = on
-- log_disconnections = on
-- log_lock_waits = on

-- Performance Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email_active 
ON users(email) WHERE is_active = true;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_keys_user_id_active 
ON api_keys(user_id) WHERE is_active = true;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_timestamp_user 
ON audit_logs(timestamp, user_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detection_results_created_at 
ON detection_results(created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detection_results_model_id 
ON detection_results(model_id, created_at);

-- Analyze table statistics
ANALYZE;

-- Create monitoring views
CREATE OR REPLACE VIEW slow_queries AS
SELECT query, mean_time, calls, total_time, rows, 
       mean_time / calls as avg_time_per_call
FROM pg_stat_statements 
WHERE calls > 100 
ORDER BY mean_time DESC 
LIMIT 20;

CREATE OR REPLACE VIEW table_stats AS
SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del,
       n_live_tup, n_dead_tup, last_vacuum, last_autovacuum, last_analyze
FROM pg_stat_user_tables 
ORDER BY n_live_tup DESC;
```

### Application Performance Configuration

```python
# src/pynomaly/infrastructure/config/performance.py
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PerformanceConfig:
    """Production performance configuration"""
    
    # Database connection pooling
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 30
    DATABASE_POOL_TIMEOUT: int = 30
    DATABASE_POOL_RECYCLE: int = 3600
    DATABASE_POOL_PRE_PING: bool = True
    
    # Redis connection pooling
    REDIS_MAX_CONNECTIONS: int = 100
    REDIS_CONNECTION_POOL_TIMEOUT: int = 20
    REDIS_SOCKET_KEEPALIVE: bool = True
    REDIS_SOCKET_KEEPALIVE_OPTIONS: Dict[str, int] = None
    
    # HTTP client configuration
    HTTP_TIMEOUT: int = 30
    HTTP_MAX_CONNECTIONS: int = 100
    HTTP_MAX_KEEPALIVE_CONNECTIONS: int = 20
    
    # Async processing
    WORKER_PROCESSES: int = 4
    WORKER_CONNECTIONS: int = 1000
    WORKER_TIMEOUT: int = 30
    WORKER_KEEPALIVE: int = 2
    
    # Caching configuration
    CACHE_DEFAULT_TIMEOUT: int = 300
    CACHE_MAX_ENTRIES: int = 10000
    MEMORY_CACHE_SIZE_MB: int = 100
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 1000
    RATE_LIMIT_WINDOW: int = 3600
    RATE_LIMIT_BURST: int = 100
    
    # Request processing
    MAX_REQUEST_SIZE: int = 10 * 1024 * 1024  # 10MB
    REQUEST_TIMEOUT: int = 60
    
    def __post_init__(self):
        if self.REDIS_SOCKET_KEEPALIVE_OPTIONS is None:
            self.REDIS_SOCKET_KEEPALIVE_OPTIONS = {
                "TCP_KEEPINTVL": 1,
                "TCP_KEEPCNT": 3,
                "TCP_KEEPIDLE": 1
            }
    
    @classmethod
    def from_environment(cls) -> 'PerformanceConfig':
        """Load configuration from environment variables"""
        return cls(
            DATABASE_POOL_SIZE=int(os.getenv('DATABASE_POOL_SIZE', '20')),
            DATABASE_MAX_OVERFLOW=int(os.getenv('DATABASE_MAX_OVERFLOW', '30')),
            REDIS_MAX_CONNECTIONS=int(os.getenv('REDIS_MAX_CONNECTIONS', '100')),
            WORKER_PROCESSES=int(os.getenv('WORKER_PROCESSES', '4')),
            WORKER_CONNECTIONS=int(os.getenv('WORKER_CONNECTIONS', '1000')),
            CACHE_DEFAULT_TIMEOUT=int(os.getenv('CACHE_DEFAULT_TIMEOUT', '300')),
            RATE_LIMIT_REQUESTS=int(os.getenv('RATE_LIMIT_REQUESTS', '1000')),
            RATE_LIMIT_WINDOW=int(os.getenv('RATE_LIMIT_WINDOW', '3600')),
        )
```

## Comprehensive Backup and Disaster Recovery

### Automated Backup System

```bash
#!/bin/bash
# scripts/comprehensive_backup.sh

set -euo pipefail

# Configuration
BACKUP_BASE_DIR="/opt/backups"
S3_BUCKET="pynomaly-backups"
RETENTION_DAYS=30
ENCRYPTION_KEY_FILE="/opt/pynomaly/keys/backup.key"

# Logging
LOG_FILE="/var/log/pynomaly/backup.log"
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Create backup directories
mkdir -p "$BACKUP_BASE_DIR"/{database,application,configs,keys}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 1. Database Backup
log "Starting database backup..."
DB_BACKUP_FILE="$BACKUP_BASE_DIR/database/pynomaly_db_$TIMESTAMP.sql"

pg_dump \
    --host="$DATABASE_HOST" \
    --username="$DATABASE_USER" \
    --dbname="$DATABASE_NAME" \
    --verbose \
    --format=custom \
    --compress=9 \
    --file="$DB_BACKUP_FILE"

# Encrypt database backup
gpg --cipher-algo AES256 --compress-algo 1 --symmetric \
    --output "$DB_BACKUP_FILE.gpg" \
    --passphrase-file "$ENCRYPTION_KEY_FILE" \
    "$DB_BACKUP_FILE"

rm "$DB_BACKUP_FILE"
log "Database backup completed: $DB_BACKUP_FILE.gpg"

# 2. Redis Backup
log "Starting Redis backup..."
REDIS_BACKUP_FILE="$BACKUP_BASE_DIR/database/redis_$TIMESTAMP.rdb"

redis-cli --rdb "$REDIS_BACKUP_FILE"
gzip "$REDIS_BACKUP_FILE"
log "Redis backup completed: $REDIS_BACKUP_FILE.gz"

# 3. Application Code Backup
log "Starting application backup..."
APP_BACKUP_FILE="$BACKUP_BASE_DIR/application/pynomaly_app_$TIMESTAMP.tar.gz"

tar -czf "$APP_BACKUP_FILE" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.env*' \
    --exclude='logs/*' \
    /opt/pynomaly

log "Application backup completed: $APP_BACKUP_FILE"

# 4. Configuration Backup
log "Starting configuration backup..."
CONFIG_BACKUP_FILE="$BACKUP_BASE_DIR/configs/configs_$TIMESTAMP.tar.gz"

tar -czf "$CONFIG_BACKUP_FILE" \
    /etc/nginx \
    /etc/systemd/system/pynomaly* \
    /opt/pynomaly/.env.production \
    /opt/pynomaly/docker-compose*.yml

log "Configuration backup completed: $CONFIG_BACKUP_FILE"

# 5. SSL Certificates Backup
log "Starting SSL certificates backup..."
SSL_BACKUP_FILE="$BACKUP_BASE_DIR/configs/ssl_$TIMESTAMP.tar.gz"

if [ -d "/etc/letsencrypt" ]; then
    tar -czf "$SSL_BACKUP_FILE" /etc/letsencrypt
    log "SSL certificates backup completed: $SSL_BACKUP_FILE"
fi

# 6. Upload to S3
log "Uploading backups to S3..."
aws s3 sync "$BACKUP_BASE_DIR" "s3://$S3_BUCKET/backups/$TIMESTAMP" \
    --storage-class STANDARD_IA \
    --server-side-encryption AES256

# 7. Verify backups
log "Verifying backup integrity..."
for backup_file in "$BACKUP_BASE_DIR"/**/*"$TIMESTAMP"*; do
    if [[ -f "$backup_file" ]]; then
        # Verify file integrity
        if [[ "$backup_file" == *.gpg ]]; then
            gpg --quiet --batch --passphrase-file "$ENCRYPTION_KEY_FILE" \
                --decrypt "$backup_file" > /dev/null
        elif [[ "$backup_file" == *.gz ]]; then
            gzip -t "$backup_file"
        elif [[ "$backup_file" == *.tar.gz ]]; then
            tar -tzf "$backup_file" > /dev/null
        fi
        log "Verified: $backup_file"
    fi
done

# 8. Cleanup old backups
log "Cleaning up old backups..."
find "$BACKUP_BASE_DIR" -type f -mtime +$RETENTION_DAYS -delete

# Delete old S3 backups
aws s3 ls "s3://$S3_BUCKET/backups/" | \
    awk '$1 <= "'$(date -d "$RETENTION_DAYS days ago" '+%Y-%m-%d')'" {print $4}' | \
    xargs -I {} aws s3 rm "s3://$S3_BUCKET/backups/{}" --recursive

log "Backup process completed successfully"

# Send notification
curl -X POST "$SLACK_WEBHOOK_URL" \
    -H 'Content-type: application/json' \
    --data '{"text":"🎯 Pynomaly backup completed successfully for timestamp: '$TIMESTAMP'"}'
```

### Disaster Recovery Procedures

```bash
#!/bin/bash
# scripts/disaster_recovery.sh

set -euo pipefail

# Configuration
RESTORE_TIMESTAMP="$1"
S3_BUCKET="pynomaly-backups"
TEMP_RESTORE_DIR="/tmp/pynomaly_restore"
ENCRYPTION_KEY_FILE="/opt/pynomaly/keys/backup.key"

if [ -z "$RESTORE_TIMESTAMP" ]; then
    echo "Usage: $0 <timestamp>"
    echo "Available backups:"
    aws s3 ls "s3://$S3_BUCKET/backups/" | grep DIR
    exit 1
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Starting disaster recovery for timestamp: $RESTORE_TIMESTAMP"

# 1. Create restore directory
mkdir -p "$TEMP_RESTORE_DIR"
cd "$TEMP_RESTORE_DIR"

# 2. Download backups from S3
log "Downloading backups from S3..."
aws s3 sync "s3://$S3_BUCKET/backups/$RESTORE_TIMESTAMP" .

# 3. Stop application services
log "Stopping application services..."
docker-compose -f /opt/pynomaly/docker-compose.prod.yml down
systemctl stop nginx

# 4. Restore database
log "Restoring database..."
DB_BACKUP_FILE="database/pynomaly_db_$RESTORE_TIMESTAMP.sql.gpg"

if [ -f "$DB_BACKUP_FILE" ]; then
    # Decrypt and restore database
    gpg --quiet --batch --passphrase-file "$ENCRYPTION_KEY_FILE" \
        --decrypt "$DB_BACKUP_FILE" | \
    pg_restore \
        --host="$DATABASE_HOST" \
        --username="$DATABASE_USER" \
        --dbname="$DATABASE_NAME" \
        --clean \
        --if-exists \
        --verbose
    
    log "Database restored successfully"
else
    log "ERROR: Database backup file not found"
    exit 1
fi

# 5. Restore Redis
log "Restoring Redis..."
REDIS_BACKUP_FILE="database/redis_$RESTORE_TIMESTAMP.rdb.gz"

if [ -f "$REDIS_BACKUP_FILE" ]; then
    systemctl stop redis
    gunzip -c "$REDIS_BACKUP_FILE" > /var/lib/redis/dump.rdb
    chown redis:redis /var/lib/redis/dump.rdb
    systemctl start redis
    log "Redis restored successfully"
fi

# 6. Restore application
log "Restoring application..."
APP_BACKUP_FILE="application/pynomaly_app_$RESTORE_TIMESTAMP.tar.gz"

if [ -f "$APP_BACKUP_FILE" ]; then
    # Backup current application
    mv /opt/pynomaly "/opt/pynomaly.backup.$(date +%Y%m%d_%H%M%S)"
    
    # Extract application backup
    tar -xzf "$APP_BACKUP_FILE" -C /
    chown -R pynomaly:pynomaly /opt/pynomaly
    log "Application restored successfully"
fi

# 7. Restore configurations
log "Restoring configurations..."
CONFIG_BACKUP_FILE="configs/configs_$RESTORE_TIMESTAMP.tar.gz"

if [ -f "$CONFIG_BACKUP_FILE" ]; then
    tar -xzf "$CONFIG_BACKUP_FILE" -C /
    log "Configurations restored successfully"
fi

# 8. Restore SSL certificates
SSL_BACKUP_FILE="configs/ssl_$RESTORE_TIMESTAMP.tar.gz"
if [ -f "$SSL_BACKUP_FILE" ]; then
    tar -xzf "$SSL_BACKUP_FILE" -C /
    log "SSL certificates restored successfully"
fi

# 9. Start services
log "Starting services..."
systemctl start nginx
docker-compose -f /opt/pynomaly/docker-compose.prod.yml up -d

# 10. Verify restoration
log "Verifying restoration..."
sleep 30

# Health checks
curl -f http://localhost/health || {
    log "ERROR: Health check failed"
    exit 1
}

# Database connectivity
docker-compose -f /opt/pynomaly/docker-compose.prod.yml exec -T app python -c "
import psycopg2
conn = psycopg2.connect('$DATABASE_URL')
cursor = conn.cursor()
cursor.execute('SELECT 1')
print('Database connection successful')
conn.close()
"

log "Disaster recovery completed successfully"

# Cleanup
rm -rf "$TEMP_RESTORE_DIR"

# Send notification
curl -X POST "$SLACK_WEBHOOK_URL" \
    -H 'Content-type: application/json' \
    --data '{"text":"🚨 Pynomaly disaster recovery completed for timestamp: '$RESTORE_TIMESTAMP'"}'
```

## Load Testing and Capacity Planning

### Load Testing Configuration

```yaml
# load_testing/k6_test.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

export let errorRate = new Rate('errors');

export let options = {
  stages: [
    { duration: '2m', target: 100 }, // Ramp up to 100 users
    { duration: '5m', target: 100 }, // Stay at 100 users
    { duration: '2m', target: 200 }, // Ramp up to 200 users
    { duration: '5m', target: 200 }, // Stay at 200 users
    { duration: '2m', target: 300 }, // Ramp up to 300 users
    { duration: '5m', target: 300 }, // Stay at 300 users
    { duration: '2m', target: 0 },   // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<1000'], // 95% of requests must complete below 1s
    http_req_failed: ['rate<0.1'],     // Error rate must be below 10%
    errors: ['rate<0.1'],              // Custom error rate
  },
};

const BASE_URL = 'https://api.pynomaly.com';

export function setup() {
  // Login and get auth token
  let loginRes = http.post(`${BASE_URL}/api/v1/auth/login`, {
    email: 'test@example.com',
    password: 'testpassword'
  });
  
  return { token: loginRes.json('access_token') };
}

export default function (data) {
  let params = {
    headers: {
      'Authorization': `Bearer ${data.token}`,
      'Content-Type': 'application/json',
    },
  };

  // Test health endpoint
  let healthRes = http.get(`${BASE_URL}/health`, params);
  let healthCheck = check(healthRes, {
    'health status is 200': (r) => r.status === 200,
    'health response time < 500ms': (r) => r.timings.duration < 500,
  });
  errorRate.add(!healthCheck);

  // Test API endpoints
  let apiRes = http.get(`${BASE_URL}/api/v1/detectors`, params);
  let apiCheck = check(apiRes, {
    'API status is 200': (r) => r.status === 200,
    'API response time < 1000ms': (r) => r.timings.duration < 1000,
  });
  errorRate.add(!apiCheck);

  // Test detection endpoint
  let detectionPayload = {
    data: [
      [1.0, 2.0, 3.0, 4.0, 5.0],
      [2.0, 3.0, 4.0, 5.0, 6.0],
      [3.0, 4.0, 5.0, 6.0, 7.0]
    ],
    detector_type: 'isolation_forest'
  };

  let detectionRes = http.post(
    `${BASE_URL}/api/v1/detect`, 
    JSON.stringify(detectionPayload), 
    params
  );
  
  let detectionCheck = check(detectionRes, {
    'detection status is 200': (r) => r.status === 200,
    'detection response time < 2000ms': (r) => r.timings.duration < 2000,
    'detection has results': (r) => r.json('results') !== undefined,
  });
  errorRate.add(!detectionCheck);

  sleep(1);
}

export function teardown(data) {
  // Logout
  http.post(`${BASE_URL}/api/v1/auth/logout`, {}, {
    headers: { 'Authorization': `Bearer ${data.token}` }
  });
}
```

### Capacity Planning Script

```python
#!/usr/bin/env python3
# scripts/capacity_planning.py

import argparse
import json
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List

class CapacityPlanner:
    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url
        self.metrics = {}
    
    def query_prometheus(self, query: str, start_time: datetime, end_time: datetime) -> Dict:
        """Query Prometheus for metrics data"""
        params = {
            'query': query,
            'start': start_time.isoformat(),
            'end': end_time.isoformat(),
            'step': '1m'
        }
        
        response = requests.get(f"{self.prometheus_url}/api/v1/query_range", params=params)
        return response.json()
    
    def analyze_cpu_usage(self, days: int = 30) -> Dict:
        """Analyze CPU usage trends"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        query = 'avg(rate(container_cpu_usage_seconds_total{container="pynomaly"}[5m])) * 100'
        data = self.query_prometheus(query, start_time, end_time)
        
        values = [float(point[1]) for point in data['data']['result'][0]['values']]
        
        return {
            'average_cpu': sum(values) / len(values),
            'max_cpu': max(values),
            'min_cpu': min(values),
            'p95_cpu': sorted(values)[int(len(values) * 0.95)],
            'trend': 'increasing' if values[-1] > values[0] else 'decreasing'
        }
    
    def analyze_memory_usage(self, days: int = 30) -> Dict:
        """Analyze memory usage trends"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        query = 'avg(container_memory_usage_bytes{container="pynomaly"}) / 1024 / 1024'
        data = self.query_prometheus(query, start_time, end_time)
        
        values = [float(point[1]) for point in data['data']['result'][0]['values']]
        
        return {
            'average_memory_mb': sum(values) / len(values),
            'max_memory_mb': max(values),
            'min_memory_mb': min(values),
            'p95_memory_mb': sorted(values)[int(len(values) * 0.95)],
            'trend': 'increasing' if values[-1] > values[0] else 'decreasing'
        }
    
    def analyze_request_rate(self, days: int = 30) -> Dict:
        """Analyze request rate trends"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        query = 'sum(rate(http_requests_total{job="pynomaly-app"}[5m]))'
        data = self.query_prometheus(query, start_time, end_time)
        
        values = [float(point[1]) for point in data['data']['result'][0]['values']]
        
        return {
            'average_rps': sum(values) / len(values),
            'max_rps': max(values),
            'min_rps': min(values),
            'p95_rps': sorted(values)[int(len(values) * 0.95)],
            'trend': 'increasing' if values[-1] > values[0] else 'decreasing'
        }
    
    def analyze_database_connections(self, days: int = 30) -> Dict:
        """Analyze database connection usage"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        query = 'avg(pg_stat_database_numbackends{datname="pynomaly"})'
        data = self.query_prometheus(query, start_time, end_time)
        
        values = [float(point[1]) for point in data['data']['result'][0]['values']]
        
        return {
            'average_connections': sum(values) / len(values),
            'max_connections': max(values),
            'p95_connections': sorted(values)[int(len(values) * 0.95)],
            'utilization_percent': (max(values) / 200) * 100  # Assuming max_connections = 200
        }
    
    def predict_capacity_needs(self, growth_rate: float = 0.2) -> Dict:
        """Predict capacity needs based on current trends and growth rate"""
        cpu_analysis = self.analyze_cpu_usage()
        memory_analysis = self.analyze_memory_usage()
        request_analysis = self.analyze_request_rate()
        db_analysis = self.analyze_database_connections()
        
        # Predict resource needs for 6 months with given growth rate
        predicted_rps = request_analysis['p95_rps'] * (1 + growth_rate)
        predicted_cpu = cpu_analysis['p95_cpu'] * (1 + growth_rate)
        predicted_memory = memory_analysis['p95_memory_mb'] * (1 + growth_rate)
        predicted_db_connections = db_analysis['p95_connections'] * (1 + growth_rate)
        
        recommendations = []
        
        # CPU recommendations
        if predicted_cpu > 80:
            recommendations.append("Scale up CPU resources or add more instances")
        
        # Memory recommendations
        if predicted_memory > 1500:  # Assuming 2GB limit
            recommendations.append("Increase memory allocation or optimize memory usage")
        
        # Database recommendations
        if predicted_db_connections > 160:  # 80% of 200 max connections
            recommendations.append("Increase database connection pool or add read replicas")
        
        # Instance scaling recommendations
        current_instances = 3  # Current number of instances
        requests_per_instance = predicted_rps / current_instances
        
        if requests_per_instance > 100:  # Threshold for instance scaling
            recommended_instances = int((predicted_rps / 100) + 1)
            recommendations.append(f"Scale to {recommended_instances} instances")
        
        return {
            'current_metrics': {
                'cpu_p95': cpu_analysis['p95_cpu'],
                'memory_p95_mb': memory_analysis['p95_memory_mb'],
                'rps_p95': request_analysis['p95_rps'],
                'db_connections_p95': db_analysis['p95_connections']
            },
            'predicted_metrics': {
                'cpu_p95': predicted_cpu,
                'memory_p95_mb': predicted_memory,
                'rps_p95': predicted_rps,
                'db_connections_p95': predicted_db_connections
            },
            'recommendations': recommendations,
            'growth_rate_used': growth_rate,
            'prediction_horizon': '6 months'
        }
    
    def generate_report(self, output_file: str = None) -> Dict:
        """Generate comprehensive capacity planning report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'cpu_analysis': self.analyze_cpu_usage(),
            'memory_analysis': self.analyze_memory_usage(),
            'request_analysis': self.analyze_request_rate(),
            'database_analysis': self.analyze_database_connections(),
            'capacity_predictions': self.predict_capacity_needs()
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Pynomaly Capacity Planning Tool')
    parser.add_argument('--prometheus-url', required=True, help='Prometheus server URL')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--growth-rate', type=float, default=0.2, help='Expected growth rate (default: 20%)')
    
    args = parser.parse_args()
    
    planner = CapacityPlanner(args.prometheus_url)
    report = planner.generate_report(args.output)
    
    print("Capacity Planning Report")
    print("=" * 50)
    print(f"Generated at: {report['timestamp']}")
    print(f"\nCurrent P95 Metrics:")
    print(f"  CPU Usage: {report['cpu_analysis']['p95_cpu']:.1f}%")
    print(f"  Memory Usage: {report['memory_analysis']['p95_memory_mb']:.1f} MB")
    print(f"  Request Rate: {report['request_analysis']['p95_rps']:.1f} RPS")
    print(f"  DB Connections: {report['database_analysis']['p95_connections']:.1f}")
    
    print(f"\nPredicted Metrics (6 months, {args.growth_rate*100}% growth):")
    predictions = report['capacity_predictions']
    print(f"  CPU Usage: {predictions['predicted_metrics']['cpu_p95']:.1f}%")
    print(f"  Memory Usage: {predictions['predicted_metrics']['memory_p95_mb']:.1f} MB")
    print(f"  Request Rate: {predictions['predicted_metrics']['rps_p95']:.1f} RPS")
    print(f"  DB Connections: {predictions['predicted_metrics']['db_connections_p95']:.1f}")
    
    if predictions['recommendations']:
        print(f"\nRecommendations:")
        for rec in predictions['recommendations']:
            print(f"  • {rec}")
    else:
        print(f"\nNo scaling recommendations at this time.")

if __name__ == '__main__':
    main()
```

## Operational Runbooks

### Incident Response Runbook

```markdown
# Incident Response Runbook

## Severity Levels

### P0 - Critical (Response: Immediate)
- Complete service outage
- Data corruption or loss
- Security breach
- Multiple availability zones down

### P1 - High (Response: 15 minutes)
- Significant service degradation
- Single availability zone down
- API error rate > 5%
- Database connection failures

### P2 - Medium (Response: 2 hours)
- Minor service degradation
- Single instance failures
- API response time > 2s
- Non-critical feature outages

### P3 - Low (Response: 24 hours)
- Documentation issues
- Minor UI bugs
- Non-critical monitoring alerts

## Response Procedures

### Step 1: Assess and Communicate
1. Acknowledge the incident in Slack #incidents channel
2. Create incident ticket in JIRA/GitHub
3. Determine severity level
4. Notify on-call team members
5. If P0/P1: Notify stakeholders via status page

### Step 2: Initial Investigation
1. Check monitoring dashboards
2. Review recent deployments
3. Check infrastructure status
4. Examine application logs
5. Verify external dependencies

### Step 3: Mitigation
1. Implement immediate workarounds if available
2. Scale resources if needed
3. Rollback recent changes if suspected cause
4. Redirect traffic if necessary
5. Document all actions taken

### Step 4: Communication
1. Update incident ticket every 30 minutes
2. Post updates in #incidents channel
3. Update status page for P0/P1 incidents
4. Communicate ETA for resolution

### Step 5: Resolution
1. Implement permanent fix
2. Verify solution works
3. Monitor for 30 minutes post-fix
4. Update all communication channels
5. Close incident ticket

### Step 6: Post-Mortem
1. Schedule post-mortem meeting within 48 hours
2. Create post-mortem document
3. Identify root cause and contributing factors
4. Define action items to prevent recurrence
5. Update runbooks and monitoring
```

This enhanced production deployment guide provides comprehensive coverage of all the acceptance criteria from issue #108, including infrastructure as code examples, advanced monitoring, security hardening, disaster recovery procedures, and operational runbooks. The guide is production-ready and follows enterprise best practices.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Create Production Deployment Guide (Issue #108)", "status": "completed", "priority": "medium", "id": "27"}, {"content": "Add AutoML Feature Flag Support (Issue #116)", "status": "pending", "priority": "medium", "id": "29"}, {"content": "Create feature flag documentation", "status": "pending", "priority": "medium", "id": "38"}, {"content": "Update documentation to reflect implemented features", "status": "pending", "priority": "medium", "id": "41"}]
