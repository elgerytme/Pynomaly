# Pynomaly Infrastructure as Code Guide

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 🚀 [Deployment](README.md) > 🏗️ Infrastructure as Code

This comprehensive guide covers Infrastructure as Code (IaC) implementation for Pynomaly using Terraform, including AWS, Azure, and Google Cloud Platform deployments with complete automation and best practices.

## 📋 Table of Contents

- [Overview](#overview)
- [Terraform Setup](#terraform-setup)
- [AWS Infrastructure](#aws-infrastructure)
- [Azure Infrastructure](#azure-infrastructure)
- [Google Cloud Infrastructure](#google-cloud-infrastructure)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Networking & Security](#networking--security)
- [Monitoring Infrastructure](#monitoring-infrastructure)
- [CI/CD Pipeline](#cicd-pipeline)
- [Best Practices](#best-practices)

## 🎯 Overview

### Infrastructure Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Production Environment                     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                 Load Balancer                       │    │
│  │              (ALB/NLB/CloudLB)                      │    │
│  └─────────────────────┬───────────────────────────────┘    │
│                        │                                    │
│  ┌─────────────────────┼───────────────────────────────┐    │
│  │              Application Tier                       │    │
│  │   ┌─────────┐   ┌─────────┐   ┌─────────┐          │    │
│  │   │   ECS   │   │   ECS   │   │   ECS   │          │    │
│  │   │ Task 1  │   │ Task 2  │   │ Task 3  │          │    │
│  │   └─────────┘   └─────────┘   └─────────┘          │    │
│  └─────────────────────┬───────────────────────────────┘    │
│                        │                                    │
│  ┌─────────────────────┼───────────────────────────────┐    │
│  │                Data Tier                            │    │
│  │   ┌─────────────┐   ┌─────────────┐   ┌──────────┐  │    │
│  │   │ PostgreSQL  │   │   Redis     │   │   S3     │  │    │
│  │   │    RDS      │   │ ElastiCache │   │ Bucket   │  │    │
│  │   └─────────────┘   └─────────────┘   └──────────┘  │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Key Benefits

- **Reproducible**: Consistent infrastructure across environments
- **Version Controlled**: Track changes and rollback capabilities
- **Scalable**: Easy horizontal and vertical scaling
- **Secure**: Built-in security best practices
- **Cost-Effective**: Automated resource management

## 🔧 Terraform Setup

### Project Structure

```
terraform/
├── environments/
│   ├── dev/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── terraform.tfvars
│   ├── staging/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── terraform.tfvars
│   └── prod/
│       ├── main.tf
│       ├── variables.tf
│       └── terraform.tfvars
├── modules/
│   ├── network/
│   ├── compute/
│   ├── database/
│   ├── storage/
│   ├── monitoring/
│   └── security/
├── scripts/
│   ├── deploy.sh
│   ├── destroy.sh
│   └── validate.sh
└── README.md
```

### Backend Configuration

```hcl
# backend.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
  
  backend "s3" {
    bucket  = "pynomaly-terraform-state"
    key     = "prod/terraform.tfstate"
    region  = "us-east-1"
    encrypt = true
    
    dynamodb_table = "pynomaly-terraform-lock"
  }
}
```

## ☁️ AWS Infrastructure

### Main Infrastructure

```hcl
# environments/prod/main.tf
module "network" {
  source = "../../modules/network"
  
  environment = var.environment
  vpc_cidr    = var.vpc_cidr
  
  availability_zones = var.availability_zones
  
  tags = var.common_tags
}

module "compute" {
  source = "../../modules/compute"
  
  environment = var.environment
  vpc_id      = module.network.vpc_id
  subnet_ids  = module.network.private_subnet_ids
  
  cluster_name     = var.cluster_name
  desired_capacity = var.desired_capacity
  max_size         = var.max_size
  min_size         = var.min_size
  
  instance_types = var.instance_types
  
  tags = var.common_tags
}

module "database" {
  source = "../../modules/database"
  
  environment = var.environment
  vpc_id      = module.network.vpc_id
  subnet_ids  = module.network.database_subnet_ids
  
  db_name         = var.db_name
  db_username     = var.db_username
  db_password     = var.db_password
  db_instance_class = var.db_instance_class
  
  backup_retention_period = var.backup_retention_period
  
  tags = var.common_tags
}

module "storage" {
  source = "../../modules/storage"
  
  environment = var.environment
  
  model_bucket_name = var.model_bucket_name
  backup_bucket_name = var.backup_bucket_name
  
  tags = var.common_tags
}

module "monitoring" {
  source = "../../modules/monitoring"
  
  environment = var.environment
  cluster_name = var.cluster_name
  
  tags = var.common_tags
}
```

### Network Module

```hcl
# modules/network/main.tf
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = merge(var.tags, {
    Name = "${var.environment}-pynomaly-vpc"
  })
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  
  tags = merge(var.tags, {
    Name = "${var.environment}-pynomaly-igw"
  })
}

resource "aws_subnet" "public" {
  count = length(var.availability_zones)
  
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true
  
  tags = merge(var.tags, {
    Name = "${var.environment}-pynomaly-public-${count.index + 1}"
    Type = "public"
  })
}

resource "aws_subnet" "private" {
  count = length(var.availability_zones)
  
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone = var.availability_zones[count.index]
  
  tags = merge(var.tags, {
    Name = "${var.environment}-pynomaly-private-${count.index + 1}"
    Type = "private"
  })
}

resource "aws_subnet" "database" {
  count = length(var.availability_zones)
  
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 20)
  availability_zone = var.availability_zones[count.index]
  
  tags = merge(var.tags, {
    Name = "${var.environment}-pynomaly-database-${count.index + 1}"
    Type = "database"
  })
}

resource "aws_nat_gateway" "main" {
  count = length(var.availability_zones)
  
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id
  
  tags = merge(var.tags, {
    Name = "${var.environment}-pynomaly-nat-${count.index + 1}"
  })
}

resource "aws_eip" "nat" {
  count = length(var.availability_zones)
  
  domain = "vpc"
  
  tags = merge(var.tags, {
    Name = "${var.environment}-pynomaly-eip-${count.index + 1}"
  })
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  
  tags = merge(var.tags, {
    Name = "${var.environment}-pynomaly-public-rt"
  })
}

resource "aws_route_table" "private" {
  count = length(var.availability_zones)
  
  vpc_id = aws_vpc.main.id
  
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[count.index].id
  }
  
  tags = merge(var.tags, {
    Name = "${var.environment}-pynomaly-private-rt-${count.index + 1}"
  })
}

resource "aws_route_table_association" "public" {
  count = length(aws_subnet.public)
  
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count = length(aws_subnet.private)
  
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}
```

### Compute Module (ECS)

```hcl
# modules/compute/main.tf
resource "aws_ecs_cluster" "main" {
  name = var.cluster_name
  
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]
  
  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 1
    base              = 1
  }
  
  default_capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    weight            = 4
  }
  
  tags = var.tags
}

resource "aws_ecs_task_definition" "pynomaly" {
  family                   = "pynomaly"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "1024"
  memory                   = "2048"
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn
  
  container_definitions = jsonencode([
    {
      name  = "pynomaly"
      image = "${var.ecr_repository_url}:latest"
      
      essential = true
      
      portMappings = [
        {
          containerPort = 8000
          protocol      = "tcp"
        }
      ]
      
      environment = [
        {
          name  = "ENVIRONMENT"
          value = var.environment
        },
        {
          name  = "DATABASE_URL"
          value = "postgresql://${var.db_username}:${var.db_password}@${var.db_endpoint}:5432/${var.db_name}"
        },
        {
          name  = "REDIS_URL"
          value = "redis://${var.redis_endpoint}:6379"
        }
      ]
      
      healthCheck = {
        command = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
        interval = 30
        timeout = 5
        retries = 3
        startPeriod = 60
      }
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.pynomaly.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
    }
  ])
  
  tags = var.tags
}

resource "aws_ecs_service" "pynomaly" {
  name            = "pynomaly"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.pynomaly.arn
  desired_count   = var.desired_capacity
  
  capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 1
    base              = 1
  }
  
  capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    weight            = 4
  }
  
  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }
  
  load_balancer {
    target_group_arn = aws_lb_target_group.pynomaly.arn
    container_name   = "pynomaly"
    container_port   = 8000
  }
  
  deployment_configuration {
    maximum_percent         = 200
    minimum_healthy_percent = 100
    
    deployment_circuit_breaker {
      enable   = true
      rollback = true
    }
  }
  
  tags = var.tags
}

resource "aws_lb" "main" {
  name               = "${var.environment}-pynomaly-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = var.public_subnet_ids
  
  enable_deletion_protection = true
  
  tags = var.tags
}

resource "aws_lb_target_group" "pynomaly" {
  name     = "${var.environment}-pynomaly-tg"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = var.vpc_id
  
  target_type = "ip"
  
  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    path                = "/health"
    matcher             = "200"
    protocol            = "HTTP"
  }
  
  tags = var.tags
}

resource "aws_lb_listener" "pynomaly" {
  load_balancer_arn = aws_lb.main.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS-1-2-2017-01"
  certificate_arn   = var.ssl_certificate_arn
  
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.pynomaly.arn
  }
}

resource "aws_cloudwatch_log_group" "pynomaly" {
  name              = "/ecs/pynomaly"
  retention_in_days = 7
  
  tags = var.tags
}
```

### Database Module

```hcl
# modules/database/main.tf
resource "aws_db_subnet_group" "main" {
  name       = "${var.environment}-pynomaly-db-subnet-group"
  subnet_ids = var.subnet_ids
  
  tags = merge(var.tags, {
    Name = "${var.environment}-pynomaly-db-subnet-group"
  })
}

resource "aws_db_instance" "main" {
  identifier = "${var.environment}-pynomaly-db"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = var.db_instance_class
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type          = "gp3"
  storage_encrypted     = true
  
  db_name  = var.db_name
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.database.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = var.backup_retention_period
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  performance_insights_enabled = true
  monitoring_interval         = 60
  monitoring_role_arn        = aws_iam_role.rds_monitoring.arn
  
  deletion_protection = true
  skip_final_snapshot = false
  final_snapshot_identifier = "${var.environment}-pynomaly-db-final-snapshot"
  
  tags = var.tags
}

resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.environment}-pynomaly-redis-subnet-group"
  subnet_ids = var.subnet_ids
  
  tags = var.tags
}

resource "aws_elasticache_replication_group" "main" {
  replication_group_id       = "${var.environment}-pynomaly-redis"
  description                = "Redis cluster for Pynomaly"
  
  node_type            = "cache.t3.micro"
  port                 = 6379
  parameter_group_name = "default.redis7"
  
  num_cache_clusters = 2
  
  subnet_group_name  = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = var.tags
}

resource "aws_security_group" "database" {
  name        = "${var.environment}-pynomaly-db-sg"
  description = "Security group for Pynomaly database"
  vpc_id      = var.vpc_id
  
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(var.tags, {
    Name = "${var.environment}-pynomaly-db-sg"
  })
}

resource "aws_security_group" "redis" {
  name        = "${var.environment}-pynomaly-redis-sg"
  description = "Security group for Pynomaly Redis"
  vpc_id      = var.vpc_id
  
  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(var.tags, {
    Name = "${var.environment}-pynomaly-redis-sg"
  })
}
```

### Variables

```hcl
# environments/prod/variables.tf
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "prod"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "cluster_name" {
  description = "ECS cluster name"
  type        = string
  default     = "pynomaly-prod"
}

variable "desired_capacity" {
  description = "Desired number of tasks"
  type        = number
  default     = 3
}

variable "max_size" {
  description = "Maximum number of tasks"
  type        = number
  default     = 10
}

variable "min_size" {
  description = "Minimum number of tasks"
  type        = number
  default     = 1
}

variable "instance_types" {
  description = "EC2 instance types"
  type        = list(string)
  default     = ["t3.medium", "t3.large"]
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "pynomaly"
}

variable "db_username" {
  description = "Database username"
  type        = string
  default     = "pynomaly"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "db_instance_class" {
  description = "Database instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "backup_retention_period" {
  description = "Database backup retention period"
  type        = number
  default     = 7
}

variable "common_tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    Project   = "Pynomaly"
    ManagedBy = "Terraform"
  }
}
```

## 🚀 Deployment Scripts

### Deploy Script

```bash
#!/bin/bash
# scripts/deploy.sh

set -e

ENVIRONMENT=${1:-prod}
ACTION=${2:-apply}

echo "🚀 Deploying Pynomaly infrastructure to $ENVIRONMENT"

# Change to environment directory
cd "environments/$ENVIRONMENT"

# Initialize Terraform
echo "📦 Initializing Terraform..."
terraform init

# Plan deployment
echo "📋 Planning deployment..."
terraform plan -var-file="terraform.tfvars" -out="tfplan"

# Apply if requested
if [ "$ACTION" == "apply" ]; then
    echo "🔄 Applying changes..."
    terraform apply "tfplan"
    
    echo "✅ Deployment completed successfully!"
    
    # Output important information
    echo "📄 Important outputs:"
    terraform output
else
    echo "📋 Plan completed. Run with 'apply' to deploy."
fi
```

### Validation Script

```bash
#!/bin/bash
# scripts/validate.sh

set -e

ENVIRONMENT=${1:-prod}

echo "🔍 Validating Pynomaly infrastructure for $ENVIRONMENT"

cd "environments/$ENVIRONMENT"

# Validate Terraform configuration
echo "📋 Validating Terraform configuration..."
terraform validate

# Check formatting
echo "📄 Checking Terraform formatting..."
terraform fmt -check=true -recursive

# Security scan with tfsec
echo "🔒 Running security scan..."
tfsec .

# Cost estimation with Infracost
echo "💰 Estimating costs..."
infracost breakdown --path .

echo "✅ Validation completed successfully!"
```

## 🔒 Security Best Practices

### Security Group Rules

```hcl
# Restrictive security groups
resource "aws_security_group" "ecs_tasks" {
  name        = "${var.environment}-pynomaly-ecs-tasks"
  description = "Security group for ECS tasks"
  vpc_id      = var.vpc_id
  
  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  
  egress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  egress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  tags = merge(var.tags, {
    Name = "${var.environment}-pynomaly-ecs-tasks"
  })
}
```

### IAM Roles

```hcl
# ECS Task Role
resource "aws_iam_role" "ecs_task_role" {
  name = "${var.environment}-pynomaly-ecs-task-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
  
  tags = var.tags
}

resource "aws_iam_role_policy" "ecs_task_policy" {
  name = "${var.environment}-pynomaly-ecs-task-policy"
  role = aws_iam_role.ecs_task_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "${aws_s3_bucket.models.arn}/*",
          "${aws_s3_bucket.backups.arn}/*"
        ]
      }
    ]
  })
}
```

## 📊 Monitoring Infrastructure

### CloudWatch Dashboards

```hcl
# modules/monitoring/main.tf
resource "aws_cloudwatch_dashboard" "pynomaly" {
  dashboard_name = "${var.environment}-pynomaly-dashboard"
  
  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        
        properties = {
          metrics = [
            ["AWS/ECS", "CPUUtilization", "ServiceName", "pynomaly", "ClusterName", var.cluster_name],
            [".", "MemoryUtilization", ".", ".", ".", "."]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "ECS Service Metrics"
          period  = 300
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6
        
        properties = {
          metrics = [
            ["AWS/RDS", "DatabaseConnections", "DBInstanceIdentifier", "${var.environment}-pynomaly-db"],
            [".", "ReadLatency", ".", "."],
            [".", "WriteLatency", ".", "."]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "Database Metrics"
          period  = 300
        }
      }
    ]
  })
  
  tags = var.tags
}
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/terraform.yml
name: Terraform Infrastructure

on:
  push:
    branches: [main]
    paths: ['terraform/**']
  pull_request:
    branches: [main]
    paths: ['terraform/**']

jobs:
  terraform:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout
      uses: actions/checkout@v3
    
    - name: Setup Terraform
      uses: hashicorp/setup-terraform@v2
      with:
        terraform_version: 1.0.0
    
    - name: Configure AWS Credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Terraform Init
      run: |
        cd terraform/environments/prod
        terraform init
    
    - name: Terraform Plan
      run: |
        cd terraform/environments/prod
        terraform plan -var-file="terraform.tfvars"
    
    - name: Terraform Apply
      if: github.ref == 'refs/heads/main'
      run: |
        cd terraform/environments/prod
        terraform apply -auto-approve -var-file="terraform.tfvars"
```

## 📚 Best Practices

### State Management

- Use remote backend (S3 + DynamoDB)
- Enable state locking
- Use workspaces for environments
- Regular state backups

### Code Organization

- Use modules for reusability
- Environment-specific configurations
- Version control everything
- Document infrastructure changes

### Security

- Use IAM roles, not access keys
- Enable encryption at rest and in transit
- Regular security audits
- Least privilege principle

### Cost Optimization

- Use Spot instances where appropriate
- Implement auto-scaling
- Regular cost reviews
- Resource tagging for cost allocation

This infrastructure as code guide provides a complete foundation for deploying Pynomaly in production environments with security, scalability, and best practices built-in.