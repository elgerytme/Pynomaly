# Production Infrastructure for MLOps Platform
# Terraform configuration for multi-cloud production deployment

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
  
  backend "s3" {
    bucket         = "mlops-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "mlops-terraform-locks"
  }
}

# AWS Provider Configuration
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment = "production"
      Project     = "MLOps-Platform"
      ManagedBy   = "Terraform"
      CostCenter  = "Engineering"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Local values for consistent naming
locals {
  cluster_name = "${var.environment}-mlops-cluster"
  common_tags = {
    Environment = var.environment
    Project     = "MLOps-Platform"
    CreatedBy   = "Terraform"
  }
}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"
  
  name = "${var.environment}-mlops-vpc"
  cidr = var.vpc_cidr
  
  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
  enable_dns_hostnames = true
  enable_dns_support = true
  
  # Flow logs for security monitoring
  enable_flow_log = true
  create_flow_log_cloudwatch_log_group = true
  create_flow_log_cloudwatch_iam_role = true
  flow_log_destination_type = "cloud-watch-logs"
  
  tags = local.common_tags
}

# Security Groups
resource "aws_security_group" "cluster_sg" {
  name_prefix = "${local.cluster_name}-cluster-"
  description = "Security group for EKS cluster"
  vpc_id      = module.vpc.vpc_id
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-cluster-sg"
  })
}

resource "aws_security_group" "node_sg" {
  name_prefix = "${local.cluster_name}-node-"
  description = "Security group for EKS worker nodes"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    description = "Allow pods to communicate with the cluster API Server"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  ingress {
    description = "Allow worker nodes to communicate with each other"
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    self        = true
  }
  
  ingress {
    description = "Allow worker Kubelets and pods to receive communication from the cluster control plane"
    from_port   = 1025
    to_port     = 65535
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-node-sg"
  })
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"
  
  cluster_name    = local.cluster_name
  cluster_version = var.kubernetes_version
  
  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true
  cluster_endpoint_public_access_cidrs = var.allowed_cidr_blocks
  
  vpc_id                   = module.vpc.vpc_id
  subnet_ids               = module.vpc.private_subnets
  control_plane_subnet_ids = module.vpc.private_subnets
  
  # Cluster security group
  cluster_security_group_id = aws_security_group.cluster_sg.id
  
  # CloudWatch logging
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  # EKS Managed Node Groups
  eks_managed_node_groups = {
    # CPU-optimized nodes for general workloads
    cpu_optimized = {
      name           = "cpu-optimized"
      instance_types = ["c5.2xlarge", "c5.4xlarge"]
      
      min_size     = 3
      max_size     = 20
      desired_size = 5
      
      ami_type       = "AL2_x86_64"
      capacity_type  = "ON_DEMAND"
      disk_size      = 100
      
      vpc_security_group_ids = [aws_security_group.node_sg.id]
      
      k8s_labels = {
        Environment = var.environment
        NodeType    = "cpu-optimized"
      }
      
      taints = {
        cpu-optimized = {
          key    = "node-type"
          value  = "cpu-optimized"
          effect = "NO_SCHEDULE"
        }
      }
    }
    
    # Memory-optimized nodes for feature store and caching
    memory_optimized = {
      name           = "memory-optimized"
      instance_types = ["m5.4xlarge", "m5.8xlarge"]
      
      min_size     = 2
      max_size     = 15
      desired_size = 3
      
      ami_type       = "AL2_x86_64"
      capacity_type  = "ON_DEMAND"
      disk_size      = 200
      
      vpc_security_group_ids = [aws_security_group.node_sg.id]
      
      k8s_labels = {
        Environment = var.environment
        NodeType    = "memory-optimized"
      }
      
      taints = {
        memory-optimized = {
          key    = "node-type"
          value  = "memory-optimized"
          effect = "NO_SCHEDULE"
        }
      }
    }
    
    # GPU nodes for ML training and inference
    gpu_enabled = {
      name           = "gpu-enabled"
      instance_types = ["p3.2xlarge", "p3.8xlarge"]
      
      min_size     = 0
      max_size     = 10
      desired_size = 2
      
      ami_type       = "AL2_x86_64_GPU"
      capacity_type  = "ON_DEMAND"
      disk_size      = 500
      
      vpc_security_group_ids = [aws_security_group.node_sg.id]
      
      k8s_labels = {
        Environment = var.environment
        NodeType    = "gpu-enabled"
        "nvidia.com/gpu" = "true"
      }
      
      taints = {
        gpu-enabled = {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }
    }
  }
  
  # aws-auth configmap
  manage_aws_auth_configmap = true
  
  aws_auth_roles = [
    {
      rolearn  = aws_iam_role.eks_admin.arn
      username = "eks-admin"
      groups   = ["system:masters"]
    },
  ]
  
  aws_auth_users = var.eks_admin_users
  
  tags = local.common_tags
}

# IAM Role for EKS administration
resource "aws_iam_role" "eks_admin" {
  name = "${local.cluster_name}-admin-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
      },
    ]
  })
  
  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "eks_admin_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_admin.name
}

# RDS Instance for PostgreSQL
resource "aws_db_subnet_group" "postgres" {
  name       = "${var.environment}-postgres-subnet-group"
  subnet_ids = module.vpc.private_subnets
  
  tags = merge(local.common_tags, {
    Name = "${var.environment}-postgres-subnet-group"
  })
}

resource "aws_security_group" "postgres" {
  name_prefix = "${var.environment}-postgres-"
  description = "Security group for PostgreSQL RDS instance"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    description = "PostgreSQL access from EKS cluster"
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
  
  tags = merge(local.common_tags, {
    Name = "${var.environment}-postgres-sg"
  })
}

resource "aws_db_instance" "postgres" {
  identifier = "${var.environment}-mlops-postgres"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = var.postgres_instance_class
  
  allocated_storage     = var.postgres_allocated_storage
  max_allocated_storage = var.postgres_max_allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = true
  
  db_name  = var.postgres_database_name
  username = var.postgres_username
  password = var.postgres_password
  
  vpc_security_group_ids = [aws_security_group.postgres.id]
  db_subnet_group_name   = aws_db_subnet_group.postgres.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  # Enhanced monitoring
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_monitoring.arn
  
  # Performance Insights
  performance_insights_enabled = true
  performance_insights_retention_period = 7
  
  skip_final_snapshot = false
  final_snapshot_identifier = "${var.environment}-mlops-postgres-final-snapshot"
  
  tags = merge(local.common_tags, {
    Name = "${var.environment}-mlops-postgres"
  })
}

# IAM role for RDS monitoring
resource "aws_iam_role" "rds_monitoring" {
  name = "${var.environment}-rds-monitoring-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      },
    ]
  })
  
  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "rds_monitoring" {
  role       = aws_iam_role.rds_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# Redis (ElastiCache) for caching
resource "aws_elasticache_subnet_group" "redis" {
  name       = "${var.environment}-redis-subnet-group"
  subnet_ids = module.vpc.private_subnets
  
  tags = local.common_tags
}

resource "aws_security_group" "redis" {
  name_prefix = "${var.environment}-redis-"
  description = "Security group for Redis ElastiCache cluster"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    description = "Redis access from EKS cluster"
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  tags = merge(local.common_tags, {
    Name = "${var.environment}-redis-sg"
  })
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "${var.environment}-mlops-redis"
  description                = "Redis cluster for MLOps platform caching"
  
  node_type                  = var.redis_node_type
  port                       = 6379
  parameter_group_name       = "default.redis7"
  
  num_cache_clusters         = var.redis_num_cache_nodes
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  subnet_group_name = aws_elasticache_subnet_group.redis.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  # Backup configuration
  snapshot_retention_limit = 5
  snapshot_window         = "03:00-05:00"
  
  tags = local.common_tags
}

# S3 Buckets for data storage
resource "aws_s3_bucket" "mlops_data" {
  bucket = "${var.environment}-mlops-data-${random_id.bucket_suffix.hex}"
  
  tags = merge(local.common_tags, {
    Name = "MLOps Data Bucket"
    Type = "data-storage"
  })
}

resource "aws_s3_bucket" "mlops_models" {
  bucket = "${var.environment}-mlops-models-${random_id.bucket_suffix.hex}"
  
  tags = merge(local.common_tags, {
    Name = "MLOps Models Bucket"
    Type = "model-storage"
  })
}

resource "aws_s3_bucket" "mlops_artifacts" {
  bucket = "${var.environment}-mlops-artifacts-${random_id.bucket_suffix.hex}"
  
  tags = merge(local.common_tags, {
    Name = "MLOps Artifacts Bucket"
    Type = "artifact-storage"
  })
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# S3 bucket configurations
resource "aws_s3_bucket_versioning" "mlops_data" {
  bucket = aws_s3_bucket.mlops_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "mlops_models" {
  bucket = aws_s3_bucket.mlops_models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "mlops_artifacts" {
  bucket = aws_s3_bucket.mlops_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 bucket encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "mlops_data" {
  bucket = aws_s3_bucket.mlops_data.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mlops_models" {
  bucket = aws_s3_bucket.mlops_models.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mlops_artifacts" {
  bucket = aws_s3_bucket.mlops_artifacts.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# S3 bucket public access blocking
resource "aws_s3_bucket_public_access_block" "mlops_data" {
  bucket = aws_s3_bucket.mlops_data.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "mlops_models" {
  bucket = aws_s3_bucket.mlops_models.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "mlops_artifacts" {
  bucket = aws_s3_bucket.mlops_artifacts.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Application Load Balancer
resource "aws_lb" "mlops_alb" {
  name               = "${var.environment}-mlops-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets           = module.vpc.public_subnets
  
  enable_deletion_protection = true
  
  access_logs {
    bucket  = aws_s3_bucket.alb_logs.bucket
    prefix  = "alb-logs"
    enabled = true
  }
  
  tags = merge(local.common_tags, {
    Name = "${var.environment}-mlops-alb"
  })
}

# ALB Security Group
resource "aws_security_group" "alb" {
  name_prefix = "${var.environment}-alb-"
  description = "Security group for Application Load Balancer"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    description = "HTTPS"
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
  
  tags = merge(local.common_tags, {
    Name = "${var.environment}-alb-sg"
  })
}

# S3 bucket for ALB access logs
resource "aws_s3_bucket" "alb_logs" {
  bucket = "${var.environment}-mlops-alb-logs-${random_id.bucket_suffix.hex}"
  
  tags = merge(local.common_tags, {
    Name = "ALB Access Logs"
    Type = "logs"
  })
}

resource "aws_s3_bucket_public_access_block" "alb_logs" {
  bucket = aws_s3_bucket.alb_logs.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_policy" "alb_logs" {
  bucket = aws_s3_bucket.alb_logs.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_elb_service_account.main.id}:root"
        }
        Action   = "s3:PutObject"
        Resource = "${aws_s3_bucket.alb_logs.arn}/alb-logs/AWSLogs/${data.aws_caller_identity.current.account_id}/*"
      },
      {
        Effect = "Allow"
        Principal = {
          Service = "delivery.logs.amazonaws.com"
        }
        Action   = "s3:PutObject"
        Resource = "${aws_s3_bucket.alb_logs.arn}/alb-logs/AWSLogs/${data.aws_caller_identity.current.account_id}/*"
        Condition = {
          StringEquals = {
            "s3:x-amz-acl" = "bucket-owner-full-control"
          }
        }
      },
      {
        Effect = "Allow"
        Principal = {
          Service = "delivery.logs.amazonaws.com"
        }
        Action   = "s3:GetBucketAcl"
        Resource = aws_s3_bucket.alb_logs.arn
      }
    ]
  })
}

data "aws_elb_service_account" "main" {}

# WAF for application protection
resource "aws_wafv2_web_acl" "mlops_waf" {
  name  = "${var.environment}-mlops-waf"
  description = "WAF for MLOps platform"
  scope = "REGIONAL"
  
  default_action {
    allow {}
  }
  
  # AWS Managed Rules
  rule {
    name     = "AWSManagedRulesCommonRuleSet"
    priority = 1
    
    override_action {
      none {}
    }
    
    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"
      }
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "CommonRuleSetMetric"
      sampled_requests_enabled   = true
    }
  }
  
  rule {
    name     = "AWSManagedRulesKnownBadInputsRuleSet"
    priority = 2
    
    override_action {
      none {}
    }
    
    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesKnownBadInputsRuleSet"
        vendor_name = "AWS"
      }
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "KnownBadInputsRuleSetMetric"
      sampled_requests_enabled   = true
    }
  }
  
  # Rate limiting rule
  rule {
    name     = "RateLimitRule"
    priority = 3
    
    action {
      block {}
    }
    
    statement {
      rate_based_statement {
        limit              = 1000
        aggregate_key_type = "IP"
      }
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "RateLimitRuleMetric"
      sampled_requests_enabled   = true
    }
  }
  
  tags = local.common_tags
  
  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "MLOpsWAFMetric"
    sampled_requests_enabled   = true
  }
}

# Associate WAF with ALB
resource "aws_wafv2_web_acl_association" "mlops_waf_alb" {
  resource_arn = aws_lb.mlops_alb.arn
  web_acl_arn  = aws_wafv2_web_acl.mlops_waf.arn
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "mlops_platform" {
  name              = "/aws/mlops/${var.environment}/platform"
  retention_in_days = 30
  
  tags = local.common_tags
}

resource "aws_cloudwatch_log_group" "model_inference" {
  name              = "/aws/mlops/${var.environment}/inference"
  retention_in_days = 14
  
  tags = local.common_tags
}

resource "aws_cloudwatch_log_group" "model_training" {
  name              = "/aws/mlops/${var.environment}/training"
  retention_in_days = 7
  
  tags = local.common_tags
}