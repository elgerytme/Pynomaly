# Regional Cluster Module for Global MLOps Platform
# This module creates a complete regional deployment

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }
}

# Local variables
locals {
  cluster_name = var.cluster_name
  common_tags = merge(var.tags, {
    Region = var.region
    Role   = var.region_role
  })
}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"
  
  name = "${var.cluster_name}-vpc"
  cidr = var.vpc_cidr
  
  azs             = var.availability_zones
  private_subnets = [for i, az in var.availability_zones : cidrsubnet(var.vpc_cidr, 8, i)]
  public_subnets  = [for i, az in var.availability_zones : cidrsubnet(var.vpc_cidr, 8, i + 100)]
  
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
  
  # CloudWatch logging
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  # Encryption
  cluster_encryption_config = [
    {
      provider_key_arn = aws_kms_key.eks.arn
      resources        = ["secrets"]
    }
  ]
  
  # EKS Managed Node Groups
  eks_managed_node_groups = {
    for name, config in var.node_groups : name => {
      name           = name
      instance_types = config.instance_types
      
      min_size     = config.min_size
      max_size     = config.max_size
      desired_size = config.desired_size
      
      ami_type       = contains(config.instance_types, "p3") || contains(config.instance_types, "p4") ? "AL2_x86_64_GPU" : "AL2_x86_64"
      capacity_type  = var.cost_optimization.spot_instances_enabled && name != "gpu_enabled" ? "SPOT" : "ON_DEMAND"
      disk_size      = name == "gpu_enabled" ? 500 : (name == "memory_optimized" ? 200 : 100)
      
      vpc_security_group_ids = [aws_security_group.node_sg.id]
      
      k8s_labels = merge({
        Environment = var.environment
        NodeType    = name
        Region      = var.region
      }, name == "gpu_enabled" ? { "nvidia.com/gpu" = "true" } : {})
      
      taints = name == "gpu_enabled" ? {
        gpu-enabled = {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      } : {
        "${name}" = {
          key    = "node-type"
          value  = name
          effect = "NO_SCHEDULE"
        }
      }
      
      # Spot instance configuration
      dynamic "remote_access" {
        for_each = var.cost_optimization.spot_instances_enabled ? [1] : []
        content {
          ec2_ssh_key = var.ssh_key_name
        }
      }
      
      # Auto Scaling Group tags
      tags = merge(local.common_tags, {
        "k8s.io/cluster-autoscaler/enabled"                = "true"
        "k8s.io/cluster-autoscaler/${local.cluster_name}" = "owned"
      })
    }
  }
  
  tags = local.common_tags
}

# KMS Key for EKS encryption
resource "aws_kms_key" "eks" {
  description             = "EKS Secret Encryption Key for ${local.cluster_name}"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-eks-encryption-key"
  })
}

resource "aws_kms_alias" "eks" {
  name          = "alias/${local.cluster_name}-eks"
  target_key_id = aws_kms_key.eks.key_id
}

# Security Groups
resource "aws_security_group" "node_sg" {
  name_prefix = "${local.cluster_name}-node-"
  description = "Security group for EKS worker nodes"
  vpc_id      = module.vpc.vpc_id
  
  # Allow nodes to communicate with each other
  ingress {
    description = "Node to node communication"
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    self        = true
  }
  
  # Allow pods to communicate with the cluster API Server
  ingress {
    description = "Cluster API Server"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  # Allow kubelet and pods to receive communication from the cluster control plane
  ingress {
    description = "Cluster control plane"
    from_port   = 1025
    to_port     = 65535
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  # Regional cross-cluster communication
  dynamic "ingress" {
    for_each = var.region_role != "disaster_recovery" ? [1] : []
    content {
      description = "Cross-region cluster communication"
      from_port   = 443
      to_port     = 443
      protocol    = "tcp"
      cidr_blocks = var.cross_region_cidrs
    }
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

# Application Load Balancer
resource "aws_security_group" "alb" {
  name_prefix = "${local.cluster_name}-alb-"
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
    Name = "${local.cluster_name}-alb-sg"
  })
}

# Regional Database (Aurora PostgreSQL)
resource "aws_rds_cluster" "regional" {
  count = var.region_role == "primary" ? 1 : 0
  
  cluster_identifier      = "${local.cluster_name}-db-cluster"
  engine                 = "aurora-postgresql"
  engine_version         = "15.4"
  database_name          = "mlops_${replace(var.region, "-", "_")}"
  master_username        = "mlops_admin"
  master_password        = var.database_password
  
  vpc_security_group_ids = [aws_security_group.database.id]
  db_subnet_group_name   = aws_db_subnet_group.regional.name
  
  backup_retention_period = var.disaster_recovery.backup_retention_days
  preferred_backup_window = "03:00-04:00"
  
  # Global cluster configuration
  global_cluster_identifier = var.global_cluster_identifier
  
  # Encryption
  storage_encrypted = true
  kms_key_id       = aws_kms_key.rds.arn
  
  # Enhanced monitoring
  enabled_cloudwatch_logs_exports = ["postgresql"]
  
  # Deletion protection for production
  deletion_protection = var.environment == "production"
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-db-cluster"
  })
}

resource "aws_rds_cluster_instance" "regional" {
  count = var.region_role == "primary" ? 2 : (var.region_role == "secondary" ? 2 : 1)
  
  identifier         = "${local.cluster_name}-db-${count.index}"
  cluster_identifier = var.region_role == "primary" ? aws_rds_cluster.regional[0].id : var.global_cluster_identifier
  instance_class     = var.database_config.instance_class
  engine             = "aurora-postgresql"
  
  # Performance Insights
  performance_insights_enabled = true
  performance_insights_retention_period = 7
  
  # Enhanced monitoring
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_monitoring.arn
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-db-instance-${count.index}"
  })
}

# Database security group
resource "aws_security_group" "database" {
  name_prefix = "${local.cluster_name}-db-"
  description = "Security group for RDS cluster"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    description = "PostgreSQL from EKS"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  # Cross-region database access for replication
  dynamic "ingress" {
    for_each = var.region_role != "disaster_recovery" ? [1] : []
    content {
      description = "Cross-region database access"
      from_port   = 5432
      to_port     = 5432
      protocol    = "tcp"
      cidr_blocks = var.cross_region_cidrs
    }
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-db-sg"
  })
}

# Database subnet group
resource "aws_db_subnet_group" "regional" {
  name       = "${local.cluster_name}-db-subnet-group"
  subnet_ids = module.vpc.private_subnets
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-db-subnet-group"
  })
}

# KMS key for RDS encryption
resource "aws_kms_key" "rds" {
  description             = "RDS encryption key for ${local.cluster_name}"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-rds-encryption-key"
  })
}

# IAM role for RDS monitoring
resource "aws_iam_role" "rds_monitoring" {
  name = "${local.cluster_name}-rds-monitoring-role"
  
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

# Redis cluster
resource "aws_elasticache_replication_group" "regional" {
  replication_group_id       = "${local.cluster_name}-redis"
  description                = "Redis cluster for ${local.cluster_name}"
  
  node_type                  = var.redis_config.node_type
  port                       = 6379
  parameter_group_name       = "default.redis7"
  
  num_cache_clusters         = var.redis_config.num_cache_nodes
  automatic_failover_enabled = var.redis_config.num_cache_nodes > 1
  multi_az_enabled          = var.redis_config.num_cache_nodes > 1
  
  subnet_group_name = aws_elasticache_subnet_group.regional.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  # Backup configuration
  snapshot_retention_limit = var.disaster_recovery.backup_retention_days
  snapshot_window         = "03:00-05:00"
  
  # Cross-region replication for primary
  dynamic "global_replication_group_id" {
    for_each = var.region_role != "primary" && var.global_redis_replication_group_id != null ? [1] : []
    content {
      global_replication_group_id = var.global_redis_replication_group_id
    }
  }
  
  tags = local.common_tags
}

# Redis security group
resource "aws_security_group" "redis" {
  name_prefix = "${local.cluster_name}-redis-"
  description = "Security group for Redis cluster"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    description = "Redis from EKS"
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  # Cross-region Redis access
  dynamic "ingress" {
    for_each = var.region_role != "disaster_recovery" ? [1] : []
    content {
      description = "Cross-region Redis access"
      from_port   = 6379
      to_port     = 6379
      protocol    = "tcp"
      cidr_blocks = var.cross_region_cidrs
    }
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-redis-sg"
  })
}

# Redis subnet group
resource "aws_elasticache_subnet_group" "regional" {
  name       = "${local.cluster_name}-redis-subnet-group"
  subnet_ids = module.vpc.private_subnets
  
  tags = local.common_tags
}

# Regional monitoring
resource "aws_cloudwatch_log_group" "regional" {
  name              = "/aws/mlops/${var.environment}/${var.region}"
  retention_in_days = var.monitoring_config.log_retention_days
  
  tags = local.common_tags
}

# Auto Scaling for the region
resource "aws_autoscaling_group" "regional_scaling" {
  count = var.global_auto_scaling.enabled ? 1 : 0
  
  name                = "${local.cluster_name}-regional-asg"
  vpc_zone_identifier = module.vpc.private_subnets
  target_group_arns   = []
  health_check_type   = "ELB"
  
  min_size         = var.global_auto_scaling.min_capacity
  max_size         = var.global_auto_scaling.max_capacity
  desired_capacity = var.global_auto_scaling.target_capacity
  
  # Predictive scaling
  dynamic "predictive_scaling_configuration" {
    for_each = var.global_auto_scaling.predictive_scaling_enabled ? [1] : []
    content {
      mode                         = "ForecastOnly"
      scheduling_buffer_time       = 600
      max_capacity_breach_behavior = "IncreaseMaxCapacity"
      max_capacity_buffer          = 10
    }
  }
  
  tag {
    key                 = "Name"
    value               = "${local.cluster_name}-regional-asg"
    propagate_at_launch = true
  }
  
  dynamic "tag" {
    for_each = local.common_tags
    content {
      key                 = tag.key
      value               = tag.value
      propagate_at_launch = true
    }
  }
}

# Regional metrics for global monitoring
resource "aws_cloudwatch_metric_alarm" "regional_health" {
  alarm_name          = "${local.cluster_name}-regional-health"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "HealthyHostCount"
  namespace           = "AWS/ApplicationELB"
  period              = "60"
  statistic           = "Average"
  threshold           = "1"
  alarm_description   = "This metric monitors regional cluster health"
  
  dimensions = {
    LoadBalancer = "app/${local.cluster_name}-alb/*"
  }
  
  alarm_actions = [aws_sns_topic.regional_alerts.arn]
  ok_actions    = [aws_sns_topic.regional_alerts.arn]
  
  tags = local.common_tags
}

# SNS topic for regional alerts
resource "aws_sns_topic" "regional_alerts" {
  name = "${local.cluster_name}-alerts"
  
  tags = local.common_tags
}