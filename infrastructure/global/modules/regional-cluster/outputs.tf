# Outputs for Regional Cluster Module

# VPC Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = module.vpc.private_subnets
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = module.vpc.public_subnets
}

output "availability_zones" {
  description = "Availability zones used"
  value       = var.availability_zones
}

# EKS Cluster Outputs
output "cluster_id" {
  description = "EKS cluster ID"
  value       = module.eks.cluster_id
}

output "cluster_arn" {
  description = "EKS cluster ARN"
  value       = module.eks.cluster_arn
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "cluster_version" {
  description = "EKS cluster Kubernetes version"
  value       = module.eks.cluster_version
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "oidc_provider_arn" {
  description = "ARN of the OIDC Provider for EKS"
  value       = module.eks.oidc_provider_arn
}

# Node Group Outputs
output "node_groups" {
  description = "EKS node groups information"
  value = {
    for name, config in var.node_groups : name => {
      arn          = module.eks.eks_managed_node_groups[name].node_group_arn
      status       = module.eks.eks_managed_node_groups[name].node_group_status
      capacity_type = module.eks.eks_managed_node_groups[name].capacity_type
      instance_types = config.instance_types
      min_size     = config.min_size
      max_size     = config.max_size
      desired_size = config.desired_size
    }
  }
}

# Database Outputs
output "database_cluster_endpoint" {
  description = "Aurora cluster endpoint"
  value       = var.region_role == "primary" ? aws_rds_cluster.regional[0].endpoint : null
}

output "database_cluster_reader_endpoint" {
  description = "Aurora cluster reader endpoint"
  value       = var.region_role == "primary" ? aws_rds_cluster.regional[0].reader_endpoint : null
}

output "database_cluster_identifier" {
  description = "Aurora cluster identifier"
  value       = var.region_role == "primary" ? aws_rds_cluster.regional[0].cluster_identifier : null
}

output "database_port" {
  description = "Database port"
  value       = 5432
}

output "database_name" {
  description = "Database name"
  value       = var.region_role == "primary" ? aws_rds_cluster.regional[0].database_name : null
}

output "database_username" {
  description = "Database master username"
  value       = var.region_role == "primary" ? aws_rds_cluster.regional[0].master_username : null
  sensitive   = true
}

# Redis Outputs
output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = aws_elasticache_replication_group.regional.primary_endpoint_address
}

output "redis_port" {
  description = "Redis port"
  value       = aws_elasticache_replication_group.regional.port
}

output "redis_cluster_id" {
  description = "Redis cluster identifier"
  value       = aws_elasticache_replication_group.regional.replication_group_id
}

# Security Group Outputs
output "node_security_group_id" {
  description = "Security group ID for EKS nodes"
  value       = aws_security_group.node_sg.id
}

output "alb_security_group_id" {
  description = "Security group ID for ALB"
  value       = aws_security_group.alb.id
}

output "database_security_group_id" {
  description = "Security group ID for database"
  value       = aws_security_group.database.id
}

output "redis_security_group_id" {
  description = "Security group ID for Redis"
  value       = aws_security_group.redis.id
}

# KMS Outputs
output "eks_kms_key_id" {
  description = "KMS key ID for EKS encryption"
  value       = aws_kms_key.eks.key_id
}

output "eks_kms_key_arn" {
  description = "KMS key ARN for EKS encryption"
  value       = aws_kms_key.eks.arn
}

output "rds_kms_key_id" {
  description = "KMS key ID for RDS encryption"
  value       = aws_kms_key.rds.key_id
}

output "rds_kms_key_arn" {
  description = "KMS key ARN for RDS encryption"
  value       = aws_kms_key.rds.arn
}

# Monitoring Outputs
output "cloudwatch_log_group_name" {
  description = "CloudWatch log group name"
  value       = aws_cloudwatch_log_group.regional.name
}

output "cloudwatch_log_group_arn" {
  description = "CloudWatch log group ARN"
  value       = aws_cloudwatch_log_group.regional.arn
}

output "sns_topic_arn" {
  description = "SNS topic ARN for alerts"
  value       = aws_sns_topic.regional_alerts.arn
}

# Regional Configuration Summary
output "region_summary" {
  description = "Summary of regional deployment"
  value = {
    region           = var.region
    environment      = var.environment
    region_role      = var.region_role
    cluster_name     = local.cluster_name
    vpc_cidr         = var.vpc_cidr
    availability_zones = var.availability_zones
    node_groups      = var.node_groups
    database_config  = var.database_config
    redis_config     = var.redis_config
  }
}

# Auto Scaling Group Outputs
output "autoscaling_group_arn" {
  description = "Auto Scaling Group ARN"
  value       = var.global_auto_scaling.enabled ? aws_autoscaling_group.regional_scaling[0].arn : null
}

output "autoscaling_group_name" {
  description = "Auto Scaling Group name"
  value       = var.global_auto_scaling.enabled ? aws_autoscaling_group.regional_scaling[0].name : null
}

# Resource ARNs for Global Management
output "resource_arns" {
  description = "ARNs of key resources for global management"
  value = {
    vpc_arn              = "arn:aws:ec2:${var.region}:${data.aws_caller_identity.current.account_id}:vpc/${module.vpc.vpc_id}"
    cluster_arn          = module.eks.cluster_arn
    database_cluster_arn = var.region_role == "primary" ? aws_rds_cluster.regional[0].arn : null
    redis_cluster_arn    = aws_elasticache_replication_group.regional.arn
    log_group_arn        = aws_cloudwatch_log_group.regional.arn
    kms_key_arns = {
      eks = aws_kms_key.eks.arn
      rds = aws_kms_key.rds.arn
    }
  }
}

# Health Check Endpoints
output "health_endpoints" {
  description = "Health check endpoints for this region"
  value = {
    cluster_endpoint = module.eks.cluster_endpoint
    database_endpoint = var.region_role == "primary" ? aws_rds_cluster.regional[0].endpoint : null
    redis_endpoint   = aws_elasticache_replication_group.regional.primary_endpoint_address
  }
}

# Cost Information
output "cost_summary" {
  description = "Cost-related information for this region"
  value = {
    spot_instances_enabled     = var.cost_optimization.spot_instances_enabled
    reserved_instances_enabled = var.cost_optimization.reserved_instances_enabled
    savings_plans_enabled      = var.cost_optimization.savings_plans_enabled
    storage_optimization       = var.cost_optimization.storage_optimization
    predictive_scaling_enabled = var.global_auto_scaling.predictive_scaling_enabled
    total_node_groups         = length(var.node_groups)
    total_desired_capacity    = sum([for ng in var.node_groups : ng.desired_size])
    total_max_capacity        = sum([for ng in var.node_groups : ng.max_size])
  }
}

# Data source for current AWS account
data "aws_caller_identity" "current" {}