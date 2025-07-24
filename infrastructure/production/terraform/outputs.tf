# Outputs for production infrastructure

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

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "cluster_version" {
  description = "EKS cluster Kubernetes version"
  value       = module.eks.cluster_version
}

# Node Group Outputs
output "node_groups" {
  description = "EKS node groups information"
  value = {
    cpu_optimized = {
      arn          = module.eks.eks_managed_node_groups["cpu_optimized"].node_group_arn
      status       = module.eks.eks_managed_node_groups["cpu_optimized"].node_group_status
      capacity_type = module.eks.eks_managed_node_groups["cpu_optimized"].capacity_type
    }
    memory_optimized = {
      arn          = module.eks.eks_managed_node_groups["memory_optimized"].node_group_arn
      status       = module.eks.eks_managed_node_groups["memory_optimized"].node_group_status
      capacity_type = module.eks.eks_managed_node_groups["memory_optimized"].capacity_type
    }
    gpu_enabled = {
      arn          = module.eks.eks_managed_node_groups["gpu_enabled"].node_group_arn
      status       = module.eks.eks_managed_node_groups["gpu_enabled"].node_group_status
      capacity_type = module.eks.eks_managed_node_groups["gpu_enabled"].capacity_type
    }
  }
}

# Database Outputs
output "postgres_endpoint" {
  description = "RDS PostgreSQL endpoint"
  value       = aws_db_instance.postgres.endpoint
  sensitive   = false
}

output "postgres_port" {
  description = "RDS PostgreSQL port"
  value       = aws_db_instance.postgres.port
}

output "postgres_database_name" {
  description = "PostgreSQL database name"
  value       = aws_db_instance.postgres.db_name
}

output "postgres_username" {
  description = "PostgreSQL master username"
  value       = aws_db_instance.postgres.username
  sensitive   = true
}

output "postgres_identifier" {
  description = "PostgreSQL instance identifier"
  value       = aws_db_instance.postgres.identifier
}

# Redis Outputs
output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = aws_elasticache_replication_group.redis.primary_endpoint_address
}

output "redis_port" {
  description = "ElastiCache Redis port"
  value       = aws_elasticache_replication_group.redis.port
}

output "redis_cluster_id" {
  description = "ElastiCache Redis cluster ID"
  value       = aws_elasticache_replication_group.redis.replication_group_id
}

# S3 Buckets
output "s3_buckets" {
  description = "S3 bucket information"
  value = {
    data_bucket = {
      id     = aws_s3_bucket.mlops_data.id
      arn    = aws_s3_bucket.mlops_data.arn
      domain_name = aws_s3_bucket.mlops_data.bucket_domain_name
    }
    models_bucket = {
      id     = aws_s3_bucket.mlops_models.id
      arn    = aws_s3_bucket.mlops_models.arn
      domain_name = aws_s3_bucket.mlops_models.bucket_domain_name
    }
    artifacts_bucket = {
      id     = aws_s3_bucket.mlops_artifacts.id
      arn    = aws_s3_bucket.mlops_artifacts.arn
      domain_name = aws_s3_bucket.mlops_artifacts.bucket_domain_name
    }
    alb_logs_bucket = {
      id     = aws_s3_bucket.alb_logs.id
      arn    = aws_s3_bucket.alb_logs.arn
      domain_name = aws_s3_bucket.alb_logs.bucket_domain_name
    }
  }
}

# Load Balancer Outputs
output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = aws_lb.mlops_alb.dns_name
}

output "alb_zone_id" {
  description = "Zone ID of the Application Load Balancer"
  value       = aws_lb.mlops_alb.zone_id
}

output "alb_arn" {
  description = "ARN of the Application Load Balancer"
  value       = aws_lb.mlops_alb.arn
}

# Security Outputs
output "security_groups" {
  description = "Security group information"
  value = {
    cluster_sg = {
      id   = aws_security_group.cluster_sg.id
      arn  = aws_security_group.cluster_sg.arn
      name = aws_security_group.cluster_sg.name
    }
    node_sg = {
      id   = aws_security_group.node_sg.id
      arn  = aws_security_group.node_sg.arn
      name = aws_security_group.node_sg.name
    }
    postgres_sg = {
      id   = aws_security_group.postgres.id
      arn  = aws_security_group.postgres.arn
      name = aws_security_group.postgres.name
    }
    redis_sg = {
      id   = aws_security_group.redis.id
      arn  = aws_security_group.redis.arn
      name = aws_security_group.redis.name
    }
    alb_sg = {
      id   = aws_security_group.alb.id
      arn  = aws_security_group.alb.arn
      name = aws_security_group.alb.name
    }
  }
}

# WAF Outputs
output "waf_web_acl_id" {
  description = "WAF Web ACL ID"
  value       = aws_wafv2_web_acl.mlops_waf.id
}

output "waf_web_acl_arn" {
  description = "WAF Web ACL ARN"
  value       = aws_wafv2_web_acl.mlops_waf.arn
}

# CloudWatch Log Groups
output "cloudwatch_log_groups" {
  description = "CloudWatch log group information"
  value = {
    platform_logs = {
      name = aws_cloudwatch_log_group.mlops_platform.name
      arn  = aws_cloudwatch_log_group.mlops_platform.arn
    }
    inference_logs = {
      name = aws_cloudwatch_log_group.model_inference.name
      arn  = aws_cloudwatch_log_group.model_inference.arn
    }
    training_logs = {
      name = aws_cloudwatch_log_group.model_training.name
      arn  = aws_cloudwatch_log_group.model_training.arn
    }
  }
}

# IAM Roles
output "iam_roles" {
  description = "IAM role information"
  value = {
    eks_admin_role = {
      name = aws_iam_role.eks_admin.name
      arn  = aws_iam_role.eks_admin.arn
    }
    rds_monitoring_role = {
      name = aws_iam_role.rds_monitoring.name
      arn  = aws_iam_role.rds_monitoring.arn
    }
  }
}

# Kubernetes Configuration
output "kubectl_config" {
  description = "kubectl configuration command"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}

# Connection Strings (for application configuration)
output "database_url" {
  description = "PostgreSQL connection URL template"
  value       = "postgresql://${aws_db_instance.postgres.username}:PASSWORD@${aws_db_instance.postgres.endpoint}/${aws_db_instance.postgres.db_name}"
  sensitive   = true
}

output "redis_url" {
  description = "Redis connection URL"
  value       = "redis://${aws_elasticache_replication_group.redis.primary_endpoint_address}:${aws_elasticache_replication_group.redis.port}"
}

# Environment Information
output "environment_summary" {
  description = "Summary of the deployed environment"
  value = {
    environment     = var.environment
    region         = var.aws_region
    cluster_name   = module.eks.cluster_name
    vpc_cidr       = module.vpc.vpc_cidr_block
    availability_zones = data.aws_availability_zones.available.names
    deployment_timestamp = timestamp()
  }
}

# Cost Optimization Information
output "cost_optimization_info" {
  description = "Information for cost optimization"
  value = {
    spot_instances_enabled = var.enable_spot_instances
    spot_percentage       = var.spot_instance_percentage
    multi_az_enabled     = var.multi_az_enabled
    backup_retention_days = var.postgres_backup_retention_period
  }
}

# Monitoring Endpoints
output "monitoring_endpoints" {
  description = "Monitoring and observability endpoints"
  value = {
    prometheus_endpoint = "http://${aws_lb.mlops_alb.dns_name}/prometheus"
    grafana_endpoint   = "http://${aws_lb.mlops_alb.dns_name}/grafana"
    kibana_endpoint    = "http://${aws_lb.mlops_alb.dns_name}/kibana"
    api_endpoint       = "http://${aws_lb.mlops_alb.dns_name}/api"
  }
}

# Backup and Recovery Information
output "backup_configuration" {
  description = "Backup and recovery configuration"
  value = {
    postgres_backup_window = aws_db_instance.postgres.backup_window
    postgres_backup_retention = aws_db_instance.postgres.backup_retention_period
    redis_snapshot_window = aws_elasticache_replication_group.redis.snapshot_window
    redis_snapshot_retention = aws_elasticache_replication_group.redis.snapshot_retention_limit
    s3_versioning_enabled = true
  }
}

# Security Configuration Summary
output "security_summary" {
  description = "Security configuration summary"
  value = {
    waf_enabled = var.enable_waf
    encryption_at_rest = var.enable_encryption_at_rest
    encryption_in_transit = var.enable_encryption_in_transit
    vpc_flow_logs = var.enable_flow_logs
    enhanced_monitoring = var.enable_enhanced_monitoring
  }
}