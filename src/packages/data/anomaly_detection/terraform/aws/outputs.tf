# Outputs for Pynomaly Detection AWS infrastructure

# EKS Cluster Outputs
output "cluster_id" {
  description = "EKS cluster ID"
  value       = module.eks.cluster_id
}

output "cluster_arn" {
  description = "EKS cluster ARN"
  value       = module.eks.cluster_arn
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_version" {
  description = "EKS cluster version"
  value       = module.eks.cluster_version
}

output "cluster_platform_version" {
  description = "EKS cluster platform version"
  value       = module.eks.cluster_platform_version
}

output "cluster_security_group_id" {
  description = "EKS cluster security group ID"
  value       = module.eks.cluster_security_group_id
}

output "cluster_iam_role_arn" {
  description = "EKS cluster IAM role ARN"
  value       = module.eks.cluster_iam_role_arn
}

output "cluster_certificate_authority_data" {
  description = "EKS cluster certificate authority data"
  value       = module.eks.cluster_certificate_authority_data
}

output "cluster_oidc_issuer_url" {
  description = "EKS cluster OIDC issuer URL"
  value       = module.eks.cluster_oidc_issuer_url
}

# Node Group Outputs
output "node_groups" {
  description = "EKS node groups"
  value       = module.eks.eks_managed_node_groups
}

output "node_security_group_id" {
  description = "EKS node security group ID"
  value       = module.eks.node_security_group_id
}

# VPC Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "vpc_arn" {
  description = "VPC ARN"
  value       = module.vpc.vpc_arn
}

output "vpc_cidr_block" {
  description = "VPC CIDR block"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnets" {
  description = "Private subnet IDs"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "Public subnet IDs"
  value       = module.vpc.public_subnets
}

output "nat_gateway_ids" {
  description = "NAT Gateway IDs"
  value       = module.vpc.natgw_ids
}

output "internet_gateway_id" {
  description = "Internet Gateway ID"
  value       = module.vpc.igw_id
}

# S3 Storage Outputs
output "s3_bucket_name" {
  description = "S3 bucket name"
  value       = aws_s3_bucket.pynomaly_storage.bucket
}

output "s3_bucket_arn" {
  description = "S3 bucket ARN"
  value       = aws_s3_bucket.pynomaly_storage.arn
}

output "s3_bucket_domain_name" {
  description = "S3 bucket domain name"
  value       = aws_s3_bucket.pynomaly_storage.bucket_domain_name
}

output "s3_bucket_hosted_zone_id" {
  description = "S3 bucket hosted zone ID"
  value       = aws_s3_bucket.pynomaly_storage.hosted_zone_id
}

# CloudWatch Outputs
output "cloudwatch_log_group_name" {
  description = "CloudWatch log group name"
  value       = aws_cloudwatch_log_group.pynomaly_logs.name
}

output "cloudwatch_log_group_arn" {
  description = "CloudWatch log group ARN"
  value       = aws_cloudwatch_log_group.pynomaly_logs.arn
}

output "cloudwatch_dashboard_url" {
  description = "CloudWatch dashboard URL"
  value       = "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${aws_cloudwatch_dashboard.pynomaly_dashboard.dashboard_name}"
}

# IAM Outputs
output "node_iam_role_arn" {
  description = "EKS node IAM role ARN"
  value       = aws_iam_role.pynomaly_node_role.arn
}

output "pynomaly_policy_arn" {
  description = "Pynomaly custom policy ARN"
  value       = aws_iam_policy.pynomaly_policy.arn
}

# Kubernetes Outputs
output "kubernetes_namespace" {
  description = "Kubernetes namespace"
  value       = kubernetes_namespace.pynomaly.metadata[0].name
}

output "kubernetes_service_account_name" {
  description = "Kubernetes service account name"
  value       = kubernetes_service_account.pynomaly.metadata[0].name
}

# Application URLs
output "application_urls" {
  description = "Application URLs"
  value = {
    api_endpoint = var.domain_name != "" ? "https://${var.domain_name}" : "Use kubectl port-forward to access the application"
    dashboard    = aws_cloudwatch_dashboard.pynomaly_dashboard.dashboard_name
    logs         = aws_cloudwatch_log_group.pynomaly_logs.name
  }
}

# Deployment Information
output "deployment_info" {
  description = "Deployment information"
  value = {
    cluster_name     = module.eks.cluster_name
    region          = var.aws_region
    environment     = var.environment
    namespace       = kubernetes_namespace.pynomaly.metadata[0].name
    storage_bucket  = aws_s3_bucket.pynomaly_storage.bucket
    log_group       = aws_cloudwatch_log_group.pynomaly_logs.name
  }
}

# Connection Information
output "connection_info" {
  description = "Connection information for kubectl"
  value = {
    aws_region      = var.aws_region
    cluster_name    = module.eks.cluster_name
    kubectl_command = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
  }
}

# Resource Tags
output "common_tags" {
  description = "Common tags applied to all resources"
  value = {
    Application = "Pynomaly-Detection"
    Environment = var.environment
    Project     = "Pynomaly"
    ManagedBy   = "Terraform"
  }
}

# Cost Information
output "cost_tags" {
  description = "Cost tracking tags"
  value = {
    CostCenter = var.cost_center
    Owner      = var.owner
    Project    = "Pynomaly-Detection"
  }
}

# Security Information
output "security_groups" {
  description = "Security group IDs"
  value = {
    cluster_security_group = module.eks.cluster_security_group_id
    node_security_group    = module.eks.node_security_group_id
  }
}

# Backup Information
output "backup_info" {
  description = "Backup configuration"
  value = {
    s3_bucket_versioning = var.enable_s3_versioning
    log_retention_days   = var.log_retention_days
    backup_enabled       = var.enable_backup
  }
}

# Scaling Information
output "scaling_info" {
  description = "Auto-scaling configuration"
  value = {
    min_nodes      = var.min_nodes
    max_nodes      = var.max_nodes
    desired_nodes  = var.desired_nodes
    instance_types = var.instance_types
  }
}

# Monitoring Information
output "monitoring_info" {
  description = "Monitoring configuration"
  value = {
    cloudwatch_namespace    = var.cloudwatch_namespace
    log_group              = aws_cloudwatch_log_group.pynomaly_logs.name
    dashboard_name         = aws_cloudwatch_dashboard.pynomaly_dashboard.dashboard_name
    detailed_monitoring    = var.enable_detailed_monitoring
  }
}

# Network Information
output "network_info" {
  description = "Network configuration"
  value = {
    vpc_id              = module.vpc.vpc_id
    vpc_cidr           = module.vpc.vpc_cidr_block
    private_subnets    = module.vpc.private_subnets
    public_subnets     = module.vpc.public_subnets
    availability_zones = module.vpc.azs
  }
}

# Random Suffix
output "random_suffix" {
  description = "Random suffix used for resource names"
  value       = random_string.suffix.result
}