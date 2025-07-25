# =============================================================================
# TERRAFORM OUTPUTS
# Comprehensive output definitions for infrastructure resources
# =============================================================================

# =============================================================================
# NETWORKING OUTPUTS
# =============================================================================

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = aws_vpc.main.cidr_block
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}

output "database_subnet_ids" {
  description = "IDs of the database subnets"
  value       = aws_subnet.database[*].id
}

output "internet_gateway_id" {
  description = "ID of the Internet Gateway"
  value       = aws_internet_gateway.main.id
}

output "nat_gateway_ids" {
  description = "IDs of the NAT Gateways"
  value       = aws_nat_gateway.main[*].id
}

output "nat_gateway_ips" {
  description = "Elastic IP addresses of NAT Gateways"
  value       = aws_eip.nat[*].public_ip
}

# =============================================================================
# SECURITY OUTPUTS
# =============================================================================

output "security_group_web_id" {
  description = "ID of the web security group"
  value       = aws_security_group.web.id
}

output "security_group_app_id" {
  description = "ID of the application security group"
  value       = aws_security_group.app.id
}

output "security_group_database_id" {
  description = "ID of the database security group"
  value       = aws_security_group.database.id
}

output "security_group_bastion_id" {
  description = "ID of the bastion security group"
  value       = aws_security_group.bastion.id
}

output "security_group_eks_cluster_id" {
  description = "ID of the EKS cluster security group"
  value       = aws_security_group.eks_cluster.id
}

# =============================================================================
# EKS CLUSTER OUTPUTS
# =============================================================================

output "eks_cluster_id" {
  description = "ID of the EKS cluster"
  value       = aws_eks_cluster.main.id
}

output "eks_cluster_arn" {
  description = "ARN of the EKS cluster"
  value       = aws_eks_cluster.main.arn
}

output "eks_cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = aws_eks_cluster.main.endpoint
}

output "eks_cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = aws_eks_cluster.main.vpc_config[0].cluster_security_group_id
}

output "eks_cluster_iam_role_name" {
  description = "IAM role name associated with EKS cluster"
  value       = aws_iam_role.eks_service_role.name
}

output "eks_cluster_iam_role_arn" {
  description = "IAM role ARN associated with EKS cluster"
  value       = aws_iam_role.eks_service_role.arn
}

output "eks_cluster_version" {
  description = "Version of the EKS cluster"
  value       = aws_eks_cluster.main.version
}

output "eks_cluster_platform_version" {
  description = "Platform version for the EKS cluster"
  value       = aws_eks_cluster.main.platform_version
}

output "eks_cluster_certificate_authority" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = aws_eks_cluster.main.certificate_authority[0].data
}

output "eks_node_group_arn" {
  description = "ARN of the EKS node group"
  value       = aws_eks_node_group.main.arn
}

output "eks_node_group_status" {
  description = "Status of the EKS node group"
  value       = aws_eks_node_group.main.status
}

output "eks_node_group_capacity_type" {
  description = "Type of capacity associated with the EKS Node Group"
  value       = aws_eks_node_group.main.capacity_type
}

output "eks_node_group_instance_types" {
  description = "Instance types associated with the EKS Node Group"
  value       = aws_eks_node_group.main.instance_types
}

# =============================================================================
# DATABASE OUTPUTS
# =============================================================================

output "db_instance_address" {
  description = "RDS instance hostname"
  value       = aws_db_instance.postgresql.address
  sensitive   = true
}

output "db_instance_arn" {
  description = "RDS instance ARN"
  value       = aws_db_instance.postgresql.arn
}

output "db_instance_availability_zone" {
  description = "RDS instance availability zone"
  value       = aws_db_instance.postgresql.availability_zone
}

output "db_instance_backup_retention_period" {
  description = "RDS instance backup retention period"
  value       = aws_db_instance.postgresql.backup_retention_period
}

output "db_instance_backup_window" {
  description = "RDS instance backup window"
  value       = aws_db_instance.postgresql.backup_window
}

output "db_instance_ca_cert_identifier" {
  description = "RDS instance CA certificate identifier"
  value       = aws_db_instance.postgresql.ca_cert_identifier
}

output "db_instance_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.postgresql.endpoint
  sensitive   = true
}

output "db_instance_engine" {
  description = "RDS instance engine"
  value       = aws_db_instance.postgresql.engine
}

output "db_instance_engine_version" {
  description = "RDS instance engine version"
  value       = aws_db_instance.postgresql.engine_version
}

output "db_instance_id" {
  description = "RDS instance ID"
  value       = aws_db_instance.postgresql.id
}

output "db_instance_class" {
  description = "RDS instance class"
  value       = aws_db_instance.postgresql.instance_class
}

output "db_instance_name" {
  description = "RDS instance database name"
  value       = aws_db_instance.postgresql.db_name
}

output "db_instance_port" {
  description = "RDS instance port"
  value       = aws_db_instance.postgresql.port
}

output "db_instance_status" {
  description = "RDS instance status"
  value       = aws_db_instance.postgresql.status
}

output "db_instance_username" {
  description = "RDS instance root username"
  value       = aws_db_instance.postgresql.username
  sensitive   = true
}

output "db_subnet_group_id" {
  description = "DB subnet group name"
  value       = aws_db_subnet_group.main.id
}

output "db_subnet_group_arn" {
  description = "DB subnet group ARN"
  value       = aws_db_subnet_group.main.arn
}

# =============================================================================
# REDIS/ELASTICACHE OUTPUTS
# =============================================================================

output "redis_cluster_address" {
  description = "Address of the replication group configuration endpoint"
  value       = aws_elasticache_replication_group.redis.configuration_endpoint_address
  sensitive   = true
}

output "redis_cluster_id" {
  description = "ID of the ElastiCache replication group"
  value       = aws_elasticache_replication_group.redis.id
}

output "redis_cluster_port" {
  description = "Port number on which the configuration endpoint will accept connections"
  value       = aws_elasticache_replication_group.redis.port
}

output "redis_cluster_arn" {
  description = "ARN of the created ElastiCache replication group"
  value       = aws_elasticache_replication_group.redis.arn
}

output "redis_primary_endpoint_address" {
  description = "Address of the endpoint for the primary node in the replication group"
  value       = aws_elasticache_replication_group.redis.primary_endpoint_address
  sensitive   = true
}

output "redis_reader_endpoint_address" {
  description = "Address of the endpoint for the reader node in the replication group"
  value       = aws_elasticache_replication_group.redis.reader_endpoint_address
  sensitive   = true
}

output "redis_member_clusters" {
  description = "Identifiers of all the nodes that are part of this replication group"
  value       = aws_elasticache_replication_group.redis.member_clusters
}

# =============================================================================
# LOAD BALANCER OUTPUTS
# =============================================================================

output "alb_id" {
  description = "ID of the load balancer"
  value       = aws_lb.main.id
}

output "alb_arn" {
  description = "ARN of the load balancer"
  value       = aws_lb.main.arn
}

output "alb_arn_suffix" {
  description = "ARN suffix for use with CloudWatch Metrics"
  value       = aws_lb.main.arn_suffix
}

output "alb_dns_name" {
  description = "DNS name of the load balancer"
  value       = aws_lb.main.dns_name
}

output "alb_hosted_zone_id" {
  description = "Canonical hosted zone ID of the load balancer"
  value       = aws_lb.main.zone_id
}

# =============================================================================
# S3 OUTPUTS
# =============================================================================

output "s3_bucket_logs_id" {
  description = "Name of the logs bucket"
  value       = aws_s3_bucket.logs.id
}

output "s3_bucket_logs_arn" {
  description = "ARN of the logs bucket"
  value       = aws_s3_bucket.logs.arn
}

output "s3_bucket_logs_domain_name" {
  description = "Bucket domain name of the logs bucket"
  value       = aws_s3_bucket.logs.bucket_domain_name
}

output "s3_bucket_logs_regional_domain_name" {
  description = "Regional domain name of the logs bucket"
  value       = aws_s3_bucket.logs.bucket_regional_domain_name
}

output "s3_bucket_model_artifacts_id" {
  description = "Name of the model artifacts bucket"
  value       = aws_s3_bucket.model_artifacts.id
}

output "s3_bucket_model_artifacts_arn" {
  description = "ARN of the model artifacts bucket"
  value       = aws_s3_bucket.model_artifacts.arn
}

output "s3_bucket_model_artifacts_domain_name" {
  description = "Bucket domain name of the model artifacts bucket"
  value       = aws_s3_bucket.model_artifacts.bucket_domain_name
}

output "s3_bucket_model_artifacts_regional_domain_name" {
  description = "Regional domain name of the model artifacts bucket"
  value       = aws_s3_bucket.model_artifacts.bucket_regional_domain_name
}

# =============================================================================
# KMS OUTPUTS
# =============================================================================

output "kms_key_eks_id" {
  description = "The globally unique identifier for the EKS KMS key"
  value       = aws_kms_key.eks.key_id
}

output "kms_key_eks_arn" {
  description = "The Amazon Resource Name (ARN) of the EKS KMS key"
  value       = aws_kms_key.eks.arn
}

output "kms_key_rds_id" {
  description = "The globally unique identifier for the RDS KMS key"
  value       = aws_kms_key.rds.key_id
}

output "kms_key_rds_arn" {
  description = "The Amazon Resource Name (ARN) of the RDS KMS key"
  value       = aws_kms_key.rds.arn
}

output "kms_key_elasticache_id" {
  description = "The globally unique identifier for the ElastiCache KMS key"
  value       = aws_kms_key.elasticache.key_id
}

output "kms_key_elasticache_arn" {
  description = "The Amazon Resource Name (ARN) of the ElastiCache KMS key"
  value       = aws_kms_key.elasticache.arn
}

output "kms_key_s3_id" {
  description = "The globally unique identifier for the S3 KMS key"
  value       = aws_kms_key.s3.key_id
}

output "kms_key_s3_arn" {
  description = "The Amazon Resource Name (ARN) of the S3 KMS key"
  value       = aws_kms_key.s3.arn
}

output "kms_key_cloudwatch_id" {
  description = "The globally unique identifier for the CloudWatch KMS key"
  value       = aws_kms_key.cloudwatch.key_id
}

output "kms_key_cloudwatch_arn" {
  description = "The Amazon Resource Name (ARN) of the CloudWatch KMS key"
  value       = aws_kms_key.cloudwatch.arn
}

# =============================================================================
# SECRETS MANAGER OUTPUTS
# =============================================================================

output "secrets_manager_db_password_arn" {
  description = "ARN of the database password secret"
  value       = aws_secretsmanager_secret.ssh_private_key.arn
}

output "secrets_manager_ssh_private_key_arn" {
  description = "ARN of the SSH private key secret"
  value       = aws_secretsmanager_secret.ssh_private_key.arn
}

# =============================================================================
# EC2 KEY PAIR OUTPUTS
# =============================================================================

output "key_pair_name" {
  description = "Name of the key pair"
  value       = aws_key_pair.main.key_name
}

output "key_pair_fingerprint" {
  description = "SHA-1 digest of the DER encoded public key"
  value       = aws_key_pair.main.fingerprint
}

# =============================================================================
# CLOUDWATCH OUTPUTS
# =============================================================================

output "cloudwatch_log_group_eks_cluster_name" {
  description = "Name of the EKS cluster CloudWatch log group"
  value       = aws_cloudwatch_log_group.eks_cluster.name
}

output "cloudwatch_log_group_eks_cluster_arn" {
  description = "ARN of the EKS cluster CloudWatch log group"
  value       = aws_cloudwatch_log_group.eks_cluster.arn
}

output "cloudwatch_log_group_redis_slow_name" {
  description = "Name of the Redis slow log CloudWatch log group"
  value       = aws_cloudwatch_log_group.redis_slow.name
}

output "cloudwatch_log_group_redis_slow_arn" {
  description = "ARN of the Redis slow log CloudWatch log group"
  value       = aws_cloudwatch_log_group.redis_slow.arn
}

# =============================================================================
# IAM OUTPUTS
# =============================================================================

output "iam_role_eks_cluster_arn" {
  description = "ARN of the EKS cluster IAM role"
  value       = aws_iam_role.eks_service_role.arn
}

output "iam_role_eks_cluster_name" {
  description = "Name of the EKS cluster IAM role"
  value       = aws_iam_role.eks_service_role.name
}

output "iam_role_eks_node_group_arn" {
  description = "ARN of the EKS node group IAM role"
  value       = aws_iam_role.eks_node_group_role.arn
}

output "iam_role_eks_node_group_name" {
  description = "Name of the EKS node group IAM role"
  value       = aws_iam_role.eks_node_group_role.name
}

# =============================================================================
# AVAILABILITY ZONES OUTPUTS
# =============================================================================

output "availability_zones" {
  description = "List of availability zones used in the current AWS region"
  value       = data.aws_availability_zones.available.names
}

# =============================================================================
# REGION AND ACCOUNT OUTPUTS
# =============================================================================

output "aws_region" {
  description = "AWS region"
  value       = data.aws_region.current.name
}

output "aws_account_id" {
  description = "AWS account ID"
  value       = data.aws_caller_identity.current.account_id
}

# =============================================================================
# ENVIRONMENT INFORMATION
# =============================================================================

output "environment" {
  description = "Environment name"
  value       = local.environment
}

output "project_name" {
  description = "Project name"
  value       = local.project_name
}

output "name_prefix" {
  description = "Resource name prefix"
  value       = local.name_prefix
}

# =============================================================================
# CONFIGURATION OUTPUTS FOR KUBECTL
# =============================================================================

output "kubectl_config" {
  description = "kubectl config for EKS cluster access"
  value = {
    cluster_name     = aws_eks_cluster.main.name
    cluster_endpoint = aws_eks_cluster.main.endpoint
    cluster_ca       = aws_eks_cluster.main.certificate_authority[0].data
    aws_region      = data.aws_region.current.name
  }
}

# =============================================================================
# DATABASE CONNECTION INFORMATION
# =============================================================================

output "database_connection_info" {
  description = "Database connection information"
  value = {
    host     = aws_db_instance.postgresql.address
    port     = aws_db_instance.postgresql.port
    database = aws_db_instance.postgresql.db_name
    username = aws_db_instance.postgresql.username
  }
  sensitive = true
}

# =============================================================================
# REDIS CONNECTION INFORMATION
# =============================================================================

output "redis_connection_info" {
  description = "Redis connection information"
  value = {
    primary_endpoint = aws_elasticache_replication_group.redis.primary_endpoint_address
    reader_endpoint  = aws_elasticache_replication_group.redis.reader_endpoint_address
    port            = aws_elasticache_replication_group.redis.port
    auth_required   = true
  }
  sensitive = true
}

# =============================================================================
# DEPLOYMENT INFORMATION
# =============================================================================

output "deployment_info" {
  description = "Information needed for application deployment"
  value = {
    cluster_name         = aws_eks_cluster.main.name
    node_group_name      = aws_eks_node_group.main.node_group_name
    load_balancer_dns    = aws_lb.main.dns_name
    vpc_id              = aws_vpc.main.id
    private_subnet_ids  = aws_subnet.private[*].id
    public_subnet_ids   = aws_subnet.public[*].id
    security_group_app  = aws_security_group.app.id
    security_group_web  = aws_security_group.web.id
  }
}

# =============================================================================
# MONITORING AND LOGGING ENDPOINTS
# =============================================================================

output "monitoring_endpoints" {
  description = "Monitoring and logging endpoints"
  value = {
    cloudwatch_log_groups = {
      eks_cluster = aws_cloudwatch_log_group.eks_cluster.name
      redis_slow  = aws_cloudwatch_log_group.redis_slow.name
    }
    s3_buckets = {
      logs            = aws_s3_bucket.logs.bucket
      model_artifacts = aws_s3_bucket.model_artifacts.bucket
    }
  }
}

# =============================================================================
# TAGS INFORMATION
# =============================================================================

output "common_tags" {
  description = "Common tags applied to all resources"
  value       = local.common_tags
}

# =============================================================================
# COST TRACKING OUTPUTS
# =============================================================================

output "cost_tracking" {
  description = "Information for cost tracking and optimization"
  value = {
    environment           = local.environment
    project_name         = local.project_name
    cost_center          = var.cost_center
    owner               = var.owner
    resource_prefix     = local.name_prefix
    eks_node_types      = aws_eks_node_group.main.instance_types
    eks_capacity_type   = aws_eks_node_group.main.capacity_type
    db_instance_class   = aws_db_instance.postgresql.instance_class
    redis_node_type     = aws_elasticache_replication_group.redis.node_type
  }
}