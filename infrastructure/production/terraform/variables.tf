# Variables for production infrastructure deployment

variable "environment" {
  description = "Environment name (production, staging, etc.)"
  type        = string
  default     = "production"
  
  validation {
    condition = contains(["production", "staging", "development"], var.environment)
    error_message = "Environment must be one of: production, staging, development."
  }
}

variable "aws_region" {
  description = "AWS region for resource deployment"
  type        = string
  default     = "us-west-2"
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access EKS cluster"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict this in production
}

# EKS Configuration
variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "eks_admin_users" {
  description = "List of AWS users with admin access to EKS cluster"
  type = list(object({
    userarn  = string
    username = string
    groups   = list(string)
  }))
  default = []
}

# PostgreSQL Configuration
variable "postgres_instance_class" {
  description = "RDS instance class for PostgreSQL"
  type        = string
  default     = "db.r5.2xlarge"
}

variable "postgres_allocated_storage" {
  description = "Initial allocated storage for PostgreSQL (GB)"
  type        = number
  default     = 500
}

variable "postgres_max_allocated_storage" {
  description = "Maximum allocated storage for PostgreSQL auto-scaling (GB)"
  type        = number
  default     = 2000
}

variable "postgres_database_name" {
  description = "Name of the PostgreSQL database"
  type        = string
  default     = "mlops"
}

variable "postgres_username" {
  description = "Master username for PostgreSQL"
  type        = string
  default     = "mlops_admin"
}

variable "postgres_password" {
  description = "Master password for PostgreSQL"
  type        = string
  sensitive   = true
}

# Redis Configuration
variable "redis_node_type" {
  description = "ElastiCache node type for Redis"
  type        = string
  default     = "cache.r6g.2xlarge"
}

variable "redis_num_cache_nodes" {
  description = "Number of cache nodes in Redis cluster"
  type        = number
  default     = 3
}

# Node Group Scaling Configuration
variable "cpu_node_min_size" {
  description = "Minimum number of CPU-optimized nodes"
  type        = number
  default     = 3
}

variable "cpu_node_max_size" {
  description = "Maximum number of CPU-optimized nodes"
  type        = number
  default     = 20
}

variable "cpu_node_desired_size" {
  description = "Desired number of CPU-optimized nodes"
  type        = number
  default     = 5
}

variable "memory_node_min_size" {
  description = "Minimum number of memory-optimized nodes"
  type        = number
  default     = 2
}

variable "memory_node_max_size" {
  description = "Maximum number of memory-optimized nodes"
  type        = number
  default     = 15
}

variable "memory_node_desired_size" {
  description = "Desired number of memory-optimized nodes"
  type        = number
  default     = 3
}

variable "gpu_node_min_size" {
  description = "Minimum number of GPU-enabled nodes"
  type        = number
  default     = 0
}

variable "gpu_node_max_size" {
  description = "Maximum number of GPU-enabled nodes"
  type        = number
  default     = 10
}

variable "gpu_node_desired_size" {
  description = "Desired number of GPU-enabled nodes"
  type        = number
  default     = 2
}

# Monitoring and Logging
variable "log_retention_days" {
  description = "CloudWatch log retention period in days"
  type        = number
  default     = 30
}

variable "enable_flow_logs" {
  description = "Enable VPC flow logs"
  type        = bool
  default     = true
}

# Security Configuration
variable "enable_waf" {
  description = "Enable WAF for application protection"
  type        = bool
  default     = true
}

variable "waf_rate_limit" {
  description = "WAF rate limit per IP (requests per 5 minutes)"
  type        = number
  default     = 1000
}

# Backup and Recovery
variable "postgres_backup_retention_period" {
  description = "PostgreSQL backup retention period in days"
  type        = number
  default     = 7
  
  validation {
    condition = var.postgres_backup_retention_period >= 1 && var.postgres_backup_retention_period <= 35
    error_message = "Backup retention period must be between 1 and 35 days."
  }
}

variable "redis_snapshot_retention_limit" {
  description = "Redis snapshot retention limit"
  type        = number
  default     = 5
}

# Tagging
variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

# Cost Management
variable "enable_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = false
}

variable "spot_instance_percentage" {
  description = "Percentage of capacity to use spot instances"
  type        = number
  default     = 50
  
  validation {
    condition = var.spot_instance_percentage >= 0 && var.spot_instance_percentage <= 100
    error_message = "Spot instance percentage must be between 0 and 100."
  }
}

# High Availability
variable "multi_az_enabled" {
  description = "Enable multi-AZ deployment for high availability"
  type        = bool
  default     = true
}

# Auto Scaling Configuration
variable "cluster_autoscaler_enabled" {
  description = "Enable cluster autoscaler"
  type        = bool
  default     = true
}

variable "metrics_server_enabled" {
  description = "Enable metrics server for HPA"
  type        = bool
  default     = true
}

# Network Security
variable "enable_network_policy" {
  description = "Enable Kubernetes network policies"
  type        = bool
  default     = true
}

variable "enable_pod_security_policy" {
  description = "Enable pod security policies"
  type        = bool
  default     = true
}

# Encryption
variable "enable_encryption_at_rest" {
  description = "Enable encryption at rest for all storage"
  type        = bool
  default     = true
}

variable "enable_encryption_in_transit" {
  description = "Enable encryption in transit"
  type        = bool
  default     = true
}

# Domain and SSL
variable "domain_name" {
  description = "Domain name for the MLOps platform"
  type        = string
  default     = ""
}

variable "ssl_certificate_arn" {
  description = "ARN of SSL certificate for HTTPS"
  type        = string
  default     = ""
}

# Disaster Recovery
variable "enable_cross_region_backup" {
  description = "Enable cross-region backup for disaster recovery"
  type        = bool
  default     = true
}

variable "backup_region" {
  description = "AWS region for cross-region backups"
  type        = string
  default     = "us-east-1"
}

# Performance Configuration
variable "enable_enhanced_monitoring" {
  description = "Enable enhanced monitoring for RDS"
  type        = bool
  default     = true
}

variable "enable_performance_insights" {
  description = "Enable Performance Insights for RDS"
  type        = bool
  default     = true
}

# Development and Testing
variable "enable_debug_mode" {
  description = "Enable debug mode for troubleshooting"
  type        = bool
  default     = false
}