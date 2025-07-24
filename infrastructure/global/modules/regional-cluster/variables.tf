# Variables for Regional Cluster Module

variable "region" {
  description = "AWS region for this cluster"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
}

variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access EKS cluster"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "cross_region_cidrs" {
  description = "CIDR blocks from other regions for cross-region communication"
  type        = list(string)
  default     = []
}

variable "node_groups" {
  description = "Configuration for EKS managed node groups"
  type = map(object({
    instance_types = list(string)
    min_size       = number
    max_size       = number
    desired_size   = number
  }))
}

variable "database_config" {
  description = "Database configuration"
  type = object({
    instance_class       = string
    allocated_storage    = number
    max_allocated_storage = number
  })
}

variable "redis_config" {
  description = "Redis configuration"
  type = object({
    node_type           = string
    num_cache_nodes     = number
  })
}

variable "database_password" {
  description = "Password for the regional database"
  type        = string
  sensitive   = true
  default     = ""
}

variable "region_role" {
  description = "Role of this region (primary, secondary, disaster_recovery)"
  type        = string
  
  validation {
    condition     = contains(["primary", "secondary", "disaster_recovery"], var.region_role)
    error_message = "Region role must be one of: primary, secondary, disaster_recovery."
  }
}

variable "global_cluster_identifier" {
  description = "Global cluster identifier for database replication"
  type        = string
  default     = null
}

variable "global_redis_replication_group_id" {
  description = "Global Redis replication group ID"
  type        = string
  default     = null
}

variable "global_s3_bucket_data" {
  description = "Global S3 bucket for data"
  type        = string
}

variable "global_s3_bucket_models" {
  description = "Global S3 bucket for models"
  type        = string
}

variable "global_s3_bucket_artifacts" {
  description = "Global S3 bucket for artifacts"
  type        = string
}

variable "tags" {
  description = "Common tags to apply to resources"
  type        = map(string)
  default     = {}
}

variable "ssh_key_name" {
  description = "EC2 Key Pair name for node access"
  type        = string
  default     = null
}

# Auto Scaling Configuration
variable "global_auto_scaling" {
  description = "Global auto-scaling configuration"
  type = object({
    enabled                    = bool
    scale_out_cooldown        = number
    scale_in_cooldown         = number
    target_capacity           = number
    max_capacity              = number
    min_capacity              = number
    predictive_scaling_enabled = bool
  })
  
  default = {
    enabled                    = true
    scale_out_cooldown        = 300
    scale_in_cooldown         = 600
    target_capacity           = 70
    max_capacity              = 200
    min_capacity              = 20
    predictive_scaling_enabled = true
  }
}

# Disaster Recovery Configuration
variable "disaster_recovery" {
  description = "Disaster recovery configuration"
  type = object({
    enabled                 = bool
    backup_retention_days   = number
    cross_region_backup     = bool
    failover_time_minutes   = number
    auto_failover_enabled   = bool
  })
  
  default = {
    enabled                 = true
    backup_retention_days   = 30
    cross_region_backup     = true
    failover_time_minutes   = 15
    auto_failover_enabled   = true
  }
}

# Cost Optimization
variable "cost_optimization" {
  description = "Cost optimization configuration"
  type = object({
    spot_instances_enabled     = bool
    reserved_instances_enabled = bool
    savings_plans_enabled      = bool
    right_sizing_enabled       = bool
    storage_optimization       = bool
    data_lifecycle_enabled     = bool
  })
  
  default = {
    spot_instances_enabled     = true
    reserved_instances_enabled = true
    savings_plans_enabled      = true
    right_sizing_enabled       = true
    storage_optimization       = true
    data_lifecycle_enabled     = true
  }
}

# Monitoring Configuration
variable "monitoring_config" {
  description = "Monitoring configuration"
  type = object({
    detailed_monitoring_enabled = bool
    custom_metrics_enabled     = bool
    log_retention_days         = number
    alerting_enabled           = bool
    dashboard_enabled          = bool
    real_user_monitoring       = bool
  })
  
  default = {
    detailed_monitoring_enabled = true
    custom_metrics_enabled     = true
    log_retention_days         = 90
    alerting_enabled           = true
    dashboard_enabled          = true
    real_user_monitoring       = true
  }
}

# Security Configuration
variable "security_config" {
  description = "Security configuration"
  type = object({
    waf_enabled               = bool
    ddos_protection          = bool
    ssl_security_policy      = string
    hsts_enabled             = bool
    csp_enabled              = bool
    rate_limiting_enabled    = bool
    geo_blocking_enabled     = bool
  })
  
  default = {
    waf_enabled               = true
    ddos_protection          = true
    ssl_security_policy      = "TLSv1.2_2021"
    hsts_enabled             = true
    csp_enabled              = true
    rate_limiting_enabled    = true
    geo_blocking_enabled     = false
  }
}

# Performance Configuration
variable "performance_config" {
  description = "Performance configuration"
  type = object({
    cache_cluster_enabled      = bool
    content_acceleration      = bool
    image_optimization        = bool
    gzip_compression          = bool
    brotli_compression        = bool
    http2_enabled             = bool
    websocket_support         = bool
  })
  
  default = {
    cache_cluster_enabled      = true
    content_acceleration      = true
    image_optimization        = true
    gzip_compression          = true
    brotli_compression        = true
    http2_enabled             = true
    websocket_support         = true
  }
}

# Feature Flags
variable "feature_flags" {
  description = "Feature flags for this region"
  type = object({
    edge_computing_enabled   = bool
    ai_optimization_enabled  = bool
    quantum_ready_enabled    = bool
    blockchain_integration   = bool
    iot_support_enabled      = bool
  })
  
  default = {
    edge_computing_enabled   = true
    ai_optimization_enabled  = true
    quantum_ready_enabled    = false
    blockchain_integration   = false
    iot_support_enabled      = true
  }
}