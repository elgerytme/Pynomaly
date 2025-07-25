# =============================================================================
# TERRAFORM VARIABLES CONFIGURATION
# Comprehensive variable definitions for multi-environment infrastructure
# =============================================================================

# =============================================================================
# CORE CONFIGURATION VARIABLES
# =============================================================================

variable "environment" {
  description = "The deployment environment (development, staging, production)"
  type        = string
  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be one of: development, staging, production."
  }
}

variable "aws_region" {
  description = "AWS region for resource deployment"
  type        = string
  default     = "us-west-2"
  validation {
    condition     = can(regex("^[a-z]{2}-[a-z]+-[0-9]$", var.aws_region))
    error_message = "AWS region must be in the format: us-west-2, eu-central-1, etc."
  }
}

variable "gcp_project_id" {
  description = "Google Cloud Project ID"
  type        = string
  default     = ""
}

variable "gcp_region" {
  description = "Google Cloud region"
  type        = string
  default     = "us-west1"
}

variable "azure_location" {
  description = "Azure location/region"
  type        = string
  default     = "West US 2"
}

# =============================================================================
# PROJECT METADATA
# =============================================================================

variable "owner" {
  description = "Owner of the infrastructure resources"
  type        = string
  default     = "platform-team"
}

variable "cost_center" {
  description = "Cost center for billing and resource tracking"
  type        = string
  default     = "engineering"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "anomaly-detection-platform"
}

# =============================================================================
# NETWORKING CONFIGURATION
# =============================================================================

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed for external access"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict in production
  validation {
    condition     = length(var.allowed_cidr_blocks) > 0
    error_message = "At least one CIDR block must be specified."
  }
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = null  # Will be set based on environment in locals
  validation {
    condition = var.vpc_cidr == null || can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid CIDR block."
  }
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for private subnets"
  type        = bool
  default     = true
}

variable "enable_vpn_gateway" {
  description = "Enable VPN Gateway for on-premises connectivity"
  type        = bool
  default     = false
}

# =============================================================================
# KUBERNETES CONFIGURATION
# =============================================================================

variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
  validation {
    condition     = can(regex("^[0-9]+\\.[0-9]+$", var.kubernetes_version))
    error_message = "Kubernetes version must be in format X.Y (e.g., 1.28)."
  }
}

variable "kubernetes_config_path" {
  description = "Path to Kubernetes config file"
  type        = string
  default     = "~/.kube/config"
}

variable "node_instance_types" {
  description = "EC2 instance types for EKS worker nodes"
  type        = list(string)
  default     = null  # Will be set based on environment
}

variable "node_desired_capacity" {
  description = "Desired number of worker nodes"
  type        = number
  default     = null  # Will be set based on environment
  validation {
    condition = var.node_desired_capacity == null || (var.node_desired_capacity >= 1 && var.node_desired_capacity <= 100)
    error_message = "Node desired capacity must be between 1 and 100."
  }
}

variable "node_min_capacity" {
  description = "Minimum number of worker nodes"
  type        = number
  default     = null  # Will be set based on environment
  validation {
    condition = var.node_min_capacity == null || (var.node_min_capacity >= 1 && var.node_min_capacity <= 100)
    error_message = "Node minimum capacity must be between 1 and 100."
  }
}

variable "node_max_capacity" {
  description = "Maximum number of worker nodes"
  type        = number
  default     = null  # Will be set based on environment
  validation {
    condition = var.node_max_capacity == null || (var.node_max_capacity >= 1 && var.node_max_capacity <= 1000)
    error_message = "Node maximum capacity must be between 1 and 1000."
  }
}

variable "node_volume_size" {
  description = "EBS volume size for worker nodes (GB)"
  type        = number
  default     = 50
  validation {
    condition     = var.node_volume_size >= 20 && var.node_volume_size <= 1000
    error_message = "Node volume size must be between 20 and 1000 GB."
  }
}

variable "enable_cluster_autoscaler" {
  description = "Enable Kubernetes cluster autoscaler"
  type        = bool
  default     = true
}

variable "enable_horizontal_pod_autoscaler" {
  description = "Enable Horizontal Pod Autoscaler"
  type        = bool
  default     = true
}

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = null  # Will be set based on environment
}

variable "db_allocated_storage" {
  description = "Initial allocated storage for RDS (GB)"
  type        = number
  default     = null  # Will be set based on environment
  validation {
    condition = var.db_allocated_storage == null || (var.db_allocated_storage >= 20 && var.db_allocated_storage <= 65536)
    error_message = "Database allocated storage must be between 20 and 65536 GB."
  }
}

variable "db_max_allocated_storage" {
  description = "Maximum allocated storage for RDS auto-scaling (GB)"
  type        = number
  default     = null  # Will be set based on environment
  validation {
    condition = var.db_max_allocated_storage == null || (var.db_max_allocated_storage >= 20 && var.db_max_allocated_storage <= 65536)
    error_message = "Database max allocated storage must be between 20 and 65536 GB."
  }
}

variable "db_backup_retention_period" {
  description = "Number of days to retain database backups"
  type        = number
  default     = null  # Will be set based on environment
  validation {
    condition = var.db_backup_retention_period == null || (var.db_backup_retention_period >= 0 && var.db_backup_retention_period <= 35)
    error_message = "Backup retention period must be between 0 and 35 days."
  }
}

variable "db_multi_az" {
  description = "Enable Multi-AZ deployment for RDS"
  type        = bool
  default     = null  # Will be set based on environment
}

variable "db_deletion_protection" {
  description = "Enable deletion protection for RDS"
  type        = bool
  default     = null  # Will be set based on environment
}

variable "enable_performance_insights" {
  description = "Enable Performance Insights for RDS"
  type        = bool
  default     = null  # Will be set based on environment
}

variable "db_monitoring_interval" {
  description = "Enhanced monitoring interval for RDS (seconds)"
  type        = number
  default     = null  # Will be set based on environment
  validation {
    condition = var.db_monitoring_interval == null || contains([0, 1, 5, 10, 15, 30, 60], var.db_monitoring_interval)
    error_message = "Monitoring interval must be one of: 0, 1, 5, 10, 15, 30, 60."
  }
}

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = null  # Will be set based on environment
}

variable "redis_num_cache_clusters" {
  description = "Number of cache clusters for Redis"
  type        = number
  default     = null  # Will be set based on environment
  validation {
    condition = var.redis_num_cache_clusters == null || (var.redis_num_cache_clusters >= 1 && var.redis_num_cache_clusters <= 6)
    error_message = "Redis cache clusters must be between 1 and 6."
  }
}

variable "redis_snapshot_retention_limit" {
  description = "Number of days to retain Redis snapshots"
  type        = number
  default     = null  # Will be set based on environment
  validation {
    condition = var.redis_snapshot_retention_limit == null || (var.redis_snapshot_retention_limit >= 0 && var.redis_snapshot_retention_limit <= 35)
    error_message = "Redis snapshot retention must be between 0 and 35 days."
  }
}

variable "enable_redis_auth" {
  description = "Enable Redis authentication"
  type        = bool
  default     = true
}

variable "enable_redis_encryption_at_rest" {
  description = "Enable Redis encryption at rest"
  type        = bool
  default     = true
}

variable "enable_redis_encryption_in_transit" {
  description = "Enable Redis encryption in transit"
  type        = bool
  default     = true
}

# =============================================================================
# MONITORING AND LOGGING
# =============================================================================

variable "enable_cloudwatch_logs" {
  description = "Enable CloudWatch logging"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch log retention period (days)"
  type        = number
  default     = null  # Will be set based on environment
  validation {
    condition = var.log_retention_days == null || contains([1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653], var.log_retention_days)
    error_message = "Log retention days must be a valid CloudWatch retention period."
  }
}

variable "enable_container_insights" {
  description = "Enable Container Insights for EKS"
  type        = bool
  default     = true
}

variable "enable_prometheus" {
  description = "Enable Prometheus monitoring"
  type        = bool
  default     = true
}

variable "enable_grafana" {
  description = "Enable Grafana dashboards"
  type        = bool
  default     = true
}

variable "enable_alertmanager" {
  description = "Enable Alertmanager for alerts"
  type        = bool
  default     = true
}

# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================

variable "enable_encryption_at_rest" {
  description = "Enable encryption at rest for all resources"
  type        = bool
  default     = true
}

variable "enable_encryption_in_transit" {
  description = "Enable encryption in transit for all resources"
  type        = bool
  default     = true
}

variable "kms_key_rotation" {
  description = "Enable automatic KMS key rotation"
  type        = bool
  default     = true
}

variable "ssl_certificate_arn" {
  description = "ARN of SSL certificate for HTTPS"
  type        = string
  default     = ""
}

variable "enable_waf" {
  description = "Enable AWS WAF for web applications"
  type        = bool
  default     = true
}

variable "enable_vpc_flow_logs" {
  description = "Enable VPC Flow Logs"
  type        = bool
  default     = true
}

variable "enable_config_rules" {
  description = "Enable AWS Config rules for compliance"
  type        = bool
  default     = true
}

variable "enable_guardduty" {
  description = "Enable AWS GuardDuty for threat detection"
  type        = bool
  default     = true
}

variable "enable_security_hub" {
  description = "Enable AWS Security Hub"
  type        = bool
  default     = true
}

# =============================================================================
# BACKUP AND DISASTER RECOVERY
# =============================================================================

variable "enable_automated_backups" {
  description = "Enable automated backups for all resources"
  type        = bool
  default     = true
}

variable "backup_retention_period" {
  description = "Backup retention period (days)"
  type        = number
  default     = null  # Will be set based on environment
  validation {
    condition = var.backup_retention_period == null || (var.backup_retention_period >= 1 && var.backup_retention_period <= 365)
    error_message = "Backup retention period must be between 1 and 365 days."
  }
}

variable "enable_cross_region_backup" {
  description = "Enable cross-region backup replication"
  type        = bool
  default     = null  # Will be set based on environment
}

variable "disaster_recovery_region" {
  description = "AWS region for disaster recovery"
  type        = string
  default     = "us-east-1"
}

# =============================================================================
# COST OPTIMIZATION
# =============================================================================

variable "enable_spot_instances" {
  description = "Enable Spot instances for cost optimization"
  type        = bool
  default     = null  # Will be set based on environment
}

variable "spot_instance_percentage" {
  description = "Percentage of Spot instances in node groups"
  type        = number
  default     = 50
  validation {
    condition     = var.spot_instance_percentage >= 0 && var.spot_instance_percentage <= 100
    error_message = "Spot instance percentage must be between 0 and 100."
  }
}

variable "enable_scheduled_scaling" {
  description = "Enable scheduled scaling for cost optimization"
  type        = bool
  default     = false
}

# =============================================================================
# FEATURE FLAGS
# =============================================================================

variable "enable_blue_green_deployment" {
  description = "Enable blue-green deployment capability"
  type        = bool
  default     = true
}

variable "enable_canary_deployment" {
  description = "Enable canary deployment capability"
  type        = bool
  default     = true
}

variable "enable_istio_service_mesh" {
  description = "Enable Istio service mesh"
  type        = bool
  default     = false
}

variable "enable_external_dns" {
  description = "Enable External DNS for automatic DNS management"
  type        = bool
  default     = true
}

variable "enable_cert_manager" {
  description = "Enable cert-manager for automatic certificate management"
  type        = bool
  default     = true
}

# =============================================================================
# APPLICATION CONFIGURATION
# =============================================================================

variable "application_image_tag" {
  description = "Docker image tag for the application"
  type        = string
  default     = "latest"
}

variable "application_replicas" {
  description = "Number of application replicas"
  type        = number
  default     = null  # Will be set based on environment
  validation {
    condition = var.application_replicas == null || (var.application_replicas >= 1 && var.application_replicas <= 100)
    error_message = "Application replicas must be between 1 and 100."
  }
}

variable "application_resources" {
  description = "Resource requests and limits for application containers"
  type = object({
    requests = object({
      cpu    = string
      memory = string
    })
    limits = object({
      cpu    = string
      memory = string
    })
  })
  default = null  # Will be set based on environment
}

variable "application_environment_variables" {
  description = "Environment variables for the application"
  type        = map(string)
  default     = {}
  sensitive   = true
}

variable "application_secrets" {
  description = "Secrets for the application stored in AWS Secrets Manager"
  type        = map(string)
  default     = {}
  sensitive   = true
}

# =============================================================================
# DOMAIN AND DNS CONFIGURATION
# =============================================================================

variable "domain_name" {
  description = "Primary domain name for the application"
  type        = string
  default     = ""
}

variable "subdomain" {
  description = "Subdomain for the environment"
  type        = string
  default     = ""
}

variable "enable_route53" {
  description = "Enable Route53 DNS management"
  type        = bool
  default     = true
}

variable "dns_zone_id" {
  description = "Route53 hosted zone ID"
  type        = string
  default     = ""
}

# =============================================================================
# INTEGRATION CONFIGURATION
# =============================================================================

variable "enable_datadog_integration" {
  description = "Enable Datadog monitoring integration"
  type        = bool
  default     = false
}

variable "datadog_api_key" {
  description = "Datadog API key"
  type        = string
  default     = ""
  sensitive   = true
}

variable "enable_newrelic_integration" {
  description = "Enable New Relic monitoring integration"
  type        = bool
  default     = false
}

variable "newrelic_license_key" {
  description = "New Relic license key"
  type        = string
  default     = ""
  sensitive   = true
}

variable "enable_slack_notifications" {
  description = "Enable Slack notifications for alerts"
  type        = bool
  default     = false
}

variable "slack_webhook_url" {
  description = "Slack webhook URL for notifications"
  type        = string
  default     = ""
  sensitive   = true
}

# =============================================================================
# DEVELOPMENT AND TESTING
# =============================================================================

variable "enable_development_tools" {
  description = "Enable development and debugging tools"
  type        = bool
  default     = null  # Will be set based on environment
}

variable "enable_load_testing" {
  description = "Enable load testing infrastructure"
  type        = bool
  default     = false
}

variable "enable_chaos_engineering" {
  description = "Enable chaos engineering tools"
  type        = bool
  default     = false
}

# =============================================================================
# COMPLIANCE AND GOVERNANCE
# =============================================================================

variable "compliance_standards" {
  description = "List of compliance standards to adhere to"
  type        = list(string)
  default     = ["SOC2", "ISO27001"]
  validation {
    condition = alltrue([
      for standard in var.compliance_standards : contains(["SOC2", "ISO27001", "PCI-DSS", "HIPAA", "GDPR"], standard)
    ])
    error_message = "Compliance standards must be from: SOC2, ISO27001, PCI-DSS, HIPAA, GDPR."
  }
}

variable "enable_audit_logging" {
  description = "Enable comprehensive audit logging"
  type        = bool
  default     = true
}

variable "enable_data_classification" {
  description = "Enable data classification and tagging"
  type        = bool
  default     = true
}

# =============================================================================
# TERRAFORM CONFIGURATION
# =============================================================================

variable "terraform_state_bucket" {
  description = "S3 bucket for Terraform state"
  type        = string
  default     = "anomaly-detection-terraform-state"
}

variable "terraform_state_lock_table" {
  description = "DynamoDB table for Terraform state locking"
  type        = string
  default     = "terraform-state-lock"
}

variable "terraform_state_region" {
  description = "AWS region for Terraform state storage"
  type        = string
  default     = "us-west-2"
}