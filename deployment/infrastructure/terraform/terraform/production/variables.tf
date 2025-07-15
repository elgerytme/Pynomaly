# Variables for Pynomaly production infrastructure

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "pynomaly-production"
}

variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.27"
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
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

# EKS Node Group Configuration
variable "node_instance_types" {
  description = "Instance types for EKS node group"
  type        = list(string)
  default     = ["m5.large", "m5.xlarge"]
}

variable "node_desired_capacity" {
  description = "Desired number of nodes in the node group"
  type        = number
  default     = 3
}

variable "node_max_capacity" {
  description = "Maximum number of nodes in the node group"
  type        = number
  default     = 10
}

variable "node_min_capacity" {
  description = "Minimum number of nodes in the node group"
  type        = number
  default     = 1
}

variable "node_disk_size" {
  description = "Disk size for EKS nodes (GB)"
  type        = number
  default     = 50
}

# Spot Node Group Configuration
variable "spot_instance_types" {
  description = "Instance types for EKS spot node group"
  type        = list(string)
  default     = ["m5.large", "m5.xlarge", "m4.large", "m4.xlarge"]
}

variable "spot_desired_capacity" {
  description = "Desired number of spot nodes"
  type        = number
  default     = 2
}

variable "spot_max_capacity" {
  description = "Maximum number of spot nodes"
  type        = number
  default     = 20
}

variable "spot_min_capacity" {
  description = "Minimum number of spot nodes"
  type        = number
  default     = 0
}

# RDS Configuration
variable "db_instance_class" {
  description = "Instance class for RDS database"
  type        = string
  default     = "db.t3.medium"
}

variable "db_allocated_storage" {
  description = "Allocated storage for RDS database (GB)"
  type        = number
  default     = 100
}

variable "db_max_allocated_storage" {
  description = "Maximum allocated storage for RDS database (GB)"
  type        = number
  default     = 1000
}

# Redis Configuration
variable "redis_instance_type" {
  description = "Instance type for Redis cluster"
  type        = string
  default     = "cache.t3.medium"
}

variable "redis_num_nodes" {
  description = "Number of nodes in Redis cluster"
  type        = number
  default     = 3
}

# Security Configuration
variable "api_server_authorized_ip_ranges" {
  description = "IP ranges authorized to access the API server"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict this in production
}

# Domain Configuration
variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "pynomaly.ai"
}

# Environment Tags
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "project" {
  description = "Project name"
  type        = string
  default     = "pynomaly"
}

variable "owner" {
  description = "Owner of the resources"
  type        = string
  default     = "devops"
}

# Backup Configuration
variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 30
}

# Monitoring Configuration
variable "enable_cloudwatch_logs" {
  description = "Enable CloudWatch logs for EKS cluster"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "Number of days to retain CloudWatch logs"
  type        = number
  default     = 30
}

# Cost Optimization
variable "enable_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = true
}

variable "enable_cluster_autoscaler" {
  description = "Enable cluster autoscaler"
  type        = bool
  default     = true
}

# High Availability Configuration
variable "multi_az_deployment" {
  description = "Enable multi-AZ deployment for high availability"
  type        = bool
  default     = true
}

variable "enable_cross_zone_load_balancing" {
  description = "Enable cross-zone load balancing"
  type        = bool
  default     = true
}

# Security Configuration
variable "enable_pod_security_policy" {
  description = "Enable Pod Security Policy"
  type        = bool
  default     = true
}

variable "enable_network_policy" {
  description = "Enable Network Policy"
  type        = bool
  default     = true
}

variable "enable_secrets_encryption" {
  description = "Enable secrets encryption at rest"
  type        = bool
  default     = true
}

# Storage Configuration
variable "enable_ebs_encryption" {
  description = "Enable EBS encryption"
  type        = bool
  default     = true
}

variable "ebs_volume_type" {
  description = "EBS volume type for storage"
  type        = string
  default     = "gp3"
}

# Application Configuration
variable "app_replicas_min" {
  description = "Minimum number of application replicas"
  type        = number
  default     = 3
}

variable "app_replicas_max" {
  description = "Maximum number of application replicas"
  type        = number
  default     = 50
}

variable "app_cpu_request" {
  description = "CPU request for application pods"
  type        = string
  default     = "500m"
}

variable "app_memory_request" {
  description = "Memory request for application pods"
  type        = string
  default     = "1Gi"
}

variable "app_cpu_limit" {
  description = "CPU limit for application pods"
  type        = string
  default     = "2000m"
}

variable "app_memory_limit" {
  description = "Memory limit for application pods"
  type        = string
  default     = "4Gi"
}

# Worker Configuration
variable "worker_replicas_min" {
  description = "Minimum number of worker replicas"
  type        = number
  default     = 2
}

variable "worker_replicas_max" {
  description = "Maximum number of worker replicas"
  type        = number
  default     = 20
}

variable "worker_cpu_request" {
  description = "CPU request for worker pods"
  type        = string
  default     = "1000m"
}

variable "worker_memory_request" {
  description = "Memory request for worker pods"
  type        = string
  default     = "2Gi"
}

variable "worker_cpu_limit" {
  description = "CPU limit for worker pods"
  type        = string
  default     = "4000m"
}

variable "worker_memory_limit" {
  description = "Memory limit for worker pods"
  type        = string
  default     = "8Gi"
}

# Load Balancer Configuration
variable "lb_type" {
  description = "Type of load balancer (application or network)"
  type        = string
  default     = "application"
}

variable "lb_scheme" {
  description = "Load balancer scheme (internet-facing or internal)"
  type        = string
  default     = "internet-facing"
}

variable "enable_waf" {
  description = "Enable WAF for the load balancer"
  type        = bool
  default     = true
}

# DNS Configuration
variable "enable_external_dns" {
  description = "Enable external DNS for automatic DNS management"
  type        = bool
  default     = true
}

variable "dns_zone_id" {
  description = "Route53 hosted zone ID"
  type        = string
  default     = ""
}

# Certificate Configuration
variable "enable_cert_manager" {
  description = "Enable cert-manager for automatic certificate management"
  type        = bool
  default     = true
}

variable "certificate_issuer" {
  description = "Certificate issuer (letsencrypt-prod or letsencrypt-staging)"
  type        = string
  default     = "letsencrypt-prod"
}

# Monitoring and Observability
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

variable "enable_jaeger" {
  description = "Enable Jaeger tracing"
  type        = bool
  default     = true
}

variable "enable_fluentbit" {
  description = "Enable Fluent Bit for log aggregation"
  type        = bool
  default     = true
}

# Disaster Recovery
variable "enable_cross_region_backup" {
  description = "Enable cross-region backup for disaster recovery"
  type        = bool
  default     = true
}

variable "backup_destination_region" {
  description = "Destination region for cross-region backups"
  type        = string
  default     = "us-west-2"
}

# Compliance and Governance
variable "enable_config_rules" {
  description = "Enable AWS Config rules for compliance"
  type        = bool
  default     = true
}

variable "enable_security_hub" {
  description = "Enable AWS Security Hub"
  type        = bool
  default     = true
}

variable "enable_guardduty" {
  description = "Enable AWS GuardDuty"
  type        = bool
  default     = true
}

# Performance Optimization
variable "enable_container_insights" {
  description = "Enable CloudWatch Container Insights"
  type        = bool
  default     = true
}

variable "enable_x_ray" {
  description = "Enable AWS X-Ray tracing"
  type        = bool
  default     = true
}

# Development and Testing
variable "enable_dev_tools" {
  description = "Enable development and debugging tools"
  type        = bool
  default     = false
}

variable "enable_chaos_engineering" {
  description = "Enable chaos engineering tools"
  type        = bool
  default     = false
}
