# =============================================================================
# DEVELOPMENT ENVIRONMENT CONFIGURATION
# Terraform variables for development environment
# =============================================================================

# Core configuration
environment = "development"
aws_region  = "us-west-2"
owner       = "development-team"
cost_center = "engineering-dev"

# Networking
allowed_cidr_blocks = ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]

# EKS Configuration
kubernetes_version    = "1.28"
node_instance_types   = ["t3.medium", "t3.large"]
node_desired_capacity = 2
node_min_capacity     = 1
node_max_capacity     = 5
node_volume_size      = 30

# Database Configuration
db_instance_class           = "db.t3.medium"
db_allocated_storage        = 20
db_max_allocated_storage    = 100
db_backup_retention_period  = 7
db_multi_az                = false
db_deletion_protection     = false
enable_performance_insights = false
db_monitoring_interval     = 0

# Redis Configuration
redis_node_type                    = "cache.t4g.micro"
redis_num_cache_clusters          = 1
redis_snapshot_retention_limit    = 1
enable_redis_auth                 = true
enable_redis_encryption_at_rest   = true
enable_redis_encryption_in_transit = true

# Monitoring and Logging
enable_cloudwatch_logs    = true
log_retention_days        = 7
enable_container_insights = false
enable_prometheus         = true
enable_grafana           = true
enable_alertmanager      = false

# Security Configuration
enable_encryption_at_rest     = true
enable_encryption_in_transit  = true
kms_key_rotation             = true
enable_waf                   = false
enable_vpc_flow_logs         = false
enable_config_rules          = false
enable_guardduty             = false
enable_security_hub          = false

# Backup and Disaster Recovery
enable_automated_backups     = true
backup_retention_period      = 7
enable_cross_region_backup   = false

# Cost Optimization
enable_spot_instances         = true
spot_instance_percentage     = 70
enable_scheduled_scaling     = false

# Feature Flags
enable_blue_green_deployment = false
enable_canary_deployment     = false
enable_istio_service_mesh    = false
enable_external_dns          = true
enable_cert_manager          = true

# Application Configuration
application_image_tag = "development"
application_replicas  = 1
application_resources = {
  requests = {
    cpu    = "100m"
    memory = "128Mi"
  }
  limits = {
    cpu    = "500m"
    memory = "512Mi"
  }
}

# Application Environment Variables
application_environment_variables = {
  ENVIRONMENT                = "development"
  LOG_LEVEL                 = "DEBUG"
  DEBUG                     = "true"
  ENABLE_SWAGGER            = "true"
  ENABLE_METRICS           = "true"
  ENABLE_PROFILING         = "true"
  DATABASE_POOL_SIZE       = "5"
  REDIS_POOL_SIZE          = "5"
  API_RATE_LIMIT           = "1000"
  FEATURE_TOGGLE_ADVANCED  = "true"
  CACHE_TTL                = "300"
}

# Domain Configuration
domain_name           = "dev.detection-platform.local"
subdomain            = "dev"
enable_route53       = false

# Integration Configuration
enable_datadog_integration  = false
enable_newrelic_integration = false
enable_slack_notifications  = true

# Development Tools
enable_development_tools   = true
enable_load_testing       = true
enable_chaos_engineering   = false

# Compliance
compliance_standards    = ["SOC2"]
enable_audit_logging   = false
enable_data_classification = false

# Terraform State
terraform_state_bucket     = "anomaly-detection-terraform-state-dev"
terraform_state_lock_table = "terraform-state-lock-dev"
terraform_state_region     = "us-west-2"