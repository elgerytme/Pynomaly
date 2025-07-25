# =============================================================================
# PRODUCTION ENVIRONMENT CONFIGURATION
# Terraform variables for production environment
# =============================================================================

# Core configuration
environment = "production"
aws_region  = "us-west-2"
owner       = "platform-team"
cost_center = "engineering-prod"

# Networking
allowed_cidr_blocks = ["10.0.0.0/16"]  # Restrict to corporate network

# EKS Configuration
kubernetes_version    = "1.28"
node_instance_types   = ["m5.large", "m5.xlarge", "c5.large"]
node_desired_capacity = 6
node_min_capacity     = 3
node_max_capacity     = 20
node_volume_size      = 100

# Database Configuration
db_instance_class           = "db.r5.xlarge"
db_allocated_storage        = 500
db_max_allocated_storage    = 2000
db_backup_retention_period  = 30
db_multi_az                = true
db_deletion_protection     = true
enable_performance_insights = true
db_monitoring_interval     = 60

# Redis Configuration
redis_node_type                    = "cache.r6g.large"
redis_num_cache_clusters          = 3
redis_snapshot_retention_limit    = 7
enable_redis_auth                 = true
enable_redis_encryption_at_rest   = true
enable_redis_encryption_in_transit = true

# Monitoring and Logging
enable_cloudwatch_logs    = true
log_retention_days        = 90
enable_container_insights = true
enable_prometheus         = true
enable_grafana           = true
enable_alertmanager      = true

# Security Configuration
enable_encryption_at_rest     = true
enable_encryption_in_transit  = true
kms_key_rotation             = true
enable_waf                   = true
enable_vpc_flow_logs         = true
enable_config_rules          = true
enable_guardduty             = true
enable_security_hub          = true

# Backup and Disaster Recovery
enable_automated_backups     = true
backup_retention_period      = 30
enable_cross_region_backup   = true
disaster_recovery_region     = "us-east-1"

# Cost Optimization
enable_spot_instances         = false
spot_instance_percentage     = 0
enable_scheduled_scaling     = true

# Feature Flags
enable_blue_green_deployment = true
enable_canary_deployment     = true
enable_istio_service_mesh    = true
enable_external_dns          = true
enable_cert_manager          = true

# Application Configuration
application_image_tag = "stable"
application_replicas  = 6
application_resources = {
  requests = {
    cpu    = "1000m"
    memory = "2Gi"
  }
  limits = {
    cpu    = "2000m"
    memory = "4Gi"
  }
}

# Application Environment Variables
application_environment_variables = {
  ENVIRONMENT                = "production"
  LOG_LEVEL                 = "INFO"
  DEBUG                     = "false"
  ENABLE_SWAGGER            = "false"
  ENABLE_METRICS           = "true"
  ENABLE_PROFILING         = "false"
  DATABASE_POOL_SIZE       = "20"
  REDIS_POOL_SIZE          = "20"
  API_RATE_LIMIT           = "10000"
  FEATURE_TOGGLE_ADVANCED  = "true"
  CACHE_TTL                = "3600"
  MAX_WORKERS              = "4"
  WORKER_TIMEOUT           = "30"
  ENABLE_CIRCUIT_BREAKER   = "true"
  CIRCUIT_BREAKER_THRESHOLD = "5"
  CIRCUIT_BREAKER_TIMEOUT   = "60"
}

# Domain Configuration
domain_name           = "detection-platform.io"
subdomain            = "api"
enable_route53       = true
dns_zone_id          = "Z1D633PJN98FT9"

# SSL Configuration
ssl_certificate_arn = "arn:aws:acm:us-west-2:123456789012:certificate/12345678-1234-1234-1234-123456789012"

# Integration Configuration
enable_datadog_integration  = true
enable_newrelic_integration = true
enable_slack_notifications  = true

# Development Tools
enable_development_tools   = false
enable_load_testing       = false
enable_chaos_engineering   = true

# Compliance
compliance_standards    = ["SOC2", "ISO27001", "PCI-DSS"]
enable_audit_logging   = true
enable_data_classification = true

# High Availability Configuration
enable_multi_region_deployment = true
enable_active_active_setup     = true

# Performance Configuration
enable_cdn                     = true
enable_edge_locations         = true
enable_global_load_balancing  = true

# Advanced Security
enable_network_segmentation   = true
enable_zero_trust_network     = true
enable_secrets_rotation       = true
enable_vulnerability_scanning = true

# Disaster Recovery
enable_automated_failover     = true
enable_backup_validation     = true
enable_disaster_recovery_testing = true

# Terraform State
terraform_state_bucket     = "anomaly-detection-terraform-state-prod"
terraform_state_lock_table = "terraform-state-lock-prod"
terraform_state_region     = "us-west-2"