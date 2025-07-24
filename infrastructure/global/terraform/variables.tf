# Variables for Global Multi-Region Infrastructure

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "domain_name" {
  description = "Primary domain name for the global deployment"
  type        = string
}

variable "primary_region" {
  description = "Primary AWS region for global deployment"
  type        = string
  default     = "us-west-2"
}

variable "secondary_region" {
  description = "Secondary AWS region for global deployment"
  type        = string
  default     = "us-east-1"
}

variable "tertiary_region" {
  description = "Tertiary AWS region for disaster recovery"
  type        = string
  default     = "eu-west-1"
}

variable "global_database_password" {
  description = "Password for global database cluster"
  type        = string
  sensitive   = true
}

variable "common_tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default = {
    Project   = "MLOps-Global"
    ManagedBy = "Terraform"
  }
}

# Regional Configuration
variable "regional_configs" {
  description = "Configuration for each region"
  type = object({
    primary = object({
      vpc_cidr = string
      database = object({
        instance_class       = string
        allocated_storage    = number
        max_allocated_storage = number
      })
      redis = object({
        node_type           = string
        num_cache_nodes     = number
      })
    })
    secondary = object({
      vpc_cidr = string
      database = object({
        instance_class       = string
        allocated_storage    = number
        max_allocated_storage = number
      })
      redis = object({
        node_type           = string
        num_cache_nodes     = number
      })
    })
    tertiary = object({
      vpc_cidr = string
      database = object({
        instance_class       = string
        allocated_storage    = number
        max_allocated_storage = number
      })
      redis = object({
        node_type           = string
        num_cache_nodes     = number
      })
    })
  })
  
  default = {
    primary = {
      vpc_cidr = "10.0.0.0/16"
      database = {
        instance_class       = "db.r5.4xlarge"
        allocated_storage    = 1000
        max_allocated_storage = 5000
      }
      redis = {
        node_type           = "cache.r6g.4xlarge"
        num_cache_nodes     = 6
      }
    }
    secondary = {
      vpc_cidr = "10.1.0.0/16"
      database = {
        instance_class       = "db.r5.2xlarge"
        allocated_storage    = 500
        max_allocated_storage = 2000
      }
      redis = {
        node_type           = "cache.r6g.2xlarge"
        num_cache_nodes     = 3
      }
    }
    tertiary = {
      vpc_cidr = "10.2.0.0/16"
      database = {
        instance_class       = "db.r5.xlarge"
        allocated_storage    = 200
        max_allocated_storage = 1000
      }
      redis = {
        node_type           = "cache.r6g.large"
        num_cache_nodes     = 2
      }
    }
  }
}

# CDN Configuration
variable "cdn_config" {
  description = "CloudFront CDN configuration"
  type = object({
    price_class                = string
    minimum_protocol_version   = string
    geo_restriction_type       = string
    geo_restriction_locations  = list(string)
    compress                   = bool
    default_ttl               = number
    max_ttl                   = number
    min_ttl                   = number
  })
  
  default = {
    price_class                = "PriceClass_All"
    minimum_protocol_version   = "TLSv1.2_2021"
    geo_restriction_type       = "none"
    geo_restriction_locations  = []
    compress                   = true
    default_ttl               = 86400
    max_ttl                   = 31536000
    min_ttl                   = 0
  }
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

# Performance Configuration
variable "performance_config" {
  description = "Global performance configuration"
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

# Security Configuration
variable "security_config" {
  description = "Global security configuration"
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

# Monitoring Configuration
variable "monitoring_config" {
  description = "Global monitoring configuration"
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

# Traffic Distribution
variable "traffic_distribution" {
  description = "Global traffic distribution configuration"
  type = object({
    primary_percentage   = number
    secondary_percentage = number
    tertiary_percentage  = number
    weighted_routing     = bool
    latency_based_routing = bool
    geo_routing_enabled  = bool
  })
  
  default = {
    primary_percentage   = 60
    secondary_percentage = 30
    tertiary_percentage  = 10
    weighted_routing     = true
    latency_based_routing = true
    geo_routing_enabled  = true
  }
}

# Feature Flags
variable "feature_flags" {
  description = "Global feature flags"
  type = object({
    multi_region_deployment  = bool
    edge_computing_enabled   = bool
    ai_optimization_enabled  = bool
    quantum_ready_enabled    = bool
    blockchain_integration   = bool
    iot_support_enabled      = bool
  })
  
  default = {
    multi_region_deployment  = true
    edge_computing_enabled   = true
    ai_optimization_enabled  = true
    quantum_ready_enabled    = false
    blockchain_integration   = false
    iot_support_enabled      = true
  }
}

# Compliance Configuration
variable "compliance_config" {
  description = "Compliance and governance configuration"
  type = object({
    gdpr_compliance          = bool
    hipaa_compliance         = bool
    sox_compliance           = bool
    pci_compliance           = bool
    data_residency_enforcement = bool
    audit_logging_enabled    = bool
  })
  
  default = {
    gdpr_compliance          = true
    hipaa_compliance         = false
    sox_compliance           = false
    pci_compliance           = false
    data_residency_enforcement = true
    audit_logging_enabled    = true
  }
}