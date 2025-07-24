# Global Multi-Region Infrastructure for MLOps Platform
# This configuration deploys the platform across multiple regions for global scale

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
  
  backend "s3" {
    bucket         = "mlops-terraform-state-global"
    key            = "global/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "mlops-terraform-locks-global"
  }
}

# Multiple AWS Provider Configurations for different regions
provider "aws" {
  alias  = "primary"
  region = var.primary_region
  
  default_tags {
    tags = var.common_tags
  }
}

provider "aws" {
  alias  = "secondary"
  region = var.secondary_region
  
  default_tags {
    tags = var.common_tags
  }
}

provider "aws" {
  alias  = "tertiary"
  region = var.tertiary_region
  
  default_tags {
    tags = var.common_tags
  }
}

# Data sources
data "aws_availability_zones" "primary" {
  provider = aws.primary
  state    = "available"
}

data "aws_availability_zones" "secondary" {
  provider = aws.secondary
  state    = "available"
}

data "aws_availability_zones" "tertiary" {
  provider = aws.tertiary
  state    = "available"
}

# Local values for consistent configuration
locals {
  regions = {
    primary   = var.primary_region
    secondary = var.secondary_region
    tertiary  = var.tertiary_region
  }
  
  cluster_names = {
    primary   = "${var.environment}-mlops-${var.primary_region}"
    secondary = "${var.environment}-mlops-${var.secondary_region}"
    tertiary  = "${var.environment}-mlops-${var.tertiary_region}"
  }
  
  common_tags = merge(var.common_tags, {
    Environment = var.environment
    Project     = "MLOps-Global"
    DeployedBy  = "Terraform"
  })
}

# Global DNS and CDN
resource "aws_route53_zone" "global" {
  provider = aws.primary
  name     = var.domain_name
  
  tags = merge(local.common_tags, {
    Name = "MLOps Global DNS Zone"
  })
}

# Global CloudFront Distribution
resource "aws_cloudfront_distribution" "global" {
  provider = aws.primary
  
  origin {
    domain_name = aws_lb.primary_global_alb.dns_name
    origin_id   = "primary-origin"
    
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }
  
  origin {
    domain_name = aws_lb.secondary_global_alb.dns_name
    origin_id   = "secondary-origin"
    
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }
  
  origin {
    domain_name = aws_lb.tertiary_global_alb.dns_name
    origin_id   = "tertiary-origin"
    
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }
  
  enabled = true
  
  # Global distribution
  price_class = "PriceClass_All"
  
  # Default cache behavior (primary region)
  default_cache_behavior {
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "primary-origin"
    compress               = true
    viewer_protocol_policy = "redirect-to-https"
    
    forwarded_values {
      query_string = true
      headers      = ["Host", "Authorization", "CloudFront-Forwarded-Proto"]
      
      cookies {
        forward = "all"
      }
    }
    
    min_ttl     = 0
    default_ttl = 86400
    max_ttl     = 31536000
  }
  
  # Geographic distribution with failover
  ordered_cache_behavior {
    path_pattern           = "/api/v1/*"
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "primary-origin"
    compress               = true
    viewer_protocol_policy = "redirect-to-https"
    
    forwarded_values {
      query_string = true
      headers      = ["*"]
      
      cookies {
        forward = "all"
      }
    }
    
    min_ttl     = 0
    default_ttl = 0
    max_ttl     = 0
  }
  
  # Geographic restrictions
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  # SSL Certificate
  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate.global.arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }
  
  # Custom error pages
  custom_error_response {
    error_code         = 404
    response_code      = 200
    response_page_path = "/index.html"
  }
  
  custom_error_response {
    error_code         = 403
    response_code      = 200
    response_page_path = "/index.html"
  }
  
  tags = merge(local.common_tags, {
    Name = "MLOps Global CDN"
  })
}

# SSL Certificate for CloudFront (must be in us-east-1)
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
}

resource "aws_acm_certificate" "global" {
  provider          = aws.us_east_1
  domain_name       = var.domain_name
  validation_method = "DNS"
  
  subject_alternative_names = [
    "*.${var.domain_name}",
    "api.${var.domain_name}",
    "models.${var.domain_name}"
  ]
  
  lifecycle {
    create_before_destroy = true
  }
  
  tags = merge(local.common_tags, {
    Name = "MLOps Global SSL Certificate"
  })
}

# Regional Infrastructure Modules
module "primary_region" {
  source = "../modules/regional-cluster"
  
  providers = {
    aws = aws.primary
  }
  
  region             = var.primary_region
  environment        = var.environment
  cluster_name       = local.cluster_names.primary
  vpc_cidr           = var.regional_configs.primary.vpc_cidr
  availability_zones = slice(data.aws_availability_zones.primary.names, 0, 3)
  
  # Scaling configuration for primary region (largest)
  node_groups = {
    cpu_optimized = {
      instance_types = ["c5.2xlarge", "c5.4xlarge"]
      min_size       = 5
      max_size       = 50
      desired_size   = 10
    }
    memory_optimized = {
      instance_types = ["m5.4xlarge", "m5.8xlarge"]
      min_size       = 3
      max_size       = 30
      desired_size   = 6
    }
    gpu_enabled = {
      instance_types = ["p3.2xlarge", "p3.8xlarge"]
      min_size       = 1
      max_size       = 20
      desired_size   = 4
    }
  }
  
  # Database configuration
  database_config = var.regional_configs.primary.database
  redis_config    = var.regional_configs.primary.redis
  
  # Regional role
  region_role = "primary"
  
  # Global resources
  global_s3_bucket_data      = aws_s3_bucket.global_data.bucket
  global_s3_bucket_models    = aws_s3_bucket.global_models.bucket
  global_s3_bucket_artifacts = aws_s3_bucket.global_artifacts.bucket
  
  tags = local.common_tags
}

module "secondary_region" {
  source = "../modules/regional-cluster"
  
  providers = {
    aws = aws.secondary
  }
  
  region             = var.secondary_region
  environment        = var.environment
  cluster_name       = local.cluster_names.secondary
  vpc_cidr           = var.regional_configs.secondary.vpc_cidr
  availability_zones = slice(data.aws_availability_zones.secondary.names, 0, 3)
  
  # Scaling configuration for secondary region
  node_groups = {
    cpu_optimized = {
      instance_types = ["c5.2xlarge", "c5.4xlarge"]
      min_size       = 3
      max_size       = 30
      desired_size   = 6
    }
    memory_optimized = {
      instance_types = ["m5.4xlarge", "m5.8xlarge"]
      min_size       = 2
      max_size       = 20
      desired_size   = 4
    }
    gpu_enabled = {
      instance_types = ["p3.2xlarge"]
      min_size       = 0
      max_size       = 10
      desired_size   = 2
    }
  }
  
  # Database configuration
  database_config = var.regional_configs.secondary.database
  redis_config    = var.regional_configs.secondary.redis
  
  # Regional role
  region_role = "secondary"
  
  # Global resources
  global_s3_bucket_data      = aws_s3_bucket.global_data.bucket
  global_s3_bucket_models    = aws_s3_bucket.global_models.bucket
  global_s3_bucket_artifacts = aws_s3_bucket.global_artifacts.bucket
  
  tags = local.common_tags
}

module "tertiary_region" {
  source = "../modules/regional-cluster"
  
  providers = {
    aws = aws.tertiary
  }
  
  region             = var.tertiary_region
  environment        = var.environment
  cluster_name       = local.cluster_names.tertiary
  vpc_cidr           = var.regional_configs.tertiary.vpc_cidr
  availability_zones = slice(data.aws_availability_zones.tertiary.names, 0, 3)
  
  # Scaling configuration for tertiary region (disaster recovery)
  node_groups = {
    cpu_optimized = {
      instance_types = ["c5.large", "c5.xlarge"]
      min_size       = 2
      max_size       = 20
      desired_size   = 3
    }
    memory_optimized = {
      instance_types = ["m5.2xlarge", "m5.4xlarge"]
      min_size       = 1
      max_size       = 15
      desired_size   = 2
    }
    gpu_enabled = {
      instance_types = ["p3.2xlarge"]
      min_size       = 0
      max_size       = 5
      desired_size   = 1
    }
  }
  
  # Database configuration
  database_config = var.regional_configs.tertiary.database
  redis_config    = var.regional_configs.tertiary.redis
  
  # Regional role
  region_role = "disaster_recovery"
  
  # Global resources
  global_s3_bucket_data      = aws_s3_bucket.global_data.bucket
  global_s3_bucket_models    = aws_s3_bucket.global_models.bucket
  global_s3_bucket_artifacts = aws_s3_bucket.global_artifacts.bucket
  
  tags = local.common_tags
}

# Global S3 Buckets with Cross-Region Replication
resource "aws_s3_bucket" "global_data" {
  provider = aws.primary
  bucket   = "${var.environment}-mlops-global-data-${random_id.global_suffix.hex}"
  
  tags = merge(local.common_tags, {
    Name = "MLOps Global Data Bucket"
    Type = "global-data-storage"
  })
}

resource "aws_s3_bucket" "global_models" {
  provider = aws.primary
  bucket   = "${var.environment}-mlops-global-models-${random_id.global_suffix.hex}"
  
  tags = merge(local.common_tags, {
    Name = "MLOps Global Models Bucket"
    Type = "global-model-storage"
  })
}

resource "aws_s3_bucket" "global_artifacts" {
  provider = aws.primary
  bucket   = "${var.environment}-mlops-global-artifacts-${random_id.global_suffix.hex}"
  
  tags = merge(local.common_tags, {
    Name = "MLOps Global Artifacts Bucket"
    Type = "global-artifact-storage"
  })
}

resource "random_id" "global_suffix" {
  byte_length = 4
}

# Cross-Region Replication for global buckets
resource "aws_s3_bucket_replication_configuration" "global_data_replication" {
  provider   = aws.primary
  depends_on = [aws_s3_bucket_versioning.global_data]
  
  role   = aws_iam_role.replication.arn
  bucket = aws_s3_bucket.global_data.id
  
  rule {
    id     = "replicate-to-secondary"
    status = "Enabled"
    
    destination {
      bucket        = aws_s3_bucket.global_data_replica_secondary.arn
      storage_class = "STANDARD_IA"
    }
  }
  
  rule {
    id     = "replicate-to-tertiary"
    status = "Enabled"
    
    destination {
      bucket        = aws_s3_bucket.global_data_replica_tertiary.arn
      storage_class = "GLACIER"
    }
  }
}

# S3 bucket replicas in other regions
resource "aws_s3_bucket" "global_data_replica_secondary" {
  provider = aws.secondary
  bucket   = "${var.environment}-mlops-global-data-replica-${var.secondary_region}-${random_id.global_suffix.hex}"
  
  tags = merge(local.common_tags, {
    Name = "MLOps Global Data Replica (Secondary)"
    Type = "global-data-replica"
  })
}

resource "aws_s3_bucket" "global_data_replica_tertiary" {
  provider = aws.tertiary
  bucket   = "${var.environment}-mlops-global-data-replica-${var.tertiary_region}-${random_id.global_suffix.hex}"
  
  tags = merge(local.common_tags, {
    Name = "MLOps Global Data Replica (Tertiary)"
    Type = "global-data-replica"
  })
}

# S3 bucket configurations
resource "aws_s3_bucket_versioning" "global_data" {
  provider = aws.primary
  bucket   = aws_s3_bucket.global_data.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "global_models" {
  provider = aws.primary
  bucket   = aws_s3_bucket.global_models.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

# IAM role for S3 replication
resource "aws_iam_role" "replication" {
  provider = aws.primary
  name     = "${var.environment}-s3-replication-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
      },
    ]
  })
  
  tags = local.common_tags
}

resource "aws_iam_policy" "replication" {
  provider = aws.primary
  name     = "${var.environment}-s3-replication-policy"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetObjectVersionForReplication",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersionTagging"
        ]
        Effect = "Allow"
        Resource = [
          "${aws_s3_bucket.global_data.arn}/*",
          "${aws_s3_bucket.global_models.arn}/*"
        ]
      },
      {
        Action = [
          "s3:ListBucket"
        ]
        Effect = "Allow"
        Resource = [
          aws_s3_bucket.global_data.arn,
          aws_s3_bucket.global_models.arn
        ]
      },
      {
        Action = [
          "s3:ReplicateObject",
          "s3:ReplicateDelete",
          "s3:ReplicateTags"
        ]
        Effect = "Allow"
        Resource = [
          "${aws_s3_bucket.global_data_replica_secondary.arn}/*",
          "${aws_s3_bucket.global_data_replica_tertiary.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "replication" {
  provider   = aws.primary
  role       = aws_iam_role.replication.name
  policy_arn = aws_iam_policy.replication.arn
}

# Global Load Balancers for regional clusters
resource "aws_lb" "primary_global_alb" {
  provider = aws.primary
  
  name               = "${var.environment}-mlops-global-primary-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [module.primary_region.alb_security_group_id]
  subnets           = module.primary_region.public_subnet_ids
  
  enable_deletion_protection = true
  enable_cross_zone_load_balancing = true
  
  tags = merge(local.common_tags, {
    Name = "MLOps Global Primary ALB"
    Region = var.primary_region
  })
}

resource "aws_lb" "secondary_global_alb" {
  provider = aws.secondary
  
  name               = "${var.environment}-mlops-global-secondary-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [module.secondary_region.alb_security_group_id]
  subnets           = module.secondary_region.public_subnet_ids
  
  enable_deletion_protection = true
  enable_cross_zone_load_balancing = true
  
  tags = merge(local.common_tags, {
    Name = "MLOps Global Secondary ALB"
    Region = var.secondary_region
  })
}

resource "aws_lb" "tertiary_global_alb" {
  provider = aws.tertiary
  
  name               = "${var.environment}-mlops-global-tertiary-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [module.tertiary_region.alb_security_group_id]
  subnets           = module.tertiary_region.public_subnet_ids
  
  enable_deletion_protection = true
  enable_cross_zone_load_balancing = true
  
  tags = merge(local.common_tags, {
    Name = "MLOps Global Tertiary ALB"
    Region = var.tertiary_region
  })
}

# Global Database Cross-Region Setup
resource "aws_rds_global_cluster" "mlops_global" {
  provider = aws.primary
  
  cluster_identifier      = "${var.environment}-mlops-global-cluster"
  engine                 = "aurora-postgresql"
  engine_version         = "15.4"
  database_name          = "mlops_global"
  master_username        = "mlops_global_admin"
  master_password        = var.global_database_password
  backup_retention_period = 7
  preferred_backup_window = "03:00-04:00"
  
  tags = merge(local.common_tags, {
    Name = "MLOps Global Database Cluster"
  })
}

# Route53 Health Checks and Failover
resource "aws_route53_health_check" "primary" {
  provider                        = aws.primary
  fqdn                           = aws_lb.primary_global_alb.dns_name
  port                           = 443
  type                           = "HTTPS"
  resource_path                  = "/health"
  failure_threshold              = "3"
  request_interval               = "30"
  cloudwatch_logs_region         = var.primary_region
  cloudwatch_alarm_region        = var.primary_region
  insufficient_data_health_status = "Failure"
  
  tags = merge(local.common_tags, {
    Name = "Primary Region Health Check"
  })
}

resource "aws_route53_health_check" "secondary" {
  provider                        = aws.primary
  fqdn                           = aws_lb.secondary_global_alb.dns_name
  port                           = 443
  type                           = "HTTPS"
  resource_path                  = "/health"
  failure_threshold              = "3"
  request_interval               = "30"
  cloudwatch_logs_region         = var.secondary_region
  cloudwatch_alarm_region        = var.secondary_region
  insufficient_data_health_status = "Failure"
  
  tags = merge(local.common_tags, {
    Name = "Secondary Region Health Check"
  })
}

# DNS Records with Failover Routing
resource "aws_route53_record" "primary" {
  provider = aws.primary
  zone_id  = aws_route53_zone.global.zone_id
  name     = "api.${var.domain_name}"
  type     = "A"
  
  set_identifier = "primary"
  
  failover_routing_policy {
    type = "PRIMARY"
  }
  
  health_check_id = aws_route53_health_check.primary.id
  
  alias {
    name                   = aws_lb.primary_global_alb.dns_name
    zone_id                = aws_lb.primary_global_alb.zone_id
    evaluate_target_health = true
  }
}

resource "aws_route53_record" "secondary" {
  provider = aws.primary
  zone_id  = aws_route53_zone.global.zone_id
  name     = "api.${var.domain_name}"
  type     = "A"
  
  set_identifier = "secondary"
  
  failover_routing_policy {
    type = "SECONDARY"
  }
  
  health_check_id = aws_route53_health_check.secondary.id
  
  alias {
    name                   = aws_lb.secondary_global_alb.dns_name
    zone_id                = aws_lb.secondary_global_alb.zone_id
    evaluate_target_health = true
  }
}

# Global monitoring and alerting
resource "aws_cloudwatch_dashboard" "global" {
  provider       = aws.primary
  dashboard_name = "${var.environment}-mlops-global-dashboard"
  
  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        
        properties = {
          metrics = [
            ["AWS/ApplicationELB", "RequestCount", "LoadBalancer", aws_lb.primary_global_alb.arn_suffix],
            ["AWS/ApplicationELB", "RequestCount", "LoadBalancer", aws_lb.secondary_global_alb.arn_suffix],
            ["AWS/ApplicationELB", "RequestCount", "LoadBalancer", aws_lb.tertiary_global_alb.arn_suffix]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.primary_region
          title   = "Global Request Volume"
          period  = 300
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6
        
        properties = {
          metrics = [
            ["AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", aws_lb.primary_global_alb.arn_suffix],
            ["AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", aws_lb.secondary_global_alb.arn_suffix],
            ["AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", aws_lb.tertiary_global_alb.arn_suffix]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.primary_region
          title   = "Global Response Times"
          period  = 300
        }
      }
    ]
  })
}