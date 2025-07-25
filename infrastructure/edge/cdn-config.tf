# Global CDN and Edge Computing Infrastructure
# Terraform configuration for CloudFlare, AWS CloudFront, and Azure CDN

terraform {
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.0"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

# CloudFlare Configuration
variable "cloudflare_zone_id" {
  description = "CloudFlare Zone ID"
  type        = string
  sensitive   = true
}

variable "cloudflare_api_token" {
  description = "CloudFlare API Token"
  type        = string
  sensitive   = true
}

# CloudFlare DNS and CDN
resource "cloudflare_record" "api_endpoint" {
  zone_id = var.cloudflare_zone_id
  name    = "api"
  value   = aws_lb.main.dns_name
  type    = "CNAME"
  ttl     = 300
  proxied = true
}

resource "cloudflare_page_rule" "api_cache" {
  zone_id  = var.cloudflare_zone_id
  target   = "api.${data.cloudflare_zone.main.name}/*"
  priority = 1

  actions {
    cache_level                = "cache_everything"
    edge_cache_ttl            = 7200
    browser_cache_ttl         = 3600
    always_online             = "on"
    ssl                       = "full"
    security_level           = "medium"
    rocket_loader            = "on"
    minify {
      css  = "on"
      js   = "on"
      html = "on"
    }
  }
}

# AWS CloudFront Distribution
resource "aws_cloudfront_distribution" "main" {
  origin {
    domain_name              = aws_lb.main.dns_name
    origin_id               = "MLOps-API-Origin"
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  enabled             = true
  is_ipv6_enabled     = true
  comment             = "MLOps Platform CDN"
  default_root_object = "index.html"

  default_cache_behavior {
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "MLOps-API-Origin"
    compress               = true
    viewer_protocol_policy = "redirect-to-https"

    forwarded_values {
      query_string = true
      headers      = ["Authorization", "Content-Type", "X-API-Key"]
      cookies {
        forward = "none"
      }
    }

    min_ttl     = 0
    default_ttl = 3600
    max_ttl     = 86400
  }

  # Cache behavior for API endpoints
  ordered_cache_behavior {
    path_pattern           = "/api/*"
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD", "OPTIONS"]
    target_origin_id       = "MLOps-API-Origin"
    compress               = true
    viewer_protocol_policy = "https-only"

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

  # Cache behavior for static assets
  ordered_cache_behavior {
    path_pattern           = "/static/*"
    allowed_methods        = ["GET", "HEAD"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "MLOps-API-Origin"
    compress               = true
    viewer_protocol_policy = "https-only"

    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }

    min_ttl     = 86400
    default_ttl = 86400
    max_ttl     = 31536000
  }

  price_class = "PriceClass_All"

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate.main.arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  web_acl_id = aws_wafv2_web_acl.main.arn

  tags = {
    Name        = "MLOps-Platform-CDN"
    Environment = "production"
    Project     = "MLOps"
  }
}

# AWS WAF for CDN Protection
resource "aws_wafv2_web_acl" "main" {
  name  = "mlops-cdn-waf"
  scope = "CLOUDFRONT"

  default_action {
    allow {}
  }

  rule {
    name     = "AWS-AWSManagedRulesCommonRuleSet"
    priority = 1

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "CommonRuleSetMetric"
      sampled_requests_enabled   = true
    }
  }

  rule {
    name     = "AWS-AWSManagedRulesKnownBadInputsRuleSet"
    priority = 2

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesKnownBadInputsRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "KnownBadInputsRuleSetMetric"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "MLOpsCDNWAF"
    sampled_requests_enabled   = true
  }

  tags = {
    Name        = "MLOps-CDN-WAF"
    Environment = "production"
  }
}

# Azure CDN Profile and Endpoint
resource "azurerm_cdn_profile" "main" {
  name                = "mlops-cdn-profile"
  location            = "global"
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "Standard_Microsoft"

  tags = {
    Environment = "production"
    Project     = "MLOps"
  }
}

resource "azurerm_cdn_endpoint" "main" {
  name                = "mlops-cdn-endpoint"
  profile_name        = azurerm_cdn_profile.main.name
  location            = azurerm_cdn_profile.main.location
  resource_group_name = azurerm_resource_group.main.name

  origin {
    name      = "mlops-origin"
    host_name = azurerm_kubernetes_cluster.main.fqdn
  }

  delivery_rule {
    name  = "httpsRedirect"
    order = 1

    request_scheme_condition {
      operator     = "Equal"
      match_values = ["HTTP"]
    }

    url_redirect_action {
      redirect_type = "PermanentRedirect"
      protocol      = "Https"
    }
  }

  delivery_rule {
    name  = "cacheOptimization"
    order = 2

    url_path_condition {
      operator     = "BeginsWith"
      match_values = ["/static/"]
    }

    cache_expiration_action {
      behavior = "Override"
      duration = "1.00:00:00"
    }
  }

  global_delivery_rule {
    cache_expiration_action {
      behavior = "Override"
      duration = "04:00:00"
    }

    cache_key_query_string_action {
      behavior   = "IncludeAll"
      parameters = "version,locale"
    }
  }

  tags = {
    Environment = "production"
    Project     = "MLOps"
  }
}

# Edge Computing Lambda Functions
resource "aws_lambda_function" "edge_analytics" {
  filename         = "edge_analytics.zip"
  function_name    = "mlops-edge-analytics"
  role            = aws_iam_role.lambda_edge.arn
  handler         = "index.handler"
  source_code_hash = data.archive_file.edge_analytics.output_base64sha256
  runtime         = "python3.9"
  timeout         = 30

  environment {
    variables = {
      ANALYTICS_ENDPOINT = var.analytics_endpoint
      API_KEY           = var.edge_api_key
    }
  }

  tags = {
    Name        = "MLOps-Edge-Analytics"
    Environment = "production"
  }
}

# CloudFlare Workers for Edge Computing
resource "cloudflare_worker_script" "edge_processor" {
  name    = "mlops-edge-processor"
  content = file("${path.module}/workers/edge-processor.js")
}

resource "cloudflare_worker_route" "edge_processor" {
  zone_id     = var.cloudflare_zone_id
  pattern     = "api.${data.cloudflare_zone.main.name}/edge/*"
  script_name = cloudflare_worker_script.edge_processor.name
}

# Outputs
output "cloudfront_distribution_id" {
  value = aws_cloudfront_distribution.main.id
}

output "cloudfront_domain_name" {
  value = aws_cloudfront_distribution.main.domain_name
}

output "azure_cdn_endpoint_fqdn" {
  value = azurerm_cdn_endpoint.main.fqdn
}

output "cloudflare_worker_url" {
  value = "https://api.${data.cloudflare_zone.main.name}/edge/*"
}