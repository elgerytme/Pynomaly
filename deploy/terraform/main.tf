# Terraform Configuration for Anomaly Detection Platform
# Multi-environment infrastructure setup with Kubernetes clusters

terraform {
  required_version = ">= 1.0"
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.4"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }

  backend "s3" {
    bucket = "anomaly-detection-terraform-state"
    key    = "infrastructure/terraform.tfstate"
    region = "us-east-1"
  }
}

# Variables
variable "environment" {
  description = "Environment name (staging, production)"
  type        = string
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "anomaly-detection"
}

variable "kubernetes_cluster_name" {
  description = "Kubernetes cluster name"
  type        = string
}

variable "kubernetes_cluster_endpoint" {
  description = "Kubernetes cluster API endpoint"
  type        = string
}

variable "kubernetes_cluster_ca_certificate" {
  description = "Kubernetes cluster CA certificate"
  type        = string
}

variable "kubernetes_token" {
  description = "Kubernetes authentication token"
  type        = string
  sensitive   = true
}

variable "domain_name" {
  description = "Base domain name"
  type        = string
  default     = "anomaly-detection.io"
}

variable "enable_monitoring" {
  description = "Enable monitoring stack"
  type        = bool
  default     = true
}

variable "enable_security_scanning" {
  description = "Enable security scanning"
  type        = bool
  default     = true
}

# Local values
locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    CreatedBy   = "anomaly-detection-platform"
  }

  namespace_name = "${var.project_name}-${var.environment}"
  
  domain_map = {
    staging    = "staging.${var.domain_name}"
    production = var.domain_name
  }
}

# Kubernetes Provider
provider "kubernetes" {
  host                   = var.kubernetes_cluster_endpoint
  cluster_ca_certificate = base64decode(var.kubernetes_cluster_ca_certificate)
  token                  = var.kubernetes_token
}

provider "helm" {
  kubernetes {
    host                   = var.kubernetes_cluster_endpoint
    cluster_ca_certificate = base64decode(var.kubernetes_cluster_ca_certificate)
    token                  = var.kubernetes_token
  }
}

# Random passwords for databases
resource "random_password" "postgresql_password" {
  length  = 32
  special = true
}

resource "random_password" "redis_password" {
  length  = 32
  special = true
}

# Create namespace
resource "kubernetes_namespace" "main" {
  metadata {
    name = local.namespace_name
    labels = merge(local.common_tags, {
      "app.kubernetes.io/name"     = var.project_name
      "app.kubernetes.io/instance" = var.environment
    })
  }
}

# Monitoring namespace
resource "kubernetes_namespace" "monitoring" {
  count = var.enable_monitoring ? 1 : 0
  
  metadata {
    name = "monitoring"
    labels = merge(local.common_tags, {
      "app.kubernetes.io/name"     = "monitoring"
      "app.kubernetes.io/instance" = var.environment
    })
  }
}

# Create secrets
resource "kubernetes_secret" "database_credentials" {
  metadata {
    name      = "${var.project_name}-database-credentials"
    namespace = kubernetes_namespace.main.metadata[0].name
  }

  type = "Opaque"

  data = {
    username = "anomaly_detection"
    password = random_password.postgresql_password.result
    database = "anomaly_detection_${var.environment}"
  }
}

resource "kubernetes_secret" "redis_credentials" {
  metadata {
    name      = "${var.project_name}-redis-credentials"
    namespace = kubernetes_namespace.main.metadata[0].name
  }

  type = "Opaque"

  data = {
    password = random_password.redis_password.result
  }
}

# Application configuration
resource "kubernetes_config_map" "app_config" {
  metadata {
    name      = "${var.project_name}-config"
    namespace = kubernetes_namespace.main.metadata[0].name
  }

  data = {
    "config.yaml" = yamlencode({
      environment = var.environment
      server = {
        host = "0.0.0.0"
        port = 8000
      }
      logging = {
        level  = var.environment == "production" ? "INFO" : "DEBUG"
        format = "json"
      }
      database = {
        host     = "${var.project_name}-postgresql"
        port     = 5432
        name     = "anomaly_detection_${var.environment}"
        ssl_mode = "require"
      }
      redis = {
        host = "${var.project_name}-redis-master"
        port = 6379
        db   = 0
      }
      monitoring = {
        enabled           = var.enable_monitoring
        metrics_port      = 9090
        health_check_path = "/api/health/ready"
      }
      security = {
        cors_origins = var.environment == "production" ? [
          "https://${local.domain_map[var.environment]}"
        ] : ["*"]
        rate_limiting = {
          enabled = true
          requests_per_minute = var.environment == "production" ? 100 : 1000
        }
      }
    })
  }
}

# Deploy PostgreSQL using Helm
resource "helm_release" "postgresql" {
  name       = "${var.project_name}-postgresql"
  repository = "https://charts.bitnami.com/bitnami"
  chart      = "postgresql"
  version    = "12.12.10"
  namespace  = kubernetes_namespace.main.metadata[0].name

  values = [
    yamlencode({
      auth = {
        postgresPassword = random_password.postgresql_password.result
        database         = "anomaly_detection_${var.environment}"
      }
      primary = {
        persistence = {
          enabled = true
          size    = var.environment == "production" ? "100Gi" : "20Gi"
        }
        resources = var.environment == "production" ? {
          requests = {
            memory = "1Gi"
            cpu    = "500m"
          }
          limits = {
            memory = "2Gi"
            cpu    = "1000m"
          }
        } : {
          requests = {
            memory = "256Mi"
            cpu    = "250m"
          }
          limits = {
            memory = "512Mi"
            cpu    = "500m"
          }
        }
      }
      metrics = {
        enabled = var.enable_monitoring
        serviceMonitor = {
          enabled = var.enable_monitoring
        }
      }
    })
  ]

  depends_on = [kubernetes_namespace.main]
}

# Deploy Redis using Helm
resource "helm_release" "redis" {
  name       = "${var.project_name}-redis"
  repository = "https://charts.bitnami.com/bitnami"
  chart      = "redis"
  version    = "18.4.0"
  namespace  = kubernetes_namespace.main.metadata[0].name

  values = [
    yamlencode({
      auth = {
        enabled  = true
        password = random_password.redis_password.result
      }
      master = {
        persistence = {
          enabled = true
          size    = var.environment == "production" ? "20Gi" : "8Gi"
        }
        resources = var.environment == "production" ? {
          requests = {
            memory = "512Mi"
            cpu    = "250m"
          }
          limits = {
            memory = "1Gi"
            cpu    = "500m"
          }
        } : {
          requests = {
            memory = "128Mi"
            cpu    = "100m"
          }
          limits = {
            memory = "256Mi"
            cpu    = "200m"
          }
        }
      }
      metrics = {
        enabled = var.enable_monitoring
        serviceMonitor = {
          enabled = var.enable_monitoring
        }
      }
    })
  ]

  depends_on = [kubernetes_namespace.main]
}

# Deploy monitoring stack (Prometheus + Grafana)
resource "helm_release" "kube_prometheus_stack" {
  count = var.enable_monitoring ? 1 : 0

  name       = "kube-prometheus-stack"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  version    = "55.5.0"
  namespace  = kubernetes_namespace.monitoring[0].metadata[0].name

  values = [
    yamlencode({
      prometheus = {
        prometheusSpec = {
          retention = "30d"
          storageSpec = {
            volumeClaimTemplate = {
              spec = {
                storageClassName = var.environment == "production" ? "ssd" : "standard"
                accessModes      = ["ReadWriteOnce"]
                resources = {
                  requests = {
                    storage = var.environment == "production" ? "100Gi" : "50Gi"
                  }
                }
              }
            }
          }
          resources = var.environment == "production" ? {
            requests = {
              memory = "2Gi"
              cpu    = "1000m"
            }
            limits = {
              memory = "4Gi"
              cpu    = "2000m"
            }
          } : {
            requests = {
              memory = "1Gi"
              cpu    = "500m"
            }
            limits = {
              memory = "2Gi"
              cpu    = "1000m"
            }
          }
        }
      }
      grafana = {
        adminPassword = random_password.postgresql_password.result # Reuse for simplicity
        persistence = {
          enabled = true
          size    = "10Gi"
        }
        dashboardProviders = {
          "dashboardproviders.yaml" = {
            apiVersion = 1
            providers = [
              {
                name    = "default"
                orgId   = 1
                folder  = ""
                type    = "file"
                options = {
                  path = "/var/lib/grafana/dashboards/default"
                }
              }
            ]
          }
        }
        dashboards = {
          default = {
            "anomaly-detection-overview" = {
              gnetId    = null
              revision  = null
              datasource = "Prometheus"
            }
          }
        }
        ingress = {
          enabled = true
          hosts = [
            "grafana.${local.domain_map[var.environment]}"
          ]
          tls = [
            {
              secretName = "grafana-tls"
              hosts      = ["grafana.${local.domain_map[var.environment]}"]
            }
          ]
        }
      }
      alertmanager = {
        alertmanagerSpec = {
          storage = {
            volumeClaimTemplate = {
              spec = {
                storageClassName = var.environment == "production" ? "ssd" : "standard"
                accessModes      = ["ReadWriteOnce"]
                resources = {
                  requests = {
                    storage = "10Gi"
                  }
                }
              }
            }
          }
        }
      }
    })
  ]

  depends_on = [kubernetes_namespace.monitoring]
}

# Deploy the main application using our Helm chart
resource "helm_release" "anomaly_detection" {
  name      = var.project_name
  chart     = "../../helm"
  namespace = kubernetes_namespace.main.metadata[0].name

  values = [
    file("../../helm/values-${var.environment}.yaml")
  ]

  set_sensitive {
    name  = "secrets.data.database-url"
    value = base64encode("postgresql://anomaly_detection:${random_password.postgresql_password.result}@${var.project_name}-postgresql:5432/anomaly_detection_${var.environment}")
  }

  set_sensitive {
    name  = "secrets.data.redis-url"
    value = base64encode("redis://:${random_password.redis_password.result}@${var.project_name}-redis-master:6379/0")
  }

  set {
    name  = "ingress.hosts[0].host"
    value = local.domain_map[var.environment]
  }

  set {
    name  = "monitoring.enabled"
    value = var.enable_monitoring
  }

  depends_on = [
    helm_release.postgresql,
    helm_release.redis,
    kubernetes_config_map.app_config
  ]
}

# Outputs
output "namespace" {
  description = "Kubernetes namespace"
  value       = kubernetes_namespace.main.metadata[0].name
}

output "application_url" {
  description = "Application URL"
  value       = "https://${local.domain_map[var.environment]}"
}

output "grafana_url" {
  description = "Grafana dashboard URL"
  value       = var.enable_monitoring ? "https://grafana.${local.domain_map[var.environment]}" : null
}

output "database_secret_name" {
  description = "Database credentials secret name"
  value       = kubernetes_secret.database_credentials.metadata[0].name
}

output "redis_secret_name" {
  description = "Redis credentials secret name"
  value       = kubernetes_secret.redis_credentials.metadata[0].name
}