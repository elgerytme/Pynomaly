# Multi-Cloud Infrastructure for MLOps Platform
# This Terraform configuration supports deployment across AWS, GCP, and Azure

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
  
  backend "s3" {
    bucket = "mlops-terraform-state"
    key    = "multi-cloud/terraform.tfstate"
    region = "us-east-1"
  }
}

# Variables
variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "mlops-platform"
}

variable "enable_aws" {
  description = "Enable AWS deployment"
  type        = bool
  default     = true
}

variable "enable_gcp" {
  description = "Enable GCP deployment"
  type        = bool
  default     = true
}

variable "enable_azure" {
  description = "Enable Azure deployment"
  type        = bool
  default     = false
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "gcp_region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "azure_location" {
  description = "Azure location"
  type        = string
  default     = "East US"
}

variable "node_count" {
  description = "Number of nodes per cluster"
  type        = number
  default     = 3
}

variable "node_instance_type" {
  description = "Instance type for worker nodes"
  type = object({
    aws   = string
    gcp   = string
    azure = string
  })
  default = {
    aws   = "t3.large"
    gcp   = "e2-standard-4"
    azure = "Standard_D4s_v3"
  }
}

# Local values
locals {
  common_tags = {
    Environment = var.environment
    Project     = var.project_name
    ManagedBy   = "terraform"
    CreatedAt   = timestamp()
  }
  
  cluster_name = "${var.project_name}-${var.environment}"
}

# Data sources
data "aws_availability_zones" "available" {
  count = var.enable_aws ? 1 : 0
  state = "available"
}

data "google_client_config" "default" {
  count = var.enable_gcp ? 1 : 0
}

data "azurerm_client_config" "current" {
  count = var.enable_azure ? 1 : 0
}

# Provider configurations
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = local.common_tags
  }
}

provider "google" {
  region = var.gcp_region
}

provider "azurerm" {
  features {}
}

# AWS Infrastructure
module "aws_infrastructure" {
  count  = var.enable_aws ? 1 : 0
  source = "./modules/aws"
  
  environment         = var.environment
  project_name        = var.project_name
  region              = var.aws_region
  availability_zones  = data.aws_availability_zones.available[0].names
  cluster_name        = "${local.cluster_name}-aws"
  node_count          = var.node_count
  node_instance_type  = var.node_instance_type.aws
  
  tags = local.common_tags
}

# GCP Infrastructure
module "gcp_infrastructure" {
  count  = var.enable_gcp ? 1 : 0
  source = "./modules/gcp"
  
  environment        = var.environment
  project_name       = var.project_name
  region             = var.gcp_region
  cluster_name       = "${local.cluster_name}-gcp"
  node_count         = var.node_count
  node_instance_type = var.node_instance_type.gcp
  
  labels = {
    environment = var.environment
    project     = var.project_name
    managed-by  = "terraform"
  }
}

# Azure Infrastructure
module "azure_infrastructure" {
  count  = var.enable_azure ? 1 : 0
  source = "./modules/azure"
  
  environment        = var.environment
  project_name       = var.project_name
  location           = var.azure_location
  cluster_name       = "${local.cluster_name}-azure"
  node_count         = var.node_count
  node_instance_type = var.node_instance_type.azure
  
  tags = local.common_tags
}

# Global DNS and Load Balancing
resource "aws_route53_zone" "main" {
  count = var.enable_aws ? 1 : 0
  name  = "mlops-platform.com"
  
  tags = merge(local.common_tags, {
    Name = "MLOps Platform DNS Zone"
  })
}

# Multi-cloud service mesh configuration
resource "helm_release" "istio_multicluster" {
  for_each = toset([
    for cluster in [
      var.enable_aws ? "aws" : null,
      var.enable_gcp ? "gcp" : null,
      var.enable_azure ? "azure" : null
    ] : cluster if cluster != null
  ])
  
  name       = "istio-multicluster-${each.key}"
  repository = "https://istio-release.storage.googleapis.com/charts"
  chart      = "istiod"
  namespace  = "istio-system"
  
  create_namespace = true
  
  values = [
    templatefile("${path.module}/templates/istio-multicluster-values.yaml", {
      cluster_name = "${local.cluster_name}-${each.key}"
      cloud_provider = each.key
    })
  ]
  
  depends_on = [
    module.aws_infrastructure,
    module.gcp_infrastructure,
    module.azure_infrastructure
  ]
}

# Outputs
output "aws_cluster_endpoint" {
  description = "AWS EKS cluster endpoint"
  value       = var.enable_aws ? module.aws_infrastructure[0].cluster_endpoint : null
}

output "gcp_cluster_endpoint" {
  description = "GCP GKE cluster endpoint"
  value       = var.enable_gcp ? module.gcp_infrastructure[0].cluster_endpoint : null
}

output "azure_cluster_endpoint" {
  description = "Azure AKS cluster endpoint"
  value       = var.enable_azure ? module.azure_infrastructure[0].cluster_endpoint : null
}

output "dns_zone_id" {
  description = "Route53 DNS zone ID"
  value       = var.enable_aws ? aws_route53_zone.main[0].zone_id : null
}

output "cluster_kubeconfig_commands" {
  description = "Commands to configure kubectl for each cluster"
  value = {
    aws = var.enable_aws ? "aws eks update-kubeconfig --region ${var.aws_region} --name ${local.cluster_name}-aws" : null
    gcp = var.enable_gcp ? "gcloud container clusters get-credentials ${local.cluster_name}-gcp --region ${var.gcp_region}" : null
    azure = var.enable_azure ? "az aks get-credentials --resource-group ${var.project_name}-${var.environment} --name ${local.cluster_name}-azure" : null
  }
}