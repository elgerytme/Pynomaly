# Terraform configuration for Pynomaly Detection AWS infrastructure
# Creates EKS cluster, S3 storage, and CloudWatch monitoring

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }

  backend "s3" {
    # Configure S3 backend for state management
    # bucket = "your-terraform-state-bucket"
    # key    = "pynomaly-detection/terraform.tfstate"
    # region = "us-east-1"
  }
}

# Configure AWS Provider
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = local.common_tags
  }
}

# Configure Kubernetes Provider
provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

# Configure Helm Provider
provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
    }
  }
}

# Local variables
locals {
  name = var.cluster_name
  
  common_tags = {
    Application   = "Pynomaly-Detection"
    Environment   = var.environment
    Project       = "Pynomaly"
    ManagedBy     = "Terraform"
    CreatedAt     = formatdate("YYYY-MM-DD", timestamp())
  }
  
  vpc_cidr = "10.0.0.0/16"
  azs      = slice(data.aws_availability_zones.available.names, 0, 3)
  
  # EKS cluster configuration
  cluster_version = var.kubernetes_version
  
  # Node group configuration
  node_groups = {
    pynomaly_nodes = {
      min_size     = var.min_nodes
      max_size     = var.max_nodes
      desired_size = var.desired_nodes
      
      instance_types = var.instance_types
      capacity_type  = "ON_DEMAND"
      
      k8s_labels = {
        Application = "pynomaly-detection"
        NodeGroup   = "pynomaly-nodes"
      }
      
      update_config = {
        max_unavailable_percentage = 33
      }
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

data "aws_caller_identity" "current" {}

# Random suffix for unique resource names
resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# ================================
# NETWORKING
# ================================

module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${local.name}-vpc"
  cidr = local.vpc_cidr

  azs             = local.azs
  private_subnets = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 4, k)]
  public_subnets  = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 8, k + 48)]

  enable_nat_gateway = true
  single_nat_gateway = var.single_nat_gateway
  enable_dns_hostnames = true
  enable_dns_support   = true

  # Required for EKS
  public_subnet_tags = {
    "kubernetes.io/cluster/${local.name}" = "shared"
    "kubernetes.io/role/elb"             = "1"
  }

  private_subnet_tags = {
    "kubernetes.io/cluster/${local.name}" = "shared"
    "kubernetes.io/role/internal-elb"    = "1"
  }

  tags = local.common_tags
}

# ================================
# EKS CLUSTER
# ================================

module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 19.15"

  cluster_name    = local.name
  cluster_version = local.cluster_version

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true

  # EKS Managed Node Groups
  eks_managed_node_groups = local.node_groups

  # aws-auth configmap
  manage_aws_auth_configmap = true

  aws_auth_roles = [
    {
      rolearn  = aws_iam_role.pynomaly_node_role.arn
      username = "system:node:{{EC2PrivateDNSName}}"
      groups   = ["system:bootstrappers", "system:nodes"]
    },
  ]

  aws_auth_users = var.aws_auth_users

  # Extend cluster security group rules
  cluster_security_group_additional_rules = {
    ingress_nodes_ephemeral_ports_tcp = {
      description                = "Nodes on ephemeral ports"
      protocol                   = "tcp"
      from_port                  = 1025
      to_port                    = 65535
      type                       = "ingress"
      source_node_security_group = true
    }
  }

  # Extend node-to-node security group rules
  node_security_group_additional_rules = {
    ingress_self_all = {
      description = "Node to node all ports/protocols"
      protocol    = "-1"
      from_port   = 0
      to_port     = 0
      type        = "ingress"
      self        = true
    }
  }

  # EKS Add-ons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  tags = local.common_tags
}

# ================================
# IAM ROLES
# ================================

resource "aws_iam_role" "pynomaly_node_role" {
  name = "${local.name}-node-role"

  assume_role_policy = jsonencode({
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
    Version = "2012-10-17"
  })
}

resource "aws_iam_role_policy_attachment" "pynomaly_node_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.pynomaly_node_role.name
}

resource "aws_iam_role_policy_attachment" "pynomaly_cni_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.pynomaly_node_role.name
}

resource "aws_iam_role_policy_attachment" "pynomaly_registry_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.pynomaly_node_role.name
}

# Custom IAM policy for Pynomaly Detection
resource "aws_iam_policy" "pynomaly_policy" {
  name        = "${local.name}-policy"
  description = "IAM policy for Pynomaly Detection"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.pynomaly_storage.arn,
          "${aws_s3_bucket.pynomaly_storage.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:ListMetrics"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "pynomaly_custom_policy" {
  policy_arn = aws_iam_policy.pynomaly_policy.arn
  role       = aws_iam_role.pynomaly_node_role.name
}

# ================================
# S3 STORAGE
# ================================

resource "aws_s3_bucket" "pynomaly_storage" {
  bucket = "${local.name}-storage-${random_string.suffix.result}"
  
  tags = local.common_tags
}

resource "aws_s3_bucket_versioning" "pynomaly_storage" {
  bucket = aws_s3_bucket.pynomaly_storage.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "pynomaly_storage" {
  bucket = aws_s3_bucket.pynomaly_storage.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "pynomaly_storage" {
  bucket = aws_s3_bucket.pynomaly_storage.id

  rule {
    id     = "data_lifecycle"
    status = "Enabled"

    filter {
      prefix = "data/"
    }

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }
  }

  rule {
    id     = "cache_lifecycle"
    status = "Enabled"

    filter {
      prefix = "cache/"
    }

    expiration {
      days = 7
    }
  }
}

# ================================
# CLOUDWATCH MONITORING
# ================================

resource "aws_cloudwatch_log_group" "pynomaly_logs" {
  name              = "/aws/pynomaly/${local.name}"
  retention_in_days = var.log_retention_days

  tags = local.common_tags
}

# CloudWatch Dashboard
resource "aws_cloudwatch_dashboard" "pynomaly_dashboard" {
  dashboard_name = "${local.name}-dashboard"

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
            ["Pynomaly/Detection", "DetectionCount"],
            ["Pynomaly/Detection", "AnomalyCount"],
            ["Pynomaly/Detection", "ErrorCount"]
          ]
          period = 300
          stat   = "Sum"
          region = var.aws_region
          title  = "Detection Metrics"
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["Pynomaly/Detection", "DetectionLatency"],
            ["Pynomaly/Detection", "ProcessingTime"]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "Performance Metrics"
        }
      }
    ]
  })
}

# ================================
# KUBERNETES RESOURCES
# ================================

resource "kubernetes_namespace" "pynomaly" {
  metadata {
    name = var.kubernetes_namespace
    
    labels = {
      name        = var.kubernetes_namespace
      application = "pynomaly-detection"
    }
  }

  depends_on = [module.eks]
}

resource "kubernetes_service_account" "pynomaly" {
  metadata {
    name      = "pynomaly-detection"
    namespace = kubernetes_namespace.pynomaly.metadata[0].name
    
    annotations = {
      "eks.amazonaws.com/role-arn" = aws_iam_role.pynomaly_node_role.arn
    }
  }

  depends_on = [module.eks]
}

# ================================
# HELM CHARTS
# ================================

resource "helm_release" "aws_load_balancer_controller" {
  name       = "aws-load-balancer-controller"
  repository = "https://aws.github.io/eks-charts"
  chart      = "aws-load-balancer-controller"
  namespace  = "kube-system"
  version    = "1.5.4"

  set {
    name  = "clusterName"
    value = module.eks.cluster_name
  }

  set {
    name  = "serviceAccount.create"
    value = "false"
  }

  depends_on = [module.eks]
}

resource "helm_release" "metrics_server" {
  name       = "metrics-server"
  repository = "https://kubernetes-sigs.github.io/metrics-server/"
  chart      = "metrics-server"
  namespace  = "kube-system"
  version    = "3.10.0"

  depends_on = [module.eks]
}

resource "helm_release" "cluster_autoscaler" {
  name       = "cluster-autoscaler"
  repository = "https://kubernetes.github.io/autoscaler"
  chart      = "cluster-autoscaler"
  namespace  = "kube-system"
  version    = "9.29.0"

  set {
    name  = "autoDiscovery.clusterName"
    value = module.eks.cluster_name
  }

  set {
    name  = "awsRegion"
    value = var.aws_region
  }

  depends_on = [module.eks]
}