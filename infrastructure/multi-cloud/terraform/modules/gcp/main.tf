# GCP Infrastructure Module for MLOps Platform

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

# Variables
variable "environment" {
  description = "Environment name"
  type        = string
}

variable "project_name" {
  description = "Project name"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "cluster_name" {
  description = "GKE cluster name"
  type        = string
}

variable "node_count" {
  description = "Number of worker nodes"
  type        = number
}

variable "node_instance_type" {
  description = "GCE instance type for worker nodes"
  type        = string
}

variable "labels" {
  description = "Resource labels"
  type        = map(string)
  default     = {}
}

# Local values
locals {
  network_name = "${var.project_name}-${var.environment}-network"
  subnet_name  = "${var.project_name}-${var.environment}-subnet"
  
  zones = [
    "${var.region}-a",
    "${var.region}-b",
    "${var.region}-c"
  ]
}

# Data sources
data "google_project" "current" {}

# VPC Network
resource "google_compute_network" "main" {
  name                    = local.network_name
  auto_create_subnetworks = false
  routing_mode           = "REGIONAL"
  
  description = "VPC network for ${var.project_name} ${var.environment}"
}

# Subnets
resource "google_compute_subnetwork" "main" {
  name          = local.subnet_name
  ip_cidr_range = "10.0.0.0/16"
  region        = var.region
  network       = google_compute_network.main.id
  
  description = "Main subnet for ${var.project_name} ${var.environment}"
  
  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.1.0.0/16"
  }
  
  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.2.0.0/16"
  }
  
  private_ip_google_access = true
}

# Cloud Router for NAT Gateway
resource "google_compute_router" "main" {
  name    = "${var.project_name}-${var.environment}-router"
  region  = var.region
  network = google_compute_network.main.id
  
  bgp {
    asn = 64514
  }
}

# NAT Gateway
resource "google_compute_router_nat" "main" {
  name                               = "${var.project_name}-${var.environment}-nat"
  router                             = google_compute_router.main.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
  
  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# Firewall Rules
resource "google_compute_firewall" "allow_internal" {
  name    = "${var.project_name}-${var.environment}-allow-internal"
  network = google_compute_network.main.name
  
  description = "Allow internal communication"
  
  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }
  
  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }
  
  allow {
    protocol = "icmp"
  }
  
  source_ranges = ["10.0.0.0/8"]
}

resource "google_compute_firewall" "allow_ssh" {
  name    = "${var.project_name}-${var.environment}-allow-ssh"
  network = google_compute_network.main.name
  
  description = "Allow SSH access"
  
  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
  
  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["allow-ssh"]
}

resource "google_compute_firewall" "allow_https" {
  name    = "${var.project_name}-${var.environment}-allow-https"
  network = google_compute_network.main.name
  
  description = "Allow HTTPS traffic"
  
  allow {
    protocol = "tcp"
    ports    = ["443", "80"]
  }
  
  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["allow-https"]
}

# Service Account for GKE
resource "google_service_account" "gke_cluster" {
  account_id   = "${var.project_name}-${var.environment}-gke"
  display_name = "GKE Cluster Service Account"
  description  = "Service account for GKE cluster nodes"
}

resource "google_project_iam_member" "gke_cluster_roles" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer",
    "roles/storage.objectViewer"
  ])
  
  project = data.google_project.current.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.gke_cluster.email}"
}

# GKE Cluster
resource "google_container_cluster" "main" {
  name     = var.cluster_name
  location = var.region
  
  description = "GKE cluster for ${var.project_name} ${var.environment}"
  
  # Network configuration
  network    = google_compute_network.main.name
  subnetwork = google_compute_subnetwork.main.name
  
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }
  
  # Security configuration
  enable_shielded_nodes = true
  
  network_policy {
    enabled  = true
    provider = "CALICO"
  }
  
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
    
    master_global_access_config {
      enabled = true
    }
  }
  
  # Master auth
  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }
  
  # Workload Identity
  workload_identity_config {
    workload_pool = "${data.google_project.current.project_id}.svc.id.goog"
  }
  
  # Addons
  addons_config {
    http_load_balancing {
      disabled = false
    }
    
    horizontal_pod_autoscaling {
      disabled = false
    }
    
    network_policy_config {
      disabled = false
    }
    
    istio_config {
      disabled = false
      auth     = "AUTH_MUTUAL_TLS"
    }
    
    cloudrun_config {
      disabled = false
    }
  }
  
  # Logging and monitoring
  logging_service    = "logging.googleapis.com/kubernetes"
  monitoring_service = "monitoring.googleapis.com/kubernetes"
  
  # Cluster autoscaling
  cluster_autoscaling {
    enabled = true
    
    resource_limits {
      resource_type = "cpu"
      minimum       = 1
      maximum       = 100
    }
    
    resource_limits {
      resource_type = "memory"
      minimum       = 1
      maximum       = 1000
    }
    
    auto_provisioning_defaults {
      oauth_scopes = [
        "https://www.googleapis.com/auth/devstorage.read_only",
        "https://www.googleapis.com/auth/logging.write",
        "https://www.googleapis.com/auth/monitoring",
        "https://www.googleapis.com/auth/servicecontrol",
        "https://www.googleapis.com/auth/service.management.readonly",
        "https://www.googleapis.com/auth/trace.append"
      ]
      
      service_account = google_service_account.gke_cluster.email
    }
  }
  
  # Database encryption
  database_encryption {
    state    = "ENCRYPTED"
    key_name = google_kms_crypto_key.gke.id
  }
  
  # Maintenance policy
  maintenance_policy {
    recurring_window {
      start_time = "2023-01-01T03:00:00Z"
      end_time   = "2023-01-01T07:00:00Z"
      recurrence = "FREQ=WEEKLY;BYDAY=SA"
    }
  }
  
  # Remove default node pool
  remove_default_node_pool = true
  initial_node_count       = 1
  
  depends_on = [
    google_project_iam_member.gke_cluster_roles
  ]
}

# Node Pool
resource "google_container_node_pool" "main" {
  name       = "${var.cluster_name}-main-pool"
  cluster    = google_container_cluster.main.name
  location   = var.region
  node_count = var.node_count
  
  # Autoscaling
  autoscaling {
    min_node_count = 1
    max_node_count = var.node_count * 2
  }
  
  # Management
  management {
    auto_repair  = true
    auto_upgrade = true
  }
  
  # Node configuration
  node_config {
    preemptible  = false
    machine_type = var.node_instance_type
    disk_type    = "pd-ssd"
    disk_size_gb = 100
    
    # Service account
    service_account = google_service_account.gke_cluster.email
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/devstorage.read_only",
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
      "https://www.googleapis.com/auth/servicecontrol",
      "https://www.googleapis.com/auth/service.management.readonly",
      "https://www.googleapis.com/auth/trace.append"
    ]
    
    # Security
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
    
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
    
    # Network tags
    tags = ["allow-ssh", "allow-https"]
    
    # Labels
    labels = merge(var.labels, {
      node-pool = "main"
    })
    
    # Metadata
    metadata = {
      disable-legacy-endpoints = "true"
    }
  }
  
  depends_on = [google_container_cluster.main]
}

# KMS Key Ring and Key for GKE encryption
resource "google_kms_key_ring" "gke" {
  name     = "${var.project_name}-${var.environment}-gke-keyring"
  location = var.region
}

resource "google_kms_crypto_key" "gke" {
  name     = "${var.project_name}-${var.environment}-gke-key"
  key_ring = google_kms_key_ring.gke.id
  
  rotation_period = "86400s"  # 24 hours
  
  lifecycle {
    prevent_destroy = true
  }
}

# Cloud SQL PostgreSQL Instance
resource "google_sql_database_instance" "main" {
  name             = "${var.project_name}-${var.environment}-db"
  database_version = "POSTGRES_15"
  region           = var.region
  
  settings {
    tier                        = "db-custom-2-4096"
    availability_type          = "REGIONAL"
    disk_type                  = "PD_SSD"
    disk_size                  = 100
    disk_autoresize            = true
    disk_autoresize_limit      = 1000
    
    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      point_in_time_recovery_enabled = true
      backup_retention_settings {
        retained_backups = 7
      }
    }
    
    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.main.id
      require_ssl     = true
    }
    
    database_flags {
      name  = "log_checkpoints"
      value = "on"
    }
    
    database_flags {
      name  = "log_connections"
      value = "on"
    }
    
    database_flags {
      name  = "log_disconnections"
      value = "on"
    }
    
    maintenance_window {
      day          = 7  # Sunday
      hour         = 3
      update_track = "stable"
    }
    
    insights_config {
      query_insights_enabled  = true
      query_string_length     = 1024
      record_application_tags = true
      record_client_address   = true
    }
  }
  
  deletion_protection = false
  
  depends_on = [google_service_networking_connection.private_vpc_connection]
}

# Private VPC connection for Cloud SQL
resource "google_compute_global_address" "private_ip_address" {
  name          = "${var.project_name}-${var.environment}-private-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.main.id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.main.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
}

# Cloud SQL Database
resource "google_sql_database" "main" {
  name     = "mlops"
  instance = google_sql_database_instance.main.name
}

# Cloud SQL User
resource "google_sql_user" "main" {
  name     = "mlops"
  instance = google_sql_database_instance.main.name
  password = "changeme123"  # Should be from Secret Manager
}

# Memorystore Redis Instance
resource "google_redis_instance" "main" {
  name           = "${var.project_name}-${var.environment}-redis"
  memory_size_gb = 1
  region         = var.region
  
  location_id             = "${var.region}-a"
  alternative_location_id = "${var.region}-b"
  
  authorized_network = google_compute_network.main.id
  connect_mode       = "PRIVATE_SERVICE_ACCESS"
  
  redis_version     = "REDIS_7_0"
  display_name      = "MLOps Redis Cache"
  reserved_ip_range = "10.3.0.0/29"
  
  auth_enabled = true
  
  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 3
        minutes = 0
      }
    }
  }
  
  persistence_config {
    persistence_mode    = "RDB"
    rdb_snapshot_period = "ONE_HOUR"
  }
  
  depends_on = [google_service_networking_connection.private_vpc_connection]
}

# Cloud Storage Bucket for backups and artifacts
resource "google_storage_bucket" "artifacts" {
  name          = "${var.project_name}-${var.environment}-artifacts-${random_string.bucket_suffix.result}"
  location      = var.region
  force_destroy = false
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  encryption {
    default_kms_key_name = google_kms_crypto_key.storage.id
  }
  
  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = 365
    }
  }
  
  lifecycle_rule {
    action {
      type = "SetStorageClass"
      storage_class = "NEARLINE"
    }
    condition {
      age = 30
    }
  }
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# KMS key for storage encryption
resource "google_kms_crypto_key" "storage" {
  name     = "${var.project_name}-${var.environment}-storage-key"
  key_ring = google_kms_key_ring.gke.id
  
  rotation_period = "7776000s"  # 90 days
}

# Load Balancer
resource "google_compute_global_address" "lb_ip" {
  name = "${var.project_name}-${var.environment}-lb-ip"
}

# SSL Certificate
resource "google_compute_managed_ssl_certificate" "main" {
  name = "${var.project_name}-${var.environment}-ssl-cert"
  
  managed {
    domains = ["gcp.mlops-platform.com"]
  }
}

# Outputs
output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = "https://${google_container_cluster.main.endpoint}"
}

output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.main.name
}

output "cluster_location" {
  description = "GKE cluster location"
  value       = google_container_cluster.main.location
}

output "cluster_ca_certificate" {
  description = "GKE cluster CA certificate"
  value       = google_container_cluster.main.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "network_name" {
  description = "VPC network name"
  value       = google_compute_network.main.name
}

output "subnet_name" {
  description = "Subnet name"
  value       = google_compute_subnetwork.main.name
}

output "sql_instance_connection_name" {
  description = "Cloud SQL instance connection name"
  value       = google_sql_database_instance.main.connection_name
}

output "sql_instance_private_ip" {
  description = "Cloud SQL instance private IP"
  value       = google_sql_database_instance.main.private_ip_address
}

output "redis_host" {
  description = "Redis instance host"
  value       = google_redis_instance.main.host
}

output "redis_port" {
  description = "Redis instance port"
  value       = google_redis_instance.main.port
}

output "storage_bucket_name" {
  description = "Storage bucket name"
  value       = google_storage_bucket.artifacts.name
}

output "load_balancer_ip" {
  description = "Load balancer IP address"
  value       = google_compute_global_address.lb_ip.address
}