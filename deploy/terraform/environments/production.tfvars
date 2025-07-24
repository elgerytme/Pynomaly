# Production Environment Configuration

environment = "production"
project_name = "anomaly-detection"

# Kubernetes Configuration
kubernetes_cluster_name = "anomaly-detection-production"
# These values should be provided via environment variables or secure parameter store
# kubernetes_cluster_endpoint = "https://production-k8s.example.com"
# kubernetes_cluster_ca_certificate = "LS0tLS1CRUdJTi..."
# kubernetes_token = "eyJhbGciOiJSUzI1NiIs..."

# Domain Configuration
domain_name = "anomaly-detection.io"

# Feature Flags
enable_monitoring = true
enable_security_scanning = true

# Additional production-specific configuration
# (Add as needed for production environment)