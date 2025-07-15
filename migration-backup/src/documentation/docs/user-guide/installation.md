# Installation Guide

This guide covers all installation methods for Pynomaly, from simple pip installation to enterprise Kubernetes deployments.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 10GB free space for models and data

### Dependencies
- NumPy >= 1.26.0
- Pandas >= 2.2.3
- Scikit-learn >= 1.7.0
- PyOD >= 2.0.5

## 🚀 Quick Installation

### Option 1: pip (Recommended)
```bash
# Install from PyPI
pip install pynomaly

# Verify installation
pynomaly --version
```

### Option 2: Development Installation
```bash
# Clone repository
git clone https://github.com/your-org/pynomaly.git
cd pynomaly

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

## 🔧 Installation Options

### Basic Installation
For basic anomaly detection capabilities:
```bash
pip install pynomaly
```

### Full Installation
For all features including web interface and monitoring:
```bash
pip install "pynomaly[full]"
```

### Development Installation
For contributing to Pynomaly:
```bash
pip install "pynomaly[dev]"
```

Available extras:
- `web`: Web interface and dashboard
- `monitoring`: Monitoring and alerting features
- `ml-ops`: ML governance and deployment tools
- `docs`: Documentation building tools
- `dev`: Development and testing tools
- `full`: All features

## 🐳 Docker Installation

### Quick Start with Docker
```bash
# Pull the official image
docker pull pynomaly/pynomaly:latest

# Run with default configuration
docker run -p 8080:8080 pynomaly/pynomaly:latest

# Access web interface at http://localhost:8080
```

### Docker Compose for Development
```bash
# Clone repository
git clone https://github.com/your-org/pynomaly.git
cd pynomaly

# Start all services
docker-compose up -d

# Services available:
# - Web UI: http://localhost:8080
# - API: http://localhost:8080/api
# - Monitoring: http://localhost:3000
```

### Custom Docker Configuration
```bash
# Create custom configuration
mkdir -p ~/.pynomaly/config

# Run with custom config
docker run -p 8080:8080 \
  -v ~/.pynomaly/config:/app/config \
  -v ~/.pynomaly/data:/app/data \
  pynomaly/pynomaly:latest
```

## ☸️ Kubernetes Installation

### Prerequisites
- Kubernetes cluster (1.20+)
- kubectl configured
- Helm 3.0+ (optional)

### Quick Deployment
```bash
# Apply basic configuration
kubectl apply -f https://raw.githubusercontent.com/your-org/pynomaly/main/deploy/k8s/quick-start.yaml

# Check deployment status
kubectl get pods -l app=pynomaly

# Access via port-forward
kubectl port-forward service/pynomaly 8080:8080
```

### Helm Installation
```bash
# Add Pynomaly Helm repository
helm repo add pynomaly https://charts.pynomaly.com
helm repo update

# Install with default values
helm install pynomaly pynomaly/pynomaly

# Install with custom values
helm install pynomaly pynomaly/pynomaly -f values.yaml
```

### Production Kubernetes Setup
```bash
# Create namespace
kubectl create namespace pynomaly-prod

# Apply production configuration
kubectl apply -f k8s/production/ -n pynomaly-prod

# Components deployed:
# - Application pods with auto-scaling
# - PostgreSQL database
# - Redis cache
# - Ingress controller
# - Monitoring stack
```

## 🔐 Security Installation

### TLS/SSL Configuration
```bash
# Generate self-signed certificates (development)
pynomaly security generate-certs

# Or use existing certificates
export PYNOMALY_TLS_CERT=/path/to/cert.pem
export PYNOMALY_TLS_KEY=/path/to/key.pem

# Start with TLS enabled
pynomaly server start --tls
```

### Authentication Setup
```bash
# Initialize authentication
pynomaly auth init

# Create admin user
pynomaly auth create-user --username admin --role admin

# Configure OAuth (optional)
pynomaly auth configure-oauth --provider google
```

## 🔧 Configuration

### Environment Variables
```bash
# Core configuration
export PYNOMALY_HOST=0.0.0.0
export PYNOMALY_PORT=8080
export PYNOMALY_DEBUG=false

# Database configuration
export PYNOMALY_DATABASE_URL=postgresql://user:pass@localhost/pynomaly

# Redis configuration
export PYNOMALY_REDIS_URL=redis://localhost:6379

# Storage configuration
export PYNOMALY_STORAGE_BACKEND=local
export PYNOMALY_STORAGE_PATH=/data/pynomaly
```

### Configuration File
Create `~/.pynomaly/config.yaml`:
```yaml
server:
  host: 0.0.0.0
  port: 8080
  workers: 4

database:
  url: postgresql://user:pass@localhost/pynomaly
  pool_size: 10

redis:
  url: redis://localhost:6379
  db: 0

logging:
  level: INFO
  format: json

ml:
  model_storage: /data/models
  max_model_size: 1GB
```

## ✅ Verification

### Basic Verification
```bash
# Check version
pynomaly --version

# Run health check
pynomaly health

# Test basic functionality
python -c "from pynomaly import AnomalyDetector; print('✅ Installation successful')"
```

### Web Interface Verification
```bash
# Start server
pynomaly server start

# Open browser to http://localhost:8080
# You should see the Pynomaly dashboard
```

### API Verification
```bash
# Test API endpoint
curl http://localhost:8080/api/health

# Expected response:
# {"status": "healthy", "version": "0.1.2"}
```

## 🐛 Troubleshooting

### Common Issues

#### Installation Fails
```bash
# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Clear pip cache
pip cache purge

# Install with verbose output
pip install -v pynomaly
```

#### Import Errors
```bash
# Check Python version
python --version  # Should be 3.8+

# Check installed packages
pip list | grep pynomaly

# Reinstall dependencies
pip install --force-reinstall pynomaly
```

#### Permission Errors
```bash
# Install for current user only
pip install --user pynomaly

# Or use virtual environment
python -m venv venv
source venv/bin/activate
pip install pynomaly
```

#### Docker Issues
```bash
# Check Docker version
docker --version

# Pull latest image
docker pull pynomaly/pynomaly:latest

# Check container logs
docker logs <container_id>
```

### Getting Help

If you encounter issues:

1. **Check the logs**: `pynomaly logs --tail 100`
2. **Verify configuration**: `pynomaly config validate`
3. **Check system resources**: Ensure adequate memory and disk space
4. **Update to latest version**: `pip install --upgrade pynomaly`
5. **Search documentation**: Check our troubleshooting guide
6. **Report issues**: Create a GitHub issue with details

## 🔄 Upgrade Guide

### Upgrading from pip
```bash
# Backup configuration
cp ~/.pynomaly/config.yaml ~/.pynomaly/config.yaml.backup

# Upgrade
pip install --upgrade pynomaly

# Verify upgrade
pynomaly --version
```

### Upgrading Docker
```bash
# Pull new image
docker pull pynomaly/pynomaly:latest

# Stop old container
docker stop pynomaly

# Start with new image
docker run -p 8080:8080 pynomaly/pynomaly:latest
```

### Upgrading Kubernetes
```bash
# Backup current deployment
kubectl get deployment pynomaly -o yaml > pynomaly-backup.yaml

# Update with Helm
helm upgrade pynomaly pynomaly/pynomaly

# Or apply new manifests
kubectl apply -f k8s/
```

## 🎯 Next Steps

Now that Pynomaly is installed:

1. 📖 Follow the [Basic Tutorial](basic-tutorial.md) to create your first anomaly detector
2. 🔧 Configure [Data Sources](data-management.md) for your use case
3. 🚀 Set up [Real-time Detection](real-time-detection.md) for streaming data
4. 📊 Explore the [Web Dashboard](web-interface.md) for visual analysis
5. 🏭 Plan your [Production Deployment](production-deployment.md)

Ready to detect some anomalies? Let's start with the [Basic Tutorial](basic-tutorial.md)!