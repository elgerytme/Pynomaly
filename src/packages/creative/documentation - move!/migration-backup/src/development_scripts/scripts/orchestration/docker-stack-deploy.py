#!/usr/bin/env python3
"""
Docker Swarm Stack Deployment Script for Pynomaly
Alternative to Kubernetes for simpler production deployments
"""

import os
import subprocess
import sys
import time
from pathlib import Path


def check_docker_swarm():
    """Check if Docker Swarm is initialized."""
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.Swarm.LocalNodeState}}"],
            capture_output=True,
            text=True,
            check=True,
        )

        if result.stdout.strip() == "active":
            print("✅ Docker Swarm is active")
            return True
        else:
            print("⚠️  Docker Swarm is not active")
            return False

    except subprocess.CalledProcessError:
        print("❌ Failed to check Docker Swarm status")
        return False


def initialize_docker_swarm():
    """Initialize Docker Swarm mode."""
    print("Initializing Docker Swarm...")

    try:
        subprocess.run(
            ["docker", "swarm", "init", "--advertise-addr", "127.0.0.1"], check=True
        )

        print("✅ Docker Swarm initialized")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to initialize Docker Swarm: {e}")
        return False


def create_docker_networks():
    """Create Docker overlay networks for the stack."""
    networks = [
        ("pynomaly-network", "attachable"),
        ("monitoring-network", "attachable"),
        ("tracing-network", "attachable"),
        ("logging-network", "attachable"),
    ]

    print("Creating Docker networks...")

    for network_name, options in networks:
        try:
            # Check if network exists
            result = subprocess.run(
                [
                    "docker",
                    "network",
                    "ls",
                    "--filter",
                    f"name={network_name}",
                    "--format",
                    "{{.Name}}",
                ],
                capture_output=True,
                text=True,
            )

            if network_name in result.stdout:
                print(f"Network '{network_name}' already exists")
                continue

            # Create network
            cmd = ["docker", "network", "create", "--driver", "overlay"]
            if options == "attachable":
                cmd.append("--attachable")
            cmd.append(network_name)

            subprocess.run(cmd, check=True)
            print(f"✅ Created network: {network_name}")

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create network {network_name}: {e}")
            return False

    return True


def create_docker_secrets():
    """Create Docker secrets for sensitive data."""
    secrets = {
        "postgres_password": "pynomaly_prod_secret_2024",
        "redis_password": "redis_prod_secret_2024",
        "jwt_secret_key": "jwt_super_secret_key_for_production_2024",
        "app_secret_key": "app_super_secret_key_for_production_2024",
        "grafana_admin_password": "grafana_admin_secret_2024",
        "flower_password": "flower_monitor_secret_2024",
    }

    print("Creating Docker secrets...")

    for secret_name, secret_value in secrets.items():
        try:
            # Check if secret exists
            result = subprocess.run(
                [
                    "docker",
                    "secret",
                    "ls",
                    "--filter",
                    f"name={secret_name}",
                    "--format",
                    "{{.Name}}",
                ],
                capture_output=True,
                text=True,
            )

            if secret_name in result.stdout:
                print(f"Secret '{secret_name}' already exists")
                continue

            # Create secret
            process = subprocess.Popen(
                ["docker", "secret", "create", secret_name, "-"],
                stdin=subprocess.PIPE,
                text=True,
            )

            process.communicate(input=secret_value)

            if process.returncode == 0:
                print(f"✅ Created secret: {secret_name}")
            else:
                print(f"❌ Failed to create secret: {secret_name}")
                return False

        except Exception as e:
            print(f"❌ Failed to create secret {secret_name}: {e}")
            return False

    return True


def create_docker_configs():
    """Create Docker configs for configuration files."""
    configs = {
        "prometheus_config": """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "/etc/prometheus/pynomaly_alerts.yml"

scrape_configs:
  - job_name: 'pynomaly-api'
    dns_sd_configs:
      - names:
          - 'tasks.pynomaly-api'
        type: 'A'
        port: 9090
    scrape_interval: 15s
    metrics_path: /metrics

  - job_name: 'pynomaly-workers'
    dns_sd_configs:
      - names:
          - 'tasks.pynomaly-worker-training'
          - 'tasks.pynomaly-worker-drift'
        type: 'A'
        port: 9090
    scrape_interval: 30s

  - job_name: 'node-exporter'
    dns_sd_configs:
      - names:
          - 'tasks.node-exporter'
        type: 'A'
        port: 9100

  - job_name: 'cadvisor'
    dns_sd_configs:
      - names:
          - 'tasks.cadvisor'
        type: 'A'
        port: 8080
""",
        "nginx_config": """
events {
    worker_connections 1024;
}

http {
    upstream pynomaly_api {
        server pynomaly-api:8000;
    }

    upstream grafana {
        server grafana:3000;
    }

    upstream jaeger {
        server jaeger:16686;
    }

    server {
        listen 80;
        server_name api.monorepo.local;

        location / {
            proxy_pass http://pynomaly_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /health {
            proxy_pass http://pynomaly_api/health;
        }
    }

    server {
        listen 80;
        server_name grafana.monorepo.local;

        location / {
            proxy_pass http://grafana;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }

    server {
        listen 80;
        server_name jaeger.monorepo.local;

        location / {
            proxy_pass http://jaeger;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
""",
        "otel_collector_config": """
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 1s
    send_batch_size: 1024
  memory_limiter:
    limit_mib: 512

exporters:
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true
  prometheus:
    endpoint: "0.0.0.0:8889"
    namespace: "monorepo"

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [jaeger]
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [prometheus]
""",
    }

    print("Creating Docker configs...")

    for config_name, config_content in configs.items():
        try:
            # Check if config exists
            result = subprocess.run(
                [
                    "docker",
                    "config",
                    "ls",
                    "--filter",
                    f"name={config_name}",
                    "--format",
                    "{{.Name}}",
                ],
                capture_output=True,
                text=True,
            )

            if config_name in result.stdout:
                print(f"Config '{config_name}' already exists")
                continue

            # Create config
            process = subprocess.Popen(
                ["docker", "config", "create", config_name, "-"],
                stdin=subprocess.PIPE,
                text=True,
            )

            process.communicate(input=config_content.strip())

            if process.returncode == 0:
                print(f"✅ Created config: {config_name}")
            else:
                print(f"❌ Failed to create config: {config_name}")
                return False

        except Exception as e:
            print(f"❌ Failed to create config {config_name}: {e}")
            return False

    return True


def create_docker_stack_file():
    """Create Docker Stack compose file."""
    stack_compose = """
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: pynomaly
      POSTGRES_USER: pynomaly
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
    secrets:
      - postgres_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - pynomaly-network
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.role == manager
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 512M
          cpus: '0.25'
      restart_policy:
        condition: any
        delay: 5s
        max_attempts: 3

  # Redis Cache
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass_file /run/secrets/redis_password
    secrets:
      - redis_password
    volumes:
      - redis_data:/data
    networks:
      - pynomaly-network
    deploy:
      replicas: 1
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 256M
          cpus: '0.1'
      restart_policy:
        condition: any
        delay: 5s

  # Pynomaly API
  pynomaly-api:
    image: pynomaly/api:1.0.0
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://pynomaly:$(cat /run/secrets/postgres_password)@postgres:5432/pynomaly
      - REDIS_URL=redis://:$(cat /run/secrets/redis_password)@redis:6379/0
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
      - JAEGER_AGENT_HOST=jaeger
    secrets:
      - postgres_password
      - redis_password
      - jwt_secret_key
      - app_secret_key
    volumes:
      - app_storage:/app/storage
      - app_logs:/app/logs
    networks:
      - pynomaly-network
    depends_on:
      - postgres
      - redis
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        order: start-first
        failure_action: rollback
        delay: 10s
      rollback_config:
        parallelism: 1
        order: stop-first
        delay: 10s
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 1G
          cpus: '0.5'
      restart_policy:
        condition: any
        delay: 5s
        max_attempts: 3

  # Training Worker
  pynomaly-worker-training:
    image: pynomaly/worker:1.0.0
    command: celery worker -A monorepo.workers -Q model_training,anomaly_detection --concurrency=4
    environment:
      - ENVIRONMENT=production
      - CELERY_BROKER_URL=redis://:$(cat /run/secrets/redis_password)@redis:6379/0
      - DATABASE_URL=postgresql://pynomaly:$(cat /run/secrets/postgres_password)@postgres:5432/pynomaly
    secrets:
      - postgres_password
      - redis_password
    volumes:
      - app_storage:/app/storage
      - app_logs:/app/logs
    networks:
      - pynomaly-network
    depends_on:
      - postgres
      - redis
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 8G
          cpus: '4.0'
        reservations:
          memory: 2G
          cpus: '1.0'
      restart_policy:
        condition: any
        delay: 5s

  # Drift Worker
  pynomaly-worker-drift:
    image: pynomaly/worker:1.0.0
    command: celery worker -A monorepo.workers -Q drift_monitoring,alert_processing --concurrency=2
    environment:
      - ENVIRONMENT=production
      - CELERY_BROKER_URL=redis://:$(cat /run/secrets/redis_password)@redis:6379/0
      - DATABASE_URL=postgresql://pynomaly:$(cat /run/secrets/postgres_password)@postgres:5432/pynomaly
    secrets:
      - postgres_password
      - redis_password
    volumes:
      - app_storage:/app/storage
      - app_logs:/app/logs
    networks:
      - pynomaly-network
    depends_on:
      - postgres
      - redis
    deploy:
      replicas: 1
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 1G
          cpus: '0.5'
      restart_policy:
        condition: any
        delay: 5s

  # Celery Scheduler
  pynomaly-scheduler:
    image: pynomaly/scheduler:1.0.0
    command: celery beat -A monorepo.workers --loglevel=info
    environment:
      - ENVIRONMENT=production
      - CELERY_BROKER_URL=redis://:$(cat /run/secrets/redis_password)@redis:6379/0
      - DATABASE_URL=postgresql://pynomaly:$(cat /run/secrets/postgres_password)@postgres:5432/pynomaly
    secrets:
      - postgres_password
      - redis_password
    volumes:
      - app_storage:/app/storage
      - app_logs:/app/logs
    networks:
      - pynomaly-network
    depends_on:
      - postgres
      - redis
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.role == manager
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 256M
          cpus: '0.1'

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    configs:
      - source: prometheus_config
        target: /etc/prometheus/prometheus.yml
    volumes:
      - prometheus_data:/prometheus
    networks:
      - monitoring-network
      - pynomaly-network
    ports:
      - "9090:9090"
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.role == manager
      resources:
        limits:
          memory: 8G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '0.5'

  # Grafana
  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD__FILE=/run/secrets/grafana_admin_password
      - GF_USERS_ALLOW_SIGN_UP=false
    secrets:
      - grafana_admin_password
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - monitoring-network
    ports:
      - "3000:3000"
    deploy:
      replicas: 1
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 256M
          cpus: '0.1'

  # Jaeger All-in-One
  jaeger:
    image: jaegertracing/all-in-one:latest
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
      - COLLECTOR_OTLP_ENABLED=true
      - SPAN_STORAGE_TYPE=badger
      - BADGER_EPHEMERAL=false
      - BADGER_DIRECTORY_VALUE=/badger/data
      - BADGER_DIRECTORY_KEY=/badger/key
    volumes:
      - jaeger_data:/badger
    networks:
      - tracing-network
      - pynomaly-network
    ports:
      - "16686:16686"
      - "14268:14268"
      - "14250:14250"
      - "6831:6831/udp"
      - "6832:6832/udp"
    deploy:
      replicas: 1
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 512M
          cpus: '0.25'

  # OpenTelemetry Collector
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    command: ["--config=/etc/otel-collector-config.yaml"]
    configs:
      - source: otel_collector_config
        target: /etc/otel-collector-config.yaml
    networks:
      - tracing-network
      - pynomaly-network
      - monitoring-network
    ports:
      - "4317:4317"
      - "4318:4318"
      - "8889:8889"
    depends_on:
      - jaeger
      - prometheus
    deploy:
      replicas: 1
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 256M
          cpus: '0.1'

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    configs:
      - source: nginx_config
        target: /etc/nginx/nginx.conf
    networks:
      - pynomaly-network
      - monitoring-network
      - tracing-network
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - pynomaly-api
      - grafana
      - jaeger
    deploy:
      replicas: 2
      update_config:
        parallelism: 1
        order: start-first
      resources:
        limits:
          memory: 256M
          cpus: '0.25'
        reservations:
          memory: 64M
          cpus: '0.05'

  # Node Exporter
  node-exporter:
    image: prom/node-exporter:latest
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    networks:
      - monitoring-network
    ports:
      - "9100:9100"
    deploy:
      mode: global
      resources:
        limits:
          memory: 128M
          cpus: '0.25'
        reservations:
          memory: 64M
          cpus: '0.05'

  # cAdvisor
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    networks:
      - monitoring-network
    ports:
      - "8080:8080"
    deploy:
      mode: global
      resources:
        limits:
          memory: 256M
          cpus: '0.25'
        reservations:
          memory: 128M
          cpus: '0.1'

networks:
  pynomaly-network:
    driver: overlay
    attachable: true
  monitoring-network:
    driver: overlay
    attachable: true
  tracing-network:
    driver: overlay
    attachable: true
  logging-network:
    driver: overlay
    attachable: true

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
  jaeger_data:
  app_storage:
  app_logs:

secrets:
  postgres_password:
    external: true
  redis_password:
    external: true
  jwt_secret_key:
    external: true
  app_secret_key:
    external: true
  grafana_admin_password:
    external: true
  flower_password:
    external: true

configs:
  prometheus_config:
    external: true
  nginx_config:
    external: true
  otel_collector_config:
    external: true
"""

    stack_dir = Path("/mnt/c/Users/andre/Pynomaly/deploy/docker-swarm")
    stack_dir.mkdir(parents=True, exist_ok=True)
    stack_file = stack_dir / "docker-stack.yml"

    with open(stack_file, "w") as f:
        f.write(stack_compose.strip())

    print(f"✅ Docker Stack file created: {stack_file}")
    return stack_file


def deploy_stack(stack_file: Path, stack_name: str = "monorepo"):
    """Deploy the Docker Stack."""
    print(f"Deploying Docker Stack '{stack_name}'...")

    try:
        subprocess.run(
            [
                "docker",
                "stack",
                "deploy",
                "--compose-file",
                str(stack_file),
                stack_name,
            ],
            check=True,
        )

        print(f"✅ Stack '{stack_name}' deployed successfully")

        # Wait for services to start
        print("Waiting for services to start...")
        time.sleep(30)

        # Show stack status
        subprocess.run(["docker", "stack", "ps", stack_name])

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to deploy stack: {e}")
        return False


def create_management_scripts():
    """Create management scripts for Docker Swarm."""
    scripts_dir = Path("/opt/pynomaly/scripts/docker-swarm")
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # Deploy script
    deploy_script = """#!/bin/bash
# Docker Swarm deployment script for Pynomaly

STACK_NAME="${STACK_NAME:-pynomaly}"
COMPOSE_FILE="${COMPOSE_FILE:-./deploy/docker-swarm/docker-stack.yml}"

echo "🚀 Deploying Pynomaly Stack to Docker Swarm..."
echo "Stack: $STACK_NAME"
echo "Compose file: $COMPOSE_FILE"

# Check if swarm is initialized
if ! docker info --format '{{.Swarm.LocalNodeState}}' | grep -q active; then
    echo "❌ Docker Swarm is not active. Please initialize swarm first:"
    echo "docker swarm init"
    exit 1
fi

# Deploy stack
docker stack deploy --compose-file "$COMPOSE_FILE" "$STACK_NAME"

echo "✅ Stack deployment initiated!"

# Wait and show status
echo "Waiting for services to start..."
sleep 30

echo ""
echo "📊 Stack Status:"
docker stack ps "$STACK_NAME"

echo ""
echo "🌐 Services:"
docker stack services "$STACK_NAME"
"""

    deploy_file = scripts_dir / "deploy.sh"
    with open(deploy_file, "w") as f:
        f.write(deploy_script.strip())
    os.chmod(deploy_file, 0o755)

    # Monitor script
    monitor_script = """#!/bin/bash
# Docker Swarm monitoring script for Pynomaly

STACK_NAME="${STACK_NAME:-pynomaly}"

echo "📊 Pynomaly Docker Swarm Monitoring"
echo "Stack: $STACK_NAME"
echo "=" * 50

# Swarm status
echo ""
echo "🏠 Swarm Status:"
docker node ls

# Stack services
echo ""
echo "🐳 Stack Services:"
docker stack services "$STACK_NAME"

# Stack tasks
echo ""
echo "📋 Stack Tasks:"
docker stack ps "$STACK_NAME" --no-trunc

# Service logs (sample)
echo ""
echo "📄 Recent Logs (API):"
docker service logs --tail 20 "${STACK_NAME}_pynomaly-api" 2>/dev/null || echo "No API service logs"

# System resource usage
echo ""
echo "📈 System Resources:"
docker system df
"""

    monitor_file = scripts_dir / "monitor.sh"
    with open(monitor_file, "w") as f:
        f.write(monitor_script.strip())
    os.chmod(monitor_file, 0o755)

    # Scale script
    scale_script = """#!/bin/bash
# Docker Swarm scaling script for Pynomaly

STACK_NAME="${STACK_NAME:-pynomaly}"

usage() {
    echo "Usage: $0 <service> <replicas>"
    echo "Services: api, worker-training, worker-drift, nginx"
    echo "Example: $0 api 5"
    exit 1
}

if [ $# -ne 2 ]; then
    usage
fi

SERVICE="$1"
REPLICAS="$2"

case $SERVICE in
    "api")
        SERVICE_NAME="${STACK_NAME}_pynomaly-api"
        ;;
    "worker-training")
        SERVICE_NAME="${STACK_NAME}_pynomaly-worker-training"
        ;;
    "worker-drift")
        SERVICE_NAME="${STACK_NAME}_pynomaly-worker-drift"
        ;;
    "nginx")
        SERVICE_NAME="${STACK_NAME}_nginx"
        ;;
    *)
        echo "Unknown service: $SERVICE"
        usage
        ;;
esac

echo "📊 Scaling $SERVICE_NAME to $REPLICAS replicas..."

docker service scale "$SERVICE_NAME=$REPLICAS"

echo "✅ Scaling completed!"
echo ""
echo "📊 Current status:"
docker service ls --filter name="$SERVICE_NAME"
"""

    scale_file = scripts_dir / "scale.sh"
    with open(scale_file, "w") as f:
        f.write(scale_script.strip())
    os.chmod(scale_file, 0o755)

    # Cleanup script
    cleanup_script = """#!/bin/bash
# Docker Swarm cleanup script for Pynomaly

STACK_NAME="${STACK_NAME:-pynomaly}"

echo "🧹 Cleaning up Pynomaly Docker Stack..."
echo "Stack: $STACK_NAME"

read -p "Are you sure you want to remove the stack? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Remove stack
docker stack rm "$STACK_NAME"

echo "Waiting for services to shut down..."
sleep 30

# Clean up volumes (optional)
read -p "Remove volumes? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "💾 Removing volumes..."
    docker volume prune -f
fi

# Clean up configs and secrets (optional)
read -p "Remove configs and secrets? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔐 Removing configs and secrets..."
    docker config rm prometheus_config nginx_config otel_collector_config 2>/dev/null || true
    docker secret rm postgres_password redis_password jwt_secret_key app_secret_key grafana_admin_password flower_password 2>/dev/null || true
fi

echo "✅ Cleanup completed!"
"""

    cleanup_file = scripts_dir / "cleanup.sh"
    with open(cleanup_file, "w") as f:
        f.write(cleanup_script.strip())
    os.chmod(cleanup_file, 0o755)

    print(f"✅ Management scripts created in {scripts_dir}")
    return True


def main():
    """Main Docker Swarm setup function."""
    print("Setting up Pynomaly Docker Swarm Deployment...")
    print("=" * 60)

    try:
        # Check Docker Swarm status
        if not check_docker_swarm():
            if input("Initialize Docker Swarm? (y/N): ").lower() == "y":
                if not initialize_docker_swarm():
                    sys.exit(1)
            else:
                print("Docker Swarm is required for stack deployment")
                sys.exit(1)

        # Create networks
        if not create_docker_networks():
            sys.exit(1)

        # Create secrets
        if not create_docker_secrets():
            sys.exit(1)

        # Create configs
        if not create_docker_configs():
            sys.exit(1)

        # Create stack file
        stack_file = create_docker_stack_file()
        if not stack_file:
            sys.exit(1)

        # Create management scripts
        if not create_management_scripts():
            sys.exit(1)

        # Ask to deploy
        if input("Deploy the stack now? (y/N): ").lower() == "y":
            if not deploy_stack(stack_file):
                sys.exit(1)

        print("\n" + "=" * 60)
        print("✅ Docker Swarm setup completed successfully!")
        print("\nNext steps:")
        print("1. Deploy the stack:")
        print("   ./scripts/docker-swarm/deploy.sh")
        print("\n2. Monitor the deployment:")
        print("   ./scripts/docker-swarm/monitor.sh")
        print("\n3. Scale services:")
        print("   ./scripts/docker-swarm/scale.sh api 5")
        print("\n4. Access services:")
        print("   - API: http://api.monorepo.local")
        print("   - Grafana: http://grafana.monorepo.local (admin/admin)")
        print("   - Jaeger: http://jaeger.monorepo.local")
        print("   - Prometheus: http://localhost:9090")
        print("\n5. Clean up:")
        print("   ./scripts/docker-swarm/cleanup.sh")

        print("\nKey files created:")
        print(f"- Docker Stack: {stack_file}")
        print("- Management scripts: /opt/pynomaly/scripts/docker-swarm/")

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
