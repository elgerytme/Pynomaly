#!/usr/bin/env python3
"""
Distributed Tracing Setup Script for Pynomaly

This script sets up and configures the distributed tracing infrastructure
including Jaeger, OTLP collectors, and OpenTelemetry instrumentation.
"""

import os
import sys
from pathlib import Path


def create_jaeger_docker_compose():
    """Create Docker Compose configuration for Jaeger."""
    jaeger_compose = """
version: '3.8'

services:
  jaeger-all-in-one:
    image: jaegertracing/all-in-one:latest
    container_name: pynomaly-jaeger
    ports:
      - "16686:16686"  # Jaeger UI
      - "14268:14268"  # Jaeger collector HTTP
      - "14250:14250"  # Jaeger collector gRPC
      - "6831:6831/udp"  # Jaeger agent UDP
      - "6832:6832/udp"  # Jaeger agent UDP
      - "5778:5778"   # Jaeger agent HTTP
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
      - COLLECTOR_OTLP_ENABLED=true
    networks:
      - pynomaly-tracing
    restart: unless-stopped
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.jaeger.rule=Host(`jaeger.local`)"
      - "traefik.http.services.jaeger.loadbalancer.server.port=16686"

  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    container_name: pynomaly-otel-collector
    command: ["--config=/etc/otel-collector-config.yaml"]
    volumes:
      - ./otel-collector-config.yaml:/etc/otel-collector-config.yaml
    ports:
      - "4317:4317"   # OTLP gRPC receiver
      - "4318:4318"   # OTLP HTTP receiver
      - "8888:8888"   # Prometheus metrics
      - "8889:8889"   # Prometheus exporter metrics
    depends_on:
      - jaeger-all-in-one
    networks:
      - pynomaly-tracing
    restart: unless-stopped

  zipkin:
    image: openzipkin/zipkin:latest
    container_name: pynomaly-zipkin
    ports:
      - "9411:9411"   # Zipkin UI and API
    networks:
      - pynomaly-tracing
    restart: unless-stopped
    environment:
      - STORAGE_TYPE=mem

  # Prometheus for metrics collection
  prometheus:
    image: prom/prometheus:latest
    container_name: pynomaly-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    networks:
      - pynomaly-tracing
    restart: unless-stopped

  # Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    container_name: pynomaly-grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
      - GF_USERS_ALLOW_SIGN_UP=false
    networks:
      - pynomaly-tracing
    restart: unless-stopped

networks:
  pynomaly-tracing:
    driver: bridge

volumes:
  prometheus_data:
  grafana_data:
"""

    compose_dir = Path("/opt/pynomaly/docker/tracing")
    compose_dir.mkdir(parents=True, exist_ok=True)

    compose_file = compose_dir / "docker-compose.yml"

    print("Creating Jaeger Docker Compose configuration...")
    with open(compose_file, "w") as f:
        f.write(jaeger_compose.strip())

    print(f"Docker Compose file created: {compose_file}")
    return compose_file


def create_otel_collector_config():
    """Create OpenTelemetry Collector configuration."""
    otel_config = """
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318
        cors:
          allowed_origins:
            - "http://*"
            - "https://*"

  jaeger:
    protocols:
      grpc:
        endpoint: 0.0.0.0:14250
      thrift_http:
        endpoint: 0.0.0.0:14268
      thrift_compact:
        endpoint: 0.0.0.0:6831

  zipkin:
    endpoint: 0.0.0.0:9411

processors:
  batch:
    timeout: 1s
    send_batch_size: 1024
    send_batch_max_size: 2048

  memory_limiter:
    limit_mib: 512
    spike_limit_mib: 128
    check_interval: 5s

  resource:
    attributes:
      - key: environment
        value: production
        action: upsert
      - key: service.namespace
        value: pynomaly
        action: upsert

  filter:
    traces:
      span:
        - 'attributes["http.url"] != nil and IsMatch(attributes["http.url"], ".*health.*")'
        - 'attributes["http.url"] != nil and IsMatch(attributes["http.url"], ".*metrics.*")'

  transform:
    traces:
      statements:
        - context: span
          statements:
            # Remove sensitive attributes
            - delete_key(attributes, "authorization")
            - delete_key(attributes, "cookie")
            - delete_key(attributes, "password")
            # Normalize HTTP status codes
            - set(attributes["http.status_class"], "2xx") where attributes["http.status_code"] >= 200 and attributes["http.status_code"] < 300
            - set(attributes["http.status_class"], "3xx") where attributes["http.status_code"] >= 300 and attributes["http.status_code"] < 400
            - set(attributes["http.status_class"], "4xx") where attributes["http.status_code"] >= 400 and attributes["http.status_code"] < 500
            - set(attributes["http.status_class"], "5xx") where attributes["http.status_code"] >= 500

exporters:
  jaeger:
    endpoint: jaeger-all-in-one:14250
    tls:
      insecure: true

  zipkin:
    endpoint: "http://zipkin:9411/api/v2/spans"

  prometheus:
    endpoint: "0.0.0.0:8889"
    namespace: "monorepo"
    const_labels:
      environment: "production"

  logging:
    loglevel: info

  file:
    path: "/tmp/traces.json"

extensions:
  health_check:
    endpoint: 0.0.0.0:13133

  pprof:
    endpoint: 0.0.0.0:1777

  zpages:
    endpoint: 0.0.0.0:55679

service:
  extensions: [health_check, pprof, zpages]

  pipelines:
    traces:
      receivers: [otlp, jaeger, zipkin]
      processors: [memory_limiter, resource, filter, transform, batch]
      exporters: [jaeger, zipkin, logging]

    metrics:
      receivers: [otlp]
      processors: [memory_limiter, resource, batch]
      exporters: [prometheus, logging]

  telemetry:
    logs:
      level: "info"
    metrics:
      address: 0.0.0.0:8888
"""

    config_dir = Path("/opt/pynomaly/docker/tracing")
    config_dir.mkdir(parents=True, exist_ok=True)

    config_file = config_dir / "otel-collector-config.yaml"

    print("Creating OpenTelemetry Collector configuration...")
    with open(config_file, "w") as f:
        f.write(otel_config.strip())

    print(f"OTEL Collector config created: {config_file}")
    return config_file


def create_prometheus_config():
    """Create Prometheus configuration."""
    prometheus_config = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "tracing_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'otel-collector'
    static_configs:
      - targets: ['otel-collector:8888', 'otel-collector:8889']
    scrape_interval: 10s
    metrics_path: /metrics

  - job_name: 'pynomaly-app'
    static_configs:
      - targets: ['host.docker.internal:8000']
    scrape_interval: 15s
    metrics_path: /metrics

  - job_name: 'jaeger'
    static_configs:
      - targets: ['jaeger-all-in-one:14269']
    scrape_interval: 15s
    metrics_path: /metrics
"""

    config_dir = Path("/opt/pynomaly/docker/tracing")
    config_file = config_dir / "prometheus.yml"

    print("Creating Prometheus configuration...")
    with open(config_file, "w") as f:
        f.write(prometheus_config.strip())

    print(f"Prometheus config created: {config_file}")
    return config_file


def create_grafana_config():
    """Create Grafana configuration and dashboards."""
    # Grafana provisioning configuration
    grafana_dir = Path("/opt/pynomaly/docker/tracing/grafana")

    # Datasources
    datasources_dir = grafana_dir / "provisioning" / "datasources"
    datasources_dir.mkdir(parents=True, exist_ok=True)

    datasources_config = """
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true

  - name: Jaeger
    type: jaeger
    access: proxy
    url: http://jaeger-all-in-one:16686
    editable: true
"""

    with open(datasources_dir / "datasources.yml", "w") as f:
        f.write(datasources_config.strip())

    # Dashboards provisioning
    dashboards_dir = grafana_dir / "provisioning" / "dashboards"
    dashboards_dir.mkdir(parents=True, exist_ok=True)

    dashboards_config = """
apiVersion: 1

providers:
  - name: 'pynomaly-dashboards'
    orgId: 1
    folder: 'Pynomaly'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
"""

    with open(dashboards_dir / "dashboards.yml", "w") as f:
        f.write(dashboards_config.strip())

    # Create dashboard directory
    dashboard_files_dir = grafana_dir / "dashboards"
    dashboard_files_dir.mkdir(parents=True, exist_ok=True)

    # Tracing overview dashboard
    tracing_dashboard = {
        "dashboard": {
            "id": None,
            "title": "Pynomaly Distributed Tracing",
            "tags": ["monorepo", "tracing"],
            "timezone": "browser",
            "panels": [
                {
                    "id": 1,
                    "title": "Traces per Second",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "rate(otelcol_exporter_sent_spans_total[5m])",
                            "legendFormat": "{{exporter}}",
                        }
                    ],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                },
                {
                    "id": 2,
                    "title": "Export Latency",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, rate(otelcol_exporter_send_failed_spans_total[5m]))",
                            "legendFormat": "95th percentile",
                        }
                    ],
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                },
            ],
            "time": {"from": "now-1h", "to": "now"},
            "refresh": "5s",
        }
    }

    with open(dashboard_files_dir / "tracing-overview.json", "w") as f:
        import json

        json.dump(tracing_dashboard, f, indent=2)

    print(f"Grafana configuration created: {grafana_dir}")


def create_tracing_scripts():
    """Create utility scripts for tracing management."""
    scripts_dir = Path("/opt/pynomaly/scripts/tracing")
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # Start tracing infrastructure script
    start_script = """#!/bin/bash
# Start Pynomaly distributed tracing infrastructure

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$SCRIPT_DIR/../../docker/tracing"

echo "Starting Pynomaly distributed tracing infrastructure..."

cd "$DOCKER_DIR"

# Pull latest images
echo "Pulling latest Docker images..."
docker-compose pull

# Start services
echo "Starting tracing services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Check service health
echo "Checking service health..."

# Check Jaeger
if curl -f http://localhost:16686/api/services > /dev/null 2>&1; then
    echo "✅ Jaeger is running: http://localhost:16686"
else
    echo "❌ Jaeger is not responding"
fi

# Check OTEL Collector
if curl -f http://localhost:13133 > /dev/null 2>&1; then
    echo "✅ OTEL Collector is running: http://localhost:13133"
else
    echo "❌ OTEL Collector is not responding"
fi

# Check Prometheus
if curl -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
    echo "✅ Prometheus is running: http://localhost:9090"
else
    echo "❌ Prometheus is not responding"
fi

# Check Grafana
if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
    echo "✅ Grafana is running: http://localhost:3000 (admin/admin123)"
else
    echo "❌ Grafana is not responding"
fi

echo ""
echo "🎉 Distributed tracing infrastructure is ready!"
echo ""
echo "Service URLs:"
echo "  - Jaeger UI: http://localhost:16686"
echo "  - Grafana: http://localhost:3000 (admin/admin123)"
echo "  - Prometheus: http://localhost:9090"
echo "  - OTEL Collector Health: http://localhost:13133"
echo ""
echo "Trace endpoints:"
echo "  - OTLP gRPC: http://localhost:4317"
echo "  - OTLP HTTP: http://localhost:4318"
echo "  - Jaeger gRPC: http://localhost:14250"
echo "  - Jaeger HTTP: http://localhost:14268"
echo "  - Zipkin: http://localhost:9411"
"""

    start_script_file = scripts_dir / "start_tracing.sh"
    with open(start_script_file, "w") as f:
        f.write(start_script.strip())
    os.chmod(start_script_file, 0o755)

    # Stop tracing infrastructure script
    stop_script = """#!/bin/bash
# Stop Pynomaly distributed tracing infrastructure

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$SCRIPT_DIR/../../docker/tracing"

echo "Stopping Pynomaly distributed tracing infrastructure..."

cd "$DOCKER_DIR"

# Stop services
docker-compose down

echo "✅ Distributed tracing infrastructure stopped"
"""

    stop_script_file = scripts_dir / "stop_tracing.sh"
    with open(stop_script_file, "w") as f:
        f.write(stop_script.strip())
    os.chmod(stop_script_file, 0o755)

    # Trace query script
    query_script = """#!/usr/bin/env python3
'''Query traces from Jaeger'''

import requests
import sys
import json
from datetime import datetime, timedelta

def query_traces(service=None, operation=None, tags=None, limit=20, lookback_hours=1):
    '''Query traces from Jaeger API'''

    jaeger_url = "http://localhost:16686"
    api_url = f"{jaeger_url}/api/traces"

    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=lookback_hours)

    params = {
        'end': int(end_time.timestamp() * 1000000),  # microseconds
        'start': int(start_time.timestamp() * 1000000),
        'limit': limit,
        'lookback': f'{lookback_hours}h'
    }

    if service:
        params['service'] = service

    if operation:
        params['operation'] = operation

    if tags:
        params['tags'] = json.dumps(tags)

    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()

        traces = response.json()

        print(f"Found {len(traces.get('data', []))} traces")
        print("-" * 50)

        for trace in traces.get('data', []):
            trace_id = trace['traceID']
            spans = trace['spans']
            duration = trace['spans'][0]['duration'] / 1000  # Convert to ms

            print(f"Trace ID: {trace_id}")
            print(f"Duration: {duration:.2f}ms")
            print(f"Spans: {len(spans)}")

            # Show root span info
            root_span = min(spans, key=lambda s: s['startTime'])
            print(f"Operation: {root_span['operationName']}")
            print(f"Service: {root_span['process']['serviceName']}")
            print()

    except requests.RequestException as e:
        print(f"Error querying Jaeger: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Query traces from Jaeger')
    parser.add_argument('--service', help='Service name filter')
    parser.add_argument('--operation', help='Operation name filter')
    parser.add_argument('--tags', help='Tags filter (JSON format)')
    parser.add_argument('--limit', type=int, default=20, help='Number of traces to return')
    parser.add_argument('--hours', type=int, default=1, help='Lookback hours')

    args = parser.parse_args()

    tags = None
    if args.tags:
        try:
            tags = json.loads(args.tags)
        except json.JSONDecodeError:
            print("Invalid JSON format for tags")
            sys.exit(1)

    query_traces(args.service, args.operation, tags, args.limit, args.hours)
"""

    query_script_file = scripts_dir / "query_traces.py"
    with open(query_script_file, "w") as f:
        f.write(query_script.strip())
    os.chmod(query_script_file, 0o755)

    print(f"Tracing management scripts created: {scripts_dir}")


def create_test_application():
    """Create a test application with tracing."""
    test_app = """#!/usr/bin/env python3
'''Test application for distributed tracing'''

import asyncio
import logging
import random
import time
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.monorepo.infrastructure.tracing.distributed_tracer import (
        initialize_distributed_tracing,
        trace_function,
        trace_async_function,
        trace_operation,
        SpanType
    )
except ImportError:
    logger.error("Could not import distributed tracer. Make sure it's installed.")
    sys.exit(1)

# Initialize tracing
config = {
    "service_name": "pynomaly-test",
    "service_version": "1.0.0",
    "environment": "test",
    "backends": ["console", "jaeger"],
    "jaeger": {
        "agent_host": "localhost",
        "agent_port": 6831
    }
}

tracer = initialize_distributed_tracing(config)

@trace_function("authenticate_user", SpanType.AUTHENTICATION)
def authenticate_user(username: str, password: str) -> bool:
    '''Simulate user authentication'''
    time.sleep(random.uniform(0.01, 0.05))  # Simulate auth delay

    # Simulate success/failure
    success = random.random() > 0.1  # 90% success rate

    if not success:
        raise ValueError("Authentication failed")

    return True

@trace_async_function("fetch_user_data", SpanType.DATABASE_QUERY)
async def fetch_user_data(user_id: int) -> dict:
    '''Simulate database query'''
    await asyncio.sleep(random.uniform(0.02, 0.1))  # Simulate DB latency

    return {
        "user_id": user_id,
        "username": f"user_{user_id}",
        "email": f"user_{user_id}@example.com",
        "preferences": {"theme": "dark", "language": "en"}
    }

@trace_async_function("ml_inference", SpanType.ML_INFERENCE)
async def run_ml_inference(input_data: List[float]) -> dict:
    '''Simulate ML model inference'''
    await asyncio.sleep(random.uniform(0.1, 0.3))  # Simulate model latency

    # Simulate prediction
    prediction = sum(input_data) / len(input_data)
    confidence = random.uniform(0.7, 0.99)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "model_version": "v1.2.3"
    }

@trace_function("process_request", SpanType.BUSINESS_LOGIC)
def process_request(user_id: int, request_data: dict) -> dict:
    '''Main request processing function'''

    # Step 1: Authentication
    with trace_operation("validate_request", SpanType.BUSINESS_LOGIC) as span:
        span.set_attribute("user_id", user_id)
        span.set_attribute("request_size", len(str(request_data)))

        if not request_data.get("username") or not request_data.get("password"):
            span.set_attribute("validation_error", "Missing credentials")
            raise ValueError("Missing username or password")

    # Authenticate user
    auth_success = authenticate_user(
        request_data["username"],
        request_data["password"]
    )

    if not auth_success:
        raise ValueError("Authentication failed")

    return {"status": "authenticated", "user_id": user_id}

async def async_process_request(user_id: int, request_data: dict) -> dict:
    '''Async request processing'''

    with trace_operation("async_request_processing", SpanType.BUSINESS_LOGIC) as span:
        span.set_attribute("user_id", user_id)
        span.set_attribute("async", True)

        # Step 1: Process sync authentication
        auth_result = process_request(user_id, request_data)

        # Step 2: Fetch user data asynchronously
        user_data = await fetch_user_data(user_id)

        # Step 3: Run ML inference if data provided
        ml_result = None
        if "ml_input" in request_data:
            ml_result = await run_ml_inference(request_data["ml_input"])

        # Step 4: Simulate external API call
        with trace_operation("external_api_call", SpanType.EXTERNAL_API) as api_span:
            api_span.set_attribute("api.endpoint", "https://api.example.com/data")
            api_span.set_attribute("http.method", "GET")

            await asyncio.sleep(random.uniform(0.05, 0.2))  # Simulate API latency

            # Simulate occasional API failures
            if random.random() < 0.05:  # 5% failure rate
                api_span.set_attribute("error", True)
                raise Exception("External API temporarily unavailable")

            api_span.set_attribute("http.status_code", 200)

        return {
            "auth": auth_result,
            "user_data": user_data,
            "ml_result": ml_result,
            "external_data": {"status": "success"}
        }

async def main():
    '''Main test function'''
    logger.info("Starting distributed tracing test application")

    # Simulate multiple concurrent requests
    tasks = []

    for i in range(10):
        request_data = {
            "username": f"user_{i}",
            "password": "test_password",
            "ml_input": [random.random() for _ in range(5)]
        }

        task = async_process_request(i, request_data)
        tasks.append(task)

    # Execute requests with some concurrency
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if not isinstance(r, Exception))
        error_count = len(results) - success_count

        logger.info(f"Completed {len(results)} requests: {success_count} successful, {error_count} errors")

        # Print sample results
        for i, result in enumerate(results[:3]):  # Show first 3 results
            if isinstance(result, Exception):
                logger.error(f"Request {i} failed: {result}")
            else:
                logger.info(f"Request {i} succeeded: {result['auth']['status']}")

    except Exception as e:
        logger.error(f"Test failed: {e}")

    # Get tracing metrics
    metrics = tracer.get_metrics_summary()
    logger.info(f"Tracing metrics: {metrics}")

    # Health check
    health = tracer.health_check()
    logger.info(f"Tracing health: {health['status']}")

    logger.info("Distributed tracing test completed")

if __name__ == "__main__":
    asyncio.run(main())
"""

    test_dir = Path("/opt/pynomaly/scripts/tracing")
    test_file = test_dir / "test_tracing.py"

    with open(test_file, "w") as f:
        f.write(test_app.strip())
    os.chmod(test_file, 0o755)

    print(f"Test application created: {test_file}")


def main():
    """Main setup function."""
    print("Setting up Pynomaly distributed tracing infrastructure...")
    print("=" * 60)

    try:
        # Create Docker Compose configuration
        compose_file = create_jaeger_docker_compose()

        # Create OTEL Collector configuration
        otel_config = create_otel_collector_config()

        # Create Prometheus configuration
        prometheus_config = create_prometheus_config()

        # Create Grafana configuration
        create_grafana_config()

        # Create management scripts
        create_tracing_scripts()

        # Create test application
        create_test_application()

        print("\n" + "=" * 60)
        print("✅ Distributed tracing infrastructure setup completed!")
        print("\nNext steps:")
        print("1. Start the tracing infrastructure:")
        print("   ./scripts/tracing/start_tracing.sh")
        print("\n2. Test the tracing setup:")
        print("   python3 ./scripts/tracing/test_tracing.py")
        print("\n3. Access the UIs:")
        print("   - Jaeger: http://localhost:16686")
        print("   - Grafana: http://localhost:3000 (admin/admin123)")
        print("   - Prometheus: http://localhost:9090")
        print("\n4. Query traces:")
        print("   ./scripts/tracing/query_traces.py --service pynomaly --hours 1")
        print("\n5. Stop the infrastructure:")
        print("   ./scripts/tracing/stop_tracing.sh")

        print("\nKey files created:")
        print(f"- Docker Compose: {compose_file}")
        print(f"- OTEL Collector config: {otel_config}")
        print(f"- Prometheus config: {prometheus_config}")
        print("- Grafana configs: /opt/pynomaly/docker/tracing/grafana/")
        print("- Management scripts: /opt/pynomaly/scripts/tracing/")

    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
