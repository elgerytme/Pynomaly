# Advanced Deployment Scenarios

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 🚀 [Deployment](README.md) > 📄 Advanced Deployment

---


This guide covers sophisticated deployment patterns for Pynomaly in enterprise environments, including multi-region setups, microservices architecture, edge computing, and hybrid cloud deployments.

## Table of Contents

1. [Multi-Region High Availability](#multi-region-high-availability)
2. [Microservices Architecture](#microservices-architecture)
3. [Edge Computing Deployment](#edge-computing-deployment)
4. [Serverless and Auto-Scaling](#serverless-and-auto-scaling)
5. [Hybrid Cloud Integration](#hybrid-cloud-integration)
6. [Zero-Downtime Deployment](#zero-downtime-deployment)
7. [Advanced Monitoring](#advanced-monitoring)
8. [Disaster Recovery](#disaster-recovery)

## Multi-Region High Availability

Deploy Pynomaly across multiple regions for maximum availability and performance with built-in resilience patterns.

### Global Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   US-East-1     │    │   EU-West-1     │    │   Asia-Pacific  │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │  Pynomaly   │ │    │ │  Pynomaly   │ │    │ │  Pynomaly   │ │
│ │  Primary    │ │    │ │  Replica    │ │    │ │  Replica    │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │  Primary    │ │    │ │  Read       │ │    │ │  Read       │ │
│ │  Database   │ │    │ │  Replica    │ │    │ │  Replica    │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Global Load    │
                    │  Balancer       │
                    │  (CloudFlare)   │
                    └─────────────────┘
```

### Global Load Balancer Configuration

```yaml
# global-load-balancer.yaml
apiVersion: networking.gke.io/v1
kind: ManagedCertificate
metadata:
  name: pynomaly-ssl-cert
spec:
  domains:
    - api.pynomaly.com
    - *.pynomaly.com
---
apiVersion: compute.googleapis.com/v1
kind: GlobalAddress
metadata:
  name: pynomaly-global-ip
  annotations:
    resilience.pynomaly.com/region-fallback: "enabled"
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pynomaly-global-ingress
  annotations:
    kubernetes.io/ingress.global-static-ip-name: pynomaly-global-ip
    networking.gke.io/managed-certificates: pynomaly-ssl-cert
    kubernetes.io/ingress.class: gce
    kubernetes.io/ingress.allow-http: "false"
    # Circuit breaker configuration
    nginx.ingress.kubernetes.io/upstream-max-fails: "3"
    nginx.ingress.kubernetes.io/upstream-fail-timeout: "30s"
spec:
  rules:
  - host: api.pynomaly.com
    http:
      paths:
      - path: /*
        pathType: ImplementationSpecific
        backend:
          service:
            name: pynomaly-api
            port:
              number: 80
```

### Cross-Region Database Replication

```sql
-- Primary region database (us-east-1)
-- Enable logical replication
ALTER SYSTEM SET wal_level = logical;
ALTER SYSTEM SET max_replication_slots = 4;
ALTER SYSTEM SET max_wal_senders = 4;

-- Create publication for all tables
CREATE PUBLICATION pynomaly_replication FOR ALL TABLES;

-- Create replication user
CREATE USER replication_user REPLICATION PASSWORD 'secure_password';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO replication_user;

-- Read replica setup (eu-west-1)
CREATE SUBSCRIPTION pynomaly_replica_eu
CONNECTION 'host=primary-db.us-east-1.rds.amazonaws.com port=5432 user=replication_user dbname=pynomaly sslmode=require'
PUBLICATION pynomaly_replication;

-- Read replica setup (asia-pacific)
CREATE SUBSCRIPTION pynomaly_replica_ap
CONNECTION 'host=primary-db.us-east-1.rds.amazonaws.com port=5432 user=replication_user dbname=pynomaly sslmode=require'
PUBLICATION pynomaly_replication;
```

### Region-Specific Configuration with Resilience

```python
# config/regions.py
from pynomaly.infrastructure.resilience import ml_resilient, database_resilient, api_resilient
import asyncio
from typing import Dict, Any

REGION_CONFIGS = {
    "us-east-1": {
        "database_url": "postgresql://user:pass@primary-db-us-east-1/pynomaly",
        "redis_url": "redis://cache-us-east-1:6379",
        "model_storage": "s3://pynomaly-models-us-east-1",
        "backup_region": "us-west-2",
        "role": "primary",
        "circuit_breaker_threshold": 5,
        "timeout_seconds": 300
    },
    "eu-west-1": {
        "database_url": "postgresql://user:pass@replica-db-eu-west-1/pynomaly",
        "redis_url": "redis://cache-eu-west-1:6379",
        "model_storage": "s3://pynomaly-models-eu-west-1",
        "backup_region": "eu-central-1",
        "role": "replica",
        "circuit_breaker_threshold": 3,
        "timeout_seconds": 200
    },
    "asia-pacific-1": {
        "database_url": "postgresql://user:pass@replica-db-ap-1/pynomaly",
        "redis_url": "redis://cache-ap-1:6379",
        "model_storage": "s3://pynomaly-models-ap-1",
        "backup_region": "asia-pacific-2",
        "role": "replica",
        "circuit_breaker_threshold": 3,
        "timeout_seconds": 200
    }
}

class RegionalConfigManager:
    """Manages regional configuration with built-in resilience."""

    def __init__(self, region: str):
        self.region = region
        self.config = REGION_CONFIGS.get(region)
        if not self.config:
            raise ValueError(f"Unknown region: {region}")

    @ml_resilient(timeout_seconds=300, max_attempts=2)
    async def get_regional_config(self) -> Dict[str, Any]:
        """Get configuration for specific region with resilience."""
        return self.config.copy()

    @database_resilient(timeout_seconds=30, max_attempts=3)
    async def test_database_connection(self) -> bool:
        """Test database connectivity with resilience patterns."""
        from sqlalchemy import create_engine
        try:
            engine = create_engine(self.config['database_url'])
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception as e:
            print(f"Database connection failed: {e}")
            return False

    @api_resilient(timeout_seconds=10, max_attempts=2)
    async def test_cache_connection(self) -> bool:
        """Test cache connectivity with resilience patterns."""
        import redis
        try:
            r = redis.from_url(self.config['redis_url'])
            r.ping()
            return True
        except Exception as e:
            print(f"Cache connection failed: {e}")
            return False

    async def health_check(self) -> Dict[str, bool]:
        """Comprehensive health check for region."""
        return {
            'database': await self.test_database_connection(),
            'cache': await self.test_cache_connection(),
            'region': self.region,
            'role': self.config['role']
        }

# Usage example
async def setup_regional_deployment():
    import os
    region = os.getenv('AWS_REGION', 'us-east-1')

    config_manager = RegionalConfigManager(region)
    config = await config_manager.get_regional_config()
    health = await config_manager.health_check()

    print(f"Region: {region}")
    print(f"Role: {config['role']}")
    print(f"Health: {health}")

    return config_manager
```

### Global Model Synchronization

```python
# models/global_sync.py
from pynomaly.infrastructure.resilience import ml_resilient, api_resilient
import asyncio
import boto3
from datetime import datetime
import json

class GlobalModelSynchronizer:
    """Synchronize trained models across regions with resilience."""

    def __init__(self, region: str, config_manager: RegionalConfigManager):
        self.region = region
        self.config_manager = config_manager
        self.s3_client = boto3.client('s3')
        self.sync_interval = 300  # 5 minutes

    @ml_resilient(timeout_seconds=600, max_attempts=2)
    async def sync_models_globally(self):
        """Synchronize models across all regions."""
        config = await self.config_manager.get_regional_config()

        if config['role'] == 'primary':
            await self._push_models_to_replicas()
        else:
            await self._pull_models_from_primary()

    async def _push_models_to_replicas(self):
        """Push models from primary to replica regions."""
        config = await self.config_manager.get_regional_config()
        primary_bucket = config['model_storage'].replace('s3://', '')

        # List all models in primary region
        models = self._list_models(primary_bucket)

        for region_config in REGION_CONFIGS.values():
            if region_config['role'] == 'replica':
                replica_bucket = region_config['model_storage'].replace('s3://', '')
                await self._copy_models_to_bucket(models, primary_bucket, replica_bucket)

    async def _pull_models_from_primary(self):
        """Pull latest models from primary region."""
        # Find primary region
        primary_config = next(
            config for config in REGION_CONFIGS.values()
            if config['role'] == 'primary'
        )

        primary_bucket = primary_config['model_storage'].replace('s3://', '')
        local_config = await self.config_manager.get_regional_config()
        local_bucket = local_config['model_storage'].replace('s3://', '')

        # Get latest models from primary
        models = self._list_models(primary_bucket)
        latest_models = self._filter_latest_models(models)

        await self._copy_models_to_bucket(latest_models, primary_bucket, local_bucket)

    @api_resilient(timeout_seconds=30, max_attempts=3)
    async def _copy_models_to_bucket(self, models: list, source_bucket: str, dest_bucket: str):
        """Copy models between S3 buckets with resilience."""
        for model in models:
            try:
                copy_source = {'Bucket': source_bucket, 'Key': model['Key']}
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.s3_client.copy_object,
                    copy_source,
                    dest_bucket,
                    model['Key']
                )
                print(f"Copied model {model['Key']} to {dest_bucket}")
            except Exception as e:
                print(f"Failed to copy model {model['Key']}: {e}")
                # Circuit breaker will handle retries

    def _list_models(self, bucket: str) -> list:
        """List all models in S3 bucket."""
        try:
            response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix='models/')
            return response.get('Contents', [])
        except Exception as e:
            print(f"Failed to list models in {bucket}: {e}")
            return []

    def _filter_latest_models(self, models: list, hours: int = 24) -> list:
        """Filter models updated in the last N hours."""
        from datetime import datetime, timedelta

        cutoff_time = datetime.now() - timedelta(hours=hours)

        return [
            model for model in models
            if model.get('LastModified', datetime.min).replace(tzinfo=None) > cutoff_time
        ]
```

## Microservices Architecture

Deploy Pynomaly as a collection of microservices for better scalability and maintainability.

### Service Mesh with Istio

```yaml
# istio-gateway.yaml
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: pynomaly-gateway
  annotations:
    resilience.pynomaly.com/circuit-breaker: "enabled"
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 443
      name: https
      protocol: HTTPS
    tls:
      mode: SIMPLE
      credentialName: pynomaly-tls
    hosts:
    - api.pynomaly.com
---
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: pynomaly-routes
spec:
  hosts:
  - api.pynomaly.com
  gateways:
  - pynomaly-gateway
  http:
  # Detector service with circuit breaker
  - match:
    - uri:
        prefix: /api/v1/detectors
    route:
    - destination:
        host: detector-service
        port:
          number: 8000
    fault:
      abort:
        percentage:
          value: 0.1
        httpStatus: 503
    timeout: 30s
    retries:
      attempts: 3
      perTryTimeout: 10s
      retryOn: 5xx,reset,connect-failure,refused-stream

  # Dataset service with load balancing
  - match:
    - uri:
        prefix: /api/v1/datasets
    route:
    - destination:
        host: dataset-service
        port:
          number: 8001
    timeout: 60s

  # Model service with caching
  - match:
    - uri:
        prefix: /api/v1/models
    route:
    - destination:
        host: model-service
        port:
          number: 8002
    headers:
      response:
        add:
          cache-control: "max-age=300"
```

### Microservice Deployment with Resilience

```yaml
# detector-service.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: detector-service
  labels:
    app: detector-service
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: detector-service
  template:
    metadata:
      labels:
        app: detector-service
        version: v1
      annotations:
        sidecar.istio.io/inject: "true"
        # Resilience configuration
        resilience.pynomaly.com/circuit-breaker-threshold: "5"
        resilience.pynomaly.com/timeout-seconds: "300"
        resilience.pynomaly.com/retry-max-attempts: "3"
    spec:
      containers:
      - name: detector-service
        image: pynomaly/detector-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: SERVICE_NAME
          value: "detector-service"
        - name: RESILIENCE_ENABLED
          value: "true"
        - name: CIRCUIT_BREAKER_ENABLED
          value: "true"
        - name: CIRCUIT_BREAKER_THRESHOLD
          value: "5"
        - name: TIMEOUT_SECONDS
          value: "300"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        resources:
          limits:
            cpu: 1000m
            memory: 1Gi
          requests:
            cpu: 500m
            memory: 512Mi
```

### Service Communication with Circuit Breakers

```python
# services/detector_service.py
from pynomaly.infrastructure.resilience import ml_resilient, api_resilient
from fastapi import FastAPI, HTTPException
import asyncio

app = FastAPI(title="Detector Service")

class DetectorService:
    """Microservice for detector management with built-in resilience."""

    def __init__(self):
        self.dataset_service_url = "http://dataset-service:8001"
        self.model_service_url = "http://model-service:8002"

    @ml_resilient(timeout_seconds=300, max_attempts=2)
    async def create_detector(self, detector_config: dict):
        """Create detector with ML resilience patterns."""
        try:
            # Create detector using domain services
            from pynomaly.domain.services import DetectorFactory

            factory = DetectorFactory()
            detector = await factory.create_detector(
                algorithm=detector_config['algorithm'],
                parameters=detector_config.get('parameters', {})
            )

            return {
                "id": detector.id,
                "algorithm": detector.algorithm,
                "parameters": detector.parameters,
                "created_at": detector.created_at.isoformat()
            }

        except Exception as e:
            # Circuit breaker will handle retries
            raise HTTPException(status_code=500, detail=f"Detector creation failed: {str(e)}")

    @api_resilient(timeout_seconds=30, max_attempts=3)
    async def get_dataset_info(self, dataset_id: str):
        """Get dataset information from dataset service with API resilience."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.dataset_service_url}/datasets/{dataset_id}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Dataset service error: {response.status}"
                    )

    @ml_resilient(timeout_seconds=600, max_attempts=1)  # No retry for training
    async def train_detector(self, detector_id: str, dataset_id: str):
        """Train detector with comprehensive resilience."""
        try:
            # Get dataset information
            dataset_info = await self.get_dataset_info(dataset_id)

            # Perform training with timeout protection
            from pynomaly.application.use_cases import TrainDetector

            train_use_case = TrainDetector()
            result = await train_use_case.execute(
                detector_id=detector_id,
                dataset_id=dataset_id
            )

            return {
                "detector_id": detector_id,
                "training_status": "completed",
                "training_time_ms": result.training_time_ms,
                "samples_processed": result.samples_processed
            }

        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Training timeout")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

# FastAPI endpoints
detector_service = DetectorService()

@app.post("/detectors")
async def create_detector(detector_config: dict):
    return await detector_service.create_detector(detector_config)

@app.post("/detectors/{detector_id}/train")
async def train_detector(detector_id: str, dataset_id: str):
    return await detector_service.train_detector(detector_id, dataset_id)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "detector-service"}

@app.get("/ready")
async def readiness_check():
    # Check dependencies
    try:
        # Quick health check of dependent services
        dataset_health = await detector_service.get_dataset_info("health-check")
    except:
        dataset_health = None

    return {
        "ready": dataset_health is not None,
        "dependencies": {
            "dataset_service": dataset_health is not None
        }
    }
```

## Edge Computing Deployment

Deploy lightweight anomaly detection at edge locations for real-time processing.

### Edge Node Configuration

```yaml
# edge-deployment.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: pynomaly-edge
  labels:
    app: pynomaly-edge
spec:
  selector:
    matchLabels:
      app: pynomaly-edge
  template:
    metadata:
      labels:
        app: pynomaly-edge
      annotations:
        # Edge-specific resilience configuration
        resilience.pynomaly.com/offline-mode: "enabled"
        resilience.pynomaly.com/local-cache: "enabled"
        resilience.pynomaly.com/sync-interval: "300"
    spec:
      nodeSelector:
        node-type: edge
      tolerations:
      - key: edge-node
        operator: Exists
        effect: NoSchedule
      containers:
      - name: pynomaly-edge
        image: pynomaly/edge:latest
        resources:
          limits:
            cpu: "0.5"
            memory: "512Mi"
          requests:
            cpu: "0.2"
            memory: "256Mi"
        env:
        - name: EDGE_MODE
          value: "true"
        - name: CENTRAL_API_URL
          value: "https://api.pynomaly.com"
        - name: SYNC_INTERVAL
          value: "300" # 5 minutes
        - name: OFFLINE_MODE
          value: "true"
        - name: LOCAL_CACHE_SIZE
          value: "100MB"
        volumeMounts:
        - name: model-cache
          mountPath: /app/models
        - name: data-buffer
          mountPath: /app/buffer
        - name: edge-config
          mountPath: /app/config
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
      volumes:
      - name: model-cache
        emptyDir:
          sizeLimit: 1Gi
      - name: data-buffer
        emptyDir:
          sizeLimit: 2Gi
      - name: edge-config
        configMap:
          name: edge-config
```

### Edge Synchronization Service

```python
# edge/sync_service.py
from pynomaly.infrastructure.resilience import api_resilient, ml_resilient
import asyncio
import logging
import json
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EdgeSyncService:
    """Service for synchronizing edge nodes with central system."""

    def __init__(self, central_api_url: str, sync_interval: int = 300):
        self.central_api_url = central_api_url
        self.sync_interval = sync_interval
        self.local_models = {}
        self.pending_results = []
        self.offline_mode = os.getenv('OFFLINE_MODE', 'false').lower() == 'true'
        self.last_sync_time = datetime.now()

    async def start_sync_loop(self):
        """Start continuous synchronization loop."""
        while True:
            try:
                await self.sync_cycle()
                await asyncio.sleep(self.sync_interval)
            except Exception as e:
                logger.error(f"Sync cycle failed: {e}")
                await asyncio.sleep(self.sync_interval * 2)  # Back off on error

    async def sync_cycle(self):
        """Perform one complete sync cycle."""
        logger.info("Starting sync cycle...")

        # Try to sync models (download)
        await self.sync_models()

        # Try to upload results
        await self.upload_results()

        # Clean up old data
        await self.cleanup_old_data()

        self.last_sync_time = datetime.now()
        logger.info("Sync cycle completed")

    @api_resilient(timeout_seconds=30, max_attempts=3)
    async def sync_models(self):
        """Sync models from central system with resilience."""
        if self.offline_mode and not await self._test_connectivity():
            logger.info("Offline mode: skipping model sync")
            return

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.central_api_url}/api/v1/models/edge") as response:
                    if response.status == 200:
                        models = await response.json()

                        for model_info in models:
                            if self._should_download_model(model_info):
                                await self.download_model(model_info)
                    else:
                        logger.warning(f"Model sync failed with status {response.status}")

        except Exception as e:
            logger.error(f"Model sync failed: {e}")
            if not self.offline_mode:
                raise  # Let circuit breaker handle retries

    @ml_resilient(timeout_seconds=60, max_attempts=2)
    async def download_model(self, model_info: dict):
        """Download individual model with ML resilience."""
        model_id = model_info['id']
        model_url = model_info['download_url']

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(model_url) as response:
                    if response.status == 200:
                        model_data = await response.read()

                        # Save model locally
                        model_path = f"/app/models/{model_id}.pkl"
                        with open(model_path, 'wb') as f:
                            f.write(model_data)

                        # Update local model registry
                        self.local_models[model_id] = {
                            'path': model_path,
                            'version': model_info['version'],
                            'downloaded_at': datetime.now().isoformat(),
                            'metadata': model_info.get('metadata', {})
                        }

                        logger.info(f"Downloaded model {model_id} version {model_info['version']}")
                    else:
                        logger.error(f"Failed to download model {model_id}: HTTP {response.status}")

        except Exception as e:
            logger.error(f"Model download failed for {model_id}: {e}")
            raise

    @api_resilient(timeout_seconds=60, max_attempts=2)
    async def upload_results(self):
        """Upload pending results to central system."""
        if not self.pending_results:
            return

        if self.offline_mode and not await self._test_connectivity():
            logger.info(f"Offline mode: queuing {len(self.pending_results)} results")
            return

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                upload_data = {
                    "edge_node_id": os.getenv('EDGE_NODE_ID', 'unknown'),
                    "results": self.pending_results[:100],  # Batch upload
                    "timestamp": datetime.now().isoformat()
                }

                async with session.post(
                    f"{self.central_api_url}/api/v1/results/edge",
                    json=upload_data
                ) as response:

                    if response.status == 200:
                        uploaded_count = len(upload_data['results'])
                        self.pending_results = self.pending_results[uploaded_count:]
                        logger.info(f"Uploaded {uploaded_count} results to central system")
                    else:
                        logger.warning(f"Result upload failed with status {response.status}")

        except Exception as e:
            logger.error(f"Result upload failed: {e}")
            # Keep results for next attempt
            if len(self.pending_results) > 10000:  # Prevent memory overflow
                self.pending_results = self.pending_results[-5000:]  # Keep recent results
                logger.warning("Trimmed pending results due to memory constraints")

    async def queue_detection_result(self, result: dict):
        """Queue detection result for upload."""
        result['edge_timestamp'] = datetime.now().isoformat()
        result['edge_node_id'] = os.getenv('EDGE_NODE_ID', 'unknown')

        self.pending_results.append(result)

        # Immediate upload for critical anomalies
        if result.get('anomaly_score', 0) > 0.9:
            await self.upload_critical_result(result)

    @api_resilient(timeout_seconds=10, max_attempts=1)
    async def upload_critical_result(self, result: dict):
        """Immediately upload critical anomaly results."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.central_api_url}/api/v1/alerts/critical",
                    json=result
                ) as response:
                    if response.status == 200:
                        logger.info(f"Critical anomaly uploaded immediately: {result.get('anomaly_score')}")
        except Exception as e:
            logger.warning(f"Critical result upload failed (will retry in batch): {e}")

    def _should_download_model(self, model_info: dict) -> bool:
        """Check if model should be downloaded."""
        model_id = model_info['id']

        if model_id not in self.local_models:
            return True

        local_version = self.local_models[model_id].get('version', 0)
        remote_version = model_info.get('version', 0)

        return remote_version > local_version

    async def _test_connectivity(self) -> bool:
        """Test connectivity to central system."""
        try:
            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{self.central_api_url}/health") as response:
                    return response.status == 200
        except:
            return False

    async def cleanup_old_data(self):
        """Clean up old models and results."""
        # Remove old model files
        cutoff_time = datetime.now() - timedelta(days=7)

        for model_id, model_info in list(self.local_models.items()):
            downloaded_at = datetime.fromisoformat(model_info['downloaded_at'])
            if downloaded_at < cutoff_time:
                try:
                    os.remove(model_info['path'])
                    del self.local_models[model_id]
                    logger.info(f"Cleaned up old model {model_id}")
                except Exception as e:
                    logger.warning(f"Failed to clean up model {model_id}: {e}")

        # Limit pending results
        if len(self.pending_results) > 5000:
            self.pending_results = self.pending_results[-2500:]
            logger.info("Trimmed pending results to manage memory")

# Edge detection service with local models
class EdgeDetectionService:
    """Local anomaly detection service for edge nodes."""

    def __init__(self, sync_service: EdgeSyncService):
        self.sync_service = sync_service
        self.loaded_models = {}

    @ml_resilient(timeout_seconds=30, max_attempts=1)
    async def detect_anomalies(self, data: dict, model_id: str = None):
        """Perform local anomaly detection with resilience."""
        try:
            # Use best available model if none specified
            if model_id is None:
                model_id = self._select_best_model(data)

            # Load model if not in memory
            if model_id not in self.loaded_models:
                await self._load_model(model_id)

            # Perform detection
            model = self.loaded_models[model_id]

            # Convert data to format expected by model
            import numpy as np
            features = np.array(list(data.values())).reshape(1, -1)

            # Run inference
            prediction = model.predict(features)[0]
            score = model.decision_function(features)[0]

            result = {
                'data': data,
                'is_anomaly': bool(prediction == -1),
                'anomaly_score': float(score),
                'model_id': model_id,
                'detected_at': datetime.now().isoformat()
            }

            # Queue result for upload
            await self.sync_service.queue_detection_result(result)

            return result

        except Exception as e:
            logger.error(f"Edge detection failed: {e}")
            # Return safe default
            return {
                'data': data,
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'model_id': None,
                'error': str(e),
                'detected_at': datetime.now().isoformat()
            }

    async def _load_model(self, model_id: str):
        """Load model into memory."""
        if model_id not in self.sync_service.local_models:
            raise ValueError(f"Model {model_id} not available locally")

        model_info = self.sync_service.local_models[model_id]
        model_path = model_info['path']

        import joblib
        model = joblib.load(model_path)
        self.loaded_models[model_id] = model

        logger.info(f"Loaded model {model_id} into memory")

    def _select_best_model(self, data: dict) -> str:
        """Select best available model for given data."""
        # Simple heuristic: use most recent model
        if not self.sync_service.local_models:
            raise ValueError("No models available locally")

        latest_model = max(
            self.sync_service.local_models.items(),
            key=lambda x: x[1]['downloaded_at']
        )

        return latest_model[0]
```

This comprehensive guide demonstrates advanced deployment scenarios with built-in resilience patterns throughout. Each deployment pattern leverages Pynomaly's infrastructure resilience features including circuit breakers, retry mechanisms, and timeout handling for maximum reliability in production environments.

---

## 🔗 **Related Documentation**

### **Getting Started**
- **[Installation Guide](../../getting-started/installation.md)** - Setup and installation
- **[Quick Start](../../getting-started/quickstart.md)** - Your first detection
- **[Platform Setup](../../getting-started/platform-specific/)** - Platform-specific guides

### **User Guides**
- **[Basic Usage](../basic-usage/README.md)** - Essential functionality
- **[Advanced Features](../advanced-features/README.md)** - Sophisticated capabilities  
- **[Troubleshooting](../troubleshooting/README.md)** - Problem solving

### **Reference**
- **[Algorithm Reference](../../reference/algorithms/README.md)** - Algorithm documentation
- **[API Documentation](../../developer-guides/api-integration/README.md)** - Programming interfaces
- **[Configuration](../../reference/configuration/)** - System configuration

### **Examples**
- **[Examples & Tutorials](../../examples/README.md)** - Real-world use cases
- **[Banking Examples](../../examples/banking/)** - Financial fraud detection
- **[Notebooks](../../examples/notebooks/)** - Interactive examples

---

## 🆘 **Getting Help**

- **[Troubleshooting Guide](../troubleshooting/troubleshooting.md)** - Common issues and solutions
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs and request features
- **[GitHub Discussions](https://github.com/your-org/pynomaly/discussions)** - Ask questions and share ideas
- **[Security Issues](mailto:security@pynomaly.org)** - Report security vulnerabilities
