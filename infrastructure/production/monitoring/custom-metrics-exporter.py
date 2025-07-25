#!/usr/bin/env python3
"""
Custom Metrics Exporter for MLOps Platform

This script exports business, security, and ML-specific metrics to Prometheus
for comprehensive monitoring and alerting.
"""

import time
import logging
import asyncio
import psycopg2
import redis
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info
import requests
from sqlalchemy import create_engine, text
from kubernetes import client, config
import ssl
import socket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
business_metrics = {
    'model_predictions_total': Counter('model_predictions_total', 'Total number of model predictions', ['model_id', 'model_type']),
    'revenue_generated_total': Counter('revenue_generated_total', 'Total revenue generated', ['source']),
    'active_users_gauge': Gauge('active_users_gauge', 'Number of active users'),
    'model_accuracy': Gauge('model_accuracy', 'Model accuracy score', ['model_id', 'model_type']),
    'data_quality_score': Gauge('data_quality_score', 'Data quality score', ['dataset_name']),
    'user_actions_total': Counter('user_actions_total', 'Total user actions', ['action_type']),
    'resource_cost_total': Gauge('resource_cost_total', 'Resource cost by type', ['resource_type']),
    'data_drift_score': Gauge('data_drift_score', 'Data drift detection score', ['model_id'])
}

security_metrics = {
    'security_alerts_total': Counter('security_alerts_total', 'Total security alerts', ['severity', 'type']),
    'failed_login_attempts_total': Counter('failed_login_attempts_total', 'Failed login attempts', ['source_ip']),
    'ssl_certificate_expiry_days': Gauge('ssl_certificate_expiry_days', 'Days until SSL certificate expiry', ['domain']),
    'vulnerability_scan_issues': Gauge('vulnerability_scan_issues', 'Number of vulnerability issues', ['severity']),
    'authentication_events_total': Counter('authentication_events_total', 'Authentication events', ['event_type']),
    'security_policy_violations_total': Counter('security_policy_violations_total', 'Security policy violations', ['policy_type']),
    'network_security_events_total': Counter('network_security_events_total', 'Network security events', ['source_ip', 'event_type']),
    'access_control_events_total': Counter('access_control_events_total', 'Access control events', ['action']),
    'data_encryption_status': Gauge('data_encryption_status', 'Data encryption status', ['encryption_type']),
    'compliance_score': Gauge('compliance_score', 'Compliance framework score', ['framework'])
}

ml_metrics = {
    'model_prediction_duration_seconds': Histogram('model_prediction_duration_seconds', 'Model prediction duration', ['model_id']),
    'model_training_duration_seconds': Histogram('model_training_duration_seconds', 'Model training duration', ['model_id']),
    'model_deployment_status': Gauge('model_deployment_status', 'Model deployment status', ['model_id', 'version']),
    'feature_store_operations_total': Counter('feature_store_operations_total', 'Feature store operations', ['operation_type']),
    'pipeline_execution_duration_seconds': Histogram('pipeline_execution_duration_seconds', 'Pipeline execution duration', ['pipeline_id']),
    'data_processing_volume_bytes': Counter('data_processing_volume_bytes', 'Data processing volume', ['stage'])
}

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str
    port: int
    database: str
    username: str
    password: str

@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str
    port: int
    password: Optional[str] = None

class MetricsCollector:
    """Collects metrics from various sources and exports to Prometheus"""
    
    def __init__(self):
        self.db_config = self._get_database_config()
        self.redis_config = self._get_redis_config()
        self.db_engine = None
        self.redis_client = None
        self.k8s_client = None
        
    def _get_database_config(self) -> DatabaseConfig:
        """Get database configuration from environment"""
        return DatabaseConfig(
            host=os.getenv('DB_HOST', 'postgres'),
            port=int(os.getenv('DB_PORT', 5432)),
            database=os.getenv('DB_NAME', 'mlops_prod'),
            username=os.getenv('DB_USER', 'mlops'),
            password=os.getenv('DB_PASSWORD', 'mlops123')
        )
    
    def _get_redis_config(self) -> RedisConfig:
        """Get Redis configuration from environment"""
        return RedisConfig(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            password=os.getenv('REDIS_PASSWORD')
        )
    
    async def initialize(self):
        """Initialize connections to data sources"""
        try:
            # Initialize database connection
            db_url = f"postgresql://{self.db_config.username}:{self.db_config.password}@{self.db_config.host}:{self.db_config.port}/{self.db_config.database}"
            self.db_engine = create_engine(db_url)
            logger.info("Database connection initialized")
            
            # Initialize Redis connection
            self.redis_client = redis.Redis(
                host=self.redis_config.host,
                port=self.redis_config.port,
                password=self.redis_config.password,
                decode_responses=True
            )
            logger.info("Redis connection initialized")
            
            # Initialize Kubernetes client
            try:
                config.load_incluster_config()
            except:
                config.load_kube_config()
            self.k8s_client = client.CoreV1Api()
            logger.info("Kubernetes client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            raise
    
    async def collect_business_metrics(self):
        """Collect business metrics from database"""
        try:
            with self.db_engine.connect() as conn:
                # Model predictions
                result = conn.execute(text("""
                    SELECT model_id, model_type, COUNT(*) as prediction_count
                    FROM model_predictions 
                    WHERE created_at >= NOW() - INTERVAL '1 hour'
                    GROUP BY model_id, model_type
                """))
                
                for row in result:
                    business_metrics['model_predictions_total'].labels(
                        model_id=row.model_id,
                        model_type=row.model_type
                    ).inc(row.prediction_count)
                
                # Revenue metrics
                result = conn.execute(text("""
                    SELECT source, SUM(amount) as total_revenue
                    FROM revenue_events 
                    WHERE created_at >= NOW() - INTERVAL '1 hour'
                    GROUP BY source
                """))
                
                for row in result:
                    business_metrics['revenue_generated_total'].labels(
                        source=row.source
                    ).inc(row.total_revenue)
                
                # Active users
                result = conn.execute(text("""
                    SELECT COUNT(DISTINCT user_id) as active_users
                    FROM user_sessions 
                    WHERE last_activity >= NOW() - INTERVAL '1 hour'
                """))
                
                row = result.fetchone()
                if row:
                    business_metrics['active_users_gauge'].set(row.active_users)
                
                # Model accuracy
                result = conn.execute(text("""
                    SELECT model_id, model_type, AVG(accuracy_score) as avg_accuracy
                    FROM model_evaluations 
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                    GROUP BY model_id, model_type
                """))
                
                for row in result:
                    business_metrics['model_accuracy'].labels(
                        model_id=row.model_id,
                        model_type=row.model_type
                    ).set(row.avg_accuracy)
                
                # Data quality scores
                result = conn.execute(text("""
                    SELECT dataset_name, AVG(quality_score) as avg_quality
                    FROM data_quality_metrics 
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                    GROUP BY dataset_name
                """))
                
                for row in result:
                    business_metrics['data_quality_score'].labels(
                        dataset_name=row.dataset_name
                    ).set(row.avg_quality)
                
                logger.info("Business metrics collected successfully")
                
        except Exception as e:
            logger.error(f"Failed to collect business metrics: {e}")
    
    async def collect_security_metrics(self):
        """Collect security metrics from various sources"""
        try:
            # SSL certificate expiry
            domains = ['api.mlops-platform.com', 'app.mlops-platform.com', 'monitoring.mlops-platform.com']
            for domain in domains:
                try:
                    expiry_days = self._check_ssl_expiry(domain)
                    security_metrics['ssl_certificate_expiry_days'].labels(domain=domain).set(expiry_days)
                except Exception as e:
                    logger.warning(f"Failed to check SSL for {domain}: {e}")
            
            # Authentication metrics from Redis
            try:
                failed_logins = self.redis_client.get('failed_logins_1h') or 0
                security_metrics['failed_login_attempts_total'].labels(source_ip='aggregate').inc(int(failed_logins))
            except Exception as e:
                logger.warning(f"Failed to get Redis metrics: {e}")
            
            # Security metrics from database
            with self.db_engine.connect() as conn:
                # Security alerts
                result = conn.execute(text("""
                    SELECT severity, alert_type, COUNT(*) as alert_count
                    FROM security_alerts 
                    WHERE created_at >= NOW() - INTERVAL '1 hour'
                    GROUP BY severity, alert_type
                """))
                
                for row in result:
                    security_metrics['security_alerts_total'].labels(
                        severity=row.severity,
                        type=row.alert_type
                    ).inc(row.alert_count)
                
                # Authentication events
                result = conn.execute(text("""
                    SELECT event_type, COUNT(*) as event_count
                    FROM authentication_events 
                    WHERE created_at >= NOW() - INTERVAL '1 hour'
                    GROUP BY event_type
                """))
                
                for row in result:
                    security_metrics['authentication_events_total'].labels(
                        event_type=row.event_type
                    ).inc(row.event_count)
                
                # Compliance scores
                result = conn.execute(text("""
                    SELECT framework, AVG(score) as avg_score
                    FROM compliance_assessments 
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                    GROUP BY framework
                """))
                
                for row in result:
                    security_metrics['compliance_score'].labels(
                        framework=row.framework
                    ).set(row.avg_score)
            
            logger.info("Security metrics collected successfully")
            
        except Exception as e:
            logger.error(f"Failed to collect security metrics: {e}")
    
    def _check_ssl_expiry(self, domain: str) -> int:
        """Check SSL certificate expiry for a domain"""
        try:
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    expiry_date = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                    days_until_expiry = (expiry_date - datetime.now()).days
                    return days_until_expiry
        except Exception as e:
            logger.warning(f"Failed to check SSL for {domain}: {e}")
            return -1
    
    async def collect_ml_metrics(self):
        """Collect ML-specific metrics"""
        try:
            with self.db_engine.connect() as conn:
                # Model prediction latency
                result = conn.execute(text("""
                    SELECT model_id, AVG(prediction_duration_ms) as avg_duration
                    FROM model_prediction_logs 
                    WHERE created_at >= NOW() - INTERVAL '1 hour'
                    GROUP BY model_id
                """))
                
                for row in result:
                    ml_metrics['model_prediction_duration_seconds'].labels(
                        model_id=row.model_id
                    ).observe(row.avg_duration / 1000.0)
                
                # Model deployment status
                result = conn.execute(text("""
                    SELECT model_id, version, 
                           CASE WHEN status = 'active' THEN 1 ELSE 0 END as is_active
                    FROM model_deployments
                """))
                
                for row in result:
                    ml_metrics['model_deployment_status'].labels(
                        model_id=row.model_id,
                        version=row.version
                    ).set(row.is_active)
                
                # Pipeline execution metrics
                result = conn.execute(text("""
                    SELECT pipeline_id, AVG(execution_duration_seconds) as avg_duration
                    FROM pipeline_executions 
                    WHERE created_at >= NOW() - INTERVAL '1 hour'
                    GROUP BY pipeline_id
                """))
                
                for row in result:
                    ml_metrics['pipeline_execution_duration_seconds'].labels(
                        pipeline_id=row.pipeline_id
                    ).observe(row.avg_duration)
                
                # Data processing volume
                result = conn.execute(text("""
                    SELECT processing_stage, SUM(data_volume_bytes) as total_bytes
                    FROM data_processing_logs 
                    WHERE created_at >= NOW() - INTERVAL '1 hour'
                    GROUP BY processing_stage
                """))
                
                for row in result:
                    ml_metrics['data_processing_volume_bytes'].labels(
                        stage=row.processing_stage
                    ).inc(row.total_bytes)
            
            logger.info("ML metrics collected successfully")
            
        except Exception as e:
            logger.error(f"Failed to collect ML metrics: {e}")
    
    async def collect_kubernetes_metrics(self):
        """Collect Kubernetes-specific metrics"""
        try:
            # Get pod status in production namespace
            pods = self.k8s_client.list_namespaced_pod(namespace='mlops-production')
            
            running_pods = 0
            failed_pods = 0
            pending_pods = 0
            
            for pod in pods.items:
                if pod.status.phase == 'Running':
                    running_pods += 1
                elif pod.status.phase == 'Failed':
                    failed_pods += 1
                elif pod.status.phase == 'Pending':
                    pending_pods += 1
            
            # Export as gauge metrics (would need to define these)
            logger.info(f"Kubernetes metrics: Running={running_pods}, Failed={failed_pods}, Pending={pending_pods}")
            
        except Exception as e:
            logger.error(f"Failed to collect Kubernetes metrics: {e}")
    
    async def run_collection_cycle(self):
        """Run a complete metrics collection cycle"""
        logger.info("Starting metrics collection cycle")
        
        try:
            await asyncio.gather(
                self.collect_business_metrics(),
                self.collect_security_metrics(),
                self.collect_ml_metrics(),
                self.collect_kubernetes_metrics(),
                return_exceptions=True
            )
            logger.info("Metrics collection cycle completed")
        except Exception as e:
            logger.error(f"Metrics collection cycle failed: {e}")

async def main():
    """Main function to run the metrics exporter"""
    logger.info("Starting MLOps Custom Metrics Exporter")
    
    # Start Prometheus HTTP server
    metrics_port = int(os.getenv('METRICS_PORT', 8080))
    start_http_server(metrics_port)
    logger.info(f"Prometheus metrics server started on port {metrics_port}")
    
    # Initialize metrics collector
    collector = MetricsCollector()
    await collector.initialize()
    
    # Collection interval in seconds
    collection_interval = int(os.getenv('COLLECTION_INTERVAL', 60))
    
    logger.info(f"Starting metrics collection loop (interval: {collection_interval}s)")
    
    while True:
        try:
            await collector.run_collection_cycle()
            await asyncio.sleep(collection_interval)
        except KeyboardInterrupt:
            logger.info("Metrics exporter stopped by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            await asyncio.sleep(30)  # Wait before retrying

if __name__ == "__main__":
    asyncio.run(main())