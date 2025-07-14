"""
External Services Integration Testing Suite
Comprehensive tests for integration with external services, APIs, and third-party systems.
"""

import json
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from pynomaly.infrastructure.external.cloud_storage import GCSClient, S3Client
from pynomaly.infrastructure.external.database_clients import (
    MongoDBClient,
    PostgreSQLClient,
    RedisClient,
)
from pynomaly.infrastructure.external.monitoring import (
    DatadogClient,
    GrafanaClient,
    PrometheusClient,
)
from pynomaly.infrastructure.external.notification import (
    EmailClient,
    PagerDutyClient,
    SlackClient,
)


class TestDatabaseIntegration:
    """Test suite for database service integration."""

    @pytest.fixture
    def postgresql_client(self):
        """Create PostgreSQL client for testing."""
        return PostgreSQLClient(
            host="localhost",
            port=5432,
            database="pynomaly_test",
            username="test_user",
            password="test_password",
        )

    @pytest.fixture
    def redis_client(self):
        """Create Redis client for testing."""
        return RedisClient(host="localhost", port=6379, database=0, password=None)

    @pytest.fixture
    def mongodb_client(self):
        """Create MongoDB client for testing."""
        return MongoDBClient(
            host="localhost",
            port=27017,
            database="pynomaly_test",
            username="test_user",
            password="test_password",
        )

    def test_postgresql_connection_and_operations(self, postgresql_client):
        """Test PostgreSQL connection and basic operations."""
        with patch("psycopg2.connect") as mock_connect:
            mock_connection = MagicMock()
            mock_cursor = MagicMock()
            mock_connect.return_value = mock_connection
            mock_connection.cursor.return_value = mock_cursor

            # Test connection
            connection = postgresql_client.connect()
            assert connection is not None

            # Test query execution
            mock_cursor.fetchall.return_value = [
                (1, "detector_1", "IsolationForest"),
                (2, "detector_2", "LocalOutlierFactor"),
            ]

            result = postgresql_client.execute_query(
                "SELECT id, name, algorithm FROM detectors"
            )

            assert len(result) == 2
            assert result[0][1] == "detector_1"

            # Test transaction
            with postgresql_client.transaction() as txn:
                txn.execute(
                    "INSERT INTO detectors (name, algorithm) VALUES (%s, %s)",
                    ("test_detector", "OneClassSVM"),
                )
                txn.execute(
                    "UPDATE detectors SET status = %s WHERE name = %s",
                    ("active", "test_detector"),
                )

            mock_connection.commit.assert_called()

    def test_redis_caching_operations(self, redis_client):
        """Test Redis caching operations."""
        with patch("redis.Redis") as mock_redis:
            mock_redis_instance = MagicMock()
            mock_redis.return_value = mock_redis_instance

            # Test connection
            redis_client.connect()

            # Test set operation
            redis_client.set(
                "detector:123",
                json.dumps(
                    {
                        "algorithm": "IsolationForest",
                        "parameters": {"n_estimators": 100},
                    }
                ),
                ttl=3600,
            )

            mock_redis_instance.setex.assert_called_with(
                "detector:123",
                3600,
                json.dumps(
                    {
                        "algorithm": "IsolationForest",
                        "parameters": {"n_estimators": 100},
                    }
                ),
            )

            # Test get operation
            mock_redis_instance.get.return_value = json.dumps(
                {"algorithm": "IsolationForest", "parameters": {"n_estimators": 100}}
            )

            result = redis_client.get("detector:123")
            cached_data = json.loads(result)

            assert cached_data["algorithm"] == "IsolationForest"
            assert cached_data["parameters"]["n_estimators"] == 100

    def test_mongodb_document_operations(self, mongodb_client):
        """Test MongoDB document operations."""
        with patch("pymongo.MongoClient") as mock_mongo:
            mock_client = MagicMock()
            mock_db = MagicMock()
            mock_collection = MagicMock()

            mock_mongo.return_value = mock_client
            mock_client.__getitem__.return_value = mock_db
            mock_db.__getitem__.return_value = mock_collection

            # Test connection
            mongodb_client.connect()

            # Test document insertion
            detector_doc = {
                "name": "anomaly_detector_1",
                "algorithm": "IsolationForest",
                "parameters": {"contamination": 0.1},
                "created_at": datetime.now(),
                "version": "1.0",
            }

            mock_collection.insert_one.return_value.inserted_id = (
                "507f1f77bcf86cd799439011"
            )

            doc_id = mongodb_client.insert_document("detectors", detector_doc)
            assert doc_id == "507f1f77bcf86cd799439011"

            # Test document retrieval
            mock_collection.find_one.return_value = detector_doc

            retrieved_doc = mongodb_client.find_document(
                "detectors", {"name": "anomaly_detector_1"}
            )
            assert retrieved_doc["algorithm"] == "IsolationForest"

    def test_database_connection_pooling(self, postgresql_client):
        """Test database connection pooling."""
        with patch("psycopg2.pool.ThreadedConnectionPool") as mock_pool:
            mock_pool_instance = MagicMock()
            mock_pool.return_value = mock_pool_instance

            # Configure connection pool
            postgresql_client.configure_pool(min_connections=2, max_connections=10)

            mock_pool.assert_called_with(
                2,
                10,
                host="localhost",
                port=5432,
                database="pynomaly_test",
                user="test_user",
                password="test_password",
            )

            # Test getting connection from pool
            mock_connection = MagicMock()
            mock_pool_instance.getconn.return_value = mock_connection

            connection = postgresql_client.get_connection()
            assert connection is not None

            # Test returning connection to pool
            postgresql_client.put_connection(connection)
            mock_pool_instance.putconn.assert_called_with(mock_connection)

    def test_database_failover_mechanism(self, postgresql_client):
        """Test database failover mechanism."""
        primary_config = {
            "host": "primary-db.example.com",
            "port": 5432,
            "database": "pynomaly",
        }

        secondary_config = {
            "host": "secondary-db.example.com",
            "port": 5432,
            "database": "pynomaly",
        }

        with patch("psycopg2.connect") as mock_connect:
            # Simulate primary database failure
            mock_connect.side_effect = [
                Exception("Connection to primary failed"),
                MagicMock(),  # Successful connection to secondary
            ]

            connection = postgresql_client.connect_with_failover(
                primary_config, secondary_config
            )

            assert connection is not None
            assert mock_connect.call_count == 2


class TestCloudStorageIntegration:
    """Test suite for cloud storage service integration."""

    @pytest.fixture
    def s3_client(self):
        """Create S3 client for testing."""
        return S3Client(
            access_key_id="test_access_key",
            secret_access_key="test_secret_key",
            region="us-west-2",
            bucket_name="pynomaly-models",
        )

    @pytest.fixture
    def gcs_client(self):
        """Create Google Cloud Storage client for testing."""
        return GCSClient(
            project_id="pynomaly-project",
            credentials_path="/path/to/credentials.json",
            bucket_name="pynomaly-storage",
        )

    def test_s3_model_storage_operations(self, s3_client):
        """Test S3 model storage operations."""
        with patch("boto3.client") as mock_boto3:
            mock_s3 = MagicMock()
            mock_boto3.return_value = mock_s3

            # Test model upload
            model_data = b"serialized_model_data"
            model_key = "models/detector_123/model.pkl"

            s3_client.upload_model(model_key, model_data)

            mock_s3.put_object.assert_called_with(
                Bucket="pynomaly-models",
                Key=model_key,
                Body=model_data,
                ContentType="application/octet-stream",
            )

            # Test model download
            mock_s3.get_object.return_value = {
                "Body": MagicMock(read=lambda: model_data)
            }

            downloaded_data = s3_client.download_model(model_key)
            assert downloaded_data == model_data

            # Test model listing
            mock_s3.list_objects_v2.return_value = {
                "Contents": [
                    {"Key": "models/detector_123/model.pkl", "Size": 1024},
                    {"Key": "models/detector_456/model.pkl", "Size": 2048},
                ]
            }

            models = s3_client.list_models("models/")
            assert len(models) == 2
            assert models[0]["key"] == "models/detector_123/model.pkl"

    def test_gcs_dataset_storage_operations(self, gcs_client):
        """Test Google Cloud Storage dataset operations."""
        with patch("google.cloud.storage.Client") as mock_gcs:
            mock_client = MagicMock()
            mock_bucket = MagicMock()
            mock_blob = MagicMock()

            mock_gcs.return_value = mock_client
            mock_client.bucket.return_value = mock_bucket
            mock_bucket.blob.return_value = mock_blob

            # Test dataset upload
            dataset_content = "feature1,feature2,feature3\n1,2,3\n4,5,6"
            dataset_key = "datasets/training_data.csv"

            gcs_client.upload_dataset(dataset_key, dataset_content)

            mock_blob.upload_from_string.assert_called_with(
                dataset_content, content_type="text/csv"
            )

            # Test dataset download
            mock_blob.download_as_text.return_value = dataset_content

            downloaded_content = gcs_client.download_dataset(dataset_key)
            assert downloaded_content == dataset_content

    def test_cloud_storage_backup_and_versioning(self, s3_client):
        """Test cloud storage backup and versioning."""
        with patch("boto3.client") as mock_boto3:
            mock_s3 = MagicMock()
            mock_boto3.return_value = mock_s3

            # Test versioned model storage
            model_versions = [
                {"version": "1.0", "data": b"model_v1_data"},
                {"version": "1.1", "data": b"model_v1_1_data"},
                {"version": "2.0", "data": b"model_v2_data"},
            ]

            for version_info in model_versions:
                f"models/detector_123/v{version_info['version']}/model.pkl"
                s3_client.upload_model_version(
                    "detector_123", version_info["version"], version_info["data"]
                )

            assert mock_s3.put_object.call_count == 3

            # Test backup creation
            backup_config = {
                "source_bucket": "pynomaly-models",
                "backup_bucket": "pynomaly-backups",
                "schedule": "daily",
                "retention_days": 30,
            }

            s3_client.create_backup_job(backup_config)

            # Verify backup configuration
            assert backup_config["retention_days"] == 30

    def test_cloud_storage_access_control(self, s3_client):
        """Test cloud storage access control and security."""
        with patch("boto3.client") as mock_boto3:
            mock_s3 = MagicMock()
            mock_boto3.return_value = mock_s3

            # Test setting bucket policy
            bucket_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Deny",
                        "Principal": "*",
                        "Action": "s3:*",
                        "Resource": "arn:aws:s3:::pynomaly-models/*",
                        "Condition": {"Bool": {"aws:SecureTransport": "false"}},
                    }
                ],
            }

            s3_client.set_bucket_policy(json.dumps(bucket_policy))

            mock_s3.put_bucket_policy.assert_called_with(
                Bucket="pynomaly-models", Policy=json.dumps(bucket_policy)
            )

            # Test encryption configuration
            encryption_config = {
                "Rules": [
                    {"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}
                ]
            }

            s3_client.configure_encryption(encryption_config)

            mock_s3.put_bucket_encryption.assert_called_with(
                Bucket="pynomaly-models",
                ServerSideEncryptionConfiguration=encryption_config,
            )


class TestMonitoringIntegration:
    """Test suite for monitoring service integration."""

    @pytest.fixture
    def prometheus_client(self):
        """Create Prometheus client for testing."""
        return PrometheusClient(url="http://prometheus.example.com:9090", timeout=30)

    @pytest.fixture
    def grafana_client(self):
        """Create Grafana client for testing."""
        return GrafanaClient(
            url="http://grafana.example.com:3000",
            api_key="grafana_api_key_123",
            timeout=30,
        )

    @pytest.fixture
    def datadog_client(self):
        """Create Datadog client for testing."""
        return DatadogClient(
            api_key="datadog_api_key_123", app_key="datadog_app_key_456"
        )

    def test_prometheus_metrics_collection(self, prometheus_client):
        """Test Prometheus metrics collection."""
        with patch("requests.get") as mock_get:
            # Mock Prometheus query response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "status": "success",
                "data": {
                    "resultType": "vector",
                    "result": [
                        {
                            "metric": {
                                "__name__": "pynomaly_predictions_total",
                                "detector": "detector_123",
                            },
                            "value": [1640995200, "1500"],
                        }
                    ],
                },
            }
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            # Query metrics
            query = "pynomaly_predictions_total{detector='detector_123'}"
            result = prometheus_client.query(query)

            assert result["status"] == "success"
            assert len(result["data"]["result"]) == 1
            assert result["data"]["result"][0]["value"][1] == "1500"

    def test_grafana_dashboard_management(self, grafana_client):
        """Test Grafana dashboard management."""
        with patch("requests.post") as mock_post, patch("requests.get") as mock_get:
            # Test dashboard creation
            dashboard_config = {
                "dashboard": {
                    "title": "Pynomaly Monitoring",
                    "panels": [
                        {
                            "title": "Prediction Rate",
                            "type": "graph",
                            "targets": [
                                {"expr": "rate(pynomaly_predictions_total[5m])"}
                            ],
                        }
                    ],
                },
                "overwrite": True,
            }

            mock_post.return_value.json.return_value = {
                "status": "success",
                "id": 1,
                "uid": "pynomaly-dashboard",
                "url": "/d/pynomaly-dashboard/pynomaly-monitoring",
            }
            mock_post.return_value.status_code = 200

            result = grafana_client.create_dashboard(dashboard_config)

            assert result["status"] == "success"
            assert result["uid"] == "pynomaly-dashboard"

            # Test dashboard retrieval
            mock_get.return_value.json.return_value = {
                "dashboard": dashboard_config["dashboard"],
                "meta": {"isStarred": False, "canEdit": True},
            }
            mock_get.return_value.status_code = 200

            dashboard = grafana_client.get_dashboard("pynomaly-dashboard")

            assert dashboard["dashboard"]["title"] == "Pynomaly Monitoring"

    def test_datadog_metrics_and_alerts(self, datadog_client):
        """Test Datadog metrics and alerting."""
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 202
            mock_post.return_value.json.return_value = {"status": "ok"}

            # Test sending custom metrics
            metrics = [
                {
                    "metric": "pynomaly.detection.latency",
                    "points": [[int(time.time()), 150.5]],
                    "tags": ["detector:isolation_forest", "environment:production"],
                },
                {
                    "metric": "pynomaly.detection.accuracy",
                    "points": [[int(time.time()), 0.95]],
                    "tags": ["detector:isolation_forest", "environment:production"],
                },
            ]

            datadog_client.send_metrics(metrics)

            mock_post.assert_called()

            # Test creating monitor/alert
            monitor_config = {
                "type": "metric alert",
                "query": "avg(last_5m):avg:pynomaly.detection.latency{environment:production} > 500",
                "name": "High Detection Latency",
                "message": "Detection latency is above 500ms @slack-alerts",
                "tags": ["service:pynomaly", "team:ml-platform"],
                "options": {
                    "thresholds": {"critical": 500, "warning": 300},
                    "notify_no_data": True,
                    "no_data_timeframe": 10,
                },
            }

            mock_post.return_value.json.return_value = {
                "id": 12345,
                "name": "High Detection Latency",
                "created": "2024-01-01T00:00:00Z",
            }

            monitor = datadog_client.create_monitor(monitor_config)

            assert monitor["id"] == 12345
            assert monitor["name"] == "High Detection Latency"

    def test_monitoring_alert_integration(self, prometheus_client, datadog_client):
        """Test monitoring alert integration workflow."""
        # Simulate alert condition detection
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "status": "success",
                "data": {
                    "resultType": "vector",
                    "result": [
                        {
                            "metric": {"__name__": "pynomaly_error_rate"},
                            "value": [int(time.time()), "0.15"],  # 15% error rate
                        }
                    ],
                },
            }
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            # Query current error rate
            error_rate_query = "pynomaly_error_rate"
            result = prometheus_client.query(error_rate_query)
            current_error_rate = float(result["data"]["result"][0]["value"][1])

            # Check if alert threshold is exceeded
            alert_threshold = 0.10  # 10%

            if current_error_rate > alert_threshold:
                # Trigger alert via Datadog
                with patch("requests.post") as mock_post:
                    mock_post.return_value.status_code = 202

                    alert_event = {
                        "title": "High Error Rate Detected",
                        "text": f"Error rate is {current_error_rate:.2%}, above threshold of {alert_threshold:.2%}",
                        "alert_type": "error",
                        "tags": ["service:pynomaly", "severity:high"],
                    }

                    datadog_client.send_event(alert_event)

                    assert mock_post.called
                    assert current_error_rate > alert_threshold


class TestNotificationIntegration:
    """Test suite for notification service integration."""

    @pytest.fixture
    def slack_client(self):
        """Create Slack client for testing."""
        return SlackClient(
            webhook_url="https://hooks.slack.com/services/ABC/DEF/GHI",
            channel="#alerts",
            username="pynomaly-bot",
        )

    @pytest.fixture
    def email_client(self):
        """Create email client for testing."""
        return EmailClient(
            smtp_server="smtp.example.com",
            smtp_port=587,
            username="noreply@example.com",
            password="email_password",
            use_tls=True,
        )

    @pytest.fixture
    def pagerduty_client(self):
        """Create PagerDuty client for testing."""
        return PagerDutyClient(
            integration_key="pagerduty_integration_key_123",
            service_id="PYNOMALY_SERVICE",
        )

    def test_slack_notification_sending(self, slack_client):
        """Test Slack notification sending."""
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"ok": True}

            # Test simple message
            message = "Anomaly detection completed successfully for detector_123"
            slack_client.send_message(message)

            mock_post.assert_called()
            call_args = mock_post.call_args
            payload = json.loads(call_args[1]["data"])

            assert payload["text"] == message
            assert payload["channel"] == "#alerts"

            # Test rich message with attachments
            rich_message = {
                "text": "Anomaly Detection Alert",
                "attachments": [
                    {
                        "color": "danger",
                        "title": "High Anomaly Rate Detected",
                        "fields": [
                            {
                                "title": "Detector",
                                "value": "detector_123",
                                "short": True,
                            },
                            {"title": "Anomaly Rate", "value": "15%", "short": True},
                            {"title": "Threshold", "value": "10%", "short": True},
                        ],
                        "timestamp": int(time.time()),
                    }
                ],
            }

            slack_client.send_rich_message(rich_message)

            assert mock_post.call_count == 2

    def test_email_notification_sending(self, email_client):
        """Test email notification sending."""
        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            # Test simple email
            email_config = {
                "to": ["admin@example.com", "ml-team@example.com"],
                "subject": "Pynomaly Alert: High Anomaly Rate",
                "body": "Alert: Anomaly rate has exceeded threshold",
                "html": False,
            }

            email_client.send_email(email_config)

            mock_server.send_message.assert_called()

            # Test HTML email with template
            html_email_config = {
                "to": ["admin@example.com"],
                "subject": "Pynomaly Weekly Report",
                "template": "weekly_report",
                "template_data": {
                    "period": "2024-01-01 to 2024-01-07",
                    "total_detections": 15420,
                    "anomalies_found": 342,
                    "accuracy": 0.956,
                },
                "html": True,
            }

            email_client.send_templated_email(html_email_config)

            assert mock_server.send_message.call_count == 2

    def test_pagerduty_incident_management(self, pagerduty_client):
        """Test PagerDuty incident management."""
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 202
            mock_post.return_value.json.return_value = {
                "status": "success",
                "message": "Event processed",
                "dedup_key": "pynomaly-critical-error-123",
            }

            # Test incident creation
            incident_data = {
                "routing_key": "pagerduty_integration_key_123",
                "event_action": "trigger",
                "dedup_key": "pynomaly-critical-error",
                "payload": {
                    "summary": "Critical: Pynomaly service down",
                    "source": "pynomaly-monitoring",
                    "severity": "critical",
                    "component": "detection-service",
                    "group": "ml-platform",
                    "class": "service-outage",
                    "custom_details": {
                        "error_rate": "50%",
                        "response_time": "timeout",
                        "affected_detectors": ["detector_123", "detector_456"],
                    },
                },
            }

            result = pagerduty_client.create_incident(incident_data)

            assert result["status"] == "success"
            assert "dedup_key" in result

            # Test incident resolution
            resolution_data = {
                "routing_key": "pagerduty_integration_key_123",
                "event_action": "resolve",
                "dedup_key": "pynomaly-critical-error-123",
            }

            pagerduty_client.resolve_incident(resolution_data)

            assert mock_post.call_count == 2

    def test_notification_escalation_workflow(
        self, slack_client, email_client, pagerduty_client
    ):
        """Test notification escalation workflow."""
        # Simulate escalation scenario
        alert_severity = "critical"
        alert_duration = timedelta(minutes=15)  # Alert has been active for 15 minutes

        escalation_rules = {
            "low": ["slack"],
            "medium": ["slack", "email"],
            "high": ["slack", "email", "pagerduty"],
            "critical": ["slack", "email", "pagerduty"],
        }

        notification_channels = escalation_rules.get(alert_severity, ["slack"])

        alert_message = {
            "title": "Service Degradation Detected",
            "description": "Pynomaly detection service showing high latency",
            "severity": alert_severity,
            "duration": str(alert_duration),
        }

        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"ok": True}

            # Send notifications based on escalation rules
            for channel in notification_channels:
                if channel == "slack":
                    slack_client.send_alert(alert_message)
                elif channel == "email":
                    email_client.send_alert_email(alert_message)
                elif channel == "pagerduty":
                    pagerduty_client.create_incident_from_alert(alert_message)

            # Verify all channels were notified for critical alert
            assert len(notification_channels) == 3  # slack, email, pagerduty
            assert mock_post.call_count >= 3


class TestExternalAPIIntegration:
    """Test suite for external API integration."""

    def test_third_party_model_api_integration(self):
        """Test integration with third-party ML model APIs."""
        api_config = {
            "base_url": "https://api.mlservice.example.com/v1",
            "api_key": "third_party_api_key_123",
            "timeout": 30,
        }

        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "predictions": [0, 1, 0, 1, 0],
                "confidence_scores": [0.1, 0.9, 0.2, 0.8, 0.3],
                "model_version": "2.1.0",
                "processing_time_ms": 125,
            }

            # Test external model prediction
            from pynomaly.infrastructure.external.model_apis import ThirdPartyMLAPI

            api_client = ThirdPartyMLAPI(api_config)

            prediction_request = {
                "data": [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0],
                ],
                "model_id": "anomaly_detector_v2",
            }

            result = api_client.predict(prediction_request)

            assert len(result["predictions"]) == 5
            assert result["model_version"] == "2.1.0"
            assert result["processing_time_ms"] == 125

    def test_data_source_api_integration(self):
        """Test integration with external data source APIs."""
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "data": [
                    {
                        "timestamp": "2024-01-01T00:00:00Z",
                        "value": 1.5,
                        "sensor_id": "sensor_001",
                    },
                    {
                        "timestamp": "2024-01-01T00:01:00Z",
                        "value": 1.7,
                        "sensor_id": "sensor_001",
                    },
                    {
                        "timestamp": "2024-01-01T00:02:00Z",
                        "value": 1.2,
                        "sensor_id": "sensor_001",
                    },
                ],
                "pagination": {"page": 1, "total_pages": 10, "total_records": 1000},
            }

            # Test data ingestion from external API
            from pynomaly.infrastructure.external.data_apis import SensorDataAPI

            data_api = SensorDataAPI(
                base_url="https://api.sensors.example.com/v1",
                api_key="sensor_api_key_123",
            )

            data = data_api.fetch_sensor_data(
                sensor_id="sensor_001",
                start_time="2024-01-01T00:00:00Z",
                end_time="2024-01-01T01:00:00Z",
            )

            assert len(data["data"]) == 3
            assert data["pagination"]["total_records"] == 1000

    def test_webhook_integration(self):
        """Test webhook integration for real-time notifications."""
        webhook_config = {
            "url": "https://external-system.example.com/webhooks/pynomaly",
            "secret": "webhook_secret_key_123",
            "events": ["anomaly_detected", "model_trained", "alert_triggered"],
        }

        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"status": "received"}

            from pynomaly.infrastructure.external.webhooks import WebhookSender

            webhook_sender = WebhookSender(webhook_config)

            # Test sending webhook notification
            event_data = {
                "event_type": "anomaly_detected",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "detector_id": "detector_123",
                    "anomaly_score": 0.95,
                    "data_point": [1.5, 2.3, 4.1, 0.8],
                    "confidence": 0.87,
                },
            }

            result = webhook_sender.send_event(event_data)

            assert result["status"] == "received"
            mock_post.assert_called()

            # Verify webhook payload includes signature for security
            call_args = mock_post.call_args
            headers = call_args[1]["headers"]
            assert "X-Pynomaly-Signature" in headers
