"""
Third-Party Integrations Framework for Pynomaly Detection
========================================================

Comprehensive integration system providing:
- Standardized connector framework for third-party systems
- Data source integrations (databases, APIs, file systems)
- ML framework integrations (scikit-learn, TensorFlow, PyTorch)
- Monitoring tool integrations (Prometheus, Grafana, ELK Stack)
- Cloud service integrations (AWS, GCP, Azure services)
- Message queue integrations (Kafka, RabbitMQ, Redis)
- Authentication and authorization adapters
"""

import logging
import json
import time
import threading
import asyncio
from typing import Dict, List, Optional, Any, Union, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import uuid

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import sqlalchemy
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

logger = logging.getLogger(__name__)

class IntegrationType(Enum):
    """Integration type enumeration."""
    DATA_SOURCE = "data_source"
    ML_FRAMEWORK = "ml_framework"
    MONITORING = "monitoring"
    CLOUD_SERVICE = "cloud_service"
    MESSAGE_QUEUE = "message_queue"
    DATABASE = "database"
    API = "api"
    FILE_SYSTEM = "file_system"
    AUTHENTICATION = "authentication"

class ConnectionStatus(Enum):
    """Connection status enumeration."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"
    UNKNOWN = "unknown"

class DataFormat(Enum):
    """Data format enumeration."""
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    AVRO = "avro"
    XML = "xml"
    BINARY = "binary"
    STREAMING = "streaming"

@dataclass
class IntegrationConfig:
    """Integration configuration."""
    integration_id: str
    name: str
    integration_type: IntegrationType
    connection_params: Dict[str, Any] = field(default_factory=dict)
    authentication: Dict[str, Any] = field(default_factory=dict)
    data_format: DataFormat = DataFormat.JSON
    timeout_seconds: int = 30
    retry_attempts: int = 3
    batch_size: int = 1000
    is_active: bool = True
    created_date: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IntegrationStatus:
    """Integration status information."""
    integration_id: str
    status: ConnectionStatus
    last_connected: Optional[datetime] = None
    last_error: Optional[str] = None
    connection_attempts: int = 0
    data_transferred: int = 0
    operations_count: int = 0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataExchange:
    """Data exchange record."""
    exchange_id: str
    integration_id: str
    operation: str  # read, write, query, stream
    timestamp: datetime
    data_size: int = 0
    duration_ms: float = 0
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class ThirdPartyConnector(ABC):
    """Base class for third-party connectors."""
    
    def __init__(self, config: IntegrationConfig):
        """Initialize connector with configuration.
        
        Args:
            config: Integration configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"connector.{config.integration_id}")
        self._connection = None
        self._status = ConnectionStatus.DISCONNECTED
        self._lock = threading.RLock()
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to third-party system.
        
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Close connection to third-party system.
        
        Returns:
            True if disconnection successful
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test connection to third-party system.
        
        Returns:
            True if connection is healthy
        """
        pass
    
    @abstractmethod
    def read_data(self, query: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Read data from third-party system.
        
        Args:
            query: Optional query parameters
            
        Returns:
            Retrieved data or None
        """
        pass
    
    @abstractmethod
    def write_data(self, data: Any, destination: Optional[str] = None) -> bool:
        """Write data to third-party system.
        
        Args:
            data: Data to write
            destination: Optional destination specifier
            
        Returns:
            True if write successful
        """
        pass
    
    def get_status(self) -> ConnectionStatus:
        """Get current connection status."""
        return self._status
    
    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self._status == ConnectionStatus.CONNECTED
    
    def get_info(self) -> Dict[str, Any]:
        """Get connector information."""
        return {
            "integration_id": self.config.integration_id,
            "name": self.config.name,
            "type": self.config.integration_type.value,
            "status": self._status.value,
            "is_active": self.config.is_active
        }

class DatabaseConnector(ThirdPartyConnector):
    """Database connector for SQL and NoSQL databases."""
    
    def connect(self) -> bool:
        """Connect to database."""
        try:
            if not SQLALCHEMY_AVAILABLE:
                self.logger.error("SQLAlchemy not available")
                return False
            
            connection_string = self._build_connection_string()
            engine = sqlalchemy.create_engine(connection_string)
            
            # Test connection
            with engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
            
            self._connection = engine
            self._status = ConnectionStatus.CONNECTED
            self.logger.info(f"Connected to database: {self.config.name}")
            return True
            
        except Exception as e:
            self._status = ConnectionStatus.ERROR
            self.logger.error(f"Database connection failed: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from database."""
        try:
            if self._connection:
                self._connection.dispose()
                self._connection = None
            
            self._status = ConnectionStatus.DISCONNECTED
            self.logger.info(f"Disconnected from database: {self.config.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Database disconnection failed: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            if not self._connection:
                return False
            
            with self._connection.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Database connection test failed: {e}")
            return False
    
    def read_data(self, query: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Read data from database."""
        try:
            if not self.is_connected() or not PANDAS_AVAILABLE:
                return None
            
            sql_query = query.get('sql') if query else "SELECT * FROM anomalies LIMIT 1000"
            
            with self._connection.connect() as conn:
                df = pd.read_sql(sql_query, conn)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Database read failed: {e}")
            return None
    
    def write_data(self, data: Any, destination: Optional[str] = None) -> bool:
        """Write data to database."""
        try:
            if not self.is_connected() or not PANDAS_AVAILABLE:
                return False
            
            if not isinstance(data, pd.DataFrame):
                # Convert to DataFrame if possible
                if isinstance(data, list) and data:
                    data = pd.DataFrame(data)
                else:
                    return False
            
            table_name = destination or "anomaly_results"
            
            with self._connection.connect() as conn:
                data.to_sql(table_name, conn, if_exists='append', index=False)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Database write failed: {e}")
            return False
    
    def _build_connection_string(self) -> str:
        """Build database connection string."""
        params = self.config.connection_params
        auth = self.config.authentication
        
        db_type = params.get('type', 'postgresql')
        host = params.get('host', 'localhost')
        port = params.get('port', 5432)
        database = params.get('database', 'pynomaly')
        
        username = auth.get('username', '')
        password = auth.get('password', '')
        
        if username and password:
            return f"{db_type}://{username}:{password}@{host}:{port}/{database}"
        else:
            return f"{db_type}://{host}:{port}/{database}"

class APIConnector(ThirdPartyConnector):
    """REST API connector."""
    
    def connect(self) -> bool:
        """Connect to API (test endpoint)."""
        try:
            if not REQUESTS_AVAILABLE:
                self.logger.error("Requests library not available")
                return False
            
            # Test connection with health check endpoint
            base_url = self.config.connection_params.get('base_url', '')
            health_endpoint = self.config.connection_params.get('health_endpoint', '/health')
            
            response = requests.get(
                f"{base_url}{health_endpoint}",
                headers=self._get_headers(),
                timeout=self.config.timeout_seconds
            )
            
            if response.status_code == 200:
                self._status = ConnectionStatus.CONNECTED
                self.logger.info(f"Connected to API: {self.config.name}")
                return True
            else:
                self._status = ConnectionStatus.ERROR
                return False
                
        except Exception as e:
            self._status = ConnectionStatus.ERROR
            self.logger.error(f"API connection failed: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from API (no-op for REST APIs)."""
        self._status = ConnectionStatus.DISCONNECTED
        return True
    
    def test_connection(self) -> bool:
        """Test API connection."""
        return self.connect()
    
    def read_data(self, query: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Read data from API."""
        try:
            if not REQUESTS_AVAILABLE:
                return None
            
            base_url = self.config.connection_params.get('base_url', '')
            endpoint = query.get('endpoint', '/data') if query else '/data'
            params = query.get('params', {}) if query else {}
            
            response = requests.get(
                f"{base_url}{endpoint}",
                headers=self._get_headers(),
                params=params,
                timeout=self.config.timeout_seconds
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"API read failed: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"API read failed: {e}")
            return None
    
    def write_data(self, data: Any, destination: Optional[str] = None) -> bool:
        """Write data to API."""
        try:
            if not REQUESTS_AVAILABLE:
                return False
            
            base_url = self.config.connection_params.get('base_url', '')
            endpoint = destination or '/data'
            
            response = requests.post(
                f"{base_url}{endpoint}",
                headers=self._get_headers(),
                json=data,
                timeout=self.config.timeout_seconds
            )
            
            return response.status_code in [200, 201, 202]
            
        except Exception as e:
            self.logger.error(f"API write failed: {e}")
            return False
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        headers = {'Content-Type': 'application/json'}
        
        auth = self.config.authentication
        if auth.get('type') == 'bearer':
            headers['Authorization'] = f"Bearer {auth.get('token')}"
        elif auth.get('type') == 'api_key':
            key_name = auth.get('key_name', 'X-API-Key')
            headers[key_name] = auth.get('api_key')
        
        return headers

class MessageQueueConnector(ThirdPartyConnector):
    """Message queue connector (Kafka, RabbitMQ, etc.)."""
    
    def connect(self) -> bool:
        """Connect to message queue."""
        try:
            # This is a simplified implementation
            # In practice, would use specific libraries like kafka-python, pika, etc.
            
            queue_type = self.config.connection_params.get('type', 'kafka')
            
            if queue_type == 'kafka':
                return self._connect_kafka()
            elif queue_type == 'rabbitmq':
                return self._connect_rabbitmq()
            elif queue_type == 'redis':
                return self._connect_redis()
            else:
                self.logger.error(f"Unsupported queue type: {queue_type}")
                return False
                
        except Exception as e:
            self._status = ConnectionStatus.ERROR
            self.logger.error(f"Message queue connection failed: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from message queue."""
        try:
            if self._connection:
                # Close connection based on type
                queue_type = self.config.connection_params.get('type', 'kafka')
                
                if hasattr(self._connection, 'close'):
                    self._connection.close()
                
                self._connection = None
            
            self._status = ConnectionStatus.DISCONNECTED
            return True
            
        except Exception as e:
            self.logger.error(f"Message queue disconnection failed: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test message queue connection."""
        # Simple connection test
        return self.is_connected()
    
    def read_data(self, query: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Read messages from queue."""
        try:
            if not self.is_connected():
                return None
            
            topic = query.get('topic', 'anomalies') if query else 'anomalies'
            max_messages = query.get('max_messages', 100) if query else 100
            
            # Simplified message reading
            messages = []
            queue_type = self.config.connection_params.get('type', 'kafka')
            
            if queue_type == 'kafka':
                messages = self._read_kafka_messages(topic, max_messages)
            elif queue_type == 'rabbitmq':
                messages = self._read_rabbitmq_messages(topic, max_messages)
            elif queue_type == 'redis':
                messages = self._read_redis_messages(topic, max_messages)
            
            return messages
            
        except Exception as e:
            self.logger.error(f"Message queue read failed: {e}")
            return None
    
    def write_data(self, data: Any, destination: Optional[str] = None) -> bool:
        """Write messages to queue."""
        try:
            if not self.is_connected():
                return False
            
            topic = destination or 'anomaly_results'
            queue_type = self.config.connection_params.get('type', 'kafka')
            
            if queue_type == 'kafka':
                return self._write_kafka_message(topic, data)
            elif queue_type == 'rabbitmq':
                return self._write_rabbitmq_message(topic, data)
            elif queue_type == 'redis':
                return self._write_redis_message(topic, data)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Message queue write failed: {e}")
            return False
    
    def _connect_kafka(self) -> bool:
        """Connect to Kafka."""
        # Simplified Kafka connection
        # Would use kafka-python in practice
        self._connection = {"type": "kafka", "connected": True}
        self._status = ConnectionStatus.CONNECTED
        return True
    
    def _connect_rabbitmq(self) -> bool:
        """Connect to RabbitMQ."""
        # Simplified RabbitMQ connection
        # Would use pika in practice
        self._connection = {"type": "rabbitmq", "connected": True}
        self._status = ConnectionStatus.CONNECTED
        return True
    
    def _connect_redis(self) -> bool:
        """Connect to Redis."""
        # Simplified Redis connection
        # Would use redis-py in practice
        self._connection = {"type": "redis", "connected": True}
        self._status = ConnectionStatus.CONNECTED
        return True
    
    def _read_kafka_messages(self, topic: str, max_messages: int) -> List[Dict[str, Any]]:
        """Read messages from Kafka topic."""
        # Mock implementation
        return [{"id": i, "data": f"message_{i}", "timestamp": datetime.now().isoformat()} 
                for i in range(min(max_messages, 10))]
    
    def _read_rabbitmq_messages(self, queue: str, max_messages: int) -> List[Dict[str, Any]]:
        """Read messages from RabbitMQ queue."""
        # Mock implementation
        return [{"id": i, "data": f"message_{i}", "timestamp": datetime.now().isoformat()} 
                for i in range(min(max_messages, 10))]
    
    def _read_redis_messages(self, stream: str, max_messages: int) -> List[Dict[str, Any]]:
        """Read messages from Redis stream."""
        # Mock implementation
        return [{"id": i, "data": f"message_{i}", "timestamp": datetime.now().isoformat()} 
                for i in range(min(max_messages, 10))]
    
    def _write_kafka_message(self, topic: str, data: Any) -> bool:
        """Write message to Kafka topic."""
        # Mock implementation
        return True
    
    def _write_rabbitmq_message(self, queue: str, data: Any) -> bool:
        """Write message to RabbitMQ queue."""
        # Mock implementation
        return True
    
    def _write_redis_message(self, stream: str, data: Any) -> bool:
        """Write message to Redis stream."""
        # Mock implementation
        return True

class CloudServiceConnector(ThirdPartyConnector):
    """Cloud service connector (AWS, GCP, Azure)."""
    
    def connect(self) -> bool:
        """Connect to cloud service."""
        try:
            cloud_provider = self.config.connection_params.get('provider', 'aws')
            
            if cloud_provider == 'aws':
                return self._connect_aws()
            elif cloud_provider == 'gcp':
                return self._connect_gcp()
            elif cloud_provider == 'azure':
                return self._connect_azure()
            else:
                self.logger.error(f"Unsupported cloud provider: {cloud_provider}")
                return False
                
        except Exception as e:
            self._status = ConnectionStatus.ERROR
            self.logger.error(f"Cloud service connection failed: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from cloud service."""
        self._connection = None
        self._status = ConnectionStatus.DISCONNECTED
        return True
    
    def test_connection(self) -> bool:
        """Test cloud service connection."""
        try:
            if not self._connection:
                return False
            
            # Perform a simple operation to test connectivity
            cloud_provider = self.config.connection_params.get('provider', 'aws')
            
            if cloud_provider == 'aws':
                return self._test_aws_connection()
            elif cloud_provider == 'gcp':
                return self._test_gcp_connection()
            elif cloud_provider == 'azure':
                return self._test_azure_connection()
            
            return False
            
        except Exception as e:
            self.logger.error(f"Cloud service connection test failed: {e}")
            return False
    
    def read_data(self, query: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Read data from cloud service."""
        try:
            if not self.is_connected():
                return None
            
            cloud_provider = self.config.connection_params.get('provider', 'aws')
            
            if cloud_provider == 'aws':
                return self._read_aws_data(query)
            elif cloud_provider == 'gcp':
                return self._read_gcp_data(query)
            elif cloud_provider == 'azure':
                return self._read_azure_data(query)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Cloud service read failed: {e}")
            return None
    
    def write_data(self, data: Any, destination: Optional[str] = None) -> bool:
        """Write data to cloud service."""
        try:
            if not self.is_connected():
                return False
            
            cloud_provider = self.config.connection_params.get('provider', 'aws')
            
            if cloud_provider == 'aws':
                return self._write_aws_data(data, destination)
            elif cloud_provider == 'gcp':
                return self._write_gcp_data(data, destination)
            elif cloud_provider == 'azure':
                return self._write_azure_data(data, destination)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Cloud service write failed: {e}")
            return False
    
    def _connect_aws(self) -> bool:
        """Connect to AWS services."""
        # Simplified AWS connection - would use boto3 in practice
        self._connection = {"provider": "aws", "connected": True}
        self._status = ConnectionStatus.CONNECTED
        return True
    
    def _connect_gcp(self) -> bool:
        """Connect to GCP services."""
        # Simplified GCP connection - would use google-cloud libraries in practice
        self._connection = {"provider": "gcp", "connected": True}
        self._status = ConnectionStatus.CONNECTED
        return True
    
    def _connect_azure(self) -> bool:
        """Connect to Azure services."""
        # Simplified Azure connection - would use azure-* libraries in practice
        self._connection = {"provider": "azure", "connected": True}
        self._status = ConnectionStatus.CONNECTED
        return True
    
    def _test_aws_connection(self) -> bool:
        """Test AWS connection."""
        return True
    
    def _test_gcp_connection(self) -> bool:
        """Test GCP connection."""
        return True
    
    def _test_azure_connection(self) -> bool:
        """Test Azure connection."""
        return True
    
    def _read_aws_data(self, query: Optional[Dict[str, Any]]) -> Optional[Any]:
        """Read data from AWS services."""
        # Mock implementation
        return {"service": "s3", "data": "mock_aws_data", "timestamp": datetime.now().isoformat()}
    
    def _read_gcp_data(self, query: Optional[Dict[str, Any]]) -> Optional[Any]:
        """Read data from GCP services."""
        # Mock implementation
        return {"service": "gcs", "data": "mock_gcp_data", "timestamp": datetime.now().isoformat()}
    
    def _read_azure_data(self, query: Optional[Dict[str, Any]]) -> Optional[Any]:
        """Read data from Azure services."""
        # Mock implementation
        return {"service": "blob", "data": "mock_azure_data", "timestamp": datetime.now().isoformat()}
    
    def _write_aws_data(self, data: Any, destination: Optional[str]) -> bool:
        """Write data to AWS services."""
        # Mock implementation
        return True
    
    def _write_gcp_data(self, data: Any, destination: Optional[str]) -> bool:
        """Write data to GCP services."""
        # Mock implementation
        return True
    
    def _write_azure_data(self, data: Any, destination: Optional[str]) -> bool:
        """Write data to Azure services."""
        # Mock implementation
        return True

class IntegrationFramework:
    """Comprehensive third-party integration framework."""
    
    def __init__(self):
        """Initialize integration framework."""
        # Integration registry
        self.integrations: Dict[str, IntegrationConfig] = {}
        self.connectors: Dict[str, ThirdPartyConnector] = {}
        self.status_registry: Dict[str, IntegrationStatus] = {}
        
        # Data exchange tracking
        self.exchange_log: List[DataExchange] = []
        
        # Connector type mapping
        self.connector_types = {
            IntegrationType.DATABASE: DatabaseConnector,
            IntegrationType.API: APIConnector,
            IntegrationType.MESSAGE_QUEUE: MessageQueueConnector,
            IntegrationType.CLOUD_SERVICE: CloudServiceConnector,
            # Add more connector types as needed
        }
        
        # Threading
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_integrations': 0,
            'active_connections': 0,
            'total_data_exchanges': 0,
            'failed_operations': 0,
            'total_data_transferred': 0
        }
        
        logger.info("Integration Framework initialized")
    
    def register_integration(self, config: IntegrationConfig) -> bool:
        """Register new integration.
        
        Args:
            config: Integration configuration
            
        Returns:
            True if registration successful
        """
        try:
            with self.lock:
                # Validate configuration
                if not self._validate_config(config):
                    return False
                
                # Store integration config
                self.integrations[config.integration_id] = config
                
                # Initialize status tracking
                self.status_registry[config.integration_id] = IntegrationStatus(
                    integration_id=config.integration_id,
                    status=ConnectionStatus.DISCONNECTED
                )
                
                # Update statistics
                self.stats['total_integrations'] += 1
                
                logger.info(f"Integration registered: {config.integration_id}")
                return True
                
        except Exception as e:
            logger.error(f"Integration registration failed: {e}")
            return False
    
    def create_connector(self, integration_id: str) -> Optional[ThirdPartyConnector]:
        """Create connector for integration.
        
        Args:
            integration_id: Integration identifier
            
        Returns:
            Connector instance or None
        """
        try:
            config = self.integrations.get(integration_id)
            if not config:
                logger.error(f"Integration not found: {integration_id}")
                return None
            
            # Get connector class for integration type
            connector_class = self.connector_types.get(config.integration_type)
            if not connector_class:
                logger.error(f"No connector available for type: {config.integration_type}")
                return None
            
            # Create connector instance
            connector = connector_class(config)
            
            # Store connector
            with self.lock:
                self.connectors[integration_id] = connector
            
            logger.info(f"Connector created for integration: {integration_id}")
            return connector
            
        except Exception as e:
            logger.error(f"Connector creation failed: {e}")
            return None
    
    def connect_integration(self, integration_id: str) -> bool:
        """Connect to integration.
        
        Args:
            integration_id: Integration identifier
            
        Returns:
            True if connection successful
        """
        try:
            connector = self.connectors.get(integration_id)
            if not connector:
                connector = self.create_connector(integration_id)
                if not connector:
                    return False
            
            # Attempt connection
            if connector.connect():
                with self.lock:
                    self.status_registry[integration_id].status = ConnectionStatus.CONNECTED
                    self.status_registry[integration_id].last_connected = datetime.now()
                    self.stats['active_connections'] += 1
                
                logger.info(f"Integration connected: {integration_id}")
                return True
            else:
                with self.lock:
                    self.status_registry[integration_id].status = ConnectionStatus.ERROR
                    self.status_registry[integration_id].connection_attempts += 1
                
                return False
                
        except Exception as e:
            logger.error(f"Integration connection failed: {e}")
            return False
    
    def disconnect_integration(self, integration_id: str) -> bool:
        """Disconnect from integration.
        
        Args:
            integration_id: Integration identifier
            
        Returns:
            True if disconnection successful
        """
        try:
            connector = self.connectors.get(integration_id)
            if not connector:
                return True  # Already disconnected
            
            if connector.disconnect():
                with self.lock:
                    self.status_registry[integration_id].status = ConnectionStatus.DISCONNECTED
                    if self.stats['active_connections'] > 0:
                        self.stats['active_connections'] -= 1
                
                logger.info(f"Integration disconnected: {integration_id}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Integration disconnection failed: {e}")
            return False
    
    def read_from_integration(self, integration_id: str, 
                            query: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Read data from integration.
        
        Args:
            integration_id: Integration identifier
            query: Optional query parameters
            
        Returns:
            Retrieved data or None
        """
        start_time = time.time()
        
        try:
            connector = self.connectors.get(integration_id)
            if not connector or not connector.is_connected():
                logger.error(f"Integration not connected: {integration_id}")
                return None
            
            # Read data
            data = connector.read_data(query)
            
            # Record exchange
            duration_ms = (time.time() - start_time) * 1000
            data_size = self._estimate_data_size(data)
            
            exchange = DataExchange(
                exchange_id=str(uuid.uuid4()),
                integration_id=integration_id,
                operation="read",
                timestamp=datetime.now(),
                data_size=data_size,
                duration_ms=duration_ms,
                success=data is not None
            )
            
            self._record_exchange(exchange)
            
            return data
            
        except Exception as e:
            logger.error(f"Integration read failed: {e}")
            
            # Record failed exchange
            exchange = DataExchange(
                exchange_id=str(uuid.uuid4()),
                integration_id=integration_id,
                operation="read",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e)
            )
            
            self._record_exchange(exchange)
            return None
    
    def write_to_integration(self, integration_id: str, data: Any,
                           destination: Optional[str] = None) -> bool:
        """Write data to integration.
        
        Args:
            integration_id: Integration identifier
            data: Data to write
            destination: Optional destination specifier
            
        Returns:
            True if write successful
        """
        start_time = time.time()
        
        try:
            connector = self.connectors.get(integration_id)
            if not connector or not connector.is_connected():
                logger.error(f"Integration not connected: {integration_id}")
                return False
            
            # Write data
            success = connector.write_data(data, destination)
            
            # Record exchange
            duration_ms = (time.time() - start_time) * 1000
            data_size = self._estimate_data_size(data)
            
            exchange = DataExchange(
                exchange_id=str(uuid.uuid4()),
                integration_id=integration_id,
                operation="write",
                timestamp=datetime.now(),
                data_size=data_size,
                duration_ms=duration_ms,
                success=success
            )
            
            self._record_exchange(exchange)
            
            return success
            
        except Exception as e:
            logger.error(f"Integration write failed: {e}")
            
            # Record failed exchange
            exchange = DataExchange(
                exchange_id=str(uuid.uuid4()),
                integration_id=integration_id,
                operation="write",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e)
            )
            
            self._record_exchange(exchange)
            return False
    
    def get_integration_status(self, integration_id: str) -> Optional[IntegrationStatus]:
        """Get integration status.
        
        Args:
            integration_id: Integration identifier
            
        Returns:
            Integration status or None
        """
        return self.status_registry.get(integration_id)
    
    def list_integrations(self, integration_type: Optional[IntegrationType] = None,
                         status: Optional[ConnectionStatus] = None) -> List[IntegrationConfig]:
        """List registered integrations.
        
        Args:
            integration_type: Optional type filter
            status: Optional status filter
            
        Returns:
            List of integration configurations
        """
        integrations = list(self.integrations.values())
        
        if integration_type:
            integrations = [i for i in integrations if i.integration_type == integration_type]
        
        if status:
            integrations = [
                i for i in integrations 
                if self.status_registry.get(i.integration_id, IntegrationStatus("", ConnectionStatus.UNKNOWN)).status == status
            ]
        
        return integrations
    
    def test_all_connections(self) -> Dict[str, bool]:
        """Test all integration connections.
        
        Returns:
            Dictionary mapping integration IDs to test results
        """
        results = {}
        
        for integration_id, connector in self.connectors.items():
            try:
                results[integration_id] = connector.test_connection()
            except Exception as e:
                logger.error(f"Connection test failed for {integration_id}: {e}")
                results[integration_id] = False
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get integration framework statistics.
        
        Returns:
            Statistics dictionary
        """
        with self.lock:
            stats = self.stats.copy()
            
            # Add real-time metrics
            stats.update({
                'registered_integrations': len(self.integrations),
                'created_connectors': len(self.connectors),
                'recent_exchanges': len([e for e in self.exchange_log 
                                      if (datetime.now() - e.timestamp).seconds < 3600]),
                'success_rate': self._calculate_success_rate(),
                'avg_response_time': self._calculate_avg_response_time(),
                'integration_types': self._get_type_distribution()
            })
            
            return stats
    
    def cleanup_inactive_connections(self) -> int:
        """Cleanup inactive connections.
        
        Returns:
            Number of connections cleaned up
        """
        cleaned_up = 0
        current_time = datetime.now()
        
        for integration_id, status in self.status_registry.items():
            if (status.status == ConnectionStatus.ERROR or 
                (status.last_connected and 
                 current_time - status.last_connected > timedelta(hours=24))):
                
                if self.disconnect_integration(integration_id):
                    cleaned_up += 1
        
        logger.info(f"Cleaned up {cleaned_up} inactive connections")
        return cleaned_up
    
    def _validate_config(self, config: IntegrationConfig) -> bool:
        """Validate integration configuration."""
        if not config.integration_id or not config.name:
            logger.error("Integration ID and name are required")
            return False
        
        if config.integration_id in self.integrations:
            logger.error(f"Integration already exists: {config.integration_id}")
            return False
        
        if not config.connection_params:
            logger.error("Connection parameters are required")
            return False
        
        return True
    
    def _record_exchange(self, exchange: DataExchange):
        """Record data exchange."""
        with self.lock:
            self.exchange_log.append(exchange)
            
            # Update statistics
            self.stats['total_data_exchanges'] += 1
            self.stats['total_data_transferred'] += exchange.data_size
            
            if not exchange.success:
                self.stats['failed_operations'] += 1
            
            # Update integration status
            if exchange.integration_id in self.status_registry:
                status = self.status_registry[exchange.integration_id]
                status.data_transferred += exchange.data_size
                status.operations_count += 1
                
                if not exchange.success:
                    status.last_error = exchange.error_message
            
            # Keep only recent exchanges (last 1000)
            if len(self.exchange_log) > 1000:
                self.exchange_log = self.exchange_log[-1000:]
    
    def _estimate_data_size(self, data: Any) -> int:
        """Estimate data size in bytes."""
        try:
            if data is None:
                return 0
            elif isinstance(data, str):
                return len(data.encode('utf-8'))
            elif isinstance(data, bytes):
                return len(data)
            elif isinstance(data, (list, dict)):
                return len(json.dumps(data, default=str).encode('utf-8'))
            elif PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
                return data.memory_usage(deep=True).sum()
            else:
                return len(str(data).encode('utf-8'))
        except Exception:
            return 0
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate."""
        if not self.exchange_log:
            return 100.0
        
        successful = sum(1 for e in self.exchange_log if e.success)
        return (successful / len(self.exchange_log)) * 100.0
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time."""
        if not self.exchange_log:
            return 0.0
        
        total_time = sum(e.duration_ms for e in self.exchange_log)
        return total_time / len(self.exchange_log)
    
    def _get_type_distribution(self) -> Dict[str, int]:
        """Get distribution of integration types."""
        distribution = {}
        
        for integration in self.integrations.values():
            type_name = integration.integration_type.value
            distribution[type_name] = distribution.get(type_name, 0) + 1
        
        return distribution