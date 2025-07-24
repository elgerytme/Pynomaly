# 🏢 Enterprise Integration & API Management Guide

## Overview

The Enterprise Integration & API Management system provides comprehensive capabilities for connecting enterprise systems, managing APIs, and orchestrating message flows. This platform includes advanced API gateway functionality, enterprise connectors for major systems like SAP, and robust message brokering for asynchronous communication.

## Architecture Components

### 1. Enterprise API Gateway (`enterprise_api_gateway.py`)
Advanced API gateway with enterprise-grade features:

- **Multi-Protocol Support**: REST, GraphQL, SOAP, gRPC, WebSocket
- **Authentication Methods**: API Key, JWT, OAuth2, Basic Auth, Mutual TLS
- **Rate Limiting**: Client-based, endpoint-based, hierarchical limits
- **Circuit Breaker**: Automatic failover and recovery
- **Request/Response Transformation**: JSON/XML conversion, field mapping, data enrichment
- **Caching**: Redis-backed response caching with TTL
- **Load Balancing**: Multiple upstream server support

### 2. SAP Connector (`sap_connector.py`)
Comprehensive SAP system integration:

- **Multiple SAP Systems**: ERP, S/4 HANA, BW, CRM, SRM, PI/PO, Fiori
- **Integration Patterns**: RFC, BAPI, IDoc, OData, REST API, SOAP
- **Connection Pooling**: Efficient connection management
- **Circuit Breaker**: SAP-specific fault tolerance
- **Authentication**: Basic, OAuth2, SAML, X.509 certificates
- **Data Transformation**: Bidirectional mapping and enrichment

### 3. Enterprise Message Broker (`enterprise_message_broker.py`)
High-performance message brokering system:

- **Queue Types**: Standard, FIFO, Priority, Delayed, Dead Letter
- **Delivery Modes**: At-most-once, At-least-once, Exactly-once
- **Message Routing**: Rule-based routing and filtering
- **Consumer Groups**: Load balancing and failover
- **Kafka Integration**: External system integration
- **Persistence**: Redis-backed message durability

## Quick Start

### 1. Initialize API Gateway

```python
from api_gateway.infrastructure.gateway.enterprise_api_gateway import (
    EnterpriseAPIGateway, APIEndpoint, ProtocolType, AuthenticationMethod
)

# Initialize gateway
gateway = EnterpriseAPIGateway()
await gateway.initialize()

# Register custom endpoint
endpoint = APIEndpoint(
    id="payment_service",
    name="Payment Processing API",
    path="/api/v1/payments",
    method="*",
    protocol=ProtocolType.REST,
    upstream_url="http://payment-service:8080",
    authentication=AuthenticationMethod.JWT_TOKEN,
    rate_limit=500,
    timeout_seconds=30,
    retry_attempts=3,
    circuit_breaker_enabled=True,
    transformation_rules=[
        {
            "type": "field_mapping",
            "mappings": {
                "card_number": "cardNumber",
                "expiry_date": "expiryDate"
            }
        }
    ],
    caching_enabled=True,
    cache_ttl_seconds=300
)

await gateway.register_endpoint(endpoint)

# Process API request
request_data = {
    "request_id": "req_12345",
    "endpoint_id": "payment_service",
    "method": "POST",
    "path": "/api/v1/payments",
    "headers": {
        "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
        "Content-Type": "application/json"
    },
    "body": {
        "amount": 100.00,
        "currency": "USD",
        "card_number": "4111111111111111",
        "expiry_date": "12/25"
    },
    "client_id": "mobile_app_v2",
    "client_ip": "192.168.1.100"
}

response = await gateway.process_request(request_data)
print(f"Payment Response: {response}")
```

### 2. Connect to SAP Systems

```python
from enterprise_connectors.infrastructure.connectors.sap_connector import (
    SAPConnector, SAPConnection, SAPOperation, SAPRequest, 
    SAPSystemType, AuthenticationType, IntegrationPattern
)

# Initialize SAP connector
connector = SAPConnector()
await connector.initialize()

# Register custom SAP connection
connection = SAPConnection(
    connection_id="sap_production",
    name="SAP Production ERP",
    host="sap-prod.company.com",
    port=8000,
    client="100",
    system_id="PRD",
    system_type=SAPSystemType.ERP,
    authentication=AuthenticationType.BASIC,
    username="API_USER",
    password="secure_password",
    pool_size=20,
    timeout_seconds=60
)

await connector.register_connection(connection)

# Register custom operation
operation = SAPOperation(
    operation_id="get_vendor_data",
    name="Get Vendor Master Data",
    pattern=IntegrationPattern.BAPI,
    bapi_name="BAPI_VENDOR_GETDETAIL",
    input_mapping={
        "vendor_number": "VENDORNO",
        "company_code": "COMPANYCODE"
    },
    output_mapping={
        "vendor_data": "VENDORDETAIL",
        "addresses": "VENDORADDRESS"
    }
)

await connector.register_operation(operation)

# Execute SAP request
sap_request = SAPRequest(
    request_id="sap_req_001",
    connection_id="sap_production",
    operation_id="get_vendor_data",
    parameters={
        "vendor_number": "0000100001",
        "company_code": "1000"
    }
)

response = await connector.execute_request(sap_request)
print(f"SAP Response: {response.data if response.success else response.error_message}")
```

### 3. Set Up Message Brokering

```python
from message_queue.infrastructure.messaging.enterprise_message_broker import (
    EnterpriseMessageBroker, Message, Queue, Consumer, 
    MessagePriority, QueueType, DeliveryMode
)

# Initialize message broker
broker = EnterpriseMessageBroker()
await broker.initialize()

# Create custom queue
queue = Queue(
    name="customer_events",
    queue_type=QueueType.PRIORITY,
    max_size=100000,
    ttl_seconds=3600,
    dead_letter_queue="customer_events_dlq",
    max_delivery_attempts=5
)

await broker.create_queue(queue)

# Publish message
message = Message(
    message_id="cust_event_001",
    topic="customer_events",
    payload={
        "event_type": "customer_created",
        "customer_id": "CUST-98765",
        "customer_data": {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "tier": "premium"
        },
        "metadata": {
            "source_system": "crm",
            "event_timestamp": "2024-01-15T10:30:00Z"
        }
    },
    headers={
        "source": "customer_service",
        "version": "2.0",
        "content_encoding": "gzip"
    },
    priority=MessagePriority.HIGH,
    delivery_mode=DeliveryMode.AT_LEAST_ONCE,
    correlation_id="workflow_abc123"
)

success = await broker.publish_message(message)
print(f"Message published: {success}")

# Register consumer
async def customer_event_handler(messages):
    for message in messages:
        event_type = message.payload.get('event_type')
        customer_id = message.payload.get('customer_id')
        
        print(f"Processing {event_type} for customer {customer_id}")
        
        # Process the event
        if event_type == "customer_created":
            # Send welcome email, update analytics, etc.
            await send_welcome_email(message.payload['customer_data'])
            await update_customer_analytics(customer_id)
        
        return True

consumer = Consumer(
    consumer_id="customer_processor_001",
    name="Customer Event Processor",
    topics=["customer_events"],
    group_id="customer_processing_group",
    handler=customer_event_handler,
    batch_processing=True,
    max_batch_size=10,
    processing_timeout=60
)

await broker.register_consumer(consumer)

# Start consuming messages
while True:
    messages = await broker.consume_messages(consumer.consumer_id)
    if messages:
        # Process messages
        success = await customer_event_handler(messages)
        
        # Acknowledge processed messages
        for message in messages:
            if success:
                await broker.acknowledge_message(message.message_id, consumer.consumer_id)
            else:
                await broker.reject_message(message.message_id, consumer.consumer_id, requeue=True)
    
    await asyncio.sleep(1)
```

## Configuration

### Environment Variables

```bash
# API Gateway Configuration
export API_GATEWAY_REDIS_URL="redis://redis-cluster:6379/7"
export API_GATEWAY_JWT_SECRET="your-jwt-secret-key"
export API_GATEWAY_RATE_LIMIT_ENABLED="true"
export API_GATEWAY_CIRCUIT_BREAKER_ENABLED="true"

# SAP Connector Configuration
export SAP_CONNECTOR_REDIS_URL="redis://redis-cluster:6379/8"
export SAP_PRODUCTION_USERNAME="API_USER"
export SAP_PRODUCTION_PASSWORD="encrypted_password"
export SAP_CONNECTION_POOL_SIZE="20"
export SAP_DEFAULT_TIMEOUT="60"

# Message Broker Configuration
export MESSAGE_BROKER_REDIS_URL="redis://redis-cluster:6379/9"
export KAFKA_BOOTSTRAP_SERVERS="kafka-cluster:9092"
export MESSAGE_BROKER_DEFAULT_TTL="3600"
export MESSAGE_BROKER_MAX_RETRIES="5"

# Security Configuration
export ENTERPRISE_ENCRYPTION_KEY="your-encryption-key"
export TLS_CERT_PATH="/etc/ssl/certs"
export TLS_KEY_PATH="/etc/ssl/private"

# Monitoring Configuration
export PROMETHEUS_URL="http://prometheus:9090"
export GRAFANA_URL="http://grafana:3000"
export JAEGER_ENDPOINT="http://jaeger:14268/api/traces"
```

### Kubernetes Deployment

```yaml
# enterprise-integration-namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: enterprise-integration
  labels:
    name: enterprise-integration

---
# api-gateway-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
  namespace: enterprise-integration
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
    spec:
      containers:
      - name: api-gateway
        image: enterprise/api-gateway:v1.0.0
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        env:
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: integration-secrets
              key: redis-url
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: integration-secrets
              key: jwt-secret
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 8000
          name: metrics
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

---
# sap-connector-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sap-connector
  namespace: enterprise-integration
spec:
  replicas: 2
  selector:
    matchLabels:
      app: sap-connector
  template:
    metadata:
      labels:
        app: sap-connector
    spec:
      containers:
      - name: sap-connector
        image: enterprise/sap-connector:v1.0.0
        resources:
          requests:
            cpu: "300m"
            memory: "512Mi"
          limits:
            cpu: "1"
            memory: "2Gi"
        env:
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: integration-secrets
              key: redis-url
        - name: SAP_USERNAME
          valueFrom:
            secretKeyRef:
              name: sap-credentials
              key: username
        - name: SAP_PASSWORD
          valueFrom:
            secretKeyRef:
              name: sap-credentials
              key: password
        ports:
        - containerPort: 8080
          name: http

---
# message-broker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: message-broker
  namespace: enterprise-integration
spec:
  replicas: 2
  selector:
    matchLabels:
      app: message-broker
  template:
    metadata:
      labels:
        app: message-broker
    spec:
      containers:
      - name: message-broker
        image: enterprise/message-broker:v1.0.0
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        env:
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: integration-secrets
              key: redis-url
        - name: KAFKA_BOOTSTRAP_SERVERS
          value: "kafka-cluster:9092"
        ports:
        - containerPort: 8080
          name: http

---
# Services
apiVersion: v1
kind: Service
metadata:
  name: api-gateway-service
  namespace: enterprise-integration
spec:
  selector:
    app: api-gateway
  ports:
  - port: 8080
    targetPort: 8080
    name: http
  - port: 8000
    targetPort: 8000
    name: metrics
  type: LoadBalancer

---
apiVersion: v1
kind: Service
metadata:
  name: sap-connector-service
  namespace: enterprise-integration
spec:
  selector:
    app: sap-connector
  ports:
  - port: 8080
    targetPort: 8080
  type: ClusterIP

---
apiVersion: v1
kind: Service
metadata:
  name: message-broker-service
  namespace: enterprise-integration
spec:
  selector:
    app: message-broker
  ports:
  - port: 8080
    targetPort: 8080
  type: ClusterIP
```

## API Gateway Features

### 1. Authentication and Authorization

#### JWT Token Authentication
```python
# JWT token validation with custom claims
jwt_config = {
    "secret": "your-jwt-secret",
    "algorithms": ["HS256", "RS256"],
    "required_claims": ["sub", "exp", "iat"],
    "custom_validation": lambda payload: payload.get("role") in ["user", "admin"]
}
```

#### API Key Management
```python
# API key with rate limiting and scoping
api_key_config = {
    "key": "ak_live_1234567890abcdef",
    "scopes": ["payments:read", "payments:write"],
    "rate_limit": {
        "requests_per_minute": 1000,
        "requests_per_hour": 10000
    },
    "ip_whitelist": ["192.168.1.0/24", "10.0.0.0/8"]
}
```

### 2. Request/Response Transformation

#### JSON to SOAP Transformation
```python
transformation_rule = {
    "type": "json_to_soap",
    "soap_action": "urn:processOrder",
    "envelope_template": "order_soap_envelope.xml",
    "field_mappings": {
        "order_id": "OrderNumber",
        "customer_id": "CustomerNumber",
        "items": "OrderItems"
    }
}
```

#### Field Mapping and Data Enrichment
```python
enrichment_rule = {
    "type": "data_enrichment",
    "sources": ["timestamp", "request_id", "client_info"],
    "mappings": {
        "customer_number": "customerId",
        "order_date": "orderTimestamp"
    },
    "lookups": [
        {
            "field": "customer_tier",
            "source": "customer_service",
            "key": "customer_id"
        }
    ]
}
```

### 3. Rate Limiting and Circuit Breaker

#### Hierarchical Rate Limiting
```python
rate_limit_config = {
    "global": {"requests_per_second": 10000},
    "per_client": {"requests_per_minute": 1000},
    "per_endpoint": {"requests_per_minute": 500},
    "burst_allowance": 50,
    "backoff_strategy": "exponential"
}
```

#### Circuit Breaker Configuration
```python
circuit_breaker_config = {
    "failure_threshold": 10,
    "success_threshold": 5,
    "timeout_seconds": 60,
    "half_open_max_calls": 3,
    "metrics_window_seconds": 300
}
```

## SAP Integration Patterns

### 1. BAPI Integration

```python
# Customer master data retrieval
bapi_request = SAPRequest(
    request_id="bapi_customer_001",
    connection_id="erp_production",
    operation_id="get_customer_data",
    parameters={
        "customer_number": "0000100001",
        "company_code": "1000",
        "key_date": "20240115"
    }
)

response = await connector.execute_request(bapi_request)

if response.success:
    customer_data = response.data.get('CUSTOMERDETAIL', {})
    addresses = response.data.get('CUSTOMERADDRESS', [])
    
    print(f"Customer: {customer_data.get('NAME')}")
    print(f"Country: {customer_data.get('COUNTRY')}")
    for address in addresses:
        print(f"Address: {address.get('STREET')}, {address.get('CITY')}")
```

### 2. RFC Function Calls

```python
# Read table data using RFC_READ_TABLE
rfc_request = SAPRequest(
    request_id="rfc_table_001",
    connection_id="erp_production",
    operation_id="get_material_info",
    parameters={
        "table": "MARA",  # Material master table
        "fields": [
            {"FIELDNAME": "MATNR"},  # Material number
            {"FIELDNAME": "MTART"},  # Material type
            {"FIELDNAME": "MAKTX"}   # Material description
        ],
        "options": [
            {"TEXT": "MTART EQ 'FERT'"}  # Finished products only
        ],
        "rowcount": 100
    }
)

response = await connector.execute_request(rfc_request)

if response.success:
    materials = response.data.get('DATA', [])
    for material in materials:
        print(f"Material: {material['WA']}")
```

### 3. IDoc Processing

```python
# Send purchase order IDoc
idoc_request = SAPRequest(
    request_id="idoc_po_001",
    connection_id="erp_production",
    operation_id="send_purchase_order_idoc",
    parameters={
        "control_record": {
            "MESTYP": "ORDERS",
            "IDOCTYP": "ORDERS05",
            "RCVPRT": "LS",
            "RCVPRN": "VENDOR_SYSTEM"
        },
        "data_records": [
            {
                "SEGNAM": "E1EDK01",
                "BELNR": "PO-12345",
                "CURCY": "USD",
                "WKURS": "1.0"
            },
            {
                "SEGNAM": "E1EDP01",
                "POSEX": "00010",
                "MATNR": "WIDGET-001",
                "MENGE": "100"
            }
        ]
    }
)

response = await connector.execute_request(idoc_request)

if response.success:
    idoc_number = response.data.get('idoc_number')
    print(f"IDoc sent successfully: {idoc_number}")
```

### 4. OData Services

```python
# Query products using OData
odata_request = SAPRequest(
    request_id="odata_products_001",
    connection_id="s4_hana_dev",
    operation_id="odata_products",
    parameters={
        "filters": "ProductType eq 'FERT' and ProductGroup eq 'ELECTRONICS'",
        "select": "Product,ProductType,ProductDescription,BaseUnit",
        "expand": "to_ProductPlant",
        "orderby": "Product asc",
        "top": 50
    }
)

response = await connector.execute_request(odata_request)

if response.success:
    products = response.data.get('d', {}).get('results', [])
    for product in products:
        print(f"Product: {product['Product']} - {product['ProductDescription']}")
```

## Message Brokering Patterns

### 1. Priority Queue Processing

```python
# High-priority order processing
urgent_order = Message(
    message_id="urgent_order_001",
    topic="orders",
    payload={
        "order_id": "ORD-URGENT-001",
        "customer_tier": "platinum",
        "order_type": "express",
        "total_amount": 5000.00
    },
    priority=MessagePriority.CRITICAL,
    headers={
        "sla": "2_hours",
        "escalation_required": "true"
    }
)

await broker.publish_message(urgent_order)
```

### 2. Delayed Message Processing

```python
# Schedule reminder email
reminder_message = Message(
    message_id="reminder_001",
    topic="notifications",
    payload={
        "type": "cart_abandonment_reminder",
        "customer_id": "CUST-12345",
        "cart_items": ["PROD-001", "PROD-002"],
        "abandoned_at": "2024-01-15T14:30:00Z"
    },
    delay_seconds=3600,  # Send after 1 hour
    priority=MessagePriority.NORMAL
)

await broker.publish_message(reminder_message)
```

### 3. Dead Letter Queue Handling

```python
# Configure dead letter queue processing
async def dlq_processor():
    while True:
        # Process messages from dead letter queue
        dlq_messages = await broker.consume_messages("dlq_processor")
        
        for message in dlq_messages:
            dlq_reason = message.headers.get('dlq_reason', 'unknown')
            original_topic = message.headers.get('original_topic', 'unknown')
            
            logger.warning(f"Processing DLQ message: {message.message_id}")
            logger.warning(f"Reason: {dlq_reason}, Original topic: {original_topic}")
            
            # Implement recovery logic
            if dlq_reason == "max_retries_exceeded":
                # Send to manual review queue
                await send_to_manual_review(message)
            elif dlq_reason == "expired":
                # Log for analytics
                await log_expired_message(message)
            
            await broker.acknowledge_message(message.message_id, "dlq_processor")
        
        await asyncio.sleep(30)  # Check every 30 seconds
```

### 4. Consumer Group Load Balancing

```python
# Configure consumer group with multiple consumers
consumers = []

for i in range(3):  # 3 consumer instances
    consumer = Consumer(
        consumer_id=f"order_processor_{i}",
        name=f"Order Processor {i}",
        topics=["orders", "order_updates"],
        group_id="order_processing_group",
        handler=process_order_batch,
        batch_processing=True,
        max_batch_size=20,
        processing_timeout=120
    )
    
    await broker.register_consumer(consumer)
    consumers.append(consumer)

# Start consuming with load balancing
async def start_consumer_group():
    tasks = []
    
    for consumer in consumers:
        task = asyncio.create_task(consumer_loop(consumer))
        tasks.append(task)
    
    await asyncio.gather(*tasks)

async def consumer_loop(consumer):
    while True:
        messages = await broker.consume_messages(consumer.consumer_id)
        
        if messages:
            try:
                success = await consumer.handler(messages)
                
                for message in messages:
                    if success:
                        await broker.acknowledge_message(
                            message.message_id, consumer.consumer_id
                        )
                    else:
                        await broker.reject_message(
                            message.message_id, consumer.consumer_id, requeue=True
                        )
                        
            except Exception as e:
                logger.error(f"Consumer error: {e}")
                
                for message in messages:
                    await broker.reject_message(
                        message.message_id, consumer.consumer_id, requeue=True
                    )
        
        await asyncio.sleep(1)
```

## Monitoring and Analytics

### Prometheus Metrics

#### API Gateway Metrics
```promql
# Request rate by endpoint
rate(api_gateway_requests_total[5m])

# Average response time
rate(api_gateway_request_duration_seconds_sum[5m]) / 
rate(api_gateway_request_duration_seconds_count[5m])

# Error rate percentage
rate(api_gateway_requests_total{status=~"5.."}[5m]) / 
rate(api_gateway_requests_total[5m]) * 100

# Circuit breaker trips
increase(api_gateway_circuit_breaker_trips_total[1h])
```

#### SAP Connector Metrics
```promql
# SAP request success rate
rate(sap_requests_total{status="success"}[5m]) / 
rate(sap_requests_total[5m]) * 100

# Average SAP response time
rate(sap_request_duration_seconds_sum[5m]) / 
rate(sap_request_duration_seconds_count[5m])

# Connection pool utilization
sap_connection_pool_size - sap_connection_pool_available
```

#### Message Broker Metrics
```promql
# Message throughput
rate(message_broker_messages_produced_total[5m])
rate(message_broker_messages_consumed_total[5m])

# Queue depth
message_broker_queue_size

# Consumer lag
message_broker_consumer_lag

# Dead letter queue growth
increase(message_broker_queue_size{queue_name=~".*_dlq"}[1h])
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "Enterprise Integration Overview",
    "panels": [
      {
        "title": "API Gateway Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(api_gateway_requests_total[5m])) by (endpoint)",
            "legendFormat": "{{endpoint}}"
          }
        ]
      },
      {
        "title": "SAP Connection Health",
        "type": "stat",
        "targets": [
          {
            "expr": "sap_circuit_breaker_state",
            "legendFormat": "{{connection}}"
          }
        ]
      },
      {
        "title": "Message Queue Sizes",
        "type": "graph",
        "targets": [
          {
            "expr": "message_broker_queue_size",
            "legendFormat": "{{queue_name}}"
          }
        ]
      },
      {
        "title": "Integration Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "sum(rate(api_gateway_requests_total{status=~\"5..\"}[5m])) / sum(rate(api_gateway_requests_total[5m])) * 100"
          }
        ]
      }
    ]
  }
}
```

### Alert Rules

```yaml
# prometheus-alerts.yaml
groups:
- name: enterprise-integration
  rules:
  - alert: APIGatewayHighErrorRate
    expr: |
      (
        sum(rate(api_gateway_requests_total{status=~"5.."}[5m])) / 
        sum(rate(api_gateway_requests_total[5m]))
      ) * 100 > 5
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "API Gateway high error rate"
      description: "Error rate is {{ $value }}% over the last 5 minutes"

  - alert: SAPConnectionDown
    expr: sap_circuit_breaker_state == 1
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "SAP connection circuit breaker open"
      description: "SAP connection {{ $labels.connection }} is unavailable"

  - alert: MessageQueueBacklog
    expr: message_broker_queue_size > 10000
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Large message queue backlog"
      description: "Queue {{ $labels.queue_name }} has {{ $value }} messages"

  - alert: ConsumerLagHigh
    expr: message_broker_consumer_lag > 1000
    for: 3m
    labels:
      severity: warning
    annotations:
      summary: "High consumer lag detected"
      description: "Consumer group {{ $labels.consumer_group }} has lag of {{ $value }} messages"
```

## Security and Compliance

### 1. Data Encryption

#### Transport Security
```python
# TLS configuration for all connections
tls_config = {
    "ssl_cert_file": "/etc/ssl/certs/integration.crt",
    "ssl_key_file": "/etc/ssl/private/integration.key",
    "ssl_ca_file": "/etc/ssl/certs/ca.crt",
    "ssl_verify_mode": "CERT_REQUIRED",
    "ssl_ciphers": "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS"
}
```

#### Data at Rest Encryption
```python
# Message payload encryption
from cryptography.fernet import Fernet

def encrypt_payload(payload: dict, key: bytes) -> str:
    fernet = Fernet(key)
    serialized = json.dumps(payload).encode()
    encrypted = fernet.encrypt(serialized)
    return encrypted.decode()

def decrypt_payload(encrypted_payload: str, key: bytes) -> dict:
    fernet = Fernet(key)
    decrypted = fernet.decrypt(encrypted_payload.encode())
    return json.loads(decrypted.decode())
```

### 2. Access Control

#### Role-Based Access Control (RBAC)
```python
rbac_config = {
    "roles": {
        "admin": {
            "permissions": ["*"],
            "resources": ["*"]
        },
        "api_user": {
            "permissions": ["read", "write"],
            "resources": ["api:*", "queue:orders", "queue:notifications"]
        },
        "sap_integration": {
            "permissions": ["read", "write"],
            "resources": ["sap:*", "queue:sap_events"]
        },
        "readonly": {
            "permissions": ["read"],
            "resources": ["api:health", "metrics:*"]
        }
    }
}
```

### 3. Audit Logging

```python
# Comprehensive audit logging
audit_config = {
    "log_level": "INFO",
    "log_format": "json",
    "include_request_body": False,  # PII protection
    "include_response_body": False,
    "log_retention_days": 90,
    "fields": [
        "timestamp",
        "request_id",
        "user_id",
        "client_ip",
        "endpoint",
        "method",
        "status_code",
        "response_time",
        "error_message"
    ]
}

async def log_audit_event(event_type: str, details: dict):
    audit_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "details": details,
        "source": "enterprise_integration"
    }
    
    # Send to centralized logging system
    await send_to_elk_stack(audit_entry)
    
    # Store in audit database
    await store_audit_record(audit_entry)
```

## Performance Optimization

### 1. Connection Pooling

```python
# Optimized connection pool configuration
pool_config = {
    "initial_size": 5,
    "max_size": 50,
    "max_idle_time": 300,  # 5 minutes
    "connection_timeout": 30,
    "validation_query": "SELECT 1",
    "retry_attempts": 3,
    "retry_delay": 1
}
```

### 2. Caching Strategies

```python
# Multi-level caching
cache_config = {
    "levels": [
        {
            "type": "memory",
            "max_size": 1000,
            "ttl": 60
        },
        {
            "type": "redis",
            "max_size": 100000,
            "ttl": 3600
        }
    ],
    "cache_key_prefix": "integration:",
    "compression": "gzip"
}
```

### 3. Batch Processing

```python
# Optimized batch processing
async def process_messages_batch(messages: List[Message]) -> bool:
    batch_size = 50
    batches = [messages[i:i + batch_size] 
              for i in range(0, len(messages), batch_size)]
    
    tasks = []
    for batch in batches:
        task = asyncio.create_task(process_single_batch(batch))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Check for failures and handle accordingly
    success_count = sum(1 for r in results if r is True)
    return success_count == len(batches)
```

## Troubleshooting

### Common Issues

#### 1. API Gateway Issues

```bash
# Check gateway health
curl http://api-gateway:8080/health

# View gateway logs
kubectl logs -f deployment/api-gateway -n enterprise-integration

# Check rate limiting
curl -H "X-API-Key: your-key" http://api-gateway:8080/api/v1/rate-limit-status

# Test circuit breaker
curl -H "X-API-Key: your-key" http://api-gateway:8080/api/v1/circuit-breaker-status
```

#### 2. SAP Connection Issues

```bash
# Test SAP connectivity
curl -X POST http://sap-connector:8080/test-connection \
  -H "Content-Type: application/json" \
  -d '{"connection_id": "erp_production"}'

# View SAP connector metrics
curl http://sap-connector:8080/metrics

# Check connection pool status
curl http://sap-connector:8080/connection-pools
```

#### 3. Message Broker Issues

```bash
# Check broker health
curl http://message-broker:8080/health

# View queue status
curl http://message-broker:8080/queues

# Check consumer lag
curl http://message-broker:8080/consumer-groups

# Monitor dead letter queues
curl http://message-broker:8080/dead-letter-queues
```

### Debugging Commands

```bash
# Enable debug logging
kubectl set env deployment/api-gateway LOG_LEVEL=DEBUG -n enterprise-integration
kubectl set env deployment/sap-connector LOG_LEVEL=DEBUG -n enterprise-integration
kubectl set env deployment/message-broker LOG_LEVEL=DEBUG -n enterprise-integration

# View detailed metrics
kubectl port-forward svc/api-gateway-service 8080:8080 -n enterprise-integration
curl http://localhost:8080/metrics | grep -E "(request_duration|error_rate)"

# Check Redis connectivity
kubectl exec -it redis-pod -- redis-cli ping

# Monitor resource usage
kubectl top pods -n enterprise-integration
kubectl describe pods -n enterprise-integration
```

## Best Practices

### 1. API Design
- **RESTful Principles**: Follow REST conventions for consistency
- **Versioning**: Use semantic versioning for API endpoints
- **Error Handling**: Provide meaningful error messages and codes
- **Documentation**: Maintain up-to-date OpenAPI specifications

### 2. Integration Patterns
- **Idempotency**: Ensure operations are idempotent where possible
- **Timeout Handling**: Set appropriate timeouts for all integrations
- **Retry Logic**: Implement exponential backoff for retries
- **Circuit Breakers**: Use circuit breakers to prevent cascade failures

### 3. Message Design
- **Schema Validation**: Validate message schemas for consistency
- **Message Versioning**: Support multiple message versions
- **Dead Letter Queues**: Configure DLQs for failed message handling
- **Message Ordering**: Use FIFO queues when order matters

### 4. Monitoring and Alerting
- **Golden Signals**: Monitor latency, traffic, errors, and saturation
- **SLA Tracking**: Track and alert on SLA violations
- **Capacity Planning**: Monitor resource usage trends
- **Incident Response**: Maintain runbooks for common issues

## Support and Resources

- **Documentation**: `integration/docs/`
- **API Documentation**: `http://localhost:8080/docs`
- **Monitoring Dashboard**: `http://grafana.integration.local`
- **Support Channel**: `#enterprise-integration`
- **Issue Tracker**: GitHub Issues with `integration` label

For detailed implementation examples and advanced configuration, see the `examples/` directory and component-specific documentation.