# Microservices Template

A comprehensive template for building production-ready microservices architecture with service discovery, distributed patterns, and observability.

## 🎯 Features

### Core Microservices Capabilities
- **Service Discovery**: Consul-based service registry and discovery
- **API Gateway**: Traefik for routing and load balancing
- **Inter-Service Communication**: gRPC and REST APIs
- **Message Queue**: Redis Streams and RabbitMQ integration
- **Circuit Breaker**: Resilience patterns for fault tolerance
- **Distributed Tracing**: OpenTelemetry with Jaeger

### Resilience & Reliability
- **Health Checks**: Comprehensive service health monitoring
- **Circuit Breakers**: Automatic failure isolation
- **Retry Mechanisms**: Exponential backoff and jitter
- **Rate Limiting**: Per-service and global rate limiting
- **Bulkhead Pattern**: Resource isolation
- **Timeout Management**: Request timeout handling

### Observability & Monitoring
- **Distributed Tracing**: End-to-end request tracking
- **Metrics Collection**: Prometheus integration
- **Logging**: Structured logging with correlation IDs
- **Service Mesh**: Istio integration (optional)
- **APM**: Application performance monitoring
- **Alerting**: Prometheus Alertmanager

### Development & Deployment
- **Clean Architecture**: Domain-driven microservice design
- **API-First Design**: OpenAPI specifications
- **Container-Native**: Docker and Kubernetes ready
- **CI/CD Pipeline**: Automated testing and deployment
- **Local Development**: Docker Compose environment
- **Testing**: Contract testing with Pact

## 🏗️ Architecture

```
microservices-platform/
├── services/                    # Individual microservices
│   ├── user-service/           # User management service
│   ├── order-service/          # Order processing service
│   ├── notification-service/   # Notification service
│   └── payment-service/        # Payment processing service
├── gateway/                    # API Gateway (Traefik)
├── discovery/                  # Service discovery (Consul)
├── monitoring/                 # Observability stack
│   ├── prometheus/            # Metrics collection
│   ├── jaeger/               # Distributed tracing
│   └── grafana/              # Monitoring dashboards
├── messaging/                  # Message brokers
│   ├── redis/                # Redis for caching/streams
│   └── rabbitmq/             # RabbitMQ for queues
└── shared/                     # Shared libraries
    ├── events/               # Event schemas
    ├── protocols/            # gRPC protocols
    └── middleware/           # Common middleware
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone template
cp -r templates/microservices-template/ my-microservices
cd my-microservices

# Start infrastructure
docker-compose up -d consul redis rabbitmq prometheus jaeger grafana

# Start services
docker-compose up -d user-service order-service notification-service
```

### 2. Service Development

```bash
# Create new service
./scripts/create-service.sh product-service

# Run service locally
cd services/product-service
pip install -e ".[dev]"
python -m product_service.main

# Test service
pytest tests/ --cov=product_service
```

### 3. Inter-Service Communication

```bash
# gRPC communication
python -c "
from shared.protocols.user_pb2_grpc import UserServiceStub
from shared.protocols.user_pb2 import GetUserRequest
import grpc

with grpc.insecure_channel('localhost:50051') as channel:
    stub = UserServiceStub(channel)
    response = stub.GetUser(GetUserRequest(user_id='123'))
    print(response)
"

# REST API communication
curl http://localhost:8080/api/users/123
```

## 📊 Service Templates

### User Service
- **Purpose**: User authentication and profile management
- **Database**: PostgreSQL with user data
- **APIs**: REST and gRPC endpoints
- **Events**: User created/updated/deleted events

### Order Service
- **Purpose**: Order processing and management
- **Database**: PostgreSQL with order data
- **APIs**: REST and gRPC endpoints
- **Events**: Order created/updated/shipped events
- **Integration**: Payment service, notification service

### Notification Service
- **Purpose**: Multi-channel notifications (email, SMS, push)
- **Database**: MongoDB for notification logs
- **APIs**: gRPC endpoints
- **Events**: Notification sent/failed events
- **Integration**: External providers (SendGrid, Twilio)

### Payment Service
- **Purpose**: Payment processing and billing
- **Database**: PostgreSQL with payment data
- **APIs**: REST and gRPC endpoints
- **Events**: Payment processed/failed events
- **Integration**: Payment gateways (Stripe, PayPal)

## 🛠️ Technology Stack

### Service Framework
- **FastAPI**: High-performance async web framework
- **gRPC**: High-performance RPC framework
- **Pydantic**: Data validation and serialization
- **SQLAlchemy**: ORM with async support
- **Alembic**: Database migrations

### Infrastructure
- **Docker**: Containerization platform
- **Kubernetes**: Container orchestration
- **Consul**: Service discovery and configuration
- **Traefik**: Modern reverse proxy and load balancer
- **Istio**: Service mesh (optional)

### Messaging & Events
- **Redis Streams**: Event streaming
- **RabbitMQ**: Message queuing
- **Apache Kafka**: High-throughput messaging (optional)
- **CloudEvents**: Event specification standard

### Observability
- **OpenTelemetry**: Distributed tracing standard
- **Prometheus**: Metrics collection
- **Jaeger**: Distributed tracing backend
- **Grafana**: Monitoring dashboards
- **ELK Stack**: Centralized logging

### Testing
- **Pytest**: Testing framework
- **Pact**: Contract testing
- **Testcontainers**: Integration testing
- **K6**: Load testing

## 📁 Service Structure

```
service-template/
├── README.md
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── k8s/                        # Kubernetes manifests
├── src/service_name/
│   ├── domain/                # Domain logic
│   │   ├── entities/         # Business entities
│   │   ├── value_objects/    # Value objects
│   │   ├── services/         # Domain services
│   │   └── events/           # Domain events
│   ├── application/          # Application layer
│   │   ├── use_cases/       # Business use cases
│   │   ├── services/        # Application services
│   │   └── dto/             # Data transfer objects
│   ├── infrastructure/      # Infrastructure layer
│   │   ├── repositories/    # Data access
│   │   ├── messaging/       # Event publishing
│   │   ├── external/        # External service clients
│   │   └── database/        # Database configuration
│   └── presentation/        # Presentation layer
│       ├── api/            # REST API endpoints
│       ├── grpc/           # gRPC services
│       └── cli/            # CLI commands
├── tests/                   # Test suite
├── scripts/                # Utility scripts
└── docs/                   # Service documentation
```

## 🔧 Configuration

### Service Discovery
```yaml
# consul.yml
datacenter: dc1
retry_join:
  - consul-1
  - consul-2
  - consul-3
ui_config:
  enabled: true
connect:
  enabled: true
```

### API Gateway
```yaml
# traefik.yml
api:
  dashboard: true
  debug: true
entryPoints:
  web:
    address: ":80"
  websecure:
    address: ":443"
providers:
  consul:
    endpoints:
      - "consul:8500"
```

### Observability
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: 'microservices'
    consul_sd_configs:
      - server: 'consul:8500'
```

## 🔄 Communication Patterns

### Synchronous Communication
- **REST APIs**: HTTP-based request/response
- **gRPC**: High-performance binary protocol
- **GraphQL**: Flexible query language (optional)

### Asynchronous Communication
- **Event Streaming**: Redis Streams for real-time events
- **Message Queues**: RabbitMQ for reliable messaging
- **Pub/Sub**: Event-driven architecture patterns

### Service Mesh (Optional)
- **Istio**: Traffic management and security
- **Envoy Proxy**: L7 proxy and communication
- **mTLS**: Mutual TLS for service security

## 🧪 Testing Strategy

### Unit Testing
```bash
# Test individual service
cd services/user-service
pytest tests/unit/ --cov=user_service
```

### Integration Testing
```bash
# Test service integration
pytest tests/integration/ --testcontainers
```

### Contract Testing
```bash
# Test service contracts
pytest tests/contracts/ --pact-broker-url=http://pact-broker:9292
```

### End-to-End Testing
```bash
# Test complete workflows
pytest tests/e2e/ --environment=staging
```

## 📈 Monitoring & Observability

### Distributed Tracing
- **OpenTelemetry**: Automatic instrumentation
- **Jaeger**: Trace collection and visualization
- **Correlation IDs**: Request tracking across services

### Metrics Collection
- **Prometheus**: Time-series metrics database
- **Grafana**: Metrics visualization and alerting
- **Custom Metrics**: Business and technical metrics

### Logging
- **Structured Logging**: JSON-formatted logs
- **Log Aggregation**: ELK stack for centralized logging
- **Log Correlation**: Trace ID integration

### Health Checks
- **Liveness Probes**: Service availability
- **Readiness Probes**: Service ready to accept traffic
- **Custom Health Checks**: Business logic validation

## 🚀 Deployment Options

### Local Development
```bash
# Start all services
docker-compose up -d

# Scale specific service
docker-compose up -d --scale user-service=3
```

### Kubernetes Deployment
```bash
# Deploy to cluster
kubectl apply -f k8s/

# Monitor deployment
kubectl get pods -l app=microservices

# Service mesh (optional)
istioctl install --set values.global.meshID=mesh1
```

### Production Deployment
```bash
# Build and push images
./scripts/build-all.sh
./scripts/push-all.sh

# Deploy with Helm
helm install microservices ./helm/microservices
```

## 🔐 Security

### Authentication & Authorization
- **JWT Tokens**: Stateless authentication
- **OAuth2**: Third-party authentication
- **RBAC**: Role-based access control
- **mTLS**: Service-to-service encryption

### Network Security
- **Service Mesh**: Encrypted service communication
- **Network Policies**: Kubernetes network isolation
- **API Gateway**: Centralized security policies
- **Rate Limiting**: DDoS protection

## 🔄 CI/CD Pipeline

### Build Pipeline
1. **Unit Tests**: Individual service testing
2. **Integration Tests**: Cross-service testing
3. **Contract Tests**: API contract validation
4. **Security Scan**: Vulnerability assessment
5. **Image Build**: Docker image creation

### Deployment Pipeline
1. **Staging Deployment**: Automated staging deployment
2. **E2E Tests**: End-to-end validation
3. **Performance Tests**: Load and stress testing
4. **Production Deployment**: Blue-green deployment
5. **Monitoring**: Post-deployment monitoring

## 📚 Documentation

### API Documentation
- **OpenAPI**: REST API specifications
- **gRPC Documentation**: Protocol buffer definitions
- **Postman Collections**: API testing collections

### Architecture Documentation
- **C4 Model**: System architecture diagrams
- **Sequence Diagrams**: Inter-service communication
- **Deployment Diagrams**: Infrastructure topology

## 🤝 Contributing

1. **Service Development**: Follow domain-driven design principles
2. **API Design**: Use OpenAPI specifications
3. **Testing**: Maintain comprehensive test coverage
4. **Documentation**: Update architecture and API docs
5. **Monitoring**: Add appropriate metrics and logging

## 📄 License

MIT License - see LICENSE file for details.

---

**⚡ Ready for microservices at scale!**
**🚀 From monolith to microservices in minutes!**