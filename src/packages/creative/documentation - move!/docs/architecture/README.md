# Architecture Documentation

**Comprehensive guide to Pynomaly's system architecture and design decisions**

## Overview

Pynomaly is built using Clean Architecture principles with Domain-Driven Design (DDD) patterns, ensuring maintainability, testability, and extensibility.

## Quick Navigation

### 🏗️ [System Design](./system-design.md)
High-level architecture overview and system components

### 📋 [Architecture Decision Records (ADRs)](./adr/)
Documented architectural decisions and their rationale

### 🎯 [Domain Model](./domain-model.md)
Core domain entities, value objects, and business rules

### 🧩 [Clean Architecture Implementation](./clean-architecture.md)
How we implement Clean Architecture principles

## Architecture Principles

### 1. Clean Architecture
- **Dependency Inversion**: Dependencies point inward toward the domain
- **Separation of Concerns**: Each layer has a single responsibility
- **Independence**: Framework, database, and UI independence

### 2. Domain-Driven Design
- **Ubiquitous Language**: Consistent terminology across the system
- **Bounded Contexts**: Clear boundaries between different domains
- **Aggregate Roots**: Consistent data modification boundaries

### 3. SOLID Principles
- **Single Responsibility**: Each class has one reason to change
- **Open/Closed**: Open for extension, closed for modification
- **Liskov Substitution**: Derived classes must be substitutable
- **Interface Segregation**: Clients depend only on interfaces they use
- **Dependency Inversion**: Depend on abstractions, not concretions

## System Layers

```
┌─────────────────────────────────────┐
│          Presentation Layer         │  ← Web UI, CLI, API Controllers
│  (FastAPI, Typer, React Frontend)  │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│         Application Layer           │  ← Use Cases, DTOs, Services
│   (Services, Orchestration)        │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│           Domain Layer              │  ← Business Logic, Entities
│  (Entities, Value Objects, Rules)  │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│        Infrastructure Layer        │  ← Database, External APIs
│  (Repositories, Adapters, etc.)    │
└─────────────────────────────────────┘
```

## Key Architectural Components

### Domain Layer (`src/pynomaly/domain/`)
- **Entities**: Core business objects (Detector, Dataset, Anomaly)
- **Value Objects**: Immutable objects (DetectionConfig, Metrics)
- **Services**: Domain logic that doesn't belong to entities
- **Protocols**: Interfaces defining contracts

### Application Layer (`src/pynomaly/application/`)
- **Services**: Use case orchestration
- **DTOs**: Data transfer objects for API boundaries
- **Ports**: Interfaces for external dependencies

### Infrastructure Layer (`src/pynomaly/infrastructure/`)
- **Repositories**: Data persistence implementations
- **Adapters**: External service integrations
- **Configuration**: System configuration management

### Presentation Layer (`src/pynomaly/presentation/`)
- **API**: REST API endpoints (FastAPI)
- **CLI**: Command-line interface (Typer)
- **Web**: Frontend application (React/HTMX)

## Architecture Decision Records

Our architectural decisions are documented in ADRs:

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-001](./adr/ADR-001-core-architecture-patterns.md) | Core Architecture Patterns | Accepted |
| [ADR-002](./adr/ADR-002-data-pipeline-architecture.md) | Data Pipeline Architecture | Accepted |
| [ADR-003](./adr/ADR-003.md) | Technology Stack Selection | Accepted |
| [ADR-004](./adr/ADR-004-api-design-and-versioning.md) | API Design and Versioning | Accepted |
| [ADR-005](./adr/ADR-005-security-architecture.md) | Security Architecture | Accepted |
| [ADR-014](./adr/ADR-014-repository-unit-of-work-pattern.md) | Repository & Unit-of-Work Pattern | Accepted |
| [ADR-015](./adr/ADR-015-production-database-technology-selection.md) | Database Technology Selection | Accepted |
| [ADR-016](./adr/ADR-016-message-queue-choice.md) | Message Queue Choice | Accepted |
| [ADR-017](./adr/ADR-017-observability-stack.md) | Observability Stack | Accepted |
| [ADR-018](./adr/ADR-018-cicd-strategy.md) | CI/CD Strategy | Accepted |
| [ADR-019](./adr/ADR-019-security-hardening-threat-model.md) | Security Hardening & Threat Model | Accepted |

## Design Patterns

### 1. Repository Pattern
```python
class DetectorRepositoryProtocol(Protocol):
    def save(self, detector: Detector) -> None: ...
    def find_by_id(self, detector_id: UUID) -> Optional[Detector]: ...
    def find_all(self) -> List[Detector]: ...
```

### 2. Factory Pattern
```python
class DetectorFactory:
    @staticmethod
    def create(algorithm: str, **config) -> Detector:
        return algorithm_registry.create_detector(algorithm, **config)
```

### 3. Strategy Pattern
```python
class AnomalyDetectionStrategy(Protocol):
    def fit(self, data: np.ndarray) -> None: ...
    def predict(self, data: np.ndarray) -> np.ndarray: ...
```

### 4. Observer Pattern
```python
class DetectionEventPublisher:
    def publish(self, event: DetectionEvent) -> None:
        for subscriber in self._subscribers:
            subscriber.handle(event)
```

## Testing Architecture

### Test Pyramid
```
        ┌─────────────────┐
        │   E2E Tests     │  ← Full system integration
        │    (Few)        │
        └─────────────────┘
      ┌─────────────────────┐
      │ Integration Tests   │  ← Service boundaries
      │     (Some)          │
      └─────────────────────┘
    ┌─────────────────────────┐
    │     Unit Tests          │  ← Domain logic
    │      (Many)             │
    └─────────────────────────┘
```

### Test Organization
- **Unit Tests**: Domain logic and individual components
- **Integration Tests**: Cross-layer interactions
- **Contract Tests**: API and interface compliance
- **End-to-End Tests**: Complete user workflows

## Scalability Considerations

### Horizontal Scaling
- Stateless application design
- Database read replicas
- Cache layer (Redis)
- Load balancing

### Vertical Scaling
- Efficient algorithms
- Memory optimization
- Parallel processing
- GPU acceleration support

## Security Architecture

### Defense in Depth
1. **Network Security**: Firewalls, VPNs, network segmentation
2. **Application Security**: Input validation, authentication, authorization
3. **Data Security**: Encryption at rest and in transit
4. **Infrastructure Security**: Container security, secrets management

### Security Patterns
- **Authentication**: JWT tokens with refresh mechanism
- **Authorization**: Role-based access control (RBAC)
- **Audit Logging**: Comprehensive security event tracking
- **Rate Limiting**: API protection against abuse

## Performance Architecture

### Caching Strategy
- **Application Cache**: In-memory model caching
- **Database Cache**: Query result caching
- **CDN**: Static asset caching
- **API Cache**: Response caching

### Optimization Techniques
- **Lazy Loading**: Load data only when needed
- **Batch Processing**: Process multiple items together
- **Async Processing**: Non-blocking operations
- **Connection Pooling**: Efficient database connections

## Monitoring and Observability

### Three Pillars
1. **Metrics**: Performance and business metrics
2. **Logs**: Structured logging with correlation IDs
3. **Traces**: Distributed tracing across components

### Implementation
- **Prometheus**: Metrics collection
- **Grafana**: Visualization and dashboards
- **Jaeger**: Distributed tracing
- **ELK Stack**: Log aggregation and analysis

## Future Architecture Considerations

### Microservices Migration
- Service decomposition strategy
- Event-driven communication
- Data consistency patterns
- Service mesh implementation

### Cloud-Native Features
- Kubernetes deployment
- Service discovery
- Circuit breakers
- Bulkhead patterns

---

For detailed information on specific architectural aspects:
- [System Design](./system-design.md) - Detailed system architecture
- [Domain Model](./domain-model.md) - Business domain modeling
- [Clean Architecture](./clean-architecture.md) - Implementation details
- [ADRs](./adr/) - Architectural decision records