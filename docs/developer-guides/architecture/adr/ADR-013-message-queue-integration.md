# ADR-013: Message Queue Integration

🍞 **Breadcrumb:** 🏠 [Home](../../../index.md) > 👨‍💻 [Developer Guides](../../README.md) > 🏗️ [Architecture](../README.md) > 📋 [ADR](README.md) > Message Queue Integration

## Status

PROPOSED

## Context

### Problem Statement
Pynomaly needs a robust message queue system to handle asynchronous task processing, event-driven communication between services, and decoupling of system components. The current synchronous processing model creates bottlenecks and limits scalability for computationally intensive anomaly detection tasks.

### Goals
- Enable asynchronous processing of computationally intensive tasks
- Implement event-driven architecture for loose coupling
- Support reliable message delivery with retry mechanisms
- Provide horizontal scaling for message processing
- Enable dead letter queues for failed message handling

### Constraints
- Must integrate with existing streaming engine and caching layer
- Must support message ordering for sequential processing requirements
- Must provide at-least-once delivery guarantees
- Must handle message persistence across system restarts
- Must support message filtering and routing capabilities

### Assumptions
- Message processing latency requirements vary by message type
- System will handle 1000+ messages per second during peak load
- Failed messages require manual intervention or retry logic
- Message payloads will typically be <1MB in size

## Decision

### Chosen Solution
Implement **Apache Kafka** as the primary message queue with **Redis Streams** for real-time messaging.

### Rationale
Kafka provides excellent durability, scalability, and ordering guarantees for high-volume message processing, while Redis Streams offers low-latency messaging for real-time scenarios. This combination enables both reliable batch processing and responsive real-time messaging.

## Architecture

### System Overview
```mermaid
graph TB
    subgraph "Message Producers"
        API[API Services]
        STREAM[Streaming Engine]
        BATCH[Batch Processors]
    end
    
    subgraph "Message Infrastructure"
        KT[Kafka Topics]
        RS[Redis Streams]
        MR[Message Router]
    end
    
    subgraph "Message Consumers"
        AD[Anomaly Detection Workers]
        AL[Alert Processors]
        AU[Audit Processors]
        ME[Metrics Collectors]
    end
    
    subgraph "Message Management"
        DLQ[Dead Letter Queues]
        MRE[Message Retry Engine]
        MM[Message Monitor]
    end
    
    API --> MR
    STREAM --> KT
    BATCH --> KT
    
    MR --> KT
    MR --> RS
    
    KT --> AD
    KT --> AL
    RS --> AU
    RS --> ME
    
    AD --> DLQ
    AL --> DLQ
    MRE --> DLQ
    MM --> KT
    MM --> RS
```

### Component Interactions
```mermaid
sequenceDiagram
    participant P as Producer
    participant MR as Message Router
    participant K as Kafka Topic
    participant CG as Consumer Group
    participant C as Consumer
    participant DLQ as Dead Letter Queue
    participant RE as Retry Engine
    
    P->>MR: send_message(content, routing_key)
    MR->>K: publish_message(topic, partition, message)
    K->>CG: deliver_message(message)
    CG->>C: process_message(message)
    
    alt Success
        C->>CG: ack_message(message)
        CG->>K: commit_offset
    else Failure
        C->>CG: nack_message(message)
        CG->>DLQ: send_to_dlq(message, error)
        DLQ->>RE: schedule_retry(message)
        RE->>K: retry_message(message)
    end
```

## Options Considered

### Pros and Cons Matrix

| Option | Pros | Cons | Score |
|--------|------|------|-------|
| **Kafka + Redis Streams** | ✅ Best of both worlds<br/>✅ Durability + Speed<br/>✅ Mature ecosystem | ❌ Operational complexity<br/>❌ Two systems to manage | **9/10** |
| RabbitMQ | ✅ Feature-rich<br/>✅ Management UI<br/>✅ Flexible routing | ❌ Limited scalability<br/>❌ Single point of failure | 7/10 |
| AWS SQS/SNS | ✅ Managed service<br/>✅ High availability | ❌ Vendor lock-in<br/>❌ Cost implications | 6/10 |

### Rejected Alternatives
- **RabbitMQ**: Limited horizontal scaling capabilities and complex clustering setup
- **AWS SQS/SNS**: Vendor lock-in concerns and cost implications for high-volume messaging

## Implementation

### Technical Approach
1. Set up Kafka cluster infrastructure
2. Implement Redis Streams for real-time messaging
3. Create message routing and handling framework
4. Add dead letter queues and retry mechanisms
5. Implement monitoring and alerting

### Migration Strategy
1. **Phase 1**: Implement message queues alongside existing synchronous processing
2. **Phase 2**: Migrate computationally intensive tasks to async processing
3. **Phase 3**: Transition remaining workflows to event-driven architecture

### Testing Strategy
- Unit tests for message producers and consumers
- Integration tests with streaming engine and caching layer
- Performance tests for message throughput and latency
- Chaos engineering for message delivery reliability

## Consequences

### Positive
- Improved system scalability and performance
- Better fault tolerance and reliability
- Decoupled system components
- Asynchronous processing capabilities
- Enhanced monitoring and observability

### Negative
- Increased operational complexity
- Additional infrastructure requirements
- Message ordering complexity
- Potential message duplication scenarios
- Learning curve for event-driven patterns

### Neutral
- Changes to deployment architecture
- New configuration and monitoring requirements
- Additional testing scenarios for message handling
- Training requirements for message queue operations

## Compliance

### Security Impact
- Message encryption for sensitive data
- Authentication and authorization for message access
- Audit logging for message processing events

### Performance Impact
- Target: >1000 messages per second throughput
- Target: <100ms message processing latency
- Reliability: >99.9% message delivery success rate

### Monitoring Requirements
- Message throughput and latency metrics
- Dead letter queue size and retry success rates
- Consumer lag and processing time
- Message delivery success rate

## Decision Log

| Date | Author | Action | Rationale |
|------|--------|--------|-----------|
| 2025-01-07 | Architecture Team | PROPOSED | Initial proposal for message queue integration |

## References

- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Redis Streams Documentation](https://redis.io/docs/data-types/streams/)
- [Event-Driven Architecture Patterns](https://microservices.io/patterns/data/event-driven-architecture.html)

---

## 🔗 **Related Documentation**

### **Architecture**
- **[Architecture Overview](../overview.md)** - System design principles
- **[Clean Architecture](../overview.md)** - Architectural patterns
- **[ADR Index](README.md)** - All architectural decisions

### **Implementation**
- **[Implementation Guide](../../contributing/IMPLEMENTATION_GUIDE.md)** - Coding standards
- **[Contributing Guidelines](../../contributing/CONTRIBUTING.md)** - Development process
- **[File Organization](../../contributing/FILE_ORGANIZATION_STANDARDS.md)** - Project structure

### **Deployment**
- **[Production Deployment](../../../deployment/README.md)** - Production setup
- **[Security](../../../deployment/SECURITY.md)** - Security configuration
- **[Monitoring](../../../user-guides/basic-usage/monitoring.md)** - System observability

---

**Authors:** Architecture Team<br/>
**Last Updated:** 2025-01-07<br/>
**Next Review:** 2025-04-07
