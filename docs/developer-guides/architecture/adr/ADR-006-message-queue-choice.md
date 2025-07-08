# ADR-006: Message Queue Choice (Redis vs. RabbitMQ vs. Kafka)

**Status:** DRAFT  
**Date:** 2025-01-08  
**GitHub Issue:** [#36](https://github.com/elgerytme/Pynomaly/issues/36)

## Context

We need to formalize our architectural decision regarding the selection of message queue technology for the Pynomaly project's async processing and event-driven architecture.

### Key Considerations:
- Throughput and latency requirements for real-time anomaly detection
- Message persistence and durability needs
- Scalability requirements for distributed processing
- Operational complexity and maintenance overhead
- Integration with existing monitoring and processing systems
- Cost considerations for cloud deployment

## TODO

- [ ] Evaluate message queue options (Redis, RabbitMQ, Kafka)
- [ ] Document throughput and latency requirements
- [ ] Define durability and persistence needs
- [ ] Document scaling considerations
- [ ] Define integration patterns with anomaly detection pipeline
- [ ] Document operational monitoring requirements
- [ ] Define disaster recovery and failover strategies
- [ ] Create deployment and configuration guidelines

## Decision

**[TO BE COMPLETED]**

## Consequences

**[TO BE COMPLETED]**

## Status

This ADR is currently in **DRAFT** status and requires completion.

## Related ADRs

- ADR-003: Clean Architecture & DDD Adoption
- ADR-007: Observability Stack (OpenTelemetry + Prometheus + Grafana)

## References

- [Redis Documentation](https://redis.io/docs/)
- [RabbitMQ Documentation](https://www.rabbitmq.com/documentation.html)
- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
