# ADR-005: Production Database Technology Selection

**Status:** DRAFT  
**Date:** 2025-01-08  
**GitHub Issue:** [#35](https://github.com/elgerytme/Pynomaly/issues/35)

## Context

We need to formalize our architectural decision regarding the selection of production database technology for the Pynomaly project.

### Key Considerations:
- Performance requirements for anomaly detection data
- Scalability needs for time-series data
- ACID compliance requirements
- Operational complexity and maintenance
- Cost considerations for cloud deployment
- Integration with existing monitoring and analytics tools

## TODO

- [ ] Evaluate database options (PostgreSQL, MySQL, SQLite, etc.)
- [ ] Document performance requirements and benchmarks
- [ ] Define scalability considerations
- [ ] Document migration strategy from current SQLite
- [ ] Define backup and recovery procedures
- [ ] Document operational monitoring requirements
- [ ] Define data retention and archiving policies
- [ ] Create deployment and configuration guidelines

## Decision

**[TO BE COMPLETED]**

## Consequences

**[TO BE COMPLETED]**

## Status

This ADR is currently in **DRAFT** status and requires completion.

## Related ADRs

- ADR-004: Repository & Unit-of-Work Pattern
- ADR-007: Observability Stack (OpenTelemetry + Prometheus + Grafana)

## References

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Database Selection Guide](https://cloud.google.com/database/database-guide)
