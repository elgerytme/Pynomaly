# Architectural Decision Record

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 👨‍💻 [Developer Guides](../README.md) > 🏗️ [Architecture](README.md)

---

## Overview

This directory contains Architectural Decision Records (ADRs) for the Pynomaly project. ADRs document important architectural decisions and their context, consequences, and status.

## ADR Index

### 📋 Draft ADRs (Awaiting Completion)

| ADR | Title | Status | GitHub Issue |
|-----|-------|--------|--------------|
| [ADR-003](ADR-003-clean-architecture-ddd-adoption.md) | Clean Architecture & DDD Adoption | 🚧 DRAFT | [#40](https://github.com/elgerytme/Pynomaly/issues/40) |
| [ADR-004](ADR-004-repository-unit-of-work-pattern.md) | Repository & Unit-of-Work Pattern | 🚧 DRAFT | [#34](https://github.com/elgerytme/Pynomaly/issues/34) |
| [ADR-005](ADR-005-production-database-technology-selection.md) | Production Database Technology Selection | 🚧 DRAFT | [#35](https://github.com/elgerytme/Pynomaly/issues/35) |
| [ADR-006](ADR-006-message-queue-choice.md) | Message Queue Choice (Redis vs. RabbitMQ vs. Kafka) | 🚧 DRAFT | [#36](https://github.com/elgerytme/Pynomaly/issues/36) |
| [ADR-007](ADR-007-observability-stack.md) | Observability Stack (OpenTelemetry + Prometheus + Grafana) | 🚧 DRAFT | [#37](https://github.com/elgerytme/Pynomaly/issues/37) |
| [ADR-008](ADR-008-cicd-strategy.md) | CI/CD Strategy (GitHub Actions + Docker + Dev/Prod envs) | 🚧 DRAFT | [#38](https://github.com/elgerytme/Pynomaly/issues/38) |
| [ADR-009](ADR-009-security-hardening-threat-model.md) | Security Hardening & Threat Model | 🚧 DRAFT | [#39](https://github.com/elgerytme/Pynomaly/issues/39) |

### 📊 ADR Status Legend

- 🚧 **DRAFT**: Initial context defined, decision pending
- ✅ **ACCEPTED**: Decision made and approved
- 🔄 **SUPERSEDED**: Replaced by newer ADR
- ❌ **DEPRECATED**: No longer relevant

## Contributing to ADRs

1. **Creating new ADRs**: Follow the format in existing stubs
2. **Updating ADRs**: Complete the TODO items and decision sections
3. **Reviewing ADRs**: Participate in GitHub issue discussions
4. **Linking ADRs**: Reference related ADRs in the "Related ADRs" section

## ADR Template Structure

Each ADR should include:
- **Status**: Current state (DRAFT, ACCEPTED, SUPERSEDED, DEPRECATED)
- **Context**: Background and motivations
- **Decision**: The architectural decision made
- **Consequences**: Positive and negative outcomes
- **Related ADRs**: Links to dependent or related decisions

---

**Note**: These ADRs are currently in DRAFT status and require completion. Please see the linked GitHub issues for current progress and discussion.
