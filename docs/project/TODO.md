# Pynomaly TODO List

## 🎯 Overview

This backlog organizes tasks by Clean Architecture layers, with each task including ID, priority, estimate, owner assignment, and dependency links. All "Recently Completed" items have been moved to CHANGELOG.md to keep this document strictly forward-looking.

---

## Technical Debt & Project Organization
*Immediate priorities for code quality and project structure*

### TD-001: Address Circular Import Issues
- **Priority**: Critical
- **Estimate**: 2 days
- **Owner**: TBD
- **Dependencies**: None
- **Description**: Resolve circular import dependencies that are blocking proper module loading. Debug files indicate this is an active issue affecting core functionality.

### TD-002: Consolidate Test Files
- **Priority**: High
- **Estimate**: 1 day
- **Owner**: TBD
- **Dependencies**: None
- **Description**: Move test files scattered in root directory to proper test organization structure. Clean up test_*.py files from project root.

### TD-003: Project Structure Cleanup
- **Priority**: High
- **Estimate**: 2 days
- **Owner**: TBD
- **Dependencies**: None
- **Description**: Organize analysis files, temporary files, and development artifacts in root directory into appropriate subdirectories.

### TD-004: Configuration Management Streamlining
- **Priority**: Medium
- **Estimate**: 1 day
- **Owner**: TBD
- **Dependencies**: None
- **Description**: Streamline config files and environment setup. Consolidate redundant configuration files and standardize environment management.

### TD-005: Core Features Testing & Validation
- **Priority**: Critical
- **Estimate**: 3 days
- **Owner**: TBD
- **Dependencies**: TD-001
- **Description**: Ensure PyOD and scikit-learn integrations are fully functional and test suite passes with claimed 85%+ coverage.

### TD-006: Development Roadmap Creation
- **Priority**: High
- **Estimate**: 1 day
- **Owner**: TBD
- **Dependencies**: None
- **Description**: Create clear development roadmap prioritizing Beta/Experimental features for implementation. Update README to separate working vs. needs-work features.

### TD-007: Pre-commit Hooks Setup
- **Priority**: Medium
- **Estimate**: 0.5 days
- **Owner**: TBD
- **Dependencies**: None
- **Description**: Ensure pre-commit hooks are properly configured and running with ruff, mypy, and other quality tools.

### TD-008: Dependency Review & Updates
- **Priority**: Medium
- **Estimate**: 1 day
- **Owner**: TBD
- **Dependencies**: None
- **Description**: Review pyproject.toml for dependency conflicts, outdated packages, and optimization opportunities.

---

## Domain Layer
*Business logic and core entities*

### D-001: Enhanced Domain Entity Validation
- **Priority**: High
- **Estimate**: 3 days  
- **Owner**: TBD
- **Dependencies**: None
- **Description**: Implement advanced validation rules for AnomalyScore, ContaminationRate, and DetectionResult entities to ensure business rule compliance

### D-002: Advanced Anomaly Classification
- **Priority**: Medium
- **Estimate**: 5 days
- **Owner**: TBD
- **Dependencies**: None
- **Description**: Extend anomaly types beyond binary classification to support severity levels and categorical anomalies

### D-003: Model Performance Degradation Detection
- **Priority**: High
- **Estimate**: 4 days
- **Owner**: TBD
- **Dependencies**: None
- **Description**: Implement domain logic for detecting when model performance drops below acceptable thresholds

---

## Application Layer
*Use cases and orchestration*

### A-001: Automated Model Retraining Workflows
- **Priority**: High
- **Estimate**: 6 days
- **Owner**: TBD
- **Dependencies**: D-003
- **Description**: Create use cases for automated model retraining based on performance degradation triggers

### A-002: Batch Processing Orchestration
- **Priority**: Medium
- **Estimate**: 4 days
- **Owner**: TBD
- **Dependencies**: None
- **Description**: Implement use cases for processing large datasets in configurable batch sizes

### A-003: Model Comparison and Selection
- **Priority**: Medium
- **Estimate**: 3 days
- **Owner**: TBD
- **Dependencies**: D-002
- **Description**: Orchestrate multi-algorithm comparison workflows with statistical significance testing

---

## Infrastructure Layer
*External integrations and technical concerns*

### I-001: Production Database Integration
- **Priority**: Critical
- **Estimate**: 8 days
- **Owner**: TBD
- **Dependencies**: None
- **Description**: Replace file-based storage with PostgreSQL/MongoDB for production scalability

### I-002: Deep Learning Framework Integration
- **Priority**: High
- **Estimate**: 10 days
- **Owner**: TBD
- **Dependencies**: None
- **Description**: Complete PyTorch/TensorFlow adapter implementations (currently stubs)

### I-003: Message Queue Integration
- **Priority**: Medium
- **Estimate**: 5 days
- **Owner**: TBD
- **Dependencies**: I-001
- **Description**: Implement Redis/RabbitMQ for asynchronous task processing

### I-004: External Monitoring System Integration
- **Priority**: Medium
- **Estimate**: 4 days
- **Owner**: TBD
- **Dependencies**: None
- **Description**: Complete Prometheus/Grafana integration with custom dashboards

### I-005: Cloud Storage Adapters
- **Priority**: Low
- **Estimate**: 6 days
- **Owner**: TBD
- **Dependencies**: I-001
- **Description**: Implement AWS S3, Azure Blob, GCP Storage adapters for large dataset handling

---

## Presentation Layer
*User interfaces and APIs*

### P-001: Advanced Analytics Dashboard
- **Priority**: High
- **Estimate**: 8 days
- **Owner**: TBD
- **Dependencies**: A-003
- **Description**: Build comprehensive analytics dashboard with real-time model performance visualization

### P-002: Mobile-Responsive UI Enhancements
- **Priority**: Medium
- **Estimate**: 5 days
- **Owner**: TBD
- **Dependencies**: None
- **Description**: Optimize web interface for mobile devices and tablet usage

### P-003: CLI Command Completion
- **Priority**: High
- **Estimate**: 3 days
- **Owner**: TBD
- **Dependencies**: I-002
- **Description**: Enable remaining disabled CLI commands (security, dashboard, governance)

### P-004: GraphQL API Layer
- **Priority**: Low
- **Estimate**: 7 days
- **Owner**: TBD
- **Dependencies**: I-001
- **Description**: Add GraphQL endpoints for flexible data querying alongside REST API

### P-005: OpenAPI Schema Fixes
- **Priority**: Medium
- **Estimate**: 2 days
- **Owner**: TBD
- **Dependencies**: None
- **Description**: Resolve Pydantic forward reference issues preventing OpenAPI documentation generation

---

## CI/CD Layer
*Build, test, and deployment automation*

### C-001: Automated Dependency Vulnerability Scanning
- **Priority**: High
- **Estimate**: 2 days
- **Owner**: TBD
- **Dependencies**: None
- **Description**: Integrate automated dependency scanning with Snyk/Dependabot for security monitoring

### C-002: Multi-Environment Deployment Pipeline
- **Priority**: High
- **Estimate**: 5 days
- **Owner**: TBD
- **Dependencies**: I-001
- **Description**: Create staging and production deployment pipelines with environment-specific configurations

### C-003: Performance Regression Testing
- **Priority**: Medium
- **Estimate**: 4 days
- **Owner**: TBD
- **Dependencies**: None
- **Description**: Implement automated performance benchmarking in CI pipeline

### C-004: Container Security Scanning
- **Priority**: Medium
- **Estimate**: 2 days
- **Owner**: TBD
- **Dependencies**: None
- **Description**: Add container vulnerability scanning with Trivy/Clair in Docker builds

---

## Documentation Layer
*Documentation and knowledge management*

### DOC-001: API Documentation Completion
- **Priority**: High
- **Estimate**: 3 days
- **Owner**: TBD
- **Dependencies**: P-005
- **Description**: Complete OpenAPI documentation with examples for all 65+ endpoints

### DOC-002: User Guide Video Tutorials
- **Priority**: Medium
- **Estimate**: 6 days
- **Owner**: TBD
- **Dependencies**: P-001
- **Description**: Create video tutorials for common workflows and dashboard usage

### DOC-003: Architecture Decision Records (ADRs)
- **Priority**: Medium
- **Estimate**: 4 days
- **Owner**: TBD
- **Dependencies**: None
- **Description**: Document architectural decisions and trade-offs for future reference

### DOC-004: Performance Benchmarking Guide
- **Priority**: Low
- **Estimate**: 2 days
- **Owner**: TBD
- **Dependencies**: C-003
- **Description**: Create comprehensive guide for performance testing and optimization

### DOC-005: Security Best Practices Guide
- **Priority**: High
- **Estimate**: 3 days
- **Owner**: TBD
- **Dependencies**: C-001
- **Description**: Document security configurations, threat model, and mitigation strategies

---

## Priority Summary

### Critical (Must Have)
- TD-001: Address Circular Import Issues
- TD-005: Core Features Testing & Validation
- I-001: Production Database Integration

### High Priority  
- TD-002: Consolidate Test Files
- TD-003: Project Structure Cleanup
- TD-006: Development Roadmap Creation
- D-001: Enhanced Domain Entity Validation
- D-003: Model Performance Degradation Detection
- A-001: Automated Model Retraining Workflows
- I-002: Deep Learning Framework Integration
- P-001: Advanced Analytics Dashboard
- P-003: CLI Command Completion
- C-001: Automated Dependency Vulnerability Scanning
- C-002: Multi-Environment Deployment Pipeline
- DOC-001: API Documentation Completion
- DOC-005: Security Best Practices Guide

### Medium Priority
- TD-004: Configuration Management Streamlining
- TD-007: Pre-commit Hooks Setup
- TD-008: Dependency Review & Updates
- D-002: Advanced Anomaly Classification
- A-002: Batch Processing Orchestration
- A-003: Model Comparison and Selection
- I-003: Message Queue Integration
- I-004: External Monitoring System Integration
- P-002: Mobile-Responsive UI Enhancements
- P-005: OpenAPI Schema Fixes
- C-003: Performance Regression Testing
- C-004: Container Security Scanning
- DOC-002: User Guide Video Tutorials
- DOC-003: Architecture Decision Records

### Low Priority
- I-005: Cloud Storage Adapters
- P-004: GraphQL API Layer
- DOC-004: Performance Benchmarking Guide

---

## Dependency Graph

```
Critical Path (Technical Debt First):
TD-001 → TD-005 → I-001 → C-002 → Production Deployment

Immediate Technical Debt Chain:
TD-001 (Circular imports) → TD-005 (Core testing) → All other development

High Priority Chains:
TD-002, TD-003, TD-006 (Project organization) → Parallel development
D-003 → A-001 (Performance monitoring → Automated retraining)
A-003 → P-001 (Model comparison → Analytics dashboard)
P-005 → DOC-001 (Schema fixes → API documentation)

Supporting Infrastructure:
I-002 → P-003 (Deep learning → CLI completion)
I-001 → I-003 → I-005 (Database → Message queue → Cloud storage)
C-001 → DOC-005 (Vulnerability scanning → Security guide)
C-003 → DOC-004 (Performance testing → Benchmarking guide)

Foundational Tasks:
TD-004, TD-007, TD-008 (Configuration, pre-commit, dependencies) → Code quality
```

---

*Last updated: 2025-01-08*
*Total estimated effort: 136.5 days*
*Critical path duration: ~26 days*
*Technical Debt tasks added: 8 (11.5 days estimated)*
