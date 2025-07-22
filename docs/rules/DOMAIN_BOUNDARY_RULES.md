# Domain Boundary Rules

This document establishes the domain boundary rules for the repository to ensure clean architecture and proper separation of concerns.

## Overview

The repository follows a domain-driven design (DDD) approach with clear boundaries between different domains. Each package must maintain its domain independence and not reference concepts from other domains.

## Domain Packages

### 1. `software` Package
**Purpose**: Contains generic software infrastructure components that are domain-agnostic.

**Allowed Concepts**:
- Generic software patterns (services, repositories, entities, value objects)
- Infrastructure concerns (databases, caching, networking)
- Application architecture patterns
- Cross-cutting concerns (logging, monitoring, security)
- Generic interfaces and protocols

**Prohibited Concepts**:
- Domain-specific business logic
- References to ML/AI concepts
- Anomaly detection terminology
- Data science specific patterns
- Domain-specific metrics or algorithms

### 2. `domain_library` Package
**Purpose**: Contains reusable domain modeling patterns and templates.

**Allowed Concepts**:
- Generic domain patterns
- Entity and value object templates
- Business logic templates
- Domain relationship patterns

**Prohibited Concepts**:
- Specific domain implementations
- Technology-specific code
- Domain-specific business rules

### 3. `enterprise` Package
**Purpose**: Contains enterprise-specific infrastructure and multi-tenancy features.

**Allowed Concepts**:
- Multi-tenant architecture
- Enterprise authentication and authorization
- Organization and subscription management
- Enterprise-grade monitoring and compliance

**Prohibited Concepts**:
- Domain-specific business logic
- Application-specific features
- Domain-specific algorithms

### 4. `anomaly_detection` Package
**Purpose**: Contains anomaly detection specific domain logic.

**Allowed Concepts**:
- Anomaly detection algorithms
- Threshold management
- Severity scoring
- Alert management
- Detection-specific entities and services

### 5. `machine_learning` Package
**Purpose**: Contains machine learning infrastructure and algorithms.

**Allowed Concepts**:
- ML model management
- Training and inference pipelines
- AutoML capabilities
- Model performance tracking
- ML-specific entities and services

### 6. `data_science` Package
**Purpose**: Contains data science workflows and analytical capabilities.

**Allowed Concepts**:
- Data processing pipelines
- Statistical analysis
- Data validation and quality
- Analytical workflows
- Data-specific entities and services

## Domain Validation Rules

### Software Package Rules
The software package must NOT contain references to:
- `anomaly_detection`, `anomaly`, `anomaly_detection`, `outlier`, `detection`
- `automl`, `ensemble`, `explainability`, `explainable_ai`
- `drift`, `model`, `training`, `dataset`, `preprocessing`
- `contamination`, `threshold`, `severity`, `confidence_interval`
- `hyperparameters`, `model_metrics`, `performance_metrics`
- `active_learning`, `continuous_learning`, `streaming_session`
- ML/AI library references (`sklearn`, `tensorflow`, `pytorch`, etc.)
- Domain-specific business concepts (`fraud`, `intrusion`, `cybersecurity`)

### Configuration Rules
All `pyproject.toml` files must:
- Use domain-appropriate package names (no "anomaly_detection" references)
- Include domain-appropriate descriptions
- Use domain-appropriate keywords and classifiers
- Reference domain-appropriate documentation URLs
- Use domain-appropriate email addresses

### Code Organization Rules
1. **Entity Placement**: Domain-specific entities must be placed in their respective domain packages
2. **Service Placement**: Domain-specific services must be placed in their respective domain packages
3. **Value Object Placement**: Domain-specific value objects must be placed in their respective domain packages
4. **Interface Separation**: Generic interfaces in software package, domain-specific implementations in domain packages

## Validation Process

### Automated Validation
Use the `scripts/domain_boundary_validator.py` script to validate domain boundaries:

```bash
python scripts/domain_boundary_validator.py
```

The validator will:
- Check all packages for domain boundary violations
- Generate a detailed report of violations
- Provide recommendations for fixes
- Exit with error code if violations are found

### Manual Review Process
1. **Code Review**: All changes must be reviewed for domain boundary compliance
2. **Architecture Review**: Significant architectural changes require architecture team approval
3. **Documentation Review**: Domain boundary documentation must be updated for architectural changes

## Common Violations and Fixes

### 1. Domain-Specific Entities in Software Package
**Problem**: ML/AI specific entities in `software/core/domain/entities/`
**Fix**: Move to appropriate domain package (e.g., `machine_learning/core/domain/entities/`)

### 2. Domain-Specific Configuration
**Problem**: References to "anomaly_detection" or "anomaly" in package names/descriptions
**Fix**: Use generic software terminology

### 3. Domain-Specific Dependencies
**Problem**: Software package depending on domain-specific libraries
**Fix**: Move domain-specific dependencies to appropriate domain packages

### 4. Cross-Domain References
**Problem**: One domain package directly referencing another's internal types
**Fix**: Use generic interfaces and dependency injection

## Enforcement

### CI/CD Integration
The domain boundary validator is integrated into the CI/CD pipeline:
- Runs on every pull request
- Blocks merging if violations are found
- Generates violation reports for review

### Git Hooks
Pre-commit hooks run domain boundary validation:
- Prevents commits with domain violations
- Provides immediate feedback during development
- Maintains code quality standards

## Benefits

1. **Maintainability**: Clear separation makes code easier to maintain
2. **Testability**: Domain isolation improves testability
3. **Reusability**: Generic components can be reused across domains
4. **Scalability**: New domains can be added without affecting existing code
5. **Team Autonomy**: Teams can work independently on their domains

## Migration Guide

When moving code between domains:

1. **Identify Dependencies**: Map all dependencies of the code being moved
2. **Update Imports**: Change import statements to reference new locations
3. **Update Tests**: Move and update corresponding test files
4. **Update Documentation**: Update references in documentation
5. **Validate**: Run domain boundary validator to ensure compliance

## Examples

### ✅ Correct Domain Usage

```python
# In software/core/domain/abstractions/base_entity.py
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseEntity(ABC):
    """Generic base entity for all domain entities"""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation"""
        pass
```

```python
# In anomaly_detection/core/domain/entities/alert.py
from software.core.domain.abstractions.base_entity import BaseEntity
from anomaly_detection.core.domain.value_objects.severity_score import SeverityScore

class Alert(BaseEntity):
    """Alert entity for anomaly detection domain"""
    
    def __init__(self, severity: SeverityScore):
        self.severity = severity
```

### ❌ Incorrect Domain Usage

```python
# WRONG: In software/core/domain/entities/anomaly_alert.py
class AnomalyAlert:
    """This should be in anomaly_detection domain"""
    
    def __init__(self, contamination_rate: float):
        self.contamination_rate = contamination_rate
```

## Contact

For questions about domain boundaries or architecture decisions:
- Architecture Team: architecture@software.io
- Domain Boundary Questions: domains@software.io
- Technical Issues: tech@software.io