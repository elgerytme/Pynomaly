# Package Isolation Rule

## Core Principle

**No package shall reference the domain, entities, business logic, or implementation details of another package except through simple import statements and the specific code or interface imported from that package.**

## Exception Rules

This rule has defined exceptions for reasonable cases. See [PACKAGE_ISOLATION_EXCEPTIONS.md](PACKAGE_ISOLATION_EXCEPTIONS.md) for comprehensive exception guidelines covering:
- Third-party dependencies (node_modules)
- Domain-appropriate usage within domain packages
- User role terminology in interface documentation
- Technical context terms in documentation

## Rules

### 1. Documentation Isolation
- Each package's README, documentation, and comments must **only** describe its own functionality
- No package documentation shall describe, explain, or reference another package's:
  - Domain entities
  - Business rules
  - Internal implementation
  - Use cases
  - Domain-specific terminology

### 2. Testing Isolation
- Each package tests **only** its own functionality
- No package shall contain tests for another package's functionality
- Integration tests must focus on the interface contract, not internal implementation

### 3. Import Isolation
- Packages may **only** import:
  - Public interfaces and contracts
  - Exported classes, functions, and types
  - Standard library modules
  - Third-party dependencies

### 4. Domain Boundary Enforcement
- No package shall reference domain-specific concepts from other packages
- Domain knowledge must remain within the owning package
- Cross-package communication happens through generic interfaces

### 5. Naming Isolation
- Package names, class names, and function names must not reference other domains
- Generic naming that describes the capability, not the domain
- Domain-agnostic terminology for shared components

## Allowed Interactions

```python
# ✅ ALLOWED: Simple import and usage
from pynomaly.math.statistics import calculate_mean
result = calculate_mean(data)

# ✅ ALLOWED: Interface contract usage
from pynomaly.interfaces.detection import DetectionInterface
detector = DetectionInterface()

# ✅ ALLOWED: Generic utility usage
from pynomaly.core.validation import validate_data
is_valid = validate_data(input_data)
```

## Prohibited Interactions

```python
# ❌ PROHIBITED: Domain-specific knowledge
# Package A knowing about Package B's domain entities
from pynomaly.anomaly_detection.domain import AnomalyScore  # Wrong domain knowledge

# ❌ PROHIBITED: Implementation details
# Package A testing Package B's internal logic
def test_anomaly_detection_scoring():  # Wrong - not your domain
    pass

# ❌ PROHIBITED: Cross-domain documentation
"""
This package handles user authentication and integrates with the
anomaly detection system to provide security for ML models.
"""  # Wrong - references another package's domain
```

## Documentation Guidelines

### Package README Structure
```markdown
# Package Name

## Overview
Brief description of THIS package's purpose and functionality.

## Features
- Feature 1 (of this package)
- Feature 2 (of this package)
- Feature 3 (of this package)

## Installation
How to install THIS package.

## Usage
How to use THIS package's functionality.

## API Reference
THIS package's API only.
```

### Prohibited Documentation Patterns
- ❌ "This package works with the anomaly detection system..."
- ❌ "Integrates with the ML pipeline to provide..."
- ❌ "Handles user data for anomaly detection..."
- ❌ "Provides authentication for the detection dashboard..."

### Allowed Documentation Patterns
- ✅ "This package provides authentication services..."
- ✅ "Generic data validation utilities..."
- ✅ "HTTP client with retry logic..."
- ✅ "Statistical computation functions..."

## Enforcement Mechanisms

### 1. Architectural Validators
```python
# Automated validation rules
- No cross-package domain references in docstrings
- No cross-package entity imports
- No cross-package test files
- No domain-specific naming in generic packages
```

### 2. Code Review Guidelines
- Review for domain boundary violations
- Check for inappropriate cross-package references
- Validate documentation isolation
- Ensure generic naming conventions

### 3. CI/CD Checks
- Automated scanning for rule violations
- Documentation content validation
- Import dependency analysis
- Domain boundary enforcement

## Migration Guidelines

### Existing Violations
1. Identify cross-package domain references
2. Extract domain-specific code to appropriate packages
3. Create generic interfaces for cross-package communication
4. Update documentation to remove domain references
5. Rename domain-specific components to generic names

### Refactoring Approach
1. **Domain Extraction**: Move domain logic to owning package
2. **Interface Creation**: Create generic interfaces for communication
3. **Documentation Cleanup**: Remove cross-domain references
4. **Naming Standardization**: Use generic names for shared components
5. **Testing Isolation**: Move tests to appropriate packages

## Examples

### Before (Violation)
```python
# In authentication package
class AnomalyDetectionAuth:
    """Authentication for anomaly detection system."""
    
    def authenticate_ml_user(self, user):
        """Authenticate user for ML model access."""
        pass

# Documentation mentions anomaly detection specifics
# Class names reference other domains
# Methods have domain-specific names
```

### After (Compliant)
```python
# In authentication package
class ServiceAuth:
    """Authentication service for API access."""
    
    def authenticate_user(self, user):
        """Authenticate user for service access."""
        pass

# Generic naming and documentation
# No domain-specific references
# Focuses on authentication capability
```

## Benefits

1. **Clear Boundaries**: Each package has well-defined responsibilities
2. **Maintainability**: Changes in one package don't affect others
3. **Testability**: Each package can be tested independently
4. **Reusability**: Generic components can be reused across domains
5. **Scalability**: New domains can be added without affecting existing packages

## Enforcement Tools

### Validation Script
```bash
# Check for package isolation violations
python tools/validate_package_isolation.py

# Scan for cross-domain references
python tools/scan_domain_violations.py

# Validate documentation isolation
python tools/validate_docs_isolation.py
```

This rule ensures clean architecture, maintainable code, and proper domain separation throughout the anomaly detection platform.