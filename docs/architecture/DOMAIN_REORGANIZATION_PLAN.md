# Domain Reorganization Plan

## Overview

This document outlines the comprehensive reorganization of the repository into logical domains following domain-driven design principles. The goal is to improve code organization, reduce coupling, and create clearer boundaries between different business domains.

## Current Issues Addressed

1. **Scattered anomaly detection logic**: `pynomaly-detection/` package moved to standard `src/packages/data/anomaly_detection/` structure
2. **Overlapping responsibilities**: Multiple packages handling similar functionality (algorithms, core detection, mlops)
3. **Inconsistent package structure**: Some packages follow standards, others don't
4. **Mixed concerns**: Packages containing multiple unrelated domains

## New Domain Structure

### Primary Domains

#### 1. `anomaly_detection/` - Core Anomaly Detection Domain
**Purpose**: Consolidated anomaly and outlier detection functionality
**Status**: ✅ Implemented

**Sources Consolidated**:
- `pynomaly-detection/src/pynomaly_detection/` → `src/packages/data/anomaly_detection/`
- `src/packages/algorithms/adapters/` → `src/packages/anomaly_detection/adapters/`
- Core detection logic from `src/packages/core/`

**Key Features**:
- 40+ detection algorithms (PyOD, scikit-learn, PyTorch, TensorFlow, JAX)
- Clean architecture with domain-driven design
- AutoML capabilities for algorithm selection
- Explainable AI with SHAP and LIME
- Real-time and batch processing
- Enterprise features (multi-tenancy, monitoring)

#### 2. `machine_learning/` - ML Operations & Lifecycle
**Purpose**: Comprehensive ML operations, training, and lifecycle management
**Status**: ✅ Implemented

**Structure**:
```
machine_learning/
├── training/           # Model training workflows
├── optimization/       # Hyperparameter optimization
├── lifecycle/          # Model versioning and registry
├── monitoring/         # Performance monitoring
├── experiments/        # Experiment tracking
└── governance/         # ML compliance
```

**Key Features**:
- Model training pipelines
- Hyperparameter optimization (Optuna, Hyperopt)
- Model registry and versioning
- Real-time performance monitoring
- Experiment tracking and management
- ML governance and compliance

#### 3. `people_ops/` - User Management & Authentication
**Purpose**: Comprehensive user operations and security
**Status**: ✅ Implemented

**Structure**:
```
people_ops/
├── authentication/     # JWT, OAuth, MFA
├── authorization/      # RBAC, permissions
├── user_management/    # User lifecycle
├── compliance/         # Audit, privacy
├── sessions/           # Session management
└── policies/           # Security policies
```

**Key Features**:
- Multi-factor authentication
- Role-based access control
- User lifecycle management
- Audit trails and compliance
- Enterprise integration (LDAP, SAML)

### Supporting Domains

#### 4. `mathematics/` - Statistical Analysis
**Purpose**: Mathematical computations and statistical analysis
**Status**: ✅ Existing, enhanced

**Responsibilities**:
- Statistical utilities
- Mathematical computations
- Shared algorithms

#### 5. `data_platform/` - Data Operations
**Purpose**: Data processing, quality, and transformation
**Status**: ✅ Existing, reorganized

**Subdomains**:
- Data profiling
- Quality assessment
- Transformation pipelines
- Data science utilities

#### 6. `infrastructure/` - Technical Infrastructure
**Purpose**: Deployment, monitoring, and technical concerns
**Status**: ✅ Existing, enhanced

**Responsibilities**:
- Deployment configurations
- Monitoring systems
- Persistence layers
- External integrations

#### 7. `interfaces/` - User Interfaces
**Purpose**: All user-facing interfaces
**Status**: ✅ Existing, organized by interface type

**Organization**:
- `api/` - REST APIs
- `cli/` - Command-line interfaces
- `web/` - Web applications
- `sdks/` - Software development kits

#### 8. `enterprise/` - Enterprise Features
**Purpose**: Enterprise-grade features and governance
**Status**: ✅ Existing, enhanced

**Responsibilities**:
- Multi-tenancy
- Governance frameworks
- Compliance tools
- Enterprise integrations

### New Infrastructure

#### 9. `pkg/` - Third-Party Packages
**Purpose**: Vendored dependencies and custom forks
**Status**: ✅ Implemented

**Structure**:
```
pkg/
├── vendor_dependencies/    # Vendored third-party packages
└── custom_forks/          # Customized open-source forks
```

## Migration Strategy

### Phase 1: Core Domain Creation ✅ Complete
- [x] Create `anomaly_detection/` package
- [x] Move `pynomaly-detection/` content to `data/anomaly_detection/`
- [x] Consolidate algorithm adapters
- [x] Create `machine_learning/` package
- [x] Create `people_ops/` package
- [x] Create `pkg/` directory

### Phase 2: Build System Updates ✅ Complete
- [x] Update main `BUCK` file with new package dependencies
- [x] Create individual `BUCK` files for new packages
- [x] Update package-specific `pyproject.toml` files
- [x] Add proper dependency declarations

### Phase 3: Governance Updates 🔄 In Progress
- [ ] Update package structure enforcer
- [ ] Update documentation checker
- [ ] Update dependency validation
- [ ] Update pre-commit hooks

### Phase 4: Testing & Validation 🕒 Pending
- [ ] Update test configurations
- [ ] Validate build system
- [ ] Run comprehensive tests
- [ ] Update CI/CD pipelines

### Phase 5: Documentation & Cleanup 🕒 Pending
- [ ] Update main README
- [ ] Create migration guides
- [ ] Archive old packages
- [ ] Update dependency documentation

## Package Dependencies

### Dependency Graph
```
Core Domain Layer:
├── core (base domain logic)
├── mathematics (statistical utilities)
└── anomaly_detection (detection algorithms)

Application Layer:
├── machine_learning → core, anomaly_detection, data_platform
├── people_ops → core, infrastructure
├── data_platform → core, infrastructure
├── services → core, infrastructure
└── enterprise → core, infrastructure, services

Infrastructure Layer:
├── infrastructure → core
└── interfaces → ALL (presentation layer)

Shared:
├── testing (test utilities)
└── pkg (third-party packages)
```

### Dependency Rules
1. **Domain packages** can only depend on other domain packages or core
2. **Application packages** can depend on domain and infrastructure
3. **Infrastructure packages** can depend on domain packages
4. **Presentation layer** can depend on all other layers
5. **No circular dependencies** allowed between packages

## Benefits of New Structure

### 1. Clear Domain Boundaries
- Each package represents a single business domain
- Reduced coupling between unrelated functionality
- Easier to understand and maintain

### 2. Improved Scalability
- Teams can work independently on different domains
- Easier to add new features within domain boundaries
- Clear interfaces between domains

### 3. Better Testing
- Domain-specific test suites
- Easier to isolate and test individual components
- Clear dependency testing

### 4. Enhanced Security
- Centralized authentication and authorization
- Clear security boundaries
- Easier to audit and comply with regulations

### 5. Simplified Deployment
- Domain-based deployment strategies
- Independent scaling of different components
- Clearer monitoring and observability

## Migration Considerations

### Backward Compatibility
- Legacy imports will need updating
- Gradual migration strategy for existing code
- Clear deprecation timeline for old packages

### Performance
- Build optimization through clearer dependencies
- Better caching strategies with Buck2
- Reduced build times through incremental compilation

### Documentation
- Updated API documentation
- Migration guides for developers
- Clear package responsibility documentation

## Implementation Status

- ✅ **Domain Design**: Complete
- ✅ **Core Packages**: Created and configured
- ✅ **Build System**: Updated for new structure
- 🔄 **Governance Scripts**: In progress
- 🕒 **Testing & Validation**: Pending
- 🕒 **Documentation**: Pending

## Next Steps

1. **Complete governance script updates**
2. **Validate build system functionality**
3. **Update test configurations**
4. **Create migration documentation**
5. **Plan deprecation timeline for old packages**

## Success Metrics

- [ ] All packages follow consistent structure
- [ ] Build times improved by >20%
- [ ] Clear dependency graph with no cycles
- [ ] All tests passing with new structure
- [ ] Developer onboarding time reduced
- [ ] Code review efficiency improved

---

*This plan represents a significant step toward a more maintainable and scalable codebase organized around clear business domains.*