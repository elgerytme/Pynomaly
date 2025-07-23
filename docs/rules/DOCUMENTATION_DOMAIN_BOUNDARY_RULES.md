# Documentation Domain Boundary Rules

This document establishes the documentation domain boundary rules for the repository to ensure clean separation of concerns in documentation and prevent documentation domain leakage.

## Overview

Documentation in this repository must follow strict domain boundaries to maintain architectural integrity and prevent coupling between different domains and packages. This ensures that documentation remains maintainable, reusable, and domain-appropriate.

## Core Principles

1. **Package Documentation Isolation**: Documentation within a package's `docs/` folder must not reference other packages
2. **Repository Documentation Genericity**: Main repository documentation must remain generic and not reference specific package implementations
3. **Self-Contained Examples**: Code examples in documentation must be self-contained and use appropriate import patterns
4. **Domain-Appropriate Content**: Documentation content must be appropriate for its domain context

## Documentation Domain Hierarchy

### 1. Repository-Level Documentation (`/docs/`)
**Purpose**: Contains generic architectural, process, and governance documentation

**Allowed Content**:
- Generic architectural patterns and principles
- Development processes and workflows
- General best practices and guidelines
- Abstract domain concepts and relationships
- Generic configuration examples
- Repository-level policies and rules

**Prohibited Content**:
- References to specific package implementations
- Package-specific configuration examples
- Domain-specific business logic documentation
- Specific package import statements or code examples
- Package-specific deployment instructions

### 2. Package-Specific Documentation (`/src/packages/*/docs/`)
**Purpose**: Contains documentation specific to individual packages

**Allowed Content**:
- Package-specific API documentation
- Package-specific getting started guides
- Package-specific configuration and deployment
- Internal package architecture
- Package-specific examples using relative imports
- Package-specific troubleshooting

**Prohibited Content**:
- References to other packages (except allowed dependencies)
- Cross-package import examples
- Cross-domain business logic references
- Implementation details of other packages

## Detailed Rules

### Rule 1: No Cross-Package References in Package Documentation

Package documentation must not reference other packages except for:
- Explicitly declared dependencies in domain configuration
- Generic infrastructure packages (when allowed by domain rules)
- Standard library and third-party packages

**❌ Violation Examples**:
```markdown
# In src/packages/ai/mlops/docs/getting-started.md
from anomaly_detection.core import DetectionService  # References other package
from monorepo.data_science import DataProcessor      # Cross-domain reference
```

**✅ Correct Examples**:
```markdown
# In src/packages/ai/mlops/docs/getting-started.md
from ..core.services import MLOpsService             # Relative import
from .models import ModelRegistry                    # Relative import
```

### Rule 2: No Package-Specific References in Repository Documentation

Repository-level documentation must remain generic and not reference specific package implementations.

**❌ Violation Examples**:
```markdown
# In docs/architecture/system-overview.md
The anomaly_detection package provides detection capabilities...
Configure the MLOps service using monorepo.mlops.config...
```

**✅ Correct Examples**:
```markdown
# In docs/architecture/system-overview.md
Detection services provide anomaly identification capabilities...
Configure ML operations using the appropriate domain service...
```

### Rule 3: Import Statement Patterns

Code examples in documentation must follow appropriate import patterns based on context.

#### Package Documentation Import Patterns
- **Use relative imports** for same-package references
- **Use absolute imports** only for external dependencies
- **Avoid monorepo-style imports** (`from monorepo.package.*`)

#### Repository Documentation Import Patterns
- **Use generic placeholder imports** when necessary
- **Avoid specific package import statements**
- **Focus on patterns rather than specific implementations**

### Rule 4: Configuration Examples

Configuration examples must be appropriate for their documentation context.

#### Package Documentation
- May include package-specific configuration
- Must use relative paths and package-appropriate values
- Should not reference other packages' configuration

#### Repository Documentation
- Must use generic, template-style configuration
- Should use placeholder values and patterns
- Must not include package-specific configuration details

## Validation Rules

### Automated Detection Patterns

The documentation boundary validator will check for these patterns:

1. **Cross-Package Import Detection**:
   ```regex
   from\s+(?!\.{1,2}|[\w_]+\.[\w_]+$)[a-zA-Z_][\w_]*\..*import
   ```

2. **Monorepo Import Pattern Detection**:
   ```regex
   from\s+monorepo\.[a-zA-Z_][\w_]*
   ```

3. **Package Name References**:
   ```regex
   (anomaly_detection|mlops|data_science|enterprise_\w+)(?!\w)
   ```

4. **Cross-Domain Business Logic References**:
   ```regex
   (detection|anomaly|mlops|training|deployment)(?=\s+service|Service|config)
   ```

### File Scope Rules

- **`/docs/**/*.md`**: Must not contain package-specific references
- **`/src/packages/*/docs/**/*.md`**: Must not contain cross-package references
- **`/src/packages/*/README.md`**: Must not contain cross-package references
- **Code blocks in markdown**: Must follow import pattern rules

## Configuration

### Domain Boundaries Configuration

Documentation rules are configured in `.domain-boundaries.yaml`:

```yaml
documentation:
  rules:
    - name: no_cross_package_references_in_package_docs
      severity: critical
      scope: "src/packages/*/docs/**/*.md"
      patterns:
        - pattern: "from\\s+(?!\\.{1,2})([a-zA-Z_][\\w_]*)\\..*import"
          message: "Package documentation must not reference other packages"
        
    - name: no_package_specific_refs_in_repo_docs
      severity: critical
      scope: "docs/**/*.md"
      patterns:
        - pattern: "(anomaly_detection|mlops|data_science|enterprise_\\w+)(?!\\w)"
          message: "Repository documentation must not reference specific packages"
          
    - name: no_monorepo_imports
      severity: warning
      scope: "**/*.md"
      patterns:
        - pattern: "from\\s+monorepo\\.[a-zA-Z_][\\w_]*"
          message: "Use relative imports instead of monorepo-style imports"

  exceptions:
    - file: "docs/architecture/DOMAIN_SEPARATION_COMPLETION_SUMMARY.md"
      reason: "Historical migration documentation"
      expires: "2025-12-31"
```

## Enforcement Mechanisms

### 1. Pre-commit Hooks
- Documentation boundary validation runs on every commit
- Blocks commits that introduce documentation domain leakage
- Provides immediate feedback with specific violation details

### 2. CI/CD Pipeline
- Documentation boundary checks run on all pull requests
- Generates detailed violation reports
- Blocks merges that introduce violations

### 3. Automated Scanning
- Regular scans of all documentation files
- Generates compliance reports
- Tracks violations over time

## Remediation Guidelines

### Fixing Cross-Package References

1. **In Package Documentation**:
   - Replace cross-package imports with relative imports
   - Remove references to other package implementations
   - Use generic examples appropriate to the package domain

2. **In Repository Documentation**:
   - Replace specific package references with generic concepts
   - Move package-specific content to appropriate package documentation
   - Use abstract examples that don't reference specific implementations

### Migration Process

1. **Identify Violations**: Run documentation boundary scanner
2. **Categorize Violations**: Separate critical from warning violations
3. **Fix Critical Violations**: Address violations that break domain boundaries
4. **Refactor Content**: Move misplaced content to appropriate locations
5. **Update Examples**: Replace problematic examples with domain-appropriate ones
6. **Validate Changes**: Run boundary scanner to confirm fixes

## Best Practices

### Writing Package Documentation
1. Focus on the package's specific domain and capabilities
2. Use relative imports in all code examples
3. Avoid referencing external package implementations
4. Keep examples self-contained and runnable within the package context

### Writing Repository Documentation
1. Focus on generic patterns and architectural principles
2. Use abstract examples that illustrate concepts without specific implementations
3. Avoid mentioning specific package names or implementations
4. Focus on interfaces and contracts rather than implementations

### Code Examples
1. Prefer conceptual examples over specific implementation examples
2. Use placeholder names that indicate intent rather than specific packages
3. Include imports only when necessary for understanding
4. Validate that examples follow import pattern rules

## Tooling Support

### Documentation Boundary Scanner
The `domain_boundary_detector` tool includes documentation scanning capabilities:

```bash
# Scan documentation for boundary violations
python -m domain_boundary_detector.cli scan --include-docs

# Generate documentation compliance report
python -m domain_boundary_detector.cli scan --docs-only --format markdown

# Fix common documentation violations
python -m domain_boundary_detector.cli fix --docs-only --dry-run
```

### Integration with Development Workflow
- IDE plugins provide real-time feedback on documentation violations
- Git hooks prevent commits with documentation boundary violations
- CI/CD pipelines block merges that introduce violations

## Violation Examples and Fixes

### Example 1: Cross-Package Import in Package Documentation

**❌ Violation**:
```markdown
# In src/packages/ai/mlops/docs/api.md
```python
from anomaly_detection.core import DetectionService
service = DetectionService()
```

**✅ Fix**:
```markdown
# In src/packages/ai/mlops/docs/api.md
```python
from ..core.services import MLOpsService
service = MLOpsService()
```

### Example 2: Package-Specific Reference in Repository Documentation

**❌ Violation**:
```markdown
# In docs/architecture/system-overview.md
The anomaly_detection package handles all detection operations.
```

**✅ Fix**:
```markdown
# In docs/architecture/system-overview.md
Detection services handle anomaly identification operations.
```

### Example 3: Monorepo Import Pattern

**❌ Violation**:
```markdown
# In src/packages/ai/mlops/docs/getting-started.md
```python
from monorepo.mlops.services import TrainingService
```

**✅ Fix**:
```markdown
# In src/packages/ai/mlops/docs/getting-started.md
```python
from .services import TrainingService
```

## Contact and Support

For questions about documentation domain boundaries:
- Architecture Team: architecture@domain.io
- Documentation Team: docs@domain.io
- Domain Boundary Questions: boundaries@domain.io

## Related Documents

- [Domain Boundary Rules](DOMAIN_BOUNDARY_RULES.md)
- [Package Isolation Rules](PACKAGE_ISOLATION_RULE.md)
- [Development Guidelines](DEVELOPMENT_GUIDELINES.md)