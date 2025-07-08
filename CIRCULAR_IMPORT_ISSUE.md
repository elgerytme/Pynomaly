# Circular Import Issues in Pynomaly Web-UI

## Issue Summary

During the setup of a reproducible development environment for Pynomaly, we have identified multiple circular import issues that occur when starting the Web-UI and container services. These circular imports can cause intermittent startup failures and make the codebase harder to maintain.

## Reproduction Environment

### Setup Method 1: Local Development Scripts
```bash
# Using development scripts
python scripts/development/run_web_app.py --port 8000
python scripts/development/run_web_ui.py --port 8080
```

### Setup Method 2: Docker Compose (Recommended for Consistency)
```bash
# Using Docker Compose for reproducible environment
docker compose -f docker-compose.dev.yml up --build
```

## Detected Circular Import Patterns

### 1. Domain Entities Circular Import
**Chain:** `pynomaly.domain.entities -> pynomaly.domain.entities`

**Location:** Domain layer self-reference
**Impact:** Critical - affects core business logic imports

### 2. Infrastructure Config Circular Imports (Multiple)
**Chain:** `pynomaly.infrastructure.config -> pynomaly.infrastructure.config`

**Frequency:** 25+ occurrences detected
**Impact:** High - affects application configuration and dependency injection

### 3. Value Objects and Exceptions Circular Dependencies
Based on the detailed trace from `debug_imports.py`:

```
pynomaly.domain.value_objects -> pynomaly.domain.exceptions -> 
pynomaly.domain.value_objects.confidence_interval -> pynomaly.domain.exceptions
```

## Current Stack Traces

### Stack Trace 1: Domain Entities Circular Import
```
⚠️  POTENTIAL CIRCULAR IMPORT: pynomaly.domain.entities
Import chain: pynomaly.presentation.web.app -> pynomaly.domain.entities -> pynomaly.domain.entities
  File "pynomaly/__init__.py", line 23, in <module>
    from pynomaly.domain.entities import Anomaly, Dataset, DetectionResult, Detector
  File "pynomaly/domain/__init__.py", line 3, in <module>
    from pynomaly.domain.entities import Anomaly, Dataset, DetectionResult, Detector
```

### Stack Trace 2: Value Objects Circular Import
```
⚠️  POTENTIAL CIRCULAR IMPORT: pynomaly.domain.value_objects
Import chain: pynomaly.domain.entities -> pynomaly.domain.value_objects -> 
              pynomaly.domain.exceptions -> pynomaly.domain.value_objects.confidence_interval -> 
              pynomaly.domain.exceptions -> pynomaly.domain.value_objects
```

### Stack Trace 3: Infrastructure Config Circular Import
```
⚠️  POTENTIAL CIRCULAR IMPORT: pynomaly.infrastructure.config
Import chain: pynomaly.infrastructure.config -> pynomaly.domain.entities -> 
              pynomaly.infrastructure.config
```

## Error Messages Encountered

### Import Error: Missing DTO
```
❌ IMPORT ERROR: pynomaly.presentation.cli - cannot import name 'DatasetCharacteristicsDTO' 
from 'pynomaly.application.dto.configuration_dto'
```

### Application Startup Issues
```
Web UI mounted successfully
✅ Web app created successfully
```

Note: Despite circular imports, the application can still start, but this indicates potential runtime issues.

## Impact Assessment

### Severity: HIGH
- **Startup Reliability:** Circular imports can cause non-deterministic startup failures
- **Development Experience:** Makes debugging difficult and IDE navigation unreliable  
- **Maintenance:** Increases complexity when adding new features or refactoring
- **Testing:** May cause test isolation issues and flaky tests

### Current Status
- Web application **CAN** start despite circular imports
- Multiple circular import patterns detected (25+ occurrences)
- Some import errors in CLI components

## Recommended Solutions

### 1. Immediate Fixes (Domain Layer)
- Move shared types to `pynomaly.domain.types` module
- Use forward references in type hints where possible
- Extract common exceptions to `pynomaly.domain.exceptions.base`

### 2. Infrastructure Layer Refactoring
- Break `pynomaly.infrastructure.config` into smaller modules
- Use dependency injection patterns instead of direct imports
- Implement lazy loading for heavy configuration objects

### 3. Value Objects Restructuring
- Move exceptions used by value objects to separate module
- Consider using Protocol classes for shared interfaces
- Implement factory patterns for complex value object creation

## Development Environment Scripts

### Debug Script Usage
```bash
# Run comprehensive circular import analysis
python debug_circular_imports.py

# Run import tracing for specific components
python debug_imports.py
```

### Reproducible Environment Setup
```bash
# Create debug reports directory
mkdir debug_reports

# Run web application with tracing
python scripts/development/run_web_app.py --port 8000

# Alternative: Run with Docker for consistency
docker compose -f docker-compose.dev.yml up --build
```

## Files for Investigation

### Key Problem Files
1. `src/pynomaly/__init__.py` - Root package imports
2. `src/pynomaly/domain/__init__.py` - Domain layer imports
3. `src/pynomaly/domain/entities/__init__.py` - Entity imports
4. `src/pynomaly/domain/value_objects/__init__.py` - Value object imports
5. `src/pynomaly/infrastructure/config/__init__.py` - Config imports

### Analysis Tools
1. `debug_circular_imports.py` - Comprehensive analysis tool
2. `debug_imports.py` - Simple import tracer
3. `docker-compose.dev.yml` - Reproducible environment setup
4. `Dockerfile.debug` - Debug container configuration

## Next Steps

1. **Immediate:** Fix the most critical circular imports in domain layer
2. **Short-term:** Refactor infrastructure configuration modules
3. **Long-term:** Implement architectural patterns to prevent future circular imports

## Testing the Fix

After implementing fixes, run:
```bash
# Test import analysis
python debug_circular_imports.py

# Test web application startup
python scripts/development/run_web_app.py --port 8000

# Verify no circular imports detected
echo $? # Should be 0 if no circular imports
```

---

**Date:** 2025-07-08  
**Environment:** Windows PowerShell, Python 3.11+  
**Status:** Issue Documented & Reproducible Environment Established
