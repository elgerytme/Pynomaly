# CLI and Import Audit Report

## Summary

This report presents the results of a static code survey to identify:
1. Commented-out or `click.command` lines marked as TODO
2. Import errors raised by pytest and static analysis

**Date:** 2025-07-08T00:39:26Z
**Total Python files analyzed:** 482
**Successfully imported:** 257 (53.3%)
**Failed imports:** 225 (46.7%)

## 1. CLI and Click Command Analysis

### 1.1 TODO Comments Related to Click/CLI
- **Location:** `src/pynomaly/presentation/api/auth_deps.py:114`
- **Content:** `# TODO: Implement actual permission checking`
- **Context:** In SimplePermissionChecker class, the permission checking is simplified and needs proper implementation

### 1.2 Commented-out Click Commands
- **Status:** No commented-out `@click.command` or `@click.group` decorators found
- **Active CLI modules:** 11 CLI modules found with active click implementations:
  - `src/pynomaly/presentation/cli/dashboard.py`
  - `src/pynomaly/presentation/cli/training_automation_commands.py`
  - `src/pynomaly/presentation/cli/governance.py`
  - `src/pynomaly/presentation/cli/tenant.py`
  - `src/pynomaly/presentation/cli/ensemble.py`
  - `src/pynomaly/presentation/cli/quality.py`
  - `src/pynomaly/presentation/cli/security.py`
  - `src/pynomaly/presentation/cli/explain.py`
  - `src/pynomaly/presentation/cli/benchmarking.py`
  - `src/pynomaly/presentation/cli/cost_optimization.py`
  - `src/pynomaly/presentation/cli/enhanced_automl.py`
  - `src/pynomaly/presentation/cli/alert.py`

## 2. Import Errors Analysis

### 2.1 Critical Import Errors

#### Missing Dependencies
- **FastAPI ecosystem:** 50+ modules fail due to missing `fastapi`, `uvicorn`, `httpx`
- **Authentication:** `jwt`, `bcrypt`, `cryptography` missing (auth modules)
- **Database:** `redis`, `psycopg2`, `aiosqlite` missing
- **HTTP/Async:** `requests`, `aiohttp`, `aiofiles` missing
- **ML/Data:** `torch`, `tensorflow`, `jax`, `polars` missing
- **Monitoring:** `opentelemetry`, `memory_profiler` missing
- **Template:** `jinja2` missing
- **Validation:** `bleach` missing

#### Module Structure Issues
- **Missing base modules:** `pynomaly.domain.models.base` not found (affects federated learning)
- **Missing protocols:** `pynomaly.shared.protocols.repository` not found
- **Missing config:** `pynomaly.shared.config` not found

### 2.2 Import Resolution Errors

#### Undefined Classes/Functions
1. **API Endpoints Issue:**
   - **File:** `src/pynomaly/presentation/api/endpoints/datasets.py:219`
   - **Error:** `NameError: name 'get_container_simple' is not defined`
   - **Solution:** Missing import from `pynomaly.presentation.api.auth_deps`

2. **Missing DTO Classes:**
   - `OptimizationConfigDTO` not found in `configuration_dto.py`
   - `ConfigurationRecommendationDTO` not found in `configuration_dto.py`
   - `DatasetCharacteristicsDTO` not found in `configuration_dto.py`

3. **Missing Entity Classes:**
   - `DriftEvent` not found in `drift_detection.py`
   - `DriftMetrics` not found in `drift_detection.py`
   - `MLNoiseFeatures` not found in `alert.py`
   - `AnomalyPoint` not found in entities `__init__.py`

4. **Missing Exception Classes:**
   - `ComplianceError` not found in `shared.exceptions`
   - `TrainingError` not found in `domain.exceptions`
   - `StreamingError` not found in `domain.exceptions`
   - `OptimizationError` not found in `shared.exceptions`
   - `DataLoadError` not found in `domain.exceptions`
   - `InvalidConfigurationError` not found in `domain.exceptions`

5. **MRO (Method Resolution Order) Conflicts:**
   - **File:** `enhanced_detection_service.py`
   - **Error:** Cannot create consistent MRO for `DetectorProtocol` and `ExplainableDetectorProtocol`

6. **SQLAlchemy Table Issues:**
   - **File:** `optimized_repositories.py`
   - **Error:** Can't place `__table_args__` on inherited class with no table

### 2.3 Missing External Dependencies

#### Required Package Installations
```bash
# Core web framework
pip install fastapi uvicorn

# Authentication & Security
pip install python-jose[cryptography] bcrypt

# Database
pip install redis psycopg2-binary aiosqlite

# HTTP & Async
pip install requests aiohttp aiofiles httpx

# ML & Data Processing
pip install torch tensorflow jax jaxlib
pip install polars pyarrow

# Monitoring & Profiling
pip install opentelemetry-api memory-profiler

# Templates & Validation
pip install jinja2 bleach

# Optional ML Libraries
pip install shap lime
```

## 3. Recommendations

### 3.1 Immediate Actions
1. **Fix critical import:** Add missing import in `datasets.py`:
   ```python
   from pynomaly.presentation.api.auth_deps import get_container_simple
   from pynomaly.infrastructure.security.rbac_middleware import require_permissions, CommonPermissions
   ```

2. **Install missing dependencies:** See package list above

3. **Define missing classes:** Create missing DTO, Entity, and Exception classes

### 3.2 Architecture Improvements
1. **Resolve MRO conflicts:** Redesign protocol inheritance hierarchy
2. **Fix SQLAlchemy models:** Correct table inheritance structure
3. **Complete permission system:** Implement TODO in `auth_deps.py:114`

### 3.3 Dependencies Audit
1. **Review optional dependencies:** Some imports are for optional features (torch, tensorflow)
2. **Create dependency groups:** Separate core, web, ml, monitoring dependencies
3. **Update setup.py/pyproject.toml:** Ensure all required dependencies are specified

## 4. Risk Assessment

### High Risk
- **Authentication system incomplete:** Missing core auth dependencies and TODO in permission checking
- **API endpoints broken:** Import errors prevent API from starting
- **Database connections failing:** Missing database drivers

### Medium Risk
- **ML models unavailable:** Missing torch/tensorflow for deep learning features
- **Monitoring disabled:** Missing telemetry and profiling tools

### Low Risk
- **Optional features:** SHAP/LIME for explainability (gracefully handled)
- **Edge case protocols:** MRO conflicts in advanced detection services

## 5. Next Steps

1. **Resolve critical imports:** Fix the immediate import errors preventing application startup
2. **Install core dependencies:** FastAPI, database drivers, authentication libraries
3. **Define missing classes:** Create the missing DTO, Entity, and Exception classes
4. **Test import resolution:** Re-run import analysis after fixes
5. **Update dependency management:** Ensure pyproject.toml includes all required packages
