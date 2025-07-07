# CLAUDE.md

## Project Overview
Pynomaly: Python 3.11+ anomaly detection package integrating PyOD, PyGOD, scikit-learn, PyTorch, TensorFlow, JAX through clean architecture.

## Architecture
**Clean Architecture + DDD + Hexagonal Architecture**
- **Domain**: Pure business logic (`Anomaly`, `Detector`, `Dataset`, `Score`)
- **Application**: Use cases (`DetectAnomalies`, `TrainDetector`, `EvaluateModel`)
- **Infrastructure**: External integrations (adapters, data sources, persistence)
- **Presentation**: FastAPI, CLI, SDK, PWA (HTMX, Tailwind, D3.js, ECharts)

## Standards
- **Type Hints**: 100% with `mypy --strict`
- **Async/Await**: All I/O operations
- **Patterns**: Repository, Factory, Strategy, Observer
- **Testing**: >90% coverage, pytest, Hypothesis, performance tests
- **Production**: OpenTelemetry, Prometheus, K8s health checks, circuit breakers

## Key Features
- **Algorithm Integration**: Adapter pattern with `DetectorProtocol`, batch/streaming modes
- **Data**: CSV/Parquet/HDF5/SQL support, streaming with backpressure, DVC versioning
- **Advanced**: AutoML, SHAP/LIME explainability, drift detection, ensemble methods
- **PWA**: HTMX + Tailwind + D3.js + ECharts, offline-capable, installable

## Structure
**Reference**: See `PROJECT_STRUCTURE.md` for complete directory layout and organization rules.

```
src/pynomaly/
├── domain/ presentation/ infrastructure/ application/ shared/
tests/ docs/ examples/ benchmarks/ deploy/docker/ deploy/kubernetes/
```

## Environment
**MANDATORY**: All virtual environments in `environments/` with dot-prefix naming:
- ✅ `environments/.venv/` ❌ `.venv/`
- Python 3.11+, Poetry dependency management

## Commands
```bash
poetry install/add/run pytest/run mypy src/
poetry run uvicorn pynomaly.presentation.api:app --reload
npm install htmx.org d3 echarts tailwindcss
```

## Workflow
1. Domain-first, test-driven development
2. Production-grade, not prototype
3. Composition over inheritance, fail fast
4. Objective and critical assessment of code quality

## File Organization (ENFORCED)
**Reference**: See `PROJECT_STRUCTURE.md` for complete directory structure and AI assistant guidelines.
**Root**: Only essential config files (README.md, pyproject.toml, etc.)
**Move**: tests/ → tests/, scripts/ → scripts/, docs/ → docs/, reports/ → reports/
**Delete**: temp files, build artifacts, stray environments
**Validate**: `pre-commit install`, `python scripts/validate_file_organization.py`
**AI Note**: Project organization is difficult to maintain consistently with AI agents - always reference PROJECT_STRUCTURE.md

## Documentation (AUTO-UPDATE)
**Every commit**: Update README.md/TODO.md for accuracy
**README.md**: Only verifiable features, no marketing language
**TODO.md**: Sync with Claude Code todos, track completion dates
**Validation**: All examples must execute, all links must exist

## Changelog
**Update**: CHANGELOG.md for complete features, bugs, infrastructure changes
**Categories**: Added, Changed, Fixed, Security, Performance, Documentation, Infrastructure, Testing
**Sync**: TODO.md with Claude Code todos (⏳ pending, 🔄 in_progress, ✅ completed)

## TDD (ENABLED: 85% coverage)
**Commands**: `pynomaly tdd init/status/validate/report`
**Enforcement**: Domain/application layers mandatory, infrastructure selective
**Workflow**: Test requirement → failing test → minimal code → refactor
**Integration**: Pre-commit hooks, CI/CD validation, coverage reporting

## Critical Notes
- Keep `requirements.txt` synced with `pyproject.toml`
- Virtual environment activation required before operations
- Follow clean architecture separation strictly