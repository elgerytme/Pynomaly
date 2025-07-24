# 🔄 Domain Migration Scripts

This directory contains automated scripts for migrating the anomaly detection package to proper domain boundaries following Domain-Driven Design principles.

## 📋 Overview

The migration process restructures the monolithic anomaly detection package into focused domain packages:

- **Core Anomaly Detection** → `core/anomaly_detection/`
- **Machine Learning Services** → `ai/machine_learning/`
- **MLOps Capabilities** → `ai/mlops/`
- **Data Engineering** → `data/data_engineering/`
- **Data Quality** → `data/data_quality/`
- **Infrastructure** → `shared/infrastructure/`
- **Observability** → `shared/observability/`

## 🛠️ Available Scripts

### 1. Phase 1 Infrastructure Migration (`phase1_infrastructure_migration.py`)

Handles the migration of foundational infrastructure components and core domain entities.

**Features:**
- Automated file migration with dependency tracking
- Import statement updates
- Backup creation and rollback capabilities
- Comprehensive logging and reporting

**Usage:**
```bash
# Dry run to preview changes
python scripts/migration/phase1_infrastructure_migration.py --dry-run

# Execute live migration
python scripts/migration/phase1_infrastructure_migration.py --execute

# Rollback if needed
python scripts/migration/phase1_infrastructure_migration.py --rollback
```

**Migrates:**
- Configuration management → `shared/infrastructure/config/`
- Logging infrastructure → `shared/infrastructure/logging/`
- Middleware components → `shared/infrastructure/middleware/`
- Core domain entities → `core/anomaly_detection/domain/entities/`
- Utilities and base classes → `shared/infrastructure/utils/`

### 2. Migration Validator (`migration_validator.py`)

Comprehensive validation tool for ensuring migration success and system integrity.

**Features:**
- Pre-migration readiness checks
- Post-migration verification
- Performance impact assessment
- Dependency analysis and circular dependency detection
- Integration testing automation

**Usage:**
```bash
# Pre-migration validation
python scripts/migration/migration_validator.py --pre-migration --phase=1

# Post-migration verification
python scripts/migration/migration_validator.py --post-migration --phase=1

# Comprehensive validation
python scripts/migration/migration_validator.py --full-validation --phase=1

# Performance testing only
python scripts/migration/migration_validator.py --performance-test
```

**Validation Checks:**
- Source code quality (flake8, linting)
- Test coverage analysis
- Dependency graph validation
- Circular dependency detection
- Backup system readiness
- Target directory structure
- Infrastructure readiness
- Performance baseline capture
- API endpoint functionality
- Import resolution verification

## 📊 Migration Phases

### Phase 1: Foundation Infrastructure (Week 1)
**Complexity:** LOW | **Risk:** LOW | **Files:** 24

- Infrastructure components (config, logging, middleware)
- Core domain entities and value objects
- Base utilities and common code

### Phase 2: Machine Learning Components (Week 2-3)
**Complexity:** MEDIUM | **Risk:** MEDIUM | **Files:** 23

- Detection services and algorithms
- Ensemble methods and explainability
- Algorithm adapters (PyOD, sklearn, deep learning)

### Phase 3: Data Engineering Components (Week 3-4)
**Complexity:** MEDIUM | **Risk:** MEDIUM | **Files:** 15

- Batch and stream processing services
- Data transformation and conversion
- Data quality and validation services

### Phase 4: Observability & Monitoring (Week 4-5)
**Complexity:** HIGH | **Risk:** HIGH | **Files:** 16

- System monitoring and health checks
- Dashboard and analytics components
- Performance monitoring and alerting

### Phase 5: MLOps & Advanced Services (Week 5-6)
**Complexity:** HIGH | **Risk:** HIGH | **Files:** 12

- Model lifecycle management
- Experimentation and A/B testing
- Concept drift detection

### Phase 6: API & Presentation Layer (Week 6-7)
**Complexity:** HIGH | **Risk:** HIGH | **Files:** 25

- Service composition and API orchestration
- Web interfaces and CLI commands
- Endpoint routing and response aggregation

### Phase 7: Application Layer (Week 7-8)
**Complexity:** MEDIUM | **Risk:** MEDIUM | **Files:** 25

- Use case distribution to appropriate domains
- Application facades and orchestration
- Final integration and optimization

## 🚀 Quick Start

### Prerequisites

1. **Python Environment**: Python 3.11+ with required packages
```bash
pip install pytest flake8 coverage psutil requests
```

2. **Backup Space**: Ensure at least 1GB free disk space for backups

3. **Repository Access**: Write permissions to target directories

### Running Your First Migration

1. **Validate Pre-Migration State**
```bash
python scripts/migration/migration_validator.py --pre-migration --phase=1
```

2. **Run Migration Dry Run**
```bash
python scripts/migration/phase1_infrastructure_migration.py --dry-run
```

3. **Execute Migration**
```bash
python scripts/migration/phase1_infrastructure_migration.py --execute
```

4. **Validate Post-Migration**
```bash
python scripts/migration/migration_validator.py --post-migration --phase=1
```

5. **Review Results**
```bash
# Check migration log
cat migration_phase1.log

# Review validation report
cat migration_validation_report_*.json
```

## 📁 File Structure After Migration

```
src/packages/
├── core/
│   └── anomaly_detection/           # Core anomaly detection domain
│       ├── domain/
│       │   ├── entities/            # Anomaly, DetectionResult
│       │   ├── value_objects/       # Threshold, Score
│       │   └── exceptions.py        # Domain exceptions
│       ├── application/
│       │   └── use_cases/           # detect_anomalies, compare_algorithms
│       └── presentation/
│           ├── cli/                 # Anomaly detection CLI commands
│           └── api/                 # Anomaly detection API endpoints
│
├── ai/
│   ├── machine_learning/            # General ML algorithms and services
│   │   ├── domain/services/         # Detection, ensemble, explainability
│   │   └── infrastructure/adapters/ # Algorithm adapters
│   └── mlops/                       # Model lifecycle management
│       ├── domain/services/         # Model management, experimentation
│       └── infrastructure/          # Model repository, deployment
│
├── data/
│   ├── data_engineering/            # Data processing and pipelines
│   │   ├── domain/services/         # Batch processing, streaming
│   │   └── application/use_cases/   # Process streaming, data transformation
│   └── data_quality/                # Data validation and profiling
│       ├── domain/services/         # Validation, profiling, sampling
│       └── infrastructure/          # Quality check implementations
│
├── shared/
│   ├── infrastructure/              # Common infrastructure components
│   │   ├── config/                  # Configuration management
│   │   ├── logging/                 # Structured logging
│   │   ├── middleware/              # Rate limiting, CORS, error handling
│   │   └── utils/                   # Shared utilities
│   └── observability/               # System monitoring and observability
│       ├── domain/services/         # Health monitoring, analytics
│       └── infrastructure/          # Monitoring infrastructure
│
└── configurations/
    └── anomaly_detection_config/    # Service composition
        ├── api/                     # API composition and routing
        ├── services.py              # Service wiring and facades
        └── server.py                # Composed application server
```

## 🔧 Advanced Usage

### Custom Migration Configuration

Create a custom migration configuration file:

```json
{
  "phase": 1,
  "custom_mappings": {
    "src/anomaly_detection/custom_service.py": "src/packages/shared/infrastructure/custom/"
  },
  "skip_files": [
    "src/anomaly_detection/temp_file.py"
  ],
  "additional_import_mappings": {
    "old_import_path": "new_import_path"
  }
}
```

Use with migration script:
```bash
python scripts/migration/phase1_infrastructure_migration.py --execute --config=custom_config.json
```

### Performance Monitoring During Migration

Monitor system performance during migration:

```bash
# Start performance monitoring
python scripts/migration/migration_validator.py --performance-test &

# Run migration
python scripts/migration/phase1_infrastructure_migration.py --execute

# Compare performance
python scripts/migration/migration_validator.py --post-migration --phase=1
```

### Batch Migration Validation

Validate multiple phases at once:

```bash
# Validate all completed phases
for phase in 1 2 3; do
    python scripts/migration/migration_validator.py --post-migration --phase=$phase
done
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Import Resolution Failures
**Problem**: Imports fail after migration
**Solution**: Check import mappings in migration script and update manually if needed

```python
# Update imports in affected files
from shared.infrastructure.config import settings  # New
# from anomaly_detection.infrastructure.config import settings  # Old
```

#### 2. Permission Errors
**Problem**: Cannot create target directories
**Solution**: Ensure write permissions and create directories manually

```bash
# Create target directories
mkdir -p src/packages/shared/infrastructure
chmod 755 src/packages/shared/infrastructure
```

#### 3. Circular Dependencies
**Problem**: Circular import dependencies detected
**Solution**: Refactor code to remove circular references

```python
# Instead of direct import, use dependency injection
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from other_module import OtherClass
```

#### 4. Test Failures After Migration
**Problem**: Unit tests fail after migration
**Solution**: Update test imports and fixture paths

```python
# Update test imports
from core.anomaly_detection.domain.entities import Anomaly  # New
# from anomaly_detection.domain.entities import Anomaly  # Old
```

### Recovery Procedures

#### Rollback Migration
```bash
python scripts/migration/phase1_infrastructure_migration.py --rollback
```

#### Manual File Recovery
```bash
# Restore from backup
cp -r migration_backups/phase1/anomaly_detection_pre_phase1/* src/packages/data/anomaly_detection/
```

#### Reset Migration State
```bash
# Clean up partial migration
rm -rf src/packages/shared/infrastructure/
rm -rf src/packages/core/anomaly_detection/
git checkout HEAD -- src/packages/data/anomaly_detection/
```

## 📊 Monitoring and Reporting

### Migration Reports

Each migration generates detailed reports:

- **Migration Log**: `migration_phase1.log`
- **Validation Report**: `migration_validation_report_*.json`
- **Performance Report**: `performance_baseline.json`

### Success Metrics

- **Migration Success Rate**: ≥95% of files migrated successfully
- **Test Pass Rate**: ≥90% of tests passing after migration
- **Performance Impact**: ≤10% degradation in response times
- **Import Resolution**: 100% of imports resolving correctly

### Reporting Dashboard

Generate migration dashboard:

```bash
python scripts/migration/generate_migration_dashboard.py --phase=1
```

View at: `file://migration_reports/dashboard.html`

## 🔗 Integration with CI/CD

### GitHub Actions Integration

Add to `.github/workflows/migration.yml`:

```yaml
name: Domain Migration
on:
  workflow_dispatch:
    inputs:
      phase:
        description: 'Migration phase to execute'
        required: true
        default: '1'

jobs:
  migrate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: pip install -r requirements-dev.txt
        
      - name: Validate pre-migration
        run: python scripts/migration/migration_validator.py --pre-migration --phase=${{ github.event.inputs.phase }}
        
      - name: Execute migration
        run: python scripts/migration/phase${{ github.event.inputs.phase }}_migration.py --execute
        
      - name: Validate post-migration
        run: python scripts/migration/migration_validator.py --post-migration --phase=${{ github.event.inputs.phase }}
        
      - name: Upload migration reports
        uses: actions/upload-artifact@v3
        with:
          name: migration-reports
          path: |
            migration_phase*.log
            migration_validation_report_*.json
```

## 📚 Additional Resources

- **[Complete Migration Plan](../DOMAIN_MIGRATION_PLAN.md)**: Comprehensive migration strategy
- **[Domain Architecture Guide](../../docs/architecture/domain-driven-design.md)**: DDD principles
- **[Testing Strategy](../../docs/testing/migration-testing.md)**: Testing approaches
- **[Performance Monitoring](../../docs/monitoring/migration-monitoring.md)**: Monitoring setup

## 🆘 Support

- **Migration Team**: [Slack #domain-migration](https://workspace.slack.com/channels/domain-migration)
- **Technical Support**: [support-migration@company.com](mailto:support-migration@company.com)
- **Emergency**: [Migration On-Call](https://pagerduty.com/migration-oncall)

---

**🎯 Ready to Begin Migration?**

Start with Phase 1 validation:
```bash
python scripts/migration/migration_validator.py --pre-migration --phase=1
```

Follow the step-by-step process in the [Complete Migration Plan](../DOMAIN_MIGRATION_PLAN.md) for detailed guidance.