# Project Tidy Rules

This document outlines the enforceable tidiness rules for the Pynomaly project to maintain a clean, organized, and well-structured repository.

## Table of Contents

1. [Directory Layout Contract](#directory-layout-contract)
2. [File Naming Conventions](#file-naming-conventions)
3. [Archiving Guidelines](#archiving-guidelines)
4. [Commit & PR Standards](#commit--pr-standards)
5. [Enforcement](#enforcement)
6. [Violation Remediation](#violation-remediation)

## Directory Layout Contract

### Core Structure

The project follows a strict directory structure that must be maintained:

```
pynomaly/
├── src/pynomaly/          # Source code only
├── tests/                 # Test files only
├── docs/                  # Documentation
├── examples/              # Example code and tutorials
├── scripts/               # Utility scripts
├── config/                # Configuration files
├── environments/          # Environment configurations
├── deploy/                # Deployment configurations
├── tools/                 # Development tools
└── README.md             # Project root documentation
```

### Forbidden Directories

The following directories are **strictly forbidden** at the root level:

- `build/` - Use `artifacts/` instead
- `dist/` - Use `artifacts/` instead
- `output/` - Use `artifacts/` instead
- `tmp/` - Use `artifacts/tmp/` instead
- `temp/` - Use `artifacts/tmp/` instead
- `cache/` - Use `artifacts/cache/` instead
- `backup/` - Use `docs/archive/` instead
- `old/` - Use `docs/archive/` instead
- `legacy/` - Use `docs/archive/legacy-*/` instead
- `vendor/` - Use `tools/vendor/` instead
- `lib/` - Use `tools/lib/` instead
- `bin/` - Use `tools/bin/` instead
- `data/` - Use `examples/data/` instead
- `samples/` - Use `examples/` instead
- `misc/` - Reorganize into appropriate directories

### Allowed Artifact Directories

- `artifacts/` - Build outputs and generated files
- `reports/` - Test reports and analysis results
- `storage/` - Temporary storage during development
- `baseline_outputs/` - Baseline files for testing
- `test_reports/` - Test result files

### Source Code Organization

```
src/pynomaly/
├── domain/               # Business logic
├── application/          # Application services
├── infrastructure/       # External integrations
├── presentation/         # UI/API layers
└── shared/              # Shared utilities
```

## File Naming Conventions

### Python Files

- **Modules**: `snake_case.py`
- **Classes**: `PascalCase` in `snake_case.py` files
- **Functions**: `snake_case()`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: `_leading_underscore`

### Test Files

- **Unit tests**: `test_*.py`
- **Integration tests**: `test_*_integration.py`
- **Performance tests**: `test_*_performance.py`
- **UI tests**: `test_*_ui.py`

### Documentation Files

- **Markdown**: `kebab-case.md`
- **API docs**: `API_*.md` (uppercase for API documentation)
- **README files**: `README.md` (exactly this name)
- **Architecture docs**: `ADR-###-description.md`

### Configuration Files

- **Environment**: `.env.example`, `.env.local`
- **Config**: `config.yaml`, `settings.toml`
- **Docker**: `Dockerfile`, `docker-compose.yml`
- **CI/CD**: `.github/workflows/*.yml`

### Script Files

- **Python scripts**: `snake_case.py`
- **Shell scripts**: `kebab-case.sh`
- **Batch files**: `kebab-case.bat`

## Archiving Guidelines

### When to Archive

Files should be archived (moved to `docs/archive/`) when:

1. **Obsolete**: No longer needed for current development
2. **Replaced**: Superseded by newer implementations
3. **Historical**: Needed for reference but not active development
4. **Experimental**: Failed experiments or proof-of-concepts

### Archive Structure

```
docs/archive/
├── historical-project-docs/    # Old project documentation
├── legacy-algorithm-docs/      # Deprecated algorithm documentation
├── experimental/               # Failed experiments
├── deprecated-features/        # Removed features
└── old-configs/               # Obsolete configuration files
```

### Archive Process

1. **Create archive directory** if it doesn't exist
2. **Move files** to appropriate archive subdirectory
3. **Update references** in active documentation
4. **Document reason** for archiving in commit message
5. **Update .gitignore** if needed

### Archive Naming

- Use descriptive names: `legacy-v1-algorithms/`
- Include dates when relevant: `2024-01-experiment-results/`
- Group related items: `historical-project-docs/`

## Commit & PR Standards

### Commit Message Format

```
type(scope): description

Optional body explaining the change in more detail.

Closes #issue-number
```

#### Types

- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style/formatting
- `refactor`: Code refactoring
- `test`: Test additions/modifications
- `chore`: Maintenance tasks
- `ci`: CI/CD changes
- `perf`: Performance improvements
- `security`: Security fixes

#### Scope Examples

- `core`: Core functionality
- `api`: API changes
- `cli`: Command-line interface
- `ui`: User interface
- `docs`: Documentation
- `tests`: Test-related changes
- `build`: Build system changes
- `deploy`: Deployment changes

### PR Requirements

#### PR Title Format

```
[type]: Brief description of changes
```

#### PR Description Template

```markdown
## Changes Made

- [ ] Feature/fix description
- [ ] Related changes
- [ ] Documentation updates

## Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No merge conflicts
- [ ] Tidy rules followed
```

#### Required Checks

1. **Automated Tests**: All CI checks must pass
2. **Code Review**: At least one approval required
3. **Tidy Check**: Structure validation must pass
4. **Documentation**: Updates must be included
5. **Atomic Changes**: Each PR should be a logical unit

## Enforcement

### Pre-commit Hooks

The following hooks are enforced on every commit:

1. **Ruff Linting**: `ruff check --fix`
2. **Import Sorting**: `ruff check --select I --fix`
3. **Secret Scanning**: `detect-secrets`
4. **Structure Validation**: `check-tidy-structure.py`
5. **File Organization**: `validate_file_organization.py`

### CI/CD Enforcement

The `project-tidy-check` job runs on every PR and enforces:

- Directory structure compliance
- File naming conventions
- Forbidden directory detection
- Archive organization
- Documentation completeness

### Automated Fixes

Some violations are automatically fixed:

- **Import sorting**: Automatically reordered
- **Code formatting**: Automatically formatted
- **Line endings**: Normalized to LF
- **Trailing whitespace**: Removed

## Violation Remediation

### Common Violations

#### 1. Forbidden Directories

**Violation**: Creating `build/`, `dist/`, `tmp/` directories
**Fix**: Move contents to `artifacts/` subdirectories

```bash
# Wrong
mkdir build/
mkdir dist/

# Right
mkdir -p artifacts/build/
mkdir -p artifacts/dist/
```

#### 2. Incorrect File Naming

**Violation**: Using `camelCase.py` or `UPPERCASE.py`
**Fix**: Rename to `snake_case.py`

```bash
# Wrong
myModule.py
MyModule.py

# Right
my_module.py
```

#### 3. Misplaced Files

**Violation**: Putting tests in `src/`
**Fix**: Move to appropriate test directory

```bash
# Wrong
src/pynomaly/test_something.py

# Right
tests/unit/test_something.py
```

#### 4. Unorganized Archives

**Violation**: Mixing old and new files
**Fix**: Move old files to `docs/archive/`

```bash
# Wrong
docs/old_algorithm_docs.md

# Right
docs/archive/legacy-algorithm-docs/old_algorithm_docs.md
```

### Remediation Steps

1. **Identify violations** using the tidy check script
2. **Plan remediation** based on violation type
3. **Execute fixes** systematically
4. **Test changes** to ensure nothing breaks
5. **Commit fixes** with appropriate messages
6. **Verify compliance** with re-running checks

### Emergency Procedures

If tidy checks block critical fixes:

1. **Temporary bypass**: Add `--no-verify` to git commit
2. **Create follow-up issue** for remediation
3. **Fix violations** in next commit
4. **Document reason** in commit message

## Quality Metrics

### Compliance Targets

- **Directory Structure**: 100% compliance
- **File Naming**: 100% compliance
- **Archive Organization**: 95% compliance
- **Documentation**: 90% coverage

### Monitoring

- **Pre-commit success rate**: Target 95%
- **CI tidy check pass rate**: Target 100%
- **Manual violations**: Target 0 per month
- **Archive organization**: Quarterly reviews

## Tools and Scripts

### Available Tools

- `scripts/validation/check-tidy-structure.py` - Structure validation
- `scripts/validation/validate_file_organization.py` - File organization check
- `scripts/cleanup/archive-old-files.py` - Automated archiving
- `scripts/cleanup/normalize-naming.py` - File naming fixes

### Usage Examples

```bash
# Check structure compliance
python scripts/validation/check-tidy-structure.py

# Validate file organization
python scripts/validation/validate_file_organization.py

# Archive old files
python scripts/cleanup/archive-old-files.py --dry-run

# Fix naming violations
python scripts/cleanup/normalize-naming.py --fix
```

## Exceptions

### Approved Exceptions

- `.github/` - GitHub-specific directory
- `node_modules/` - Node.js dependencies (if using)
- `.venv/` - Python virtual environments
- `htmlcov/` - Coverage reports

### Requesting Exceptions

1. **Create issue** explaining the need
2. **Justify exception** with technical reasoning
3. **Propose alternative** if possible
4. **Get approval** from maintainers
5. **Document exception** in this file

## References

- [Clean Architecture Guidelines](architecture/overview.md)
- [Contributing Guidelines](contributing/CONTRIBUTING.md)
- [File Organization Standards](contributing/FILE_ORGANIZATION_STANDARDS.md)
- [Development Setup Guide](DEVELOPMENT_SETUP.md)

---

*This document is enforced by automated tooling and should be updated when rules change.*
