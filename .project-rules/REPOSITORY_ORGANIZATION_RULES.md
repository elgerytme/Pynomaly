# Repository Organization Rules
# ============================

## Core Principles

### 1. Clean Root Directory Policy
**ENFORCED**: Only these directories are allowed in the repository root:
- `.claude`, `.github`, `.hypothesis`, `.project-rules`, `.ruff_cache`, `.storybook`, `.vscode`
- `docs`, `pkg`, `scripts`, `src`

**PROHIBITED** in root:
- Any `test_*` directories or files
- Any `temp*`, `tmp*`, `scratch*` directories
- Any environment directories (`env*`, `venv*`, `.env*`)
- Any build artifacts (`build/`, `dist/`, `*.egg-info/`)
- Any configuration files (must go in `scripts/config/`)
- Any documentation files except `README.md`, `CHANGELOG.md`, `LICENSE`

### 2. File Organization Rules

#### Configuration Files → `scripts/config/`
- Build configs: `scripts/config/buck/`
- Environment files: `scripts/config/env/`
- Linting configs: `scripts/config/linting/`
- Git configs: `scripts/config/git/`
- Security configs: `scripts/config/security/`
- Testing configs: `scripts/config/testing/`

#### Documentation → `docs/`
- Architecture docs: `docs/architecture/`
- Migration docs: `docs/migration/`
- API docs: `docs/api/`
- Deployment docs: `docs/deployment/`

#### Source Code → `src/`
- Main packages: `src/packages/`
- Applications: `src/apps/`
- Client SDKs: `src/client_sdks/`
- Tests: `src/tests/`

#### Scripts → `scripts/`
- Build scripts: `scripts/build/`
- Deployment: `scripts/deployment/`
- Development: `scripts/development/`
- Validation: `scripts/validation/`

### 3. Naming Conventions

#### Files
- Use snake_case for Python files
- Use kebab-case for documentation files
- Use UPPERCASE for project-level files (README, LICENSE, CHANGELOG)
- Prefix temporary files with `temp_` or `tmp_`

#### Directories
- Use snake_case for Python package directories
- Use kebab-case for documentation directories
- Use lowercase for all directory names
- No spaces in directory names

### 4. Temporary File Management

#### Prohibited Patterns
- No files ending in `.tmp`, `.temp`, `.bak`, `.backup` in main directories
- No directories named `temp`, `tmp`, `scratch`, `debug` in organized areas
- No cache directories except in designated areas
- No build artifacts in source directories

#### Allowed Temporary Locations
- `scripts/temp/` (auto-cleaned)
- `src/packages/*/temp/` (auto-cleaned)
- Individual package temp directories (auto-cleaned)

## Enforcement Mechanisms

### 1. Automated Validation
- Pre-commit hooks validate structure
- CI pipeline checks organization
- Daily cleanup automation
- Developer workflow helpers

### 2. Git Hooks
- **pre-commit**: Structure validation
- **pre-push**: Clean state verification
- **post-commit**: Cleanup suggestions

### 3. Continuous Monitoring
- File organization scanner
- Rule violation alerts
- Automated cleanup suggestions
- Developer guidance

## Violation Handling

### Level 1: Warnings
- Misplaced configuration files
- Temporary files in wrong locations
- Minor naming convention violations

### Level 2: Blocks
- Files in prohibited root locations
- Critical organizational violations
- Security-sensitive misplacements

### Level 3: Auto-fix
- Move files to correct locations
- Clean up temporary files
- Organize known patterns

## Developer Workflow

### Before Committing
1. Run `scripts/validation/validate_organization.py`
2. Use `scripts/cleanup/auto_organize.py` if needed
3. Review organization report

### Daily Maintenance
1. Automated cleanup runs
2. Organization report generated
3. Violations flagged for review

### Weekly Review
1. Comprehensive organization audit
2. Rule effectiveness review
3. Process improvements

## Emergency Procedures

### Organization Drift Recovery
1. Run `scripts/recovery/restore_organization.py`
2. Review and approve changes
3. Update rules if needed

### Mass File Movement
1. Use `scripts/migration/bulk_organize.py`
2. Validate with `scripts/validation/verify_migration.py`
3. Test build system integrity

## Rule Updates

### Process
1. Propose rule changes via GitHub issue
2. Team review and discussion
3. Update automation scripts
4. Deploy with monitoring

### Versioning
- Rules versioned with semantic versioning
- Breaking changes require team approval
- Backward compatibility maintained where possible

## Metrics and Monitoring

### Organization Health Score
- Root directory cleanliness: 100%
- Proper file placement: 95%+
- Naming convention compliance: 95%+
- Temporary file management: 100%

### Automated Reports
- Daily organization status
- Weekly trend analysis
- Monthly rule effectiveness review

---

**Last Updated**: Auto-updated by repository automation
**Version**: 1.0.0
**Next Review**: Monthly