# CI/CD Branch Compliance Integration

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 👨‍💻 [Developer Guides](../README.md) > 🤝 [Contributing](README.md) > 🔄 Branch Compliance

---

## Overview

The Pynomaly CI/CD pipeline now includes comprehensive branch compliance validation to ensure consistent development practices and prevent direct pushes to protected branches. This system enforces branch naming conventions and validates workflow compliance before any other CI jobs run.

## Branch Compliance Job

### Purpose
- **First-stage validation** in all CI pipelines
- **Early failure** to save CI/CD resources
- **Consistent enforcement** of branch naming conventions
- **Protection of main/develop** branches from direct pushes

### Job Configuration

```yaml
branch-compliance:
  name: Branch Compliance
  runs-on: ubuntu-latest
  steps:
  - name: Check branch protection
    run: |
      # Prevents direct pushes to main/develop
      if [ "$EVENT_NAME" = "push" ]; then
        if [ "$CURRENT_BRANCH" = "main" ] || [ "$CURRENT_BRANCH" = "develop" ]; then
          echo "❌ ERROR: Direct push to protected branch '$CURRENT_BRANCH' is not allowed"
          exit 1
        fi
      fi
  
  - name: Validate branch name
    run: |
      make branch-validate
```

## Branch Naming Conventions

### Valid Branch Formats

| Branch Type | Pattern | Example | Description |
|-------------|---------|---------|-------------|
| **Feature** | `feature/<name>` | `feature/user-auth` | New feature development |
| **Bugfix** | `bugfix/<name>` | `bugfix/memory-leak` | Bug fixes |
| **Hotfix** | `hotfix/<name>` | `hotfix/security-patch` | Critical production fixes |
| **Release** | `release/<version>` | `release/1.2.0` | Release preparation |
| **Chore** | `chore/<name>` | `chore/update-deps` | Maintenance tasks |
| **Docs** | `docs/<name>` | `docs/api-guide` | Documentation updates |

### Protected Branches

- **main**: Production-ready code (PR-only)
- **develop**: Development integration branch (PR-only)

### Validation Rules

```bash
# Branch name must match pattern:
^(feature|bugfix|hotfix|release|chore|docs)/[a-z0-9-]+$

# Examples:
✅ feature/user-authentication
✅ bugfix/memory-leak-fix
✅ hotfix/security-vulnerability
✅ release/1.2.0
✅ chore/dependency-update
✅ docs/api-documentation

# Invalid examples:
❌ feature/User-Authentication (uppercase)
❌ bugfix/memory_leak_fix (underscores)
❌ MyFeature (no type prefix)
❌ feature/ (empty name)
```

## Pipeline Integration

### Job Dependencies

All CI jobs now depend on `branch-compliance` passing:

```yaml
jobs:
  branch-compliance:
    name: Branch Compliance
    runs-on: ubuntu-latest
    # ... validation steps ...

  pre-commit:
    name: Pre-commit Hooks
    runs-on: ubuntu-latest
    needs: [branch-compliance]  # ← Dependency

  code-quality:
    name: Code Quality & Linting
    runs-on: ubuntu-latest
    needs: [branch-compliance]  # ← Dependency

  # ... all other jobs depend on branch-compliance
```

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CI/CD Pipeline Flow                             │
└─────────────────────────────────────────────────────────────────────┘

1. 🔍 branch-compliance     [FIRST - Required for all branches]
   ├─ Check branch protection (prevent direct push to main/develop)
   ├─ Validate branch naming convention
   └─ Early failure if invalid
   
2. 🎯 All other jobs run in parallel (if branch-compliance passes)
   ├─ pre-commit
   ├─ validate-structure
   ├─ docs-lint
   ├─ code-quality
   ├─ build
   ├─ test
   ├─ api-cli-test
   ├─ security
   └─ docker (depends on build)
   
3. 📊 ci-summary            [LAST - Depends on all jobs]
   └─ Generate comprehensive pipeline report
```

## Developer Workflow

### Creating New Branches

#### Option 1: Using Make Commands (Recommended)

```bash
# Create new branch with validation
make branch-new TYPE=feature NAME=user-dashboard

# This automatically:
# 1. Validates the branch name format
# 2. Checks current git status
# 3. Creates branch from develop
# 4. Switches to new branch
```

#### Option 2: Manual Creation

```bash
# Create branch manually
git checkout develop
git pull origin develop
git checkout -b feature/user-dashboard

# Validate the branch name
make branch-validate
```

### Switching Branches

```bash
# Safe branch switching with validation
make branch-switch NAME=feature/user-dashboard

# This automatically:
# 1. Checks for uncommitted changes
# 2. Validates target branch exists
# 3. Switches to branch
# 4. Validates branch name
```

### Pre-Push Validation

```bash
# Always validate before pushing
make branch-validate

# If validation passes:
git push origin feature/user-dashboard
```

## Makefile Commands

### Branch Management Commands

```bash
# Create new branch with validation
make branch-new TYPE=<type> NAME=<name>
# Example: make branch-new TYPE=feature NAME=user-auth

# Switch branches safely
make branch-switch NAME=<branch-name>
# Example: make branch-switch NAME=feature/user-auth

# Validate current branch
make branch-validate
```

### Command Implementation

The `make branch-validate` command implements the following logic:

```bash
# Get current branch
CURRENT_BRANCH=$(git branch --show-current)

# Check if it's a protected branch (allowed)
if [ "$CURRENT_BRANCH" = "main" ] || [ "$CURRENT_BRANCH" = "develop" ]; then
    echo "✅ Main branch '$CURRENT_BRANCH' - validation passed"
    exit 0
fi

# Validate naming convention
if echo "$CURRENT_BRANCH" | grep -qE '^(feature|bugfix|hotfix|release|chore|docs)/[a-z0-9-]+$'; then
    echo "✅ Branch name '$CURRENT_BRANCH' follows naming convention"
else
    echo "❌ ERROR: Branch name '$CURRENT_BRANCH' does not follow naming convention"
    echo "Expected format: <type>/<name>"
    echo "Valid types: feature, bugfix, hotfix, release, chore, docs"
    exit 1
fi
```

## Error Handling

### Common Validation Errors

#### 1. Invalid Branch Name Format

```bash
❌ Error: Branch name 'MyFeature' does not follow naming convention
Expected format: <type>/<name>
Valid types: feature, bugfix, hotfix, release, chore, docs
Name must contain only lowercase letters, numbers, and hyphens
Examples: feature/anomaly-detection, bugfix/memory-leak, docs/api-updates
```

**Solution**: Rename branch or create new one with valid format.

#### 2. Direct Push to Protected Branch

```bash
❌ ERROR: Direct push to protected branch 'main' is not allowed
Please use Pull Requests for changes to main/develop branches
```

**Solution**: Create feature branch and use PR workflow.

#### 3. Branch Creation Failure

```bash
❌ Error: Both TYPE and NAME are required
Usage: make branch-new TYPE=<type> NAME=<name>
Valid types: feature, bugfix, hotfix, release, chore, docs
Example: make branch-new TYPE=feature NAME=anomaly-detection
```

**Solution**: Provide both TYPE and NAME parameters.

### Troubleshooting

#### Check Current Branch Status

```bash
# Show current branch and validation status
make branch-validate

# Show git status
git status

# Show all branches
git branch -a
```

#### Fix Invalid Branch Name

```bash
# Option 1: Rename current branch
git branch -m feature/new-valid-name

# Option 2: Create new branch and delete old one
git checkout -b feature/new-valid-name
git branch -D old-invalid-name
```

## CI/CD Best Practices

### 1. Always Validate Before Push

```bash
# Recommended workflow
make branch-validate
git add .
git commit -m "feat: implement user authentication"
make branch-validate  # Double-check
git push origin feature/user-auth
```

### 2. Use Make Commands for Safety

```bash
# Use make commands instead of raw git
make branch-new TYPE=feature NAME=user-auth  # vs git checkout -b
make branch-switch NAME=feature/user-auth    # vs git checkout
make branch-validate                         # Always validate
```

### 3. Monitor CI Pipeline

```bash
# Check CI status after push
gh pr status  # If using GitHub CLI

# View detailed CI logs
gh run view --web  # Opens CI run in browser
```

### 4. Handle CI Failures

If branch compliance fails:

1. **Check error message** in CI logs
2. **Fix branch name** if invalid
3. **Ensure no direct pushes** to main/develop
4. **Re-run validation** locally: `make branch-validate`
5. **Force push** if branch was renamed: `git push -f origin feature/fixed-name`

## Integration with Git Hooks

### Pre-commit Hook

The branch compliance is also enforced via pre-commit hooks:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: branch-naming
        name: Branch Naming Convention
        entry: make branch-validate
        language: system
        always_run: true
        pass_filenames: false
```

### Pre-push Hook

```bash
#!/bin/bash
# .git/hooks/pre-push
echo "🔍 Validating branch name before push..."
make branch-validate
```

## Configuration

### GitHub Actions Configuration

The branch compliance job is configured in `.github/workflows/ci.yml`:

```yaml
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  branch-compliance:
    name: Branch Compliance
    runs-on: ubuntu-latest
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: Check branch protection
      run: |
        CURRENT_BRANCH="${{ github.ref_name }}"
        EVENT_NAME="${{ github.event_name }}"
        
        if [ "$EVENT_NAME" = "push" ]; then
          if [ "$CURRENT_BRANCH" = "main" ] || [ "$CURRENT_BRANCH" = "develop" ]; then
            echo "❌ ERROR: Direct push to protected branch '$CURRENT_BRANCH' is not allowed"
            exit 1
          fi
        fi

    - name: Validate branch name
      run: |
        make branch-validate
```

### Branch Protection Rules

Configure in GitHub repository settings:

```yaml
# main branch protection
- Required status checks: branch-compliance
- Require pull request reviews: 2 reviewers
- Restrict pushes: maintainers only
- Require branches to be up to date

# develop branch protection  
- Required status checks: branch-compliance
- Require pull request reviews: 1 reviewer
- Restrict pushes: developers and above
```

## Migration Guide

### For Existing Branches

If you have existing branches that don't follow the naming convention:

```bash
# Check current branch
git branch --show-current

# If invalid, rename it
git branch -m feature/old-name feature/new-valid-name

# Update remote
git push origin -u feature/new-valid-name
git push origin --delete feature/old-name
```

### For Teams

1. **Communicate changes** to all team members
2. **Update documentation** and onboarding materials
3. **Train developers** on new make commands
4. **Gradually migrate** existing branches
5. **Monitor CI** for compliance failures

## Advanced Usage

### Custom Validation Rules

Extend the validation in `Makefile`:

```makefile
branch-validate-custom: ## Custom branch validation
	@echo "🔍 Running custom branch validation..."
	@# Add your custom validation logic here
	@make branch-validate
	@echo "✅ Custom validation passed"
```

### Integration with External Tools

```bash
# JIRA ticket validation
make branch-validate-jira TICKET=PROJ-123

# Slack notifications
make branch-validate-slack CHANNEL=#dev-team
```

## Monitoring and Metrics

### CI/CD Metrics

Track branch compliance metrics:

- **Validation success rate**
- **Common naming violations**
- **Direct push attempts**
- **CI pipeline failure rates**

### Reporting

Generate compliance reports:

```bash
# Weekly compliance report
make branch-compliance-report WEEK=2024-W12

# Team compliance summary
make branch-compliance-summary TEAM=backend
```

## Future Enhancements

### Planned Features

1. **Automated branch naming** suggestions
2. **Integration with issue trackers** (JIRA, GitHub Issues)
3. **Custom validation rules** per project
4. **Dashboard for compliance** metrics
5. **Slack/Teams integration** for notifications

### Feedback and Improvements

To suggest improvements or report issues:

1. **Create GitHub issue** with `[branch-compliance]` label
2. **Propose changes** via PR to this documentation
3. **Join discussions** in team channels

---

## Related Documentation

- [Git Branching Strategy](../../../developer-guides/git-workflow/branching-strategy.md)
- [Contributing Guidelines](../../../CONTRIBUTING.md)
- [Development Workflow](./README.md)
- [CI/CD Pipeline](./CI_CD_VERSION_UPDATE_SUMMARY.md)

---

**Last Updated**: 2024-07-08  
**Version**: 1.0.0  
**Maintainer**: Development Team
