# Git Hooks for Pynomaly Project

This directory contains Git hooks that automate workflow enforcement for the Pynomaly project.

## Available Hooks

### `pre-commit`
- **Purpose**: Validates branch naming conventions and runs partial linting
- **Actions**:
  - Validates current branch follows naming convention: `<type>/<name>`
  - Valid types: `feature`, `bugfix`, `hotfix`, `release`, `chore`, `docs`
  - Runs `ruff` linting on staged Python files only
  - Checks for merge conflict markers
  - Warns about debug statements

### `pre-push`
- **Purpose**: Runs unit tests before pushing
- **Actions**:
  - Executes `make test-unit` to run unit tests
  - Prevents push if tests fail

### `post-checkout`
- **Purpose**: Reminds about restarting long-running services
- **Actions**:
  - Displays reminder to restart development services
  - Suggests using `make dev` or `make dev-storage`

## Installation

### Using Make (if available)
```bash
make git-hooks
```

### Using Scripts (cross-platform)

**Windows (PowerShell):**
```powershell
scripts/git/install-hooks.ps1
```

**Unix-like systems (Bash):**
```bash
scripts/git/install-hooks.sh
```

### Manual Installation
```bash
git config core.hooksPath scripts/git/hooks
```

## Branch Naming Convention

The hooks enforce the following branch naming convention:

- Format: `<type>/<name>`
- Valid types: `feature`, `bugfix`, `hotfix`, `release`, `chore`, `docs`
- Name must contain only lowercase letters, numbers, and hyphens
- Examples:
  - `feature/anomaly-detection`
  - `bugfix/memory-leak`
  - `docs/api-updates`
  - `chore/dependency-update`

## Features

- **Cross-platform**: Works on Windows, macOS, and Linux
- **Non-intrusive**: Only checks staged files for linting
- **Smart validation**: Skips branch validation for main branches (`main`, `master`, `develop`)
- **Informative output**: Colored output with clear success/failure messages
- **Fail-safe**: Gracefully handles missing tools (warns instead of failing)

## Customization

You can customize the hooks by editing the files in this directory:

- Modify branch naming patterns in `pre-commit`
- Change test commands in `pre-push`
- Update service restart reminders in `post-checkout`

## Troubleshooting

If hooks fail to execute:

1. **Check permissions**: Ensure hooks are executable
2. **Verify tools**: Ensure `hatch`, `ruff`, and `python` are available
3. **Test manually**: Run individual hook scripts to debug issues
4. **Check paths**: Verify `core.hooksPath` is set correctly

```bash
git config --get core.hooksPath
# Should output: scripts/git/hooks
```

## Integration with CI/CD

These hooks complement the CI/CD pipeline by:

- Catching issues early in the development process
- Enforcing consistent code quality standards
- Reducing failed CI builds
- Maintaining branch naming consistency

The hooks are designed to work alongside the existing `make` targets and Hatch environments.
