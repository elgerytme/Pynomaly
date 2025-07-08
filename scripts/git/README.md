# Git Helper Scripts

Cross-platform Git helper scripts for safe branch management.

## Scripts

### `git_new_branch.sh` / `git_new_branch.ps1`
Creates a new Git branch with validation.

**Usage:**
```bash
# Using Makefile (recommended)
make branch-new TYPE=feature NAME=my-feature

# Direct usage (Bash)
./scripts/git/git_new_branch.sh feature my-feature

# Direct usage (PowerShell)
powershell -ExecutionPolicy Bypass -File scripts/git/git_new_branch.ps1 -Type feature -Name my-feature

# Using hatch env run
hatch env run git:git_new_branch.sh feature my-feature
hatch env run git:git_new_branch.ps1 feature my-feature
```

**Valid branch types:**
- `feature` - New features
- `bugfix` - Bug fixes
- `hotfix` - Critical fixes
- `release` - Release preparation
- `chore` - Maintenance tasks
- `docs` - Documentation changes

**Branch naming rules:**
- Must contain only lowercase letters, numbers, and hyphens
- Examples: `feature/user-auth`, `bugfix/memory-leak`, `docs/api-updates`

### `git_switch_safe.sh` / `git_switch_safe.ps1`
Switches to a Git branch safely with uncommitted changes check.

**Usage:**
```bash
# Using Makefile (recommended)
make branch-switch NAME=feature/my-feature

# Direct usage (Bash)
./scripts/git/git_switch_safe.sh feature/my-feature

# Direct usage (PowerShell)
powershell -ExecutionPolicy Bypass -File scripts/git/git_switch_safe.ps1 -Name feature/my-feature

# Using hatch env run
hatch env run git:git_switch_safe.sh feature/my-feature
hatch env run git:git_switch_safe.ps1 feature/my-feature
```

## Environment Detection

The Makefile automatically detects the OS and uses the appropriate script:

- **Windows**: Uses PowerShell scripts (`.ps1`)
- **Linux/macOS**: Uses Bash scripts (`.sh`) via `hatch env run`

## Integration with Hatch

The scripts are integrated with Hatch environments for cross-platform compatibility:

```toml
[tool.hatch.envs.git]
detached = true
dependencies = []

[tool.hatch.envs.git.scripts]
git_new_branch.sh = "bash scripts/git/git_new_branch.sh {args}"
git_new_branch.ps1 = "powershell scripts/git/git_new_branch.ps1 {args}"
git_switch_safe.sh = "bash scripts/git/git_switch_safe.sh {args}"
git_switch_safe.ps1 = "powershell scripts/git/git_switch_safe.ps1 {args}"
```

## Examples

### Creating a Feature Branch
```bash
# Create a new feature branch
make branch-new TYPE=feature NAME=anomaly-detection

# Create a bugfix branch
make branch-new TYPE=bugfix NAME=memory-leak

# Create a documentation branch
make branch-new TYPE=docs NAME=api-updates
```

### Switching Branches
```bash
# Switch to an existing branch
make branch-switch NAME=feature/anomaly-detection

# Switch to main branch
make branch-switch NAME=main
```

### Error Handling

The scripts include validation and error handling:

- **Branch type validation**: Only valid types are accepted
- **Branch name validation**: Enforces naming conventions
- **Duplicate branch check**: Prevents creating existing branches
- **Uncommitted changes check**: Prevents switching with uncommitted changes
- **Cross-platform compatibility**: Works on Windows, Linux, and macOS

### Integration with CI/CD

The scripts follow the same branch naming conventions used by the Makefile's `branch-validate` target, ensuring consistency across development and CI environments.

### Branch Compliance Integration

These scripts are integrated with Pynomaly's CI/CD pipeline:

- **Automated Validation**: The `branch-validate` target runs these validation rules
- **CI/CD Integration**: Branch compliance job in GitHub Actions uses same validation
- **Protection**: Direct pushes to `main`/`develop` are blocked
- **Early Failure**: Invalid branch names fail CI before other jobs run

**Related Commands:**
```bash
# Validate current branch (used by CI)
make branch-validate

# Install Git hooks for local validation
make git-hooks

# Check branch compliance before CI
make ci  # Includes branch validation
```

**Documentation**: See [CI/CD Branch Compliance](../../docs/developer-guides/contributing/CI_CD_BRANCH_COMPLIANCE.md) for complete integration details.
