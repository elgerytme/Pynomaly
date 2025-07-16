# Git Workflow and History Management

This document outlines the git workflow and history management practices for the Pynomaly project.

## Overview

The Pynomaly project follows a clean git history approach with:
- **Feature-based commits** - Each commit represents a complete feature or logical change
- **Logical version tags** - Version tags mark significant milestones and releases
- **Conventional commit messages** - Standardized commit message format for clarity
- **Clean history** - Avoiding noise commits and maintaining readable history

## Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>: <description> (Issue #<number>)

[optional body]

[optional footer]
```

### Commit Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that don't affect the meaning of the code
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **chore**: Changes to the build process or auxiliary tools
- **ci**: Changes to CI configuration files and scripts
- **revert**: Reverts a previous commit

### Examples

```bash
feat: Add anomaly detection algorithm (Issue #123)
fix: Resolve memory leak in data processing (Issue #456)
docs: Update API documentation for new endpoints
refactor: Simplify data transformation pipeline
test: Add unit tests for anomaly detection module
chore: Update build configuration for new dependencies
```

## Version Tagging Strategy

### Version Format

We use semantic versioning (SemVer) with the format `vMAJOR.MINOR.PATCH`:

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality in a backwards compatible manner
- **PATCH**: Backwards compatible bug fixes

### Current Version Tags

- `v1.1.0`: PyPI Package Release Preparation Complete
- `v1.2.0`: MLOps Phase 4 - Pipeline Orchestration System Complete
- `v1.3.0`: Comprehensive Data Transformation Package Integration
- `v1.4.0`: MLOps Phase 4 Pipeline Orchestration Exports Finalized

### Creating Version Tags

```bash
# Create an annotated tag
git tag -a v1.5.0 -m "v1.5.0: Major feature release with new algorithms"

# Push tags to remote
git push origin --tags
```

## Git Workflow Scripts

### Git Cleanup Script

Use `scripts/git-cleanup.sh` for history management:

```bash
# Show commit statistics
./scripts/git-cleanup.sh stats

# Create backup tag
./scripts/git-cleanup.sh backup

# Create logical version tags
./scripts/git-cleanup.sh tags

# Configure commit message template
./scripts/git-cleanup.sh configure
```

### Feature-Based Commit Script

Use `scripts/commit-by-feature.sh` for structured commits:

```bash
# Commit a new feature
./scripts/commit-by-feature.sh feature 123 "Add new anomaly detection algorithm"

# Commit a bug fix
./scripts/commit-by-feature.sh fix 456 "Resolve memory leak in data processing"

# Commit documentation changes
./scripts/commit-by-feature.sh docs "Update API documentation"

# Create version tag
./scripts/commit-by-feature.sh tag v1.5.0 "Major feature release"
```

## Best Practices

### 1. Commit Organization

- **One feature per commit**: Each commit should represent a complete, logical change
- **Atomic commits**: Commits should be self-contained and not break the build
- **Descriptive messages**: Use clear, concise commit messages that explain what and why

### 2. Branch Strategy

- **Feature branches**: Create branches for new features or major changes
- **Clean merges**: Use squash merges to maintain clean history
- **Regular rebasing**: Keep feature branches up to date with main

### 3. History Maintenance

- **Avoid noise commits**: Don't commit auto-generated or temporary changes
- **Squash related commits**: Group related changes into single commits
- **Use meaningful tags**: Tag significant milestones and releases

### 4. Commit Template

The project uses a git commit template (`.gitmessage`) that provides structure:

```bash
# Configure git to use the template
git config commit.template .gitmessage

# Use the template (don't use -m flag)
git commit
```

## Troubleshooting

### Cleaning Up History

If the history becomes cluttered with noise commits:

1. Create a backup:
   ```bash
   ./scripts/git-cleanup.sh backup
   ```

2. Use interactive rebase to clean up:
   ```bash
   git rebase -i HEAD~20  # Last 20 commits
   ```

3. Remove or squash unwanted commits in the interactive editor

### Recovering from Mistakes

If you make a mistake with git history:

1. Find your backup tag:
   ```bash
   git tag -l | grep backup
   ```

2. Reset to the backup:
   ```bash
   git reset --hard backup-YYYYMMDD-HHMMSS
   ```

## Automated Tools

### Pre-commit Hooks

The project uses pre-commit hooks to enforce quality:

- Commit message format validation
- Code formatting and linting
- Test execution on staged changes

### GitHub Actions

Automated workflows that run on commits:

- CI/CD pipeline for testing and deployment
- Automated issue synchronization
- Release automation

## Migration from Old History

The project has migrated from a cluttered history with many auto-sync commits to a cleaner, feature-based approach. The old history is preserved in backup tags, and new development follows the clean history guidelines.

### Historical Backup Tags

- `backup-before-history-cleanup-YYYYMMDD`: Backup before implementing clean history
- Archive tags: Various archive tags for historical reference

## Contributing

When contributing to the project:

1. Follow the commit message format
2. Use the provided scripts for consistent commits
3. Create meaningful commit messages
4. Group related changes into single commits
5. Test your changes before committing

For more information on contributing, see the [CONTRIBUTING.md](../CONTRIBUTING.md) file.