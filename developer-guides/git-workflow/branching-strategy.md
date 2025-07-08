# Git Branching Strategy

## Overview

This document defines the formal branching strategy for our development workflow. It establishes clear guidelines for branch naming, lifecycle management, and merge patterns to ensure consistent and maintainable code delivery.

## Branch Types

### Base Branches

#### `main`
- **Purpose**: Production-ready code
- **Stability**: Always stable and deployable
- **Protection**: Protected branch with required reviews
- **Deployment**: Automatically deployed to production

#### `develop`
- **Purpose**: Integration branch for ongoing development
- **Stability**: Generally stable, contains latest development changes
- **Protection**: Protected branch with required reviews
- **Deployment**: Deployed to staging/development environments

### Short-lived Branches

#### `feat/*`
- **Purpose**: New feature development
- **Naming**: `feat/feature-name` or `feat/ticket-number-feature-name`
- **Lifecycle**: Created from `develop`, merged back to `develop`
- **Examples**: `feat/user-authentication`, `feat/JIRA-123-payment-gateway`

#### `fix/*`
- **Purpose**: Bug fixes for existing features
- **Naming**: `fix/bug-description` or `fix/ticket-number-bug-description`
- **Lifecycle**: Created from `develop`, merged back to `develop`
- **Examples**: `fix/login-validation`, `fix/BUG-456-memory-leak`

#### `chore/*`
- **Purpose**: Maintenance tasks, dependency updates, tooling changes
- **Naming**: `chore/task-description`
- **Lifecycle**: Created from `develop`, merged back to `develop`
- **Examples**: `chore/update-dependencies`, `chore/ci-optimization`

#### `docs/*`
- **Purpose**: Documentation updates and improvements
- **Naming**: `docs/documentation-area`
- **Lifecycle**: Created from `develop`, merged back to `develop`
- **Examples**: `docs/api-documentation`, `docs/readme-update`

### Release Branches

#### `release/*`
- **Purpose**: Prepare new production releases
- **Naming**: `release/version-number` (following semantic versioning)
- **Lifecycle**: Created from `develop`, merged to both `main` and `develop`
- **Examples**: `release/1.2.0`, `release/2.0.0-beta.1`

### Hot-fix Branches

#### `hotfix/*`
- **Purpose**: Critical fixes for production issues
- **Naming**: `hotfix/issue-description` or `hotfix/version-number`
- **Lifecycle**: Created from `main`, merged to both `main` and `develop`
- **Examples**: `hotfix/security-vulnerability`, `hotfix/1.2.1`

## Visual Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Git Branching Strategy                     │
└─────────────────────────────────────────────────────────────────────┘

main      ●─────●─────────────●─────────────●─────────────●
          │     │             │             │             │
          │     │             │             │             │
release   │     │       ┌─────●─────┐       │             │
          │     │       │           │       │             │
          │     │       │           ▼       │             │
develop   ●─────●─────●─●─────●─────●─────●─●─────●─────●─●
          │     │     │             │     │       │     │
          │     │     │             │     │       │     │
feat/*    │     ▼     │             │     │       │     │
          │   ┌─●─┐   │             │     │       │     │
          │   │   │   │             │     │       │     │
          │   └─▲─┘   │             │     │       │     │
          │     │     │             │     │       │     │
fix/*     │     │     ▼             │     │       │     │
          │     │   ┌─●─┐           │     │       │     │
          │     │   │   │           │     │       │     │
          │     │   └─▲─┘           │     │       │     │
          │     │     │             │     │       │     │
hotfix/*  │     │     │             │     ▼       │     │
          │     │     │             │   ┌─●─┐     │     │
          │     │     │             │   │   │     │     │
          │     │     │             │   └─▲─┘     │     │
          │     │     │             │     │       │     │
chore/*   │     │     │             │     │       ▼     │
          │     │     │             │     │     ┌─●─┐   │
          │     │     │             │     │     │   │   │
          │     │     │             │     │     └─▲─┘   │
          │     │     │             │     │       │     │
docs/*    │     │     │             │     │       │     ▼
          │     │     │             │     │       │   ┌─●─┐
          │     │     │             │     │       │   │   │
          │     │     │             │     │       │   └─▲─┘
          │     │     │             │     │       │     │
          ▼     ▼     ▼             ▼     ▼       ▼     ▼
        Time ──────────────────────────────────────────────▶

Legend:
● = Commit/Merge point
┌─●─┐ = Branch lifecycle (create → work → merge)
▼ = Branch creation direction
▲ = Merge direction
```

## Branch State Machine

### Branch Compliance and Validation

All branches must pass branch compliance checks in CI/CD: 
- Names must follow established conventions (e.g., `feat/*`, `fix/*`)
- Direct pushes to `main` and `develop` are restricted
- `make branch-validate` to ensure correct naming before any push

### Allowed Branch Transitions

| From Branch | To Branch | Operation | Conditions |
|-------------|-----------|-----------|------------|
| `develop` | `feat/*` | Create | Always allowed |
| `develop` | `fix/*` | Create | Always allowed |
| `develop` | `chore/*` | Create | Always allowed |
| `develop` | `docs/*` | Create | Always allowed |
| `develop` | `release/*` | Create | When ready for release |
| `main` | `hotfix/*` | Create | Critical production issue |
| `feat/*` | `develop` | Merge | After code review and CI pass |
| `fix/*` | `develop` | Merge | After code review and CI pass |
| `chore/*` | `develop` | Merge | After code review and CI pass |
| `docs/*` | `develop` | Merge | After code review and CI pass |
| `release/*` | `main` | Merge | After release testing |
| `release/*` | `develop` | Merge | After merging to main |
| `hotfix/*` | `main` | Merge | After critical fix validation |
| `hotfix/*` | `develop` | Merge | After merging to main |

### State Machine Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Branch State Machine                             │
└─────────────────────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────────────────────────┐
        │                                                         │
        │    ┌─────────┐     create     ┌─────────────────────┐   │
        │    │         │  ──────────▶   │                     │   │
        │    │ develop │                │    Short-lived      │   │
        │    │         │  ◀──────────   │  (feat/fix/chore/   │   │
        │    └─────────┘     merge      │     docs)           │   │
        │         │                     │                     │   │
        │         │                     └─────────────────────┘   │
        │         │                                               │
        │         │ create                                        │
        │         ▼                                               │
        │    ┌─────────┐     merge      ┌─────────────────────┐   │
        │    │         │  ──────────▶   │                     │   │
        │    │release/*│                │        main         │   │
        │    │         │  ◀──merge──    │                     │   │
        │    └─────────┘     back       └─────────────────────┘   │
        │                                          │               │
        │                                          │ create        │
        │                                          ▼               │
        │                                    ┌─────────┐           │
        │                                    │         │           │
        │                                    │hotfix/* │           │
        │                                    │         │           │
        │                                    └─────────┘           │
        │                                          │               │
        │                                          │ merge         │
        │                                          ▼               │
        └──────────────────────────────────────────────────────────┘
```

## Merge Strategies

### Fast-Forward Merges
- **Used for**: Clean linear history
- **Branches**: `hotfix/*` → `main`
- **Condition**: When no divergent commits exist

### Merge Commits
- **Used for**: Preserving branch history
- **Branches**: `feat/*`, `fix/*`, `chore/*`, `docs/*` → `develop`
- **Benefits**: Clear feature boundaries, easier rollbacks

### Squash Merges
- **Used for**: Clean commit history
- **Branches**: Small feature branches with multiple commits
- **Benefits**: Simplified history, atomic changes

## Branch Protection Rules

### `main` Branch
- ✅ Require pull request reviews (minimum 2 reviewers)
- ✅ Require status checks to pass
- ✅ Require branches to be up to date before merging
- ✅ Restrict pushes to specific roles (maintainers only)
- ✅ Require linear history

### `develop` Branch
- ✅ Require pull request reviews (minimum 1 reviewer)
- ✅ Require status checks to pass
- ✅ Require branches to be up to date before merging
- ✅ Allow force pushes (for integration fixes)

### Release Branches (`release/*`)
- ✅ Require pull request reviews (minimum 2 reviewers)
- ✅ Require status checks to pass
- ✅ Restrict pushes to release managers

## Workflow Examples

### Feature Development
```bash
# Create feature branch (with validation)
make branch-new TYPE=feature NAME=user-profile-page
# OR manually:
# git checkout develop
# git pull origin develop
# git checkout -b feat/user-profile-page

# Validate branch name before starting work
make branch-validate

# Development work
git add .
git commit -m "Add user profile component"

# Validate again before push (recommended)
make branch-validate
git push origin feat/user-profile-page

# Create pull request to develop
# CI Pipeline runs:
# 1. Branch compliance (validates name, checks for direct push)
# 2. All other jobs run only if branch compliance passes
# After approval and CI pass, merge to develop
```

### Release Process
```bash
# Create release branch
git checkout develop
git pull origin develop
git checkout -b release/1.2.0

# Finalize release (version bumps, changelog, etc.)
git add .
git commit -m "Prepare release 1.2.0"
git push origin release/1.2.0

# Create pull request to main
# After approval, merge to main and back to develop
```

### Hotfix Process
```bash
# Create hotfix branch
git checkout main
git pull origin main
git checkout -b hotfix/security-patch

# Apply critical fix
git add .
git commit -m "Fix security vulnerability"
git push origin hotfix/security-patch

# Create pull request to main
# After approval, merge to main and back to develop
```

## Best Practices

### Branch Naming
- Use lowercase with hyphens
- Include ticket numbers when applicable
- Be descriptive but concise
- Follow the established prefixes

### Commit Messages
- Use conventional commit format
- Write clear, descriptive messages
- Reference ticket numbers
- Use imperative mood

### Code Reviews
- Review code thoroughly before merging
- Check for code quality and standards
- Verify tests pass and coverage is adequate
- Ensure documentation is updated

### Cleanup
- Delete merged branches promptly
- Keep branch count manageable
- Regular maintenance of stale branches
- Use automated cleanup tools when possible

## Integration with CI/CD

### Branch Compliance Pipeline

**New First-Stage Validation (2024):**
- **Branch Compliance Job**: Runs first in all CI pipelines
- **Naming Validation**: Executes `make branch-validate` automatically
- **Direct Push Protection**: Blocks direct pushes to `main`/`develop`
- **Early Failure**: Stops entire pipeline if branch name is invalid
- **Dependency Chain**: All subsequent jobs depend on branch compliance

#### CI Job Order
```
1. branch-compliance     ← NEW: First validation step
2. pre-commit           ← Depends on branch-compliance
3. validate-structure   ← Depends on branch-compliance
4. docs-lint           ← Depends on branch-compliance
5. code-quality        ← Depends on branch-compliance
6. build               ← Depends on branch-compliance
7. test                ← Depends on branch-compliance
8. api-cli-test        ← Depends on branch-compliance
9. security            ← Depends on branch-compliance
10. docker             ← Depends on branch-compliance + build
11. ci-summary         ← Depends on all jobs
```

### Continuous Integration
- All branches trigger CI builds
- Branch compliance checked before any other tests
- Tests must pass before merging
- Code quality checks enforced
- Security scans on all commits

### Continuous Deployment
- `main` branch auto-deploys to production
- `develop` branch auto-deploys to staging
- `release/*` branches deploy to UAT environment
- Feature branches deploy to preview environments

## Troubleshooting

### Common Issues
1. **Merge conflicts**: Regularly sync with target branch
2. **Failed CI**: Fix issues before requesting review
3. **Stale branches**: Regular cleanup and maintenance
4. **Accidental commits**: Use `git revert` for safe rollbacks

### Emergency Procedures
1. **Production hotfix**: Follow hotfix workflow immediately
2. **Broken main**: Revert commits or deploy previous version
3. **Corrupted develop**: Recreate from last stable point
4. **Lost commits**: Use `git reflog` for recovery

## Working While Services Are Running

When working with long-running services, it's essential to ensure that your workflow remains efficient and safe. Here are some best practices to consider:

### Using `git worktree` for Parallel Branches

`git worktree` allows you to work on multiple branches simultaneously without changing your current directory, which is particularly useful when services are running on a specific branch.

```bash
# Create a new worktree for a new feature
git worktree add ../feature-worktree feat/new-feature

# Check out an existing branch to a new worktree
git worktree add ../bugfix-worktree fix/critical-bug

# List all worktrees
git worktree list

# Remove a worktree when done
git worktree remove ../feature-worktree
```

**Benefits:**
- Keep services running on one branch while developing on another
- Avoid stopping/starting services when switching branches
- Parallel development and testing workflows

### Safe Stash/Commit Patterns Before Long-Running Jobs

Before starting long-running jobs (builds, tests, deployments), always save your work to avoid losing changes.

#### Stashing Changes
```bash
# Stash all changes with a descriptive message
git stash push -m "WIP: Before long-running job"

# Stash only specific files
git stash push -m "WIP: Database changes" -- src/database/

# Include untracked files
git stash push -u -m "WIP: Including new files"

# Apply stash after job completion
git stash pop
```

#### Committing Changes
```bash
# Create a temporary commit
git add .
git commit -m "WIP: Save work before build"

# After job completion, you can:
# 1. Continue working and amend the commit
git commit --amend -m "Complete feature implementation"

# 2. Or reset to continue with uncommitted changes
git reset HEAD~1
```

### Automatic Rebase vs Merge Guidance

Choose the appropriate integration strategy based on your workflow and branch state:

#### When to Use Rebase
- **Personal feature branches** that haven't been shared
- **Clean up commit history** before merging
- **Linear history preference** for easier tracking

```bash
# Interactive rebase to clean up commits
git rebase -i HEAD~3

# Rebase feature branch onto latest develop
git checkout feat/your-feature
git rebase develop

# Handle conflicts during rebase
git add resolved-file.py
git rebase --continue
```

#### When to Use Merge
- **Shared branches** with multiple contributors
- **Preserve complete history** including branch context
- **Safer option** when uncertain about conflicts

```bash
# Standard merge
git checkout develop
git merge feat/your-feature

# Merge with explicit commit message
git merge feat/your-feature -m "Merge feature: User authentication"

# No-fast-forward merge to preserve branch history
git merge feat/your-feature --no-ff
```

#### Automated Merge Strategies
```bash
# Configure automatic rebase for pulls
git config pull.rebase true

# Configure merge strategy for specific branches
git config branch.develop.mergeoptions "--no-ff"

# Use merge tool for conflicts
git config merge.tool vimdiff
```

### Best Practices for Concurrent Development

1. **Frequent Integration**: Regularly sync with base branches
2. **Atomic Commits**: Keep commits focused and complete
3. **Clear Messaging**: Use descriptive commit messages
4. **Backup Work**: Always save work before risky operations
5. **Service Isolation**: Use worktrees to isolate running services

---

*This document should be reviewed and updated regularly to reflect changes in our development workflow and tooling.*
