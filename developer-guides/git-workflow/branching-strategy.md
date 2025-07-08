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
# Create feature branch
git checkout develop
git pull origin develop
git checkout -b feat/user-profile-page

# Development work
git add .
git commit -m "Add user profile component"
git push origin feat/user-profile-page

# Create pull request to develop
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

### Continuous Integration
- All branches trigger CI builds
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

---

*This document should be reviewed and updated regularly to reflect changes in our development workflow and tooling.*
