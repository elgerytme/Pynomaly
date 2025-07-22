# Git Branch Isolation Rule

## Overview
This rule ensures that git branches are created and managed in isolated environments to prevent interference between different users, agents, and development workstreams.

## Core Principles

### 1. Branch Naming Convention
All branches must follow the isolation pattern:
```
<type>/<scope>/<description>
```

Where:
- `type`: feature, bugfix, hotfix, experiment, agent, user
- `scope`: isolated identifier (user-id, agent-id, package-name, etc.)
- `description`: brief description using kebab-case

### 2. Isolation Scopes

#### User Isolation
- **Pattern**: `feature/user-<username>/<description>`
- **Example**: `feature/user-andre/quality-gates-validation`
- **Purpose**: Personal development branches

#### Agent Isolation  
- **Pattern**: `feature/agent-<agent-id>/<description>`
- **Example**: `feature/agent-claude/automated-refactoring`
- **Purpose**: AI agent development branches

#### Package Isolation
- **Pattern**: `feature/pkg-<package>/<description>`
- **Example**: `feature/pkg-anomaly-detection/new-algorithm`
- **Purpose**: Package-specific development

### 3. Protected Environments

#### Main Branch Protection
- `main`: Production-ready code only
- Requires PR approval and quality gate validation
- No direct pushes allowed

#### Integration Branches
- `develop`: Development integration branch
- `staging`: Staging environment branch
- Protected from direct manipulation

### 4. Temporary Branch Lifecycle

#### Automatic Cleanup
- Feature branches > 30 days: flagged for review
- Merged branches: automatically deleted after 7 days
- Stale branches: owner notified for cleanup

#### Branch Limits
- Maximum 5 active branches per user/agent
- Maximum 3 long-running experiment branches
- Automatic enforcement via git hooks

## Implementation Rules

### 1. Branch Creation Validation
```bash
# Valid examples
git checkout -b feature/user-andre/implement-caching
git checkout -b bugfix/agent-claude/fix-memory-leak
git checkout -b experiment/pkg-mlops/new-deployment-strategy

# Invalid examples (will be rejected)
git checkout -b my-feature
git checkout -b fix-bug
git checkout -b temp-branch
```

### 2. Commit Message Requirements
All commits in isolated branches must include:
```
<type>(<scope>): <description>

Isolation: <isolation-id>
Branch: <branch-name>
```

### 3. Merge Requirements
- All branches must pass quality gates before merging
- No cross-contamination between isolation scopes
- Automated conflict resolution for isolated changes

## Automation Features

### 1. Branch Validation Hook
- Pre-commit validation of branch names
- Isolation scope verification
- Conflict detection between isolation scopes

### 2. Automated Branch Management
- Stale branch detection and notification
- Automatic cleanup of merged branches
- Branch limit enforcement

### 3. Environment Isolation
- Separate test environments per isolation scope
- Isolated CI/CD pipelines for concurrent development
- Resource allocation per isolation boundary

## Enforcement Mechanisms

### 1. Git Hooks
- `pre-commit`: Branch name validation
- `pre-push`: Isolation boundary checks
- `post-merge`: Cleanup and notification

### 2. CI/CD Integration
- Branch-specific pipeline execution
- Isolated test environments
- Automated quality gate validation

### 3. Monitoring and Alerts
- Branch proliferation monitoring
- Isolation boundary violation alerts
- Resource usage tracking per scope

## Exception Handling

### Emergency Hotfixes
- Temporary bypass for critical production issues
- Requires explicit approval and justification
- Automated cleanup and audit trail

### Cross-Package Dependencies
- Coordinated development across package boundaries
- Requires dependency declaration and validation
- Automated integration testing

## Benefits

1. **Conflict Prevention**: Eliminates merge conflicts between users/agents
2. **Resource Isolation**: Separate environments prevent interference
3. **Parallel Development**: Multiple workstreams without coordination overhead
4. **Automated Management**: Reduces manual branch maintenance burden
5. **Quality Assurance**: Consistent validation across all development streams
6. **Audit Trail**: Clear tracking of changes by isolation scope

## Compliance

This rule is enforced through:
- Pre-commit hooks (mandatory)
- CI/CD pipeline validation
- Automated monitoring and reporting
- Regular compliance audits

Violations result in:
1. Commit rejection
2. Notification to development team
3. Automatic remediation where possible
4. Escalation for repeated violations