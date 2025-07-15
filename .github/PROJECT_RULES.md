# Pynomaly Project Rules

## Issue Management

### Single Source of Truth: GitHub Issues Only

**RULE**: The project uses **GitHub Issues exclusively** for task and issue tracking. 

### Prohibited Practices

❌ **No local TODO.md files**
- Local TODO.md files create synchronization problems
- Leads to duplicate tracking and confusion
- Creates maintenance overhead

❌ **No duplicate issue tracking systems**
- Only GitHub Issues should be used
- No local markdown files for issue tracking
- No separate task management files

### Required Practices

✅ **Use GitHub Issues for all tracking**
- All tasks, bugs, features, and documentation items go in GitHub Issues
- Use GitHub labels for categorization (P0-Critical, P1-High, P2-Medium, P3-Low)
- Use GitHub milestones for project phases
- Use GitHub Projects for kanban-style tracking

✅ **Issue Templates and Labels**
- Use consistent labeling: bug, enhancement, documentation, Infrastructure, Application, Presentation
- Priority labels: P0-Critical, P1-High, P2-Medium, P3-Low
- Status tracking via issue state (open/closed) and labels (In-Progress, Blocked)

✅ **Reference Issues in Commits**
- Always reference relevant issue numbers in commit messages
- Use "fixes #123" or "closes #123" to auto-close issues
- Use "relates to #123" for partial progress

## Rationale

This rule was established because:
1. Maintaining both local TODO.md and GitHub Issues led to synchronization issues
2. Information became scattered across multiple sources
3. Automation complexity increased maintenance burden
4. GitHub Issues provide better collaboration, linking, and project management features

## Implementation Date
July 14, 2025

## Enforcement
All contributors must follow this rule. Any pull requests containing local TODO.md files or alternative tracking systems will be rejected.