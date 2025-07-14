# GitHub Issues to TODO.md Synchronization

This document describes the automated system that keeps the TODO.md file synchronized with GitHub issues.

## Overview

The automation system ensures that the "GitHub Issues" section of TODO.md always reflects the current state of GitHub issues. This eliminates manual maintenance overhead and ensures accuracy.

## Architecture

### Components

1. **GitHub Actions Workflow** (`.github/workflows/issue-sync.yml`)
   - Triggers on issue events (opened, closed, labeled, etc.)
   - Runs the synchronization script
   - Commits changes back to the repository

2. **Synchronization Script** (`scripts/automation/sync_github_issues_to_todo.py`)
   - Fetches issues from GitHub API
   - Formats issues for TODO.md
   - Updates the Issues section in TODO.md

3. **Manual Sync Script** (`scripts/automation/manual_sync.py`)
   - Allows manual triggering of sync
   - Useful for testing and development

## Features

### Automatic Triggers

The system automatically runs when:

- Issues are opened, edited, closed, or reopened
- Issues are labeled or unlabeled
- Issue comments are created, edited, or deleted
- Manual workflow dispatch is triggered

### Priority Mapping

GitHub issue labels are automatically mapped to priorities:

- `P1-High` → 🔥 P1-High
- `P2-Medium` → 🔶 P2-Medium  
- `P3-Low` → 🟢 P3-Low
- No priority label → 📋 General (defaults to medium)

### Status Detection

Issue status is determined automatically:

- **Closed issues** → ✅ COMPLETED
- **Issues with "in-progress" label** → 🔄 IN PROGRESS
- **Issues with "blocked" label** → 🚫 BLOCKED
- **All other open issues** → ⏳ PENDING

### Category Classification

Issues are categorized based on labels:

- `presentation` → 🎨 Presentation
- `application` → ⚙️ Application
- `infrastructure` → 🏗️ Infrastructure
- `documentation` → 📚 Documentation
- `ci/cd` → 🚀 CI/CD
- `enhancement` → ✨ Enhancement
- `bug` → 🐛 Bug
- `feature` → 🚀 Feature

## Usage

### Automatic Operation

The system works automatically. No manual intervention required for normal operation.

### Manual Sync

To manually trigger a sync:

```bash
# Set your GitHub token
export GITHUB_TOKEN="your_token_here"

# Run manual sync
cd /path/to/pynomaly
python scripts/automation/manual_sync.py
```

### Manual Workflow Trigger

You can also trigger the workflow manually from GitHub:

1. Go to Actions tab in GitHub
2. Select "GitHub Issues Sync to TODO.md" workflow
3. Click "Run workflow"

## Configuration

### Environment Variables

- `GITHUB_TOKEN`: Required for API access
- `GITHUB_REPOSITORY`: Repository in format "owner/repo" (auto-set in Actions)

### Customization

To modify the sync behavior, edit `scripts/automation/sync_github_issues_to_todo.py`:

- **Priority mapping**: Update `get_priority_from_labels()` method
- **Status detection**: Update `get_status_from_issue()` method  
- **Category mapping**: Update `get_category_from_labels()` method
- **Formatting**: Update `format_issue_for_todo()` method

## Output Format

The automation generates a structured Issues section with:

```markdown
## 📋 **GitHub Issues** (Auto-Synchronized)

**Total Open Issues**: X  
**Completed**: Y  
**In Progress**: Z  
**Pending**: W

**Last Sync**: Date and time

### 🔥 **P1-High Priority Issues**
#### **Issue #123: Title**
**Labels**: label1, label2  
**Priority**: 🔥 P1-High  
**Status**: ⏳ PENDING  
**Category**: 🎨 Presentation  
**Created**: Date  
**Updated**: Date

- **Scope**: Issue description preview...
- **GitHub**: [View Issue](link)
```

## Troubleshooting

### Common Issues

1. **Sync not triggering**: Check GitHub Actions logs
2. **Permission errors**: Ensure GITHUB_TOKEN has proper permissions
3. **Formatting issues**: Check TODO.md syntax and section markers

### Manual Recovery

If automation fails, you can:

1. Run manual sync script
2. Manually trigger workflow from GitHub Actions
3. Check and fix TODO.md format issues

### Logs and Monitoring

- GitHub Actions provides execution logs
- Script outputs status messages for debugging
- Failed runs will show in Actions tab

## Permissions Required

The GitHub token needs these permissions:

- `contents: write` (to update TODO.md)
- `issues: read` (to fetch issue data)
- `pull-requests: read` (to distinguish PRs from issues)

## Security Considerations

- GitHub token is stored as repository secret
- Script only reads public issue data
- Commits are made by GitHub Actions bot account
- No sensitive data is exposed in logs

## Integration with Development Workflow

### Best Practices

1. **Use consistent labels**: Apply P1-High, P2-Medium, P3-Low labels
2. **Update issue status**: Use "in-progress" and "blocked" labels appropriately
3. **Close completed issues**: Mark issues as closed when work is done
4. **Review sync results**: Check that TODO.md updates are accurate

### Label Standards

For optimal automation results, use these labels:

- **Priority**: P1-High, P2-Medium, P3-Low
- **Status**: in-progress, blocked
- **Category**: presentation, application, infrastructure, documentation, ci/cd
- **Type**: enhancement, bug, feature

## Future Enhancements

Potential improvements to the automation system:

- Milestone integration
- Assignee tracking
- Due date support
- Integration with project boards
- Slack/Discord notifications
- Progress tracking metrics

## Maintenance

### Regular Tasks

- Monitor GitHub Actions for failures
- Review label consistency across issues
- Update automation scripts as needed
- Validate TODO.md formatting remains correct

### Updates

When updating the automation:

1. Test changes with manual sync script
2. Review output format carefully
3. Ensure backward compatibility
4. Update documentation

---

**Last Updated**: July 11, 2025  
**Automation Version**: 1.0  
**Status**: ✅ Active and functional
