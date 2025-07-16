# GitHub Issues - Todo List Synchronization Rules

## Overview

This document defines the automation rules and procedures for maintaining synchronization between internal todo lists and GitHub issues. This ensures consistency between project management tools and development workflow.

## 📋 **Core Synchronization Rules**

### 1. **Automatic Synchronization Triggers**
- **On Issue Creation**: Automatically add P1-High and P2-Medium issues to todo list
- **On Label Changes**: Update todo status when issue labels change (in-progress, completed)
- **On Issue Closure**: Mark corresponding todo as completed
- **Daily Sync**: Run synchronization script daily at 9 AM UTC
- **Session Start**: Auto-sync when Claude Code sessions begin

### 2. **Priority Mapping Rules**
| GitHub Label | Todo Priority | Inclusion Rule |
|--------------|---------------|----------------|
| `P1-High` | `high` | Always included in active todos |
| `P2-Medium` | `medium` | Included if in-progress or recently created |
| `P3-Low` | `low` | Excluded unless marked in-progress |

### 3. **Status Mapping Rules**
| GitHub State/Label | Todo Status | Description |
|-------------------|-------------|-------------|
| `closed` | `completed` | Issue is finished and verified |
| `In-Progress` label | `in_progress` | Currently being worked on |
| `open` (no in-progress) | `pending` | Ready to work but not started |

### 4. **Content Formatting Rules**
- **Todo Content Format**: `{Issue Title} (#{Issue Number})`
- **GitHub Reference**: Always include issue number for traceability
- **Priority Indication**: Use priority field to match GitHub labels
- **Length Limit**: Truncate titles to 80 characters with "..." if needed

### 5. **Filtering and Inclusion Rules**
- **Include**: Issues with P1-High, P2-Medium priority labels
- **Include**: All in-progress issues regardless of priority
- **Exclude**: Closed issues (mark as completed then archive after 7 days)
- **Exclude**: P3-Low issues unless specifically marked in-progress
- **Limit**: Maximum 15 active todos to maintain focus

### 6. **Update Frequency and Triggers**
- **Real-time**: On GitHub webhook events (issue updates, label changes)
- **Scheduled**: Daily synchronization at 9 AM UTC via GitHub Actions
- **On-demand**: Manual sync via CLI command
- **Before Sessions**: Auto-sync when development sessions start

## 🔧 **Implementation Components**

### Synchronization Script
**Location**: `.github/scripts/sync-todos-with-issues.py`

**Features**:
- Fetches GitHub issues using `gh` CLI
- Maps priorities and statuses according to rules
- Formats todos for Claude's TodoWrite tool
- Supports dry-run mode for testing
- Comprehensive logging and error handling

**Usage**:
```bash
# Normal sync
python .github/scripts/sync-todos-with-issues.py --verbose

# Preview changes without applying
python .github/scripts/sync-todos-with-issues.py --dry-run --verbose

# Emergency reset from GitHub
python .github/scripts/sync-todos-with-issues.py --reset-from-github
```

### GitHub Actions Workflow
**Location**: `.github/workflows/sync-todos.yml`

**Triggers**:
- Issue events: opened, closed, labeled, unlabeled
- Daily schedule: 9 AM UTC
- Manual dispatch: workflow_dispatch

**Features**:
- Automated sync on issue changes
- Validation of sync rules compliance
- Error reporting and alerts
- Artifact preservation for troubleshooting

## 🚨 **Conflict Resolution Rules**

### Priority Conflicts
1. **GitHub is Source of Truth**: GitHub issue labels override todo priorities
2. **Manual Todo Changes**: Preserved until next GitHub sync event
3. **Multiple Labels**: Highest priority label takes precedence
4. **Missing Labels**: Default to medium priority with warning

### Status Conflicts
1. **GitHub State Priority**: GitHub open/closed state overrides manual status
2. **Label vs State**: `In-Progress` label overrides open state for status
3. **Race Conditions**: GitHub timestamp determines most recent change
4. **Sync Failures**: Preserve last known good state until next successful sync

### Content Conflicts
1. **Title Changes**: GitHub issue title updates override todo content
2. **Manual Edits**: Manual todo content changes preserved until issue title changes
3. **Issue Number**: Always preserved and used for linking
4. **Character Limits**: Automatic truncation with preservation of issue number

## 📊 **Monitoring and Compliance**

### Automated Validation
- **Daily Reports**: Generate todo-issue alignment reports
- **Drift Detection**: Alert when todos diverge from GitHub issues >24 hours
- **Audit Trail**: Log all synchronization actions with timestamps
- **Performance Tracking**: Monitor sync execution time and success rate

### Compliance Metrics
- **Sync Success Rate**: Target >99% successful syncs
- **Drift Time**: Average time between GitHub change and todo update <1 hour
- **Coverage**: Percentage of priority issues represented in todos >95%
- **Accuracy**: Percentage of todo statuses matching GitHub state >98%

### Alert Conditions
- **Sync Failure**: 2+ consecutive failed syncs
- **High Drift**: >10 todos with outdated status from GitHub
- **Missing Issues**: P1-High issues not represented in todos
- **API Rate Limits**: GitHub API rate limit approaching

## 🛠 **Emergency Procedures**

### Sync Failure Recovery
1. **Immediate**: Check GitHub API status and rate limits
2. **Short-term**: Fall back to manual todo management
3. **Resolution**: Fix underlying issue (permissions, API limits, script bugs)
4. **Recovery**: Run full resync with `--reset-from-github` flag

### Data Corruption Recovery
1. **Source of Truth**: Always restore from GitHub as authoritative source
2. **Backup**: Preserve previous todo state for manual review
3. **Validation**: Run sync in dry-run mode to preview changes
4. **Rollback**: Maintain ability to revert to previous working state

### Service Outage Handling
1. **GitHub Down**: Queue sync operations for later execution
2. **CI/CD Failure**: Continue with manual sync processes
3. **Script Errors**: Implement exponential backoff and retry logic
4. **Communication**: Alert stakeholders of sync service interruption

## 📈 **Performance Optimization**

### Caching Strategy
- **API Responses**: Cache GitHub API responses for 5 minutes
- **Issue Metadata**: Cache issue labels and status for quick lookups
- **Batch Processing**: Process multiple issue updates in single operation
- **Incremental Sync**: Only sync changed issues when possible

### Resource Management
- **Rate Limiting**: Respect GitHub API rate limits (5000/hour)
- **Concurrent Requests**: Limit to 10 concurrent API requests
- **Memory Usage**: Process issues in batches of 50 to manage memory
- **Execution Time**: Target sync completion <30 seconds

### Scalability Considerations
- **Issue Volume**: Support up to 1000+ issues in repository
- **Update Frequency**: Handle up to 100 issue updates per day
- **Concurrent Users**: Support multiple development sessions
- **Geographic Distribution**: Consider API latency for global teams

## 🔄 **Maintenance Procedures**

### Regular Maintenance
- **Weekly**: Review sync logs for errors or performance degradation
- **Monthly**: Validate rule effectiveness and update if needed
- **Quarterly**: Performance review and optimization
- **Annually**: Complete rule review and process improvement

### Script Updates
- **Version Control**: All script changes tracked in git
- **Testing**: Comprehensive testing in dry-run mode before deployment
- **Rollback Plan**: Ability to revert to previous script version
- **Documentation**: Update rules documentation with script changes

### Rule Evolution
- **Feedback Loop**: Collect user feedback on sync effectiveness
- **Metrics Analysis**: Regular review of compliance and performance metrics
- **Process Improvement**: Iterate on rules based on usage patterns
- **Stakeholder Review**: Quarterly review with development team

## 🎯 **Success Criteria**

### Operational Excellence
- **Reliability**: 99.9% sync success rate
- **Performance**: <30 second sync execution time
- **Accuracy**: >98% todo-issue alignment
- **Availability**: <1 hour downtime per quarter

### User Experience
- **Transparency**: Clear visibility into sync status and changes
- **Predictability**: Consistent behavior across all sync scenarios
- **Flexibility**: Support for manual overrides when needed
- **Responsiveness**: Real-time updates for critical priority changes

### Business Value
- **Productivity**: Reduced manual effort in project management
- **Consistency**: Unified view of work across tools
- **Traceability**: Clear audit trail from todos to GitHub issues
- **Scalability**: Support for growing team and project size

---

## Quick Reference Commands

```bash
# Manual sync
python .github/scripts/sync-todos-with-issues.py --verbose

# Dry run
python .github/scripts/sync-todos-with-issues.py --dry-run

# Emergency reset
python .github/scripts/sync-todos-with-issues.py --reset-from-github

# Check sync status
gh workflow run sync-todos.yml

# View recent syncs
gh run list --workflow=sync-todos.yml --limit=10
```

---

*Last Updated: 2025-07-15*  
*Next Review: 2025-10-15*