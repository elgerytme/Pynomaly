# Branch & Stash Cleanup Automation

This document describes the automated branch and stash cleanup system that helps maintain repository hygiene by:

- 🍂 **Flagging branches >30 days stale** - Automatically detects and reports branches that haven't been updated
- 🗂️ **Failing if new stashes exist on `main`** - Prevents stash accumulation on the main branch
- 🔒 **Enforcing protected `main` with required checks** - Maintains branch protection rules

## 📋 Overview

The system consists of:

1. **GitHub Actions Workflow** (`.github/workflows/branch-stash-cleanup.yml`)
2. **Manual Cleanup Script** (`scripts/maintenance/branch_cleanup.py`)
3. **Automated Issue Creation** for tracking cleanup tasks

## 🔄 Automated Workflow

### Schedule
- **Runs weekly** on Mondays at 2:00 AM UTC
- **Can be triggered manually** via GitHub Actions UI

### Features

#### 1. Stash Detection
- Checks for stashes on the `main` branch
- **Fails the workflow** if stashes are found
- Provides clear instructions for cleanup

#### 2. Branch Staleness Analysis
- Analyzes all remote branches (excluding `main`)
- Identifies branches with no commits for >30 days
- Creates detailed reports with branch information

#### 3. Automated Issue Creation
- Creates GitHub issues for stale branch cleanup
- Updates existing issues instead of creating duplicates
- Includes actionable cleanup commands

#### 4. Branch Protection Enforcement
- Ensures `main` branch has proper protection rules
- Requires status checks before merging
- Enforces pull request reviews

## 🛠️ Manual Cleanup Tools

### Branch Cleanup Script

The `scripts/maintenance/branch_cleanup.py` script provides manual tools for branch management:

#### Analyze Branches
```bash
# Analyze all branches for staleness
python scripts/maintenance/branch_cleanup.py analyze

# Show all branches, not just stale ones
python scripts/maintenance/branch_cleanup.py analyze --show-all

# Save analysis to file
python scripts/maintenance/branch_cleanup.py analyze --output reports/branch-analysis.json

# Analyze local branches instead of remote
python scripts/maintenance/branch_cleanup.py analyze --local
```

#### Clean Up Stale Branches
```bash
# Interactive cleanup of stale branches
python scripts/maintenance/branch_cleanup.py cleanup

# Dry run (show what would be deleted)
python scripts/maintenance/branch_cleanup.py cleanup --dry-run

# Only delete merged branches
python scripts/maintenance/branch_cleanup.py cleanup --merged-only

# Force cleanup without confirmation
python scripts/maintenance/branch_cleanup.py cleanup --force
```

#### Check for Stashes
```bash
# Check for stashes on main branch
python scripts/maintenance/branch_cleanup.py stash-check

# Check for stashes on specific branch
python scripts/maintenance/branch_cleanup.py stash-check --branch feature/my-feature

# Automatically clear stashes
python scripts/maintenance/branch_cleanup.py stash-check --fix
```

#### Generate Reports
```bash
# Generate comprehensive branch analysis report
python scripts/maintenance/branch_cleanup.py report

# Custom output location
python scripts/maintenance/branch_cleanup.py report --output my-report.json

# Custom stale threshold
python scripts/maintenance/branch_cleanup.py report --stale-days 14
```

## 📊 Workflow Jobs

### 1. `stash-check`
- **Purpose**: Verify no stashes exist on `main` branch
- **Failure**: Stops workflow if stashes are found
- **Output**: Clear instructions for stash cleanup

### 2. `branch-analysis` 
- **Purpose**: Analyze all branches for staleness
- **Output**: JSON report of stale branches
- **Artifacts**: Analysis reports uploaded for 90 days

### 3. `create-cleanup-issue`
- **Purpose**: Create/update GitHub issues for stale branches
- **Trigger**: Only runs if stale branches are found
- **Features**: Updates existing issues instead of creating duplicates

### 4. `enforce-branch-protection`
- **Purpose**: Ensure `main` branch protection rules are active
- **Rules**: 
  - Require status checks
  - Require pull request reviews
  - Prevent force pushes
  - Prevent deletions

### 5. `summary`
- **Purpose**: Provide comprehensive workflow summary
- **Output**: GitHub Actions summary with recommendations
- **Exit**: Fails if stashes found on `main`

## 🔧 Configuration

### Environment Variables
```yaml
env:
  STALE_DAYS: 30          # Days after which branch is considered stale
  MAIN_BRANCH: main       # Main branch name
```

### Branch Protection Rules
The workflow enforces these protection rules on `main`:

- **Required Status Checks**:
  - `CI / test`
  - `CI / lint`
  - `CI / security-scan`
  - `Quality Gates / quality-check`
  - `Validation Suite / comprehensive-validation`

- **Pull Request Requirements**:
  - 1 approving review required
  - Dismiss stale reviews
  - Require code owner reviews

- **Restrictions**:
  - No force pushes
  - No deletions
  - Require conversation resolution

## 📈 Reports and Artifacts

### Branch Analysis Report
Located at `reports/branch-analysis/stale-branches.json`:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "stale_days_threshold": 30,
  "repository_health": {
    "total_branches": 15,
    "stale_branches": 3,
    "merged_branches": 2,
    "active_branches": 12,
    "has_stashes_on_main": false
  },
  "stale_branches": [
    {
      "name": "feature/old-feature",
      "last_commit": "2023-12-01T12:00:00Z",
      "days_stale": 45,
      "is_merged": false
    }
  ]
}
```

### Workflow Artifacts
- **branch-analysis-report**: Detailed analysis results (90 days retention)
- **Maintenance Summary**: Overall workflow results (90 days retention)

## 🚨 Troubleshooting

### Common Issues

#### 1. Stashes Found on Main
**Error**: `New stashes found on main branch`
**Solution**:
```bash
# List all stashes
git stash list

# Clear all stashes
git stash clear

# Or drop specific stash
git stash drop stash@{0}
```

#### 2. Branch Protection Update Failed
**Error**: `Error updating branch protection`
**Cause**: Insufficient permissions or repository settings
**Solution**: Ensure the GitHub token has admin permissions

#### 3. No Stale Branches Found
**Info**: This is good! Your repository has healthy branch hygiene.

#### 4. Failed to Delete Branch
**Error**: `Failed to delete remote branch`
**Causes**:
- Branch has recent commits
- Branch is protected
- Insufficient permissions
**Solution**: Review branch manually or adjust permissions

## 🎯 Best Practices

### For Developers
1. **Delete branches after merging**: Don't leave merged branches hanging
2. **Avoid stashing on main**: Use feature branches for development
3. **Regular cleanup**: Review and clean up your own branches periodically
4. **Use descriptive branch names**: Makes cleanup decisions easier

### For Maintainers
1. **Review cleanup issues**: Address stale branch issues promptly
2. **Monitor workflow runs**: Check for patterns in branch staleness
3. **Adjust stale threshold**: Modify `STALE_DAYS` if needed for your workflow
4. **Enforce branch protection**: Ensure the protection rules match your requirements

### For Repository Admins
1. **Configure notifications**: Set up alerts for workflow failures
2. **Review permissions**: Ensure the workflow has necessary permissions
3. **Customize protection rules**: Adjust status checks to match your CI/CD pipeline
4. **Monitor repository health**: Use the reports to track improvements

## 🔄 Integration with Existing Workflows

### Maintenance Workflow
The branch & stash cleanup works alongside the existing maintenance workflow:

```yaml
# maintenance.yml runs repository cleanup
# branch-stash-cleanup.yml runs branch hygiene checks
```

### CI/CD Pipeline
The branch protection rules integrate with:
- **CI workflow**: `CI / test`, `CI / lint`, `CI / security-scan`
- **Quality Gates**: `Quality Gates / quality-check`
- **Validation Suite**: `Validation Suite / comprehensive-validation`

## 📅 Schedule and Timing

### Weekly Schedule
- **Monday 2:00 AM UTC**: Branch & Stash Cleanup
- **Monday 3:00 AM UTC**: Regular Maintenance

### Manual Triggers
- Use GitHub Actions UI to run manually
- Useful for testing or immediate cleanup needs

## 🔍 Monitoring and Alerts

### Success Indicators
- ✅ No stashes on main branch
- ✅ Minimal stale branches (<5)
- ✅ Branch protection rules active
- ✅ Regular cleanup issue resolution

### Warning Signs
- ⚠️ Increasing number of stale branches
- ⚠️ Stashes accumulating on main
- ⚠️ Branch protection rules failing to apply
- ⚠️ Cleanup issues not being addressed

## 📚 Additional Resources

- [GitHub Branch Protection Documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches)
- [Git Stash Documentation](https://git-scm.com/docs/git-stash)
- [GitHub Actions Scheduling](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#schedule)

---

*This automation system helps maintain a clean and organized repository by enforcing good Git hygiene practices automatically.*
