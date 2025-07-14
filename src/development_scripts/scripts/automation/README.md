# GitHub Issues to TODO.md Automation

This directory contains the automation system that keeps the `TODO.md` file synchronized with GitHub issues.

## 📁 Files

- **`sync_github_issues_to_todo.py`** - Main synchronization script
- **`test_sync.py`** - Validation and testing script  
- **`github_todo_sync.py`** - Legacy sync script (deprecated)
- **`manual_sync.py`** - Manual synchronization utilities

## 🔧 Setup

The automation is already configured and active via:

- **GitHub Actions Workflow**: `.github/workflows/issue-sync.yml`
- **Triggers**: Automatic on issue changes, manual dispatch available
- **Dependencies**: Python 3.11+, `requests`, `python-dateutil`

## 🚀 Usage

### Automatic Synchronization

The system automatically syncs when:

- Issues are opened, edited, closed, or reopened
- Issue labels are added or removed
- Issue comments are created, edited, or deleted

### Manual Synchronization

```bash
# Trigger via GitHub CLI
gh workflow run "GitHub Issues Sync to TODO.md"

# Run locally (requires GITHUB_TOKEN)
export GITHUB_TOKEN="your_token"
export GITHUB_REPOSITORY="elgerytme/Pynomaly"
python scripts/automation/sync_github_issues_to_todo.py

# Test the setup
python scripts/automation/test_sync.py
```

## 📋 Features

### Priority Mapping

- **P0-Critical** → 🚨 (immediate attention)
- **P1-High** → 🔥 (1-2 weeks)
- **P2-Medium** → 🔶 (2-4 weeks)  
- **P3-Low** → 🟢 (1-2 months)

### Status Detection

- **Closed issues** → ✅ COMPLETED
- **"In-Progress" label** → 🔄 IN PROGRESS
- **"Blocked" label** → 🚫 BLOCKED
- **Default** → ⏳ PENDING

### Category Classification

Based on issue labels:

- 🎨 Presentation
- ⚙️ Application
- 🏗️ Infrastructure
- 📚 Documentation
- 🚀 CI/CD
- ✨ Enhancement
- 🐛 Bug

## 🔄 Workflow

1. **Issue Event** → GitHub webhook triggers workflow
2. **Fetch Issues** → Script retrieves all open issues via GitHub API
3. **Process & Format** → Issues grouped by priority with consistent formatting
4. **Update TODO.md** → Replace Issues section with current data
5. **Commit Changes** → Auto-commit with standardized message

## 🎯 Output Format

The generated section includes:

```markdown
## 📋 **GitHub Issues** (Auto-Synchronized)

### **Overview**
- **Total Open Issues**: X
- **P0-Critical**: X issues (requires immediate attention)
- **P1-High**: X issues (fix within 1-2 weeks)
- **P2-Medium**: X issues (fix within 2-4 weeks)
- **P3-Low**: X issues (fix within 1-2 months)

### 🚨 **P0-Critical Priority Issues**
[Generated issue list...]
```

## ⚡ Performance

- **Sync Time**: < 30 seconds for 100+ issues
- **Update Frequency**: Within 2-3 minutes of issue changes
- **Rate Limits**: Respects GitHub API limits (5000/hour)

## 🔒 Security

- Uses GitHub Actions `GITHUB_TOKEN` (automatic)
- No secrets stored in code
- Read-only access to issues
- Write access only to `TODO.md`

## 🧪 Testing

```bash
# Validate setup
python scripts/automation/test_sync.py

# Check workflow status  
gh run list --workflow="GitHub Issues Sync to TODO.md"

# View latest run logs
gh run view --log
```

## 🚨 Troubleshooting

### Common Issues

1. **Workflow not triggering**
   - Check `.github/workflows/issue-sync.yml` exists
   - Verify GitHub Actions are enabled
   - Check webhook permissions

2. **Script fails**
   - Verify `GITHUB_TOKEN` has issues:read permission
   - Check Python dependencies installed
   - Validate repository format (`owner/repo`)

3. **TODO.md not updating**
   - Check Git permissions in workflow
   - Verify TODO.md exists and is writable
   - Review workflow run logs

### Debug Commands

```bash
# Check workflow exists
gh workflow list | grep -i sync

# View recent runs  
gh run list --workflow="GitHub Issues Sync to TODO.md" --limit 5

# Manual test
python scripts/automation/test_sync.py
```

## 📊 Monitoring

Track automation health via:

- GitHub Actions workflow runs
- Commit history on `TODO.md`
- Issue synchronization timestamps
- Workflow run success/failure rates

---

**Last Updated**: July 11, 2025  
**Status**: ✅ Active and operational  
**Maintainer**: Automated CI/CD system
