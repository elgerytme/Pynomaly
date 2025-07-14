# GitHub Issues to TODO.md Auto-Sync System

This directory contains the automated synchronization system between GitHub Issues and TODO.md. The system provides real-time, bidirectional synchronization to keep your TODO.md file always up-to-date with your GitHub project management.

## 🌟 Features

- **Real-time Sync**: Automatically updates TODO.md when GitHub issues change
- **Bidirectional Updates**: Sync works both ways (planned for future enhancement)
- **Smart Prioritization**: Automatically maps GitHub labels to priority levels
- **Status Detection**: Intelligently determines issue status from labels and state
- **Category Classification**: Organizes issues by type (bug, enhancement, documentation, etc.)
- **Webhook Support**: Real-time updates via GitHub webhooks
- **Manual Sync**: On-demand synchronization capability
- **GitHub Actions Integration**: Automated workflow triggers

## 📁 Files

### Core Scripts
- `sync_github_issues_to_todo.py` - Main synchronization engine
- `manual_sync.py` - Manual sync with CLI options
- `webhook_server.py` - Webhook server for real-time updates

### Configuration
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

### GitHub Actions
- `../.github/workflows/issue-sync.yml` - Automated workflow

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd scripts/automation
pip install -r requirements.txt
```

### 2. Set Up GitHub Authentication
```bash
# Option 1: Use GitHub CLI (recommended)
gh auth login

# Option 2: Set environment variable
export GITHUB_TOKEN="your_token_here"
```

### 3. Run Manual Sync
```bash
python manual_sync.py
```

## 🔧 Configuration

### Repository Settings
The system is configured for the `elgerytme/Pynomaly` repository by default. To change this:

1. Edit the configuration variables in each script:
   ```python
   repo_owner = "your-username"
   repo_name = "your-repo-name"
   ```

2. Update the GitHub Actions workflow

### Priority Mapping
Issues are automatically prioritized based on GitHub labels:

- `P1-High` → 🔥 P1-High (Critical)
- `P2-Medium` → 🔶 P2-Medium (Important)
- `P3-Low` → 🟢 P3-Low (Nice to have)

### Status Detection
Issue status is determined by:

- **Closed** → ✅ COMPLETED
- **In-Progress label** → 🔄 IN PROGRESS
- **Blocked label** → 🚫 BLOCKED
- **Open (default)** → ⏳ PENDING

### Category Classification
Issues are categorized by labels:

- `bug` → 🐛 Bug
- `enhancement` → ✨ Enhancement
- `documentation` → 📚 Documentation
- `Presentation` → 🎨 Presentation
- `Application` → ⚙️ Application
- `Infrastructure` → 🏗️ Infrastructure
- `CI/CD` → 🔄 CI/CD

## 📚 Usage

### Manual Sync
```bash
# Basic sync
python manual_sync.py

# Dry run (preview changes)
python manual_sync.py --dry-run

# Verbose output
python manual_sync.py --verbose

# Custom repository
python manual_sync.py --repo-owner myuser --repo-name myrepo
```

### Webhook Server
```bash
# Start webhook server
python webhook_server.py

# Custom port
PORT=8080 python webhook_server.py

# With webhook secret
GITHUB_WEBHOOK_SECRET="your_secret" python webhook_server.py
```

### GitHub Actions
The system automatically triggers on:
- Issue opened, edited, closed, reopened
- Issue labeled or unlabeled
- Issue comments created, edited, deleted
- Manual workflow dispatch

## 🔄 Sync Process

### 1. Issue Collection
- Fetches all issues from GitHub API
- Handles pagination for large repositories
- Filters out pull requests

### 2. Data Processing
- Extracts priority from labels
- Determines status from state and labels
- Categorizes by issue type
- Truncates long descriptions

### 3. TODO.md Generation
- Groups issues by priority level
- Formats with consistent structure
- Includes links back to GitHub
- Adds sync metadata

### 4. File Update
- Atomic write to prevent corruption
- Preserves encoding and formatting
- Commits changes with descriptive message

## 📊 TODO.md Structure

The generated TODO.md follows this structure:

```markdown
# Pynomaly GitHub Issues

**Auto-Synchronized GitHub Issues List**

**Total Open Issues**: 50
**Completed**: 7 (examples...)
**In Progress**: 1
**Pending**: 48

**Last Sync**: July 14, 2025 at 21:30 UTC

---

## 🔥 **P1-High Priority Issues**

### **Issue #1: Critical Bug Fix**
**Labels**: bug, P1-High
**Priority**: 🔥 P1-High
**Status**: ⏳ PENDING
**Category**: 🐛 Bug
**Created**: Jul 08, 2025
**Updated**: Jul 11, 2025

- **Scope**: This is a critical bug that needs immediate attention...
- **GitHub**: [View Issue](https://github.com/owner/repo/issues/1)

...
```

## 🔐 Security

### Authentication
- Uses GitHub personal access tokens
- Supports both environment variables and GitHub CLI
- Requires minimal permissions (`repo` scope)

### Webhook Security
- Validates webhook signatures using HMAC
- Configurable webhook secret
- Logs all webhook events

### Data Privacy
- No sensitive data stored locally
- Issue content truncated in TODO.md
- Full data available via GitHub links

## 🐛 Troubleshooting

### Common Issues

**Authentication Error**
```bash
# Solution: Set up GitHub token
gh auth login
# OR
export GITHUB_TOKEN="your_token"
```

**Rate Limiting**
```bash
# Solution: Use authenticated requests
# The system automatically handles rate limiting
```

**Permission Denied**
```bash
# Solution: Ensure token has repo permissions
gh auth refresh -h github.com -s repo
```

**Webhook Not Working**
```bash
# Check webhook configuration in repository settings
# Verify webhook secret matches environment variable
# Check server logs for errors
```

### Debug Mode
```bash
# Enable verbose logging
python manual_sync.py --verbose

# Check webhook server logs
python webhook_server.py
```

## 🔮 Future Enhancements

### Planned Features
- **Bidirectional Sync**: Update GitHub issues from TODO.md changes
- **Custom Templates**: Configurable TODO.md formatting
- **Slack Integration**: Notifications for sync events
- **Dashboard**: Web interface for sync management
- **Conflict Resolution**: Handle simultaneous updates
- **Batch Operations**: Bulk issue updates

### Configuration Options
- Custom priority mappings
- Filtered sync (by labels, milestones)
- Multiple repository support
- Custom sync schedules

## 🤝 Contributing

To improve the sync system:

1. **Test Changes**: Use `--dry-run` flag for testing
2. **Update Documentation**: Keep README current
3. **Add Tests**: Include unit tests for new features
4. **Follow Conventions**: Maintain code style consistency

## 📞 Support

For issues with the sync system:

1. **Check Logs**: Review script output and GitHub Actions logs
2. **Verify Configuration**: Ensure tokens and permissions are correct
3. **Test Manually**: Use `manual_sync.py` for debugging
4. **Create Issue**: Report problems via GitHub Issues

## 📜 License

This sync system is part of the Pynomaly project and follows the same license terms.

---

**Last Updated**: July 14, 2025  
**Version**: 1.0.0  
**Maintainer**: Pynomaly Development Team