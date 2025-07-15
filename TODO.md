# Issue Management

## ⚠️ No TODOs or Issue Tracking in This File

This file is **NOT** for tracking issues or TODO items. All issue management must be done through GitHub Issues using the GitHub CLI.

## 📋 How to View and Manage Issues

### Install GitHub CLI
```bash
# Install gh CLI (if not already installed)
# macOS
brew install gh

# Windows
winget install GitHub.cli

# Linux
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh
```

### Authenticate with GitHub
```bash
gh auth login
```

### View All Issues
```bash
# List all open issues
gh issue list

# List all issues (including closed)
gh issue list --state=all

# List issues by priority labels
gh issue list --label="P1-High"
gh issue list --label="P2-Medium"
gh issue list --label="P3-Low"
```

### View Issue Details
```bash
# View specific issue
gh issue view 123

# View issue in browser
gh issue view 123 --web
```

### Create New Issues
```bash
# Create new issue interactively
gh issue create

# Create issue with title and body
gh issue create --title "Bug: Fix authentication error" --body "Description of the issue"
```

### Manage Issue Status
```bash
# Close an issue
gh issue close 123

# Reopen an issue
gh issue reopen 123

# Add labels to issue
gh issue edit 123 --add-label "bug,P1-High"

# Remove labels from issue
gh issue edit 123 --remove-label "P2-Medium"
```

### Search and Filter Issues
```bash
# Search issues by text
gh issue list --search "authentication"

# Filter by assignee
gh issue list --assignee="@me"

# Filter by milestone
gh issue list --milestone="v1.0"

# Filter by state
gh issue list --state=closed
```

### Advanced Issue Management
```bash
# List issues with JSON output for scripting
gh issue list --json number,title,state,labels

# Create issue from template
gh issue create --template=bug_report.md

# Edit issue
gh issue edit 123 --title "New title" --body "Updated description"

# Comment on issue
gh issue comment 123 --body "Adding a comment"
```

### Project Management
```bash
# List issues in project board format
gh issue list --json number,title,state,labels | jq '.[] | {number, title, state}'

# View issue activity
gh issue view 123 --comments

# List issues by priority
gh issue list --label="P0-Critical"
gh issue list --label="P1-High"
gh issue list --label="P2-Medium"
gh issue list --label="P3-Low"
```

## 🔒 Repository Rule

**This file serves only as a reference for GitHub CLI usage. It must not contain:**
- Issue tracking content
- TODO lists
- Project status updates
- Issue summaries or duplicated GitHub data

**All project management must be done through GitHub Issues directly.**

## 📚 Additional Resources

- [GitHub CLI Documentation](https://cli.github.com/manual/)
- [GitHub Issues Guide](https://docs.github.com/en/issues)
- [GitHub Project Management](https://docs.github.com/en/issues/organizing-your-work-with-project-boards)

---

**Last Updated**: July 15, 2025  
**Purpose**: GitHub CLI reference only - No issue tracking permitted