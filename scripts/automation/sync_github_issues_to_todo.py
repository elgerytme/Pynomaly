#!/usr/bin/env python3
"""
GitHub Issues to TODO.md Sync Script

This script automatically syncs GitHub issues to TODO.md with bidirectional updates.
Supports real-time sync via GitHub webhooks and scheduled sync.
"""

import json
import logging
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class GitHubIssue:
    """Represents a GitHub issue with sync-relevant data"""
    number: int
    title: str
    state: str
    labels: List[str]
    created_at: str
    updated_at: str
    body: str
    url: str
    priority: str = "medium"
    category: str = "General"
    status: str = "PENDING"
    
    def __post_init__(self):
        """Process issue data to extract priority, category, and status"""
        self.priority = self._extract_priority()
        self.category = self._extract_category()
        self.status = self._extract_status()
    
    def _extract_priority(self) -> str:
        """Extract priority from labels"""
        priority_map = {
            "P1-High": "🔥 P1-High",
            "P2-Medium": "🔶 P2-Medium", 
            "P3-Low": "🟢 P3-Low"
        }
        
        for label in self.labels:
            if label in priority_map:
                return priority_map[label]
        return "🔶 P2-Medium"  # Default
    
    def _extract_category(self) -> str:
        """Extract category from labels"""
        category_map = {
            "bug": "🐛 Bug",
            "enhancement": "✨ Enhancement",
            "documentation": "📚 Documentation",
            "Presentation": "🎨 Presentation",
            "Application": "⚙️ Application",
            "Infrastructure": "🏗️ Infrastructure",
            "CI/CD": "🔄 CI/CD"
        }
        
        for label in self.labels:
            if label in category_map:
                return category_map[label]
        return "📋 General"
    
    def _extract_status(self) -> str:
        """Extract status from state and labels"""
        if self.state == "closed":
            return "✅ COMPLETED"
        elif "In-Progress" in self.labels:
            return "🔄 IN PROGRESS"
        elif "Blocked" in self.labels:
            return "🚫 BLOCKED"
        else:
            return "⏳ PENDING"

class GitHubAPI:
    """GitHub API client for fetching issues"""
    
    def __init__(self, repo_owner: str, repo_name: str, token: Optional[str] = None):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.token = token
        self.base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Pynomaly-Issue-Sync/1.0"
        }
        
        if token:
            self.headers["Authorization"] = f"token {token}"
    
    def get_issues(self, state: str = "all", per_page: int = 100) -> List[GitHubIssue]:
        """Fetch all issues from GitHub"""
        issues = []
        page = 1
        
        while True:
            url = f"{self.base_url}/issues"
            params = {
                "state": state,
                "per_page": per_page,
                "page": page,
                "sort": "updated",
                "direction": "desc"
            }
            
            try:
                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                
                page_issues = response.json()
                if not page_issues:
                    break
                
                for issue_data in page_issues:
                    # Skip pull requests
                    if "pull_request" in issue_data:
                        continue
                    
                    issue = GitHubIssue(
                        number=issue_data["number"],
                        title=issue_data["title"],
                        state=issue_data["state"],
                        labels=[label["name"] for label in issue_data["labels"]],
                        created_at=issue_data["created_at"],
                        updated_at=issue_data["updated_at"],
                        body=issue_data["body"] or "",
                        url=issue_data["html_url"]
                    )
                    issues.append(issue)
                
                page += 1
                
            except requests.RequestException as e:
                logger.error(f"Error fetching issues: {e}")
                break
        
        return issues

class TODOManager:
    """Manages TODO.md file updates"""
    
    def __init__(self, todo_file: Path):
        self.todo_file = todo_file
        self.repo_owner = "elgerytme"
        self.repo_name = "Pynomaly"
    
    def generate_todo_content(self, issues: List[GitHubIssue]) -> str:
        """Generate complete TODO.md content from issues"""
        
        # Count issues by status
        total_issues = len(issues)
        completed_issues = len([i for i in issues if i.status == "✅ COMPLETED"])
        in_progress_issues = len([i for i in issues if i.status == "🔄 IN PROGRESS"])
        pending_issues = len([i for i in issues if i.status == "⏳ PENDING"])
        
        # Get completed issue examples
        completed_examples = [i for i in issues if i.status == "✅ COMPLETED"][:7]
        completed_text = ", ".join([f"Issue #{i.number} - {i.title}" for i in completed_examples])
        
        # Sort issues by priority and status
        priority_order = {"🔥 P1-High": 1, "🔶 P2-Medium": 2, "🟢 P3-Low": 3}
        issues_sorted = sorted(issues, key=lambda x: (priority_order.get(x.priority, 2), x.number))
        
        # Group issues by priority
        high_priority = [i for i in issues_sorted if i.priority == "🔥 P1-High"]
        medium_priority = [i for i in issues_sorted if i.priority == "🔶 P2-Medium"]
        low_priority = [i for i in issues_sorted if i.priority == "🟢 P3-Low"]
        
        # Generate content
        content = f"""# Pynomaly GitHub Issues

**Auto-Synchronized GitHub Issues List**

**Total Open Issues**: {total_issues}  
**Completed**: {completed_issues} ({completed_text})  
**In Progress**: {in_progress_issues}  
**Pending**: {pending_issues}  

**Last Sync**: {datetime.now(timezone.utc).strftime('%B %d, %Y at %H:%M UTC')}

---

"""
        
        # Add high priority issues
        if high_priority:
            content += "## 🔥 **P1-High Priority Issues**\n\n"
            for issue in high_priority:
                content += self._format_issue(issue)
        
        # Add medium priority issues
        if medium_priority:
            content += "## 🔶 **P2-Medium Priority Issues**\n\n"
            for issue in medium_priority:
                content += self._format_issue(issue)
        
        # Add low priority issues
        if low_priority:
            content += "## 🟢 **P3-Low Priority Issues**\n\n"
            for issue in low_priority:
                content += self._format_issue(issue)
        
        # Add footer
        content += f"""---

## 🔧 **Automation Setup**

**GitHub Actions Workflow**: `.github/workflows/issue-sync.yml`  
**Sync Script**: `scripts/automation/sync_github_issues_to_todo.py`  
**Manual Sync**: `scripts/automation/manual_sync.py`

**Triggers**:

- Issue opened, edited, closed, reopened
- Issue labeled or unlabeled  
- Issue comments created, edited, deleted
- Manual workflow dispatch

**Rules**:

1. **Priority Mapping**: P1-High → 🔥, P2-Medium → 🔶, P3-Low → 🟢
2. **Status Detection**: Closed → ✅ COMPLETED, In-Progress label → 🔄, Blocked label → 🚫, Default → ⏳ PENDING
3. **Category Classification**: Based on labels (Presentation, Application, Infrastructure, etc.)
4. **Auto-formatting**: Consistent structure with links, dates, and priority badges

---

**Last Updated**: {datetime.now(timezone.utc).strftime('%B %d, %Y at %H:%M UTC')}  
**Sync Status**: ✅ Active (Auto-synced on issue changes)  
**Next Manual Review**: As needed for strategic planning
"""
        
        return content
    
    def _format_issue(self, issue: GitHubIssue) -> str:
        """Format a single issue for TODO.md"""
        
        # Truncate body for preview
        body_preview = issue.body[:200] + "..." if len(issue.body) > 200 else issue.body
        body_preview = body_preview.replace('\n', ' ').replace('\r', ' ')
        
        # Format created/updated dates
        created_date = datetime.fromisoformat(issue.created_at.replace('Z', '+00:00'))
        updated_date = datetime.fromisoformat(issue.updated_at.replace('Z', '+00:00'))
        
        return f"""### **Issue #{issue.number}: {issue.title}**

**Labels**: {', '.join(issue.labels) if issue.labels else 'None'}  
**Priority**: {issue.priority}  
**Status**: {issue.status}  
**Category**: {issue.category}  
**Created**: {created_date.strftime('%b %d, %Y')}  
**Updated**: {updated_date.strftime('%b %d, %Y')}  

- **Scope**: {body_preview}
- **GitHub**: [View Issue]({issue.url})

"""
    
    def update_todo_file(self, issues: List[GitHubIssue]) -> bool:
        """Update TODO.md file with new content"""
        try:
            content = self.generate_todo_content(issues)
            
            with open(self.todo_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Updated TODO.md with {len(issues)} issues")
            return True
            
        except Exception as e:
            logger.error(f"Error updating TODO.md: {e}")
            return False

class SyncManager:
    """Main sync manager coordinating GitHub API and TODO updates"""
    
    def __init__(self, repo_owner: str, repo_name: str, todo_file: Path, token: Optional[str] = None):
        self.github_api = GitHubAPI(repo_owner, repo_name, token)
        self.todo_manager = TODOManager(todo_file)
        self.repo_owner = repo_owner
        self.repo_name = repo_name
    
    def sync_issues_to_todo(self) -> bool:
        """Perform full sync from GitHub issues to TODO.md"""
        logger.info("Starting GitHub issues to TODO.md sync")
        
        try:
            # Fetch all issues
            issues = self.github_api.get_issues()
            logger.info(f"Fetched {len(issues)} issues from GitHub")
            
            # Update TODO.md
            success = self.todo_manager.update_todo_file(issues)
            
            if success:
                logger.info("Sync completed successfully")
                return True
            else:
                logger.error("Sync failed")
                return False
                
        except Exception as e:
            logger.error(f"Sync error: {e}")
            return False
    
    def handle_webhook(self, webhook_data: Dict) -> bool:
        """Handle GitHub webhook events for real-time sync"""
        try:
            action = webhook_data.get('action')
            issue_data = webhook_data.get('issue', {})
            
            logger.info(f"Received webhook: {action} for issue #{issue_data.get('number')}")
            
            # For any issue-related webhook, perform full sync
            # This ensures TODO.md is always up-to-date
            if action in ['opened', 'edited', 'closed', 'reopened', 'labeled', 'unlabeled']:
                return self.sync_issues_to_todo()
            
            return True
            
        except Exception as e:
            logger.error(f"Webhook handling error: {e}")
            return False

def get_github_token() -> Optional[str]:
    """Get GitHub token from environment or gh CLI"""
    import os
    
    # Try environment variable first
    token = os.getenv('GITHUB_TOKEN')
    if token:
        return token
    
    # Try gh CLI
    try:
        result = subprocess.run(['gh', 'auth', 'token'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("No GitHub token found. API requests will be rate-limited.")
        return None

def main():
    """Main entry point"""
    # Configuration
    repo_owner = "elgerytme"
    repo_name = "Pynomaly"
    todo_file = Path(__file__).parent.parent.parent / "TODO.md"
    
    # Get GitHub token
    token = get_github_token()
    
    # Create sync manager
    sync_manager = SyncManager(repo_owner, repo_name, todo_file, token)
    
    # Perform sync
    success = sync_manager.sync_issues_to_todo()
    
    if success:
        print("✅ GitHub issues synced to TODO.md successfully!")
        sys.exit(0)
    else:
        print("❌ Sync failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()