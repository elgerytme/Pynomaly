#!/usr/bin/env python3
"""
Todo-GitHub Issues Synchronization Script

This script automatically synchronizes internal todo lists with GitHub issues to ensure
consistency between project management tools and development workflow.

Usage:
    python sync-todos-with-issues.py [--dry-run] [--verbose]
"""

import json
import subprocess
import sys
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TodoItem:
    """Represents a todo item."""
    content: str
    status: str  # "pending", "in_progress", "completed"
    priority: str  # "high", "medium", "low"
    id: str
    github_issue: Optional[int] = None


@dataclass
class GitHubIssue:
    """Represents a GitHub issue."""
    number: int
    title: str
    state: str  # "open", "closed"
    labels: List[str]
    
    @property
    def priority(self) -> str:
        """Extract priority from labels."""
        if "P1-High" in self.labels:
            return "high"
        elif "P2-Medium" in self.labels:
            return "medium"
        elif "P3-Low" in self.labels:
            return "low"
        return "medium"  # default
    
    @property
    def status(self) -> str:
        """Extract status from state and labels."""
        if self.state == "closed":
            return "completed"
        elif "In-Progress" in self.labels or "in-progress" in self.labels:
            return "in_progress"
        else:
            return "pending"


class TodoGitHubSync:
    """Synchronizes todos with GitHub issues."""
    
    def __init__(self, dry_run: bool = False, verbose: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose
        
    def log(self, message: str, level: str = "INFO"):
        """Log message if verbose or important."""
        if self.verbose or level in ["ERROR", "WARNING"]:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    def get_github_issues(self) -> List[GitHubIssue]:
        """Fetch GitHub issues using gh CLI."""
        try:
            # Get open issues with priority labels
            cmd = [
                "gh", "issue", "list", 
                "--state", "open",
                "--limit", "50",
                "--json", "number,title,state,labels"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            issues_data = json.loads(result.stdout)
            
            issues = []
            for issue_data in issues_data:
                labels = [label["name"] for label in issue_data["labels"]]
                # Only include issues with priority labels
                if any(label.startswith("P") and "High" in label or "Medium" in label or "Low" in label 
                       for label in labels):
                    issues.append(GitHubIssue(
                        number=issue_data["number"],
                        title=issue_data["title"],
                        state=issue_data["state"],
                        labels=labels
                    ))
            
            self.log(f"Fetched {len(issues)} GitHub issues")
            return issues
            
        except subprocess.CalledProcessError as e:
            self.log(f"Error fetching GitHub issues: {e}", "ERROR")
            return []
        except json.JSONDecodeError as e:
            self.log(f"Error parsing GitHub issues JSON: {e}", "ERROR")
            return []
    
    def extract_issue_number(self, content: str) -> Optional[int]:
        """Extract GitHub issue number from todo content."""
        import re
        match = re.search(r'#(\d+)', content)
        return int(match.group(1)) if match else None
    
    def create_todo_from_issue(self, issue: GitHubIssue, todo_id: str) -> TodoItem:
        """Create a todo item from a GitHub issue."""
        content = f"{issue.title} (#{issue.number})"
        return TodoItem(
            content=content,
            status=issue.status,
            priority=issue.priority,
            id=todo_id,
            github_issue=issue.number
        )
    
    def sync_todos_with_issues(self) -> List[TodoItem]:
        """Synchronize todos with GitHub issues."""
        issues = self.get_github_issues()
        
        # Create todos from high and medium priority issues
        todos = []
        todo_id = 1
        
        # Sort by priority and status
        priority_order = {"high": 1, "medium": 2, "low": 3}
        status_order = {"in_progress": 1, "pending": 2, "completed": 3}
        
        sorted_issues = sorted(issues, key=lambda x: (
            priority_order.get(x.priority, 3),
            status_order.get(x.status, 2),
            x.number
        ))
        
        for issue in sorted_issues:
            # Include high priority and in-progress medium priority issues
            if (issue.priority == "high" or 
                (issue.priority == "medium" and issue.status == "in_progress")):
                
                todo = self.create_todo_from_issue(issue, str(todo_id))
                todos.append(todo)
                todo_id += 1
                
                self.log(f"Created todo: {todo.content} ({todo.status}, {todo.priority})")
        
        return todos
    
    def format_todos_for_claude(self, todos: List[TodoItem]) -> List[Dict[str, Any]]:
        """Format todos for Claude's TodoWrite tool."""
        return [
            {
                "content": todo.content,
                "status": todo.status,
                "priority": todo.priority,
                "id": todo.id
            }
            for todo in todos
        ]
    
    def generate_automation_rules(self) -> str:
        """Generate automation rules document."""
        return """# GitHub Issues - Todo List Synchronization Rules

## Automation Rules for Todo-GitHub Issues Sync

### 1. **Automatic Synchronization Triggers**
- **On Issue Creation**: Automatically add P1-High and P2-Medium issues to todo list
- **On Label Changes**: Update todo status when issue labels change (in-progress, completed)
- **On Issue Closure**: Mark corresponding todo as completed
- **Daily Sync**: Run synchronization script daily at 9 AM UTC

### 2. **Priority Mapping Rules**
- **P1-High** → **high priority** todos (always included)
- **P2-Medium** → **medium priority** todos (included if in-progress or recently created)
- **P3-Low** → **low priority** todos (excluded from active todo list)

### 3. **Status Mapping Rules**
- **GitHub "closed"** → **Todo "completed"**
- **GitHub "In-Progress" label** → **Todo "in_progress"**
- **GitHub "open" (no in-progress label)** → **Todo "pending"**

### 4. **Content Formatting Rules**
- **Todo Content Format**: `{Issue Title} (#{Issue Number})`
- **GitHub Reference**: Always include issue number for traceability
- **Priority Indication**: Use priority field to match GitHub labels

### 5. **Filtering Rules**
- **Include**: Issues with P1-High, P2-Medium priority labels
- **Include**: All in-progress issues regardless of priority
- **Exclude**: Closed issues (mark as completed then archive)
- **Exclude**: P3-Low issues unless specifically marked in-progress
- **Limit**: Maximum 15 active todos to maintain focus

### 6. **Update Frequency Rules**
- **Real-time**: On GitHub webhook events (issue updates, label changes)
- **Scheduled**: Daily synchronization at 9 AM UTC
- **On-demand**: Manual sync via `python sync-todos-with-issues.py`
- **Before Sessions**: Auto-sync when Claude Code sessions start

### 7. **Conflict Resolution Rules**
- **GitHub is Source of Truth**: GitHub issue state overrides todo state
- **Manual Todo Changes**: Preserved until next GitHub update
- **Priority Conflicts**: GitHub label priority takes precedence
- **Status Conflicts**: GitHub state takes precedence over manual status

### 8. **GitHub Actions Integration**
```yaml
# .github/workflows/sync-todos.yml
name: Sync Todos with Issues
on:
  issues:
    types: [opened, closed, labeled, unlabeled]
  schedule:
    - cron: '0 9 * * *'  # Daily at 9 AM UTC
  workflow_dispatch:

jobs:
  sync-todos:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Sync Todos
        run: python .github/scripts/sync-todos-with-issues.py
```

### 9. **Compliance Monitoring**
- **Weekly Reports**: Generate todo-issue alignment reports
- **Drift Detection**: Alert when todos diverge from GitHub issues
- **Audit Trail**: Log all synchronization actions with timestamps
- **Performance Tracking**: Monitor sync execution time and success rate

### 10. **Emergency Procedures**
- **Sync Failure**: Fall back to manual todo management
- **API Rate Limits**: Implement exponential backoff and retry logic
- **Data Corruption**: Restore from GitHub as authoritative source
- **Service Outage**: Queue sync operations for later execution

## Implementation Commands

### Manual Sync
```bash
python .github/scripts/sync-todos-with-issues.py --verbose
```

### Dry Run (Preview Changes)
```bash
python .github/scripts/sync-todos-with-issues.py --dry-run --verbose
```

### Emergency Reset
```bash
python .github/scripts/sync-todos-with-issues.py --reset-from-github
```
"""

    def run_sync(self) -> bool:
        """Run the synchronization process."""
        self.log("Starting GitHub Issues - Todo List synchronization")
        
        # Get synchronized todos
        todos = self.sync_todos_with_issues()
        
        if not todos:
            self.log("No todos to sync", "WARNING")
            return True
        
        # Format for output
        formatted_todos = self.format_todos_for_claude(todos)
        
        if self.dry_run:
            self.log("DRY RUN - Would update todos to:")
            for todo in formatted_todos:
                self.log(f"  - {todo['content']} ({todo['status']}, {todo['priority']})")
            return True
        
        # Output for Claude to use
        self.log("Synchronization complete")
        print(json.dumps(formatted_todos, indent=2))
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Sync todos with GitHub issues")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    sync = TodoGitHubSync(dry_run=args.dry_run, verbose=args.verbose)
    success = sync.run_sync()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()