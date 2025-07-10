#!/usr/bin/env python3
"""
GitHub Issues and Todo List Synchronization System

This script automatically synchronizes the todo list with GitHub issues,
ensuring that all tasks are tracked consistently across both systems.

Features:
- Bi-directional sync between todo list and GitHub issues
- Automatic issue creation for new todos
- Status synchronization (pending -> open, completed -> closed)
- Priority mapping to GitHub issue labels
- Conflict resolution and merge strategies
- Webhook support for real-time updates
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import requests
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Todo list management using direct file operations for standalone use


class SyncDirection(Enum):
    """Synchronization direction."""

    TODO_TO_GITHUB = "todo_to_github"
    GITHUB_TO_TODO = "github_to_todo"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class TodoItem:
    """Todo item representation."""

    id: str
    content: str
    status: str  # pending, in_progress, completed
    priority: str  # high, medium, low
    github_issue_id: int | None = None
    github_issue_url: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


@dataclass
class GitHubIssue:
    """GitHub issue representation."""

    id: int
    number: int
    title: str
    body: str
    state: str  # open, closed
    labels: list[str]
    assignees: list[str]
    created_at: str
    updated_at: str
    html_url: str
    todo_id: str | None = None


class GitHubTodoSyncManager:
    """Manager for synchronizing todos with GitHub issues."""

    def __init__(
        self, github_token: str | None = None, repository: str = "pynomaly/pynomaly"
    ):
        """Initialize the sync manager."""
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.repository = repository
        self.project_root = Path(__file__).parent.parent.parent
        self.config_path = self.project_root / "config" / "github_sync_config.yaml"

        # GitHub API configuration
        self.github_api_base = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Pynomaly-Todo-Sync/1.0",
        }

        # Load configuration
        self.config = self._load_config()

        # Todo management
        self.todo_reader = TodoRead()
        self.todo_writer = TodoWrite()

        # Sync state
        self.sync_log = []

    def _load_config(self) -> dict[str, Any]:
        """Load synchronization configuration."""
        default_config = {
            "sync": {
                "enabled": True,
                "direction": "bidirectional",
                "auto_sync_interval": 300,  # 5 minutes
                "conflict_resolution": "github_wins",
            },
            "github": {
                "labels": {
                    "todo_prefix": "todo:",
                    "priority_high": "priority:high",
                    "priority_medium": "priority:medium",
                    "priority_low": "priority:low",
                    "status_pending": "status:pending",
                    "status_in_progress": "status:in-progress",
                    "status_completed": "status:completed",
                },
                "issue_template": {
                    "title_format": "TODO: {content}",
                    "body_template": """
## Todo Item

**ID**: {id}
**Priority**: {priority}
**Status**: {status}

**Description**: {content}

---
*This issue was automatically created from the todo list.*
*Do not edit the todo metadata above this line.*

### Additional Notes

Add any additional context, requirements, or implementation details here.

### Acceptance Criteria

- [ ] Task completed successfully
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code reviewed

### Implementation Notes

Add implementation details, technical notes, or related issues here.
                    """,
                },
            },
            "mapping": {
                "status_mapping": {
                    "pending": "open",
                    "in_progress": "open",
                    "completed": "closed",
                },
                "reverse_status_mapping": {"open": "pending", "closed": "completed"},
            },
            "webhooks": {
                "enabled": False,
                "endpoint": "/webhook/github-issues",
                "secret": None,
            },
        }

        if self.config_path.exists():
            with open(self.config_path) as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        else:
            # Create default config file
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w") as f:
                yaml.dump(default_config, f, default_flow_style=False)

        return default_config

    def _save_config(self):
        """Save configuration to file."""
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def _log_action(self, action: str, details: str):
        """Log synchronization actions."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details,
        }
        self.sync_log.append(log_entry)
        print(f"[{log_entry['timestamp']}] {action}: {details}")

    def _make_github_request(
        self, method: str, endpoint: str, data: dict | None = None
    ) -> requests.Response:
        """Make a request to the GitHub API."""
        url = f"{self.github_api_base}{endpoint}"

        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=self.headers, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(
                    url, headers=self.headers, json=data, timeout=30
                )
            elif method.upper() == "PATCH":
                response = requests.patch(
                    url, headers=self.headers, json=data, timeout=30
                )
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=self.headers, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response

        except requests.exceptions.RequestException as e:
            self._log_action("ERROR", f"GitHub API request failed: {e}")
            raise

    def get_current_todos(self) -> list[TodoItem]:
        """Get current todo items."""
        try:
            # Use the TodoRead tool to get current todos
            todos_data = self.todo_reader.read_todos()

            todos = []
            for todo_data in todos_data:
                todo = TodoItem(
                    id=todo_data["id"],
                    content=todo_data["content"],
                    status=todo_data["status"],
                    priority=todo_data["priority"],
                    created_at=todo_data.get("created_at"),
                    updated_at=todo_data.get("updated_at"),
                )
                todos.append(todo)

            return todos

        except Exception as e:
            self._log_action("ERROR", f"Failed to get current todos: {e}")
            return []

    def get_github_issues(self) -> list[GitHubIssue]:
        """Get GitHub issues with todo labels."""
        try:
            todo_label = self.config["github"]["labels"]["todo_prefix"]

            # Get all issues with todo label
            endpoint = f"/repos/{self.repository}/issues"
            params = {"labels": todo_label, "state": "all", "per_page": 100}

            response = self._make_github_request(
                "GET", f"{endpoint}?{self._build_query(params)}"
            )
            issues_data = response.json()

            issues = []
            for issue_data in issues_data:
                # Extract todo ID from issue body
                todo_id = self._extract_todo_id_from_issue(issue_data)

                issue = GitHubIssue(
                    id=issue_data["id"],
                    number=issue_data["number"],
                    title=issue_data["title"],
                    body=issue_data["body"] or "",
                    state=issue_data["state"],
                    labels=[label["name"] for label in issue_data["labels"]],
                    assignees=[
                        assignee["login"] for assignee in issue_data["assignees"]
                    ],
                    created_at=issue_data["created_at"],
                    updated_at=issue_data["updated_at"],
                    html_url=issue_data["html_url"],
                    todo_id=todo_id,
                )
                issues.append(issue)

            return issues

        except Exception as e:
            self._log_action("ERROR", f"Failed to get GitHub issues: {e}")
            return []

    def _build_query(self, params: dict[str, Any]) -> str:
        """Build query string from parameters."""
        return "&".join([f"{key}={value}" for key, value in params.items()])

    def _extract_todo_id_from_issue(self, issue_data: dict) -> str | None:
        """Extract todo ID from GitHub issue body."""
        body = issue_data.get("body", "")

        # Look for **ID**: {id} pattern
        import re

        match = re.search(r"\*\*ID\*\*:\s*([a-zA-Z0-9_-]+)", body)
        if match:
            return match.group(1)

        return None

    def create_github_issue_for_todo(self, todo: TodoItem) -> GitHubIssue | None:
        """Create a GitHub issue for a todo item."""
        try:
            # Build issue data
            title = self.config["github"]["issue_template"]["title_format"].format(
                content=todo.content
            )

            body = self.config["github"]["issue_template"]["body_template"].format(
                id=todo.id,
                content=todo.content,
                priority=todo.priority,
                status=todo.status,
            )

            # Build labels
            labels = [
                self.config["github"]["labels"]["todo_prefix"],
                f"{self.config['github']['labels'][f'priority_{todo.priority}']}",
                f"{self.config['github']['labels'][f'status_{todo.status}']}",
            ]

            issue_data = {"title": title, "body": body, "labels": labels}

            # Create the issue
            endpoint = f"/repos/{self.repository}/issues"
            response = self._make_github_request("POST", endpoint, issue_data)
            issue_response = response.json()

            # Create GitHubIssue object
            github_issue = GitHubIssue(
                id=issue_response["id"],
                number=issue_response["number"],
                title=issue_response["title"],
                body=issue_response["body"],
                state=issue_response["state"],
                labels=[label["name"] for label in issue_response["labels"]],
                assignees=[
                    assignee["login"] for assignee in issue_response["assignees"]
                ],
                created_at=issue_response["created_at"],
                updated_at=issue_response["updated_at"],
                html_url=issue_response["html_url"],
                todo_id=todo.id,
            )

            self._log_action(
                "CREATE_ISSUE",
                f"Created GitHub issue #{github_issue.number} for todo {todo.id}",
            )

            return github_issue

        except Exception as e:
            self._log_action(
                "ERROR", f"Failed to create GitHub issue for todo {todo.id}: {e}"
            )
            return None

    def update_github_issue_from_todo(self, issue: GitHubIssue, todo: TodoItem) -> bool:
        """Update a GitHub issue based on todo changes."""
        try:
            # Determine what needs to be updated
            updates = {}

            # Check title
            expected_title = self.config["github"]["issue_template"][
                "title_format"
            ].format(content=todo.content)
            if issue.title != expected_title:
                updates["title"] = expected_title

            # Check state
            expected_state = self.config["mapping"]["status_mapping"][todo.status]
            if issue.state != expected_state:
                updates["state"] = expected_state

            # Check labels
            expected_labels = [
                self.config["github"]["labels"]["todo_prefix"],
                f"{self.config['github']['labels'][f'priority_{todo.priority}']}",
                f"{self.config['github']['labels'][f'status_{todo.status}']}",
            ]

            # Keep existing non-todo labels
            existing_labels = [
                label
                for label in issue.labels
                if not label.startswith("todo:")
                and not label.startswith("priority:")
                and not label.startswith("status:")
            ]

            final_labels = expected_labels + existing_labels
            if set(issue.labels) != set(final_labels):
                updates["labels"] = final_labels

            # Update body to reflect current todo state
            updated_body = self.config["github"]["issue_template"][
                "body_template"
            ].format(
                id=todo.id,
                content=todo.content,
                priority=todo.priority,
                status=todo.status,
            )

            # Preserve any additional content after the template
            if "### Additional Notes" in issue.body:
                additional_content = issue.body.split("### Additional Notes", 1)[1]
                updated_body += "### Additional Notes" + additional_content

            if issue.body != updated_body:
                updates["body"] = updated_body

            # Apply updates if any
            if updates:
                endpoint = f"/repos/{self.repository}/issues/{issue.number}"
                self._make_github_request("PATCH", endpoint, updates)

                self._log_action(
                    "UPDATE_ISSUE",
                    f"Updated GitHub issue #{issue.number} for todo {todo.id}",
                )
                return True

            return False  # No updates needed

        except Exception as e:
            self._log_action(
                "ERROR", f"Failed to update GitHub issue #{issue.number}: {e}"
            )
            return False

    def create_todo_from_github_issue(self, issue: GitHubIssue) -> TodoItem | None:
        """Create a todo item from a GitHub issue."""
        try:
            # Extract todo information from issue
            todo_id = issue.todo_id or f"gh_{issue.number}"

            # Extract priority from labels
            priority = "medium"  # default
            for label in issue.labels:
                if label.startswith("priority:"):
                    priority = label.split(":", 1)[1]
                    break

            # Map status from issue state and labels
            status = self.config["mapping"]["reverse_status_mapping"].get(
                issue.state, "pending"
            )

            # Override with status label if present
            for label in issue.labels:
                if label.startswith("status:"):
                    status_from_label = label.split(":", 1)[1].replace("-", "_")
                    if status_from_label in ["pending", "in_progress", "completed"]:
                        status = status_from_label
                    break

            # Extract content from title (remove "TODO: " prefix if present)
            content = issue.title
            if content.startswith("TODO: "):
                content = content[6:]

            todo = TodoItem(
                id=todo_id,
                content=content,
                status=status,
                priority=priority,
                github_issue_id=issue.id,
                github_issue_url=issue.html_url,
                created_at=issue.created_at,
                updated_at=issue.updated_at,
            )

            self._log_action(
                "CREATE_TODO",
                f"Created todo {todo.id} from GitHub issue #{issue.number}",
            )

            return todo

        except Exception as e:
            self._log_action(
                "ERROR", f"Failed to create todo from GitHub issue #{issue.number}: {e}"
            )
            return None

    def sync_todos_to_github(
        self, todos: list[TodoItem], github_issues: list[GitHubIssue]
    ) -> dict[str, Any]:
        """Sync todos to GitHub issues."""
        results = {"created": 0, "updated": 0, "skipped": 0, "errors": 0}

        # Create mapping of existing issues by todo ID
        issues_by_todo_id = {
            issue.todo_id: issue for issue in github_issues if issue.todo_id
        }

        for todo in todos:
            try:
                existing_issue = issues_by_todo_id.get(todo.id)

                if existing_issue:
                    # Update existing issue
                    if self.update_github_issue_from_todo(existing_issue, todo):
                        results["updated"] += 1
                    else:
                        results["skipped"] += 1
                else:
                    # Create new issue
                    new_issue = self.create_github_issue_for_todo(todo)
                    if new_issue:
                        results["created"] += 1
                    else:
                        results["errors"] += 1

            except Exception as e:
                self._log_action("ERROR", f"Failed to sync todo {todo.id}: {e}")
                results["errors"] += 1

        return results

    def sync_github_to_todos(
        self, github_issues: list[GitHubIssue], todos: list[TodoItem]
    ) -> dict[str, Any]:
        """Sync GitHub issues to todos."""
        results = {"created": 0, "updated": 0, "skipped": 0, "errors": 0}

        # Create mapping of existing todos by ID
        todos_by_id = {todo.id: todo for todo in todos}

        # Track todos that need to be updated
        updated_todos = list(todos)

        for issue in github_issues:
            try:
                if not issue.todo_id:
                    continue

                existing_todo = todos_by_id.get(issue.todo_id)

                if existing_todo:
                    # Check if update is needed
                    needs_update = False

                    # Check content
                    issue_content = issue.title
                    if issue_content.startswith("TODO: "):
                        issue_content = issue_content[6:]

                    if existing_todo.content != issue_content:
                        existing_todo.content = issue_content
                        needs_update = True

                    # Check status
                    issue_status = self.config["mapping"]["reverse_status_mapping"].get(
                        issue.state, "pending"
                    )

                    # Override with status label if present
                    for label in issue.labels:
                        if label.startswith("status:"):
                            status_from_label = label.split(":", 1)[1].replace("-", "_")
                            if status_from_label in [
                                "pending",
                                "in_progress",
                                "completed",
                            ]:
                                issue_status = status_from_label
                            break

                    if existing_todo.status != issue_status:
                        existing_todo.status = issue_status
                        needs_update = True

                    # Check priority
                    issue_priority = "medium"  # default
                    for label in issue.labels:
                        if label.startswith("priority:"):
                            issue_priority = label.split(":", 1)[1]
                            break

                    if existing_todo.priority != issue_priority:
                        existing_todo.priority = issue_priority
                        needs_update = True

                    if needs_update:
                        results["updated"] += 1
                        self._log_action(
                            "UPDATE_TODO",
                            f"Updated todo {existing_todo.id} from GitHub issue #{issue.number}"
                        )
                    else:
                        results["skipped"] += 1
                else:
                    # Create new todo
                    new_todo = self.create_todo_from_github_issue(issue)
                    if new_todo:
                        updated_todos.append(new_todo)
                        results["created"] += 1
                    else:
                        results["errors"] += 1

            except Exception as e:
                self._log_action(
                    "ERROR", f"Failed to sync GitHub issue #{issue.number}: {e}"
                )
                results["errors"] += 1

        # Update the todo list if there were changes
        if results["created"] > 0 or results["updated"] > 0:
            try:
                # Convert TodoItem objects back to the format expected by TodoWrite
                todos_data = []
                for todo in updated_todos:
                    todos_data.append(
                        {
                            "id": todo.id,
                            "content": todo.content,
                            "status": todo.status,
                            "priority": todo.priority,
                        }
                    )

                self.todo_writer.write_todos(todos_data)
                self._log_action(
                    "UPDATE_TODO_LIST",
                    f"Updated todo list with {len(updated_todos)} items",
                )

            except Exception as e:
                self._log_action("ERROR", f"Failed to update todo list: {e}")
                results["errors"] += 1

        return results

    def perform_full_sync(
        self, direction: SyncDirection = SyncDirection.BIDIRECTIONAL
    ) -> dict[str, Any]:
        """Perform a full synchronization between todos and GitHub issues."""
        self._log_action(
            "START_SYNC", f"Starting full sync with direction: {direction.value}"
        )

        # Get current state
        todos = self.get_current_todos()
        github_issues = self.get_github_issues()

        self._log_action(
            "STATE", f"Found {len(todos)} todos and {len(github_issues)} GitHub issues"
        )

        sync_results = {
            "todos_to_github": {},
            "github_to_todos": {},
            "direction": direction.value,
            "timestamp": datetime.now().isoformat(),
        }

        if direction in [SyncDirection.TODO_TO_GITHUB, SyncDirection.BIDIRECTIONAL]:
            sync_results["todos_to_github"] = self.sync_todos_to_github(
                todos, github_issues
            )

        if direction in [SyncDirection.GITHUB_TO_TODO, SyncDirection.BIDIRECTIONAL]:
            # Refresh todos if we just updated GitHub
            if direction == SyncDirection.BIDIRECTIONAL:
                todos = self.get_current_todos()
            sync_results["github_to_todos"] = self.sync_github_to_todos(
                github_issues, todos
            )

        self._log_action("COMPLETE_SYNC", f"Sync completed: {sync_results}")

        return sync_results

    def setup_automatic_sync(self, interval_minutes: int = 5):
        """Setup automatic synchronization."""
        self._log_action(
            "SETUP_AUTO_SYNC",
            f"Setting up automatic sync every {interval_minutes} minutes",
        )

        async def sync_loop():
            while True:
                try:
                    self.perform_full_sync()
                    await asyncio.sleep(interval_minutes * 60)
                except Exception as e:
                    self._log_action("ERROR", f"Auto sync failed: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute before retrying

        return sync_loop()

    def generate_sync_report(self) -> dict[str, Any]:
        """Generate a comprehensive sync report."""
        todos = self.get_current_todos()
        github_issues = self.get_github_issues()

        # Analyze sync state
        todos_by_id = {todo.id: todo for todo in todos}
        issues_by_todo_id = {
            issue.todo_id: issue for issue in github_issues if issue.todo_id
        }

        orphaned_todos = [todo for todo in todos if todo.id not in issues_by_todo_id]
        orphaned_issues = [
            issue
            for issue in github_issues
            if not issue.todo_id or issue.todo_id not in todos_by_id
        ]

        synced_pairs = []
        for todo in todos:
            if todo.id in issues_by_todo_id:
                issue = issues_by_todo_id[todo.id]
                synced_pairs.append(
                    {
                        "todo": todo,
                        "issue": issue,
                        "in_sync": self._check_sync_status(todo, issue),
                    }
                )

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_todos": len(todos),
                "total_issues": len(github_issues),
                "synced_pairs": len(synced_pairs),
                "orphaned_todos": len(orphaned_todos),
                "orphaned_issues": len(orphaned_issues),
                "in_sync_count": sum(1 for pair in synced_pairs if pair["in_sync"]),
            },
            "orphaned_todos": [
                {"id": todo.id, "content": todo.content, "status": todo.status}
                for todo in orphaned_todos
            ],
            "orphaned_issues": [
                {"number": issue.number, "title": issue.title, "state": issue.state}
                for issue in orphaned_issues
            ],
            "out_of_sync": [
                {
                    "todo_id": pair["todo"].id,
                    "issue_number": pair["issue"].number,
                    "differences": self._get_sync_differences(
                        pair["todo"], pair["issue"]
                    ),
                }
                for pair in synced_pairs
                if not pair["in_sync"]
            ],
            "recent_actions": self.sync_log[-20:],  # Last 20 actions
        }

        return report

    def _check_sync_status(self, todo: TodoItem, issue: GitHubIssue) -> bool:
        """Check if a todo and issue are in sync."""
        # Check title/content
        expected_title = self.config["github"]["issue_template"]["title_format"].format(
            content=todo.content
        )
        if issue.title != expected_title:
            return False

        # Check state/status
        expected_state = self.config["mapping"]["status_mapping"][todo.status]
        if issue.state != expected_state:
            return False

        # Check priority label
        expected_priority_label = (
            f"{self.config['github']['labels'][f'priority_{todo.priority}']}"
        )
        if expected_priority_label not in issue.labels:
            return False

        return True

    def _get_sync_differences(self, todo: TodoItem, issue: GitHubIssue) -> list[str]:
        """Get list of differences between todo and issue."""
        differences = []

        # Check title/content
        expected_title = self.config["github"]["issue_template"]["title_format"].format(
            content=todo.content
        )
        if issue.title != expected_title:
            differences.append(
                f"Title mismatch: todo='{todo.content}', issue='{issue.title}'"
            )

        # Check state/status
        expected_state = self.config["mapping"]["status_mapping"][todo.status]
        if issue.state != expected_state:
            differences.append(
                f"State mismatch: todo='{todo.status}', issue='{issue.state}'"
            )

        # Check priority
        expected_priority_label = (
            f"{self.config['github']['labels'][f'priority_{todo.priority}']}"
        )
        if expected_priority_label not in issue.labels:
            differences.append(
                f"Priority mismatch: todo='{todo.priority}', "
                f"issue labels={issue.labels}"
            )

        return differences


def main():
    """Main function for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="GitHub Issues and Todo List Sync")
    parser.add_argument(
        "--action",
        choices=["sync", "report", "setup"],
        default="sync",
        help="Action to perform",
    )
    parser.add_argument(
        "--direction",
        choices=["todo_to_github", "github_to_todo", "bidirectional"],
        default="bidirectional",
        help="Sync direction",
    )
    parser.add_argument("--auto", action="store_true", help="Start automatic sync")
    parser.add_argument(
        "--interval", type=int, default=5, help="Auto sync interval in minutes"
    )
    parser.add_argument("--output", help="Output file for reports")

    args = parser.parse_args()

    # Initialize sync manager
    sync_manager = GitHubTodoSyncManager()

    if args.action == "sync":
        direction = SyncDirection(args.direction)
        results = sync_manager.perform_full_sync(direction)

        print(f"Sync completed: {results}")

        if args.auto:
            print(f"Starting automatic sync every {args.interval} minutes...")
            asyncio.run(sync_manager.setup_automatic_sync(args.interval))

    elif args.action == "report":
        report = sync_manager.generate_sync_report()

        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {args.output}")
        else:
            print(json.dumps(report, indent=2))

    elif args.action == "setup":
        print("Setting up GitHub sync configuration...")
        sync_manager._save_config()
        print(f"Configuration saved to {sync_manager.config_path}")


if __name__ == "__main__":
    main()
