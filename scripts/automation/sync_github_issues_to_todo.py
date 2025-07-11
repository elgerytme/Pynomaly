#!/usr/bin/env python3
"""
GitHub Issues to TODO.md Synchronization Script

This script fetches GitHub issues and automatically updates the Issues section
of TODO.md to keep it synchronized with the current state of GitHub issues.

Usage:
    python sync_github_issues_to_todo.py

Environment Variables:
    GITHUB_TOKEN: GitHub personal access token
    GITHUB_REPOSITORY: Repository in format "owner/repo"
"""

import os
import re
import sys
from datetime import datetime
from typing import Any

import requests
from dateutil.parser import parse as parse_date


class GitHubIssuesSync:
    """Synchronizes GitHub issues with TODO.md Issues section."""

    def __init__(self):
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repository = os.getenv("GITHUB_REPOSITORY")
        self.base_url = "https://api.github.com"

        if not self.github_token:
            raise ValueError("GITHUB_TOKEN environment variable is required")
        if not self.repository:
            raise ValueError("GITHUB_REPOSITORY environment variable is required")

        self.headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Pynomaly-Issues-Sync/1.0",
        }

    def fetch_issues(self) -> list[dict[str, Any]]:
        """Fetch all open issues from GitHub."""
        issues = []
        page = 1
        per_page = 100

        while True:
            url = f"{self.base_url}/repos/{self.repository}/issues"
            params = {
                "state": "open",
                "per_page": per_page,
                "page": page,
                "sort": "created",
                "direction": "asc",
            }

            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()

            page_issues = response.json()
            if not page_issues:
                break

            # Filter out pull requests (GitHub API returns both issues and PRs)
            issues.extend(
                [issue for issue in page_issues if "pull_request" not in issue]
            )
            page += 1

        return issues

    def get_priority_from_labels(self, labels: list[dict[str, Any]]) -> str:
        """Extract priority from issue labels."""
        priority_map = {"P1-High": "high", "P2-Medium": "medium", "P3-Low": "low"}

        for label in labels:
            label_name = label.get("name", "")
            if label_name in priority_map:
                return priority_map[label_name]

        return "medium"  # Default priority

    def get_status_from_issue(self, issue: dict[str, Any]) -> str:
        """Determine status from issue state and labels."""
        if issue["state"] == "closed":
            return "completed"

        # Check for in-progress or blocked labels
        for label in issue.get("labels", []):
            label_name = label.get("name", "").lower()
            if "in-progress" in label_name or "in progress" in label_name:
                return "in_progress"
            if "blocked" in label_name:
                return "blocked"

        return "pending"

    def get_category_from_labels(self, labels: list[dict[str, Any]]) -> str:
        """Extract category from issue labels."""
        categories = {
            "presentation": "🎨 Presentation",
            "application": "⚙️ Application",
            "infrastructure": "🏗️ Infrastructure",
            "documentation": "📚 Documentation",
            "ci/cd": "🚀 CI/CD",
            "enhancement": "✨ Enhancement",
            "bug": "🐛 Bug",
            "feature": "🚀 Feature",
        }

        for label in labels:
            label_name = label.get("name", "").lower()
            if label_name in categories:
                return categories[label_name]

        return "📋 General"

    def format_issue_for_todo(self, issue: dict[str, Any]) -> str:
        """Format a single issue for the TODO.md Issues section."""
        number = issue["number"]
        title = issue["title"]
        labels = issue.get("labels", [])
        created_at = parse_date(issue["created_at"])
        updated_at = parse_date(issue["updated_at"])

        priority = self.get_priority_from_labels(labels)
        status = self.get_status_from_issue(issue)
        category = self.get_category_from_labels(labels)

        # Format priority badge
        priority_badge = {
            "high": "🔥 P1-High",
            "medium": "🔶 P2-Medium",
            "low": "🟢 P3-Low",
        }.get(priority, "📋 General")

        # Format status badge
        status_badge = {
            "completed": "✅ COMPLETED",
            "in_progress": "🔄 IN PROGRESS",
            "blocked": "🚫 BLOCKED",
            "pending": "⏳ PENDING",
        }.get(status, "⏳ PENDING")

        # Extract label names for display
        label_names = [
            label["name"] for label in labels if not label["name"].startswith("P")
        ]
        labels_text = ", ".join(label_names) if label_names else "General"

        # Format dates
        created_date = created_at.strftime("%b %d, %Y")
        updated_date = updated_at.strftime("%b %d, %Y")

        return f"""
#### **Issue #{number}: {title}**

**Labels**: {labels_text}
**Priority**: {priority_badge}
**Status**: {status_badge}
**Category**: {category}
**Created**: {created_date}
**Updated**: {updated_date}

- **Scope**: {issue.get("body", "")[:200].strip()}{"..." if len(issue.get("body", "")) > 200 else ""}
- **GitHub**: [View Issue](https://github.com/{self.repository}/issues/{number})
"""

    def generate_issues_section(self, issues: list[dict[str, Any]]) -> str:
        """Generate the complete Issues section for TODO.md."""
        if not issues:
            return """
## 📋 **GitHub Issues**

No open issues found.

---
"""

        # Group issues by priority
        issues_by_priority = {"high": [], "medium": [], "low": []}

        for issue in issues:
            priority = self.get_priority_from_labels(issue.get("labels", []))
            issues_by_priority[priority].append(issue)

        # Sort issues within each priority by number
        for priority in issues_by_priority:
            issues_by_priority[priority].sort(key=lambda x: x["number"])

        # Count statistics
        total_issues = len(issues)
        completed_issues = len(
            [i for i in issues if self.get_status_from_issue(i) == "completed"]
        )
        in_progress_issues = len(
            [i for i in issues if self.get_status_from_issue(i) == "in_progress"]
        )

        # Generate section
        section = f"""
## 📋 **GitHub Issues** (Auto-Synchronized)

**Total Open Issues**: {total_issues}
**Completed**: {completed_issues}
**In Progress**: {in_progress_issues}
**Pending**: {total_issues - completed_issues - in_progress_issues}

**Last Sync**: {datetime.now().strftime("%B %d, %Y at %H:%M UTC")}

### 🔥 **P1-High Priority Issues**
"""

        if issues_by_priority["high"]:
            for issue in issues_by_priority["high"]:
                section += self.format_issue_for_todo(issue)
        else:
            section += "\nNo high priority issues.\n"

        section += "\n### 🔶 **P2-Medium Priority Issues**\n"

        if issues_by_priority["medium"]:
            for issue in issues_by_priority["medium"]:
                section += self.format_issue_for_todo(issue)
        else:
            section += "\nNo medium priority issues.\n"

        section += "\n### 🟢 **P3-Low Priority Issues**\n"

        if issues_by_priority["low"]:
            for issue in issues_by_priority["low"]:
                section += self.format_issue_for_todo(issue)
        else:
            section += "\nNo low priority issues.\n"

        section += "\n---\n"

        return section

    def update_todo_file(self, issues_section: str) -> bool:
        """Update the TODO.md file with the new Issues section."""
        todo_path = "TODO.md"

        if not os.path.exists(todo_path):
            print(f"ERROR: {todo_path} not found")
            return False

        # Read current TODO.md
        with open(todo_path, encoding="utf-8") as f:
            content = f.read()

        # Find existing Issues section using regex
        issues_pattern = r"## 📋 \*\*GitHub Issues\*\*.*?(?=##|\Z)"

        if re.search(issues_pattern, content, re.DOTALL):
            # Replace existing Issues section
            new_content = re.sub(
                issues_pattern, issues_section.strip(), content, flags=re.DOTALL
            )
        else:
            # Add Issues section before the last section (usually before ---)
            if "---" in content:
                parts = content.rsplit("---", 1)
                new_content = parts[0] + issues_section + "\n---" + parts[1]
            else:
                # Append to end
                new_content = content + "\n" + issues_section

        # Write updated content
        with open(todo_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        print(
            f"✅ Successfully updated {todo_path} with {len(re.findall(r'#### \*\*Issue #', issues_section))} issues"
        )
        return True

    def run(self) -> bool:
        """Main execution method."""
        try:
            print("🔄 Fetching GitHub issues...")
            issues = self.fetch_issues()
            print(f"📊 Found {len(issues)} open issues")

            print("📝 Generating Issues section...")
            issues_section = self.generate_issues_section(issues)

            print("💾 Updating TODO.md...")
            success = self.update_todo_file(issues_section)

            if success:
                print("✅ GitHub Issues sync completed successfully!")
                return True
            else:
                print("❌ Failed to update TODO.md")
                return False

        except Exception as e:
            print(f"❌ Error during sync: {e}")
            return False


def main():
    """Main entry point."""
    try:
        syncer = GitHubIssuesSync()
        success = syncer.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
