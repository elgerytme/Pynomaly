#!/usr/bin/env python3
"""
TODO.md and GitHub Issues Synchronization Script

This script compares TODO.md with GitHub issues and detects drift between them.
It can create sync alert issues and optionally auto-update based on configuration.

Required environment variables:
- GITHUB_REPO: Repository in format 'owner/repo' 
- GITHUB_TOKEN: GitHub personal access token with issues read/write permissions

Optional environment variables:
- AUTO_UPDATE: Set to 'true' to enable automatic updates (default: false)
- DRY_RUN: Set to 'true' to run without making changes (default: false)
- TODO_FILE_PATH: Path to TODO.md file (default: '../docs/project/TODO.md')
"""

import os
import sys
import re
import logging
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from github import Github
from github.Repository import Repository
from github.Issue import Issue

# Configuration
TODO_FILE_PATH = os.getenv('TODO_FILE_PATH', '../docs/project/TODO.md')
GITHUB_REPO = os.getenv('GITHUB_REPO')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
AUTO_UPDATE = os.getenv('AUTO_UPDATE', 'false').lower() == 'true'
DRY_RUN = os.getenv('DRY_RUN', 'false').lower() == 'true'

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TodoItem:
    """Represents a TODO item from TODO.md"""
    id: str
    title: str
    priority: str
    estimate: str
    owner: str
    dependencies: List[str]
    description: str
    layer: str


class TodoParser:
    """Parser for TODO.md file"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def parse_todo_file(self) -> List[TodoItem]:
        """Parse TODO.md file and extract structured TODO items"""
        if not os.path.exists(self.file_path):
            logger.error(f"TODO file not found: {self.file_path}")
            return []
            
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except Exception as e:
            logger.error(f"Error reading TODO file: {e}")
            return []
            
        return self._extract_todo_items(content)
    
    def _extract_todo_items(self, content: str) -> List[TodoItem]:
        """Extract TODO items from file content"""
        items = []
        current_layer = ""
        
        lines = content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Detect layer sections
            if line.startswith('## ') and 'Layer' in line:
                current_layer = line.replace('## ', '').replace(' Layer', '')
                i += 1
                continue
            
            # Detect TODO items (### format)
            if line.startswith('### '):
                todo_item = self._parse_todo_item(lines, i, current_layer)
                if todo_item:
                    items.append(todo_item)
                i += 1
                continue
                
            i += 1
            
        return items
    
    def _parse_todo_item(self, lines: List[str], start_index: int, layer: str) -> Optional[TodoItem]:
        """Parse a single TODO item starting at given index"""
        if start_index >= len(lines):
            return None
            
        # Extract title and ID from header
        header = lines[start_index].strip()
        if not header.startswith('### '):
            return None
            
        header = header[4:]  # Remove '### '
        
        # Extract ID and title
        id_match = re.match(r'([A-Z]+-\d+):\s*(.+)', header)
        if not id_match:
            return None
            
        todo_id = id_match.group(1)
        title = id_match.group(2)
        
        # Parse details
        priority = ""
        estimate = ""
        owner = ""
        dependencies = []
        description = ""
        
        i = start_index + 1
        while i < len(lines) and not lines[i].startswith('### '):
            line = lines[i].strip()
            
            if line.startswith('- **Priority**:'):
                priority = line.split(':', 1)[1].strip()
            elif line.startswith('- **Estimate**:'):
                estimate = line.split(':', 1)[1].strip()
            elif line.startswith('- **Owner**:'):
                owner = line.split(':', 1)[1].strip()
            elif line.startswith('- **Dependencies**:'):
                deps_str = line.split(':', 1)[1].strip()
                if deps_str.lower() != 'none':
                    dependencies = [dep.strip() for dep in deps_str.split(',')]
            elif line.startswith('- **Description**:'):
                description = line.split(':', 1)[1].strip()
            
            i += 1
            
        return TodoItem(
            id=todo_id,
            title=title,
            priority=priority,
            estimate=estimate,
            owner=owner,
            dependencies=dependencies,
            description=description,
            layer=layer
        )


class IssueSyncer:
    """Handles synchronization between TODO items and GitHub issues"""
    
    def __init__(self, repo: Repository):
        self.repo = repo
        
    def get_open_issues(self) -> Dict[str, Issue]:
        """Get all open issues from the repository"""
        issues = {}
        try:
            for issue in self.repo.get_issues(state='open'):
                issues[issue.title] = issue
        except Exception as e:
            logger.error(f"Error fetching issues: {e}")
            
        return issues
    
    def find_drift(self, todo_items: List[TodoItem], open_issues: Dict[str, Issue]) -> Tuple[List[TodoItem], List[str]]:
        """Find drift between TODO items and GitHub issues"""
        # Create sets for comparison
        todo_titles = {f"{item.id}: {item.title}" for item in todo_items}
        issue_titles = set(open_issues.keys())
        
        # Find missing items
        missing_issues = []
        for item in todo_items:
            item_title = f"{item.id}: {item.title}"
            if item_title not in issue_titles:
                missing_issues.append(item)
        
        # Find extra issues (issues not in TODO)
        extra_issues = []
        for issue_title in issue_titles:
            if issue_title not in todo_titles and not issue_title.startswith("Sync Alert:"):
                extra_issues.append(issue_title)
        
        return missing_issues, extra_issues
    
    def create_sync_alert_issue(self, missing_issues: List[TodoItem], extra_issues: List[str]) -> None:
        """Create or update a sync alert issue"""
        if not missing_issues and not extra_issues:
            logger.info("No drift detected - TODO.md and issues are in sync")
            return
        
        # Check if sync alert issue already exists
        existing_alert = None
        for issue in self.repo.get_issues(state='open'):
            if issue.title.startswith("Sync Alert:"):
                existing_alert = issue
                break
        
        # Build issue body
        body = self._build_sync_alert_body(missing_issues, extra_issues)
        
        if DRY_RUN:
            logger.info("DRY RUN: Would create/update sync alert issue")
            logger.info(f"Body: {body}")
            return
        
        try:
            if existing_alert:
                existing_alert.edit(body=body)
                logger.info(f"Updated existing sync alert issue: {existing_alert.html_url}")
            else:
                new_issue = self.repo.create_issue(
                    title="Sync Alert: TODO List vs Issues Mismatch",
                    body=body,
                    labels=["sync-alert", "maintenance"]
                )
                logger.info(f"Created new sync alert issue: {new_issue.html_url}")
        except Exception as e:
            logger.error(f"Error creating/updating sync alert issue: {e}")
    
    def _build_sync_alert_body(self, missing_issues: List[TodoItem], extra_issues: List[str]) -> str:
        """Build the body content for sync alert issue"""
        body = "# TODO.md and GitHub Issues Synchronization Alert\n\n"
        body += "This issue was automatically generated to alert about drift between TODO.md and GitHub issues.\n\n"
        
        if missing_issues:
            body += "## 🔍 TODO Items Missing as Issues\n\n"
            body += "The following TODO items should be created as GitHub issues:\n\n"
            
            for item in missing_issues:
                body += f"### {item.id}: {item.title}\n"
                body += f"- **Layer**: {item.layer}\n"
                body += f"- **Priority**: {item.priority}\n"
                body += f"- **Estimate**: {item.estimate}\n"
                body += f"- **Owner**: {item.owner}\n"
                body += f"- **Dependencies**: {', '.join(item.dependencies) if item.dependencies else 'None'}\n"
                body += f"- **Description**: {item.description}\n\n"
        
        if extra_issues:
            body += "## 📋 Issues Not in TODO.md\n\n"
            body += "The following GitHub issues are not documented in TODO.md:\n\n"
            
            for issue_title in extra_issues:
                body += f"- {issue_title}\n"
        
        body += "\n## 🔧 Recommended Actions\n\n"
        
        if missing_issues:
            body += "1. **Create missing issues**: Convert TODO items to GitHub issues\n"
            
        if extra_issues:
            body += "2. **Update TODO.md**: Add missing issues to TODO.md or close unnecessary issues\n"
            
        body += "3. **Review and reconcile**: Ensure all work items are properly tracked\n"
        body += "\n*This issue will be automatically updated when the sync script runs.*"
        
        return body
    
    def auto_create_issues(self, missing_issues: List[TodoItem]) -> None:
        """Automatically create GitHub issues for missing TODO items"""
        if not AUTO_UPDATE or not missing_issues:
            return
            
        logger.info(f"Auto-creating {len(missing_issues)} missing issues")
        
        for item in missing_issues:
            if DRY_RUN:
                logger.info(f"DRY RUN: Would create issue for {item.id}: {item.title}")
                continue
                
            try:
                # Build issue body
                body = f"**Layer**: {item.layer}\n"
                body += f"**Priority**: {item.priority}\n"
                body += f"**Estimate**: {item.estimate}\n"
                body += f"**Owner**: {item.owner}\n"
                body += f"**Dependencies**: {', '.join(item.dependencies) if item.dependencies else 'None'}\n\n"
                body += f"**Description**: {item.description}\n\n"
                body += f"*This issue was automatically created from TODO.md item {item.id}*"
                
                # Create labels based on priority and layer
                labels = ["todo-sync", item.layer.lower().replace(' ', '-')]
                if item.priority.lower() in ['critical', 'high', 'medium', 'low']:
                    labels.append(f"priority-{item.priority.lower()}")
                
                new_issue = self.repo.create_issue(
                    title=f"{item.id}: {item.title}",
                    body=body,
                    labels=labels
                )
                
                logger.info(f"Created issue for {item.id}: {new_issue.html_url}")
                
            except Exception as e:
                logger.error(f"Error creating issue for {item.id}: {e}")


def validate_environment() -> bool:
    """Validate required environment variables"""
    if not GITHUB_TOKEN:
        logger.error("GITHUB_TOKEN environment variable is required")
        return False
        
    if not GITHUB_REPO:
        logger.error("GITHUB_REPO environment variable is required (format: owner/repo)")
        return False
        
    return True


def main():
    """Main function to run the sync process"""
    logger.info("Starting TODO.md and GitHub Issues synchronization")
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Parse TODO.md
    parser = TodoParser(TODO_FILE_PATH)
    todo_items = parser.parse_todo_file()
    
    if not todo_items:
        logger.warning("No TODO items found in TODO.md")
        return
    
    logger.info(f"Found {len(todo_items)} TODO items")
    
    # Initialize GitHub client
    try:
        github_client = Github(GITHUB_TOKEN)
        repo = github_client.get_repo(GITHUB_REPO)
    except Exception as e:
        logger.error(f"Error initializing GitHub client: {e}")
        sys.exit(1)
    
    # Sync with issues
    syncer = IssueSyncer(repo)
    open_issues = syncer.get_open_issues()
    
    logger.info(f"Found {len(open_issues)} open issues")
    
    # Find drift
    missing_issues, extra_issues = syncer.find_drift(todo_items, open_issues)
    
    logger.info(f"Found {len(missing_issues)} missing issues and {len(extra_issues)} extra issues")
    
    # Create sync alert if there's drift
    if missing_issues or extra_issues:
        syncer.create_sync_alert_issue(missing_issues, extra_issues)
        
        # Auto-create issues if enabled
        if AUTO_UPDATE:
            syncer.auto_create_issues(missing_issues)
    else:
        logger.info("✅ TODO.md and GitHub issues are in sync")
    
    logger.info("Synchronization completed")


if __name__ == "__main__":
    main()
