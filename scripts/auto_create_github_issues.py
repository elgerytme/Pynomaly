#!/usr/bin/env python3
"""
Auto-create GitHub issues for test failures with git blame to assign to last committer.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
from github import Github


def get_git_blame_info(file_path: str, line_number: int = None) -> dict:
    """Get git blame information for a file."""
    try:
        if line_number:
            cmd = ['git', 'blame', '-L', f'{line_number},{line_number}', '--porcelain', file_path]
        else:
            cmd = ['git', 'log', '-1', '--format=%H,%an,%ae,%s', '--', file_path]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if line_number:
            # Parse git blame output
            lines = result.stdout.strip().split('\n')
            if lines:
                commit_info = lines[0].split()
                commit_hash = commit_info[0]
                author_cmd = ['git', 'show', '-s', '--format=%an,%ae', commit_hash]
                author_result = subprocess.run(author_cmd, capture_output=True, text=True, check=True)
                author_name, author_email = author_result.stdout.strip().split(',')
                
                return {
                    'commit_hash': commit_hash,
                    'author_name': author_name,
                    'author_email': author_email,
                    'github_username': get_github_username(author_email)
                }
        else:
            # Parse git log output
            if result.stdout.strip():
                commit_hash, author_name, author_email, subject = result.stdout.strip().split(',', 3)
                return {
                    'commit_hash': commit_hash,
                    'author_name': author_name,
                    'author_email': author_email,
                    'github_username': get_github_username(author_email),
                    'subject': subject
                }
    except subprocess.CalledProcessError as e:
        print(f"Error getting git blame info for {file_path}: {e}")
    
    return {}


def get_github_username(email: str) -> str:
    """Get GitHub username from email (simplified mapping)."""
    # This is a simplified mapping - in reality, you'd want a more sophisticated approach
    email_to_username = {
        'user@example.com': 'username',
        # Add more mappings as needed
    }
    
    return email_to_username.get(email, email.split('@')[0])


def parse_test_results(file_path: str) -> list:
    """Parse JUnit XML report and extract failure information."""
    if not os.path.exists(file_path):
        print(f"Test results file not found: {file_path}")
        return []
    
    tree = ET.parse(file_path)
    root = tree.getroot()
    failures = []

    # Collect failed testcases
    for testcase in root.findall('.//testcase'):
        failure_elem = testcase.find('failure')
        error_elem = testcase.find('error')
        
        if failure_elem is not None or error_elem is not None:
            # Extract test file path from classname
            classname = testcase.get('classname', '')
            test_name = testcase.get('name', '')
            
            # Convert classname to file path
            test_file_path = classname.replace('.', '/') + '.py'
            if classname.startswith('tests.'):
                test_file_path = classname.replace('.', '/') + '.py'
            elif 'test_' in classname:
                # Handle different test naming conventions
                test_file_path = f"tests/{classname.replace('.', '/')}.py"
            
            # Get git blame info
            blame_info = get_git_blame_info(test_file_path)
            
            failure_info = {
                'classname': classname,
                'name': test_name,
                'file_path': test_file_path,
                'message': (failure_elem.get('message') if failure_elem is not None 
                           else error_elem.get('message')),
                'text': (failure_elem.text if failure_elem is not None 
                        else error_elem.text),
                'type': (failure_elem.get('type') if failure_elem is not None 
                        else error_elem.get('type')),
                'blame_info': blame_info
            }
            
            failures.append(failure_info)
    
    return failures


def create_github_issues(failures: list, repo) -> None:
    """Create GitHub issues for test failures."""
    for failure in failures:
        title = f"🔴 Test Failure: {failure['classname']}.{failure['name']}"
        
        # Create detailed issue body
        body = f"""## Test Failure Report

**Test:** `{failure['classname']}.{failure['name']}`
**File:** `{failure['file_path']}`
**Type:** `{failure['type'] or 'Unknown'}`
**Timestamp:** {datetime.now().isoformat()}

### Failure Details

```
{failure['message'] or 'No message available'}
```

### Full Error Output

```
{failure['text'] or 'No detailed output available'}
```

### Git Blame Information

"""
        
        assignee = None
        if failure['blame_info']:
            blame = failure['blame_info']
            body += f"""**Last Modified By:** {blame.get('author_name', 'Unknown')} ({blame.get('author_email', 'Unknown')})
**Commit:** {blame.get('commit_hash', 'Unknown')}
**GitHub User:** @{blame.get('github_username', 'Unknown')}
"""
            assignee = blame.get('github_username')
        else:
            body += "*Git blame information not available*"
        
        body += f"""

### Action Required

- [ ] Investigate the test failure
- [ ] Fix the underlying issue
- [ ] Verify the fix doesn't break other tests
- [ ] Update test if needed

### Additional Context

This issue was automatically created by the nightly test suite validation.

**Labels:** `bug`, `test-failure`, `auto-created`
"""
        
        # Check if issue already exists
        existing_issues = repo.get_issues(state='open')
        issue_exists = any(issue.title == title for issue in existing_issues)
        
        if not issue_exists:
            try:
                # Create the issue
                issue = repo.create_issue(
                    title=title,
                    body=body,
                    labels=['bug', 'test-failure', 'auto-created']
                )
                
                # Try to assign to the person who last modified the test
                if assignee and assignee != 'Unknown':
                    try:
                        issue.add_to_assignees(assignee)
                        print(f"✅ Issue created and assigned to {assignee}: {title}")
                    except Exception as e:
                        print(f"⚠️  Issue created but failed to assign to {assignee}: {e}")
                        print(f"   Issue: {title}")
                else:
                    print(f"✅ Issue created (no assignee): {title}")
                    
            except Exception as e:
                print(f"❌ Failed to create issue: {title}")
                print(f"   Error: {e}")
        else:
            print(f"ℹ️  Issue already exists: {title}")


def main():
    parser = argparse.ArgumentParser(description='Auto-create GitHub issues for test failures')
    parser.add_argument('--test-results', default='test-results.xml', 
                       help='Path to JUnit XML test results file')
    parser.add_argument('--repository', help='GitHub repository (owner/repo)')
    parser.add_argument('--previous-run-id', help='Previous GitHub Actions run ID')
    
    args = parser.parse_args()
    
    # Get GitHub token
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("❌ GITHUB_TOKEN environment variable not set")
        return 1
    
    # Get repository name
    repo_name = args.repository or os.getenv('GITHUB_REPOSITORY')
    if not repo_name:
        print("❌ Repository name not provided and GITHUB_REPOSITORY not set")
        return 1
    
    # Initialize GitHub client
    try:
        g = Github(token)
        repo = g.get_repo(repo_name)
    except Exception as e:
        print(f"❌ Failed to initialize GitHub client: {e}")
        return 1
    
    # Parse test results
    failures = parse_test_results(args.test_results)
    
    if failures:
        print(f"📋 Found {len(failures)} test failures")
        create_github_issues(failures, repo)
    else:
        print("✅ No test failures found")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
