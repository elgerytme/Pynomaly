#!/usr/bin/env python3
"""
Changelog Update Checker for Pull Requests

Checks if CHANGELOG.md has been updated for significant changes in a PR.
This is used by the GitHub Actions workflow.
"""

import os
import sys
import re
from pathlib import Path

def read_file_list(filename):
    """Read file list from a file."""
    try:
        with open(filename, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return []

def parse_diff_stats(filename):
    """Parse git diff --numstat output."""
    stats = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 3:
                    try:
                        added = int(parts[0]) if parts[0] != '-' else 0
                        deleted = int(parts[1]) if parts[1] != '-' else 0
                        filename = parts[2]
                        stats.append((added, deleted, filename))
                    except ValueError:
                        continue
    except FileNotFoundError:
        pass
    
    return stats

def is_significant_change_pr(changed_files, diff_stats):
    """Determine if PR changes are significant enough to require changelog update."""
    
    # Files that always require changelog updates
    critical_paths = [
        'src/',
        'pynomaly/',
        'examples/',
        'docs/',
        'scripts/',
        'docker/',
        'tests/',
        'pyproject.toml',
        'requirements.txt',
        'README.md',
        'Dockerfile'
    ]
    
    # Files that don't require changelog updates
    ignore_paths = [
        '.gitignore',
        '.github/',
        'TODO.md',
        'CLAUDE.md',
        '.pytest_cache/',
        '__pycache__/',
        '.venv/',
        'node_modules/',
        '.coverage',
        'htmlcov/',
        '.mypy_cache/',
        'CHANGELOG.md'  # Don't require changelog for changelog changes
    ]
    
    significant_files = []
    total_lines_changed = 0
    
    # Check significant files
    for file_path in changed_files:
        if not file_path:
            continue
            
        # Skip ignored files
        if any(file_path.startswith(ignore) for ignore in ignore_paths):
            continue
            
        # Check if file is in critical paths
        if any(file_path.startswith(critical) for critical in critical_paths):
            significant_files.append(file_path)
    
    # Calculate total lines changed
    for added, deleted, filename in diff_stats:
        # Skip ignored files in stats
        if not any(filename.startswith(ignore) for ignore in ignore_paths):
            total_lines_changed += added + deleted
    
    # Determine significance
    is_significant = (
        len(significant_files) > 0 or  # Any critical files changed
        total_lines_changed > 20  # More than 20 lines changed total
    )
    
    return is_significant, significant_files, total_lines_changed

def check_changelog_updated(changed_files):
    """Check if CHANGELOG.md was updated in the PR."""
    return 'CHANGELOG.md' in changed_files

def analyze_changes(significant_files):
    """Analyze the types of changes made."""
    change_types = []
    
    categories = {
        'Core functionality': ['src/', 'pynomaly/'],
        'Examples': ['examples/'],
        'Documentation': ['docs/', 'README.md'],
        'Testing': ['tests/'],
        'Scripts': ['scripts/'],
        'Infrastructure': ['docker/', 'Dockerfile', '.github/'],
        'Dependencies': ['pyproject.toml', 'requirements.txt']
    }
    
    for category, patterns in categories.items():
        if any(any(f.startswith(pattern) for pattern in patterns) for f in significant_files):
            change_types.append(category)
    
    return change_types

def main():
    """Main PR changelog checker."""
    print("🔍 Checking if CHANGELOG.md update is required for this PR...")
    
    # Read changed files and diff stats
    changed_files = read_file_list('changed_files.txt')
    diff_stats = parse_diff_stats('diff_stats.txt')
    
    if not changed_files:
        print("✅ No files changed in PR")
        return 0
    
    print(f"📁 Files changed: {len(changed_files)}")
    
    # Check if changes are significant
    is_significant, significant_files, total_lines_changed = is_significant_change_pr(changed_files, diff_stats)
    
    if not is_significant:
        print("✅ Changes don't require CHANGELOG.md update")
        print(f"   Changed files: {', '.join(changed_files[:3])}")
        if len(changed_files) > 3:
            print(f"   ... and {len(changed_files) - 3} more")
        return 0
    
    # Check if changelog was updated
    changelog_updated = check_changelog_updated(changed_files)
    
    if changelog_updated:
        print("✅ CHANGELOG.md has been updated in this PR")
        return 0
    
    # Changelog update required but missing
    print("❌ CHANGELOG.md update required but missing!")
    print(f"   Significant files changed: {len(significant_files)}")
    print(f"   Total lines changed: {total_lines_changed}")
    
    # Analyze change types
    change_types = analyze_changes(significant_files)
    if change_types:
        print(f"   Change categories: {', '.join(change_types)}")
    
    print(f"   Key files changed:")
    for file in significant_files[:10]:  # Show first 10 files
        print(f"     - {file}")
    if len(significant_files) > 10:
        print(f"     ... and {len(significant_files) - 10} more")
    
    print(f"\n💡 To fix this:")
    print(f"   1. Run: python3 scripts/update_changelog_helper.py")
    print(f"   2. Update CHANGELOG.md with details about your changes")
    print(f"   3. Follow the format in CLAUDE.md > Changelog Management Rules")
    print(f"   4. Commit the changelog update to this PR")
    
    # Suggest sections based on change types
    if change_types:
        print(f"\n📝 Suggested changelog sections based on your changes:")
        section_mapping = {
            'Core functionality': 'Added/Changed/Fixed',
            'Examples': 'Added/Documentation', 
            'Documentation': 'Documentation',
            'Testing': 'Testing',
            'Scripts': 'Infrastructure',
            'Infrastructure': 'Infrastructure',
            'Dependencies': 'Changed'
        }
        
        suggested_sections = set()
        for change_type in change_types:
            if change_type in section_mapping:
                suggested_sections.add(section_mapping[change_type])
        
        for section in suggested_sections:
            print(f"   - {section}")
    
    return 1

if __name__ == "__main__":
    sys.exit(main())