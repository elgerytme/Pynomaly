#!/usr/bin/env python3
"""
Changelog Update Helper

Interactive script to help create properly formatted changelog entries
following the project's changelog management rules.
"""

import sys
import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

def get_current_version():
    """Get current version from CHANGELOG.md or pyproject.toml."""
    
    # Try to get version from CHANGELOG.md first
    changelog_path = Path('CHANGELOG.md')
    if changelog_path.exists():
        with open(changelog_path, 'r') as f:
            content = f.read()
        
        # Look for version pattern [X.Y.Z]
        version_pattern = r'\[(\d+\.\d+\.\d+)\]'
        matches = re.findall(version_pattern, content)
        
        if matches:
            latest_version = matches[0]
            # Parse version components
            major, minor, patch = map(int, latest_version.split('.'))
            return major, minor, patch
    
    # Fallback to 0.1.0 if no version found
    return 0, 1, 0

def increment_version(major: int, minor: int, patch: int, change_type: str) -> tuple:
    """Increment version based on change type."""
    if change_type == 'major':
        return major + 1, 0, 0
    elif change_type == 'minor':
        return major, minor + 1, 0
    elif change_type == 'patch':
        return major, minor, patch + 1
    else:
        return major, minor, patch

def get_change_categories():
    """Get available change categories."""
    return {
        'added': 'Added - New features, capabilities, or functionality',
        'changed': 'Changed - Changes in existing functionality or behavior',
        'deprecated': 'Deprecated - Soon-to-be removed features',
        'removed': 'Removed - Features removed in this release',
        'fixed': 'Fixed - Bug fixes and issue resolutions',
        'security': 'Security - Security-related changes and vulnerability fixes',
        'performance': 'Performance - Performance improvements and optimizations',
        'documentation': 'Documentation - Documentation additions, improvements, or restructuring',
        'infrastructure': 'Infrastructure - CI/CD, build system, or deployment changes',
        'testing': 'Testing - Test additions, improvements, or infrastructure changes'
    }

def get_user_input(prompt: str, options: Optional[List[str]] = None) -> str:
    """Get user input with optional validation."""
    while True:
        if options:
            print(f"\nOptions: {', '.join(options)}")
        
        response = input(f"{prompt}: ").strip()
        
        if not response:
            print("Please provide a response.")
            continue
        
        if options and response.lower() not in [opt.lower() for opt in options]:
            print(f"Please choose from: {', '.join(options)}")
            continue
        
        return response

def get_change_type():
    """Determine the type of change for versioning."""
    print("\n🔢 What type of changes are you making?")
    print("1. patch - Bug fixes, small improvements (X.Y.Z+1)")
    print("2. minor - New features, backwards compatible (X.Y+1.0)")
    print("3. major - Breaking changes, major features (X+1.0.0)")
    
    change_types = ['patch', 'minor', 'major']
    choice = get_user_input("Select change type", ['1', '2', '3'])
    
    return change_types[int(choice) - 1]

def collect_changelog_entries():
    """Collect changelog entries from user."""
    categories = get_change_categories()
    entries = {}
    
    print("\n📝 Let's create your changelog entries.")
    print("For each category, provide entries (one per line). Press Enter twice when done.")
    
    for category_key, category_desc in categories.items():
        print(f"\n{category_desc}")
        category_entries = []
        
        while True:
            entry = input(f"  - ").strip()
            if not entry:
                break
            category_entries.append(entry)
        
        if category_entries:
            entries[category_key] = category_entries
    
    return entries

def format_changelog_entry(version: str, date: str, entries: Dict[str, List[str]]) -> str:
    """Format changelog entry according to project standards."""
    
    # Map category keys to display names
    category_names = {
        'added': 'Added',
        'changed': 'Changed',
        'deprecated': 'Deprecated',
        'removed': 'Removed',
        'fixed': 'Fixed',
        'security': 'Security',
        'performance': 'Performance',
        'documentation': 'Documentation',
        'infrastructure': 'Infrastructure',
        'testing': 'Testing'
    }
    
    lines = [f"## [{version}] - {date}", ""]
    
    # Order categories by importance
    category_order = ['added', 'changed', 'deprecated', 'removed', 'fixed', 'security', 
                     'performance', 'documentation', 'infrastructure', 'testing']
    
    for category in category_order:
        if category in entries and entries[category]:
            lines.append(f"### {category_names[category]}")
            for entry in entries[category]:
                lines.append(f"- {entry}")
            lines.append("")
    
    return "\n".join(lines)

def update_changelog_file(new_entry: str):
    """Update CHANGELOG.md with new entry."""
    changelog_path = Path('CHANGELOG.md')
    
    if not changelog_path.exists():
        print("❌ CHANGELOG.md not found!")
        return False
    
    # Read existing content
    with open(changelog_path, 'r') as f:
        content = f.read()
    
    # Find the position to insert new entry
    # Look for ## [Unreleased] or first ## [version] entry
    unreleased_pattern = r'## \[Unreleased\]'
    version_pattern = r'## \[\d+\.\d+\.\d+\]'
    
    unreleased_match = re.search(unreleased_pattern, content)
    version_match = re.search(version_pattern, content)
    
    if unreleased_match:
        # Insert after [Unreleased] section
        insert_pos = unreleased_match.end()
        # Find the next section
        next_section = content.find('\n## ', insert_pos)
        if next_section != -1:
            insert_pos = next_section
        else:
            insert_pos = len(content)
        
        # Insert new entry
        new_content = content[:insert_pos] + "\n\n" + new_entry + content[insert_pos:]
    
    elif version_match:
        # Insert before first version entry
        insert_pos = version_match.start()
        new_content = content[:insert_pos] + new_entry + "\n\n" + content[insert_pos:]
    
    else:
        # Append to end of file
        new_content = content + "\n\n" + new_entry
    
    # Write updated content
    with open(changelog_path, 'w') as f:
        f.write(new_content)
    
    return True

def main():
    """Main changelog update helper."""
    print("📋 CHANGELOG.MD UPDATE HELPER")
    print("=" * 50)
    print("This script helps you create properly formatted changelog entries")
    print("following Pynomaly's changelog management rules.")
    
    # Get current version
    major, minor, patch = get_current_version()
    current_version = f"{major}.{minor}.{patch}"
    print(f"\n📊 Current version: {current_version}")
    
    # Get change type
    change_type = get_change_type()
    
    # Calculate new version
    new_major, new_minor, new_patch = increment_version(major, minor, patch, change_type)
    new_version = f"{new_major}.{new_minor}.{new_patch}"
    
    print(f"📈 New version: {new_version}")
    
    # Confirm version
    confirm = get_user_input(f"Use version {new_version}?", ['y', 'n'])
    if confirm.lower() == 'n':
        custom_version = get_user_input("Enter custom version (X.Y.Z)")
        # Validate version format
        if not re.match(r'^\d+\.\d+\.\d+$', custom_version):
            print("❌ Invalid version format. Use X.Y.Z")
            return 1
        new_version = custom_version
    
    # Get today's date
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Collect changelog entries
    entries = collect_changelog_entries()
    
    if not entries:
        print("❌ No changelog entries provided!")
        return 1
    
    # Format changelog entry
    changelog_entry = format_changelog_entry(new_version, today, entries)
    
    # Preview entry
    print(f"\n📖 Preview of changelog entry:")
    print("=" * 50)
    print(changelog_entry)
    print("=" * 50)
    
    # Confirm and update
    confirm = get_user_input("Add this entry to CHANGELOG.md?", ['y', 'n'])
    if confirm.lower() == 'y':
        if update_changelog_file(changelog_entry):
            print("✅ CHANGELOG.md updated successfully!")
            print(f"📝 Added version {new_version} entry")
            
            # Remind about TODO.md update
            print(f"\n💡 Don't forget to:")
            print(f"   1. Update TODO.md to mark completed items")
            print(f"   2. Commit both CHANGELOG.md and TODO.md together")
            print(f"   3. Consider creating a git tag: git tag v{new_version}")
            
            return 0
        else:
            print("❌ Failed to update CHANGELOG.md")
            return 1
    else:
        print("❌ Changelog update cancelled")
        return 1

if __name__ == "__main__":
    sys.exit(main())