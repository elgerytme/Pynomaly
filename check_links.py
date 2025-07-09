#!/usr/bin/env python3
"""Simple link checker for documentation files."""

import os
import re
from pathlib import Path
from typing import List, Tuple, Set

def find_markdown_links(content: str) -> List[Tuple[str, str]]:
    """Find all markdown links in content."""
    # Pattern to match [text](link)
    pattern = r'\[([^\]]*)\]\(([^)]+)\)'
    return re.findall(pattern, content)

def check_file_exists(link: str, base_path: Path) -> bool:
    """Check if a file link exists."""
    if link.startswith(('http://', 'https://', 'mailto:', '#')):
        return True  # Skip external links and anchors
    
    # Remove anchor from link
    link = link.split('#')[0]
    
    # Make path relative to base
    full_path = base_path / link
    return full_path.exists()

def check_markdown_file(file_path: Path) -> List[str]:
    """Check all links in a markdown file."""
    errors = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return [f"Error reading {file_path}: {e}"]
    
    links = find_markdown_links(content)
    base_path = file_path.parent
    
    for text, link in links:
        if not check_file_exists(link, base_path):
            errors.append(f"Broken link in {file_path}: [{text}]({link})")
    
    return errors

def main():
    """Main link checker."""
    project_docs = Path('docs/project')
    
    if not project_docs.exists():
        print(f"Project docs directory not found: {project_docs}")
        return
    
    all_errors = []
    
    # Check the main documentation files
    main_files = [
        'requirements/REQUIREMENTS.md',
        'FEATURE_BACKLOG.md',
        'DEVELOPMENT_ROADMAP.md'
    ]
    
    for file_name in main_files:
        file_path = project_docs / file_name
        if file_path.exists():
            errors = check_markdown_file(file_path)
            all_errors.extend(errors)
        else:
            all_errors.append(f"File not found: {file_path}")
    
    if all_errors:
        print("Found broken links:")
        for error in all_errors:
            print(f"  {error}")
    else:
        print("All links checked successfully!")

if __name__ == '__main__':
    main()
