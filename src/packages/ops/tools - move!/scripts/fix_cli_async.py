#!/usr/bin/env python3
"""
Script to fix asyncio.run() calls in CLI modules by replacing them with cli_runner.run()
"""

import os
import re
import sys
from pathlib import Path

def fix_cli_module(file_path):
    """Fix asyncio.run() calls in a single CLI module."""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Check if async_utils is already imported
    if "from pynomaly.presentation.cli.async_utils import cli_runner" not in content:
        # Add import after other pynomaly imports
        import_pattern = r'(from pynomaly\.presentation\.cli\.container import[^\n]+\n)'
        if re.search(import_pattern, content):
            content = re.sub(
                import_pattern,
                r'\1from pynomaly.presentation.cli.async_utils import cli_runner\n',
                content
            )
        else:
            # Add after rich imports
            rich_pattern = r'(from rich\.[^\n]+\n)'
            last_rich_match = None
            for match in re.finditer(rich_pattern, content):
                last_rich_match = match
            
            if last_rich_match:
                pos = last_rich_match.end()
                content = content[:pos] + '\nfrom pynomaly.presentation.cli.async_utils import cli_runner\n' + content[pos:]
    
    # Replace asyncio.run( with cli_runner.run(
    content = re.sub(r'asyncio\.run\(', 'cli_runner.run(', content)
    
    # Replace use_case.execute(request) patterns with cli_runner.run_use_case(use_case, request)
    content = re.sub(
        r'asyncio\.run\(([^.]+)\.execute\(([^)]+)\)\)',
        r'cli_runner.run_use_case(\1, \2)',
        content
    )
    
    # Remove unused asyncio imports if they're only used for asyncio.run
    if 'asyncio.run' not in content and 'asyncio.create_task' not in content:
        # Check if asyncio is used for other purposes
        if not re.search(r'asyncio\.(?!run)', content):
            content = re.sub(r'import asyncio\n', '', content)
            content = re.sub(r'from asyncio import[^\n]+\n', '', content)
    
    # Only write if content changed
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Fixed: {file_path}")
        return True
    
    return False

def main():
    """Main function to fix all CLI modules."""
    cli_dir = Path("/mnt/c/Users/andre/Pynomaly/src/pynomaly/presentation/cli")
    
    if not cli_dir.exists():
        print(f"CLI directory not found: {cli_dir}")
        return
    
    fixed_files = []
    
    # Process all Python files in CLI directory
    for file_path in cli_dir.glob("*.py"):
        if file_path.name == "__init__.py":
            continue
        if file_path.name == "async_utils.py":
            continue
        if file_path.name == "app.py":
            continue  # Already fixed manually
        if file_path.name == "detection.py":
            continue  # Already fixed manually
        if file_path.name == "automl.py":
            continue  # Already fixed manually
            
        if fix_cli_module(file_path):
            fixed_files.append(file_path)
    
    print(f"\nFixed {len(fixed_files)} CLI modules:")
    for file_path in fixed_files:
        print(f"  {file_path.name}")

if __name__ == "__main__":
    main()