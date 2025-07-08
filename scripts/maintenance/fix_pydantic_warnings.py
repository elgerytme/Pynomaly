#!/usr/bin/env python3
"""
Script to fix Pydantic v2 deprecation warnings by updating deprecated syntax.
"""

import os
import re
from pathlib import Path


def fix_file(file_path):
    """Fix Pydantic v2 deprecation warnings in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix min_items -> min_length
        content = re.sub(r'\bmin_items\s*=', 'min_length=', content)
        
        # Fix max_items -> max_length
        content = re.sub(r'\bmax_items\s*=', 'max_length=', content)
        
        # Fix regex -> pattern
        content = re.sub(r'\bregex\s*=', 'pattern=', content)
        
        # Fix class Config pattern
        content = re.sub(
            r'class\s+Config\s*:\s*\n\s*([^\n]+)',
            r'model_config = ConfigDict(\1)',
            content,
            flags=re.MULTILINE
        )
        
        # Fix json_encoders usage
        content = re.sub(
            r'json_encoders\s*=\s*{[^}]*}',
            'json_schema_extra={}',
            content
        )
        
        # Fix enum usage in Field
        content = re.sub(
            r'Field\(([^,)]+),\s*enum\s*=\s*([^,)]+)',
            r'Field(\1, json_schema_extra={"enum": \2}',
            content
        )
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {file_path}")
            return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False
    
    return False


def main():
    """Main function to process all Python files."""
    src_dir = Path("src")
    if not src_dir.exists():
        print("src directory not found")
        return
    
    fixed_count = 0
    
    # Process all Python files
    for py_file in src_dir.rglob("*.py"):
        if fix_file(py_file):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")


if __name__ == "__main__":
    main()
