#!/usr/bin/env python3
"""
Script to migrate Pydantic class-based Config to ConfigDict for V3.0 compatibility.
This addresses GitHub Issue #199: Tech Debt: Update Pydantic Configurations for V3.0 Compatibility
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

def find_files_with_class_config(root_dir: str) -> List[Path]:
    """Find all Python files containing 'class Config:' pattern."""
    files_with_config = []
    
    for root, dirs, files in os.walk(root_dir):
        # Skip virtual environments and other non-source directories
        dirs[:] = [d for d in dirs if d not in ['venv', '.venv', 'env', '.env', 'node_modules', '__pycache__', '.git']]
        
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'class Config:' in content:
                            files_with_config.append(file_path)
                except (UnicodeDecodeError, PermissionError):
                    continue
    
    return files_with_config

def analyze_config_content(content: str) -> List[Tuple[str, str]]:
    """Analyze the content of class Config and extract configuration options."""
    configs = []
    
    # Find all class Config blocks
    config_pattern = re.compile(r'(\s+)class Config:\s*\n((?:\1[ ]+.*\n?)*)', re.MULTILINE)
    matches = config_pattern.findall(content)
    
    for indent, config_body in matches:
        config_options = []
        
        # Parse configuration options
        lines = config_body.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('"""') or line.startswith('#'):
                continue
            if line.startswith('"""') and line.endswith('"""'):
                continue
            if '=' in line:
                config_options.append(line)
        
        if config_options:
            configs.append((indent, '\n'.join(config_options)))
    
    return configs

def convert_config_to_configdict(config_content: str) -> str:
    """Convert class Config content to ConfigDict format."""
    lines = config_content.split('\n')
    converted_options = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Handle common configuration patterns
        if '=' in line:
            # Handle json_schema_extra
            if 'json_schema_extra' in line:
                converted_options.append(line)
            # Handle json_encoders
            elif 'json_encoders' in line:
                converted_options.append(line)
            # Handle simple assignments
            elif any(attr in line for attr in [
                'allow_mutation', 'validate_assignment', 'arbitrary_types_allowed',
                'use_enum_values', 'validate_all', 'extra', 'frozen', 'allow_reuse'
            ]):
                converted_options.append(line)
            else:
                converted_options.append(line)
    
    return ',\n        '.join(converted_options)

def migrate_file(file_path: Path) -> bool:
    """Migrate a single file from class Config to ConfigDict."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        content = original_content
        
        # Check if ConfigDict is already imported
        needs_configdict_import = 'ConfigDict' not in content
        
        # Find and replace class Config patterns
        config_pattern = re.compile(r'(\s+)class Config:\s*\n((?:\1[ ]+.*\n?)*)', re.MULTILINE)
        
        def replace_config(match):
            indent = match.group(1)
            config_body = match.group(2)
            
            # Extract configuration options
            config_options = []
            lines = config_body.split('\n')
            
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith('"""') or stripped.startswith('#'):
                    continue
                if stripped.startswith('"""') and stripped.endswith('"""'):
                    continue
                if '=' in stripped:
                    config_options.append(stripped)
            
            if not config_options:
                return ''
            
            # Convert to ConfigDict format
            config_dict_content = ',\n        '.join(config_options)
            
            return f'{indent}model_config = ConfigDict(\n        {config_dict_content}\n    )'
        
        # Replace all class Config occurrences
        new_content = config_pattern.sub(replace_config, content)
        
        # Add ConfigDict import if needed and if we made changes
        if new_content != content and needs_configdict_import:
            # Find pydantic import line and add ConfigDict
            pydantic_import_pattern = re.compile(r'from pydantic import ([^)]+)')
            
            def add_configdict_import(match):
                imports = match.group(1)
                if 'ConfigDict' not in imports:
                    # Clean up the imports and add ConfigDict
                    import_list = [imp.strip() for imp in imports.split(',')]
                    if 'ConfigDict' not in import_list:
                        import_list.append('ConfigDict')
                    return f'from pydantic import {", ".join(sorted(import_list))}'
                return match.group(0)
            
            new_content = pydantic_import_pattern.sub(add_configdict_import, new_content)
        
        # Only write if content changed
        if new_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main migration function."""
    root_dir = "src"
    
    if not os.path.exists(root_dir):
        print(f"Directory {root_dir} not found!")
        sys.exit(1)
    
    print("Finding files with class Config pattern...")
    files_with_config = find_files_with_class_config(root_dir)
    
    print(f"Found {len(files_with_config)} files with class Config pattern")
    
    migrated_count = 0
    
    for file_path in files_with_config:
        print(f"Processing: {file_path}")
        if migrate_file(file_path):
            migrated_count += 1
            print(f"  ✓ Migrated")
        else:
            print(f"  - No changes needed")
    
    print(f"\nMigration complete!")
    print(f"Migrated {migrated_count} files out of {len(files_with_config)} total files")

if __name__ == "__main__":
    main()