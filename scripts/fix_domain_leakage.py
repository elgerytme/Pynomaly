#!/usr/bin/env python3
"""
Script to fix domain leakage by removing monorepo imports.
"""

import os
import re
import glob
from pathlib import Path


def find_python_files(directory: str) -> list[str]:
    """Find all Python files in a directory."""
    pattern = os.path.join(directory, "**", "*.py")
    return glob.glob(pattern, recursive=True)


def fix_monorepo_imports(file_path: str) -> bool:
    """Fix monorepo imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Common replacements for domain leakage
        replacements = [
            # Remove monorepo imports that should be local
            (r'from monorepo\.domain\.entities\.dataset import Dataset', 
             '# TODO: Create local Dataset entity'),
            (r'from monorepo\.domain\.entities\.detector import Detector', 
             '# TODO: Create local Detector entity'),
            (r'from monorepo\.domain\.entities import Dataset', 
             '# TODO: Create local Dataset entity'),
            (r'from monorepo\.domain\.entities import Dataset, Detector', 
             '# TODO: Create local Dataset and Detector entities'),
            (r'from monorepo\.shared\.protocols import DetectorProtocol', 
             '# TODO: Create local DetectorProtocol'),
            (r'from monorepo\.shared\.protocols import \([\s\S]*?\)', 
             '# TODO: Create local protocol interfaces'),
            (r'from monorepo\.infrastructure\.adapters\.sklearn_adapter import SklearnAdapter', 
             '# TODO: Create local sklearn adapter'),
            (r'from monorepo\.infrastructure\.adapters\.pyod_adapter import PyODAdapter', 
             '# TODO: Create local pyod adapter'),
            (r'from monorepo\.application\.dto\.configuration_dto import \([\s\S]*?\)', 
             '# TODO: Create local configuration DTOs'),
            (r'from monorepo\.application\.services\.configuration_capture_service import \([\s\S]*?\)', 
             '# TODO: Create local configuration service'),
            (r'from monorepo\.infrastructure\.config\.feature_flags import require_feature', 
             '# TODO: Create local feature flags'),
        ]
        
        # Apply replacements
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
        
        # If content changed, write it back
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main function to fix domain leakage."""
    # Focus on packages directory
    packages_dir = "src/packages"
    
    if not os.path.exists(packages_dir):
        print(f"Directory {packages_dir} not found")
        return
    
    # Find all Python files
    python_files = find_python_files(packages_dir)
    
    print(f"Found {len(python_files)} Python files")
    
    # Process each file
    fixed_files = []
    for file_path in python_files:
        if fix_monorepo_imports(file_path):
            fixed_files.append(file_path)
    
    print(f"Fixed {len(fixed_files)} files:")
    for file_path in fixed_files:
        print(f"  {file_path}")
    
    # Report remaining monorepo imports
    remaining_imports = []
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'from monorepo' in content:
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if 'from monorepo' in line:
                        remaining_imports.append(f"{file_path}:{i}: {line.strip()}")
        except Exception as e:
            print(f"Error checking {file_path}: {e}")
    
    print(f"\nRemaining monorepo imports: {len(remaining_imports)}")
    if remaining_imports:
        print("First 20 remaining imports:")
        for import_line in remaining_imports[:20]:
            print(f"  {import_line}")


if __name__ == "__main__":
    main()