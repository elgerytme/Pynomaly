#!/usr/bin/env python3
"""Comprehensive fix script for UserModel imports."""

import os
import re

# List of endpoint files to fix
endpoint_files = [
    "src/pynomaly/presentation/api/endpoints/detectors.py",
    "src/pynomaly/presentation/api/endpoints/experiments.py", 
    "src/pynomaly/presentation/api/endpoints/datasets.py",
    "src/pynomaly/presentation/api/endpoints/monitoring.py",
    "src/pynomaly/presentation/api/endpoints/auth.py"
]

def fix_file(filepath):
    """Fix UserModel imports in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix direct UserModel imports in import statements
        content = re.sub(
            r'from pynomaly\.infrastructure\.auth import \(\s*UserModel,\s*',
            'from pynomaly.infrastructure.auth import (\n    ',
            content
        )
        
        # Remove standalone UserModel imports
        content = re.sub(
            r'from pynomaly\.infrastructure\.auth import \(\s*UserModel,([^)]+)\)',
            r'from pynomaly.infrastructure.auth import (\1)',
            content
        )
        
        # Add try-except UserModel import if not already present
        if 'from pynomaly.infrastructure.auth.jwt_auth import UserModel' not in content:
            # Find the infrastructure auth import section
            import_match = re.search(r'from pynomaly\.infrastructure\.auth import \([^)]+\)', content)
            if import_match:
                insert_pos = import_match.end()
                usermodel_import = '\n\ntry:\n    from pynomaly.infrastructure.auth.jwt_auth import UserModel\nexcept ImportError:\n    # Fallback for testing or when auth is not available\n    UserModel = None'
                content = content[:insert_pos] + usermodel_import + content[insert_pos:]
        
        # Fix TYPE_CHECKING pattern if it exists
        content = re.sub(
            r'if TYPE_CHECKING:\s*\n\s*from pynomaly\.infrastructure\.auth\.jwt_auth import UserModel',
            'try:\n    from pynomaly.infrastructure.auth.jwt_auth import UserModel\nexcept ImportError:\n    # Fallback for testing or when auth is not available\n    UserModel = None',
            content
        )
        
        # Replace function parameter annotations
        content = re.sub(
            r'(\w+): UserModel = Depends\(',
            r'\1 = Depends(',
            content
        )
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"Fixed {filepath}")
        return True
        
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def main():
    """Fix all endpoint files."""
    success_count = 0
    
    for filepath in endpoint_files:
        if os.path.exists(filepath):
            if fix_file(filepath):
                success_count += 1
        else:
            print(f"File not found: {filepath}")
    
    print(f"\nFixed {success_count} out of {len(endpoint_files)} files")

if __name__ == "__main__":
    main()
