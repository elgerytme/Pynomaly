#!/usr/bin/env python3
"""Script to fix duplicate model_config declarations in DTO files."""

import re
from pathlib import Path

def fix_duplicate_model_configs():
    """Fix duplicate model_config declarations in all DTO files."""
    dto_dir = Path("src/pynomaly/application/dto")
    
    # Pattern to match the problematic duplicate declarations
    pattern = r'(model_config = ConfigDict\(extra="forbid"\))\s*\n\s*(model_config = ConfigDict\(from_attributes=True, extra="forbid"\))'
    
    # Replacement - just keep the second one
    replacement = r'\2'
    
    for py_file in dto_dir.glob("*.py"):
        if py_file.stem == "__init__":
            continue
            
        try:
            content = py_file.read_text(encoding="utf-8")
            
            # Check if file has the problematic pattern
            if re.search(pattern, content):
                print(f"Fixing {py_file.name}")
                
                # Fix the duplicate declarations
                fixed_content = re.sub(pattern, replacement, content)
                
                # Write back
                py_file.write_text(fixed_content, encoding="utf-8")
                print(f"Fixed {py_file.name}")
            else:
                print(f"No issues found in {py_file.name}")
                
        except Exception as e:
            print(f"Error processing {py_file.name}: {e}")

if __name__ == "__main__":
    fix_duplicate_model_configs()
