#!/usr/bin/env python3
"""
Fix broken documentation links in Pynomaly documentation.
This script identifies and fixes common broken link patterns.
"""

import os
import re
import sys
from pathlib import Path

def find_and_fix_links(root_dir):
    """Find and fix broken documentation links."""
    fixes_made = 0
    
    # Common link patterns to fix
    link_fixes = {
        # Missing quickstart.md - should point to getting-started/quickstart.md
        r'\[([^\]]+)\]\(quickstart\.md\)': r'[\1](getting-started/quickstart.md)',
        r'\[([^\]]+)\]\(\.\./quickstart\.md\)': r'[\1](../getting-started/quickstart.md)',
        r'\[([^\]]+)\]\(\.\./\.\./quickstart\.md\)': r'[\1](../../getting-started/quickstart.md)',
        
        # Fix user-guides paths
        r'\[([^\]]+)\]\(\.\./user-guides/basic-usage/([^)]+)\)': r'[\1](../user-guides/basic-usage/\2)',
        r'\[([^\]]+)\]\(\.\./\.\./user-guides/basic-usage/([^)]+)\)': r'[\1](../../user-guides/basic-usage/\2)',
        
        # Fix CLI reference paths
        r'\[([^\]]+)\]\(\.\./cli/command-reference\.md\)': r'[\1](../cli/command-reference.md)',
        r'\[([^\]]+)\]\(cli/command-reference\.md\)': r'[\1](cli/command-reference.md)',
        
        # Fix algorithm reference paths
        r'\[([^\]]+)\]\(\.\./reference/algorithms/([^)]+)\)': r'[\1](../reference/algorithms/\2)',
        r'\[([^\]]+)\]\(\.\./\.\./reference/algorithms/([^)]+)\)': r'[\1](../../reference/algorithms/\2)',
        
        # Fix deployment paths
        r'\[([^\]]+)\]\(\.\./deployment/([^)]+)\)': r'[\1](../deployment/\2)',
        r'\[([^\]]+)\]\(\.\./\.\./deployment/([^)]+)\)': r'[\1](../../deployment/\2)',
        
        # Fix examples paths  
        r'\[([^\]]+)\]\(\.\./examples/([^)]+)\)': r'[\1](../examples/\2)',
        r'\[([^\]]+)\]\(\.\./\.\./examples/([^)]+)\)': r'[\1](../../examples/\2)',
        
        # Fix developer-guides paths
        r'\[([^\]]+)\]\(\.\./developer-guides/([^)]+)\)': r'[\1](../developer-guides/\2)',
        r'\[([^\]]+)\]\(\.\./\.\./developer-guides/([^)]+)\)': r'[\1](../../developer-guides/\2)',
    }
    
    # Walk through all markdown files
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # Apply fixes
                    for pattern, replacement in link_fixes.items():
                        content = re.sub(pattern, replacement, content)
                    
                    # Write back if changed
                    if content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        fixes_made += 1
                        print(f"Fixed links in: {file_path}")
                
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    return fixes_made

def verify_critical_files():
    """Verify that critical documentation files exist."""
    docs_dir = Path("/mnt/c/Users/andre/Pynomaly/docs")
    
    critical_files = [
        "getting-started/quickstart.md",
        "user-guides/README.md", 
        "user-guides/basic-usage/README.md",
        "user-guides/basic-usage/autonomous-mode.md",
        "user-guides/basic-usage/datasets.md",
        "user-guides/basic-usage/monitoring.md",
        "user-guides/advanced-features/README.md",
        "user-guides/troubleshooting/README.md",
        "examples/README.md",
        "examples/banking/Banking_Anomaly_Detection_Guide.md",
        "examples/tutorials/README.md",
    ]
    
    missing_files = []
    for file_path in critical_files:
        full_path = docs_dir / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("Missing critical files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
    else:
        print("All critical files exist!")
    
    return missing_files

if __name__ == "__main__":
    docs_root = "/mnt/c/Users/andre/Pynomaly/docs"
    
    print("Verifying critical documentation files...")
    missing = verify_critical_files()
    
    print("\nFixing documentation links...")
    fixes = find_and_fix_links(docs_root)
    
    print(f"\nSummary:")
    print(f"- Missing critical files: {len(missing)}")
    print(f"- Files with fixed links: {fixes}")
    
    if missing:
        print("\nPlease create missing files or update links to point to existing files.")