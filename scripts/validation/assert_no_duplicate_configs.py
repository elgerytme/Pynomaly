#!/usr/bin/env python3
"""
Utility script to detect duplicate configuration files in the project.

This script scans the project directory for configuration files and detects
duplicates based on filename and content hash. It helps maintain a clean
project structure by identifying potentially redundant configuration files.
"""

from pathlib import Path
import sys
import hashlib


def main():
    """Main function to scan for duplicate configuration files."""
    ROOT = Path(__file__).resolve().parents[2]
    config_patterns = ["*.yml", "*.yaml", "*.json", "*.toml", "*.ini"]
    files = [p for pattern in config_patterns for p in ROOT.rglob(pattern)]
    
    hash_map = {}
    dup_list = []
    
    for p in files:
        # Skip files in certain directories that might contain test or temp files
        if any(part in p.parts for part in ['.git', '__pycache__', '.pytest_cache', 'node_modules']):
            continue
            
        try:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            key = (p.name.lower(), h)
            
            if key in hash_map:
                dup_list.append((hash_map[key], p))
            else:
                hash_map[key] = p
        except (OSError, IOError) as e:
            print(f"Warning: Could not read file {p}: {e}", file=sys.stderr)
            continue
    
    if dup_list:
        print("❌ Duplicate configuration files found:")
        for a, b in dup_list:
            print(f"  - {a.relative_to(ROOT)}  <==>  {b.relative_to(ROOT)}")
        sys.exit(1)
    
    print("✅ No duplicate configuration files detected.")


if __name__ == "__main__":
    main()
