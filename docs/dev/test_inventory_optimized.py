#!/usr/bin/env python3
"""
Test Files Inventory Script

This script creates a comprehensive inventory of all test files with:
- File size
- Last modified date
- Whether the file is imported by other files (using ripgrep for performance)
- File type statistics
"""

import os
import subprocess
import time
from pathlib import Path
from collections import defaultdict
import json

def run_ripgrep_search(pattern, directory):
    """Use ripgrep to search for import patterns efficiently"""
    try:
        result = subprocess.run([
            'rg', '--type', 'py', '--no-heading', '--line-number', 
            pattern, str(directory)
        ], capture_output=True, text=True, timeout=30)
        return result.stdout.strip().split('\n') if result.stdout.strip() else []
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        # Fallback to grep if ripgrep is not available
        try:
            result = subprocess.run([
                'grep', '-r', '--include=*.py', '-n', pattern, str(directory)
            ], capture_output=True, text=True, timeout=30)
            return result.stdout.strip().split('\n') if result.stdout.strip() else []
        except:
            return []

def check_if_imported(file_path, test_dir):
    """Check if a Python file is imported by other files"""
    if not file_path.suffix == '.py':
        return False
    
    # Convert file path to module name
    try:
        relative_path = file_path.relative_to(test_dir.parent)
        module_name = str(relative_path.with_suffix('')).replace(os.sep, '.')
        
        # Search for direct imports
        import_patterns = [
            f'import {module_name}',
            f'from {module_name}',
            f'import.*{module_name}',
            f'from.*{module_name}'
        ]
        
        for pattern in import_patterns:
            matches = run_ripgrep_search(pattern, test_dir.parent)
            if matches and matches != ['']:
                # Filter out self-imports
                for match in matches:
                    if ':' in match:
                        match_file = match.split(':')[0]
                        if Path(match_file) != file_path:
                            return True
        return False
    except:
        return False

def get_file_stats(test_dir):
    """Get statistics about file types and sizes"""
    stats = {
        'total_files': 0,
        'by_extension': defaultdict(int),
        'by_size_range': defaultdict(int),
        'largest_files': [],
        'oldest_files': [],
        'newest_files': []
    }
    
    all_files = []
    
    for file_path in test_dir.rglob('*'):
        if file_path.is_file():
            try:
                file_size = file_path.stat().st_size
                mod_time = file_path.stat().st_mtime
                
                stats['total_files'] += 1
                stats['by_extension'][file_path.suffix or 'no_extension'] += 1
                
                # Size ranges
                if file_size < 1024:
                    stats['by_size_range']['< 1KB'] += 1
                elif file_size < 10240:
                    stats['by_size_range']['1KB - 10KB'] += 1
                elif file_size < 102400:
                    stats['by_size_range']['10KB - 100KB'] += 1
                else:
                    stats['by_size_range']['> 100KB'] += 1
                
                all_files.append((file_path, file_size, mod_time))
            except:
                continue
    
    # Sort and get top files
    all_files.sort(key=lambda x: x[1], reverse=True)
    stats['largest_files'] = all_files[:10]
    
    all_files.sort(key=lambda x: x[2])
    stats['oldest_files'] = all_files[:5]
    stats['newest_files'] = all_files[-5:]
    
    return stats

def main():
    # Directory containing the test files
    test_dir = Path("C:/Users/andre/Pynomaly/tests")
    
    # Path to save the inventory
    output_file = Path("C:/Users/andre/Pynomaly/docs/dev/test_inventory.md")
    
    print("Starting test file inventory...")
    
    # Find all test files (only files, not directories)
    test_files = [f for f in test_dir.rglob('*') if f.is_file()]
    
    print(f"Found {len(test_files)} test files...")
    
    # Get file statistics
    print("Generating file statistics...")
    stats = get_file_stats(test_dir)
    
    # Initialize inventory
    inventory = []
    imported_count = 0
    
    print("Processing files for imports...")
    
    for i, file_path in enumerate(test_files):
        if i % 50 == 0:
            print(f"Processed {i}/{len(test_files)} files...")
        
        try:
            # Get file size and last modified time
            file_size = file_path.stat().st_size
            last_modified_time = time.ctime(file_path.stat().st_mtime)
            
            # Check if file is imported by other files
            is_imported = check_if_imported(file_path, test_dir)
            if is_imported:
                imported_count += 1
            
            # Format file path to be relative to project root
            rel_path = file_path.relative_to(test_dir.parent)
            inventory.append({
                'path': str(rel_path),
                'size': file_size,
                'modified': last_modified_time,
                'imported': is_imported
            })
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    print(f"Processed {len(inventory)} files successfully")
    print(f"Found {imported_count} files that are imported by other files")
    
    # Write inventory to markdown file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Test Files Inventory\n\n")
        f.write("This inventory contains all files in the tests directory with their properties.\n\n")
        
        # Write statistics
        f.write("## Statistics\n\n")
        f.write(f"- **Total Files**: {stats['total_files']}\n")
        f.write(f"- **Files with Imports**: {imported_count}\n")
        f.write(f"- **Import Percentage**: {(imported_count/len(inventory)*100):.1f}%\n\n")
        
        # File type breakdown
        f.write("### File Types\n\n")
        for ext, count in sorted(stats['by_extension'].items(), key=lambda x: x[1], reverse=True):
            f.write(f"- **{ext}**: {count} files\n")
        f.write("\n")
        
        # Size breakdown
        f.write("### File Size Distribution\n\n")
        for size_range, count in stats['by_size_range'].items():
            f.write(f"- **{size_range}**: {count} files\n")
        f.write("\n")
        
        # Largest files
        f.write("### Largest Files\n\n")
        for file_path, size, _ in stats['largest_files']:
            rel_path = file_path.relative_to(test_dir.parent)
            f.write(f"- **{rel_path}**: {size:,} bytes\n")
        f.write("\n")
        
        # Main inventory table
        f.write("## Complete File Inventory\n\n")
        f.write("| File Path | Size (bytes) | Last Modified | Imported By Other Files |\n")
        f.write("|-----------|-------------|---------------|-------------------------|\n")
        
        for item in inventory:
            f.write(f"| {item['path']} | {item['size']:,} | {item['modified']} | {item['imported']} |\n")
    
    print(f"Inventory saved to {output_file}")
    
    # Also save as JSON for programmatic access
    json_file = output_file.with_suffix('.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'stats': dict(stats),
            'inventory': inventory,
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2, default=str)
    
    print(f"JSON inventory saved to {json_file}")

if __name__ == "__main__":
    main()
