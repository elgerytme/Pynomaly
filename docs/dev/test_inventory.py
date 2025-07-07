import os
import glob
import time
import re
from pathlib import Path

# Directory containing the test files
test_dir = Path("C:/Users/andre/Pynomaly/tests")

# Path to save the inventory
output_file = Path("C:/Users/andre/Pynomaly/docs/dev/test_inventory.md")

# Find all test files (only files, not directories)
test_files = [f for f in test_dir.rglob('*') if f.is_file()]

# Initialize inventory
inventory = []

print(f"Found {len(test_files)} test files...")

for file_path in test_files:
    try:
        # Get file size and last modified time
        file_size = file_path.stat().st_size
        last_modified_time = time.ctime(file_path.stat().st_mtime)
        
        # Check if file is imported by other files
        is_imported = False
        
        # Only check Python files for imports
        if file_path.suffix == '.py':
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # Extract module name from file path
                    relative_path = file_path.relative_to(test_dir.parent)
                    module_name = str(relative_path.with_suffix('')).replace(os.sep, '.')
                    
                    # Check if this module is imported by other files
                    for other_file in test_files:
                        if other_file != file_path and other_file.suffix == '.py':
                            try:
                                with open(other_file, 'r', encoding='utf-8', errors='ignore') as other_f:
                                    other_content = other_f.read()
                                    if (f"import {module_name}" in other_content or 
                                        f"from {module_name}" in other_content):
                                        is_imported = True
                                        break
                            except Exception:
                                continue
            except Exception:
                pass
        
        # Format file path to be relative to project root
        rel_path = file_path.relative_to(test_dir.parent)
        inventory.append(f"| {rel_path} | {file_size} | {last_modified_time} | {is_imported} |")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        continue

print(f"Processed {len(inventory)} files successfully")

# Write inventory to markdown file
with open(output_file, 'w') as f:
    f.write("# Test Files Inventory\n\n")
    f.write("This inventory contains all files in the tests directory with their properties.\n\n")
    f.write("| File Path | Size (bytes) | Last Modified | Imported By Other Files |\n")
    f.write("|-----------|-------------|---------------|-------------------------|\n")
    f.write("\n".join(inventory))

print(f"Inventory saved to {output_file}")
