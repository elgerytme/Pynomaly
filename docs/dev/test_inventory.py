import os
import glob
import time
import re

# Directory containing the test files
test_dir = "C:/Users/andre/Pynomaly/tests"

# Path to save the inventory
output_file = "C:/Users/andre/Pynomaly/docs/dev/test_inventory.md"

# Find all test files
test_files = glob.glob(os.path.join(test_dir, '**', '*'), recursive=True)

# Initialize inventory
inventory = []

for file_path in test_files:
    if os.path.isfile(file_path):
        # Get file size and last modified time
        file_size = os.path.getsize(file_path)
        last_modified_time = time.ctime(os.path.getmtime(file_path))
        
        # Check if file is imported by other files
        is_imported = False
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            imports = re.findall(r'from\s+([\w.]+)\s+import|import\s+([\w.]+)', content)
            import_statements = set(sum(imports, ()))
            # Simple heuristic to check for imports
            for other_file in test_files:
                if other_file != file_path:
                    with open(other_file, 'r', encoding='utf-8', errors='ignore') as other_f:
                        other_content = other_f.read()
                        if any(im in other_content for im in import_statements):
                            is_imported = True
                            break

        inventory.append(f"| {file_path} | {file_size} | {last_modified_time} | {is_imported} |")

# Write inventory to markdown file
with open(output_file, 'w') as f:
    f.write("| File Path | Size (bytes) | Last Modified | Imported By Other Files |\n")
    f.write("|-----------|-------------|---------------|-------------------------|\n")
    f.write("\n".join(inventory))
