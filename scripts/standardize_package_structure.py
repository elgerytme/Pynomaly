#!/usr/bin/env python3
"""
Script to standardize package directory structure.
"""

import os
import shutil
from pathlib import Path


def standardize_package_structure():
    """Standardize package directory structure."""
    packages_dir = Path("src/packages")
    
    if not packages_dir.exists():
        print(f"Directory {packages_dir} not found")
        return
    
    # Define the standard structure
    standard_structure = [
        "domain",
        "application", 
        "infrastructure",
        "interfaces",
        "tests",
        "docs",
        "examples",
    ]
    
    # Optional directories that may exist
    optional_dirs = [
        "scripts",
        "deploy",
        "web",
        "mobile",
        "cli",
        "api",
        "python_sdk",
        "mlops",
        "use_cases",
    ]
    
    changes_made = []
    
    # Process each package
    for package_path in packages_dir.rglob("*"):
        if package_path.is_dir() and package_path.name in ["src", "anomaly_detection"]:
            # Handle packages with extra src/ layer
            parent = package_path.parent
            
            # Check if this is a problematic src directory
            if (package_path.name == "src" and 
                len(list(package_path.iterdir())) == 1 and
                (package_path / "anomaly_detection").exists()):
                
                # Move contents of src/anomaly_detection up one level
                src_content = package_path / "anomaly_detection"
                
                # Create a backup note
                backup_note = f"# Moved from {package_path}/anomaly_detection to {parent}\n"
                
                try:
                    # Move all contents
                    for item in src_content.iterdir():
                        target = parent / item.name
                        if target.exists():
                            print(f"Warning: {target} already exists, skipping")
                            continue
                        shutil.move(str(item), str(target))
                    
                    # Remove the now empty directories
                    src_content.rmdir()
                    package_path.rmdir()
                    
                    # Create a note about the change
                    note_file = parent / "STRUCTURE_CHANGE.md"
                    with open(note_file, "w") as f:
                        f.write(f"# Structure Change\n\n")
                        f.write(f"Moved contents from `{package_path.name}/{src_content.name}/` to root level\n")
                        f.write(f"for consistency with standard package structure.\n\n")
                        f.write(f"Date: {datetime.datetime.now().isoformat()}\n")
                    
                    changes_made.append(f"Flattened {package_path} structure in {parent}")
                    
                except Exception as e:
                    print(f"Error flattening {package_path}: {e}")
    
    # Report on non-standard directories
    non_standard_dirs = []
    for package_path in packages_dir.rglob("*"):
        if (package_path.is_dir() and 
            package_path.parent.name in ["packages", "ai", "data"] and
            package_path.name not in ["ai", "data", "temp", "tools", "archive", "__pycache__", ".github"]):
            
            # Check subdirectories
            for subdir in package_path.iterdir():
                if (subdir.is_dir() and 
                    subdir.name not in standard_structure + optional_dirs + 
                    [".github", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__", 
                     "htmlcov", "build", "dist", ".git", "data-platform"]):
                    non_standard_dirs.append(str(subdir))
    
    print(f"Made {len(changes_made)} structural changes:")
    for change in changes_made:
        print(f"  {change}")
    
    print(f"\nFound {len(non_standard_dirs)} non-standard directories:")
    for dir_path in non_standard_dirs[:20]:  # Show first 20
        print(f"  {dir_path}")
    
    if len(non_standard_dirs) > 20:
        print(f"  ... and {len(non_standard_dirs) - 20} more")


if __name__ == "__main__":
    import datetime
    standardize_package_structure()