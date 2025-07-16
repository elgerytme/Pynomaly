#!/usr/bin/env python3
"""
Repository Cleanup Script - Phase 1 Implementation
Removes build artifacts, cache files, and organizes root directory
"""

import os
import shutil
import sys
from pathlib import Path
from typing import List, Set


class RepositoryCleanup:
    """Handles comprehensive repository cleanup operations."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.removed_files = 0
        self.removed_dirs = 0
        self.freed_space = 0
        
    def remove_cache_files(self) -> None:
        """Remove Python cache files and directories."""
        print("🧹 Removing Python cache files...")
        
        # Remove __pycache__ directories
        for cache_dir in self.repo_root.rglob("__pycache__"):
            if cache_dir.is_dir():
                try:
                    size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                    shutil.rmtree(cache_dir)
                    self.removed_dirs += 1
                    self.freed_space += size
                except (OSError, PermissionError) as e:
                    print(f"  ⚠️  Could not remove {cache_dir}: {e}")
        
        # Remove .pyc files
        for pyc_file in self.repo_root.rglob("*.pyc"):
            try:
                size = pyc_file.stat().st_size
                pyc_file.unlink()
                self.removed_files += 1
                self.freed_space += size
            except (OSError, PermissionError) as e:
                print(f"  ⚠️  Could not remove {pyc_file}: {e}")
    
    def remove_build_artifacts(self) -> None:
        """Remove build artifacts and temporary directories."""
        print("🗑️  Removing build artifacts...")
        
        artifacts_to_remove = [
            "src/temporary",
            "src/build_artifacts", 
            "dist",
            "build",
            ".pytest_cache",
            ".coverage",
            "htmlcov",
            ".mypy_cache",
            ".ruff_cache",
            "node_modules"
        ]
        
        for artifact in artifacts_to_remove:
            artifact_path = self.repo_root / artifact
            if artifact_path.exists():
                try:
                    if artifact_path.is_dir():
                        size = sum(f.stat().st_size for f in artifact_path.rglob('*') if f.is_file())
                        shutil.rmtree(artifact_path)
                        self.removed_dirs += 1
                        print(f"  ✅ Removed directory: {artifact}")
                    else:
                        size = artifact_path.stat().st_size
                        artifact_path.unlink()
                        self.removed_files += 1
                        print(f"  ✅ Removed file: {artifact}")
                    self.freed_space += size
                except (OSError, PermissionError) as e:
                    print(f"  ⚠️  Could not remove {artifact}: {e}")
    
    def clean_virtual_environments(self) -> None:
        """Remove virtual environment directories."""
        print("🐍 Removing virtual environments...")
        
        venv_patterns = ["venv", ".venv", "env", ".env", "virtualenv"]
        
        for pattern in venv_patterns:
            for venv_dir in self.repo_root.rglob(pattern):
                if venv_dir.is_dir() and any(
                    (venv_dir / check).exists() 
                    for check in ["bin/python", "Scripts/python.exe", "lib", "Lib"]
                ):
                    try:
                        size = sum(f.stat().st_size for f in venv_dir.rglob('*') if f.is_file())
                        shutil.rmtree(venv_dir)
                        self.removed_dirs += 1
                        self.freed_space += size
                        print(f"  ✅ Removed virtual environment: {venv_dir}")
                    except (OSError, PermissionError) as e:
                        print(f"  ⚠️  Could not remove {venv_dir}: {e}")
    
    def remove_test_artifacts(self) -> None:
        """Remove test artifacts and databases."""
        print("🧪 Removing test artifacts...")
        
        test_artifacts = [
            "test_audit.db",
            "test_tenants.db", 
            "*.log",
            ".coverage*",
            "junit.xml"
        ]
        
        for pattern in test_artifacts:
            for artifact in self.repo_root.rglob(pattern):
                if artifact.is_file():
                    try:
                        size = artifact.stat().st_size
                        artifact.unlink()
                        self.removed_files += 1
                        self.freed_space += size
                        print(f"  ✅ Removed test artifact: {artifact}")
                    except (OSError, PermissionError) as e:
                        print(f"  ⚠️  Could not remove {artifact}: {e}")
    
    def organize_root_directory(self) -> None:
        """Organize files in root directory."""
        print("📁 Organizing root directory...")
        
        # Create organization directories
        org_dirs = ["reports", "scripts", "configs", "deployment"]
        for dir_name in org_dirs:
            (self.repo_root / dir_name).mkdir(exist_ok=True)
        
        # Move stray files to appropriate locations
        moves = {
            "*.md": "reports/analysis",
            "*.py": "scripts/analysis",
            "docker-compose*.yml": "deployment/docker",
            "*.yaml": "configs",
            "*.yml": "configs"
        }
        
        for pattern, destination in moves.items():
            dest_dir = self.repo_root / destination
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            for file_path in self.repo_root.glob(pattern):
                if file_path.is_file() and file_path.parent == self.repo_root:
                    # Skip essential files
                    if file_path.name in ["README.md", "pyproject.toml", "nx.json"]:
                        continue
                    
                    try:
                        dest_path = dest_dir / file_path.name
                        if not dest_path.exists():
                            shutil.move(str(file_path), str(dest_path))
                            print(f"  ✅ Moved {file_path.name} to {destination}/")
                    except (OSError, PermissionError) as e:
                        print(f"  ⚠️  Could not move {file_path}: {e}")
    
    def run_cleanup(self) -> None:
        """Execute complete cleanup process."""
        print("🚀 Starting Repository Cleanup - Phase 1")
        print(f"📂 Repository: {self.repo_root}")
        print("-" * 60)
        
        # Execute cleanup phases
        self.remove_cache_files()
        self.remove_build_artifacts()
        self.clean_virtual_environments()
        self.remove_test_artifacts()
        self.organize_root_directory()
        
        # Report results
        print("-" * 60)
        print("✅ Cleanup Complete!")
        print(f"📊 Files removed: {self.removed_files:,}")
        print(f"📁 Directories removed: {self.removed_dirs:,}")
        print(f"💾 Space freed: {self.freed_space / (1024*1024):.1f} MB")
        

def main():
    """Main entry point for cleanup script."""
    repo_root = Path(__file__).parent.parent.parent
    
    # Confirm before running
    print("⚠️  This will permanently remove build artifacts and cache files.")
    response = input("Continue? (y/N): ").lower().strip()
    
    if response != 'y':
        print("❌ Cleanup cancelled.")
        return
    
    cleanup = RepositoryCleanup(repo_root)
    cleanup.run_cleanup()


if __name__ == "__main__":
    main()