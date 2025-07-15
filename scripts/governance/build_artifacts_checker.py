#!/usr/bin/env python3
"""
Build Artifacts Checker
Prevents build artifacts and cache files from being committed.
"""

import os
import sys
from pathlib import Path
from typing import List, Set


class BuildArtifactsChecker:
    """Checks for and prevents build artifacts from being committed."""
    
    FORBIDDEN_PATTERNS = {
        # Python artifacts
        "__pycache__", "*.pyc", "*.pyo", "*.pyd", ".Python",
        # Build directories
        "build/", "dist/", "eggs/", "*.egg-info/", ".eggs/",
        # Virtual environments
        "venv/", ".venv/", "env/", ".env/", "virtualenv/",
        # IDE files
        ".vscode/", ".idea/", "*.swp", "*.swo", "*~",
        # OS files  
        ".DS_Store", "Thumbs.db", "desktop.ini",
        # Test artifacts
        ".pytest_cache/", ".coverage", "htmlcov/", ".tox/",
        # Cache directories
        ".cache/", ".mypy_cache/", ".ruff_cache/",
        # Node.js artifacts
        "node_modules/", "npm-debug.log*", "yarn-debug.log*",
        # Database files
        "*.db", "*.sqlite", "*.sqlite3",
        # Log files
        "*.log", "logs/",
        # Temporary files
        "tmp/", "temp/", "*.tmp", "*.temp",
        # Backup files  
        "*.bak", "*.backup", "*~"
    }
    
    ALLOWED_EXCEPTIONS = {
        # Configuration examples
        "example.db", "sample.log", "template.tmp",
        # Documentation assets
        "docs/**/*.db", "examples/**/*.log"
    }
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.violations: List[str] = []
        
    def check_staged_files(self) -> List[str]:
        """Check staged files for build artifacts."""
        import subprocess
        
        try:
            # Get list of staged files
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                capture_output=True,
                text=True,
                cwd=self.repo_root
            )
            
            if result.returncode != 0:
                return ["Error getting staged files"]
            
            staged_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
        except subprocess.SubprocessError:
            return ["Error running git command"]
        
        violations = []
        
        for file_path in staged_files:
            if self._is_forbidden(file_path):
                violations.append(f"Build artifact staged for commit: {file_path}")
        
        return violations
    
    def _is_forbidden(self, file_path: str) -> bool:
        """Check if a file path matches forbidden patterns."""
        path = Path(file_path)
        
        # Check against forbidden patterns
        for pattern in self.FORBIDDEN_PATTERNS:
            if pattern.endswith('/'):
                # Directory pattern
                if pattern[:-1] in path.parts:
                    return True
            elif pattern.startswith('*.'):
                # Extension pattern
                if path.suffix == pattern[1:]:
                    return True
            else:
                # Exact match or substring
                if pattern in str(path) or path.name == pattern:
                    return True
        
        # Check for allowed exceptions
        for exception in self.ALLOWED_EXCEPTIONS:
            if exception in file_path:
                return False
        
        return False
    
    def scan_repository(self) -> List[str]:
        """Scan entire repository for build artifacts."""
        violations = []
        
        for root, dirs, files in os.walk(self.repo_root):
            # Skip .git directory
            if '.git' in dirs:
                dirs.remove('.git')
            
            root_path = Path(root)
            
            # Check directories
            for dir_name in dirs[:]:  # Use slice to allow modification
                if self._is_forbidden(str(root_path / dir_name)):
                    violations.append(f"Build artifact directory: {root_path / dir_name}")
                    dirs.remove(dir_name)  # Don't recurse into forbidden dirs
            
            # Check files
            for file_name in files:
                file_path = root_path / file_name
                if self._is_forbidden(str(file_path)):
                    violations.append(f"Build artifact file: {file_path}")
        
        return violations
    
    def run_check(self, scan_all: bool = False) -> bool:
        """Run the build artifacts check."""
        if scan_all:
            print("🔍 Scanning repository for build artifacts...")
            violations = self.scan_repository()
        else:
            print("🔍 Checking staged files for build artifacts...")
            violations = self.check_staged_files()
        
        if violations:
            print("❌ Build artifacts detected:")
            for violation in violations:
                print(f"  • {violation}")
            
            if not scan_all:
                print("\n💡 To remove build artifacts:")
                print("  git reset HEAD <file>  # Unstage file")
                print("  rm -rf <artifact>      # Remove artifact")
                print("  git add .gitignore     # Update .gitignore if needed")
            
            return False
        
        if not scan_all:
            print("✅ No build artifacts in staged files")
        else:
            print("✅ No build artifacts found in repository")
        
        return True


def main():
    """Main entry point for build artifacts checker."""
    repo_root = Path(__file__).parent.parent.parent
    checker = BuildArtifactsChecker(repo_root)
    
    # Check if --scan-all flag is provided
    scan_all = len(sys.argv) > 1 and sys.argv[1] == "--scan-all"
    
    success = checker.run_check(scan_all=scan_all)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()