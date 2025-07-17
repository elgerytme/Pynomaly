"""
Tidiness checker for repository governance.
Checks for backup files, temporary files, and other clutter.
"""

import os
import re
from pathlib import Path
from typing import Dict, List

from .base_checker import BaseChecker


class TidinessChecker(BaseChecker):
    """Checker for repository tidiness issues."""
    
    def __init__(self, root_path: Path):
        """Initialize the tidiness checker."""
        super().__init__(root_path)
        self.backup_patterns = [
            r".*_old\.(py|md|txt|json|yaml|yml)$",
            r".*_backup\.(py|md|txt|json|yaml|yml)$",
            r".*_temp\.(py|md|txt|json|yaml|yml)$",
            r".*_orig\.(py|md|txt|json|yaml|yml)$",
            r".*_original_backup\.(py|md|txt|json|yaml|yml)$",
            r".*\.bak$",
            r".*\.backup$",
            r".*\.tmp$",
            r".*\.temp$",
            r".*~$",
        ]
        
        self.temp_directories = [
            "temp", "tmp", "temporary", "scratch", "backup", "backups",
            "htmlcov", "__pycache__", ".pytest_cache", ".mypy_cache",
            ".ruff_cache", "node_modules", ".venv", "venv", "env"
        ]
        
        self.generated_files = [
            "*.pyc", "*.pyo", "*.pyd", "*.so", "*.dll", "*.dylib",
            "*.egg-info", "*.log", "*.coverage", ".coverage.*",
            "coverage.xml", "*.prof", "*.lprof"
        ]
    
    def check(self) -> Dict:
        """Run tidiness checks."""
        violations = []
        
        # Check for backup files
        backup_files = self._find_backup_files()
        if backup_files:
            violations.append({
                "type": "backup_files",
                "severity": "high",
                "message": f"Found {len(backup_files)} backup files",
                "files": backup_files[:20],  # Limit to first 20 for readability
                "total_count": len(backup_files)
            })
        
        # Check for temporary directories
        temp_dirs = self._find_temp_directories()
        if temp_dirs:
            violations.append({
                "type": "temp_directories",
                "severity": "medium",
                "message": f"Found {len(temp_dirs)} temporary directories",
                "directories": temp_dirs[:10],
                "total_count": len(temp_dirs)
            })
        
        # Check for generated files
        generated_files = self._find_generated_files()
        if generated_files:
            violations.append({
                "type": "generated_files",
                "severity": "medium",
                "message": f"Found {len(generated_files)} generated files",
                "files": generated_files[:20],
                "total_count": len(generated_files)
            })
        
        # Check for large files
        large_files = self._find_large_files()
        if large_files:
            violations.append({
                "type": "large_files",
                "severity": "low",
                "message": f"Found {len(large_files)} large files (>10MB)",
                "files": [{"path": str(f), "size_mb": round(f.stat().st_size / (1024*1024), 2)} for f in large_files],
                "total_count": len(large_files)
            })
        
        # Check for deep directory nesting
        deep_dirs = self._find_deep_directories()
        if deep_dirs:
            violations.append({
                "type": "deep_directories",
                "severity": "low",
                "message": f"Found {len(deep_dirs)} deeply nested directories (>8 levels)",
                "directories": [{"path": str(d), "depth": self._calculate_depth(d)} for d in deep_dirs],
                "total_count": len(deep_dirs)
            })
        
        return {
            "violations": violations,
            "total_violations": len(violations),
            "score": max(0, 100 - len(violations) * 10),
            "recommendations": self._generate_recommendations(violations)
        }
    
    def _find_backup_files(self) -> List[str]:
        """Find backup files matching backup patterns."""
        backup_files = []
        
        for pattern in self.backup_patterns:
            regex = re.compile(pattern)
            for file_path in self.root_path.rglob("*"):
                if file_path.is_file() and regex.match(file_path.name):
                    backup_files.append(str(file_path.relative_to(self.root_path)))
        
        return sorted(backup_files)
    
    def _find_temp_directories(self) -> List[str]:
        """Find temporary directories."""
        temp_dirs = []
        
        for dir_path in self.root_path.rglob("*"):
            if (dir_path.is_dir() and 
                dir_path.name in self.temp_directories and
                not self._is_in_gitignore(dir_path)):
                temp_dirs.append(str(dir_path.relative_to(self.root_path)))
        
        return sorted(temp_dirs)
    
    def _find_generated_files(self) -> List[str]:
        """Find generated files that shouldn't be committed."""
        generated_files = []
        
        for pattern in self.generated_files:
            for file_path in self.root_path.rglob(pattern):
                if file_path.is_file() and not self._is_in_gitignore(file_path):
                    generated_files.append(str(file_path.relative_to(self.root_path)))
        
        return sorted(generated_files)
    
    def _find_large_files(self, size_limit_mb: int = 10) -> List[Path]:
        """Find files larger than the specified size limit."""
        large_files = []
        size_limit_bytes = size_limit_mb * 1024 * 1024
        
        for file_path in self.root_path.rglob("*"):
            if (file_path.is_file() and 
                file_path.stat().st_size > size_limit_bytes and
                not self._is_in_gitignore(file_path)):
                large_files.append(file_path)
        
        return sorted(large_files, key=lambda f: f.stat().st_size, reverse=True)
    
    def _find_deep_directories(self, depth_limit: int = 8) -> List[Path]:
        """Find directories nested deeper than the limit."""
        deep_dirs = []
        
        for dir_path in self.root_path.rglob("*"):
            if dir_path.is_dir() and self._calculate_depth(dir_path) > depth_limit:
                deep_dirs.append(dir_path)
        
        return sorted(deep_dirs, key=lambda d: self._calculate_depth(d), reverse=True)
    
    def _calculate_depth(self, path: Path) -> int:
        """Calculate the depth of a path relative to root."""
        return len(path.relative_to(self.root_path).parts)
    
    def _is_in_gitignore(self, path: Path) -> bool:
        """Check if a path is likely ignored by git."""
        # Simple heuristic - check if path contains common ignored patterns
        path_str = str(path)
        ignored_patterns = [
            ".git/", "__pycache__/", ".pytest_cache/", ".mypy_cache/",
            ".ruff_cache/", "node_modules/", ".venv/", "venv/", "env/"
        ]
        return any(pattern in path_str for pattern in ignored_patterns)
    
    def _generate_recommendations(self, violations: List[Dict]) -> List[str]:
        """Generate recommendations based on violations."""
        recommendations = []
        
        for violation in violations:
            if violation["type"] == "backup_files":
                recommendations.append(
                    f"Remove {violation['total_count']} backup files using: "
                    f"scripts/repository_governance/fixes/backup_file_fixer.py"
                )
            elif violation["type"] == "temp_directories":
                recommendations.append(
                    f"Remove {violation['total_count']} temporary directories and "
                    f"update .gitignore to prevent future commits"
                )
            elif violation["type"] == "generated_files":
                recommendations.append(
                    f"Remove {violation['total_count']} generated files and "
                    f"update .gitignore patterns"
                )
            elif violation["type"] == "large_files":
                recommendations.append(
                    f"Review {violation['total_count']} large files - consider "
                    f"Git LFS or alternative storage"
                )
            elif violation["type"] == "deep_directories":
                recommendations.append(
                    f"Consider flattening {violation['total_count']} deeply nested "
                    f"directory structures"
                )
        
        return recommendations