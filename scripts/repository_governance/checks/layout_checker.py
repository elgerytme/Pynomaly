"""
Layout checker for repository governance.
Checks for consistent directory structure and organization.
"""

import json
from pathlib import Path
from typing import Dict, List, Set

from .base_checker import BaseChecker


class LayoutChecker(BaseChecker):
    """Checker for repository layout and organization."""
    
    def __init__(self, root_path: Path):
        """Initialize the layout checker."""
        super().__init__(root_path)
        
        # Standard package structure
        self.standard_dirs = [
            "domain",
            "application", 
            "infrastructure",
            "interfaces"
        ]
        
        # Optional directories
        self.optional_dirs = [
            "tests",
            "docs",
            "examples",
            "scripts", 
            "deploy",
            "web",
            "mobile",
            "cli",
            "api",
            "python_sdk",
            "mlops",
            "use_cases"
        ]
        
        # Directories that should not exist
        self.forbidden_dirs = [
            "temp",
            "tmp", 
            "temporary",
            "backup",
            "backups",
            "scratch",
            "archive",
            "old"
        ]
        
        # Files that should be in specific locations
        self.file_location_rules = {
            "README.md": ["root", "package_root"],
            "CHANGELOG.md": ["root", "package_root"],
            "LICENSE": ["root", "package_root"],
            "pyproject.toml": ["root", "package_root"],
            "requirements.txt": ["root", "package_root"],
            "setup.py": ["root", "package_root"],
            "Dockerfile": ["root", "deploy"],
            "docker-compose.yml": ["root", "deploy"]
        }
    
    def check(self) -> Dict:
        """Run layout checks."""
        violations = []
        
        # Check package structure consistency
        structure_violations = self._check_package_structure()
        if structure_violations:
            violations.append({
                "type": "inconsistent_package_structure",
                "severity": "medium",
                "message": f"Found {len(structure_violations)} packages with inconsistent structure",
                "packages": structure_violations,
                "total_count": len(structure_violations)
            })
        
        # Check for forbidden directories
        forbidden_dirs = self._find_forbidden_directories()
        if forbidden_dirs:
            violations.append({
                "type": "forbidden_directories",
                "severity": "high",
                "message": f"Found {len(forbidden_dirs)} forbidden directories",
                "directories": forbidden_dirs,
                "total_count": len(forbidden_dirs)
            })
        
        # Check file locations
        misplaced_files = self._check_file_locations()
        if misplaced_files:
            violations.append({
                "type": "misplaced_files",
                "severity": "low",
                "message": f"Found {len(misplaced_files)} misplaced files",
                "files": misplaced_files,
                "total_count": len(misplaced_files)
            })
        
        # Check for duplicate files
        duplicate_files = self._find_duplicate_files()
        if duplicate_files:
            violations.append({
                "type": "duplicate_files",
                "severity": "medium",
                "message": f"Found {len(duplicate_files)} sets of duplicate files",
                "duplicates": duplicate_files,
                "total_count": len(duplicate_files)
            })
        
        # Check directory naming consistency
        naming_violations = self._check_naming_consistency()
        if naming_violations:
            violations.append({
                "type": "naming_inconsistency",
                "severity": "low",
                "message": f"Found {len(naming_violations)} naming inconsistencies",
                "violations": naming_violations,
                "total_count": len(naming_violations)
            })
        
        # Check for nested src directories
        nested_src = self._find_nested_src_directories()
        if nested_src:
            violations.append({
                "type": "nested_src_directories", 
                "severity": "medium",
                "message": f"Found {len(nested_src)} packages with nested src directories",
                "packages": nested_src,
                "total_count": len(nested_src)
            })
        
        return {
            "violations": violations,
            "total_violations": len(violations),
            "score": self.calculate_score(violations, penalty_per_violation=10),
            "recommendations": self._generate_recommendations(violations)
        }
    
    def _check_package_structure(self) -> List[Dict]:
        """Check each package for consistent structure."""
        violations = []
        
        for package_dir in self.get_package_directories():
            existing_dirs = [d.name for d in package_dir.iterdir() if d.is_dir()]
            
            # Check for missing standard directories
            missing_standard = [d for d in self.standard_dirs if d not in existing_dirs]
            
            # Check for non-standard directories
            non_standard = [
                d for d in existing_dirs 
                if d not in self.standard_dirs + self.optional_dirs + [
                    ".git", ".github", "__pycache__", ".pytest_cache", 
                    ".mypy_cache", ".ruff_cache", "htmlcov", "build", "dist"
                ]
            ]
            
            if missing_standard or non_standard:
                violations.append({
                    "package": str(package_dir.relative_to(self.root_path)),
                    "missing_standard": missing_standard,
                    "non_standard": non_standard,
                    "existing_dirs": existing_dirs
                })
        
        return violations
    
    def _find_forbidden_directories(self) -> List[str]:
        """Find directories that should not exist."""
        forbidden = []
        
        for forbidden_name in self.forbidden_dirs:
            for dir_path in self.root_path.rglob(forbidden_name):
                if dir_path.is_dir():
                    forbidden.append(str(dir_path.relative_to(self.root_path)))
        
        return sorted(forbidden)
    
    def _check_file_locations(self) -> List[Dict]:
        """Check if files are in correct locations."""
        misplaced = []
        
        for file_name, allowed_locations in self.file_location_rules.items():
            for file_path in self.root_path.rglob(file_name):
                if file_path.is_file():
                    location = self._classify_file_location(file_path)
                    if location not in allowed_locations:
                        misplaced.append({
                            "file": str(file_path.relative_to(self.root_path)),
                            "current_location": location,
                            "allowed_locations": allowed_locations
                        })
        
        return misplaced
    
    def _find_duplicate_files(self) -> List[Dict]:
        """Find duplicate files by name and size."""
        file_info = {}
        
        for file_path in self.root_path.rglob("*"):
            if file_path.is_file():
                key = (file_path.name, file_path.stat().st_size)
                if key not in file_info:
                    file_info[key] = []
                file_info[key].append(str(file_path.relative_to(self.root_path)))
        
        duplicates = []
        for (name, size), paths in file_info.items():
            if len(paths) > 1:
                duplicates.append({
                    "file_name": name,
                    "file_size": size,
                    "paths": paths,
                    "count": len(paths)
                })
        
        return duplicates
    
    def _check_naming_consistency(self) -> List[Dict]:
        """Check for naming consistency issues."""
        violations = []
        
        # Check for mixed case in directory names
        for dir_path in self.root_path.rglob("*"):
            if dir_path.is_dir():
                name = dir_path.name
                if name and not name.startswith(".") and not name.startswith("__"):
                    # Check for mixed case (should be lowercase with underscores)
                    if name != name.lower() or " " in name or "-" in name:
                        violations.append({
                            "type": "directory_naming",
                            "path": str(dir_path.relative_to(self.root_path)),
                            "current_name": name,
                            "suggested_name": name.lower().replace(" ", "_").replace("-", "_")
                        })
        
        # Check for inconsistent README capitalization
        readme_files = list(self.root_path.rglob("readme*"))
        if readme_files:
            readme_names = [f.name for f in readme_files]
            if len(set(readme_names)) > 1:
                violations.append({
                    "type": "readme_capitalization",
                    "files": [str(f.relative_to(self.root_path)) for f in readme_files],
                    "suggestion": "Use consistent capitalization (README.md)"
                })
        
        return violations
    
    def _find_nested_src_directories(self) -> List[Dict]:
        """Find packages with problematic nested src directories."""
        nested_src = []
        
        for package_dir in self.get_package_directories():
            src_dirs = [d for d in package_dir.rglob("src") if d.is_dir()]
            
            for src_dir in src_dirs:
                # Check if this src directory has only one subdirectory
                subdirs = [d for d in src_dir.iterdir() if d.is_dir()]
                if len(subdirs) == 1:
                    nested_src.append({
                        "package": str(package_dir.relative_to(self.root_path)),
                        "src_path": str(src_dir.relative_to(self.root_path)),
                        "subdirectory": subdirs[0].name,
                        "can_flatten": True
                    })
        
        return nested_src
    
    def _classify_file_location(self, file_path: Path) -> str:
        """Classify where a file is located."""
        parts = file_path.parts
        
        if len(parts) == 1:
            return "root"
        elif "packages" in parts and len(parts) == parts.index("packages") + 3:
            return "package_root"
        elif "deploy" in parts or "scripts" in parts:
            return "deploy"
        else:
            return "other"
    
    def _generate_recommendations(self, violations: List[Dict]) -> List[str]:
        """Generate recommendations based on violations."""
        recommendations = []
        
        for violation in violations:
            if violation["type"] == "inconsistent_package_structure":
                recommendations.append(
                    f"Standardize structure for {violation['total_count']} packages: "
                    f"ensure domain/, application/, infrastructure/, interfaces/ exist"
                )
            elif violation["type"] == "forbidden_directories":
                recommendations.append(
                    f"Remove {violation['total_count']} forbidden directories: "
                    f"temp/, backup/, archive/, etc."
                )
            elif violation["type"] == "misplaced_files":
                recommendations.append(
                    f"Move {violation['total_count']} misplaced files to correct locations"
                )
            elif violation["type"] == "duplicate_files":
                recommendations.append(
                    f"Consolidate {violation['total_count']} sets of duplicate files"
                )
            elif violation["type"] == "naming_inconsistency":
                recommendations.append(
                    f"Fix {violation['total_count']} naming inconsistencies: "
                    f"use lowercase with underscores"
                )
            elif violation["type"] == "nested_src_directories":
                recommendations.append(
                    f"Flatten {violation['total_count']} nested src/ directory structures"
                )
        
        return recommendations