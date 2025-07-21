#!/usr/bin/env python3
"""
Repository Structure Validator

This script validates the repository structure to prevent creation of prohibited 
folders and files that could lead to architectural drift.

Usage:
    python scripts/validate_repository_structure.py

Exit codes:
    0: No violations found
    1: Violations found
    2: Script error
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set

class RepositoryValidator:
    """Validates repository structure against defined rules."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root).resolve()
        self.violations = []
        
        # Prohibited folder patterns
        self.prohibited_folders = {
            "core": "Use specific domain names instead of generic 'core'",
            "common": "Use specific domain names instead of generic 'common'",
            "shared": "Use specific shared functionality names",
            "utils": "Use specific utility names (e.g., 'validation_utils')",
            "helpers": "Use specific helper names",
            "lib": "Use specific library names",
            "misc": "Miscellaneous folders are prohibited"
        }
        
        # Prohibited file patterns
        self.prohibited_files = {
            "core.py": "Use specific module names instead of generic 'core.py'",
            "common.py": "Use specific functionality names",
            "utils.py": "Use specific utility names",
            "helpers.py": "Use specific helper names",
            "misc.py": "Miscellaneous files are prohibited"
        }
        
        # Exempt paths (these are allowed to have prohibited patterns)
        self.exempt_paths = {
            "node_modules",
            ".venv",
            "venv",
            ".env",
            "env",
            "__pycache__",
            ".git",
            ".pytest_cache",
            ".mypy_cache",
            ".coverage",
            "htmlcov",
            "build",
            "dist",
            "*.egg-info"
        }
    
    def is_path_exempt(self, path: Path) -> bool:
        """Check if a path is exempt from validation."""
        path_str = str(path.relative_to(self.repo_root))
        
        for exempt in self.exempt_paths:
            if exempt in path_str:
                return True
            if path_str.startswith(exempt):
                return True
        
        return False
    
    def validate_folder_names(self) -> List[Dict]:
        """Validate folder names against prohibited patterns."""
        violations = []
        
        for root, dirs, files in os.walk(self.repo_root):
            root_path = Path(root)
            
            # Skip exempt paths
            if self.is_path_exempt(root_path):
                continue
            
            for dir_name in dirs:
                dir_path = root_path / dir_name
                
                # Skip exempt directories
                if self.is_path_exempt(dir_path):
                    continue
                
                # Check for prohibited folder names
                if dir_name.lower() in self.prohibited_folders:
                    violations.append({
                        "type": "prohibited_folder",
                        "path": str(dir_path.relative_to(self.repo_root)),
                        "name": dir_name,
                        "reason": self.prohibited_folders[dir_name.lower()],
                        "severity": "high"
                    })
        
        return violations
    
    def validate_file_names(self) -> List[Dict]:
        """Validate file names against prohibited patterns."""
        violations = []
        
        for root, dirs, files in os.walk(self.repo_root):
            root_path = Path(root)
            
            # Skip exempt paths
            if self.is_path_exempt(root_path):
                continue
            
            for file_name in files:
                file_path = root_path / file_name
                
                # Skip exempt files
                if self.is_path_exempt(file_path):
                    continue
                
                # Check for prohibited file names
                if file_name.lower() in self.prohibited_files:
                    violations.append({
                        "type": "prohibited_file",
                        "path": str(file_path.relative_to(self.repo_root)),
                        "name": file_name,
                        "reason": self.prohibited_files[file_name.lower()],
                        "severity": "high"
                    })
        
        return violations
    
    def validate_structure_patterns(self) -> List[Dict]:
        """Validate against problematic structure patterns."""
        violations = []
        
        # Check for duplicate domain structures
        domain_paths = set()
        for root, dirs, files in os.walk(self.repo_root / "src" / "packages"):
            if "domain" in dirs:
                domain_path = Path(root) / "domain"
                if not self.is_path_exempt(domain_path):
                    parent_structure = str(Path(root).relative_to(self.repo_root))
                    if parent_structure in domain_paths:
                        violations.append({
                            "type": "duplicate_domain",
                            "path": str(domain_path.relative_to(self.repo_root)),
                            "name": "domain",
                            "reason": "Duplicate domain structure detected",
                            "severity": "medium"
                        })
                    domain_paths.add(parent_structure)
        
        return violations
    
    def validate_import_consistency(self) -> List[Dict]:
        """Validate that imports don't reference prohibited structures."""
        violations = []
        
        # This is a simplified check - could be expanded
        for root, dirs, files in os.walk(self.repo_root):
            root_path = Path(root)
            
            if self.is_path_exempt(root_path):
                continue
            
            for file_name in files:
                if file_name.endswith('.py'):
                    file_path = root_path / file_name
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # Check for imports from prohibited modules
                        lines = content.split('\n')
                        for line_num, line in enumerate(lines, 1):
                            if 'import' in line and 'core' in line:
                                # Simple check for core imports
                                if re.search(r'from\s+\w*\.core\s+import|import\s+\w*\.core', line):
                                    violations.append({
                                        "type": "prohibited_import",
                                        "path": str(file_path.relative_to(self.repo_root)),
                                        "line": line_num,
                                        "content": line.strip(),
                                        "reason": "Import from prohibited 'core' module",
                                        "severity": "medium"
                                    })
                    except Exception as e:
                        # Skip files that can't be read
                        continue
        
        return violations
    
    def run_validation(self) -> bool:
        """Run all validation checks."""
        print("🔍 Running repository structure validation...")
        
        # Run all validation checks
        folder_violations = self.validate_folder_names()
        file_violations = self.validate_file_names()
        structure_violations = self.validate_structure_patterns()
        import_violations = self.validate_import_consistency()
        
        # Combine all violations
        all_violations = (
            folder_violations + 
            file_violations + 
            structure_violations + 
            import_violations
        )
        
        # Report results
        if not all_violations:
            print("✅ No violations found. Repository structure is clean.")
            return True
        
        print(f"❌ Found {len(all_violations)} violation(s):")
        print()
        
        # Group violations by type
        violations_by_type = {}
        for violation in all_violations:
            vtype = violation["type"]
            if vtype not in violations_by_type:
                violations_by_type[vtype] = []
            violations_by_type[vtype].append(violation)
        
        # Display violations by type
        for vtype, violations in violations_by_type.items():
            print(f"🚨 {vtype.replace('_', ' ').title()} ({len(violations)} violations):")
            for violation in violations:
                severity_icon = "🔴" if violation["severity"] == "high" else "🟡"
                print(f"  {severity_icon} {violation['path']}")
                print(f"     Reason: {violation['reason']}")
                if 'content' in violation:
                    print(f"     Line {violation['line']}: {violation['content']}")
                print()
        
        return False
    
    def fix_violations(self) -> bool:
        """Attempt to automatically fix certain violations."""
        print("🔧 Attempting to fix violations...")
        
        # This would implement automatic fixes for certain types of violations
        # For now, just provide guidance
        print("⚠️  Automatic fixing not implemented yet.")
        print("📋 Please review the violations above and fix them manually.")
        print()
        print("💡 Quick fix suggestions:")
        print("   - Rename 'core' folders to specific domain names")
        print("   - Replace generic file names with specific functionality names")
        print("   - Update imports to use new structure")
        print("   - Remove duplicate domain structures")
        
        return False

def main():
    """Main entry point."""
    repo_root = os.getcwd()
    
    # Change to repository root if script is run from scripts directory
    if os.path.basename(repo_root) == "scripts":
        repo_root = os.path.dirname(repo_root)
    
    validator = RepositoryValidator(repo_root)
    
    try:
        is_valid = validator.run_validation()
        
        if not is_valid:
            print("🛠️  To fix violations, run:")
            print("    python scripts/validate_repository_structure.py --fix")
            print()
            print("📖 For more information, see:")
            print("    - docs/REPLACEMENT_FOLDERS_ISSUE.md")
            print("    - REPOSITORY_RULES.md")
            print("    - GitHub Issue: https://github.com/elgerytme/anomaly_detection/issues/830")
            
            sys.exit(1)
        
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ Script error: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()