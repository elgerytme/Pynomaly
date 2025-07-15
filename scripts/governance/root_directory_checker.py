#!/usr/bin/env python3
"""
Root Directory Organization Checker
Enforces clean root directory organization according to best practices.
"""

import sys
from pathlib import Path
from typing import Dict, List, Set


class RootDirectoryChecker:
    """Enforces root directory organization standards."""
    
    ALLOWED_ROOT_FILES = {
        # Essential project files
        "README.md", "LICENSE", "CONTRIBUTING.md", "CHANGELOG.md",
        # Configuration files
        "pyproject.toml", "nx.json", "workspace.json", 
        ".gitignore", ".gitattributes", ".pre-commit-config.yaml",
        # Build files
        "BUCK", "Makefile", "justfile", "docker-compose.yml",
        # CI/CD
        ".github/", "Dockerfile",
        # Migration documentation (temporary)
        "BUCK2_MIGRATION_COMPLETE.md", "MONOREPO_MIGRATION_PLAN.md"
    }
    
    ALLOWED_ROOT_DIRECTORIES = {
        # Source code
        "src/",
        # Documentation
        "docs/", "documentation/",
        # Testing
        "tests/",
        # Configuration
        "configs/", "config/",
        # Deployment
        "deployment/", "deploy/",
        # Scripts and automation
        "scripts/", "tools/",
        # Reports and analysis
        "reports/",
        # Templates
        "templates/",
        # CI/CD
        ".github/",
        # Development environment
        "environments/"
    }
    
    SUGGESTED_MOVES = {
        # Analysis and reports
        r'.*_ANALYSIS\.md$': 'reports/analysis/',
        r'.*_REPORT\.md$': 'reports/analysis/', 
        r'.*_ASSESSMENT\.md$': 'reports/analysis/',
        # Scripts
        r'.*\.py$': 'scripts/',
        r'.*_script\..*$': 'scripts/',
        # Docker files
        r'docker-compose\..*\.yml$': 'deployment/docker/',
        r'Dockerfile\..*$': 'deployment/docker/',
        # Kubernetes
        r'.*\.k8s\.ya?ml$': 'deployment/kubernetes/',
        # Configuration
        r'.*\.ya?ml$': 'configs/',
        r'.*\.json$': 'configs/',
        r'.*\.toml$': 'configs/',
        # Temporary files
        r'.*\.tmp$': 'tmp/',
        r'.*\.temp$': 'tmp/',
        r'.*\.log$': 'logs/',
    }
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.violations: List[Dict] = []
        
    def check_root_organization(self) -> List[Dict]:
        """Check root directory organization."""
        violations = []
        
        for item in self.repo_root.iterdir():
            if item.name.startswith('.git'):
                continue
                
            item_path = item.name + ('/' if item.is_dir() else '')
            
            # Check if item is allowed in root
            if item.is_file():
                if item.name not in self.ALLOWED_ROOT_FILES:
                    suggestion = self._get_suggested_location(item.name)
                    violations.append({
                        "type": "misplaced_file",
                        "item": item.name,
                        "suggestion": suggestion,
                        "severity": "warning"
                    })
            else:  # Directory
                if item_path not in self.ALLOWED_ROOT_DIRECTORIES:
                    suggestion = self._get_suggested_location(item.name)
                    violations.append({
                        "type": "misplaced_directory", 
                        "item": item.name + "/",
                        "suggestion": suggestion,
                        "severity": "warning"
                    })
        
        return violations
    
    def _get_suggested_location(self, filename: str) -> str:
        """Get suggested location for misplaced file."""
        import re
        
        for pattern, location in self.SUGGESTED_MOVES.items():
            if re.match(pattern, filename, re.IGNORECASE):
                return location
        
        # Default suggestions based on file type
        if filename.endswith(('.md', '.txt')):
            if any(keyword in filename.lower() for keyword in ['analysis', 'report', 'assessment']):
                return 'reports/'
            else:
                return 'docs/'
        elif filename.endswith('.py'):
            return 'scripts/'
        elif filename.endswith(('.yml', '.yaml', '.json', '.toml')):
            if 'docker' in filename.lower():
                return 'deployment/docker/'
            elif 'k8s' in filename.lower() or 'kubernetes' in filename.lower():
                return 'deployment/kubernetes/'
            else:
                return 'configs/'
        else:
            return 'appropriate subdirectory'
    
    def check_directory_structure(self) -> List[Dict]:
        """Check for proper directory structure."""
        violations = []
        
        required_dirs = ['src/', 'docs/', 'tests/', 'scripts/']
        
        for req_dir in required_dirs:
            dir_path = self.repo_root / req_dir.rstrip('/')
            if not dir_path.exists():
                violations.append({
                    "type": "missing_directory",
                    "directory": req_dir,
                    "severity": "info",
                    "description": f"Consider creating {req_dir} for better organization"
                })
        
        return violations
    
    def auto_organize(self) -> None:
        """Automatically organize root directory."""
        print("🔧 Auto-organizing root directory...")
        
        # Create standard directories
        standard_dirs = ['reports/analysis', 'scripts/analysis', 'configs', 'deployment/docker']
        for dir_path in standard_dirs:
            (self.repo_root / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Move misplaced files
        for item in self.repo_root.iterdir():
            if item.is_file() and item.name not in self.ALLOWED_ROOT_FILES:
                if item.name.startswith('.'):
                    continue  # Skip hidden files
                    
                suggestion = self._get_suggested_location(item.name)
                dest_dir = self.repo_root / suggestion
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                dest_path = dest_dir / item.name
                if not dest_path.exists():
                    try:
                        item.rename(dest_path)
                        print(f"  ✅ Moved {item.name} to {suggestion}")
                    except OSError as e:
                        print(f"  ⚠️  Could not move {item.name}: {e}")
    
    def run_check(self) -> bool:
        """Run root directory organization check."""
        print("📁 Checking root directory organization...")
        
        violations = []
        violations.extend(self.check_root_organization())
        violations.extend(self.check_directory_structure())
        
        if violations:
            errors = [v for v in violations if v.get("severity") == "error"]
            warnings = [v for v in violations if v.get("severity") == "warning"]
            info = [v for v in violations if v.get("severity") == "info"]
            
            if errors:
                print(f"❌ Found {len(errors)} error(s):")
                for error in errors:
                    print(f"  • {error['type']}: {error.get('item', error.get('directory'))}")
            
            if warnings:
                print(f"⚠️  Found {len(warnings)} warning(s):")
                for warning in warnings:
                    item = warning.get('item', warning.get('directory'))
                    suggestion = warning.get('suggestion', '')
                    print(f"  • {item} → suggested: {suggestion}")
            
            if info:
                print(f"💡 Found {len(info)} suggestion(s):")
                for suggestion in info:
                    print(f"  • {suggestion['description']}")
            
            print("\n💡 Run 'python3 scripts/governance/root_directory_checker.py --fix' to auto-organize")
            
            return len(errors) == 0
        
        print("✅ Root directory is well organized!")
        return True


def main():
    """Main entry point for root directory checker."""
    repo_root = Path(__file__).parent.parent.parent
    checker = RootDirectoryChecker(repo_root)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--fix":
        checker.auto_organize()
    
    success = checker.run_check()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()