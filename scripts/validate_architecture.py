#!/usr/bin/env python3
"""
Architecture validation script for enterprise vs domain separation.

Enforces the configuration-based architecture rules to ensure proper
separation of concerns between core, enterprise, integrations, and configurations.
"""

import argparse
import ast
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Set, Tuple
import re


class ArchitectureViolation:
    """Represents an architecture rule violation."""
    
    def __init__(self, file_path: str, rule: str, message: str, line: int = None):
        self.file_path = file_path
        self.rule = rule
        self.message = message
        self.line = line
    
    def __str__(self):
        location = f"{self.file_path}:{self.line}" if self.line else self.file_path
        return f"[{self.rule}] {location}: {self.message}"


class ArchitectureValidator:
    """Validates architecture rules for enterprise vs domain separation."""
    
    def __init__(self, rules_file: str = ".github/ARCHITECTURE_RULES.yml"):
        self.rules_file = Path(rules_file)
        self.rules = self._load_rules()
        self.violations: List[ArchitectureViolation] = []
        self.src_packages = Path("src/packages")
    
    def _load_rules(self) -> Dict:
        """Load architecture rules from YAML file."""
        if not self.rules_file.exists():
            raise FileNotFoundError(f"Architecture rules file not found: {self.rules_file}")
        
        with open(self.rules_file) as f:
            return yaml.safe_load(f)
    
    def validate_all(self) -> bool:
        """Run all validation checks."""
        success = True
        
        print("🏗️  Validating Enterprise vs Domain Architecture...")
        print("=" * 60)
        
        success &= self.check_forbidden_directories()
        success &= self.check_import_restrictions()
        success &= self.check_dependency_restrictions()
        success &= self.check_package_concerns()
        
        self._report_results()
        return success
    
    def check_forbidden_directories(self) -> bool:
        """Check for forbidden directory patterns."""
        print("\n📁 Checking forbidden directory patterns...")
        
        forbidden_patterns = self.rules["rules"]["forbidden_directories"]["patterns"]
        violations_found = False
        
        for pattern in forbidden_patterns:
            # Convert pattern to glob pattern
            glob_pattern = pattern.replace("*", "**")
            matching_paths = list(Path(".").glob(glob_pattern))
            
            for path in matching_paths:
                if path.exists():
                    self.violations.append(ArchitectureViolation(
                        str(path),
                        "forbidden_directories",
                        f"Forbidden directory pattern: {pattern}"
                    ))
                    violations_found = True
        
        if not violations_found:
            print("✅ No forbidden directory patterns found")
        
        return not violations_found
    
    def check_import_restrictions(self) -> bool:
        """Check import restrictions between packages."""
        print("\n📦 Checking import restrictions...")
        
        import_rules = self.rules["rules"]["import_restrictions"]["rules"]
        violations_found = False
        
        for rule in import_rules:
            source_pattern = rule["source"]
            forbidden_imports = rule["forbidden_imports"]
            message = rule["message"]
            
            # Find all Python files matching source pattern
            source_files = self._find_python_files(source_pattern)
            
            for file_path in source_files:
                violations_found |= self._check_file_imports(
                    file_path, forbidden_imports, message
                )
        
        if not violations_found:
            print("✅ No import restriction violations found")
        
        return not violations_found
    
    def check_dependency_restrictions(self) -> bool:
        """Check dependency restrictions in pyproject.toml files."""
        print("\n🔗 Checking dependency restrictions...")
        
        violations_found = False
        
        # Check core packages
        core_rules = self.rules["rules"]["core_packages"]
        allowed_deps = set(core_rules["allowed_dependencies"])
        forbidden_deps = set(core_rules["forbidden_dependencies"])
        
        core_files = list(Path("src/packages/core").rglob("pyproject.toml"))
        
        for pyproject_file in core_files:
            violations_found |= self._check_dependencies(
                pyproject_file, allowed_deps, forbidden_deps, "core_packages"
            )
        
        if not violations_found:
            print("✅ No dependency restriction violations found")
        
        return not violations_found
    
    def check_package_concerns(self) -> bool:
        """Check that packages contain only allowed concerns."""
        print("\n🎯 Checking package concerns...")
        
        violations_found = False
        
        # Check enterprise packages
        enterprise_rules = self.rules["rules"]["enterprise_packages"]
        allowed_concerns = enterprise_rules["allowed_concerns"]
        forbidden_concerns = enterprise_rules["forbidden_concerns"]
        
        enterprise_dirs = list(Path("src/packages/enterprise").iterdir())
        
        for enterprise_dir in enterprise_dirs:
            if enterprise_dir.is_dir():
                violations_found |= self._check_enterprise_concerns(
                    enterprise_dir, allowed_concerns, forbidden_concerns
                )
        
        if not violations_found:
            print("✅ No package concern violations found")
        
        return not violations_found
    
    def _find_python_files(self, pattern: str) -> List[Path]:
        """Find Python files matching a pattern."""
        # Convert pattern to glob pattern
        glob_pattern = pattern.replace("**", "*").rstrip("*") + "**/*.py"
        return list(Path(".").glob(glob_pattern))
    
    def _check_file_imports(self, file_path: Path, forbidden_imports: List[str], message: str) -> bool:
        """Check imports in a single file."""
        violations_found = False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST to find imports
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_name = self._get_import_name(node)
                    
                    for forbidden_pattern in forbidden_imports:
                        if self._matches_import_pattern(import_name, forbidden_pattern):
                            self.violations.append(ArchitectureViolation(
                                str(file_path),
                                "import_restrictions", 
                                f"{message}: {import_name}",
                                node.lineno
                            ))
                            violations_found = True
        
        except (SyntaxError, UnicodeDecodeError):
            # Skip files that can't be parsed
            pass
        
        return violations_found
    
    def _get_import_name(self, node) -> str:
        """Extract import name from AST node."""
        if isinstance(node, ast.Import):
            return node.names[0].name if node.names else ""
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            return module
        return ""
    
    def _matches_import_pattern(self, import_name: str, pattern: str) -> bool:
        """Check if import matches forbidden pattern."""
        # Convert pattern to regex
        regex_pattern = pattern.replace("*", ".*").replace(".", r"\.")
        return bool(re.match(regex_pattern, import_name))
    
    def _check_dependencies(self, pyproject_file: Path, allowed: Set[str], forbidden: Set[str], rule_name: str) -> bool:
        """Check dependencies in a pyproject.toml file."""
        violations_found = False
        
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                print("⚠️  Warning: tomllib/tomli not available, skipping dependency checks")
                return False
        
        try:
            with open(pyproject_file, 'rb') as f:
                data = tomllib.load(f)
            
            dependencies = data.get("project", {}).get("dependencies", [])
            
            for dep in dependencies:
                # Extract package name (before version specifiers)
                pkg_name = re.split(r'[<>=!]', dep)[0].strip()
                
                if pkg_name in forbidden:
                    self.violations.append(ArchitectureViolation(
                        str(pyproject_file),
                        rule_name,
                        f"Forbidden dependency: {pkg_name}"
                    ))
                    violations_found = True
        
        except Exception as e:
            print(f"⚠️  Warning: Could not parse {pyproject_file}: {e}")
        
        return violations_found
    
    def _check_enterprise_concerns(self, enterprise_dir: Path, allowed: List[str], forbidden: List[str]) -> bool:
        """Check that enterprise directory contains only allowed concerns."""
        violations_found = False
        dir_name = enterprise_dir.name.lower()
        
        # Check if directory name matches forbidden concerns
        for forbidden_concern in forbidden:
            if forbidden_concern in dir_name:
                self.violations.append(ArchitectureViolation(
                    str(enterprise_dir),
                    "enterprise_packages",
                    f"Enterprise package contains forbidden domain-specific concern: {forbidden_concern}"
                ))
                violations_found = True
        
        return violations_found
    
    def _report_results(self):
        """Report validation results."""
        print("\n" + "=" * 60)
        
        if self.violations:
            print(f"❌ Found {len(self.violations)} architecture violations:")
            print()
            
            # Group violations by rule
            violations_by_rule = {}
            for violation in self.violations:
                if violation.rule not in violations_by_rule:
                    violations_by_rule[violation.rule] = []
                violations_by_rule[violation.rule].append(violation)
            
            for rule, violations in violations_by_rule.items():
                print(f"🚫 {rule.replace('_', ' ').title()} ({len(violations)} violations):")
                for violation in violations:
                    print(f"   {violation}")
                print()
            
            print("📖 See .github/ARCHITECTURE_RULES.yml for detailed rules")
            print("📚 See src/packages/ARCHITECTURE.md for architecture guide")
        
        else:
            print("✅ No architecture violations found!")
            print("🏗️  Configuration-based architecture is properly maintained")


def main():
    parser = argparse.ArgumentParser(description="Validate enterprise vs domain architecture")
    parser.add_argument("--check-dependencies", action="store_true", help="Check dependency restrictions")
    parser.add_argument("--check-imports", action="store_true", help="Check import restrictions")  
    parser.add_argument("--check-directories", action="store_true", help="Check directory structure")
    parser.add_argument("--rules-file", default=".github/ARCHITECTURE_RULES.yml", help="Path to rules file")
    
    args = parser.parse_args()
    
    try:
        validator = ArchitectureValidator(args.rules_file)
        
        if args.check_dependencies:
            success = validator.check_dependency_restrictions()
        elif args.check_imports:
            success = validator.check_import_restrictions()
        elif args.check_directories:
            success = validator.check_forbidden_directories()
        else:
            success = validator.validate_all()
        
        sys.exit(0 if success else 1)
    
    except Exception as e:
        print(f"❌ Architecture validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()