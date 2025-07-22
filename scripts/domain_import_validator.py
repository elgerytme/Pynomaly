#!/usr/bin/env python3
"""
Domain Import Validator

This script validates that import statements don't violate domain boundaries
and enforces import consolidation rules (single import per package).
It's designed to be used as a pre-commit hook for individual files.
"""

import ast
import sys
from pathlib import Path
from typing import List, Dict, Set, Optional
from collections import defaultdict

class DomainImportValidator:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.packages_dir = self.project_root / "src" / "packages"
        
        # Define domain dependency rules
        self.dependency_rules = {
            # Business domains can depend on technical domains
            "business": ["software", "data", "ai", "ops", "formal_sciences"],
            
            # Technical domains have limited dependencies  
            "data": ["software", "formal_sciences"],
            "ai": ["software", "data", "formal_sciences"],
            "software": ["formal_sciences"],
            "ops": ["software"],
            "formal_sciences": [],
            
            # Specific package rules
            "business/governance": ["software/core", "business/administration"],
            "business/cost_optimization": ["software/core", "data/data_observability"],
            "business/administration": ["software/core", "software/enterprise"],
            "data/anomaly_detection": ["software/core", "formal_sciences/mathematics", "ai/mlops"],
            "data/data_quality": ["software/core", "formal_sciences/mathematics"],
            "ai/mlops": ["software/core", "data/data_observability", "formal_sciences/mathematics"],
            "software/enterprise": ["software/core"],
            "software/cybersecurity": ["software/core"],
        }
        
        # Define forbidden imports (specific violations)
        self.forbidden_imports = {
            "software/core": [
                "business.*",  # Core cannot depend on business logic
                "data/anomaly_detection.*",  # Core cannot depend on specific domains
                "ai/mlops.*"   # Core cannot depend on AI domains
            ],
            "data/anomaly_detection": [
                "business.*",  # Data domains cannot depend on business logic
                "software/enterprise.*",  # Cannot depend on enterprise features
                "ai/mlops.*"   # Cannot depend on MLOps (should be reverse)
            ],
            "ai/mlops": [
                "business.*",  # AI domains cannot depend on business logic
                "software/enterprise.*"  # Cannot depend on enterprise features
            ]
        }
    
    def validate_file(self, file_path: Path) -> List[str]:
        """Validate imports in a single file"""
        violations = []
        
        # Skip non-Python files
        if file_path.suffix != '.py':
            return violations
            
        # Get current domain
        current_domain = self._get_domain_from_path(file_path)
        if not current_domain:
            return violations
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse imports
            tree = ast.parse(content)
            imports = self._extract_imports(tree)
            
            # Check import consolidation (single import per package rule)
            consolidation_violations = self._check_import_consolidation(imports)
            violations.extend(consolidation_violations)
            
            # Check each import for domain boundary violations
            for import_info in imports:
                import_domain = self._get_domain_from_import(import_info['module'])
                
                # Check domain dependency rules
                if import_domain and not self._is_allowed_dependency(current_domain, import_domain):
                    violations.append(
                        f"Line {import_info['line']}: Invalid import '{import_info['module']}' "
                        f"- {current_domain} cannot import from {import_domain}"
                    )
                
                # Check forbidden imports
                if self._is_forbidden_import(current_domain, import_info['module']):
                    violations.append(
                        f"Line {import_info['line']}: Forbidden import '{import_info['module']}' "
                        f"- violates domain boundary for {current_domain}"
                    )
                    
        except Exception as e:
            violations.append(f"Error analyzing file: {str(e)}")
            
        return violations
    
    def _get_domain_from_path(self, file_path: Path) -> Optional[str]:
        """Extract domain from file path"""
        try:
            # Get relative path from packages directory
            rel_path = file_path.relative_to(self.packages_dir)
            parts = rel_path.parts
            
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
            elif len(parts) >= 1:
                return parts[0]
                
        except ValueError:
            pass
            
        return None
    
    def _extract_imports(self, tree: ast.AST) -> List[Dict]:
        """Extract import statements from AST"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'module': alias.name,
                        'line': node.lineno,
                        'type': 'import'
                    })
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append({
                        'module': node.module,
                        'line': node.lineno,
                        'type': 'from_import'
                    })
                    
        return imports
    
    def _get_domain_from_import(self, module: str) -> Optional[str]:
        """Get domain from import module path"""
        # Check if it's a local import
        if module.startswith('src.packages.'):
            parts = module.split('.')
            if len(parts) >= 4:
                return f"{parts[2]}/{parts[3]}"
            elif len(parts) >= 3:
                return parts[2]
                
        return None
    
    def _is_allowed_dependency(self, from_domain: str, to_domain: str) -> bool:
        """Check if dependency is allowed"""
        # Self-references are allowed
        if from_domain == to_domain:
            return True
            
        # Check specific package rules
        if from_domain in self.dependency_rules:
            allowed = self.dependency_rules[from_domain]
            if to_domain in allowed:
                return True
                
        # Check general domain rules
        from_type = from_domain.split('/')[0]
        to_type = to_domain.split('/')[0]
        
        if from_type in self.dependency_rules:
            allowed = self.dependency_rules[from_type]
            if to_type in allowed:
                return True
                
        return False
    
    def _is_forbidden_import(self, current_domain: str, import_module: str) -> bool:
        """Check if import is explicitly forbidden"""
        if current_domain not in self.forbidden_imports:
            return False
            
        forbidden_patterns = self.forbidden_imports[current_domain]
        
        for pattern in forbidden_patterns:
            if pattern.endswith('.*'):
                # Wildcard pattern
                prefix = pattern[:-2]
                if import_module.startswith(prefix):
                    return True
            else:
                # Exact match
                if import_module == pattern:
                    return True
                    
        return False

    def _check_import_consolidation(self, imports: List[Dict]) -> List[str]:
        """Check for import consolidation violations (multiple imports from same package)"""
        violations = []
        
        # Group imports by package
        imports_by_package = defaultdict(list)
        
        for import_info in imports:
            module = import_info['module']
            
            # Skip standard library and relative imports
            if self._is_standard_library_or_relative(module):
                continue
            
            # Extract base package name
            package_name = self._extract_base_package(module)
            if package_name:
                imports_by_package[package_name].append(import_info)
        
        # Check for multiple imports from same package
        for package, import_list in imports_by_package.items():
            if len(import_list) > 1:
                lines = [str(imp['line']) for imp in import_list]
                statements = [f"'{imp['module']}'" for imp in import_list]
                
                violations.append(
                    f"Lines {', '.join(lines)}: Multiple imports from package '{package}' - "
                    f"consolidate {', '.join(statements[:3])}{'...' if len(statements) > 3 else ''}"
                )
        
        return violations
    
    def _is_standard_library_or_relative(self, module: str) -> bool:
        """Check if module is standard library or relative import"""
        if not module or module.startswith('.'):
            return True
            
        # Common standard library modules
        stdlib_modules = {
            'os', 'sys', 'json', 'yaml', 'datetime', 'typing', 'pathlib', 're',
            'subprocess', 'logging', 'collections', 'itertools', 'functools',
            'asyncio', 'concurrent', 'multiprocessing', 'threading', 'unittest',
            'ast', 'argparse', 'tempfile', 'shutil', 'base64', 'hashlib'
        }
        
        root_module = module.split('.')[0]
        return root_module in stdlib_modules
    
    def _extract_base_package(self, module: str) -> Optional[str]:
        """Extract base package name for consolidation checking"""
        if not module:
            return None
            
        # Handle internal packages (src.packages.domain.package)
        if module.startswith('src.packages.'):
            parts = module.split('.')
            if len(parts) >= 4:
                return f"{parts[2]}.{parts[3]}"  # domain.package
            elif len(parts) >= 3:
                return parts[2]  # domain
                
        # Handle packages.domain.package imports
        if module.startswith('packages.'):
            parts = module.split('.')
            if len(parts) >= 3:
                return f"{parts[1]}.{parts[2]}"  # domain.package
            elif len(parts) >= 2:
                return parts[1]  # domain
        
        # For external packages, use root module name
        return module.split('.')[0]

def main():
    """Main entry point for pre-commit hook"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate domain imports")
    parser.add_argument("files", nargs="*", help="Files to validate")
    parser.add_argument("--fail-on-violation", action="store_true", help="Exit with non-zero code on violations")
    
    args = parser.parse_args()
    
    validator = DomainImportValidator(".")
    
    total_violations = 0
    
    for file_path in args.files:
        violations = validator.validate_file(Path(file_path))
        
        if violations:
            print(f"❌ {file_path}:")
            for violation in violations:
                print(f"  {violation}")
            print()
            total_violations += len(violations)
    
    if total_violations > 0:
        print(f"Found {total_violations} domain import violations")
        if args.fail_on_violation:
            sys.exit(1)
    else:
        print("✅ No domain import violations found")

if __name__ == "__main__":
    main()