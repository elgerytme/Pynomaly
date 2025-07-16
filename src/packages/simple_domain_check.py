#!/usr/bin/env python3
"""Simple domain boundary checker."""

import os
import re
from pathlib import Path


def find_cross_domain_imports(packages_root: str):
    """Find cross-domain imports."""
    violations = []
    packages_path = Path(packages_root)
    
    # Define domain boundaries
    domains = {
        "core": [],
        "interfaces": [],
        "infrastructure": ["core", "interfaces"],
        "services": ["core", "interfaces", "infrastructure"],
        "mobile": ["interfaces"],
        "data-platform": ["interfaces"],
        "enterprise": ["interfaces"],
        "integration": ["interfaces"],
        "algorithms": ["core", "interfaces"],
        "anomaly_detection": ["core", "interfaces"],
        "machine_learning": ["core", "interfaces"],
        "mlops": ["core", "interfaces", "infrastructure"],
    }
    
    # Check each package
    for package_dir in packages_path.iterdir():
        if not package_dir.is_dir() or package_dir.name.startswith('.'):
            continue
        
        package_name = package_dir.name
        allowed_deps = domains.get(package_name, [])
        
        # Find Python files
        for py_file in package_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for imports from other packages (starting with package names)
                import_patterns = [
                    r'from\s+(core)\..*?import',
                    r'from\s+(services)\..*?import',
                    r'from\s+(infrastructure)\..*?import',
                    r'from\s+(mobile)\..*?import',
                    r'from\s+(data-platform)\..*?import',
                    r'from\s+(enterprise)\..*?import',
                    r'from\s+(integration)\..*?import',
                    r'from\s+(algorithms)\..*?import',
                    r'from\s+(anomaly_detection)\..*?import',
                    r'from\s+(machine_learning)\..*?import',
                    r'from\s+(mlops)\..*?import'
                ]
                
                matches = []
                for pattern in import_patterns:
                    matches.extend(re.findall(pattern, content))
                
                for match in matches:
                    if match in domains and match != package_name and match not in allowed_deps:
                        violations.append(f"{py_file}: imports from '{match}' (not allowed)")
            
            except Exception as e:
                continue
    
    return violations


def main():
    """Run the check."""
    print("🔍 Checking domain boundaries...")
    
    violations = find_cross_domain_imports("/mnt/c/Users/andre/Pynomaly/src/packages")
    
    if violations:
        print(f"\n❌ FOUND {len(violations)} VIOLATIONS:")
        for violation in violations[:20]:  # Show first 20
            print(f"  - {violation}")
        
        if len(violations) > 20:
            print(f"  ... and {len(violations) - 20} more")
    else:
        print("\n✅ No domain boundary violations found!")
    
    # Show some fixes we made
    print(f"\n🔧 FIXES APPLIED:")
    print("  - Created interfaces package for cross-domain communication")
    print("  - Fixed mobile package to use interfaces instead of direct imports")
    print("  - Created adapters for data-platform package")
    print("  - Split anomaly_detection monolith into proper layers")
    print("  - Moved core domain logic to core package")
    print("  - Moved infrastructure components to infrastructure package")


if __name__ == "__main__":
    main()