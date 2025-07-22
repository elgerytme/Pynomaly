#!/usr/bin/env python3
"""
PyProject to Buck2 Migration Script
Converts pyproject.toml configurations to Buck2 BUCK files
"""

import argparse
import re
import sys
import toml
from pathlib import Path
from typing import Dict, List, Optional, Set

class PyProjectToBuck2Migrator:
    """Converts pyproject.toml files to Buck2 BUCK files"""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.third_party_deps: Set[str] = set()
    
    def parse_pyproject(self, pyproject_path: Path) -> Dict:
        """Parse a pyproject.toml file"""
        try:
            with open(pyproject_path, 'r') as f:
                return toml.load(f)
        except Exception as e:
            print(f"❌ Error parsing {pyproject_path}: {e}")
            return {}
    
    def extract_dependencies(self, pyproject_data: Dict) -> List[str]:
        """Extract dependencies from pyproject.toml data"""
        deps = []
        
        # Main dependencies
        if 'project' in pyproject_data and 'dependencies' in pyproject_data['project']:
            for dep in pyproject_data['project']['dependencies']:
                # Clean dependency specification (remove version constraints)
                clean_dep = re.sub(r'[><=!~].*', '', dep.strip())
                clean_dep = clean_dep.replace('-', '_').replace('.', '_')
                deps.append(f"//third-party/python:{clean_dep}")
                self.third_party_deps.add(clean_dep)
        
        # Optional dependencies 
        if 'project' in pyproject_data and 'optional-dependencies' in pyproject_data['project']:
            for group, group_deps in pyproject_data['project']['optional-dependencies'].items():
                for dep in group_deps:
                    clean_dep = re.sub(r'[><=!~].*', '', dep.strip())
                    clean_dep = clean_dep.replace('-', '_').replace('.', '_')
                    deps.append(f"//third-party/python:{clean_dep}")
                    self.third_party_deps.add(clean_dep)
        
        return deps
    
    def extract_cli_entrypoints(self, pyproject_data: Dict) -> Dict[str, str]:
        """Extract CLI entry points from pyproject.toml"""
        entry_points = {}
        
        if 'project' in pyproject_data and 'scripts' in pyproject_data['project']:
            for script_name, script_path in pyproject_data['project']['scripts'].items():
                # Convert module.function to file path
                if ':' in script_path:
                    module_path, function = script_path.split(':', 1)
                    file_path = module_path.replace('.', '/') + '.py'
                    entry_points[script_name] = file_path
        
        return entry_points
    
    def generate_buck_content(
        self, 
        package_name: str,
        package_path: Path,
        pyproject_data: Dict,
        dependencies: List[str]
    ) -> str:
        """Generate BUCK file content for a package"""
        
        # Extract metadata
        description = pyproject_data.get('project', {}).get('description', '')
        version = pyproject_data.get('project', {}).get('version', '0.1.0')
        
        # CLI entry points
        cli_entrypoints = self.extract_cli_entrypoints(pyproject_data)
        
        content = f'''# Buck2 configuration for {package_name}
# Generated from pyproject.toml - DO NOT EDIT MANUALLY
# Description: {description}
# Version: {version}

load("@prelude//python:defs.bzl", "python_binary", "python_library", "python_test")

# Main library target
python_library(
    name = "{package_name}",
    srcs = glob([
        "**/*.py",
    ], exclude = [
        "**/test_*.py",
        "**/tests/**/*.py",
        "**/*_test.py",
    ]),
    deps = [
'''
        
        # Add dependencies
        if dependencies:
            for dep in sorted(dependencies):
                content += f'        "{dep}",\n'
        
        content += f'''    ],
    visibility = ["PUBLIC"],
)

# Test target
python_test(
    name = "{package_name}-tests",
    srcs = glob([
        "**/test_*.py",
        "**/tests/**/*.py", 
        "**/*_test.py",
    ]),
    deps = [
        ":{package_name}",
        "//third-party/python:pytest",
    ],
    visibility = ["PUBLIC"],
)
'''
        
        # Add CLI binary targets
        for cli_name, cli_path in cli_entrypoints.items():
            content += f'''
# CLI binary: {cli_name}
python_binary(
    name = "{cli_name}",
    main = "{cli_path}",
    deps = [
        ":{package_name}",
    ],
    visibility = ["PUBLIC"],
)
'''
        
        return content
    
    def create_package_buck_file(self, package_path: Path) -> bool:
        """Create BUCK file for a single package"""
        
        pyproject_path = package_path / "pyproject.toml"
        if not pyproject_path.exists():
            print(f"⚠️ No pyproject.toml found in {package_path}")
            return False
        
        # Parse pyproject.toml
        pyproject_data = self.parse_pyproject(pyproject_path)
        if not pyproject_data:
            return False
        
        # Extract package name
        package_name = pyproject_data.get('project', {}).get('name', package_path.name)
        package_name = package_name.replace('-', '_').replace('.', '_')
        
        # Extract dependencies
        dependencies = self.extract_dependencies(pyproject_data)
        
        # Generate BUCK content
        buck_content = self.generate_buck_content(
            package_name, package_path, pyproject_data, dependencies
        )
        
        # Write BUCK file
        buck_path = package_path / "BUCK"
        if self.dry_run:
            print(f"📄 Would create {buck_path}")
            print(f"Content preview:\n{buck_content[:200]}...")
        else:
            with open(buck_path, 'w') as f:
                f.write(buck_content)
            print(f"✅ Created {buck_path}")
        
        return True
    
    def update_third_party_buck(self) -> None:
        """Update third-party/python/BUCK with collected dependencies"""
        
        if not self.third_party_deps:
            return
        
        third_party_path = Path("third-party/python/BUCK")
        if not third_party_path.exists():
            print(f"⚠️ {third_party_path} not found")
            return
        
        # Read existing content
        with open(third_party_path, 'r') as f:
            existing_content = f.read()
        
        # Find missing dependencies
        missing_deps = []
        for dep in sorted(self.third_party_deps):
            if f'name = "{dep}"' not in existing_content:
                missing_deps.append(dep)
        
        if not missing_deps:
            print("✅ All dependencies already defined in third-party/python/BUCK")
            return
        
        # Generate new dependency definitions
        new_content = "\n# Additional dependencies from pyproject.toml migration\n"
        for dep in missing_deps:
            new_content += f'''
python_library(
    name = "{dep}",
    srcs = [],
    visibility = ["PUBLIC"],
)
'''
        
        if self.dry_run:
            print(f"📄 Would add {len(missing_deps)} dependencies to {third_party_path}")
            print(f"New dependencies: {', '.join(missing_deps)}")
        else:
            with open(third_party_path, 'a') as f:
                f.write(new_content)
            print(f"✅ Added {len(missing_deps)} dependencies to {third_party_path}")
    
    def migrate_all_packages(self, base_path: Path) -> None:
        """Migrate all packages in the monorepo"""
        
        print(f"🔍 Scanning for pyproject.toml files in {base_path}")
        
        migrated_count = 0
        failed_count = 0
        
        # Find all pyproject.toml files
        for pyproject_path in base_path.rglob("pyproject.toml"):
            package_path = pyproject_path.parent
            print(f"\n📦 Processing package: {package_path.relative_to(base_path)}")
            
            try:
                if self.create_package_buck_file(package_path):
                    migrated_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                print(f"❌ Failed to migrate {package_path}: {e}")
                failed_count += 1
        
        # Update third-party dependencies
        print(f"\n🔧 Updating third-party dependencies...")
        self.update_third_party_buck()
        
        print(f"\n📊 Migration Summary:")
        print(f"  ✅ Successfully migrated: {migrated_count}")
        print(f"  ❌ Failed: {failed_count}")
        print(f"  🔗 Third-party deps found: {len(self.third_party_deps)}")

def main():
    parser = argparse.ArgumentParser(
        description="Migrate pyproject.toml files to Buck2 BUCK files"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("src/packages"),
        help="Base path to scan for packages (default: src/packages)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--package",
        type=Path,
        help="Migrate a single package path instead of scanning all"
    )
    
    args = parser.parse_args()
    
    if not args.path.exists() and not args.package:
        print(f"❌ Path {args.path} does not exist")
        sys.exit(1)
    
    migrator = PyProjectToBuck2Migrator(dry_run=args.dry_run)
    
    if args.package:
        print(f"🔄 Migrating single package: {args.package}")
        if not migrator.create_package_buck_file(args.package):
            sys.exit(1)
        migrator.update_third_party_buck()
    else:
        print(f"🔄 Migrating all packages from pyproject.toml to Buck2...")
        migrator.migrate_all_packages(args.path)
    
    print(f"\n🎉 Migration {'simulation' if args.dry_run else 'completed'}!")

if __name__ == "__main__":
    main()