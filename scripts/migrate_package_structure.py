#!/usr/bin/env python3
"""
Package Structure Migration Script
=================================
Migrates existing packages to the new DDD-based folder structure.

Usage:
    python scripts/migrate_package_structure.py --package anomaly_detection --dry-run
    python scripts/migrate_package_structure.py --package anomaly_detection --execute
"""

import os
import sys
import shutil
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import subprocess

@dataclass
class MigrationPlan:
    """Plan for migrating a package to new structure"""
    package_name: str
    source_path: Path
    target_path: Path
    file_mappings: List[Tuple[Path, Path]]  # (source, target)
    import_updates: List[Tuple[str, str]]   # (old_import, new_import)

class PackageMigrator:
    """Migrates packages to new DDD structure"""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.packages_dir = root_dir / "src" / "packages"
        self.migration_rules = self._load_migration_rules()
    
    def _load_migration_rules(self) -> Dict[str, str]:
        """Define rules for mapping files to DDD layers"""
        return {
            # Domain layer patterns
            "domain/entities": "src/{package}/domain/entities",
            "domain/value_objects": "src/{package}/domain/value_objects", 
            "domain/services": "src/{package}/domain/services",
            "domain/repositories": "src/{package}/domain/repositories",
            "domain/exceptions": "src/{package}/domain/exceptions",
            
            # Application layer patterns
            "application/services": "src/{package}/application/services",
            "application/use_cases": "src/{package}/application/use_cases",
            "application/dto": "src/{package}/application/dto",
            
            # Infrastructure layer patterns
            "infrastructure/adapters": "src/{package}/infrastructure/adapters",
            "infrastructure/repositories": "src/{package}/infrastructure/repositories",
            "infrastructure/api": "src/{package}/infrastructure/api",
            "infrastructure/cli": "src/{package}/infrastructure/cli",
            "infrastructure/persistence": "src/{package}/infrastructure/persistence",
            "infrastructure/security": "src/{package}/infrastructure/security",
            "infrastructure/monitoring": "src/{package}/infrastructure/monitoring",
            "infrastructure/config": "src/{package}/infrastructure/config",
            
            # Presentation layer patterns
            "presentation/api": "src/{package}/presentation/api",
            "presentation/cli": "src/{package}/presentation/cli",
            "presentation/web": "src/{package}/presentation/web",
            
            # Root level mappings
            "services": "application/services",  # Most services go to application
            "core": "infrastructure/config",     # Core configs to infrastructure
            "ecosystem": "infrastructure/integrations",
            "enterprise": "infrastructure/enterprise",
            "enhanced_features": "application/services",  # Enhanced features to application
            "performance": "infrastructure/monitoring",
            "sdk": "presentation/sdk",
            
            # Build and deployment
            "docker": "deploy/docker",
            "k8s": "deploy/k8s", 
            "monitoring": "deploy/monitoring",
            "scripts": "scripts",
            "docs": "docs",
            "tests": "tests"
        }
    
    def analyze_package(self, package_path: Path) -> MigrationPlan:
        """Analyze package structure and create migration plan"""
        print(f"📋 Analyzing package: {package_path.name}")
        
        package_name = package_path.name
        target_path = self.packages_dir / f"{package_name}_new"
        file_mappings = []
        import_updates = []
        
        # Walk through all files in package
        for root, dirs, files in os.walk(package_path):
            root_path = Path(root)
            relative_path = root_path.relative_to(package_path)
            
            for file in files:
                if file.endswith(('.py', '.yaml', '.yml', '.toml', '.md', '.txt')):
                    source_file = root_path / file
                    target_file = self._map_file_to_target(
                        source_file, package_path, target_path, package_name
                    )
                    file_mappings.append((source_file, target_file))
        
        # Generate import updates
        import_updates = self._generate_import_updates(package_name, file_mappings)
        
        return MigrationPlan(
            package_name=package_name,
            source_path=package_path,
            target_path=target_path,
            file_mappings=file_mappings,
            import_updates=import_updates
        )
    
    def _map_file_to_target(self, source_file: Path, package_path: Path, 
                           target_path: Path, package_name: str) -> Path:
        """Map source file to target location in new structure"""
        
        relative_path = source_file.relative_to(package_path)
        path_parts = list(relative_path.parts)
        
        # Special handling for root level files
        if len(path_parts) == 1:
            filename = path_parts[0]
            
            # Configuration files go to root
            if filename in ['pyproject.toml', 'README.md', 'LICENSE', 'CHANGELOG.md']:
                return target_path / filename
            
            # Python files need layer assignment
            if filename.endswith('.py'):
                if 'cli' in filename.lower():
                    return target_path / "src" / package_name / "presentation" / "cli" / filename
                elif any(word in filename.lower() for word in ['service', 'engine']):
                    return target_path / "src" / package_name / "application" / "services" / filename
                else:
                    return target_path / "src" / package_name / "domain" / filename
        
        # Handle directory-based mappings
        first_dir = path_parts[0] if path_parts else ""
        
        # Map based on migration rules
        if first_dir in self.migration_rules:
            rule = self.migration_rules[first_dir]
            if "{package}" in rule:
                new_path = rule.format(package=package_name)
            else:
                new_path = rule
            
            # Reconstruct path
            remaining_parts = path_parts[1:] if len(path_parts) > 1 else [path_parts[0]]
            return target_path / new_path / Path(*remaining_parts)
        
        # Default mapping to src/{package}/infrastructure
        return target_path / "src" / package_name / "infrastructure" / relative_path
    
    def _generate_import_updates(self, package_name: str, 
                                file_mappings: List[Tuple[Path, Path]]) -> List[Tuple[str, str]]:
        """Generate import statement updates for new structure"""
        import_updates = []
        
        # Common import patterns to update
        old_patterns = [
            f"from {package_name}.services",
            f"from {package_name}.core", 
            f"from {package_name}.ecosystem",
            f"from {package_name}.enterprise",
            f"from {package_name}.enhanced_features",
            f"from src.packages.data.{package_name}"
        ]
        
        new_patterns = [
            f"from {package_name}.application.services",
            f"from {package_name}.infrastructure.config",
            f"from {package_name}.infrastructure.integrations", 
            f"from {package_name}.infrastructure.enterprise",
            f"from {package_name}.application.services",
            f"from {package_name}"
        ]
        
        for old, new in zip(old_patterns, new_patterns):
            import_updates.append((old, new))
        
        return import_updates
    
    def create_target_structure(self, plan: MigrationPlan) -> None:
        """Create target directory structure"""
        print(f"🏗️  Creating target structure: {plan.target_path}")
        
        # Create main directories
        main_dirs = ["build", "deploy", "docs", "src", "scripts", "tests"]
        for dir_name in main_dirs:
            (plan.target_path / dir_name).mkdir(parents=True, exist_ok=True)
        
        # Create deploy subdirectories  
        deploy_dirs = ["docker", "k8s", "monitoring"]
        for dir_name in deploy_dirs:
            (plan.target_path / "deploy" / dir_name).mkdir(parents=True, exist_ok=True)
        
        # Create DDD structure in src
        package_src = plan.target_path / "src" / plan.package_name
        ddd_dirs = ["application", "domain", "infrastructure", "presentation"]
        for dir_name in ddd_dirs:
            (package_src / dir_name).mkdir(parents=True, exist_ok=True)
        
        # Create layer subdirectories
        layer_subdirs = {
            "domain": ["entities", "value_objects", "services", "repositories", "exceptions"],
            "application": ["services", "use_cases", "dto"],
            "infrastructure": ["adapters", "repositories", "api", "cli", "persistence", 
                             "security", "monitoring", "config"],
            "presentation": ["api", "cli", "web"]
        }
        
        for layer, subdirs in layer_subdirs.items():
            for subdir in subdirs:
                (package_src / layer / subdir).mkdir(parents=True, exist_ok=True)
    
    def migrate_files(self, plan: MigrationPlan, dry_run: bool = True) -> None:
        """Migrate files according to the plan"""
        print(f"📦 Migrating {len(plan.file_mappings)} files...")
        
        for i, (source, target) in enumerate(plan.file_mappings):
            print(f"  [{i+1:3d}/{len(plan.file_mappings):3d}] {source.name} -> {target.relative_to(plan.target_path)}")
            
            if not dry_run:
                # Ensure target directory exists
                target.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                if source.exists():
                    shutil.copy2(source, target)
                    
                    # Update imports in Python files
                    if target.suffix == '.py':
                        self._update_imports_in_file(target, plan.import_updates)
    
    def _update_imports_in_file(self, file_path: Path, import_updates: List[Tuple[str, str]]) -> None:
        """Update import statements in a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply import updates
            for old_import, new_import in import_updates:
                content = re.sub(
                    re.escape(old_import) + r'(\s|\.|\b)',
                    new_import + r'\1',
                    content
                )
            
            # Write updated content if changed
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"    ✅ Updated imports in {file_path.name}")
                
        except Exception as e:
            print(f"    ⚠️  Failed to update imports in {file_path.name}: {e}")
    
    def create_package_init_files(self, plan: MigrationPlan) -> None:
        """Create __init__.py files with proper imports"""
        package_src = plan.target_path / "src" / plan.package_name
        
        # Main package __init__.py
        main_init = package_src / "__init__.py"
        main_init_content = f'''"""
{plan.package_name} - {plan.package_name.title()} Package

Following Domain-Driven Design principles.
"""

__version__ = "0.1.0"

from .application import *
from .domain import *
from .infrastructure import *
from .presentation import *
'''
        main_init.write_text(main_init_content)
        
        # Layer __init__.py files
        layers = ["application", "domain", "infrastructure", "presentation"]
        layer_descriptions = {
            "application": "Use cases and application services",
            "domain": "Core business logic and entities", 
            "infrastructure": "External concerns and implementations",
            "presentation": "User interfaces and external APIs"
        }
        
        for layer in layers:
            layer_init = package_src / layer / "__init__.py"
            layer_init_content = f'"""\n{layer.title()} layer - {layer_descriptions[layer]}\n"""\n'
            layer_init.write_text(layer_init_content)
    
    def generate_migration_report(self, plan: MigrationPlan, output_path: Path) -> None:
        """Generate detailed migration report"""
        report = {
            "package_name": plan.package_name,
            "source_path": str(plan.source_path),
            "target_path": str(plan.target_path),
            "files_migrated": len(plan.file_mappings),
            "import_updates": len(plan.import_updates),
            "file_mappings": [
                {
                    "source": str(source.relative_to(plan.source_path)),
                    "target": str(target.relative_to(plan.target_path))
                }
                for source, target in plan.file_mappings
            ],
            "import_updates": [
                {"old": old, "new": new}
                for old, new in plan.import_updates
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Migration report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Migrate package to new DDD structure")
    parser.add_argument('--package', required=True, help='Package name to migrate')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without executing')
    parser.add_argument('--execute', action='store_true', help='Execute the migration')
    parser.add_argument('--report', help='Output path for migration report')
    
    args = parser.parse_args()
    
    if not (args.dry_run or args.execute):
        print("❌ Must specify either --dry-run or --execute")
        sys.exit(1)
    
    # Initialize migrator
    root_dir = Path(__file__).parent.parent
    migrator = PackageMigrator(root_dir)
    
    # Find package
    package_path = migrator.packages_dir / "data" / args.package
    if not package_path.exists():
        print(f"❌ Package not found: {package_path}")
        sys.exit(1)
    
    print(f"🚀 Starting migration for package: {args.package}")
    print(f"   Source: {package_path}")
    
    # Analyze package
    plan = migrator.analyze_package(package_path)
    
    # Create target structure
    migrator.create_target_structure(plan)
    
    # Show plan
    print(f"\n📋 Migration Plan:")
    print(f"   Files to migrate: {len(plan.file_mappings)}")
    print(f"   Import updates: {len(plan.import_updates)}")
    
    if args.dry_run:
        print(f"\n🔍 DRY RUN - showing what would be done:")
        migrator.migrate_files(plan, dry_run=True)
    
    if args.execute:
        print(f"\n⚡ EXECUTING migration:")
        migrator.migrate_files(plan, dry_run=False)
        migrator.create_package_init_files(plan)
        print(f"✅ Migration completed!")
    
    # Generate report
    if args.report:
        report_path = Path(args.report)
        migrator.generate_migration_report(plan, report_path)
    
    print(f"\n🎉 Migration {'planned' if args.dry_run else 'completed'} for {args.package}")

if __name__ == "__main__":
    main()