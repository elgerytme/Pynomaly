#!/usr/bin/env python3
"""
Targeted Package Migration Script
===============================
Focuses on migrating core structure while preserving existing DDD organization.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List
import argparse

class TargetedMigrator:
    """Migrates anomaly_detection with targeted approach"""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.source_pkg = root_dir / "src/packages/data/anomaly_detection"
        self.target_pkg = root_dir / "src/packages/data/anomaly_detection_restructured"
    
    def migrate_package(self, dry_run: bool = True):
        """Migrate anomaly_detection package with new folder structure"""
        print(f"🚀 Starting targeted migration...")
        print(f"   Source: {self.source_pkg}")
        print(f"   Target: {self.target_pkg}")
        
        if not dry_run:
            # Create target structure
            self._create_target_structure()
            
            # Copy files with better organization
            self._migrate_files()
            
            # Create proper __init__ files
            self._create_init_files()
        
        self._show_migration_plan()
    
    def _create_target_structure(self):
        """Create new DDD-compliant folder structure"""
        print("🏗️  Creating target structure...")
        
        # Main directories
        dirs = [
            "build",
            "deploy/docker", 
            "deploy/k8s",
            "deploy/monitoring",
            "docs",
            "scripts", 
            "tests",
            "src/anomaly_detection/application/services",
            "src/anomaly_detection/application/use_cases",
            "src/anomaly_detection/application/dto",
            "src/anomaly_detection/domain/entities",
            "src/anomaly_detection/domain/value_objects", 
            "src/anomaly_detection/domain/services",
            "src/anomaly_detection/domain/repositories",
            "src/anomaly_detection/domain/exceptions",
            "src/anomaly_detection/infrastructure/adapters",
            "src/anomaly_detection/infrastructure/repositories",
            "src/anomaly_detection/infrastructure/api",
            "src/anomaly_detection/infrastructure/cli",
            "src/anomaly_detection/infrastructure/config",
            "src/anomaly_detection/infrastructure/security",
            "src/anomaly_detection/infrastructure/monitoring",
            "src/anomaly_detection/infrastructure/persistence", 
            "src/anomaly_detection/presentation/api",
            "src/anomaly_detection/presentation/cli",
            "src/anomaly_detection/presentation/web"
        ]
        
        for dir_path in dirs:
            (self.target_pkg / dir_path).mkdir(parents=True, exist_ok=True)
    
    def _migrate_files(self):
        """Migrate files with improved organization"""
        print("📦 Migrating files...")
        
        # File mapping rules (source -> target)
        mappings = {
            # Root config files
            "pyproject.toml": "pyproject.toml",
            "README.md": "docs/README.md", 
            "CHANGELOG.md": "docs/CHANGELOG.md",
            "LICENSE": "LICENSE",
            
            # Build files  
            "Dockerfile": "deploy/docker/Dockerfile",
            
            # Keep existing DDD structure but move to src/
            "domain/": "src/anomaly_detection/domain/",
            "application/": "src/anomaly_detection/application/", 
            "infrastructure/": "src/anomaly_detection/infrastructure/",
            
            # Move root-level services to application
            "services/": "src/anomaly_detection/application/services/",
            
            # Move core configs to infrastructure
            "core/dependency_injection.py": "src/anomaly_detection/infrastructure/config/",
            "core/domain_entities.py": "src/anomaly_detection/infrastructure/config/",
            "core/security_configuration.py": "src/anomaly_detection/infrastructure/security/",
            "core/performance_optimization.py": "src/anomaly_detection/infrastructure/monitoring/",
            
            # Move ecosystem to infrastructure/integrations
            "ecosystem/": "src/anomaly_detection/infrastructure/integrations/",
            
            # Move enterprise to infrastructure  
            "enterprise/": "src/anomaly_detection/infrastructure/enterprise/",
            
            # Move enhanced_features to application/services
            "enhanced_features/": "src/anomaly_detection/application/services/enhanced/",
            
            # Move docker files
            "docker/": "deploy/docker/",
            
            # Move k8s files
            "k8s/": "deploy/k8s/",
            
            # Move monitoring
            "monitoring/": "deploy/monitoring/",
            
            # Move scripts
            "scripts/": "scripts/",
            
            # Move SDK to presentation
            "sdk/": "src/anomaly_detection/presentation/sdk/",
            
            # Move examples to docs
            "examples/": "docs/examples/",
        }
        
        for source_pattern, target_path in mappings.items():
            self._copy_matching_files(source_pattern, target_path)
    
    def _copy_matching_files(self, source_pattern: str, target_path: str):
        """Copy files matching pattern to target"""
        source_path = self.source_pkg / source_pattern
        target_full = self.target_pkg / target_path
        
        if source_path.exists():
            if source_path.is_file():
                target_full.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_full)
                print(f"  📄 {source_pattern} -> {target_path}")
            elif source_path.is_dir():
                if target_full.exists():
                    shutil.rmtree(target_full)
                shutil.copytree(source_path, target_full)
                print(f"  📁 {source_pattern} -> {target_path}")
    
    def _create_init_files(self):
        """Create proper __init__.py files"""
        print("📝 Creating __init__.py files...")
        
        # Main package init
        main_init = self.target_pkg / "src/anomaly_detection/__init__.py"
        main_init.write_text('''"""
Anomaly Detection Package

Domain-Driven Design structure for anomaly detection.
"""

__version__ = "0.1.0"

from .application import *
from .domain import *  
from .infrastructure import *
from .presentation import *
''')
        
        # Layer inits
        layer_inits = {
            "application": "Use cases and application services",
            "domain": "Core business logic and entities",
            "infrastructure": "External concerns and implementations", 
            "presentation": "User interfaces and external APIs"
        }
        
        for layer, description in layer_inits.items():
            init_file = self.target_pkg / f"src/anomaly_detection/{layer}/__init__.py"
            init_file.write_text(f'"""\n{layer.title()} layer - {description}\n"""\n')
    
    def _show_migration_plan(self):
        """Show what will be migrated"""
        print(f"\n📋 Migration Plan:")
        print(f"   ✅ Preserve existing DDD structure (domain/, application/, infrastructure/)")
        print(f"   ✅ Move to src/anomaly_detection/ for clean separation")
        print(f"   ✅ Reorganize root-level files into proper layers")
        print(f"   ✅ Create build/, deploy/, docs/, scripts/, tests/ structure")
        print(f"   ✅ Move configs, docker, k8s to appropriate folders")
        
        # Count files that would be affected
        key_dirs = ["services", "core", "ecosystem", "enterprise", "enhanced_features"]
        total_files = 0
        for dir_name in key_dirs:
            dir_path = self.source_pkg / dir_name
            if dir_path.exists():
                files = list(dir_path.rglob("*.py"))
                total_files += len(files)
                print(f"   📁 {dir_name}/: {len(files)} Python files")
        
        print(f"\n   Total files to reorganize: {total_files}")

def main():
    parser = argparse.ArgumentParser(description="Targeted package migration")
    parser.add_argument('--dry-run', action='store_true', help='Show plan without executing')
    parser.add_argument('--execute', action='store_true', help='Execute the migration')
    
    args = parser.parse_args()
    
    root_dir = Path(__file__).parent.parent
    migrator = TargetedMigrator(root_dir)
    
    if args.execute:
        migrator.migrate_package(dry_run=False)
        print("✅ Migration completed!")
    else:
        migrator.migrate_package(dry_run=True)
        print("🔍 Dry run completed. Use --execute to perform migration.")

if __name__ == "__main__":
    main()