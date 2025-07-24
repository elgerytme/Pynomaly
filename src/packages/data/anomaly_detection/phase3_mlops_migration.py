#!/usr/bin/env python3
"""
Phase 3: MLOps Services Migration Script

This script migrates MLOps components from anomaly_detection to ai/mlops domain
according to the domain migration plan.
"""

import os
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class MigrationItem:
    source_path: str
    target_path: str
    migration_type: str
    dependencies: List[str]
    complexity: str

class Phase3MLOpsMigration:
    def __init__(self, base_path: str = "/mnt/c/Users/andre/monorepo"):
        self.base_path = Path(base_path)
        self.backup_path = self.base_path / "migration_backups" / f"phase3_{int(time.time())}"
        self.migration_items = self._define_migration_items()
        
    def _define_migration_items(self) -> List[MigrationItem]:
        """Define Phase 3 MLOps component migration items"""
        return [
            # MLOps Operations Interface
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/domain/interfaces/mlops_operations.py",
                target_path="src/packages/ai/mlops/domain/interfaces/mlops_operations.py",
                migration_type="file",
                dependencies=[],
                complexity="medium"
            ),
            
            # MLOps Service
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/domain/services/mlops_service.py",
                target_path="src/packages/ai/mlops/domain/services/mlops_service.py",
                migration_type="file",
                dependencies=["mlops_operations.py"],
                complexity="high"
            ),
            
            # A/B Testing Service
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/domain/services/ab_testing_service.py",
                target_path="src/packages/ai/mlops/domain/services/ab_testing_service.py",
                migration_type="file",
                dependencies=["mlops_operations.py"],
                complexity="high"
            ),
            
            # MLOps Stubs
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/infrastructure/adapters/stubs/mlops_stubs.py",
                target_path="src/packages/ai/mlops/infrastructure/adapters/stubs/mlops_stubs.py",
                migration_type="file",
                dependencies=["mlops_operations.py"],
                complexity="medium"
            ),
            
            # Model Repository
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/infrastructure/repositories/model_repository.py",
                target_path="src/packages/ai/mlops/infrastructure/repositories/model_repository.py",
                migration_type="file",
                dependencies=[],
                complexity="medium"
            ),
            
            # MLOps Integration Tests
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/tests/integration/test_mlops_integration.py",
                target_path="src/packages/ai/mlops/tests/integration/test_mlops_integration.py",
                migration_type="file",
                dependencies=["mlops_service.py"],
                complexity="high"
            ),
        ]
    
    def create_backup(self):
        """Create backup of source files before migration"""
        print(f"📦 Creating backup at {self.backup_path}")
        self.backup_path.mkdir(parents=True, exist_ok=True)
        
        for item in self.migration_items:
            source_file = self.base_path / item.source_path
            if source_file.exists():
                backup_file = self.backup_path / item.source_path
                backup_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, backup_file)
                print(f"  ✅ Backed up: {item.source_path}")
    
    def create_target_directories(self):
        """Create target directory structure"""
        print("📁 Creating target directory structure...")
        
        directories = set()
        for item in self.migration_items:
            target_dir = (self.base_path / item.target_path).parent
            directories.add(target_dir)
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ Created: {directory}")
            
            # Create __init__.py files for Python packages
            init_file = directory / "__init__.py"
            if not init_file.exists():
                init_file.write_text("# MLOps package\n")
    
    def migrate_files(self):
        """Execute file migration"""
        print("🚚 Starting file migration...")
        
        for item in self.migration_items:
            source_file = self.base_path / item.source_path
            target_file = self.base_path / item.target_path
            
            if source_file.exists():
                print(f"  📁 Migrating: {item.source_path}")
                print(f"     → {item.target_path}")
                
                # Ensure target directory exists
                target_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(source_file, target_file)
                
                # Update imports in the copied file
                self._update_imports(target_file)
                
                print(f"  ✅ Migrated: {target_file.name}")
            else:
                print(f"  ⚠️  Source not found: {item.source_path}")
    
    def _update_imports(self, file_path: Path):
        """Update import statements in migrated files"""
        if file_path.suffix != '.py':
            return
            
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Update imports from anomaly_detection to appropriate domains
            updates = [
                # MLOps operations to ai/mlops
                ("from anomaly_detection.domain.interfaces.mlops_operations", 
                 "from ai.mlops.domain.interfaces.mlops_operations"),
                ("from anomaly_detection.domain.services.mlops_service",
                 "from ai.mlops.domain.services.mlops_service"),
                ("from anomaly_detection.domain.services.ab_testing_service",
                 "from ai.mlops.domain.services.ab_testing_service"),
                ("from anomaly_detection.infrastructure.adapters.stubs.mlops_stubs",
                 "from ai.mlops.infrastructure.adapters.stubs.mlops_stubs"),
                ("from anomaly_detection.infrastructure.repositories.model_repository",
                 "from ai.mlops.infrastructure.repositories.model_repository"),
                
                # ML operations to ai/machine_learning
                ("from anomaly_detection.domain.interfaces.ml_operations", 
                 "from ai.machine_learning.domain.interfaces.ml_operations"),
                ("from anomaly_detection.infrastructure.adapters.stubs.ml_stubs",
                 "from ai.machine_learning.infrastructure.adapters.stubs.ml_stubs"),
                
                # Core entities to core domain
                ("from anomaly_detection.domain.entities",
                 "from core.anomaly_detection.domain.entities"),
                 
                # Infrastructure to shared
                ("from anomaly_detection.infrastructure.config",
                 "from shared.infrastructure.config"),
                ("from anomaly_detection.infrastructure.logging",
                 "from shared.infrastructure.logging"),
                ("from anomaly_detection.infrastructure.middleware",
                 "from shared.infrastructure.middleware"),
            ]
            
            for old_import, new_import in updates:
                content = content.replace(old_import, new_import)
            
            file_path.write_text(content, encoding='utf-8')
            
        except Exception as e:
            print(f"  ⚠️  Failed to update imports in {file_path}: {e}")
    
    def remove_source_files(self):
        """Remove source files after successful migration"""
        print("🗑️ Removing source files...")
        
        for item in self.migration_items:
            source_file = self.base_path / item.source_path
            if source_file.exists():
                source_file.unlink()
                print(f"  ✅ Removed: {item.source_path}")
                
                # Remove empty directories
                parent = source_file.parent
                try:
                    if not any(parent.iterdir()):
                        parent.rmdir()
                        print(f"  ✅ Removed empty directory: {parent}")
                except OSError:
                    pass  # Directory not empty
    
    def generate_migration_report(self) -> Dict[str, Any]:
        """Generate migration report"""
        report = {
            "phase": "Phase 3: MLOps Services",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_items": len(self.migration_items),
            "migrated_files": [],
            "failed_migrations": [],
            "backup_location": str(self.backup_path)
        }
        
        for item in self.migration_items:
            source_file = self.base_path / item.source_path
            target_file = self.base_path / item.target_path
            
            if target_file.exists():
                report["migrated_files"].append({
                    "source": item.source_path,
                    "target": item.target_path,
                    "complexity": item.complexity
                })
            else:
                report["failed_migrations"].append({
                    "source": item.source_path,
                    "target": item.target_path,
                    "reason": "Target file not found after migration"
                })
        
        return report
    
    def run_migration(self):
        """Execute complete Phase 3 migration"""
        print("🚀 Starting Phase 3: MLOps Services Migration")
        print("=" * 60)
        
        try:
            # Step 1: Create backup
            self.create_backup()
            
            # Step 2: Create target directories
            self.create_target_directories()
            
            # Step 3: Migrate files
            self.migrate_files()
            
            # Step 4: Remove source files
            self.remove_source_files()
            
            # Step 5: Generate report
            report = self.generate_migration_report()
            
            print("\n" + "=" * 60)
            print("✅ Phase 3 Migration Complete!")
            print(f"📊 Migrated: {len(report['migrated_files'])} files")
            print(f"❌ Failed: {len(report['failed_migrations'])} files")
            print(f"💾 Backup: {report['backup_location']}")
            
            return report
            
        except Exception as e:
            print(f"\n❌ Migration failed: {e}")
            print(f"💾 Backup available at: {self.backup_path}")
            raise

if __name__ == "__main__":
    migration = Phase3MLOpsMigration()
    migration.run_migration()