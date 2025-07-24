#!/usr/bin/env python3
"""
Phase 1: Infrastructure Migration Script
Automated migration of infrastructure components and core entities.

This script handles the migration of:
- Configuration management → shared/infrastructure/config/
- Logging infrastructure → shared/infrastructure/logging/  
- Middleware components → shared/infrastructure/middleware/
- Core domain entities → core/anomaly_detection/domain/entities/

Usage:
    python scripts/migration/phase1_infrastructure_migration.py --execute
    python scripts/migration/phase1_infrastructure_migration.py --dry-run
    python scripts/migration/phase1_infrastructure_migration.py --rollback
"""

import os
import shutil
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration_phase1.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MigrationItem:
    """Represents a single file/directory migration."""
    source_path: str
    target_path: str
    migration_type: str  # 'file', 'directory', 'partial'
    dependencies: List[str]
    complexity: str  # 'low', 'medium', 'high'
    backup_path: Optional[str] = None

class Phase1Migrator:
    """Handles Phase 1 infrastructure migration."""
    
    def __init__(self, base_path: str = "/mnt/c/Users/andre/monorepo"):
        self.base_path = Path(base_path)
        self.anomaly_detection_path = self.base_path / "src/packages/data/anomaly_detection"
        self.backup_path = self.base_path / "migration_backups/phase1"
        self.migration_log = []
        
        # Define migration mapping
        self.migration_items = self._define_migration_items()
        
    def _define_migration_items(self) -> List[MigrationItem]:
        """Define all migration items for Phase 1."""
        return [
            # Configuration Management
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/infrastructure/config",
                target_path="src/packages/shared/infrastructure/config",
                migration_type="directory",
                dependencies=[],
                complexity="low"
            ),
            
            # Logging Infrastructure
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/infrastructure/logging",
                target_path="src/packages/shared/infrastructure/logging",
                migration_type="directory", 
                dependencies=["config"],
                complexity="low"
            ),
            
            # Middleware Components
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/infrastructure/middleware",
                target_path="src/packages/shared/infrastructure/middleware",
                migration_type="directory",
                dependencies=["config", "logging"],
                complexity="medium"
            ),
            
            # Core Domain Entities
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/domain/entities",
                target_path="src/packages/core/anomaly_detection/domain/entities",
                migration_type="directory",
                dependencies=[],
                complexity="low"
            ),
            
            # Infrastructure Utilities
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/infrastructure/utils",
                target_path="src/packages/shared/infrastructure/utils",
                migration_type="directory",
                dependencies=[],
                complexity="low"
            ),
        ]
    
    def create_backup(self) -> bool:
        """Create complete backup of current state."""
        try:
            logger.info("Creating Phase 1 migration backup...")
            
            # Create backup directory
            self.backup_path.mkdir(parents=True, exist_ok=True)
            
            # Backup entire anomaly_detection package
            backup_target = self.backup_path / "anomaly_detection_pre_phase1"
            if backup_target.exists():
                shutil.rmtree(backup_target)
                
            shutil.copytree(self.anomaly_detection_path, backup_target)
            logger.info(f"Backup created at: {backup_target}")
            
            # Save migration state
            state_file = self.backup_path / "migration_state.json"
            with open(state_file, 'w') as f:
                json.dump({
                    "phase": 1,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "items": [
                        {
                            "source": item.source_path,
                            "target": item.target_path,
                            "type": item.migration_type,
                            "complexity": item.complexity
                        }
                        for item in self.migration_items
                    ]
                }, f, indent=2)
                
            return True
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return False
    
    def validate_prerequisites(self) -> bool:
        """Validate prerequisites for Phase 1 migration."""
        logger.info("Validating Phase 1 prerequisites...")
        
        validation_results = []
        
        # Check source paths exist
        for item in self.migration_items:
            source_full_path = self.base_path / item.source_path
            if not source_full_path.exists():
                validation_results.append(f"❌ Source not found: {source_full_path}")
            else:
                validation_results.append(f"✅ Source exists: {source_full_path}")
        
        # Check target directories can be created
        target_dirs = set()
        for item in self.migration_items:
            target_dir = Path(self.base_path / item.target_path).parent
            target_dirs.add(target_dir)
            
        for target_dir in target_dirs:
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
                validation_results.append(f"✅ Target accessible: {target_dir}")
            except Exception as e:
                validation_results.append(f"❌ Target inaccessible: {target_dir} - {e}")
        
        # Check for conflicting files
        conflicts = []
        for item in self.migration_items:
            target_full_path = self.base_path / item.target_path
            if target_full_path.exists():
                conflicts.append(f"⚠️  Target exists: {target_full_path}")
        
        # Print results
        for result in validation_results:
            logger.info(result)
            
        for conflict in conflicts:
            logger.warning(conflict)
        
        # Return success if no errors found
        return not any("❌" in result for result in validation_results)
    
    def migrate_item(self, item: MigrationItem, dry_run: bool = False) -> bool:
        """Migrate a single item."""
        source_full_path = self.base_path / item.source_path
        target_full_path = self.base_path / item.target_path
        
        logger.info(f"Migrating {item.migration_type}: {item.source_path} → {item.target_path}")
        
        if dry_run:
            logger.info(f"[DRY RUN] Would migrate: {source_full_path} → {target_full_path}")
            return True
            
        try:
            # Create target directory
            target_full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Handle different migration types
            if item.migration_type == "directory":
                if target_full_path.exists():
                    shutil.rmtree(target_full_path)
                shutil.copytree(source_full_path, target_full_path)
                
            elif item.migration_type == "file":
                if target_full_path.exists():
                    target_full_path.unlink()
                shutil.copy2(source_full_path, target_full_path)
                
            else:
                logger.error(f"Unknown migration type: {item.migration_type}")
                return False
            
            # Update imports in migrated files
            self._update_imports_in_path(target_full_path, item)
            
            # Log successful migration
            self.migration_log.append({
                "item": item.source_path,
                "status": "success",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            logger.info(f"✅ Successfully migrated: {item.source_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration failed for {item.source_path}: {e}")
            self.migration_log.append({
                "item": item.source_path,
                "status": "failed",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            return False
    
    def _update_imports_in_path(self, path: Path, item: MigrationItem):
        """Update import statements in migrated files."""
        if path.is_file() and path.suffix == '.py':
            self._update_imports_in_file(path)
        elif path.is_dir():
            for py_file in path.rglob('*.py'):
                self._update_imports_in_file(py_file)
    
    def _update_imports_in_file(self, file_path: Path):
        """Update import statements in a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Define import mapping patterns
            import_mappings = {
                # Infrastructure mappings
                r'from \.\.\.infrastructure\.config': 'from shared.infrastructure.config',
                r'from \.\.infrastructure\.config': 'from shared.infrastructure.config',
                r'from anomaly_detection\.infrastructure\.config': 'from shared.infrastructure.config',
                
                r'from \.\.\.infrastructure\.logging': 'from shared.infrastructure.logging',
                r'from \.\.infrastructure\.logging': 'from shared.infrastructure.logging', 
                r'from anomaly_detection\.infrastructure\.logging': 'from shared.infrastructure.logging',
                
                r'from \.\.\.infrastructure\.middleware': 'from shared.infrastructure.middleware',
                r'from \.\.infrastructure\.middleware': 'from shared.infrastructure.middleware',
                r'from anomaly_detection\.infrastructure\.middleware': 'from shared.infrastructure.middleware',
                
                r'from \.\.\.infrastructure\.utils': 'from shared.infrastructure.utils',
                r'from \.\.infrastructure\.utils': 'from shared.infrastructure.utils',
                r'from anomaly_detection\.infrastructure\.utils': 'from shared.infrastructure.utils',
                
                # Domain mappings
                r'from \.\.\.domain\.entities': 'from core.anomaly_detection.domain.entities',
                r'from \.\.domain\.entities': 'from core.anomaly_detection.domain.entities',
                r'from anomaly_detection\.domain\.entities': 'from core.anomaly_detection.domain.entities',
                
                r'from \.\.\.domain\.value_objects': 'from core.anomaly_detection.domain.value_objects',
                r'from \.\.domain\.value_objects': 'from core.anomaly_detection.domain.value_objects',
                r'from anomaly_detection\.domain\.value_objects': 'from core.anomaly_detection.domain.value_objects',
                
                r'from \.\.\.domain\.exceptions': 'from core.anomaly_detection.domain.exceptions',
                r'from \.\.domain\.exceptions': 'from core.anomaly_detection.domain.exceptions',
                r'from anomaly_detection\.domain\.exceptions': 'from core.anomaly_detection.domain.exceptions',
            }
            
            # Apply import mappings
            modified = False
            for old_pattern, new_import in import_mappings.items():
                new_content = re.sub(old_pattern, new_import, content)
                if new_content != content:
                    content = new_content
                    modified = True
            
            # Write updated content if modified
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.debug(f"Updated imports in: {file_path}")
                
        except Exception as e:
            logger.warning(f"Failed to update imports in {file_path}: {e}")
    
    def execute_migration(self, dry_run: bool = False) -> bool:
        """Execute complete Phase 1 migration."""
        logger.info(f"Starting Phase 1 migration {'(DRY RUN)' if dry_run else '(LIVE)'}")
        
        # Validate prerequisites
        if not self.validate_prerequisites():
            logger.error("Prerequisites validation failed. Aborting migration.")
            return False
        
        # Create backup (not in dry run)
        if not dry_run and not self.create_backup():
            logger.error("Backup creation failed. Aborting migration.")
            return False
        
        # Execute migrations in dependency order
        success_count = 0
        total_count = len(self.migration_items)
        
        # Sort by dependencies (items with no dependencies first)
        sorted_items = sorted(self.migration_items, key=lambda x: len(x.dependencies))
        
        for item in sorted_items:
            if self.migrate_item(item, dry_run):
                success_count += 1
            else:
                logger.error(f"Migration failed for {item.source_path}, stopping migration.")
                break
        
        # Generate migration report
        self._generate_migration_report(success_count, total_count, dry_run)
        
        success_rate = success_count / total_count
        logger.info(f"Phase 1 migration completed. Success rate: {success_rate:.1%} ({success_count}/{total_count})")
        
        return success_rate >= 0.95  # 95% success threshold
    
    def _generate_migration_report(self, success_count: int, total_count: int, dry_run: bool):
        """Generate detailed migration report."""
        report_path = self.backup_path / f"phase1_migration_report{'_dry_run' if dry_run else ''}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "phase": 1,
            "dry_run": dry_run,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_items": total_count,
                "successful": success_count,
                "failed": total_count - success_count,
                "success_rate": success_count / total_count if total_count > 0 else 0
            },
            "migration_log": self.migration_log,
            "next_steps": [
                "Review migration log for any failures",
                "Run integration tests to verify functionality",
                "Update CI/CD pipelines for new structure",
                "Begin Phase 2: Machine Learning Components migration"
            ]
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Migration report saved to: {report_path}")
    
    def rollback_migration(self) -> bool:
        """Rollback Phase 1 migration using backup."""
        logger.info("Starting Phase 1 migration rollback...")
        
        backup_source = self.backup_path / "anomaly_detection_pre_phase1"
        if not backup_source.exists():
            logger.error(f"Backup not found at: {backup_source}")
            return False
        
        try:
            # Remove current anomaly_detection directory
            if self.anomaly_detection_path.exists():
                shutil.rmtree(self.anomaly_detection_path)
            
            # Restore from backup
            shutil.copytree(backup_source, self.anomaly_detection_path)
            
            # Remove migrated files from target locations
            for item in self.migration_items:
                target_path = self.base_path / item.target_path
                if target_path.exists():
                    if target_path.is_dir():
                        shutil.rmtree(target_path)
                    else:
                        target_path.unlink()
                    logger.info(f"Removed migrated: {target_path}")
            
            logger.info("✅ Phase 1 migration rollback completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            return False

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Phase 1 Infrastructure Migration")
    parser.add_argument('--execute', action='store_true', help='Execute live migration')
    parser.add_argument('--dry-run', action='store_true', help='Perform dry run (no actual changes)')
    parser.add_argument('--rollback', action='store_true', help='Rollback migration using backup')
    parser.add_argument('--base-path', default='/mnt/c/Users/andre/monorepo', help='Base repository path')
    
    args = parser.parse_args()
    
    if not any([args.execute, args.dry_run, args.rollback]):
        parser.error("Must specify one of: --execute, --dry-run, or --rollback")
    
    migrator = Phase1Migrator(args.base_path)
    
    if args.rollback:
        success = migrator.rollback_migration()
    elif args.dry_run:
        success = migrator.execute_migration(dry_run=True)
    else:  # execute
        success = migrator.execute_migration(dry_run=False)
    
    if success:
        logger.info("🎉 Phase 1 migration operation completed successfully!")
        exit(0)
    else:
        logger.error("💥 Phase 1 migration operation failed!")
        exit(1)

if __name__ == "__main__":
    main()