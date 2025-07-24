#!/usr/bin/env python3
"""
Phase 5: System Monitoring Migration Script

This script migrates monitoring and observability components from anomaly_detection 
to shared/observability domain according to the domain migration plan.
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

class Phase5MonitoringMigration:
    def __init__(self, base_path: str = "/mnt/c/Users/andre/monorepo"):
        self.base_path = Path(base_path)
        self.backup_path = self.base_path / "migration_backups" / f"phase5_{int(time.time())}"
        self.migration_items = self._define_migration_items()
        
    def _define_migration_items(self) -> List[MigrationItem]:
        """Define Phase 5 System Monitoring component migration items"""
        return [
            # Health Monitoring Service
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/domain/services/health_monitoring_service.py",
                target_path="src/packages/shared/observability/domain/services/health_monitoring_service.py",
                migration_type="file",
                dependencies=[],
                complexity="medium"
            ),
            
            # Infrastructure Monitoring Components
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/infrastructure/monitoring/health_checker.py",
                target_path="src/packages/shared/observability/infrastructure/monitoring/health_checker.py",
                migration_type="file",
                dependencies=[],
                complexity="low"
            ),
            
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/infrastructure/monitoring/metrics_collector.py",
                target_path="src/packages/shared/observability/infrastructure/monitoring/metrics_collector.py",
                migration_type="file",
                dependencies=[],
                complexity="medium"
            ),
            
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/infrastructure/monitoring/performance_monitor.py",
                target_path="src/packages/shared/observability/infrastructure/monitoring/performance_monitor.py",
                migration_type="file",
                dependencies=[],
                complexity="medium"
            ),
            
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/infrastructure/monitoring/monitoring_integration.py",
                target_path="src/packages/shared/observability/infrastructure/monitoring/monitoring_integration.py",
                migration_type="file",
                dependencies=[],
                complexity="high"
            ),
            
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/infrastructure/monitoring/alerting_system.py",
                target_path="src/packages/shared/observability/infrastructure/monitoring/alerting_system.py",
                migration_type="file",
                dependencies=[],
                complexity="medium"
            ),
            
            # Dashboard Components
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/infrastructure/monitoring/dashboard.py",
                target_path="src/packages/shared/observability/infrastructure/dashboards/dashboard.py",
                migration_type="file",
                dependencies=[],
                complexity="medium"
            ),
            
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/infrastructure/monitoring/monitoring_dashboard.py",
                target_path="src/packages/shared/observability/infrastructure/dashboards/monitoring_dashboard.py",
                migration_type="file",
                dependencies=[],
                complexity="medium"
            ),
            
            # Observability Services
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/infrastructure/observability/business_metrics_service.py",
                target_path="src/packages/shared/observability/domain/services/business_metrics_service.py",
                migration_type="file",
                dependencies=[],
                complexity="high"
            ),
            
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/infrastructure/observability/intelligent_alerting_service.py",
                target_path="src/packages/shared/observability/domain/services/intelligent_alerting_service.py",
                migration_type="file",
                dependencies=[],
                complexity="high"
            ),
            
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/infrastructure/observability/monitoring_orchestrator.py",
                target_path="src/packages/shared/observability/domain/services/monitoring_orchestrator.py",
                migration_type="file",
                dependencies=[],
                complexity="high"
            ),
            
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/infrastructure/observability/performance_profiler.py",
                target_path="src/packages/shared/observability/infrastructure/profiling/performance_profiler.py",
                migration_type="file",
                dependencies=[],
                complexity="medium"
            ),
            
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/infrastructure/observability/realtime_monitoring_service.py",
                target_path="src/packages/shared/observability/domain/services/realtime_monitoring_service.py",
                migration_type="file",
                dependencies=[],
                complexity="high"
            ),
            
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/infrastructure/observability/tracing_service.py",
                target_path="src/packages/shared/observability/infrastructure/tracing/tracing_service.py",
                migration_type="file",
                dependencies=[],
                complexity="medium"
            ),
            
            # API Endpoints
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/api/health.py",
                target_path="src/packages/shared/observability/api/health.py",
                migration_type="file",
                dependencies=["health_monitoring_service.py"],
                complexity="low"
            ),
            
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/api/v1/health.py",
                target_path="src/packages/shared/observability/api/v1/health.py",
                migration_type="file",
                dependencies=["health_monitoring_service.py"],
                complexity="medium"
            ),
            
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/api/v1/monitoring.py",
                target_path="src/packages/shared/observability/api/v1/monitoring.py",
                migration_type="file",
                dependencies=["monitoring_orchestrator.py"],
                complexity="medium"
            ),
            
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/api/v1/performance_monitoring.py",
                target_path="src/packages/shared/observability/api/v1/performance_monitoring.py",
                migration_type="file",
                dependencies=["performance_profiler.py"],
                complexity="medium"
            ),
            
            # CLI Commands
            MigrationItem(
                source_path="src/packages/data/anomaly_detection/src/anomaly_detection/cli/commands/health.py",
                target_path="src/packages/shared/observability/cli/commands/health.py",
                migration_type="file",
                dependencies=["health_monitoring_service.py"],
                complexity="low"
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
                init_file.write_text("# Observability package\n")
    
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
                # Observability to shared/observability
                ("from anomaly_detection.domain.services.health_monitoring_service", 
                 "from shared.observability.domain.services.health_monitoring_service"),
                ("from anomaly_detection.infrastructure.monitoring.health_checker",
                 "from shared.observability.infrastructure.monitoring.health_checker"),
                ("from anomaly_detection.infrastructure.monitoring.metrics_collector",
                 "from shared.observability.infrastructure.monitoring.metrics_collector"),
                ("from anomaly_detection.infrastructure.monitoring.performance_monitor",
                 "from shared.observability.infrastructure.monitoring.performance_monitor"),
                ("from anomaly_detection.infrastructure.monitoring.monitoring_integration",
                 "from shared.observability.infrastructure.monitoring.monitoring_integration"),
                ("from anomaly_detection.infrastructure.monitoring.alerting_system",
                 "from shared.observability.infrastructure.monitoring.alerting_system"),
                ("from anomaly_detection.infrastructure.monitoring.dashboard",
                 "from shared.observability.infrastructure.dashboards.dashboard"),
                ("from anomaly_detection.infrastructure.monitoring.monitoring_dashboard",
                 "from shared.observability.infrastructure.dashboards.monitoring_dashboard"),
                ("from anomaly_detection.infrastructure.observability",
                 "from shared.observability.domain.services"),
                
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
                 
                # Data processing
                ("from anomaly_detection.domain.services.data_processing_service",
                 "from data.processing.domain.services.data_processing_service"),
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
            "phase": "Phase 5: System Monitoring",
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
        """Execute complete Phase 5 migration"""
        print("🚀 Starting Phase 5: System Monitoring Migration")
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
            print("✅ Phase 5 Migration Complete!")
            print(f"📊 Migrated: {len(report['migrated_files'])} files")
            print(f"❌ Failed: {len(report['failed_migrations'])} files")
            print(f"💾 Backup: {report['backup_location']}")
            
            return report
            
        except Exception as e:
            print(f"\n❌ Migration failed: {e}")
            print(f"💾 Backup available at: {self.backup_path}")
            raise

if __name__ == "__main__":
    migration = Phase5MonitoringMigration()
    migration.run_migration()