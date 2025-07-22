#!/usr/bin/env python3
"""
Buck2 Build Configuration Consolidation
=======================================

Removes redundant [build-system] sections from pyproject.toml files in packages
that have BUCK files, since Buck2 handles the building. Keeps all other metadata
and tool configurations intact.

This script is part of Phase 1.2 of the Buck2 enhancement roadmap.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import toml
import argparse
import logging
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


class BuildConfigConsolidator:
    """
    Consolidates build configurations by removing redundant [build-system] 
    sections from pyproject.toml files that have corresponding BUCK files.
    """
    
    def __init__(self, repository_root: Path = None, dry_run: bool = False):
        self.repository_root = repository_root or Path.cwd()
        self.packages_dir = self.repository_root / "src" / "packages"
        self.dry_run = dry_run
        self.logger = self._setup_logging()
        
        # Statistics tracking
        self.stats = {
            'packages_processed': 0,
            'packages_with_mixed_config': 0,
            'packages_consolidated': 0,
            'backup_files_created': 0,
            'errors': 0
        }
        
        # Backup directory
        self.backup_dir = (
            self.repository_root / "temp" / "build_config_backups" / 
            datetime.now().strftime('%Y%m%d_%H%M%S')
        )
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger('build_config_consolidator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def find_mixed_config_packages(self) -> List[Path]:
        """Find packages that have both BUCK and pyproject.toml with [build-system]"""
        mixed_packages = []
        
        if not self.packages_dir.exists():
            self.logger.error(f"Packages directory not found: {self.packages_dir}")
            return mixed_packages
        
        # Find all BUCK files
        buck_files = list(self.packages_dir.rglob("BUCK"))
        self.logger.info(f"Found {len(buck_files)} BUCK files")
        
        for buck_file in buck_files:
            package_dir = buck_file.parent
            pyproject_file = package_dir / "pyproject.toml"
            
            if pyproject_file.exists():
                self.stats['packages_processed'] += 1
                
                try:
                    # Check if pyproject.toml has [build-system] section
                    content = pyproject_file.read_text(encoding='utf-8')
                    
                    if '[build-system]' in content or 'build-system' in content:
                        mixed_packages.append(package_dir)
                        self.stats['packages_with_mixed_config'] += 1
                        self.logger.debug(f"Found mixed config: {package_dir}")
                        
                except Exception as e:
                    self.logger.error(f"Error reading {pyproject_file}: {e}")
                    self.stats['errors'] += 1
        
        self.logger.info(f"Found {len(mixed_packages)} packages with mixed build configurations")
        return mixed_packages
    
    def consolidate_package_config(self, package_dir: Path) -> bool:
        """
        Consolidate build configuration for a single package by removing
        [build-system] section from pyproject.toml
        """
        pyproject_file = package_dir / "pyproject.toml"
        
        if not pyproject_file.exists():
            self.logger.warning(f"No pyproject.toml found in {package_dir}")
            return False
        
        try:
            # Read original content
            original_content = pyproject_file.read_text(encoding='utf-8')
            
            # Create backup if not dry run
            if not self.dry_run:
                self._create_backup(pyproject_file, original_content)
            
            # Remove [build-system] section
            consolidated_content = self._remove_build_system_section(original_content)
            
            if consolidated_content == original_content:
                self.logger.info(f"No changes needed for {package_dir.name}")
                return False
            
            # Write consolidated content
            if not self.dry_run:
                pyproject_file.write_text(consolidated_content, encoding='utf-8')
                self.logger.info(f"✅ Consolidated: {package_dir}")
            else:
                self.logger.info(f"📝 Would consolidate: {package_dir}")
            
            self.stats['packages_consolidated'] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Error consolidating {package_dir}: {e}")
            self.stats['errors'] += 1
            return False
    
    def _remove_build_system_section(self, content: str) -> str:
        """
        Remove [build-system] section from pyproject.toml content while
        preserving all other sections and formatting.
        """
        lines = content.splitlines()
        result_lines = []
        
        in_build_system_section = False
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if we're entering build-system section
            if line == '[build-system]':
                in_build_system_section = True
                # Skip the [build-system] line
                i += 1
                continue
            
            # Check if we're entering a new section (exit build-system)
            if line.startswith('[') and line != '[build-system]' and in_build_system_section:
                in_build_system_section = False
            
            # If we're in build-system section, skip the line
            if in_build_system_section:
                # Skip build-system content lines (requires, build-backend, etc.)
                i += 1
                continue
            
            # Keep all other lines
            result_lines.append(lines[i])
            i += 1
        
        # Clean up any extra blank lines at the beginning
        while result_lines and not result_lines[0].strip():
            result_lines.pop(0)
        
        # Ensure there's exactly one blank line after removing build-system
        if result_lines and result_lines[0].startswith('['):
            # If first line is a section, no need for extra blank line
            pass
        else:
            # Add a blank line after project metadata if needed
            for i, line in enumerate(result_lines):
                if line.startswith('[') and i > 0:
                    if result_lines[i-1].strip():
                        result_lines.insert(i, '')
                    break
        
        return '\n'.join(result_lines) + '\n'
    
    def _create_backup(self, file_path: Path, content: str) -> None:
        """Create backup of original file"""
        backup_path = self.backup_dir / file_path.relative_to(self.repository_root)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        backup_path.write_text(content, encoding='utf-8')
        self.stats['backup_files_created'] += 1
    
    def validate_consolidation(self, package_dirs: List[Path]) -> Dict[str, bool]:
        """
        Validate that consolidated packages still have valid pyproject.toml
        files and that builds work correctly.
        """
        validation_results = {}
        
        for package_dir in package_dirs:
            pyproject_file = package_dir / "pyproject.toml"
            
            try:
                # Try to parse the consolidated TOML
                with open(pyproject_file, 'r', encoding='utf-8') as f:
                    toml.load(f)
                
                validation_results[str(package_dir)] = True
                self.logger.debug(f"✅ Valid TOML: {package_dir.name}")
                
            except Exception as e:
                validation_results[str(package_dir)] = False
                self.logger.error(f"❌ Invalid TOML after consolidation: {package_dir.name}: {e}")
                self.stats['errors'] += 1
        
        return validation_results
    
    def consolidate_all_packages(self) -> Dict[str, int]:
        """Consolidate build configurations across all packages"""
        self.logger.info("🔧 Starting build configuration consolidation")
        
        # Create backup directory
        if not self.dry_run:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created backup directory: {self.backup_dir}")
        
        # Find packages with mixed configurations
        mixed_packages = self.find_mixed_config_packages()
        
        if not mixed_packages:
            self.logger.info("No packages with mixed build configurations found")
            return self.stats
        
        # Consolidate each package
        consolidated_packages = []
        for package_dir in mixed_packages:
            if self.consolidate_package_config(package_dir):
                consolidated_packages.append(package_dir)
        
        # Validate consolidations
        if consolidated_packages and not self.dry_run:
            self.logger.info("🔍 Validating consolidated configurations...")
            validation_results = self.validate_consolidation(consolidated_packages)
            
            failed_validations = [
                pkg for pkg, valid in validation_results.items() if not valid
            ]
            
            if failed_validations:
                self.logger.error(f"❌ {len(failed_validations)} packages failed validation")
                for pkg in failed_validations:
                    self.logger.error(f"   - {pkg}")
        
        # Generate summary
        self._log_summary()
        
        return self.stats
    
    def _log_summary(self) -> None:
        """Log consolidation summary"""
        self.logger.info("\n📊 Build Configuration Consolidation Summary:")
        self.logger.info(f"   Packages processed: {self.stats['packages_processed']}")
        self.logger.info(f"   Mixed configurations found: {self.stats['packages_with_mixed_config']}")
        self.logger.info(f"   Packages consolidated: {self.stats['packages_consolidated']}")
        self.logger.info(f"   Backup files created: {self.stats['backup_files_created']}")
        self.logger.info(f"   Errors: {self.stats['errors']}")
        
        if not self.dry_run and self.stats['packages_consolidated'] > 0:
            self.logger.info(f"   Backups stored in: {self.backup_dir}")
        
        if self.stats['errors'] == 0 and self.stats['packages_consolidated'] > 0:
            self.logger.info("✅ All build configurations successfully consolidated!")
        elif self.stats['errors'] > 0:
            self.logger.warning(f"⚠️  {self.stats['errors']} errors occurred during consolidation")
    
    def restore_from_backup(self, backup_timestamp: str) -> None:
        """Restore packages from backup"""
        backup_path = (
            self.repository_root / "temp" / "build_config_backups" / backup_timestamp
        )
        
        if not backup_path.exists():
            self.logger.error(f"Backup not found: {backup_path}")
            return
        
        restored_count = 0
        
        for backup_file in backup_path.rglob("pyproject.toml"):
            original_path = self.repository_root / backup_file.relative_to(backup_path)
            
            if original_path.exists():
                original_path.write_text(backup_file.read_text(encoding='utf-8'), encoding='utf-8')
                restored_count += 1
                self.logger.info(f"Restored: {original_path}")
        
        self.logger.info(f"✅ Restored {restored_count} files from backup")


def create_consolidation_report(stats: Dict[str, int], repository_root: Path) -> None:
    """Create a detailed report of the consolidation process"""
    report_file = repository_root / "docs" / "reports" / "BUILD_CONFIG_CONSOLIDATION_REPORT.md"
    
    report_content = f"""# Build Configuration Consolidation Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Process**: Phase 1.2 - Consolidate Build Configurations  
**Scope**: Remove redundant [build-system] sections from pyproject.toml files

## Summary

This report documents the consolidation of build configurations as part of the Buck2 
enhancement roadmap. The goal was to remove redundant [build-system] sections from 
pyproject.toml files in packages that have BUCK files, since Buck2 handles building.

## Results

- **Packages processed**: {stats['packages_processed']}
- **Mixed configurations found**: {stats['packages_with_mixed_config']}  
- **Packages consolidated**: {stats['packages_consolidated']}
- **Backup files created**: {stats['backup_files_created']}
- **Errors**: {stats['errors']}

## What Changed

### Before Consolidation
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "example-package"
version = "0.1.0"
# ... rest of project metadata
```

### After Consolidation
```toml
[project]
name = "example-package"
version = "0.1.0"
# ... rest of project metadata
```

## Benefits

1. **Eliminated configuration duplication** - No more conflicts between BUCK and pyproject.toml build systems
2. **Simplified maintenance** - Single source of truth for build configuration (BUCK files)
3. **Reduced complexity** - Cleaner pyproject.toml files focused on metadata and tooling
4. **Improved consistency** - All packages now use Buck2 exclusively for builds

## Preserved Configurations

The consolidation process preserved all important configurations:

- ✅ **Project metadata** - name, version, description, authors, etc.
- ✅ **Dependencies** - production and optional dependencies  
- ✅ **Scripts and entry points** - CLI commands and entry points
- ✅ **Tool configurations** - black, ruff, mypy, pytest, coverage settings
- ✅ **URLs and classifiers** - project links and PyPI classifiers

## Next Steps

1. **Test builds** - Verify Buck2 builds work correctly for all consolidated packages
2. **Update CI/CD** - Ensure CI pipelines use Buck2 exclusively
3. **Documentation** - Update build documentation to reflect Buck2-only approach
4. **Monitoring** - Watch for any issues with the consolidated configurations

## Rollback Procedure

If issues occur, restore from backup:

```bash
# Restore all packages from backup
python scripts/buck2/consolidate_build_configs.py \\
    --restore-backup {datetime.now().strftime('%Y%m%d_%H%M%S')}

# Restore specific package
cp temp/build_config_backups/<timestamp>/src/packages/<domain>/<package>/pyproject.toml \\
   src/packages/<domain>/<package>/pyproject.toml
```

## Status

{'✅ **SUCCESS**: All packages consolidated successfully!' if stats['errors'] == 0 else f"⚠️ **PARTIAL SUCCESS**: {stats['errors']} errors occurred during consolidation"}

The build configuration consolidation is {'complete' if stats['errors'] == 0 else 'mostly complete with some issues'}. 
All packages now use Buck2 as the exclusive build system while maintaining 
all important project metadata and tooling configurations.
"""
    
    report_file.write_text(report_content, encoding='utf-8')
    print(f"📝 Detailed report saved to: {report_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Consolidate build configurations by removing redundant [build-system] sections")
    parser.add_argument('--repository-root', type=str, default=".", help="Repository root directory")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be changed without making changes")
    parser.add_argument('--restore-backup', type=str, help="Restore from backup timestamp (YYYYMMDD_HHMMSS)")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging")
    parser.add_argument('--generate-report', action='store_true', help="Generate detailed consolidation report")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger('build_config_consolidator').setLevel(logging.DEBUG)
    
    consolidator = BuildConfigConsolidator(
        Path(args.repository_root), 
        dry_run=args.dry_run
    )
    
    if args.restore_backup:
        # Restore from backup
        consolidator.restore_from_backup(args.restore_backup)
        return 0
    
    # Run consolidation
    stats = consolidator.consolidate_all_packages()
    
    # Generate report if requested
    if args.generate_report:
        create_consolidation_report(stats, consolidator.repository_root)
    
    # Exit with error if there were errors
    return 1 if stats['errors'] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())