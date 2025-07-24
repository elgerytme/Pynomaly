#!/usr/bin/env python3
"""
Buck2 Standardized Macro Migration
==================================

Migrates existing package BUCK files to use the new standardized
anomalies_python_package macros and rules.

Part of Phase 2.2 of the Buck2 enhancement roadmap.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
import argparse
import logging
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


class BuckMacroMigrator:
    """
    Migrates existing BUCK files to use standardized Anomaly Detection macros.
    """
    
    def __init__(self, repository_root: Path = None, dry_run: bool = False):
        self.repository_root = repository_root or Path.cwd()
        self.packages_dir = self.repository_root / "src" / "packages"
        self.dry_run = dry_run
        self.logger = self._setup_logging()
        
        # Statistics tracking
        self.stats = {
            'buck_files_processed': 0,
            'buck_files_migrated': 0,
            'packages_updated': 0,
            'errors': 0
        }
        
        # Backup directory
        self.backup_dir = (
            self.repository_root / "temp" / "buck_macro_migration" / 
            datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        
        # Domain mapping
        self.domain_mapping = {
            'ai': ['machine_learning', 'mlops', 'neuro_symbolic'],
            'data': [
                'anomaly_detection', 'data_analytics', 'data_architecture', 
                'data_engineering', 'data_ingestion', 'data_lineage',
                'data_modeling', 'data_pipelines', 'data_quality', 
                'data_science', 'data_visualization', 'observability',
                'profiling', 'quality', 'statistics', 'transformation'
            ],
            'enterprise': [
                'enterprise_auth', 'enterprise_governance', 'enterprise_scalability'
            ]
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger('buck_macro_migrator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def find_package_buck_files(self) -> List[Path]:
        """Find all BUCK files in packages"""
        buck_files = []
        
        if not self.packages_dir.exists():
            self.logger.error(f"Packages directory not found: {self.packages_dir}")
            return buck_files
        
        # Find all BUCK files in package directories
        for buck_file in self.packages_dir.rglob("BUCK"):
            buck_files.append(buck_file)
        
        self.logger.info(f"Found {len(buck_files)} BUCK files to process")
        return buck_files
    
    def detect_package_info(self, buck_file: Path) -> Dict[str, str]:
        """Detect package information from path and content"""
        package_dir = buck_file.parent
        parts = package_dir.relative_to(self.packages_dir).parts
        
        package_info = {
            'path': str(package_dir),
            'domain': 'unknown',
            'package_name': 'unknown',
            'type': 'standard'
        }
        
        if len(parts) >= 2:
            domain = parts[0]
            package_name = parts[1]
            
            # Validate domain
            if domain in self.domain_mapping:
                package_info['domain'] = domain
                package_info['package_name'] = package_name
            
        # Detect package type from content
        try:
            content = buck_file.read_text()
            
            if 'python_binary' in content and 'main =' in content:
                if 'server' in content.lower() or 'api' in content.lower():
                    package_info['type'] = 'microservice'
                elif 'cli' in content.lower():
                    package_info['type'] = 'cli'
                else:
                    package_info['type'] = 'standard'
            
            if any(ml_term in content.lower() for ml_term in ['sklearn', 'torch', 'tensorflow', 'mlflow']):
                package_info['type'] = 'ml'
                
        except Exception as e:
            self.logger.debug(f"Could not analyze content of {buck_file}: {e}")
        
        return package_info
    
    def migrate_buck_file(self, buck_file: Path) -> bool:
        """Migrate a single BUCK file to use standardized macros"""
        self.stats['buck_files_processed'] += 1
        
        try:
            # Read original content
            original_content = buck_file.read_text()
            
            # Detect package information
            package_info = self.detect_package_info(buck_file)
            
            if package_info['domain'] == 'unknown':
                self.logger.warning(f"Could not determine domain for {buck_file}")
                return False
            
            # Create backup if not dry run
            if not self.dry_run:
                self._create_backup(buck_file, original_content)
            
            # Generate migrated content
            migrated_content = self._generate_standardized_buck_content(
                package_info, original_content
            )
            
            if migrated_content == original_content:
                self.logger.info(f"No changes needed for {package_info['package_name']}")
                return False
            
            # Write migrated content
            if not self.dry_run:
                buck_file.write_text(migrated_content)
                self.logger.info(f"✅ Migrated: {package_info['package_name']}")
            else:
                self.logger.info(f"📝 Would migrate: {package_info['package_name']}")
            
            self.stats['buck_files_migrated'] += 1
            self.stats['packages_updated'] += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Error migrating {buck_file}: {e}")
            self.stats['errors'] += 1
            return False
    
    def _generate_standardized_buck_content(
        self, 
        package_info: Dict[str, str], 
        original_content: str
    ) -> str:
        """Generate standardized BUCK content using macros"""
        
        domain = package_info['domain']
        package_name = package_info['package_name']
        package_type = package_info['type']
        
        # Extract dependencies from original content
        deps = self._extract_dependencies(original_content)
        
        # Generate new content
        content_parts = [
            "# Anomaly Detection Python Package - Standardized Buck2 Configuration",
            f"# Domain: {domain}",
            f"# Package: {package_name}",
            f"# Type: {package_type}",
            "",
            'load("//tools/buck:anomaly_detection_python_package.bzl", "anomaly_detection_python_package")',
            ""
        ]
        
        # Generate package rule based on type
        if package_type == 'standard':
            content_parts.extend([
                "anomaly_detection_python_package(",
                f'    name = "{package_name}",',
                f'    domain = "{domain}",',
            ])
            
        elif package_type == 'ml':
            content_parts.extend([
                'load("//tools/buck:anomaly_detection_python_package.bzl", "anomaly_detection_ml_package")',
                "",
                "anomaly_detection_ml_package(",
                f'    name = "{package_name}",',
                '    frameworks = ["sklearn"],  # Adjust as needed',
            ])
            
        elif package_type == 'microservice':
            content_parts.extend([
                'load("//tools/buck:anomaly_detection_python_package.bzl", "anomaly_detection_microservice")', 
                "",
                "anomaly_detection_microservice(",
                f'    name = "{package_name}",',
                f'    domain = "{domain}",',
                f'    main_module = "{package_name}.main",  # Adjust as needed',
            ])
            
        elif package_type == 'cli':
            content_parts.extend([
                'load("//tools/buck:anomaly_detection_python_package.bzl", "anomaly_detection_cli_package")',
                "",
                "anomaly_detection_cli_package(",
                f'    name = "{package_name}",',
                f'    domain = "{domain}",',
                f'    cli_module = "{package_name}.cli.main",  # Adjust as needed',
            ])
        
        # Add dependencies if found
        if deps:
            content_parts.extend([
                "    deps = [",
                *[f'        "{dep}",' for dep in deps],
                "    ],",
            ])
        
        # Add common configurations
        content_parts.extend([
            '    visibility = ["PUBLIC"],',
            ")"
        ])
        
        return '\n'.join(content_parts) + '\n'
    
    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract dependencies from original BUCK content"""
        deps = []
        
        # Look for deps = [ ... ] patterns
        deps_pattern = r'deps\s*=\s*\[(.*?)\]'
        deps_match = re.search(deps_pattern, content, re.DOTALL)
        
        if deps_match:
            deps_content = deps_match.group(1)
            
            # Extract individual dependency strings
            dep_pattern = r'"([^"]+)"'
            for match in re.finditer(dep_pattern, deps_content):
                dep = match.group(1)
                # Filter out common patterns that will be handled by macros
                if not any(common in dep for common in ['pytest', 'third-party/python']):
                    deps.append(dep)
        
        return deps
    
    def _create_backup(self, file_path: Path, content: str) -> None:
        """Create backup of original file"""
        backup_path = self.backup_dir / file_path.relative_to(self.repository_root)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        backup_path.write_text(content)
    
    def migrate_all_packages(self) -> Dict[str, int]:
        """Migrate all package BUCK files to use standardized macros"""
        self.logger.info("🔧 Starting Buck2 macro migration")
        
        # Create backup directory
        if not self.dry_run:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created backup directory: {self.backup_dir}")
        
        # Find all BUCK files
        buck_files = self.find_package_buck_files()
        
        if not buck_files:
            self.logger.info("No BUCK files found to migrate")
            return self.stats
        
        # Migrate each BUCK file
        for buck_file in buck_files:
            self.migrate_buck_file(buck_file)
        
        # Generate summary
        self._log_summary()
        
        return self.stats
    
    def _log_summary(self) -> None:
        """Log migration summary"""
        self.logger.info("\n📊 Buck2 Macro Migration Summary:")
        self.logger.info(f"   BUCK files processed: {self.stats['buck_files_processed']}")
        self.logger.info(f"   BUCK files migrated: {self.stats['buck_files_migrated']}")
        self.logger.info(f"   Packages updated: {self.stats['packages_updated']}")
        self.logger.info(f"   Errors: {self.stats['errors']}")
        
        if not self.dry_run and self.stats['buck_files_migrated'] > 0:
            self.logger.info(f"   Backups stored in: {self.backup_dir}")
        
        if self.stats['errors'] == 0 and self.stats['buck_files_migrated'] > 0:
            self.logger.info("✅ All BUCK files successfully migrated to standardized macros!")
        elif self.stats['errors'] > 0:
            self.logger.warning(f"⚠️  {self.stats['errors']} errors occurred during migration")

    def validate_migrations(self, buck_files: List[Path]) -> Dict[str, bool]:
        """Validate that migrated BUCK files are syntactically correct"""
        validation_results = {}
        
        self.logger.info("🔍 Validating migrated BUCK files...")
        
        for buck_file in buck_files:
            try:
                # Basic syntax validation - check if file is readable
                content = buck_file.read_text()
                
                # Check for basic Buck2 syntax
                if 'anomaly_detection_python_package' in content or 'anomaly_detection_' in content:
                    # Check for balanced parentheses
                    if content.count('(') == content.count(')'):
                        validation_results[str(buck_file)] = True
                        self.logger.debug(f"✅ Valid: {buck_file.parent.name}")
                    else:
                        validation_results[str(buck_file)] = False
                        self.logger.error(f"❌ Unbalanced parentheses: {buck_file.parent.name}")
                else:
                    validation_results[str(buck_file)] = False
                    self.logger.error(f"❌ Missing macro usage: {buck_file.parent.name}")
                    
            except Exception as e:
                validation_results[str(buck_file)] = False
                self.logger.error(f"❌ Validation failed for {buck_file.parent.name}: {e}")
        
        return validation_results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Migrate BUCK files to use standardized Monorepo macros")
    parser = argparse.ArgumentParser(description="Migrate BUCK files to use standardized Anomaly Detection macros")
    parser.add_argument('--repository-root', type=str, default=".", help="Repository root directory")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be changed without making changes")
    parser.add_argument('--validate', action='store_true', help="Validate migrated BUCK files")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger('buck_macro_migrator').setLevel(logging.DEBUG)
    
    migrator = BuckMacroMigrator(
        Path(args.repository_root), 
        dry_run=args.dry_run
    )
    
    # Run migration
    stats = migrator.migrate_all_packages()
    
    # Run validation if requested
    if args.validate and not args.dry_run and stats['buck_files_migrated'] > 0:
        buck_files = migrator.find_package_buck_files()
        validation_results = migrator.validate_migrations(buck_files)
        
        failed_validations = [
            path for path, valid in validation_results.items() if not valid
        ]
        
        if failed_validations:
            print(f"\n❌ {len(failed_validations)} BUCK files failed validation:")
            for path in failed_validations:
                print(f"   - {path}")
    
    # Exit with error if there were errors
    return 1 if stats['errors'] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())