#!/usr/bin/env python3
"""
Import Consolidation Refactoring Tool
====================================

Automatically refactor Python files to consolidate imports and fix import 
consolidation violations. This tool safely merges multiple import statements
from the same package into single consolidated statements.

Features:
- Safe import consolidation with backup creation
- Multiple import style handling (import vs from-import)
- PEP 8 compliant import organization
- Dry-run mode for preview
- Batch processing across packages
"""

import ast
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.import_consolidation_validator import ImportConsolidationValidator, ImportViolation
except ImportError:
    # Handle case where the validator is not available
    ImportConsolidationValidator = None
    ImportViolation = None


class ImportConsolidationRefactor:
    """
    Refactors Python files to consolidate imports according to the 
    single import per package rule.
    """
    
    def __init__(self, repository_root: Path = None, dry_run: bool = False):
        self.repository_root = repository_root or Path.cwd()
        self.packages_dir = self.repository_root / "src" / "packages"
        self.dry_run = dry_run
        self.logger = self._setup_logging()
        
        # Statistics tracking
        self.stats = {
            'files_processed': 0,
            'files_modified': 0,
            'imports_consolidated': 0,
            'violations_fixed': 0,
            'errors': 0
        }
        
        # Backup directory
        self.backup_dir = self.repository_root / "temp" / "import_refactor_backups" / datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger('import_consolidation_refactor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def refactor_all_packages(self) -> Dict[str, int]:
        """Refactor imports across all packages"""
        self.logger.info("🔧 Starting import consolidation refactoring")
        
        if not self.packages_dir.exists():
            self.logger.error(f"Packages directory not found: {self.packages_dir}")
            return self.stats
        
        # Create backup directory
        if not self.dry_run:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each domain directory
        for domain_dir in self.packages_dir.iterdir():
            if not domain_dir.is_dir():
                continue
                
            # Process each package in the domain
            for package_dir in domain_dir.iterdir():
                if not package_dir.is_dir():
                    continue
                    
                package_name = f"{domain_dir.name}.{package_dir.name}"
                self.logger.info(f"   Processing package: {package_name}")
                
                self._refactor_package(package_dir)
        
        # Generate summary
        self._log_summary()
        
        return self.stats
    
    def refactor_package(self, package_path: str) -> Dict[str, int]:
        """Refactor imports in a specific package"""
        package_dir = Path(package_path)
        
        if not package_dir.exists():
            self.logger.error(f"Package directory not found: {package_dir}")
            return self.stats
        
        if not self.dry_run:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self._refactor_package(package_dir)
        self._log_summary()
        
        return self.stats
    
    def refactor_files(self, file_paths: List[str]) -> Dict[str, int]:
        """Refactor imports in specific files"""
        if not self.dry_run:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        for file_path_str in file_paths:
            file_path = Path(file_path_str)
            if file_path.exists() and file_path.suffix == '.py':
                self._refactor_file(file_path)
        
        self._log_summary()
        
        return self.stats
    
    def _refactor_package(self, package_dir: Path) -> None:
        """Refactor imports in all Python files within a package"""
        python_files = list(package_dir.rglob("*.py"))
        
        for py_file in python_files:
            self._refactor_file(py_file)
    
    def _refactor_file(self, file_path: Path) -> None:
        """Refactor imports in a single Python file"""
        self.stats['files_processed'] += 1
        
        try:
            # Read original content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Parse AST and analyze imports
            tree = ast.parse(original_content)
            imports_to_consolidate = self._analyze_imports_for_consolidation(tree)
            
            if not imports_to_consolidate:
                return  # No consolidation needed
            
            # Generate refactored content
            refactored_content = self._consolidate_imports_in_content(
                original_content, imports_to_consolidate
            )
            
            if refactored_content == original_content:
                return  # No changes needed
            
            # Create backup if not dry run
            if not self.dry_run:
                self._create_backup(file_path, original_content)
                
                # Write refactored content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(refactored_content)
                
                self.logger.info(f"✅ Refactored: {file_path}")
            else:
                self.logger.info(f"📝 Would refactor: {file_path}")
            
            self.stats['files_modified'] += 1
            self.stats['violations_fixed'] += len(imports_to_consolidate)
            
        except Exception as e:
            self.logger.error(f"❌ Error refactoring {file_path}: {e}")
            self.stats['errors'] += 1
    
    def _analyze_imports_for_consolidation(self, tree: ast.AST) -> Dict[str, List[Dict]]:
        """Analyze AST to find imports that need consolidation"""
        imports_by_package = defaultdict(list)
        
        # Extract all imports with their details
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    package_name = self._extract_package_name(alias.name)
                    if package_name and not self._is_standard_library(package_name):
                        imports_by_package[package_name].append({
                            'type': 'import',
                            'line': node.lineno,
                            'module': alias.name,
                            'name': alias.asname,
                            'node': node
                        })
                        
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    package_name = self._extract_package_name(node.module)
                    if package_name and not self._is_standard_library(package_name):
                        items = []
                        for alias in node.names:
                            items.append({
                                'name': alias.name,
                                'asname': alias.asname
                            })
                        
                        imports_by_package[package_name].append({
                            'type': 'from',
                            'line': node.lineno,
                            'module': node.module,
                            'level': node.level,
                            'items': items,
                            'node': node
                        })
        
        # Return only packages with multiple imports
        return {
            package: import_list 
            for package, import_list in imports_by_package.items() 
            if len(import_list) > 1
        }
    
    def _consolidate_imports_in_content(
        self, 
        content: str, 
        imports_to_consolidate: Dict[str, List[Dict]]
    ) -> str:
        """Consolidate imports in file content"""
        lines = content.splitlines(keepends=True)
        
        # Track lines to remove and lines to add
        lines_to_remove = set()
        lines_to_add = {}  # line_number -> new_import_statement
        
        for package, import_list in imports_to_consolidate.items():
            # Determine consolidation strategy
            consolidated_import = self._create_consolidated_import(import_list)
            
            if consolidated_import:
                # Mark all original import lines for removal
                for import_info in import_list:
                    lines_to_remove.add(import_info['line'] - 1)  # Convert to 0-based
                
                # Add consolidated import at the position of the first import
                first_line = min(imp['line'] for imp in import_list) - 1
                lines_to_add[first_line] = consolidated_import
        
        # Apply changes
        new_lines = []
        for i, line in enumerate(lines):
            if i in lines_to_remove:
                # Skip this line (it's being consolidated)
                if i in lines_to_add:
                    # Add consolidated import instead
                    new_lines.append(lines_to_add[i] + '\n')
                continue
            elif i in lines_to_add:
                # Add consolidated import before this line
                new_lines.append(lines_to_add[i] + '\n')
            
            new_lines.append(line)
        
        return ''.join(new_lines)
    
    def _create_consolidated_import(self, import_list: List[Dict]) -> Optional[str]:
        """Create a consolidated import statement from multiple imports"""
        if not import_list:
            return None
        
        # Separate import types
        regular_imports = [imp for imp in import_list if imp['type'] == 'import']
        from_imports = [imp for imp in import_list if imp['type'] == 'from']
        
        # Strategy: prefer from-imports for consolidation
        if from_imports:
            # Consolidate from-imports
            return self._consolidate_from_imports(from_imports)
        elif regular_imports:
            # Multiple regular imports - convert to from-import if possible
            return self._consolidate_regular_imports(regular_imports)
        
        return None
    
    def _consolidate_from_imports(self, from_imports: List[Dict]) -> str:
        """Consolidate multiple from-imports into a single statement"""
        # Use the most specific module path as base
        base_module = from_imports[0]['module']
        level = from_imports[0].get('level', 0)
        
        # Collect all imported items
        all_items = []
        for imp in from_imports:
            for item in imp['items']:
                item_str = item['name']
                if item['asname']:
                    item_str += f" as {item['asname']}"
                all_items.append(item_str)
        
        # Remove duplicates while preserving order
        unique_items = []
        seen = set()
        for item in all_items:
            if item not in seen:
                unique_items.append(item)
                seen.add(item)
        
        # Format the consolidated import
        if len(unique_items) <= 3:
            items_str = ', '.join(unique_items)
        else:
            # Use parentheses for long import lists
            items_str = '(\n    ' + ',\n    '.join(unique_items) + '\n)'
        
        # Handle relative imports
        dots = '.' * level if level > 0 else ''
        
        return f"from {dots}{base_module} import {items_str}"
    
    def _consolidate_regular_imports(self, regular_imports: List[Dict]) -> str:
        """Consolidate multiple regular imports"""
        # For now, keep them as separate imports but group them
        # This is a conservative approach to avoid breaking existing code
        
        import_statements = []
        for imp in regular_imports:
            if imp['name']:
                import_statements.append(f"import {imp['module']} as {imp['name']}")
            else:
                import_statements.append(f"import {imp['module']}")
        
        return '\n'.join(import_statements)
    
    def _extract_package_name(self, module_path: str) -> Optional[str]:
        """Extract the base package name from a module path"""
        if not module_path or module_path.startswith('.'):
            return None
        
        # Handle src.packages.* imports
        if module_path.startswith('src.packages.'):
            parts = module_path.split('.')
            if len(parts) >= 4:
                return f"{parts[2]}.{parts[3]}"
            elif len(parts) >= 3:
                return parts[2]
        
        # Handle packages.* imports
        if module_path.startswith('packages.'):
            parts = module_path.split('.')
            if len(parts) >= 3:
                return f"{parts[1]}.{parts[2]}"
            elif len(parts) >= 2:
                return parts[1]
        
        # Return root package name
        return module_path.split('.')[0]
    
    def _is_standard_library(self, package_name: str) -> bool:
        """Check if package is part of Python standard library"""
        stdlib_modules = {
            'os', 'sys', 'json', 'yaml', 'datetime', 'typing', 'pathlib', 're',
            'subprocess', 'logging', 'collections', 'itertools', 'functools',
            'asyncio', 'concurrent', 'multiprocessing', 'threading', 'unittest',
            'ast', 'argparse', 'tempfile', 'shutil', 'base64', 'hashlib',
            'socket', 'http', 'urllib', 'email', 'csv', 'sqlite3'
        }
        
        root_name = package_name.split('.')[0]
        return root_name in stdlib_modules
    
    def _create_backup(self, file_path: Path, content: str) -> None:
        """Create a backup of the original file"""
        backup_path = self.backup_dir / file_path.relative_to(self.repository_root)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _log_summary(self) -> None:
        """Log refactoring summary"""
        self.logger.info("\n📊 Import Consolidation Summary:")
        self.logger.info(f"   Files processed: {self.stats['files_processed']}")
        self.logger.info(f"   Files modified: {self.stats['files_modified']}")
        self.logger.info(f"   Violations fixed: {self.stats['violations_fixed']}")
        self.logger.info(f"   Errors: {self.stats['errors']}")
        
        if not self.dry_run and self.stats['files_modified'] > 0:
            self.logger.info(f"   Backups created in: {self.backup_dir}")
    
    def restore_from_backup(self, backup_timestamp: str) -> None:
        """Restore files from a specific backup"""
        backup_path = self.repository_root / "temp" / "import_refactor_backups" / backup_timestamp
        
        if not backup_path.exists():
            self.logger.error(f"Backup not found: {backup_path}")
            return
        
        restored_count = 0
        
        for backup_file in backup_path.rglob("*.py"):
            original_path = self.repository_root / backup_file.relative_to(backup_path)
            
            if original_path.exists():
                shutil.copy2(backup_file, original_path)
                restored_count += 1
                self.logger.info(f"Restored: {original_path}")
        
        self.logger.info(f"✅ Restored {restored_count} files from backup")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Refactor imports to consolidate multiple imports from same package")
    parser.add_argument('--repository-root', type=str, default=".", help="Repository root directory")
    parser.add_argument('--package', type=str, help="Refactor specific package (domain.package format)")
    parser.add_argument('--files', nargs='*', help="Refactor specific files")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be changed without making changes")
    parser.add_argument('--restore-backup', type=str, help="Restore from backup timestamp (YYYYMMDD_HHMMSS)")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger('import_consolidation_refactor').setLevel(logging.DEBUG)
    
    refactor = ImportConsolidationRefactor(Path(args.repository_root), dry_run=args.dry_run)
    
    if args.restore_backup:
        # Restore from backup
        refactor.restore_from_backup(args.restore_backup)
        return 0
    
    if args.files:
        # Refactor specific files
        stats = refactor.refactor_files(args.files)
    elif args.package:
        # Refactor specific package
        domain, package = args.package.split('.')
        package_path = refactor.packages_dir / domain / package
        stats = refactor.refactor_package(str(package_path))
    else:
        # Refactor all packages
        stats = refactor.refactor_all_packages()
    
    # Exit with error if there were errors
    return 1 if stats['errors'] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())