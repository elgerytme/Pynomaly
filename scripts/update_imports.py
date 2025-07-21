#!/usr/bin/env python3
"""
Import Statement Update Script
============================
Updates import statements for the new DDD structure.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict
import argparse

class ImportUpdater:
    """Updates import statements for new package structure"""
    
    def __init__(self, package_path: Path):
        self.package_path = package_path
        self.import_mappings = self._create_import_mappings()
    
    def _create_import_mappings(self) -> List[Tuple[str, str]]:
        """Create mapping of old imports to new imports"""
        return [
            # Core module relocations
            (r"from \.core\.dependency_injection import", "from .infrastructure.config.dependency_injection import"),
            (r"from \.core\.domain_entities import", "from .infrastructure.config.domain_entities import"),
            (r"from \.core\.security_configuration import", "from .infrastructure.security.security_configuration import"),
            (r"from \.core\.performance_optimization import", "from .infrastructure.monitoring.performance_optimization import"),
            
            # Services relocations  
            (r"from \.services\.([a-zA-Z_]+) import", r"from .application.services.\1 import"),
            (r"from services\.([a-zA-Z_]+) import", r"from application.services.\1 import"),
            
            # Ecosystem to integrations
            (r"from \.ecosystem\.([a-zA-Z_]+) import", r"from .infrastructure.integrations.\1 import"),
            (r"from ecosystem\.([a-zA-Z_]+) import", r"from infrastructure.integrations.\1 import"),
            
            # Enterprise to infrastructure/enterprise
            (r"from \.enterprise\.([a-zA-Z_]+) import", r"from .infrastructure.enterprise.\1 import"),
            (r"from enterprise\.([a-zA-Z_]+) import", r"from infrastructure.enterprise.\1 import"),
            
            # Enhanced features to application/services/enhanced
            (r"from \.enhanced_features\.([a-zA-Z_]+) import", r"from .application.services.enhanced.\1 import"),
            (r"from enhanced_features\.([a-zA-Z_]+) import", r"from application.services.enhanced.\1 import"),
            
            # SDK to presentation
            (r"from \.sdk\.([a-zA-Z_]+) import", r"from .presentation.sdk.\1 import"),
            (r"from sdk\.([a-zA-Z_]+) import", r"from presentation.sdk.\1 import"),
            
            # Long absolute imports from old structure
            (r"from src\.packages\.data\.anomaly_detection\.services\.([a-zA-Z_]+) import", 
             r"from anomaly_detection.application.services.\1 import"),
            (r"from src\.packages\.data\.anomaly_detection\.core\.([a-zA-Z_]+) import",
             r"from anomaly_detection.infrastructure.config.\1 import"),
            (r"from src\.packages\.data\.anomaly_detection\.ecosystem\.([a-zA-Z_]+) import",
             r"from anomaly_detection.infrastructure.integrations.\1 import"),
            (r"from src\.packages\.data\.anomaly_detection\.enterprise\.([a-zA-Z_]+) import",
             r"from anomaly_detection.infrastructure.enterprise.\1 import"),
            
            # Relative imports cleanup
            (r"from \.\.\.([a-zA-Z_]+)\.([a-zA-Z_]+) import", r"from anomaly_detection.\1.\2 import"),
            (r"from \.\.([a-zA-Z_]+)\.([a-zA-Z_]+) import", r"from .\1.\2 import"),
            
            # Common problematic patterns
            (r"from pynomaly_detection\.services\.([a-zA-Z_]+) import", 
             r"from anomaly_detection.application.services.\1 import"),
            (r"from pynomaly_detection\.core\.([a-zA-Z_]+) import",
             r"from anomaly_detection.infrastructure.config.\1 import"),
        ]
    
    def update_file_imports(self, file_path: Path) -> bool:
        """Update imports in a single file"""
        if not file_path.suffix == '.py':
            return False
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            changes_made = False
            
            # Apply import mappings
            for old_pattern, new_pattern in self.import_mappings:
                new_content = re.sub(old_pattern, new_pattern, content)
                if new_content != content:
                    content = new_content
                    changes_made = True
            
            # Write back if changes were made
            if changes_made:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  ✅ Updated: {file_path.relative_to(self.package_path)}")
                return True
            
            return False
            
        except Exception as e:
            print(f"  ⚠️  Error updating {file_path}: {e}")
            return False
    
    def update_all_imports(self) -> Dict[str, int]:
        """Update imports in all Python files"""
        print(f"🔄 Updating imports in: {self.package_path}")
        
        stats = {"files_checked": 0, "files_updated": 0, "errors": 0}
        
        # Walk through all Python files
        for py_file in self.package_path.rglob("*.py"):
            if py_file.name == '__pycache__':
                continue
                
            stats["files_checked"] += 1
            
            try:
                if self.update_file_imports(py_file):
                    stats["files_updated"] += 1
            except Exception as e:
                stats["errors"] += 1
                print(f"  ❌ Error processing {py_file}: {e}")
        
        return stats
    
    def validate_imports(self) -> List[str]:
        """Validate that imports are working after updates"""
        print("🔍 Validating imports...")
        
        issues = []
        
        # Try importing main module
        try:
            import sys
            sys.path.insert(0, str(self.package_path / "src"))
            
            import anomaly_detection
            print("  ✅ Main package imports successfully")
            
        except ImportError as e:
            issues.append(f"Main package import failed: {e}")
            print(f"  ❌ Main package import failed: {e}")
        
        return issues
    
    def create_import_report(self, stats: Dict[str, int]) -> str:
        """Create a summary report of import updates"""
        report = f"""
Import Update Report
==================

Package: {self.package_path.name}
Files checked: {stats['files_checked']}
Files updated: {stats['files_updated']}  
Errors encountered: {stats['errors']}

Update rate: {(stats['files_updated']/stats['files_checked']*100):.1f}%

Import Mappings Applied:
{chr(10).join(f"  • {old} -> {new}" for old, new in self.import_mappings[:5])}
  ... and {len(self.import_mappings) - 5} more mappings

Status: {'✅ SUCCESS' if stats['errors'] == 0 else '⚠️ COMPLETED WITH ERRORS'}
"""
        return report

def main():
    parser = argparse.ArgumentParser(description="Update imports for new package structure")
    parser.add_argument('--package-path', required=True, help='Path to the migrated package')
    parser.add_argument('--validate', action='store_true', help='Validate imports after update')
    parser.add_argument('--report', help='Output path for update report')
    
    args = parser.parse_args()
    
    package_path = Path(args.package_path)
    if not package_path.exists():
        print(f"❌ Package path not found: {package_path}")
        return 1
    
    # Update imports
    updater = ImportUpdater(package_path)
    stats = updater.update_all_imports()
    
    # Validate imports if requested
    if args.validate:
        issues = updater.validate_imports()
        if issues:
            print("❌ Import validation issues found:")
            for issue in issues:
                print(f"  • {issue}")
    
    # Generate report
    report = updater.create_import_report(stats)
    print(report)
    
    if args.report:
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to: {args.report}")
    
    print(f"\n🎉 Import update completed!")
    return 0 if stats['errors'] == 0 else 1

if __name__ == "__main__":
    exit(main())