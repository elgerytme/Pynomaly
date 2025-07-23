#!/usr/bin/env python3
"""
Comprehensive Pynomaly Reference Cleanup
========================================
Fix ALL remaining pynomaly references in the repository.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set


class ComprehensivePynomalyFixer:
    """Fixes ALL remaining pynomaly references"""
    
    def __init__(self, repository_root: Path):
        self.repository_root = repository_root
        
        # More aggressive replacement patterns
        self.patterns = [
            # Case-insensitive word boundary replacements
            (r'\bpynomaly\b', 'anomaly_detection', re.IGNORECASE),
            (r'\bPynomaly\b', 'Anomaly Detection Platform', re.IGNORECASE),
            
            # Python package/module patterns
            (r'anomaly_detection', 'anomaly_detection', 0),
            (r'anomaly_detection', 'anomaly_detection', 0),
            
            # Import patterns  
            (r'from anomaly_detection', 'from anomaly_detection', 0),
            (r'import anomaly_detection', 'import anomaly_detection', 0),
            
            # URL/path patterns
            (r'/anomaly_detection/', '/anomaly-detection/', 0),
            (r'anomaly_detection/', 'anomaly-detection/', 0),
            
            # Configuration patterns
            (r'anomaly_detection-', 'anomaly-detection-', 0),
            (r'anomaly_detection_', 'anomaly_detection_', 0),
            
            # Docker/service patterns
            (r'anomaly_detection:', 'anomaly-detection:', 0),
            
            # Special cases for specific contexts
            (r'AnomalyDetectionClient', 'AnomalyDetectionClient', 0),
            (r'anomaly-detection', 'anomaly-detection', re.IGNORECASE),  # Fix typos
            
            # Database/service names
            (r'anomaly_detection_db', 'anomaly_detection_db', 0),
            (r'anomaly_detection-db', 'anomaly-detection-db', 0),
            
            # Environment variables
            (r'ANOMALY_DETECTION_', 'ANOMALY_DETECTION_', 0),
        ]
        
        # File extensions to process
        self.text_extensions = {
            '.py', '.md', '.rst', '.txt', '.yml', '.yaml', '.json',
            '.toml', '.cfg', '.ini', '.sh', '.bat', '.ps1', '.dockerfile',
            '.ts', '.js', '.tsx', '.jsx', '.vue', '.java', '.xml', '.env',
            '.template', '.sql', '.html', '.css'
        }
        
        # Directories to skip
        self.skip_dirs = {
            '.git', '__pycache__', '.pytest_cache', 'node_modules',
            '.venv', 'venv', 'env', '.env', 'environments', 'buck-out',
            '.mypy_cache', '.tox', 'htmlcov'
        }
        
        # Files to skip
        self.skip_files = {
            'package-lock.json',  # Large auto-generated file
        }
    
    def should_process_file(self, file_path: Path) -> bool:
        """Determine if a file should be processed"""
        # Skip if filename is excluded
        if file_path.name in self.skip_files:
            return False
            
        # Skip if in excluded directory
        for part in file_path.parts:
            if part in self.skip_dirs:
                return False
        
        # Skip if not a text file
        if file_path.suffix.lower() not in self.text_extensions:
            return False
            
        # Skip if too large (10MB limit)
        try:
            if file_path.stat().st_size > 10 * 1024 * 1024:
                return False
        except OSError:
            return False
            
        return True
    
    def contains_pynomaly(self, content: str) -> bool:
        """Check if content contains pynomaly references"""
        return bool(re.search(r'pynomaly', content, re.IGNORECASE))
    
    def fix_content(self, content: str, file_path: Path) -> tuple[str, List[str]]:
        """Fix all pynomaly references in content"""
        modified_content = content
        changes = []
        
        # Apply all patterns
        for pattern, replacement, flags in self.patterns:
            matches = re.findall(pattern, modified_content, flags)
            if matches:
                count = len(matches)
                modified_content = re.sub(pattern, replacement, modified_content, flags=flags)
                changes.append(f"Replaced {count} occurrences of '{pattern}' with '{replacement}'")
        
        return modified_content, changes
    
    def process_file(self, file_path: Path) -> tuple[bool, List[str]]:
        """Process a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                original_content = f.read()
        except (UnicodeDecodeError, OSError) as e:
            return False, [f"Could not read file: {e}"]
        
        # Skip if no pynomaly references
        if not self.contains_pynomaly(original_content):
            return False, []
        
        # Fix content
        modified_content, changes = self.fix_content(original_content, file_path)
        
        # Write back if changed
        if modified_content != original_content:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                return True, changes
            except OSError as e:
                return False, [f"Could not write file: {e}"]
        
        return False, []
    
    def fix_all_files(self) -> Dict[str, any]:
        """Fix all files in repository"""
        results = {
            'processed': 0,
            'changed': 0,
            'errors': [],
            'changed_files': []
        }
        
        print("🔍 Scanning for remaining pynomaly references...")
        
        for file_path in self.repository_root.rglob('*'):
            if not file_path.is_file():
                continue
                
            if not self.should_process_file(file_path):
                continue
            
            results['processed'] += 1
            relative_path = file_path.relative_to(self.repository_root)
            
            try:
                changed, changes = self.process_file(file_path)
                if changed:
                    results['changed'] += 1
                    results['changed_files'].append({
                        'file': str(relative_path),
                        'changes': changes
                    })
                    print(f"✅ Fixed {relative_path}")
                    for change in changes:
                        print(f"   • {change}")
                
            except Exception as e:
                error_msg = f"Error processing {relative_path}: {e}"
                results['errors'].append(error_msg)
                print(f"❌ {error_msg}")
        
        return results


def main():
    """Main entry point"""
    repository_root = Path(__file__).parent.parent
    
    print("🔧 Starting comprehensive pynomaly reference cleanup...")
    print(f"📁 Repository root: {repository_root}")
    
    fixer = ComprehensivePynomalyFixer(repository_root)
    results = fixer.fix_all_files()
    
    print(f"\n📊 Final Summary:")
    print(f"   Files processed: {results['processed']}")
    print(f"   Files changed: {results['changed']}")
    print(f"   Errors: {len(results['errors'])}")
    
    # Verify remaining count
    print(f"\n🔍 Checking for remaining pynomaly references...")
    import subprocess
    try:
        result = subprocess.run(['grep', '-ri', 'pynomaly', str(repository_root)], 
                              capture_output=True, text=True)
        remaining_lines = result.stdout.count('\n') if result.stdout else 0
        print(f"   Remaining pynomaly references: {remaining_lines}")
        
        if remaining_lines == 0:
            print("✅ All pynomaly references successfully removed!")
        else:
            print("⚠️ Some references may still remain")
            
    except FileNotFoundError:
        print("   Could not verify remaining count (grep not available)")
    
    if results['errors']:
        print(f"\n⚠️  {len(results['errors'])} errors occurred during processing")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())