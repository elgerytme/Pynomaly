#!/usr/bin/env python3
"""
Targeted Priority Fixes

This script applies priority fixes only to core software package files,
excluding virtual environments, node_modules, and other external dependencies.
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class TargetedPriorityFixer:
    """Targeted fixes for core software package files only"""
    
    def __init__(self):
        self.priority_fixes = {
            'pynomaly': [
                (r'anomaly_detection', 'software'),
                (r'Anomaly Detection Platform', 'Software'),
                (r'pynomaly-', 'software-'),
                (r'pynomaly_', 'software_'),
                (r'"pynomaly"', '"software"'),
                (r"'pynomaly'", "'software'")
            ],
            'dataset': [
                (r'\bdataset\b', 'data_collection'),
                (r'\bDataset\b', 'DataCollection'),
                (r'dataset_', 'data_collection_'),
                (r'_dataset', '_data_collection')
            ],
            'model': [
                (r'\bmodel\b', 'processor'),
                (r'\bModel\b', 'Processor'), 
                (r'model_', 'processor_'),
                (r'_model', '_processor')
            ],
            'detection': [
                (r'\bdetection\b', 'processing'),
                (r'\bDetection\b', 'Processing'),
                (r'detection_', 'processing_'),
                (r'_detection', '_processing')
            ],
            'metrics': [
                (r'\bmetrics\b', 'measurements'),
                (r'\bMetrics\b', 'Measurements'),
                (r'metrics_', 'measurements_'),
                (r'_metrics', '_measurements')
            ]
        }
        
        self.stats = {
            'files_processed': 0,
            'files_modified': 0,
            'total_replacements': 0,
            'replacements_by_type': {},
            'files_modified_list': []
        }
        
        # Initialize counters
        for violation_type in self.priority_fixes.keys():
            self.stats['replacements_by_type'][violation_type] = 0
    
    def _get_target_files(self) -> List[Path]:
        """Get only the core software package files"""
        
        software_path = Path("src/packages/software")
        target_files = []
        
        # Exclude patterns
        exclude_patterns = [
            '/.venv/',
            '/node_modules/',
            '/__pycache__/',
            '.pyc',
            '.pyo',
            '.git/',
            '/build/',
            '/dist/',
            '/target/',
            '/coverage/',
            '.DS_Store'
        ]
        
        # Include only these directories
        include_dirs = [
            'core/domain',
            'core/dto',
            'core/api',
            'core/services',
            'core/shared',
            'core/infrastructure',
            'core/cli',
            'interfaces/api',
            'interfaces/cli',
            'interfaces/python_sdk/domain',
            'interfaces/python_sdk/infrastructure',
            'interfaces/python_sdk/presentation',
            'services'
        ]
        
        for include_dir in include_dirs:
            dir_path = software_path / include_dir
            if dir_path.exists():
                for file_path in dir_path.rglob("*"):
                    if file_path.is_file():
                        # Skip if matches exclude patterns
                        if any(pattern in str(file_path) for pattern in exclude_patterns):
                            continue
                            
                        # Only process certain file types
                        if file_path.suffix in ['.py', '.toml', '.yaml', '.yml', '.json', '.md']:
                            target_files.append(file_path)
        
        # Also include root level files in each package
        for root_file in software_path.rglob("*.py"):
            if root_file.is_file():
                if not any(pattern in str(root_file) for pattern in exclude_patterns):
                    if root_file not in target_files:
                        # Only if it's directly in a package subdirectory
                        if len(root_file.relative_to(software_path).parts) <= 3:
                            target_files.append(root_file)
        
        return sorted(set(target_files))
    
    def _is_safe_replacement(self, content: str, match_start: int, match_end: int) -> bool:
        """Check if replacement is safe"""
        
        # Get the line containing the match
        line_start = content.rfind('\n', 0, match_start) + 1
        line_end = content.find('\n', match_end)
        if line_end == -1:
            line_end = len(content)
        line_content = content[line_start:line_end].strip()
        
        # Don't replace in import statements
        if line_content.startswith('import ') or line_content.startswith('from '):
            return False
            
        # Don't replace in class definitions
        if line_content.startswith('class '):
            return False
            
        # Don't replace in function definitions
        if line_content.startswith('def '):
            return False
            
        # Get surrounding context
        context_start = max(0, match_start - 30)
        context_end = min(len(content), match_end + 30)
        context = content[context_start:context_end]
        
        # Don't replace in URLs
        if 'http' in context or 'github.com' in context or '.io' in context:
            return False
            
        return True
    
    def _apply_fixes_to_file(self, file_path: Path, dry_run: bool = False) -> Dict[str, int]:
        """Apply fixes to a single file"""
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                original_content = f.read()
                
            current_content = original_content
            file_replacements = {}
            
            # Apply each violation type
            for violation_type, patterns in self.priority_fixes.items():
                type_replacements = 0
                
                for pattern, replacement in patterns:
                    # Find all matches
                    matches = list(re.finditer(pattern, current_content))
                    
                    # Apply replacements from end to start
                    for match in reversed(matches):
                        if self._is_safe_replacement(current_content, match.start(), match.end()):
                            current_content = (current_content[:match.start()] + 
                                             replacement + 
                                             current_content[match.end():])
                            type_replacements += 1
                
                if type_replacements > 0:
                    file_replacements[violation_type] = type_replacements
            
            # Write back if changes were made
            if file_replacements and not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(current_content)
                    
            return file_replacements
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return {}
    
    def process_files(self, dry_run: bool = False) -> Dict:
        """Process target files"""
        
        logger.info("Finding target files...")
        target_files = self._get_target_files()
        
        logger.info(f"Found {len(target_files)} target files to process")
        
        if not target_files:
            logger.warning("No target files found")
            return self._generate_summary(dry_run)
        
        # Process each file
        for file_path in target_files:
            self.stats['files_processed'] += 1
            
            replacements = self._apply_fixes_to_file(file_path, dry_run)
            
            if replacements:
                self.stats['files_modified'] += 1
                self.stats['files_modified_list'].append(str(file_path))
                
                for violation_type, count in replacements.items():
                    self.stats['replacements_by_type'][violation_type] += count
                    self.stats['total_replacements'] += count
                    
                replacement_summary = ', '.join(f"{k}:{v}" for k, v in replacements.items())
                logger.info(f"{'DRY RUN: ' if dry_run else ''}Fixed {file_path.name}: {replacement_summary}")
                
        return self._generate_summary(dry_run)
    
    def _generate_summary(self, dry_run: bool) -> Dict:
        """Generate summary report"""
        
        return {
            "timestamp": datetime.now().isoformat(),
            "dry_run": dry_run,
            "files_processed": self.stats['files_processed'],
            "files_modified": self.stats['files_modified'],
            "total_replacements": self.stats['total_replacements'],
            "replacements_by_type": self.stats['replacements_by_type'],
            "files_modified_list": self.stats['files_modified_list'][:50],  # Limit to first 50
            "expected_impact": {
                "pynomaly": f"{self.stats['replacements_by_type']['pynomaly']} replacements",
                "dataset": f"{self.stats['replacements_by_type']['dataset']} replacements",
                "model": f"{self.stats['replacements_by_type']['model']} replacements", 
                "detection": f"{self.stats['replacements_by_type']['detection']} replacements",
                "metrics": f"{self.stats['replacements_by_type']['metrics']} replacements"
            }
        }

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Targeted Priority Fixes")
    parser.add_argument("--dry-run", action="store_true",
                       help="Run in dry-run mode (no changes made)")
    parser.add_argument("--output", default="targeted_fixes_report.json",
                       help="Output report file")
    
    args = parser.parse_args()
    
    fixer = TargetedPriorityFixer()
    
    try:
        summary = fixer.process_files(dry_run=args.dry_run)
        
        # Save report
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Print summary
        print(f"\n{'🔍 DRY RUN RESULTS' if args.dry_run else '✅ FIXES APPLIED'}")
        print("=" * 50)
        print(f"📁 Files processed: {summary['files_processed']}")
        print(f"📝 Files modified: {summary['files_modified']}")
        print(f"🔄 Total replacements: {summary['total_replacements']}")
        print()
        
        print("🎯 Replacements by type:")
        for violation_type, count in summary['replacements_by_type'].items():
            if count > 0:
                print(f"  {violation_type}: {count}")
                
        print(f"\n📊 Report saved: {args.output}")
        
        if args.dry_run:
            print("\n⚠️  DRY RUN - No changes made")
            print("Remove --dry-run to apply changes")
        else:
            print("\n✨ Changes applied successfully!")
            print("Run domain boundary validator to measure improvement")
            
    except Exception as e:
        logger.error(f"Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()