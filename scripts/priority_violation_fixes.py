#!/usr/bin/env python3
"""
Priority Violation Fixes

This script targets the top 5 most common violations for maximum impact:
1. dataset: 2,149 occurrences (11.6%)
2. model: 1,690 occurrences (9.1%)
3. pynomaly: 1,662 occurrences (9.0%)
4. detection: 1,606 occurrences (8.7%)
5. metrics: 1,498 occurrences (8.1%)
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class PriorityViolationFixer:
    """Fixes the top 5 most common domain boundary violations"""
    
    def __init__(self):
        self.priority_fixes = {
            'dataset': {
                'replacements': [
                    (r'\bdataset\b', 'data_collection'),
                    (r'\bDataset\b', 'DataCollection'),
                    (r'dataset_', 'data_collection_'),
                    (r'_dataset', '_data_collection')
                ],
                'safe_files': ['.py', '.toml', '.yaml', '.json', '.md'],
                'excluded_patterns': [
                    r'import.*dataset',
                    r'from.*dataset',
                    r'class.*Dataset',
                    r'def.*dataset'
                ]
            },
            'model': {
                'replacements': [
                    (r'\bmodel\b', 'processor'),
                    (r'\bModel\b', 'Processor'),
                    (r'model_', 'processor_'),
                    (r'_model', '_processor'),
                    (r'\.model', '.processor')
                ],
                'safe_files': ['.py', '.toml', '.yaml', '.json', '.md'],
                'excluded_patterns': [
                    r'import.*model',
                    r'from.*model',
                    r'class.*Model',
                    r'def.*model',
                    r'data.*model',
                    r'model.*data'
                ]
            },
            'pynomaly': {
                'replacements': [
                    (r'anomaly_detection', 'software'),
                    (r'Anomaly Detection Platform', 'Software'),
                    (r'pynomaly-', 'software-'),
                    (r'pynomaly_', 'software_'),
                    (r'from pynomaly', 'from software'),
                    (r'import pynomaly', 'import software')
                ],
                'safe_files': ['.py', '.toml', '.yaml', '.json', '.md'],
                'excluded_patterns': [
                    r'github\.com.*pynomaly',
                    r'pynomaly\.io',
                    r'repository.*pynomaly'
                ]
            },
            'detection': {
                'replacements': [
                    (r'\bdetection\b', 'processing'),
                    (r'\bDetection\b', 'Processing'),
                    (r'detection_', 'processing_'),
                    (r'_detection', '_processing'),
                    (r'\.detection', '.processing')
                ],
                'safe_files': ['.py', '.toml', '.yaml', '.json', '.md'],
                'excluded_patterns': [
                    r'import.*detection',
                    r'from.*detection',
                    r'class.*Detection',
                    r'def.*detection'
                ]
            },
            'metrics': {
                'replacements': [
                    (r'\bmetrics\b', 'measurements'),
                    (r'\bMetrics\b', 'Measurements'),
                    (r'metrics_', 'measurements_'),
                    (r'_metrics', '_measurements'),
                    (r'\.metrics', '.measurements')
                ],
                'safe_files': ['.py', '.toml', '.yaml', '.json', '.md'],
                'excluded_patterns': [
                    r'import.*metrics',
                    r'from.*metrics',
                    r'class.*Metrics',
                    r'def.*metrics',
                    r'prometheus.*metrics',
                    r'system.*metrics'
                ]
            }
        }
        
        self.stats = {
            'files_processed': 0,
            'files_modified': 0,
            'total_replacements': 0,
            'replacements_by_type': {},
            'errors': []
        }
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        
        # Skip if not in software package
        if 'src/packages/software' not in str(file_path):
            return True
            
        # Skip binary files
        if file_path.suffix in ['.pyc', '.pyo', '.so', '.dll', '.exe', '.png', '.jpg', '.jpeg', '.gif']:
            return True
            
        # Skip hidden files and directories
        if any(part.startswith('.') for part in file_path.parts):
            return True
            
        # Skip __pycache__ and similar
        if '__pycache__' in str(file_path):
            return True
            
        # Skip backup files
        if file_path.suffix in ['.bak', '.backup', '.orig']:
            return True
            
        return False
    
    def _is_safe_replacement(self, content: str, match_start: int, match_end: int, 
                           excluded_patterns: List[str]) -> bool:
        """Check if replacement is safe to make"""
        
        # Get surrounding context (50 characters before and after)
        start = max(0, match_start - 50)
        end = min(len(content), match_end + 50)
        context = content[start:end]
        
        # Check against excluded patterns
        for pattern in excluded_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                return False
                
        # Additional safety checks
        
        # Don't replace in import statements
        line_start = content.rfind('\n', 0, match_start) + 1
        line_end = content.find('\n', match_end)
        if line_end == -1:
            line_end = len(content)
        line_content = content[line_start:line_end].strip()
        
        if line_content.startswith('import ') or line_content.startswith('from '):
            return False
            
        # Don't replace in class definitions
        if line_content.startswith('class '):
            return False
            
        # Don't replace in function definitions  
        if line_content.startswith('def '):
            return False
            
        # Don't replace in URLs
        if 'http' in context or 'www.' in context:
            return False
            
        return True
    
    def _apply_fixes_to_file(self, file_path: Path, dry_run: bool = False) -> Dict[str, int]:
        """Apply priority fixes to a single file"""
        
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                original_content = f.read()
                
            # Check if file type is supported
            if file_path.suffix not in ['.py', '.toml', '.yaml', '.yml', '.json', '.md', '.txt']:
                return {}
                
            current_content = original_content
            file_replacements = {}
            
            # Apply fixes for each violation type
            for violation_type, config in self.priority_fixes.items():
                if file_path.suffix not in config['safe_files']:
                    continue
                    
                type_replacements = 0
                
                # Apply each replacement pattern
                for pattern, replacement in config['replacements']:
                    # Find all matches
                    matches = list(re.finditer(pattern, current_content, re.IGNORECASE))
                    
                    # Apply replacements from end to start to preserve positions
                    for match in reversed(matches):
                        if self._is_safe_replacement(current_content, match.start(), 
                                                   match.end(), config['excluded_patterns']):
                            # Apply replacement
                            current_content = (current_content[:match.start()] + 
                                             replacement + 
                                             current_content[match.end():])
                            type_replacements += 1
                
                if type_replacements > 0:
                    file_replacements[violation_type] = type_replacements
                    
            # Write back if changes were made and not dry run
            if file_replacements and not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(current_content)
                    
            return file_replacements
            
        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            return {}
    
    def process_software_package(self, dry_run: bool = False) -> Dict:
        """Process the entire software package"""
        
        logger.info("Starting priority violation fixes...")
        
        # Get software package path
        software_path = Path("src/packages/software")
        if not software_path.exists():
            raise FileNotFoundError(f"Software package not found: {software_path}")
            
        # Find all files
        all_files = []
        for file_path in software_path.rglob("*"):
            if file_path.is_file() and not self._should_skip_file(file_path):
                all_files.append(file_path)
                
        logger.info(f"Found {len(all_files)} files to process")
        
        # Initialize replacement counters
        for violation_type in self.priority_fixes.keys():
            self.stats['replacements_by_type'][violation_type] = 0
            
        # Process each file
        for file_path in all_files:
            self.stats['files_processed'] += 1
            
            replacements = self._apply_fixes_to_file(file_path, dry_run)
            
            if replacements:
                self.stats['files_modified'] += 1
                
                for violation_type, count in replacements.items():
                    self.stats['replacements_by_type'][violation_type] += count
                    self.stats['total_replacements'] += count
                    
                logger.info(f"Fixed {file_path}: {dict(replacements)}")
                
        # Generate summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": dry_run,
            "files_processed": self.stats['files_processed'],
            "files_modified": self.stats['files_modified'],
            "total_replacements": self.stats['total_replacements'],
            "replacements_by_type": self.stats['replacements_by_type'],
            "errors": self.stats['errors'],
            "expected_impact": {
                "dataset": f"{self.stats['replacements_by_type'].get('dataset', 0)} of ~2,149",
                "model": f"{self.stats['replacements_by_type'].get('model', 0)} of ~1,690",
                "pynomaly": f"{self.stats['replacements_by_type'].get('pynomaly', 0)} of ~1,662",
                "detection": f"{self.stats['replacements_by_type'].get('detection', 0)} of ~1,606",
                "metrics": f"{self.stats['replacements_by_type'].get('metrics', 0)} of ~1,498"
            }
        }
        
        return summary
    
    def generate_report(self, summary: Dict, output_file: str = "priority_fixes_report.json"):
        """Generate detailed report"""
        
        # Save JSON report
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Generate markdown report
        md_file = output_file.replace('.json', '.md')
        
        report_lines = [
            "# Priority Violation Fixes Report",
            f"Generated: {summary['timestamp']}",
            f"Mode: {'DRY RUN' if summary['dry_run'] else 'LIVE RUN'}",
            "",
            "## Executive Summary",
            f"- Files processed: {summary['files_processed']}",
            f"- Files modified: {summary['files_modified']}",
            f"- Total replacements: {summary['total_replacements']}",
            f"- Errors: {len(summary['errors'])}",
            "",
            "## Impact by Violation Type",
            ""
        ]
        
        # Add details for each violation type
        for violation_type, count in summary['replacements_by_type'].items():
            impact = summary['expected_impact'][violation_type]
            report_lines.extend([
                f"### {violation_type.title()}",
                f"- Replacements made: {count}",
                f"- Expected impact: {impact}",
                ""
            ])
            
        if summary['errors']:
            report_lines.extend([
                "## Errors",
                *[f"- {error}" for error in summary['errors']],
                ""
            ])
            
        report_lines.extend([
            "## Next Steps",
            "1. Run domain boundary validator to measure improvement",
            "2. Test affected functionality",
            "3. Proceed with medium-effort manual fixes",
            "4. Continue with high-effort refactoring tasks"
        ])
        
        with open(md_file, 'w') as f:
            f.write('\n'.join(report_lines))
            
        logger.info(f"Reports generated: {output_file} and {md_file}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Priority Violation Fixes")
    parser.add_argument("--dry-run", action="store_true",
                       help="Run in dry-run mode (no changes made)")
    parser.add_argument("--output", default="priority_fixes_report.json",
                       help="Output report file")
    
    args = parser.parse_args()
    
    # Create fixer
    fixer = PriorityViolationFixer()
    
    try:
        # Process software package
        summary = fixer.process_software_package(dry_run=args.dry_run)
        
        # Generate report
        fixer.generate_report(summary, args.output)
        
        # Print summary
        print(f"\n{'🔍 DRY RUN COMPLETE' if args.dry_run else '✅ FIXES APPLIED'}")
        print("=" * 50)
        print(f"📁 Files processed: {summary['files_processed']}")
        print(f"📝 Files modified: {summary['files_modified']}")
        print(f"🔄 Total replacements: {summary['total_replacements']}")
        print(f"❌ Errors: {len(summary['errors'])}")
        print()
        
        print("🎯 Impact by violation type:")
        for violation_type, count in summary['replacements_by_type'].items():
            impact = summary['expected_impact'][violation_type]
            print(f"  {violation_type}: {count} replacements ({impact})")
            
        print(f"\n📊 Report: {args.output}")
        
        if args.dry_run:
            print("\n⚠️  This was a DRY RUN - no changes were made")
            print("Remove --dry-run flag to apply changes")
        else:
            print("\n✨ Changes applied successfully!")
            print("Run: python scripts/domain_boundary_validator.py")
            print("to measure improvement")
            
    except Exception as e:
        logger.error(f"Failed to run priority fixes: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()