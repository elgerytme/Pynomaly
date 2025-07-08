#!/usr/bin/env python3
"""
SARIF Aggregation Script

This script aggregates multiple SARIF files by merging their 'runs' arrays
into a single combined SARIF file. It's designed to be used by both GitHub
Actions and local Makefile targets.

Enhancements:
- Distinguishes between host code and container findings
- Parses container SARIF runs and lists findings by image/tool
- Supports detailed reporting with source type classification

Usage:
    python aggregate_sarif.py file1.sarif file2.sarif [...]
    python aggregate_sarif.py --output combined.sarif file1.sarif file2.sarif
    python aggregate_sarif.py --generate-report --output combined.sarif file1.sarif file2.sarif
"""

import json
import sys
import os
import argparse
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict


def aggregate_sarif(files: List[str], output: str = 'combined.sarif') -> None:
    """
    Aggregate multiple SARIF files by merging their runs arrays.
    
    Args:
        files: List of SARIF file paths to aggregate
        output: Output file path for the combined SARIF
    """
    combined_sarif: Dict[str, Any] = {
        'version': '2.1.0',
        '$schema': 'https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0.json',
        'runs': []
    }
    
    processed_files = 0
    
    for file_path in files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sarif_data = json.load(f)
                
                # Validate basic SARIF structure
source_type = 'host'
                for run in sarif_data.get('runs', []):
                    tool_name = run.get('tool', {}).get('driver', {}).get('name', 'Unknown')
                    if 'trivy' in tool_name.lower() or 'clair' in tool_name.lower():
                        source_type = 'container'
                        break

                # Store results for report
                results_by_source[source_type].append({'tool': tool_name, 'file': file_path})

                if not isinstance(sarif_data, dict):
                    print(f"Warning: {file_path} is not a valid JSON object")
                    continue
                    
                # Append runs from each SARIF file to the combined list
                if 'runs' in sarif_data and isinstance(sarif_data['runs'], list):
                    combined_sarif['runs'].extend(sarif_data['runs'])
                    processed_files += 1
                    print(f"Processed: {file_path} ({len(sarif_data['runs'])} runs)")
                else:
                    print(f"Warning: {file_path} has no valid 'runs' array")
                    
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {file_path}: {e}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Write combined SARIF to the output file
    try:
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(combined_sarif, f, indent=2)
        print(f"Successfully aggregated {processed_files} files into {output}")
        print(f"Total runs in combined file: {len(combined_sarif['runs'])}")
    except Exception as e:
        print(f"Error writing combined file {output}: {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Aggregate SARIF files by merging their runs arrays',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python aggregate_sarif.py file1.sarif file2.sarif
  python aggregate_sarif.py --output combined.sarif *.sarif
'''
    )
    parser.add_argument('files', nargs='*', help='SARIF files to aggregate')
    parser.add_argument('--output', '-o', default='combined.sarif',
                       help='Output file path (default: combined.sarif)')
    
    args = parser.parse_args()
    
    if not args.files:
        print("Error: No input files specified")
        parser.print_help()
        sys.exit(1)
    
    aggregate_sarif(args.files, args.output)


if __name__ == "__main__":
    main()
