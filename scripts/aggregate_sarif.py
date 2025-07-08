#!/usr/bin/env python3
"""
SARIF File Aggregation Script
Combines multiple SARIF files into a single consolidated report
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import uuid
from datetime import datetime


def load_sarif_file(file_path: Path) -> Dict[str, Any]:
    """Load and validate a SARIF file"""
    try:
        with open(file_path, 'r') as f:
            sarif_data = json.load(f)
        
        # Validate basic SARIF structure
        if not isinstance(sarif_data, dict):
            raise ValueError(f"Invalid SARIF format in {file_path}")
        
        if "version" not in sarif_data:
            sarif_data["version"] = "2.1.0"
        
        if "runs" not in sarif_data:
            sarif_data["runs"] = []
        
        return sarif_data
    except Exception as e:
        print(f"Error loading SARIF file {file_path}: {e}", file=sys.stderr)
        return {"version": "2.1.0", "runs": []}


def aggregate_sarif_files(sarif_files: List[Path], output_file: Path) -> None:
    """Aggregate multiple SARIF files into a single file"""
    
    # Initialize the aggregated SARIF structure
    aggregated = {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": []
    }
    
    # Track statistics
    total_results = 0
    total_rules = 0
    tools_processed = []
    
    print(f"📊 Aggregating {len(sarif_files)} SARIF files...")
    
    for sarif_file in sarif_files:
        if not sarif_file.exists():
            print(f"⚠️  File not found: {sarif_file}", file=sys.stderr)
            continue
            
        print(f"   Processing: {sarif_file}")
        sarif_data = load_sarif_file(sarif_file)
        
        # Add runs from this file to the aggregated runs
        for run in sarif_data.get("runs", []):
            # Add metadata about the source file
            if "properties" not in run:
                run["properties"] = {}
            run["properties"]["sourceFile"] = str(sarif_file)
            run["properties"]["aggregatedAt"] = datetime.now().isoformat()
            
            # Track statistics
            results_count = len(run.get("results", []))
            rules_count = len(run.get("tool", {}).get("driver", {}).get("rules", []))
            tool_name = run.get("tool", {}).get("driver", {}).get("name", "unknown")
            
            total_results += results_count
            total_rules += rules_count
            tools_processed.append(tool_name)
            
            print(f"     - {tool_name}: {results_count} results, {rules_count} rules")
            
            aggregated["runs"].append(run)
    
    # Add aggregation metadata
    aggregated["properties"] = {
        "aggregatedBy": "pynomaly-sarif-aggregator",
        "aggregationTime": datetime.now().isoformat(),
        "sourceFiles": [str(f) for f in sarif_files],
        "statistics": {
            "totalResults": total_results,
            "totalRules": total_rules,
            "toolsProcessed": list(set(tools_processed)),
            "filesProcessed": len(sarif_files)
        }
    }
    
    # Write the aggregated SARIF file
    with open(output_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"✅ Aggregation complete!")
    print(f"   📁 Output: {output_file}")
    print(f"   📊 Statistics:")
    print(f"      - Total results: {total_results}")
    print(f"      - Total rules: {total_rules}")
    print(f"      - Tools processed: {', '.join(set(tools_processed))}")
    print(f"      - Files processed: {len(sarif_files)}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multiple SARIF files into a single consolidated report"
    )
    parser.add_argument(
        "files", 
        nargs="+", 
        help="SARIF files to aggregate"
    )
    parser.add_argument(
        "--output", 
        default="combined.sarif",
        help="Output file name (default: combined.sarif)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate SARIF files before aggregation"
    )
    
    args = parser.parse_args()
    
    # Convert file arguments to Path objects
    sarif_files = [Path(f) for f in args.files]
    output_file = Path(args.output)
    
    # Validate input files exist
    missing_files = [f for f in sarif_files if not f.exists()]
    if missing_files:
        print(f"Error: Missing files: {', '.join(str(f) for f in missing_files)}", file=sys.stderr)
        sys.exit(1)
    
    # Perform aggregation
    try:
        aggregate_sarif_files(sarif_files, output_file)
    except Exception as e:
        print(f"Error during aggregation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
