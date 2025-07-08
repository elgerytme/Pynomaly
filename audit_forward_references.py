#!/usr/bin/env python3
"""
Comprehensive audit script to find all FastAPI helper type hint forward reference issues.

This script scans the codebase for patterns that could cause TypeAdapter forward-reference 
issues with FastAPI helpers like Query, Depends, Body, Path, etc.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


def scan_file_for_patterns(file_path: Path) -> Dict[str, List[Tuple[int, str]]]:
    """
    Scan a file for forward reference patterns related to FastAPI helpers.
    
    Returns:
        Dict mapping pattern names to list of (line_number, line_content) tuples
    """
    patterns = {
        'query_usage': [
            r'Query\(',
            r'from fastapi import.*Query',
            r'typing\.Annotated.*Query',
            r'Annotated.*Query',
        ],
        'depends_usage': [
            r'Depends\(',
            r'from fastapi import.*Depends',
            r'typing\.Annotated.*Depends',
            r'Annotated.*Depends',
        ],
        'body_usage': [
            r'Body\(',
            r'from fastapi import.*Body',
            r'typing\.Annotated.*Body',
            r'Annotated.*Body',
        ],
        'path_usage': [
            r'Path\(',
            r'from fastapi import.*Path',
            r'typing\.Annotated.*Path',
            r'Annotated.*Path',
        ],
        'header_usage': [
            r'Header\(',
            r'from fastapi import.*Header',
            r'typing\.Annotated.*Header',
            r'Annotated.*Header',
        ],
        'cookie_usage': [
            r'Cookie\(',
            r'from fastapi import.*Cookie',
            r'typing\.Annotated.*Cookie',
            r'Annotated.*Cookie',
        ],
        'file_usage': [
            r'File\(',
            r'from fastapi import.*File',
            r'typing\.Annotated.*File',
            r'Annotated.*File',
        ],
        'form_usage': [
            r'Form\(',
            r'from fastapi import.*Form',
            r'typing\.Annotated.*Form',
            r'Annotated.*Form',
        ],
        'security_usage': [
            r'Security\(',
            r'from fastapi import.*Security',
            r'typing\.Annotated.*Security',
            r'Annotated.*Security',
        ],
        'forward_ref_patterns': [
            r'ForwardRef\(',
            r'typing\.ForwardRef',
            r'TypeAdapter\[.*ForwardRef',
            r'PydanticUndefined',
        ],
        'type_adapter_usage': [
            r'TypeAdapter\(',
            r'from pydantic import.*TypeAdapter',
            r'TypeAdapter\[.*\]',
        ],
        'request_parameter_patterns': [
            r'Request.*=.*Query',
            r'Request.*=.*Depends',
            r'Request.*=.*Body',
            r'Request.*=.*Path',
            r'.*Request.*Query\(',
            r'.*Request.*Depends\(',
        ],
        'model_rebuild_patterns': [
            r'\.model_rebuild\(',
            r'\.rebuild\(',
        ],
        'function_parameter_annotations': [
            r':\s*Annotated\[.*Query',
            r':\s*Annotated\[.*Depends',
            r':\s*Annotated\[.*Body',
            r':\s*Annotated\[.*Path',
        ]
    }
    
    results = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return results
    
    for pattern_name, pattern_list in patterns.items():
        matches = []
        for line_num, line in enumerate(lines, 1):
            for pattern in pattern_list:
                if re.search(pattern, line):
                    matches.append((line_num, line.strip()))
                    break  # Only count once per line
        
        if matches:
            results[pattern_name] = matches
    
    return results


def find_python_files(root_dir: Path) -> List[Path]:
    """Find all Python files in the codebase."""
    python_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # Skip certain directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    return python_files


def main():
    """Main audit function."""
    print("=" * 80)
    print("PYNOMALY FASTAPI FORWARD REFERENCE AUDIT")
    print("=" * 80)
    
    # Get project root
    script_dir = Path(__file__).parent
    src_dir = script_dir / "src"
    
    if not src_dir.exists():
        print(f"Error: Source directory {src_dir} not found")
        return
    
    print(f"Scanning directory: {src_dir}")
    print(f"Looking for FastAPI helper forward reference issues...")
    print("-" * 80)
    
    # Find all Python files
    python_files = find_python_files(src_dir)
    print(f"Found {len(python_files)} Python files to scan")
    
    # Scan each file
    all_results = {}
    total_issues = 0
    
    for file_path in python_files:
        results = scan_file_for_patterns(file_path)
        if results:
            relative_path = file_path.relative_to(script_dir)
            all_results[relative_path] = results
            
            # Count total issues
            for pattern_matches in results.values():
                total_issues += len(pattern_matches)
    
    print(f"\n📊 AUDIT SUMMARY")
    print("=" * 80)
    print(f"Files scanned: {len(python_files)}")
    print(f"Files with issues: {len(all_results)}")
    print(f"Total pattern matches: {total_issues}")
    
    # Generate detailed report
    if all_results:
        print(f"\n🔍 DETAILED FINDINGS")
        print("=" * 80)
        
        for file_path, file_results in sorted(all_results.items()):
            print(f"\n📁 {file_path}")
            print("-" * 60)
            
            for pattern_name, matches in file_results.items():
                if matches:
                    print(f"  🔸 {pattern_name.upper()} ({len(matches)} matches):")
                    for line_num, line_content in matches:
                        print(f"    Line {line_num}: {line_content}")
                    print()
    
    # Generate prioritized action items
    print(f"\n🎯 PRIORITIZED ACTION ITEMS")
    print("=" * 80)
    
    priority_files = []
    
    for file_path, file_results in all_results.items():
        priority_score = 0
        issue_types = []
        
        if 'forward_ref_patterns' in file_results:
            priority_score += 10
            issue_types.append("ForwardRef/TypeAdapter issues")
        
        if 'type_adapter_usage' in file_results:
            priority_score += 8
            issue_types.append("TypeAdapter usage")
        
        if 'request_parameter_patterns' in file_results:
            priority_score += 7
            issue_types.append("Request parameter patterns")
        
        if 'query_usage' in file_results:
            priority_score += 5
            issue_types.append("Query usage")
        
        if 'depends_usage' in file_results:
            priority_score += 5
            issue_types.append("Depends usage")
        
        if priority_score > 0:
            priority_files.append((priority_score, file_path, issue_types))
    
    # Sort by priority
    priority_files.sort(key=lambda x: x[0], reverse=True)
    
    print("Files requiring immediate attention (sorted by priority):")
    print()
    
    for i, (score, file_path, issue_types) in enumerate(priority_files[:10], 1):
        print(f"{i:2d}. {file_path} (Priority: {score})")
        print(f"    Issues: {', '.join(issue_types)}")
        print()
    
    # Generate file list for later steps
    print(f"\n📝 FILE LIST FOR REFACTORING")
    print("=" * 80)
    
    refactor_files = []
    for file_path, file_results in all_results.items():
        line_numbers = []
        for pattern_matches in file_results.values():
            for line_num, _ in pattern_matches:
                line_numbers.append(line_num)
        
        if line_numbers:
            refactor_files.append({
                'file': str(file_path),
                'lines': sorted(set(line_numbers)),
                'total_matches': len(line_numbers)
            })
    
    # Sort by number of matches
    refactor_files.sort(key=lambda x: x['total_matches'], reverse=True)
    
    print("Files needing refactoring (sorted by number of issues):")
    print()
    
    for i, file_info in enumerate(refactor_files[:20], 1):
        print(f"{i:2d}. {file_info['file']}")
        print(f"    Lines: {', '.join(map(str, file_info['lines']))}")
        print(f"    Total matches: {file_info['total_matches']}")
        print()
    
    # Generate summary report for next steps
    print(f"\n📋 REFACTORING GUIDANCE")
    print("=" * 80)
    
    print("Common patterns found that need attention:")
    print()
    
    pattern_summary = {}
    for file_results in all_results.values():
        for pattern_name, matches in file_results.items():
            if pattern_name not in pattern_summary:
                pattern_summary[pattern_name] = 0
            pattern_summary[pattern_name] += len(matches)
    
    sorted_patterns = sorted(pattern_summary.items(), key=lambda x: x[1], reverse=True)
    
    for pattern_name, count in sorted_patterns:
        print(f"  • {pattern_name}: {count} occurrences")
    
    print(f"\n🚨 IMMEDIATE ACTION REQUIRED")
    print("=" * 80)
    
    print("1. Fix TypeAdapter forward reference issues in:")
    for score, file_path, issue_types in priority_files:
        if "ForwardRef/TypeAdapter issues" in issue_types:
            print(f"   - {file_path}")
    
    print("\n2. Review and update FastAPI dependency patterns in:")
    for score, file_path, issue_types in priority_files:
        if "Request parameter patterns" in issue_types:
            print(f"   - {file_path}")
    
    print("\n3. Ensure proper model rebuilding in files with:")
    for file_path, file_results in all_results.items():
        if 'model_rebuild_patterns' in file_results:
            print(f"   - {file_path}")
    
    print(f"\n✅ AUDIT COMPLETE")
    print("=" * 80)
    print("This report should guide the refactoring efforts in later steps.")


if __name__ == "__main__":
    main()
