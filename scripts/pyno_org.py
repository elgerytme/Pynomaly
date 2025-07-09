#!/usr/bin/env python3
"""
Unified CLI entry point for Pynomaly file organization.

This script provides a unified command-line interface for validating and organizing
project files according to Pynomaly standards.

Usage:
    pyno-org validate        # exits 0/1
    pyno-org organize --dry  # prints plan  
    pyno-org organize --fix  # executes
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

# Import existing functionality from the validation and organization modules
try:
    from validation.validate_file_organization import validate_file_organization
    from analysis.organize_files import FileOrganizer
except ImportError:
    # Fallback to relative imports if the modules are not in the path
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from validation.validate_file_organization import validate_file_organization
    from analysis.organize_files import FileOrganizer


def validate_command() -> int:
    """
    Execute file organization validation.
    
    Returns:
        0 if validation passes, 1 if validation fails
    """
    print("[*] Validating file organization...")
    
    is_valid, violations, suggestions = validate_file_organization()
    
    if is_valid:
        print("[+] File organization validation PASSED")
        return 0
    else:
        print("[-] File organization validation FAILED")
        print(f"\n[!] Found {len(violations)} violations:")
        for violation in violations:
            print(f"  • {violation}")
        
        if suggestions:
            print(f"\n[!] Suggested fixes:")
            for suggestion in suggestions:
                print(f"  • {suggestion}")
        
        print(f"\n[>] Run 'pyno-org organize --fix' to apply automated fixes")
        return 1


def organize_command(dry_run: bool = True) -> int:
    """
    Execute file organization operations.
    
    Args:
        dry_run: If True, only show what would be done without making changes
        
    Returns:
        0 on success, 1 on failure
    """
    try:
        organizer = FileOrganizer(dry_run=dry_run)
        
        print("[*] Pynomaly File Organization Tool")
        print("=" * 50)
        
        # Analyze current state
        analysis = organizer.analyze_repository()
        
        if not analysis["stray_files"] and not analysis["stray_directories"]:
            print("[+] Repository is already well-organized!")
            return 0
        
        print(f"[*] Analysis Results:")
        print(f"  • Stray files: {len(analysis['stray_files'])}")
        print(f"  • Stray directories: {len(analysis['stray_directories'])}")
        
        # Plan operations
        operations = organizer.plan_organization(analysis)
        
        if not operations:
            print("[!] No operations needed")
            return 0
        
        print(f"\n[*] Planned Operations ({len(operations)} total):")
        for i, operation in enumerate(operations, 1):
            action = operation["action"].upper()
            source = operation["source"]
            target = operation.get("target", "DELETED")
            print(f"  {i:2d}. {action}: {source} -> {target}")
        
        # Execute operations
        results = organizer.execute_operations(operations)
        
        # Print summary
        summary = results["summary"]
        status = "DRY RUN" if dry_run else "EXECUTED"
        print(f"\n[*] Operation Summary ({status})")
        print("=" * 50)
        print(f"[+] Executed: {summary['executed']}")
        print(f"[>] Skipped: {summary['skipped']}")
        print(f"[-] Errors: {summary['errors']}")
        print(f"[>] Moves: {summary['moves']}")
        print(f"[X] Deletions: {summary['deletions']}")
        
        if results["errors"]:
            print("\n[-] Errors encountered:")
            for error in results["errors"]:
                print(f"  • {error['operation']['source']}: {error['error']}")
            return 1
        
        if dry_run:
            print(f"\n[>] Run 'pyno-org organize --fix' to execute these changes")
        else:
            print(f"\n[+] File organization completed successfully!")
            
        return 0
        
    except Exception as e:
        print(f"[-] Error during organization: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="pyno-org",
        description="Unified CLI for Pynomaly file organization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pyno-org validate              # Validate file organization
  pyno-org organize --dry        # Show what would be organized
  pyno-org organize --fix        # Execute organization changes
  pyno-org organize --fix --force # Execute without confirmation
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate file organization against project standards"
    )
    
    # Organize command
    organize_parser = subparsers.add_parser(
        "organize", 
        help="Organize files according to project standards"
    )
    organize_parser.add_argument(
        "--dry",
        action="store_true",
        help="Show what would be done without making changes (default behavior)"
    )
    organize_parser.add_argument(
        "--fix",
        action="store_true", 
        help="Execute the organization changes"
    )
    organize_parser.add_argument(
        "--force",
        action="store_true",
        help="Execute without confirmation prompts"
    )
    organize_parser.add_argument(
        "--output",
        type=str,
        help="Save detailed report to file"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle no command provided
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute commands
    if args.command == "validate":
        return validate_command()
    
    elif args.command == "organize":
        # Default to dry run unless --fix is specified
        dry_run = not args.fix
        
        if args.fix and not args.force:
            # Ask for confirmation in fix mode
            response = input(f"\n[!] Execute file organization changes? [y/N]: ")
            if response.lower() != "y":
                print("[-] Operation cancelled")
                return 1
        
        return organize_command(dry_run=dry_run)
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
