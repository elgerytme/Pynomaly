#!/usr/bin/env python3
"""
Validate project structure against FILE_ORGANIZATION_STANDARDS.

This script validates the project structure and blocks commits that introduce
new violations to the organization standards.
"""

import json
import sys
from pathlib import Path

# Import the existing validation logic
sys.path.insert(0, str(Path(__file__).parent))
from validate_file_organization import print_results, validate_file_organization


def main():
    """Main validation function for pre-commit hook."""
    print("Validating project structure against FILE_ORGANIZATION_STANDARDS...")

    # Run the validation
    is_valid, violations, suggestions = validate_file_organization()

    # Print results
    print_results(is_valid, violations, suggestions)

    # Save validation report
    report = {
        "is_valid": is_valid,
        "violations": violations,
        "suggestions": suggestions,
        "validation_type": "pre-commit",
        "timestamp": str(Path.cwd()),
    }

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    with open(reports_dir / "structure_validation.json", "w") as f:
        json.dump(report, f, indent=2)

    # Exit with error code if validation failed to block commit
    if not is_valid:
        print("\nCOMMIT BLOCKED: Structure validation failed")
        print("Please fix the violations above before committing")
        sys.exit(1)
    else:
        print("\nStructure validation passed - commit allowed")
        sys.exit(0)


if __name__ == "__main__":
    main()
