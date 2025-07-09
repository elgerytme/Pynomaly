#!/usr/bin/env python3
"""
Validate project structure against FILE_ORGANIZATION_STANDARDS.

This script validates the project structure and blocks commits that introduce
new violations to the organization standards. It generates a SARIF report for
GitHub code scanning.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
import uuid

# Import the existing validation logic
sys.path.insert(0, str(Path(__file__).parent))
from validate_file_organization import print_results, validate_file_organization


SARIF_VERSION = "2.1.0"


def generate_sarif(violations):
    """Generate SARIF report from validation violations."""
    runs = []
    tool = {
        "driver": {
            "name": "validate_structure",
            "informationUri": "https://github.com/pynomaly/pynomaly",
            "rules": []
        }
    }
    results = []
    for violation in violations:
        result = {
            "ruleId": str(uuid.uuid4()),
            "level": "error",
            "message": {
                "text": violation
            },
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": "file://pathname"
                        },
                        "region": {
                            "startLine": 0
                        }
                    }
                }
            ]
        }
        results.append(result)

    run = {
        "tool": tool,
        "results": results
    }
    runs.append(run)

    sarif_report = {
        "$schema": "https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0.json",
        "version": SARIF_VERSION,
        "runs": runs
    }

    return sarif_report


def main():
    """Main validation function for pre-commit hook."""
    print("Validating project structure against FILE_ORGANIZATION_STANDARDS...")

    # Run the validation
    is_valid, violations, suggestions = validate_file_organization()

    # Print results
    print_results(is_valid, violations, suggestions)

    # Save JSON and SARIF validation report
    timestamp = datetime.now().isoformat()

    json_report = {
        "is_valid": is_valid,
        "violations": violations,
        "suggestions": suggestions,
        "validation_type": "pre-commit",
        "timestamp": timestamp
    }

    sarif_report = generate_sarif(violations)

    reports_dir = Path("reports/quality")
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(reports_dir / "structure_validation.json", "w") as f:
        json.dump(json_report, f, indent=2)

    with open(reports_dir / "structure_validation.sarif", "w") as f:
        json.dump(sarif_report, f, indent=2)

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
