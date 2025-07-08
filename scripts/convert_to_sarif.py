#!/usr/bin/env python3
"""
Convert security tool outputs to SARIF format
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
import uuid


def convert_bandit_to_sarif(bandit_json_file: Path, output_file: Path) -> None:
    """Convert bandit JSON output to SARIF format"""
    
    with open(bandit_json_file, 'r') as f:
        bandit_data = json.load(f)
    
    # Create SARIF structure
    sarif = {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "bandit",
                        "version": "1.7.5",
                        "informationUri": "https://bandit.readthedocs.io/",
                        "rules": []
                    }
                },
                "results": []
            }
        ]
    }
    
    # Process bandit results
    run = sarif["runs"][0]
    rules_seen = set()
    
    for result in bandit_data.get("results", []):
        # Create rule if not seen before
        rule_id = result.get("test_id", "unknown")
        if rule_id not in rules_seen:
            rule = {
                "id": rule_id,
                "shortDescription": {
                    "text": result.get("test_name", "Unknown test")
                },
                "fullDescription": {
                    "text": result.get("issue_text", "No description available")
                },
                "help": {
                    "text": result.get("issue_text", "No help available")
                },
                "defaultConfiguration": {
                    "level": convert_severity_to_sarif_level(result.get("issue_severity", "medium"))
                }
            }
            run["tool"]["driver"]["rules"].append(rule)
            rules_seen.add(rule_id)
        
        # Create result
        sarif_result = {
            "ruleId": rule_id,
            "message": {
                "text": result.get("issue_text", "Security issue detected")
            },
            "level": convert_severity_to_sarif_level(result.get("issue_severity", "medium")),
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": result.get("filename", "unknown")
                        },
                        "region": {
                            "startLine": result.get("line_number", 1),
                            "startColumn": 1
                        }
                    }
                }
            ],
            "properties": {
                "confidence": result.get("issue_confidence", "medium"),
                "severity": result.get("issue_severity", "medium"),
                "lineRange": result.get("line_range", [])
            }
        }
        
        run["results"].append(sarif_result)
    
    # Add metadata
    run["properties"] = {
        "converted_at": datetime.now().isoformat(),
        "source_file": str(bandit_json_file),
        "converter": "pynomaly-sarif-converter"
    }
    
    # Write SARIF file
    with open(output_file, 'w') as f:
        json.dump(sarif, f, indent=2)
    
    print(f"✅ Converted {len(run['results'])} results to SARIF format")
    print(f"   📁 Output: {output_file}")


def convert_severity_to_sarif_level(severity: str) -> str:
    """Convert bandit severity to SARIF level"""
    mapping = {
        "low": "note",
        "medium": "warning", 
        "high": "error"
    }
    return mapping.get(severity.lower(), "warning")


def create_sample_sarif_files():
    """Create sample SARIF files for testing aggregation"""
    
    # Sample SARIF file 1 - CodeQL results
    codeql_sarif = {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "CodeQL",
                        "version": "2.15.0",
                        "informationUri": "https://github.com/github/codeql"
                    }
                },
                "results": [
                    {
                        "ruleId": "py/sql-injection",
                        "message": {
                            "text": "Potential SQL injection vulnerability"
                        },
                        "level": "error",
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {
                                        "uri": "src/pynomaly/infrastructure/database/models.py"
                                    },
                                    "region": {
                                        "startLine": 42,
                                        "startColumn": 1
                                    }
                                }
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    # Sample SARIF file 2 - Trivy results
    trivy_sarif = {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "Trivy",
                        "version": "0.47.0",
                        "informationUri": "https://trivy.dev/"
                    }
                },
                "results": [
                    {
                        "ruleId": "CVE-2023-1234",
                        "message": {
                            "text": "Vulnerable dependency detected"
                        },
                        "level": "warning",
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {
                                        "uri": "requirements.txt"
                                    },
                                    "region": {
                                        "startLine": 5,
                                        "startColumn": 1
                                    }
                                }
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    # Write sample files
    Path("artifacts/security").mkdir(parents=True, exist_ok=True)
    
    with open("artifacts/security/codeql_results.sarif", "w") as f:
        json.dump(codeql_sarif, f, indent=2)
    
    with open("artifacts/security/trivy_results.sarif", "w") as f:
        json.dump(trivy_sarif, f, indent=2)
    
    print("✅ Created sample SARIF files:")
    print("   - artifacts/security/codeql_results.sarif")
    print("   - artifacts/security/trivy_results.sarif")


def main():
    parser = argparse.ArgumentParser(
        description="Convert security tool outputs to SARIF format"
    )
    parser.add_argument(
        "--bandit-json",
        help="Bandit JSON file to convert"
    )
    parser.add_argument(
        "--output",
        help="Output SARIF file"
    )
    parser.add_argument(
        "--create-samples",
        action="store_true",
        help="Create sample SARIF files for testing"
    )
    
    args = parser.parse_args()
    
    if args.create_samples:
        create_sample_sarif_files()
        return
    
    if args.bandit_json and args.output:
        convert_bandit_to_sarif(Path(args.bandit_json), Path(args.output))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
