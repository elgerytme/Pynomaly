import json
import sys

# Script to convert Bandit JSON output to SARIF

def convert_bandit_to_sarif(input_file, output_file):
    with open(input_file, 'r') as infile:
        bandit_data = json.load(infile)

    sarif_data = {
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "Bandit",
                        "rules": []
                    }
                },
                "results": []
            }
        ]
    }

    for result in bandit_data.get("results", []):
        rule_id = result["test_id"]

        # Add rules if not already added
        if not any(rule["id"] == rule_id for rule in sarif_data["runs"][0]["tool"]["driver"]["rules"]):
            # Map severity to SARIF levels
            severity_map = {
                "HIGH": "error",
                "MEDIUM": "warning", 
                "LOW": "note"
            }
            
            sarif_data["runs"][0]["tool"]["driver"]["rules"].append({
                "id": rule_id,
                "shortDescription": {
                    "text": result["issue_text"]
                },
                "fullDescription": {
                    "text": result.get("more_info", result["issue_text"])
                },
                "defaultConfiguration": {
                    "level": severity_map.get(result["issue_severity"], "note")
                }
            })

        sarif_data["runs"][0]["results"].append({
            "ruleId": rule_id,
            "message": {
                "text": result["issue_text"]
            },
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": result["filename"]
                        },
                        "region": {
                            "startLine": result["line_number"],
                            "endLine": result["line_number"]
                        }
                    }
                }
            ]
        })

    with open(output_file, 'w') as outfile:
        json.dump(sarif_data, outfile, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <bandit_json_input> <sarif_output>")
        sys.exit(1)

    convert_bandit_to_sarif(sys.argv[1], sys.argv[2])

