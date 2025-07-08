import json
import sys
from typing import List


def aggregate_sarif(files: List[str], output: str = 'combined.sarif'):
    combined_sarif = {
        'version': '2.1.0',
        'runs': []
    }

    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                sarif_data = json.load(f)
                # Append runs from each SARIF file to the combined list
                if 'runs' in sarif_data:
                    combined_sarif['runs'].extend(sarif_data['runs'])
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Write combined SARIF to the output file
    with open(output, 'w') as f:
        json.dump(combined_sarif, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python aggregate_sarif.py <list_of_sarif_files>")
        sys.exit(1)

    sarif_files = sys.argv[1:]
    aggregate_sarif(sarif_files)
