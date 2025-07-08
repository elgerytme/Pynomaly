import json
import sys

# Script to aggregate multiple SARIF files into one

def aggregate_sarif(input_files, output_file):
    aggregated_sarif = {
        "version": "2.1.0",
        "runs": []
    }

    for input_file in input_files:
        with open(input_file, 'r') as infile:
            sarif_part = json.load(infile)
            aggregated_sarif["runs"].extend(sarif_part.get("runs", []))

    with open(output_file, 'w') as outfile:
        json.dump(aggregated_sarif, outfile, indent=2)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} cinput_sarif1c ... cinput_sarifNc coutput_sarifc")
        sys.exit(1)

    aggregate_sarif(sys.argv[1:-1], sys.argv[-1])

